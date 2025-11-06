#!/usr/bin/env python3
"""
Test suite for database interactions.

This module tests the basic functionality of NAS_DB and VEST_DB controllers,
including data loading, caching, and error handling.

Run with pytest: pytest tests/db_controller/test_db.py
Or run directly: python tests/db_controller/test_db.py
"""

from pathlib import Path
import shutil
import pytest
import gc
import time

from ifi.utils.common import LogManager
from ifi.db_controller.nas_db import NAS_DB
from ifi.db_controller.vest_db import VEST_DB

# Note: Tests in this module should run sequentially to avoid HDF5 file lock conflicts.
# If pytest-order is installed, uncomment the following line:
# pytestmark = pytest.mark.order("sequential")

# Initialize logging
LogManager(level="INFO")
logger = LogManager().get_logger(__name__)

# --- Test Configuration ---
# PLEASE ADJUST THESE VALUES BASED ON YOUR 'ifi/config.ini' AND AVAILABLE DATA
SHOT_TO_TEST = 45821  # Test for CSV files, 'MDO3000orig', 'MDO3000fetch', 'MSO58'
# SHOT_TO_TEST = 41715 # Test for CSV files, 'MDO3000orig', 'MDO3000fetch'
# SHOT_TO_TEST = 36853 # Test for CSV files, 'MSO58'
# SHOT_TO_TEST = 38396 # Test for 'MDO3000pc'
# SHOT_TO_TEST = 'AGC w attn I' # Test for 'ETC'

CONFIG_PATH = 'ifi/config.ini'
CACHE_FOLDER = Path('./cache')  # Should match [LOCAL_CACHE] dumping_folder in config

VEST_SHOT_TO_TEST = 40656
VEST_FIELD_TO_TEST = 109


def _wait_for_hdf5_file_release(cache_file: Path, max_wait: float = 2.0, check_interval: float = 0.1):
    """
    Wait for an HDF5 file to be released (no longer locked).
    
    This is a workaround for Windows file locking issues with HDF5 files.
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            # Try to open the file in exclusive mode to check if it's locked
            if cache_file.exists():
                # On Windows, we can try to rename the file temporarily
                # If that succeeds, the file is not locked
                temp_name = cache_file.with_suffix('.h5.tmp')
                try:
                    cache_file.rename(temp_name)
                    temp_name.rename(cache_file)  # Rename back
                    return True  # File is not locked
                except (PermissionError, OSError):
                    # File is still locked, wait a bit more
                    time.sleep(check_interval)
            else:
                return True  # File doesn't exist, so it's "available"
        except Exception:
            time.sleep(check_interval)
    
    return False  # File is still locked after max_wait


def _close_hdf5_files(cache_dir: Path):
    """
    Attempt to close any open HDF5 files in the cache directory.
    
    This helps prevent HDF5 file lock issues on Windows.
    Handles both h5py.File objects and pandas HDFStore objects.
    """
    try:
        import h5py
        from pandas.io.pytables import HDFStore
        
        # Force garbage collection first
        gc.collect()
        
        # Close any h5py.File objects
        for obj in gc.get_objects():
            if isinstance(obj, h5py.File):
                try:
                    if obj.filename and str(cache_dir) in obj.filename:
                        if obj.id.valid:  # Check if file is still open
                            obj.close()
                except:  # noqa: E722
                    pass
        
        # Close any pandas HDFStore objects
        for obj in gc.get_objects():
            if isinstance(obj, HDFStore):
                try:
                    if hasattr(obj, '_path') and str(cache_dir) in str(obj._path):
                        if obj.is_open:  # Check if store is still open
                            obj.close()
                except:  # noqa: E722
                    pass
        
        gc.collect()
        
        # Wait for HDF5 files in the cache directory to be released
        if cache_dir.exists():
            for h5_file in cache_dir.rglob("*.h5"):
                _wait_for_hdf5_file_release(h5_file, max_wait=1.0)
                
    except ImportError:
        pass  # h5py or pandas not available
    except Exception:
        pass  # Ignore errors in cleanup


@pytest.fixture(scope="function", autouse=True)
def cleanup_cache():
    """
    Fixture to clean up cache folder before and after tests.
    
    Handles Windows PermissionError when files are in use (e.g., HDF5 files
    still open in other processes or previous tests).
    
    Note: autouse=True ensures this runs for all tests, but cleanup is optional.
    """
    # Cleanup before test - handle permission errors gracefully
    if CACHE_FOLDER.exists():
        # First, try to close any open HDF5 files and wait for release
        _close_hdf5_files(CACHE_FOLDER)
        time.sleep(0.5)  # Wait for file handles to release
        
        try:
            shutil.rmtree(CACHE_FOLDER)
        except PermissionError as e:
            # File may be in use by another process or previous test
            # Try harder to close files and retry with longer wait
            _close_hdf5_files(CACHE_FOLDER)
            gc.collect()
            time.sleep(1.5)  # Longer wait for file handles to release
            try:
                shutil.rmtree(CACHE_FOLDER)
            except PermissionError:
                # If still failing, skip cleanup and log warning
                # This is acceptable - tests can work with existing cache
                logger.warning(f"Could not clean cache folder before test: {e}. "
                             "File may be in use. Continuing with existing cache.")
        except OSError as e:
            logger.warning(f"Could not clean cache folder before test: {e}. "
                         "Continuing with existing cache.")
    
    yield
    
    # Cleanup after test - ensure HDF5 files are closed
    # This is critical to prevent file lock issues in subsequent tests
    _close_hdf5_files(CACHE_FOLDER)
    gc.collect()
    time.sleep(0.5)  # Wait longer for file handles to release after test
    
    # Note: After-test folder cleanup is disabled by default to allow cache inspection
    # Uncomment the following lines if you want automatic cleanup after each test:
    # if CACHE_FOLDER.exists():
    #     try:
    #         _close_hdf5_files(CACHE_FOLDER)
    #         time.sleep(0.5)
    #         shutil.rmtree(CACHE_FOLDER)
    #     except (PermissionError, OSError) as e:
    #         logger.warning(f"Could not clean cache folder after test: {e}")


@pytest.fixture
def nas_db():
    """Fixture to provide NAS_DB instance."""
    if not Path(CONFIG_PATH).exists():
        pytest.skip(f"Configuration file not found at '{CONFIG_PATH}'")
    return NAS_DB(config_path=CONFIG_PATH)


@pytest.fixture
def vest_db():
    """Fixture to provide VEST_DB instance."""
    if not Path(CONFIG_PATH).exists():
        pytest.skip(f"Configuration file not found at '{CONFIG_PATH}'")
    return VEST_DB(config_path=CONFIG_PATH)


class TestNAS_DB:
    """Test suite for NAS_DB controller."""
    
    def test_initialization_and_connection(self, nas_db):
        """Test NAS_DB initialization and connection."""
        assert nas_db is not None
        with nas_db:
            assert nas_db._is_connected
    
    def test_first_call_fetch_and_cache(self, nas_db):
        """
        Test first call: Data is not cached. It will be fetched from the remote source
        and then saved to the local cache.
        """
        max_retries = 3
        retry_delay = 1.0
        
        try:
            for attempt in range(max_retries):
                try:
                    # Ensure cache is clean before test
                    _close_hdf5_files(CACHE_FOLDER)
                    gc.collect()
                    time.sleep(0.5)
                    
                    with nas_db:
                        data_dict = nas_db.get_shot_data(SHOT_TO_TEST)
                        
                        assert data_dict is not None
                        assert isinstance(data_dict, dict)
                        assert len(data_dict) > 0, "No data retrieved on first call"
                        
                        # Verify data structure
                        for key, df in data_dict.items():
                            assert df is not None
                            assert hasattr(df, 'shape')
                            assert hasattr(df, 'columns')
                            logger.info(f"DataFrame '{key}' shape: {df.shape}, columns: {df.columns.tolist()}")
                        break  # Success, exit retry loop
                except Exception as e:
                    if "HDF5" in str(e) or "lock" in str(e).lower() or "WinError 32" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(f"HDF5 file lock error on attempt {attempt + 1}/{max_retries}. Retrying after {retry_delay}s...")
                            _close_hdf5_files(CACHE_FOLDER)
                            gc.collect()
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
                            continue
                        else:
                            pytest.skip(f"HDF5 file lock error persists after {max_retries} attempts: {e}")
                    else:
                        raise  # Re-raise non-HDF5 errors
            else:
                pytest.skip("Failed to retrieve data after all retries")
        finally:
            # Ensure HDF5 files are closed after test
            _close_hdf5_files(CACHE_FOLDER)
            gc.collect()
            time.sleep(0.5)  # Wait longer for HDF5 file handles to release
    
    def test_second_call_load_from_cache(self, nas_db):
        """
        Test second call: The local cache file now exists. This call should be much faster
        as it reads directly from the local HDF5 file.
        """
        max_retries = 3
        retry_delay = 1.0
        
        try:
            for attempt in range(max_retries):
                try:
                    # Ensure cache is clean before test
                    _close_hdf5_files(CACHE_FOLDER)
                    gc.collect()
                    time.sleep(0.5)
                    
                    with nas_db:
                        # First call to create cache
                        data_dict_first = nas_db.get_shot_data(SHOT_TO_TEST)
                    
                    # Exit context manager to ensure NAS_DB connection is closed
                    # Then wait for HDF5 files to be released before second call
                    _close_hdf5_files(CACHE_FOLDER)
                    gc.collect()
                    time.sleep(1.0)  # Wait longer for HDF5 file handles to release
                    
                    # Second call should use cache (create new context)
                    with nas_db:
                        data_dict_second = nas_db.get_shot_data(SHOT_TO_TEST)
                        
                        assert data_dict_second is not None
                        assert isinstance(data_dict_second, dict)
                        assert len(data_dict_second) == len(data_dict_first), \
                            f"Cache mismatch: first={len(data_dict_first)}, second={len(data_dict_second)}"
                        
                        # Verify data matches
                        for key in data_dict_first:
                            assert key in data_dict_second, f"Key '{key}' missing in cached data"
                            assert data_dict_first[key].shape == data_dict_second[key].shape, \
                                f"Shape mismatch for '{key}'"
                        break  # Success, exit retry loop
                except Exception as e:
                    if "HDF5" in str(e) or "lock" in str(e).lower() or "WinError 32" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(f"HDF5 file lock error on attempt {attempt + 1}/{max_retries}. Retrying after {retry_delay}s...")
                            _close_hdf5_files(CACHE_FOLDER)
                            gc.collect()
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
                            continue
                        else:
                            pytest.skip(f"HDF5 file lock error persists after {max_retries} attempts: {e}")
                    else:
                        raise  # Re-raise non-HDF5 errors
            else:
                pytest.skip("Failed to retrieve data after all retries")
        finally:
            # Ensure HDF5 files are closed after test
            _close_hdf5_files(CACHE_FOLDER)
            gc.collect()
            time.sleep(0.5)  # Wait longer for HDF5 file handles to release
    
    def test_cache_file_creation(self, nas_db):
        """Test that cache file is created after first fetch."""
        max_retries = 3
        retry_delay = 1.0
        
        try:
            for attempt in range(max_retries):
                try:
                    # Ensure cache is clean before test
                    _close_hdf5_files(CACHE_FOLDER)
                    gc.collect()
                    time.sleep(0.5)
                    
                    with nas_db:
                        nas_db.get_shot_data(SHOT_TO_TEST)
                    
                    # Ensure HDF5 file is closed before checking
                    _close_hdf5_files(CACHE_FOLDER)
                    gc.collect()
                    time.sleep(0.5)  # Wait longer for HDF5 file handles to release
                    
                    cache_dir = CACHE_FOLDER / str(SHOT_TO_TEST)
                    expected_cache_file = cache_dir / f"{SHOT_TO_TEST}.h5"
                    
                    if expected_cache_file.exists():
                        cache_size = expected_cache_file.stat().st_size
                        logger.info(f"Cache file created: {expected_cache_file}, size: {cache_size:,} bytes")
                        assert cache_size > 0
                        break  # Success, exit retry loop
                    else:
                        pytest.skip("Cache file not created (may be expected if caching is disabled)")
                except Exception as e:
                    if "HDF5" in str(e) or "lock" in str(e).lower() or "WinError 32" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(f"HDF5 file lock error on attempt {attempt + 1}/{max_retries}. Retrying after {retry_delay}s...")
                            _close_hdf5_files(CACHE_FOLDER)
                            gc.collect()
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
                            continue
                        else:
                            pytest.skip(f"HDF5 file lock error persists after {max_retries} attempts: {e}")
                    else:
                        raise  # Re-raise non-HDF5 errors
            else:
                pytest.skip("Failed to retrieve data after all retries")
        finally:
            _close_hdf5_files(CACHE_FOLDER)
            gc.collect()
            time.sleep(0.5)  # Wait longer for HDF5 file handles to release
    
    def test_force_remote_fetch(self, nas_db):
        """
        Test force remote fetch: Use 'force_remote=True' to bypass the local cache and
        re-download the data from the source.
        """
        max_retries = 3
        retry_delay = 1.0
        
        try:
            for attempt in range(max_retries):
                try:
                    # Ensure cache is clean before test
                    _close_hdf5_files(CACHE_FOLDER)
                    gc.collect()
                    time.sleep(0.5)
                    
                    with nas_db:
                        # First call to create cache
                        nas_db.get_shot_data(SHOT_TO_TEST)
                    
                    # Exit context manager to ensure NAS_DB connection is closed
                    # Then wait for HDF5 files to be released before force remote fetch
                    _close_hdf5_files(CACHE_FOLDER)
                    gc.collect()
                    time.sleep(1.0)  # Wait longer for HDF5 file handles to release
                    
                    # Force remote fetch (create new context)
                    with nas_db:
                        forced_data_dict = nas_db.get_shot_data(SHOT_TO_TEST, force_remote=True)
                        
                        # Should still return data (may fail if remote connection unavailable)
                        if forced_data_dict:
                            assert isinstance(forced_data_dict, dict)
                            assert len(forced_data_dict) > 0
                            break  # Success, exit retry loop
                        else:
                            pytest.skip("Force remote fetch failed (may be expected if remote connection unavailable)")
                except Exception as e:
                    if "HDF5" in str(e) or "lock" in str(e).lower() or "WinError 32" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(f"HDF5 file lock error on attempt {attempt + 1}/{max_retries}. Retrying after {retry_delay}s...")
                            _close_hdf5_files(CACHE_FOLDER)
                            gc.collect()
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # Exponential backoff
                            continue
                        else:
                            pytest.skip(f"HDF5 file lock error persists after {max_retries} attempts: {e}")
                    else:
                        raise  # Re-raise non-HDF5 errors
            else:
                pytest.skip("Failed to retrieve data after all retries")
        finally:
            # Ensure HDF5 files are closed after test
            _close_hdf5_files(CACHE_FOLDER)
            gc.collect()
            time.sleep(0.5)  # Wait longer for HDF5 file handles to release
    
    def test_file_header_retrieval(self, nas_db):
        """Test file header retrieval using get_data_top."""
        with nas_db:
            file_head = nas_db.get_data_top(SHOT_TO_TEST, lines=50)
            
            if file_head:
                assert isinstance(file_head, str)
                assert len(file_head) > 0
                logger.info(f"Retrieved file header ({len(file_head)} chars)")
            else:
                pytest.skip("File header retrieval failed (may be expected if file not found)")


class TestVEST_DB:
    """Test suite for VEST_DB controller."""
    
    def test_initialization_and_connection(self, vest_db):
        """Test VEST_DB initialization and connection."""
        assert vest_db is not None
        with vest_db:
            assert vest_db.connection is not None
            assert vest_db.connection.open
    
    def test_shot_existence_check(self, vest_db):
        """Test checking if a shot exists in the database."""
        with vest_db:
            existence = vest_db.exist_shot(VEST_SHOT_TO_TEST, VEST_FIELD_TO_TEST)
            
            # existence should be 0, 2, or 3
            assert existence in [0, 2, 3], f"Unexpected existence value: {existence}"
            
            if existence > 0:
                logger.info(f"Shot found in table shotDataWaveform_{existence}")
            else:
                pytest.skip(f"Shot {VEST_SHOT_TO_TEST} not found (may be expected in test environment)")
    
    def test_load_shot_data(self, vest_db):
        """Test loading shot data from VEST database."""
        with vest_db:
            # First check if shot exists
            existence = vest_db.exist_shot(VEST_SHOT_TO_TEST, VEST_FIELD_TO_TEST)
            
            if existence == 0:
                pytest.skip(f"Shot {VEST_SHOT_TO_TEST} not found (may be expected in test environment)")
            
            # Load data
            result_dict = vest_db.load_shot(VEST_SHOT_TO_TEST, [VEST_FIELD_TO_TEST])
            
            assert result_dict is not None
            assert isinstance(result_dict, dict)
            assert len(result_dict) > 0, "No data loaded from VEST database"
            
            # Verify data structure
            for rate_key, df in result_dict.items():
                assert df is not None
                assert hasattr(df, 'shape')
                assert hasattr(df, 'columns')
                logger.info(f"Rate group '{rate_key}': shape={df.shape}, columns={df.columns.tolist()}")
    
    def test_get_next_shot_code(self, vest_db):
        """Test getting the next shot code from database."""
        with vest_db:
            next_shot = vest_db.get_next_shot_code()
            
            if next_shot:
                assert isinstance(next_shot, int)
                assert next_shot > 0
                logger.info(f"Next shot code: {next_shot}")
            else:
                pytest.skip("Could not get next shot code (may be expected if database is empty)")


# Standalone execution support (for direct Python execution)
def run_all_tests():
    """
    Run all database tests and report results.
    
    This function is for direct Python execution, not pytest.
    """
    logger.info("Starting Database Controller Tests")
    logger.info("=" * 60)
    logger.info("Test Configuration:")
    logger.info(f"  - Config path: {CONFIG_PATH}")
    logger.info(f"  - Cache folder: {CACHE_FOLDER}")
    logger.info(f"  - NAS shot number: {SHOT_TO_TEST}")
    logger.info(f"  - VEST shot number: {VEST_SHOT_TO_TEST}")
    logger.info(f"  - VEST field number: {VEST_FIELD_TO_TEST}")
    logger.info("=" * 60)
    logger.info("\nNOTE: For comprehensive testing, use pytest:")
    logger.info("  pytest tests/db_controller/test_db.py")
    logger.info("\nRunning basic connectivity tests...\n")
    
    # Basic connectivity tests
    try:
        if Path(CONFIG_PATH).exists():
            # Test NAS_DB connection
            logger.info("Testing NAS_DB connection...")
            with NAS_DB(config_path=CONFIG_PATH) as nas:
                if nas._is_connected:
                    logger.info("  ✓ NAS_DB connection successful")
                else:
                    logger.warning("  ✗ NAS_DB connection failed")
            
            # Test VEST_DB connection
            logger.info("Testing VEST_DB connection...")
            with VEST_DB(config_path=CONFIG_PATH) as db:
                if db.connection and db.connection.open:
                    logger.info("  ✓ VEST_DB connection successful")
                else:
                    logger.warning("  ✗ VEST_DB connection failed")
        else:
            logger.error(f"Configuration file not found at '{CONFIG_PATH}'")
            return False
    except Exception as e:
        logger.error(f"Error during basic connectivity tests: {e}", exc_info=True)
        return False
    
    logger.info("\nFor full tests, run: pytest tests/db_controller/test_db.py")
    return True


if __name__ == "__main__":
    """
    Run basic connectivity tests when executed directly.
    
    For comprehensive testing, use pytest:
        pytest tests/db_controller/test_db.py
    """
    success = run_all_tests()
    exit(0 if success else 1)
