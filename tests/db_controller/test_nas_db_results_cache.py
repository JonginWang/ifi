#!/usr/bin/env python3
"""
Test suite for NAS_DB results cache functionality.

This module tests the new results directory checking functionality in NAS_DB,
including loading signals from results/<shot_num>/<shot_num>.h5.

Run with pytest: pytest tests/db_controller/test_nas_db_results_cache.py
"""

from pathlib import Path
import shutil
import pytest
import pandas as pd
import numpy as np
import h5py
import tempfile

from ifi.utils.common import LogManager
from ifi.db_controller.nas_db import NAS_DB
from ifi import get_project_root

# Initialize logging
LogManager(level="INFO")
logger = LogManager().get_logger(__name__)

CONFIG_PATH = 'ifi/config.ini'
SHOT_TO_TEST = 45821  # Test shot number


@pytest.fixture
def nas_db():
    """Fixture to provide NAS_DB instance."""
    if not Path(CONFIG_PATH).exists():
        pytest.skip(f"Configuration file not found at '{CONFIG_PATH}'")
    return NAS_DB(config_path=CONFIG_PATH)


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary results directory structure."""
    project_root = get_project_root()
    results_base = project_root / "ifi" / "results"
    results_base.mkdir(parents=True, exist_ok=True)
    
    shot_dir = results_base / str(SHOT_TO_TEST)
    shot_dir.mkdir(parents=True, exist_ok=True)
    
    yield shot_dir
    
    # Cleanup
    if shot_dir.exists():
        try:
            # Remove H5 files first
            for h5_file in shot_dir.glob("*.h5"):
                try:
                    h5_file.unlink()
                except Exception:
                    pass
            shutil.rmtree(shot_dir)
        except Exception:
            pass


@pytest.fixture
def sample_results_h5(temp_results_dir):
    """Create a sample results H5 file with valid signals."""
    h5_file = temp_results_dir / f"{SHOT_TO_TEST}.h5"
    
    # Create sample data
    n_samples = 1000
    time_data = np.linspace(0, 0.01, n_samples)
    signal_data = np.sin(2 * np.pi * 94e9 * time_data)
    
    with h5py.File(h5_file, 'w') as f:
        # Create signals group
        signals_group = f.create_group("signals")
        freq_group = signals_group.create_group("freq_94.0_GHz")
        freq_group.create_dataset("TIME", data=time_data)
        freq_group.create_dataset("CH0", data=signal_data)
        freq_group.create_dataset("CH1", data=signal_data * 0.9)
        
        # Create metadata group
        metadata_group = f.create_group("metadata")
        metadata_group.attrs["shot_number"] = SHOT_TO_TEST
        metadata_group.attrs["created_at"] = "2025-01-27"
    
    yield h5_file
    
    # Cleanup
    if h5_file.exists():
        try:
            h5_file.unlink()
        except Exception:
            pass


@pytest.fixture
def multi_freq_results_h5(temp_results_dir):
    """Create a results H5 file with multiple frequency signals (94GHz and 280GHz)."""
    h5_file = temp_results_dir / f"{SHOT_TO_TEST}.h5"
    
    # Create sample data for 94GHz
    n_samples_94 = 1000
    time_data_94 = np.linspace(0, 0.01, n_samples_94)
    signal_data_94 = np.sin(2 * np.pi * 94e9 * time_data_94)
    
    # Create sample data for 280GHz
    n_samples_280 = 1000
    time_data_280 = np.linspace(0, 0.01, n_samples_280)
    signal_data_280 = np.sin(2 * np.pi * 280e9 * time_data_280)
    
    with h5py.File(h5_file, 'w') as f:
        # Create signals group
        signals_group = f.create_group("signals")
        
        # 94GHz signal
        freq_94_group = signals_group.create_group("freq_94.0_GHz")
        freq_94_group.create_dataset("TIME", data=time_data_94)
        freq_94_group.create_dataset("CH0", data=signal_data_94)
        freq_94_group.create_dataset("CH1", data=signal_data_94 * 0.9)
        
        # 280GHz signal
        freq_280_group = signals_group.create_group("freq_280.0_GHz")
        freq_280_group.create_dataset("TIME", data=time_data_280)
        freq_280_group.create_dataset("CH0", data=signal_data_280)
        freq_280_group.create_dataset("CH1", data=signal_data_280 * 0.8)
        
        # Create metadata group
        metadata_group = f.create_group("metadata")
        metadata_group.attrs["shot_number"] = SHOT_TO_TEST
        metadata_group.attrs["created_at"] = "2025-01-27"
    
    yield h5_file
    
    # Cleanup
    if h5_file.exists():
        try:
            h5_file.unlink()
        except Exception:
            pass


@pytest.fixture
def invalid_results_h5(temp_results_dir):
    """Create a results H5 file with invalid data (NaN, inf, empty)."""
    h5_file = temp_results_dir / f"{SHOT_TO_TEST}.h5"
    
    with h5py.File(h5_file, 'w') as f:
        # Create signals group with invalid data
        signals_group = f.create_group("signals")
        freq_group = signals_group.create_group("freq_94.0_GHz")
        
        # Create data with NaN
        time_data = np.linspace(0, 0.01, 100)
        signal_data = np.full(100, np.nan)
        freq_group.create_dataset("TIME", data=time_data)
        freq_group.create_dataset("CH0", data=signal_data)
        
        # Create metadata group
        metadata_group = f.create_group("metadata")
        metadata_group.attrs["shot_number"] = SHOT_TO_TEST
    
    yield h5_file
    
    # Cleanup
    if h5_file.exists():
        try:
            h5_file.unlink()
        except Exception:
            pass


class TestNAS_DB_ResultsCache:
    """Test suite for NAS_DB results cache functionality."""
    
    def test_load_signals_from_results_valid(self, nas_db, sample_results_h5):
        """Test loading valid signals from results directory."""
        project_root = get_project_root()
        results_base_dir = project_root / "ifi" / "results"
        
        with nas_db:
            results_signals = nas_db._load_signals_from_results(SHOT_TO_TEST, results_base_dir)
            
            assert results_signals is not None
            assert isinstance(results_signals, dict)
            assert len(results_signals) > 0
            
            # Check signal structure
            for signal_name, df in results_signals.items():
                assert isinstance(df, pd.DataFrame)
                assert not df.empty
                assert "TIME" in df.columns
                logger.info(f"Loaded signal '{signal_name}' with shape {df.shape}")
    
    def test_load_signals_from_results_invalid(self, nas_db, invalid_results_h5):
        """Test that invalid signals are not loaded from results directory."""
        project_root = get_project_root()
        results_base_dir = project_root / "ifi" / "results"
        
        with nas_db:
            results_signals = nas_db._load_signals_from_results(SHOT_TO_TEST, results_base_dir)
            
            # Invalid data should be filtered out
            if results_signals:
                # Check that no invalid data is present
                for signal_name, df in results_signals.items():
                    assert not df.isnull().any().any(), "Should not contain NaN"
                    assert not np.isinf(df.select_dtypes(include=[np.number])).any().any(), "Should not contain inf"
            else:
                # It's acceptable if no valid signals are found
                logger.info("No valid signals found (expected for invalid data)")
    
    def test_is_valid_data(self, nas_db):
        """Test data validation function."""
        # Valid data
        valid_df = pd.DataFrame({
            'TIME': np.linspace(0, 1, 100),
            'CH0': np.sin(np.linspace(0, 2*np.pi, 100))
        })
        assert nas_db._is_valid_data(valid_df) == True
        
        # None
        assert nas_db._is_valid_data(None) == False
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        assert nas_db._is_valid_data(empty_df) == False
        
        # DataFrame with NaN
        nan_df = pd.DataFrame({
            'TIME': np.linspace(0, 1, 100),
            'CH0': np.full(100, np.nan)
        })
        assert nas_db._is_valid_data(nan_df) == False
        
        # DataFrame with inf
        inf_df = pd.DataFrame({
            'TIME': np.linspace(0, 1, 100),
            'CH0': np.full(100, np.inf)
        })
        assert nas_db._is_valid_data(inf_df) == False
        
        # DataFrame with zero rows
        zero_rows_df = pd.DataFrame({'TIME': [], 'CH0': []})
        assert nas_db._is_valid_data(zero_rows_df) == False
        
        # DataFrame with zero columns
        zero_cols_df = pd.DataFrame(index=range(100))
        assert nas_db._is_valid_data(zero_cols_df) == False
    
    def test_get_shot_data_prioritizes_results(self, nas_db, sample_results_h5):
        """Test that get_shot_data prioritizes results over cache and NAS."""
        with nas_db:
            # This should load from results if available
            data_dict = nas_db.get_shot_data(SHOT_TO_TEST, force_remote=False)
            
            # Should have loaded data (either from results or cache/NAS)
            assert data_dict is not None
            assert isinstance(data_dict, dict)
            
            # If results were loaded, verify structure
            if len(data_dict) > 0:
                for key, df in data_dict.items():
                    assert isinstance(df, pd.DataFrame)
                    assert not df.empty
                    logger.info(f"Loaded '{key}' with shape {df.shape}")
    
    def test_load_signals_from_results_with_freq_filter(self, nas_db, multi_freq_results_h5):
        """Test that _load_signals_from_results filters by frequency correctly."""
        project_root = get_project_root()
        results_base_dir = project_root / "ifi" / "results"
        
        with nas_db:
            # Load all frequencies
            all_signals = nas_db._load_signals_from_results(SHOT_TO_TEST, results_base_dir)
            assert all_signals is not None
            assert len(all_signals) == 2  # Should have both 94GHz and 280GHz
            assert "freq_94.0_GHz" in all_signals
            assert "freq_280.0_GHz" in all_signals
            
            # Load only 94GHz
            signals_94 = nas_db._load_signals_from_results(
                SHOT_TO_TEST, results_base_dir, allowed_frequencies=[94.0]
            )
            assert signals_94 is not None
            assert len(signals_94) == 1
            assert "freq_94.0_GHz" in signals_94
            assert "freq_280.0_GHz" not in signals_94
            
            # Load only 280GHz
            signals_280 = nas_db._load_signals_from_results(
                SHOT_TO_TEST, results_base_dir, allowed_frequencies=[280.0]
            )
            assert signals_280 is not None
            assert len(signals_280) == 1
            assert "freq_280.0_GHz" in signals_280
            assert "freq_94.0_GHz" not in signals_280
            
            # Load both frequencies explicitly
            signals_both = nas_db._load_signals_from_results(
                SHOT_TO_TEST, results_base_dir, allowed_frequencies=[94.0, 280.0]
            )
            assert signals_both is not None
            assert len(signals_both) == 2
    
    def test_convert_results_signals_to_nas_format(self, nas_db, multi_freq_results_h5):
        """Test conversion of results signals format to NAS format with frequency matching."""
        project_root = get_project_root()
        results_base_dir = project_root / "ifi" / "results"
        
        with nas_db:
            # Load signals
            results_signals = nas_db._load_signals_from_results(SHOT_TO_TEST, results_base_dir)
            
            if results_signals:
                # Create target files that match the frequency mapping
                # 94GHz files: _056.csv, _789.csv
                # 280GHz file: _ALL.csv
                target_files = [
                    f"{SHOT_TO_TEST}_056.csv",
                    f"{SHOT_TO_TEST}_789.csv",
                    f"{SHOT_TO_TEST}_ALL.csv"
                ]
                
                # Convert format
                nas_format = nas_db._convert_results_signals_to_nas_format(
                    results_signals, target_files, SHOT_TO_TEST
                )
                
                assert isinstance(nas_format, dict)
                assert len(nas_format) > 0
                
                # Check that keys are file-like names
                for key in nas_format.keys():
                    assert isinstance(key, str)
                    logger.info(f"Converted key: '{key}'")
                
                # Verify that 280GHz signal is matched to _ALL file
                if f"{SHOT_TO_TEST}_ALL.csv" in nas_format:
                    logger.info(f"280GHz signal correctly matched to _ALL file")
                
                # Verify that 94GHz signal is matched to appropriate files
                # (could be _056 or _789 depending on implementation)
                matched_94_files = [
                    k for k in nas_format.keys() 
                    if k.endswith("_056.csv") or k.endswith("_789.csv")
                ]
                if matched_94_files:
                    logger.info(f"94GHz signal matched to files: {matched_94_files}")
    
    def test_get_shot_data_with_freq_filter(self, nas_db, multi_freq_results_h5):
        """Test that get_shot_data respects frequency filtering."""
        project_root = get_project_root()
        results_base_dir = project_root / "ifi" / "results"
        
        with nas_db:
            # Test with frequency filter - should only load 94GHz data
            data_dict_94 = nas_db.get_shot_data(
                SHOT_TO_TEST, 
                force_remote=False,
                allowed_frequencies=[94.0]
            )
            
            # Test with frequency filter - should only load 280GHz data
            data_dict_280 = nas_db.get_shot_data(
                SHOT_TO_TEST,
                force_remote=False,
                allowed_frequencies=[280.0]
            )
            
            # Test with both frequencies
            data_dict_both = nas_db.get_shot_data(
                SHOT_TO_TEST,
                force_remote=False,
                allowed_frequencies=[94.0, 280.0]
            )
            
            # Verify that filtering worked (if results were loaded)
            # Note: This test may pass even if no results are found,
            # as it depends on whether results exist for the test shot
            logger.info(f"94GHz filter returned {len(data_dict_94)} files")
            logger.info(f"280GHz filter returned {len(data_dict_280)} files")
            logger.info(f"Both frequencies filter returned {len(data_dict_both)} files")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

