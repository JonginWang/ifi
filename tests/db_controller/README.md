# Database Controller Tests

This directory contains tests for IFI database controllers (NAS_DB and VEST_DB).

## Test Files

### Basic Database Tests

- **`test_db.py`**: Pytest-compatible test suite for database functionality:
  - **Full pytest support** with fixtures and class-based test structure
  - Run with pytest: `pytest tests/db_controller/test_db.py`
  - Run directly: `python tests/db_controller/test_db.py` (basic connectivity test only)
  - **NAS_DB Tests**:
    - Initialization and connection
    - First call: Data retrieval and cache creation
    - Second call: Cache loading verification
    - Cache file creation verification
    - Force remote fetch (bypass cache)
    - File header retrieval
  - **VEST_DB Tests**:
    - Initialization and connection
    - Shot existence check
    - Data loading
    - Next shot code retrieval
  - Uses hardcoded test values (configurable in script)
  - Handles connection failures gracefully with `pytest.skip()`

- **`test_db.ipynb`**: Jupyter notebook for interactive database testing:
  - Same test coverage as `test_db.py`
  - Interactive execution and variable inspection
  - Visual output and debugging capabilities
  - Useful for development and troubleshooting

- **`test_db_error_cases.ipynb`**: Jupyter notebook for comprehensive error handling:
  - Invalid shot numbers
  - Empty shot numbers
  - Connection failures
  - SQL injection attempts
  - Invalid queries
  - File I/O errors
  - Invalid data types

- **`test_db_error_cases.py`**: Original Python script (converted to notebook, kept for reference)

### Parallel Processing Tests

- **`test_dask_integration.ipynb`**: Jupyter notebook for Dask integration tests:
  - Different Dask schedulers (threads, processes, single-threaded)
  - DB controller compatibility with Dask
  - Simulated file processing parallelization
  - Simulated DB operations parallelization
  - Performance measurements and speedup calculations

- **`test_dask_integration.py`**: Original Python script (converted to notebook, kept for reference)

## Running Tests

### Pytest Execution (Recommended)

```bash
# Run all database tests
pytest tests/db_controller/test_db.py

# Run specific test class
pytest tests/db_controller/test_db.py::TestNAS_DB
pytest tests/db_controller/test_db.py::TestVEST_DB

# Run specific test method
pytest tests/db_controller/test_db.py::TestNAS_DB::test_first_call_fetch_and_cache
pytest tests/db_controller/test_db.py::TestVEST_DB::test_load_shot_data

# Verbose output
pytest tests/db_controller/test_db.py -v

# Show print statements
pytest tests/db_controller/test_db.py -s
```

### Direct Python Script Execution

```bash
# Run basic connectivity test
python tests/db_controller/test_db.py
```

**Note**: Direct execution runs only basic connectivity tests. For comprehensive testing, use pytest.

### Jupyter Notebooks

Run notebooks interactively:

```bash
jupyter notebook tests/db_controller/
```

## Configuration

Tests require `ifi/config.ini` with:
- NAS connection settings
- SSH tunnel configuration
- VEST database connection settings
- Cache directory configuration

## Test Data

### `test_db.py` Configuration

The script uses hardcoded test values at the top of the file:
- `SHOT_TO_TEST = 45821` (NAS shot number)
- `VEST_SHOT_TO_TEST = 40656` (VEST shot number)
- `VEST_FIELD_TO_TEST = 109` (VEST field number)
- `CONFIG_PATH = 'ifi/config.ini'` (config file path)
- `CACHE_FOLDER = Path('./cache')` (cache directory)

Modify these values directly in the script before running.

### Notebook Configuration

- Notebooks use configurable test values within cells
- NAS tests require network access to NAS
- VEST tests require MySQL database access
- Cache files are created during tests

## Notes

- **Pytest Compatibility**: `test_db.py` is fully pytest-compatible with fixtures and class-based structure
- Tests automatically skip if configuration file is missing or connections are unavailable
- Uses `pytest.skip()` for graceful handling of missing resources
- Cache files persist after tests (for inspection)
- Jupyter notebook (`test_db.ipynb`) provides interactive testing environment

## Troubleshooting

### Connection Issues

- Check `ifi/config.ini` configuration
- Verify network connectivity
- Check SSH tunnel settings for remote NAS access
- Verify MySQL credentials for VEST database

### Cache Issues

- Cache files are stored in configured cache directory (default: `./cache`)
- Old cache files may need to be cleared
- Check file permissions for cache directory

### Shot Number Issues

- Update shot numbers in `test_db.py` or notebook if test shots are unavailable
- Use `test_db_error_cases.ipynb` to test error handling with invalid shots

### Pytest Issues

If pytest has issues running `test_db.py`:
- Ensure `pytest` is installed: `pip install pytest`
- Check that all dependencies are available
- Verify `ifi/config.ini` exists and is properly configured
- For connection issues, tests will automatically skip (expected behavior)
- Run with `-v` flag for verbose output: `pytest tests/db_controller/test_db.py -v`
- Run with `-s` flag to see print/logging output: `pytest tests/db_controller/test_db.py -s`
