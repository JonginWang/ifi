# IFI Test Suite

This directory contains comprehensive test suites for the IFI (Interferometer Data Analysis) package.

## Directory Structure

```
tests/
├── utils/                    # Utility function tests
│   ├── test_cache_fix.ipynb          # Cache setup and ssqueezepy import tests
│   ├── test_cache_fix.py             # Original Python script (reference)
│   ├── test_ide_access.ipynb         # IDE access and import tests
│   ├── test_ide_access.py            # Original Python script (reference)
│   ├── test_logging_behavior.ipynb   # Logging behavior tests
│   ├── test_logging_behavior.py      # Original Python script (reference)
│   └── test_torch_reload.ipynb       # PyTorch enable/reload tests
│
├── db_controller/            # Database controller tests
│   ├── test_db.py                    # Standalone DB test script (NOT for pytest)
│   ├── test_db_error_cases.ipynb     # Error handling tests (notebook)
│   ├── test_db_error_cases.py        # Original Python script (reference)
│   ├── test_dask_integration.ipynb   # Dask parallel processing tests (notebook)
│   └── test_dask_integration.py      # Original Python script (reference)
│
└── analysis/                 # Analysis module tests
    ├── test_main_analysis_integration.ipynb  # Main analysis workflow
    ├── test_phase_analysis_precision.ipynb   # Phase analysis precision
    ├── test_phase_analysis_real_data.ipynb   # Real data phase analysis
    └── test_vest_integration.ipynb            # VEST database integration
```

## Test Organization

### Utility Tests (`tests/utils/`)

These tests verify core utility functions and infrastructure:

- **`test_cache_fix.ipynb`**: Tests cache setup, ssqueezepy import, and CWT functionality
  - Cache directory creation
  - Numba threading layer configuration (TBB, OpenMP, safe)
  - ssqueezepy import after cache setup
  - CWT operations with proper time axis handling

- **`test_torch_reload.ipynb`**: Tests PyTorch enable functionality
  - Enabling real PyTorch after dummy module creation
  - Handling of partially loaded modules
  - Version verification
  - Safe module reloading via `ifi.utils.cache_setup.enable_torch()`

- **`test_ide_access.ipynb`**: Verifies IDE access to IFI modules
  - Module import tests
  - Path configuration
  - Functionality verification

- **`test_logging_behavior.ipynb`**: Tests logging configuration and behavior
  - LogManager singleton behavior
  - Log level configuration
  - Specialized logger creation
  - Output destinations

**Note**: Python script versions (`.py`) are kept for reference but notebooks are the preferred testing method.

### Database Controller Tests (`tests/db_controller/`)

These tests verify database connectivity and data loading:

- **`test_db.py`**: Standalone Python script for basic database tests
  - **NOT meant to be run with pytest** (uses `pytest.skip` marker)
  - Run directly: `python tests/db_controller/test_db.py`
  - Tests NAS_DB initialization, connection, and data loading
  - Tests VEST_DB initialization, connection, and data loading
  - Tests cache file creation and usage
  - Uses hardcoded test values (configurable in script)
  - Handles connection failures gracefully

- **`test_db_error_cases.ipynb`**: Error handling tests for:
  - Invalid shot numbers
  - Connection failures
  - SQL injection attempts
  - File I/O errors

- **`test_dask_integration.ipynb`**: Tests Dask parallel processing:
  - Different schedulers (threads, processes)
  - DB controller integration
  - Simulated DB operations parallelization

**Note**: Python script versions (`.py`) are kept for reference but notebooks are preferred.

### Analysis Tests (`tests/analysis/`)

See `tests/analysis/README.md` for detailed documentation.

## Running Tests

### Jupyter Notebooks

All tests are provided as Jupyter notebooks for interactive execution:

```bash
# Activate virtual environment
.\.venv\Scripts\activate  # Windows PowerShell
# or
source .venv/bin/activate  # Linux/Mac

# Start Jupyter
jupyter notebook tests/

# Navigate to specific test notebook and run cells
```

### Python Scripts

Some standalone Python scripts are available for direct execution:

```bash
# Run standalone DB test script (NOT with pytest)
python tests/db_controller/test_db.py
```

**Important**: `test_db.py` is configured to skip pytest execution. Run it directly with Python.

## Test Requirements

All tests require:
- Python 3.10+
- IFI package installed (development mode recommended)
- Required dependencies (see `requirements.txt`)
- Access to NAS and/or VEST database (for database tests)
- Configuration file at `ifi/config.ini`

## Test Maintenance

### Adding New Tests

1. Create a new Jupyter notebook in the appropriate directory
2. Follow the naming convention: `test_<feature>_<aspect>.ipynb`
3. Include markdown cells for documentation
4. Add import/setup cell at the beginning
5. Structure tests logically with markdown section headers
6. Update this README

### Removing Tests

- Obsolete tests should be removed, not just commented out
- Update this README when removing tests
- Consider migrating useful test code to new notebooks

### Test Data

- Tests should use minimal, reproducible test data
- Real data tests should be clearly marked
- Large test datasets should be cached or downloaded on-demand

## Notes

- **Notebook Priority**: Jupyter notebooks are the preferred testing method
- **Python Scripts**: Original Python scripts are kept for reference but notebooks are preferred
- **Pytest Compatibility**: `test_db.py` explicitly skips pytest (uses `pytest.mark.skip`)
- **Test Organization**: Tests are organized by functional area (utils, db_controller, analysis)

## Troubleshooting

### Import Errors

If you encounter import errors:
1. Run `test_ide_access.ipynb` to verify IDE configuration
2. Check that project root is in Python path
3. Verify virtual environment is activated

### Database Connection Issues

If database tests fail:
1. Check `ifi/config.ini` configuration
2. Verify network connectivity to NAS/VEST
3. Check SSH tunnel configuration for remote access
4. See `test_db_error_cases.ipynb` for error handling examples

### Cache Issues

If cache-related tests fail:
1. Check file permissions for cache directory
2. Run `test_cache_fix.ipynb` to diagnose issues
3. Try clearing cache and re-running tests
4. Verify Numba threading layer configuration (TBB/OpenMP installation)

### Pytest Issues

If pytest tries to run `test_db.py`:
- The script is configured to skip pytest execution automatically
- If issues persist, ensure `pytest` is installed and the skip marker is working
- Run directly with `python tests/db_controller/test_db.py` instead

## Contributing

When adding new functionality:
1. Add corresponding tests in appropriate notebook
2. Update this README
3. Ensure tests pass before committing
