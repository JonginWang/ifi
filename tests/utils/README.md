# Utility Tests

This directory contains tests for IFI utility functions and infrastructure components.

## Test Files

### Infrastructure Tests

- **`test_cache_fix.ipynb`**: Jupyter notebook testing cache setup and ssqueezepy import:
  - Cache directory creation
  - ssqueezepy import after cache setup
  - CWT (Continuous Wavelet Transform) functionality
  - Time axis control in CWT
  - Numba threading layer configuration

- **`test_cache_fix.py`**: Original Python script (converted to notebook, kept for reference)

- **`test_torch_reload.ipynb`**: Jupyter notebook testing PyTorch enable functionality:
  - Enabling real PyTorch after dummy module creation
  - Handling of partially loaded modules
  - Version verification
  - Safe module reloading via `ifi.utils.cache_setup.enable_torch()`

### Validation Tests

- **`test_ide_access.ipynb`**: Jupyter notebook for IDE compatibility verification:
  - Module import tests
  - Path configuration
  - Functionality verification
  - Project access validation

- **`test_ide_access.py`**: Original Python script (converted to notebook, kept for reference)

- **`test_logging_behavior.ipynb`**: Jupyter notebook for logging system tests:
  - LogManager configuration
  - Specialized logger creation
  - Log level behavior
  - Output destinations
  - Singleton behavior verification

- **`test_logging_behavior.py`**: Original Python script (converted to notebook, kept for reference)

## Running Tests

All tests are provided as Jupyter notebooks for interactive execution:

```bash
# Activate virtual environment
.\.venv\Scripts\activate  # Windows PowerShell
# or
source .venv/bin/activate  # Linux/Mac

# Start Jupyter
jupyter notebook tests/utils/
```

## Test Coverage

- ✅ Cache setup and configuration
- ✅ ssqueezepy import and CWT functionality
- ✅ PyTorch enable/reload functionality
- ✅ IDE access and imports
- ✅ Logging configuration and behavior
- ✅ Path handling and utilities

## Test Details

### Cache and Import Tests

- **`test_cache_fix.ipynb`**: Verifies that cache setup works correctly and that ssqueezepy can be imported after cache configuration
- Tests Numba threading layer configuration (TBB, OpenMP, safe)
- Validates CWT operations with proper time axis handling

### PyTorch Tests

- **`test_torch_reload.ipynb`**: Tests the `enable_torch()` function from `ifi.utils.cache_setup`
- Verifies safe reloading of PyTorch when dummy modules exist
- Handles DLL conflicts and module initialization issues

### IDE Access Tests

- **`test_ide_access.ipynb`**: Verifies that the IFI package can be imported correctly in IDE environments
- Tests path configuration and module accessibility

### Logging Tests

- **`test_logging_behavior.ipynb`**: Tests LogManager singleton behavior
- Verifies log level configuration
- Tests specialized logger creation
- Validates output destinations

## Notes

- Tests use minimal test data to ensure reproducibility
- Some tests may require specific dependencies (ssqueezepy, torch)
- Cache tests may create temporary directories
- Clean up after running tests if needed
- Python script versions are kept for reference but notebooks are preferred

## Troubleshooting

### Import Errors

If you encounter import errors:
1. Run `test_ide_access.ipynb` to verify IDE configuration
2. Check that project root is in Python path
3. Verify virtual environment is activated

### Cache Issues

If cache-related tests fail:
1. Check file permissions for cache directory
2. Run `test_cache_fix.ipynb` to diagnose issues
3. Try clearing cache and re-running tests
4. Verify Numba threading layer configuration (TBB/OpenMP installation)

### PyTorch Issues

If PyTorch tests fail:
1. Ensure PyTorch is installed in the environment
2. Check `ifi.utils.cache_setup.enable_torch()` function
3. Verify DLL conflicts are handled correctly
4. Run `test_torch_reload.ipynb` to diagnose issues
