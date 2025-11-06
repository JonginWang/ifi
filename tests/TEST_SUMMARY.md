# Tests Directory Summary

This document summarizes the functionality of all test files in the `tests/` directory.
Test files with duplicate functionality are identified for consolidation/deletion.

## Test File Structure

```
tests/
├── analysis/          # Analysis-related tests
├── db_controller/     # Database controller tests
├── scope/             #TODO: Tek scope function tests
├── gui/               #TODO: Interactive GUI and its layout tests
└── utils/             # Utility function tests
```

---

## 1. Analysis Tests

### 1.1 Spectrum Analysis Tests

#### `test_spectrum_simple.py`
- **Purpose**: Simple test for basic SpectrumAnalysis functionality
- **Test Content**:
  - Synthetic signal generation and noise handling
  - FFT-based center frequency detection
  - STFT analysis
- **Duplication Risk**: Functional overlap with `test_spectrum_comprehensive.py`

#### `test_spectrum_comprehensive.py`
- **Purpose**: Comprehensive SpectrumAnalysis test suite
- **Test Content**:
  - Various synthetic signal generation (sine, chirp, multi-tone, etc.)
  - FFT analysis
  - STFT analysis (various parameters)
  - CWT analysis
- **Duplication Risk**: Can include `test_spectrum_simple.py` functionality

#### `test_spectrum_error_cases.py`
- **Purpose**: SpectrumAnalysis error handling test
- **Test Content**:
  - Invalid input parameter handling
  - Edge case handling
  - Exception handling
- **Duplication Risk**: Low (dedicated error case tests)

#### `test_spectrum_dynamic_stft.py`
- **Purpose**: Dynamic STFT parameter selection test
- **Test Content**:
  - Automatic nperseg selection based on signal length
  - Automatic noverlap calculation
  - Dynamic selection validation for various signal lengths
- **Duplication Risk**: Low (dedicated feature test)

#### `test_analysis_accuracy.py`
- **Purpose**: Time-frequency analysis accuracy test
- **Test Content**:
  - STFT single-tone frequency detection accuracy
  - CWT linear chirp tracking accuracy
- **Duplication Risk**: Possible partial overlap with `test_spectrum_comprehensive.py`

#### `test_analysis_performance.py`
- **Purpose**: Time-frequency analysis performance benchmark
- **Test Content**:
  - STFT performance measurement (light/heavy modes)
  - CWT performance measurement
- **Duplication Risk**: Possible overlap with `test_performance_comprehensive.py`

---

### 1.2 Phase Analysis Tests

#### `test_phase_analysis.py`
- **Purpose**: Basic PhaseChangeDetector functionality test
- **Test Content**:
  - PhaseChangeDetector initialization
  - Phase change detection
- **Duplication Risk**: Possible overlap with `test_unified_phase_analysis.py`

#### `test_phase_reconstruction.py`
- **Purpose**: PhaseConverter phase reconstruction method test
- **Test Content**:
  - CDM method phase difference verification
  - IQ method phase difference verification
  - FPGA method phase difference verification
- **Duplication Risk**: Low (specific reconstruction method test)

#### `test_phase_modulation_comparison.py`
- **Purpose**: Comparison of various phase modulation methods (MATLAB vs Python)
- **Test Content**:
  - Constant phase offset comparison
  - Linear phase evolution comparison
  - Sinusoidal phase modulation comparison
  - MATLAB hilbert vs Python hilbert vs CDM vs IQ vs FPGA comparison
- **Duplication Risk**: Low (dedicated MATLAB comparison)

#### `test_unified_phase_analysis.py`
- **Purpose**: Unified phase analysis test
- **Test Content**:
  - Unified phase detection functionality
- **Duplication Risk**: Possible overlap with `test_phase_analysis.py`

#### `test_simple_phase.py`
- **Purpose**: Simple test for basic phase analysis functionality
- **Test Content**:
  - Basic phase analysis functionality
- **Duplication Risk**: Can be integrated into `test_phase_analysis.py` or `test_unified_phase_analysis.py`

---

### 1.3 Phi2ne (Phase-to-Density) Tests

#### `test_phi2ne_comprehensive.py`
- **Purpose**: Comprehensive phi2ne module test suite
- **Test Content**:
  - Numba cache setup
  - PhaseConverter initialization and basic functionality
  - Phase-to-density conversion
  - Various phase calculation methods (CDM, IQ, FPGA)
- **Duplication Risk**: Low (comprehensive test suite)

#### `test_phi2ne_error_cases.py`
- **Purpose**: phi2ne module error handling test
- **Test Content**:
  - PhaseConverter error cases
  - Phase-to-density conversion error handling
- **Duplication Risk**: Low (dedicated error case tests)

#### `test_interferometry_params.py`
- **Purpose**: Interferometry parameter calculation test
- **Test Content**:
  - get_interferometry_params standalone function test
  - Class method tests
  - Phase-to-density conversion integration test
  - Numba optimization test
- **Duplication Risk**: Low (dedicated parameter calculation test)

---

### 1.4 Integration Tests

#### `test_integration_comprehensive.py`
- **Purpose**: Comprehensive IFI package integration test
- **Test Content**:
  - End-to-end workflow tests
  - Inter-module integration tests
  - Integration scenarios using mock data
- **Duplication Risk**: Possible overlap with `test_integration_full.py`

#### `test_integration_full.py`
- **Purpose**: Full analysis pipeline integration test
- **Test Content**:
  - Complete analysis pipeline test
  - Spectrum analysis integration
  - Density calculation integration
  - Plotting integration
- **Duplication Risk**: Possible overlap with `test_integration_comprehensive.py`

#### `test_main_analysis.py`
- **Purpose**: Basic main_analysis module test
- **Test Content**:
  - Module import test
  - Numba function tests
  - Minimal analysis functionality test
- **Duplication Risk**: Low (dedicated main_analysis test)

#### `test_main_analysis_error_cases.py`
- **Purpose**: main_analysis error handling test
- **Test Content**:
  - main_analysis error cases
- **Duplication Risk**: Possible overlap with `test_analysis_error_cases.py`

#### `test_simple_unified.py`
- **Purpose**: Simple integration test
- **Test Content**:
  - Simple integration functionality test
- **Duplication Risk**: Can be included in other integration tests

#### `test_signal_analysis_comprehensive.py`
- **Purpose**: Comprehensive signal analysis test
- **Test Content**:
  - Comprehensive signal analysis functionality test
- **Duplication Risk**: Possible overlap with other integration tests

---

### 1.5 Performance Tests

#### `test_performance.py`
- **Purpose**: Basic performance test
- **Test Content**:
  - Basic performance measurement
- **Duplication Risk**: Possible overlap with `test_performance_comprehensive.py`

#### `test_performance_comprehensive.py`
- **Purpose**: Comprehensive performance and parallel processing test
- **Test Content**:
  - Dask integration performance
  - Caching performance
  - Overall system performance
- **Duplication Risk**: Possible partial overlap with `test_analysis_performance.py`

---

### 1.6 Plotting Tests

#### `test_plots_comprehensive.py`
- **Purpose**: Comprehensive plotting functionality test
- **Test Content**:
  - Waveform plots
  - Spectrum plots
  - Filter response plots
  - Density plots
  - Comparison plots
- **Duplication Risk**: Possible overlap with `test_shot_visualization.py`

#### `test_shot_visualization.py`
- **Purpose**: Shot data visualization test
- **Test Content**:
  - Waveform plots
  - Spectrum plots
  - Filter response plots
  - Density plots
  - Comparison plots
  - Real shot data visualization
- **Duplication Risk**: Nearly identical to `test_plots_comprehensive.py`

---

### 1.7 Comparison Tests

#### `test_hilbert_matlab_comparison.py`
- **Purpose**: Hilbert transform MATLAB comparison test
- **Test Content**:
  - Direct MATLAB hilbert comparison
  - CDM hilbert usage comparison
- **Duplication Risk**: Low (dedicated MATLAB comparison)

#### `test_cdm_full_pipeline_comparison.py`
- **Purpose**: CDM full pipeline MATLAB comparison
- **Test Content**:
  - CDM full pipeline MATLAB comparison
  - CDM test using PhaseConverter
- **Duplication Risk**: Low (dedicated CDM pipeline comparison)

---

### 1.8 Error Case Tests

#### `test_analysis_error_cases.py`
- **Purpose**: Comprehensive analysis module error handling test
- **Test Content**:
  - PhaseAnalysis error cases
  - SpectrumAnalysis error cases
  - Plotting error cases
  - MainAnalysis error cases
- **Duplication Risk**: Overlap with `test_spectrum_error_cases.py`, `test_phi2ne_error_cases.py`, `test_main_analysis_error_cases.py`

---

### 1.9 Specialized Tests

#### `test_linear_drift.py`
- **Purpose**: Linear drift analysis test
- **Test Content**:
  - Linear drift analysis
- **Duplication Risk**: Low (dedicated specialized feature test)

---

## 2. Database Controller Tests

#### `test_db.py`
- **Purpose**: Database interaction test suite (pytest-compatible)
- **Test Content**:
  - NAS_DB tests:
    - Initialization and connection
    - First call: Data retrieval and cache creation
    - Second call: Cache loading verification
    - Cache file creation verification
    - Force remote fetch (bypass cache)
    - File header retrieval
  - VEST_DB tests:
    - Initialization and connection
    - Shot existence check
    - Data loading
    - Next shot code retrieval
- **Execution**: 
  - Run with pytest: `pytest tests/db_controller/test_db.py`
  - Run directly: `python tests/db_controller/test_db.py` (basic connectivity test only)
- **Pytest Compatibility**: Full pytest support with fixtures and class-based test structure
- **Duplication Risk**: Low (comprehensive DB test suite)

#### `test_db_error_cases.ipynb` / `test_db_error_cases.py`
- **Purpose**: Database error handling test
- **Test Content**:
  - NAS_DB error cases
  - VEST_DB error cases
  - File I/O error cases
- **Format**: Jupyter notebook (preferred) and Python script (reference)
- **Duplication Risk**: Low (dedicated error case tests)

#### `test_dask_integration.ipynb` / `test_dask_integration.py`
- **Purpose**: Dask parallel processing integration test
- **Test Content**:
  - Dask scheduler comparison
  - DB controller integration
  - DB operations with Dask integration
- **Format**: Jupyter notebook (preferred) and Python script (reference)
- **Duplication Risk**: Low (dedicated Dask integration test)

**Note**: Backup files (`test_dask_integration_backup.py`, `test_simple_dask.py`) have been removed.

---

## 3. Utils Tests

#### `test_cache_fix.ipynb` / `test_cache_fix.py`
- **Purpose**: Cache setup and ssqueezepy import test
- **Test Content**:
  - Cache directory creation
  - Numba threading layer configuration (TBB, OpenMP, safe)
  - ssqueezepy import after cache setup
  - CWT operations with proper time axis handling
- **Format**: Jupyter notebook (preferred) and Python script (reference)
- **Duplication Risk**: Low (dedicated cache and import test)

#### `test_torch_reload.ipynb`
- **Purpose**: PyTorch enable/reload functionality test
- **Test Content**:
  - Enabling real PyTorch after dummy module creation
  - Handling of partially loaded modules
  - Version verification
  - Safe module reloading via `ifi.utils.cache_setup.enable_torch()`
- **Format**: Jupyter notebook only
- **Duplication Risk**: Low (dedicated PyTorch test)

#### `test_ide_access.ipynb` / `test_ide_access.py`
- **Purpose**: IDE access and import test
- **Test Content**:
  - Module import tests
  - Path configuration
  - Functionality verification
  - Project access validation
- **Format**: Jupyter notebook (preferred) and Python script (reference)
- **Duplication Risk**: Low (dedicated IDE test)

#### `test_logging_behavior.ipynb` / `test_logging_behavior.py`
- **Purpose**: Logging system behavior test
- **Test Content**:
  - LogManager singleton behavior
  - Log level configuration
  - Specialized logger creation
  - Output destinations
- **Format**: Jupyter notebook (preferred) and Python script (reference)
- **Duplication Risk**: Low (dedicated logging test)

**Note**: `test_core_functionality.py` has been removed (functionality distributed across specialized test notebooks).

---

## Duplicate Test Consolidation Proposal

### High Priority (Clear Duplicates)

1. **Spectrum Tests**
   - `test_spectrum_simple.py` → Integrate into or delete in favor of `test_spectrum_comprehensive.py`
   - `test_analysis_accuracy.py` → Integrate into `test_spectrum_comprehensive.py` or maintain separately

2. **Phase Tests**
   - `test_phase_analysis.py`, `test_unified_phase_analysis.py`, `test_simple_phase.py` → Consolidate into one
   - Recommendation: Consolidate into `test_phase_analysis.py` and delete the others

3. **Integration Tests**
   - `test_integration_comprehensive.py` and `test_integration_full.py` → Consolidate into one
   - `test_simple_unified.py` → Include in integration tests or delete
   - `test_signal_analysis_comprehensive.py` → Include in integration tests or delete

4. **Performance Tests**
   - `test_performance.py` → Integrate into or delete in favor of `test_performance_comprehensive.py`

5. **Plotting Tests**
   - `test_plots_comprehensive.py` and `test_shot_visualization.py` → Consolidate into one

6. **Error Cases**
   - Since `test_analysis_error_cases.py` is already comprehensive, individual error tests may overlap, but functional separation may be preferable

7. **Backup Files** ✅ **COMPLETED**
   - `test_dask_integration_backup.py` → ✅ Deleted
   - `test_simple_dask.py` → ✅ Deleted

8. **Utils** ✅ **COMPLETED**
   - Tests organized into specialized notebooks:
     - `test_cache_fix.ipynb` - Cache and import tests
     - `test_torch_reload.ipynb` - PyTorch enable/reload tests
     - `test_ide_access.ipynb` - IDE access tests
     - `test_logging_behavior.ipynb` - Logging behavior tests
   - `test_core_functionality.py` → ✅ Removed (functionality distributed across specialized notebooks)

### Medium Priority (Needs Review)

- `test_main_analysis_error_cases.py` vs `test_analysis_error_cases.py`
- `test_analysis_performance.py` vs `test_performance_comprehensive.py`

### Low Priority (Recommended to Keep)

- MATLAB comparison tests (`test_phase_modulation_comparison.py`, `test_hilbert_matlab_comparison.py`, `test_cdm_full_pipeline_comparison.py`)
- Specialized feature tests (`test_linear_drift.py`, `test_spectrum_dynamic_stft.py`, `test_interferometry_params.py`)
- Module-specific error cases (maintain functional separation)

---

## Expected Structure After Consolidation

```
tests/
├── analysis/
│   ├── test_spectrum_comprehensive.py      # comprehensive spectrum tests
│   ├── test_spectrum_error_cases.py        # spectrum error cases
│   ├── test_spectrum_dynamic_stft.py       # dynamic STFT (keep)
│   ├── test_phase_analysis.py              # integrated phase tests
│   ├── test_phase_reconstruction.py        # phase reconstruction (keep)
│   ├── test_phase_modulation_comparison.py # MATLAB comparison (keep)
│   ├── test_phi2ne_comprehensive.py        # comprehensive phi2ne
│   ├── test_phi2ne_error_cases.py          # phi2ne errors
│   ├── test_interferometry_params.py       # interferometry params (keep)
│   ├── test_integration_full.py            # integration tests (consolidated)
│   ├── test_main_analysis.py               # main_analysis
│   ├── test_analysis_error_cases.py        # comprehensive analysis errors
│   ├── test_performance_comprehensive.py   # comprehensive performance
│   ├── test_shot_visualization.py          # visualization (consolidated)
│   ├── test_linear_drift.py                # linear drift (keep)
│   ├── test_hilbert_matlab_comparison.py   # MATLAB comparison (keep)
│   └── test_cdm_full_pipeline_comparison.py # CDM comparison (keep)
├── db_controller/
│   ├── test_db.py                          # basic DB (pytest-compatible test suite)
│   ├── test_db.ipynb                      # basic DB (Jupyter notebook for interactive testing)
│   ├── test_db_error_cases.ipynb          # DB errors (notebook)
│   ├── test_db_error_cases.py             # DB errors (reference script)
│   ├── test_dask_integration.ipynb        # Dask integration (notebook)
│   └── test_dask_integration.py           # Dask integration (reference script)
└── utils/
    ├── test_cache_fix.ipynb               # cache and import (notebook)
    ├── test_cache_fix.py                  # cache and import (reference script)
    ├── test_torch_reload.ipynb            # PyTorch enable/reload (notebook)
    ├── test_ide_access.ipynb              # IDE access (notebook)
    ├── test_ide_access.py                 # IDE access (reference script)
    ├── test_logging_behavior.ipynb        # logging behavior (notebook)
    └── test_logging_behavior.py           # logging behavior (reference script)
```

---

## Document Information

- **Date**: 2025-01-31
- **Last Updated**: 2025-01-31
- **Purpose**: Identify test file duplicates and establish consolidation plan
- **Status**: 
  - ✅ Database Controller tests organized (backup files removed)
  - ✅ Utils tests organized (specialized notebooks created)
  - ✅ Test structure documented in README files
- **Next Steps**: Continue consolidation of analysis tests according to priorities above

