#!/usr/bin/env python3
"""
Test script for main_analysis.py with proper numba cache setup.
"""

# ============================================================================
# CRITICAL: Set up numba cache BEFORE any imports
# ============================================================================
import os
import tempfile
import sys

# Add project to path
sys.path.insert(0, os.path.abspath('.'))

# Configure numba cache
project_cache = os.path.join(os.path.abspath('.'), 'cache', 'numba_cache')
try:
    os.makedirs(project_cache, exist_ok=True)
    cache_dir = project_cache
    print(f"Using project cache: {cache_dir}")
except (PermissionError, OSError):
    # Fallback to user temp directory
    cache_dir = os.path.join(tempfile.gettempdir(), 'ifi_numba_cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using temp cache: {cache_dir}")

os.environ['NUMBA_CACHE_DIR'] = cache_dir
os.environ['NUMBA_THREADING_LAYER'] = 'safe'
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'

print("Numba cache configured, now testing main_analysis...")

def test_main_analysis_import():
    """Test if main_analysis can be imported without cache errors."""
    try:
        print("Testing imports...")
        
        # Test individual modules first
        print("   - Testing numba...")
        import numba
        print(f"Numba cache: {numba.config.CACHE_DIR}")
        
        print("   - Testing pandas, numpy...")
        import pandas as pd
        import numpy as np
        
        print("   - Testing dask...")
        import dask
        
        print("   - Testing ifi modules...")
        from ifi.db_controller.nas_db import NAS_DB
        from ifi.db_controller.vest_db import VEST_DB
        print("Database modules")
        
        from ifi.analysis.phi2ne import PhaseConverter
        print("Phase converter (with numba optimizations)")
        
        from ifi.analysis import spectrum
        print("Spectrum analysis")
        
        print("   - Testing main analysis...")
        from ifi.analysis import main_analysis
        print("Main analysis module")
        
        return True
        
    except Exception as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numba_functions():
    """Test numba-optimized functions."""
    try:
        print("\nTesting numba-optimized functions...")
        
        from ifi.analysis.phi2ne import _normalize_iq_signals, _calculate_differential_phase
        
        # Generate small test data
        i_signal = np.random.randn(1000)
        q_signal = np.random.randn(1000)
        
        print("   - Testing _normalize_iq_signals...")
        i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)
        print(f"Shape: {i_norm.shape}")
        
        print("   - Testing _calculate_differential_phase...")
        phase_diff = _calculate_differential_phase(i_norm, q_norm)
        print(f"Shape: {phase_diff.shape}")
        
        return True
        
    except Exception as e:
        print(f"Numba function error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_analysis():
    """Test minimal analysis workflow."""
    try:
        print("\nTesting minimal analysis workflow...")
        
        # Import after cache setup
        from ifi.analysis.phi2ne import PhaseConverter
        
        # Test phase converter
        print("   - Testing PhaseConverter...")
        converter = PhaseConverter()
        
        # Test with dummy data
        dummy_phase = np.random.randn(100) * 0.1
        density = converter.phase_to_density(dummy_phase, 94)
        print(f"Density calculation: {density.shape}")
        
        return True
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Main Analysis with Cache Setup")
    print("=" * 50)
    
    success = True
    
    # Test 1: Import modules
    if not test_main_analysis_import():
        success = False
    
    # Test 2: Numba functions
    if not test_numba_functions():
        success = False
    
    # Test 3: Minimal analysis
    if not test_minimal_analysis():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed! Main analysis is ready to use.")
        print(f"Tip: Run with environment variable: NUMBA_CACHE_DIR={cache_dir}")
    else:
        print("Some tests failed. Check the errors above.")
    
    return success

if __name__ == '__main__':
    import numpy as np  # Import numpy after cache setup
    success = main()
    sys.exit(0 if success else 1)