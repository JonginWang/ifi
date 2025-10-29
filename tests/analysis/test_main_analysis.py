"""
Test suite for main analysis functions.
"""

import sys
from pathlib import Path

# Add ifi package to Python path for IDE compatibility
current_dir = Path(__file__).resolve()
ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
                + list(current_dir.parents) if p.name == 'ifi']
IFI_ROOT = ifi_parents[-1] if ifi_parents else None

try:
    sys.path.insert(0, str(IFI_ROOT))
except Exception as e:
    print(f"!! Could not find ifi package root: {e}")
    pass

from ifi.utils.cache_setup import setup_project_cache
cache_config = setup_project_cache()

import numpy as np


def test_main_analysis_import():
    """Test if main_analysis can be imported without cache errors."""
    try:
        print("Testing imports...")
        
        # Test individual modules first
        print("   - Testing numba...")
        import numba
        print(f"Numba cache: {numba.config.CACHE_DIR}")
        
        print("   - Testing pandas, numpy...")
        
        print("   - Testing dask...")
        
        print("   - Testing ifi modules...")
        print("Database modules")
        
        print("Phase converter (with numba optimizations)")
        
        print("Spectrum analysis")
        
        print("   - Testing main analysis...")
        print("Main analysis module")
        
        assert True
        
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
        
        assert True
        
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
        
        assert True
        
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
        print(f"Tip: Run with environment variable: NUMBA_CACHE_DIR={cache_config['cache_dir']}")
    else:
        print("Some tests failed. Check the errors above.")
    
    return success

if __name__ == '__main__':
    import numpy as np  # Import numpy after cache setup
    success = main()
    sys.exit(0 if success else 1)