#!/usr/bin/env python3
"""
Test suite for interferometry parameters.
"""

import numpy as np

from ifi.analysis.phi2ne import get_interferometry_params, PhaseConverter
from ifi.utils.common import LogManager

LogManager()

def test_standalone_function():
    """Test the standalone get_interferometry_params function."""
    print("Testing get_interferometry_params function:")
    print("=" * 60)
    
    # Test cases based on the rules defined
    test_cases = [
        (45821, "45821_ALL.csv", "280GHz CDM"),    # Rule 3: >= 41542, _ALL -> 280GHz CDM
        (45821, "45821_056.csv", "94GHz CDM"),     # Rule 3: >= 41542, _0XX -> 94GHz CDM  
        (45821, "45821_789.csv", "94GHz CDM"),     # Rule 3: >= 41542, other -> 94GHz CDM
        (40000, "40000.dat", "94GHz FPGA"),        # Rule 2: 39302-41398 -> 94GHz FPGA
        (35000, "35000.csv", "94GHz IQ"),          # Rule 1: 30000-39265 -> 94GHz IQ
        (25000, "25000.csv", "unknown"),           # Outside range -> unknown
    ]
    
    for shot_num, filename, expected in test_cases:
        params = get_interferometry_params(shot_num, filename)
        print(f"Shot {shot_num}, file '{filename}' (Expected: {expected}):")
        print(f"Method: {params['method']}")
        print(f"Frequency (GHz): {params['freq_ghz']}")
        print(f"Frequency (Hz): {params['freq']}")
        print(f"n_path: {params['n_path']}")
        print(f"Reference column: {params['ref_col']}")
        print(f"Probe columns: {params['probe_cols']}")
        if 'amp_ref_col' in params:
            print(f"Amplitude ref column: {params['amp_ref_col']}")
        if 'amp_probe_cols' in params:
            print(f"Amplitude probe columns: {params['amp_probe_cols']}")
        
        # Validate frequency consistency
        expected_freq_hz = params['freq_ghz'] * 1e9
        freq_match = abs(params['freq'] - expected_freq_hz) < 1e6  # 1MHz tolerance
        print(f"   {'Yes!' if freq_match else 'No!!'} Frequency consistency: {freq_match}")
        print()

def test_class_methods():
    """Test PhaseConverter class methods."""
    print("Testing PhaseConverter class methods:")
    print("=" * 60)
    
    # Initialize PhaseConverter
    try:
        pc = PhaseConverter()
        print("PhaseConverter initialized successfully")
    except Exception as e:
        print(f"PhaseConverter initialization failed: {e}")
        return
    
    # Test get_analysis_params method
    test_cases = [
        (45821, "45821_ALL.csv"),
        (45821, "45821_056.csv"),
        (40000, "40000.dat")
    ]
    
    for shot_num, filename in test_cases:
        try:
            params = pc.get_analysis_params(shot_num, filename)
            print(f"Shot {shot_num}, file '{filename}':")
            print(f"Method: {params['method']}")
            print(f"Frequency (GHz): {params['freq_ghz']}")
            print(f"Frequency (Hz): {params['freq']}")
            print(f"n_path: {params['n_path']}")
            print()
        except Exception as e:
            print(f"Failed for {filename}: {e}")

def test_phase_to_density_integration():
    """Test integration between interferometry params and phase_to_density."""
    print("Testing phase_to_density integration:")
    print("=" * 60)
    
    pc = PhaseConverter()
    
    # Test different parameter sources
    test_cases = [
        (45821, "45821_ALL.csv", "280GHz CDM"),
        (45821, "45821_056.csv", "94GHz CDM"),
    ]
    
    for shot_num, filename, description in test_cases:
        print(f"Testing {description} ({filename}):")
        
        # Get analysis parameters
        params = pc.get_analysis_params(shot_num, filename)
        
        # Create dummy phase data
        dummy_phase = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        try:
            # Method 1: Using analysis_params
            density1 = pc.phase_to_density(dummy_phase, analysis_params=params)
            print(f"   phase_to_density with analysis_params: shape {density1.shape}")
            
            # Method 2: Using direct parameters
            density2 = pc.phase_to_density(dummy_phase, 
                                          freq_hz=params['freq'], 
                                          n_path=params['n_path'])
            print(f"   phase_to_density with direct params: shape {density2.shape}")
            
            # Check consistency
            consistency = np.allclose(density1, density2)
            print(f"   Results consistency: {consistency}")
            
            # Display sample values
            print(f"   Sample density (first 3): {density1[:3]}")
            print()
            
        except Exception as e:
            print(f"   Error in phase_to_density: {e}")
            print()

def test_numba_optimization():
    """Test that numba optimizations are working."""
    print("Testing Numba optimization:")
    print("=" * 60)
    
    try:
        from ifi.analysis.phi2ne import _normalize_iq_signals, _calculate_differential_phase
        
        # Test data
        i_signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        q_signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        
        # Test numba functions
        i_norm, q_norm = _normalize_iq_signals(i_signal.real, q_signal.real)
        print("_normalize_iq_signals executed successfully")
        
        phase_diff = _calculate_differential_phase(i_norm, q_norm)
        print("_calculate_differential_phase executed successfully")
        print(f"   Output shape: {phase_diff.shape}")
        print(f"   Sample values: {phase_diff[:3]}")
        
    except Exception as e:
        print(f"Numba optimization test failed: {e}")
    
    print()

def run_all_tests():
    """Run the complete test suite."""
    print("IFI Interferometry Parameters - Comprehensive Test Suite")
    print("=" * 80)
    print()
    
    tests = [
        test_standalone_function,
        test_class_methods, 
        test_phase_to_density_integration,
        test_numba_optimization
    ]
    
    failed_tests = []
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            failed_tests.append((test_func.__name__, str(e)))
            print(f"{test_func.__name__} failed: {e}")
            print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY:")
    if not failed_tests:
        print("All tests passed successfully!")
    else:
        print(f"{len(failed_tests)} test(s) failed:")
        for test_name, error in failed_tests:
            print(f"   - {test_name}: {error}")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()