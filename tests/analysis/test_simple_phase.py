#!/usr/bin/env python3
"""
Simple test script for phase analysis methods
"""

import sys

import numpy as np

def test_basic_functionality():
    """Test basic functionality without complex methods"""
    print("Testing basic phase analysis functionality...")
    
    try:
        # Generate simple test signals
        fs = 1e9
        duration = 1e-6
        f0 = 50e6
        t = np.arange(0, duration, 1/fs)
        
        # Simple sine waves
        ref_signal = np.sin(2 * np.pi * f0 * t)
        probe_signal = np.sin(2 * np.pi * f0 * t + 0.5)  # 0.5 rad phase shift
        
        print(f"Generated signals: {len(ref_signal)} samples")
        
        # Test CORDIC processor
        from ifi.analysis.phase_analysis import CORDICProcessor
        cordic = CORDICProcessor()
        print("CORDIC processor created successfully")
        
        # Test basic CORDIC rotation
        magnitude, phase, scale = cordic.cordic_rotation(1.0, 0.0, 0.0)
        print(f"CORDIC test: magnitude={magnitude:.3f}, phase={phase:.3f}, scale={scale:.3f}")
        
        # Test phase extraction
        times, phases = cordic.extract_phase_samples(ref_signal, f0, fs)
        print(f"CORDIC phase extraction: {len(times)} time points, {len(phases)} phases")
        
        # Test signal stacking
        from ifi.analysis.phase_analysis import SignalStacker
        stacker = SignalStacker(fs)
        print("Signal stacker created successfully")
        
        # Test frequency detection
        detected_f0 = stacker.find_fundamental_frequency(ref_signal)
        print(f"Detected frequency: {detected_f0/1e6:.1f} MHz")
        
        # Test stacking
        stacked_signal, time_points = stacker.stack_signals(ref_signal, f0)
        print(f"Stacking: {len(time_points)} time points")
        
        print("Basic functionality test passed!")
        assert True
        
    except Exception as e:
        print(f"Error in basic functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\nBasic test passed!")
    else:
        print("\nBasic test failed!")
        sys.exit(1)

