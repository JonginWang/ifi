#!/usr/bin/env python3
"""
Simple test for unified phase detection
"""

import numpy as np

from ifi.utils.common import LogManager
from ifi.analysis.phase_analysis import PhaseChangeDetector

# Initialize logging
LogManager(level="INFO")

def test_simple_unified():
    """Test simple unified phase detection."""
    
    print("Simple Unified Phase Detection Test")
    print("=" * 40)
    
    # Initialize detector
    fs = 1e9  # 1 GHz sampling
    f0 = 50e6  # 50 MHz signal
    detector = PhaseChangeDetector(fs)
    
    # Generate simple test signals
    signal_length = 1000
    t = np.arange(signal_length) / fs
    
    # Constant phase change
    ref_signal = np.sin(2 * np.pi * f0 * t)
    probe_signal = np.sin(2 * np.pi * f0 * t + 0.5)  # 0.5 rad constant phase change
    
    print("Testing constant phase change (0.5 rad)...")
    
    try:
        # Test unified detection
        result = detector.detect_phase_changes_unified(ref_signal, probe_signal, f0, ['stacking'])
        
        if 'error' not in result['unified_analysis']['stacking']:
            analysis = result['unified_analysis']['stacking']['analysis']
            change_type = analysis['change_classification']
            constant_phase = analysis['constant_phase']['mean']
            linear_slope = analysis['linear_phase']['slope_rad_per_sec']
            r_squared = analysis['linear_phase']['r_squared']
            
            print(f"  Detected type: {change_type}")
            print(f"  Constant phase: {constant_phase:.3f} rad")
            print(f"  Linear slope: {linear_slope:.3f} rad/s")
            print(f"  R² (linear): {r_squared:.3f}")
            print("  ??Unified detection successful!")
        else:
            print(f"  ??Error: {result['unified_analysis']['stacking']['error']}")
            
    except Exception as e:
        print(f"  ??Exception: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_simple_unified()

