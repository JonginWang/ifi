#!/usr/bin/env python3
"""
Test Linear Phase Drift Analysis
================================

Simple test script to verify linear phase drift analysis functionality.
"""

import numpy as np

from ifi.utils.common import LogManager
from ifi.analysis.phase_analysis import PhaseChangeDetector

# Initialize logging
LogManager(level="INFO")

def test_linear_drift_analysis():
    """Test linear phase drift analysis with different drift rates."""
    
    print("Testing Linear Phase Drift Analysis")
    print("=" * 40)
    
    # Initialize detector
    fs = 1e9  # 1 GHz sampling
    f0 = 50e6  # 50 MHz signal
    detector = PhaseChangeDetector(fs)
    
    # Test parameters
    signal_length = 1000
    t = np.arange(signal_length) / fs
    
    # Test different drift rates
    drift_rates = [0.0, 0.1, 0.5, 1.0, 2.0]  # rad/s
    
    for drift_rate in drift_rates:
        print(f"\nTesting drift rate: {drift_rate} rad/s")
        
        # Generate signals with linear drift
        ref_signal = np.sin(2 * np.pi * f0 * t)
        probe_signal = np.sin(2 * np.pi * f0 * t + 0.5 + drift_rate * t)
        
        try:
            # Test stacking method
            result = detector.analyze_linear_phase_drift(ref_signal, probe_signal, f0, ['stacking'])
            
            if 'error' not in result['linear_drift_analysis']['stacking']:
                analysis = result['linear_drift_analysis']['stacking']
                detected_slope = analysis['linear_slope_rad_per_sec']
                r_squared = analysis['r_squared_linear']
                
                print(f"  Detected slope: {detected_slope:.3f} rad/s")
                print(f"  True slope: {drift_rate:.3f} rad/s")
                print(f"  Error: {abs(detected_slope - drift_rate):.3f} rad/s")
                print(f"  R-squared: {r_squared:.3f}")
            else:
                print(f"  Error: {result['linear_drift_analysis']['stacking']['error']}")
                
        except Exception as e:
            print(f"  Exception: {e}")
    
    print("\nLinear drift analysis test completed!")

if __name__ == "__main__":
    test_linear_drift_analysis()

