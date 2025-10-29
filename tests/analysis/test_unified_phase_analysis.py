#!/usr/bin/env python3
"""
Unified Phase Analysis Test
==========================

Test script for the unified phase change detection algorithm that can handle
constant, linear, and nonlinear phase changes using a single method.
"""

import numpy as np
import matplotlib.pyplot as plt

from ifi.utils.common import LogManager
from ifi.analysis.phase_analysis import PhaseChangeDetector

# Initialize logging
LogManager(level="INFO")

def generate_test_signals(signal_length: int, fs: float = 1e9, f0: float = 50e6, 
                         phase_change_type: str = 'constant', phase_params: dict = None) -> tuple:
    """
    Generate test signals with different types of phase changes.
    
    Args:
        signal_length: Length of the signal in samples
        fs: Sampling frequency in Hz
        f0: Signal frequency in Hz
        phase_change_type: Type of phase change ('constant', 'linear', 'quadratic', 'cubic')
        phase_params: Parameters for phase change
        
    Returns:
        Tuple of (time_axis, reference_signal, probe_signal, true_phase_change)
    """
    if phase_params is None:
        phase_params = {}
    
    # Time axis
    t = np.arange(signal_length) / fs
    
    # Reference signal (clean)
    ref_signal = np.sin(2 * np.pi * f0 * t)
    
    # Generate phase change based on type
    if phase_change_type == 'constant':
        phase_offset = phase_params.get('offset', 0.5)
        phase_change = np.full_like(t, phase_offset)
        
    elif phase_change_type == 'linear':
        slope = phase_params.get('slope', 0.1)  # rad/s
        offset = phase_params.get('offset', 0.0)
        phase_change = slope * t + offset
        
    elif phase_change_type == 'quadratic':
        a = phase_params.get('a', 0.01)  # rad/s^2
        b = phase_params.get('b', 0.1)   # rad/s
        c = phase_params.get('c', 0.0)   # rad
        phase_change = a * t**2 + b * t + c
        
    elif phase_change_type == 'cubic':
        a = phase_params.get('a', 0.001)  # rad/s^3
        b = phase_params.get('b', 0.01)  # rad/s^2
        c = phase_params.get('c', 0.1)   # rad/s
        d = phase_params.get('d', 0.0)   # rad
        phase_change = a * t**3 + b * t**2 + c * t + d
        
    else:
        raise ValueError(f"Unknown phase change type: {phase_change_type}")
    
    # Probe signal with phase change
    probe_signal = np.sin(2 * np.pi * f0 * t + phase_change)
    
    # Add noise
    noise_level = phase_params.get('noise', 0.05)
    if noise_level > 0:
        ref_signal += np.random.normal(0, noise_level, signal_length)
        probe_signal += np.random.normal(0, noise_level, signal_length)
    
    return t, ref_signal, probe_signal, phase_change

def test_unified_phase_detection():
    """Test unified phase change detection with different change types."""
    
    print("Unified Phase Change Detection Test")
    print("=" * 50)
    
    # Initialize detector
    fs = 1e9  # 1 GHz sampling
    f0 = 50e6  # 50 MHz signal
    detector = PhaseChangeDetector(fs)
    
    # Test parameters
    signal_length = 2000
    test_cases = [
        {
            'type': 'constant',
            'params': {'offset': 0.5, 'noise': 0.05},
            'description': 'Constant phase change (0.5 rad)'
        },
        {
            'type': 'linear',
            'params': {'slope': 0.2, 'offset': 0.1, 'noise': 0.05},
            'description': 'Linear phase change (0.2 rad/s slope)'
        },
        {
            'type': 'quadratic',
            'params': {'a': 0.01, 'b': 0.1, 'c': 0.0, 'noise': 0.05},
            'description': 'Quadratic phase change'
        },
        {
            'type': 'cubic',
            'params': {'a': 0.001, 'b': 0.01, 'c': 0.1, 'd': 0.0, 'noise': 0.05},
            'description': 'Cubic phase change'
        }
    ]
    
    results = {}
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['description']}")
        print("-" * 40)
        
        # Generate test signals
        t, ref_signal, probe_signal, true_phase_change = generate_test_signals(
            signal_length, fs, f0, test_case['type'], test_case['params']
        )
        
        # Test unified detection with different methods
        methods = ['stacking', 'cdm', 'cordic']
        method_results = {}
        
        for method in methods:
            try:
                print(f"  Testing {method} method...")
                
                # Use unified detection
                result = detector.detect_phase_changes_unified(
                    ref_signal, probe_signal, f0, [method])
                
                if 'error' not in result['unified_analysis'][method]:
                    analysis = result['unified_analysis'][method]['analysis']
                    
                    # Extract key metrics
                    change_type = analysis['change_classification']
                    constant_phase = analysis['constant_phase']['mean']
                    linear_slope = analysis['linear_phase']['slope_rad_per_sec']
                    r_squared_linear = analysis['linear_phase']['r_squared']
                    r_squared_quad = analysis['nonlinear_phase']['r_squared_quadratic']
                    snr_db = analysis['signal_quality']['snr_db']
                    
                    print(f"    Detected type: {change_type}")
                    print(f"    Constant phase: {constant_phase:.3f} rad")
                    print(f"    Linear slope: {linear_slope:.3f} rad/s")
                    print(f"    R² (linear): {r_squared_linear:.3f}")
                    print(f"    R² (quadratic): {r_squared_quad:.3f}")
                    print(f"    SNR: {snr_db:.1f} dB")
                    
                    method_results[method] = {
                        'change_type': change_type,
                        'constant_phase': constant_phase,
                        'linear_slope': linear_slope,
                        'r_squared_linear': r_squared_linear,
                        'r_squared_quad': r_squared_quad,
                        'snr_db': snr_db,
                        'analysis': analysis
                    }
                else:
                    print(f"    Error: {result['unified_analysis'][method]['error']}")
                    method_results[method] = {'error': result['unified_analysis'][method]['error']}
                    
            except Exception as e:
                print(f"    Exception: {e}")
                method_results[method] = {'error': str(e)}
        
        results[test_case['type']] = {
            'description': test_case['description'],
            'true_phase_change': true_phase_change,
            'methods': method_results
        }
    
    return results

def plot_results(results: dict):
    """Plot the results of unified phase detection."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    test_types = list(results.keys())
    
    for i, test_type in enumerate(test_types):
        if i >= 4:
            break
            
        ax = axes[i]
        test_result = results[test_type]
        
        # Plot true phase change
        t = np.arange(len(test_result['true_phase_change'])) / 1e9
        ax.plot(t, test_result['true_phase_change'], 'k-', linewidth=2, label='True phase change')
        
        # Plot detected phase changes for each method
        colors = ['red', 'blue', 'green']
        for j, (method, method_result) in enumerate(test_result['methods'].items()):
            if 'error' not in method_result:
                analysis = method_result['analysis']
                phase_diff = analysis['phase_difference']
                times = analysis['time_points']
                
                ax.plot(times, phase_diff, color=colors[j], alpha=0.7, 
                       label=f'{method} (detected)')
        
        ax.set_title(f'{test_result["description"]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unified_phase_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main test function."""
    print("Starting Unified Phase Analysis Test...")
    
    # Run tests
    results = test_unified_phase_detection()
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for test_type, test_result in results.items():
        print(f"\n{test_result['description']}:")
        for method, method_result in test_result['methods'].items():
            if 'error' not in method_result:
                print(f"  {method:8s}: {method_result['change_type']:15s} "
                      f"(R²={method_result['r_squared_linear']:.3f}, "
                      f"SNR={method_result['snr_db']:.1f}dB)")
            else:
                print(f"  {method:8s}: ERROR - {method_result['error']}")
    
    # Plot results
    try:
        plot_results(results)
        print("\nResults plotted and saved to 'unified_phase_analysis_results.png'")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nUnified phase analysis test completed!")

if __name__ == "__main__":
    main()

