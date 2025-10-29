#!/usr/bin/env python3
"""
Phase Analysis Profiling Script
==============================

This script profiles the phase analysis algorithms to identify bottlenecks
and optimization opportunities.

Usage:
    python profile_phase_analysis.py [options]

Options:
    --signal-length N    Signal length in samples (default: 10000)
    --methods LIST       Comma-separated list of methods to profile (default: all)
    --iterations N       Number of profiling iterations (default: 3)
    --output FILE        Output file for profiling results (default: profile_results.json)
"""

import time
import json
import argparse
import cProfile
import pstats
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ifi.utils.common import LogManager
from ifi.analysis.phase_analysis import PhaseChangeDetector

# Initialize logging
LogManager(level="INFO")

def generate_test_signals(signal_length: int, fs: float = 1e9, f0: float = 50e6, 
                         phase_change: float = 0.5, noise_level: float = 0.1,
                         linear_phase_drift: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test signals with known phase change for profiling.
    
    Args:
        signal_length: Length of the signal in samples
        fs: Sampling frequency in Hz
        f0: Signal frequency in Hz
        phase_change: Constant phase change in radians
        noise_level: Noise level (standard deviation)
        linear_phase_drift: Linear phase drift rate (radians per second)
        
    Returns:
        Tuple of (time_axis, reference_signal, probe_signal)
    """
    # Time axis
    t = np.arange(signal_length) / fs
    
    # Reference signal (clean)
    ref_signal = np.sin(2 * np.pi * f0 * t)
    
    # Probe signal with phase change and optional linear drift
    phase_change_total = phase_change + linear_phase_drift * t
    probe_signal = np.sin(2 * np.pi * f0 * t + phase_change_total)
    
    # Add noise
    if noise_level > 0:
        ref_signal += np.random.normal(0, noise_level, signal_length)
        probe_signal += np.random.normal(0, noise_level, signal_length)
    
    return t, ref_signal, probe_signal

def profile_method(detector: PhaseChangeDetector, ref_signal: np.ndarray, 
                   probe_signal: np.ndarray, f0: float, method: str, 
                   iterations: int = 3) -> Dict[str, Any]:
    """
    Profile a specific method for performance analysis.
    
    Args:
        detector: PhaseChangeDetector instance
        ref_signal: Reference signal
        probe_signal: Probe signal
        f0: Fundamental frequency
        method: Method to profile
        iterations: Number of iterations
        
    Returns:
        Profiling results
    """
    print(f"Profiling {method} method...")
    
    # Time profiling
    times = []
    for i in range(iterations):
        start_time = time.time()
        
        try:
            results = detector.detect_phase_changes(ref_signal, probe_signal, f0, [method])
            end_time = time.time()
            times.append(end_time - start_time)
            
            if 'error' in results['methods'][method]:
                print(f"  Error in {method}: {results['methods'][method]['error']}")
                return {'error': results['methods'][method]['error']}
                
        except Exception as e:
            print(f"  Exception in {method}: {e}")
            return {'error': str(e)}
    
    # Statistical analysis
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Memory profiling (approximate)
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run one more iteration for memory measurement
    detector.detect_phase_changes(ref_signal, probe_signal, f0, [method])
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before
    
    return {
        'method': method,
        'iterations': iterations,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'memory_used_mb': memory_used,
        'throughput': 1.0 / mean_time if mean_time > 0 else 0
    }

def profile_cpu_intensive_functions(detector: PhaseChangeDetector, ref_signal: np.ndarray, 
                                   probe_signal: np.ndarray, f0: float) -> Dict[str, Any]:
    """
    Profile CPU-intensive functions within the phase analysis.
    
    Args:
        detector: PhaseChangeDetector instance
        ref_signal: Reference signal
        probe_signal: Probe signal
        f0: Fundamental frequency
        
    Returns:
        CPU profiling results
    """
    print("Running CPU profiling...")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the main detection function
    profiler.enable()
    try:
        results = detector.detect_phase_changes(ref_signal, probe_signal, f0)
    except Exception as e:
        print(f"Error during profiling: {e}")
        return {'error': str(e)}
    finally:
        profiler.disable()
    
    # Get profiling statistics
    stats = pstats.Stats(profiler)
    
    # Extract top functions by cumulative time
    top_functions = []
    stats.sort_stats('cumulative')
    
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        if ct > 0.001:  # Only include functions taking > 1ms
            top_functions.append({
                'function': f"{func[0]}:{func[1]}:{func[2]}",
                'cumulative_time': ct,
                'total_time': tt,
                'call_count': cc,
                'primitive_calls': nc
            })
    
    # Sort by cumulative time
    top_functions.sort(key=lambda x: x['cumulative_time'], reverse=True)
    
    return {
        'top_functions': top_functions[:20],  # Top 20 functions
        'total_functions': len(stats.stats),
        'total_time': sum(f['cumulative_time'] for f in top_functions)
    }

def analyze_linear_phase_drift(detector: PhaseChangeDetector, signal_length: int = 10000,
                              fs: float = 1e9, f0: float = 50e6) -> Dict[str, Any]:
    """
    Analyze performance with linear phase drift.
    
    Args:
        detector: PhaseChangeDetector instance
        signal_length: Signal length
        fs: Sampling frequency
        f0: Fundamental frequency
        
    Returns:
        Linear phase drift analysis results
    """
    print("Analyzing linear phase drift performance...")
    
    drift_rates = [0.0, 0.1, 0.5, 1.0, 2.0]  # radians per second
    results = {}
    
    for drift_rate in drift_rates:
        print(f"  Testing drift rate: {drift_rate} rad/s")
        
        # Generate signals with linear drift
        t, ref_signal, probe_signal = generate_test_signals(
            signal_length, fs, f0, phase_change=0.5, 
            noise_level=0.1, linear_phase_drift=drift_rate
        )
        
        # Test each method
        method_results = {}
        for method in ['stacking', 'stft', 'cwt', 'cdm', 'cordic']:
            try:
                start_time = time.time()
                result = detector.detect_phase_changes(ref_signal, probe_signal, f0, [method])
                end_time = time.time()
                
                if 'error' not in result['methods'][method]:
                    phase_diff = result['methods'][method]['phase_difference']
                    
                    # Analyze phase drift detection
                    if len(phase_diff) > 1:
                        # Calculate linear fit to phase difference
                        x = np.arange(len(phase_diff))
                        coeffs = np.polyfit(x, phase_diff, 1)
                        linear_slope = coeffs[0]
                        r_squared = 1 - (np.sum((phase_diff - (coeffs[0] * x + coeffs[1]))**2) / 
                                        np.sum((phase_diff - np.mean(phase_diff))**2))
                        
                        method_results[method] = {
                            'processing_time': end_time - start_time,
                            'detected_slope': linear_slope,
                            'true_slope': drift_rate,
                            'slope_error': abs(linear_slope - drift_rate),
                            'r_squared': r_squared,
                            'phase_std': np.std(phase_diff)
                        }
                    else:
                        method_results[method] = {'error': 'Insufficient data points'}
                else:
                    method_results[method] = {'error': result['methods'][method]['error']}
                    
            except Exception as e:
                method_results[method] = {'error': str(e)}
        
        results[f'drift_{drift_rate}'] = method_results
    
    return results

def main():
    """Main profiling function."""
    parser = argparse.ArgumentParser(description='Phase Analysis Profiling')
    parser.add_argument('--signal-length', type=int, default=10000, help='Signal length in samples')
    parser.add_argument('--methods', type=str, default='all', 
                       help='Comma-separated list of methods to profile')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    parser.add_argument('--output', type=str, default='profile_results.json', 
                       help='Output file for results')
    parser.add_argument('--fs', type=float, default=1e9, help='Sampling frequency in Hz')
    parser.add_argument('--f0', type=float, default=50e6, help='Signal frequency in Hz')
    parser.add_argument('--linear-drift', action='store_true', 
                       help='Include linear phase drift analysis')
    
    args = parser.parse_args()
    
    # Parse methods
    if args.methods == 'all':
        methods = ['stacking', 'stft', 'cwt', 'cdm', 'cordic']
    else:
        methods = [m.strip() for m in args.methods.split(',')]
    
    print("Phase Analysis Profiling")
    print("=" * 50)
    print(f"Signal length: {args.signal_length}")
    print(f"Methods: {methods}")
    print(f"Iterations: {args.iterations}")
    print(f"Output: {args.output}")
    
    # Initialize detector
    detector = PhaseChangeDetector(args.fs)
    
    # Generate test signals
    print("\nGenerating test signals...")
    t, ref_signal, probe_signal = generate_test_signals(
        args.signal_length, args.fs, args.f0, 
        phase_change=0.5, noise_level=0.1
    )
    
    # Profile each method
    profiling_results = {}
    for method in methods:
        profiling_results[method] = profile_method(
            detector, ref_signal, probe_signal, args.f0, method, args.iterations
        )
    
    # CPU profiling
    print("\nRunning CPU profiling...")
    cpu_results = profile_cpu_intensive_functions(detector, ref_signal, probe_signal, args.f0)
    
    # Linear phase drift analysis
    linear_drift_results = {}
    if args.linear_drift:
        print("\nRunning linear phase drift analysis...")
        linear_drift_results = analyze_linear_phase_drift(
            detector, args.signal_length, args.fs, args.f0
        )
    
    # Compile results
    results = {
        'parameters': {
            'signal_length': args.signal_length,
            'sampling_frequency': args.fs,
            'signal_frequency': args.f0,
            'methods_tested': methods,
            'iterations': args.iterations
        },
        'profiling_results': profiling_results,
        'cpu_profiling': cpu_results,
        'linear_drift_analysis': linear_drift_results
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PROFILING SUMMARY")
    print("=" * 50)
    
    for method, result in profiling_results.items():
        if 'error' not in result:
            print(f"{method:12s}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s "
                  f"({result['throughput']:.2f} signals/sec)")
        else:
            print(f"{method:12s}: ERROR - {result['error']}")
    
    if cpu_results and 'top_functions' in cpu_results:
        print(f"\nTop CPU-intensive functions:")
        for i, func in enumerate(cpu_results['top_functions'][:5]):
            print(f"  {i+1}. {func['function']}: {func['cumulative_time']:.4f}s")

if __name__ == "__main__":
    main()
