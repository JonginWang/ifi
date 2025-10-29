#!/usr/bin/env python3
"""
Phase Detection Algorithm Benchmark Script
========================================

This script benchmarks the performance of phase change detection algorithms
in the IFI analysis pipeline. It measures processing times for different
signal lengths and algorithm methods.

Usage:
    python benchmark_phase_detection.py [options]

Options:
    --signal-length N    Signal length in samples (default: 10000)
    --methods LIST       Comma-separated list of methods to test (default: all)
    --iterations N       Number of benchmark iterations (default: 5)
    --output FILE        Output file for results (default: phase_benchmark_results.json)
"""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ifi.utils.common import LogManager

# Initialize logging
LogManager(level="INFO")

# Import with error handling for conda environment issues
PHASE_ANALYSIS_AVAILABLE = False
PhaseChangeDetector = None

try:
    from ifi.analysis.phase_analysis import PhaseChangeDetector
    PHASE_ANALYSIS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Phase analysis module not available: {e}")
    print("This is likely due to NumPy version compatibility issues in conda environment.")
    print("Please run: conda install numpy --force-reinstall")
    print("Continuing with mock implementation...")

# Mock PhaseChangeDetector for testing when real module is not available
class MockPhaseChangeDetector:
    """Mock implementation for testing when phase analysis module is not available."""
    
    def __init__(self, fs: float):
        self.fs = fs
    
    def detect_phase_changes(self, ref_signal: np.ndarray, probe_signal: np.ndarray, 
                            f0: float = None, methods: list = None) -> dict:
        """Mock phase change detection."""
        if methods is None:
            methods = ['stacking', 'stft', 'cwt', 'cdm', 'cordic']
        
        results = {
            'fundamental_frequency': f0 or 50e6,
            'sampling_frequency': self.fs,
            'methods': {}
        }
        
        for method in methods:
            # Simulate processing time
            time.sleep(0.01)
            
            # Mock results
            phase_diff = np.random.randn(len(ref_signal)) * 0.1 + 0.5
            results['methods'][method] = {
                'phase_difference': phase_diff,
                'detected_frequency': f0 or 50e6,
                'times': np.arange(len(phase_diff)) / self.fs
            }
        
        return results

def generate_test_signals(signal_length: int, fs: float = 1e9, f0: float = 50e6, 
                         phase_change: float = 0.5, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate test signals with known phase change for benchmarking.
    
    Args:
        signal_length: Length of the signal in samples
        fs: Sampling frequency in Hz
        f0: Signal frequency in Hz
        phase_change: Phase change in radians
        noise_level: Noise level (standard deviation)
        
    Returns:
        Tuple of (time_axis, reference_signal, probe_signal)
    """
    t = np.arange(0, signal_length / fs, 1 / fs)
    
    # Reference signal
    ref_signal = np.sin(2 * np.pi * f0 * t)
    
    # Probe signal with phase change
    probe_signal = np.sin(2 * np.pi * f0 * t + phase_change)
    
    # Add noise
    ref_signal += noise_level * np.random.randn(len(ref_signal))
    probe_signal += noise_level * np.random.randn(len(probe_signal))
    
    return t, ref_signal, probe_signal

def benchmark_method(detector: PhaseChangeDetector, ref_signal: np.ndarray, 
                    probe_signal: np.ndarray, method: str, iterations: int = 5) -> Dict[str, Any]:
    """
    Benchmark a specific phase detection method.
    
    Args:
        detector: PhaseChangeDetector instance
        ref_signal: Reference signal
        probe_signal: Probe signal
        method: Method name to test
        iterations: Number of benchmark iterations
        
    Returns:
        Dictionary containing benchmark results
    """
    print(f"\nBenchmarking {method} method...")
    
    times = []
    accuracies = []
    
    for iteration in range(iterations):
        start_time = time.time()
        
        try:
            # Run phase detection
            results = detector.detect_phase_changes(ref_signal, probe_signal, methods=[method])
            
            end_time = time.time()
            processing_time = end_time - start_time
            times.append(processing_time)
            
            # Calculate accuracy (compare with known phase change)
            if method in results['methods'] and 'error' not in results['methods'][method]:
                phase_diff = results['methods'][method]['phase_difference']
                mean_phase = np.mean(phase_diff)
                accuracy = 1.0 - abs(mean_phase - 0.5) / 0.5  # 0.5 is the known phase change
                accuracies.append(max(0, accuracy))
            else:
                accuracies.append(0.0)
                
        except Exception as e:
            print(f"  Error in iteration {iteration + 1}: {e}")
            times.append(float('inf'))
            accuracies.append(0.0)
    
    # Calculate statistics
    valid_times = [t for t in times if t != float('inf')]
    valid_accuracies = [a for a in accuracies if a > 0]
    
    result = {
        'method': method,
        'iterations': iterations,
        'times': times,
        'accuracies': accuracies,
        'mean_time': np.mean(valid_times) if valid_times else float('inf'),
        'std_time': np.std(valid_times) if valid_times else 0.0,
        'mean_accuracy': np.mean(valid_accuracies) if valid_accuracies else 0.0,
        'std_accuracy': np.std(valid_accuracies) if valid_accuracies else 0.0,
        'success_rate': len(valid_times) / iterations,
        'throughput': 1.0 / np.mean(valid_times) if valid_times else 0.0
    }
    
    print(f"  Mean time: {result['mean_time']:.4f}s")
    print(f"  Mean accuracy: {result['mean_accuracy']:.4f}")
    print(f"  Success rate: {result['success_rate']:.2%}")
    print(f"  Throughput: {result['throughput']:.2f} signals/sec")
    
    return result

def benchmark_signal_lengths(detector: PhaseChangeDetector, methods: List[str], 
                           signal_lengths: List[int], iterations: int = 3) -> List[Dict[str, Any]]:
    """
    Benchmark different signal lengths.
    
    Args:
        detector: PhaseChangeDetector instance
        methods: List of methods to test
        signal_lengths: List of signal lengths to test
        iterations: Number of iterations per test
        
    Returns:
        List of benchmark results
    """
    results = []
    
    for signal_length in signal_lengths:
        print(f"\n{'='*60}")
        print(f"Testing signal length: {signal_length}")
        print(f"{'='*60}")
        
        # Generate test signals
        t, ref_signal, probe_signal = generate_test_signals(signal_length)
        
        for method in methods:
            result = benchmark_method(detector, ref_signal, probe_signal, method, iterations)
            result['signal_length'] = signal_length
            results.append(result)
    
    return results

def create_performance_plots(results: List[Dict[str, Any]], output_dir: str = "benchmark_plots"):
    """
    Create performance comparison plots.
    
    Args:
        results: List of benchmark results
        output_dir: Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Group results by method
    methods = list(set(r['method'] for r in results))
    signal_lengths = sorted(list(set(r['signal_length'] for r in results)))
    
    # Processing time vs signal length
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        times = [r['mean_time'] for r in method_results]
        std_times = [r['std_time'] for r in method_results]
        
        ax1.errorbar(signal_lengths, times, yerr=std_times, label=method, marker='o')
    
    ax1.set_xlabel('Signal Length (samples)')
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Processing Time vs Signal Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Accuracy vs signal length
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        accuracies = [r['mean_accuracy'] for r in method_results]
        std_accuracies = [r['std_accuracy'] for r in method_results]
        
        ax2.errorbar(signal_lengths, accuracies, yerr=std_accuracies, label=method, marker='o')
    
    ax2.set_xlabel('Signal Length (samples)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Signal Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    
    # Method comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    mean_times = [np.mean([r['mean_time'] for r in results if r['method'] == method]) for method in methods]
    mean_accuracies = [np.mean([r['mean_accuracy'] for r in results if r['method'] == method]) for method in methods]
    
    ax1.bar(methods, mean_times, alpha=0.7)
    ax1.set_ylabel('Mean Processing Time (seconds)')
    ax1.set_title('Method Performance Comparison')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(methods, mean_accuracies, alpha=0.7)
    ax2.set_ylabel('Mean Accuracy')
    ax2.set_title('Method Accuracy Comparison')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'method_comparison.png', dpi=300, bbox_inches='tight')
    
    print(f"\nPerformance plots saved to: {output_dir}")

def main():
    """Main benchmark function."""
    # Use mock implementation if real module is not available
    if not PHASE_ANALYSIS_AVAILABLE:
        print("Using mock implementation due to dependency issues.")
        print("Results will be simulated for demonstration purposes.")
        global PhaseChangeDetector
        PhaseChangeDetector = MockPhaseChangeDetector
    
    parser = argparse.ArgumentParser(description='Phase Detection Algorithm Benchmark')
    parser.add_argument('--signal-length', type=int, default=10000, help='Signal length in samples')
    parser.add_argument('--methods', type=str, default='all', 
                       help='Comma-separated list of methods to test')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
    parser.add_argument('--output', type=str, default='phase_benchmark_results.json', 
                       help='Output file for results')
    parser.add_argument('--plot-dir', type=str, default='benchmark_plots',
                       help='Output directory for plots')
    parser.add_argument('--signal-lengths', type=str, default='1000,5000,10000,20000,50000',
                       help='Comma-separated list of signal lengths to test')
    parser.add_argument('--fs', type=float, default=1e9, help='Sampling frequency in Hz')
    parser.add_argument('--f0', type=float, default=50e6, help='Signal frequency in Hz')
    
    args = parser.parse_args()
    
    # Parse methods
    if args.methods == 'all':
        methods = ['stacking', 'stft', 'cwt', 'cdm', 'cordic']
    else:
        methods = [m.strip() for m in args.methods.split(',')]
    
    # Parse signal lengths
    signal_lengths = [int(s.strip()) for s in args.signal_lengths.split(',')]
    
    print("Phase Detection Algorithm Benchmark")
    print("=" * 50)
    print(f"Methods: {methods}")
    print(f"Signal lengths: {signal_lengths}")
    print(f"Iterations: {args.iterations}")
    print(f"Output: {args.output}")
    
    # Initialize detector
    detector = PhaseChangeDetector(fs=args.fs)
    
    # Run benchmarks
    results = benchmark_signal_lengths(detector, methods, signal_lengths, args.iterations)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    create_performance_plots(results, args.plot_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<12} | {'Mean Time':<10} | {'Mean Accuracy':<12} | {'Success Rate':<12}")
    print("-" * 60)
    
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        if method_results:
            mean_time = np.mean([r['mean_time'] for r in method_results])
            mean_accuracy = np.mean([r['mean_accuracy'] for r in method_results])
            success_rate = np.mean([r['success_rate'] for r in method_results])
            print(f"{method:<12} | {mean_time:<10.4f} | {mean_accuracy:<12.4f} | {success_rate:<12.2%}")
    
    print(f"\nResults saved to: {args.output}")
    print(f"Plots saved to: {args.plot_dir}")

if __name__ == '__main__':
    main()
