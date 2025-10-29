#!/usr/bin/env python3
"""
Dask Performance Benchmark Script
================================

This script benchmarks the performance of Dask parallel processing
for the IFI analysis pipeline. It compares different schedulers and
measures processing times for various file counts.

Usage:
    python benchmark_dask_performance.py [options]

Options:
    --files N          Number of test files to process (default: 10)
    --scheduler TYPE   Scheduler type: 'threads', 'processes', 'synchronous' (default: 'threads')
    --iterations N     Number of benchmark iterations (default: 3)
    --output FILE      Output file for results (default: benchmark_results.json)
"""

import time
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import dask
from dask import delayed
import matplotlib.pyplot as plt

from ifi.utils.common import LogManager

# Initialize logging
LogManager(level="INFO")

@delayed
def mock_file_processing(file_path: str, processing_time: float = 0.1) -> Dict[str, Any]:
    """
    Mock file processing function that simulates the load_and_process_file function.
    
    Args:
        file_path: Path to the file being processed
        processing_time: Simulated processing time in seconds
        
    Returns:
        Dictionary containing processing results
    """
    # Simulate file processing time
    time.sleep(processing_time)
    
    # Simulate data processing
    data_size = np.random.randint(1000, 10000)
    processed_data = np.random.randn(data_size)
    
    # Simulate STFT analysis
    stft_result = {
        'frequencies': np.linspace(0, 100, 100),
        'magnitude': np.random.randn(100, 50),
        'phase': np.random.randn(100, 50)
    }
    
    # Simulate CWT analysis
    cwt_result = {
        'scales': np.linspace(1, 50, 50),
        'coefficients': np.random.randn(50, 100) + 1j * np.random.randn(50, 100)
    }
    
    return {
        'file_path': file_path,
        'data_size': data_size,
        'processed_data': processed_data,
        'stft_result': stft_result,
        'cwt_result': cwt_result,
        'processing_time': processing_time
    }

def benchmark_scheduler(scheduler: str, num_files: int, iterations: int = 3) -> Dict[str, Any]:
    """
    Benchmark a specific scheduler with given parameters.
    
    Args:
        scheduler: Dask scheduler type
        num_files: Number of files to process
        iterations: Number of benchmark iterations
        
    Returns:
        Dictionary containing benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {scheduler} scheduler with {num_files} files")
    print(f"{'='*60}")
    
    results = {
        'scheduler': scheduler,
        'num_files': num_files,
        'iterations': iterations,
        'times': [],
        'throughput': [],
        'efficiency': []
    }
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}")
        
        # Create mock file paths
        file_paths = [f"test_file_{i:04d}.csv" for i in range(num_files)]
        
        # Create Dask tasks
        tasks = [mock_file_processing(fp, processing_time=0.1) for fp in file_paths]
        
        # Execute with timing
        start_time = time.time()
        results_data = dask.compute(*tasks, scheduler=scheduler)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = num_files / total_time
        efficiency = throughput / num_files  # Files per second per file
        
        results['times'].append(total_time)
        results['throughput'].append(throughput)
        results['efficiency'].append(efficiency)
        
        print(f"  Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} files/sec")
        print(f"  Efficiency: {efficiency:.4f}")
    
    # Calculate statistics
    results['mean_time'] = np.mean(results['times'])
    results['std_time'] = np.std(results['times'])
    results['mean_throughput'] = np.mean(results['throughput'])
    results['std_throughput'] = np.std(results['throughput'])
    results['mean_efficiency'] = np.mean(results['efficiency'])
    results['std_efficiency'] = np.std(results['efficiency'])
    
    print(f"\nResults for {scheduler}:")
    print(f"  Mean time: {results['mean_time']:.2f} ± {results['std_time']:.2f}s")
    print(f"  Mean throughput: {results['mean_throughput']:.2f} ± {results['std_throughput']:.2f} files/sec")
    print(f"  Mean efficiency: {results['mean_efficiency']:.4f} ± {results['std_efficiency']:.4f}")
    
    return results

def create_performance_plot(results: List[Dict[str, Any]], output_file: str = "benchmark_plot.png"):
    """
    Create a performance comparison plot.
    
    Args:
        results: List of benchmark results
        output_file: Output file for the plot
    """
    schedulers = [r['scheduler'] for r in results]
    mean_times = [r['mean_time'] for r in results]
    std_times = [r['std_time'] for r in results]
    mean_throughput = [r['mean_throughput'] for r in results]
    std_throughput = [r['std_throughput'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Processing time plot
    ax1.bar(schedulers, mean_times, yerr=std_times, capsize=5, alpha=0.7)
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Processing Time Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Throughput plot
    ax2.bar(schedulers, mean_throughput, yerr=std_throughput, capsize=5, alpha=0.7)
    ax2.set_ylabel('Throughput (files/sec)')
    ax2.set_title('Throughput Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {output_file}")

def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Dask Performance Benchmark')
    parser.add_argument('--files', type=int, default=10, help='Number of test files')
    parser.add_argument('--scheduler', type=str, default='threads', 
                       choices=['threads', 'processes', 'synchronous'],
                       help='Scheduler type')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    parser.add_argument('--output', type=str, default='benchmark_results.json', 
                       help='Output file for results')
    parser.add_argument('--plot', type=str, default='benchmark_plot.png',
                       help='Output file for performance plot')
    parser.add_argument('--all-schedulers', action='store_true',
                       help='Test all schedulers')
    
    args = parser.parse_args()
    
    print("Dask Performance Benchmark")
    print("=" * 50)
    print(f"Files: {args.files}")
    print(f"Iterations: {args.iterations}")
    print(f"Output: {args.output}")
    
    if args.all_schedulers:
        # Test all schedulers
        schedulers = ['threads', 'processes', 'synchronous']
        all_results = []
        
        for scheduler in schedulers:
            result = benchmark_scheduler(scheduler, args.files, args.iterations)
            all_results.append(result)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create comparison plot
        create_performance_plot(all_results, args.plot)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for result in all_results:
            print(f"{result['scheduler']:12} | {result['mean_time']:6.2f}s | {result['mean_throughput']:8.2f} files/sec")
        
    else:
        # Test single scheduler
        result = benchmark_scheduler(args.scheduler, args.files, args.iterations)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump([result], f, indent=2)
        
        print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
