#!/usr/bin/env python3
"""
Performance test suite for IFI package components.
"""

import os
import sys
from pathlib import Path
import time
import numpy as np

from ifi.analysis.phi2ne import PhaseConverter, _normalize_iq_signals, _calculate_differential_phase, _accumulate_phase_diff, _phase_to_density
from ifi.utils.cache_setup import setup_project_cache

cache_config = setup_project_cache()

class PerformanceTester:
    """Comprehensive performance testing suite."""
    
    def __init__(self):
        self.results = {}
        
    def generate_test_data(self, n_samples=1000000):
        """Generate realistic I/Q signals for testing."""
        print(f"Generating test data: {n_samples:,} samples")
        
        # Create realistic interferometry-like signals
        t = np.linspace(0, 1, n_samples)
        freq1, freq2 = 5e6, 10e6  # 5MHz and 10MHz components
        
        # Add phase modulation and noise
        phase_shift = 0.5 * np.sin(2 * np.pi * 1e5 * t)  # 100kHz phase modulation
        i_signal = np.cos(2 * np.pi * freq1 * t + phase_shift) + 0.1 * np.random.randn(n_samples)
        q_signal = np.sin(2 * np.pi * freq1 * t + phase_shift) + 0.1 * np.random.randn(n_samples)
        
        return i_signal, q_signal
    
    def benchmark_function(self, func, *args, name="Function", n_runs=3):
        """Benchmark a function over multiple runs."""
        times = []
        
        print(f"\nBenchmarking {name}:")
        
        # Warm-up run (important for numba JIT)
        print("   - Warming up (JIT compilation)...")
        result = func(*args)
        
        # Actual benchmark runs
        for i in range(n_runs):
            start_time = time.perf_counter()
            result = func(*args)
            end_time = time.perf_counter()
            
            run_time = end_time - start_time
            times.append(run_time)
            print(f"   - Run {i+1}: {run_time:.4f} seconds")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Average: {avg_time:.4f} ± {std_time:.4f} seconds")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}")
        
        return avg_time, result
    
    def test_cache_configuration(self):
        """Test numba cache configuration."""
        print("\nTesting Cache Configuration")
        print("=" * 40)
        
        try:
            import numba
            print(f"Numba version: {numba.__version__}")
            print(f"Cache directory: {numba.config.CACHE_DIR}")
            print(f"Threading layer: {os.environ.get('NUMBA_THREADING_LAYER', 'default')}")
            
            # Test if cache directory is writable
            test_file = Path(numba.config.CACHE_DIR) / 'test_write.tmp'
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                test_file.unlink()
                print("Cache directory is writable")
            except Exception as e:
                print(f"Cache directory not writable: {e}")
                
            assert True
            
        except Exception as e:
            print(f"Cache configuration error: {e}")
            return False
    
    def test_numba_functions(self):
        """Test individual numba-optimized functions."""
        print("\nTesting Numba-Optimized Functions")
        print("=" * 40)
        
        # Generate test data
        i_signal, q_signal = self.generate_test_data(n_samples=500000)
        
        results = {}
        
        # Test 1: Normalization
        time_norm, (i_norm, q_norm) = self.benchmark_function(
            _normalize_iq_signals, i_signal, q_signal,
            name="IQ Normalization", n_runs=3
        )
        results['normalization'] = time_norm
        
        # Test 2: Differential phase calculation
        time_diff, phase_diff = self.benchmark_function(
            _calculate_differential_phase, i_norm, q_norm,
            name="Differential Phase", n_runs=3
        )
        results['differential_phase'] = time_diff
        
        # Test 3: Cumulative sum
        time_cumsum, phase_accum = self.benchmark_function(
            _accumulate_phase_diff, phase_diff,
            name="Cumulative Sum", n_runs=3
        )
        results['cumulative_sum'] = time_cumsum
        
        # Test 4: Phase to density conversion
        # Use realistic physical constants
        c = 2.998e8          # speed of light
        m_e = 9.109e-31      # electron mass
        eps0 = 8.854e-12     # permittivity of free space
        qe = 1.602e-19       # elementary charge
        freq = 94e9          # 94 GHz
        n_path = 2           # number of passes
        
        phase_test = np.random.randn(len(phase_accum)) * 0.1
        time_density, density = self.benchmark_function(
            _phase_to_density, phase_test, freq, c, m_e, eps0, qe, n_path,
            name="Phase to Density", n_runs=3
        )
        results['phase_to_density'] = time_density
        
        # Overall performance metrics
        total_time = sum(results.values())
        data_size_mb = len(i_signal) * 16 / 1024 / 1024  # 2 arrays * 8 bytes
        processing_rate = data_size_mb / total_time
        
        print("\nPerformance Summary:")
        print(f"Data processed: {data_size_mb:.1f} MB")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Processing rate: {processing_rate:.1f} MB/s")
        
        self.results['numba_functions'] = results
        self.results['processing_rate_mb_s'] = processing_rate
        
        return results
    
    def test_phase_converter_methods(self):
        """Test PhaseConverter class methods."""
        print("\nTesting PhaseConverter Methods")
        print("=" * 40)
        
        # Generate test data
        i_signal, q_signal = self.generate_test_data(n_samples=200000)
        
        # Initialize phase converter
        phase_converter = PhaseConverter()
        
        results = {}
        
        # Test calc_phase_iq_asin2 (numba-optimized)
        time_asin2, phase_asin2 = self.benchmark_function(
            phase_converter.calc_phase_iq_asin2, 
            i_signal, q_signal, False,
            name="calc_phase_iq_asin2 (Numba Optimized)", 
            n_runs=3
        )
        results['asin2_optimized'] = time_asin2
        
        # Test calc_phase_iq_atan2 (not optimized)
        time_atan2, phase_atan2 = self.benchmark_function(
            phase_converter.calc_phase_iq_atan2, 
            i_signal, q_signal, False,
            name="calc_phase_iq_atan2 (Original)",
            n_runs=3
        )
        results['atan2_original'] = time_atan2
        
        # Test phase_to_density (numba-optimized)
        dummy_phase = np.random.randn(len(phase_asin2)) * 0.1
        time_density, density = self.benchmark_function(
            phase_converter.phase_to_density,
            dummy_phase, 94,  # 94 GHz
            name="phase_to_density (Numba Optimized)",
            n_runs=3
        )
        results['density_conversion'] = time_density
        
        # Performance comparison
        speedup = time_atan2 / time_asin2
        
        print("\nMethod Comparison:")
        print(f"asin2 (optimized): {time_asin2:.4f} seconds")
        print(f"atan2 (original):  {time_atan2:.4f} seconds")
        print(f"Speedup factor:    {speedup:.2f}x")
        
        self.results['phase_converter'] = results
        self.results['speedup_factor'] = speedup
        
        return results
    
    def test_jit_compilation_effect(self):
        """Test the effect of JIT compilation over multiple runs."""
        print("\nTesting JIT Compilation Effect")
        print("=" * 40)
        
        # Generate smaller dataset for multiple runs
        i_signal, q_signal = self.generate_test_data(n_samples=100000)
        
        print("Running full pipeline 10 times to measure JIT effect:")
        
        times = []
        for i in range(10):
            start = time.perf_counter()
            
            # Full pipeline
            i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)
            phase_diff = _calculate_differential_phase(i_norm, q_norm)
            phase_accum = _accumulate_phase_diff(phase_diff)
            
            end = time.perf_counter()
            run_time = end - start
            times.append(run_time)
            print(f"   Run {i+1:2d}: {run_time:.4f} seconds")
        
        # Analysis
        first_run = times[0]
        steady_state = np.mean(times[2:])  # Skip first 2 runs
        jit_speedup = first_run / steady_state
        
        print("\nJIT Analysis:")
        print(f"First run (cold):     {first_run:.4f} seconds")
        print(f"Steady state (warm):  {steady_state:.4f} seconds")
        print(f"JIT speedup:          {jit_speedup:.1f}x")
        
        self.results['jit_speedup'] = jit_speedup
        
        return {'first_run': first_run, 'steady_state': steady_state, 'jit_speedup': jit_speedup}
    
    def run_full_test_suite(self):
        """Run the complete performance test suite."""
        print("IFI Comprehensive Performance Test Suite")
        print("=" * 60)
        print(f"Cache directory: {cache_config['cache_dir']}")
        
        success = True
        
        # Test 1: Cache configuration
        if not self.test_cache_configuration():
            success = False
        
        # Test 2: Individual numba functions
        try:
            self.test_numba_functions()
        except Exception as e:
            print(f"Numba functions test failed: {e}")
            success = False
        
        # Test 3: PhaseConverter methods
        try:
            self.test_phase_converter_methods()
        except Exception as e:
            print(f"PhaseConverter test failed: {e}")
            success = False
        
        # Test 4: JIT compilation effect
        try:
            self.test_jit_compilation_effect()
        except Exception as e:
            print(f"JIT compilation test failed: {e}")
            success = False
        
        # Final summary
        print("\n" + "=" * 60)
        if success:
            print("All performance tests passed!")
            print(f"Peak processing rate: {self.results.get('processing_rate_mb_s', 0):.1f} MB/s")
            print(f"Numba speedup: {self.results.get('speedup_factor', 1):.1f}x")
            print(f"JIT speedup: {self.results.get('jit_speedup', 1):.1f}x")
        else:
            print("Some performance tests failed. Check the errors above.")
        
        return success, self.results

def main():
    """Main test runner."""
    tester = PerformanceTester()
    success, results = tester.run_full_test_suite()
    return 0 if success else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)