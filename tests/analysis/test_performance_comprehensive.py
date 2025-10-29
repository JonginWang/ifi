#!/usr/bin/env python3
"""
Performance and Parallel Processing Test Suite
=============================================

Comprehensive performance testing for IFI package.
Tests Dask integration, caching, and overall system performance.
"""

import sys
import unittest
import time
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from ifi.utils.common import LogManager
from ifi.analysis import processing, spectrum

# Initialize logging
LogManager(level="INFO")

class TestDaskPerformance(unittest.TestCase):
    """Test Dask parallel processing performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = self.generate_test_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def generate_test_data(self, num_files=8, signal_length=10000):
        """Generate test data for performance testing."""
        data = {}
        for i in range(num_files):
            t = np.linspace(0, 10, signal_length)  # 10 seconds, 10000 samples
            signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
            data[f"file_{i:03d}.csv"] = pd.DataFrame({
                'TIME': t,
                'SIGNAL': signal
            })
        return data
    
    def test_dask_scheduler_comparison(self):
        """Test different Dask schedulers."""
        import dask
        import dask.delayed
        
        @dask.delayed
        def process_signal(signal, fs):
            """Simulate signal processing."""
            time.sleep(0.1)  # Simulate processing time
            return np.fft.fft(signal)
        
        # Test parameters
        num_tasks = 8
        signal_length = 1000
        
        # Generate test signals
        signals = [np.random.randn(signal_length) for _ in range(num_tasks)]
        fs = 1000
        
        # Test different schedulers
        schedulers = ['threads', 'processes', 'single-threaded']
        results = {}
        
        for scheduler in schedulers:
            print(f"\nTesting {scheduler} scheduler...")
            
            # Create delayed tasks
            tasks = [process_signal(signal, fs) for signal in signals]
            
            # Measure execution time
            start_time = time.time()
            try:
                computed_results = dask.compute(*tasks, scheduler=scheduler)
                end_time = time.time()
                
                total_time = end_time - start_time
                expected_time = num_tasks * 0.1  # 0.1s per task
                speedup = expected_time / total_time if total_time > 0 else 0
                
                results[scheduler] = {
                    'total_time': total_time,
                    'expected_time': expected_time,
                    'speedup': speedup,
                    'success': True
                }
                
                print(f"  Total time: {total_time:.3f}s")
                print(f"  Expected time: {expected_time:.3f}s")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Results: {len(computed_results)} tasks completed")
                
            except Exception as e:
                results[scheduler] = {
                    'total_time': float('inf'),
                    'expected_time': expected_time,
                    'speedup': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"  Error: {e}")
        
        # Verify results
        self.assertGreater(len(results), 0)
        
        # Check that at least one scheduler worked
        successful_schedulers = [s for s, r in results.items() if r['success']]
        self.assertGreater(len(successful_schedulers), 0, "No schedulers worked successfully")
    
    def test_dask_with_real_processing(self):
        """Test Dask with real signal processing operations."""
        import dask
        import dask.delayed
        
        @dask.delayed
        def process_file_data(file_name, df):
            """Process a single file's data."""
            # Simulate processing steps
            time.sleep(0.05)  # Simulate I/O time
            
            # Refine data
            df_refined = processing.refine_data(df)
            
            # Remove offset
            df_processed = processing.remove_offset(df_refined, window_size=100)
            
            # Compute STFT
            signal = df_processed['SIGNAL'].values
            fs = 1000
            analyzer = spectrum.SpectrumAnalysis()
            freq, time_stft, stft_matrix = analyzer.compute_stft(signal, fs)
            
            return {
                'file_name': file_name,
                'processed_shape': df_processed.shape,
                'stft_shape': stft_matrix.shape,
                'mean_signal': signal.mean(),
                'std_signal': signal.std()
            }
        
        # Test with multiple files
        num_files = 6
        tasks = []
        
        for i, (file_name, df) in enumerate(list(self.test_data.items())[:num_files]):
            task = process_file_data(file_name, df)
            tasks.append(task)
        
        print(f"\nProcessing {num_files} files with Dask...")
        
        start_time = time.time()
        results = dask.compute(*tasks, scheduler='threads')
        end_time = time.time()
        
        total_time = end_time - start_time
        expected_time = num_files * 0.05  # 0.05s per file
        
        print(f"Total processing time: {total_time:.3f}s")
        print(f"Expected time: {expected_time:.3f}s")
        print(f"Speedup: {expected_time / total_time:.2f}x")
        print(f"Files processed: {len(results)}")
        
        # Verify results
        self.assertEqual(len(results), num_files)
        
        for result in results:
            self.assertIn('file_name', result)
            self.assertIn('processed_shape', result)
            self.assertIn('stft_shape', result)
            self.assertIsInstance(result['mean_signal'], (int, float))
            self.assertIsInstance(result['std_signal'], (int, float))
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        import dask
        import dask.delayed
        
        @dask.delayed
        def process_large_signal(signal_length):
            """Process a large signal."""
            # Generate large signal
            t = np.linspace(0, 1, signal_length)
            signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
            
            # Process signal
            df = pd.DataFrame({'TIME': t, 'SIGNAL': signal})
            df_refined = processing.refine_data(df)
            df_processed = processing.remove_offset(df_refined, window_size=100)
            
            return {
                'signal_length': signal_length,
                'processed_length': len(df_processed),
                'memory_usage': signal.nbytes
            }
        
        # Test with different signal sizes
        signal_lengths = [1000, 5000, 10000]
        tasks = [process_large_signal(length) for length in signal_lengths]
        
        print(f"\nTesting memory efficiency with signal lengths: {signal_lengths}")
        
        start_time = time.time()
        results = dask.compute(*tasks, scheduler='threads')
        end_time = time.time()
        
        total_time = end_time - start_time
        print(f"Total processing time: {total_time:.3f}s")
        
        # Verify results
        self.assertEqual(len(results), len(signal_lengths))
        
        for i, result in enumerate(results):
            expected_length = signal_lengths[i]
            self.assertEqual(result['signal_length'], expected_length)
            self.assertGreater(result['processed_length'], 0)
            self.assertGreater(result['memory_usage'], 0)

class TestCachingPerformance(unittest.TestCase):
    """Test caching system performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_hdf5_caching_performance(self):
        """Test HDF5 caching performance."""
        from ifi.utils.file_io import save_results_to_hdf5
        
        # Generate test data
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        data = pd.DataFrame({'TIME': t, 'SIGNAL': signal})
        
        # Test saving performance
        cache_file = self.cache_dir / "1.h5"  # This matches the shot_num=1
        
        start_time = time.time()
        # Create a non-empty signals dictionary
        signals_dict = {'test_signal': data}
        
        save_results_to_hdf5(
            output_dir=str(cache_file.parent),
            shot_num=1,
            signals=signals_dict,
            stft_results={},
            cwt_results={},
            density_data=pd.DataFrame(),
            vest_data=pd.DataFrame()
        )
        save_time = time.time() - start_time
        
        print(f"HDF5 save time: {save_time:.4f}s")
        
        # Test loading performance
        import h5py
        start_time = time.time()
        with h5py.File(cache_file, 'r') as f:
            # Check if the file was created and has the expected structure
            self.assertIn('metadata', f.keys())
            loaded_metadata = dict(f.attrs)
        load_time = time.time() - start_time
        
        print(f"HDF5 load time: {load_time:.4f}s")
        
        # Verify file was created
        self.assertTrue(cache_file.exists())
        
        # Performance assertions
        self.assertLess(save_time, 1.0, "HDF5 save should be fast")
        self.assertLess(load_time, 0.5, "HDF5 load should be very fast")
    
    def test_cache_hit_performance(self):
        """Test cache hit performance."""
        # This would test the actual caching system in NAS_DB
        # For now, we'll simulate the concept
        
        cache_file = self.cache_dir / "999.h5"  # This matches the shot_num=999
        
        # First access (cache miss)
        start_time = time.time()
        # Simulate data processing
        time.sleep(0.1)  # Simulate processing time
        # Save to cache
        from ifi.utils.file_io import save_results_to_hdf5
        data = {'test_signal': pd.DataFrame({'TIME': [1, 2, 3], 'CH0': [0.1, 0.2, 0.3]})}
        save_results_to_hdf5(
            output_dir=str(cache_file.parent),
            shot_num=999,
            signals=data,
            stft_results={},
            cwt_results={},
            density_data=pd.DataFrame(),
            vest_data=pd.DataFrame()
        )
        first_access_time = time.time() - start_time
        
        # Second access (cache hit)
        start_time = time.time()
        # Load from cache
        import h5py
        with h5py.File(cache_file, 'r') as f:
            loaded_data = dict(f.attrs)
        second_access_time = time.time() - start_time
        
        print(f"First access (cache miss): {first_access_time:.4f}s")
        print(f"Second access (cache hit): {second_access_time:.4f}s")
        print(f"Speedup: {first_access_time / second_access_time:.2f}x")
        
        # Verify cache hit is much faster
        self.assertLess(second_access_time, first_access_time / 2, 
                       "Cache hit should be significantly faster")

class TestSystemPerformance(unittest.TestCase):
    """Test overall system performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analysis_pipeline_performance(self):
        """Test complete analysis pipeline performance."""
        # Generate test data (longer signal for STFT)
        fs = 1000
        t = np.linspace(0, 10, 10 * fs)  # 10 seconds, 10000 samples
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        df = pd.DataFrame({'TIME': t, 'SIGNAL': signal})
        
        # Test complete pipeline
        start_time = time.time()
        
        # Step 1: Data refinement
        df_refined = processing.refine_data(df)
        refine_time = time.time() - start_time
        
        # Step 2: Offset removal
        start_time = time.time()
        df_processed = processing.remove_offset(df_refined, window_size=100)
        offset_time = time.time() - start_time
        
        # Step 3: STFT analysis
        start_time = time.time()
        analyzer = spectrum.SpectrumAnalysis()
        freq, time_stft, stft_matrix = analyzer.compute_stft(
            df_processed['SIGNAL'].values, fs
        )
        stft_time = time.time() - start_time
        
        total_time = refine_time + offset_time + stft_time
        
        print("\nAnalysis Pipeline Performance:")
        print(f"  Data refinement: {refine_time:.4f}s")
        print(f"  Offset removal: {offset_time:.4f}s")
        print(f"  STFT analysis: {stft_time:.4f}s")
        print(f"  Total time: {total_time:.4f}s")
        
        # Performance assertions
        self.assertLess(total_time, 5.0, "Complete pipeline should be fast")
        self.assertGreater(stft_matrix.shape[0], 0)
        self.assertGreater(stft_matrix.shape[1], 0)
    
    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large dataset
        large_signal_length = 50000
        t = np.linspace(0, 1, large_signal_length)
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        df = pd.DataFrame({'TIME': t, 'SIGNAL': signal})
        
        # Process large dataset
        df_refined = processing.refine_data(df)
        df_processed = processing.remove_offset(df_refined, window_size=1000)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print("\nMemory Usage Test:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Signal length: {large_signal_length}")
        
        # Memory usage should be reasonable
        self.assertLess(memory_increase, 100, "Memory usage should be reasonable")
        self.assertEqual(len(df_processed), large_signal_length)

def run_performance_tests():
    """Run all performance tests."""
    print("Running Performance Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDaskPerformance,
        TestCachingPerformance,
        TestSystemPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_performance_tests()
    sys.exit(0 if success else 1)
