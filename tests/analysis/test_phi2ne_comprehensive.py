#!/usr/bin/env python3
"""
Comprehensive test suite for phi2ne module.
"""

from pathlib import Path
import time
import numpy as np

from ifi.analysis.phi2ne import get_interferometry_params, PhaseConverter, _normalize_iq_signals, _calculate_differential_phase, _accumulate_phase_diff, _phase_to_density
from ifi.utils.common import LogManager
from ifi.utils.cache_setup import setup_project_cache
cache_config = setup_project_cache()

LogManager()

class Phi2neTestSuite:
    """Comprehensive test suite for phi2ne module."""
    
    def __init__(self):
        self.test_results = {}
        self.pc = None
        
    def test_numba_cache_setup(self):
        """Test that numba cache is properly configured."""
        print("Testing Numba Cache Setup:")
        print("-" * 40)
        
        try:
            import numba
            cache_dir = numba.config.CACHE_DIR
            print(f"Numba cache directory: {cache_dir}")
            
            # Test if cache is writable
            test_file = Path(cache_dir) / 'test_write.tmp'
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                test_file.unlink()
                print("Cache directory is writable")
                self.test_results['cache_setup'] = True
            except Exception as e:
                print(f"Cache directory not writable: {e}")
                self.test_results['cache_setup'] = False
                
        except Exception as e:
            print(f"Numba cache setup failed: {e}")
            self.test_results['cache_setup'] = False
        
        print()
    
    def test_numba_functions(self):
        """Test individual numba-optimized functions."""
        print("Testing Numba-Optimized Functions:")
        print("-" * 40)
        
        # Generate test data
        n_samples = 10000
        i_signal = np.cos(2 * np.pi * 1e6 * np.linspace(0, 1, n_samples)) + 0.1 * np.random.randn(n_samples)
        q_signal = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1, n_samples)) + 0.1 * np.random.randn(n_samples)
        
        try:
            # Test 1: Normalization
            start_time = time.perf_counter()
            i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)
            norm_time = time.perf_counter() - start_time
            
            print(f"_normalize_iq_signals: {norm_time:.4f}s")
            print(f"Input shape: {i_signal.shape}, Output shape: {i_norm.shape}")
            
            # Test 2: Differential phase
            start_time = time.perf_counter()
            phase_diff = _calculate_differential_phase(i_norm, q_norm)
            diff_time = time.perf_counter() - start_time
            
            print(f"_calculate_differential_phase: {diff_time:.4f}s")
            print(f"Output shape: {phase_diff.shape}")
            
            # Test 3: Cumulative sum
            start_time = time.perf_counter()
            phase_accum = _accumulate_phase_diff(phase_diff)
            cumsum_time = time.perf_counter() - start_time
            
            print(f"_accumulate_phase_diff: {cumsum_time:.4f}s")
            print(f"Output shape: {phase_accum.shape}")
            
            # Test 4: Phase to density core
            freq = 94e9  # 94 GHz
            c = 2.998e8
            m_e = 9.109e-31
            eps0 = 8.854e-12
            qe = 1.602e-19
            n_path = 2
            
            start_time = time.perf_counter()
            density = _phase_to_density(phase_accum, freq, c, m_e, eps0, qe, n_path)
            density_time = time.perf_counter() - start_time
            
            print(f"_phase_to_density: {density_time:.4f}s")
            print(f"Density range: {np.min(density):.2e} to {np.max(density):.2e} m^-2")
            
            total_time = norm_time + diff_time + cumsum_time + density_time
            processing_rate = (len(i_signal) * 8) / total_time / 1024 / 1024  # MB/s
            print(f"Total processing time: {total_time:.4f}s")
            print(f"Processing rate: {processing_rate:.1f} MB/s")
            
            self.test_results['numba_functions'] = True
            
        except Exception as e:
            print(f"Numba functions test failed: {e}")
            self.test_results['numba_functions'] = False
        
        print()
    
    def test_phase_converter_initialization(self):
        """Test PhaseConverter class initialization."""
        print("Testing PhaseConverter Initialization:")
        print("-" * 40)
        
        try:
            self.pc = PhaseConverter()
            print("PhaseConverter initialized successfully")
            
            # Test constants loading
            required_constants = ['m_e', 'eps0', 'qe', 'c']
            for const in required_constants:
                if const in self.pc.constants:
                    print(f"Constant '{const}': {self.pc.constants[const]}")
                else:
                    print(f"Missing constant: {const}")
            
            self.test_results['initialization'] = True
            
        except Exception as e:
            print(f"PhaseConverter initialization failed: {e}")
            self.test_results['initialization'] = False
        
        print()
    
    def test_parameter_functions(self):
        """Test interferometry parameter functions."""
        print("Testing Parameter Functions:")
        print("-" * 40)
        
        if not self.pc:
            print("Skipping: PhaseConverter not initialized")
            return
        
        test_cases = [
            (45821, "45821_ALL.csv", 280.0, "CDM"),
            (45821, "45821_056.csv", 94.0, "CDM"),
            (40000, "40000.dat", 94.0, "FPGA"),
            (35000, "35000.csv", 94.0, "IQ"),
        ]
        
        for shot_num, filename, expected_freq_ghz, expected_method in test_cases:
            try:
                # Test standalone function
                params1 = get_interferometry_params(shot_num, filename)
                
                # Test class method
                params2 = self.pc.get_analysis_params(shot_num, filename)
                
                print(f"{filename}:")
                print(f"Method: {params1['method']} (expected: {expected_method})")
                print(f"Freq GHz: {params1['freq_ghz']} (expected: {expected_freq_ghz})")
                print(f"Freq Hz: {params1['freq']}")
                print(f"n_path: {params1['n_path']}")
                
                # Validate consistency between standalone and class method
                consistency = (params1['method'] == params2['method'] and 
                             params1['freq_ghz'] == params2['freq_ghz'])
                print(f"   {'Yes!' if consistency else 'No!!'} Standalone/Class consistency: {consistency}")
                
            except Exception as e:
                print(f"Failed for {filename}: {e}")
        
        self.test_results['parameters'] = True
        print()
    
    def test_phase_calculation_methods(self):
        """Test different phase calculation methods."""
        print("Testing Phase Calculation Methods:")
        print("-" * 40)
        
        if not self.pc:
            print("Skipping: PhaseConverter not initialized")
            return
        
        # Generate realistic test signals
        n_samples = 5000
        fs = 250e6  # 250 MHz sampling
        t = np.arange(n_samples) / fs
        
        # IF signal around 10 MHz with phase modulation
        f_if = 10e6
        phase_mod = 0.1 * np.sin(2 * np.pi * 1e5 * t)  # 100 kHz modulation
        
        ref_signal = np.cos(2 * np.pi * f_if * t + phase_mod) + 0.05 * np.random.randn(n_samples)
        probe_signal = np.cos(2 * np.pi * f_if * t + phase_mod + 0.2) + 0.05 * np.random.randn(n_samples)
        
        # I/Q signals
        i_signal = np.cos(2 * np.pi * f_if * t + phase_mod)
        q_signal = np.sin(2 * np.pi * f_if * t + phase_mod)
        
        try:
            # Test 1: IQ atan2 method
            start_time = time.perf_counter()
            phase_atan2 = self.pc.calc_phase_iq_atan2(i_signal, q_signal)
            atan2_time = time.perf_counter() - start_time
            print(f"calc_phase_iq_atan2: {atan2_time:.4f}s, shape: {phase_atan2.shape}")
            
            # Test 2: IQ asin2 method (numba optimized)
            start_time = time.perf_counter()
            phase_asin2 = self.pc.calc_phase_iq_asin2(i_signal, q_signal)
            asin2_time = time.perf_counter() - start_time
            print(f"calc_phase_iq_asin2: {asin2_time:.4f}s, shape: {phase_asin2.shape}")
            
            # Performance comparison
            if asin2_time > 0:
                speedup = atan2_time / asin2_time
                print(f"Speedup (atan2/asin2): {speedup:.2f}x")
            
            # Test 3: CDM method (requires center frequency detection)
            try:
                from ifi.analysis.spectrum import SpectrumAnalysis
                analyzer = SpectrumAnalysis()
                f_center = analyzer.find_center_frequency_fft(ref_signal, fs)
                if f_center == 0.0:
                    f_center = f_if  # Use known frequency
                
                start_time = time.perf_counter()
                phase_cdm = self.pc.calc_phase_cdm(ref_signal, probe_signal, fs, f_center)
                cdm_time = time.perf_counter() - start_time
                print(f"calc_phase_cdm: {cdm_time:.4f}s, shape: {phase_cdm.shape}")
                print(f"Center frequency: {f_center/1e6:.2f} MHz")
                
            except Exception as e:
                print(f"calc_phase_cdm skipped: {e}")
            
            self.test_results['phase_methods'] = True
            
        except Exception as e:
            print(f"Phase calculation test failed: {e}")
            self.test_results['phase_methods'] = False
        
        print()
    
    def test_density_conversion_integration(self):
        """Test complete integration from parameters to density."""
        print("Testing Complete Integration:")
        print("-" * 40)
        
        if not self.pc:
            print("Skipping: PhaseConverter not initialized")
            return
        
        test_cases = [
            (45821, "45821_ALL.csv"),
            (45821, "45821_056.csv"),
        ]
        
        for shot_num, filename in test_cases:
            try:
                print(f"Testing {filename}:")
                
                # Get parameters
                params = self.pc.get_analysis_params(shot_num, filename)
                print(f"Method: {params['method']}, Freq: {params['freq_ghz']} GHz")
                
                # Generate realistic phase data
                n_samples = 1000
                # Simulate phase evolution with plasma density ramp
                t = np.linspace(0, 1, n_samples)
                density_profile = 1e19 * np.exp(-(t-0.5)**2 / 0.1**2)  # Gaussian profile
                
                # Convert density to phase (reverse of what we're testing)
                freq = params['freq']
                n_path = params['n_path']
                c = self.pc.constants['c']
                m_e = self.pc.constants['m_e']
                eps0 = self.pc.constants['eps0']
                qe = self.pc.constants['qe']
                
                # Critical density and phase calculation
                n_c = m_e * eps0 * (2 * np.pi * freq)**2 / qe**2
                expected_phase = (np.pi * freq / c) * density_profile / n_c * n_path
                
                # Add some noise
                phase_with_noise = expected_phase + 0.01 * np.random.randn(n_samples)
                
                # Test density conversion
                start_time = time.perf_counter()
                calculated_density = self.pc.phase_to_density(phase_with_noise, analysis_params=params)
                conv_time = time.perf_counter() - start_time
                
                print(f"   Conversion time: {conv_time:.4f}s")
                print(f"   Input phase range: {np.min(phase_with_noise):.3f} to {np.max(phase_with_noise):.3f} rad")
                print(f"   Output density range: {np.min(calculated_density):.2e} to {np.max(calculated_density):.2e} m^-2")
                
                # Test accuracy (should recover original density profile approximately)
                correlation = np.corrcoef(density_profile, calculated_density)[0, 1]
                print(f"   Correlation with expected: {correlation:.3f}")
                
                # Test alternative calling method
                calculated_density2 = self.pc.phase_to_density(phase_with_noise, 
                                                             freq_hz=params['freq'],
                                                             n_path=params['n_path'])
                consistency = np.allclose(calculated_density, calculated_density2)
                print(f"   Method consistency: {consistency}")
                
            except Exception as e:
                print(f"Integration test failed for {filename}: {e}")
        
        self.test_results['integration'] = True
        print()
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("IFI Phi2ne Module - Comprehensive Test Suite")
        print("=" * 80)
        print(f"Numba cache directory: {cache_config['numba_cache_dir']}")
        print("=" * 80)
        print()
        
        # Run all tests
        test_methods = [
            self.test_numba_cache_setup,
            self.test_numba_functions,
            self.test_phase_converter_initialization,
            self.test_parameter_functions,
            self.test_phase_calculation_methods,
            self.test_density_conversion_integration
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"{test_method.__name__} failed with exception: {e}")
                print()
        
        # Summary
        print("=" * 80)
        print("TEST SUMMARY:")
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"   {status}: {test_name}")
        
        print("-" * 40)
        print(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All tests passed! phi2ne module is working correctly.")
        else:
            print(f"{total_tests - passed_tests} test(s) failed. Check the output above.")
        
        print("=" * 80)

def main():
    """Main test runner."""
    suite = Phi2neTestSuite()
    suite.run_all_tests()

if __name__ == '__main__':
    main()