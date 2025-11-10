#!/usr/bin/env python3
"""
Test script for analysis error handling cases
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ifi.utils.common import LogManager

# Initialize logging
LogManager(level="INFO")
logger = LogManager().get_logger(__name__)

def test_phase_analysis_errors():
    """Test phase analysis error handling cases"""
    print("=" * 50)
    print("Testing Phase Analysis Error Cases")
    print("=" * 50)
    
    try:
        from ifi.analysis.phase_analysis import PhaseChangeDetector, SignalStacker
        
        # Test 1: Normal initialization
        print("\n1. Testing normal PhaseChangeDetector initialization...")
        detector = PhaseChangeDetector(fs=1000)
        print(f"   [OK] PhaseChangeDetector initialized successfully")
        
        # Test 2: Invalid sampling frequency
        print("\n2. Testing with invalid sampling frequency...")
        try:
            detector_invalid = PhaseChangeDetector(fs=0)
            print(f"   [WARN] PhaseChangeDetector with fs=0 succeeded")
        except Exception as e:
            print(f"   [OK] PhaseChangeDetector with fs=0 properly failed: {e}")
        
        # Test 3: Negative sampling frequency
        print("\n3. Testing with negative sampling frequency...")
        try:
            detector_negative = PhaseChangeDetector(fs=-1000)
            print(f"   [WARN] PhaseChangeDetector with negative fs succeeded")
        except Exception as e:
            print(f"   [OK] PhaseChangeDetector with negative fs properly failed: {e}")
        
        # Test 4: Empty signals
        print("\n4. Testing with empty signals...")
        try:
            empty_signal = np.array([])
            result = detector.detect_phase_changes_unified(
                empty_signal, empty_signal, 10.0, methods=['stacking']
            )
            print(f"   [WARN] Empty signals processing succeeded: {result}")
        except Exception as e:
            print(f"   [OK] Empty signals properly failed: {e}")
        
        # Test 5: Mismatched signal lengths
        print("\n5. Testing with mismatched signal lengths...")
        try:
            ref_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            probe_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 500))  # Different length
            result = detector.detect_phase_changes_unified(
                ref_signal, probe_signal, 10.0, methods=['stacking']
            )
            print(f"   [WARN] Mismatched signals processing succeeded: {result}")
        except Exception as e:
            print(f"   [OK] Mismatched signals properly failed: {e}")
        
        # Test 6: Invalid frequency
        print("\n6. Testing with invalid frequency...")
        try:
            ref_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            probe_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            result = detector.detect_phase_changes_unified(
                ref_signal, probe_signal, 0.0, methods=['stacking']
            )
            print(f"   [WARN] Invalid frequency processing succeeded: {result}")
        except Exception as e:
            print(f"   [OK] Invalid frequency properly failed: {e}")
        
        # Test 7: Negative frequency
        print("\n7. Testing with negative frequency...")
        try:
            ref_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            probe_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            result = detector.detect_phase_changes_unified(
                ref_signal, probe_signal, -10.0, methods=['stacking']
            )
            print(f"   [WARN] Negative frequency processing succeeded: {result}")
        except Exception as e:
            print(f"   [OK] Negative frequency properly failed: {e}")
        
        # Test 8: Invalid methods
        print("\n8. Testing with invalid methods...")
        try:
            ref_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            probe_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            result = detector.detect_phase_changes_unified(
                ref_signal, probe_signal, 10.0, methods=['invalid_method']
            )
            print(f"   [WARN] Invalid methods processing succeeded: {result}")
        except Exception as e:
            print(f"   [OK] Invalid methods properly failed: {e}")
        
        # Test 9: Empty methods list
        print("\n9. Testing with empty methods list...")
        try:
            ref_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            probe_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            result = detector.detect_phase_changes_unified(
                ref_signal, probe_signal, 10.0, methods=[]
            )
            print(f"   [WARN] Empty methods processing succeeded: {result}")
        except Exception as e:
            print(f"   [OK] Empty methods properly failed: {e}")
        
        # Test 10: None signals
        print("\n10. Testing with None signals...")
        try:
            result = detector.detect_phase_changes_unified(
                None, None, 10.0, methods=['stacking']
            )
            print(f"   [WARN] None signals processing succeeded: {result}")
        except Exception as e:
            print(f"   [OK] None signals properly failed: {e}")
            
    except Exception as e:
        print(f"   [ERROR] Critical error in phase analysis testing: {e}")
        logger.exception("Phase analysis critical error")

def test_spectrum_analysis_errors():
    """Test spectrum analysis error handling cases"""
    print("\n" + "=" * 50)
    print("Testing Spectrum Analysis Error Cases")
    print("=" * 50)
    
    try:
        from ifi.analysis.spectrum import SpectrumAnalysis
        
        # Test 1: Normal initialization
        print("\n1. Testing normal SpectrumAnalysis initialization...")
        analyzer = SpectrumAnalysis()
        print(f"   [OK] SpectrumAnalysis initialized successfully")
        
        # Test 2: Invalid parameters
        print("\n2. Testing with invalid parameters...")
        try:
            # Test with invalid window size
            result = analyzer.compute_stft(np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)), 
                                        window_size=0)
            print(f"   [WARN] Invalid window size processing succeeded")
        except Exception as e:
            print(f"   [OK] Invalid window size properly failed: {e}")
        
        # Test 3: Empty signal
        print("\n3. Testing with empty signal...")
        try:
            result = analyzer.compute_stft(np.array([]))
            print(f"   [WARN] Empty signal processing succeeded")
        except Exception as e:
            print(f"   [OK] Empty signal properly failed: {e}")
        
        # Test 4: Single sample signal
        print("\n4. Testing with single sample signal...")
        try:
            result = analyzer.compute_stft(np.array([1.0]))
            print(f"   [WARN] Single sample signal processing succeeded")
        except Exception as e:
            print(f"   [OK] Single sample signal properly failed: {e}")
        
        # Test 5: NaN values
        print("\n5. Testing with NaN values...")
        try:
            signal_with_nan = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            signal_with_nan[500] = np.nan
            result = analyzer.compute_stft(signal_with_nan)
            print(f"   [WARN] NaN values processing succeeded")
        except Exception as e:
            print(f"   [OK] NaN values properly failed: {e}")
        
        # Test 6: Inf values
        print("\n6. Testing with Inf values...")
        try:
            signal_with_inf = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            signal_with_inf[500] = np.inf
            result = analyzer.compute_stft(signal_with_inf)
            print(f"   [WARN] Inf values processing succeeded")
        except Exception as e:
            print(f"   [OK] Inf values properly failed: {e}")
            
    except Exception as e:
        print(f"   [ERROR] Critical error in spectrum analysis testing: {e}")
        logger.exception("Spectrum analysis critical error")

def test_plotting_errors():
    """Test plotting error handling cases"""
    print("\n" + "=" * 50)
    print("Testing Plotting Error Cases")
    print("=" * 50)
    
    try:
        from ifi.analysis.plots import Plotter
        
        # Test 1: Normal initialization
        print("\n1. Testing normal Plotter initialization...")
        plotter = Plotter()
        print(f"   [OK] Plotter initialized successfully")
        
        # Test 2: Empty DataFrame
        print("\n2. Testing with empty DataFrame...")
        try:
            empty_df = pd.DataFrame()
            fig = plotter.plot_waveforms(empty_df, title="Empty DataFrame")
            print(f"   [WARN] Empty DataFrame plotting succeeded")
        except Exception as e:
            print(f"   [OK] Empty DataFrame properly failed: {e}")
        
        # Test 3: DataFrame with no numeric columns
        print("\n3. Testing with DataFrame with no numeric columns...")
        try:
            non_numeric_df = pd.DataFrame({'text': ['a', 'b', 'c'], 'category': ['x', 'y', 'z']})
            fig = plotter.plot_waveforms(non_numeric_df, title="Non-numeric DataFrame")
            print(f"   [WARN] Non-numeric DataFrame plotting succeeded")
        except Exception as e:
            print(f"   [OK] Non-numeric DataFrame properly failed: {e}")
        
        # Test 4: DataFrame with all NaN values
        print("\n4. Testing with DataFrame with all NaN values...")
        try:
            nan_df = pd.DataFrame({'TIME': [1, 2, 3], 'SIGNAL': [np.nan, np.nan, np.nan]})
            fig = plotter.plot_waveforms(nan_df, title="All NaN DataFrame")
            print(f"   [WARN] All NaN DataFrame plotting succeeded")
        except Exception as e:
            print(f"   [OK] All NaN DataFrame properly failed: {e}")
        
        # Test 5: Invalid time-frequency data
        print("\n5. Testing with invalid time-frequency data...")
        try:
            # Mismatched dimensions
            time = np.linspace(0, 1, 100)
            freq = np.linspace(0, 50, 50)
            stft_data = np.random.randn(100, 100)  # Wrong shape
            fig = plotter.plot_time_frequency(time, freq, stft_data, method='stft', title="Invalid STFT")
            print(f"   [WARN] Invalid time-frequency data plotting succeeded")
        except Exception as e:
            print(f"   [OK] Invalid time-frequency data properly failed: {e}")
        
        # Test 6: None data
        print("\n6. Testing with None data...")
        try:
            fig = plotter.plot_waveforms(None, title="None data")
            print(f"   [WARN] None data plotting succeeded")
        except Exception as e:
            print(f"   [OK] None data properly failed: {e}")
            
    except Exception as e:
        print(f"   [ERROR] Critical error in plotting testing: {e}")
        logger.exception("Plotting critical error")

def test_main_analysis_errors():
    """Test main analysis error handling cases"""
    print("\n" + "=" * 50)
    print("Testing Main Analysis Error Cases")
    print("=" * 50)
    
    try:
        from ifi.analysis.main_analysis import run_analysis
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        # Test 1: Invalid arguments
        print("\n1. Testing with invalid arguments...")
        try:
            # Test with None arguments
            result = run_analysis(None)
            print(f"   [WARN] None arguments processing succeeded")
        except Exception as e:
            print(f"   [OK] None arguments properly failed: {e}")
        
        # Test 2: Invalid file paths
        print("\n2. Testing with invalid file paths...")
        try:
            # Create a mock args object with invalid paths
            class MockArgs:
                def __init__(self):
                    self.data_folders = ['/invalid/path']
                    self.add_path = False
                    self.force_remote = False
                    self.results_dir = str(temp_dir)
                    self.no_offset_removal = False
                    self.offset_window = 2001
                    self.stft = True
                    self.stft_cols = None
                    self.cwt = False
                    self.cwt_cols = None
                    self.scheduler = 'threads'
                    self.save_plots = False
                    self.save_data = False
            
            args = MockArgs()
            result = run_analysis(args)
            print(f"   [WARN] Invalid file paths processing succeeded")
        except Exception as e:
            print(f"   [OK] Invalid file paths properly failed: {e}")
        
        # Test 3: Empty data folders
        print("\n3. Testing with empty data folders...")
        try:
            class MockArgs:
                def __init__(self):
                    self.data_folders = []
                    self.add_path = False
                    self.force_remote = False
                    self.results_dir = str(temp_dir)
                    self.no_offset_removal = False
                    self.offset_window = 2001
                    self.stft = True
                    self.stft_cols = None
                    self.cwt = False
                    self.cwt_cols = None
                    self.scheduler = 'threads'
                    self.save_plots = False
                    self.save_data = False
            
            args = MockArgs()
            result = run_analysis(args)
            print(f"   [WARN] Empty data folders processing succeeded")
        except Exception as e:
            print(f"   [OK] Empty data folders properly failed: {e}")
        
        # Test 4: Invalid scheduler
        print("\n4. Testing with invalid scheduler...")
        try:
            class MockArgs:
                def __init__(self):
                    self.data_folders = None
                    self.add_path = False
                    self.force_remote = False
                    self.results_dir = str(temp_dir)
                    self.no_offset_removal = False
                    self.offset_window = 2001
                    self.stft = True
                    self.stft_cols = None
                    self.cwt = False
                    self.cwt_cols = None
                    self.scheduler = 'invalid_scheduler'
                    self.save_plots = False
                    self.save_data = False
            
            args = MockArgs()
            result = run_analysis(args)
            print(f"   [WARN] Invalid scheduler processing succeeded")
        except Exception as e:
            print(f"   [OK] Invalid scheduler properly failed: {e}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"   [OK] Cleanup completed")
        
    except Exception as e:
        print(f"   [ERROR] Critical error in main analysis testing: {e}")
        logger.exception("Main analysis critical error")

if __name__ == "__main__":
    print("Starting Analysis Error Handling Tests")
    print("=" * 60)
    
    # Test phase analysis error cases
    test_phase_analysis_errors()
    
    # Test spectrum analysis error cases
    test_spectrum_analysis_errors()
    
    # Test plotting error cases
    test_plotting_errors()
    
    # Test main analysis error cases
    test_main_analysis_errors()
    
    print("\n" + "=" * 60)
    print("Analysis Error Handling Tests Completed")

