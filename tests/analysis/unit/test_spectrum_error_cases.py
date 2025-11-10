#!/usr/bin/env python3
"""
Test Error Cases for Spectrum Analysis Module
============================================

This script tests error handling in the spectrum analysis module,
focusing on STFT and CWT operations with various edge cases.
"""

import os
import sys
import numpy as np
from pathlib import Path

from ifi.utils.common import LogManager
LogManager(level="INFO")
logger = LogManager().get_logger(__name__)

# Add the project root to sys.path for module imports
try:
    project_root = Path(__file__).resolve().parents[2]  # Adjust ifi to be 2 levels up
    if str(project_root) not in os.environ.get('PYTHONPATH', ''):
        os.environ['PYTHONPATH'] = f"{project_root};{os.environ.get('PYTHONPATH', '')}"
except Exception as e:
    logger.error(f"Error setting project root: {e}")
    # Fallback for IDEs or other environments
    current_dir = Path(__file__).parent
    if current_dir.name == 'test':  # if in ifi/test
        project_root = current_dir.parent
    elif current_dir.name == 'ifi':  # if in ifi
        project_root = current_dir
    else:  # Assume current_dir is project root
        project_root = current_dir
    
    if str(project_root) not in os.environ.get('PYTHONPATH', ''):
        os.environ['PYTHONPATH'] = f"{project_root};{os.environ.get('PYTHONPATH', '')}"

# Ensure the project root is in sys.path for immediate imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import modules from ifi
try:
    from ifi.analysis.spectrum import SpectrumAnalysis
    from ifi.utils.cache_setup import setup_project_cache
    cache_config = setup_project_cache()
except ImportError as e:
    logger.error(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    sys.exit(1)


def test_spectrum_analysis_errors():
    """Test SpectrumAnalysis error handling cases"""
    print("\n" + "=" * 50)
    print("Testing SpectrumAnalysis Error Cases")
    print("=" * 50)
    
    try:
        # Test 1: Normal initialization
        print("\n1. Testing normal SpectrumAnalysis initialization...")
        analyzer = SpectrumAnalysis()
        print(f"   [OK] SpectrumAnalysis initialized successfully")
        
        # Test 2: STFT with invalid parameters
        print("\n2. Testing STFT with invalid parameters...")
        try:
            # Empty signal
            empty_signal = np.array([])
            result = analyzer.compute_stft(empty_signal, fs=1000)
            print(f"   [WARN] Empty signal STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Empty signal STFT properly failed: {e}")
        
        try:
            # Single sample signal
            single_sample = np.array([1.0])
            result = analyzer.compute_stft(single_sample, fs=1000)
            print(f"   [WARN] Single sample STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Single sample STFT properly failed: {e}")
        
        try:
            # Invalid sampling frequency
            signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            result = analyzer.compute_stft(signal, fs=0)
            print(f"   [WARN] Invalid fs=0 STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Invalid fs=0 STFT properly failed: {e}")
        
        try:
            # Negative sampling frequency
            result = analyzer.compute_stft(signal, fs=-1000)
            print(f"   [WARN] Negative fs STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Negative fs STFT properly failed: {e}")
        
        # Test 3: STFT with NaN values
        print("\n3. Testing STFT with NaN values...")
        try:
            signal_nan = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            signal_nan[100:200] = np.nan
            result = analyzer.compute_stft(signal_nan, fs=1000)
            print(f"   [WARN] NaN signal STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] NaN signal STFT properly failed: {e}")
        
        # Test 4: STFT with Inf values
        print("\n4. Testing STFT with Inf values...")
        try:
            signal_inf = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            signal_inf[100:200] = np.inf
            result = analyzer.compute_stft(signal_inf, fs=1000)
            print(f"   [WARN] Inf signal STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Inf signal STFT properly failed: {e}")
        
        # Test 5: STFT with very large values
        print("\n5. Testing STFT with very large values...")
        try:
            signal_large = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)) * 1e10
            result = analyzer.compute_stft(signal_large, fs=1000)
            print(f"   [OK] Large values STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] Large values STFT failed: {e}")
        
        # Test 6: CWT with invalid parameters
        print("\n6. Testing CWT with invalid parameters...")
        try:
            # Empty signal
            result = analyzer.compute_cwt(empty_signal, fs=1000)
            print(f"   [WARN] Empty signal CWT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Empty signal CWT properly failed: {e}")
        
        try:
            # Single sample signal
            result = analyzer.compute_cwt(single_sample, fs=1000)
            print(f"   [WARN] Single sample CWT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Single sample CWT properly failed: {e}")
        
        # Test 7: CWT with NaN values
        print("\n7. Testing CWT with NaN values...")
        try:
            result = analyzer.compute_cwt(signal_nan, fs=1000)
            print(f"   [WARN] NaN signal CWT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] NaN signal CWT properly failed: {e}")
        
        # Test 8: CWT with Inf values
        print("\n8. Testing CWT with Inf values...")
        try:
            result = analyzer.compute_cwt(signal_inf, fs=1000)
            print(f"   [WARN] Inf signal CWT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Inf signal CWT properly failed: {e}")
        
        # Test 9: Normal operation with valid signal
        print("\n9. Testing normal operation with valid signal...")
        try:
            signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
            stft_result = analyzer.compute_stft(signal, fs=1000)
            cwt_result = analyzer.compute_cwt(signal, fs=1000)
            print(f"   [OK] Normal STFT operation: {type(stft_result)}")
            print(f"   [OK] Normal CWT operation: {type(cwt_result)}")
        except Exception as e:
            print(f"   [ERROR] Normal operation failed: {e}")
            logger.exception("Normal operation error")
        
        # Test 10: Edge case - very short signal
        print("\n10. Testing with very short signal...")
        try:
            short_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 0.01, 10))
            result = analyzer.compute_stft(short_signal, fs=1000)
            print(f"   [WARN] Very short signal STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Very short signal STFT properly failed: {e}")
        
        # Test 11: Edge case - very high frequency
        print("\n11. Testing with very high frequency...")
        try:
            high_freq_signal = np.sin(2 * np.pi * 500 * np.linspace(0, 1, 1000))  # 500 Hz at 1kHz fs
            result = analyzer.compute_stft(high_freq_signal, fs=1000)
            print(f"   [OK] High frequency signal STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] High frequency signal STFT failed: {e}")
        
        # Test 12: Edge case - very low frequency
        print("\n12. Testing with very low frequency...")
        try:
            low_freq_signal = np.sin(2 * np.pi * 0.1 * np.linspace(0, 1, 1000))  # 0.1 Hz at 1kHz fs
            result = analyzer.compute_stft(low_freq_signal, fs=1000)
            print(f"   [OK] Low frequency signal STFT succeeded: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] Low frequency signal STFT failed: {e}")
        
    except Exception as e:
        print(f"   [ERROR] Critical error in SpectrumAnalysis testing: {e}")
        logger.exception("SpectrumAnalysis critical error")


if __name__ == "__main__":
    print("Starting Spectrum Analysis Error Handling Tests")
    print("=" * 60)
    test_spectrum_analysis_errors()
    print("=" * 60)
    print("Spectrum Analysis Error Handling Tests Completed")

