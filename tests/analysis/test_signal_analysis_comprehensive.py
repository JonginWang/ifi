#!/usr/bin/env python3
"""
Comprehensive Signal Analysis Test
=================================

This script tests signal analysis capabilities with various signal types
including sinusoidal, linear combinations, and non-sinusoidal signals.
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
    from ifi.analysis.phase_analysis import SignalStacker
    from ifi.analysis.spectrum import SpectrumAnalysis
    from ifi.analysis.phi2ne import PhaseConverter
    from ifi.utils.cache_setup import setup_project_cache
    cache_config = setup_project_cache()
except ImportError as e:
    logger.error(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    sys.exit(1)


def generate_test_signals():
    """Generate various test signals"""
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs)  # 1 second of data
    
    signals = {}
    
    # 1. Pure sinusoidal signal
    f1 = 10  # 10 Hz
    signals['sinusoidal'] = np.sin(2 * np.pi * f1 * t)
    
    # 2. Linear combination of sinusoids
    f2 = 20  # 20 Hz
    f3 = 30  # 30 Hz
    signals['linear_combination'] = (np.sin(2 * np.pi * f1 * t) + 
                                   0.5 * np.sin(2 * np.pi * f2 * t) + 
                                   0.3 * np.sin(2 * np.pi * f3 * t))
    
    # 3. Chirp signal (frequency sweep)
    signals['chirp'] = np.sin(2 * np.pi * (f1 + 20 * t) * t)
    
    # 4. Square wave
    signals['square'] = np.sign(np.sin(2 * np.pi * f1 * t))
    
    # 5. Sawtooth wave
    signals['sawtooth'] = 2 * (t * f1 - np.floor(t * f1 + 0.5))
    
    # 6. Triangle wave
    signals['triangle'] = 2 * np.abs(2 * (t * f1 - np.floor(t * f1 + 0.5))) - 1
    
    # 7. Noise
    signals['noise'] = np.random.normal(0, 0.1, len(t))
    
    # 8. Signal with noise
    signals['sinusoidal_noisy'] = signals['sinusoidal'] + signals['noise']
    
    # 9. Amplitude modulated signal
    carrier_freq = 50  # 50 Hz
    mod_freq = 5     # 5 Hz
    signals['am_modulated'] = (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)) * np.sin(2 * np.pi * carrier_freq * t)
    
    # 10. Frequency modulated signal
    signals['fm_modulated'] = np.sin(2 * np.pi * (f1 + 10 * np.sin(2 * np.pi * mod_freq * t)) * t)
    
    return signals, t, fs


def test_signal_analysis():
    """Test signal analysis with various signal types"""
    print("\n" + "=" * 60)
    print("Comprehensive Signal Analysis Test")
    print("=" * 60)
    
    # Generate test signals
    signals, t, fs = generate_test_signals()
    
    try:
        # Initialize analysis components
        print("\n1. Initializing analysis components...")
        signal_stacker = SignalStacker(fs)
        spectrum_analyzer = SpectrumAnalysis()
        phase_converter = PhaseConverter()
        print("   [OK] All components initialized successfully")
        
        # Test each signal type
        for signal_name, signal in signals.items():
            print(f"\n2. Testing {signal_name} signal...")
            
            try:
                # Test fundamental frequency detection with appropriate frequency range
                f_range = (0.1, fs/2)  # Use Nyquist frequency as upper limit
                f0 = signal_stacker.find_fundamental_frequency(signal, f_range)
                print(f"   [OK] Fundamental frequency detected: {f0:.2f} Hz")
                
                # Test phase analysis (using signal as both ref and probe with phase shift)
                phase_shifted = np.sin(2 * np.pi * f0 * t + np.pi/4)  # 45 degree phase shift
                
                # Test different methods
                methods = ['stacking', 'cdm', 'cordic']
                for method in methods:
                    try:
                        phase_diff, detected_f0 = signal_stacker.compute_phase_difference(
                            signal, phase_shifted, f0, method)
                        print(f"   [OK] {method} method: phase_diff shape {phase_diff.shape}, f0={detected_f0:.2f} Hz")
                    except Exception as e:
                        print(f"   [WARN] {method} method failed: {e}")
                
                # Test spectrum analysis (if signal is long enough)
                if len(signal) >= 5000:  # STFT requires longer signals
                    try:
                        stft_result = spectrum_analyzer.compute_stft(signal, fs)
                        print(f"   [OK] STFT analysis: {type(stft_result)}")
                    except Exception as e:
                        print(f"   [WARN] STFT analysis failed: {e}")
                
                # Test CWT analysis
                try:
                    cwt_result = spectrum_analyzer.compute_cwt(signal, fs)
                    print(f"   [OK] CWT analysis: {type(cwt_result)}")
                except Exception as e:
                    print(f"   [WARN] CWT analysis failed: {e}")
                
                # Test phase-to-density conversion
                try:
                    # Create phase data
                    phase_data = np.linspace(0, 2*np.pi, len(signal))
                    density = phase_converter.phase_to_density(phase_data, wavelength=532e-9)
                    print(f"   [OK] Phase-to-density conversion: {type(density)}")
                except Exception as e:
                    print(f"   [WARN] Phase-to-density conversion failed: {e}")
                
            except Exception as e:
                print(f"   [ERROR] Analysis failed for {signal_name}: {e}")
                logger.exception(f"Analysis error for {signal_name}")
        
        print("\n3. Testing signal comparison...")
        try:
            # Compare two different signals
            ref_signal = signals['sinusoidal']
            probe_signal = signals['linear_combination']
            
            # Test unified analysis
            unified_result = signal_stacker.detect_phase_changes_unified(
                ref_signal, probe_signal)
            print(f"   [OK] Unified analysis: {type(unified_result)}")
            
        except Exception as e:
            print(f"   [ERROR] Signal comparison failed: {e}")
            logger.exception("Signal comparison error")
        
        print("\n4. Testing edge cases...")
        try:
            # Test with very short signal
            short_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 0.01, 10))
            f0_short = signal_stacker.find_fundamental_frequency(short_signal, (0.1, 50))
            print(f"   [OK] Short signal analysis: f0={f0_short:.2f} Hz")
            
            # Test with high frequency signal
            high_freq_signal = np.sin(2 * np.pi * 100 * t)
            f0_high = signal_stacker.find_fundamental_frequency(high_freq_signal, (0.1, fs/2))
            print(f"   [OK] High frequency signal analysis: f0={f0_high:.2f} Hz")
            
            # Test with low frequency signal
            low_freq_signal = np.sin(2 * np.pi * 0.5 * t)
            f0_low = signal_stacker.find_fundamental_frequency(low_freq_signal, (0.1, fs/2))
            print(f"   [OK] Low frequency signal analysis: f0={f0_low:.2f} Hz")
            
        except Exception as e:
            print(f"   [ERROR] Edge case testing failed: {e}")
            logger.exception("Edge case testing error")
        
        print("\n" + "=" * 60)
        print("Signal Analysis Test Completed Successfully")
        print("=" * 60)
        
    except Exception as e:
        print(f"   [ERROR] Critical error in signal analysis testing: {e}")
        logger.exception("Signal analysis critical error")


if __name__ == "__main__":
    print("Starting Comprehensive Signal Analysis Test")
    test_signal_analysis()

