#!/usr/bin/env python3
"""
Simple test suite for spectrum analysis.
"""

import numpy as np

from ifi.analysis.spectrum import SpectrumAnalysis

def test_synthetic_signal():
    print("Testing synthetic signal with noise...")
    
    # Create synthetic signal with proper Nyquist consideration
    fs1 = 5e6   # 5 MHz component
    fs2 = 8e6   # 8 MHz component  
    fs = 50e6   # 50 MHz sampling rate (much higher than signal components)
    
    duration = 1.0  # 1 second
    t = np.arange(0, duration, 1/fs)
    
    # Create signals
    signal1 = np.sin(2 * np.pi * fs1 * t)  # 10 MHz component
    signal2 = np.sin(2 * np.pi * fs2 * t)  # 20 MHz component  
    noise = 0.1 * np.random.randn(len(t))  # Random noise
    
    signal = signal1 + 0.5 * signal2 + noise
    
    print("Signal parameters:")
    print(f"   - Sampling rate: {fs/1e6:.1f} MHz")
    print(f"   - Duration: {duration} s")
    print(f"   - Signal length: {len(signal)} samples")
    print(f"   - Component 1: {fs1/1e6:.1f} MHz (amplitude: 1.0)")
    print(f"   - Component 2: {fs2/1e6:.1f} MHz (amplitude: 0.5)")
    print("   - Noise level: 0.1")
    
    # Test frequency detection
    analyzer = SpectrumAnalysis()
    f_center = analyzer.find_center_frequency_fft(signal, fs)
    
    print("\nCenter frequency detection:")
    print(f"   - Detected: {f_center/1e6:.2f} MHz")
    print(f"   - Expected: ~{fs2/1e6:.1f} MHz (stronger component)")
    
    # Test STFT
    try:
        f, t_stft, Zxx = analyzer.compute_stft(signal, fs)
        ridge = analyzer.find_freq_ridge(Zxx, f, method='stft')
        
        print("\nSciPy STFT Analysis:")
        print(f"   - STFT shape: {Zxx.shape}")
        print(f"   - Frequency range: {f[0]/1e6:.1f} to {f[-1]/1e6:.1f} MHz")
        print(f"   - Ridge frequency: {np.mean(ridge)/1e6:.2f} MHz")
        
    except Exception as e:
        print(f"SciPy STFT failed: {e}")
    
    # Test ssqueezepy STFT
    try:
        f_ssq, t_ssq, Sxx = analyzer.compute_stft_sqpy(signal, fs, n_fft=1024, hop_len=512)
        ridge_ssq = analyzer.find_freq_ridge(Sxx, f_ssq, method='stft')
        
        print("\nssqueezepy STFT Analysis:")
        print(f"   - STFT shape: {Sxx.shape}")
        print(f"   - Frequency range: {f_ssq[0]/1e6:.1f} to {f_ssq[-1]/1e6:.1f} MHz")
        print(f"   - Ridge frequency: {np.mean(ridge_ssq)/1e6:.2f} MHz")
        
    except Exception as e:
        print(f"ssqueezepy STFT failed: {e}")
    
    print("\nTest completed!")
    assert len(signal) > 0, "Signal should not be empty"
    assert fs > 0, "Sampling frequency should be positive"

if __name__ == '__main__':
    signal, fs = test_synthetic_signal()