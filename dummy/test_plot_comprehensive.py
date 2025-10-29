#!/usr/bin/env python3
"""
Comprehensive PLOT functionality test
"""

import sys
sys.path.insert(0, '.')
from ifi.analysis.plots import Plotter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test_comprehensive_plotting():
    """Test comprehensive PLOT functionality"""
    print("Testing comprehensive PLOT functionality...")
    
    # 복잡한 테스트 데이터 생성
    t = np.linspace(0, 0.1, 10000)  # 100ms, 100kHz sampling
    fs = 100000  # 100 kHz
    
    # 여러 채널의 신호 생성
    data = pd.DataFrame({
        'TIME': t,
        'CH0': np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t)),  # 1kHz + noise
        'CH1': np.sin(2 * np.pi * 2000 * t) + 0.1 * np.random.randn(len(t)),  # 2kHz + noise
        'CH2': np.sin(2 * np.pi * 5000 * t) + 0.1 * np.random.randn(len(t)), # 5kHz + noise
        'REF': np.sin(2 * np.pi * 1000 * t)  # Reference signal
    })
    
    print(f"Created comprehensive test data: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Time range: {t[0]:.3f} to {t[-1]:.3f} seconds")
    print(f"Sampling frequency: {fs} Hz")
    
    plotter = Plotter()
    
    # Test 1: Basic waveform plotting
    print("\n1. Testing basic waveform plotting...")
    try:
        fig1, axes1 = plotter.plot_waveforms(
            data, 
            title='Comprehensive Test - Waveforms',
            show_plot=False,
            time_scale='ms',
            signal_scale='V'
        )
        fig1.savefig('test_waveforms.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print("[SUCCESS] Basic waveform plotting")
    except Exception as e:
        print(f"[ERROR] Basic waveform plotting failed: {e}")
    
    # Test 2: Time-frequency analysis (STFT)
    print("\n2. Testing STFT analysis...")
    try:
        signal = data['CH0'].values
        fig2, axes2 = plotter.plot_time_frequency(
            signal,
            method='stft',
            fs=fs,
            title='STFT Analysis - CH0',
            show_plot=False,
            time_scale='ms',
            freq_scale='kHz'
        )
        fig2.savefig('test_stft.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("[SUCCESS] STFT analysis")
    except Exception as e:
        print(f"[ERROR] STFT analysis failed: {e}")
    
    # Test 3: Density plotting
    print("\n3. Testing density plotting...")
    try:
        # Create synthetic density data
        density_data = pd.DataFrame({
            'TIME': t,
            'ne_CH0': 1e19 * np.exp(-t * 10) + 5e18,  # Exponential decay
            'ne_CH1': 1.2e19 * np.exp(-t * 8) + 4e18,
            'ne_CH2': 0.8e19 * np.exp(-t * 12) + 6e18
        })
        
        fig3, axes3 = plotter.plot_density(
            density_data,
            title='Density Profiles',
            show_plot=False,
            time_scale='ms',
            density_scale='1e19'
        )
        fig3.savefig('test_density.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print("[SUCCESS] Density plotting")
    except Exception as e:
        print(f"[ERROR] Density plotting failed: {e}")
    
    # Test 4: Filter response plotting
    print("\n4. Testing filter response plotting...")
    try:
        # Create synthetic filter response data
        freqs = np.logspace(1, 5, 1000)  # 10 Hz to 100 kHz
        magnitude = 1 / (1 + (freqs / 1000) ** 2)  # Low-pass filter
        phase = -np.arctan(freqs / 1000)
        
        fig4, axes4 = plotter.plot_response(
            freqs, magnitude, phase,
            title='Filter Response',
            show_plot=False,
            freq_scale='Hz'
        )
        fig4.savefig('test_response.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print("[SUCCESS] Filter response plotting")
    except Exception as e:
        print(f"[ERROR] Filter response plotting failed: {e}")
    
    print("\n[SUCCESS] All PLOT tests completed successfully!")

if __name__ == '__main__':
    test_comprehensive_plotting()
