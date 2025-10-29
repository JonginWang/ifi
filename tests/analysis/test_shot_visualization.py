#!/usr/bin/env python3
"""
Integrated Shot Visualization and Plotting Test Suite
===================================================

This test suite combines comprehensive plotting tests with real shot data visualization.
It tests both synthetic data plotting functions and real shot data processing.
"""

# ============================================================================
# CRITICAL: Set up numba cache BEFORE any imports
# ============================================================================
from pathlib import Path

from ifi.utils.cache_setup import setup_project_cache
cache_config = setup_project_cache()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging

from ifi.analysis.plots import (
    plot_response, plot_waveforms, plot_spectrogram, plot_density_results,
    create_shot_results_directory, plot_shot_spectrograms, plot_shot_waveforms, plot_shot_density_evolution,
    plot_shot_overview, load_cached_shot_data
)
from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.utils.common import LogManager, ensure_dir_exists
from ifi.db_controller.vest_db import VEST_DB
from ifi.analysis.params.params_plot import FontStyle


def setup_logging():
    """Setup logging for test execution."""
    LogManager(level="INFO")
    return logging.getLogger(__name__)

# Pytest fixtures
import pytest

@pytest.fixture
def logger():
    """Provide logger fixture for tests."""
    return setup_logging()


def create_synthetic_interferometer_data():
    """Create synthetic interferometer data for plotting tests."""
    # Parameters
    fs = 250e6  # 250 MHz sampling
    duration = 0.02  # 20 ms
    t = np.arange(0, duration, 1/fs)
    
    # Base frequencies
    f_ref = 10e6    # 10 MHz reference
    f_probe1 = 10.5e6  # 10.5 MHz probe 1 (slightly shifted)
    f_probe2 = 11e6    # 11 MHz probe 2
    
    # Create signals with realistic phase evolution
    phase_evolution = 2 * np.pi * np.cumsum(0.1 * np.random.randn(len(t))) / fs
    
    ref_signal = np.sin(2 * np.pi * f_ref * t)
    probe1_signal = np.sin(2 * np.pi * f_probe1 * t + phase_evolution * 0.5)
    probe2_signal = np.sin(2 * np.pi * f_probe2 * t + phase_evolution)
    
    # Add noise
    noise_level = 0.05
    ref_signal += noise_level * np.random.randn(len(t))
    probe1_signal += noise_level * np.random.randn(len(t))
    probe2_signal += noise_level * np.random.randn(len(t))
    
    # Create DataFrame
    data = pd.DataFrame({
        'TIME': t,
        'CH0': ref_signal,    # Reference
        'CH1': probe1_signal, # Probe 1
        'CH2': probe2_signal  # Probe 2
    })
    data.set_index('TIME', inplace=True)
    
    return data, fs


def create_synthetic_density_data():
    """Create synthetic density data for plotting."""
    t = np.linspace(-0.005, 0.035, 1000)  # -5ms to +35ms
    
    # Plasma density evolution (realistic shape)
    ramp_up = np.where(t < 0, 0, np.exp(-((t - 0.005) / 0.003)**2))
    flat_top = np.where((t >= 0.005) & (t <= 0.025), 1.0, 0)
    ramp_down = np.where(t > 0.025, np.exp(-((t - 0.025) / 0.005)**2), 0)
    
    base_density = (ramp_up + flat_top + ramp_down) * 5e18  # 5e18 m^-3 peak
    
    # Add realistic fluctuations
    fluctuations = 0.1 * base_density * np.sin(2 * np.pi * 50 * t)  # 50 Hz fluctuations
    noise = 0.05 * base_density * np.random.randn(len(t))
    
    density_profiles = {
        'ne_CH1_test': base_density + fluctuations + noise,
        'ne_CH2_test': 0.8 * base_density + 0.8 * fluctuations + noise,
    }
    
    # Create VEST-like data
    vest_data = pd.DataFrame({
        'TIME': t,
        'Ip_raw ([V])': 2.0 * (flat_top + 0.5 * ramp_up + 0.3 * ramp_down),  # Plasma current
        'H-alpha ([a.u.])': 1.5 * base_density / 5e18 + 0.2 * np.random.randn(len(t))  # H-alpha
    })
    vest_data.set_index('TIME', inplace=True)
    
    return density_profiles, vest_data


def test_waveform_plots(logger):
    """Test waveform plotting functions."""
    logger.info("Testing Waveform Plots")
    logger.info("=" * 50)
    
    # Create synthetic data
    data, fs = create_synthetic_interferometer_data()
    
    output_dir = "ifi/results/test_plots/waveforms"
    ensure_dir_exists(output_dir)
    
    test_cases = [
        {
            'name': 'basic_waveforms',
            'description': 'Basic 3-channel waveform plot',
            'channels': ['CH0', 'CH1', 'CH2'],
            'time_range': None
        },
        {
            'name': 'time_windowed',
            'description': 'Time-windowed waveform plot',
            'channels': ['CH0', 'CH1'],
            'time_range': (0.005, 0.015)  # 10ms window
        },
        {
            'name': 'single_channel',
            'description': 'Single channel detail plot',
            'channels': ['CH1'],
            'time_range': (0.000, 0.005)  # First 5ms
        }
    ]
    
    for case in test_cases:
        logger.info(f" {case['description']}")
        
        try:
            # Filter data by time range if specified
            plot_data = data[case['channels']].copy()
            if case['time_range'] is not None:
                time_mask = (data.index >= case['time_range'][0]) & (data.index <= case['time_range'][1])
                plot_data = plot_data[time_mask]
            
            fig, axes = plot_waveforms(
                plot_data, 
                title=f"Test Waveforms - {case['description']}"
            )
            
            output_path = Path(output_dir) / f"{case['name']}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f" Saved: {output_path}")
            
        except Exception as e:
            logger.error(f" Failed: {e}")


def test_spectrum_plots(logger):
    """Test spectrum and spectrogram plotting."""
    logger.info("Testing Spectrum Plots")
    logger.info("=" * 50)
    
    # Create test signal
    data, fs = create_synthetic_interferometer_data()
    signal = data['CH1'].values
    
    analyzer = SpectrumAnalysis()
    output_dir = "ifi/results/test_plots/spectra"
    ensure_dir_exists(output_dir)
    
    # Test STFT spectrogram
    try:
        logger.info(" STFT Spectrogram")
        f, t_stft, Zxx = analyzer.compute_stft(signal, fs, nperseg=1024, noverlap=512)
        
        # Filter frequency range if needed
        freq_mask = (f >= 5e6) & (f <= 15e6)
        f_filtered = f[freq_mask]
        Zxx_filtered = np.abs(Zxx)[freq_mask, :]
        
        fig, ax = plot_spectrogram(
            f_filtered, t_stft, Zxx_filtered,
            title="STFT Spectrogram - Synthetic Signal"
        )
        
        output_path = Path(output_dir) / "stft_spectrogram.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f" Saved: {output_path}")
        
    except Exception as e:
        logger.error(f" STFT failed: {e}")
    
    # Test CWT spectrogram (with smaller signal to avoid memory issues)
    try:
        logger.info(" CWT Spectrogram")
        # Use only first 10000 samples to avoid memory issues
        signal_short = signal[:10000]
        freqs, Wx = analyzer.compute_cwt(signal_short, fs, nv=8, scales='log')
        
        # Create time array for the short signal
        t_cwt = np.arange(len(signal_short)) / fs
        
        fig, ax = plot_spectrogram(
            freqs, t_cwt, np.abs(Wx).T,
            title="CWT Spectrogram - Synthetic Signal"
        )
        
        output_path = Path(output_dir) / "cwt_spectrogram.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f" Saved: {output_path}")
        
    except Exception as e:
        logger.error(f" CWT failed: {e}")


def test_filter_response_plots(logger):
    """Test filter response plotting."""
    logger.info("Testing Filter Response Plots")
    logger.info("=" * 50)
    
    output_dir = "ifi/results/test_plots/filters"
    ensure_dir_exists(output_dir)
    
    # Test different filter configurations
    fs = 250e6  # 250 MHz sampling
    filter_configs = [
        {
            'name': 'bandpass_10mhz',
            'description': '10 MHz Bandpass Filter',
            'freqs': [8e6, 9.5e6, 10.5e6, 12e6],  # [stop1, pass1, pass2, stop2]
            'amps': [0, 1, 1, 0],
            'rips': [0.01, 0.001, 0.01]
        },
        {
            'name': 'lowpass_50mhz',
            'description': '50 MHz Lowpass Filter',
            'freqs': [45e6, 50e6],
            'amps': [1, 0],
            'rips': [0.001, 0.01]
        },
        {
            'name': 'highpass_5mhz',
            'description': '5 MHz Highpass Filter',
            'freqs': [5e6, 8e6],
            'amps': [0, 1],
            'rips': [0.01, 0.001]
        }
    ]
    
    for config in filter_configs:
        logger.info(f" {config['description']}")
        
        try:
            from scipy.signal import remez, freqz
            
            # Design filter using remez
            numtaps = 101
            h = remez(numtaps, config['freqs'], config['amps'], fs=fs)
            
            # Get frequency response
            w, H = freqz(h, worN=8000, fs=fs)
            
            fig = plot_response(
                w, H, title=config['description']
            )
            
            output_path = Path(output_dir) / f"{config['name']}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f" Saved: {output_path}")
            
        except Exception as e:
            logger.error(f" Failed: {e}")


def test_density_plots(logger):
    """Test density plotting functions."""
    logger.info("Testing Density Plots")
    logger.info("=" * 50)
    
    # Create synthetic density data
    density_profiles, vest_data = create_synthetic_density_data()
    
    output_dir = "ifi/results/test_plots/density"
    ensure_dir_exists(output_dir)
    
    # Test density profile plots
    try:
        logger.info(" Density Profiles")
        
        # Convert to DataFrame for plotting
        time_data = list(density_profiles.values())[0]  # Get time from first entry
        t = np.arange(len(time_data)) / 1000  # Assume 1kHz sampling
        
        density_df = pd.DataFrame(density_profiles, index=t)
        
        fig, ax = plot_density_results(
            density_df,
            title="Synthetic Plasma Density Evolution"
        )
        
        output_path = Path(output_dir) / "density_evolution.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f" Saved: {output_path}")
        
    except Exception as e:
        logger.error(f" Failed: {e}")
    
    # Test individual channel plots
    for channel, ne_data in density_profiles.items():
        try:
            logger.info(f" Individual plot: {channel}")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Density plot
            time_axis = vest_data.index.values
            ax1.plot(time_axis * 1000, ne_data / 1e18, 'b-', linewidth=2, label=f'Density {channel}')
            ax1.set_ylabel(r'LID [$10^{18} m^{-2}$]', **FontStyle.label)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title(f'Plasma Density - {channel}', **FontStyle.title)
            
            # VEST data plot
            ax2.plot(time_axis * 1000, vest_data['Ip_raw ([V])'], 'r-', label='Plasma Current')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(time_axis * 1000, vest_data['H-alpha ([a.u.])'], 'g-', label='H-alpha')
            
            ax2.set_xlabel('Time [ms]', **FontStyle.label)
            ax2.set_ylabel(r'$I_{p}$ [V]', color='r', **FontStyle.label)
            ax2_twin.set_ylabel(r'$H_{\alpha}$ [a.u.]', color='g', **FontStyle.label)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = Path(output_dir) / f"{channel}_individual.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f" Saved: {output_path}")
            
        except Exception as e:
            logger.error(f" Failed for {channel}: {e}")


def test_comparison_plots(logger):
    """Test comparative plotting functions."""
    logger.info("Testing Comparison Plots")
    logger.info("=" * 50)
    
    output_dir = "ifi/results/test_plots/comparisons"
    ensure_dir_exists(output_dir)
    
    # Create comparison data
    data, fs = create_synthetic_interferometer_data()
    analyzer = SpectrumAnalysis()
    
    # Method comparison
    try:
        logger.info(" Method Comparison")
        
        signal = data['CH1'].values
        
        # Different analysis methods
        f_fft = analyzer.find_center_frequency_fft(signal, fs)
        f_stft, t_stft, Sxx_stft = analyzer.compute_stft_sqpy(signal, fs, n_fft=1024, hop_len=512)
        freqs_cwt, Wx_cwt = analyzer.compute_cwt(signal, fs, nv=32, scales='log-piecewise')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain
        axes[0, 0].plot(data.index.values * 1000, signal)
        axes[0, 0].set_title('Time Domain Signal', **FontStyle.title)
        axes[0, 0].set_xlabel('Time [ms]', **FontStyle.label)
        axes[0, 0].set_ylabel('Amplitude', **FontStyle.label)
        axes[0, 0].grid(True, alpha=0.3)
        
        # FFT
        fft_signal = np.fft.fft(signal)
        freqs_fft = np.fft.fftfreq(len(signal), 1/fs)
        axes[0, 1].plot(freqs_fft[:len(freqs_fft)//2] / 1e6, np.abs(fft_signal[:len(fft_signal)//2]))
        axes[0, 1].axvline(f_fft / 1e6, color='r', linestyle='--', label=f'Peak: {f_fft/1e6:.1f} MHz')
        axes[0, 1].set_title('FFT Spectrum', **FontStyle.title)
        axes[0, 1].set_xlabel('Frequency [MHz]', **FontStyle.label)
        axes[0, 1].set_ylabel('Magnitude', **FontStyle.label)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # STFT
        im1 = axes[1, 0].imshow(np.abs(Sxx_stft), aspect='auto', origin='lower', 
                               extent=[t_stft[0]*1000, t_stft[-1]*1000,
                                      f_stft[0]/1e6, f_stft[-1]/1e6])
        axes[1, 0].set_title('STFT Spectrogram', **FontStyle.title)
        axes[1, 0].set_xlabel('Time [ms]', **FontStyle.label)
        axes[1, 0].set_ylabel('Frequency [MHz]', **FontStyle.label)
        plt.colorbar(im1, ax=axes[1, 0])
        
        # CWT
        im2 = axes[1, 1].imshow(np.abs(Wx_cwt).T, aspect='auto', origin='lower',
                               extent=[data.index.values[0]*1000, data.index.values[-1]*1000,
                                      freqs_cwt[0]/1e6, freqs_cwt[-1]/1e6])
        axes[1, 1].set_title('CWT Spectrogram', **FontStyle.title)
        axes[1, 1].set_xlabel('Time [ms]', **FontStyle.label)
        axes[1, 1].set_ylabel('Frequency [MHz]', **FontStyle.label)
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / "method_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f" Saved: {output_path}")
        
    except Exception as e:
        logger.error(f" Failed: {e}")


def performance_test(logger):
    """Test plotting performance with different data sizes."""
    logger.info("Performance Test")
    logger.info("=" * 50)
    
    data_sizes = [1000, 10000, 100000, 1000000]
    
    for size in data_sizes:
        logger.info(f" Testing {size:,} samples")
        
        # Create data
        t = np.linspace(0, 0.1, size)
        signal = np.sin(2 * np.pi * 10e6 * t)
        data = pd.DataFrame({'TIME': t, 'CH0': signal})
        data.set_index('TIME', inplace=True)
        
        # Test plotting speed
        start_time = time.time()
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index.values, data['CH0'].values)
            ax.set_title(f'Performance Test - {size:,} samples', **FontStyle.title)
            plt.close(fig)
            
            elapsed = time.time() - start_time
            logger.info(f" Plot time: {elapsed*1000:.1f} ms")
            
        except Exception as e:
            logger.error(f" Failed: {e}")


def load_vest_data(shot_num):
    """Load VEST data for the shot."""
    logger = logging.getLogger(__name__)
    
    try:
        with VEST_DB() as vest_db:
            logger.info(f"Loading VEST data for shot {shot_num}")
            # Load common diagnostic fields
            fields = [109, 101]  # Ip, H-alpha
            
            vest_data = {}
            for field_id in fields:
                try:
                    time_data, signal_data = vest_db.load_shot(shot_num, field_id)
                    if time_data is not None and signal_data is not None:
                        vest_data[f'field_{field_id}'] = pd.Series(signal_data, index=time_data)
                except Exception as e:
                    logger.warning(f"Failed to load VEST field {field_id}: {e}")
            
            if vest_data:
                return pd.DataFrame(vest_data)
            else:
                return None
                
    except Exception as e:
        logger.error(f"Failed to connect to VEST DB: {e}")
        return None


def test_real_shot_visualization(logger, shot_num=45821):
    """Test visualization with real shot data."""
    logger.info("Testing Real Shot Visualization")
    logger.info("=" * 50)
    
    logger.info(f"Processing shot {shot_num}")
    
    # Load cached data
    shot_data = load_cached_shot_data(shot_num)
    if not shot_data:
        logger.warning(f"No cached data found for shot {shot_num}, skipping real shot test")
        return
    
    # Load VEST data
    vest_data = load_vest_data(shot_num)
    
    # Create results directory
    results_dir = create_shot_results_directory(shot_num, "ifi/results/test_shot_vis")
    logger.info(f"Results will be saved to: {results_dir}")
    
    try:
        # Generate all visualizations
        plot_shot_waveforms(shot_data, results_dir, shot_num)
        plot_shot_spectrograms(shot_data, results_dir, shot_num)
        plot_shot_density_evolution(shot_data, vest_data, results_dir, shot_num)
        plot_shot_overview(shot_data, vest_data, results_dir, shot_num)
        
        logger.info(f"Real shot visualization complete! Results saved in: {results_dir}")
        
    except Exception as e:
        logger.error(f"Real shot visualization failed: {e}")


def main():
    """Main test execution."""
    logger = setup_logging()
    
    logger.info("IFI Shot Visualization - Integrated Test Suite")
    logger.info("=" * 80)
    logger.info("")
    
    # Ensure output directory exists
    ensure_dir_exists("ifi/results/test_plots")
    ensure_dir_exists("ifi/results/test_shot_vis")
    
    try:
        # Test 1: Synthetic data plotting
        test_waveform_plots(logger)
        logger.info("")
        
        test_spectrum_plots(logger)
        logger.info("")
        
        test_filter_response_plots(logger)
        logger.info("")
        
        test_density_plots(logger)
        logger.info("")
        
        test_comparison_plots(logger)
        logger.info("")
        
        performance_test(logger)
        logger.info("")
        
        # Test 2: Real shot data visualization
        test_real_shot_visualization(logger)
        logger.info("")
        
        logger.info("All tests completed successfully!")
        logger.info("Synthetic test results saved in: ifi/results/test_plots/")
        logger.info("Real shot results saved in: ifi/results/test_shot_vis/")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == '__main__':
    main()
