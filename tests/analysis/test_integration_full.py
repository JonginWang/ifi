#!/usr/bin/env python3
"""
Full integration test suite for the IFI package.
"""

import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.analysis.phi2ne import get_interferometry_params, PhaseConverter
from ifi.analysis.plots import plot_waveforms, plot_spectrogram, plot_density_results
from ifi.utils.common import LogManager, ensure_dir_exists
from ifi.utils.cache_setup import setup_project_cache

cache_config = setup_project_cache()

def setup_logging():
    """Setup logging for integration test."""
    LogManager(level="INFO")
    return logging.getLogger(__name__)

@pytest.fixture
def logger():
    """Provide logger fixture for tests."""
    return setup_logging()

@pytest.fixture
def results():
    """Provide results fixture for tests."""
    return create_full_synthetic_dataset()

@pytest.fixture
def spectrum_results(results):
    """Provide spectrum analysis results."""
    # Extract datasets from results
    datasets = results['datasets']
    fs = results['fs']
    
    # Perform spectrum analysis on first dataset
    first_dataset = list(datasets.values())[0]
    spectrum_analyzer = SpectrumAnalysis()
    spectrum_results = {}
    
    for channel in first_dataset.columns:
        if channel != 'TIME':
            signal_data = first_dataset[channel].values
            time_axis = first_dataset.index.values
            spectrum_results[channel] = spectrum_analyzer.analyze_signal(
                signal_data, time_axis, method='fft'
            )
    
    return spectrum_results

@pytest.fixture
def density_results(results):
    """Provide density calculation results."""
    # Extract datasets from results
    datasets = results['datasets']
    fs = results['fs']
    
    # Perform density calculation on first dataset
    first_dataset = list(datasets.values())[0]
    phase_converter = PhaseConverter()
    density_results = {}
    
    # Convert phase to density for each channel
    for channel in first_dataset.columns:
        if channel != 'TIME' and 'phase' in channel.lower():
            signal_data = first_dataset[channel].values
            time_axis = first_dataset.index.values
            density_results[channel] = phase_converter.phase_to_density(
                signal_data, time_axis
            )
    
    return density_results

@pytest.fixture
def vest_data():
    """Provide VEST data fixture."""
    # Create mock VEST data
    time_points = np.linspace(0, 0.05, 1000)
    vest_data = pd.DataFrame({
        'time': time_points,
        'ip': 1000 * np.sin(2 * np.pi * 10 * time_points) + 500,  # Plasma current
        'ne': 1e19 * np.exp(-time_points * 10) + 5e18,  # Electron density
        'te': 1000 * np.exp(-time_points * 5) + 500  # Electron temperature
    })
    return vest_data

def create_full_synthetic_dataset():
    """Create a complete synthetic dataset mimicking real interferometer data."""
    # Setup cache first
    cache_config = setup_project_cache()
    
    # Signal parameters
    fs = 250e6  # 250 MHz sampling
    duration = 0.05  # 50ms
    t = np.arange(0, duration, 1/fs)
    
    # Create realistic interferometer signals
    datasets = {}
    
    # 94 GHz data (two files: _056.csv and _789.csv)
    f_base_94 = 10e6  # 10 MHz base frequency
    
    # File 1: 45821_056.csv (has reference)
    ref_signal_94 = np.sin(2 * np.pi * f_base_94 * t)
    phase_94_1 = 0.1 * np.cumsum(np.random.randn(len(t))) / fs  # Random walk phase
    phase_94_2 = 0.15 * np.cumsum(np.random.randn(len(t))) / fs
    
    probe1_94 = np.sin(2 * np.pi * f_base_94 * t + phase_94_1)
    probe2_94 = np.sin(2 * np.pi * f_base_94 * t + phase_94_2)
    
    # Add realistic noise
    noise_level = 0.02
    ref_signal_94 += noise_level * np.random.randn(len(t))
    probe1_94 += noise_level * np.random.randn(len(t))
    probe2_94 += noise_level * np.random.randn(len(t))
    
    datasets['45821_056.csv'] = pd.DataFrame({
        'TIME': t,
        'CH0': ref_signal_94,   # Reference
        'CH1': probe1_94,       # Probe 1
        'CH2': probe2_94        # Probe 2
    })
    
    # File 2: 45821_789.csv (no reference, additional probes)
    # These signals are measured at different probe locations
    phase_94_3 = 0.08 * np.cumsum(np.random.randn(len(t))) / fs
    phase_94_4 = 0.12 * np.cumsum(np.random.randn(len(t))) / fs
    phase_94_5 = 0.18 * np.cumsum(np.random.randn(len(t))) / fs
    
    probe3_94 = np.sin(2 * np.pi * f_base_94 * t + phase_94_3)
    probe4_94 = np.sin(2 * np.pi * f_base_94 * t + phase_94_4)
    probe5_94 = np.sin(2 * np.pi * f_base_94 * t + phase_94_5)
    
    probe3_94 += noise_level * np.random.randn(len(t))
    probe4_94 += noise_level * np.random.randn(len(t))
    probe5_94 += noise_level * np.random.randn(len(t))
    
    datasets['45821_789.csv'] = pd.DataFrame({
        'TIME': t,
        'CH0': probe3_94,       # Probe 3
        'CH1': probe4_94,       # Probe 4
        'CH2': probe5_94        # Probe 5
    })
    
    # 280 GHz data: 45821_ALL.csv (single file, higher frequency)
    f_base_280 = 25e6  # 25 MHz base frequency
    ref_signal_280 = np.sin(2 * np.pi * f_base_280 * t)
    phase_280 = 0.2 * np.cumsum(np.random.randn(len(t))) / fs
    probe_280 = np.sin(2 * np.pi * f_base_280 * t + phase_280)
    
    ref_signal_280 += noise_level * np.random.randn(len(t))
    probe_280 += noise_level * np.random.randn(len(t))
    
    datasets['45821_ALL.csv'] = pd.DataFrame({
        'TIME': t,
        'CH0': ref_signal_280,  # Reference
        'CH1': probe_280,       # Probe
        'CH2': 0.1 * np.random.randn(len(t))  # Unused channel
    })
    
    # Set TIME as index for all datasets
    for filename, data in datasets.items():
        data.set_index('TIME', inplace=True)
    
    return {'datasets': datasets, 'fs': fs}

def create_synthetic_vest_data():
    """Create synthetic VEST data for comparison."""
    t = np.linspace(-0.005, 0.045, 1000)  # -5ms to +45ms
    
    # Realistic plasma current evolution
    ramp_up = np.where(t < 0, 0, np.exp(-((t - 0.002) / 0.003)**2))
    flat_top = np.where((t >= 0.005) & (t <= 0.035), 1.0, 0)
    ramp_down = np.where(t > 0.035, np.exp(-((t - 0.035) / 0.008)**2), 0)
    
    ip_profile = (ramp_up + flat_top + ramp_down) * 2.5  # 2.5V peak
    
    # H-alpha evolution (follows density)
    halpha_profile = 1.2 * ip_profile + 0.1 * np.random.randn(len(t))
    
    vest_data = pd.DataFrame({
        'TIME': t,
        'Ip_raw ([V])': ip_profile + 0.05 * np.random.randn(len(t)),
        'H-alpha ([a.u.])': halpha_profile
    })
    vest_data.set_index('TIME', inplace=True)
    
    return vest_data

def test_complete_analysis_pipeline(logger):
    """Test the complete analysis pipeline from data to results."""
    logger.info("Testing Complete Analysis Pipeline")
    logger.info("=" * 60)
    
    # Create synthetic datasets
    logger.info("Creating synthetic datasets...")
    datasets, fs = create_full_synthetic_dataset()
    vest_data = create_synthetic_vest_data()
    
    output_dir = "ifi/results/integration_test"
    ensure_dir_exists(output_dir)
    
    # Test each dataset processing
    results = {}
    
    for filename, data in datasets.items():
        logger.info(f"Processing {filename}")
        
        try:
            # Get interferometry parameters
            shot_num = int(filename.split('_')[0])
            params = get_interferometry_params(shot_num, filename)
            
            logger.info(f"Method: {params['method']}")
            logger.info(f"Frequency: {params['freq_ghz']} GHz")
            logger.info(f"Reference: {params['ref_col']}")
            logger.info(f"Probes: {params['probe_cols']}")
            
            # Store for later analysis
            results[filename] = {
                'data': data,
                'params': params,
                'fs': fs
            }
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            continue
    
    assert len(results) > 0, "Results should not be empty"
    assert len(vest_data) > 0, "VEST data should not be empty"

def test_spectrum_analysis_integration(results, logger):
    """Test spectrum analysis on synthetic data."""
    logger.info("Testing Spectrum Analysis Integration")
    logger.info("=" * 60)
    
    analyzer = SpectrumAnalysis()
    spectrum_results = {}
    
    datasets = results['datasets']
    for filename, file_data in datasets.items():
        data = file_data['data']
        params = file_data['params']
        fs = file_data['fs']
        
        logger.info(f"Analyzing {filename}")
        
        try:
            if params['ref_col']:
                # Analyze reference signal
                ref_signal = data[params['ref_col']].values
                
                # FFT analysis
                f_center = analyzer.find_center_frequency_fft(ref_signal, fs)
                logger.info(f"Center frequency: {f_center/1e6:.2f} MHz")
                
                # STFT analysis
                f_stft, t_stft, Sxx = analyzer.compute_stft_sqpy(ref_signal, fs, n_fft=1024, hop_len=512)
                ridge_stft = analyzer.find_freq_ridge(Sxx, f_stft, method='stft')
                logger.info(f"   STFT ridge: {np.mean(ridge_stft)/1e6:.2f} MHz")
                
                # CWT analysis (shorter for speed)
                if len(ref_signal) <= 1000000:  # Only for manageable sizes
                    freqs_cwt, Wx = analyzer.compute_cwt(ref_signal, fs, f_min=5e6, f_max=50e6, nv=16, scales='log')
                    ridge_cwt = analyzer.find_freq_ridge(Wx, freqs_cwt, method='cwt')
                    logger.info(f"   CWT ridge: {np.mean(ridge_cwt)/1e6:.2f} MHz")
                else:
                    logger.info("   CWT skipped (signal too long)")
                
                spectrum_results[filename] = {
                    'f_center': f_center,
                    'stft_data': (f_stft, t_stft, Sxx),
                    'ridge_stft': ridge_stft
                }
                
            else:
                logger.info("     No reference signal for analysis")
                
        except Exception as e:
            logger.error(f"    Spectrum analysis failed: {e}")
    
    assert len(spectrum_results) > 0, "Spectrum results should not be empty"

def test_density_calculation_integration(results, logger):
    """Test density calculation on synthetic data."""
    logger.info("  Testing Density Calculation Integration")
    logger.info("=" * 60)
    
    phase_converter = PhaseConverter()
    density_results = {}
    
    datasets = results['datasets']
    for filename, file_data in datasets.items():
        data = file_data['data']
        params = file_data['params']
        fs = file_data['fs']
        
        logger.info(f" Calculating density for {filename}")
        
        try:
            if params['method'] == 'CDM' and params['ref_col']:
                ref_signal = data[params['ref_col']].values
                
                # Use a reasonable center frequency for testing
                f_center = 20e6  # 20 MHz
                
                density_data = {}
                
                for probe_col in params['probe_cols']:
                    if probe_col in data.columns:
                        probe_signal = data[probe_col].values
                        
                        # Calculate phase
                        phase, _ = phase_converter.calc_phase_cdm(ref_signal, probe_signal, fs, f_center)
                        
                        # Convert to density
                        density = phase_converter.phase_to_density(phase, analysis_params=params)
                        
                        density_data[f"ne_{probe_col}_{filename}"] = density
                        
                        logger.info(f"{probe_col}: density range {np.min(density):.2e} to {np.max(density):.2e} m⁻³")
                
                density_results[filename] = density_data
                
            else:
                logger.info(f"Skipping (method: {params['method']}, ref: {params['ref_col']})")
                
        except Exception as e:
            logger.error(f"Density calculation failed: {e}")
    
    assert len(density_results) > 0, "Density results should not be empty"

def test_plotting_integration(results, spectrum_results, density_results, vest_data, logger):
    """Test plotting functions with integration data."""
    logger.info("Testing Plotting Integration")
    logger.info("=" * 60)
    
    output_dir = "ifi/results/integration_test/plots"
    ensure_dir_exists(output_dir)
    
    # Test waveform plots
    for filename, file_data in results.items():
        data = file_data['data']
        
        try:
            logger.info(f"Plotting waveforms for {filename}")
            
            fig, axes = plot_waveforms(
                data,
                title=f"Synthetic Interferometer Data - {filename}",
                time_range=(0.000, 0.020)  # First 20ms
            )
            
            output_path = Path(output_dir) / f"waveforms_{filename.replace('.csv', '')}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Waveform plot failed: {e}")
    
    # Test spectrum plots
    for filename, spectrum_data in spectrum_results.items():
        try:
            logger.info(f"Plotting spectrum for {filename}")
            
            f_stft, t_stft, Sxx = spectrum_data['stft_data']
            time_axis = t_stft
            
            fig = plot_spectrogram(
                f_stft, time_axis, np.abs(Sxx).T,
                title=f"STFT Spectrogram - {filename}",
                freq_range=(5e6, 50e6)
            )
            
            output_path = Path(output_dir) / f"spectrum_{filename.replace('.csv', '')}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Spectrum plot failed: {e}")
    
    # Test density plots
    if density_results:
        try:
            logger.info("Plotting density results")
            
            # Combine all density data
            all_density = {}
            for file_densities in density_results.values():
                all_density.update(file_densities)
            
            fig = plot_density_results(
                all_density, vest_data,
                title="Synthetic Plasma Density - Integration Test",
                time_range=(-0.005, 0.045)
            )
            
            output_path = Path(output_dir) / "density_combined.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Density plot failed: {e}")

def performance_benchmark_integration(logger):
    """Benchmark the complete integration performance."""
    logger.info("Integration Performance Benchmark")
    logger.info("=" * 60)
    
    # Test with different data sizes
    data_sizes = [10000, 100000, 1000000]  # 10k, 100k, 1M samples
    
    for size in data_sizes:
        logger.info(f"Testing {size:,} samples")
        
        try:
            # Create data
            fs = 250e6
            t = np.arange(size) / fs
            ref_signal = np.sin(2 * np.pi * 10e6 * t)
            probe_signal = np.sin(2 * np.pi * 10e6 * t + 0.1 * np.sin(2 * np.pi * 1000 * t))
            
            # Timing tests
            timings = {}
            
            # Spectrum analysis
            start_time = time.time()
            analyzer = SpectrumAnalysis()
            f_center = analyzer.find_center_frequency_fft(ref_signal, fs)
            timings['fft'] = time.time() - start_time
            
            # Phase calculation
            start_time = time.time()
            phase_converter = PhaseConverter()
            phase, _ = phase_converter.calc_phase_cdm(ref_signal, probe_signal, fs, 20e6)
            timings['phase'] = time.time() - start_time
            
            # Density calculation
            start_time = time.time()
            params = {'freq_ghz': 94.0, 'n_path': 2}
            density = phase_converter.phase_to_density(phase, analysis_params=params)
            timings['density'] = time.time() - start_time
            
            # Total pipeline time
            total_time = sum(timings.values())
            
            logger.info(f"FFT:     {timings['fft']*1000:6.1f} ms")
            logger.info(f"Phase:   {timings['phase']*1000:6.1f} ms")
            logger.info(f"Density: {timings['density']*1000:6.1f} ms")
            logger.info(f"Total:   {total_time*1000:6.1f} ms")
            logger.info(f"Rate:     {size/total_time/1e6:.1f} MSamples/s")
            logger.info("")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

def main():
    """Main integration test execution."""
    logger = setup_logging()
    
    logger.info("IFI Integration Test - Full Pipeline")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Test complete pipeline
        results, vest_data = test_complete_analysis_pipeline(logger)
        logger.info("")
        
        # Test spectrum analysis
        spectrum_results = test_spectrum_analysis_integration(results, logger)
        logger.info("")
        
        # Test density calculation
        density_results = test_density_calculation_integration(results, logger)
        logger.info("")
        
        # Test plotting
        test_plotting_integration(results, spectrum_results, density_results, vest_data, logger)
        logger.info("")
        
        # Performance benchmark
        performance_benchmark_integration(logger)
        logger.info("")
        
        logger.info("Full integration test completed successfully!")
        logger.info("Results saved in: ifi/results/integration_test/")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise

if __name__ == '__main__':
    main()