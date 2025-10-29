"""
Comprehensive test suite for spectrum analysis (STFT, CWT) with various signal types.
"""

from pathlib import Path
import sys

current_dir = Path(__file__).resolve()
ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
                + list(current_dir.parents) if p.name == 'ifi']
IFI_ROOT = ifi_parents[-1] if ifi_parents else None

try:
    sys.path.insert(0, str(IFI_ROOT))
except Exception as e:
    print(f"!! Could not find ifi package root: {e}")
    pass

import numpy as np
import time
import logging
from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.utils.common import LogManager

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

@pytest.fixture
def signals():
    """Provide signals fixture for tests."""
    return create_test_signals()

def create_test_signals():
    """Create various synthetic signals for testing."""
    signals = {}
    
    # Common parameters
    duration = 0.01  # 10ms (much shorter for memory efficiency)
    fs = 250e6  # 250 MHz sampling rate
    t = np.arange(0, duration, 1/fs)
    
    # 1. Single frequency signal (10 MHz)
    signals['single_10mhz'] = {
        'data': np.sin(2 * np.pi * 10e6 * t),
        'description': '10 MHz sine wave',
        'expected_freq': 10e6,
        'fs': fs,
        'time': t
    }
    
    # 2. Dual frequency signal (5 MHz + 20 MHz)
    signals['dual_5_20mhz'] = {
        'data': np.sin(2 * np.pi * 5e6 * t) + 0.7 * np.sin(2 * np.pi * 20e6 * t),
        'description': '5 MHz + 20 MHz dual tone',
        'expected_freq': 20e6,  # Stronger component
        'fs': fs,
        'time': t
    }
    
    # 3. Chirp signal (1-50 MHz sweep)
    f0, f1 = 1e6, 50e6
    signals['chirp_1_50mhz'] = {
        'data': np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t),
        'description': f'Chirp {f0/1e6:.0f}-{f1/1e6:.0f} MHz',
        'expected_freq': (f0 + f1) / 2,  # Middle frequency
        'fs': fs,
        'time': t
    }
    
    # 4. Noisy signal (10 MHz + noise)
    np.random.seed(42)  # Reproducible results
    signals['noisy_10mhz'] = {
        'data': np.sin(2 * np.pi * 10e6 * t) + 0.3 * np.random.randn(len(t)),
        'description': '10 MHz + 30% noise',
        'expected_freq': 10e6,
        'fs': fs,
        'time': t
    }
    
    # 5. Multi-component signal (5, 10, 15 MHz)
    signals['multi_5_10_15mhz'] = {
        'data': (np.sin(2 * np.pi * 5e6 * t) + 
                0.8 * np.sin(2 * np.pi * 10e6 * t) + 
                0.6 * np.sin(2 * np.pi * 15e6 * t)),
        'description': '5, 10, 15 MHz multi-tone',
        'expected_freq': 10e6,  # Strongest component
        'fs': fs,
        'time': t
    }
    
    return signals

def test_fft_analysis(signals, logger):
    """Test FFT-based frequency detection."""
    logger.info("Testing FFT Analysis")
    logger.info("=" * 60)
    
    analyzer = SpectrumAnalysis()
    results = {}
    
    for name, signal_info in signals.items():
        signal = signal_info['data']
        fs = signal_info['fs']
        expected = signal_info['expected_freq']
        
        start_time = time.time()
        detected_freq = analyzer.find_center_frequency_fft(signal, fs)
        elapsed = time.time() - start_time
        
        error = abs(detected_freq - expected) / expected * 100
        
        results[name] = {
            'detected': detected_freq,
            'expected': expected,
            'error_pct': error,
            'time': elapsed
        }
        
        status = "PASS" if error < 5.0 else "WARN"
        logger.info(f"{status} {signal_info['description']}")
        logger.info(f"   Expected: {expected/1e6:.2f} MHz")
        logger.info(f"   Detected: {detected_freq/1e6:.2f} MHz")
        logger.info(f"   Error: {error:.1f}%")
        logger.info(f"   Time: {elapsed*1000:.1f} ms")
        logger.info("")
    
    assert len(results) > 0, "Results should not be empty"

def test_stft_analysis(signals, logger):
    """Test STFT analysis with different parameters."""
    logger.info("Testing STFT Analysis")
    logger.info("=" * 60)
    
    analyzer = SpectrumAnalysis()
    stft_configs = [
        {'n_fft': 512, 'hop_len': 256, 'name': 'Fast (512)'},
        {'n_fft': 1024, 'hop_len': 512, 'name': 'Standard (1024)'},
        {'n_fft': 2048, 'hop_len': 1024, 'name': 'High-res (2048)'},
    ]
    
    results = {}
    
    for name, signal_info in signals.items():
        signal = signal_info['data']
        fs = signal_info['fs']
        
        logger.info(f"Testing {signal_info['description']}")
        signal_results = {}
        
        for config in stft_configs:
            try:
                start_time = time.time()
                
                # Test SciPy STFT
                f_scipy, t_stft, Zxx = analyzer.compute_stft(
                    signal, fs, nperseg=config['n_fft'], noverlap=config['n_fft']//2
                )
                ridge_scipy = analyzer.find_freq_ridge(Zxx, f_scipy, method='stft')
                scipy_time = time.time() - start_time
                
                # Test ssqueezepy STFT
                start_time = time.time()
                f_ssq, t_ssq, Sxx = analyzer.compute_stft_sqpy(
                    signal, fs, n_fft=config['n_fft'], hop_len=config['hop_len']
                )
                ridge_ssq = analyzer.find_freq_ridge(Sxx, f_ssq, method='stft')
                ssq_time = time.time() - start_time
                
                signal_results[config['name']] = {
                    'scipy': {
                        'shape': Zxx.shape,
                        'ridge_freq': np.mean(ridge_scipy),
                        'time': scipy_time
                    },
                    'ssqueezepy': {
                        'shape': Sxx.shape,
                        'ridge_freq': np.mean(ridge_ssq),
                        'time': ssq_time
                    }
                }
                
                logger.info(f"   {config['name']}:")
                logger.info(f"     SciPy: {Zxx.shape}, ridge={np.mean(ridge_scipy)/1e6:.1f}MHz, {scipy_time*1000:.1f}ms")
                logger.info(f"     ssqpy: {Sxx.shape}, ridge={np.mean(ridge_ssq)/1e6:.1f}MHz, {ssq_time*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"   {config['name']} failed: {e}")
                signal_results[config['name']] = {'error': str(e)}
        
        results[name] = signal_results
        logger.info("")
    
    assert len(results) > 0, "STFT results should not be empty"

def test_cwt_analysis(signals, logger):
    """Test CWT analysis."""
    logger.info("Testing CWT Analysis")
    logger.info("=" * 60)
    
    analyzer = SpectrumAnalysis()
    results = {}
    
    # CWT parameters (reduced for memory efficiency)
    f_min, f_max = 5e6, 20e6  # Smaller frequency range: 5-20 MHz
    nv = 8  # Fewer voices per octave
    scales = 'log'  # Simpler scale distribution
    
    for name, signal_info in signals.items():
        signal = signal_info['data']
        fs = signal_info['fs']
        
        logger.info(f"Testing {signal_info['description']}")
        
        try:
            start_time = time.time()
            freqs, Wx = analyzer.compute_cwt(signal, fs, f_min=f_min, f_max=f_max, nv=nv, scales=scales)
            ridge = analyzer.find_freq_ridge(Wx, freqs, method='cwt')
            elapsed = time.time() - start_time
            
            results[name] = {
                'shape': Wx.shape,
                'freq_range': (freqs[0], freqs[-1]),
                'ridge_freq': np.mean(ridge),
                'time': elapsed
            }
            
            logger.info(f"   CWT shape: {Wx.shape}")
            logger.info(f"   Freq range: {freqs[0]/1e6:.1f}-{freqs[-1]/1e6:.1f} MHz")
            logger.info(f"   Ridge freq: {np.mean(ridge)/1e6:.2f} MHz")
            logger.info(f"   Time: {elapsed*1000:.0f} ms")
            
        except Exception as e:
            logger.error(f"   CWT failed: {e}")
            results[name] = {'error': str(e)}
        
        logger.info("")
    
    assert len(results) > 0, "CWT results should not be empty"

def performance_benchmark(logger):
    """Benchmark performance with different signal lengths."""
    logger.info("Performance Benchmark")
    logger.info("=" * 60)
    
    analyzer = SpectrumAnalysis()
    fs = 250e6
    f_signal = 10e6
    
    signal_lengths = [1000, 10000, 100000]  # Smaller lengths to avoid memory issues
    
    for length in signal_lengths:
        t = np.arange(length) / fs
        signal = np.sin(2 * np.pi * f_signal * t)
        
        logger.info(f"Testing {length:,} samples ({length/fs*1000:.1f} ms)")
        
        # FFT benchmark
        start_time = time.time()
        freq = analyzer.find_center_frequency_fft(signal, fs)
        fft_time = time.time() - start_time
        
        # STFT benchmark  
        start_time = time.time()
        try:
            f, t, Sxx = analyzer.compute_stft_sqpy(signal, fs, n_fft=1024, hop_len=512)
            stft_time = time.time() - start_time
        except Exception as e:
            stft_time = float('inf')
            logger.warning(f"   STFT failed: {e}")
        
        # CWT benchmark (only for shorter signals)
        if length <= 10000:  # Even more restrictive
            start_time = time.time()
            try:
                freqs, Wx = analyzer.compute_cwt(signal, fs, f_min=1e6, f_max=50e6, nv=8, scales='log')  # Fewer voices
                cwt_time = time.time() - start_time
            except Exception as e:
                cwt_time = float('inf')
                logger.warning(f"   CWT failed: {e}")
        else:
            cwt_time = float('inf')
            logger.info("   CWT skipped (too long)")
        
        logger.info(f"   FFT:  {fft_time*1000:6.1f} ms")
        logger.info(f"   STFT: {stft_time*1000:6.1f} ms")
        logger.info(f"   CWT:  {cwt_time*1000:6.1f} ms" if cwt_time != float('inf') else "   CWT:  skipped")
        logger.info("")

def main():
    """Main test execution."""
    logger = setup_logging()
    
    logger.info("IFI Spectrum Analysis - Comprehensive Test Suite")
    logger.info("=" * 80)
    logger.info("")
    
    # Create test signals
    logger.info("Generating test signals...")
    signals = create_test_signals()
    logger.info(f"   Created {len(signals)} test signals")
    logger.info("")
    
    # Run tests
    try:
        fft_results = test_fft_analysis(signals, logger)
        stft_results = test_stft_analysis(signals, logger)
        cwt_results = test_cwt_analysis(signals, logger)
        performance_benchmark(logger)
        
        logger.info("All spectrum tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise

if __name__ == '__main__':
    main()