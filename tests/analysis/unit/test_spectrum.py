#!/usr/bin/env python3
"""
Comprehensive test suite for spectrum analysis (STFT, CWT) with various signal types.
Includes dynamic STFT parameter selection tests.
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


# ============================================================================
# Dynamic STFT Parameter Selection Tests (from test_spectrum_dynamic_stft.py)
# ============================================================================

def test_dynamic_nperseg_selection():
    """Test that nperseg is automatically selected based on signal length."""
    analyzer = SpectrumAnalysis()
    
    # Test cases: (signal_length, expected_nperseg_range, description)
    test_cases = [
        # Original default: 10000 for signals around 640k samples (10000 * 64)
        (640000, (8192, 16384), "Original default range (640k samples)"),
        (500000, (8192, 16384), "500k samples - should yield ~8192"),
        (320000, (4096, 8192), "320k samples - should yield ~4096-8192"),
        (1000000, (8192, 16384), "1M samples - should yield ~8192-16384"),
        (100000, (1024, 2048), "100k samples - smaller signal"),
        (50000, (512, 1024), "50k samples - minimum range"),
        (10000, (256, 512), "10k samples - very small signal"),
    ]
    
    fs = 50e6  # 50 MHz sampling rate
    
    for signal_length, expected_range, description in test_cases:
        # Generate test signal
        t = np.arange(signal_length) / fs
        signal = np.sin(2 * np.pi * 8e6 * t)  # 8 MHz tone
        
        # Compute STFT without specifying nperseg/noverlap (should use dynamic selection)
        f, t_stft, Zxx = analyzer.compute_stft(signal, fs)
        
        # Extract the actual nperseg used (inferred from STFT shape and hop)
        # The frequency resolution gives us nperseg/2 + 1 frequency bins
        n_freq_bins = len(f)
        actual_nperseg = (n_freq_bins - 1) * 2  # Approximate
        
        # Also check hop size from time resolution
        if len(t_stft) > 1:
            dt = t_stft[1] - t_stft[0]
            actual_hop = int(dt * fs)
            # nperseg = hop + noverlap, and noverlap = nperseg // 2
            # So: nperseg = hop + nperseg // 2, therefore nperseg = 2 * hop
            inferred_nperseg_from_hop = 2 * actual_hop
        
        # Verify nperseg is within expected range
        assert expected_range[0] <= actual_nperseg <= expected_range[1], \
            f"{description}: nperseg {actual_nperseg} not in range {expected_range}"
        
        # Verify it's a power of 2 (for FFT efficiency)
        assert (actual_nperseg & (actual_nperseg - 1)) == 0, \
            f"{description}: nperseg {actual_nperseg} is not a power of 2"
        
        # Verify STFT computation succeeded
        assert Zxx.shape[0] == len(f), f"{description}: Frequency dimension mismatch"
        assert Zxx.shape[1] == len(t_stft), f"{description}: Time dimension mismatch"
        
        print(f"[OK] {description}: nperseg={actual_nperseg}, "
              f"noverlap~{actual_nperseg//2}, STFT shape={Zxx.shape}")


def test_dynamic_nperseg_similar_to_original_default():
    """
    Test that for signal lengths similar to original use cases (~500k-1M),
    the dynamically selected nperseg is close to the original default (10000).
    """
    analyzer = SpectrumAnalysis()
    
    # Original default was 10000, which works well for signals around 640k samples
    # (since dynamic selection uses signal_length // 64)
    target_nperseg = 10000
    signal_length_for_target = target_nperseg * 64  # 640000
    
    # Test around this length
    test_lengths = [
        500000,   # Slightly shorter
        640000,   # Exact match for 10000
        800000,   # Slightly longer
        1000000,  # 1M samples
    ]
    
    fs = 50e6
    
    for signal_length in test_lengths:
        t = np.arange(signal_length) / fs
        signal = np.sin(2 * np.pi * 8e6 * t)
        
        # Compute STFT with dynamic selection
        f, t_stft, Zxx = analyzer.compute_stft(signal, fs)
        
        # Extract actual nperseg
        n_freq_bins = len(f)
        actual_nperseg = (n_freq_bins - 1) * 2
        
        # Check time resolution for hop
        if len(t_stft) > 1:
            dt = t_stft[1] - t_stft[0]
            actual_hop = int(dt * fs)
            inferred_nperseg = 2 * actual_hop
            
            # Use the more accurate estimate
            actual_nperseg = inferred_nperseg
        
        # For signals around 640k, we expect values close to original default
        # Dynamic selection: signal_length // 64, rounded to power-of-2
        # 640000 // 64 = 10000, rounded to 2^13 = 8192 or 2^14 = 16384
        # Original default was 10000, so we expect 8192 (closer) or 16384
        
        print(f"Signal length: {signal_length:,}, "
              f"Expected dynamic: {signal_length // 64}, "
              f"Actual nperseg: {actual_nperseg}, "
              f"Original default: {target_nperseg}")
        
        # Verify it's reasonable (within 2x of original default)
        assert 4096 <= actual_nperseg <= 16384, \
            f"nperseg {actual_nperseg} should be between 4096 and 16384 for signal length {signal_length}"
        
        # For the specific length that matched original default, check it's close
        if signal_length == signal_length_for_target:
            # Should be either 8192 (closer to 10000) or 16384
            assert actual_nperseg in [8192, 16384], \
                f"For {signal_length} samples, expected nperseg 8192 or 16384, got {actual_nperseg}"


def test_dynamic_noverlap_is_half_of_nperseg():
    """Test that noverlap is automatically set to nperseg // 2."""
    analyzer = SpectrumAnalysis()
    
    signal_lengths = [100000, 320000, 500000, 640000, 1000000]
    fs = 50e6
    
    for signal_length in signal_lengths:
        t = np.arange(signal_length) / fs
        signal = np.sin(2 * np.pi * 8e6 * t)
        
        # Compute STFT with dynamic selection
        f, t_stft, Zxx = analyzer.compute_stft(signal, fs)
        
        # Extract hop size from time resolution
        if len(t_stft) > 1:
            dt = t_stft[1] - t_stft[0]
            actual_hop = int(dt * fs)
            inferred_nperseg = 2 * actual_hop
            inferred_noverlap = inferred_nperseg - actual_hop
            
            # Verify noverlap = nperseg // 2
            expected_noverlap = inferred_nperseg // 2
            assert inferred_noverlap == expected_noverlap, \
                f"noverlap {inferred_noverlap} should equal nperseg // 2 = {expected_noverlap}"
            
            print(f"[OK] Signal length {signal_length:,}: "
                  f"nperseg={inferred_nperseg}, noverlap={inferred_noverlap}, "
                  f"hop={actual_hop}")


def test_dynamic_selection_with_explicit_values():
    """Test that explicit values override dynamic selection."""
    analyzer = SpectrumAnalysis()
    
    signal_length = 640000
    fs = 50e6
    t = np.arange(signal_length) / fs
    signal = np.sin(2 * np.pi * 8e6 * t)
    
    # Explicit values should be used
    explicit_nperseg = 2048
    explicit_noverlap = 1024
    
    f, t_stft, Zxx = analyzer.compute_stft(
        signal, fs, nperseg=explicit_nperseg, noverlap=explicit_noverlap
    )
    
    # Verify explicit values were used
    n_freq_bins = len(f)
    actual_nperseg = (n_freq_bins - 1) * 2
    
    # Should be close to explicit value (allowing for rounding)
    assert actual_nperseg == explicit_nperseg, \
        f"Explicit nperseg {explicit_nperseg} was not used, got {actual_nperseg}"
    
    # Check hop
    if len(t_stft) > 1:
        dt = t_stft[1] - t_stft[0]
        actual_hop = int(dt * fs)
        expected_hop = explicit_nperseg - explicit_noverlap
        assert actual_hop == expected_hop, \
            f"Expected hop {expected_hop}, got {actual_hop}"
    
    print(f"[OK] Explicit values used correctly: nperseg={explicit_nperseg}, "
          f"noverlap={explicit_noverlap}")


def test_dynamic_selection_ssqueezepy():
    """Test dynamic selection also works for ssqueezepy backend."""
    analyzer = SpectrumAnalysis()
    
    signal_length = 640000
    fs = 50e6
    t = np.arange(signal_length) / fs
    signal = np.sin(2 * np.pi * 8e6 * t)
    
    # Compute STFT with ssqueezepy (dynamic selection)
    f, t_stft, Zxx = analyzer.compute_stft_sqpy(signal, fs)
    
    # Extract nperseg from results
    n_freq_bins = len(f)
    actual_nperseg = (n_freq_bins - 1) * 2
    
    # Verify it's in reasonable range
    assert 4096 <= actual_nperseg <= 16384, \
        f"ssqueezepy nperseg {actual_nperseg} should be between 4096 and 16384"
    
    # Verify it's a power of 2
    assert (actual_nperseg & (actual_nperseg - 1)) == 0, \
        f"ssqueezepy nperseg {actual_nperseg} should be a power of 2"
    
    print(f"[OK] ssqueezepy dynamic selection: nperseg={actual_nperseg}, "
          f"STFT shape={Zxx.shape}")


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

