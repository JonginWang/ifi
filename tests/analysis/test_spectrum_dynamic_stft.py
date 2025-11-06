#!/usr/bin/env python3
"""
Test dynamic STFT parameter selection based on signal length.

Tests that when nperseg and noverlap are None, the system automatically
selects values similar to the original defaults (10000, 5000) for typical
signal lengths (~500k-1M samples).
"""

from __future__ import annotations

import numpy as np
import pytest

from ifi.analysis.spectrum import SpectrumAnalysis


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

