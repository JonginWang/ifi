#!/usr/bin/env python3
"""
Accuracy tests for core time-frequency analysis utilities.
"""

from __future__ import annotations

import numpy as np

from ifi.analysis.spectrum import SpectrumAnalysis

from tests.analysis.fixtures.synthetic_signals import sine_signal, linear_chirp


def test_stft_detects_single_tone_peak_close_to_true_freq():
    fs = 250e6
    f0 = 10e6
    duration = 0.010  # 10 ms -> enough frames with dynamic nperseg

    _, x = sine_signal(fs=fs, freq=f0, duration=duration)

    analyzer = SpectrumAnalysis()
    f, t, Z = analyzer.compute_stft(x, fs)

    # Magnitude spectrogram
    S = np.abs(Z)

    # Peak per time frame
    peak_idxs = np.argmax(S, axis=0)
    peak_freqs = f[peak_idxs]
    _, ridge_idxs = analyzer.find_freq_ridge(Z, f, method='stft', return_idx=True)
    ridge_freqs = f[ridge_idxs]
    # Assert mean peak frequency close to f0 within 1% relative tolerance
    assert np.isfinite(peak_freqs).all()
    assert np.allclose(peak_freqs.mean(), f0, rtol=1e-2)
    assert np.allclose(peak_freqs.mean(), ridge_freqs.mean(), rtol=1e-2)


def test_cwt_tracks_linear_chirp_ridge_within_tolerance():
    fs = 20e6
    f0 = 2e6
    f1 = 6e6
    duration = 0.010  # 10 ms

    t, x = linear_chirp(fs=fs, f0=f0, f1=f1, duration=duration)

    analyzer = SpectrumAnalysis()
    freqs_cwt, Wx = analyzer.compute_cwt(x, fs, wavelet="gmw", nv=32)

    # Ridge detection: argmax over frequency for each time column
    S = np.abs(Wx)
    peak_idxs = np.argmax(S, axis=0)
    peak_freqs = freqs_cwt[peak_idxs]
    _, ridge_idxs = analyzer.find_freq_ridge(Wx, freqs_cwt, method='cwt', return_idx=True)
    ridge_freqs = freqs_cwt[ridge_idxs]
    # Expected instantaneous frequency: linear ramp from f0 to f1 across samples
    expected_freqs = f0 + (f1 - f0) * (t / t[-1])

    # Align lengths (Wx time axis equals len(x))
    n = min(len(peak_freqs), len(expected_freqs))
    peak_freqs = peak_freqs[:n]
    expected_freqs = expected_freqs[:n]

    # Allow a modest tolerance due to wavelet resolution
    assert np.isfinite(peak_freqs).all()
    assert np.allclose(peak_freqs, expected_freqs, rtol=5e-2, atol=1e5)
    assert np.allclose(peak_freqs.mean(), ridge_freqs.mean(), rtol=1e-2, atol=1e4)


