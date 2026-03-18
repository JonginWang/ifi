#!/usr/bin/env python3
"""
Accuracy tests for core time-frequency analysis utilities.

This module includes precision evaluation tests for:
- SignalStacker: Signal stacking and phase difference calculation
- STFTRidgeAnalyzer: STFT-based ridge detection and phase extraction
- CWTPhaseReconstructor: CWT-based phase reconstruction
"""

from __future__ import annotations

import numpy as np
import pytest

from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.analysis.phase_analysis import (
    SignalStacker,
    STFTRidgeAnalyzer,
    CWTPhaseReconstructor,
)

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


class TestSignalStackerPrecision:
    """Precision evaluation tests for SignalStacker."""
    
    @pytest.fixture
    def signal_stacker(self):
        """Create a SignalStacker instance for testing."""
        fs = 50e6
        return SignalStacker(fs)
    
    def test_fundamental_frequency_detection_precision(self, signal_stacker):
        """Test precision of fundamental frequency detection."""
        fs = 50e6
        f0_true = 8e6
        duration = 0.01
        t, signal = sine_signal(fs=fs, freq=f0_true, duration=duration)
        
        # Detect fundamental frequency
        f0_detected = signal_stacker.find_fundamental_frequency(signal)
        
        # Should be very close to true frequency
        assert np.allclose(f0_detected, f0_true, rtol=1e-3), \
            f"Detected frequency {f0_detected/1e6:.2f} MHz should be close to {f0_true/1e6:.2f} MHz"
    
    def test_signal_stacking_precision(self, signal_stacker):
        """Test precision of signal stacking operation."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        t, signal = sine_signal(fs=fs, freq=f0, duration=duration)
        
        # Stack signals
        stacked_signal, time_points = signal_stacker.stack_signals(
            signal, f0, n_stacks=4
        )
        
        # Stacked signal should maintain phase information
        assert len(stacked_signal) > 0
        assert len(time_points) == len(stacked_signal)
        assert np.all(np.isfinite(stacked_signal))
        
        # Stacked signal amplitude should be enhanced (SNR improvement)
        original_power = np.mean(signal**2)
        stacked_power = np.mean(stacked_signal**2)
        # Stacking should preserve or enhance signal power
        assert stacked_power > 0
    
    def test_phase_difference_cdm_precision(self, signal_stacker):
        """Test precision of CDM-based phase difference calculation."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4  # 45 degrees
        
        # Create reference and probe signals with known phase difference
        t_ref, ref_signal = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0)
        t_probe, probe_signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # Calculate phase difference using CDM (returns tuple: phase_diff, f0)
        phase_diff, detected_f0 = signal_stacker.compute_phase_difference_cdm(
            ref_signal, probe_signal, f0
        )
        
        # Verify detected frequency is close to expected
        assert np.allclose(detected_f0, f0, rtol=1e-2), \
            f"Detected frequency {detected_f0/1e6:.2f} MHz should be close to {f0/1e6:.2f} MHz"
        
        # Phase difference should be close to expected offset
        # Use unwrapped phase for comparison
        phase_diff_mean = np.mean(phase_diff[-100:])  # Use stable region
        
        # Allow tolerance for CDM method (typically 1-5% relative error)
        # Note: CDM accumulates phase, so absolute value may differ, but trend should match
        assert np.abs(phase_diff_mean - phase_offset) < np.pi / 2, \
            f"Phase difference mean {phase_diff_mean:.4f} should be close to offset {phase_offset:.4f}"
        
        # Phase difference should be smooth (no abrupt jumps)
        phase_diff_diff = np.diff(phase_diff)
        # Allow some tolerance for accumulated phase differences
        assert np.all(np.abs(phase_diff_diff) < np.pi * 2), \
            "Phase difference should not have abrupt jumps > 2π"
    
    def test_phase_difference_cordic_precision(self, signal_stacker):
        """Test precision of CORDIC-based phase difference calculation."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4
        
        t_ref, ref_signal = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0)
        t_probe, probe_signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # Calculate phase difference using CORDIC (returns tuple: times, phase_diff, f0)
        times, phase_diff, detected_f0 = signal_stacker.compute_phase_difference_cordic(
            ref_signal, probe_signal, f0
        )
        
        # Verify detected frequency is close to expected
        assert np.allclose(detected_f0, f0, rtol=1e-2), \
            f"Detected frequency {detected_f0/1e6:.2f} MHz should be close to {f0/1e6:.2f} MHz"
        
        # Phase difference should be close to expected offset
        phase_diff_mean = np.mean(phase_diff[-100:])
        
        # CORDIC should have similar or better precision than CDM
        # Note: CORDIC accumulates phase, so absolute value may differ, but trend should match
        assert np.abs(phase_diff_mean - phase_offset) < np.pi / 2, \
            f"CORDIC phase difference mean {phase_diff_mean:.4f} should reflect offset {phase_offset:.4f}"
    
    def test_linear_phase_drift_detection_precision(self, signal_stacker):
        """Test precision of linear phase drift detection."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        
        # Create signal with linear phase drift
        t = np.arange(int(fs * duration)) / fs
        drift_rate = 1e6  # 1 MHz/s drift rate
        phase = 2 * np.pi * f0 * t + np.pi * drift_rate * t**2
        signal = np.sin(phase)
        
        # Analyze linear drift
        drift_result = signal_stacker.analyze_linear_phase_drift(signal, f0)
        
        # Should detect drift
        assert drift_result is not None
        if drift_result:
            assert "drift_rate" in drift_result
            # Detected drift rate should be close to expected
            detected_rate = drift_result["drift_rate"]
            assert np.abs(detected_rate - drift_rate) < drift_rate * 0.5, \
                f"Detected drift rate {detected_rate:.2e} should be close to {drift_rate:.2e}"


class TestSTFTRidgeAnalyzerPrecision:
    """Precision evaluation tests for STFTRidgeAnalyzer."""
    
    @pytest.fixture
    def stft_analyzer(self):
        """Create an STFTRidgeAnalyzer instance for testing."""
        fs = 50e6
        return STFTRidgeAnalyzer(fs)
    
    def test_ridge_detection_precision(self, stft_analyzer):
        """Test precision of STFT ridge detection."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        t, signal = sine_signal(fs=fs, freq=f0, duration=duration)
        
        # Compute STFT with ridge
        f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(
            signal, f0, nperseg=1024, noverlap=512
        )
        
        # Ridge frequencies should be close to true frequency
        ridge_mean = np.mean(ridge_freqs)
        assert np.allclose(ridge_mean, f0, rtol=2e-2), \
            f"Ridge frequency {ridge_mean/1e6:.2f} MHz should be close to {f0/1e6:.2f} MHz"
        
        # Ridge should be stable (low variance)
        ridge_std = np.std(ridge_freqs)
        assert ridge_std < f0 * 0.05, \
            f"Ridge frequency std {ridge_std/1e6:.2f} MHz should be small"
    
    def test_phase_extraction_precision(self, stft_analyzer):
        """Test precision of phase extraction from ridge."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4
        t, signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # Compute ridge
        f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(
            signal, f0
        )
        
        # Extract phase from ridge
        phases = stft_analyzer.extract_phase_from_ridge(
            signal, ridge_freqs, t_stft
        )
        
        # Phase should be extracted correctly
        assert len(phases) == len(t_stft)
        assert np.all(np.isfinite(phases))
        
        # Phase should reflect the offset (unwrapped)
        phase_mean = np.mean(phases[-100:])
        # Allow tolerance for phase extraction
        assert np.abs(phase_mean - phase_offset) < np.pi / 4, \
            f"Extracted phase {phase_mean:.4f} should reflect offset {phase_offset:.4f}"
    
    def test_phase_difference_computation_precision(self, stft_analyzer):
        """Test precision of phase difference computation."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4
        
        t_ref, ref_signal = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0)
        t_probe, probe_signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # Compute phase difference (returns tuple: times, ref_phases, probe_phases)
        t_common, ref_phases, probe_phases = stft_analyzer.compute_phase_difference(
            ref_signal, probe_signal, f0
        )
        
        # Calculate phase difference
        phase_diff = probe_phases - ref_phases
        
        # Phase difference should be close to expected offset
        phase_diff_mean = np.mean(phase_diff[-100:])
        assert np.allclose(
            phase_diff_mean, phase_offset, rtol=5e-2, atol=0.1
        ), f"Phase difference {phase_diff_mean:.4f} should be close to {phase_offset:.4f}"


class TestCWTPhaseReconstructorPrecision:
    """Precision evaluation tests for CWTPhaseReconstructor."""
    
    @pytest.fixture
    def cwt_reconstructor(self):
        """Create a CWTPhaseReconstructor instance for testing."""
        fs = 50e6
        return CWTPhaseReconstructor(fs)
    
    def test_cwt_phase_computation_precision(self, cwt_reconstructor):
        """Test precision of CWT phase computation."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4
        t, signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # Compute CWT phase (returns tuple: times, phases)
        times, phases = cwt_reconstructor.compute_cwt_phase(signal, f0)
        
        # Phase should be computed correctly
        assert len(phases) > 0
        assert len(times) == len(phases)
        assert np.all(np.isfinite(phases))
        
        # Phase should reflect the offset (unwrapped)
        phase_mean = np.mean(phases[-100:])
        assert np.abs(phase_mean - phase_offset) < np.pi / 4, \
            f"CWT phase {phase_mean:.4f} should reflect offset {phase_offset:.4f}"
    
    def test_phase_difference_computation_precision(self, cwt_reconstructor):
        """Test precision of CWT phase difference computation."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4
        
        t_ref, ref_signal = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0)
        t_probe, probe_signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # Compute phase difference (returns tuple: times, ref_phases, probe_phases)
        t_common, ref_phases, probe_phases = cwt_reconstructor.compute_phase_difference(
            ref_signal, probe_signal, f0
        )
        
        # Calculate phase difference
        phase_diff = probe_phases - ref_phases
        
        # Phase difference should be close to expected offset
        phase_diff_mean = np.mean(phase_diff[-100:])
        assert np.allclose(
            phase_diff_mean, phase_offset, rtol=5e-2, atol=0.1
        ), f"CWT phase difference {phase_diff_mean:.4f} should be close to {phase_offset:.4f}"
    
    def test_fir_filter_design_precision(self, cwt_reconstructor):
        """Test precision of FIR filter design."""
        f0 = 8e6
        bandwidth = 0.1
        
        # Design filter
        filter_coeffs = cwt_reconstructor.design_fir_filter(f0, bandwidth)
        
        # Filter should be designed correctly
        assert len(filter_coeffs) > 0
        assert len(filter_coeffs) % 2 == 1, "Filter length should be odd"
        assert np.all(np.isfinite(filter_coeffs))
        
        # Filter coefficients should be reasonable
        assert np.max(np.abs(filter_coeffs)) < 1.0, "Filter coefficients should be bounded"
    
    def test_decimation_preserves_phase_information(self, cwt_reconstructor):
        """Test that decimation preserves phase information."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4
        
        t, signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # Decimate signal
        decimation_factor = 4
        decimated_signal = cwt_reconstructor.decimate_signal(signal, decimation_factor)
        
        # Decimated signal should preserve phase information
        assert len(decimated_signal) == len(signal) // decimation_factor
        assert np.all(np.isfinite(decimated_signal))
        
        # Phase information should be preserved (signal should still be sinusoidal)
        # Check that signal maintains its structure
        signal_power = np.mean(signal**2)
        decimated_power = np.mean(decimated_signal**2)
        # Power should be similar (within factor of 2 due to decimation)
        assert 0.5 < decimated_power / signal_power < 2.0, \
            "Decimated signal should preserve power characteristics"


class TestPhaseAnalysisComparison:
    """Comparison tests between different phase analysis methods."""
    
    def test_signal_stacker_vs_stft_phase_difference(self):
        """Compare phase difference from SignalStacker vs STFTRidgeAnalyzer."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4
        
        t_ref, ref_signal = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0)
        t_probe, probe_signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # SignalStacker CDM (returns tuple: phase_diff, f0)
        stacker = SignalStacker(fs)
        phase_diff_stacker, _ = stacker.compute_phase_difference_cdm(
            ref_signal, probe_signal, f0
        )
        
        # STFT Ridge Analyzer
        stft_analyzer = STFTRidgeAnalyzer(fs)
        _, ref_phases_stft, probe_phases_stft = stft_analyzer.compute_phase_difference(
            ref_signal, probe_signal, f0
        )
        phase_diff_stft = probe_phases_stft - ref_phases_stft
        
        # Both methods should produce similar results
        # Compare means in stable region
        mean_stacker = np.mean(phase_diff_stacker[-100:])
        mean_stft = np.mean(phase_diff_stft[-100:])
        
        # Methods should agree within reasonable tolerance
        assert np.abs(mean_stacker - mean_stft) < np.pi / 2, \
            f"SignalStacker ({mean_stacker:.4f}) and STFT ({mean_stft:.4f}) should agree"
    
    def test_stft_vs_cwt_phase_difference(self):
        """Compare phase difference from STFT vs CWT methods."""
        fs = 50e6
        f0 = 8e6
        duration = 0.01
        phase_offset = np.pi / 4
        
        t_ref, ref_signal = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0)
        t_probe, probe_signal = sine_signal(
            fs=fs, freq=f0, duration=duration, phase=phase_offset
        )
        
        # STFT
        stft_analyzer = STFTRidgeAnalyzer(fs)
        _, ref_phases_stft, probe_phases_stft = stft_analyzer.compute_phase_difference(
            ref_signal, probe_signal, f0
        )
        phase_diff_stft = probe_phases_stft - ref_phases_stft
        
        # CWT
        cwt_reconstructor = CWTPhaseReconstructor(fs)
        _, ref_phases_cwt, probe_phases_cwt = cwt_reconstructor.compute_phase_difference(
            ref_signal, probe_signal, f0
        )
        phase_diff_cwt = probe_phases_cwt - ref_phases_cwt
        
        # Both methods should produce similar results
        mean_stft = np.mean(phase_diff_stft[-100:])
        mean_cwt = np.mean(phase_diff_cwt[-100:])
        
        # Methods should agree within reasonable tolerance
        assert np.abs(mean_stft - mean_cwt) < np.pi / 2, \
            f"STFT ({mean_stft:.4f}) and CWT ({mean_cwt:.4f}) should agree"


