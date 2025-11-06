#!/usr/bin/env python3
"""
Pytest tests for phase detection algorithms (STFT Ridge and CWT).

This test suite covers:
- STFTRidgeAnalyzer class and its methods
- CWTPhaseReconstructor class and its methods
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from ifi.analysis.phase_analysis import STFTRidgeAnalyzer, CWTPhaseReconstructor


class TestSTFTRidgeAnalyzer:
    """Test suite for STFTRidgeAnalyzer class."""

    @pytest.fixture
    def stft_analyzer(self):
        """Create STFTRidgeAnalyzer instance."""
        return STFTRidgeAnalyzer(fs=50e6)

    def test_stft_analyzer_initialization(self, stft_analyzer):
        """Test STFTRidgeAnalyzer initialization."""
        assert stft_analyzer is not None
        assert stft_analyzer.fs == 50e6
        assert hasattr(stft_analyzer, "spectrum_analyzer")
        assert hasattr(stft_analyzer, "phase_converter")

    def test_compute_stft_with_ridge_basic(self, stft_analyzer, reference_signal):
        """Test STFT ridge computation with basic signal."""
        t, signal = reference_signal
        f0 = 8e6  # 8 MHz

        f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(signal, f0)

        # Check output shapes
        assert len(f) > 0
        assert len(t_stft) > 0
        assert len(ridge_freqs) == len(t_stft)

        # Ridge frequencies should be close to target frequency
        assert np.all(np.abs(ridge_freqs - f0) < f0 * 0.5)  # Within 50% tolerance

    def test_compute_stft_with_ridge_custom_params(self, stft_analyzer, reference_signal):
        """Test STFT ridge computation with custom parameters."""
        t, signal = reference_signal
        f0 = 8e6
        nperseg = 512
        noverlap = 256

        f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(
            signal, f0, nperseg=nperseg, noverlap=noverlap
        )

        assert len(f) > 0
        assert len(t_stft) > 0
        assert len(ridge_freqs) == len(t_stft)

    def test_extract_phase_from_ridge_basic(self, stft_analyzer, reference_signal):
        """Test phase extraction from ridge frequencies."""
        t, signal = reference_signal
        f0 = 8e6

        # First compute STFT and ridge
        f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(signal, f0)

        # Extract phases
        phases = stft_analyzer.extract_phase_from_ridge(signal, ridge_freqs, t_stft)

        # Check output
        assert len(phases) == len(t_stft)
        assert np.all(np.isfinite(phases))
        # Phases should be in [-π, π] range (before unwrap)
        assert np.all(np.abs(phases) <= np.pi + 0.1)  # Small tolerance

    def test_compute_phase_difference_basic(self, stft_analyzer, reference_signal, probe_signal_constant_phase):
        """Test phase difference computation using STFT ridge method."""
        t, ref = reference_signal
        _, probe = probe_signal_constant_phase
        f0 = 8e6

        times, ref_phases, probe_phases = stft_analyzer.compute_phase_difference(
            ref, probe, f0
        )

        # Check output shapes
        assert len(times) > 0
        assert len(ref_phases) == len(times)
        assert len(probe_phases) == len(times)

        # Phases should be finite and unwrapped
        assert np.all(np.isfinite(ref_phases))
        assert np.all(np.isfinite(probe_phases))

        # Phase difference should be computed
        phase_diff = probe_phases - ref_phases
        assert len(phase_diff) == len(times)
        assert np.all(np.isfinite(phase_diff))

    def test_compute_phase_difference_linear_phase(self, stft_analyzer, reference_signal, probe_signal_linear_phase):
        """Test phase difference computation with linear phase evolution."""
        t, ref = reference_signal
        _, probe = probe_signal_linear_phase
        f0 = 8e6

        times, ref_phases, probe_phases = stft_analyzer.compute_phase_difference(
            ref, probe, f0
        )

        assert len(times) > 0
        assert np.all(np.isfinite(ref_phases))
        assert np.all(np.isfinite(probe_phases))

        # For linear phase, phase difference should show a trend
        phase_diff = probe_phases - ref_phases
        phase_diff_diff = np.diff(phase_diff)
        # Should not be all zeros (unless perfectly constant)
        assert np.any(np.abs(phase_diff_diff) > 1e-10)

    def test_compute_stft_with_ridge_short_signal(self, stft_analyzer):
        """Test STFT ridge computation with very short signal."""
        fs = 50e6
        f0 = 8e6
        n = 100  # Very short signal
        t = np.arange(n) / fs
        signal = np.sin(2 * np.pi * f0 * t)

        # Should handle short signals gracefully
        # For very short signals, use smaller nperseg and noverlap
        try:
            f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(
                signal, f0, nperseg=64, noverlap=32
            )
            assert len(f) > 0
            assert len(ridge_freqs) > 0
        except ValueError:
            # If signal is too short even for minimal params, skip test
            pytest.skip("Signal too short for STFT computation")

    def test_compute_stft_with_ridge_zero_signal(self, stft_analyzer):
        """Test STFT ridge computation with zero signal."""
        fs = 50e6
        f0 = 8e6
        n = 1000
        signal = np.zeros(n)

        # Should handle zero signal (may produce warnings/errors)
        try:
            f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(signal, f0)
            # If successful, check outputs
            assert len(f) > 0
            assert len(ridge_freqs) > 0
        except (ValueError, RuntimeError):
            # Raising an error for zero signal is acceptable
            pass

    def test_compute_phase_difference_mismatched_lengths(self, stft_analyzer, reference_signal):
        """Test phase difference computation with mismatched signal lengths."""
        t, ref = reference_signal
        probe = ref[:-100]  # Shorter probe signal
        f0 = 8e6

        # Should handle mismatched lengths (may use shorter length)
        times, ref_phases, probe_phases = stft_analyzer.compute_phase_difference(
            ref, probe, f0
        )

        # Outputs should have consistent lengths
        assert len(ref_phases) == len(probe_phases)
        assert len(times) == len(ref_phases)


class TestCWTPhaseReconstructor:
    """Test suite for CWTPhaseReconstructor class."""

    @pytest.fixture
    def cwt_reconstructor(self):
        """Create CWTPhaseReconstructor instance."""
        return CWTPhaseReconstructor(fs=50e6)

    def test_cwt_reconstructor_initialization(self, cwt_reconstructor):
        """Test CWTPhaseReconstructor initialization."""
        assert cwt_reconstructor is not None
        assert cwt_reconstructor.fs == 50e6
        assert hasattr(cwt_reconstructor, "spectrum_analyzer")

    def test_design_fir_filter_basic(self, cwt_reconstructor):
        """Test FIR filter design."""
        f0 = 8e6
        filter_coeffs = cwt_reconstructor.design_fir_filter(f0)

        # Check filter coefficients
        assert len(filter_coeffs) > 0
        assert np.all(np.isfinite(filter_coeffs))

        # Filter length should be odd (Type I filter requirement)
        assert len(filter_coeffs) % 2 == 1, "Filter length should be odd for Type I filter"
        
        # Filter length should be reasonable (remezord calculates optimal length)
        # Typical range: 50-1000 taps depending on frequency requirements
        assert 50 <= len(filter_coeffs) <= 1000, f"Filter length {len(filter_coeffs)} out of reasonable range"

    def test_design_fir_filter_custom_bandwidth(self, cwt_reconstructor):
        """Test FIR filter design with custom bandwidth."""
        f0 = 8e6
        bandwidth = 0.2  # 20% bandwidth

        filter_coeffs = cwt_reconstructor.design_fir_filter(f0, bandwidth=bandwidth)

        assert len(filter_coeffs) > 0
        assert np.all(np.isfinite(filter_coeffs))

    def test_decimate_signal_basic(self, cwt_reconstructor, reference_signal):
        """Test signal decimation."""
        t, signal = reference_signal
        decimation_factor = 4

        decimated = cwt_reconstructor.decimate_signal(signal, decimation_factor)

        # Decimated signal should be shorter
        assert len(decimated) <= len(signal) / decimation_factor + 1
        assert np.all(np.isfinite(decimated))

    def test_decimate_signal_custom_factor(self, cwt_reconstructor, reference_signal):
        """Test signal decimation with custom decimation factor."""
        t, signal = reference_signal
        decimation_factor = 8

        decimated = cwt_reconstructor.decimate_signal(signal, decimation_factor)

        assert len(decimated) <= len(signal) / decimation_factor + 1
        assert np.all(np.isfinite(decimated))

    def test_compute_cwt_phase_basic(self, cwt_reconstructor, reference_signal):
        """Test CWT phase computation."""
        t, signal = reference_signal
        f0 = 8e6

        times, phases = cwt_reconstructor.compute_cwt_phase(signal, f0)

        # Check outputs
        assert len(times) > 0
        assert len(phases) == len(times)
        assert np.all(np.isfinite(phases))
        # Phases should be in reasonable range
        assert np.all(np.abs(phases) <= np.pi + 0.1)

    def test_compute_cwt_phase_custom_voices(self, cwt_reconstructor, reference_signal):
        """Test CWT phase computation with custom voices_per_octave."""
        t, signal = reference_signal
        f0 = 8e6
        voices_per_octave = 32

        times, phases = cwt_reconstructor.compute_cwt_phase(
            signal, f0, voices_per_octave=voices_per_octave
        )

        assert len(times) > 0
        assert len(phases) == len(times)
        assert np.all(np.isfinite(phases))

    def test_compute_phase_difference_basic(self, cwt_reconstructor, reference_signal, probe_signal_constant_phase):
        """Test phase difference computation using CWT method."""
        t, ref = reference_signal
        _, probe = probe_signal_constant_phase
        f0 = 8e6

        times, ref_phases, probe_phases = cwt_reconstructor.compute_phase_difference(
            ref, probe, f0
        )

        # Check output shapes
        assert len(times) > 0
        assert len(ref_phases) == len(times)
        assert len(probe_phases) == len(times)

        # Phases should be finite and unwrapped
        assert np.all(np.isfinite(ref_phases))
        assert np.all(np.isfinite(probe_phases))

        # Phase difference should be computed
        phase_diff = probe_phases - ref_phases
        assert len(phase_diff) == len(times)
        assert np.all(np.isfinite(phase_diff))

    def test_compute_phase_difference_custom_decimation(self, cwt_reconstructor, reference_signal, probe_signal_constant_phase):
        """Test phase difference computation with custom decimation factor."""
        t, ref = reference_signal
        _, probe = probe_signal_constant_phase
        f0 = 8e6
        decimation_factor = 8

        times, ref_phases, probe_phases = cwt_reconstructor.compute_phase_difference(
            ref, probe, f0, decimation_factor=decimation_factor
        )

        assert len(times) > 0
        assert len(ref_phases) == len(times)
        assert len(probe_phases) == len(times)
        assert np.all(np.isfinite(ref_phases))
        assert np.all(np.isfinite(probe_phases))

    def test_decimate_signal_very_short(self, cwt_reconstructor):
        """Test decimation with very short signal."""
        signal = np.sin(2 * np.pi * 8e6 * np.arange(50) / 50e6)
        decimation_factor = 4

        # Should handle short signals (may produce very short output)
        # Very short signals may fail due to filter padding requirements
        try:
            decimated = cwt_reconstructor.decimate_signal(signal, decimation_factor)
            assert len(decimated) > 0
            assert np.all(np.isfinite(decimated))
        except ValueError:
            # If signal is too short for filter padding, skip test
            pytest.skip("Signal too short for decimation filter")

    def test_compute_cwt_phase_short_signal(self, cwt_reconstructor):
        """Test CWT phase computation with short signal."""
        fs = 50e6
        n = 200  # Short signal
        t = np.arange(n) / fs
        signal = np.sin(2 * np.pi * 8e6 * t)
        f0 = 8e6

        times, phases = cwt_reconstructor.compute_cwt_phase(signal, f0)

        # Should handle short signals
        assert len(times) > 0
        assert len(phases) == len(times)

    def test_compute_phase_difference_linear_phase(self, cwt_reconstructor, reference_signal, probe_signal_linear_phase):
        """Test phase difference computation with linear phase evolution."""
        t, ref = reference_signal
        _, probe = probe_signal_linear_phase
        f0 = 8e6

        times, ref_phases, probe_phases = cwt_reconstructor.compute_phase_difference(
            ref, probe, f0
        )

        assert len(times) > 0
        assert np.all(np.isfinite(ref_phases))
        assert np.all(np.isfinite(probe_phases))

        # For linear phase, phase difference should show a trend
        phase_diff = probe_phases - ref_phases
        phase_diff_diff = np.diff(phase_diff)
        # Should not be all zeros
        assert np.any(np.abs(phase_diff_diff) > 1e-10)

    def test_compute_cwt_phase_noisy_signal(self, cwt_reconstructor, noisy_signal):
        """Test CWT phase computation with noisy signal."""
        t, signal = noisy_signal
        f0 = 8e6

        times, phases = cwt_reconstructor.compute_cwt_phase(signal, f0)

        # Should handle noisy signals
        assert len(times) > 0
        assert len(phases) == len(times)
        assert np.all(np.isfinite(phases))

    def test_compute_phase_difference_mismatched_lengths(self, cwt_reconstructor, reference_signal):
        """Test phase difference computation with mismatched signal lengths."""
        t, ref = reference_signal
        probe = ref[:-100]  # Shorter probe signal
        f0 = 8e6

        # Should handle mismatched lengths
        times, ref_phases, probe_phases = cwt_reconstructor.compute_phase_difference(
            ref, probe, f0
        )

        # Outputs should have consistent lengths
        assert len(ref_phases) == len(probe_phases)
        assert len(times) == len(ref_phases)


class TestPhaseDetectionAlgorithmsIntegration:
    """Integration tests for phase detection algorithms."""

    def test_stft_vs_cwt_comparison(self, reference_signal, probe_signal_constant_phase):
        """Compare STFT and CWT methods on the same signals."""
        t, ref = reference_signal
        _, probe = probe_signal_constant_phase
        f0 = 8e6
        fs = 50e6

        stft_analyzer = STFTRidgeAnalyzer(fs)
        cwt_reconstructor = CWTPhaseReconstructor(fs)

        # Compute phase differences
        times_stft, ref_phases_stft, probe_phases_stft = stft_analyzer.compute_phase_difference(
            ref, probe, f0
        )
        times_cwt, ref_phases_cwt, probe_phases_cwt = cwt_reconstructor.compute_phase_difference(
            ref, probe, f0
        )

        # Both should produce valid results
        assert len(times_stft) > 0
        assert len(times_cwt) > 0

        # Phase differences should be finite
        phase_diff_stft = probe_phases_stft - ref_phases_stft
        phase_diff_cwt = probe_phases_cwt - ref_phases_cwt

        assert np.all(np.isfinite(phase_diff_stft))
        assert np.all(np.isfinite(phase_diff_cwt))

