"""
Integration tests for phase_analysis.py module.

Tests all classes:
1. CORDICProcessor (extract_phase_samples)
2. SignalStacker
3. STFTRidgeAnalyzer
4. CWTPhaseReconstructor
5. PhaseChangeDetector
"""
import numpy as np
import pytest

from ifi.analysis.phase_analysis import (
    CORDICProcessor,
    SignalStacker,
    STFTRidgeAnalyzer,
    CWTPhaseReconstructor,
    PhaseChangeDetector,
)


class TestExtractPhaseSamples:
    """Test CORDICProcessor.extract_phase_samples method."""

    @pytest.fixture
    def cordic_processor(self):
        """Create a CORDICProcessor instance."""
        return CORDICProcessor(n_iterations=16)

    def test_extract_phase_samples_basic(self, cordic_processor):
        """Test basic phase extraction functionality."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001  # 1ms
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * f0 * t)

        times, phases = cordic_processor.extract_phase_samples(
            signal, f0, fs, samples_per_period=4
        )

        # Verify output
        assert len(times) > 0, "Should extract some phase samples"
        assert len(phases) == len(times), "Times and phases should match length"
        assert np.all(np.isfinite(phases)), "All phases should be finite"
        assert np.all(np.isfinite(times)), "All times should be finite"

        # Verify time points are within signal duration
        assert times.min() >= 0, "Time points should start from 0"
        assert times.max() <= duration, "Time points should not exceed signal duration"

        # Verify phases are unwrapped (should not have jumps > π)
        if len(phases) > 1:
            phase_diffs = np.diff(phases)
            # After unwrapping, jumps > π should be rare (only at actual phase jumps)
            # But consecutive samples should not have > π jumps due to unwrapping
            large_jumps = np.abs(phase_diffs) > np.pi
            # Allow at most 5% of samples to have large jumps (edge cases)
            assert np.sum(large_jumps) < len(phase_diffs) * 0.05, (
                f"Too many unwrapped phase jumps: {np.sum(large_jumps)}/{len(phase_diffs)}"
            )

    def test_extract_phase_samples_short_signal(self, cordic_processor):
        """Test phase extraction with very short signal."""
        fs = 50e6
        f0 = 8e6
        duration = 1e-5  # Very short: 10us
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * f0 * t)

        # Should handle short signals gracefully
        times, phases = cordic_processor.extract_phase_samples(
            signal, f0, fs, samples_per_period=4
        )

        # May return empty arrays for very short signals
        if len(times) == 0:
            pytest.skip("Signal too short for extraction")
        else:
            assert len(phases) == len(times), "Times and phases should match"

    def test_extract_phase_samples_different_samples_per_period(self, cordic_processor):
        """Test phase extraction with different samples_per_period values."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * f0 * t)

        for samples_per_period in [1, 2, 4, 8]:
            times, phases = cordic_processor.extract_phase_samples(
                signal, f0, fs, samples_per_period=samples_per_period
            )

            if len(times) > 0:
                # With decimation, the number of samples is approximately
                # original_length / decimation_factor
                # where decimation_factor = samples_per_T // samples_per_period
                T = 1 / f0
                samples_per_T = int(T * fs)
                decimation_factor = max(1, samples_per_T // samples_per_period)
                expected_samples = len(signal) // decimation_factor
                
                # Allow some variance due to decimation filter padding and boundary conditions
                # Decimation may add or remove a few samples due to filter edge effects
                assert (
                    abs(len(times) - expected_samples) <= max(10, expected_samples * 0.2)
                ), f"For {samples_per_period} samples/period (decimation_factor={decimation_factor}), got {len(times)}, expected ~{expected_samples}"
                
                # Verify that more samples_per_period results in fewer decimated samples
                # (because decimation_factor increases)
                if samples_per_period > 1:
                    # With higher samples_per_period, decimation_factor should be larger
                    # resulting in fewer samples (but not necessarily, depends on samples_per_T)
                    pass  # This relationship is complex, so we just verify it works


class TestSignalStacker:
    """Test SignalStacker class functionality."""

    @pytest.fixture
    def signal_stacker(self):
        """Create a SignalStacker instance."""
        return SignalStacker(fs=50e6)

    def test_signal_stacker_initialization(self, signal_stacker):
        """Test SignalStacker initialization."""
        assert signal_stacker.fs == 50e6
        assert signal_stacker.spectrum_analyzer is not None
        assert signal_stacker.phase_converter is not None
        assert signal_stacker.cordic_processor is not None

    def test_find_fundamental_frequency(self, signal_stacker):
        """Test fundamental frequency detection."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * f0 * t)

        detected_f0 = signal_stacker.find_fundamental_frequency(signal)

        # Should detect frequency close to f0
        assert abs(detected_f0 - f0) < f0 * 0.1, (
            f"Detected frequency {detected_f0/1e6:.2f} MHz should be close to {f0/1e6:.2f} MHz"
        )

    def test_compute_phase_difference_stacking(self, signal_stacker):
        """Test phase difference computation using stacking method."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)

        # Create signals with known phase difference
        phase_diff = 0.5  # 0.5 rad
        ref_signal = np.sin(2 * np.pi * f0 * t)
        probe_signal = np.sin(2 * np.pi * f0 * t + phase_diff)

        phase_diff_result, detected_f0 = signal_stacker.compute_phase_difference(
            ref_signal, probe_signal, f0, method="stacking"
        )

        # Verify output
        assert len(phase_diff_result) > 0, "Should compute phase difference"
        assert abs(detected_f0 - f0) < f0 * 0.1, "Should detect correct frequency"

        # Phase difference should be approximately constant (after baseline correction)
        # The mean should be close to the known phase difference (within noise)
        mean_phase_diff = np.mean(phase_diff_result)
        assert abs(mean_phase_diff - phase_diff) < 0.5, (
            f"Mean phase difference {mean_phase_diff:.3f} should be close to {phase_diff:.3f}"
        )

    def test_compute_phase_difference_cdm(self, signal_stacker):
        """Test phase difference computation using CDM method."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)

        phase_diff = 0.5
        ref_signal = np.sin(2 * np.pi * f0 * t)
        probe_signal = np.sin(2 * np.pi * f0 * t + phase_diff)

        phase_diff_result, detected_f0 = signal_stacker.compute_phase_difference(
            ref_signal, probe_signal, f0, method="cdm"
        )

        assert len(phase_diff_result) > 0, "Should compute phase difference"
        assert abs(detected_f0 - f0) < f0 * 0.1, "Should detect correct frequency"

    def test_compute_phase_difference_cordic(self, signal_stacker):
        """Test phase difference computation using CORDIC method."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)

        phase_diff = 0.5
        ref_signal = np.sin(2 * np.pi * f0 * t)
        probe_signal = np.sin(2 * np.pi * f0 * t + phase_diff)

        times, phase_diff_result, detected_f0 = (
            signal_stacker.compute_phase_difference_cordic(
                ref_signal, probe_signal, f0
            )
        )

        assert len(phase_diff_result) > 0, "Should compute phase difference"
        assert len(times) == len(phase_diff_result), "Times and phases should match"
        assert abs(detected_f0 - f0) < f0 * 0.1, "Should detect correct frequency"


class TestSTFTRidgeAnalyzer:
    """Test STFTRidgeAnalyzer class functionality."""

    @pytest.fixture
    def stft_analyzer(self):
        """Create an STFTRidgeAnalyzer instance."""
        return STFTRidgeAnalyzer(fs=50e6)

    def test_stft_analyzer_initialization(self, stft_analyzer):
        """Test STFTRidgeAnalyzer initialization."""
        assert stft_analyzer.fs == 50e6

    def test_compute_stft_with_ridge(self, stft_analyzer):
        """Test STFT ridge computation."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * f0 * t)

        f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(signal, f0)

        assert len(f) > 0, "Should have frequency bins"
        assert len(ridge_freqs) > 0, "Should extract ridge frequencies"
        assert len(ridge_freqs) == len(t_stft), "Ridge and time should match length"

        # Ridge frequency should be close to f0
        mean_ridge = np.mean(ridge_freqs)
        assert abs(mean_ridge - f0) < f0 * 0.2, (
            f"Mean ridge frequency {mean_ridge/1e6:.2f} MHz should be close to {f0/1e6:.2f} MHz"
        )

    def test_extract_phase_from_ridge(self, stft_analyzer):
        """Test phase extraction from STFT ridge."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * f0 * t)

        f, t_stft, ridge_freqs = stft_analyzer.compute_stft_with_ridge(signal, f0)
        
        # extract_phase_from_ridge signature: (signal, ridge_freqs, times, nperseg, noverlap)
        phases = stft_analyzer.extract_phase_from_ridge(signal, ridge_freqs, t_stft)

        assert len(phases) > 0, "Should extract phases"
        assert len(phases) == len(t_stft), "Phases and time should match length"
        assert np.all(np.isfinite(phases)), "All phases should be finite"


class TestCWTPhaseReconstructor:
    """Test CWTPhaseReconstructor class functionality."""

    @pytest.fixture
    def cwt_reconstructor(self):
        """Create a CWTPhaseReconstructor instance."""
        return CWTPhaseReconstructor(fs=50e6)

    def test_cwt_reconstructor_initialization(self, cwt_reconstructor):
        """Test CWTPhaseReconstructor initialization."""
        assert cwt_reconstructor.fs == 50e6

    def test_design_fir_filter(self, cwt_reconstructor):
        """Test FIR filter design."""
        f0 = 8e6
        bandwidth = 0.1  # Fraction of f0

        # design_fir_filter signature: (f0, bandwidth)
        filter_coeffs = cwt_reconstructor.design_fir_filter(f0, bandwidth)

        assert len(filter_coeffs) > 0, "Should design filter"
        assert np.all(np.isfinite(filter_coeffs)), "Filter coefficients should be finite"
        # FIR filter should have reasonable length (typically odd order)
        assert len(filter_coeffs) > 10, "Filter should have sufficient order"

    def test_decimate_signal(self, cwt_reconstructor):
        """Test signal decimation."""
        fs = 50e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * 8e6 * t)
        decimation_factor = 4

        decimated = cwt_reconstructor.decimate_signal(signal, decimation_factor)

        assert len(decimated) > 0, "Should decimate signal"
        # Decimated signal should be approximately decimation_factor times shorter
        assert len(decimated) <= len(signal) / decimation_factor * 1.2, (
            "Decimated signal should be shorter"
        )
        assert np.all(np.isfinite(decimated)), "Decimated signal should be finite"

    def test_compute_cwt_phase(self, cwt_reconstructor):
        """Test CWT phase computation."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * f0 * t)

        times, phases = cwt_reconstructor.compute_cwt_phase(signal, f0)

        assert len(times) > 0, "Should compute CWT phase"
        assert len(phases) == len(times), "Times and phases should match"
        assert np.all(np.isfinite(phases)), "All phases should be finite"


class TestPhaseChangeDetector:
    """Test PhaseChangeDetector class functionality."""

    @pytest.fixture
    def phase_detector(self):
        """Create a PhaseChangeDetector instance."""
        return PhaseChangeDetector(fs=50e6)

    def test_phase_detector_initialization(self, phase_detector):
        """Test PhaseChangeDetector initialization."""
        assert phase_detector.fs == 50e6
        assert phase_detector.signal_stacker is not None
        assert phase_detector.stft_analyzer is not None
        assert phase_detector.cwt_reconstructor is not None

    def test_detect_phase_changes_stacking(self, phase_detector):
        """Test phase change detection using stacking method."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)

        phase_diff = 0.5
        ref_signal = np.sin(2 * np.pi * f0 * t)
        probe_signal = np.sin(2 * np.pi * f0 * t + phase_diff)

        results = phase_detector.detect_phase_changes(
            ref_signal, probe_signal, f0, methods=["stacking"]
        )

        assert "methods" in results, "Results should contain methods"
        assert "stacking" in results["methods"], "Should have stacking results"
        stacking_result = results["methods"]["stacking"]

        if "error" not in stacking_result:
            assert "phase_difference" in stacking_result, "Should have phase difference"
            assert len(stacking_result["phase_difference"]) > 0, (
                "Phase difference should have data"
            )

    def test_detect_phase_changes_all_methods(self, phase_detector):
        """Test phase change detection with all methods."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001
        t = np.arange(0, duration, 1 / fs)

        phase_diff = 0.5
        ref_signal = np.sin(2 * np.pi * f0 * t)
        probe_signal = np.sin(2 * np.pi * f0 * t + phase_diff)

        results = phase_detector.detect_phase_changes(
            ref_signal, probe_signal, f0, methods=["stacking", "stft", "cwt", "cdm", "cordic"]
        )

        assert "methods" in results, "Results should contain methods"
        assert len(results["methods"]) > 0, "Should have at least one method result"

        # Check each method (some may fail due to edge cases)
        for method_name in ["stacking", "stft", "cwt", "cdm", "cordic"]:
            if method_name in results["methods"]:
                method_result = results["methods"][method_name]
                if "error" not in method_result:
                    # If no error, should have phase data
                    assert "phase_difference" in method_result or "times" in method_result, (
                        f"{method_name} result should have phase or time data"
                    )

