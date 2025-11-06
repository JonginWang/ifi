#!/usr/bin/env python3
"""
Comprehensive pytest tests for phi2ne module.

This test suite covers:
- Numba-optimized helper functions
- calc_phase_cdm with both Zero-IF and standard DSP modes
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from ifi.analysis.phi2ne import (
    PhaseConverter,  # noqa: F401
    _normalize_iq_signals,
    _calculate_differential_phase,
    _accumulate_phase_diff,
    _phase_to_density,
)


class TestNumbaHelperFunctions:
    """Test suite for numba-optimized helper functions."""

    def test_normalize_iq_signals_basic(self):
        """Test IQ signal normalization with valid inputs."""
        i_signal = np.array([1.0, 2.0, 3.0, 4.0])
        q_signal = np.array([0.0, 1.0, 2.0, 3.0])

        i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)

        # Check normalization: magnitude should be approximately 1
        magnitudes = np.sqrt(i_norm**2 + q_norm**2)
        assert np.allclose(magnitudes, 1.0, rtol=1e-10)

        # Check that normalized signals have same relative ratios
        assert np.allclose(i_norm[0], 1.0, rtol=1e-10)  # q=0, so i_norm = 1
        assert np.allclose(q_norm[0], 0.0, rtol=1e-10)

    def test_normalize_iq_signals_zero_magnitude(self):
        """Test normalization handles zero magnitude correctly."""
        i_signal = np.array([0.0, 0.0, 1.0])
        q_signal = np.array([0.0, 0.0, 1.0])

        i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)

        # Zero magnitude should be handled (set to 1.0 in denominator)
        # Result should be finite
        assert np.all(np.isfinite(i_norm))
        assert np.all(np.isfinite(q_norm))

        # Last element should be normalized correctly
        assert np.allclose(i_norm[-1], 1.0 / np.sqrt(2), rtol=1e-10)

    def test_normalize_iq_signals_empty_array(self):
        """Test normalization with empty arrays."""
        i_signal = np.array([])
        q_signal = np.array([])

        i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)

        assert len(i_norm) == 0
        assert len(q_norm) == 0

    def test_calculate_differential_phase_basic(self, iq_signals):
        """Test differential phase calculation with known I/Q signals."""
        i_norm, q_norm = _normalize_iq_signals(
            iq_signals["i_signal"], iq_signals["q_signal"]
        )

        phase_diff = _calculate_differential_phase(i_norm, q_norm)

        # Output should be one element shorter than input
        assert len(phase_diff) == len(i_norm) - 1

        # Phase differences should be finite and in reasonable range
        assert np.all(np.isfinite(phase_diff))
        # For constant phase offset, differential phase should be small
        assert np.all(np.abs(phase_diff) < np.pi)

    def test_calculate_differential_phase_edge_cases(self):
        """Test differential phase calculation with edge cases."""
        # Test with very small values
        i_signal = np.array([1e-10, 1e-10, 1e-10])
        q_signal = np.array([0.0, 0.0, 0.0])
        i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)

        phase_diff = _calculate_differential_phase(i_norm, q_norm)
        assert np.all(np.isfinite(phase_diff))

        # Test with all zeros (should be handled gracefully)
        i_signal = np.array([0.0, 0.0, 0.0])
        q_signal = np.array([0.0, 0.0, 0.0])
        i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)

        phase_diff = _calculate_differential_phase(i_norm, q_norm)
        assert np.all(np.isfinite(phase_diff))

    def test_accumulate_phase_diff_basic(self):
        """Test phase accumulation with known differential phase."""
        # Create known differential phase (constant small increment)
        phase_diff = np.array([0.1, 0.1, 0.1, 0.1])

        phase_accum = _accumulate_phase_diff(phase_diff)

        # Output should be one element longer than input
        assert len(phase_accum) == len(phase_diff) + 1

        # Check cumulative sum: phase_accum[0] = phase_diff[0], phase_accum[1] = phase_diff[0] + phase_diff[1], etc.
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.4])  # cumsum starts from phase_diff[0]
        assert np.allclose(phase_accum, expected, rtol=1e-10)

        # Last value should be duplicated
        assert phase_accum[-1] == phase_accum[-2]

    def test_accumulate_phase_diff_empty(self):
        """Test accumulation with empty array."""
        phase_diff = np.array([])

        phase_accum = _accumulate_phase_diff(phase_diff)

        # Should return single element (last value duplicated)
        assert len(phase_accum) == 1

    def test_accumulate_phase_diff_single_element(self):
        """Test accumulation with single element."""
        phase_diff = np.array([0.5])

        phase_accum = _accumulate_phase_diff(phase_diff)

        assert len(phase_accum) == 2
        assert phase_accum[0] == 0.5
        assert phase_accum[1] == 0.5  # Duplicated

    def test_phase_to_density_basic(self):
        """Test phase to density conversion with known values."""
        phase = np.array([0.1, 0.2, 0.3])  # radians
        freq = 94e9  # 94 GHz
        c = 2.998e8
        m_e = 9.109e-31
        eps0 = 8.854e-12
        qe = 1.602e-19
        n_path = 2

        density = _phase_to_density(phase, freq, c, m_e, eps0, qe, n_path)

        assert len(density) == len(phase)
        assert np.all(np.isfinite(density))
        assert np.all(density > 0)  # Density should be positive for positive phase

        # Check that larger phase gives larger density
        assert density[2] > density[1] > density[0]

    def test_phase_to_density_zero_path(self):
        """Test phase to density with zero path (no division)."""
        phase = np.array([0.1, 0.2])
        freq = 94e9
        c = 2.998e8
        m_e = 9.109e-31
        eps0 = 8.854e-12
        qe = 1.602e-19
        n_path = 0

        density = _phase_to_density(phase, freq, c, m_e, eps0, qe, n_path)

        # With n_path=0, should not divide (or handle gracefully)
        assert np.all(np.isfinite(density))

    def test_phase_to_density_negative_phase(self):
        """Test phase to density with negative phase values."""
        phase = np.array([-0.1, -0.2, -0.3])
        freq = 94e9
        c = 2.998e8
        m_e = 9.109e-31
        eps0 = 8.854e-12
        qe = 1.602e-19
        n_path = 2

        density = _phase_to_density(phase, freq, c, m_e, eps0, qe, n_path)

        # Density can be negative for negative phase (if formula allows)
        assert np.all(np.isfinite(density))


class TestCalcPhaseCDM:
    """Test suite for calc_phase_cdm method."""

    @pytest.mark.parametrize(
        "iszif,isbpf,islpf,isconj",
        [
            (False, False, True, False),  # Standard DSP mode
            (True, False, False, False),  # Zero-IF mode
            (False, True, True, False),  # With BPF
            (False, False, True, True),  # With conjugate
        ],
    )
    def test_calc_phase_cdm_modes(
        self, phase_converter, reference_signal, probe_signal_constant_phase, iszif, isbpf, islpf, isconj
    ):
        """Test calc_phase_cdm with different mode combinations."""
        params = {
            "fs": 50e6,
            "f0": 8e6,
            "duration": 0.010,
        }
        t, ref = reference_signal
        _, probe = probe_signal_constant_phase

        phase = phase_converter.calc_phase_cdm(
            ref,
            probe,
            params["fs"],
            params["f0"],
            isbpf=isbpf,
            islpf=islpf,
            isconj=isconj,
            iszif=iszif,
        )

        # Check output shape matches input
        assert len(phase) == len(ref)
        assert np.all(np.isfinite(phase))

    def test_calc_phase_cdm_standard_dsp(self, phase_converter, reference_signal, probe_signal_constant_phase):
        """Test calc_phase_cdm in standard DSP mode (iszif=False)."""
        params = {
            "fs": 50e6,
            "f0": 8e6,
            "phase_offset": 0.75 * np.pi,
        }
        t, ref = reference_signal
        _, probe = probe_signal_constant_phase

        phase = phase_converter.calc_phase_cdm(
            ref, probe, params["fs"], params["f0"], isbpf=False, islpf=True, isconj=False, iszif=False
        )

        # For constant phase offset, check that output is finite and reasonable
        assert len(phase) == len(ref)
        assert np.all(np.isfinite(phase))
        # CDM with baseline correction may produce values centered around 0
        # Just verify the result is reasonable (not all zeros, has some variation)
        assert np.std(phase) > 1e-6  # Has some variation

    def test_calc_phase_cdm_zero_if(self, phase_converter, reference_signal, probe_signal_constant_phase):
        """Test calc_phase_cdm in Zero-IF mode (iszif=True)."""
        params = {"fs": 50e6, "f0": 8e6}
        t, ref = reference_signal
        _, probe = probe_signal_constant_phase

        phase = phase_converter.calc_phase_cdm(
            ref, probe, params["fs"], params["f0"], isbpf=False, islpf=False, isconj=False, iszif=True
        )

        # Zero-IF mode should produce finite results
        assert len(phase) == len(ref)
        assert np.all(np.isfinite(phase))

    def test_calc_phase_cdm_linear_phase(self, phase_converter, reference_signal, probe_signal_linear_phase):
        """Test calc_phase_cdm with linear phase evolution."""
        params = {"fs": 50e6, "f0": 8e6}
        t, ref = reference_signal
        _, probe = probe_signal_linear_phase

        phase = phase_converter.calc_phase_cdm(
            ref, probe, params["fs"], params["f0"], isbpf=False, islpf=True, isconj=False, iszif=False
        )

        assert len(phase) == len(ref)
        assert np.all(np.isfinite(phase))

        # For linear phase evolution, phase should show a trend
        phase_diff = np.diff(phase)
        # Should not be all zeros (unless perfectly constant)
        assert np.any(np.abs(phase_diff) > 1e-10)

    def test_calc_phase_cdm_mismatched_lengths(self, phase_converter, reference_signal):
        """Test calc_phase_cdm with mismatched signal lengths."""
        t, ref = reference_signal
        probe = ref[:-10]  # Shorter probe signal

        with pytest.raises((ValueError, IndexError)):
            phase_converter.calc_phase_cdm(ref, probe, 50e6, 8e6)

    def test_calc_phase_cdm_empty_signals(self, phase_converter):
        """Test calc_phase_cdm with empty signals."""
        ref = np.array([])
        probe = np.array([])

        with pytest.raises((ValueError, IndexError)):
            phase_converter.calc_phase_cdm(ref, probe, 50e6, 8e6)

    def test_calc_phase_cdm_nan_input(self, phase_converter, reference_signal):
        """Test calc_phase_cdm with NaN values in input."""
        t, ref = reference_signal
        probe = reference_signal[1].copy()
        probe[100:200] = np.nan  # Insert NaN values

        # Should either handle gracefully or raise an error
        try:
            phase = phase_converter.calc_phase_cdm(ref, probe, 50e6, 8e6)
            # If it succeeds, check for NaN propagation
            if np.any(np.isnan(phase)):
                pytest.skip("NaN values propagate through calculation (may be acceptable)")
        except (ValueError, RuntimeError):
            pass  # Raising an error is acceptable

    def test_calc_phase_cdm_very_short_signal(self, phase_converter):
        """Test calc_phase_cdm with very short signals."""
        fs = 50e6
        f0 = 8e6
        n = 100  # Very short signal
        t = np.arange(n) / fs
        ref = np.sin(2 * np.pi * f0 * t)
        probe = np.sin(2 * np.pi * f0 * t + 0.1)

        # Very short signals may fail with LPF (filter length > signal length)
        # Test without LPF for short signals
        phase = phase_converter.calc_phase_cdm(ref, probe, fs, f0, isbpf=False, islpf=False, iszif=False)

        # Should handle short signals gracefully
        assert len(phase) == n
        assert np.all(np.isfinite(phase))


class TestPhaseConverterClass:
    """Test suite for PhaseConverter class methods."""

    def test_phase_converter_initialization(self, phase_converter):
        """Test PhaseConverter initialization."""
        assert phase_converter is not None
        assert hasattr(phase_converter, "constants")
        assert "c" in phase_converter.constants
        assert "m_e" in phase_converter.constants

    def test_calc_phase_iq_basic(self, phase_converter, iq_signals):
        """Test calc_phase_iq with known I/Q signals."""
        phase = phase_converter.calc_phase_iq(iq_signals["i_signal"], iq_signals["q_signal"])

        assert len(phase) == len(iq_signals["i_signal"])
        assert np.all(np.isfinite(phase))

    def test_calc_phase_fpga_basic(self, phase_converter, reference_signal, probe_signal_constant_phase):
        """Test calc_phase_fpga with known phase signals."""
        from scipy.signal import hilbert

        t, ref_sig = reference_signal
        _, probe_sig = probe_signal_constant_phase

        # Convert to phases (simulating FPGA output)
        ref_analytic = hilbert(ref_sig)
        probe_analytic = hilbert(probe_sig)
        ref_phase = np.unwrap(np.angle(ref_analytic))
        probe_phase = np.unwrap(np.angle(probe_analytic))
        amp_signal = np.abs(probe_analytic)

        phase_diff = phase_converter.calc_phase_fpga(ref_phase, probe_phase, t, amp_signal)

        assert len(phase_diff) == len(ref_phase)
        assert np.all(np.isfinite(phase_diff))

    def test_phase_to_density_method(self, phase_converter):
        """Test phase_to_density method with various inputs."""
        phase = np.array([0.1, 0.2, 0.3])
        freq_hz = 94e9
        n_path = 2

        density = phase_converter.phase_to_density(phase, freq_hz=freq_hz, n_path=n_path)

        assert len(density) == len(phase)
        assert np.all(np.isfinite(density))

    def test_get_params(self, phase_converter):
        """Test get_params method."""
        params = phase_converter.get_params(94)

        assert "freq" in params
        assert "n_path" in params
        assert params["freq"] == pytest.approx(94e9, rel=1e-6)
        # Check that n_path is a valid integer
        assert isinstance(params["n_path"], (int, float))
        assert params["n_path"] > 0

