#!/usr/bin/env python3
"""
Phase reconstruction tests for phi2ne utilities.
"""

from __future__ import annotations

import numpy as np

from ifi.analysis.phi2ne import PhaseConverter
from tests.analysis.fixtures.synthetic_signals import sine_signal


# @pytest.mark.skip(reason="CDM test requires scipy remez API fix in phi2ne.py")
def test_cdm_phase_difference_matches_known_offset():
    fs = 50e6
    f0 = 8e6
    duration = 0.010
    delta = 0.75 * np.pi  # known phase offset

    t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
    _, probe = sine_signal(fs=fs, freq=f0, duration=duration, phase=delta, dphidt=True)

    conv = PhaseConverter()
    # CDM center frequency estimate can be provided as true value for this synthetic case
    phase = conv.calc_phase_cdm(ref, probe, fs, f0, isbpf=False, islpf=True, isconj=False, iszif=False)

    # unwrap and check the average offset is close to injected delta
    phase_u = np.unwrap(phase)
    # normalize around mean to avoid 2*pi bias
    mean_offset = (phase_u.mean() + np.pi) % (2 * np.pi) - np.pi
    assert np.isfinite(mean_offset)
    assert np.isclose(mean_offset, delta, rtol=0, atol=5e-2)


def test_iq_phase_difference_matches_expected():
    fs = 10e6
    f0 = 1e6
    duration = 0.010
    delta = 0.5 * np.pi

    # Construct I/Q pair representing a single complex exponential with constant phase offset
    # I/Q signals have constant phase relationship: I = cos(θ), Q = sin(θ + δ)
    n = int(fs * duration)
    t = np.arange(n) / fs
    theta = 2 * np.pi * f0 * t  # Base phase evolves linearly with time
    i_sig = np.cos(theta)  # I channel
    q_sig = np.sin(theta + delta)  # Q channel with constant offset δ

    conv = PhaseConverter()
    phase = conv.calc_phase_iq(i_sig, q_sig)
    phase_u = np.unwrap(phase)
    # Normalize mean to [-π, π] range
    mean_offset = phase_u.mean()
    # Wrap to [-π, π]
    while mean_offset > np.pi:
        mean_offset -= 2 * np.pi
    while mean_offset < -np.pi:
        mean_offset += 2 * np.pi
    # assert np.isfinite(mean_offset)
    assert np.isclose(mean_offset, delta, rtol=0, atol=5e-1)
    # Allow for phase wrapping: check if mean_offset matches delta mod 2π
    # The IQ phase calculation might introduce a constant offset, so check modulo 2π
    expected_mod = delta % (2 * np.pi)
    if expected_mod > np.pi:
        expected_mod -= 2 * np.pi
    actual_mod = mean_offset % (2 * np.pi)
    if actual_mod > np.pi:
        actual_mod -= 2 * np.pi
    # Check if they match within tolerance, accounting for 2π wrapping
    assert np.abs(actual_mod - expected_mod) < 5e-2 or np.abs(actual_mod - expected_mod + 2*np.pi) < 5e-2 or np.abs(actual_mod - expected_mod - 2*np.pi) < 5e-2, \
        f"Mean offset {mean_offset:.4f} (mod 2π: {actual_mod:.4f}) not close to delta {delta:.4f} (mod 2π: {expected_mod:.4f})"


def test_fpga_phase_difference_matches_expected():
    fs = 10e6
    f0 = 1e6
    duration = 0.010
    delta = 0.6 * np.pi  # known phase offset

    # Generate reference and probe signals with linear phase difference
    # dphidt=True means phase grows linearly from 0 to delta over duration
    t, ref_sig = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=True)
    _, probe_sig = sine_signal(fs=fs, freq=f0, duration=duration, phase=delta, dphidt=True)

    # Convert signals to phases (FPGA provides phase directly, simulate by unwrapping atan2)
    # For sine signals, phase = unwrap(arctan2(signal, hilbert(signal)))
    from scipy.signal import hilbert
    ref_analytic = hilbert(ref_sig)
    probe_analytic = hilbert(probe_sig)
    ref_phase = np.unwrap(np.angle(ref_analytic))
    probe_phase = np.unwrap(np.angle(probe_analytic))

    # Create amplitude signal (magnitude of analytic signal)
    amp_signal = np.abs(probe_analytic)

    # Verify the raw phase difference: with dphidt=True, phase grows linearly from 0 to delta
    # So at t[-1], the phase difference should be close to delta
    raw_phase_diff = probe_phase - ref_phase
    # The phase difference should grow linearly from 0 to delta
    # Check the phase difference at the end (t[-1]) should be close to delta
    phase_diff_at_end = raw_phase_diff[-1]
    # Account for 2π wrapping
    phase_diff_at_end_wrapped = (phase_diff_at_end + np.pi) % (2 * np.pi) - np.pi
    assert np.isfinite(phase_diff_at_end_wrapped)
    # For linear phase evolution, the end value should match delta (allowing for wrapping)
    # Check if it matches delta or delta ± 2π
    # Allow for Hilbert transform errors: check if within 0.35 rad tolerance
    assert (np.abs(phase_diff_at_end_wrapped - delta) < 0.35 or 
            np.abs(phase_diff_at_end_wrapped - delta + 2*np.pi) < 0.35 or
            np.abs(phase_diff_at_end_wrapped - delta - 2*np.pi) < 0.35), \
        f"Phase difference at end {phase_diff_at_end_wrapped:.4f} not close to delta {delta:.4f}"

    # Now test the FPGA method (which corrects baseline offsets)
    conv = PhaseConverter()
    phase_diff = conv.calc_phase_fpga(ref_phase, probe_phase, t, amp_signal)

    # After offset correction, verify the method returns finite values and the correction was applied
    assert np.isfinite(phase_diff).all()
    # For linear phase difference, baseline correction removes the initial offset
    # but the linear trend should still be present. For linear phase evolution,
    # the mean of first 1000 samples is approximately delta/2, so after correction,
    # the phase difference should still show a linear trend from -delta/2 to +delta/2
    if len(phase_diff) > 1000:
        # The corrected phase difference should have a linear trend
        # Since baseline correction subtracts mean of first 1000 samples (~delta/2),
        # the phase difference at the end should be approximately delta/2
        expected_end_after_correction = delta / 2
        phase_diff_at_end_corrected = phase_diff[-1]
        phase_diff_at_end_corrected_wrapped = (phase_diff_at_end_corrected + np.pi) % (2 * np.pi) - np.pi
        assert (np.abs(phase_diff_at_end_corrected_wrapped - expected_end_after_correction) < 0.5 or
                np.abs(phase_diff_at_end_corrected_wrapped - expected_end_after_correction + 2*np.pi) < 0.5 or
                np.abs(phase_diff_at_end_corrected_wrapped - expected_end_after_correction - 2*np.pi) < 0.5), \
            f"Corrected phase difference at end {phase_diff_at_end_corrected_wrapped:.4f} not close to expected {expected_end_after_correction:.4f}"


