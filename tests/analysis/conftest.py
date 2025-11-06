#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for analysis module tests.

This module provides common fixtures and test utilities used across
multiple test files in the analysis test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.analysis.fixtures.synthetic_signals import (
    sine_signal,
    linear_chirp,
    phase_jump_signal,
)


@pytest.fixture
def standard_signal_params():
    """
    Standard signal parameters for testing.

    Returns:
        dict: Dictionary with standard parameters:
            - fs: Sampling frequency (50 MHz)
            - f0: Center frequency (8 MHz)
            - duration: Signal duration (0.01 seconds)
            - phase_offset: Phase offset for testing (0.75 * π)
    """
    return {
        "fs": 50e6,  # 50 MHz sampling frequency
        "f0": 8e6,  # 8 MHz center frequency
        "duration": 0.010,  # 10 ms duration
        "phase_offset": 0.75 * np.pi,  # Standard phase offset for testing
    }


@pytest.fixture
def reference_signal(standard_signal_params):
    """
    Generate a reference signal (zero phase) for phase difference testing.

    Args:
        standard_signal_params: Standard signal parameters fixture

    Returns:
        tuple: (time, signal) arrays
    """
    params = standard_signal_params
    return sine_signal(
        fs=params["fs"],
        freq=params["f0"],
        duration=params["duration"],
        phase=0.0,
        dphidt=False,
    )


@pytest.fixture
def probe_signal_constant_phase(standard_signal_params):
    """
    Generate a probe signal with constant phase offset.

    Args:
        standard_signal_params: Standard signal parameters fixture

    Returns:
        tuple: (time, signal) arrays with constant phase offset
    """
    params = standard_signal_params
    return sine_signal(
        fs=params["fs"],
        freq=params["f0"],
        duration=params["duration"],
        phase=params["phase_offset"],
        dphidt=False,
    )


@pytest.fixture
def probe_signal_linear_phase(standard_signal_params):
    """
    Generate a probe signal with linear phase evolution.

    Args:
        standard_signal_params: Standard signal parameters fixture

    Returns:
        tuple: (time, signal) arrays with linear phase evolution
    """
    params = standard_signal_params
    return sine_signal(
        fs=params["fs"],
        freq=params["f0"],
        duration=params["duration"],
        phase=params["phase_offset"],
        dphidt=True,
    )


@pytest.fixture
def iq_signals(standard_signal_params):
    """
    Generate I/Q signal pair for IQ-based phase calculation tests.

    Args:
        standard_signal_params: Standard signal parameters fixture

    Returns:
        dict: Dictionary with 'i_signal', 'q_signal', 'time', and 'expected_phase'
    """
    params = standard_signal_params
    fs = params["fs"]
    f0 = params["f0"]
    duration = params["duration"]
    delta = params["phase_offset"]

    n = int(fs * duration)
    t = np.arange(n) / fs
    theta = 2 * np.pi * f0 * t
    i_sig = np.cos(theta)
    q_sig = np.sin(theta + delta)

    return {
        "i_signal": i_sig,
        "q_signal": q_sig,
        "time": t,
        "expected_phase": delta,
    }


@pytest.fixture
def phase_converter():
    """
    Create a PhaseConverter instance for testing.

    Returns:
        PhaseConverter: Instance of PhaseConverter class
    """
    from ifi.analysis.phi2ne import PhaseConverter

    return PhaseConverter()


@pytest.fixture
def small_signal_params():
    """
    Small signal parameters for quick tests.

    Returns:
        dict: Dictionary with smaller signal parameters for fast testing
    """
    return {
        "fs": 10e6,  # 10 MHz sampling frequency
        "f0": 1e6,  # 1 MHz center frequency
        "duration": 0.001,  # 1 ms duration (shorter for speed)
        "phase_offset": 0.5 * np.pi,
    }


@pytest.fixture
def chirp_signal(standard_signal_params):
    """
    Generate a linear chirp signal for frequency ridge detection tests.

    Args:
        standard_signal_params: Standard signal parameters fixture

    Returns:
        tuple: (time, signal) arrays for chirp signal
    """
    params = standard_signal_params
    fs = params["fs"]
    f0 = params["f0"]
    duration = params["duration"]
    # Chirp from f0 to 1.5*f0 over duration
    f1 = 1.5 * f0

    return linear_chirp(fs=fs, f0=f0, f1=f1, duration=duration)


@pytest.fixture
def phase_jump_signal_data(standard_signal_params):
    """
    Generate a signal with a phase jump for phase change detection tests.

    Args:
        standard_signal_params: Standard signal parameters fixture

    Returns:
        dict: Dictionary with 'time', 'signal', 'jump_time', and 'jump_rad'
    """
    params = standard_signal_params
    fs = params["fs"]
    f0 = params["f0"]
    duration = params["duration"]
    jump_time = duration / 2  # Jump at midpoint
    jump_rad = np.pi / 2  # 90 degree jump

    t, sig = phase_jump_signal(
        fs=fs, freq=f0, duration=duration, jump_time=jump_time, jump_rad=jump_rad
    )

    return {
        "time": t,
        "signal": sig,
        "jump_time": jump_time,
        "jump_rad": jump_rad,
    }


@pytest.fixture
def noisy_signal(reference_signal, standard_signal_params):
    """
    Add noise to a reference signal for robustness testing.

    Args:
        reference_signal: Reference signal fixture
        standard_signal_params: Standard signal parameters fixture

    Returns:
        tuple: (time, noisy_signal) arrays
    """
    t, sig = reference_signal
    snr_db = 30  # Signal-to-noise ratio in dB
    signal_power = np.mean(sig**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.RandomState(42).normal(0, np.sqrt(noise_power), len(sig))
    noisy_sig = sig + noise
    return t, noisy_sig


@pytest.fixture(scope="session")
def matlab_engine():
    """
    Create MATLAB engine for tests that require MATLAB comparison.

    Yields:
        MATLAB Engine object or None if not available

    Note:
        This fixture uses session scope to reuse the engine across tests.
        The engine is properly cleaned up after all tests complete.
    """
    try:
        import matlab.engine

        eng = matlab.engine.start_matlab()
        yield eng
        eng.quit()
    except ImportError:
        pytest.skip("MATLAB Engine not available")
        yield None


@pytest.fixture
def has_matlab():
    """
    Check if MATLAB Engine is available.

    Returns:
        bool: True if MATLAB Engine is available, False otherwise
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("matlab.engine")
        return spec is not None
    except Exception:
        return False

