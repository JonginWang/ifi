#!/usr/bin/env python3
"""
Synthetic signal fixtures for analysis tests.

Provides deterministic signals for accuracy and performance tests.
"""

from __future__ import annotations

import numpy as np


def sine_signal(
    fs: float, freq: float, duration: float, phase: float = 0.0, dphidt: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pure sine wave.

    Args:
        fs(float): Sampling frequency (Hz)
        freq(float): Tone frequency (Hz)
        duration(float): Signal duration (seconds)
        phase(float): Initial phase (radians)

    Returns:
        (t(np.ndarray), x(np.ndarray)):
            tuple of time vector and signal array
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    if dphidt:
        phase = 2 * np.pi * freq * t + phase
    else:
        phase = 2 * np.pi * freq * t + phase * t / n
    x = np.sin(phase)
    return t, x


def linear_chirp(
    fs: float, f0: float, f1: float, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linear chirp from f0 to f1 over duration.

    Args:
        fs(float): Sampling frequency (Hz)
        f0(float): Start frequency (Hz)
        f1(float): End frequency (Hz)
        duration(float): Signal duration (seconds)

    Returns:
        (t(np.ndarray), x(np.ndarray)):
            tuple of time vector and signal array
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    k = (f1 - f0) / duration  # sweep rate (Hz/s)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    x = np.sin(phase)
    return t, x


def phase_jump_signal(
    fs: float, freq: float, duration: float, jump_time: float, jump_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sine wave with a phase jump at jump_time.

    Args:
        fs(float): Sampling frequency (Hz)
        freq(float): Tone frequency (Hz)
        duration(float): Signal duration (seconds)
        jump_time(float): Time of phase jump (seconds)
        jump_rad(float): Phase jump magnitude (radians)

    Returns:
        (t(np.ndarray), x(np.ndarray)):
            tuple of time vector and signal array
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    phase = 2 * np.pi * freq * t
    idx = int(jump_time * fs)
    if 0 < idx < n:
        phase[idx:] += jump_rad
    x = np.sin(phase)
    return t, x
