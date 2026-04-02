#!/usr/bin/env python3
"""
Power Conversion Functions
===========================

This module contains the functions for power conversion.
The functions are used in the main_analysis.py module.

Functions:
    - pow2db: Convert power [W] to decibels [dB].
    - db2pow: Convert decibels [dB] to power [W].
    - amp2db: Convert amplitude [V] to decibels [dB].
    - db2amp: Convert decibels [dB] to amplitude [V].
    - mag2db: Convert magnitude to decibels [dB].
    - db2mag: Convert decibels [dB] to magnitude.

Performance Note:
    Functions use Numba JIT compilation (@njit) for arrays longer than 100k elements.
    For shorter arrays, pure NumPy implementations are used to avoid JIT overhead.

Examples:
    ```python
    import numpy as np
    from ifi.analysis.functions.power_conversion import pow2db, db2pow, amp2db, db2amp
    x = np.array([1, 2, 3])
    print(pow2db(x))
    print(db2pow(x))
    print(amp2db(x))
    print(db2amp(x))
    ```
"""

import numpy as np
from numba import njit

# ============================================================================
# Threshold for Numba JIT compilation
# ============================================================================
N_JIT_THRESHOLD = 100_000  # Use JIT for arrays longer than this


# ============================================================================
# Core Implementation Functions (without decorators)
# ============================================================================

def _pow2db_impl(x: np.ndarray, dbm: bool = True) -> np.ndarray:
    """Core implementation of pow2db (without decorator)."""
    result = 10 * np.log10(x)
    if dbm:
        result += 30
    return result


def _db2pow_impl(x: np.ndarray, dbm: bool = True) -> np.ndarray:
    """Core implementation of db2pow (without decorator)."""
    result = 10 ** ((x - 30) / 10) if dbm else 10 ** (x / 10)
    return result


def _amp2db_impl(x: np.ndarray, impedance: float = 50, dbm: bool = True) -> np.ndarray:
    """Core implementation of amp2db (without decorator)."""
    result = 20 * np.log10(x / np.sqrt(impedance * 2))
    if dbm:
        result += 30
    return result


def _db2amp_impl(x: np.ndarray, impedance: float = 50, dbm: bool = True) -> np.ndarray:
    """Core implementation of db2amp (without decorator)."""
    result = 10 ** ((x - 30) / 20) * np.sqrt(impedance * 2) if dbm else 10 ** (x / 20) * np.sqrt(impedance * 2)
    return result


def _mag2db_impl(x: np.ndarray) -> np.ndarray:
    """Core implementation of mag2db (without decorator)."""
    return 20 * np.log10(x)


def _db2mag_impl(x: np.ndarray) -> np.ndarray:
    """Core implementation of db2mag (without decorator)."""
    return 10 ** (x / 20)


# ============================================================================
# JIT-compiled versions (for large arrays)
# ============================================================================

@njit(cache=True, fastmath=True)
def _pow2db_jit(x: np.ndarray, dbm: bool = True) -> np.ndarray:
    """JIT-compiled version of pow2db."""
    result = 10 * np.log10(x)
    result += 30 if dbm else 0
    return result


@njit(cache=True, fastmath=True)
def _db2pow_jit(x: np.ndarray, dbm: bool = True) -> np.ndarray:
    """JIT-compiled version of db2pow."""
    result = 10 ** ((x - 30) / 10) if dbm else 10 ** (x / 10)
    return result


@njit(cache=True, fastmath=True)
def _amp2db_jit(x: np.ndarray, impedance: float = 50, dbm: bool = True) -> np.ndarray:
    """JIT-compiled version of amp2db."""
    result = 20 * np.log10(x / np.sqrt(impedance * 2))
    result += 30 if dbm else 0
    return result


@njit(cache=True, fastmath=True)
def _db2amp_jit(x: np.ndarray, impedance: float = 50, dbm: bool = True) -> np.ndarray:
    """JIT-compiled version of db2amp."""
    result = 10 ** ((x - 30) / 20) * np.sqrt(impedance * 2) if dbm else 10 ** (x / 20) * np.sqrt(impedance * 2)
    return result


@njit(cache=True, fastmath=True)
def _mag2db_jit(x: np.ndarray) -> np.ndarray:
    """JIT-compiled version of mag2db."""
    return 20 * np.log10(x)


@njit(cache=True, fastmath=True)
def _db2mag_jit(x: np.ndarray) -> np.ndarray:
    """JIT-compiled version of db2mag."""
    return 10 ** (x / 20)


# ============================================================================
# Public API Functions (with conditional JIT)
# ============================================================================

def pow2db(x: np.ndarray, dbm: bool = True) -> np.ndarray:
    """
    Convert power [W] to decibels [dB].

    This function automatically selects between JIT-compiled and pure NumPy
    implementations based on array length. For arrays longer than 100k elements,
    JIT compilation is used for better performance.

    Args:
        x (np.ndarray): Power in watts [W]. Can be scalar or array.
        dbm (bool, optional): If True, convert to dBm [dBm] (adds 30 dB).
            If False, convert to dB [dB]. Default is True.

    Returns:
        np.ndarray: Power in decibels [dB] or [dBm]. Same shape as input.

    Notes:
        - For arrays with length <= 100k: Uses pure NumPy (no JIT overhead).
        - For arrays with length > 100k: Uses Numba JIT compilation.
        - Handles zero and negative values (may produce NaN or warnings).
    """
    x = np.asarray(x)
    return _pow2db_jit(x, dbm) if x.size > N_JIT_THRESHOLD else _pow2db_impl(x, dbm)


def db2pow(x: np.ndarray, dbm: bool = True) -> np.ndarray:
    """
    Convert decibels [dB] to power [W].

    This function automatically selects between JIT-compiled and pure NumPy
    implementations based on array length. For arrays longer than 100k elements,
    JIT compilation is used for better performance.

    Args:
        x (np.ndarray): Power in decibels [dB] or [dBm]. Can be scalar or array.
        dbm (bool, optional): If True, input is in dBm [dBm] (subtracts 30 dB).
            If False, input is in dB [dB]. Default is True.

    Returns:
        np.ndarray: Power in watts [W]. Same shape as input.

    Notes:
        - For arrays with length <= 100k: Uses pure NumPy (no JIT overhead).
        - For arrays with length > 100k: Uses Numba JIT compilation.
    """
    x = np.asarray(x)
    return _db2pow_jit(x, dbm) if x.size > N_JIT_THRESHOLD else _db2pow_impl(x, dbm)


def amp2db(x: np.ndarray, impedance: float = 50, dbm: bool = True) -> np.ndarray:
    """
    Convert amplitude [V] to decibels [dB].

    This function converts voltage amplitude to decibels, accounting for system
    impedance. It automatically selects between JIT-compiled and pure NumPy
    implementations based on array length.

    Args:
        x (np.ndarray): Amplitude in volts [V]. Can be scalar or array.
        impedance (float, optional): Impedance of the system in ohms [Ω].
            Default is 50 Ω (typical RF system impedance).
        dbm (bool, optional): If True, convert to dBm [dBm] (adds 30 dB).
            If False, convert to dB [dB]. Default is True.

    Returns:
        np.ndarray: Amplitude in decibels [dB] or [dBm]. Same shape as input.

    Notes:
        - For arrays with length <= 100k: Uses pure NumPy (no JIT overhead).
        - For arrays with length > 100k: Uses Numba JIT compilation.
        - Formula: dB = 20 * log10(V / sqrt(Z * 2)) + (30 if dbm else 0)
        - The sqrt(Z * 2) factor converts voltage to power-equivalent.
    """
    x = np.asarray(x)
    return _amp2db_jit(x, impedance, dbm) if x.size > N_JIT_THRESHOLD else _amp2db_impl(x, impedance, dbm)


def db2amp(x: np.ndarray, impedance: float = 50, dbm: bool = True) -> np.ndarray:
    """
    Convert decibels [dB] to amplitude [V].

    This function converts decibels to voltage amplitude, accounting for system
    impedance. It automatically selects between JIT-compiled and pure NumPy
    implementations based on array length.

    Args:
        x (np.ndarray): Amplitude in decibels [dB] or [dBm]. Can be scalar or array.
        impedance (float, optional): Impedance of the system in ohms [Ω].
            Default is 50 Ω (typical RF system impedance).
        dbm (bool, optional): If True, input is in dBm [dBm] (subtracts 30 dB).
            If False, input is in dB [dB]. Default is True.

    Returns:
        np.ndarray: Amplitude in volts [V]. Same shape as input.

    Notes:
        - For arrays with length <= 100k: Uses pure NumPy (no JIT overhead).
        - For arrays with length > 100k: Uses Numba JIT compilation.
        - Formula: V = 10^(dB/20) * sqrt(Z * 2) (with -30 dB offset if dbm)
    """
    x = np.asarray(x)
    return _db2amp_jit(x, impedance, dbm) if x.size > N_JIT_THRESHOLD else _db2amp_impl(x, impedance, dbm)


def mag2db(x: np.ndarray) -> np.ndarray:
    """
    Convert magnitude to decibels [dB].

    This function converts magnitude (e.g., from FFT or STFT) to decibels.
    It automatically selects between JIT-compiled and pure NumPy
    implementations based on array length.

    Args:
        x (np.ndarray): Magnitude values. Can be scalar or array.

    Returns:
        np.ndarray: Magnitude in decibels [dB]. Same shape as input.

    Notes:
        - For arrays with length <= 100k: Uses pure NumPy (no JIT overhead).
        - For arrays with length > 100k: Uses Numba JIT compilation.
        - Formula: dB = 20 * log10(magnitude)
        - Commonly used for spectrum magnitude visualization.
    """
    x = np.asarray(x)
    return _mag2db_jit(x) if x.size > N_JIT_THRESHOLD else _mag2db_impl(x)


def db2mag(x: np.ndarray) -> np.ndarray:
    """
    Convert decibels [dB] to magnitude.

    This function converts decibels back to magnitude values.
    It automatically selects between JIT-compiled and pure NumPy
    implementations based on array length.

    Args:
        x (np.ndarray): Magnitude in decibels [dB]. Can be scalar or array.

    Returns:
        np.ndarray: Magnitude values. Same shape as input.

    Notes:
        - For arrays with length <= 100k: Uses pure NumPy (no JIT overhead).
        - For arrays with length > 100k: Uses Numba JIT compilation.
        - Formula: magnitude = 10^(dB / 20)
        - Inverse operation of mag2db.
    """
    x = np.asarray(x)
    return _db2mag_jit(x) if x.size > N_JIT_THRESHOLD else _db2mag_impl(x)
