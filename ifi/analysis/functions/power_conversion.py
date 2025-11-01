#!/usr/bin/env python3
"""
Power Conversion Functions
==========================

This module contains the functions for power conversion.
The functions are used in the main_analysis.py module.

Functions:
    - pow2db: Convert power [W] to decibels [dB].
    - db2pow: Convert decibels [dB] to power [W].
    - amp2db: Convert amplitude [V] to decibels [dB].
    - db2amp: Convert decibels [dB] to amplitude [V].

Examples:
    ```python
    import numpy as np
    from .power_conversion import pow2db, db2pow, amp2db, db2amp
    x = np.array([1, 2, 3])
    print(pow2db(x))
    print(db2pow(x))
    print(amp2db(x))
    print(db2amp(x))
    ```

    Output:
    ```
    [0.         3.01029996 4.77121255]
    [1.25892541 1.58489319 1.99526231]
    [-20.         -13.97940009 -10.45757491]
    [11.22018454 12.58925412 14.12537545]
    ```
"""

import numpy as np

# ============================================================================
# Utility Functions
# ============================================================================


def pow2db(x: np.ndarray, dbm: bool = True) -> np.ndarray:
    """
    Convert power [W] to decibels [dB].

    Args:
        x(np.ndarray): Power in watts [W]
        dbm(bool): If True, convert to dBm [dBm], otherwise convert to dB [dB]

    Returns:
        np.ndarray: Power in decibels [dB]
    """
    return 10 * np.log10(x) if not dbm else 10 * np.log10(x) + 30


def db2pow(x: np.ndarray, dbm: bool = True) -> np.ndarray:
    """Convert decibels [dB] to power [W].

    Args:
        x(np.ndarray): Power in decibels [dB]
        dbm(bool): If True, convert to dBm [dBm], otherwise convert to dB [dB]

    Returns:
        np.ndarray: Power in watts [W]
    """
    return 10 ** (x / 10) if not dbm else 10 ** ((x - 30) / 10)


def amp2db(x: np.ndarray, impedance: float = 50, dbm: bool = True) -> np.ndarray:
    """Convert amplitude [V] to decibels [dB].

    Args:
        x(np.ndarray): Amplitude in volts [V]
        impedance(float): Impedance of the system in ohms [Ω]
        dbm(bool): If True, convert to dBm [dBm], otherwise convert to dB [dB]

    Returns:
        np.ndarray: Amplitude in decibels [dB]
    """
    return (
        20 * np.log10(x / np.sqrt(impedance * 2))
        if not dbm
        else 20 * np.log10(x / np.sqrt(impedance * 2)) + 30
    )


def db2amp(x: np.ndarray, impedance: float = 50, dbm: bool = True) -> np.ndarray:
    """Convert decibels [dB] to amplitude [V].

    Args:
        x(np.ndarray): Amplitude in decibels [dB]
        impedance(float): Impedance of the system in ohms [Ω]
        dbm(bool): If True, convert to dBm [dBm], otherwise convert to dB [dB]

    Returns:
        np.ndarray: Amplitude in volts [V]
    """
    return (
        10 ** (x / 20) * np.sqrt(impedance * 2)
        if not dbm
        else 10 ** ((x - 30) / 20) * np.sqrt(impedance * 2)
    )


def mag2db(x: np.ndarray) -> np.ndarray:
    """Convert magnitude to decibels [dB].

    Args:
        x(np.ndarray): Magnitude

    Returns:
        np.ndarray: Magnitude in decibels [dB]
    """
    return 20 * np.log10(x)


def db2mag(x: np.ndarray) -> np.ndarray:
    """Convert decibels [dB] to magnitude.

    Args:
        x(np.ndarray): Magnitude in decibels [dB]

    Returns:
        np.ndarray: Magnitude
    """
    return 10 ** (x / 20)
