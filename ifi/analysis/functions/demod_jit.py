#!/usr/bin/env python3
"""
Signal Utilities (Numba-optimized)
==================================

This module provides Numba-optimized functions for signal processing.
- Normalize I and Q signals to unit circle
- Calculate differential phase
- Compute magnitude statistics
- Apply magnitude threshold
- Adjust interpolated baseline
- Accumulate phase differences
- Convert phase to density

Author: J. Wang
Date: 2025-01-16
"""

import numba
import numpy as np


@numba.jit(nopython=True, cache=True, fastmath=True)
def normalize_iq_signals_jit(
    i_signal: np.ndarray, q_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize I and Q signals to unit circle (Numba-optimized).

    This function normalizes I/Q signals by dividing by their magnitude, 
    ensuring the resulting signals lie on the unit circle.
    - Zero magnitude values are replaced with 1.0 to avoid ZeroDivisionError

    Returns:
        - i_norm: Normalized I signal (unit circle).
        - q_norm: Normalized Q signal (unit circle).
        - iq_mag: Magnitude array sqrt(i^2 + q^2) before normalization.
    """
    iq_mag = np.sqrt(i_signal**2 + q_signal**2)
    iq_mag_original = iq_mag.copy()
    # Avoid division by zero
    for i in range(len(iq_mag)):
        if iq_mag[i] == 0:
            iq_mag[i] = 1.0

    i_norm = i_signal / iq_mag
    q_norm = q_signal / iq_mag
    return i_norm, q_norm, iq_mag_original


@numba.jit(nopython=True, cache=True, fastmath=True)
def calculate_differential_phase_jit(i_norm: np.ndarray, q_norm: np.ndarray) -> np.ndarray:
    r"""
    Calculate differential phase using cross-product method (Numba-optimized).

    This function computes the phase difference between sequential samples using
    the cross-product method (mathematical formula): 
    $\Delta \phi = 2 * \arcsin((I[n] * Q[n+1] - Q[n] * I[n+1]) / (|I[n],Q[n]| * |I[n+1],Q[n+1]|))$

    Returns:
        Differential phase array in radians (length = len(i_norm) - 1).
    """
    # Vectorized differential phase calculation (cross-product method)
    denominator = np.sqrt(i_norm[:-1] ** 2 + q_norm[:-1] ** 2) * np.sqrt(
        i_norm[1:] ** 2 + q_norm[1:] ** 2
    )
    denominator[denominator == 0] = 1e-12  # Avoid division by zero

    # Clip argument to arcsin to handle potential floating point errors
    ratio = np.clip(
        (i_norm[:-1] * q_norm[1:] - q_norm[:-1] * i_norm[1:]) / denominator, -1.0, 1.0
    )
    phase_diff = np.arcsin(ratio)

    return phase_diff


@numba.jit(nopython=True, cache=True, fastmath=True)
def compute_magnitude_stats_jit(iq_mag: np.ndarray) -> tuple[float, float]:
    """
    Compute statistics for I/Q magnitude array (Numba-optimized).

    This function calculates mean and minimum values of the magnitude array.
    - Non-finite values (NaN, inf) are excluded from statistics.

    Returns:
        Tuple[float, float]: Mean and minimum magnitude values.
    """
    # Filter out non-finite values
    finite_mask = np.isfinite(iq_mag)
    if np.sum(finite_mask) == 0:
        return np.nan, np.nan
    
    finite_mag = iq_mag[finite_mask]
    mean_mag = np.mean(finite_mag)
    min_mag = np.min(finite_mag)
    return mean_mag, min_mag


@numba.jit(nopython=True, cache=True, fastmath=True)
def apply_magnitude_threshold_jit(iq_mag: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create a mask for magnitude values above threshold (Numba-optimized).
    - Non-finite values (NaN, inf) are always masked as False.

    Args:
        iq_mag: Magnitude array sqrt(i^2 + q^2) (1D array).
        threshold: Minimum magnitude threshold. Values below this are masked.

    Returns:
        Boolean mask array (True for values >= threshold, False otherwise).
    """
    mask = np.ones(len(iq_mag), dtype=numba.types.boolean)
    for i in range(len(iq_mag)):
        if not np.isfinite(iq_mag[i]) or iq_mag[i] < threshold:
            mask[i] = False
    return mask


def adjust_interpolated_baseline_jit(
    phase_diff_interp: np.ndarray, phase_diff_original: np.ndarray,
) -> np.ndarray:
    """
    Adjust interpolated values by subtracting the last finite value before each NaN segment.
    
    For consecutive NaN segments (i, i+1, i+2, ...), subtract the actual value at index i-1
    from each interpolated value in that segment. This ensures the interpolated values
    are relative to the baseline before the gap.
    
    Args:
        phase_diff_interp (np.ndarray): Interpolated phase difference array (all NaN filled).
        phase_diff_original (np.ndarray): Original phase difference array (with NaN markers).
            Used to identify which indices were interpolated.
    
    Returns:
        np.ndarray: Baseline-adjusted interpolated array.
    
    Notes:
        - Only adjusts values that were originally NaN (interpolated).
        - For degree=0 (nearest), this effectively sets interpolated values to 0.
        - For degree>=1, this removes the baseline offset from interpolated segments.
        - If a NaN segment starts at index 0, interpolated values are set to 0.
    
    Examples:
        ```python
        original = np.array([1.0, 2.0, np.nan, np.nan, 5.0])
        interp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adjusted = adjust_interpolated_baseline_jit(interp, original)
        # Result: [1.0, 2.0, 1.0, 2.0, 5.0]  # 3.0-2.0=1.0, 4.0-2.0=2.0
        ```
    """
    if len(phase_diff_interp) != len(phase_diff_original):
        raise ValueError("Interpolated and original arrays must have same length")
    
    adjusted = phase_diff_interp.copy()
    nan_mask_original = np.isnan(phase_diff_original)
    
    if not np.any(nan_mask_original):
        return adjusted
    
    # Find consecutive NaN segments
    i = 0
    while i < len(nan_mask_original):
        if nan_mask_original[i]:
            # Start of a NaN segment
            segment_start = i
            # Find the end of this segment
            while i < len(nan_mask_original) and nan_mask_original[i]:
                i += 1
            segment_end = i
            # Find the last finite value before this segment (from original array)
            baseline_value = None
            if segment_start > 0:
                # Look backwards from segment_start to find the last finite value
                # Use original array value (y(i-)) as requested
                for k in range(segment_start - 1, -1, -1):
                    if not np.isnan(phase_diff_original[k]):
                        baseline_value = phase_diff_original[k]  # Use original actual value
                        break
            # Adjust all interpolated values in this segment
            if baseline_value is not None:
                for j in range(segment_start, segment_end):
                    adjusted[j] = phase_diff_interp[j] - baseline_value
            else:
                # No baseline available (segment starts at index 0 or all previous values are NaN)
                # Set interpolated values to 0
                for j in range(segment_start, segment_end):
                    adjusted[j] = 0.0
        else:
            i += 1
    return adjusted


@numba.jit(nopython=True, cache=True, fastmath=True)
def accumulate_phase_diff_jit(phase_diff: np.ndarray, duplicate_last: bool = True) -> np.ndarray:
    """
    Accumulate phase differences to reconstruct absolute phase (Numba-optimized).
    Args:
        phase_diff: Length = (n - 1) for an original signal of length n.
        duplicate_last: Whether to duplicate the last value or the first value.
    Returns:
        Accumulated phase array: length = n as the original signal length.

    Notes:
        - Uses cumulative sum: phase_accum[i] = sum(phase_diff[0:i+1])
        - duplicate_last is True: phase_accum[n-1] = phase_accum[n-2]
        - duplicate_last is False: phase_accum[0] = phase_accum[1]
    """
    n = len(phase_diff)
    if n == 0:
        return np.zeros(1, dtype=phase_diff.dtype)

    cumsum = np.cumsum(phase_diff)
    out = np.empty(n + 1, dtype=cumsum.dtype)

    if duplicate_last:
        out[:-1] = cumsum
        out[-1] = cumsum[-1]
    else:
        out[0] = cumsum[0]
        out[1:] = cumsum

    return out


@numba.jit(nopython=True, cache=True, fastmath=True)
def phase_to_density_jit(
    phase: np.ndarray, freq: float, c: float, m_e: float, eps0: float, qe: float, n_path: int,
) -> np.ndarray:
    r"""
    Convert phase to line-integrated electron density (Numba-optimized core calculation).

    Mathematical formula:
        $n_{c} = \frac{m_{e} \varepsilon_{0} (2\pi f)^{2}}{q_{e}^{2}}$
        $\int n_{e}\, dl =\frac{c\, n_{c}}{\pi f}\, \frac{\text{phase}}{n_{\text{path}}}$

    Returns:
        np.ndarray: Line-integrated electron density in m^-2 (same length as phase).
    """
    # Calculate critical density
    n_c = m_e * eps0 * (2 * np.pi * freq)**2 / (qe**2)
    # Calculate line-integrated density
    nedl = (c * n_c / (np.pi * freq)) * phase
    if n_path > 0:
        nedl /= n_path
    return nedl

r"""
The Mathematical formula: 
    $\int {n_{e}\, dl}$
    $= \frac{2 * c * n_{c}}{2 * \pi * freqRF}\, \frac{\phi}{n_{\text{path}}}$
    - The phase (degree) : $\phi * \frac{2*\pi}{360}$
    - The phase (radians): $\phi$
    $Nedl = \frac{2 * c * n_{c}}{\omega}\, \frac{\phi}{n_{\text{path}}}$

Confirmation of the formula:
    d\phi
    &= \frac{\omega}{c} (k_{plasma} - k_{vacuum})\, dl \\
    &= \frac{\omega}{c} (N - 1)\, dl \\
    &= \frac{\omega}{c} \left( \sqrt{1 - \frac{\omega_{p}^2}{\omega^2}} - 1 \right) dl \\
    &= \frac{\omega}{c} \left( \sqrt{1 - \frac{n_{e}}{n_{c}}} - 1 \right) dl
    - $omega_{p}^2 = \frac{n_{e} * q{e}^2}{m_{e} * \varepsilon_{0}}$
    - $n_{c} = \frac{m_{e} * \varepsilon_{0} * \omega^2}{q{e}^2}$

    d\phi
    &=           \frac{\omega}{c} \left( \sqrt{1 - \frac{n_{e}}{n_{c}}} - 1 \right) dl
    &\approx     \frac{\omega}{c} * \frac{n_{e}}{n_{c}} * \frac{1}{2} * dl
    &=           \frac{\omega}{2 * c * n_{c}} * n_{e} * dl

    \Rightarrow  \int {n_{e}\, dl} 
    &= d\phi * \frac{2 * c * n_{c}}{\omega}
    &= d\phi * \frac{2 * c * m_{e} * \varepsilon_{0} * 2 * \pi * freq}{q{e}^2}
"""
