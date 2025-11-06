#!/usr/bin/env python3
"""Test the proposed CORDIC vectoring mode with quadrant folding."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from numba import njit, prange
from typing import Tuple

@njit(parallel=True)
def _cordic_vectoring_proposed(
    x: np.ndarray,
    y: np.ndarray,
    angle_table: np.ndarray,
    scale_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Proposed CORDIC vectoring mode with quadrant folding."""
    n = x.shape[0]
    n_iter = angle_table.shape[0]
    EPS = 0.0

    # Precompute 2^-i
    pow2 = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        pow2[i] = 1.0 / (1 << i)

    xr = x.copy()
    yr = y.copy()
    z = np.zeros(n, dtype=np.float64)

    # Quadrant folding: Move to right half-plane (xr >= 0)
    for j in range(n):
        x0 = xr[j]
        y0 = yr[j]
        if x0 < 0.0:
            # Flip to right half-plane
            xr[j] = -x0
            yr[j] = -y0
            # Phase correction: +pi if y>=0, -pi if y<0
            if y0 >= 0.0:
                z[j] = np.pi
            else:
                z[j] = -np.pi
        else:
            z[j] = 0.0

    # Main CORDIC loop
    for i in range(n_iter):
        a = angle_table[i]
        f = pow2[i]

        for j in prange(n):
            # Handle zero vector
            if abs(xr[j]) <= EPS and abs(yr[j]) <= EPS:
                continue

            # Vectoring: d = -sign(y)
            dj = -1.0 if yr[j] >= 0.0 else 1.0

            x_new = xr[j] - dj * yr[j] * f
            y_new = yr[j] + dj * xr[j] * f
            z[j] = z[j] + dj * a

            xr[j] = x_new
            yr[j] = y_new

    # Magnitude = |xr| / scale_factor
    mags = np.abs(xr) / scale_factor

    # Phase wrapping to [-pi, pi]
    for j in range(n):
        zj = z[j]
        while zj <= -np.pi:
            zj += 2.0 * np.pi
        while zj > np.pi:
            zj -= 2.0 * np.pi
        z[j] = zj

    return mags, z

# Test with proposed implementation
print("Testing proposed CORDIC with quadrant folding:")
n_iter = 16
angles = np.array([np.arctan(2.0 ** (-i)) for i in range(n_iter)])
scale_factor = np.prod(np.sqrt(1.0 + 2.0 ** (-2 * np.arange(n_iter))))

test_cases = [
    (1.0, 0.0, "0°"),
    (-1.0, 0.0, "180°"),
    (0.0, 1.0, "90°"),
    (1.0, 1.0, "45°"),
    (-1.0, -1.0, "-135°"),
    (np.sqrt(3), 1.0, "30°"),
    (1.0, -1.0, "-45°"),
]

print("\nResults:")
for x, y, desc in test_cases:
    mags, phases = _cordic_vectoring_proposed(
        np.array([x]), np.array([y]), angles, scale_factor
    )
    expected_phase = np.arctan2(y, x)
    expected_mag = np.sqrt(x * x + y * y)
    
    phase_diff = phases[0] - expected_phase
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    
    print(f"  {desc:>6}: x={x:7.4f}, y={y:7.4f}")
    print(f"    mag: {mags[0]:.10f} (expected: {expected_mag:.10f}, error: {abs(mags[0]-expected_mag):.2e})")
    print(f"    phase: {phases[0]:.10f} (expected: {expected_phase:.10f}, error: {abs(phase_diff):.2e})")
    print()


