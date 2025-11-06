#!/usr/bin/env python3
"""Verify why accumulation + baseline calibration works for constant phase offset."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from tests.analysis.fixtures.synthetic_signals import sine_signal
from scipy.signal import hilbert
import numpy as np

fs = 50e6
f0 = 8e6
duration = 0.010
phase_offset = 0.75 * np.pi

t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
_, probe = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=False)

ref_h = hilbert(ref)
probe_h = hilbert(probe)
demod = ref_h * probe

re = np.real(demod)
im = np.imag(demod)

denom = np.sqrt(re[:-1]**2 + im[:-1]**2) * np.sqrt(re[1:]**2 + im[1:]**2)
denom[denom == 0] = 1e-12
ratio = (re[:-1]*im[1:] - im[:-1]*re[1:]) / denom
d_phase = np.arcsin(np.clip(ratio, -1, 1))

phase_accum = np.zeros(len(d_phase) + 1)
for i in range(len(d_phase)):
    phase_accum[i+1] = phase_accum[i] + d_phase[i]

baseline = np.mean(phase_accum[:10000])
phase_calib = phase_accum - baseline

print('Verification: Why accumulation + baseline calibration works')
print(f'\nAccumulated phase (before calibration):')
print(f'  phase_accum[0:5] = {phase_accum[0:5]}')
print(f'  phase_accum[-5:] = {phase_accum[-5:]}')
print(f'  Baseline (mean of first 10000) = {baseline:.6f}')

print(f'\nAfter baseline calibration:')
print(f'  phase_calib[0:5] = {phase_calib[0:5]}')
print(f'  phase_calib[-5:] = {phase_calib[-5:]}')
print(f'  Mean of phase_calib = {np.mean(phase_calib):.6f}')
print(f'  Expected: constant phase offset = {phase_offset:.6f} rad')

print(f'\nThe baseline calibration removes the linear ramp from')
print(f'accumulating d_phase values (which include base freq evolution)')

