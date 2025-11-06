#!/usr/bin/env python3
"""
Test to investigate the systematic offset difference (~1.00531 rad) 
between constant and linear phase evolution in calc_phase_cdm results.
"""

from __future__ import annotations

import numpy as np
import pytest

from ifi.analysis.phi2ne import PhaseConverter
from tests.analysis.fixtures.synthetic_signals import sine_signal


def test_cdm_constant_vs_linear_phase_difference():
    """
    Investigate why constant and linear phase evolution yield 
    systematic offset differences in calc_phase_cdm results.
    """
    fs = 50e6
    f0 = 8e6
    duration = 0.010
    phase_offset = 0.75 * np.pi
    
    # Generate signals
    t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
    _, probe_const = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=False)
    _, probe_linear = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=True)
    
    conv = PhaseConverter()
    
    # Compute CDM phase for both
    phase_const = conv.calc_phase_cdm(ref, probe_const, fs, f0, isbpf=False, islpf=True, isconj=False, isold=False)
    phase_linear = conv.calc_phase_cdm(ref, probe_linear, fs, f0, isbpf=False, islpf=True, isconj=False, isold=False)
    
    # Normalize to [-π, π] range
    def normalize_phase(phi):
        return (phi + np.pi) % (2 * np.pi) - np.pi
    
    # Compare different statistics
    print("\n" + "=" * 70)
    print("CDM Phase Comparison: Constant vs Linear Phase Evolution")
    print("=" * 70)
    
    print(f"\nRaw values:")
    print(f"  Constant first and last: {phase_const[0]:.6f}, {phase_const[-1]:.6f}")
    print(f"  Linear first and last: {phase_linear[0]:.6f}, {phase_linear[-1]:.6f}")
    print(f"  Difference first and last: {phase_const[0] - phase_linear[0]:.6f}, {phase_const[-1] - phase_linear[-1]:.6f}")
    
    print(f"\nNormalized values:")
    first_const_norm = normalize_phase(phase_const[0])
    last_const_norm = normalize_phase(phase_const[-1])
    first_linear_norm = normalize_phase(phase_linear[0])
    last_linear_norm = normalize_phase(phase_linear[-1])
    print(f"  Constant first and last (normalized): {first_const_norm:.6f}, {last_const_norm:.6f}")
    print(f"  Linear first and last (normalized): {first_linear_norm:.6f}, {last_linear_norm:.6f}")
    print(f"  Difference first (normalized): {first_const_norm - first_linear_norm:.6f}")
    print(f"  Difference last (normalized): {last_const_norm - last_linear_norm:.6f}")
    
    print(f"\nPhase diff array (difference between consecutive samples):")
    phase_const_diff = np.diff(phase_const)
    phase_linear_diff = np.diff(phase_linear)
    print(f"  Constant diff - first 10: {phase_const_diff[:10]}")
    print(f"  Linear diff - first 10: {phase_linear_diff[:10]}")
    print(f"  Constant diff - mean: {np.mean(phase_const_diff):.12f}, std: {np.std(phase_const_diff):.12f}")
    print(f"  Linear diff - mean: {np.mean(phase_linear_diff):.12f}, std: {np.std(phase_linear_diff):.12f}")
    print(f"  Diff difference (const - linear) - first 10: {phase_const_diff[:10] - phase_linear_diff[:10]}")
    print(f"  Diff difference - mean: {np.mean(phase_const_diff - phase_linear_diff):.12f}")
    print(f"  Diff difference - std: {np.std(phase_const_diff - phase_linear_diff):.12f}")
    print(f"  Diff difference - is constant? std < 1e-6: {np.std(phase_const_diff - phase_linear_diff) < 1e-6}")
    
    print(f"\nEnd values:")
    print(f"  Constant end: {phase_const[-1]:.6f}")
    print(f"  Linear end: {phase_linear[-1]:.6f}")
    print(f"  End difference (raw): {phase_const[-1] - phase_linear[-1]:.6f}")
    print(f"  End difference (normalized): {last_const_norm - last_linear_norm:.6f}")
    
    # Check if difference is close to 1.00531
    mean_const_norm = normalize_phase(np.mean(phase_const))
    mean_linear_norm = normalize_phase(np.mean(phase_linear))
    diff_mean = abs(mean_const_norm - mean_linear_norm)
    diff_end_normalized = abs(last_const_norm - last_linear_norm)
    diff_end_raw = abs(phase_const[-1] - phase_linear[-1])
    
    target_diff = 1.00531
    tolerance = 0.01
    
    print(f"\nTarget difference: {target_diff:.6f} rad")
    print(f"  Mean difference (normalized): {diff_mean:.6f} rad (target match: {abs(diff_mean - target_diff) < tolerance})")
    print(f"  End difference (normalized): {diff_end_normalized:.6f} rad (target match: {abs(diff_end_normalized - target_diff) < tolerance})")
    print(f"  End difference (raw): {diff_end_raw:.6f} rad")
    
    # Also check if it's related to π/3 or other known constants
    print(f"\nPossible related constants:")
    print(f"  π/3 = {np.pi/3:.6f} rad")
    print(f"  1.00531 rad = {np.degrees(1.00531):.2f} degrees")
    print(f"  π/3 rad = {np.degrees(np.pi/3):.2f} degrees")
    
    # Analyze baseline calibration effects
    print(f"\nBaseline analysis (first 10000 samples):")
    baseline_const = np.mean(phase_const[:10000])
    baseline_linear = np.mean(phase_linear[:10000])
    print(f"  Constant baseline: {baseline_const:.6f}")
    print(f"  Linear baseline: {baseline_linear:.6f}")
    print(f"  Baseline difference: {baseline_const - baseline_linear:.6f}")
    
    # After baseline correction (which subtracts the baseline)
    # The difference should be in the remaining signal
    phase_const_after_calib = phase_const - baseline_const
    phase_linear_after_calib = phase_linear - baseline_linear
    
    print(f"\nAfter baseline correction (should be 0 for first samples):")
    print(f"  Constant first 1000 mean: {np.mean(phase_const_after_calib[:1000]):.6f}")
    print(f"  Linear first 1000 mean: {np.mean(phase_linear_after_calib[:1000]):.6f}")
    
    # Check if the systematic offset appears in the signal shape
    print(f"\nSignal shape analysis:")
    print(f"  Constant: min={phase_const.min():.6f}, max={phase_const.max():.6f}, std={phase_const.std():.6f}")
    print(f"  Linear: min={phase_linear.min():.6f}, max={phase_linear.max():.6f}, std={phase_linear.std():.6f}")
    
    # Test with different phase offsets to see if the difference is consistent
    print(f"\nTesting with different phase offsets:")
    for test_phase in [0.5*np.pi, 0.6*np.pi, 0.75*np.pi, np.pi]:
        _, probe_test_const = sine_signal(fs=fs, freq=f0, duration=duration, phase=test_phase, dphidt=False)
        _, probe_test_linear = sine_signal(fs=fs, freq=f0, duration=duration, phase=test_phase, dphidt=True)
        
        phase_test_const = conv.calc_phase_cdm(ref, probe_test_const, fs, f0, isbpf=False, islpf=True, isconj=False, isold=False)
        phase_test_linear = conv.calc_phase_cdm(ref, probe_test_linear, fs, f0, isbpf=False, islpf=True, isconj=False, isold=False)
        
        diff_test = abs(normalize_phase(np.mean(phase_test_const)) - normalize_phase(np.mean(phase_test_linear)))
        print(f"  Phase offset {test_phase/np.pi:.2f}π: difference = {diff_test:.6f} rad")
    
    # Verify if the difference is constant across all samples
    phase_diff_absolute = phase_const - phase_linear
    
    # Vectorized normalization
    def normalize_phase_array(phi_array):
        return (phi_array + np.pi) % (2 * np.pi) - np.pi
    
    phase_const_normalized = normalize_phase_array(phase_const)
    phase_linear_normalized = normalize_phase_array(phase_linear)
    phase_diff_normalized = phase_const_normalized - phase_linear_normalized
    
    print(f"\nAbsolute difference (const - linear) across all samples:")
    print(f"  Mean: {np.mean(phase_diff_absolute):.6f}")
    print(f"  Std: {np.std(phase_diff_absolute):.12f}")
    print(f"  Min: {np.min(phase_diff_absolute):.6f}, Max: {np.max(phase_diff_absolute):.6f}")
    print(f"  Is constant? (std < 1e-6): {np.std(phase_diff_absolute) < 1e-6}")
    
    print(f"\nNormalized difference (const - linear) across all samples:")
    print(f"  Mean: {np.mean(phase_diff_normalized):.6f}")
    print(f"  Std: {np.std(phase_diff_normalized):.12f}")
    print(f"  Min: {np.min(phase_diff_normalized):.6f}, Max: {np.max(phase_diff_normalized):.6f}")
    print(f"  Is constant? (std < 1e-6): {np.std(phase_diff_normalized) < 1e-6}")
    
    # Check if 1.00531 appears in the difference
    diff_values_unique = np.unique(np.round(phase_diff_normalized, decimals=5))
    print(f"\nUnique normalized difference values (rounded to 5 decimals):")
    print(f"  Number of unique values: {len(diff_values_unique)}")
    print(f"  First 20 unique values: {diff_values_unique[:20]}")
    
    # Check if 1.00531 or -1.00531 appears
    target_values = [1.00531, -1.00531]
    for target in target_values:
        matches = np.isclose(np.abs(phase_diff_normalized), abs(target), atol=0.01)
        count = np.sum(matches)
        if count > 0:
            print(f"\n  Found {count} samples with difference close to {target:.6f} rad")
            indices = np.where(matches)[0][:10]  # First 10 indices
            print(f"    First 10 indices: {indices}")
            print(f"    Actual values at these indices: {phase_diff_normalized[indices]}")
    
    # The difference is not constant, so we just report findings
    print(f"\nConclusion: The normalized difference is NOT constant (std={np.std(phase_diff_normalized):.6f})")
    print(f"  This suggests the phase evolution differs between constant and linear cases.")

