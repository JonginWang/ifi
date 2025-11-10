#!/usr/bin/env python3
"""
Test script to compare MATLAB hilbert and scipy.signal.hilbert.

This script uses MATLAB Engine to directly call MATLAB's hilbert function
and compares it with scipy's implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert as scipy_hilbert

try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False
    print("Warning: MATLAB Engine not available. Install with: python -m pip install matlabengine")


def test_hilbert_direct_comparison():
    """Directly compare MATLAB hilbert and scipy hilbert."""
    if not HAS_MATLAB:
        print("MATLAB Engine not available. Skipping comparison.")
        return
    
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    try:
        # Generate test signal: cos(2*pi*f*t)
        fs = 1e6
        f0 = 100e3
        duration = 0.001
        t = np.arange(int(fs * duration)) / fs
        signal = np.cos(2 * np.pi * f0 * t)
        
        print(f"Test signal:")
        print(f"  Sampling frequency: {fs/1e6:.1f} MHz")
        print(f"  Signal frequency: {f0/1e3:.1f} kHz")
        print(f"  Duration: {duration*1e3:.1f} ms")
        print(f"  Signal length: {len(signal)}")
        
        # MATLAB hilbert
        signal_matlab = matlab.double(signal.tolist())
        hilbert_matlab_result = eng.hilbert(signal_matlab)
        # Convert MATLAB complex array to numpy: np.asarray() automatically handles complex arrays
        hilbert_matlab = np.asarray(hilbert_matlab_result).flatten()
        
        # Scipy hilbert
        hilbert_scipy = scipy_hilbert(signal)
        
        # Compare
        print(f"\nHilbert Transform Comparison:")
        print(f"  MATLAB result shape: {hilbert_matlab.shape}")
        print(f"  Scipy result shape: {hilbert_scipy.shape}")
        
        # Check if they are conjugates
        hilbert_scipy_conj = np.conj(hilbert_scipy)
        
        diff_direct = np.abs(hilbert_matlab - hilbert_scipy)
        diff_conjugate = np.abs(hilbert_matlab - hilbert_scipy_conj)
        
        print(f"\n  Direct difference (MATLAB - Scipy):")
        print(f"    Max: {diff_direct.max():.6e}")
        print(f"    Mean: {diff_direct.mean():.6e}")
        
        print(f"\n  Conjugate difference (MATLAB - conj(Scipy)):")
        print(f"    Max: {diff_conjugate.max():.6e}")
        print(f"    Mean: {diff_conjugate.mean():.6e}")
        
        # Show first few samples
        print(f"\n  First 5 samples:")
        print(f"    Signal:           {signal[:5]}")
        print(f"    MATLAB hilbert:   {hilbert_matlab[:5]}")
        print(f"    Scipy hilbert:    {hilbert_scipy[:5]}")
        print(f"    conj(Scipy):      {hilbert_scipy_conj[:5]}")
        
        # Determine relationship
        if diff_conjugate.max() < diff_direct.max() * 0.01:
            print(f"\n  [OK] MATLAB hilbert == conj(scipy hilbert)")
            print(f"    Use: scipy_hilbert(signal).conj() to match MATLAB")
        elif diff_direct.max() < diff_conjugate.max() * 0.01:
            print(f"\n  [OK] MATLAB hilbert == scipy hilbert")
            print(f"    Use: scipy_hilbert(signal) directly to match MATLAB")
        else:
            print(f"\n  [WARN] Relationship unclear. Manual inspection needed.")
        
        assert min(diff_direct.max(), diff_conjugate.max()) < 1e-12
        
    finally:
        eng.quit()


def test_cdm_hilbert_usage():
    """Test how hilbert is used in CDM algorithm."""
    if not HAS_MATLAB:
        print("MATLAB Engine not available. Skipping CDM test.")
        return
    
    eng = matlab.engine.start_matlab()
    
    try:
        # Test signals as in CDM_VEST_check.m
        fs = 50e6
        f0 = 8e6
        duration = 0.010
        t = np.arange(int(fs * duration)) / fs
        
        # Reference signal: cos(2*pi*f0*t)
        ref = np.cos(2 * np.pi * f0 * t)
        # Probe signal: cos(2*pi*f0*t + phase)
        phase_offset = 0.75 * np.pi
        probe = np.cos(2 * np.pi * f0 * t + phase_offset)
        
        print(f"\nCDM Hilbert Usage Test:")
        print(f"  fs: {fs/1e6:.1f} MHz, f0: {f0/1e6:.1f} MHz")
        print(f"  Phase offset: {phase_offset/np.pi:.3f}π rad")
        
        # MATLAB: x1_94 = hilbert(ref94); ych5 = x1_94.*ch5;
        ref_matlab = matlab.double(ref.tolist())
        x1_matlab = eng.hilbert(ref_matlab)
        # Convert MATLAB complex array to numpy
        x1_matlab_array = np.asarray(x1_matlab).flatten()
        
        probe_matlab = matlab.double(probe.tolist())
        y_matlab = eng.times(x1_matlab, probe_matlab)  # MATLAB .* operator
        # Convert MATLAB complex array to numpy
        y_matlab_array = np.asarray(y_matlab).flatten()
        
        # Python/Scipy equivalent
        x1_scipy = scipy_hilbert(ref)
        
        # Try both: direct and conjugate
        y_scipy_direct = x1_scipy * probe
        y_scipy_conj = x1_scipy.conj() * probe
        
        # Compare
        diff_direct = np.abs(y_matlab_array - y_scipy_direct)
        diff_conj = np.abs(y_matlab_array - y_scipy_conj)
        
        print(f"\n  MATLAB: hilbert(ref) .* probe")
        print(f"  Scipy direct: scipy_hilbert(ref) * probe")
        print(f"    Max diff: {diff_direct.max():.6e}")
        print(f"  Scipy conj: scipy_hilbert(ref).conj() * probe")
        print(f"    Max diff: {diff_conj.max():.6e}")
        
        if diff_conj.max() < diff_direct.max() * 0.01:
            print(f"\n  [OK] Use: scipy_hilbert(ref).conj() * probe to match MATLAB")
            assert diff_conj.max() < diff_direct.max() * 0.01
        elif diff_direct.max() < diff_conj.max() * 0.01:
            print(f"\n  [OK] Use: scipy_hilbert(ref) * probe to match MATLAB")
            assert diff_direct.max() < diff_conj.max() * 0.01
        else:
            print(f"\n  [WARN] Relationship unclear")
            assert False
    finally:
        eng.quit()


if __name__ == "__main__":
    print("=" * 60)
    print("MATLAB hilbert vs Scipy hilbert Comparison Test")
    print("=" * 60)
    
    if HAS_MATLAB:
        print("\n1. Direct Hilbert Transform Comparison:")
        test_hilbert_direct_comparison()
        
        print("\n2. CDM Usage Comparison:")
        test_cdm_hilbert_usage()
        
        print("\n" + "=" * 60)
        print("Test completed!")
    else:
        print("\nMATLAB Engine not available.")
        print("Install with: python -m pip install matlabengine")
        print("Or install from MATLAB: cd (fullfile(matlabroot, 'extern', 'engines', 'python')); system('python -m pip install .')")

