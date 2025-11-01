#!/usr/bin/env python3
"""
Phase modulation comparison test between MATLAB and Python implementations.

Tests various phase modulation scenarios:
1. Constant phase offset
2. Linear phase evolution
3. Sinusoidal phase modulation

Compares:
- MATLAB hilbert + argumentation
- Python hilbert + argumentation
- Python CDM method
- Python IQ method
- Python FPGA method
"""

from __future__ import annotations

# Set environment variables BEFORE any imports that might trigger PyTorch
import os
# Prevent duplicate library errors on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Disable OpenMP if causing issues
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from scipy.signal import hilbert
import pytest

# Import directly to avoid PyTorch DLL issues
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# IMPORTANT: Import PhaseConverter BEFORE MATLAB Engine to avoid DLL conflicts
# MATLAB Engine and PyTorch share DLLs that can conflict if MATLAB is loaded first
HAS_PHASE_CONVERTER = False
PhaseConverter = None
_accumulate_phase_diff = None

def _lazy_import_phase_converter():
    """
    Lazy import of PhaseConverter to avoid PyTorch DLL issues.
    
    IMPORTANT: This must be called BEFORE starting MATLAB Engine,
    as MATLAB Engine interferes with PyTorch DLL loading.
    """
    global HAS_PHASE_CONVERTER, PhaseConverter, _accumulate_phase_diff
    if HAS_PHASE_CONVERTER:
        return True
    
    try:
        from ifi.analysis.phi2ne import PhaseConverter, _accumulate_phase_diff
        HAS_PHASE_CONVERTER = True
        return True
    except (ImportError, OSError) as e:
        if not hasattr(_lazy_import_phase_converter, '_warned'):
            print(f"Warning: Could not import PhaseConverter: {e}")
            print("Will test only MATLAB and Python hilbert methods.")
            _lazy_import_phase_converter._warned = True
        HAS_PHASE_CONVERTER = False
        return False

# Try to pre-import PhaseConverter before MATLAB Engine
# This ensures PyTorch DLLs are loaded first
_lazy_import_phase_converter()

# Now import MATLAB Engine (must be after PhaseConverter import)
try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False

from tests.analysis.fixtures.synthetic_signals import sine_signal


def test_constant_phase_offset():
    """Test constant phase offset: φ(t) = constant."""
    if not HAS_MATLAB:
        pytest.skip("MATLAB Engine not available")
    
    eng = matlab.engine.start_matlab()
    
    try:
        print("\n" + "=" * 70)
        print("Test 1: Constant Phase Offset")
        print("=" * 70)
        
        fs = 50e6
        f0 = 8e6
        duration = 0.010
        phase_offset = 0.75 * np.pi  # Known constant offset
        
        # Generate signals
        t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
        _, probe = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=False)
        
        print(f"Parameters: fs={fs/1e6:.1f}MHz, f0={f0/1e6:.1f}MHz, duration={duration*1e3:.1f}ms")
        print(f"Expected phase offset: {phase_offset/np.pi:.3f}*pi rad = {phase_offset:.4f} rad")
        
        # MATLAB: Hilbert + argumentation
        ref_matlab = matlab.double(ref.tolist())
        probe_matlab = matlab.double(probe.tolist())
        
        ref_hilbert_matlab = eng.hilbert(ref_matlab)
        probe_hilbert_matlab = eng.hilbert(probe_matlab)
        
        ref_hilbert_array = np.asarray(ref_hilbert_matlab).flatten()
        probe_hilbert_array = np.asarray(probe_hilbert_matlab).flatten()
        
        # MATLAB phase = angle(probe_hilbert) - angle(ref_hilbert)
        phase_matlab = np.unwrap(np.angle(probe_hilbert_array)) - np.unwrap(np.angle(ref_hilbert_array))
        
        # Python: Hilbert + argumentation
        ref_hilbert_python = hilbert(ref)
        probe_hilbert_python = hilbert(probe)
        phase_python_hilbert = np.unwrap(np.angle(probe_hilbert_python)) - np.unwrap(np.angle(ref_hilbert_python))
        
        # Python: CDM method (lazy import)
        if _lazy_import_phase_converter():
            conv = PhaseConverter()
            phase_cdm = conv.calc_phase_cdm(ref, probe, fs, f0, isbpf=False, islpf=False, isconj=False, isold=False)
            
            # Python: IQ method (construct I/Q from signals)
            phase_iq = conv.calc_phase_iq(ref, probe)
            
            # Python: FPGA method (use hilbert-based phases as input)
            ref_phase_fpga = np.unwrap(np.angle(hilbert(ref)))
            probe_phase_fpga = np.unwrap(np.angle(hilbert(probe)))
            amp_signal = np.abs(hilbert(probe))
            phase_fpga = conv.calc_phase_fpga(ref_phase_fpga, probe_phase_fpga, t, amp_signal)
        else:
            phase_cdm = None
            phase_iq = None
            phase_fpga = None
        
        # Compare results
        print(f"\nResults comparison:")
        print(f"  Method                      | Mean Phase    | Expected")
        print(f"  {'-' * 65}")
        
        mean_matlab = np.mean(phase_matlab)
        mean_python_hilbert = np.mean(phase_python_hilbert)
        
        print(f"  MATLAB hilbert              | {mean_matlab:10.6f}  | {phase_offset:.6f}")
        print(f"  Python hilbert              | {mean_python_hilbert:10.6f}  | {phase_offset:.6f}")
        
        # Check accuracy (allowing for wrapping and small errors)
        tolerance = 0.1  # 0.1 rad tolerance
        
        # Normalize to [-π, π] range
        def normalize_phase(phi):
            return (phi + np.pi) % (2 * np.pi) - np.pi
        
        expected_normalized = normalize_phase(phase_offset)
        
        matlab_ok = np.abs(normalize_phase(mean_matlab) - expected_normalized) < tolerance
        python_hilbert_ok = np.abs(normalize_phase(mean_python_hilbert) - expected_normalized) < tolerance
        
        # Use assertions for pytest
        assert matlab_ok, f"MATLAB hilbert mean phase {mean_matlab:.6f} not close to expected {expected_normalized:.6f}"
        assert python_hilbert_ok, f"Python hilbert mean phase {mean_python_hilbert:.6f} not close to expected {expected_normalized:.6f}"
        
        print(f"\nAccuracy check (tolerance: {tolerance} rad):")
        print(f"  MATLAB hilbert:    [PASS]")
        print(f"  Python hilbert:    [PASS]")
        
        if _lazy_import_phase_converter():
            mean_cdm = np.mean(phase_cdm)
            mean_iq = np.mean(phase_iq)
            mean_fpga = np.mean(phase_fpga)
            print(f"  Python CDM                  | {mean_cdm:10.6f}  | {phase_offset:.6f}")
            print(f"  Python IQ                   | {mean_iq:10.6f}  | {phase_offset:.6f}")
            print(f"  Python FPGA                 | {mean_fpga:10.6f}  | {phase_offset:.6f}")
            
            cdm_ok = np.abs(normalize_phase(mean_cdm) - expected_normalized) < tolerance
            iq_ok = np.abs(normalize_phase(mean_iq) - expected_normalized) < tolerance
            fpga_ok = np.abs(normalize_phase(mean_fpga) - expected_normalized) < tolerance
            
            assert cdm_ok, f"Python CDM mean phase {mean_cdm:.6f} not close to expected {expected_normalized:.6f}"
            assert iq_ok, f"Python IQ mean phase {mean_iq:.6f} not close to expected {expected_normalized:.6f}"
            assert fpga_ok, f"Python FPGA mean phase {mean_fpga:.6f} not close to expected {expected_normalized:.6f}"
            
            print(f"  Python CDM:        [PASS]")
            print(f"  Python IQ:         [PASS]")
            print(f"  Python FPGA:       [PASS]")
        
    finally:
        eng.quit()


def test_linear_phase_evolution():
    """
    Test linear phase evolution: φ(t) = k*t.
    
    IMPORTANT: PhaseConverter must be imported BEFORE starting MATLAB Engine
    to avoid DLL conflicts. This is handled at module import time above.
    """
    if not HAS_MATLAB:
        pytest.skip("MATLAB Engine not available")
    
    # Verify PhaseConverter was loaded successfully
    if not HAS_PHASE_CONVERTER:
        pytest.skip("PhaseConverter not available (PyTorch DLL issue)")
    
    eng = matlab.engine.start_matlab()
    
    try:
        print("\n" + "=" * 70)
        print("Test 2: Linear Phase Evolution")
        print("=" * 70)
        
        fs = 50e6
        f0 = 8e6
        duration = 0.010
        final_phase = 0.6 * np.pi  # Phase at t[-1]
        
        # Generate signals with linear phase evolution
        t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=True)
        _, probe = sine_signal(fs=fs, freq=f0, duration=duration, phase=final_phase, dphidt=True)
        
        print(f"Parameters: fs={fs/1e6:.1f}MHz, f0={f0/1e6:.1f}MHz, duration={duration*1e3:.1f}ms")
        print(f"Phase evolution: 0 -> {final_phase/np.pi:.3f}*pi rad over duration")
        print(f"Expected phase at t[-1]: {final_phase:.6f} rad")
        
        # MATLAB: Hilbert + argumentation
        ref_matlab = matlab.double(ref.tolist())
        probe_matlab = matlab.double(probe.tolist())
        
        ref_hilbert_matlab = eng.hilbert(ref_matlab)
        probe_hilbert_matlab = eng.hilbert(probe_matlab)
        
        ref_hilbert_array = np.asarray(ref_hilbert_matlab).flatten()
        probe_hilbert_array = np.asarray(probe_hilbert_matlab).flatten()
        
        phase_matlab = np.unwrap(np.angle(probe_hilbert_array)) - np.unwrap(np.angle(ref_hilbert_array))
        
        # Python: Hilbert + argumentation
        ref_hilbert_python = hilbert(ref)
        probe_hilbert_python = hilbert(probe)
        phase_python_hilbert = np.unwrap(np.angle(probe_hilbert_python)) - np.unwrap(np.angle(ref_hilbert_python))
        
        # Python: CDM method (lazy import)
        if _lazy_import_phase_converter():
            conv = PhaseConverter()
            phase_cdm = conv.calc_phase_cdm(ref, probe, fs, f0, isbpf=False, islpf=False, isconj=False, isold=False)
            
            # Python: IQ method
            phase_iq = conv.calc_phase_iq(ref, probe)
            
            # Python: FPGA method
            ref_phase_fpga = np.unwrap(np.angle(hilbert(ref)))
            probe_phase_fpga = np.unwrap(np.angle(hilbert(probe)))
            amp_signal = np.abs(hilbert(probe))
            phase_fpga = conv.calc_phase_fpga(ref_phase_fpga, probe_phase_fpga, t, amp_signal)
        else:
            phase_cdm = None
            phase_iq = None
            phase_fpga = None
        
        # Check phase at the end (t[-1])
        print(f"\nPhase at t[-1] comparison:")
        print(f"  Method                      | Phase at end  | Expected")
        print(f"  {'-' * 65}")
        
        def normalize_phase(phi):
            return (phi + np.pi) % (2 * np.pi) - np.pi
        
        end_matlab = normalize_phase(phase_matlab[-1])
        end_python_hilbert = normalize_phase(phase_python_hilbert[-1])
        
        expected_normalized = normalize_phase(final_phase)
        
        print(f"  MATLAB hilbert              | {end_matlab:10.6f}  | {expected_normalized:.6f}")
        print(f"  Python hilbert              | {end_python_hilbert:10.6f}  | {expected_normalized:.6f}")
        
        tolerance = 0.5  # Larger tolerance for linear phase due to baseline correction
        
        matlab_ok = np.abs(end_matlab - expected_normalized) < tolerance
        python_hilbert_ok = np.abs(end_python_hilbert - expected_normalized) < tolerance
        
        # Use assertions for pytest
        assert matlab_ok, f"MATLAB hilbert phase at end {end_matlab:.6f} not close to expected {expected_normalized:.6f}"
        assert python_hilbert_ok, f"Python hilbert phase at end {end_python_hilbert:.6f} not close to expected {expected_normalized:.6f}"
        
        print(f"\nAccuracy check (tolerance: {tolerance} rad):")
        print(f"  MATLAB hilbert:    [PASS]")
        print(f"  Python hilbert:    [PASS]")
        
        if _lazy_import_phase_converter():
            end_cdm = normalize_phase(phase_cdm[-1])
            end_iq = normalize_phase(phase_iq[-1])
            end_fpga = normalize_phase(phase_fpga[-1])
            
            print(f"  Python CDM                  | {end_cdm:10.6f}  | {expected_normalized:.6f}")
            print(f"  Python IQ                   | {end_iq:10.6f}  | {expected_normalized:.6f}")
            print(f"  Python FPGA                 | {end_fpga:10.6f}  | {expected_normalized:.6f}")
            
            cdm_ok = np.abs(end_cdm - expected_normalized) < tolerance
            iq_ok = np.abs(end_iq - expected_normalized) < tolerance
            fpga_ok = np.abs(end_fpga - expected_normalized) < tolerance
            
            assert cdm_ok, f"Python CDM phase at end {end_cdm:.6f} not close to expected {expected_normalized:.6f}"
            assert iq_ok, f"Python IQ phase at end {end_iq:.6f} not close to expected {expected_normalized:.6f}"
            assert fpga_ok, f"Python FPGA phase at end {end_fpga:.6f} not close to expected {expected_normalized:.6f}"
            
            print(f"  Python CDM:        [PASS]")
            print(f"  Python IQ:         [PASS]")
            print(f"  Python FPGA:       [PASS]")
        
    finally:
        eng.quit()


def test_sinusoidal_phase_modulation():
    """
    Test sinusoidal phase modulation: φ(t) = A*sin(2π*f_mod*t).
    
    IMPORTANT: PhaseConverter must be imported BEFORE starting MATLAB Engine
    to avoid DLL conflicts. This is handled at module import time above.
    """
    if not HAS_MATLAB:
        pytest.skip("MATLAB Engine not available")
    
    # Verify PhaseConverter was loaded successfully
    if not HAS_PHASE_CONVERTER:
        pytest.skip("PhaseConverter not available (PyTorch DLL issue)")
    
    eng = matlab.engine.start_matlab()
    
    try:
        print("\n" + "=" * 70)
        print("Test 3: Sinusoidal Phase Modulation")
        print("=" * 70)
        
        fs = 50e6
        f0 = 8e6
        duration = 0.010
        mod_amplitude = 0.5 * np.pi  # Modulation amplitude
        mod_freq = 1e3  # 1 kHz modulation frequency
        
        # Generate signals
        t = np.arange(int(fs * duration)) / fs
        ref = np.cos(2 * np.pi * f0 * t)
        # Probe with sinusoidal phase modulation
        phase_mod = mod_amplitude * np.sin(2 * np.pi * mod_freq * t)
        probe = np.cos(2 * np.pi * f0 * t + phase_mod)
        
        print(f"Parameters: fs={fs/1e6:.1f}MHz, f0={f0/1e6:.1f}MHz, duration={duration*1e3:.1f}ms")
        print(f"Modulation: phi(t) = {mod_amplitude/np.pi:.2f}*pi * sin(2*pi * {mod_freq/1e3:.1f}kHz * t)")
        
        # MATLAB: Hilbert + argumentation
        ref_matlab = matlab.double(ref.tolist())
        probe_matlab = matlab.double(probe.tolist())
        
        ref_hilbert_matlab = eng.hilbert(ref_matlab)
        probe_hilbert_matlab = eng.hilbert(probe_matlab)
        
        ref_hilbert_array = np.asarray(ref_hilbert_matlab).flatten()
        probe_hilbert_array = np.asarray(probe_hilbert_matlab).flatten()
        
        phase_matlab = np.unwrap(np.angle(probe_hilbert_array)) - np.unwrap(np.angle(ref_hilbert_array))
        
        # Python: Hilbert + argumentation
        ref_hilbert_python = hilbert(ref)
        probe_hilbert_python = hilbert(probe)
        phase_python_hilbert = np.unwrap(np.angle(probe_hilbert_python)) - np.unwrap(np.angle(ref_hilbert_python))
        
        # Python: CDM method
        if HAS_PHASE_CONVERTER:
            conv = PhaseConverter()
            phase_cdm = conv.calc_phase_cdm(ref, probe, fs, f0, isbpf=False, islpf=False, isconj=False, isold=False)
            
            # Python: IQ method
            phase_iq = conv.calc_phase_iq(ref, probe)
            
            # Python: FPGA method
            ref_phase_fpga = np.unwrap(np.angle(hilbert(ref)))
            probe_phase_fpga = np.unwrap(np.angle(hilbert(probe)))
            amp_signal = np.abs(hilbert(probe))
            phase_fpga = conv.calc_phase_fpga(ref_phase_fpga, probe_phase_fpga, t, amp_signal)
        else:
            phase_cdm = None
            phase_iq = None
            phase_fpga = None
        
        # Compare with expected modulation
        print(f"\nComparison with expected modulation:")
        print(f"  Method                      | Correlation   | RMSE")
        print(f"  {'-' * 65}")
        
        # Normalize to [-π, π] range
        def normalize_phase(phi):
            phi_norm = (phi + np.pi) % (2 * np.pi) - np.pi
            return phi_norm
        
        # Normalize expected phase to same range
        expected_phase_norm = normalize_phase(phase_mod)
        
        # Compare with expected (only up to min length)
        lengths = [len(phase_matlab), len(phase_python_hilbert), len(expected_phase_norm)]
        if _lazy_import_phase_converter():
            lengths.extend([len(phase_cdm), len(phase_iq), len(phase_fpga)])
        min_len = min(lengths)
        
        corr_matlab = np.corrcoef(normalize_phase(phase_matlab[:min_len]), expected_phase_norm[:min_len])[0,1]
        corr_python_hilbert = np.corrcoef(normalize_phase(phase_python_hilbert[:min_len]), expected_phase_norm[:min_len])[0,1]
        
        rmse_matlab = np.sqrt(np.mean((normalize_phase(phase_matlab[:min_len]) - expected_phase_norm[:min_len])**2))
        rmse_python_hilbert = np.sqrt(np.mean((normalize_phase(phase_python_hilbert[:min_len]) - expected_phase_norm[:min_len])**2))
        
        print(f"  MATLAB hilbert              | {corr_matlab:10.6f}  | {rmse_matlab:.6f}")
        print(f"  Python hilbert              | {corr_python_hilbert:10.6f}  | {rmse_python_hilbert:.6f}")
        
        correlation_threshold = 0.9
        
        # Use assertions for pytest
        assert corr_matlab > correlation_threshold, f"MATLAB hilbert correlation {corr_matlab:.6f} below threshold {correlation_threshold}"
        assert corr_python_hilbert > correlation_threshold, f"Python hilbert correlation {corr_python_hilbert:.6f} below threshold {correlation_threshold}"
        
        print(f"\nAccuracy check (correlation > {correlation_threshold}):")
        print(f"  MATLAB hilbert:    [PASS]")
        print(f"  Python hilbert:    [PASS]")
        
        if _lazy_import_phase_converter():
            corr_cdm = np.corrcoef(normalize_phase(phase_cdm[:min_len]), expected_phase_norm[:min_len])[0,1]
            corr_iq = np.corrcoef(normalize_phase(phase_iq[:min_len]), expected_phase_norm[:min_len])[0,1]
            corr_fpga = np.corrcoef(normalize_phase(phase_fpga[:min_len]), expected_phase_norm[:min_len])[0,1]
            
            rmse_cdm = np.sqrt(np.mean((normalize_phase(phase_cdm[:min_len]) - expected_phase_norm[:min_len])**2))
            rmse_iq = np.sqrt(np.mean((normalize_phase(phase_iq[:min_len]) - expected_phase_norm[:min_len])**2))
            rmse_fpga = np.sqrt(np.mean((normalize_phase(phase_fpga[:min_len]) - expected_phase_norm[:min_len])**2))
            
            print(f"  Python CDM                  | {corr_cdm:10.6f}  | {rmse_cdm:.6f}")
            print(f"  Python IQ                   | {corr_iq:10.6f}  | {rmse_iq:.6f}")
            print(f"  Python FPGA                 | {corr_fpga:10.6f}  | {rmse_fpga:.6f}")
            
            assert corr_cdm > correlation_threshold, f"Python CDM correlation {corr_cdm:.6f} below threshold {correlation_threshold}"
            assert corr_iq > correlation_threshold, f"Python IQ correlation {corr_iq:.6f} below threshold {correlation_threshold}"
            assert corr_fpga > correlation_threshold, f"Python FPGA correlation {corr_fpga:.6f} below threshold {correlation_threshold}"
            
            print(f"  Python CDM:        [PASS]")
            print(f"  Python IQ:         [PASS]")
            print(f"  Python FPGA:       [PASS]")
        
    finally:
        eng.quit()


# Script execution mode (when run directly, not via pytest)
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Phase Modulation Comparison Test")
    print("MATLAB vs Python: Hilbert + Argumentation")
    print("Python Methods: CDM, IQ, FPGA")
    print("=" * 70)
    print("\nNote: Run with pytest for proper test execution:")
    print("  python -m pytest tests/analysis/test_phase_modulation_comparison.py -v")
    print("\nExecuting tests directly...\n")
    
    if HAS_MATLAB:
        # Run tests and catch exceptions
        try:
            test_constant_phase_offset()
        except Exception as e:
            print(f"Test 1 (Constant Phase) failed: {e}")
        
        try:
            test_linear_phase_evolution()
        except Exception as e:
            print(f"Test 2 (Linear Phase) failed: {e}")
        
        try:
            test_sinusoidal_phase_modulation()
        except Exception as e:
            print(f"Test 3 (Sinusoidal Phase) failed: {e}")
        
        print("\n" + "=" * 70)
        print("Direct execution completed!")
        print("=" * 70)
    else:
        print("\nMATLAB Engine not available.")
        print("Install with: python -m pip install matlabengine")

