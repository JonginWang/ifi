#!/usr/bin/env python3
"""
Comprehensive CDM comparison tests.

This module consolidates CDM-related tests:
- Full pipeline comparison (Python vs MATLAB)
- Constant vs linear phase offset analysis
- Phase modulation comparison
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False

from ifi.analysis.phi2ne import PhaseConverter, _accumulate_phase_diff
from tests.analysis.fixtures.synthetic_signals import sine_signal


class TestCDMFullPipeline:
    """Test full CDM pipeline comparison between Python and MATLAB."""
    
    @pytest.mark.skipif(not HAS_MATLAB, reason="MATLAB Engine not available")
    def test_cdm_full_pipeline_matlab_comparison(self):
        """Compare full CDM pipeline between Python and MATLAB."""
        eng = matlab.engine.start_matlab()
        
        try:
            fs = 50e6
            f0 = 8e6
            duration = 0.010
            phase_offset = 0.75 * np.pi
            
            t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
            _, probe = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=False)
            
            # MATLAB: hilbert + demodulation
            ref_matlab = matlab.double(ref.tolist())
            probe_matlab = matlab.double(probe.tolist())
            x1_matlab = eng.hilbert(ref_matlab)
            ych5_matlab = eng.times(x1_matlab, probe_matlab)
            ych5_matlab_array = np.asarray(ych5_matlab).flatten()
            
            # Python equivalent
            x1_python = hilbert(ref)
            ych5_python = x1_python * probe
            
            # Compare demodulation
            assert np.allclose(ych5_matlab_array, ych5_python, rtol=1e-10, atol=1e-10)
            
            # Phase difference calculation
            re_matlab = np.real(ych5_matlab_array)
            im_matlab = np.imag(ych5_matlab_array)
            re_python = np.real(ych5_python)
            im_python = np.imag(ych5_python)
            
            # MATLAB method
            n_matlab = len(re_matlab) - 1
            denominator_matlab = (
                np.sqrt(re_matlab[:-1]**2 + im_matlab[:-1]**2) * 
                np.sqrt(re_matlab[1:]**2 + im_matlab[1:]**2)
            )
            denominator_matlab[denominator_matlab == 0] = 1e-12
            ratio_matlab = np.clip(
                (re_matlab[:-1] * im_matlab[1:] - im_matlab[:-1] * re_matlab[1:]) / denominator_matlab,
                -1.0, 1.0
            )
            th94_diff_matlab = np.arcsin(ratio_matlab)
            
            # Python method
            denominator_python = (
                np.sqrt(re_python[:-1]**2 + im_python[:-1]**2) * 
                np.sqrt(re_python[1:]**2 + im_python[1:]**2)
            )
            denominator_python[denominator_python == 0] = 1e-12
            ratio_python = np.clip(
                (re_python[:-1] * im_python[1:] - im_python[:-1] * re_python[1:]) / denominator_python,
                -1.0, 1.0
            )
            d_phase_python = np.arcsin(ratio_python)
            
            assert np.allclose(th94_diff_matlab, d_phase_python, rtol=1e-10, atol=1e-10)
            
            # Accumulation
            th94_accum_matlab = np.zeros(len(th94_diff_matlab))
            th94_accum_matlab[0] = th94_diff_matlab[0]
            for i in range(1, len(th94_diff_matlab)):
                th94_accum_matlab[i] = th94_diff_matlab[i] + th94_accum_matlab[i-1]
            
            phase_accum_python = _accumulate_phase_diff(d_phase_python)
            
            assert np.allclose(th94_accum_matlab, phase_accum_python[1:], rtol=1e-10, atol=1e-10)
            
        finally:
            eng.quit()
    
    @pytest.mark.skipif(not HAS_MATLAB, reason="MATLAB Engine not available")
    def test_cdm_using_phaseconverter(self):
        """Test using PhaseConverter class for complete comparison."""
        eng = matlab.engine.start_matlab()
        
        try:
            fs = 50e6
            f0 = 8e6
            duration = 0.010
            phase_offset = 0.75 * np.pi
            
            t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
            _, probe = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=False)
            
            # MATLAB CDM
            ref_matlab = matlab.double(ref.tolist())
            probe_matlab = matlab.double(probe.tolist())
            x1_matlab = eng.hilbert(ref_matlab)
            ych5_matlab = eng.times(x1_matlab, probe_matlab)
            ych5_matlab_array = np.asarray(ych5_matlab).flatten()
            
            re_matlab = np.real(ych5_matlab_array)
            im_matlab = np.imag(ych5_matlab_array)
            
            denominator = (np.sqrt(re_matlab[:-1]**2 + im_matlab[:-1]**2) * 
                          np.sqrt(re_matlab[1:]**2 + im_matlab[1:]**2))
            denominator[denominator == 0] = 1e-12
            ratio = np.clip((re_matlab[:-1] * im_matlab[1:] - im_matlab[:-1] * re_matlab[1:]) / 
                           denominator, -1.0, 1.0)
            th94_diff = np.arcsin(ratio)
            
            th94_accum = np.zeros(len(th94_diff))
            th94_accum[0] = th94_diff[0]
            for i in range(1, len(th94_diff)):
                th94_accum[i] = th94_diff[i] + th94_accum[i-1]
            
            baseline_window = min(10000, len(th94_accum))
            th94_accum_corrected = th94_accum - np.mean(th94_accum[:baseline_window])
            
            # Python PhaseConverter
            conv = PhaseConverter()
            phase_python = conv.calc_phase_cdm(
                ref, probe, fs, f0,
                isbpf=False, islpf=True, isconj=False, isold=False, isflip=False
            )
            
            assert np.allclose(th94_accum_corrected, phase_python[1:], rtol=1e-8, atol=1e-8)
            
        finally:
            eng.quit()


class TestCDMPhaseOffset:
    """Test constant vs linear phase offset differences."""
    
    def test_cdm_constant_vs_linear_phase_difference(self):
        """Investigate constant vs linear phase evolution differences."""
        fs = 50e6
        f0 = 8e6
        duration = 0.010
        phase_offset = 0.75 * np.pi
        
        t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
        _, probe_const = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=False)
        _, probe_linear = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=True)
        
        conv = PhaseConverter()
        
        phase_const = conv.calc_phase_cdm(ref, probe_const, fs, f0, isbpf=False, islpf=True, isconj=False, isold=False)
        phase_linear = conv.calc_phase_cdm(ref, probe_linear, fs, f0, isbpf=False, islpf=True, isconj=False, isold=False)
        
        # Check that both produce valid phase results
        assert len(phase_const) > 0
        assert len(phase_linear) > 0
        assert not np.any(np.isnan(phase_const))
        assert not np.any(np.isnan(phase_linear))
        
        # Check that differences exist (expected behavior)
        phase_const_diff = np.diff(phase_const)
        phase_linear_diff = np.diff(phase_linear)
        
        # Both should have similar structure but may differ in offset
        assert np.std(phase_const_diff) < 0.1  # Relatively stable
        assert np.std(phase_linear_diff) < 0.1


class TestPhaseModulation:
    """Test various phase modulation scenarios."""
    
    @pytest.mark.skipif(not HAS_MATLAB, reason="MATLAB Engine not available")
    def test_constant_phase_offset(self):
        """Test constant phase offset: φ(t) = constant."""
        eng = matlab.engine.start_matlab()
        
        try:
            fs = 50e6
            f0 = 8e6
            duration = 0.010
            phase_offset = 0.75 * np.pi
            
            t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
            _, probe = sine_signal(fs=fs, freq=f0, duration=duration, phase=phase_offset, dphidt=False)
            
            # MATLAB hilbert
            ref_matlab = matlab.double(ref.tolist())
            probe_matlab = matlab.double(probe.tolist())
            x1_matlab = eng.hilbert(ref_matlab)
            ych5_matlab = eng.times(x1_matlab, probe_matlab)
            ych5_matlab_array = np.asarray(ych5_matlab).flatten()
            
            # Python hilbert
            x1_python = hilbert(ref)
            ych5_python = x1_python * probe
            
            # Compare
            assert np.allclose(ych5_matlab_array, ych5_python, rtol=1e-10, atol=1e-10)
            
            # Python CDM
            if HAS_MATLAB:  # PhaseConverter should be available
                conv = PhaseConverter()
                phase_cdm = conv.calc_phase_cdm(ref, probe, fs, f0, isbpf=False, islpf=True, isconj=False, isold=False)
                assert len(phase_cdm) > 0
                assert not np.any(np.isnan(phase_cdm))
            
        finally:
            eng.quit()
    
    @pytest.mark.skipif(not HAS_MATLAB, reason="MATLAB Engine not available")
    def test_linear_phase_evolution(self):
        """Test linear phase evolution: φ(t) = k*t."""
        eng = matlab.engine.start_matlab()
        
        try:
            fs = 50e6
            f0 = 8e6
            duration = 0.010
            
            t, ref = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=False)
            _, probe = sine_signal(fs=fs, freq=f0, duration=duration, phase=0.0, dphidt=True)
            
            # MATLAB hilbert
            ref_matlab = matlab.double(ref.tolist())
            probe_matlab = matlab.double(probe.tolist())
            x1_matlab = eng.hilbert(ref_matlab)
            ych5_matlab = eng.times(x1_matlab, probe_matlab)
            ych5_matlab_array = np.asarray(ych5_matlab).flatten()
            
            # Python hilbert
            x1_python = hilbert(ref)
            ych5_python = x1_python * probe
            
            # Compare
            assert np.allclose(ych5_matlab_array, ych5_python, rtol=1e-10, atol=1e-10)
            
        finally:
            eng.quit()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

