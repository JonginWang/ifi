#!/usr/bin/env python3
"""
CDM 분석 수정 버전 - 적절한 필터 파라미터 사용
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add ifi package to path
sys.path.insert(0, str(Path(__file__).parent))

from ifi.analysis.phi2ne import PhaseConverter
from ifi.analysis.plots import Plotter
from ifi.utils.common import LogManager

# Setup logging
LogManager(level="INFO")
logger = logging.getLogger(__name__)

def test_fixed_cdm():
    """수정된 CDM 분석 테스트"""
    print("=== 수정된 CDM 분석 테스트 ===")
    
    # 1. 합성 데이터 생성
    print("1. 합성 데이터 생성...")
    fs = 100e6  # 100 MHz
    duration = 10e-6  # 10 μs
    t = np.arange(0, duration, 1/fs)
    
    # IF 주파수와 변조 주파수
    f_if = 10e6  # 10 MHz IF
    f_mod = 1e6  # 1 MHz 변조
    
    # 기준 신호 (일정한 주파수)
    ref_signal = np.sin(2 * np.pi * f_if * t)
    
    # 프로브 신호 (위상 변조)
    phase_shift = 0.1 * np.sin(2 * np.pi * f_mod * t)  # 작은 위상 변조
    probe_signal = np.sin(2 * np.pi * f_if * t + phase_shift)
    
    print(f"  - 샘플링 주파수: {fs/1e6:.1f} MHz")
    print(f"  - 신호 길이: {len(t)} 포인트")
    print(f"  - IF 주파수: {f_if/1e6:.1f} MHz")
    
    # 2. 수정된 CDM 분석
    print("\n2. 수정된 CDM 분석...")
    phase_converter = PhaseConverter()
    
    try:
        # 중심 주파수 설정
        f_center = f_if
        print(f"  - 중심 주파수: {f_center/1e6:.1f} MHz")
        
        # 수정된 필터 파라미터 (10 MHz IF에 적합)
        # 10 MHz IF 신호를 위한 더 넓은 BPF
        fcl, f0l, f0r, fcr = 5e6, 8e6, 12e6, 15e6  # 5-15 MHz 범위
        lpf_pass = 0.5e6
        lpf_stop = 1e6
        
        print(f"  - BPF 범위: {fcl/1e6:.1f}-{fcr/1e6:.1f} MHz")
        print(f"  - LPF 범위: {lpf_pass/1e6:.1f}-{lpf_stop/1e6:.1f} MHz")
        
        # 필터 계수 생성
        signal_length = len(ref_signal)
        max_taps = min(signal_length // 4, 1000)
        
        bpf_coeffs = phase_converter._create_bpf(fcl, f0l, f0r, fcr, fs)
        lpf_coeffs = phase_converter._create_lpf(lpf_pass, lpf_stop, fs, max_taps=max_taps)
        
        print(f"  - BPF 계수 길이: {len(bpf_coeffs)}")
        print(f"  - LPF 계수 길이: {len(lpf_coeffs)}")
        
        # BPF 적용
        print("  - BPF 적용 중...")
        from scipy.signal import filtfilt
        ref_bpf = filtfilt(bpf_coeffs, 1, ref_signal)
        prob_bpf = filtfilt(bpf_coeffs, 1, probe_signal)
        
        print(f"  - BPF 후 기준 신호 범위: {ref_bpf.min():.6f} ~ {ref_bpf.max():.6f}")
        print(f"  - BPF 후 프로브 신호 범위: {prob_bpf.min():.6f} ~ {prob_bpf.max():.6f}")
        
        # 복소수 복조
        print("  - 복소수 복조 중...")
        from scipy.signal import hilbert
        ref_hilbert = hilbert(ref_bpf)
        demod_signal = ref_hilbert.conj() * prob_bpf
        
        print(f"  - 복조 신호 범위 (실부): {np.real(demod_signal).min():.6f} ~ {np.real(demod_signal).max():.6f}")
        print(f"  - 복조 신호 범위 (허부): {np.imag(demod_signal).min():.6f} ~ {np.imag(demod_signal).max():.6f}")
        
        # LPF 적용
        print("  - LPF 적용 중...")
        demod_lpf = filtfilt(lpf_coeffs, 1, demod_signal)
        
        print(f"  - LPF 후 복조 신호 범위 (실부): {np.real(demod_lpf).min():.6f} ~ {np.real(demod_lpf).max():.6f}")
        print(f"  - LPF 후 복조 신호 범위 (허부): {np.imag(demod_lpf).min():.6f} ~ {np.imag(demod_lpf).max():.6f}")
        
        # 위상 계산
        print("  - 위상 계산 중...")
        re = np.real(demod_lpf)
        im = np.imag(demod_lpf)
        
        print(f"  - 실부 범위: {re.min():.6f} ~ {re.max():.6f}")
        print(f"  - 허부 범위: {im.min():.6f} ~ {im.max():.6f}")
        
        # 미분 위상 계산
        denominator = (np.sqrt(re[:-1]**2 + im[:-1]**2) * np.sqrt(re[1:]**2 + im[1:]**2))
        print(f"  - 분모 범위: {denominator.min():.6f} ~ {denominator.max():.6f}")
        print(f"  - 분모 0 개수: {np.sum(denominator == 0)}")
        
        denominator[denominator == 0] = 1e-12  # 0으로 나누기 방지
        
        # arcsin 계산
        ratio = (re[:-1] * im[1:] - im[:-1] * re[1:]) / denominator
        print(f"  - 비율 범위: {ratio.min():.6f} ~ {ratio.max():.6f}")
        
        # arcsin 입력 범위 확인
        ratio_clipped = np.clip(ratio, -1.0, 1.0)
        print(f"  - 클리핑된 비율 범위: {ratio_clipped.min():.6f} ~ {ratio_clipped.max():.6f}")
        
        d_phase = np.arcsin(ratio_clipped)
        print(f"  - 미분 위상 범위: {d_phase.min():.6f} ~ {d_phase.max():.6f}")
        print(f"  - 미분 위상 NaN 개수: {np.sum(np.isnan(d_phase))}")
        
        # 위상 누적
        from ifi.analysis.phi2ne import _accumulate_phase_diff
        phase_accum = np.concatenate(([0], _accumulate_phase_diff(d_phase)))
        
        print(f"  - 누적 위상 범위: {phase_accum.min():.6f} ~ {phase_accum.max():.6f}")
        print(f"  - 누적 위상 NaN 개수: {np.sum(np.isnan(phase_accum))}")
        
        # 기준점 조정
        if len(phase_accum) > 1000:
            phase_accum -= np.mean(phase_accum[:1000])
        
        print(f"  - 조정 후 위상 범위: {phase_accum.min():.6f} ~ {phase_accum.max():.6f}")
        print(f"  - 조정 후 위상 NaN 개수: {np.sum(np.isnan(phase_accum))}")
        
        # 3. 밀도 계산
        print("\n3. 밀도 계산...")
        freq_hz = 94e9  # 94 GHz
        n_path = 1
        
        density = phase_converter.phase_to_density(phase_accum, freq_hz=freq_hz, n_path=n_path)
        print(f"  - 밀도 범위: {density.min()/1e20:.2e} ~ {density.max()/1e20:.2e} m^-2")
        print(f"  - 밀도 NaN 개수: {np.sum(np.isnan(density))}")
        
        # 4. 결과 시각화
        print("\n4. 결과 시각화...")
        
        # 시간축 생성
        time_for_density = t[:len(density)]
        
        # 데이터프레임 생성
        density_df = pd.DataFrame({
            'TIME': time_for_density,
            'Density': density
        }).set_index('TIME')
        
        # 플롯 생성
        plotter = Plotter()
        
        fig, ax = plotter.plot_density(density_df, title="수정된 CDM 밀도 분석", show_plot=False)
        fig.savefig('fixed_cdm_density.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  - 수정된 CDM 밀도 플롯 저장: fixed_cdm_density.png")
        
        print("\n[SUCCESS] 수정된 CDM 분석 완료!")
        
    except Exception as e:
        print(f"[ERROR] 수정된 CDM 분석 실패: {e}")
        logger.error(f"Fixed CDM analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_fixed_cdm()
