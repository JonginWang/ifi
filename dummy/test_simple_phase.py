#!/usr/bin/env python3
"""
단순 위상 분석 테스트 - CDM 대신 기본 위상 계산 사용
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

def test_simple_phase_analysis():
    """단순 위상 분석 테스트"""
    print("=== 단순 위상 분석 테스트 ===")
    
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
    print(f"  - 기준 신호 범위: {ref_signal.min():.6f} ~ {ref_signal.max():.6f}")
    print(f"  - 프로브 신호 범위: {probe_signal.min():.6f} ~ {probe_signal.max():.6f}")
    
    # 2. 단순 위상 계산
    print("\n2. 단순 위상 계산...")
    
    # 복소수 신호 생성
    complex_signal = ref_signal + 1j * probe_signal
    
    # 위상 계산
    phase = np.unwrap(np.angle(complex_signal))
    
    # 기준점 조정 (첫 번째 샘플을 0으로)
    phase = phase - phase[0]
    
    print(f"  - 위상 범위: {phase.min():.6f} ~ {phase.max():.6f} rad")
    print(f"  - 위상 NaN 개수: {np.sum(np.isnan(phase))}")
    
    # 3. 밀도 계산
    print("\n3. 밀도 계산...")
    phase_converter = PhaseConverter()
    
    # 물리적 상수
    freq_hz = 94e9  # 94 GHz
    n_path = 1
    
    density = phase_converter.phase_to_density(phase, freq_hz=freq_hz, n_path=n_path)
    print(f"  - 밀도 범위: {density.min()/1e20:.2e} ~ {density.max()/1e20:.2e} m^-2")
    print(f"  - 밀도 NaN 개수: {np.sum(np.isnan(density))}")
    
    # 4. 결과 시각화
    print("\n4. 결과 시각화...")
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'TIME': t,
        'CH0': ref_signal,
        'CH1': probe_signal
    })
    
    # 밀도 데이터프레임
    density_df = pd.DataFrame({
        'TIME': t,
        'Density': density
    }).set_index('TIME')
    
    # 플롯 생성
    plotter = Plotter()
    
    # 파형 플롯
    fig1, ax1 = plotter.plot_waveforms(df, fs=fs, title="합성 신호 파형", show_plot=False)
    fig1.savefig('simple_waveforms.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  - 파형 플롯 저장: simple_waveforms.png")
    
    # 밀도 플롯
    fig2, ax2 = plotter.plot_density(density_df, title="단순 위상 밀도 분석", show_plot=False)
    fig2.savefig('simple_density.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  - 밀도 플롯 저장: simple_density.png")
    
    # 5. 통계 정보
    print("\n5. 통계 정보...")
    print(f"  - 위상 통계:")
    print(f"    * 평균: {phase.mean():.6f} rad")
    print(f"    * 표준편차: {phase.std():.6f} rad")
    print(f"    * RMS: {np.sqrt(np.mean(phase**2)):.6f} rad")
    
    print(f"  - 밀도 통계:")
    print(f"    * 평균: {density.mean()/1e20:.2e} m^-2")
    print(f"    * 표준편차: {density.std()/1e20:.2e} m^-2")
    print(f"    * RMS: {np.sqrt(np.mean(density**2))/1e20:.2e} m^-2")
    
    print("\n[SUCCESS] 단순 위상 분석 완료!")
    
    # 6. CDM과 비교
    print("\n6. CDM과 비교...")
    try:
        # CDM 방식 시도
        print("  - CDM 방식 시도 중...")
        phase_cdm = phase_converter.calc_phase_cdm(
            ref_signal, probe_signal, fs, f_if,
            isbpf=False, isconj=True, plot_filters=False
        )
        
        print(f"  - CDM 위상 범위: {phase_cdm.min():.6f} ~ {phase_cdm.max():.6f} rad")
        print(f"  - CDM 위상 NaN 개수: {np.sum(np.isnan(phase_cdm))}")
        
        # CDM 밀도 계산
        density_cdm = phase_converter.phase_to_density(phase_cdm, freq_hz=freq_hz, n_path=n_path)
        print(f"  - CDM 밀도 범위: {density_cdm.min()/1e20:.2e} ~ {density_cdm.max()/1e20:.2e} m^-2")
        print(f"  - CDM 밀도 NaN 개수: {np.sum(np.isnan(density_cdm))}")
        
        # 비교 플롯
        fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax3.plot(t, phase, label='단순 위상', alpha=0.7)
        ax3.plot(t[:len(phase_cdm)], phase_cdm, label='CDM 위상', alpha=0.7)
        ax3.set_title('위상 비교')
        ax3.set_ylabel('위상 (rad)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(t, density, label='단순 밀도', alpha=0.7)
        ax4.plot(t[:len(density_cdm)], density_cdm, label='CDM 밀도', alpha=0.7)
        ax4.set_title('밀도 비교')
        ax4.set_xlabel('시간 (s)')
        ax4.set_ylabel('밀도 (m^-2)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig3.savefig('phase_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print("  - 비교 플롯 저장: phase_comparison.png")
        
    except Exception as e:
        print(f"  - CDM 방식 실패: {e}")
        print("  - 단순 위상 방식만 사용")

if __name__ == '__main__':
    test_simple_phase_analysis()
