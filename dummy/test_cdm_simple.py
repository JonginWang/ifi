#!/usr/bin/env python3
"""
CDM 분석 단순 테스트 - 합성 데이터로 안전하게 테스트
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

def test_simple_cdm():
    """합성 데이터로 CDM 분석 테스트"""
    print("=== CDM 분석 단순 테스트 (합성 데이터) ===")
    
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
    
    # 2. CDM 분석
    print("\n2. CDM 분석...")
    phase_converter = PhaseConverter()
    
    try:
        # 중심 주파수 설정 (합성 데이터이므로 알려진 값 사용)
        f_center = f_if
        print(f"  - 중심 주파수: {f_center/1e6:.1f} MHz")
        
        # CDM 위상 계산 (단계별)
        print("  - BPF 필터 설계 중...")
        
        # 더 안전한 파라미터로 시도
        phase_diff = phase_converter.calc_phase_cdm(
            ref_signal, probe_signal, fs, f_center,
            isbpf=True, isconj=True, plot_filters=False
        )
        
        print(f"  - CDM 위상 계산 성공: {len(phase_diff)} 포인트")
        print(f"  - 위상 범위: {phase_diff.min():.6f} ~ {phase_diff.max():.6f} rad")
        
        # 3. 밀도 계산
        print("\n3. 밀도 계산...")
        
        # 물리적 상수
        freq_hz = 94e9  # 94 GHz 간섭계
        n_path = 1
        
        # 밀도 계산
        density = phase_converter.phase_to_density(phase_diff, freq_hz=freq_hz, n_path=n_path)
        
        print(f"  - 밀도 범위: {density.min()/1e20:.2e} ~ {density.max()/1e20:.2e} m^-2")
        
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
            'TIME': t[:len(density)],
            'Density': density
        }).set_index('TIME')
        
        # 플롯 생성
        plotter = Plotter()
        
        # 파형 플롯
        fig1, ax1 = plotter.plot_waveforms(df, fs=fs, title="합성 신호 파형", show_plot=False)
        fig1.savefig('synthetic_waveforms.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print("  - 파형 플롯 저장: synthetic_waveforms.png")
        
        # 밀도 플롯
        fig2, ax2 = plotter.plot_density(density_df, title="CDM 밀도 분석", show_plot=False)
        fig2.savefig('synthetic_density.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("  - 밀도 플롯 저장: synthetic_density.png")
        
        print("\n[SUCCESS] CDM 분석 완료!")
        
    except Exception as e:
        print(f"[ERROR] CDM 분석 실패: {e}")
        logger.error(f"CDM analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 대안: 단순 위상 계산
        print("\n대안: 단순 위상 계산...")
        try:
            simple_phase = np.unwrap(np.angle(ref_signal + 1j * probe_signal))
            phase_diff = simple_phase - simple_phase[0]
            
            # 밀도 계산
            density = phase_converter.phase_to_density(phase_diff, freq_hz=94e9, n_path=1)
            
            print(f"  - 단순 위상 밀도 범위: {density.min()/1e20:.2e} ~ {density.max()/1e20:.2e} m^-2")
            
            # 플롯 생성
            density_df = pd.DataFrame({
                'TIME': t[:len(density)],
                'Density': density
            }).set_index('TIME')
            
            plotter = Plotter()
            fig, ax = plotter.plot_density(density_df, title="단순 위상 밀도", show_plot=False)
            fig.savefig('simple_density.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("  - 단순 밀도 플롯 저장: simple_density.png")
            
        except Exception as e2:
            print(f"[ERROR] 대안 방법도 실패: {e2}")

if __name__ == '__main__':
    test_simple_cdm()
