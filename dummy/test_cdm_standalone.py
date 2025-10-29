#!/usr/bin/env python3
"""
외부 네트워크 환경에서 MVP 패키지 테스트
NAS 연결 없이 합성 데이터를 사용한 CDM 밀도 분석 테스트
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add ifi package to path
sys.path.insert(0, str(Path(__file__).parent))

from ifi.analysis.plots import Plotter
from ifi.analysis.phi2ne import PhaseConverter
from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.utils.cache_setup import setup_project_cache
from ifi.utils.common import LogManager

# Setup logging and cache
LogManager(level="INFO")
cache_config = setup_project_cache()

def create_synthetic_shot_data(shot_num=46790, fs=100e6, duration=100e-6):
    """
    합성 Shot 데이터 생성
    실제 Shot 데이터와 유사한 특성을 가진 합성 데이터 생성
    """
    print(f"=== Shot {shot_num} 합성 데이터 생성 ===")
    
    # 시간 축 생성
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # IF 주파수 (실제 데이터와 유사하게)
    f_if = 10e6  # 10 MHz IF frequency
    f_mod = 1e6  # 1 MHz modulation
    
    # Reference signal (CH0) - 안정적인 IF 신호
    ref_signal = np.sin(2 * np.pi * f_if * t) + 0.1 * np.random.randn(n_samples)
    
    # Probe signal (CH1) - 위상 변화가 있는 IF 신호
    # 실제 플라즈마 밀도 변화를 시뮬레이션
    phase_shift_max = np.pi / 4  # 최대 45도 위상 변화
    phase_shift = phase_shift_max * (1 - np.cos(2 * np.pi * f_mod * t)) / 2
    probe_signal = np.sin(2 * np.pi * f_if * t + phase_shift) + 0.1 * np.random.randn(n_samples)
    
    # CH2 (추가 채널)
    ch2_signal = 0.5 * np.sin(2 * np.pi * f_if * t + np.pi/3) + 0.05 * np.random.randn(n_samples)
    
    # DataFrame 생성
    df = pd.DataFrame({
        'TIME': t,
        'CH0': ref_signal,
        'CH1': probe_signal,
        'CH2': ch2_signal
    })
    
    # 메타데이터 설정 (실제 데이터와 유사하게)
    df.attrs['metadata'] = {
        'time_resolution': 1/fs,
        'record_length': n_samples,
        'interferometer_frequency_hz': 94e9,  # 94 GHz 간섭계
        'interferometer_n_path': 1,  # 단일 경로
        'shot_number': shot_num
    }
    df.attrs['source_file_type'] = 'synthetic'
    df.attrs['source_file_format'] = 'test'
    
    print(f"합성 데이터 생성 완료:")
    print(f"  - 샘플 수: {n_samples:,}")
    print(f"  - 샘플링 주파수: {fs/1e6:.1f} MHz")
    print(f"  - 시간 범위: {t[0]:.6f} ~ {t[-1]:.6f} s")
    print(f"  - CH0 범위: {ref_signal.min():.6f} ~ {ref_signal.max():.6f}")
    print(f"  - CH1 범위: {probe_signal.min():.6f} ~ {probe_signal.max():.6f}")
    
    return df, fs

def test_cdm_standalone_analysis():
    """외부 네트워크 환경에서 CDM 밀도 분석 테스트"""
    print("=== 외부 네트워크 환경 CDM 밀도 분석 테스트 ===")
    
    shot_num = 46790
    
    try:
        # 1. 합성 데이터 생성
        df, fs = create_synthetic_shot_data(shot_num)
        
        # 2. 분석기 초기화
        phase_converter = PhaseConverter()
        spectrum_analyzer = SpectrumAnalysis()
        plotter = Plotter()
        
        print(f"\n=== Shot {shot_num} CDM 밀도 분석 ===")
        
        # 3. CDM 분석 수행
        ref_signal = df['CH0'].to_numpy()
        probe_signal = df['CH1'].to_numpy()
        
        # 중심 주파수 감지
        f_center = spectrum_analyzer.find_center_frequency_fft(ref_signal, fs)
        if f_center == 0.0:
            f_center = min(fs / 8, 20e6)  # 기본값 사용
            print(f"중심 주파수 감지 실패, 기본값 사용: {f_center/1e6:.2f} MHz")
        else:
            print(f"감지된 중심 주파수: {f_center/1e6:.2f} MHz")
        
        # CDM 위상 계산
        print("CDM 위상 계산 중...")
        phase_cdm = phase_converter.calc_phase_cdm(ref_signal, probe_signal, fs, f_center)
        print(f"CDM 위상 계산 완료: {phase_cdm.min():.6f} ~ {phase_cdm.max():.6f} rad")
        
        # 4. 밀도 변환
        analysis_params = {
            'freq': df.attrs['metadata']['interferometer_frequency_hz'],
            'n_path': df.attrs['metadata']['interferometer_n_path']
        }
        
        print("위상-밀도 변환 중...")
        density_cdm = phase_converter.phase_to_density(phase_cdm, analysis_params=analysis_params)
        print(f"밀도 변환 완료: {density_cdm.min():.3e} ~ {density_cdm.max():.3e} m^-2")
        
        # 5. 결과 시각화
        print("\n=== 결과 시각화 ===")
        
        # 파형 플롯
        fig_wave, ax_wave = plotter.plot_waveforms(df, fs=fs, 
                                                  title=f"Shot {shot_num} 합성 파형", 
                                                  show_plot=False)
        fig_wave.savefig(f'shot_{shot_num}_synthetic_waveforms.png', dpi=150, bbox_inches='tight')
        plt.close(fig_wave)
        print(f"[SUCCESS] 합성 파형 플롯: shot_{shot_num}_synthetic_waveforms.png")
        
        # 밀도 플롯
        density_df = pd.DataFrame({'TIME': df['TIME'], 'DENSITY': density_cdm})
        density_df.attrs.update(df.attrs)  # 메타데이터 복사
        
        fig_density, ax_density = plotter.plot_density(density_df, 
                                                      title=f"Shot {shot_num} CDM 밀도", 
                                                      show_plot=False)
        fig_density.savefig(f'shot_{shot_num}_cdm_density.png', dpi=150, bbox_inches='tight')
        plt.close(fig_density)
        print(f"[SUCCESS] CDM 밀도 플롯: shot_{shot_num}_cdm_density.png")
        
        # 6. 결과 요약
        print(f"\n=== Shot {shot_num} 분석 결과 요약 ===")
        print(f"✅ 데이터 생성: {len(df):,} 샘플")
        print(f"✅ CDM 위상 계산: {phase_cdm.min():.3f} ~ {phase_cdm.max():.3f} rad")
        print(f"✅ 밀도 변환: {density_cdm.min():.3e} ~ {density_cdm.max():.3e} m^-2")
        print(f"✅ 플롯 생성: 2개 파일")
        print(f"✅ 외부 네트워크 환경에서 MVP 패키지 정상 동작 확인")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 외부 네트워크 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_cdm_standalone_analysis()
    if success:
        print("\n[SUCCESS] 외부 네트워크 환경에서 MVP 패키지 테스트 성공!")
    else:
        print("\n[ERROR] 외부 네트워크 환경에서 MVP 패키지 테스트 실패!")
