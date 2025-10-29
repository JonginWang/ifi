#!/usr/bin/env python3
"""
CDM 방식 밀도 분석 개선 테스트
Shot 46789 데이터를 사용하여 CDM 방식으로 밀도 분석 수행
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add ifi package to path
sys.path.insert(0, str(Path(__file__).parent))

from ifi.db_controller.nas_db import NAS_DB
from ifi.analysis.plots import Plotter
from ifi.analysis.phi2ne import PhaseConverter
from ifi.analysis.phase_analysis import SignalStacker
from ifi.utils.cache_setup import setup_project_cache
from ifi.utils.common import LogManager

# Setup logging and cache with DEBUG level
LogManager(level="INFO")
cache_config = setup_project_cache()

# Get logger for detailed debugging
logger = logging.getLogger(__name__)

def test_cdm_density_analysis():
    """CDM 방식 밀도 분석 테스트"""
    print("=== CDM 방식 밀도 분석 테스트 ===")
    
    shot_num = 46790
    
    try:
        with NAS_DB() as nas_db:
            # 데이터 로드
            print(f"Shot {shot_num} 데이터 로드...")
            data_dict = nas_db.get_shot_data(shot_num)
            
            if not data_dict:
                print("데이터를 로드할 수 없습니다.")
                return
            
            # 첫 번째 데이터셋 사용
            file_path, df = next(iter(data_dict.items()))
            filename = Path(file_path).name
            print(f"분석 파일: {filename}")
            print(f"데이터 Shape: {df.shape}")
            print(f"컬럼: {list(df.columns)}")
            
            # 시간축과 신호 추출
            time_axis = df['TIME'].values
            fs = 1.0 / (time_axis[1] - time_axis[0])
            print(f"샘플링 주파수: {fs/1e6:.2f} MHz")
            
            # CH0을 기준 신호, CH1을 프로브 신호로 사용
            ref_signal = df['CH0'].values
            probe_signal = df['CH1'].values
            
            print(f"기준 신호 (CH0): min={ref_signal.min():.6f}, max={ref_signal.max():.6f}")
            print(f"프로브 신호 (CH1): min={probe_signal.min():.6f}, max={probe_signal.max():.6f}")
            
            # 1. 기본 CDM 분석
            print("\n1. 기본 CDM 분석...")
            phase_converter = PhaseConverter()
            
            # 중심 주파수 자동 검출 (STFT 사용)
            from ifi.analysis.spectrum import SpectrumAnalysis
            analyzer = SpectrumAnalysis()
            f_center = analyzer.find_center_frequency_fft(ref_signal, fs)
            
            if f_center == 0.0:
                f_center = min(fs / 8, 20e6)
                print(f"중심 주파수 검출 실패, 기본값 사용: {f_center/1e6:.2f} MHz")
            else:
                print(f"검출된 중심 주파수: {f_center/1e6:.2f} MHz")
            
            # CDM 위상 계산
            try:
                print(f"CDM 분석 시작 - 중심주파수: {f_center/1e6:.2f} MHz")
                print(f"샘플링 주파수: {fs/1e6:.2f} MHz")
                print(f"신호 길이: {len(ref_signal)} 포인트")
                
                # 단계별 CDM 분석
                print("1단계: BPF 필터 설계...")
                phase_diff = phase_converter.calc_phase_cdm(
                    ref_signal, probe_signal, fs, f_center, 
                    isbpf=True, isconj=True, plot_filters=False
                )
                print(f"CDM 위상 계산 성공: {len(phase_diff)} 포인트")
                print(f"위상 범위: {phase_diff.min():.6f} ~ {phase_diff.max():.6f} rad")
                
                # 2. 개선된 밀도 계산
                print("\n2. 개선된 밀도 계산...")
                
                # 위상 차이를 적절한 범위로 제한
                phase_diff_limited = np.clip(phase_diff, -np.pi, np.pi)
                
                # 물리적 상수 설정 (282GHz 간섭계)
                freq_ghz = 282.0
                freq_hz = freq_ghz * 1e9
                n_path = 1  # 단일 경로
                
                # 밀도 계산 (개선된 공식)
                c = 3e8  # 빛의 속도
                m_e = 9.109e-31  # 전자 질량
                eps0 = 8.854e-12  # 진공 유전율
                qe = 1.602e-19  # 전자 전하
                
                # 임계 밀도
                n_c = m_e * eps0 * (2 * np.pi * freq_hz)**2 / qe**2
                print(f"임계 밀도: {n_c/1e20:.2e} m^-3")
                
                # 선적분 밀도 (개선된 공식)
                # nedl = (c * n_c / (np.pi * freq_hz)) * phase_diff_limited / n_path
                nedl = (c * n_c / (2 * np.pi * freq_hz)) * phase_diff_limited / n_path
                
                print(f"선적분 밀도 범위: {nedl.min()/1e20:.2e} ~ {nedl.max()/1e20:.2e} m^-2")
                
                # 3. 결과 시각화
                print("\n3. 결과 시각화...")
                plotter = Plotter()
                
                # 시간축 생성 (밀도 계산용)
                time_for_density = time_axis[:len(nedl)]
                
                # 밀도 데이터프레임 생성
                density_df = pd.DataFrame({
                    'TIME': time_for_density,
                    'Density_CDM': nedl
                }).set_index('TIME')
                
                # 밀도 PLOT 생성
                fig, ax = plotter.plot_density(
                    density_df, 
                    title=f"Shot {shot_num} - CDM 밀도 분석\n{filename}",
                    show_plot=False
                )
                
                # 메타데이터 정보 추가
                metadata_info = plotter._extract_metadata_info(df)
                if metadata_info:
                    fig.suptitle(f"Shot {shot_num} - CDM 밀도 분석\n{filename}\n{metadata_info}", fontsize=12)
                
                # Figure 저장
                output_file = f"shot_{shot_num}_cdm_density.png"
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"[SUCCESS] CDM 밀도 PLOT: {output_file}")
                
                # 4. 통계 정보
                print("\n4. 통계 정보...")
                print(f"위상 차이 통계:")
                print(f"  - 평균: {phase_diff_limited.mean():.6f} rad")
                print(f"  - 표준편차: {phase_diff_limited.std():.6f} rad")
                print(f"  - RMS: {np.sqrt(np.mean(phase_diff_limited**2)):.6f} rad")
                
                print(f"선적분 밀도 통계:")
                print(f"  - 평균: {nedl.mean()/1e20:.2e} m^-2")
                print(f"  - 표준편차: {nedl.std()/1e20:.2e} m^-2")
                print(f"  - RMS: {np.sqrt(np.mean(nedl**2))/1e20:.2e} m^-2")
                
                # 5. 비교 분석 (기존 방식 vs CDM)
                print("\n5. 비교 분석...")
                
                # 기존 방식 (단순 위상 계산)
                simple_phase = np.unwrap(np.angle(df['CH0'] + 1j * df['CH1']))
                simple_density = phase_converter.phase_to_density(simple_phase, freq_hz=freq_hz, n_path=n_path)
                
                print(f"기존 방식 밀도 범위: {simple_density.min()/1e20:.2e} ~ {simple_density.max()/1e20:.2e} m^-2")
                print(f"CDM 방식 밀도 범위: {nedl.min()/1e20:.2e} ~ {nedl.max()/1e20:.2e} m^-2")
                
                # 개선도 계산
                improvement_factor = np.abs(simple_density).max() / np.abs(nedl).max()
                print(f"CDM 방식 개선도: {improvement_factor:.1f}배 (기존 대비 {improvement_factor:.1f}배 작은 값)")
                
            except Exception as e:
                print(f"[ERROR] CDM 분석 실패: {e}")
                logger.error(f"CDM analysis failed: {e}")
                import traceback
                traceback.print_exc()
                
                # 대안 방법 시도
                print("\n대안 방법 시도: 단순 위상 계산...")
                try:
                    # 단순 위상 계산으로 대체
                    simple_phase = np.unwrap(np.angle(ref_signal + 1j * probe_signal))
                    phase_diff = simple_phase - simple_phase[0]  # 기준점 조정
                    print(f"단순 위상 계산 성공: {len(phase_diff)} 포인트")
                    print(f"위상 범위: {phase_diff.min():.6f} ~ {phase_diff.max():.6f} rad")
                    
                    # 밀도 계산 계속 진행
                    print("\n2. 개선된 밀도 계산...")
                    
                    # 위상 차이를 적절한 범위로 제한
                    phase_diff_limited = np.clip(phase_diff, -np.pi, np.pi)
                    
                    # 물리적 상수 설정 (282GHz 간섭계)
                    freq_ghz = 282.0
                    freq_hz = freq_ghz * 1e9
                    n_path = 1  # 단일 경로
                    
                    # 밀도 계산 (개선된 공식)
                    c = 3e8  # 빛의 속도
                    m_e = 9.109e-31  # 전자 질량
                    eps0 = 8.854e-12  # 진공 유전율
                    qe = 1.602e-19  # 전자 전하
                    
                    # 임계 밀도
                    n_c = m_e * eps0 * (2 * np.pi * freq_hz)**2 / qe**2
                    print(f"임계 밀도: {n_c/1e20:.2e} m^-3")
                    
                    # 선적분 밀도 (개선된 공식)
                    nedl = (c * n_c / (2 * np.pi * freq_hz)) * phase_diff_limited / n_path
                    
                    print(f"선적분 밀도 범위: {nedl.min()/1e20:.2e} ~ {nedl.max()/1e20:.2e} m^-2")
                    
                    # 결과 시각화
                    print("\n3. 결과 시각화...")
                    plotter = Plotter()
                    
                    # 시간축 생성 (밀도 계산용)
                    time_for_density = time_axis[:len(nedl)]
                    
                    # 밀도 데이터프레임 생성
                    density_df = pd.DataFrame({
                        'TIME': time_for_density,
                        'Density_Simple': nedl
                    }).set_index('TIME')
                    
                    # 밀도 PLOT 생성
                    fig, ax = plotter.plot_density(
                        density_df, 
                        title=f"Shot {shot_num} - 단순 위상 밀도 분석\n{filename}",
                        show_plot=False
                    )
                    
                    # Figure 저장
                    output_file = f"shot_{shot_num}_simple_density.png"
                    fig.savefig(output_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"[SUCCESS] 단순 위상 밀도 PLOT: {output_file}")
                    
                except Exception as e2:
                    print(f"[ERROR] 대안 방법도 실패: {e2}")
                    logger.error(f"Alternative method also failed: {e2}")
                    import traceback
                    traceback.print_exc()
    
    except Exception as e:
        print(f"[ERROR] 분석 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_cdm_density_analysis()
