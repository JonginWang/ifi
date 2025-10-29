#!/usr/bin/env python3
"""
Test script to debug PLOT functionality
"""

import sys
sys.path.insert(0, '.')
from ifi.analysis.plots import Plotter
import numpy as np
import pandas as pd

def test_plot_functionality():
    """Test PLOT functionality"""
    print("Testing PLOT functionality...")
    
    # 간단한 테스트 데이터 생성
    t = np.linspace(0, 1, 1000)
    data = pd.DataFrame({
        'TIME': t,
        'CH0': np.sin(2 * np.pi * 10 * t),
        'CH1': np.cos(2 * np.pi * 5 * t)
    })
    
    print(f"Created test data: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Plotter 테스트
    try:
        plotter = Plotter()
        print('[SUCCESS] Plotter 인스턴스 생성 성공')
        
        fig, axes = plotter.plot_waveforms(data, title='Test Plot', show_plot=False)
        print('[SUCCESS] plot_waveforms 성공')
        print(f'Figure type: {type(fig)}')
        print(f'Axes type: {type(axes)}')
        
        # Figure 저장 테스트
        fig.savefig('test_plot.png', dpi=150, bbox_inches='tight')
        print('[SUCCESS] Figure 저장 성공: test_plot.png')
        
    except Exception as e:
        print(f'[ERROR] 오류 발생: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_plot_functionality()
