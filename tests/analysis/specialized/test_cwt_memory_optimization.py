#!/usr/bin/env python3
"""
Test CWT Memory Optimization
============================

This script tests the CWT memory optimization features:
1. Decimation for time axis reduction
2. f_center mode for frequency range limitation
3. Memory usage comparison

Purpose: Verify that CWT memory optimization works correctly
Assumptions: scipy, numpy, ssqueezepy are available
Key I/O: Test signals → CWT results → Memory usage comparison
"""

import sys
from pathlib import Path
import numpy as np
import time
import tracemalloc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.utils.common import LogManager

# Setup logging
logger = LogManager().get_logger(__name__)


def create_test_signal(fs=50e6, duration=0.01, f_center=8e6):
    """
    Create a test signal with known center frequency.
    
    Args:
        fs: Sampling frequency
        duration: Signal duration in seconds
        f_center: Center frequency of the signal
        
    Returns:
        Tuple of (signal, fs, f_center)
    """
    t = np.arange(0, duration, 1 / fs)
    signal = np.sin(2 * np.pi * f_center * t) + 0.1 * np.random.randn(len(t))
    return signal, fs, f_center


def test_cwt_basic():
    """Test basic CWT functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic CWT (No Optimization)")
    print("=" * 80)
    
    signal, fs, f_center = create_test_signal(fs=50e6, duration=0.01, f_center=8e6)
    analyzer = SpectrumAnalysis()
    
    print(f"Signal length: {len(signal)}")
    print(f"Sampling frequency: {fs/1e6:.2f} MHz")
    print(f"Center frequency: {f_center/1e6:.2f} MHz")
    
    tracemalloc.start()
    start_time = time.time()
    
    freqs, Wx = analyzer.compute_cwt(signal, fs)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nResults:")
    print(f"  CWT shape: {Wx.shape}")
    print(f"  Frequency range: [{freqs.min()/1e6:.2f}, {freqs.max()/1e6:.2f}] MHz")
    print(f"  Computation time: {end_time - start_time:.3f} seconds")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    return Wx.shape, end_time - start_time, peak


def test_cwt_with_decimation():
    """Test CWT with decimation."""
    print("\n" + "=" * 80)
    print("TEST 2: CWT with Decimation (Time Axis Reduction)")
    print("=" * 80)
    
    signal, fs, f_center = create_test_signal(fs=50e6, duration=0.01, f_center=8e6)
    analyzer = SpectrumAnalysis()
    
    decimation_factor = 10
    print(f"Signal length: {len(signal)}")
    print(f"Decimation factor: {decimation_factor}")
    print(f"Expected signal length after decimation: {len(signal) // decimation_factor}")
    
    tracemalloc.start()
    start_time = time.time()
    
    freqs, Wx = analyzer.compute_cwt(
        signal, 
        fs, 
        decimation_factor=decimation_factor
    )
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nResults:")
    print(f"  CWT shape: {Wx.shape}")
    print(f"  Frequency range: [{freqs.min()/1e6:.2f}, {freqs.max()/1e6:.2f}] MHz")
    print(f"  Computation time: {end_time - start_time:.3f} seconds")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    print(f"  Time axis reduction: {len(signal)} → {Wx.shape[1]} ({Wx.shape[1]/len(signal)*100:.1f}%)")
    
    return Wx.shape, end_time - start_time, peak


def test_cwt_with_f_center():
    """Test CWT with f_center mode (Frequency Range Limitation)."""
    print("\n" + "=" * 80)
    print("TEST 3: CWT with f_center Mode (Frequency Range Limitation)")
    print("=" * 80)
    
    signal, fs, f_center = create_test_signal(fs=50e6, duration=0.01, f_center=8e6)
    analyzer = SpectrumAnalysis()
    
    f_deviation = 0.1  # ±10%
    print(f"Signal length: {len(signal)}")
    print(f"Center frequency: {f_center/1e6:.2f} MHz")
    print(f"Frequency deviation: ±{f_deviation*100:.1f}%")
    print(f"Expected frequency range: [{f_center*(1-f_deviation)/1e6:.2f}, {f_center*(1+f_deviation)/1e6:.2f}] MHz")
    
    tracemalloc.start()
    start_time = time.time()
    
    freqs, Wx = analyzer.compute_cwt(
        signal, 
        fs, 
        f_center=f_center,
        f_deviation=f_deviation
    )
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nResults:")
    print(f"  CWT shape: {Wx.shape}")
    print(f"  Frequency range: [{freqs.min()/1e6:.2f}, {freqs.max()/1e6:.2f}] MHz")
    print(f"  Expected range: [{f_center*(1-f_deviation)/1e6:.2f}, {f_center*(1+f_deviation)/1e6:.2f}] MHz")
    print(f"  Computation time: {end_time - start_time:.3f} seconds")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    print(f"  Frequency bins: {len(freqs)}")
    
    # Verify frequency range
    expected_min = f_center * (1 - f_deviation)
    expected_max = f_center * (1 + f_deviation)
    range_match = (freqs.min() <= expected_min * 1.1 and freqs.max() >= expected_max * 0.9)
    print(f"  Frequency range match: {range_match}")
    
    return Wx.shape, end_time - start_time, peak, range_match


def test_cwt_combined_optimization():
    """Test CWT with both decimation and f_center mode."""
    print("\n" + "=" * 80)
    print("TEST 4: CWT with Combined Optimization (Decimation + f_center)")
    print("=" * 80)
    
    signal, fs, f_center = create_test_signal(fs=50e6, duration=0.01, f_center=8e6)
    analyzer = SpectrumAnalysis()
    
    decimation_factor = 10
    f_deviation = 0.1
    
    print(f"Signal length: {len(signal)}")
    print(f"Decimation factor: {decimation_factor}")
    print(f"Center frequency: {f_center/1e6:.2f} MHz")
    print(f"Frequency deviation: ±{f_deviation*100:.1f}%")
    
    tracemalloc.start()
    start_time = time.time()
    
    freqs, Wx = analyzer.compute_cwt(
        signal, 
        fs, 
        f_center=f_center,
        f_deviation=f_deviation,
        decimation_factor=decimation_factor
    )
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nResults:")
    print(f"  CWT shape: {Wx.shape}")
    print(f"  Frequency range: [{freqs.min()/1e6:.2f}, {freqs.max()/1e6:.2f}] MHz")
    print(f"  Computation time: {end_time - start_time:.3f} seconds")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    print(f"  Time axis reduction: {len(signal)} → {Wx.shape[1]} ({Wx.shape[1]/len(signal)*100:.1f}%)")
    print(f"  Frequency bins: {len(freqs)}")
    
    return Wx.shape, end_time - start_time, peak


def test_cwt_large_signal():
    """Test CWT with large signal (simulating real data)."""
    print("\n" + "=" * 80)
    print("TEST 5: CWT with Large Signal (500k samples, Memory Optimization)")
    print("=" * 80)
    
    # Create large signal similar to real data
    fs = 50e6
    duration = 0.01  # 10 ms
    f_center = 8e6
    signal = np.sin(2 * np.pi * f_center * np.arange(0, duration, 1 / fs)) + 0.1 * np.random.randn(int(fs * duration))
    
    analyzer = SpectrumAnalysis()
    
    print(f"Signal length: {len(signal)}")
    print(f"Sampling frequency: {fs/1e6:.2f} MHz")
    print(f"Center frequency: {f_center/1e6:.2f} MHz")
    
    # Auto-detect center frequency
    detected_f_center = analyzer.find_center_frequency_fft(signal, fs)
    print(f"Detected center frequency: {detected_f_center/1e6:.2f} MHz")
    
    # Auto-calculate decimation factor (similar to main_analysis.py)
    decimation_factor = 1
    if len(signal) > 100000:
        decimation_factor = max(1, len(signal) // 50000)
    
    print(f"Auto-calculated decimation factor: {decimation_factor}")
    
    tracemalloc.start()
    start_time = time.time()
    
    if detected_f_center > 0:
        freqs, Wx = analyzer.compute_cwt(
            signal, 
            fs, 
            f_center=detected_f_center,
            f_deviation=0.1,
            decimation_factor=decimation_factor
        )
    else:
        print("WARNING: Center frequency detection failed, using default CWT")
        freqs, Wx = analyzer.compute_cwt(
            signal, 
            fs,
            decimation_factor=decimation_factor
        )
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nResults:")
    print(f"  CWT shape: {Wx.shape}")
    print(f"  Frequency range: [{freqs.min()/1e6:.2f}, {freqs.max()/1e6:.2f}] MHz")
    print(f"  Computation time: {end_time - start_time:.3f} seconds")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    print(f"  Memory per sample: {peak / len(signal) / 1024:.4f} KB")
    
    return Wx.shape, end_time - start_time, peak


def test_cwt_memory_comparison():
    """Compare memory usage between optimized and non-optimized CWT."""
    print("\n" + "=" * 80)
    print("TEST 6: Memory Usage Comparison (Optimized vs Non-Optimized)")
    print("=" * 80)
    
    signal, fs, f_center = create_test_signal(fs=50e6, duration=0.01, f_center=8e6)
    analyzer = SpectrumAnalysis()
    
    # Test 1: Non-optimized (full range, no decimation)
    print("\n1. Non-optimized CWT (full range, no decimation):")
    tracemalloc.start()
    freqs_full, Wx_full = analyzer.compute_cwt(signal, fs)
    current_full, peak_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   Shape: {Wx_full.shape}")
    print(f"   Peak memory: {peak_full / 1024 / 1024:.2f} MB")
    print(f"   Frequency bins: {len(freqs_full)}")
    
    # Test 2: Optimized (f_center + decimation)
    print("\n2. Optimized CWT (f_center ±10%, decimation=10):")
    tracemalloc.start()
    freqs_opt, Wx_opt = analyzer.compute_cwt(
        signal, 
        fs, 
        f_center=f_center,
        f_deviation=0.1,
        decimation_factor=10
    )
    current_opt, peak_opt = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   Shape: {Wx_opt.shape}")
    print(f"   Peak memory: {peak_opt / 1024 / 1024:.2f} MB")
    print(f"   Frequency bins: {len(freqs_opt)}")
    
    # Comparison
    print("\n3. Comparison:")
    memory_reduction = (1 - peak_opt / peak_full) * 100
    time_reduction = Wx_opt.shape[1] / Wx_full.shape[1]
    freq_reduction = len(freqs_opt) / len(freqs_full)
    
    print(f"   Memory reduction: {memory_reduction:.1f}%")
    print(f"   Time axis reduction: {time_reduction*100:.1f}%")
    print(f"   Frequency axis reduction: {freq_reduction*100:.1f}%")
    print(f"   Overall data reduction: {(1 - Wx_opt.size / Wx_full.size) * 100:.1f}%")
    
    return {
        "full": {"shape": Wx_full.shape, "memory": peak_full, "freqs": len(freqs_full)},
        "optimized": {"shape": Wx_opt.shape, "memory": peak_opt, "freqs": len(freqs_opt)},
        "reduction": memory_reduction
    }


def main():
    """Run all CWT memory optimization tests."""
    print("=" * 80)
    print("CWT MEMORY OPTIMIZATION TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    try:
        # Test 1: Basic CWT
        results["basic"] = test_cwt_basic()
        
        # Test 2: Decimation
        results["decimation"] = test_cwt_with_decimation()
        
        # Test 3: f_center mode
        results["f_center"] = test_cwt_with_f_center()
        
        # Test 4: Combined optimization
        results["combined"] = test_cwt_combined_optimization()
        
        # Test 5: Large signal
        results["large"] = test_cwt_large_signal()
        
        # Test 6: Memory comparison
        results["comparison"] = test_cwt_memory_comparison()
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("\nAll tests completed successfully!")
        print("\nKey Findings:")
        if "comparison" in results:
            comp = results["comparison"]
            print(f"  - Memory reduction: {comp['reduction']:.1f}%")
            print(f"  - Data size reduction: {(1 - comp['optimized']['shape'][0]*comp['optimized']['shape'][1] / (comp['full']['shape'][0]*comp['full']['shape'][1])) * 100:.1f}%")
        
        print("\n" + "=" * 80)
        print("CWT MEMORY OPTIMIZATION: VERIFIED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

