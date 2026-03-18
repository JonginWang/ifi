#!/usr/bin/env python3
"""
PhaseChangeDetector Performance Profiling Script
===============================================

This script profiles the PhaseChangeDetector methods to identify performance bottlenecks.
Uses cProfile and snakeviz for detailed analysis.

Usage:
    python profile_phase_detector.py
"""

import cProfile
import pstats
import io
import numpy as np
import time

from ifi.utils.cache_setup import setup_project_cache
from ifi.analysis.phase_analysis import PhaseChangeDetector, SignalStacker
from ifi.analysis.spectrum import SpectrumAnalysis

# Setup project cache
cache_config = setup_project_cache()

def generate_large_test_signal(fs=1000, duration=10, signal_type='chirp'):
    """
    Generate a large test signal for profiling.
    
    Args:
        fs: Sampling frequency
        duration: Duration in seconds
        signal_type: Type of signal ('chirp', 'multi_tone', 'noise')
    
    Returns:
        Tuple of (time_axis, signal)
    """
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    if signal_type == 'chirp':
        # Linear chirp from 10Hz to 100Hz
        f0, f1 = 10, 100
        signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
    elif signal_type == 'multi_tone':
        # Multiple frequency components
        signal = (np.sin(2 * np.pi * 10 * t) + 
                 0.5 * np.sin(2 * np.pi * 50 * t) + 
                 0.3 * np.sin(2 * np.pi * 100 * t))
    elif signal_type == 'noise':
        # White noise with some structure
        signal = np.random.randn(n_samples) * 0.1
        signal += np.sin(2 * np.pi * 25 * t) * 0.5
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    return t, signal

def profile_spectrum_analysis():
    """Profile SpectrumAnalysis methods."""
    print("[PROFILE] Profiling SpectrumAnalysis...")
    
    # Generate test signal
    t, signal = generate_large_test_signal(fs=1000, duration=5, signal_type='multi_tone')
    
    # Create analyzer
    analyzer = SpectrumAnalysis()
    
    def run_analysis():
        # STFT analysis
        f, t_stft, Zxx = analyzer.compute_stft(signal, fs=1000)
        
        # CWT analysis
        cwt_result = analyzer.compute_cwt(signal, fs=1000)
        
        # Find frequency ridges (skip if not available)
        try:
            ridges = analyzer.find_freq_ridge(Zxx, f)
        except:
            ridges = None
        
        return f, t_stft, Zxx, cwt_result, ridges
    
    # Profile the analysis
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    try:
        results = run_analysis()
    finally:
        profiler.disable()
    
    end_time = time.time()
    
    print(f"   [TIME] Execution time: {end_time - start_time:.3f} seconds")
    print(f"   [DATA] Signal length: {len(signal):,} samples")
    print(f"   [SHAPE] STFT shape: {results[2].shape}")
    
    return profiler

def profile_signal_stacker():
    """Profile SignalStacker methods."""
    print("[PROFILE] Profiling SignalStacker...")
    
    # Generate test signals
    t, ref_signal = generate_large_test_signal(fs=1000, duration=5, signal_type='chirp')
    _, probe_signal = generate_large_test_signal(fs=1000, duration=5, signal_type='multi_tone')
    
    # Create stacker
    stacker = SignalStacker(fs=1000)
    
    def run_stacking_analysis():
        # Find fundamental frequency
        f0 = stacker.find_fundamental_frequency(ref_signal)
        
        # Stack signals
        stacked_signal, time_points = stacker.stack_signals(ref_signal, f0, n_stacks=5)
        
        # Compute phase difference using different methods
        phase_diff_stacking, _ = stacker.compute_phase_difference(ref_signal, probe_signal, f0, 'stacking')
        phase_diff_cordic, _, _ = stacker.compute_phase_difference_cordic(ref_signal, probe_signal, f0)
        
        return f0, stacked_signal, time_points, phase_diff_stacking, phase_diff_cordic
    
    # Profile the analysis
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    try:
        results = run_stacking_analysis()
    finally:
        profiler.disable()
    
    end_time = time.time()
    
    print(f"   [TIME] Execution time: {end_time - start_time:.3f} seconds")
    print(f"   [DATA] Signal length: {len(ref_signal):,} samples")
    print(f"   [FREQ] Fundamental frequency: {results[0]:.2f} Hz")
    
    return profiler

def profile_phase_change_detector():
    """Profile PhaseChangeDetector methods."""
    print("[PROFILE] Profiling PhaseChangeDetector...")
    
    # Generate test signals
    t, ref_signal = generate_large_test_signal(fs=1000, duration=5, signal_type='chirp')
    _, probe_signal = generate_large_test_signal(fs=1000, duration=5, signal_type='multi_tone')
    
    # Create detector
    detector = PhaseChangeDetector(fs=1000)
    
    def run_detection():
        # Unified phase change detection (without method parameter)
        result = detector.detect_phase_changes_unified(ref_signal, probe_signal)
        
        return result
    
    # Profile the analysis
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    try:
        results = run_detection()
    finally:
        profiler.disable()
    
    end_time = time.time()
    
    print(f"   [TIME] Execution time: {end_time - start_time:.3f} seconds")
    print(f"   [DATA] Signal length: {len(ref_signal):,} samples")
    print(f"   [KEYS] Results keys: {list(results.keys())}")
    
    return profiler

def analyze_profile_results(profiler, component_name):
    """Analyze and display profile results."""
    print(f"\n[ANALYSIS] Profile Analysis for {component_name}")
    print("=" * 50)
    
    # Create string buffer for stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    # Get the stats string
    stats_output = s.getvalue()
    
    # Print top functions
    lines = stats_output.split('\n')
    print("Top 20 functions by cumulative time:")
    print("-" * 50)
    
    for line in lines[5:25]:  # Skip header lines
        if line.strip():
            print(line)
    
    # Find the most time-consuming functions
    ps.sort_stats('tottime')
    s = io.StringIO()
    ps.print_stats(10)
    tottime_output = s.getvalue()
    
    print(f"\nTop 10 functions by total time:")
    print("-" * 50)
    for line in tottime_output.split('\n')[5:15]:
        if line.strip():
            print(line)

def save_profile_data(profiler, filename):
    """Save profile data to file for snakeviz analysis."""
    profiler.dump_stats(filename)
    print(f"[SAVE] Profile data saved to: {filename}")
    print(f"[VIZ] To visualize with snakeviz, run: snakeviz {filename}")

def main():
    """Main profiling function."""
    print("[START] Starting PhaseChangeDetector Performance Profiling")
    print("=" * 60)
    
    # Profile each component
    components = [
        ("SpectrumAnalysis", profile_spectrum_analysis),
        ("SignalStacker", profile_signal_stacker),
        ("PhaseChangeDetector", profile_phase_change_detector)
    ]
    
    all_profilers = {}
    
    for name, profile_func in components:
        try:
            profiler = profile_func()
            all_profilers[name] = profiler
            analyze_profile_results(profiler, name)
            save_profile_data(profiler, f"profile_{name.lower()}.prof")
            print()
        except Exception as e:
            print(f"[ERROR] Error profiling {name}: {e}")
            print()
    
    # Combined analysis
    if len(all_profilers) > 1:
        print("[COMBINED] Combined Profile Analysis")
        print("=" * 50)
        
        # Create combined profiler
        combined_profiler = cProfile.Profile()
        for profiler in all_profilers.values():
            combined_profiler.add(profiler)
        
        analyze_profile_results(combined_profiler, "Combined Analysis")
        save_profile_data(combined_profiler, "profile_combined.prof")
    
    print("\n[COMPLETE] Profiling completed!")
    print("[FILES] Profile files created:")
    print("   - profile_spectrumanalysis.prof")
    print("   - profile_signalstacker.prof") 
    print("   - profile_phasechangedetector.prof")
    print("   - profile_combined.prof")
    print("\n[VIZ] To visualize results, run:")
    print("   snakeviz profile_combined.prof")

if __name__ == "__main__":
    main()
