#!/usr/bin/env python3
"""
Test Frequency Detection and Plotting Functionality
====================================================

This script tests:
1. Frequency detection from config file (93-95GHz → 94GHz, 275-285GHz → 280GHz)
2. CWT plotting functionality
3. Data loading and caching with frequency information

Purpose: Verify that frequency detection works correctly and plots are generated properly
Assumptions: ifi package is installed, config files exist
Key I/O: Test results printed to console
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from ifi.analysis.phi2ne import get_interferometry_params
from ifi.analysis.plots import plot_cwt
from ifi.utils.common import LogManager
from ifi.utils.cache_setup import setup_project_cache

# Setup cache and logging
setup_project_cache()
logger = LogManager().get_logger(__name__)


def test_frequency_detection():
    """Test frequency detection from config file."""
    print("\n" + "=" * 80)
    print("TEST 1: Frequency Detection from Config File")
    print("=" * 80)
    
    # Test cases: (shot_num, filename, expected_freq_ghz, expected_group)
    test_cases = [
        (45821, "45821_056.csv", 94.0, 94.0),  # Should map to 94GHz group
        (45821, "45821_789.csv", 94.0, 94.0),  # Should map to 94GHz group
        (45821, "45821_ALL.csv", 282.0, 280.0),  # 282GHz should map to 280GHz group
    ]
    
    print("\nTesting frequency detection and grouping:")
    all_passed = True
    
    for shot_num, filename, expected_freq, expected_group in test_cases:
        params = get_interferometry_params(shot_num, filename)
        actual_freq = params.get("freq_ghz", None)
        
        # Check if frequency is in expected range for grouping
        if 93.0 <= actual_freq <= 95.0:
            actual_group = 94.0
        elif 275.0 <= actual_freq <= 285.0:
            actual_group = 280.0
        else:
            actual_group = actual_freq
        
        print(f"\n  File: {filename}")
        print(f"    Config frequency: {actual_freq} GHz")
        print(f"    Expected frequency: {expected_freq} GHz")
        print(f"    Expected group: {expected_group} GHz")
        print(f"    Actual group: {actual_group} GHz")
        
        if abs(actual_freq - expected_freq) < 0.1:
            print(f"    [OK] Frequency matches expected value")
        else:
            print(f"    [FAIL] Frequency mismatch!")
            all_passed = False
        
        if abs(actual_group - expected_group) < 0.1:
            print(f"    [OK] Group mapping correct")
        else:
            print(f"    [FAIL] Group mapping incorrect!")
            all_passed = False
    
    if all_passed:
        print("\n[PASS] All frequency detection tests passed")
    else:
        print("\n[FAIL] Some frequency detection tests failed")
    
    return all_passed


def test_cwt_plot_data_structure():
    """Test CWT plotting with mock data structure."""
    print("\n" + "=" * 80)
    print("TEST 2: CWT Plot Data Structure")
    print("=" * 80)
    
    # Create mock CWT data in the format expected by plot_cwt
    # Format: {filename: {col_name: {freq_CWT, time_CWT, CWT_matrix}}}
    print("\nCreating mock CWT data...")
    
    # Generate test signal
    fs = 250e6  # 250 MHz sampling rate
    duration = 0.001  # 1 ms
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Create a simple signal with frequency content
    f0 = 20e6  # 20 MHz center frequency
    signal = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * f0 * 1.1 * t)
    
    # Create mock CWT result
    n_freqs = 16
    n_times = n_samples // 10  # Decimated time axis
    
    freq_CWT = np.linspace(f0 * 0.9, f0 * 1.1, n_freqs)  # ±10% around f0
    time_CWT = t[::10][:n_times]  # Decimated time axis
    CWT_matrix = np.random.randn(n_freqs, n_times) + 1j * np.random.randn(n_freqs, n_times)
    CWT_matrix = np.abs(CWT_matrix)  # Magnitude
    
    # Create data structure matching main_analysis.py output
    cwt_results = {
        "test_file.csv": {
            "CH0": {
                "freq_CWT": freq_CWT,
                "time_CWT": time_CWT,
                "CWT_matrix": CWT_matrix,
            }
        }
    }
    
    print(f"  Created CWT data structure:")
    print(f"    File: test_file.csv")
    print(f"    Channel: CH0")
    print(f"    Frequency range: {freq_CWT.min()/1e6:.2f} - {freq_CWT.max()/1e6:.2f} MHz")
    print(f"    Time range: {time_CWT.min()*1e6:.2f} - {time_CWT.max()*1e6:.2f} μs")
    print(f"    CWT matrix shape: {CWT_matrix.shape}")
    
    # Test if plot_cwt can handle this structure
    print("\nTesting plot_cwt function...")
    try:
        # Note: This will try to show plots, but we'll catch any errors
        # In actual testing, you might want to disable plotting
        plot_cwt(cwt_results, title_prefix="Test: ")
        print("  [OK] plot_cwt executed without errors")
        return True
    except Exception as e:
        print(f"  [FAIL] plot_cwt failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frequency_grouping_logic():
    """Test the frequency grouping logic used in main_analysis."""
    print("\n" + "=" * 80)
    print("TEST 3: Frequency Grouping Logic")
    print("=" * 80)
    
    # Simulate the grouping logic from main_analysis.py
    test_frequencies = [
        (94.0, 94.0),   # Exact 94GHz → 94GHz group
        (93.5, 94.0),   # 93.5GHz → 94GHz group
        (95.0, 94.0),   # 95.0GHz → 94GHz group
        (282.0, 280.0), # 282GHz → 280GHz group
        (275.0, 280.0), # 275GHz → 280GHz group
        (285.0, 280.0), # 285GHz → 280GHz group
        (100.0, 100.0), # 100GHz → 100GHz group (outside ranges)
    ]
    
    print("\nTesting frequency grouping:")
    all_passed = True
    
    for freq_ghz, expected_group in test_frequencies:
        # Apply grouping logic
        if 93.0 <= freq_ghz <= 95.0:
            group_freq = 94.0
        elif 275.0 <= freq_ghz <= 285.0:
            group_freq = 280.0
        else:
            group_freq = freq_ghz
        
        print(f"  {freq_ghz} GHz → {group_freq} GHz group (expected: {expected_group} GHz)")
        
        if abs(group_freq - expected_group) < 0.1:
            print(f"    [OK]")
        else:
            print(f"    [FAIL] Expected {expected_group} GHz, got {group_freq} GHz")
            all_passed = False
    
    if all_passed:
        print("\n[PASS] All frequency grouping tests passed")
    else:
        print("\n[FAIL] Some frequency grouping tests failed")
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Frequency Detection and Plotting Test Suite")
    print("=" * 80)
    
    results = []
    
    # Test 1: Frequency detection
    results.append(("Frequency Detection", test_frequency_detection()))
    
    # Test 2: CWT plot data structure
    results.append(("CWT Plot Data Structure", test_cwt_plot_data_structure()))
    
    # Test 3: Frequency grouping logic
    results.append(("Frequency Grouping Logic", test_frequency_grouping_logic()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

