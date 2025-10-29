#!/usr/bin/env python3
"""
CORDIC Algorithm Validation Test
===============================

This script validates that vectorized CORDIC produces the same results as individual CORDIC operations.
Tests both accuracy and performance improvements.

Usage:
    python test_cordic_validation.py
"""

import numpy as np
import time

from ifi.utils.cache_setup import setup_project_cache
from ifi.analysis.phase_analysis import CORDICProcessor

# Setup project cache
cache_config = setup_project_cache()

def generate_test_data(n_samples=1000):
    """Generate test data for CORDIC validation."""
    np.random.seed(42)  # For reproducible results
    
    # Generate random complex numbers
    x = np.random.randn(n_samples) * 10
    y = np.random.randn(n_samples) * 10
    target_angles = np.random.uniform(-np.pi, np.pi, n_samples)
    
    return x, y, target_angles

def test_individual_cordic(x=None, y=None, target_angles=None):
    """Test individual CORDIC operations."""
    print("[TEST] Testing individual CORDIC operations...")
    
    # Generate test data if not provided
    if x is None or y is None or target_angles is None:
        n_samples = 1000
        x, y, target_angles = generate_test_data(n_samples)
    
    # Initialize processor
    processor = CORDICProcessor()
    
    start_time = time.time()
    
    magnitudes = []
    phases = []
    
    for i in range(len(x)):
        mag, phase, _ = processor.cordic_rotation(x[i], y[i], target_angles[i])
        magnitudes.append(mag)
        phases.append(phase)
    
    end_time = time.time()
    
    return np.array(magnitudes), np.array(phases), end_time - start_time

def test_vectorized_cordic(x=None, y=None, target_angles=None):
    """Test vectorized CORDIC operations."""
    print("[TEST] Testing vectorized CORDIC operations...")
    
    # Generate test data if not provided
    if x is None or y is None or target_angles is None:
        n_samples = 1000
        x, y, target_angles = generate_test_data(n_samples)
    
    # Initialize processor
    processor = CORDICProcessor()
    
    start_time = time.time()
    
    magnitudes, phases, _ = processor.cordic_rotation_vectorized(x, y, target_angles)
    
    end_time = time.time()
    
    return magnitudes, phases, end_time - start_time

def compare_results(mag1, phase1, mag2, phase2, tolerance=1e-10):
    """Compare results from individual vs vectorized CORDIC."""
    print("[COMPARE] Comparing individual vs vectorized results...")
    
    # Compare magnitudes
    mag_diff = np.abs(mag1 - mag2)
    mag_max_diff = np.max(mag_diff)
    mag_mean_diff = np.mean(mag_diff)
    
    # Compare phases
    phase_diff = np.abs(phase1 - phase2)
    # Handle phase wrapping
    phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
    phase_max_diff = np.max(phase_diff)
    phase_mean_diff = np.mean(phase_diff)
    
    print(f"   [MAGNITUDE] Max diff: {mag_max_diff:.2e}, Mean diff: {mag_mean_diff:.2e}")
    print(f"   [PHASE] Max diff: {phase_max_diff:.2e}, Mean diff: {phase_mean_diff:.2e}")
    
    # Check if results are within tolerance
    mag_ok = mag_max_diff < tolerance
    phase_ok = phase_max_diff < tolerance
    
    if mag_ok and phase_ok:
        print("   [RESULT] [OK] Results match within tolerance!")
        assert True
    else:
        print("   [RESULT] [ERROR] Results differ beyond tolerance!")
        if not mag_ok:
            print(f"      Magnitude max difference: {mag_max_diff:.2e} > {tolerance:.2e}")
        if not phase_ok:
            print(f"      Phase max difference: {phase_max_diff:.2e} > {tolerance:.2e}")
        return False

def test_performance_scaling():
    """Test performance scaling with different array sizes."""
    print("\n[PERFORMANCE] Testing performance scaling...")
    
    processor = CORDICProcessor()
    
    sizes = [100, 500, 1000, 2000, 5000]
    
    print("Array Size | Individual (s) | Vectorized (s) | Speedup")
    print("-" * 55)
    
    for size in sizes:
        # Generate test data for this size
        x, y, target_angles = generate_test_data(size)
        
        # Test individual
        _, _, time_individual = test_individual_cordic(x, y, target_angles)
        
        # Test vectorized
        _, _, time_vectorized = test_vectorized_cordic(x, y, target_angles)
        
        speedup = time_individual / time_vectorized if time_vectorized > 0 else float('inf')
        
        print(f"{size:10d} | {time_individual:13.4f} | {time_vectorized:12.4f} | {speedup:7.2f}x")

def test_accuracy_with_different_inputs():
    """Test accuracy with various input types."""
    print("\n[ACCURACY] Testing accuracy with different input types...")
    
    processor = CORDICProcessor()
    
    # Test cases
    test_cases = [
        ("Small values", np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])),
        ("Large values", np.array([100, 200, 300]), np.array([100, 200, 300]), np.array([0.1, 0.2, 0.3])),
        ("Mixed signs", np.array([1, -1, 2]), np.array([-1, 1, -2]), np.array([0.5, -0.5, 1.0])),
        ("Zero angles", np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([0, 0, 0])),
        ("Extreme angles", np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([np.pi, -np.pi, np.pi/2]))
    ]
    
    for name, x, y, target_angles in test_cases:
        print(f"\n   Testing {name}...")
        
        # Individual CORDIC
        mag1, phase1, _ = test_individual_cordic(x, y, target_angles)
        
        # Vectorized CORDIC
        mag2, phase2, _ = test_vectorized_cordic(x, y, target_angles)
        
        # Compare results
        success = compare_results(mag1, phase1, mag2, phase2, tolerance=1e-8)
        
        if not success:
            print(f"   [WARNING] {name} test failed!")
            return False
    
    print("   [SUCCESS] All accuracy tests passed!")
    return True

def main():
    """Main validation function."""
    print("=" * 60)
    print("CORDIC Algorithm Validation Test")
    print("=" * 60)
    
    # Initialize CORDIC processor
    processor = CORDICProcessor()
    
    # Generate test data
    print("[SETUP] Generating test data...")
    x, y, target_angles = generate_test_data(1000)
    print(f"   Generated {len(x)} test samples")
    
    # Test individual CORDIC
    print("\n[TEST 1] Individual CORDIC operations...")
    mag1, phase1, time1 = test_individual_cordic(x, y, target_angles)
    print(f"   Time: {time1:.4f} seconds")
    
    # Test vectorized CORDIC
    print("\n[TEST 2] Vectorized CORDIC operations...")
    mag2, phase2, time2 = test_vectorized_cordic(x, y, target_angles)
    print(f"   Time: {time2:.4f} seconds")
    
    # Compare results
    print("\n[COMPARISON] Comparing results...")
    success = compare_results(mag1, phase1, mag2, phase2)
    
    if success:
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"\n[PERFORMANCE] Speedup: {speedup:.2f}x")
        
        # Additional tests
        test_accuracy_with_different_inputs()
        test_performance_scaling()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] CORDIC validation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[FAILED] CORDIC validation failed!")
        print("=" * 60)
        return False
    
    return True

if __name__ == "__main__":
    main()

