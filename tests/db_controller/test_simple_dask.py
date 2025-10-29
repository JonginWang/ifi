#!/usr/bin/env python3
"""
Simple Dask Test
================

Test basic Dask functionality without external dependencies.
"""

import dask
import dask.delayed
import time
import numpy as np

def _test_function(x, delay=0.1):
    """Test function with delay."""
    time.sleep(delay)
    return x * 2

@dask.delayed
def delayed_test_function(x, delay=0.1):
    """Delayed test function with delay."""
    return _test_function(x, delay)

def main():
    """Test Dask functionality."""
    print("Simple Dask Test")
    print("=" * 30)
    
    # Test parameters
    num_tasks = 4
    delay = 0.1
    
    print(f"Testing {num_tasks} tasks with {delay}s delay each...")
    
    # Create delayed tasks
    tasks = [delayed_test_function(i, delay) for i in range(num_tasks)]
    
    # Test with threads scheduler
    print("Testing threads scheduler...")
    start_time = time.time()
    results = dask.compute(*tasks, scheduler='threads')
    end_time = time.time()
    
    total_time = end_time - start_time
    expected_time = num_tasks * delay
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Expected time: {expected_time:.3f}s")
    print(f"  Speedup: {expected_time / total_time:.2f}x")
    print(f"  Results: {results}")
    
    if len(results) == num_tasks:
        print("  [OK] All tasks completed successfully")
    else:
        print(f"  [ERROR] Expected {num_tasks} results, got {len(results)}")
    
    print("\n[OK] Dask integration test completed!")

if __name__ == "__main__":
    main()

