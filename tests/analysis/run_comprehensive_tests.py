#!/usr/bin/env python3
"""
Comprehensive Test Runner
=========================

Runs all comprehensive tests for the IFI package.
Provides detailed reporting and coverage analysis.
"""

import sys
import subprocess
import time

def run_test_suite(test_file, test_name):
    """Run a specific test suite."""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Duration: {duration:.2f}s")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        return {
            'name': test_name,
            'file': test_file,
            'success': result.returncode == 0,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"Test {test_name} timed out after 300 seconds")
        return {
            'name': test_name,
            'file': test_file,
            'success': False,
            'duration': 300,
            'stdout': '',
            'stderr': 'Test timed out'
        }
    except Exception as e:
        print(f"Error running {test_name}: {e}")
        return {
            'name': test_name,
            'file': test_file,
            'success': False,
            'duration': 0,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    """Main test runner."""
    print("IFI Package Comprehensive Test Suite")
    print("=" * 60)
    
    # Define test suites
    test_suites = [
        {
            'file': 'ifi/test/test_core_functionality.py',
            'name': 'Core Functionality Tests'
        },
        {
            'file': 'ifi/test/test_performance_comprehensive.py',
            'name': 'Performance Tests'
        },
        {
            'file': 'ifi/test/test_integration_comprehensive.py',
            'name': 'Integration Tests'
        }
    ]
    
    # Run all test suites
    results = []
    total_start_time = time.time()
    
    for suite in test_suites:
        result = run_test_suite(suite['file'], suite['name'])
        results.append(result)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"Total test suites: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {len(successful_tests) / len(results) * 100:.1f}%")
    print(f"Total duration: {total_duration:.2f}s")
    
    print(f"\n{'='*60}")
    print("DETAILED RESULTS")
    print(f"{'='*60}")
    
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{status:4} | {result['name']:30} | {result['duration']:6.2f}s")
        
        if not result['success'] and result['stderr']:
            print(f"      | Error: {result['stderr'][:100]}...")
    
    if failed_tests:
        print(f"\n{'='*60}")
        print("FAILED TESTS DETAILS")
        print(f"{'='*60}")
        
        for result in failed_tests:
            print(f"\n{result['name']}:")
            print(f"  File: {result['file']}")
            print(f"  Duration: {result['duration']:.2f}s")
            if result['stderr']:
                print(f"  Error: {result['stderr']}")
    
    # Overall success
    overall_success = len(failed_tests) == 0
    
    print(f"\n{'='*60}")
    print("OVERALL RESULT")
    print(f"{'='*60}")
    
    if overall_success:
        print("[OK] ALL TESTS PASSED")
        print("The IFI package is functioning correctly!")
    else:
        print("[ERROR] SOME TESTS FAILED")
        print("Please review the failed tests and fix the issues.")
    
    return overall_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
