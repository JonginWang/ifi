#!/usr/bin/env python3
"""
Test Error Cases for Phase-to-Density Conversion Module
======================================================

This script tests error handling in the phi2ne module,
focusing on phase-to-density conversion operations with various edge cases.
"""

import os
import sys
import numpy as np
from pathlib import Path

from ifi.utils.common import LogManager
LogManager(level="INFO")
logger = LogManager().get_logger(__name__)

# Add the project root to sys.path for module imports
try:
    project_root = Path(__file__).resolve().parents[2]  # Adjust ifi to be 2 levels up
    if str(project_root) not in os.environ.get('PYTHONPATH', ''):
        os.environ['PYTHONPATH'] = f"{project_root};{os.environ.get('PYTHONPATH', '')}"
except Exception as e:
    logger.error(f"Error setting project root: {e}")
    # Fallback for IDEs or other environments
    current_dir = Path(__file__).parent
    if current_dir.name == 'test':  # if in ifi/test
        project_root = current_dir.parent
    elif current_dir.name == 'ifi':  # if in ifi
        project_root = current_dir
    else:  # Assume current_dir is project root
        project_root = current_dir
    
    if str(project_root) not in os.environ.get('PYTHONPATH', ''):
        os.environ['PYTHONPATH'] = f"{project_root};{os.environ.get('PYTHONPATH', '')}"

# Ensure the project root is in sys.path for immediate imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import modules from ifi
try:
    from ifi.analysis.phi2ne import PhaseConverter
    from ifi.utils.cache_setup import setup_project_cache
    cache_config = setup_project_cache()
except ImportError as e:
    logger.error(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    sys.exit(1)


def test_phase_converter_errors():
    """Test PhaseConverter error handling cases"""
    print("\n" + "=" * 50)
    print("Testing PhaseConverter Error Cases")
    print("=" * 50)
    
    try:
        # Test 1: Normal initialization
        print("\n1. Testing normal PhaseConverter initialization...")
        converter = PhaseConverter()
        print(f"   [OK] PhaseConverter initialized successfully")
        
        # Test 2: Phase-to-density conversion with invalid parameters
        print("\n2. Testing phase-to-density conversion with invalid parameters...")
        try:
            # Empty phase array
            empty_phase = np.array([])
            result = converter.phase_to_density(empty_phase, wavelength=532e-9)
            print(f"   [WARN] Empty phase conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Empty phase conversion properly failed: {e}")
        
        try:
            # Single phase value
            single_phase = np.array([0.0])
            result = converter.phase_to_density(single_phase, wavelength=532e-9)
            print(f"   [WARN] Single phase conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Single phase conversion properly failed: {e}")
        
        try:
            # Invalid wavelength (zero)
            phase = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
            result = converter.phase_to_density(phase, wavelength=0)
            print(f"   [WARN] Zero wavelength conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Zero wavelength conversion properly failed: {e}")
        
        try:
            # Negative wavelength
            result = converter.phase_to_density(phase, wavelength=-532e-9)
            print(f"   [WARN] Negative wavelength conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Negative wavelength conversion properly failed: {e}")
        
        # Test 3: Phase-to-density conversion with NaN values
        print("\n3. Testing phase-to-density conversion with NaN values...")
        try:
            phase_nan = np.array([0.0, np.nan, np.pi, 3*np.pi/2])
            result = converter.phase_to_density(phase_nan, wavelength=532e-9)
            print(f"   [WARN] NaN phase conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] NaN phase conversion properly failed: {e}")
        
        # Test 4: Phase-to-density conversion with Inf values
        print("\n4. Testing phase-to-density conversion with Inf values...")
        try:
            phase_inf = np.array([0.0, np.inf, np.pi, 3*np.pi/2])
            result = converter.phase_to_density(phase_inf, wavelength=532e-9)
            print(f"   [WARN] Inf phase conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Inf phase conversion properly failed: {e}")
        
        # Test 5: Phase-to-density conversion with very large values
        print("\n5. Testing phase-to-density conversion with very large values...")
        try:
            phase_large = np.array([0.0, 1000*np.pi, 2000*np.pi, 3000*np.pi])
            result = converter.phase_to_density(phase_large, wavelength=532e-9)
            print(f"   [OK] Large phase values conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] Large phase values conversion failed: {e}")
        
        # Test 6: Density-to-phase conversion with invalid parameters
        print("\n6. Testing density-to-phase conversion with invalid parameters...")
        try:
            # Empty density array
            empty_density = np.array([])
            result = converter.density_to_phase(empty_density, wavelength=532e-9)
            print(f"   [WARN] Empty density conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Empty density conversion properly failed: {e}")
        
        try:
            # Single density value
            single_density = np.array([1e18])
            result = converter.density_to_phase(single_density, wavelength=532e-9)
            print(f"   [WARN] Single density conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Single density conversion properly failed: {e}")
        
        try:
            # Negative density values
            density_negative = np.array([1e18, -1e18, 2e18, 3e18])
            result = converter.density_to_phase(density_negative, wavelength=532e-9)
            print(f"   [WARN] Negative density conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Negative density conversion properly failed: {e}")
        
        # Test 7: Density-to-phase conversion with NaN values
        print("\n7. Testing density-to-phase conversion with NaN values...")
        try:
            density_nan = np.array([1e18, np.nan, 2e18, 3e18])
            result = converter.density_to_phase(density_nan, wavelength=532e-9)
            print(f"   [WARN] NaN density conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] NaN density conversion properly failed: {e}")
        
        # Test 8: Density-to-phase conversion with Inf values
        print("\n8. Testing density-to-phase conversion with Inf values...")
        try:
            density_inf = np.array([1e18, np.inf, 2e18, 3e18])
            result = converter.density_to_phase(density_inf, wavelength=532e-9)
            print(f"   [WARN] Inf density conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Inf density conversion properly failed: {e}")
        
        # Test 9: Normal operation with valid data
        print("\n9. Testing normal operation with valid data...")
        try:
            # Valid phase-to-density conversion
            phase = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
            density = converter.phase_to_density(phase, wavelength=532e-9)
            print(f"   [OK] Normal phase-to-density conversion: {type(density)}")
            
            # Valid density-to-phase conversion
            phase_back = converter.density_to_phase(density, wavelength=532e-9)
            print(f"   [OK] Normal density-to-phase conversion: {type(phase_back)}")
        except Exception as e:
            print(f"   [ERROR] Normal operation failed: {e}")
            logger.exception("Normal operation error")
        
        # Test 10: Edge case - very small density values
        print("\n10. Testing with very small density values...")
        try:
            small_density = np.array([1e10, 1e11, 1e12, 1e13])  # Very small densities
            result = converter.density_to_phase(small_density, wavelength=532e-9)
            print(f"   [OK] Small density conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] Small density conversion failed: {e}")
        
        # Test 11: Edge case - very large density values
        print("\n11. Testing with very large density values...")
        try:
            large_density = np.array([1e20, 1e21, 1e22, 1e23])  # Very large densities
            result = converter.density_to_phase(large_density, wavelength=532e-9)
            print(f"   [OK] Large density conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] Large density conversion failed: {e}")
        
        # Test 12: Edge case - very small wavelength
        print("\n12. Testing with very small wavelength...")
        try:
            phase = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
            result = converter.phase_to_density(phase, wavelength=1e-12)  # Very small wavelength
            print(f"   [OK] Small wavelength conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] Small wavelength conversion failed: {e}")
        
        # Test 13: Edge case - very large wavelength
        print("\n13. Testing with very large wavelength...")
        try:
            phase = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
            result = converter.phase_to_density(phase, wavelength=1e-3)  # Very large wavelength
            print(f"   [OK] Large wavelength conversion succeeded: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] Large wavelength conversion failed: {e}")
        
    except Exception as e:
        print(f"   [ERROR] Critical error in PhaseConverter testing: {e}")
        logger.exception("PhaseConverter critical error")


if __name__ == "__main__":
    print("Starting Phase-to-Density Conversion Error Handling Tests")
    print("=" * 60)
    test_phase_converter_errors()
    print("=" * 60)
    print("Phase-to-Density Conversion Error Handling Tests Completed")

