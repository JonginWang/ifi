#!/usr/bin/env python3
"""
Test Script for IDE Access to IFI Package
=========================================

Run this script in your IDE to verify that you can access all ifi modules.
This script should work in Spyder, PyCharm, VS Code, etc.
"""

import sys
from pathlib import Path

print("Testing IFI Package Access...")
print(f"Current working directory: {Path.cwd()}")
print(f"Script location: {Path(__file__).resolve()}")

# ============================================================================
# NEW PATTERN: Find ifi package root dynamically
# ============================================================================
from pathlib import Path

# Add ifi package to Python path for IDE compatibility
current_dir = Path(__file__).resolve()
ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
                + list(current_dir.parents) if p.name == 'ifi']
IFI_ROOT = ifi_parents[-1] if ifi_parents else None

try:
    sys.path.insert(0, str(IFI_ROOT))
    print(f"Found ifi package at: {IFI_ROOT}")
    print(f"Added to Python path: {IFI_ROOT.parent}")
except Exception as e:
    print(f"!! Could not find ifi package root: {e}")
    sys.exit(1)

print(f"Python path now starts with: {sys.path[0]}")

# ============================================================================
# Test all major ifi imports
# ============================================================================
print("\nTesting imports...")

try:
    from ifi.utils.common import LogManager, ensure_dir_exists
    print("ifi.utils imported successfully")
except ImportError as e:
    print(f"ifi.utils failed: {e}")

try:
    from ifi.analysis.plots import plot_waveforms, create_shot_results_directory
    print("ifi.analysis.plots imported successfully")
except ImportError as e:
    print(f"ifi.analysis.plots failed: {e}")

try:
    from ifi.db_controller.nas_db import NAS_DB
    print("ifi.db_controller.nas_db imported successfully")
except ImportError as e:
    print(f"ifi.db_controller.nas_db failed: {e}")

try:
    from ifi.analysis.spectrum import SpectrumAnalysis
    print("ifi.analysis.spectrum imported successfully")
except ImportError as e:
    print(f"ifi.analysis.spectrum failed: {e}")

try:
    from ifi.db_controller.vest_db import VEST_DB
    print("ifi.db_controller.vest_db imported successfully")
except ImportError as e:
    print(f"ifi.db_controller.vest_db failed: {e}")

try:
    from ifi.analysis.phi2ne import get_interferometry_params
    print("ifi.analysis.phi2ne imported successfully")
except ImportError as e:
    print(f"ifi.analysis.phi2ne failed: {e}")

# ============================================================================
# Test actual functionality
# ============================================================================
print("\nTesting functionality...")

try:
    # Test logging setup
    LogManager(level="INFO")
    print("LogManager setup successful")
except Exception as e:
    print(f"LogManager setup failed: {e}")

try:
    # Test spectrum analysis
    analyzer = SpectrumAnalysis()
    print("SpectrumAnalysis instantiation successful")
except Exception as e:
    print(f"SpectrumAnalysis failed: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*50)
print("IDE ACCESS TEST COMPLETE")
print("="*50)

print("\nNext Steps:")
print("1. If all tests passed, your IDE is properly configured!")
print("2. You can now run any ifi script directly in your IDE")
print("3. Use the new pattern in your own scripts:")
print("   - Add the new path finding code at the top")
print("   - Import ifi modules using absolute imports")
print("   - No need to set working directories")

print("\nFor more details, see: ifi/test/README_IDE_COMPATIBILITY.md")
