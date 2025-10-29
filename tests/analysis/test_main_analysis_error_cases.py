#!/usr/bin/env python3
"""
Test Error Cases for Main Analysis Module
=========================================

This script tests error handling in the main analysis module,
focusing on the complete analysis pipeline with various edge cases.
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

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
    from ifi.analysis.main_analysis import run_analysis
    from ifi.db_controller.nas_db import NAS_DB
    from ifi.db_controller.vest_db import VEST_DB
    from ifi.utils.cache_setup import setup_project_cache
    cache_config = setup_project_cache()
except ImportError as e:
    logger.error(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    sys.exit(1)


def test_main_analysis_errors():
    """Test main analysis error handling cases"""
    print("\n" + "=" * 50)
    print("Testing Main Analysis Error Cases")
    print("=" * 50)
    
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        
        # Test 1: Normal initialization
        print("\n1. Testing normal main analysis initialization...")
        try:
            # Create mock args
            class MockArgs:
                def __init__(self):
                    self.shot_number = "40245"
                    self.output_dir = str(temp_dir)
                    self.methods = ["stacking"]
                    self.fundamental_frequency = None
                    self.sampling_frequency = 1000
                    self.analysis_type = "phase"
                    self.save_results = True
                    self.verbose = True
            
            args = MockArgs()
            nas_db = NAS_DB()
            vest_db = VEST_DB()
            
            print(f"   [OK] Main analysis components initialized successfully")
        except Exception as e:
            print(f"   [ERROR] Main analysis initialization failed: {e}")
            logger.exception("Main analysis initialization error")
        
        # Test 2: Analysis with invalid shot number
        print("\n2. Testing analysis with invalid shot number...")
        try:
            args.shot_number = "999999"  # Invalid shot
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Invalid shot analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Invalid shot analysis properly failed: {e}")
        
        # Test 3: Analysis with empty shot number
        print("\n3. Testing analysis with empty shot number...")
        try:
            args.shot_number = ""
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Empty shot analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Empty shot analysis properly failed: {e}")
        
        # Test 4: Analysis with invalid output directory
        print("\n4. Testing analysis with invalid output directory...")
        try:
            args.shot_number = "40245"
            args.output_dir = "/invalid/nonexistent/path"
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Invalid output directory analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Invalid output directory analysis properly failed: {e}")
        
        # Test 5: Analysis with invalid methods
        print("\n5. Testing analysis with invalid methods...")
        try:
            args.output_dir = str(temp_dir)
            args.methods = ["invalid_method"]
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Invalid methods analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Invalid methods analysis properly failed: {e}")
        
        # Test 6: Analysis with empty methods list
        print("\n6. Testing analysis with empty methods list...")
        try:
            args.methods = []
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Empty methods analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Empty methods analysis properly failed: {e}")
        
        # Test 7: Analysis with invalid sampling frequency
        print("\n7. Testing analysis with invalid sampling frequency...")
        try:
            args.methods = ["stacking"]
            args.sampling_frequency = 0
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Invalid sampling frequency analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Invalid sampling frequency analysis properly failed: {e}")
        
        # Test 8: Analysis with negative sampling frequency
        print("\n8. Testing analysis with negative sampling frequency...")
        try:
            args.sampling_frequency = -1000
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Negative sampling frequency analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Negative sampling frequency analysis properly failed: {e}")
        
        # Test 9: Analysis with invalid fundamental frequency
        print("\n9. Testing analysis with invalid fundamental frequency...")
        try:
            args.sampling_frequency = 1000
            args.fundamental_frequency = 0
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Invalid fundamental frequency analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Invalid fundamental frequency analysis properly failed: {e}")
        
        # Test 10: Analysis with negative fundamental frequency
        print("\n10. Testing analysis with negative fundamental frequency...")
        try:
            args.fundamental_frequency = -10
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Negative fundamental frequency analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Negative fundamental frequency analysis properly failed: {e}")
        
        # Test 11: Analysis with invalid analysis type
        print("\n11. Testing analysis with invalid analysis type...")
        try:
            args.fundamental_frequency = None
            args.analysis_type = "invalid_type"
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [WARN] Invalid analysis type succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] Invalid analysis type properly failed: {e}")
        
        # Test 12: Analysis with None arguments
        print("\n12. Testing analysis with None arguments...")
        try:
            result = run_analysis(None, nas_db, vest_db)
            print(f"   [WARN] None arguments analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] None arguments analysis properly failed: {e}")
        
        # Test 13: Analysis with None databases
        print("\n13. Testing analysis with None databases...")
        try:
            result = run_analysis(args, None, None)
            print(f"   [WARN] None databases analysis succeeded: {type(result)}")
        except Exception as e:
            print(f"   [OK] None databases analysis properly failed: {e}")
        
        # Test 14: Normal operation with valid parameters
        print("\n14. Testing normal operation with valid parameters...")
        try:
            # Reset to valid parameters
            args.shot_number = "40245"
            args.output_dir = str(temp_dir)
            args.methods = ["stacking"]
            args.fundamental_frequency = None
            args.sampling_frequency = 1000
            args.analysis_type = "phase"
            args.save_results = True
            args.verbose = True
            
            result = run_analysis(args, nas_db, vest_db)
            print(f"   [OK] Normal analysis operation: {type(result)}")
        except Exception as e:
            print(f"   [ERROR] Normal analysis operation failed: {e}")
            logger.exception("Normal analysis operation error")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"   [OK] Cleanup completed")
        
    except Exception as e:
        print(f"   [ERROR] Critical error in main analysis testing: {e}")
        logger.exception("Main analysis critical error")


if __name__ == "__main__":
    print("Starting Main Analysis Error Handling Tests")
    print("=" * 60)
    test_main_analysis_errors()
    print("=" * 60)
    print("Main Analysis Error Handling Tests Completed")

