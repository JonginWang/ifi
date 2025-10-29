#!/usr/bin/env python3
"""
Test script for DB error handling cases
"""

from pathlib import Path
from ifi.utils.common import LogManager

# Initialize logging
LogManager(level="INFO")
logger = LogManager().get_logger(__name__)

def test_nas_db_errors():
    """Test NAS_DB error handling cases"""
    print("=" * 50)
    print("Testing NAS_DB Error Cases")
    print("=" * 50)
    
    try:
        from ifi.db_controller.nas_db import NAS_DB
        
        # Test 1: Normal initialization
        print("\n1. Testing normal NAS_DB initialization...")
        nas_db = NAS_DB()
        print(f"   [OK] NAS_DB initialized successfully")
        print(f"   - NAS path: {nas_db.nas_path}")
        print(f"   - Access mode: {nas_db.access_mode}")
        
        # Test 2: Connection test
        print("\n2. Testing NAS_DB connection...")
        connected = nas_db.connect()
        print(f"   Connection result: {connected}")
        
        if connected:
            # Test 3: File search with valid shot
            print("\n3. Testing file search with valid shot (40245)...")
            try:
                files = nas_db.find_files('40245')
                print(f"   Found {len(files)} files for shot 40245")
                
                if files:
                    # Test 4: Data loading
                    print("\n4. Testing data loading...")
                    data = nas_db.get_shot_data('40245')
                    if data is not None:
                        print(f"   [OK] Data loaded successfully")
                        if hasattr(data, 'shape'):
                            print(f"   - Data shape: {data.shape}")
                        if hasattr(data, 'columns'):
                            print(f"   - Columns: {list(data.columns)}")
                    else:
                        print("   [WARN] No data loaded")
                else:
                    print("   [WARN] No files found")
            except Exception as e:
                print(f"   [ERROR] Error during file search/data loading: {e}")
                logger.exception("File search/data loading error")
        else:
            print("   [WARN] Connection failed - this is expected in test environment")
            
        # Test 5: Invalid shot number
        print("\n5. Testing with invalid shot number...")
        try:
            files = nas_db.find_files('999999')
            print(f"   Found {len(files)} files for invalid shot (expected: 0)")
        except Exception as e:
            print(f"   [ERROR] Error with invalid shot: {e}")
            
        # Test 6: Empty shot number
        print("\n6. Testing with empty shot number...")
        try:
            files = nas_db.find_files('')
            print(f"   Found {len(files)} files for empty shot (expected: 0)")
        except Exception as e:
            print(f"   [ERROR] Error with empty shot: {e}")
            
    except Exception as e:
        print(f"   [ERROR] Critical error in NAS_DB testing: {e}")
        logger.exception("NAS_DB critical error")

def test_vest_db_errors():
    """Test VEST_DB error handling cases"""
    print("\n" + "=" * 50)
    print("Testing VEST_DB Error Cases")
    print("=" * 50)
    
    try:
        from ifi.db_controller.vest_db import VEST_DB
        
        # Test 1: Normal initialization
        print("\n1. Testing normal VEST_DB initialization...")
        vest_db = VEST_DB()
        print(f"   [OK] VEST_DB initialized successfully")
        
        # Test 2: Connection test
        print("\n2. Testing VEST_DB connection...")
        try:
            connected = vest_db.connect()
            print(f"   Connection result: {connected}")
            
            if connected:
                # Test 3: Valid query
                print("\n3. Testing valid query...")
                try:
                    result = vest_db.query("SELECT COUNT(*) as count FROM shot_info LIMIT 1")
                    if result is not None:
                        print(f"   [OK] Query successful")
                        print(f"   - Result: {result}")
                    else:
                        print("   [WARN] Query returned None")
                except Exception as e:
                    print(f"   [ERROR] Error during valid query: {e}")
                    logger.exception("Valid query error")
                
                # Test 4: Invalid query
                print("\n4. Testing invalid query...")
                try:
                    result = vest_db.query("SELECT * FROM non_existent_table")
                    print(f"   [WARN] Invalid query unexpectedly succeeded: {result}")
                except Exception as e:
                    print(f"   [OK] Invalid query properly failed: {e}")
                
                # Test 5: SQL injection attempt
                print("\n5. Testing SQL injection protection...")
                try:
                    malicious_input = "'; DROP TABLE shot_info; --"
                    result = vest_db.query(f"SELECT * FROM shot_info WHERE shot_number = '{malicious_input}'")
                    print(f"   [WARN] SQL injection attempt succeeded: {result}")
                except Exception as e:
                    print(f"   [OK] SQL injection properly blocked: {e}")
                    
            else:
                print("   [WARN] Connection failed - this is expected in test environment")
                
        except Exception as e:
            print(f"   [ERROR] Error during VEST_DB connection: {e}")
            logger.exception("VEST_DB connection error")
            
    except Exception as e:
        print(f"   [ERROR] Critical error in VEST_DB testing: {e}")
        logger.exception("VEST_DB critical error")

def test_file_io_errors():
    """Test file I/O error handling cases"""
    print("\n" + "=" * 50)
    print("Testing File I/O Error Cases")
    print("=" * 50)
    
    try:
        from ifi.utils.file_io import save_results_to_hdf5
        import pandas as pd
        import numpy as np
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        print(f"\n1. Testing HDF5 save with valid data...")
        
        try:
            # Test with valid data
            t = np.linspace(0, 1, 1000)
            signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
            data = pd.DataFrame({'TIME': t, 'SIGNAL': signal})
            
            save_results_to_hdf5(
                output_dir=str(temp_dir),
                shot_num=1,
                signals={'test_signal': data},
                stft_results={},
                cwt_results={},
                density_data=pd.DataFrame(),
                vest_data=pd.DataFrame()
            )
            print("   [OK] HDF5 save successful")
            
        except Exception as e:
            print(f"   [ERROR] Error during HDF5 save: {e}")
            logger.exception("HDF5 save error")
        
        # Test 2: Invalid directory
        print("\n2. Testing HDF5 save with invalid directory...")
        try:
            save_results_to_hdf5(
                output_dir="/invalid/nonexistent/path",
                shot_num=2,
                signals={'test_signal': data},
                stft_results={},
                cwt_results={},
                density_data=pd.DataFrame(),
                vest_data=pd.DataFrame()
            )
            print("   [WARN] HDF5 save with invalid directory succeeded")
        except Exception as e:
            print(f"   [OK] HDF5 save with invalid directory properly failed: {e}")
        
        # Test 3: Invalid data types
        print("\n3. Testing HDF5 save with invalid data types...")
        try:
            save_results_to_hdf5(
                output_dir=str(temp_dir),
                shot_num=3,
                signals={'invalid_signal': "not_a_dataframe"},
                stft_results={},
                cwt_results={},
                density_data=pd.DataFrame(),
                vest_data=pd.DataFrame()
            )
            print("   [WARN] HDF5 save with invalid data succeeded")
        except Exception as e:
            print(f"   [OK] HDF5 save with invalid data properly failed: {e}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"   [OK] Cleanup completed")
        
    except Exception as e:
        print(f"   [ERROR] Critical error in file I/O testing: {e}")
        logger.exception("File I/O critical error")

if __name__ == "__main__":
    print("Starting DB Error Handling Tests")
    print("=" * 60)
    
    # Test NAS_DB error cases
    test_nas_db_errors()
    
    # Test VEST_DB error cases  
    test_vest_db_errors()
    
    # Test file I/O error cases
    test_file_io_errors()
    
    print("\n" + "=" * 60)
    print("DB Error Handling Tests Completed")

