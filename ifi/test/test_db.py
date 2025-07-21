import os
import sys
import logging
import shutil

# Add the parent directory to the path to allow direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ifi.db_controller.nas_db import NAS_DB
from ifi.db_controller.vest_db import VEST_DB

# --- Test Configuration ---
# PLEASE ADJUST THESE VALUES BASED ON YOUR 'ifi/config.ini' AND AVAILABLE DATA
# SHOT_TO_TEST = 45821 # Test for CSV files, 'MDO3000orig', 'MDO3000fetch', 'MSO58'
# SHOT_TO_TEST = 41715 # Test for CSV files, 'MDO3000orig', 'MDO3000fetch'
SHOT_TO_TEST = 36853 # Test for CSV files, 'MSO58'
# SHOT_TO_TEST = 38396 # Test for 'MDO3000pc'
# SHOT_TO_TEST = 'AGC w attn I' # Test for 'ETC'
# Define the list of folders to search within the NAS mount point
DATA_FOLDER_TO_SEARCH = [
    r'6. user\Jongin Wang\IFO\Data',
    r'6. user\Byun JunHyeok\IF'
]
CONFIG_PATH = 'ifi/config.ini'
CACHE_FOLDER = './cache' # Should match [LOCAL_CACHE] dumping_folder in config

VEST_SHOT_TO_TEST = 40656
VEST_FIELD_TO_TEST = 109

def cleanup():
    """Removes the cache folder created during the test."""
    if os.path.exists(CACHE_FOLDER):
        logging.info(f"Cleaning up cache folder: {CACHE_FOLDER}")
        shutil.rmtree(CACHE_FOLDER)

def run_nas_db_test():
    """
    Executes a series of tests on the new refactored NAS_DB controller.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not os.path.exists(CONFIG_PATH):
        logging.error(f"Configuration file not found at '{CONFIG_PATH}'. Please create it from the template.")
        return

    # Start with a clean slate
    cleanup()

    try:
        logging.info("--- Starting NAS_DB Refactored Test ---")
        
        with NAS_DB(config_path=CONFIG_PATH) as nas:
            logging.info(f"Successfully initialized NAS_DB with access mode: {nas.access_mode}")

            # Test 1: Get data (should find files, parse them, and create a cache)
            logging.info(f"\n[Test 1] Fetching data for shot #{SHOT_TO_TEST} from source...")
            data_dict = nas.get_shot_data(SHOT_TO_TEST, DATA_FOLDER_TO_SEARCH)
            
            if data_dict:
                logging.info(f"   -> SUCCESS: Data loaded. Found {len(data_dict)} file(s).")
                for name, df in data_dict.items():
                    logging.info(f"      - DataFrame '{name}': Shape={df.shape}, Columns={df.columns.tolist()}")
                print("--- Data Head (first file) ---")
                print(list(data_dict.values())[0].head())
                print("------------------------------")
            else:
                logging.error("   -> FAILED: Could not retrieve any data on the first call.")
                return

            # Verify cache file creation
            expected_cache_file = os.path.join(CACHE_FOLDER, f'{SHOT_TO_TEST}.h5')
            if os.path.exists(expected_cache_file):
                logging.info(f"   -> SUCCESS: Cache file created at '{expected_cache_file}'")
            else:
                logging.warning(f"   -> WARNING: Cache file was not created at '{expected_cache_file}'.")

            # Test 2: Get data again (should load from HDF5 cache)
            logging.info(f"\n[Test 2] Fetching data for shot #{SHOT_TO_TEST} again (should use cache)...")
            cached_data_dict = nas.get_shot_data(SHOT_TO_TEST, DATA_FOLDER_TO_SEARCH)

            if cached_data_dict and isinstance(cached_data_dict, dict):
                logging.info(f"   -> SUCCESS: Data loaded from cache. Found {len(cached_data_dict)} file(s).")
                # Compare keys to ensure consistency
                if sorted(data_dict.keys()) == sorted(cached_data_dict.keys()):
                    logging.info("      - SUCCESS: Cached keys match original keys.")
                else:
                    logging.error("      - FAILED: Cached keys do not match original keys.")
            else:
                logging.error("   -> FAILED: Could not retrieve data from cache or result is not a dictionary.")

            # Test 3: Get file header
            logging.info(f"\n[Test 3] Fetching top lines for the first file of shot #{SHOT_TO_TEST}...")
            file_head = nas.get_data_top(SHOT_TO_TEST, DATA_FOLDER_TO_SEARCH, lines=10)

            if file_head:
                logging.info("   -> SUCCESS: Retrieved file head.")
                print("--- File Head ---")
                print(file_head)
                print("-----------------")
            else:
                logging.error("   -> FAILED: Could not retrieve file head.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
    finally:
        # cleanup() # Comment out to inspect cache file after test
        logging.info("\n--- NAS_DB Test Finished ---")

def run_vest_db_test():
    """
    Executes a test on the VEST_DB controller.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not os.path.exists(CONFIG_PATH):
        logging.error(f"Configuration file not found at '{CONFIG_PATH}'. Please create it from the template.")
        return

    try:
        logging.info("\n--- Starting VEST_DB Test ---")
        
        with VEST_DB(config_path=CONFIG_PATH) as db:
            logging.info(f"Checking for existence of VEST shot #{VEST_SHOT_TO_TEST}, Field #{VEST_FIELD_TO_TEST}...")
            existence = db.exist_shot(VEST_SHOT_TO_TEST, VEST_FIELD_TO_TEST)
            
            if existence > 0:
                logging.info(f"  -> SUCCESS: Shot found in table (shotDataWaveform_{existence}).")
                logging.info("\nAttempting to load data...")
                result = db.load_shot(VEST_SHOT_TO_TEST, VEST_FIELD_TO_TEST)
                
                if result:
                    time_array, data_array = result
                    logging.info("  -> SUCCESS: VEST data loaded.")
                    logging.info(f"     Time array shape: {time_array.shape}, Data array shape: {data_array.shape}")
                else:
                    logging.error("  -> FAILED: Could not load VEST data.")
            else:
                logging.warning("  -> Shot not found in VEST DB.")

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during VEST_DB test: {e}", exc_info=True)
    finally:
        logging.info("\n--- VEST_DB Test Finished ---")


if __name__ == '__main__':
    # run_nas_db_test()
    run_vest_db_test() 