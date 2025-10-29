"""
    VEST_DB
    ======

    This module contains the VEST_DB class for accessing the VEST database.
"""


import sys
import logging
from pathlib import Path

# Add ifi package to Python path for IDE compatibility
current_dir = Path(__file__).resolve()
ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
                + list(current_dir.parents) if p.name == 'ifi']
IFI_ROOT = ifi_parents[-1] if ifi_parents else None

try:
    sys.path.insert(0, str(IFI_ROOT))
except Exception as e:
    logging.error(f"Could not find ifi package root: {e}")
    pass

import pymysql
import numpy as np
import configparser
import time
from sshtunnel import SSHTunnelForwarder, BaseSSHTunnelForwarderError
import pandas as pd
from collections import defaultdict


class VEST_DB:
    """
    Handles connection and queries to the VEST database using pymysql.
    Can connect directly or through an SSH tunnel based on configuration.
    Reads connection info from a config file.
    """
    def __init__(self, config_path='ifi/config.ini'):
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please create it from 'config.ini.template'.")
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

        config = configparser.ConfigParser()
        config.read(config_path)
        
        # --- Database Configuration ---
        db_cfg = config['VEST_DB']
        self.db_host = db_cfg.get('host')
        self.db_user = db_cfg.get('user')
        self.db_password = db_cfg.get('password')
        self.db_name = db_cfg.get('database')
        self.db_port = db_cfg.getint('port', 3306)

        # This dictionary will only contain connection-specific arguments
        self.db_connection_args = {
            'host': self.db_host,
            'user': self.db_user,
            'password': self.db_password,
            'database': self.db_name,
            'port': self.db_port
        }

        # --- VEST Field Label Configuration ---
        self.field_label_file = db_cfg.get('field_label_file', fallback=None)
        self.field_labels = {}
        if self.field_label_file and Path(self.field_label_file).exists():
            try:
                label_df = pd.read_csv(self.field_label_file)
                # Create a dictionary mapping field_id to a formatted name with units
                self.field_labels = {
                    row['field_id']: f"{row['field_name']} ({row['field_unit']})"
                    for _, row in label_df.iterrows()
                }
                self.logger.info(f"Successfully loaded {len(self.field_labels)} VEST field labels from {self.field_label_file}.")
            except Exception as e:
                self.logger.error(f"Failed to load or parse VEST field label file '{self.field_label_file}': {e}")
        else:
            self.logger.warning("VEST field label file not specified or not found. Column names will be field IDs.")


        # SSH Tunnel configuration
        self.tunnel_enabled = config.getboolean('SSH_TUNNEL', 'enabled', fallback=False)
        if self.tunnel_enabled:
            ssh_cfg = config['SSH_TUNNEL']
            conn_cfg = config['CONNECTION_SETTINGS']
            
            ssh_key_path = Path(ssh_cfg.get('ssh_pkey_path')).expanduser()
            self.logger.info(f"SSH key path resolved to: {ssh_key_path}")
            if not ssh_key_path.exists():
                self.logger.warning(f"SSH private key file does not exist at '{ssh_key_path}'!")

            self.ssh_config = {
                'ssh_address_or_host': (ssh_cfg.get('ssh_host'), ssh_cfg.getint('ssh_port')),
                'ssh_username': ssh_cfg.get('ssh_user'),
                'ssh_pkey': str(ssh_key_path),
                'remote_bind_address': (ssh_cfg.get('remote_mysql_host'), self.db_port),
                'set_keepalive': 60.0
            }
            SSHTunnelForwarder.SSH_TIMEOUT = conn_cfg.getfloat('ssh_connect_timeout')
            self.ssh_max_retries = conn_cfg.getint('ssh_max_retries')
        
        self.direct_connect_timeout = config.getint('CONNECTION_SETTINGS', 'direct_connect_timeout', fallback=3)
        self.connection = None
        self.tunnel = None

    def connect(self):
        """
        Establishes a connection to the database.
        It first tries a direct connection. If that fails, it attempts
        to connect through an SSH tunnel with a retry mechanism.
        """
        # Return if already connected
        if self.connection and self.connection.open:
            return True

        # 1. Attempt direct connection
        self.logger.info("Attempting direct connection to VEST DB...")
        try:
            self.connection = pymysql.connect(**self.db_connection_args, connect_timeout=self.direct_connect_timeout)
            if self.connection.open:
                self.logger.info("Direct connection successful.")
                return True
        except pymysql.Error as err:
            self.logger.warning(f"Direct connection failed: {err}")
            if not self.tunnel_enabled:
                self.logger.error("SSH tunnel is disabled. Cannot proceed.")
                return False
            # Explicitly log the fallback attempt
            self.logger.info("Direct connection failed. Now attempting fallback to SSH tunnel.")

        # 2. Fallback to SSH tunnel connection
        self.logger.info("Falling back to SSH tunnel connection...")
        for attempt in range(self.ssh_max_retries):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{self.ssh_max_retries}...")
                self.tunnel = SSHTunnelForwarder(**self.ssh_config)
                self.tunnel.start()

                self.logger.info(f"SSH tunnel established (localhost:{self.tunnel.local_bind_port}).")
                
                # Connect to MySQL through the tunnel
                tunneled_config = self.db_connection_args.copy()
                tunneled_config['host'] = '127.0.0.1'
                tunneled_config['port'] = self.tunnel.local_bind_port
                
                self.connection = pymysql.connect(**tunneled_config)
                
                if self.connection.open:
                    self.logger.info("MySQL connection through tunnel successful.")
                    return True

            except BaseSSHTunnelForwarderError as e:
                self.logger.error(f"SSH Tunnel Error on attempt {attempt + 1}: {e}", exc_info=True)
                self.disconnect() # Cleanup
            except pymysql.Error as e:
                self.logger.error(f"MySQL Connection Error (via Tunnel) on attempt {attempt + 1}: {e}", exc_info=True)
                self.disconnect() # Cleanup
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during SSH tunnel connection on attempt {attempt+1}: {e}", exc_info=True)
                self.disconnect() # Cleanup

            if attempt < self.ssh_max_retries - 1:
                self.logger.info("Retrying in 3 seconds...")
                time.sleep(3)
        
        self.logger.error("Failed to establish a database connection.")
        return False


    def disconnect(self):
        """
        Closes the database connection and the SSH tunnel if it's active.
        """
        if self.connection and self.connection.open:
            self.connection.close()
            self.logger.info("MySQL connection closed.")
        self.connection = None

        if self.tunnel and self.tunnel.is_active:
            self.tunnel.stop()
            self.logger.info("SSH tunnel closed.")
        self.tunnel = None

    def get_next_shot_code(self) -> int | None:
        """
        Fetches the latest shot number from 'shotDataWaveform_3' and adds 1.
        :return: The next shot number as an integer, or None if failed.
        """
        if not (self.connection and self.connection.open):
            self.logger.error("Not connected to the database.")
            return None
        try:
            with self.connection.cursor() as cursor:
                query = 'SELECT shotCode FROM shotDataWaveform_3 ORDER BY shotCode DESC LIMIT 1'
                cursor.execute(query)
                result_temp = cursor.fetchone()
            
                if result_temp:
                    last_shotnum = result_temp[0]
                    next_shotnum = last_shotnum + 1
                    self.logger.info(f"Next shot number: {next_shotnum}")
                    return next_shotnum
                else:
                    # Handle case where the table is empty
                    self.logger.warning("Table 'shotDataWaveform_3' is empty or shotCode not found.")
                    return 1 # Start with 1 if table is empty

        except pymysql.Error as err:
            self.logger.error(f"Query Error: {err}")
            return None

    def exist_shot(self, shot: int, field: int) -> int:
        """
        Checks if data for a given shot and field exists in the database.
        Replicates the logic of vest_exist.m.
        Returns:
            - 3 if data is in shotDataWaveform_3 (new table)
            - 2 if data is in shotDataWaveform_2 (old table)
            - 0 if data does not exist
        """
        if not self.connect():
            self.logger.error("Failed to connect to database.")
            return 0

        try:
            with self.connection.cursor() as cursor:
                # Check shotDataWaveform_3 first as it's the newer table
                query3 = "SELECT 1 FROM shotDataWaveform_3 WHERE shotCode = %s AND shotDataFieldCode = %s LIMIT 1"
                cursor.execute(query3, (shot, field))
                if cursor.fetchone():
                    return 3

                # If not in table 3, check shotDataWaveform_2
                query2 = "SELECT 1 FROM shotDataWaveform_2 WHERE shotCode = %s AND shotDataFieldCode = %s LIMIT 1"
                cursor.execute(query2, (shot, field))
                if cursor.fetchone():
                    return 2
            
            return 0 # Not found in either table
        except pymysql.Error as err:
            self.logger.error(f"Query Error: {err}")
            return 0

    def _classify_sample_rate(self, fs: float) -> str:
        """Classifies a float sampling rate into a string key (e.g., '25k', '2M')."""
        if 20_000 <= fs < 40_000:
            return '25k'
        if 200_000 <= fs < 400_000:
            return '250k'
        if fs >= 1_500_000:
            return '2M'
        
        # Fallback for other rates
        if fs >= 1000:
            return f'{round(fs / 1000)}k'
        else:
            return f'{round(fs)}Hz'

    def load_shot(self, shot: int, fields: list[int]) -> dict[str, pd.DataFrame]:
        """
        Loads time and data for a given shot and a list of fields from the VEST database.
        It groups the resulting signals by their sampling rate into separate DataFrames.

        Returns:
            A dictionary where keys are sampling rates (e.g., '25k', '2M') and
            values are pandas DataFrames containing all signals for that rate.
        """
        if not self.connect():
            self.logger.error("Failed to connect to database.")
            return {}
        
        if shot <= 29349:
            self.logger.warning(f"Shot {shot} is too old. Only shots > 29349 in MySQL are supported in this version.")
            return {}

        grouped_series = defaultdict(list)
        for field in fields:
            time_raw, data_raw = None, None
            try:
                with self.connection.cursor() as cursor:
                    table_num = self.exist_shot(shot, field)

                    if table_num == 3:
                        query = "SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_3 WHERE shotCode = %s AND shotDataFieldCode = %s"
                        cursor.execute(query, (shot, field))
                        result = cursor.fetchone()

                        if result:
                            time_str, val_str = result
                            time_raw = np.fromstring(time_str.strip('[]'), sep=',')
                            data_raw = np.fromstring(val_str.strip('[]'), sep=',')

                    elif table_num == 2:
                        query = "SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_2 WHERE shotCode = %s AND shotDataFieldCode = %s ORDER BY shotDataWaveformTime ASC"
                        cursor.execute(query, (shot, field))
                        results = cursor.fetchall()

                        if results:
                            time_raw = np.array([row[0] for row in results])
                            data_raw = np.array([row[1] for row in results])
                    
                    if time_raw is not None and data_raw is not None:
                        # Process the loaded data and get the sampling rate
                        time_corr, data_corr, sample_rate = self.process_vest_data(shot, field, time_raw, data_raw)
                        
                        # Classify the sample rate into a key
                        rate_key = self._classify_sample_rate(sample_rate)
                        
                        series_name = self.field_labels.get(field, str(field))
                        series = pd.Series(data_corr, index=time_corr, name=series_name)
                        
                        grouped_series[rate_key].append(series)
                        self.logger.info(f"Successfully loaded and processed Shot {shot} Field {field} as '{series_name}' (Rate Group: {rate_key}).")
                    else:
                        self.logger.warning(f"Shot {shot} field {field} not found in database.")

            except pymysql.Error as err:
                self.logger.error(f"Query Error for field {field}: {err}")
                continue # Move to the next field
        
        if not grouped_series:
            return {}

        # Concatenate all series for each group into a single dataframe.
        final_dfs = {}
        for rate_key, series_list in grouped_series.items():
            final_dfs[rate_key] = pd.concat(series_list, axis=1)
            self.logger.info(f"Created DataFrame for '{rate_key}' group with {len(series_list)} signal(s).")
        
        return final_dfs


    def process_vest_data(self, shot_num: int, field_id: int, time: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Processes loaded VEST data to apply sign corrections and recalculate the time axis,

        Args:
            shot_num: The shot number.
            field_id: The data field ID.
            time: The original time array from the database.
            data: The original data array from the database.

        Returns:
            A tuple containing the corrected time array, data array, and the estimated sampling rate in Hz.
        """
        # 1. Apply sign correction based on field_id
        if field_id in [101, 214, 140]:
            data = -data
            self.logger.info(f"Flipping sign for field_id {field_id}.")

        # 2. Estimate sampling rate from the final time axis
        sample_rate = 0.0
        if len(time) > 1:
            dt = np.mean(np.diff(time))
            if dt > 0:
                sample_rate = 1.0 / dt
        
        # 3. Determine t0 and tE from tFastDAQ logic
        if round(sample_rate) >= 2e6: # 2MHz DAQ
            if shot_num >= 41660: # indentical to 250kHz DAQ
                t_start = 0.26
                t_end = 0.36
            else:
                t_start = 0.24
                t_end = 0.34
        elif round(sample_rate) >= 250e3: # 250kHz DAQ
            if shot_num >= 41660: # Includes shots >= 43685
                t_start = 0.26
                t_end = 0.36
            else:
                t_start = 0.24
                t_end = 0.34
        
        # 4. Recalculate time axis
        # The MATLAB condition `diff(time) < 1/25e3` is a bit ambiguous for a whole array.
        # It likely checks if the sampling rate is high (e.g., > 25 kHz).
        # We can check the mean difference.
        if len(time) > 1 and np.mean(np.diff(time)) < (1 / 25e3):
            # High-speed DAQ: create a new time axis
            self.logger.info(f"High-speed DAQ ({sample_rate:.0e} Hz) detected. Recalculating time axis.")
            new_time = np.linspace(t_start, t_end, len(time) + 1)
            time = new_time[:-1]
        else:
            # Low-speed DAQ: time values are already in seconds
            self.logger.info("normal-speed DAQ detected. Using original time values.")
            pass # Time is already correct


        return time, data, sample_rate

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def query(self, sql_query: str, params=None):
        """
        Execute a SQL query and return the results.
        
        Args:
            sql_query: SQL query string
            params: Optional parameters for the query
            
        Returns:
            Query results as a list of tuples, or None if failed
        """
        if not (self.connection and self.connection.open):
            self.logger.error("Not connected to the database.")
            return None
        
        try:
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(sql_query, params)
                else:
                    cursor.execute(sql_query)
                
                # Fetch all results
                results = cursor.fetchall()
                self.logger.info(f"Query executed successfully. Returned {len(results)} rows.")
                return results
                
        except pymysql.Error as e:
            self.logger.error(f"Database query error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during query execution: {e}")
            return None

if __name__ == '__main__':
    # Example usage and test for the VEST_DB class.
    # Note: This requires a valid 'ifi/config.ini' file with database credentials.
    
    # Define a test shot and a list of fields that exist in the label CSV.
    test_shot = 45821 
    test_fields = [109, 171] # Corresponds to I_p_raw and Mirnov coil

    logging.info(f"--- Testing VEST_DB with Shot #{test_shot}, Fields #{test_fields} ---")
    
    try:
        # Explicitly provide the config path for robust execution
        with VEST_DB(config_path='ifi/config.ini') as db:
            # 0. Check if the database is connected and returns the last shot number
            logging.info("Checking for last shot number...")
            last_shot_num = db.get_next_shot_code()
            if last_shot_num:
                logging.info(f"Last shot number: {last_shot_num}")
            else:
                logging.error("Failed to get last shot number.")
            
            # 1. Check if the shot data exists for the first field
            logging.info(f"Checking for existence of first field ({test_fields[0]})...")
            existence = db.exist_shot(test_shot, test_fields[0])
            if existence > 0:
                logging.info("  -> Shot seems to exist. Attempting to load multiple fields...")
                
                # 2. Load all fields into a DataFrame
                result_dfs = db.load_shot(test_shot, test_fields)
                
                if result_dfs:
                    logging.info("  -> Data loaded successfully into groups.")
                    for rate, df in result_dfs.items():
                        logging.info(f"--- Group: {rate} ---")
                        logging.info(f"     DataFrame shape: {df.shape}")
                        logging.info(f"     Columns: {df.columns.tolist()}")
                        logging.info("--- Head of DataFrame ---")
                        logging.info(f"\n{df.head()}")
                        logging.info("-------------------------")
                else:
                    logging.info("  -> Data loading failed or returned no data.")
            else:
                logging.info(f"  -> Shot {test_shot} with field {test_fields[0]} not found.")

    except FileNotFoundError as e:
        logging.error(f"\nError: {e}")
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}", exc_info=True) 