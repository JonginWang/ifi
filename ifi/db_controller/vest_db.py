import pymysql
import numpy as np
import configparser
import os
import time
import logging
from sshtunnel import SSHTunnelForwarder, BaseSSHTunnelForwarderError
import pandas as pd

class VEST_DB:
    """
    Handles connection and queries to the VEST database using pymysql.
    Can connect directly or through an SSH tunnel based on configuration.
    Reads connection info from a config file.
    """
    def __init__(self, config_path='ifi/config.ini'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please create it from 'config.ini.template'.")
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        config = configparser.ConfigParser()
        config.read(config_path)
        
        db_cfg = config['VEST_DB']
        self.db_config = {
            'host': db_cfg.get('host'),
            'user': db_cfg.get('user'),
            'password': db_cfg.get('password'),
            'database': db_cfg.get('database'),
            'port': db_cfg.getint('port', 3306) # Default MySQL port
        }

        # SSH Tunnel configuration
        self.tunnel_enabled = config.getboolean('SSH_TUNNEL', 'enabled', fallback=False)
        if self.tunnel_enabled:
            ssh_cfg = config['SSH_TUNNEL']
            conn_cfg = config['CONNECTION_SETTINGS']
            
            self.ssh_config = {
                'ssh_address_or_host': (ssh_cfg.get('ssh_host'), ssh_cfg.getint('ssh_port')),
                'ssh_username': ssh_cfg.get('ssh_user'),
                'ssh_pkey': os.path.expanduser(ssh_cfg.get('ssh_pkey_path')),
                'remote_bind_address': (ssh_cfg.get('remote_mysql_host'), self.db_config['port']),
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
            self.connection = pymysql.connect(**self.db_config, connect_timeout=self.direct_connect_timeout)
            if self.connection.open:
                self.logger.info("Direct connection successful.")
                return True
        except pymysql.Error as err:
            self.logger.warning(f"Direct connection failed: {err}")
            if not self.tunnel_enabled:
                self.logger.error("SSH tunnel is disabled. Cannot proceed.")
                return False

        # 2. Fallback to SSH tunnel connection
        self.logger.info("Falling back to SSH tunnel connection...")
        for attempt in range(self.ssh_max_retries):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{self.ssh_max_retries}...")
                self.tunnel = SSHTunnelForwarder(**self.ssh_config)
                self.tunnel.start()

                self.logger.info(f"SSH tunnel established (localhost:{self.tunnel.local_bind_port}).")
                
                # Connect to MySQL through the tunnel
                tunneled_config = self.db_config.copy()
                tunneled_config['host'] = '127.0.0.1'
                tunneled_config['port'] = self.tunnel.local_bind_port
                
                self.connection = pymysql.connect(**tunneled_config)
                
                if self.connection.open:
                    self.logger.info("MySQL connection through tunnel successful.")
                    return True

            except BaseSSHTunnelForwarderError as e:
                self.logger.error(f"SSH Tunnel Error: {e}")
                self.disconnect() # Cleanup
            except pymysql.Error as e:
                self.logger.error(f"MySQL Connection Error (via Tunnel): {e}")
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
                query3 = f"SELECT 1 FROM shotDataWaveform_3 WHERE shotCode = {shot} AND shotDataFieldCode = {field} LIMIT 1"
                cursor.execute(query3)
                if cursor.fetchone():
                    return 3

                # If not in table 3, check shotDataWaveform_2
                query2 = f"SELECT 1 FROM shotDataWaveform_2 WHERE shotCode = {shot} AND shotDataFieldCode = {field} LIMIT 1"
                cursor.execute(query2)
                if cursor.fetchone():
                    return 2
            
            return 0 # Not found in either table
        except pymysql.Error as err:
            self.logger.error(f"Query Error: {err}")
            return 0

    def load_shot(self, shot: int, fields: list[int]) -> pd.DataFrame:
        """
        Loads time and data for a given shot and a list of fields from the VEST database.
        It combines the results into a single pandas DataFrame.
        Each column in the DataFrame corresponds to a field, and its index is the time.
        """
        if not self.connect():
            self.logger.error("Failed to connect to database.")
            return pd.DataFrame()
        
        if shot <= 29349:
            self.logger.warning(f"Shot {shot} is too old. Only shots > 29349 are supported in this version.")
            return pd.DataFrame()

        all_series = []
        for field in fields:
            time_raw, data_raw = None, None
            try:
                with self.connection.cursor() as cursor:
                    existence = self.exist_shot(shot, field)

                    if existence == 3:
                        # Data is in the new shotDataWaveform_3 table.
                        query = f"SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_3 WHERE shotCode = {shot} AND shotDataFieldCode = {field}"
                        cursor.execute(query)
                        result = cursor.fetchone()

                        if result:
                            time_str, val_str = result
                            time_raw = np.fromstring(time_str.strip('[]'), sep=',')
                            data_raw = np.fromstring(val_str.strip('[]'), sep=',')

                    elif existence == 2:
                        # Data is in the old shotDataWaveform_2 table.
                        query = f"SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_2 WHERE shotCode = {shot} AND shotDataFieldCode = {field} ORDER BY shotDataWaveformTime ASC"
                        cursor.execute(query)
                        results = cursor.fetchall()

                        if results:
                            time_raw = np.array([row[0] for row in results])
                            data_raw = np.array([row[1] for row in results])
                    
                    if time_raw is not None and data_raw is not None:
                        # Process the loaded data
                        time_corr, data_corr = self.process_vest_data(shot, field, time_raw, data_raw)
                        # Create a Series with time as index
                        series = pd.Series(data_corr, index=time_corr, name=str(field))
                        all_series.append(series)
                        self.logger.info(f"Successfully loaded and processed Shot {shot} Field {field}.")
                    else:
                        self.logger.warning(f"Shot {shot} field {field} not found in database.")

            except pymysql.Error as err:
                self.logger.error(f"Query Error for field {field}: {err}")
                continue # Move to the next field
        
        if not all_series:
            return pd.DataFrame()

        # Concatenate all series into a single dataframe.
        # This will result in a DataFrame with different time indices per column, which is fine.
        # The calling function is responsible for aligning them to a common time axis.
        return pd.concat(all_series, axis=1)


    def process_vest_data(self, shot_num: int, field_id: int, time: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes loaded VEST data to apply sign corrections and recalculate the time axis,
        replicating the logic from the getVESTattr.m and tFastDAQ.m scripts.

        Args:
            shot_num: The shot number.
            field_id: The data field ID.
            time: The original time array from the database.
            data: The original data array from the database.

        Returns:
            A tuple containing the corrected time and data arrays.
        """
        # 1. Apply sign correction based on field_id
        if field_id in [101, 214, 140]:
            data = -data
            self.logger.info(f"Flipping sign for field_id {field_id}.")

        # 2. Determine t0 and tE from tFastDAQ logic
        if shot_num >= 41660: # Includes shots >= 43685
            t0 = 0.26
            tE = 0.36
        else:
            t0 = 0.24
            tE = 0.34
        
        # 3. Recalculate time axis
        # The MATLAB condition `diff(time) < 1/25e3` is a bit ambiguous for a whole array.
        # It likely checks if the sampling rate is high (e.g., > 25 kHz).
        # We can check the mean difference.
        if len(time) > 1 and np.mean(np.diff(time)) < (1 / 25e3):
            # High-speed DAQ: create a new time axis
            self.logger.info("High-speed DAQ detected. Recalculating time axis.")
            new_time = np.linspace(t0, tE, len(time) + 1)
            time = new_time[:-1]
        else:
            # Low-speed DAQ: time values are already in seconds
            # The MATLAB script multiplies by 1e3 for ms, but we keep it in seconds.
            self.logger.info("Low-speed DAQ detected. Using original time values.")
            pass # Time is already correct

        return time, data

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

if __name__ == '__main__':
    # Example usage and test for the VEST_DB class.
    # Note: This requires a valid 'ifi/config.ini' file with database credentials.
    
    # Define a test shot and a list of fields.
    test_shot = 40656
    test_fields = [109, 101] # Ip and a sign-flipped field

    logging.info(f"--- Testing VEST_DB with Shot #{test_shot}, Fields #{test_fields} ---")
    
    try:
        with VEST_DB() as db:
            # 1. Check if the shot data exists for the first field
            logging.info(f"Checking for existence of first field ({test_fields[0]})...")
            existence = db.exist_shot(test_shot, test_fields[0])
            if existence > 0:
                logging.info("  -> Shot seems to exist. Attempting to load multiple fields...")
                
                # 2. Load all fields into a DataFrame
                result_df = db.load_shot(test_shot, test_fields)
                
                if not result_df.empty:
                    logging.info("  -> Data loaded successfully into DataFrame.")
                    logging.info(f"     DataFrame shape: {result_df.shape}")
                    logging.info(f"     Columns: {result_df.columns.tolist()}")
                    logging.info("--- Head of DataFrame ---")
                    print(result_df.head())
                    logging.info("-------------------------")
                else:
                    logging.info("  -> Data loading failed or returned no data.")
            else:
                logging.info(f"  -> Shot {test_shot} with field {test_fields[0]} not found.")

    except FileNotFoundError as e:
        logging.error(f"\nError: {e}")
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}", exc_info=True) 