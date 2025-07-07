import mysql.connector
import numpy as np
import configparser
import os

class VEST_DB:
    """
    Handles connection and queries to the VEST database using mysql.connector.
    Reads connection info from a config file.
    """
    def __init__(self, config_path='ifi/config.ini'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please create it from 'config.ini.template'.")
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        db_config = config['VEST_DB']
        self.config = {
            'host': db_config.get('host'),
            'user': db_config.get('user'),
            'password': db_config.get('password'),
            'database': db_config.get('database'),
        }
        self.connection = None

    def connect(self):
        """
        Establishes a connection to the database.
        :return: True if connection is successful, False otherwise.
        """
        try:
            # Prevent reconnection if already connected
            if self.connection and self.connection.is_connected():
                return True
            self.connection = mysql.connector.connect(**self.config)
            return self.connection.is_connected()
        except mysql.connector.Error as err:
            print(f"Database Connection Error: {err}")
            self.connection = None
            return False

    def disconnect(self):
        """
        Closes the database connection.
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
        self.connection = None

    def get_next_shot_code(self) -> int | None:
        """
        Fetches the latest shot number from 'shotDataWaveform_3' and adds 1.
        :return: The next shot number as an integer, or None if failed.
        """
        if not (self.connection and self.connection.is_connected()):
            print("Not connected to the database.")
            return None
        try:
            cursor = self.connection.cursor()
            query = 'SELECT shotCode FROM shotDataWaveform_3 ORDER BY shotCode DESC LIMIT 1'
            cursor.execute(query)
            result_temp = cursor.fetchone()
            cursor.close()
            
            if result_temp:
                last_shotnum = result_temp[0]
                next_shotnum = last_shotnum + 1
                print(f"Next shot number: {next_shotnum}")
                return next_shotnum
            else:
                # Handle case where the table is empty
                print("Table 'shotDataWaveform_3' is empty or shotCode not found.")
                return 1 # Start with 1 if table is empty

        except mysql.connector.Error as err:
            print(f"Query Error: {err}")
            return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect() 