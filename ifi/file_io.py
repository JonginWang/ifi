import os
import numpy as np
import pandas as pd
import argparse
import logging
from typing import Tuple, Dict

from . import utils


"""
    This section contains the functions for reading and writing data to files when using 
    the oscilloscope as VISA instruments for data logging.
"""


def save_waveform_to_csv(filepath: str, time_data: np.ndarray, voltage_data: np.ndarray, channel_name: str = "Voltage (V)"):
    """
    Saves waveform data to a CSV file using pandas.

    Args:
        filepath: The full path to save the file to.
        time_data: A NumPy array containing time values.
        voltage_data: A NumPy array containing voltage values.
        channel_name: The name for the voltage data column.
    """
    try:
        # Create a directory for the file if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df = pd.DataFrame({
            'Time (s)': time_data,
            channel_name: voltage_data
        })
        df.to_csv(filepath, index=False, float_format='%.6g') # Use scientific notation for precision
        print(f"Waveform saved to {filepath}")
    except Exception as e:
        print(f"Error saving waveform to {filepath}: {e}")

def read_waveform_file(filepath: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Reads a waveform file and returns time and voltage data.
    Supports .csv format.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    _, extension = os.path.splitext(filepath)
    extension = extension.lower()

    try:
        if extension == '.csv':
            # Use pandas to efficiently read CSV data
            # Assumes a header row and two columns: Time, Voltage
            df = pd.read_csv(filepath)
            time = df.iloc[:, 0].to_numpy()
            voltage = df.iloc[:, 1].to_numpy()
            return time, voltage
            
        else:
            print(f"Unsupported file format: {extension}. Only .csv is supported.")
            return None

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None 

def read_csv_chunked(filepath: str, chunksize: int = 1_000_000):
    """
    Reads a large CSV file in chunks and yields each chunk as a DataFrame.
    This is memory-efficient for very large files.

    :param filepath: Path to the CSV file.
    :param chunksize: Number of rows per chunk.
    :return: A generator that yields pandas DataFrames.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        # Create a generator object
        chunk_generator = pd.read_csv(filepath, chunksize=chunksize, header=0)
        yield from chunk_generator
            
    except Exception as e:
        print(f"Error reading file {filepath} in chunks: {e}")
        return


"""
    This section contains the functions for reading and writing analysis results to and from files.
"""


def get_interferometry_params(shot_num: int, filename: str) -> Dict:
    """
    Returns interferometry analysis parameters based on shot number and filename.
    
    Args:
        shot_num: Shot number
        filename: Name of the data file (e.g., "45821_056.csv", "45821_ALL.csv")
    
    Returns:
        Dictionary containing:
        - method: Analysis method ('CDM', 'FPGA', 'IQ')
        - freq: Interferometer frequency in GHz (from if_config.ini)
        - ref_col: Reference channel column name
        - probe_cols: List of probe channel column names
        - amp_ref_col: Amplitude reference column (for FPGA method)
        - amp_probe_cols: List of amplitude probe columns (for FPGA method)
    """
    import os
    import configparser
    
    basename = os.path.basename(filename)
    
    # Load interferometry configuration
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'analysis', 'if_config.ini')
    config.read(config_path)
    
    # Extract frequency values from config (convert from Hz to GHz)
    freq_94ghz = float(config.get('94GHz', 'freq')) / 1e9  # Convert Hz to GHz
    freq_280ghz = float(config.get('280GHz', 'freq')) / 1e9  # Convert Hz to GHz
    n_path_94ghz = int(config.get('94GHz', 'n_path'))
    n_path_280ghz = int(config.get('280GHz', 'n_path'))

    # Rule 3: Shots 41542 and above
    if shot_num >= 41542:
        if "_ALL" in basename:
            # <shot_num>_ALL.csv: 280GHz / CDM method
            # "CH0" in file containing "_0" or "_ALL" is "ref. signal"
            return {
                'method': 'CDM',
                'freq': freq_280ghz,
                'ref_col': 'CH0',
                'probe_cols': ['CH1'],
                'n_path': n_path_280ghz
            }
        elif "_0" in basename:
            # <shot_num>_0XX.csv: 94GHz / CDM method
            # "CH0" in file containing "_0" or "_ALL" is "ref. signal"
            return {
                'method': 'CDM',
                'freq': freq_94ghz,
                'ref_col': 'CH1',
                'probe_cols': ['CH2', 'CH3'],  # Mapping for channels 5, 6
                'n_path': n_path_94ghz
            }
        else:
            # <shot_num>_XXX.csv: 94GHz / CDM method
            # Other probe channels
            return {
                'method': 'CDM',
                'freq': freq_94ghz,
                'ref_col': '',
                'probe_cols': ['CH0', 'CH1', 'CH2'],  # Mapping for channels 7-9
                'n_path': n_path_94ghz
            }

    # Rule 2: Shots 39302–41398
    elif 39302 <= shot_num <= 41398:
        # 94GHz / FPGA method
        # The first 8 channels: phase ref(CH0) and probes (CH1-CH7) [rad]
        # The second 8 channels: nothing
        # The last 8 channels: amplitude of ref(CH16) and probes(CH17-CH23) [V]
        return {
            'method': 'FPGA',
            'freq': freq_94ghz,
            'ref_col': 'CH0', 
            'probe_cols': ['CH1', 'CH2', 'CH3', 'CH4', 'CH5'],  # ch. 5-9 of 94G interferometer
            'amp_ref_col': 'CH16',
            'amp_probe_cols': ['CH17', 'CH18', 'CH19', 'CH20', 'CH21'],  # ch. 5-9 of 94G interferometer
            'n_path': n_path_94ghz
        }

    # Rule 1: Shots 0–39265
    elif 0 <= shot_num <= 39265:
        # 94GHz / IQ method
        # Assuming the two columns for I and Q are named 'CH0' and 'CH1' for simplicity.
        return {
            'method': 'IQ',
            'freq': freq_94ghz,
            'ref_col': None,  # IQ method does not use a separate reference signal from another channel
            'probe_cols': [('CH0', 'CH1')],
            'n_path': n_path_94ghz
        }
    
    # Default case for shots outside defined ranges
    else:
        return {
            'method': 'unknown',
            'freq': freq_94ghz,  # Default to 94GHz if unknown
            'ref_col': None,
            'probe_cols': [],
            'n_path': n_path_94ghz
        }

def save_results_to_hdf5(output_dir, shot_num, signals, stft_results, cwt_results, density_data, vest_data):
    """Saves all analysis results to an HDF5 file."""
    if shot_num == 0 and signals is not None and not signals.empty:
        # For 'unknown' shots, create a filename from the first source file
        first_source_file = list(signals.keys())[0]
        filename = f"{os.path.splitext(first_source_file)[0]}.h5"
    else:
        filename = f"{shot_num}.h5"
    
    filepath = os.path.join(output_dir, filename)
    utils.ensure_dir(output_dir)

    # ... (rest of the HDF5 saving logic) 