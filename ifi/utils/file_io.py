import os
import numpy as np
import pandas as pd
import argparse
import logging
from typing import Tuple, Dict

from .common import ensure_dir_exists


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




def save_results_to_hdf5(output_dir, shot_num, signals, stft_results, cwt_results, density_data, vest_data):
    """Saves all analysis results to an HDF5 file."""
    if shot_num == 0 and signals is not None and not signals.empty:
        # For 'unknown' shots, create a filename from the first source file
        first_source_file = list(signals.keys())[0]
        filename = f"{os.path.splitext(first_source_file)[0]}.h5"
    else:
        filename = f"{shot_num}.h5"
    
    filepath = os.path.join(output_dir, filename)
    ensure_dir_exists(output_dir)

    # ... (rest of the HDF5 saving logic) 