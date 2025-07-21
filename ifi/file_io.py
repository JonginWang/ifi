import os
import numpy as np
import pandas as pd

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

# --- Example of how to use the chunked reader in main_window.py ---
# This would replace the simple call to read_waveform_file
#
# def process_large_file(filepath):
#     # This function would be called in the worker thread
#     total_rows = 0
#     # Assume column names are 'Time (s)', 'CH1', 'CH2', 'CH3', 'CH4'
#     means = {'CH1': 0, 'CH2': 0, 'CH3': 0, 'CH4': 0}
#     
#     chunk_gen = read_csv_chunked(filepath)
#     for chunk_df in chunk_gen:
#         # Process each chunk here
#         # For example, calculate running average
#         total_rows += len(chunk_df)
#         for col in means.keys():
#             means[col] += chunk_df[col].sum()
#
#     # Final calculations
#     for col in means.keys():
#         means[col] /= total_rows
#
#     # Send result to GUI
#     # self.gui_queue.put(('analysis_result', {'means': means})) 