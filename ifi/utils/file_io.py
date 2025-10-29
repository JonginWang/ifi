#!/usr/bin/env python3
"""
    File I/O
    ========

    This module contains the functions for reading and writing data to files.
    It includes functions for reading and writing waveform data to CSV files,
    and reading and writing analysis results to HDF5 files.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from ifi.utils.common import ensure_dir_exists


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
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
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
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return None

    extension = Path(filepath).suffix.lower()

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
    if not Path(filepath).exists():
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
    """
    Saves all analysis results to an HDF5 file.
    
    Args:
        output_dir: Directory to save the HDF5 file
        shot_num: Shot number (0 for unknown shots)
        signals: Dictionary of signal data (DataFrames)
        stft_results: Dictionary of STFT analysis results
        cwt_results: Dictionary of CWT analysis results  
        density_data: DataFrame containing density analysis results
        vest_data: DataFrame containing VEST data
    """
    import h5py
    
    # if shot_num == 0 and signals is not None and not signals.empty:
    if shot_num == 0 and signals is not None and signals:
        # For 'unknown' shots, create a filename from the first source file
        first_source_file = list(signals.keys())[0]
        filename = f"{Path(first_source_file).stem}.h5"
    else:
        filename = f"{shot_num}.h5"
    
    filepath = Path(output_dir) / filename
    ensure_dir_exists(str(output_dir))

    try:
        with h5py.File(filepath, 'w') as hf:
            # Save metadata
            metadata = hf.create_group('metadata')
            metadata.attrs['shot_number'] = shot_num
            metadata.attrs['created_at'] = pd.Timestamp.now().isoformat()
            metadata.attrs['ifi_version'] = '1.0'
            
            # Save signals data
            # if signals is not None and not signals.empty:
            if signals is not None and signals:
                signals_group = hf.create_group('signals')
                for signal_name, signal_data in signals.items():
                    if isinstance(signal_data, pd.DataFrame):
                        # # Convert DataFrame to structured array for HDF5
                        # signal_data.to_hdf(hf, f'signals/{signal_name}', mode='a', format='table')
                        # Save DataFrame as HDF5 dataset
                        signal_group = signals_group.create_group(signal_name)
                        for col in signal_data.columns:
                            signal_group.create_dataset(col, data=signal_data[col].values)
            else:
                # Create empty signals group if no signals provided
                signals_group = hf.create_group('signals')
                signals_group.attrs['empty'] = True
            
            # Save STFT results
            if stft_results is not None and stft_results:
                stft_group = hf.create_group('stft_results')
                for signal_name, stft_data in stft_results.items():
                    if isinstance(stft_data, dict):
                        signal_stft_group = stft_group.create_group(signal_name)
                        for key, value in stft_data.items():
                            if isinstance(value, np.ndarray):
                                signal_stft_group.create_dataset(key, data=value)
                            elif isinstance(value, (int, float, str)):
                                signal_stft_group.attrs[key] = value
            
            # Save CWT results
            if cwt_results is not None and cwt_results:
                cwt_group = hf.create_group('cwt_results')
                for signal_name, cwt_data in cwt_results.items():
                    if isinstance(cwt_data, dict):
                        signal_cwt_group = cwt_group.create_group(signal_name)
                        for key, value in cwt_data.items():
                            if isinstance(value, np.ndarray):
                                signal_cwt_group.create_dataset(key, data=value)
                            elif isinstance(value, (int, float, str)):
                                signal_cwt_group.attrs[key] = value
            
            # Save density data
            if density_data is not None and not density_data.empty:
                density_group = hf.create_group('density_data')
                # density_data.to_hdf(hf, 'density_data/data', mode='a', format='table')
                for col in density_data.columns:
                    density_group.create_dataset(col, data=density_data[col].values)

            # Save VEST data
            if vest_data is not None and not vest_data.empty:
                vest_group = hf.create_group('vest_data')
                # vest_data.to_hdf(hf, 'vest_data/data', mode='a', format='table')
                for col in vest_data.columns:
                    vest_group.create_dataset(col, data=vest_data[col].values)
        
        print(f"Results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"Error saving results to HDF5: {e}")
        return None 