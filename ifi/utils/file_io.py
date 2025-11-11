#!/usr/bin/env python3
"""
File I/O
========

This module contains the functions for reading and writing data to files.
It includes functions for reading and writing waveform data to CSV files,
and reading and writing analysis results to HDF5 files.

Functions:
    save_waveform_to_csv: Save waveform data to a CSV file.
    read_waveform_file: Read waveform data from a CSV file.
    read_csv_chunked: Read a large CSV file in chunks and yield each chunk as a DataFrame.
    create_shot_results_directory: Create the results directory for a shot.
    save_results_to_hdf5: Save analysis results to an HDF5 file.
    load_cached_shot_data: Load the cached shot data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple
import h5py
try:
    from .common import ensure_dir_exists
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import ensure_dir_exists


"""
    This section contains the functions for reading and writing data to files when using 
    the oscilloscope as VISA instruments for data logging.
"""


def save_waveform_to_csv(
    filepath: str,
    time_data: np.ndarray,
    voltage_data: np.ndarray,
    channel_name: str = "Voltage (V)",
) -> None:
    """
    Saves waveform data to a CSV file using pandas.

    Args:
        filepath(str): The full path to save the file to.
        time_data(np.ndarray): A NumPy array containing time values.
        voltage_data(np.ndarray): A NumPy array containing voltage values.
        channel_name(str): The name for the voltage data column.

    Returns:
        None

    Raises:
        Exception: If an error occurs while saving the waveform to the CSV file.
    """
    try:
        # Create a directory for the file if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"Time (s)": time_data, channel_name: voltage_data})
        df.to_csv(
            filepath, index=False, float_format="%.6g"
        )  # Use scientific notation for precision
        print(f"Waveform saved to {filepath}")
    except Exception as e:
        print(f"Error saving waveform to {filepath}: {e}")


def read_waveform_file(filepath: str) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Reads a waveform file and returns time and voltage data.
    Supports .csv format.

    Args:
        filepath(str): The full path to the waveform file to read.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the time and voltage data.
        None: If the file is not found or an error occurs.

    Raises:
        Exception: If an error occurs while reading the waveform file.
    """
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return None

    extension = Path(filepath).suffix.lower()

    try:
        if extension == ".csv":
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

    Args:
        filepath(str): Path to the CSV file.
        chunksize(int): Number of rows per chunk.

    Returns:
        Generator[pd.DataFrame, None, None]: A generator that yields pandas DataFrames.

    Raises:
        Exception: If an error occurs while reading the CSV file in chunks.
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


def create_shot_results_directory(shot_num: int, base_dir: str = "./results") -> Path:
    """Create results directory structure for a shot.

    Args:
        shot_num(int): Shot number
        base_dir(str): Base directory for the results

    Returns:
        Path: Path to the results directory
    """
    results_dir = Path(base_dir) / str(shot_num)
    subdirs = ["waveforms", "spectra", "density", "overview", "etc"]

    ensure_dir_exists(str(results_dir))
    for subdir in subdirs:
        ensure_dir_exists(str(results_dir / subdir))

    return results_dir


def save_results_to_hdf5(
    output_dir: str,
    shot_num: int,
    signals: dict,
    stft_results: dict,
    cwt_results: dict,
    density_data: pd.DataFrame,
    vest_data: pd.DataFrame,
) -> str:
    """
    Saves all analysis results to an HDF5 file.

    Args:
        output_dir(str): Directory to save the HDF5 file
        shot_num(int): Shot number (0 for unknown shots)
        signals(dict): Dictionary of signal data (DataFrames)
        stft_results(dict): Dictionary of STFT analysis results
        cwt_results(dict): Dictionary of CWT analysis results
        density_data(pd.DataFrame): DataFrame containing density analysis results
        vest_data(pd.DataFrame): DataFrame containing VEST data

    Returns:
        str | None: The path to the saved HDF5 file or None if an error occurs.

    Raises:
        Exception: If an error occurs while saving the results to the HDF5 file.
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
        with h5py.File(filepath, "w") as hf:
            # Save metadata
            metadata = hf.create_group("metadata")
            metadata.attrs["shot_number"] = shot_num
            metadata.attrs["created_at"] = pd.Timestamp.now().isoformat()
            metadata.attrs["ifi_version"] = "1.0"

            # Save signals data
            # if signals is not None and not signals.empty:
            if signals is not None and signals:
                signals_group = hf.create_group("signals")
                for signal_name, signal_data in signals.items():
                    if isinstance(signal_data, pd.DataFrame):
                        # # Convert DataFrame to structured array for HDF5
                        # signal_data.to_hdf(hf, f'signals/{signal_name}', mode='a', format='table')
                        # Save DataFrame as HDF5 dataset
                        signal_group = signals_group.create_group(signal_name)
                        for col in signal_data.columns:
                            signal_group.create_dataset(
                                col, data=signal_data[col].values
                            )
            else:
                # Create empty signals group if no signals provided
                signals_group = hf.create_group("signals")
                signals_group.attrs["empty"] = True

            # Save STFT results
            if stft_results is not None and stft_results:
                stft_group = hf.create_group("stft_results")
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
                cwt_group = hf.create_group("cwt_results")
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
                density_group = hf.create_group("density_data")
                # density_data.to_hdf(hf, 'density_data/data', mode='a', format='table')
                for col in density_data.columns:
                    density_group.create_dataset(col, data=density_data[col].values)

            # Save VEST data
            if vest_data is not None and not vest_data.empty:
                vest_group = hf.create_group("vest_data")
                # vest_data.to_hdf(hf, 'vest_data/data', mode='a', format='table')
                for col in vest_data.columns:
                    vest_group.create_dataset(col, data=vest_data[col].values)

        print(f"Results saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        print(f"Error saving results to HDF5: {e}")
        return None


def load_results_from_hdf5(shot_num: int, base_dir: str = "results") -> dict:
    """Load results from HDF5 files.

    Args:
        shot_num(int): Shot number
        base_dir(str): Base directory to load the results from

    Returns:
        dict | None: Dictionary of results or None if no results are found

    Examples:
    ```python
    from .file_io import load_results_from_hdf5

    # Load results for shot 46789
    results = load_results_from_hdf5(46789)
    if results:
        print(f"Loaded {len(results)} datasets")
    ```
    """
    results_dir = Path(base_dir) / str(shot_num)
    if not results_dir.exists():
        print(f"No results found for shot {shot_num}")
        return None

    h5_files = list(results_dir.glob("*.h5"))

    if not h5_files:
        print(f"No HDF5 files found in {results_dir}")
        return None

    results = {}

    for h5_file in h5_files:
        print(f"Loading results from {h5_file}")

        try:
            with h5py.File(h5_file, "r") as hf:
                # Load metadata
                if "metadata" in hf:
                    metadata = {}
                    for key, value in hf["metadata"].attrs.items():
                        metadata[key] = value
                    results["metadata"] = metadata
                    print(f"Loaded metadata: {metadata}")

                # Load signals data
                if "signals" in hf:
                    signals = {}
                    signals_group = hf["signals"]

                    # Check if signals group is empty
                    if signals_group.attrs.get("empty", False):
                        print("Signals group is empty")
                    else:
                        for signal_name in signals_group.keys():
                            signal_group = signals_group[signal_name]
                            signal_data = {}
                            for col_name in signal_group.keys():
                                signal_data[col_name] = signal_group[col_name][:]
                            signals[signal_name] = pd.DataFrame(signal_data)
                            print(
                                f"Loaded signal '{signal_name}' with shape {signals[signal_name].shape}"
                            )

                    if signals:
                        results["signals"] = signals

                # Load STFT results
                if "stft_results" in hf:
                    stft_results = {}
                    stft_group = hf["stft_results"]
                    for signal_name in stft_group.keys():
                        signal_stft_group = stft_group[signal_name]
                        stft_data = {}
                        for key in signal_stft_group.keys():
                            stft_data[key] = signal_stft_group[key][:]
                        for key, value in signal_stft_group.attrs.items():
                            stft_data[key] = value
                        stft_results[signal_name] = stft_data
                        print(f"Loaded STFT results for '{signal_name}'")

                    if stft_results:
                        results["stft_results"] = stft_results

                # Load CWT results
                if "cwt_results" in hf:
                    cwt_results = {}
                    cwt_group = hf["cwt_results"]
                    for signal_name in cwt_group.keys():
                        signal_cwt_group = cwt_group[signal_name]
                        cwt_data = {}
                        for key in signal_cwt_group.keys():
                            cwt_data[key] = signal_cwt_group[key][:]
                        for key, value in signal_cwt_group.attrs.items():
                            cwt_data[key] = value
                        cwt_results[signal_name] = cwt_data
                        print(f"Loaded CWT results for '{signal_name}'")

                    if cwt_results:
                        results["cwt_results"] = cwt_results

                # Load density data
                if "density_data" in hf:
                    density_group = hf["density_data"]
                    density_data = {}
                    for col_name in density_group.keys():
                        density_data[col_name] = density_group[col_name][:]
                    results["density_data"] = pd.DataFrame(density_data)
                    print(
                        f"Loaded density data with shape {results['density_data'].shape}"
                    )

                # Load VEST data
                if "vest_data" in hf:
                    vest_group = hf["vest_data"]
                    vest_data = {}
                    for col_name in vest_group.keys():
                        vest_data[col_name] = vest_group[col_name][:]
                    results["vest_data"] = pd.DataFrame(vest_data)
                    print(f"Loaded VEST data with shape {results['vest_data'].shape}")

        except Exception as e:
            print(f"Failed to load results from {h5_file}: {e}")

    return results


def load_cached_shot_data(shot_num: int, cache_base_dir: str = "cache") -> dict:
    """Legacy function - now uses load_results_from_hdf5 with improved functionality.

    This function is deprecated. Use load_results_from_hdf5 instead.

    Args:
        shot_num(int): Shot number
        cache_base_dir(str): Base directory for the cache

    Returns:
        dict | None: Dictionary of cached shot data or None if no cached data is found

    Examples:
    ```python
    from .file_io import load_cached_shot_data

    # Load cached data for shot 46789 (legacy function)
    cached_data = load_cached_shot_data(46789)
    ```
    """
    print("load_cached_shot_data is deprecated. Use load_results_from_hdf5 instead.")
    return load_results_from_hdf5(shot_num, base_dir=cache_base_dir)
