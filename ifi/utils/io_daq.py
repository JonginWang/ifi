#!/usr/bin/env python3
"""
DAQ/VISA waveform CSV/HDF5 conversion helpers
=============================================

This module contains the functions for DAQ/VISA waveform CSV/HDF5 conversion helpers.
It includes the functions for saving waveform data to a CSV file and reading waveform data from a CSV file.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# Helper functions #TODO: Will be modified and fixed later with VISA connection
def _save_waveform_to_csv(
    filepath: Path | str,
    time_data: np.ndarray,
    voltage_data: np.ndarray,
    channel_name: str = "Voltage [V]",
) -> None:
    df = pd.DataFrame({"Time [s]": time_data, "Voltage [V]": voltage_data})
    df.to_csv(filepath, index=False, float_format="%.6g")

def _save_waveform_to_hdf5(
    filepath: Path | str,
    time_data: np.ndarray,
    voltage_data: np.ndarray,
    channel_name: str = "Voltage [V]",
) -> None:
    with h5py.File(filepath, "w") as h5f:
        h5f.create_dataset("Time [s]", data=time_data)
        h5f.create_dataset("Voltage [V]", data=voltage_data)

def _save_waveform_to_wfm(
    filepath: Path | str,
    time_data: np.ndarray,
    voltage_data: np.ndarray,
    channel_name: str = "Voltage [V]",
) -> None:
    pass    


def _read_waveform_from_csv(filepath: Path | str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
    time = df["Time [s]"].to_numpy()
    voltage = df.iloc[:, df.columns[1:]].to_numpy()
    return time, voltage

def _read_waveform_from_hdf5(filepath: Path | str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(filepath, "r") as h5f:
        time = h5f["Time [s]"][:]
        voltage = h5f["Voltage [V]"][:]
        return time, voltage

def _read_waveform_from_wfm(filepath: Path | str) -> tuple[np.ndarray, np.ndarray]:
    pass    


def save_waveform(
    filepath: Path | str,
    time_data: np.ndarray,
    voltage_data: np.ndarray,
    channel_name: str = "Voltage [V]",
    extension: str = "csv",
) -> None:
    """
    Saves waveform data to a CSV file using pandas.

    Args:
        filepath(str): The full path to save the file to.
        time_data(np.ndarray): A NumPy array containing time values.
        voltage_data(np.ndarray): A NumPy array containing voltage values.
        channel_name(str): The name for the voltage data column.
        extension(str): The extension of the file to save: "csv", "h5", "wfm".
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        if extension == "csv":
            _save_waveform_to_csv(filepath, time_data, voltage_data)
        elif extension == "h5":
            _save_waveform_to_hdf5(filepath, time_data, voltage_data)
        elif extension == "wfm":
            _save_waveform_to_wfm(filepath, time_data, voltage_data)
        print(f"Waveform saved to {filepath}")
    except Exception as e:
        print(f"Error saving waveform to {filepath}: {e}")


def read_waveform(filepath: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Reads a waveform file and returns time and voltage data.
    Supports .csv format.
    """
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return None

    extension = Path(filepath).suffix.lower()

    try:
        if extension == ".csv":
            return _read_waveform_from_csv(filepath)
        elif extension == ".h5":
            return _read_waveform_from_hdf5(filepath)
        elif extension == ".wfm":
            return _read_waveform_from_wfm(filepath)
        else:
            raise ValueError(f"Unsupported file format: {extension}. Only .csv, .h5, and .wfm are supported.")
    except Exception as e:
        print(f"Error reading waveform file {filepath}: {e}")
        return None


def convert_to_hdf5(source_path: Path | str, destination_path: Path | str) -> None:
    """Convert a CSV waveform file to a simple HDF5 representation."""
    src = Path(source_path)
    dst = Path(destination_path)

    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    if src.suffix.lower() == ".csv":
        _convert_csv_to_hdf5(src, dst)
        return

    raise ValueError(f"Unsupported source format for HDF5 conversion: {src.suffix}")


def _convert_csv_to_hdf5(source_csv: Path, destination_h5: Path) -> None:
    """Internal helper to convert a CSV file to HDF5."""
    df = pd.read_csv(source_csv)

    with h5py.File(destination_h5, "w") as h5f:
        group = h5f.create_group("waveform")
        for col in df.columns:
            data = df[col].to_numpy(dtype=np.float64)
            group.create_dataset(col, data=data)


def read_csv_chunked(filepath: str, chunksize: int = 1_000_000):
    """
    Reads a large CSV file in chunks and yields each chunk as a DataFrame.
    """
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        return

    try:
        chunk_generator = pd.read_csv(filepath, chunksize=chunksize, header=0)
        yield from chunk_generator
    except Exception as e:
        print(f"Error reading file {filepath} in chunks: {e}")
        return


__all__ = [
    "save_waveform",
    "read_waveform",
    "convert_to_hdf5",
    "read_csv_chunked",
]

