#!/usr/bin/env python3
"""
File I/O Utilities
==================

Compatibility facade for file I/O utilities.

This module keeps the legacy import path (`ifi.utils.io_utils`) stable while
the implementation is split into focused modules:
    - `io_h5.py`: HDF5 structure naming/sanitization and metadata attrs helpers
    - `io_daq.py`: DAQ/VISA waveform CSV/HDF5 conversion helpers
    - `io_process.py`: post-processing result HDF5 save/load helpers

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from .io_daq import (
    convert_to_hdf5,
    read_csv_chunked,
    read_waveform,
    save_waveform,
)
from .io_h5 import (
    H5_GROUP_CWT,
    H5_GROUP_DENSITY,
    H5_GROUP_RAWDATA,
    H5_GROUP_STFT,
    H5_GROUP_VEST,
    flatten_metadata_attrs,
    make_cache_h5_key,
    make_h5_group_name,
    normalize_source_name,
    unflatten_metadata_attrs,
)
from .io_process import (
    append_signals_to_hdf5,
    append_vest_to_hdf5,
    load_results_from_hdf5,
    save_results_to_hdf5,
)

__all__ = [
    "convert_to_hdf5",
    "read_csv_chunked",
    "read_waveform",
    "save_waveform",
    "H5_GROUP_CWT",
    "H5_GROUP_DENSITY",
    "H5_GROUP_RAWDATA",
    "H5_GROUP_STFT",
    "H5_GROUP_VEST",
    "flatten_metadata_attrs",
    "make_cache_h5_key",
    "make_h5_group_name",
    "normalize_source_name",
    "unflatten_metadata_attrs",
    "append_signals_to_hdf5",
    "append_vest_to_hdf5",
    "load_results_from_hdf5",
    "save_results_to_hdf5",
]
