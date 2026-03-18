#!/usr/bin/env python3
"""
Post-processing HDF5 files I/O Utilities
========================================

Compatibility facade for post-processing HDF5 read/write helpers.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from .io_h5_append import (
    append_signals_to_hdf5,
    append_vest_to_hdf5,
)
from .io_process_read import load_results_from_hdf5
from .io_process_write import (
    save_results_to_hdf5,
)

__all__ = [
    "append_signals_to_hdf5",
    "append_vest_to_hdf5",
    "save_results_to_hdf5",
    "load_results_from_hdf5",
]
