#!/usr/bin/env python3
"""
IFI Utilities
=============

This module contains the functions for the IFI project.
It includes the cache setup functions, common utilities,
and file I/O functions.

Functions:
    setup_project_cache: Function to set up the project cache.
    assign_kwargs: Inject keyword arguments into a method call.
    LogManager: Singleton class to manage logging configuration.
    FlatShotList: Class to parse and flatten a list of shot numbers and file paths.
    ensure_dir_exists: Function to ensure a directory exists.
    ensure_str_path: Function to coerce a path to a string.
    normalize_to_forward_slash: Function to normalize a path to use forward slashes.
    save_results_to_hdf5: Function to save results to an HDF5 file.

Variables:
    __all__: List of public functions and variables.
"""

from ifi.utils.cache_setup import setup_project_cache
from ifi.utils.common import (
    assign_kwargs,
    LogManager,
    FlatShotList,
    ensure_dir_exists,
    ensure_str_path,
    normalize_to_forward_slash,
)
from ifi.utils.file_io import save_results_to_hdf5

__all__ = [
    "setup_project_cache",
    "assign_kwargs",
    "LogManager",
    "FlatShotList",
    "ensure_dir_exists",
    "ensure_str_path",
    "normalize_to_forward_slash",
    "save_results_to_hdf5",
]
