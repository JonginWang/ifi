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

Author: J. Wang
Date: 2025-01-16
"""

from .func_helper import assign_kwargs
from .if_utils import (
    assign_interferometry_params_to_shot,
    get_default_interferometry_config_path,
    get_interferometry_params_by_section,
)
from .io_process_read import load_cached_shot_data
from .io_utils import (
    append_signals_to_hdf5,
    append_vest_to_hdf5,
    load_results_from_hdf5,
    save_results_to_hdf5,
)
from .log_manager import LogManager, log_tag
from .path_utils import ensure_dir_exists, ensure_str_path, normalize_to_forward_slash
from .vest_utils import FlatShotList

# def setup_project_cache():
#     """Lazy import of setup_project_cache to avoid circular imports."""
#     from .cache_setup import setup_project_cache as _setup_project_cache

#     return _setup_project_cache()


__all__ = [
    "assign_kwargs",
    "assign_interferometry_params_to_shot",
    "get_default_interferometry_config_path",
    "get_interferometry_params_by_section",
    "append_signals_to_hdf5",
    "append_vest_to_hdf5",
    "load_cached_shot_data",
    "load_results_from_hdf5",
    "save_results_to_hdf5",
    "LogManager",
    "log_tag",
    "ensure_dir_exists",
    "ensure_str_path",
    "normalize_to_forward_slash",
    "FlatShotList",
]
