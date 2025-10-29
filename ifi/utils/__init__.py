#!/usr/bin/env python3
"""
    IFI Utilities
    ============

    This module contains the functions for the IFI project.
    It includes the cache setup functions, common utilities,
    and file I/O functions.
"""

from ifi.utils.cache_setup import setup_project_cache
from ifi.utils.common import assign_kwargs, LogManager, FlatShotList, ensure_dir_exists, ensure_str_path, normalize_to_forward_slash
from ifi.utils.file_io import save_results_to_hdf5

__all__ = [
    'setup_project_cache',
    'assign_kwargs',
    'LogManager',
    'FlatShotList',
    'ensure_dir_exists',
    'ensure_str_path',
    'normalize_to_forward_slash',
    'save_results_to_hdf5'
]