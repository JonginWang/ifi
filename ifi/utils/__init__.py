"""
    IFI Utilities
    ============

    This module contains the functions for the IFI project.
    It includes the cache setup functions, common utilities,
    and file I/O functions.
"""
import sys
import logging
from pathlib import Path

# Add ifi package to Python path for IDE compatibility
current_dir = Path(__file__).resolve()
ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
                + list(current_dir.parents) if p.name == 'ifi']
IFI_ROOT = ifi_parents[-1] if ifi_parents else None

try:
    sys.path.insert(0, str(IFI_ROOT))
except Exception as e:
    print(f"!! Could not find ifi package root: {e}")
    pass

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