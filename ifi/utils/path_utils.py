#!/usr/bin/env python3
"""
Path Utilities
==============

This module contains the functions for path utilities.
It includes the functions for ensuring a directory exists,
coercing a path to a string,
normalizing a path to use forward slashes,
and getting the absolute path to a resource.

Functions:
    ensure_dir_exists: Ensure a directory exists, creating it if necessary.
    ensure_str_path: Coerce a path to a string.
    normalize_to_forward_slash: Normalize a path to use forward slashes.
    resource_path: Get the absolute path to a resource.

Author: J. Wang
Date: 2025-01-16
"""

import sys
from pathlib import Path


def ensure_dir_exists(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_str_path(path_like: str | Path) -> str:
    """Coerce to string and normalize separators to forward slashes."""
    return normalize_to_forward_slash(path_like)


def normalize_to_forward_slash(path_like: str | Path) -> str:
    """Return a string path using forward slashes. Safe for Windows and pandas."""
    if isinstance(path_like, Path):
        return path_like.as_posix()
    return str(path_like).replace("\\", "/")


def resource_path(relative_path: str) -> Path:
    """
    Get absolute path to resource, works for dev and for PyInstaller.

    Args:
        relative_path(str): Relative path to resource

    Returns:
        Path object to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent.parent

    return base_path / relative_path
