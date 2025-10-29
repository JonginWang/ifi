#!/usr/bin/env python3
"""
Cache Setup Utilities
=====================

This module is used to set up the cache for the IFI package.

Args:
    _cache_initialized(bool): Flag to track if cache has been initialized.
    _cache_config(Dict[str, Any]): Dictionary containing the cache configuration.
    _project_root(Path): Path to the project root.

Functions:
    setup_project_cache: Set up the cache for the IFI package.
    get_cache_config: Get the current cache configuration without setting up.
    is_cache_initialized: Check if cache has been initialized.
    force_disable_jit: Force disable JIT compilation as a last resort for permission issues.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from ifi import get_project_root

# Global flag to track if cache has been initialized
_cache_initialized = False
_cache_config = None

# Get the project root
_project_root = get_project_root()


def setup_project_cache() -> Dict[str, Any]:
    """
    Set up project cache for numba and other components.

    This function uses lazy initialization - it only sets up the cache once
    per Python process, even if called multiple times.

    Args:
        None

    Returns:
        Dict containing cache configuration

    Raises:
        ValueError: If an error occurs while setting up the project cache.
        PermissionError: If the cache directory is not writable.
        OSError: If the cache directory cannot be created.
        Exception: If an error occurs while setting up the project cache.
    """
    global _cache_initialized, _cache_config

    # If already initialized, return existing config
    if _cache_initialized:
        return _cache_config

    # Try multiple cache directory options in order of preference
    cache_options = [
        _project_root / "cache" / "numba_cache",  # Project cache
        Path.home() / ".ifi_cache" / "numba_cache",  # User home cache
        Path(tempfile.gettempdir()) / "ifi_numba_cache",  # System temp cache
    ]

    cache_dir = None
    for option in cache_options:
        try:
            option.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = option / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            cache_dir = option
            print(f"Using cache directory: {cache_dir}")
            break
        except (PermissionError, OSError) as e:
            print(f"Failed to use cache directory {option}: {e}")
            continue

    if cache_dir is None:
        # Last resort: disable JIT compilation
        print(
            "Warning: Could not create any cache directory. Disabling JIT compilation."
        )
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        cache_dir = Path(tempfile.gettempdir()) / "ifi_disabled_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Set up Numba environment variables
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        os.environ["NUMBA_THREADING_LAYER"] = "safe"
        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

        # Additional safety settings for Windows
        if os.name == "nt":  # Windows
            os.environ["NUMBA_DISABLE_JIT"] = (
                "0"  # Keep JIT enabled but with safe settings
            )
            os.environ["NUMBA_DEBUG"] = (
                "0"  # Disable debug mode to reduce file operations
            )

    # Store configuration
    _cache_config = {
        "cache_dir": cache_dir,
        "numba_cache_dir": str(cache_dir),
        "threading_layer": "safe",
        "disable_intel_svml": "1",
        "jit_disabled": os.environ.get("NUMBA_DISABLE_JIT", "0") == "1",
    }

    # Mark as initialized
    _cache_initialized = True

    print("Project cache configured successfully.")
    return _cache_config


def get_cache_config() -> Dict[str, Any]:
    """
    Get the current cache configuration without setting up.

    Returns:
        Dict containing cache configuration, or None if not initialized

    Raises:
        Exception: If an error occurs while getting the cache configuration.
    """
    return _cache_config if _cache_initialized else None


def is_cache_initialized() -> bool:
    """
    Check if cache has been initialized.

    Returns:
        True if cache is initialized, False otherwise

    Raises:
        Exception: If an error occurs while checking if the cache is initialized.
    """
    return _cache_initialized


def force_disable_jit():
    """
    Force disable JIT compilation as a last resort for permission issues.

    Raises:
        Exception: If an error occurs while forcing disable JIT compilation.
    """
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    print("JIT compilation disabled due to permission issues.")
