#!/usr/bin/env python3
"""
Cache Setup Utilities
======================

Lightweight cache setup utilities.

This module keeps runtime side effects minimal. Legacy environment workarounds
that monkey-patch imports or aggressively tune Numba internals were moved to:
    ifi/olds/cache_setup_legacy.py

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

_cache_initialized = False
_cache_config: dict[str, Any] | None = None
_project_root: Path | None = None


def _get_project_root() -> Path | None:
    """Lazy import of get_project_root to avoid circular imports."""
    try:
        from .. import get_project_root
        return get_project_root()
    except ImportError as e:
        print(f"Failed to import get_project_root from parent directory: {e}")
        try:
            from ifi import get_project_root
            return get_project_root()
        except ImportError as e:
            print(f"Failed to import get_project_root from ifi package: {e}")
            return None


def _resolve_cache_dir() -> Path:
    """Pick the first writable cache directory from a short fallback list."""
    global _project_root
    if _project_root is None:
        _project_root = _get_project_root() or Path(__file__).parent.parent.parent

    candidates = [
        _project_root / "cache" / "numba_cache",
        Path.home() / ".ifi_cache" / "numba_cache",
        Path(tempfile.gettempdir()) / "ifi_numba_cache",
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            return candidate
        except (PermissionError, OSError):
            continue

    fallback = Path(tempfile.gettempdir()) / "ifi_numba_cache_fallback"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def setup_project_cache() -> dict[str, Any]:
    """
    Configure only essential cache-related env vars.

    Returns:
        A dictionary describing the effective cache configuration.
    """
    global _cache_initialized, _cache_config
    if _cache_initialized and _cache_config is not None:
        return _cache_config

    cache_dir = _resolve_cache_dir()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))

    _cache_config = {
        "cache_dir": cache_dir,
        "numba_cache_dir": str(cache_dir),
        "threading_layer": os.environ.get("NUMBA_THREADING_LAYER"),
        "jit_disabled": os.environ.get("NUMBA_DISABLE_JIT", "0") == "1",
        "legacy_workarounds_enabled": False,
    }
    _cache_initialized = True
    return _cache_config


def get_cache_config() -> dict[str, Any] | None:
    """Return the cache configuration if initialized."""
    return _cache_config if _cache_initialized else None


def is_cache_initialized() -> bool:
    """Check if cache setup has already run."""
    return _cache_initialized


def force_disable_jit() -> None:
    """Disable Numba JIT for this process."""
    os.environ["NUMBA_DISABLE_JIT"] = "1"


def enable_torch():
    """
    Import and return real torch.

    Legacy dummy-module replacement logic was moved to
    ifi/olds/cache_setup_legacy.py.
    """
    return importlib.import_module("torch")

