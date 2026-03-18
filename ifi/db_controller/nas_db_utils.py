#!/usr/bin/env python3
"""
NasDB Utilities
===============

NasDB shared utilities and helper functions.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from ..utils.io_h5 import normalize_source_name
from ..utils.path_utils import normalize_to_forward_slash

# Allowed source extensions for NAS file discovery.
# ALLOWED_EXTENSIONS = [".csv", ".dat", ".mat", ".wfm", ".isf"]
ALLOWED_EXTENSIONS = [".csv", ".dat", ".wfm", ".isf"]

# Priority for same-stem files in the same folder.
# Lower value means higher priority.
EXTENSION_PRIORITY = {
    ".csv": 0,
    ".dat": 1,
    ".mat": 2,
    # TODO: add .wfm parser support in read dispatch/parser mixins.
    ".wfm": 3,
    ".isf": 4,
}

# Process-global cache file locks shared by all NasDB instances.
_process_cache_locks: dict[str, threading.Lock] = {}
_process_cache_locks_individual = threading.Lock()

# This script finds files and returns a newline-separated list.
REMOTE_LIST_SCRIPT = r"""
import sys
import os
import glob
def find_files(base_path_str, patterns_str):
    base_paths = base_path_str.split(';')
    patterns = patterns_str.split(' ')
    all_file_paths = set()
    for base_path in base_paths:
        for pattern in patterns:
            search_pattern = os.path.join(base_path, '**', pattern)
            file_paths = glob.glob(search_pattern, recursive=True)
            all_file_paths.update(file_paths)

    sorted_paths = sorted(list(all_file_paths))
    for path in sorted_paths:
        print(path)

if __name__ == "__main__":
    base_path_str = sys.argv[1]
    patterns_str = sys.argv[2]
    find_files(base_path_str, patterns_str)
"""

# This script reads top N lines from a remote file.
REMOTE_HEAD_SCRIPT = r"""
import sys
import os

def get_top_lines(file_path, lines_to_read):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i in range(lines_to_read):
                line = f.readline()
                if not line: break
                print(line, end='')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    file_path = sys.argv[1]
    lines_to_read = int(sys.argv[2])
    get_top_lines(file_path, lines_to_read)
"""


def _generate_unique_script_name(prefix: str = "script") -> str:
    """Generate a unique filename for remote scripts."""
    unique_id = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_id}.py"


def _is_drive_or_unc_path(path_str: str) -> bool:
    """Check if a string looks like a drive or UNC path."""
    if not isinstance(path_str, str):
        return False
    if re.match(r"^[A-Z]:[/\\]", path_str, re.IGNORECASE):
        return True
    return bool(path_str.startswith("\\\\") or path_str.startswith("//"))


def _extract_filename_from_path(path_str: str) -> str:
    """Extract basename from drive/UNC path strings."""
    if not isinstance(path_str, str):
        return str(path_str)
    normalized = normalize_to_forward_slash(path_str)
    filename = normalize_source_name(normalized)
    return filename if filename else path_str


def _looks_like_path(value: str) -> bool:
    """Check if a value looks like a file path."""
    if not isinstance(value, str):
        return False
    if _is_drive_or_unc_path(value):
        return True
    normalized = normalize_to_forward_slash(value)
    has_sep = "/" in normalized
    has_ext = Path(normalized).suffix.lower() in ALLOWED_EXTENSIONS
    return has_sep or has_ext


def _select_preferred_extension_files(
    file_paths: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Keep one file per (folder, stem) using extension priority.

    Returns:
        selected_files: filtered file list.
        dropped_pairs: tuples of (kept_file, dropped_file) for logging.
    """
    selected_by_key: dict[tuple[str, str], str] = {}
    dropped_pairs: list[tuple[str, str]] = []

    for path_str in sorted(file_paths):
        normalized = normalize_to_forward_slash(path_str)
        if "/" in normalized:
            parent, name = normalized.rsplit("/", 1)
        else:
            parent, name = "", normalized

        stem = Path(name).stem.lower()
        ext = Path(name).suffix.lower()
        key = (parent.lower(), stem)

        if key not in selected_by_key:
            selected_by_key[key] = path_str
            continue

        current = selected_by_key[key]
        cur_ext = Path(current).suffix.lower()
        cur_pri = EXTENSION_PRIORITY.get(cur_ext, 999)
        new_pri = EXTENSION_PRIORITY.get(ext, 999)

        # Only collapse when extensions differ; keep first when same extension.
        if ext == cur_ext:
            continue

        if new_pri < cur_pri:
            selected_by_key[key] = path_str
            dropped_pairs.append((path_str, current))
        else:
            dropped_pairs.append((current, path_str))

    return sorted(selected_by_key.values()), dropped_pairs
