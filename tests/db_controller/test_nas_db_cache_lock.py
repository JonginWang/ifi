#!/usr/bin/env python3
"""
Test per-file thread lock for NasDB cache (process-global lock).

Verifies that when multiple threads access the same cache file (e.g. for shot 45822)
each with a different file path (same as run_analysis: one task per file), the
process-global per-file lock serializes read/write and prevents OSError (unable to lock file).

- Test uses one thread per file (like run_analysis), not N threads each loading the full shot,
  so there is no redundant NAS fetch.
- Prerequisite: Shot data removed from cache so threads fetch from NAS and write to cache.
- Requires: config, NAS reachable, and at least one file for the test shot.
- Note: Test can be slow (NAS I/O); each thread may take tens of seconds. Thread join
  timeout is 180s; if NAS is slow or shot has many files, run with a generous pytest
  timeout or run locally when needed.

Run with pytest:
  pytest tests/db_controller/test_nas_db_cache_lock.py -v -s
  pytest tests/db_controller/test_nas_db_cache_lock.py -v -s -k "concurrent"
"""

import threading
from pathlib import Path

import pytest

from ifi.utils.log_manager import LogManager
from ifi.db_controller.nas_db import NasDB

LogManager(level="INFO")
logger = LogManager().get_logger(__name__)

CONFIG_PATH = "ifi/config.ini"
SHOT_FOR_LOCK_TEST = 45822
MAX_FILES_FOR_TEST = 4
THREAD_JOIN_TIMEOUT_S = 180


def _fetch_one_file(nas_db: NasDB, file_path: str, results: dict, index: int) -> None:
    """Call get_shot_data(file_path) and store result or exception in results[index]."""
    try:
        data = nas_db.get_shot_data(file_path)
        results[index] = {"data": data, "error": None}
    except Exception as e:
        results[index] = {"data": None, "error": e}


@pytest.fixture
def nas_db():
    """Single NasDB instance (shared across threads; process-global cache locks apply)."""
    if not Path(CONFIG_PATH).exists():
        pytest.skip(f"Configuration file not found at '{CONFIG_PATH}'")
    return NasDB(config_path=CONFIG_PATH)


def test_concurrent_same_shot_cache_writes(nas_db):
    """
    Multiple threads each request one file of the same shot (like run_analysis).
    All touch the same cache file; process-global per-file lock serializes access.
    """
    if not nas_db._is_connected and not nas_db.connect():
        pytest.skip("NAS not reachable; cannot run cache lock test")

    target_files = nas_db.find_files(SHOT_FOR_LOCK_TEST)
    if not target_files:
        pytest.skip(f"No files found for shot {SHOT_FOR_LOCK_TEST} on NAS")

    # Limit number of files to keep test time bounded
    files_to_use = target_files[:MAX_FILES_FOR_TEST]
    n_files = len(files_to_use)

    cache_dir = Path(nas_db.dumping_folder) / str(SHOT_FOR_LOCK_TEST)
    cache_file = cache_dir / f"{SHOT_FOR_LOCK_TEST}.h5"
    if cache_file.exists():
        try:
            cache_file.unlink()
        except OSError:
            pass
    if cache_dir.exists() and not any(cache_dir.iterdir()):
        try:
            cache_dir.rmdir()
        except OSError:
            pass

    results = {}
    threads = []
    for i, file_path in enumerate(files_to_use):
        t = threading.Thread(
            target=_fetch_one_file,
            args=(nas_db, file_path, results, i),
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=THREAD_JOIN_TIMEOUT_S)

    # Check no thread is still alive (join timed out)
    still_alive = [t for t in threads if t.is_alive()]
    assert not still_alive, (
        f"Thread(s) did not finish within {THREAD_JOIN_TIMEOUT_S}s. "
        "Cache lock or NAS may be blocking."
    )

    errors = [results[i]["error"] for i in range(n_files) if results[i]["error"] is not None]
    assert not errors, (
        f"One or more threads raised: {errors}. "
        "Process-global per-file cache lock should prevent OSError when multiple threads "
        "write to the same cache file."
    )
    for i in range(n_files):
        assert results[i]["data"] is not None, f"Thread {i} returned no data"
        assert isinstance(results[i]["data"], dict), f"Thread {i} did not return a dict"
    assert cache_file.exists(), "Cache file should exist after concurrent writes"
