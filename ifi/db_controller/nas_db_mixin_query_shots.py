#!/usr/bin/env python3
"""
NasDB shot-data query mixin
===========================

This mixin is responsible for loading shot data from the NAS.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pandas as pd

from .. import get_project_root
from ..utils.log_manager import log_tag
from .nas_db_base import NasDBBase
from .nas_db_utils import (
    _extract_filename_from_path,
    _looks_like_path,
    _process_cache_locks,
    _process_cache_locks_individual,
)


class NasDBMixinQueryShots(NasDBBase):
    """High-level shot data loading flow with results/cache reuse."""

    def get_shot_data(
        self,
        query: int | str | list[int | str],
        data_folders: list[str] | str | None = None,
        add_path: bool = False,
        force_remote: bool = False,
        allowed_frequencies: list[float] | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        self._ensure_logger(component=__name__)
        can_use_nas = self._is_connected or self.connect()
        if not can_use_nas:
            if force_remote:
                raise ConnectionError(
                    f"{log_tag('NASDB','QSHOT')} Failed to establish connection to NAS."
                )
            self.logger.warning(
                f"{log_tag('NASDB','QSHOT')} NAS connection unavailable. Falling back to local results-only mode."
            )
            return self._load_local_results_only(query, allowed_frequencies)

        is_query_path = False
        if isinstance(query, str) and _looks_like_path(query):
            is_query_path = True
        if isinstance(query, list) and query and all(_looks_like_path(q) for q in query):
            is_query_path = True

        if is_query_path:
            target_files = self.find_files(
                query if isinstance(query, list) else [query],
                data_folders,
                add_path,
                force_remote,
                **kwargs,
            )
            self.logger.info(
                f"{log_tag('NASDB','QSHOT')} Using provided file path(s) directly: {target_files}"
            )
        else:
            target_files = self.find_files(query, data_folders, add_path, force_remote, **kwargs)
            if not target_files:
                self.logger.warning(f"{log_tag('NASDB','QSHOT')} No files found on NAS for query: {query}")
                return {}

        data_dict: dict[str, pd.DataFrame] = {}
        files_to_fetch: list[str] = []

        if force_remote:
            files_to_fetch = target_files
        else:
            shot_numbers_in_query = set()
            for file_path in target_files:
                basename = _extract_filename_from_path(file_path)
                match = re.match(r"(\d+)", basename)
                if match:
                    shot_numbers_in_query.add(int(match.group(1)))

            project_root = get_project_root()
            results_base_dir = project_root / "ifi" / "results"

            results_signals_loaded = {}
            for shot_num in shot_numbers_in_query:
                results_signals = self._load_signals_from_results(
                    shot_num, results_base_dir, allowed_frequencies=allowed_frequencies
                )
                if results_signals:
                    results_signals_loaded[shot_num] = results_signals
                    self.logger.info(
                        f"{log_tag('NASDB','RSLT')} Found valid signals in results for shot {shot_num}"
                    )

            if results_signals_loaded:
                for shot_num, signals in results_signals_loaded.items():
                    shot_target_files = [
                        f
                        for f in target_files
                        if re.match(r"(\d+)", _extract_filename_from_path(f))
                        and int(re.match(r"(\d+)", _extract_filename_from_path(f)).group(1)) == shot_num
                    ]
                    converted_signals = self._convert_results_signals_to_nas_format(
                        signals, shot_target_files, shot_num
                    )
                    data_dict.update(converted_signals)

            for file_path in target_files:
                basename = _extract_filename_from_path(file_path)
                if basename not in data_dict:
                    files_to_fetch.append(file_path)
                else:
                    self.logger.info(
                        f"{log_tag('NASDB','RSLT')} Skipping '{basename}' - already loaded from results"
                    )

            remaining_files_to_fetch: list[str] = []
            for file_path in files_to_fetch:
                basename = _extract_filename_from_path(file_path)
                match = re.match(r"(\d+)", basename)
                shot_num_for_cache = int(match.group(1)) if match else None
                if shot_num_for_cache is None:
                    self.logger.warning(
                        f"{log_tag('NASDB','QSHOT')} Could not determine shot number "
                        f"for '{basename}'. Will not use cache."
                    )
                    remaining_files_to_fetch.append(file_path)
                    continue

                cache_file = self._get_results_raw_cache_file(shot_num_for_cache, basename)
                cache_file_str = str(cache_file.resolve())
                with _process_cache_locks_individual:
                    file_lock = _process_cache_locks.setdefault(cache_file_str, threading.Lock())
                with file_lock:
                    if not cache_file.exists():
                        remaining_files_to_fetch.append(file_path)
                        continue
                    try:
                        df = self._load_single_raw_cache_h5(cache_file, basename)
                        if df is None:
                            remaining_files_to_fetch.append(file_path)
                            continue
                        data_dict[basename] = df
                    except Exception as e:
                        self.logger.error(
                            f"{log_tag('NASDB','CACHE')} Error reading cache file '{cache_file}': {e}. Will refetch."
                        )
                        remaining_files_to_fetch.append(file_path)
            files_to_fetch = remaining_files_to_fetch

        if not files_to_fetch:
            self.logger.info(f"{log_tag('NASDB','CACHE')} All required files were found in the cache.")
            return data_dict

        if self.max_load_size_gb > 0:
            total_size_bytes = self._get_files_total_size(files_to_fetch)
            total_size_gb = total_size_bytes / (1024**3)
            if total_size_bytes < 0:
                self.logger.warning(
                    f"{log_tag('NASDB','QSHOT')} Could not determine total size "
                    "of files to fetch. Proceeding with caution."
                )
            elif total_size_gb > self.max_load_size_gb:
                raise MemoryError(
                    f"{log_tag('NASDB','QSHOT')} Total size of files to fetch ({total_size_gb:.2f} GB) "
                    f"exceeds the configured limit of {self.max_load_size_gb:.2f} GB."
                )

        self.logger.info(f"{log_tag('NASDB','QSHOT')} Fetching {len(files_to_fetch)} files from NAS...")
        for file_path in files_to_fetch:
            df = self._read_shot_file(file_path, **kwargs)
            if df is None:
                continue

            basename = Path(file_path).name
            data_dict[basename] = df

            match = re.match(r"(\d+)", basename)
            shot_num_for_cache = int(match.group(1)) if match else None
            if shot_num_for_cache is None:
                self.logger.warning(
                    f"{log_tag('NASDB','CACHE')} Could not determine shot number for '{basename}'. Skipping cache."
                )
                continue

            cache_file = self._get_results_raw_cache_file(shot_num_for_cache, basename)
            cache_file_str = str(cache_file.resolve())
            with _process_cache_locks_individual:
                file_lock = _process_cache_locks.setdefault(cache_file_str, threading.Lock())
            with file_lock:
                self._write_single_raw_cache_h5(cache_file, basename, df)

        return dict(sorted(data_dict.items()))

    def _extract_shot_numbers_from_query(
        self, query: int | str | list[int | str]
    ) -> list[int]:
        """Extract candidate shot numbers from query inputs."""
        candidates = query if isinstance(query, list) else [query]

        shot_numbers: set[int] = set()
        for item in candidates:
            if isinstance(item, int):
                shot_numbers.add(item)
                continue

            text = str(item).strip()
            if not text:
                continue

            if _looks_like_path(text):
                text = _extract_filename_from_path(text)

            match = re.match(r"(\d+)", text)
            if match:
                shot_numbers.add(int(match.group(1)))

        return sorted(shot_numbers)

    def _load_local_results_only(
        self,
        query: int | str | list[int | str],
        allowed_frequencies: list[float] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load available data only from canonical local results files."""
        shot_numbers = self._extract_shot_numbers_from_query(query)
        if not shot_numbers:
            return {}

        project_root = get_project_root()
        results_base_dir = project_root / "ifi" / "results"

        local_data: dict[str, pd.DataFrame] = {}
        for shot_num in shot_numbers:
            signals = self._load_signals_from_results(
                shot_num, results_base_dir, allowed_frequencies=allowed_frequencies
            )
            if not signals:
                continue
            if len(shot_numbers) == 1:
                local_data.update(signals)
                continue
            for signal_name, df in signals.items():
                local_data[f"{shot_num}_{signal_name}"] = df

        return dict(sorted(local_data.items()))
