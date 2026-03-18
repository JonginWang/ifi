#!/usr/bin/env python3
"""
NasDB file query mixin
======================

This mixin is for querying/estimating the size of files on the NAS.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import glob
import os
import time
from contextlib import nullcontext
from pathlib import Path

from ..utils.log_manager import log_tag
from .nas_db_base import NasDBBase
from .nas_db_utils import (
    ALLOWED_EXTENSIONS,
    REMOTE_LIST_SCRIPT,
    _extract_filename_from_path,
    _generate_unique_script_name,
    _is_drive_or_unc_path,
    _select_preferred_extension_files,
)


class NasDBMixinQueryFiles(NasDBBase):
    """Search helpers for local/remote NAS files."""

    def find_files(
        self,
        query: int | str | list[int | str],
        data_folders: list[str] | str | None = None,
        add_path: bool = False,
        force_remote: bool = False,
        **kwargs,
    ) -> tuple[list[str], list[str]]:
        _ = (force_remote, kwargs)
        self._ensure_logger(component=__name__)
        if not self._is_connected and not self.connect():
            raise ConnectionError(f"{log_tag('NASDB','QFILE')} Failed to establish connection to NAS.")

        if data_folders is not None:
            if isinstance(data_folders, str):
                data_folders = [data_folders]
            elif isinstance(data_folders, list):
                for idx, folder in enumerate(data_folders):
                    if not isinstance(folder, str):
                        self.logger.warning(
                            f"{log_tag('NASDB','QFILE')} Invalid data folder type: {type(folder)}. "
                            f"Converting to string. {folder}"
                        )
                        data_folders[idx] = str(folder)
            else:
                self.logger.warning(
                    f"{log_tag('NASDB','QFILE')} Invalid data folder type: {type(data_folders)}. {data_folders}"
                )
                data_folders = [str(data_folders)]
        else:
            data_folders = self.default_data_folders

        if add_path:
            data_folders = list(set(self.default_data_folders + data_folders))

        cache_key = (str(query), tuple(sorted(data_folders)))
        if cache_key in self._file_cache:
            self.logger.info(f"{log_tag('NASDB','QFILE')} Found file list for {cache_key} in cache.")
            return self._file_cache[cache_key]

        self.logger.info(
            f"{log_tag('NASDB','QFILE')} Searching for files for query: {query} in folders: {data_folders}."
        )

        base_path = str(self.nas_mount) if self.access_mode == "local" else str(self.nas_path)

        query_items = query if isinstance(query, list) else [query]
        has_paths = any(
            isinstance(item, str) and _is_drive_or_unc_path(item) for item in query_items
        )

        if has_paths and self.access_mode == "local":
            valid_paths = []
            for item in query_items:
                if isinstance(item, str) and _is_drive_or_unc_path(item):
                    if Path(item).exists():
                        valid_paths.append(item)
                    else:
                        self.logger.warning(
                            f"{log_tag('NASDB','QFILE')} Path does not exist: {item}"
                        )
                else:
                    valid_paths.append(item)

            if valid_paths and all(
                isinstance(p, str) and _is_drive_or_unc_path(p) for p in valid_paths
            ):
                selected, dropped = _select_preferred_extension_files(valid_paths)
                if dropped:
                    self.logger.info(
                        f"{log_tag('NASDB','QFILE')} Extension priority applied to direct paths. "
                        f"Dropped {len(dropped)} lower-priority file(s)."
                    )
                return selected

        search_patterns: list[str] = []
        for item in query_items:
            if isinstance(item, str) and _is_drive_or_unc_path(item):
                item = _extract_filename_from_path(item)

            if isinstance(item, int) or (
                isinstance(item, str) and "*" not in item and "." not in item
            ):
                search_patterns.append(f"{item}*.*")
            else:
                search_patterns.append(str(item))

        all_files: set[str] = set()
        if self.access_mode == "local":
            for folder in data_folders:
                for pattern in search_patterns:
                    search_path = Path(base_path) / folder / "**" / pattern
                    all_files.update(glob.glob(str(search_path), recursive=True))
        else:
            all_files.update(self._find_files_remote(data_folders, search_patterns))

        sorted_files = sorted(list(all_files))
        filtered_files = [f for f in sorted_files if Path(f).suffix.lower() in ALLOWED_EXTENSIONS]
        if len(filtered_files) < len(sorted_files):
            self.logger.info(
                f"{log_tag('NASDB','QFILE')} Filtered file list from {len(sorted_files)} to {len(filtered_files)} "
                f"based on allowed extensions: {ALLOWED_EXTENSIONS}"
            )

        normalized_files = [f.replace("\\", "/") for f in filtered_files]
        prioritized_files, dropped_pairs = _select_preferred_extension_files(normalized_files)
        if dropped_pairs:
            self.logger.info(
                f"{log_tag('NASDB','QFILE')} Applied suffix priority (csv > dat > mat > wfm). "
                f"Dropped {len(dropped_pairs)} lower-priority same-stem file(s)."
            )

        if prioritized_files:
            self.logger.info(
                f"{log_tag('NASDB','QFILE')} Found {len(prioritized_files)} files. Caching result."
            )
            self._file_cache[cache_key] = prioritized_files
        else:
            self.logger.warning(
                f"{log_tag('NASDB','QFILE')} No files with allowed extensions found for query '{query}' "
                f"in {data_folders}. Caching empty list."
            )
            self._file_cache[cache_key] = []

        return prioritized_files

    def _find_files_remote(self, data_folders: list[str], patterns: list[str]) -> list[str]:
        """Execute one remote Python script for recursive file discovery."""
        patterns_str = " ".join(patterns)
        search_paths = [
            os.path.join(self.nas_path, folder).replace("\\", "/") for folder in data_folders
        ]
        search_paths_str = ";".join(search_paths)

        script_filename = _generate_unique_script_name(f"list_{int(time.time())}")
        remote_script_path = os.path.join(self.remote_temp_dir, script_filename).replace("\\", "/")

        # In the case of max_concurrent_ssh_commands == 1, we ensure that only one SSH operation at a time.
        mutex_context = self.ssh_operation_lock if self.ssh_operation_lock else nullcontext()
        with mutex_context:
            self.logger.debug(
                f"{log_tag('NASDB','QFILE')} Waiting for SSH operation queue "
                f"(max concurrent: {self.max_concurrent_ssh_commands})"
            )
            with self.ssh_command_semaphore:
                self.logger.debug(
                    f"{log_tag('NASDB','QFILE')} SSH semaphore acquired. "
                    f"Starting remote file search for patterns: {patterns}"
                )
                self._ensure_remote_dir_exists(self.remote_temp_dir, use_semaphore=False)
                try:
                    with self.sftp_client.open(remote_script_path, "w") as f:
                        f.write(REMOTE_LIST_SCRIPT)
                except Exception as e:
                    self.logger.error(f"{log_tag('NASDB','QFILE')} Failed to write remote list script: {e}")
                    return []

                cmd = f'python "{remote_script_path}" "{search_paths_str}" "{patterns_str}"'
                stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
                _ = stdin
                files = stdout.read().decode("utf-8").strip().splitlines()
                err_output = stderr.read().decode("utf-8", errors="ignore").strip()
                if err_output:
                    self.logger.error(f"{log_tag('NASDB','QFILE')} Remote list script error: {err_output}")
                return files

    def _get_files_total_size(self, file_list: list[str]) -> int:
        """Calculate total file size in bytes."""
        if not file_list:
            return 0

        self.logger.info(f"{log_tag('NASDB','QBYTE')} Calculating total size of {len(file_list)} files...")
        total_size = 0
        try:
            if self.access_mode == "local":
                for file_path in file_list:
                    if Path(file_path).exists():
                        total_size += Path(file_path).stat().st_size
            else:
                for file_path in file_list:
                    try:
                        with self.ssh_command_semaphore:
                            total_size += self.sftp_client.stat(file_path).st_size
                    except FileNotFoundError:
                        self.logger.warning(
                            f"{log_tag('NASDB','QBYTE')} Remote file not found during size calculation: {file_path}"
                        )
            return total_size
        except Exception as e:
            self.logger.error(f"{log_tag('NASDB','QBYTE')} Error calculating file sizes: {e}")
            return -1
