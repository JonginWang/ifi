#!/usr/bin/env python3
"""
NasDB top-lines read mixin
==========================

This mixin is for reading the top lines of a file from the NAS.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext

from ..utils.log_manager import log_tag
from .nas_db_base import NasDBBase
from .nas_db_utils import REMOTE_HEAD_SCRIPT, _generate_unique_script_name


class NasDBMixinReadTop(NasDBBase):
    """Helpers to inspect top lines of local/remote data files."""

    def get_data_top(
        self,
        query: int | str | list[int | str],
        data_folders: list[str] | str | None = None,
        lines: int = 50,
    ) -> str | None:
        self._ensure_logger(component=__name__)
        if not self._is_connected and not self.connect():
            raise ConnectionError("Failed to establish connection to NAS.")

        if isinstance(data_folders, str):
            data_folders = [data_folders]

        file_paths = self.find_files(query, data_folders)
        if not file_paths:
            self.logger.warning(
                f"{log_tag('NASDB','QTOP')} No files found for query {query} to get header from."
            )
            return None

        first_file = file_paths[0]
        if self.access_mode == "local":
            return self._get_data_top_local(first_file, lines)
        return self._get_data_top_remote(first_file, lines)

    def _get_data_top_local(self, file_path: str, lines: int) -> str | None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                head = [f.readline() for _ in range(lines)]
            return "".join(line for line in head if line)
        except Exception as e:
            self.logger.error(f"{log_tag('NASDB','QTOP')} Error reading top lines from {file_path}: {e}")
            return None

    def _get_data_top_remote(
        self,
        file_path: str,
        lines: int,
        use_semaphore: bool = True,
    ) -> str | None:
        script_filename = _generate_unique_script_name(f"head_{int(time.time())}")
        remote_script_path = os.path.join(self.remote_temp_dir, script_filename).replace("\\", "/")

        semaphore_context = self.ssh_command_semaphore if use_semaphore else nullcontext()
        with semaphore_context:
            self._ensure_remote_dir_exists(self.remote_temp_dir, use_semaphore=False)
            try:
                with self.sftp_client.open(remote_script_path, "w") as f:
                    f.write(REMOTE_HEAD_SCRIPT)
            except Exception as e:
                self.logger.error(f"{log_tag('NASDB','QTOPR')} Failed to write remote script: {e}")
                return None

            cmd = f'python "{remote_script_path}" "{file_path}" "{lines}"'
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
            _ = stdin
            output = stdout.read().decode("utf-8", errors="ignore")
            err_output = stderr.read().decode("utf-8", errors="ignore").strip()
            if err_output:
                self.logger.error(f"{log_tag('NASDB','QTOPR')} Remote script error: {err_output}")

            try:
                self.sftp_client.remove(remote_script_path)
            except Exception as e:
                self.logger.warning(
                    f"{log_tag('NASDB','QTOPR')} Failed to remove remote script {remote_script_path}: {e}"
                )
            return output if output else None
