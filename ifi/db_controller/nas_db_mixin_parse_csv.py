#!/usr/bin/env python3
"""
NasDB CSV parser mixin for Tektronix MDO3000pc and MSO58
========================================================

This mixin is for parsing CSV files for Tektronix MDO3000pc and MSO58.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from ..utils.log_manager import log_tag
from ..utils.path_utils import ensure_str_path
from .nas_db_base import NasDBBase


@contextmanager
def _staged_local_csv(file_path: str | Path, prefix: str) -> Iterator[Path]:
    """Stage a mounted/local CSV into a temporary local path for stable parsing."""
    source_path = Path(file_path)
    with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
        staged_path = Path(temp_dir) / source_path.name
        shutil.copyfile(source_path, staged_path)
        yield staged_path


def _has_crcrlf_line_endings(file_path: str | Path, sample_size: int = 1024 * 1024) -> bool:
    """Return whether the file appears to use CRCRLF line endings."""
    try:
        with open(file_path, "rb") as fh:
            sample = fh.read(sample_size)
        return b"\r\r\n" in sample
    except OSError:
        return False


class NasDBMixinParseCsv(NasDBBase):
    """CSV parser implementations with header-shape specific logic."""

    def _parse_mdo3000pc(
        self, file_path: str, header_content: list[str], **kwargs
    ) -> pd.DataFrame | None:
        _ = kwargs
        self.logger.info(f"{log_tag('NASDB','P3KPC')} Parsing as MDO3000pc: {file_path}")
        metadata = {}
        source_line_content: list[str] = []
        source_line_index = -1

        for i, line in enumerate(header_content):
            parts = line.strip().split(",")
            if len(parts) > 1:
                key = parts[0].strip()
                val = parts[1].strip()
                if "Record Length" in key:
                    metadata["record_length"] = int(val)
                elif "Sample Interval" in key:
                    metadata["time_resolution"] = float(val)
            if "Source" in line:
                source_line_content = parts
                source_line_index = i

        if source_line_index == -1:
            self.logger.error(
                f"{log_tag('NASDB','P3KPC')} Could not find 'Source' line in header for MDO3000pc file: {file_path}"
            )
            return None

        data_col_indices = []
        for idx, part in enumerate(source_line_content):
            try:
                float(part.strip())
                data_col_indices.append(idx)
            except ValueError:
                continue

        final_data_indices = [data_col_indices[0]]
        final_data_indices.extend(
            [data_col_indices[i] for i in range(len(data_col_indices)) if i % 2 == 1]
        )
        if not final_data_indices:
            self.logger.error(
                f"{log_tag('NASDB','P3KPC')} Could not extract data column indices for MDO3000pc file: {file_path}"
            )
            return None

        read_target = file_path
        temp_file = None
        staged_context = None
        try:
            if self.access_mode == "remote":
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")  # noqa: SIM115
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                staged_context = _staged_local_csv(read_target, prefix="ifi_mdo3k_")
                read_target = staged_context.__enter__()

            p = ensure_str_path(read_target)
            try:
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=17,
                        header=None,
                        usecols=final_data_indices,
                        low_memory=False,
                        on_bad_lines="warn",
                        encoding_errors="ignore",
                        memory_map=False,
                    )
            except Exception as e_c:
                self.logger.warning(
                    f"{log_tag('NASDB','P3KPC')} C engine parse failed for "
                    f"MDO3000pc: {e_c}. Retrying with python engine."
                )
                p = ensure_str_path(read_target)
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=17,
                        header=None,
                        usecols=final_data_indices,
                        on_bad_lines="warn",
                        encoding_errors="ignore",
                        engine="python",
                    )

            df.columns = ["TIME"] + [f"CH{i}" for i in range(len(df.columns) - 1)]
            df.attrs["metadata"] = metadata
            return df
        except Exception as e:
            self.logger.error(
                f"{log_tag('NASDB','P3KPC')} Pandas failed to parse MDO3000pc file {file_path}. Error: {e}"
            )
            return None
        finally:
            if staged_context:
                staged_context.__exit__(None, None, None)
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _parse_mso58(
        self, file_path: str, header_content: list[str], **kwargs
    ) -> pd.DataFrame | None:
        _ = kwargs
        self.logger.info(f"{log_tag('NASDB','PMSO5')} Parsing as MSO58: {file_path}")

        metadata = {}
        header_row_index = -1
        for i, line in enumerate(header_content):
            try:
                parts = line.strip().split(",")
                if len(parts) > 1:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if "Record Length" in key:
                        metadata["record_length"] = int(val)
                    elif "Sample Interval" in key:
                        metadata["time_resolution"] = float(val)
            except (ValueError, IndexError):
                pass

            if "TIME" in line.upper() and "CH" in line.upper():
                header_row_index = i
                break

        if header_row_index == -1:
            self.logger.error(
                f"{log_tag('NASDB','PMSO5')} Could not dynamically determine the header row for MSO58 file: {file_path}"
            )
            return None

        read_target = file_path
        temp_file = None
        staged_context = None
        try:
            if self.access_mode == "remote":
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")  # noqa: SIM115
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                staged_context = _staged_local_csv(read_target, prefix="ifi_mso5_")
                read_target = staged_context.__enter__()

            p = ensure_str_path(read_target)
            skiprows = header_row_index
            if _has_crcrlf_line_endings(p):
                # Tektronix MSO5 `_ALL` exports may use CRCRLF (`\r\r\n`).
                # pandas treats this differently from Python's splitlines(),
                # so the detected TIME header line is one row earlier for
                # read_csv(skiprows=...). Without this correction, the first
                # data row becomes the header and the actual row count is short
                # by one.
                skiprows = max(0, header_row_index - 1)
            try:
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=skiprows,
                        header=0,
                        encoding_errors="ignore",
                        low_memory=False,
                        on_bad_lines="warn",
                        memory_map=False,
                    )
            except Exception as e_c:
                self.logger.warning(
                    f"{log_tag('NASDB','PMSO5')} C engine parse failed for "
                    f"MSO58: {e_c}. Retrying with python engine. "
                    f"({header_row_index=}, {skiprows=})"
                )
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=skiprows,
                        header=0,
                        encoding_errors="ignore",
                        engine="python",
                        on_bad_lines="warn",
                    )
            df.columns = ["TIME"] + [f"CH{i}" for i in range(len(df.columns) - 1)]

            actual_len = len(df)
            expected_len = metadata.get("record_length")
            if expected_len is not None and expected_len != actual_len:
                self.logger.warning(
                    f"{log_tag('NASDB','PMSO5')} RECORD LENGTH MISMATCH "
                    f"for {Path(file_path).name}! "
                    f"Header: {expected_len}, Actual: {actual_len}"
                )

            df.attrs["metadata"] = metadata
            return df
        except Exception as e:
            self.logger.error(
                f"{log_tag('NASDB','PMSO5')} Pandas failed to parse MSO58 file {file_path}. Error: {e}"
            )
            return None
        finally:
            if staged_context:
                staged_context.__exit__(None, None, None)
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)
