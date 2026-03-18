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
import tempfile
from pathlib import Path

import pandas as pd

from ..utils.log_manager import log_tag
from ..utils.path_utils import ensure_str_path
from .nas_db_base import NasDBBase


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
        try:
            if self.access_mode == "remote":
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")  # noqa: SIM115
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                read_target = Path(read_target)

            try:
                p = ensure_str_path(read_target)
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
        try:
            if self.access_mode == "remote":
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")  # noqa: SIM115
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                read_target = Path(read_target)

            df = pd.read_csv(
                read_target,
                skiprows=header_row_index,
                header=0,
                encoding_errors="ignore",
                engine="python",
                skipfooter=0,
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
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)
