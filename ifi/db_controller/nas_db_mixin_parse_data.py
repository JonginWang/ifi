#!/usr/bin/env python3
"""
NasDB data parser mixin for standard CSV, DAT, and MAT
======================================================

This mixin is for parsing data from standard CSV, DAT, and MAT files.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .. import get_project_root
from ..utils.log_manager import log_tag
from ..utils.path_utils import ensure_str_path
from .nas_db_base import NasDBBase


class NasDBMixinParseData(NasDBBase):
    """Parser implementations for standard CSV, DAT, and MAT."""

    def _parse_standard_csv(
        self, file_path: str, header_content: list[str], csv_type: str, **kwargs
    ) -> pd.DataFrame | None:
        _ = kwargs
        self.logger.info(f"{log_tag('NASDB','P3K')} Parsing as {csv_type}: {file_path}")
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
                if "TIME" in line.upper() and "CH" in line.upper():
                    header_row_index = i
            except (ValueError, IndexError):
                continue

        if header_row_index == -1:
            self.logger.error(
                f"{log_tag('NASDB','P3K')} Could not find a valid header row ('TIME', 'CH...') in {file_path}"
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
                        skiprows=header_row_index,
                        header=0,
                        encoding="utf-8",
                        on_bad_lines="warn",
                        low_memory=False,
                        encoding_errors="ignore",
                        memory_map=False,
                    )
            except Exception as e_c:
                self.logger.warning(
                    f"{log_tag('NASDB','P3K')} C engine parse failed for standard CSV: {e_c}. "
                    "Retrying with python engine."
                )
                p = ensure_str_path(read_target)
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=header_row_index,
                        header=0,
                        encoding="utf-8",
                        on_bad_lines="warn",
                        encoding_errors="ignore",
                        engine="python",
                    )

            df.columns = ["TIME"] + [f"CH{i}" for i in range(len(df.columns) - 1)]
            df.attrs["metadata"] = metadata
            return df
        except Exception as e:
            self.logger.error(
                f"{log_tag('NASDB','P3K')} Pandas failed to parse {file_path} with detected header. Error: {e}"
            )
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _read_fpga_dat(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        _ = kwargs
        self.logger.info(f"{log_tag('NASDB','PFPGA')} Parsing as FPGA .dat: {file_path}")
        read_target = file_path
        temp_file = None
        try:
            if self.access_mode == "remote":
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")  # noqa: SIM115
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                read_target = Path(read_target)

            p = ensure_str_path(read_target)
            with open(p, "rb") as fh:
                df = pd.read_csv(fh, sep=r"\s+", header=None)

            df.columns = ["TIME"] + [f"CH{i}" for i in range(len(df.columns) - 1)]
            df.attrs["source_file_type"] = "dat"
            df.attrs["source_file_format"] = "FPGA"
            return df
        except Exception as e:
            self.logger.error(f"{log_tag('NASDB','PFPGA')} Failed to parse FPGA file {file_path}: {e}")
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _build_results_mat_temp_path(self, remote_file_path: str) -> Path:
        """Build a unique MAT temp file path under results."""
        file_name = Path(remote_file_path).name
        match = re.match(r"(\d+)", file_name)
        project_root = get_project_root()

        if match:
            temp_dir = project_root / "ifi" / "results" / match.group(1) / "_tmp"
        else:
            temp_dir = project_root / "ifi" / "results" / "_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        stem = re.sub(r'[<>:"/\\|?*]+', "_", Path(file_name).stem).strip("._") or "mat"
        return temp_dir / f"{stem}_{uuid.uuid4().hex[:8]}.mat"

    def _read_matlab_mat(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        _ = kwargs
        self.logger.info(f"{log_tag('NASDB','PMAT')} Parsing as MATLAB .mat: {file_path}")

        local_mat_path: str | Path = file_path
        temp_download_path: Path | None = None
        try:
            if self.access_mode == "remote":
                self.logger.warning(
                    f"{log_tag('NASDB','PMAT')} Remote .mat file will be downloaded to results temp path."
                )
                temp_download_path = self._build_results_mat_temp_path(file_path)
                self.sftp_client.get(file_path, str(temp_download_path))
                local_mat_path = temp_download_path

            from scipy.io import loadmat

            mat = loadmat(ensure_str_path(local_mat_path))
            data_key = [
                k for k in mat if isinstance(mat[k], np.ndarray) and not k.startswith("__")
            ][0]
            data = mat[data_key]

            cols = ["TIME"] + [f"CH{i + 1}" for i in range(data.shape[1] - 1)]
            df = pd.DataFrame(data, columns=cols)
            df.attrs["source_file_type"] = "mat"
            df.attrs["source_file_format"] = "MATLAB"
            return df
        except Exception as e:
            self.logger.error(f"{log_tag('NASDB','PMAT')} Failed to parse MATLAB file {file_path}: {e}")
            return None
        finally:
            if temp_download_path and temp_download_path.exists():
                try:
                    temp_download_path.unlink()
                except Exception as e:
                    self.logger.warning(
                        f"{log_tag('NASDB','PMAT')} Failed to remove temp MAT file '{temp_download_path}': {e}"
                    )
