#!/usr/bin/env python3
"""
NasDB dispatch and parser routing mixin
=======================================

This mixin is for dispatching reads based on file extension and routing to the appropriate parser.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.log_manager import log_tag
from .nas_db_base import NasDBBase


class NasDBMixinReadDispatch(NasDBBase):
    """Dispatch by extension and basic CSV parser routing."""

    def _read_shot_file(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        if self.access_mode == "remote":
            with self.ssh_command_semaphore:
                return self._dispatch_read(file_path, **kwargs)
        return self._dispatch_read(file_path, **kwargs)

    def _dispatch_read(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            return self._read_scope_csv(file_path, **kwargs)
        if ext == ".dat":
            return self._read_fpga_dat(file_path, **kwargs)
        if ext == ".mat":
            return self._read_matlab_mat(file_path, **kwargs)
        if ext == ".wfm":
            # TODO: implement .wfm parser and wire it here.
            self.logger.warning(
                f"{log_tag('NASDB','QEXTN')} .wfm parser is not implemented yet: {file_path}"
            )
            return None
        if ext == ".isf":
            # TODO: implement .isf parser and wire it here.
            self.logger.warning(
                f"{log_tag('NASDB','QEXTN')} .isf parser is not implemented yet: {file_path}"
            )
            return None

        self.logger.warning(
            f"{log_tag('NASDB','QEXTN')} Unsupported file extension '{ext}' for file: {file_path}"
        )
        return None

    def _read_scope_csv(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        self.logger.info(f"{log_tag('NASDB','QCSV')} Reading CSV file: {file_path}")
        header_text = (
            self._get_data_top_local(file_path, lines=40)
            if self.access_mode == "local"
            else self._get_data_top_remote(file_path, lines=40, use_semaphore=False)
        )
        if not header_text:
            self.logger.error(f"{log_tag('NASDB','QCSV')} Could not read header of {file_path}")
            return None

        header_content = header_text.splitlines()
        csv_type = self._identify_csv_type(header_content)
        self.logger.info(
            f"{log_tag('NASDB','QCSV')} Identified CSV type for {Path(file_path).name} as: {csv_type}"
        )
        try:
            df = self._parse_csv_with_type(csv_type, file_path, header_content, **kwargs)
            if df is not None:
                df.attrs["source_file_type"] = "csv"
                df.attrs["source_file_format"] = csv_type
                return df
        except Exception as read_error:
            self.logger.error(
                f"{log_tag('NASDB','QCSV')} Failed to parse file '{file_path}'. Error: {read_error}"
            )
        return None

    def _parse_csv_with_type(
        self, csv_type: str, file_path: str, header_content: list[str], **kwargs
    ) -> pd.DataFrame | None:
        if csv_type == "MDO3000pc":
            return self._parse_mdo3000pc(file_path, header_content, **kwargs)
        if csv_type == "MSO58":
            return self._parse_mso58(file_path, header_content, **kwargs)
        if csv_type in ["MDO3000orig", "MDO3000fetch", "ETC"]:
            return self._parse_standard_csv(file_path, header_content, csv_type, **kwargs)
        self.logger.error(f"{log_tag('NASDB','PCSV')} Unknown CSV type '{csv_type}' for {file_path}")
        return None

    def _identify_csv_type(self, header_content: list[str]) -> str:
        try:
            lines = [line.strip().split(",") for line in header_content]
            if not lines:
                return "UNKNOWN"

            if "model" in lines[0][0].lower() and len(lines[0]) > 1:
                model_name = lines[0][1]
                if "MDO3" in model_name:
                    if len(lines) > 1 and "firmware version" in lines[1][0].lower():
                        return "MDO3000orig"
                    return "MDO3000fetch"
                if "MSO5" in model_name:
                    return "MSO58"

            if len(lines) > 16 and len(lines[16]) > 1 and "MDO3" in lines[16][1]:
                return "MDO3000pc"
        except IndexError:
            pass
        return "ETC"
