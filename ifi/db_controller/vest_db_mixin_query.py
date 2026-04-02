#!/usr/bin/env python3
"""
VestDB Mixins for query
=======================

Query mixin for `VestDB`.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..utils.log_manager import log_tag
from .vest_db_base import VestDBBase


class VestDBMixinQuery(VestDBBase):
    """Shot- and waveform-level queries and transformations."""

    def get_next_shot_code(self) -> int | None:
        """Return next shot code from shotDataWaveform_3."""
        self._ensure_logger(component=__name__)
        if not (self.connection and self.connection.open) and not self.connect():
            self.logger.error(
                f"{log_tag('VESTD', 'QLAST')} Failed to connect to the database."
            )
            return None

        query_next_shot = "SELECT shotCode FROM shotDataWaveform_3 ORDER BY shotCode DESC LIMIT 1"
        rows = self.query(query_next_shot)
        if rows is None:
            return None
        if rows:
            next_shotnum = rows[0][0] + 1
            self.logger.info(
                f"{log_tag('VESTD', 'QLAST')} Next shot number: {next_shotnum}"
            )
            return next_shotnum

        self.logger.warning(
            f"{log_tag('VESTD', 'QLAST')} Table 'shotDataWaveform_3' is empty or shotCode not found."
        )
        return 1

    def exist_shot(self, shot: int, field: int) -> int:
        """Check whether shot/field exists (3: table3, 2: table2, 0: none)."""
        self._ensure_logger(component=__name__)
        if not self.connect():
            self.logger.error(
                f"{log_tag('VESTD', 'QEXST')} Failed to connect to database."
            )
            return 0

        query3 = (
            "SELECT 1 FROM shotDataWaveform_3 "
            "WHERE shotCode = %s AND shotDataFieldCode = %s LIMIT 1"
        )
        rows3 = self.query(query3, (shot, field))
        if rows3 is None:
            return 0
        if rows3:
            return 3

        query2 = (
            "SELECT 1 FROM shotDataWaveform_2 "
            "WHERE shotCode = %s AND shotDataFieldCode = %s LIMIT 1"
        )
        rows2 = self.query(query2, (shot, field))
        if rows2 is None:
            return 0
        if rows2:
            return 2
        return 0

    def _classify_sample_rate(self, fs: float) -> str:
        """Classify numeric sample rate into display key."""
        if 20_000 <= fs < 40_000:
            return "25k"
        if 200_000 <= fs < 400_000:
            return "250k"
        if fs >= 1_500_000:
            return "2M"
        if fs >= 1000:
            return f"{round(fs / 1000)}k"
        return f"{round(fs)}Hz"

    def load_shot(self, shot: int, fields: list[int]) -> dict[str, pd.DataFrame]:
        """Load shot waveform data and return grouped DataFrames by rate key."""
        self._ensure_logger(component=__name__)
        if not self.connect():
            self.logger.error(
                f"{log_tag('VESTD', 'LOAD')} Failed to connect to database."
            )
            return {}

        if shot <= 29349:
            self.logger.warning(
                f"{log_tag('VESTD', 'LOAD')} Shot {shot} is too old. "
                "Only shots > 29349 in MySQL are supported in this version."
            )
            return {}

        grouped_series = defaultdict(list)
        for field in fields:
            time_raw, data_raw = None, None
            table_num = self.exist_shot(shot, field)

            if table_num == 3:
                query_load_3 = (
                    "SELECT shotDataWaveformTime, shotDataWaveformValue "
                    "FROM shotDataWaveform_3 WHERE shotCode = %s AND shotDataFieldCode = %s"
                )
                rows = self.query(query_load_3, (shot, field))
                if rows:
                    time_str, val_str = rows[0]
                    time_raw = np.fromstring(time_str.strip("[]"), sep=",")
                    data_raw = np.fromstring(val_str.strip("[]"), sep=",")

            elif table_num == 2:
                query_load_2 = (
                    "SELECT shotDataWaveformTime, shotDataWaveformValue "
                    "FROM shotDataWaveform_2 WHERE shotCode = %s AND shotDataFieldCode = %s "
                    "ORDER BY shotDataWaveformTime ASC"
                )
                rows = self.query(query_load_2, (shot, field))
                if rows:
                    time_raw = np.array([row[0] for row in rows])
                    data_raw = np.array([row[1] for row in rows])

            if time_raw is not None and data_raw is not None:
                time_corr, data_corr, sample_rate = self.process_vest_data(
                    shot, field, time_raw, data_raw
                )
                rate_key = self._classify_sample_rate(sample_rate)
                series_name = self.field_labels.get(field, str(field))
                series = pd.Series(data_corr, index=time_corr, name=series_name)
                grouped_series[rate_key].append(series)
                self.logger.info(
                    f"{log_tag('VESTD', 'LOAD')} Successfully loaded and processed Shot {shot} "
                    f"Field {field} as '{series_name}' (Rate Group: {rate_key})."
                )

            else:
                self.logger.warning(
                    f"{log_tag('VESTD', 'LOAD')} Shot {shot} field {field} not found in database."
                )

        if not grouped_series:
            return {}

        final_dfs = {}
        for rate_key, series_list in grouped_series.items():
            final_dfs[rate_key] = pd.concat(series_list, axis=1)
            self.logger.info(
                f"{log_tag('VESTD', 'LOAD')} Created DataFrame for "
                f"'{rate_key}' group with {len(series_list)} signal(s)."
            )
        return final_dfs

    def process_vest_data(
        self, shot_num: int, field_id: int, time: np.ndarray, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Apply field sign correction and DAQ time-axis correction."""
        self._ensure_logger(component=__name__)
        if field_id in [101, 214, 140]:
            data = -data
            self.logger.info(
                f"{log_tag('VESTD', 'PROC')} Flipping sign for field_id {field_id}."
            )

        sample_rate = 0.0
        t_start = None
        t_end = None
        if len(time) > 1:
            dt = np.mean(np.diff(time))
            if dt > 0:
                sample_rate = 1.0 / dt

        if round(sample_rate) >= 1.99e6:  # 2MHz
            sample_rate = 2e6
            if shot_num >= 41660:
                t_start, t_end = 0.26, 0.36
            else:
                t_start, t_end = 0.24, 0.34
        elif round(sample_rate) >= 249e3:  # 250kHz
            sample_rate = 250e3
            if shot_num >= 41660:
                t_start, t_end = 0.26, 0.36
            else:
                t_start, t_end = 0.24, 0.34
        elif round(sample_rate) >= 24.9e3:  # 25kHz
            sample_rate = 25e3

        if t_start is not None and t_end is not None and len(time) > 1:
            self.logger.info(
                f"{log_tag('VESTD', 'PROC')} High-speed DAQ "
                f"({sample_rate:.0e} Hz) detected. Recalculating time axis"
            )
            new_time = np.linspace(t_start, t_end, len(time) + 1)
            time = new_time[:-1]
        else:
            self.logger.info(
                f"{log_tag('VESTD', 'PROC')} Normal-speed DAQ detected. Using original time values."
            )

        return time, data, sample_rate
