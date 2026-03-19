#!/usr/bin/env python3
"""
NasDB results/cache helper mixin
================================

This mixin is responsible for loading/writing raw-cache and reusing results signals.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .. import get_project_root
from ..utils.if_utils import assign_interferometry_params_to_shot
from ..utils.io_h5 import (
    H5_GROUP_RAWDATA,
    ensure_unique_name,
    flatten_metadata_attrs,
    h5_safe_name,
    normalize_source_name,
    unflatten_metadata_attrs,
)
from ..utils.io_process_common import build_raw_cache_file_path, make_raw_cache_group_name
from ..utils.log_manager import log_tag
from .nas_db_base import NasDBBase


class NasDBMixinCache(NasDBBase):
    """Helpers for loading/writing raw-cache and reusing results signals."""

    def _is_valid_data(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        if df.isnull().any().any():
            return False
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and np.isinf(df[numeric_cols]).any().any():
            return False
        return not (df.shape[0] == 0 or df.shape[1] == 0)

    def _get_results_raw_cache_file(self, shot_num: int, source_name: str) -> Path:
        """
        Build per-source raw cache path under `ifi/results/<shot>/`.
        Reserved analysis filename `<shot>.h5` is avoided.
        """
        project_root = get_project_root()
        results_dir = project_root / "ifi" / "results" / str(shot_num)
        return build_raw_cache_file_path(results_dir, source_name, shot_num=shot_num)

    def _load_single_raw_cache_h5(self, cache_file: Path, source_name: str) -> pd.DataFrame | None:
        """Load one source DataFrame from per-source raw cache H5."""
        if not cache_file.exists():
            return None

        canonical_source = normalize_source_name(source_name)
        try:
            with h5py.File(cache_file, "r") as hf:
                raw_group = hf.get(H5_GROUP_RAWDATA)
                if raw_group is None or raw_group.attrs.get("empty", False):
                    return None

                candidate_groups: list[h5py.Group] = []
                for group_name in raw_group.keys():
                    grp = raw_group[group_name]
                    if not isinstance(grp, h5py.Group):
                        continue
                    grp_original = normalize_source_name(str(grp.attrs.get("original_name", "")))
                    grp_canonical = normalize_source_name(str(grp.attrs.get("canonical_name", "")))
                    if canonical_source in (grp_original, grp_canonical):
                        candidate_groups.append(grp)

                if not candidate_groups:
                    for group_name in raw_group.keys():
                        grp = raw_group[group_name]
                        if isinstance(grp, h5py.Group):
                            candidate_groups.append(grp)
                            break
                if not candidate_groups:
                    return None

                grp = candidate_groups[0]
                signal_data: dict[str, np.ndarray] = {}
                for col_name in grp.keys():
                    if str(col_name).startswith("_"):
                        continue
                    dset = grp[col_name]
                    if isinstance(dset, h5py.Dataset):
                        col_original = str(dset.attrs.get("original_name", col_name))
                        signal_data[col_original] = dset[:]
                if not signal_data:
                    return None

                df = pd.DataFrame(signal_data)
                raw_attrs = {
                    str(k): v
                    for k, v in grp.attrs.items()
                    if k not in ("original_name", "canonical_name")
                }
                if raw_attrs:
                    flat_meta = unflatten_metadata_attrs(raw_attrs)
                    if flat_meta:
                        df.attrs.update(flat_meta)
                return df
        except Exception as e:
            self.logger.error(
                f"{log_tag('NASDB','CACHE')} Failed to read raw cache '{cache_file}': {e}"
            )
            return None

    def _write_single_raw_cache_h5(self, cache_file: Path, source_name: str, df: pd.DataFrame) -> None:
        """Write one source DataFrame to per-source raw cache H5."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        canonical_source = normalize_source_name(source_name)
        signal_group_name = make_raw_cache_group_name(canonical_source)

        with h5py.File(cache_file, "a") as hf:
            raw_group = hf.require_group(H5_GROUP_RAWDATA)
            raw_group.attrs.pop("empty", None)

            for group_name in list(raw_group.keys()):
                grp = raw_group[group_name]
                if not isinstance(grp, h5py.Group):
                    continue
                grp_original = normalize_source_name(str(grp.attrs.get("original_name", "")))
                grp_canonical = normalize_source_name(str(grp.attrs.get("canonical_name", "")))
                if group_name == signal_group_name or canonical_source in (grp_original, grp_canonical):
                    del raw_group[group_name]

            signal_group = raw_group.create_group(signal_group_name)
            signal_group.attrs["original_name"] = str(source_name)
            signal_group.attrs["canonical_name"] = str(canonical_source)

            if hasattr(df, "attrs") and df.attrs:
                for attr_key, attr_value in flatten_metadata_attrs(dict(df.attrs)).items():
                    signal_group.attrs[attr_key] = attr_value

            used_col_names: set[str] = set()
            for col in df.columns:
                safe_col = ensure_unique_name(h5_safe_name(str(col)), used_col_names)
                dset = signal_group.create_dataset(safe_col, data=df[col].to_numpy())
                dset.attrs["original_name"] = str(col)

    def _load_signals_from_results(
        self,
        shot_num: int,
        results_dir: Path,
        allowed_frequencies: list[float] | None = None,
    ) -> dict[str, pd.DataFrame] | None:
        """Load valid rawdata signals from canonical results H5."""
        results_file = results_dir / str(shot_num) / f"{shot_num}.h5"
        if not results_file.exists():
            return None

        try:
            with h5py.File(results_file, "r") as hf:
                signals_group = hf.get(H5_GROUP_RAWDATA)
                if signals_group is None or signals_group.attrs.get("empty", False):
                    return None

                signals_dict: dict[str, pd.DataFrame] = {}
                for sub_name in signals_group.keys():
                    signal_group = signals_group[sub_name]
                    if not isinstance(signal_group, h5py.Group):
                        continue
                    signal_name = signal_group.attrs.get("original_name", sub_name)

                    raw_freq = signal_group.attrs.get("freq")
                    if raw_freq is not None and allowed_frequencies is not None:
                        signal_freq = float(raw_freq) / 1.0e9
                        if 93.0 <= signal_freq <= 95.0:
                            group_freq = 94.0
                        elif 275.0 <= signal_freq <= 285.0:
                            group_freq = 280.0
                        else:
                            group_freq = signal_freq
                        if group_freq not in allowed_frequencies:
                            continue

                    signal_data = {}
                    for col_name in signal_group.keys():
                        if col_name.startswith("_"):
                            continue
                        dset = signal_group[col_name]
                        col_orig = dset.attrs.get("original_name", col_name)
                        signal_data[col_orig] = dset[:]

                    df = pd.DataFrame(signal_data)
                    if not self._is_valid_data(df):
                        self.logger.warning(
                            f"{log_tag('NASDB','RSLT')} Signal '{signal_name}' in results has invalid data. Skipping."
                        )
                        continue

                    signals_dict[signal_name] = df
                    self.logger.info(
                        f"{log_tag('NASDB','RSLT')} Loaded signal '{signal_name}' from results with shape {df.shape}"
                    )
                return signals_dict or None
        except Exception as e:
            self.logger.warning(
                f"{log_tag('NASDB','RSLT')} Error loading signals from results file '{results_file}': {e}"
            )
            return None

    def _convert_results_signals_to_nas_format(
        self,
        results_signals: dict[str, pd.DataFrame],
        target_files: list[str],
        shot_num: int,
    ) -> dict[str, pd.DataFrame]:
        """Map cached results signals to filename-keyed NAS dict."""
        nas_format_dict: dict[str, pd.DataFrame] = {}
        file_freq_map: dict[str, float] = {}
        for file_path in target_files:
            basename = Path(file_path).name
            try:
                params = assign_interferometry_params_to_shot(shot_num, basename)
                freq_ghz = params.get("freq_ghz", 94.0)
                if 93.0 <= freq_ghz <= 95.0:
                    group_freq = 94.0
                elif 275.0 <= freq_ghz <= 285.0:
                    group_freq = 280.0
                else:
                    group_freq = freq_ghz
                file_freq_map[basename] = group_freq
            except Exception as e:
                self.logger.warning(
                    f"{log_tag('NASDB','RSLT')} Could not determine frequency for file '{basename}': {e}"
                )
                basename_upper = basename.upper()
                file_freq_map[basename] = 280.0 if "_ALL" in basename_upper else 94.0

        for signal_name, df in results_signals.items():
            raw_freq = getattr(df, "attrs", {}).get("freq")
            if raw_freq is None:
                basename_upper = str(signal_name).upper()
                signal_freq = 280.0 if "_ALL" in basename_upper else 94.0
            else:
                signal_freq = float(raw_freq) / 1.0e9
            if 93.0 <= signal_freq <= 95.0:
                group_freq = 94.0
            elif 275.0 <= signal_freq <= 285.0:
                group_freq = 280.0
            else:
                group_freq = signal_freq

            matching_files = [
                basename for basename, freq in file_freq_map.items() if abs(freq - group_freq) < 0.1
            ]
            if not matching_files:
                continue

            if group_freq == 280.0:
                all_file = next((f for f in matching_files if "_ALL" in f), None)
                if all_file:
                    nas_format_dict[all_file] = df
            else:
                nas_format_dict[matching_files[0]] = df

        return nas_format_dict
