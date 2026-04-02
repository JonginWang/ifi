#!/usr/bin/env python3
"""
Process-specific HDF5 naming helpers
=====================================

This module contains the functions for process-specific HDF5 naming helpers.
It includes the functions for creating named datasets, updating group attrs,
and finding existing signal group names.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .io_h5 import (
    ensure_unique_name,
    h5_affixed_name,
    h5_safe_name,
    make_cache_h5_key,
    normalize_source_name,
    parse_h5_affixed_name,
)


def update_group_attrs(group: h5py.Group, attrs: dict[str, Any]) -> None:
    for attr_key, attr_value in attrs.items():
        group.attrs[attr_key] = attr_value


def create_named_dataset(
    parent: h5py.Group,
    name: str,
    data: np.ndarray | pd.Series | pd.DataFrame,
    *,
    used_names: set[str],
    original_name: str,
) -> h5py.Dataset:
    """Create a named dataset in an HDF5 group."""
    safe_name = ensure_unique_name(h5_safe_name(name), used_names)
    dset = parent.create_dataset(safe_name, data=data)
    dset.attrs["original_name"] = str(original_name)
    return dset


def make_density_group_name(freq_value: str | float | int) -> str:
    """Build canonical density subgroup name like `freq_94G`."""
    parsed_existing = parse_density_group_name(str(freq_value))
    if parsed_existing is not None:
        return h5_affixed_name(f"{parsed_existing:.0f}", prefix="freq", suffix="G", suffix_sep="")
    try:
        return h5_affixed_name(f"{float(freq_value):.0f}", prefix="freq", suffix="G", suffix_sep="")
    except (TypeError, ValueError):
        return h5_affixed_name(h5_safe_name(str(freq_value)), prefix="freq")


def parse_density_group_name(group_name: str) -> float | None:
    """Parse canonical density subgroup name like `freq_94G` into GHz."""
    stem = parse_h5_affixed_name(group_name, prefix="freq", suffix="G", suffix_sep="")
    if stem is None:
        return None
    try:
        return float(stem)
    except ValueError:
        return None


def find_existing_signal_group_name(raw_group: h5py.Group, source_name: str) -> str | None:
    """Find existing source group name inside `/rawdata`."""
    canonical = normalize_source_name(source_name)
    for group_name in raw_group.keys():
        obj = raw_group[group_name]
        if not isinstance(obj, h5py.Group):
            continue
        candidates = {
            str(obj.attrs.get("original_name", "")),
            str(obj.attrs.get("canonical_name", "")),
        }
        if source_name in candidates or canonical in candidates:
            return group_name
    return None


def make_raw_cache_group_name(source_name: str) -> str:
    """Build deterministic raw-cache subgroup name."""
    return make_cache_h5_key(normalize_source_name(source_name), prefix="sig")


def make_raw_cache_filename(source_name: str, shot_num: int | None = None) -> str:
    """Build per-source raw-cache filename within one shot results directory."""
    canonical_source = normalize_source_name(source_name)
    stem = str(Path(canonical_source).stem).strip()
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip("._")
    if not stem:
        stem = make_cache_h5_key(canonical_source, prefix="src")
    if shot_num is not None and stem == str(shot_num):
        stem = f"{stem}_raw"
    return f"{stem}.h5"


def build_raw_cache_file_path(
    results_dir: Path | str,
    source_name: str,
    shot_num: int | None = None,
) -> Path:
    """Build per-source raw-cache file path inside a shot results directory."""
    return Path(results_dir) / make_raw_cache_filename(source_name, shot_num=shot_num)


def ensure_time_indexed_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """Return a copy indexed by `TIME` and the corresponding time axis."""
    out = df.copy()
    if "TIME" in out.columns:
        time_axis = out["TIME"]
        out.index = time_axis
        out = out.drop("TIME", axis=1)
        out.index.name = "TIME"
        return out, time_axis
    return out, out.index


def ensure_time_column_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an explicit `TIME` column."""
    out = df.copy()
    if out.index.name == "TIME":
        return out.reset_index()
    if "TIME" in out.columns:
        return out
    out = out.reset_index()
    first_col = str(out.columns[0]) if len(out.columns) > 0 else "index"
    if first_col != "TIME":
        out.rename(columns={first_col: "TIME"}, inplace=True)
    return out


def append_source_stem_to_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Suffix signal columns with source stem to avoid collisions after merging."""
    stem = Path(source_name).stem
    out = df.copy()
    out.columns = [f"{col}_{stem}" for col in out.columns]
    return out


__all__ = [
    "append_source_stem_to_columns",
    "build_raw_cache_file_path",
    "update_group_attrs",
    "create_named_dataset",
    "ensure_time_column_df",
    "ensure_time_indexed_df",
    "make_raw_cache_filename",
    "make_raw_cache_group_name",
    "make_density_group_name",
    "parse_density_group_name",
    "find_existing_signal_group_name",
]
