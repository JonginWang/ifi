#!/usr/bin/env python3
"""
Post-processing HDF5 read utilities
===================================

Result read helpers for analysis outputs when post-processing.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .io_h5 import (
    H5_GROUP_CWT,
    H5_GROUP_DENSITY,
    H5_GROUP_RAWDATA,
    H5_GROUP_STFT,
    H5_GROUP_VEST,
    decode_attr_value,
    unflatten_metadata_attrs,
)
from .io_process_common import make_density_group_name, parse_density_group_name


def _decode_group_attrs(
    group: h5py.Group,
    excluded_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Decode HDF5 group attrs into plain Python values."""
    excluded = excluded_keys or set()
    attrs = {
        str(key): decode_attr_value(value)
        for key, value in group.attrs.items()
        if str(key) not in excluded
    }
    return unflatten_metadata_attrs(attrs)


def _load_metadata(hf: h5py.File) -> dict[str, Any]:
    return unflatten_metadata_attrs(hf.attrs)


def _load_raw_signals(hf: h5py.File) -> dict[str, pd.DataFrame]:
    signals_group = hf.get(H5_GROUP_RAWDATA)
    if signals_group is None or signals_group.attrs.get("empty", False):
        return {}

    signals: dict[str, pd.DataFrame] = {}
    for sub_name in signals_group.keys():
        signal_group = signals_group[sub_name]
        if not isinstance(signal_group, h5py.Group):
            continue
        orig_name = signal_group.attrs.get(
            "canonical_name",
            signal_group.attrs.get("original_name", sub_name),
        )
        signal_data: dict[str, np.ndarray] = {}
        for col_name in signal_group.keys():
            if str(col_name).startswith("_"):
                continue
            dset = signal_group[col_name]
            if isinstance(dset, h5py.Dataset):
                col_orig = dset.attrs.get("original_name", col_name)
                signal_data[str(col_orig)] = dset[:]
        signal_df = pd.DataFrame(signal_data)
        signal_df.attrs.update(
            _decode_group_attrs(
                signal_group,
                excluded_keys={"original_name", "canonical_name"},
            )
        )
        signals[str(orig_name)] = signal_df
    return signals


def _load_transform_group(
    hf: h5py.File,
    group_name: str,
) -> dict[str, dict[str, Any]]:
    group = hf.get(group_name)
    if group is None:
        return {}

    results: dict[str, dict[str, Any]] = {}
    for sub_name in group.keys():
        signal_group = group[sub_name]
        if not isinstance(signal_group, h5py.Group):
            continue
        orig_name = signal_group.attrs.get(
            "canonical_name",
            signal_group.attrs.get("original_name", sub_name),
        )
        data: dict[str, Any] = {}
        for key in signal_group.keys():
            dset = signal_group[key]
            if isinstance(dset, h5py.Dataset):
                key_orig = dset.attrs.get("original_name", key)
                data[str(key_orig)] = dset[:]
        data.update(
            _decode_group_attrs(
                signal_group,
                excluded_keys={"original_name", "canonical_name"},
            )
        )
        results[str(orig_name)] = data
    return results


def _density_freq_key_from_group(group_name: str, group: h5py.Group) -> str:
    """Recover canonical density key for structured density subgroups."""
    parsed = parse_density_group_name(group_name)
    if parsed is not None:
        return make_density_group_name(parsed)
    return make_density_group_name(group_name)


def _load_density(hf: h5py.File) -> pd.DataFrame | dict[str, pd.DataFrame] | None:
    density_group = hf.get(H5_GROUP_DENSITY)
    if density_group is None:
        return None

    structured_density: dict[str, pd.DataFrame] = {}
    for sub_name in density_group.keys():
        sub_obj = density_group[sub_name]
        if not isinstance(sub_obj, h5py.Group):
            continue
        freq_density_data: dict[str, np.ndarray] = {}
        for col_name in sub_obj.keys():
            dset = sub_obj[col_name]
            if isinstance(dset, h5py.Dataset):
                orig_col = dset.attrs.get("original_name", col_name)
                freq_density_data[str(orig_col)] = dset[:]
        if freq_density_data:
            freq_key = _density_freq_key_from_group(str(sub_name), sub_obj)
            freq_df = pd.DataFrame(freq_density_data)
            freq_df.attrs.update(_decode_group_attrs(sub_obj))
            structured_density[freq_key] = freq_df
    return structured_density or None


def load_vest_structured(vest_group: h5py.Group) -> dict[str, pd.DataFrame]:
    """Load structured VEST groups from `/vest_data`."""
    vest_by_rate: dict[str, pd.DataFrame] = {}
    for group_name in vest_group.keys():
        sr_group = vest_group[group_name]
        if not isinstance(sr_group, h5py.Group):
            continue
        if not str(group_name).upper().startswith("SR_"):
            continue

        columns: dict[str, np.ndarray] = {}
        n_points = None
        for ds_name in sr_group.keys():
            dset = sr_group[ds_name]
            if not isinstance(dset, h5py.Dataset):
                continue
            col_name = dset.attrs.get("file_name", dset.attrs.get("original_name", ds_name))
            arr = np.asarray(dset[:])
            columns[str(col_name)] = arr
            if n_points is None:
                n_points = len(arr)

        if not columns:
            continue

        df = pd.DataFrame(columns)
        if n_points and "t_start" in sr_group.attrs and "t_end" in sr_group.attrs:
            try:
                t_start = float(sr_group.attrs["t_start"])
                t_end = float(sr_group.attrs["t_end"])
                df.index = np.linspace(t_start, t_end, n_points, endpoint=False)
            except Exception:
                pass

        rate_key = str(group_name)[3:]
        vest_by_rate[rate_key] = df
    return vest_by_rate


def _load_vest(hf: h5py.File) -> dict[str, pd.DataFrame]:
    vest_group = hf.get(H5_GROUP_VEST)
    if vest_group is None or not isinstance(vest_group, h5py.Group):
        return {}

    vest_by_rate = load_vest_structured(vest_group)
    return vest_by_rate


def load_results_from_hdf5(shot_num: int, base_dir: str = "results") -> dict | None:
    """Load results from canonical `base_dir/<shot>/<shot>.h5`."""
    results_dir = Path(base_dir) / str(shot_num)
    if not results_dir.exists():
        print(f"No results found for shot {shot_num}")
        return None

    canonical_h5 = results_dir / f"{shot_num}.h5"
    if not canonical_h5.exists():
        print(f"No canonical results file found: {canonical_h5}")
        return None

    results: dict[str, Any] = {}
    print(f"Loading results from {canonical_h5}")
    try:
        with h5py.File(canonical_h5, "r") as hf:
            metadata = _load_metadata(hf)
            if metadata:
                results["metadata"] = metadata

            rawdata = _load_raw_signals(hf)
            if rawdata:
                results["rawdata"] = rawdata

            stft_results = _load_transform_group(
                hf,
                H5_GROUP_STFT,
            )
            if stft_results:
                results["stft"] = stft_results

            cwt_results = _load_transform_group(
                hf,
                H5_GROUP_CWT,
            )
            if cwt_results:
                results["cwt"] = cwt_results

            density = _load_density(hf)
            if density is not None:
                results["density"] = density

            vestdata = _load_vest(hf)
            if vestdata:
                results["vestdata"] = vestdata
    except Exception as e:
        print(f"Failed to load results from {canonical_h5}: {e}")

    return results


def load_cached_shot_data(shot_num: int, base_dir: str = "results") -> dict | None:
    """Deprecated compatibility wrapper."""
    print("load_cached_shot_data is deprecated. Use load_results_from_hdf5 instead.")
    return load_results_from_hdf5(shot_num, base_dir=base_dir)


__all__ = [
    "load_results_from_hdf5",
    "load_cached_shot_data",
]
