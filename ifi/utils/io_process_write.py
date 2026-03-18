#!/usr/bin/env python3
"""
Post-processing HDF5 write utilities
====================================

Result save and append helpers for analysis outputs when post-processing.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import sys
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
    ensure_unique_name,
    flatten_metadata_attrs,
    h5_safe_name,
    make_h5_group_name,
    normalize_source_name,
)
from .io_process_common import (
    create_named_dataset,
    find_existing_signal_group_name,
    make_density_group_name,
    parse_density_group_name,
    update_group_attrs,
)
from .path_utils import ensure_dir_exists
from .vest_utils import (
    extract_analysis_attrs,
    infer_field_meta,
    infer_sample_rate_from_index,
    load_vest_field_maps,
    normalize_sr_group_name,
    parse_rate_hz_from_key,
)


def _write_dataframe_datasets(parent: h5py.Group, df: pd.DataFrame) -> None:
    used_names: set[str] = set()
    for col in df.columns:
        create_named_dataset(
            parent,
            str(col),
            df[col].values,
            used_names=used_names,
            original_name=str(col),
        )


def _resolve_output_filename(shot_num: int, signals: dict | None) -> str:
    if shot_num == 0 and signals:
        first_source_file = list(signals.keys())[0]
        return f"{Path(first_source_file).stem}.h5"
    return f"{shot_num}.h5"


def _write_transform_results_group(
    parent: h5py.Group,
    group_results: dict,
    analysis_params_by_source: dict[str, dict[str, Any]] | None = None,
) -> None:
    used_group_names: set[str] = set()
    for signal_name, signal_data in group_results.items():
        if not isinstance(signal_data, dict):
            continue
        signal_group_name = make_h5_group_name(
            original_name=signal_name,
            prefix="sig",
            used_names=used_group_names,
            canonicalize_path=True,
        )
        signal_group = parent.create_group(signal_group_name)
        signal_group.attrs["original_name"] = str(signal_name)
        signal_group.attrs["canonical_name"] = normalize_source_name(signal_name)
        update_group_attrs(
            signal_group,
            extract_analysis_attrs((analysis_params_by_source or {}).get(str(signal_name))),
        )

        used_keys: set[str] = set()
        for key, value in signal_data.items():
            if isinstance(value, np.ndarray):
                create_named_dataset(
                    signal_group,
                    str(key),
                    value,
                    used_names=used_keys,
                    original_name=str(key),
                )
            elif isinstance(value, (int, float, str)):
                safe_key = ensure_unique_name(h5_safe_name(str(key)), used_keys)
                signal_group.attrs[safe_key] = value


def _write_density_group(
    hf: h5py.File,
    density_data: pd.DataFrame | dict[str, pd.DataFrame] | None,
    density_meta_by_freq: dict[float, dict[str, Any]] | None = None,
) -> None:
    if density_data is None:
        return
    density_group = hf.create_group(H5_GROUP_DENSITY)

    if isinstance(density_data, dict):
        for freq_key, freq_df in density_data.items():
            if not isinstance(freq_df, pd.DataFrame) or freq_df.empty:
                continue
            subgroup_name = make_density_group_name(freq_key)
            if subgroup_name in density_group:
                del density_group[subgroup_name]
            freq_group = density_group.create_group(subgroup_name)
            meta_key = parse_density_group_name(str(freq_key))
            if meta_key is None:
                try:
                    meta_key = float(freq_key)
                except (TypeError, ValueError):
                    meta_key = None
            update_group_attrs(
                freq_group,
                extract_analysis_attrs((density_meta_by_freq or {}).get(meta_key)),
            )
            _write_dataframe_datasets(freq_group, freq_df)
        return

    if isinstance(density_data, pd.DataFrame) and not density_data.empty:
        _write_dataframe_datasets(density_group, density_data)


def write_raw_signals_group(hf: h5py.File, signals: dict[str, pd.DataFrame]) -> int:
    """Write or replace `/rawdata` signal groups."""
    raw_group = hf.require_group(H5_GROUP_RAWDATA)
    raw_group.attrs.pop("empty", None)

    used_names = set(raw_group.keys())
    written = 0
    for signal_name, signal_df in signals.items():
        if not isinstance(signal_df, pd.DataFrame):
            continue

        existing_name = find_existing_signal_group_name(raw_group, str(signal_name))
        if existing_name is not None:
            del raw_group[existing_name]
            group_name = existing_name
        else:
            group_name = make_h5_group_name(
                original_name=str(signal_name),
                prefix="sig",
                used_names=used_names,
                canonicalize_path=True,
            )

        signal_group = raw_group.create_group(group_name)
        signal_group.attrs["original_name"] = str(signal_name)
        signal_group.attrs["canonical_name"] = normalize_source_name(signal_name)

        if hasattr(signal_df, "attrs") and signal_df.attrs:
            update_group_attrs(signal_group, flatten_metadata_attrs(dict(signal_df.attrs)))

        _write_dataframe_datasets(signal_group, signal_df)
        written += 1

    if written == 0:
        raw_group.attrs["empty"] = True
    return written


def write_structured_vest_groups(
    vest_root: h5py.Group,
    shot_num: int,
    vest_by_rate: dict[str, pd.DataFrame],
) -> int:
    """Write structured sample-rate grouped VEST datasets."""
    field_by_id, label_to_id = load_vest_field_maps()
    total_written = 0

    for rate_key, rate_df in vest_by_rate.items():
        if not isinstance(rate_df, pd.DataFrame) or rate_df.empty:
            continue

        sr_group_name = normalize_sr_group_name(rate_key)
        if sr_group_name in vest_root:
            del vest_root[sr_group_name]
        sr_group = vest_root.create_group(sr_group_name)

        sample_rate = parse_rate_hz_from_key(rate_key) or infer_sample_rate_from_index(rate_df.index)
        if sample_rate is not None:
            sr_group.attrs["sample_rate"] = float(sample_rate)
        if len(rate_df.index) > 0:
            try:
                sr_group.attrs["t_start"] = float(rate_df.index.min())
                sr_group.attrs["t_end"] = float(rate_df.index.max())
            except Exception:
                pass

        used_dataset_names: set[str] = set()
        for col in rate_df.columns:
            field_id, field_name, field_unit = infer_field_meta(
                str(col), field_by_id, label_to_id
            )
            if field_id is not None:
                dataset_base = f"shot_{shot_num}_{field_id}"
            else:
                dataset_base = f"shot_{shot_num}_{h5_safe_name(str(col))}"
            dset = create_named_dataset(
                sr_group,
                dataset_base,
                rate_df[col].to_numpy(),
                used_names=used_dataset_names,
                original_name=str(col),
            )
            dset.attrs["file_name"] = str(field_name)
            dset.attrs["field_unit"] = str(field_unit)
            if field_id is not None:
                dset.attrs["field_id"] = int(field_id)
            total_written += 1

    return total_written


def _write_vest_group(
    hf: h5py.File,
    shot_num: int,
    vest_data: pd.DataFrame | dict[str, pd.DataFrame] | None,
) -> None:
    if isinstance(vest_data, dict) and vest_data:
        vest_root = hf.create_group(H5_GROUP_VEST)
        write_structured_vest_groups(vest_root, shot_num, vest_data)
        return
    if isinstance(vest_data, pd.DataFrame) and not vest_data.empty:
        vest_root = hf.create_group(H5_GROUP_VEST)
        write_structured_vest_groups(
            vest_root,
            shot_num,
            {"A.U.": vest_data},
        )


def save_results_to_hdf5(
    output_dir: str,
    shot_num: int,
    signals: dict,
    stft_results: dict,
    cwt_results: dict,
    density_data: pd.DataFrame | dict[str, pd.DataFrame],
    vest_data: pd.DataFrame | dict[str, pd.DataFrame] | None,
    analysis_params_by_source: dict[str, dict[str, Any]] | None = None,
    density_meta_by_freq: dict[float, dict[str, Any]] | None = None,
) -> str | None:
    """Save all analysis results to an HDF5 file."""
    filename = _resolve_output_filename(shot_num, signals)
    filepath = Path(output_dir) / filename
    ensure_dir_exists(str(output_dir))

    try:
        with h5py.File(filepath, "w") as hf:
            run_metadata: dict[str, Any] = {
                "shot_number": shot_num,
                "created_at": pd.Timestamp.now().isoformat(),
            }
            try:
                run_metadata["ifi_version"] = __import__("ifi").__version__
            except Exception as e:
                print(f"Error getting ifi version: {e}")
                if "ifi" in sys.modules and hasattr(sys.modules["ifi"], "__version__"):
                    run_metadata["ifi_version"] = sys.modules["ifi"].__version__
                else:
                    run_metadata["ifi_version"] = "0.1.0"
            for attr_key, attr_value in flatten_metadata_attrs(run_metadata).items():
                hf.attrs[attr_key] = attr_value

            write_raw_signals_group(hf, signals or {})

            if stft_results:
                stft_group = hf.create_group(H5_GROUP_STFT)
                _write_transform_results_group(
                    stft_group,
                    stft_results,
                    analysis_params_by_source=analysis_params_by_source,
                )
            if cwt_results:
                cwt_group = hf.create_group(H5_GROUP_CWT)
                _write_transform_results_group(
                    cwt_group,
                    cwt_results,
                    analysis_params_by_source=analysis_params_by_source,
                )

            _write_density_group(
                hf,
                density_data,
                density_meta_by_freq=density_meta_by_freq,
            )
            _write_vest_group(hf, shot_num, vest_data)

        print(f"Results saved to: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"Error saving results to HDF5: {e}")
        return None


__all__ = [
    "save_results_to_hdf5",
]
