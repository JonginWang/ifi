#!/usr/bin/env python3
"""
Analysis Workflow Utilities
===========================

Workflow utility helpers for analysis orchestration.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..utils.if_utils import (
    build_combined_signals_by_frequency,
    extract_available_frequency_groups,
    map_frequency_to_group,
)
from ..utils.io_h5 import (
    H5_GROUP_CWT,
    H5_GROUP_DENSITY,
    H5_GROUP_RAWDATA,
    H5_GROUP_STFT,
)
from ..utils.io_process_common import ensure_time_column_df, parse_density_group_name
from ..utils.vest_utils import extract_analysis_attrs


def has_density_data(density_obj: object) -> bool:
    """Return True when density payload contains at least one non-empty DataFrame."""
    if isinstance(density_obj, pd.DataFrame):
        return not density_obj.empty
    if isinstance(density_obj, dict):
        return any(
            isinstance(v, pd.DataFrame) and not v.empty
            for v in density_obj.values()
        )
    return False


def build_cached_analysis_bundles(
    results_data_by_shot: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Format cached results into run_analysis return structure."""
    bundles: dict[int, dict[str, Any]] = {}
    for shot_num, results in results_data_by_shot.items():
        bundles[shot_num] = {"analysis_results": results}
    return bundles


def evaluate_cached_results_summary(
    results: dict[str, Any],
    requested_freqs: list[float] | None,
    need_stft: bool,
    need_cwt: bool,
    need_density: bool,
) -> dict[str, Any]:
    """Evaluate cache completeness for one shot result payload."""
    requested = set(requested_freqs or [])

    has_signals = False
    available_freqs: set[float] = set()
    missing_requested_freqs: set[float] = set()

    rawdata = results.get(H5_GROUP_RAWDATA)
    if isinstance(rawdata, dict) and rawdata:
        for signal_name, signal_df in rawdata.items():
            freq_value = None
            if isinstance(signal_df, pd.DataFrame):
                freq_value = signal_df.attrs.get("freq")
            if freq_value is None:
                available_freqs.update(
                    extract_available_frequency_groups([str(signal_name)])
                )
                continue
            try:
                available_freqs.add(map_frequency_to_group(float(freq_value) / 1.0e9))
            except (TypeError, ValueError):
                available_freqs.update(
                    extract_available_frequency_groups([str(signal_name)])
                )
        if requested:
            if requested.issubset(available_freqs):
                has_signals = True
            else:
                missing_requested_freqs = requested - available_freqs
        else:
            has_signals = bool(available_freqs) or bool(rawdata)

    has_stft = bool(results.get(H5_GROUP_STFT)) if need_stft else True
    has_cwt = bool(results.get(H5_GROUP_CWT)) if need_cwt else True
    has_density = has_density_data(results.get(H5_GROUP_DENSITY)) if need_density else True

    return {
        "has_signals": has_signals,
        "has_stft": has_stft,
        "has_cwt": has_cwt,
        "has_density": has_density,
        "available_freqs": available_freqs,
        "missing_requested_freqs": missing_requested_freqs,
    }

def build_signals_dict_for_hdf5(
    shot_raw_data: dict[str, pd.DataFrame],
    shot_interferometry_params: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Build per-source raw signal payload for HDF5 save."""
    signals_dict: dict[str, pd.DataFrame] = {}
    for source_name, raw_df in shot_raw_data.items():
        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            continue
        out_df = ensure_time_column_df(raw_df)
        out_df.attrs.update(extract_analysis_attrs(shot_interferometry_params.get(source_name)))
        signals_dict[str(source_name)] = out_df
    return signals_dict


def build_frequency_metadata_map(
    freq_groups: dict[float, dict[str, list]],
) -> dict[float, dict[str, Any]]:
    """Build canonical density frequency metadata map from grouped params."""
    metadata_by_freq: dict[float, dict[str, Any]] = {}
    for freq_key, group_info in freq_groups.items():
        params_list = group_info.get("params", [])
        if not params_list:
            continue
        metadata = extract_analysis_attrs(params_list[0])
        if metadata:
            metadata_by_freq[float(freq_key)] = metadata
    return metadata_by_freq


def merge_cached_results_into_analysis_bundle(
    analysis_bundle: dict[str, Any],
    cached_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Fill missing fields in a newly analyzed bundle from cached results.

    Newly computed values always take precedence.
    """
    merged = dict(analysis_bundle)
    processed = dict(merged.get("processed_data", {}))
    analysis = dict(merged.get("analysis_results", {}))
    raw_data = dict(merged.get("raw_data", {}))

    current_signals = processed.get("signals")
    if not current_signals:
        processed["signals"] = cached_results.get("rawdata", {})

    current_density = processed.get("density")
    if not has_density_data(current_density):
        cached_density = cached_results.get("density")
        if cached_density is not None:
            processed["density"] = cached_density

    if not analysis.get("stft"):
        analysis["stft"] = cached_results.get("stft", {})
    if not analysis.get("cwt"):
        analysis["cwt"] = cached_results.get("cwt", {})

    if "vest" not in raw_data and "vestdata" in cached_results:
        raw_data["vest"] = cached_results.get("vestdata")

    merged["processed_data"] = processed
    merged["analysis_results"] = analysis
    merged["raw_data"] = raw_data
    return merged


def build_plot_overview_maps(
    combined_signals: object,
    density_data: object,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build `plot_analysis_overview` inputs from single- or multi-frequency payloads."""
    signal_map: dict[str, pd.DataFrame] = {}
    density_map: dict[str, pd.DataFrame] = {}

    if isinstance(combined_signals, dict):
        for freq, df in sorted(combined_signals.items()):
            if isinstance(df, pd.DataFrame) and not df.empty:
                signal_map[f"Processed Signals ({freq:g} GHz)"] = df
    elif isinstance(combined_signals, pd.DataFrame) and not combined_signals.empty:
        signal_map["Processed Signals"] = combined_signals

    if isinstance(density_data, dict):
        for freq_key, df in density_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            parsed = parse_density_group_name(str(freq_key))
            if parsed is not None:
                freq_label = f"{parsed:g}"
            else:
                freq_label = str(freq_key)
            density_map[f"Density ({freq_label} GHz)"] = df
    elif isinstance(density_data, pd.DataFrame) and not density_data.empty:
        density_map["Density"] = density_data

    return signal_map, density_map


__all__ = [
    "build_combined_signals_by_frequency",
    "build_cached_analysis_bundles",
    "build_plot_overview_maps",
    "build_signals_dict_for_hdf5",
    "evaluate_cached_results_summary",
    "has_density_data",
    "merge_cached_results_into_analysis_bundle",
]
