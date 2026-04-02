#!/usr/bin/env python3
"""
Plot/save/final-merge phase
============================

This module contains the "the plot/save/final-merge phase" for `run_analysis`.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...plots.plot import Plotter, interactive_plotting
from ...utils.dsp_amplitude import (
    export_probe_envelope_segments_json,
    extract_probe_amplitudes_from_signals,
)
from ...utils.if_utils import get_frequency_df
from ...utils.io_process_common import parse_density_group_name
from ...utils.io_process_write import save_results_to_hdf5
from ...utils.log_manager import log_tag
from ..workflow import (
    build_frequency_metadata_map,
    build_plot_overview_maps,
    build_signals_dict_for_hdf5,
    merge_cached_results_into_analysis_bundle,
)


def plot_shot_outputs(
    *,
    shot_num: int,
    args: argparse.Namespace,
    shot_stft_data: dict[str, Any],
    shot_cwt_data: dict[str, Any],
    combined_signals: dict[float, pd.DataFrame],
    density_data: dict[str, pd.DataFrame],
    vest_ip_data: pd.DataFrame | None,
) -> None:
    """Render plot outputs for one shot based on CLI flags."""
    if not (args.plot or args.save_plots):
        return

    logging.info(f"{log_tag('ANALY','RUN')} Generating plots...")
    title_prefix = f"Shot #{shot_num} - " if shot_num else ""
    plotter = Plotter()

    with interactive_plotting(
        show_plots=args.plot,
        save_dir=Path(args.results_dir) / str(shot_num) if args.save_plots else None,
        save_prefix=title_prefix,
        block=not args.no_plot_block,
    ):
        if not args.no_plot_ft:
            if shot_stft_data:
                plotter.plot_spectrograms(
                    shot_stft_data,
                    title_prefix=title_prefix,
                    trigger_time=args.trigger_time,
                    downsample=args.downsample,
                )
            if shot_cwt_data:
                plotter.plot_cwt(
                    shot_cwt_data,
                    trigger_time=args.trigger_time,
                    title_prefix=title_prefix,
                )

        plot_signals, plot_density = build_plot_overview_maps(
            combined_signals=combined_signals,
            density_data=density_data,
        )
        if args.no_plot_raw:
            plot_signals = {}

        plot_probe_amplitudes = None
        if (
            args.color_density_by_amplitude
            and isinstance(density_data, dict)
            and isinstance(combined_signals, dict)
        ):
            merged_probe_amplitudes: dict[str, np.ndarray] = {}
            for freq_key, density_df in density_data.items():
                if not isinstance(density_df, pd.DataFrame) or density_df.empty:
                    continue
                freq_value = parse_density_group_name(str(freq_key))
                if freq_value is None:
                    continue
                signal_df = get_frequency_df(combined_signals, freq_value)
                if signal_df.empty:
                    continue
                probe_amp = extract_probe_amplitudes_from_signals(
                    density_df,
                    signal_df,
                    freq_value,
                )
                if probe_amp:
                    merged_probe_amplitudes.update(probe_amp)
            if merged_probe_amplitudes:
                plot_probe_amplitudes = merged_probe_amplitudes

        plotter.plot_analysis_overview(
            shot_num,
            plot_signals,
            plot_density,
            vest_ip_data,
            trigger_time=args.trigger_time,
            title_prefix=title_prefix,
            downsample=args.downsample,
            plot_envelope=getattr(args, "plot_envelope", False),
            color_density_by_amplitude=args.color_density_by_amplitude,
            probe_amplitudes=plot_probe_amplitudes,
            amplitude_colormap=args.amplitude_colormap,
            amplitude_impedance=args.amplitude_impedance,
        )


def _iter_envelope_probe_channels(params: dict[str, Any]) -> list[str]:
    amp_probe_cols = params.get("amp_probe_cols")
    if isinstance(amp_probe_cols, list) and amp_probe_cols:
        return [str(col) for col in amp_probe_cols if isinstance(col, str)]

    probe_cols = params.get("probe_cols", [])
    return [str(col) for col in probe_cols if isinstance(col, str)]


def export_envelope_outputs(
    *,
    shot_num: int,
    args: argparse.Namespace,
    shot_nas_data: dict[str, pd.DataFrame],
    shot_interferometry_params: dict[str, dict[str, Any]],
) -> None:
    """Export low-envelope probe segments as per-channel JSON files."""
    if not getattr(args, "envelope", False):
        return

    output_dir = Path(args.results_dir) / str(shot_num) / "envelope"
    output_dir.mkdir(parents=True, exist_ok=True)

    for basename, signal_df in shot_nas_data.items():
        if not isinstance(signal_df, pd.DataFrame) or signal_df.empty:
            continue

        params = shot_interferometry_params.get(basename, {})
        if not params:
            continue

        time_axis = (
            signal_df["TIME"].to_numpy(dtype=float)
            if "TIME" in signal_df.columns
            else signal_df.index.to_numpy(dtype=float)
        )
        if len(time_axis) == 0:
            continue

        for probe_col in _iter_envelope_probe_channels(params):
            if probe_col not in signal_df.columns:
                continue

            output_path = output_dir / f"{Path(basename).stem}_{probe_col}_envelope.json"
            payload = export_probe_envelope_segments_json(
                output_path,
                channel_name=probe_col,
                time_axis=time_axis,
                signal=signal_df[probe_col].to_numpy(dtype=float),
                threshold_ratio=0.7,
                min_duration_us=100.0,
                baseline_fraction=0.2,
                include_samples=True,
            )
            payload["shot_num"] = int(shot_num)
            payload["source_name"] = str(basename)
            payload["meas_name"] = str(params.get("meas_name", ""))
            payload["freq"] = float(params.get("freq_ghz", params.get("freq", float("nan"))))
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logging.info(f"{log_tag('ANALY','ENVL')} Saved envelope JSON: {output_path}")


def save_shot_outputs(
    *,
    shot_num: int,
    args: argparse.Namespace,
    shot_raw_data: dict[str, pd.DataFrame],
    freq_groups: dict[float, dict[str, list]],
    shot_interferometry_params: dict[str, dict[str, Any]],
    shot_stft_data: dict[str, Any],
    shot_cwt_data: dict[str, Any],
    density_data: dict[str, pd.DataFrame],
    current_vest_data: dict[str, pd.DataFrame],
) -> None:
    """Persist shot outputs into HDF5 if requested."""
    if not args.save_data:
        return

    output_dir = Path(args.results_dir) / str(shot_num) if shot_num != 0 else "unknown_shots"
    signals_dict = build_signals_dict_for_hdf5(
        shot_raw_data,
        shot_interferometry_params,
    )
    density_metadata = build_frequency_metadata_map(freq_groups)

    save_results_to_hdf5(
        output_dir,
        shot_num,
        signals_dict,
        shot_stft_data,
        shot_cwt_data,
        density_data,
        current_vest_data,
        analysis_params_by_source=shot_interferometry_params,
        density_meta_by_freq=density_metadata,
    )


def merge_cached_results_with_bundles(
    all_analysis_bundles: dict[int, dict[str, Any]],
    results_data_by_shot: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Merge cached results into computed bundles, preserving newly computed values."""
    if not results_data_by_shot:
        return all_analysis_bundles

    for shot_num, cached_results in results_data_by_shot.items():
        if shot_num in all_analysis_bundles:
            logging.info(
                f"{log_tag('ANALY','RSLT')} Merging cached and newly analyzed results for shot {shot_num}"
            )
            all_analysis_bundles[shot_num] = merge_cached_results_into_analysis_bundle(
                all_analysis_bundles[shot_num],
                cached_results,
            )
            continue
        all_analysis_bundles[shot_num] = {"analysis_results": cached_results}

    return all_analysis_bundles


__all__ = [
    "export_envelope_outputs",
    "merge_cached_results_with_bundles",
    "plot_shot_outputs",
    "save_shot_outputs",
]


