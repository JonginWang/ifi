#!/usr/bin/env python3
"""
Shot-level post-processing context phase
========================================

This module contains the "the shot-level post-processing context phase" for `run_analysis`.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ...utils.if_utils import (
    build_combined_signals_by_frequency,
    build_frequency_groups_from_params,
    filter_frequency_groups,
)
from ...utils.log_manager import log_tag


def collect_shot_interferometry_params(
    shot_files: list[str],
    interferometry_params_by_file: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collect per-shot basename->interferometry parameter mapping."""
    shot_interferometry_params: dict[str, dict[str, Any]] = {}
    for file_path in shot_files:
        basename = Path(file_path).name
        if file_path not in interferometry_params_by_file:
            continue
        params = interferometry_params_by_file[file_path]
        shot_interferometry_params[basename] = params
        logging.info(f"{log_tag('ANALY','RUN')} File {basename}: ")
        logging.info(
            f"{log_tag('ANALY','RUN')} {params['method']} method, {params['freq_ghz']}GHz, ref={params['ref_col']}, probes={params['probe_cols']}"
        )
    return shot_interferometry_params


def build_frequency_context_for_shot(
    shot_nas_data: dict[str, pd.DataFrame],
    shot_interferometry_params: dict[str, dict[str, Any]],
    requested_freqs: list[float],
) -> tuple[dict[float, dict[str, list]], dict[float, pd.DataFrame]]:
    """Build frequency groups and combined frequency-keyed signal DataFrames."""
    freq_groups, mapped_infos, out_of_standard_infos = build_frequency_groups_from_params(
        shot_interferometry_params
    )

    for basename, freq_ghz in out_of_standard_infos:
        logging.warning(
            f"{log_tag('ANALY','RUN')} Frequency {freq_ghz} GHz outside standard ranges "
            f"(93-95 or 275-285). Using actual frequency as group key."
        )
    for basename, freq_ghz, group_freq in mapped_infos:
        logging.info(
            f"{log_tag('ANALY','RUN')} Mapped frequency {freq_ghz} GHz -> {group_freq} GHz group for {basename}"
        )

    logging.info(
        f"{log_tag('ANALY','RUN')} Frequency groups found: {list(freq_groups.keys())} GHz"
    )

    if requested_freqs:
        filtered_freq_groups, has_match = filter_frequency_groups(freq_groups, requested_freqs)
        if has_match:
            freq_groups = filtered_freq_groups
            logging.info(
                f"{log_tag('ANALY','RUN')} Filtered to frequencies: {list(freq_groups.keys())} GHz (requested: {requested_freqs})"
            )
        else:
            logging.warning(
                f"{log_tag('ANALY','RUN')} No files match requested frequencies {requested_freqs}. "
                f"Available frequencies: {list(freq_groups.keys())} GHz. Processing all frequencies."
            )

    for freq_ghz, group_info in freq_groups.items():
        logging.info(
            f"{log_tag('ANALY','RUN')} Processing {freq_ghz}GHz group with files: {group_info['files']}"
        )
        if freq_ghz not in (94.0, 280.0):
            logging.warning(f"{log_tag('ANALY','RUN')} Unknown frequency group: {freq_ghz} GHz")

    freq_combined_signals = build_combined_signals_by_frequency(
        shot_nas_data=shot_nas_data,
        freq_groups=freq_groups,
    )

    if 280.0 in freq_groups and 280.0 not in freq_combined_signals:
        logging.warning(f"{log_tag('ANALY','RUN')} No _ALL file found for 280GHz group")
    if 94.0 in freq_groups and 94.0 not in freq_combined_signals:
        logging.warning(f"{log_tag('ANALY','RUN')} No reference _0XX file found for 94GHz group")
    for freq_ghz, combined_df in freq_combined_signals.items():
        logging.info(
            f"{log_tag('ANALY','RUN')} {freq_ghz}GHz combined shape: {combined_df.shape}"
        )

    return freq_groups, freq_combined_signals


def resolve_vest_ip_data(current_vest_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Pick preferred VEST Ip DataFrame by rate key."""
    vest_ip_data = current_vest_data.get("25k")
    if vest_ip_data is None:
        vest_ip_data = current_vest_data.get("250k")
    return vest_ip_data


__all__ = [
    "build_frequency_context_for_shot",
    "collect_shot_interferometry_params",
    "resolve_vest_ip_data",
]
