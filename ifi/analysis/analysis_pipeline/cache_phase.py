#!/usr/bin/env python3
"""
Cache validation phase
=======================


This module contains helpers of "the cache validation phase" for `run_analysis`.

Author: Jongin Wang
Date: 2025-01-16
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

from ...utils.io_process_read import load_results_from_hdf5
from ...utils.log_manager import log_tag
from ..workflow import evaluate_cached_results_summary

if TYPE_CHECKING:
    from ...utils.vest_utils import FlatShotList

def resolve_analysis_requirements(
    args: argparse.Namespace,
) -> tuple[bool, bool, bool, list[float]]:
    """Resolve requested analysis outputs from CLI arguments."""
    need_stft = args.stft if hasattr(args, "stft") else False
    need_cwt = args.cwt if hasattr(args, "cwt") else False
    need_density = args.density if hasattr(args, "density") else False
    requested_freqs = list(getattr(args, "freq", []) or [])
    return need_stft, need_cwt, need_density, requested_freqs


def scan_cached_results_for_query(
    flat_list: "FlatShotList",
    args: argparse.Namespace,
    requested_freqs: list[float],
    need_stft: bool,
    need_cwt: bool,
    need_density: bool,
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    """Inspect per-shot cached results and return reusable shots + shots needing analysis."""
    results_data_by_shot: dict[int, dict[str, Any]] = {}
    shots_to_analyze: set[int] = set()

    for shot_num in flat_list.nums:
        if shot_num == 0:
            continue

        results = load_results_from_hdf5(shot_num, base_dir=args.results_dir)
        if not results:
            shots_to_analyze.add(shot_num)
            logging.info(
                f"{log_tag('ANALY','RSLT')} No results found for shot {shot_num}. Will perform full analysis."
            )
            continue

        cache_summary = evaluate_cached_results_summary(
            results=results,
            requested_freqs=requested_freqs,
            need_stft=need_stft,
            need_cwt=need_cwt,
            need_density=need_density,
        )
        has_signals = cache_summary["has_signals"]
        has_stft = cache_summary["has_stft"]
        has_cwt = cache_summary["has_cwt"]
        has_density = cache_summary["has_density"]
        available_freqs = cache_summary["available_freqs"]
        missing_freqs = cache_summary["missing_requested_freqs"]

        if requested_freqs and not has_signals:
            if available_freqs:
                logging.info(
                    f"{log_tag('ANALY','RSLT')} Shot {shot_num} has signals for frequencies {sorted(available_freqs)} GHz, "
                    f"but missing frequencies {sorted(missing_freqs)} GHz. Will analyze missing frequencies."
                )
            else:
                logging.info(
                    f"{log_tag('ANALY','RSLT')} Shot {shot_num} has no matching signals for requested frequencies {sorted(set(requested_freqs))} GHz. Will analyze."
                )

        if has_signals and has_stft and has_cwt and has_density:
            results_data_by_shot[shot_num] = results
            logging.info(
                f"{log_tag('ANALY','RSLT')} Found complete analysis results for shot {shot_num} in results directory. "
                f"Will use cached results instead of re-analyzing."
            )
            continue

        shots_to_analyze.add(shot_num)
        missing = []
        if not has_signals:
            missing.append("rawdata")
        if need_stft and not has_stft:
            missing.append("stft")
        if need_cwt and not has_cwt:
            missing.append("cwt")
        if need_density and not has_density:
            missing.append("density")
        logging.info(
            f"{log_tag('ANALY','RSLT')} Shot {shot_num} has partial results. Missing: {', '.join(missing)}. "
            f"Will analyze to fill gaps."
        )

    return results_data_by_shot, shots_to_analyze


__all__ = [
    "resolve_analysis_requirements",
    "scan_cached_results_for_query",
]
