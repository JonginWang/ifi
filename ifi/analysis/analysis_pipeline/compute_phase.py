#!/usr/bin/env python3
"""
Parallel compute and aggregation helpers
=========================================

This module contains helpers of "the parallel compute and aggregation phase" for
`run_analysis`.

Author: Jongin Wang
Date: 2025-01-16
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask
import pandas as pd

try:
    from ...utils.log_manager import log_tag
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.log_manager import log_tag


def run_parallel_file_processing(
    target_files: list[str],
    args: argparse.Namespace,
    load_and_process_fn: Callable[[str, str, argparse.Namespace], Any],
) -> list[tuple[Any, Any, Any, Any, Any]]:
    """Run load/process tasks in parallel via Dask."""
    config_path = "ifi/config.ini"
    tasks: list[Any] = []
    for file_path in target_files:
        maybe_task = load_and_process_fn(config_path, file_path, args)
        if hasattr(maybe_task, "dask"):
            tasks.append(maybe_task)
        else:
            tasks.append(dask.delayed(load_and_process_fn)(config_path, file_path, args))

    logging.info(
        f"{log_tag('ANALY','RUN')} Starting Dask computation for {len(tasks)} tasks..." + "\n"
    )
    logging.info(f"{log_tag('ANALY','RUN')} Requested scheduler: {args.scheduler}")
    logging.info(f"{log_tag('ANALY','RUN')} Target files: {len(target_files)}")

    start_time = time.time()
    scheduler = args.scheduler if args.scheduler else "threads"
    logging.info(f"{log_tag('ANALY','RUN')} Using scheduler: {scheduler}")
    results = dask.compute(*tasks, scheduler=scheduler)
    end_time = time.time()

    logging.info(f"{log_tag('ANALY','RUN')} Dask computation finished.")
    logging.info(f"{log_tag('ANALY','RUN')} Processing time: {end_time - start_time:.2f} seconds")
    logging.info(
        f"{log_tag('ANALY','RUN')} Average time per file: {(end_time - start_time) / len(target_files):.2f} seconds"
    )
    return list(results)


def aggregate_dask_results(
    results: list[tuple[Any, Any, Any, Any, Any]],
) -> tuple[
    dict[int, dict[str, pd.DataFrame]],
    dict[int, dict[str, pd.DataFrame]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    """Aggregate per-file processed outputs into per-shot dictionaries."""
    analysis_data: dict[int, dict[str, pd.DataFrame]] = defaultdict(dict)
    raw_data: dict[int, dict[str, pd.DataFrame]] = defaultdict(dict)
    stft_results: dict[int, dict[str, Any]] = defaultdict(dict)
    cwt_results: dict[int, dict[str, Any]] = defaultdict(dict)

    for result in results:
        if not result or result[0] is None:
            continue

        file_path, df_processed, df_raw_single, stft_result, cwt_result = result
        if df_processed is None:
            continue

        match = re.search(r"(\d{5,})", Path(file_path).name)
        shot_num = int(match.group(1)) if match else 0

        analysis_data[shot_num][Path(file_path).name] = df_processed
        if df_raw_single is not None:
            raw_data[shot_num][Path(file_path).name] = df_raw_single
        if stft_result:
            stft_results[shot_num].update(stft_result)
        if cwt_result:
            cwt_results[shot_num].update(cwt_result)

    return analysis_data, raw_data, stft_results, cwt_results


__all__ = [
    "aggregate_dask_results",
    "run_parallel_file_processing",
]
