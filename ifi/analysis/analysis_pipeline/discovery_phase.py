#!/usr/bin/env python3
"""
Target discovery and metadata loading phase
============================================

This module contains the "the target discovery and metadata loading phase" for `run_analysis`.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ...db_controller.vest_db import VestDB
from ...utils.if_utils import assign_interferometry_params_to_shot
from ...utils.log_manager import log_tag

if TYPE_CHECKING:
    from ...utils.vest_utils import FlatShotList

def build_analysis_query(
    flat_list: "FlatShotList",
    shots_to_analyze: set[int],
) -> list[int | str]:
    """Build query list containing only shots requiring analysis plus explicit file paths."""
    shots_query = [s for s in flat_list.all if isinstance(s, int) and s in shots_to_analyze]
    shots_query.extend([p for p in flat_list.paths])
    return shots_query if shots_query else flat_list.all


def group_files_and_interferometry(
    target_files: list[str],
) -> tuple[dict[int | str, list[str]], dict[str, dict[str, Any]]]:
    """Group found files by shot and resolve per-file interferometry params."""
    files_by_shot: dict[int | str, list[str]] = defaultdict(list)
    interferometry_params_by_file: dict[str, dict[str, Any]] = {}

    for file_path in target_files:
        match = re.search(r"(\d{5,})", Path(file_path).name)
        if match:
            shot_num = int(match.group(1))
            files_by_shot[shot_num].append(file_path)
            params = assign_interferometry_params_to_shot(shot_num, Path(file_path).name)
            interferometry_params_by_file[file_path] = params
            logging.info(
                f"{log_tag('ANALY','RUN')} Interferometry (Shot #{shot_num}) params for {Path(file_path).name}: "
            )
            logging.info(
                f"{log_tag('ANALY','RUN')} {params['method']} method, {params['freq_ghz']} GHz"
            )
            continue

        files_by_shot["unknown"].append(file_path)
        params = assign_interferometry_params_to_shot(0, Path(file_path).name)
        interferometry_params_by_file[file_path] = params
        logging.info(
            f"{log_tag('ANALY','RUN')} Interferometry (Shot #00000) params for {Path(file_path).name}: "
        )
        logging.info(
            f"{log_tag('ANALY','RUN')} {params['method']} method, {params['freq_ghz']} GHz"
        )

    logging.info(
        f"{log_tag('ANALY','RUN')} Grouped files into {len(files_by_shot)} shot(s)." + "\n"
    )
    return files_by_shot, interferometry_params_by_file


def load_vest_data_for_shots(
    flat_list: "FlatShotList",
    vest_db: VestDB,
    args: argparse.Namespace,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Load VEST data for shot numbers present in query."""
    vest_data_by_shot: dict[int, dict[str, pd.DataFrame]] = defaultdict(dict)
    if not flat_list.nums:
        return vest_data_by_shot

    logging.info(f"{log_tag('ANALY','RUN')} Loading VEST data for shots: {flat_list.nums}")
    shot_num = None
    try:
        for shot_num in flat_list.nums:
            vest_data_by_shot[shot_num] = (
                vest_db.load_shot(shot=shot_num, fields=args.vest_fields)
                if shot_num > 0
                else {}
            )
    except Exception as e:
        logging.error(
            f"{log_tag('ANALY','RUN')} Error loading VEST data for shot {shot_num}: {e}"
        )
        if shot_num is not None:
            vest_data_by_shot[shot_num] = {}
    finally:
        vest_db.disconnect()
        logging.info(f"{log_tag('ANALY','RUN')} Disconnected from VEST database.")

    return vest_data_by_shot


__all__ = [
    "build_analysis_query",
    "group_files_and_interferometry",
    "load_vest_data_for_shots",
]
