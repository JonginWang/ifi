#!/usr/bin/env python3
"""
Data Append Utility
====================

Append VEST/NAS data to existing result HDF5 files.

Examples:
    python ifi/utils/io_h5_append.py "47232:47241" --vest 101 140
    python ifi/utils/io_h5_append.py "47232,47235" --nas --suffix "056 789 ALL"
    python ifi/utils/io_h5_append.py "47232:47241" --nas --suffix "056 789 ALL" --vest 101 140

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import pandas as pd

from ..utils.io_h5 import H5_GROUP_VEST
from .io_process_write import (
    write_raw_signals_group,
    write_structured_vest_groups,
)

if TYPE_CHECKING:
    from ..db_controller.nas_db import NasDB
    from ..db_controller.vest_db import VestDB


def _parse_shots(text: str) -> list[int]:
    if not text:
        return set[int]()
    shots: list[int] = []
    for part in re.split(r"[,\s]+", text.strip()):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            if token.count(":") == 2:
                start_s, end_s, step_s = token.split(":")
                shots.extend(range(int(start_s), int(end_s) + 1, int(step_s)))
            else:
                start_s, end_s = token.split(":", 1)
                if int(start_s) <= int(end_s):
                    shots.extend(range(int(start_s), int(end_s) + 1))
        else:
            shots.append(int(token))
    return sorted(set(shots))


def _parse_suffix_tokens(text: str) -> set[str]:
    if not text:
        return set()
    tokens = re.split(r"[,\s]+", text.strip())
    return {tok.upper() for tok in tokens if tok}


def _match_suffix(filename: str, suffix_tokens: set[str]) -> bool:
    if not suffix_tokens:
        return True
    stem = Path(filename).stem
    parts = re.split(r"[_\-.]+", stem)
    parts_upper = {p.upper() for p in parts if p}
    return any(tok in parts_upper for tok in suffix_tokens)


def append_signals_to_hdf5(h5_path: str | Path, signals: dict[str, pd.DataFrame]) -> int:
    """Append or update raw NAS signals in an existing results HDF5 file."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "a") as hf:
        return write_raw_signals_group(hf, signals or {})


def append_vest_to_hdf5(
    h5_path: str | Path,
    shot_num: int,
    vest_data: pd.DataFrame | dict[str, pd.DataFrame],
) -> int:
    """Append VEST data to an existing results HDF5 file."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "a") as hf:
        vest_root = hf.require_group(H5_GROUP_VEST)
        if isinstance(vest_data, dict):
            return write_structured_vest_groups(vest_root, shot_num, vest_data)
    return 0


def _append_vest_for_shot(
    shot_num: int,
    h5_path: Path,
    vest_db: "VestDB",
    vest_fields: list[int],
) -> int:
    if not vest_fields:
        return 0
    vest_groups = vest_db.load_shot(shot=shot_num, fields=vest_fields)
    if not vest_groups:
        print(f"[VEST][SKIP] {shot_num}: no VEST data for fields {vest_fields}")
        return 0
    written = append_vest_to_hdf5(h5_path=h5_path, shot_num=shot_num, vest_data=vest_groups)
    print(f"[VEST][OK] {shot_num}: appended {written} dataset(s)")
    return written


def _append_nas_for_shot(
    shot_num: int,
    h5_path: Path,
    nas_db: "NasDB",
    suffix_tokens: set[str],
    force_remote: bool = False,
) -> int:
    nas_data = nas_db.load_shot(query=shot_num, force_remote=force_remote)
    if not nas_data:
        print(f"[NAS][SKIP] {shot_num}: no NAS/cache data")
        return 0

    filtered = {
        name: df for name, df in nas_data.items() if _match_suffix(name, suffix_tokens)
    }
    if not filtered:
        print(f"[NAS][SKIP] {shot_num}: no data matched suffix filter {sorted(suffix_tokens)}")
        return 0

    written = append_signals_to_hdf5(h5_path=h5_path, signals=filtered)
    print(f"[NAS][OK] {shot_num}: appended {written} signal group(s)")
    return written


def main() -> int:
    from ..db_controller.nas_db import NasDB
    from ..db_controller.vest_db import VestDB

    parser = argparse.ArgumentParser(
        description="Append VEST/NAS data to existing ifi/results/<shot>/<shot>.h5 files."
    )
    parser.add_argument(
        "shots",
        help="Shot list expression, e.g. '47232:47241:1' or '47232,47235 47240'",
    )
    parser.add_argument(
        "--results-dir",
        default="ifi/results",
        help="Base results directory (default: ifi/results)",
    )
    parser.add_argument(
        "--config",
        default="ifi/config.ini",
        help="Path to config.ini (default: ifi/config.ini)",
    )
    parser.add_argument(
        "--vest",
        nargs="*",
        type=int,
        default=[],
        help="VEST field IDs to append, e.g. --vest 101 140 214",
    )
    parser.add_argument(
        "--nas",
        action="store_true",
        help="Append NAS (or cached) raw signals to /rawdata",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix token filter for NAS filenames, e.g. '056 789 ALL'",
    )
    parser.add_argument(
        "--force-remote",
        action="store_true",
        help="Force NAS remote fetch (bypass cache/results priority in NasDB.load_shot)",
    )
    args = parser.parse_args()

    shots = _parse_shots(args.shots)
    results_dir = Path(args.results_dir)
    suffix_tokens = _parse_suffix_tokens(args.suffix)

    if not args.vest and not args.nas:
        print("Nothing to do: specify at least one of --vest or --nas")
        return 1

    total_shots = 0
    updated_shots = 0

    vest_db = VestDB(config_path=args.config) if args.vest else None
    nas_db = NasDB(config_path=args.config) if args.nas else None

    try:
        for shot_num in shots:
            total_shots += 1
            h5_path = results_dir / str(shot_num) / f"{shot_num}.h5"
            if not h5_path.exists():
                print(f"[SKIP] {shot_num}: no h5 file at {h5_path}")
                continue

            changed = 0
            if vest_db is not None:
                changed += _append_vest_for_shot(
                    shot_num=shot_num,
                    h5_path=h5_path,
                    vest_db=vest_db,
                    vest_fields=args.vest,
                )
            if nas_db is not None:
                changed += _append_nas_for_shot(
                    shot_num=shot_num,
                    h5_path=h5_path,
                    nas_db=nas_db,
                    suffix_tokens=suffix_tokens,
                    force_remote=args.force_remote,
                )

            if changed > 0:
                updated_shots += 1

    finally:
        if vest_db is not None:
            vest_db.disconnect()
        if nas_db is not None:
            nas_db.disconnect()

    print(f"Done. Updated {updated_shots}/{total_shots} shots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
