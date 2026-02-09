#!/usr/bin/env python3
"""
Appending VEST Shot data 
=========================


This script loads shotdata with the shot code(s) via VEST_DB
and appends it to /vest_data in ifi/results/<shot>/<shot>.h5

Fucntions:
    append_shotdata_for_shot(shot_num, results_dir, vest_db, field_id=140)
    main()
    _parse_shots(text)
    _rate_key_to_hz(rate_key)
    _select_rate_group(vest_groups)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py

from ifi.db_controller.vest_db import VEST_DB


def _parse_shots(text: str) -> list[int]:
    shots: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_s, end_s = part.split(":", 1)
            start = int(start_s)
            end = int(end_s)
            shots.extend(range(start, end + 1))
        else:
            shots.append(int(part))
    return sorted(set(shots))


def _rate_key_to_hz(rate_key: str) -> int | None:
    if rate_key.endswith("k"):
        try:
            return int(rate_key[:-1]) * 1000
        except ValueError:
            return None
    if rate_key.endswith("M"):
        try:
            return int(rate_key[:-1]) * 1_000_000
        except ValueError:
            return None
    return None


def _select_rate_group(vest_groups: dict[str, object]) -> tuple[str, object] | None:
    if "250k" in vest_groups:
        return "250k", vest_groups["250k"]
    if "25k" in vest_groups:
        return "25k", vest_groups["25k"]
    if vest_groups:
        key = sorted(vest_groups.keys())[0]
        return key, vest_groups[key]
    return None


def append_shotdata_for_shot(
    shot_num: int,
    results_dir: Path,
    vest_db: VEST_DB,
    field_id: int = 140,
) -> bool:
    h5_path = results_dir / str(shot_num) / f"{shot_num}.h5"
    if not h5_path.exists():
        print(f"[SKIP] {shot_num}: no h5 file at {h5_path}")
        return False

    vest_groups = vest_db.load_shot(shot=shot_num, fields=[field_id])
    if not vest_groups:
        print(f"[SKIP] {shot_num}: no VEST data for field {field_id}")
        return False

    selection = _select_rate_group(vest_groups)
    if selection is None:
        print(f"[SKIP] {shot_num}: no usable VEST rate group")
        return False

    rate_key, df = selection
    if df is None or df.empty:
        print(f"[SKIP] {shot_num}: empty VEST data for {rate_key}")
        return False

    series_name = df.columns[0]
    data = df[series_name].to_numpy()
    rate_hz = _rate_key_to_hz(rate_key)

    with h5py.File(h5_path, "a") as hf:
        vest_group = hf.require_group("vest_data")
        if series_name in vest_group:
            del vest_group[series_name]
        ds = vest_group.create_dataset(series_name, data=data)
        if rate_hz is not None:
            ds.attrs["sampling_rate_hz"] = rate_hz
        ds.attrs["rate_key"] = rate_key
        ds.attrs["field_id"] = field_id

    print(f"[OK] {shot_num}: appended {series_name} ({rate_key}) to {h5_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append VEST shot data (field) to result HDF5 files."
    )
    parser.add_argument(
        "--shots",
        default="47232:47241",
        help="Shot list, e.g. 47232:47241 or 47232,47234,47240",
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
        "--field-id",
        type=int,
        default=140,
        help="VEST field id for shot data (default: 140)",
    )
    args = parser.parse_args()

    shots = _parse_shots(args.shots)
    results_dir = Path(args.results_dir)

    vest_db = VEST_DB(config_path=args.config)

    total = 0
    ok = 0
    for shot in shots:
        total += 1
        if append_shotdata_for_shot(shot, results_dir, vest_db, field_id=args.field_id):
            ok += 1

    print(f"Done. Updated {ok}/{total} shots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
