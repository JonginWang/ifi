#!/usr/bin/env python3
"""
Summarize longest saved envelope segment per shot into CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ifi.utils.vest_postprocess import FlatShotList


def _resolve_query_text(positional_query: list[str], query_opt: str | None) -> str:
    if query_opt and str(query_opt).strip():
        return str(query_opt).strip()
    return " ".join(str(token).strip() for token in positional_query if str(token).strip()).strip()


def _normalize_query_items(query_text: str) -> list[str]:
    text = str(query_text).strip()
    if not text:
        return []
    if ":" in text and "," not in text and " " not in text:
        return [text]
    return [token for token in text.replace(",", " ").split() if token.strip()]


def _parse_shots(query_text: str) -> list[int]:
    if not query_text:
        return []
    return FlatShotList(_normalize_query_items(query_text)).nums


def _load_longest_duration_us(envelope_dir: Path) -> float:
    json_files = sorted(envelope_dir.glob("*.json"))
    if not json_files:
        return 0.0

    longest = 0.0
    found_segment = False
    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for segment in payload.get("segments", []):
            try:
                duration_us = float(segment.get("duration_us", 0.0))
            except (TypeError, ValueError):
                duration_us = 0.0
            longest = max(longest, duration_us)
            found_segment = True
    return longest if found_segment else 0.0


def summarize_envelope_segments(
    shots: list[int],
    *,
    results_dir: str | Path = "ifi/results",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    results_root = Path(results_dir)

    for shot_num in shots:
        envelope_dir = results_root / str(shot_num) / "envelope"
        if not envelope_dir.exists():
            max_duration = math.nan
        else:
            max_duration = _load_longest_duration_us(envelope_dir)
        rows.append(
            {
                "shot_num": int(shot_num),
                "max_duration_us": max_duration,
            }
        )

    return pd.DataFrame(rows, columns=["shot_num", "max_duration_us"])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the longest saved envelope low-signal segment per shot. "
            "If results/<shot>/envelope is missing, writes NaN. "
            "If folder exists but no segment exists, writes 0."
        )
    )
    parser.add_argument(
        "query",
        nargs="*",
        help='Shot query, e.g. "47807 47808", "47807,47808", or "47807:47840".',
    )
    parser.add_argument(
        "--query",
        dest="query_opt",
        default=None,
        help="Shot query override, same format as positional query.",
    )
    parser.add_argument(
        "--results_dir",
        default="ifi/results",
        help="Base results directory containing <shot>/envelope/*.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <results_dir>/envelope_summary_<first>_<last>.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    query_text = _resolve_query_text(args.query, args.query_opt)
    shots = _parse_shots(query_text)
    if not shots:
        parser.error("No valid shots parsed from query.")

    summary_df = summarize_envelope_segments(
        shots,
        results_dir=args.results_dir,
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.results_dir) / f"envelope_summary_{shots[0]}_{shots[-1]}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved envelope summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
