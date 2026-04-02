#!/usr/bin/env python3
"""
Run local VEST shot monitoring post-processing and plotting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ifi.db_controller.vest_db import VestDB
from ifi.utils.log_manager import LogManager
from ifi.utils.vest_monitoring import run_vest_shot_monitoring
from ifi.utils.vest_postprocess import FlatShotList

logger = LogManager().get_logger(__name__, level="INFO")


def _parse_float_pair(text: str) -> tuple[float, float]:
    tokens = str(text).replace(",", " ").split()
    if len(tokens) != 2:
        raise ValueError("Expected two float values.")
    return float(tokens[0]), float(tokens[1])


def _parse_int_list(text: str) -> list[int]:
    return [int(tok) for tok in str(text).replace(",", " ").split() if tok]


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
    items: list[str] = []
    for chunk in text.replace(",", " ").split():
        token = chunk.strip()
        if token:
            items.append(token)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Translate VEST MATLAB shot monitoring into local HDF5 postprocess "
            "and monitoring plots. Writes only to results/<shot>/<shot>.h5. "
            "Default plot mode compares all requested shots and saves under "
            "results/<last-shot>/monitoring/*.png. Use --plot_each to save "
            "per-shot monitoring plots under each results/<shot>/monitoring."
        )
    )
    parser.add_argument(
        "query",
        nargs="*",
        help='Shot query, e.g. "47805 47807 47808" or "47807:47840"',
    )
    parser.add_argument(
        "--query",
        dest="query_opt",
        default=None,
        help="Shot query override, same format as positional query.",
    )
    parser.add_argument("--config", default="ifi/config.ini")
    parser.add_argument("--results_dir", default="ifi/results")
    parser.add_argument("--xrange", default="0.28 0.35")
    parser.add_argument("--xcoil", default="1 5 6 10")
    parser.add_argument("--gas", default="H2", choices=["H2", "He"])
    parser.add_argument("--overwrite_local", action="store_true")
    parser.add_argument("--no_save_plots", action="store_true")
    parser.add_argument(
        "--plot_each",
        action="store_true",
        help=(
            "Plot each shot independently instead of creating one multi-shot "
            "comparison set. Saves each monitoring result under results/<shot>/monitoring "
            "and shows figures shot-by-shot."
        ),
    )
    parser.add_argument(
        "--auto_close_sec",
        type=float,
        default=None,
        help=(
            "When used with --plot_each, show each figure non-blocking and "
            "close it automatically after the given seconds."
        ),
    )
    args = parser.parse_args()

    query_text = _resolve_query_text(args.query, args.query_opt)
    shots = FlatShotList(_normalize_query_items(query_text)).nums if query_text else []
    if not shots:
        raise SystemExit("No valid shots parsed from query.")

    vest_db = VestDB(config_path=args.config)
    try:
        payload = run_vest_shot_monitoring(
            vest_db=vest_db,
            shots=shots,
            results_dir=args.results_dir,
            xcoil=_parse_int_list(args.xcoil),
            gas=args.gas,
            xrange_s=_parse_float_pair(args.xrange),
            overwrite_local=args.overwrite_local,
            save_plots=not args.no_save_plots,
            plot_each=args.plot_each,
            auto_close_sec=args.auto_close_sec,
        )
    finally:
        vest_db.disconnect()

    for shot_num, written in payload["written"].items():
        logger.info("shot=%s written=%s", shot_num, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
