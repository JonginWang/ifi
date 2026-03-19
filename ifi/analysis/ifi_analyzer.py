#!/usr/bin/env python3
"""Interactive-friendly wrapper around the main IFI analysis CLI."""

from __future__ import annotations

import logging
import re
import sys
from argparse import Namespace
from typing import Any

import matplotlib.pyplot as plt

from ..db_controller.nas_db import NasDB
from ..db_controller.vest_db import VestDB
from ..utils.log_manager import LogManager
from .main_analysis import build_argument_parser, main as main_analysis_main, run_analysis

LogManager(level="DEBUG")

_MULTI_VALUE_OPTIONS = {
    "--freq": "--freq",
    "--stft-cols": "--stft_cols",
    "--stft_cols": "--stft_cols",
    "--cwt-cols": "--cwt_cols",
    "--cwt_cols": "--cwt_cols",
    "--vest-fields": "--vest_fields",
    "--vest_fields": "--vest_fields",
}


def create_mock_args(**overrides: Any) -> Namespace:
    """Create a default analysis namespace for notebooks and IDE inspection."""
    args = Namespace(
        query=["45821"],
        freq=[94.0, 280.0],
        density=True,
        plot=False,
        overview_plot=False,
        data_folders=None,
        add_path=False,
        force_remote=False,
        vest_fields=[109, 101],
        no_offset_removal=True,
        offset_window=2001,
        baseline="ip",
        plot_raw=False,
        no_plot_block=True,
        no_plot_raw=False,
        no_plot_ft=False,
        trigger_time=0.290,
        downsample=10,
        color_density_by_amplitude=False,
        amplitude_colormap="coolwarm",
        amplitude_impedance=50.0,
        stft=True,
        stft_cols=[0, 1],
        cwt=False,
        cwt_cols=[0, 1],
        results_dir="ifi/results",
        save_plots=False,
        save_data=True,
        scheduler="threads",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_with_args(args: Namespace) -> dict | None:
    """Run analysis using a prepared namespace object."""
    try:
        nas_db = NasDB(config_path="ifi/config.ini")
        vest_db = VestDB(config_path="ifi/config.ini")
        logging.info("Database controllers initialized successfully.")
    except FileNotFoundError:
        logging.error("Configuration file 'ifi/config.ini' not found. Exiting.")
        return None
    except Exception as exc:
        logging.error(f"Failed to initialize database controllers: {exc}")
        return None

    logging.info("Starting IFI analyzer run...")
    try:
        results = run_analysis(
            query=args.query,
            args=args,
            nas_db=nas_db,
            vest_db=vest_db,
        )
    finally:
        nas_db.disconnect()
        vest_db.disconnect()

    if results and (getattr(args, "plot", False) or getattr(args, "overview_plot", False)):
        logging.info("Displaying plots. Close plot windows to end script.")
        plt.show()
    return results


def _split_cli_values(raw: str) -> list[str]:
    return [token for token in re.split(r"[\s,]+", str(raw).strip()) if token]


def normalize_cli_argv(argv: list[str]) -> list[str]:
    """Normalize PowerShell-friendly quoted args into argparse-friendly tokens."""
    normalized: list[str] = []
    i = 0

    while i < len(argv):
        token = argv[i]

        if token == "--query":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                normalized.extend(_split_cli_values(argv[i]))
                i += 1
            continue

        canonical = _MULTI_VALUE_OPTIONS.get(token)
        if canonical is not None:
            normalized.append(canonical)
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                normalized.extend(_split_cli_values(argv[i]))
                i += 1
            continue

        if not token.startswith("--"):
            normalized.extend(_split_cli_values(token))
            i += 1
            continue

        normalized.append(token)
        i += 1

    return normalized


def main(argv: list[str] | None = None) -> dict | None:
    """Run main_analysis directly when CLI args exist, else use interactive defaults."""
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if not cli_args:
        return run_with_args(create_mock_args())

    normalized_argv = normalize_cli_argv(cli_args)
    parser = build_argument_parser()
    parser.parse_args(normalized_argv)
    return main_analysis_main(normalized_argv)


if __name__ == "__main__":
    main()
