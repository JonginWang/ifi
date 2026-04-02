#!/usr/bin/env python3
"""
CLI wrapper for plotting stacked traces from saved IFI HDF5 results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ifi.plots.plot_saved_results import (
    load_config_defaults,
    load_figure_requests,
    print_available_series,
    run_saved_results_plot,
)
from ifi.plots.style import set_plot_style
from ifi.utils.io_process_read import load_results_from_hdf5


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot stacked traces from saved canonical IFI HDF5 results.",
    )
    parser.add_argument("shot_num", nargs="?", type=int, help="Shot number to load.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="INI config file for saved-results plotting.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ifi/results",
        help="Base results directory containing <shot>/<shot>.h5.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available rawdata/density/vestdata keys and columns, then exit.",
    )
    parser.add_argument(
        "--ax",
        action="append",
        default=[],
        help=(
            "One subplot specification. Repeat per axis. "
            "Each axis accepts one or more series separated by ';'. "
            "Series format: group:source:column[@label], "
            "where group is one of raw, density, vest."
        ),
    )
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    parser.add_argument(
        "--time_unit",
        choices=["s", "ms", "us"],
        default="ms",
        help="Displayed x-axis unit.",
    )
    parser.add_argument(
        "--trigger_time",
        type=float,
        default=0.0,
        help="Optional time offset in seconds added before plotting.",
    )
    parser.add_argument("--save", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively.")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(cli_argv)

    defaults, config = load_config_defaults(pre_args.config)
    parser = build_argument_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args(cli_argv)

    set_plot_style()

    shot_num = args.shot_num
    load_results_dir = str(args.results_dir)
    if config is not None:
        plot_sections = [
            name for name in config.sections()
            if name.lower().endswith(".plot") and "shot_num" in config[name]
        ]
        if shot_num is None and plot_sections:
            shot_num = int(config[plot_sections[0]].get("shot_num"))
        if plot_sections:
            load_results_dir = config[plot_sections[0]].get("results_dir", load_results_dir)

    if shot_num is None:
        parser.error("shot_num is required, either positionally or via [plot]/[fig*.plot] shot_num in --config.")

    if args.list and config is None:
        results = load_results_from_hdf5(int(shot_num), base_dir=load_results_dir)
        if not results:
            print(f"Failed to load results for shot {shot_num}.")
            return 1
        print_available_series(results)
        return 0

    results = load_results_from_hdf5(int(shot_num), base_dir=load_results_dir)
    if not results:
        print(f"Failed to load results for shot {shot_num}.")
        return 1

    figure_requests = load_figure_requests(args, config, results=results)
    return run_saved_results_plot(
        figure_requests=figure_requests,
        results=results,
        shot_num=int(shot_num),
        results_dir=load_results_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
