#!/usr/bin/env python3
"""
CLI wrapper for plotting stacked traces from saved IFI HDF5 results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import replace

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
from ifi.utils.vest_postprocess import FlatShotList


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot stacked traces from saved canonical IFI HDF5 results.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Shot query to load, e.g. '47807', '47807 47808', or '47807:47840'.",
    )
    parser.add_argument(
        "--query",
        dest="query_opt",
        type=str,
        default=None,
        help="Shot query override, same format as positional query.",
    )
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

    shot_query = args.query_opt or args.query
    load_results_dir = str(args.results_dir)
    if config is not None:
        plot_sections = [
            name for name in config.sections()
            if name.lower().endswith(".plot") and "shot_num" in config[name]
        ]
        if plot_sections:
            load_results_dir = config[plot_sections[0]].get("results_dir", load_results_dir)

    shot_numbers: list[int]
    if shot_query:
        shot_numbers = FlatShotList([shot_query]).nums
    elif config is not None:
        plot_sections = [
            name for name in config.sections()
            if name.lower().endswith(".plot") and "shot_num" in config[name]
        ]
        shot_numbers = []
        if plot_sections:
            shot_numbers = [int(config[plot_sections[0]].get("shot_num"))]
    else:
        shot_numbers = []

    if not shot_numbers:
        parser.error(
            "A shot query is required, either positionally/--query or via [fig*.plot] shot_num in --config."
        )

    if args.list and config is None and len(shot_numbers) == 1:
        results = load_results_from_hdf5(int(shot_numbers[0]), base_dir=load_results_dir)
        if not results:
            print(f"Failed to load results for shot {shot_numbers[0]}.")
            return 1
        print_available_series(results)
        return 0

    exit_code = 0
    for idx, shot_num in enumerate(shot_numbers):
        results = load_results_from_hdf5(int(shot_num), base_dir=load_results_dir)
        if not results:
            print(f"Failed to load results for shot {shot_num}.")
            exit_code = 1
            continue

        if args.list:
            print(f"=== Shot {shot_num} ===")
            print_available_series(results)
            continue

        args_for_shot = replace(args) if hasattr(args, "__dataclass_fields__") else argparse.Namespace(**vars(args))
        args_for_shot.shot_num = int(shot_num)
        args_for_shot.query = str(shot_num)
        args_for_shot.query_opt = None
        figure_requests = load_figure_requests(args_for_shot, config, results=results)

        if len(shot_numbers) > 1:
            adjusted_requests = []
            for request in figure_requests:
                save_path = request.save
                if save_path:
                    save_obj = Path(save_path)
                    stem = save_obj.stem
                    if "{shot_num}" in save_path:
                        save_path = save_path.replace("{shot_num}", str(shot_num))
                    else:
                        save_path = str(save_obj.with_name(f"{stem}_{shot_num}{save_obj.suffix}"))
                adjusted_requests.append(
                    replace(
                        request,
                        shot_num=int(shot_num),
                        save=save_path,
                        title=(request.title or f"Shot {shot_num}") if request.title is None else f"{request.title} ({shot_num})",
                    )
                )
            figure_requests = adjusted_requests

        result_code = run_saved_results_plot(
            figure_requests=figure_requests,
            results=results,
            shot_num=int(shot_num),
            results_dir=load_results_dir,
        )
        exit_code = max(exit_code, result_code)
        if len(shot_numbers) > 1 and idx < len(shot_numbers) - 1:
            print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
