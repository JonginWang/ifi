#!/usr/bin/env python3
"""
Plot stacked traces from saved IFI HDF5 results.

Examples:
    python scripts/plot_saved_results.py 47807 --list
    python scripts/plot_saved_results.py 47807 ^
        --ax "density:freq_94G:ne_056;density:freq_280G:ne_ALL" ^
        --ax "vest:10000:Ip_raw ([V]);vest:10000:TF current" ^
        --save ifi/results/47807/stacked_plot.png
"""

from __future__ import annotations

import argparse
import configparser
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ifi.plots.style import FontStyle, set_plot_style
from ifi.utils.io_process_read import load_results_from_hdf5


@dataclass
class SeriesRequest:
    group: str
    source: str
    column: str
    label: str


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
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
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
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively.",
    )
    return parser


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_config_defaults(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}

    parser = configparser.ConfigParser()
    loaded = parser.read(config_path, encoding="utf-8")
    if not loaded:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    defaults: dict[str, object] = {}
    if parser.has_section("plot"):
        section = parser["plot"]
        if "shot_num" in section:
            defaults["shot_num"] = int(section["shot_num"])
        if "results_dir" in section:
            defaults["results_dir"] = section["results_dir"]
        if "title" in section:
            defaults["title"] = section["title"]
        if "time_unit" in section:
            defaults["time_unit"] = section["time_unit"]
        if "trigger_time" in section:
            defaults["trigger_time"] = float(section["trigger_time"])
        if "save" in section:
            defaults["save"] = section["save"]
        if "show" in section:
            defaults["show"] = _parse_bool(section.get("show"), default=False)
        if "list" in section:
            defaults["list"] = _parse_bool(section.get("list"), default=False)

    ax_sections = [
        name
        for name in parser.sections()
        if name.lower().startswith("ax")
    ]
    if ax_sections:
        def _ax_sort_key(name: str) -> tuple[int, str]:
            suffix = name[2:]
            return (int(suffix) if suffix.isdigit() else 10**9, name)

        defaults["ax"] = [
            parser[name]["series"]
            for name in sorted(ax_sections, key=_ax_sort_key)
            if "series" in parser[name]
        ]

    return defaults


def _time_scale(time_unit: str) -> tuple[float, str]:
    if time_unit == "us":
        return 1e6, "Time [us]"
    if time_unit == "ms":
        return 1e3, "Time [ms]"
    return 1.0, "Time [s]"


def _describe_mapping(name: str, mapping: dict[str, pd.DataFrame]) -> list[str]:
    lines = [f"[{name}]"]
    if not mapping:
        lines.append("  <empty>")
        return lines
    for key, df in mapping.items():
        cols = ", ".join(map(str, df.columns))
        lines.append(f"  {key}")
        lines.append(f"    cols: {cols}")
    return lines


def print_available_series(results: dict) -> None:
    print(f"Shot metadata keys: {sorted(results.get('metadata', {}).keys())}")
    print()
    for group_name, key in (
        ("raw", "rawdata"),
        ("density", "density"),
        ("vest", "vestdata"),
    ):
        payload = results.get(key, {})
        if isinstance(payload, dict):
            for line in _describe_mapping(group_name, payload):
                print(line)
        else:
            print(f"[{group_name}]")
            print("  <empty>")
        print()


def parse_series_spec(spec: str) -> list[SeriesRequest]:
    requests: list[SeriesRequest] = []
    for token in [part.strip() for part in str(spec).split(";") if part.strip()]:
        body, _, label = token.partition("@")
        parts = body.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid series spec '{token}'. Expected group:source:column[@label]."
            )
        group, source, column = parts
        requests.append(
            SeriesRequest(
                group=group.strip().lower(),
                source=source.strip(),
                column=column.strip(),
                label=label.strip() or column.strip(),
            )
        )
    return requests


def _resolve_payload(results: dict, group: str) -> dict[str, pd.DataFrame]:
    key_map = {
        "raw": "rawdata",
        "rawdata": "rawdata",
        "density": "density",
        "vest": "vestdata",
        "vestdata": "vestdata",
    }
    result_key = key_map.get(group)
    if result_key is None:
        raise KeyError(f"Unsupported group '{group}'. Use raw, density, or vest.")
    payload = results.get(result_key, {})
    if not isinstance(payload, dict):
        raise KeyError(f"No mapping payload found for group '{group}'.")
    return payload


def _extract_time_and_series(
    results: dict,
    request: SeriesRequest,
    *,
    trigger_time: float,
) -> tuple[pd.Series, pd.Series]:
    payload = _resolve_payload(results, request.group)
    if request.source not in payload:
        raise KeyError(
            f"Source '{request.source}' not found in group '{request.group}'. "
            f"Available: {list(payload.keys())}"
        )

    df = payload[request.source]
    if request.column not in df.columns:
        raise KeyError(
            f"Column '{request.column}' not found in {request.group}:{request.source}. "
            f"Available: {list(df.columns)}"
        )

    if "TIME" in df.columns:
        time = pd.Series(df["TIME"], copy=False)
    else:
        time = pd.Series(df.index, copy=False)
    time = pd.to_numeric(time, errors="coerce") + float(trigger_time)
    values = pd.to_numeric(df[request.column], errors="coerce")
    mask = time.notna() & values.notna()
    return time[mask], values[mask]


def _plot_axis(
    ax: plt.Axes,
    results: dict,
    requests: Iterable[SeriesRequest],
    *,
    trigger_time: float,
    time_factor: float,
) -> tuple[float | None, float | None]:
    min_x = None
    max_x = None
    for request in requests:
        time, values = _extract_time_and_series(
            results,
            request,
            trigger_time=trigger_time,
        )
        if time.empty:
            continue
        x = time.to_numpy(dtype=float) * time_factor
        y = values.to_numpy(dtype=float)
        ax.plot(x, y, label=request.label, linewidth=1.2)
        local_min = float(x.min())
        local_max = float(x.max())
        min_x = local_min if min_x is None else min(min_x, local_min)
        max_x = local_max if max_x is None else max(max_x, local_max)

    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    return min_x, max_x


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(cli_argv)

    defaults = _load_config_defaults(pre_args.config)
    parser = build_argument_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args(cli_argv)
    if args.shot_num is None:
        parser.error("shot_num is required, either positionally or via [plot] shot_num in --config.")

    set_plot_style()

    results = load_results_from_hdf5(args.shot_num, base_dir=args.results_dir)
    if not results:
        print(f"Failed to load results for shot {args.shot_num}.")
        return 1

    if args.list:
        print_available_series(results)
        return 0

    if not args.ax:
        parser.error("Specify at least one --ax panel, or use --list.")

    axis_specs = [parse_series_spec(spec) for spec in args.ax]
    time_factor, x_label = _time_scale(args.time_unit)

    fig, axes = plt.subplots(len(axis_specs), 1, sharex=True, figsize=(12, 3.2 * len(axis_specs)))
    if len(axis_specs) == 1:
        axes = [axes]

    global_min = None
    global_max = None

    for idx, (ax, requests) in enumerate(zip(axes, axis_specs), start=1):
        local_min, local_max = _plot_axis(
            ax,
            results,
            requests,
            trigger_time=args.trigger_time,
            time_factor=time_factor,
        )
        if local_min is not None:
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)
        ax.set_ylabel(f"ax[{idx - 1}]", **FontStyle.label)

    if global_min is not None and global_max is not None:
        for ax in axes:
            ax.set_xlim(global_min, global_max)

    axes[-1].set_xlabel(x_label, **FontStyle.label)
    fig.suptitle(args.title or f"Shot {args.shot_num} - Saved Results Stack Plot", **FontStyle.suptitle)
    fig.tight_layout()

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    if args.show or not args.save:
        plt.show(block=False)
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
