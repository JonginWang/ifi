#!/usr/bin/env python3
"""
Saved-results stack plotting helpers.

Provides figure-layout construction and rendering for canonical IFI results HDF5
files, including optional envelope overlay and saved low-envelope interval shading.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.dsp_amplitude import compute_signal_envelope
from ..utils.if_utils import map_frequency_to_group, parse_frequency_group_from_signal_name
from ..utils.io_process_read import load_results_from_hdf5
from .plot_density import resolve_density_time_data
from .plot_waveform import load_envelope_payload, shade_envelope_segments
from .style import FontStyle


@dataclass
class SavedSeriesRequest:
    group: str
    source: str
    column: str
    label: str


@dataclass
class SavedAxisRequest:
    title: str | None = None
    series: list[SavedSeriesRequest] = field(default_factory=list)
    plot_envelope: bool = False
    shade_envelope: bool = False
    allow_missing: bool = True


@dataclass
class SavedFigureRequest:
    shot_num: int
    results_dir: str
    title: str | None
    time_unit: str
    trigger_time: float
    save: str | None
    show: bool
    list_only: bool
    axes: list[SavedAxisRequest]


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
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


def parse_float_list(text: str | None) -> list[float]:
    if not text:
        return []
    cleaned = str(text).replace(",", " ")
    return [float(token) for token in cleaned.split() if token.strip()]


def load_config_defaults(config_path: str | None) -> tuple[dict[str, object], configparser.ConfigParser | None]:
    if not config_path:
        return {}, None

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
            defaults["show"] = parse_bool(section.get("show"), default=False)
        if "list" in section:
            defaults["list"] = parse_bool(section.get("list"), default=False)

    ax_sections = [name for name in parser.sections() if name.lower().startswith("ax")]
    if ax_sections:
        def _ax_sort_key(name: str) -> tuple[int, str]:
            suffix = name[2:]
            return (int(suffix) if suffix.isdigit() else 10**9, name)

        defaults["ax"] = [
            parser[name]["series"]
            for name in sorted(ax_sections, key=_ax_sort_key)
            if "series" in parser[name]
        ]

    return defaults, parser


def time_scale(time_unit: str) -> tuple[float, str]:
    if time_unit == "us":
        return 1e6, "Time [us]"
    if time_unit == "ms":
        return 1e3, "Time [ms]"
    return 1.0, "Time [s]"


def describe_mapping(name: str, mapping: dict[str, pd.DataFrame]) -> list[str]:
    lines = [f"[{name}]"]
    if not mapping:
        lines.append("  <empty>")
        return lines
    for key, df in mapping.items():
        cols = ", ".join(map(str, df.columns))
        attrs = []
        for attr_name in ("freq", "meas_name", "n_ch", "n_path"):
            if attr_name in getattr(df, "attrs", {}):
                attrs.append(f"{attr_name}={df.attrs[attr_name]}")
        attr_text = f" ({', '.join(attrs)})" if attrs else ""
        lines.append(f"  {key}{attr_text}")
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
            for line in describe_mapping(group_name, payload):
                print(line)
        else:
            print(f"[{group_name}]")
            print("  <empty>")
        print()


def parse_series_spec(spec: str) -> list[SavedSeriesRequest]:
    requests: list[SavedSeriesRequest] = []
    for token in [part.strip() for part in str(spec).split(";") if part.strip()]:
        body, _, label = token.partition("@")
        parts = body.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid series spec '{token}'. Expected group:source:column[@label]."
            )
        group, source, column = parts
        requests.append(
            SavedSeriesRequest(
                group=group.strip().lower(),
                source=source.strip(),
                column=column.strip(),
                label=label.strip() or column.strip(),
            )
        )
    return requests


def resolve_payload(results: dict, group: str) -> dict[str, pd.DataFrame]:
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


def normalize_freq_ghz(value: Any) -> float | None:
    if value is None:
        return None
    try:
        freq = float(value)
    except (TypeError, ValueError):
        return None
    if freq > 1.0e6:
        freq /= 1.0e9
    return map_frequency_to_group(freq)


def raw_sources_by_frequency(results: dict) -> dict[float, list[tuple[str, pd.DataFrame]]]:
    grouped: dict[float, list[tuple[str, pd.DataFrame]]] = {}
    for source_name, df in resolve_payload(results, "raw").items():
        freq = normalize_freq_ghz(getattr(df, "attrs", {}).get("freq"))
        if freq is None:
            freq = parse_frequency_group_from_signal_name(str(source_name))
        if freq is None:
            continue
        grouped.setdefault(freq, []).append((str(source_name), df))
    for freq in grouped:
        grouped[freq] = sorted(grouped[freq], key=lambda item: item[0])
    return grouped


def _normalize_label(text: str) -> str:
    normalized = re.sub(r"[\[\]\(\)\{\}_/\\]+", " ", str(text).lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def numeric_signal_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if str(col).upper() == "TIME":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            columns.append(str(col))
    return columns


def choose_reference_column(df: pd.DataFrame, preferred: str = "CH0") -> str | None:
    columns = numeric_signal_columns(df)
    if preferred in columns:
        return preferred
    for candidate in ("REF", "I", "CH0"):
        if candidate in columns:
            return candidate
    return columns[0] if columns else None


def choose_representative_source(
    sources: list[tuple[str, pd.DataFrame]],
    freq_ghz: float,
) -> tuple[str, pd.DataFrame] | None:
    if not sources:
        return None

    def _source_priority(item: tuple[str, pd.DataFrame]) -> tuple[int, int, str]:
        source_name, df = item
        stem = Path(source_name).stem.upper()
        has_ch0 = 0 if "CH0" in numeric_signal_columns(df) else 1

        if map_frequency_to_group(freq_ghz) == 280.0:
            if "_ALL" in stem:
                return (0, has_ch0, stem)
            return (10, has_ch0, stem)

        if map_frequency_to_group(freq_ghz) == 94.0:
            if stem.endswith("_056") or re.search(r"_056(?:_|$)", stem):
                return (0, has_ch0, stem)
            if stem.endswith("_789") or re.search(r"_789(?:_|$)", stem):
                return (1, has_ch0, stem)
            if re.search(r"_\d{3}(?:_|$)", stem):
                return (2, has_ch0, stem)
            return (10, has_ch0, stem)

        return (5, has_ch0, stem)

    return sorted(sources, key=_source_priority)[0]


def _find_suffix_source(
    sources: list[tuple[str, pd.DataFrame]],
    suffix: str,
) -> tuple[str, pd.DataFrame] | None:
    suffix_upper = suffix.upper()
    for source_name, df in sources:
        stem = Path(source_name).stem.upper()
        if stem.endswith(suffix_upper) or re.search(rf"{re.escape(suffix_upper)}(?:_|$)", stem):
            return source_name, df
    return None


def _find_94_source_pair(
    sources: list[tuple[str, pd.DataFrame]],
) -> tuple[tuple[str, pd.DataFrame] | None, tuple[str, pd.DataFrame] | None]:
    src_056 = _find_suffix_source(sources, "_056")
    src_789 = _find_suffix_source(sources, "_789")
    return src_056, src_789


def best_vest_series(results: dict) -> SavedSeriesRequest | None:
    vest_payload = resolve_payload(results, "vest")
    preferred_patterns = (
        ("ip ka", "Ip [kA]"),
        ("ip raw v", "Ip"),
        ("ip", "Ip"),
    )
    for pattern, label in preferred_patterns:
        for source_name, df in vest_payload.items():
            for column_name in df.columns:
                if pattern in _normalize_label(str(column_name)):
                    return SavedSeriesRequest(
                        group="vest",
                        source=str(source_name),
                        column=str(column_name),
                        label=label,
                    )
    for source_name, df in vest_payload.items():
        for col in df.columns:
            if "ip" in str(col).lower():
                return SavedSeriesRequest(
                    group="vest",
                    source=str(source_name),
                    column=str(col),
                    label="Ip",
                )
    return None


def best_density_series(results: dict, freq_ghz: float) -> list[SavedSeriesRequest]:
    density_payload = resolve_payload(results, "density")
    group_name = f"freq_{int(map_frequency_to_group(freq_ghz)):d}G"
    df = density_payload.get(group_name)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return [
        SavedSeriesRequest(
            group="density",
            source=group_name,
            column=str(col),
            label=str(col),
        )
        for col in df.columns
        if str(col).startswith("ne_")
    ]


def best_raw_axis_series(results: dict, freq_ghz: float) -> list[SavedSeriesRequest]:
    raw_by_freq = raw_sources_by_frequency(results)
    sources = raw_by_freq.get(map_frequency_to_group(freq_ghz), [])
    if map_frequency_to_group(freq_ghz) == 94.0:
        src_056, src_789 = _find_94_source_pair(sources)
        selected_sources = [item for item in (src_056, src_789) if item is not None]
        if not selected_sources:
            selection = choose_representative_source(sources, freq_ghz)
            selected_sources = [selection] if selection is not None else []
    else:
        selection = choose_representative_source(sources, freq_ghz)
        selected_sources = [selection] if selection is not None else []

    requests: list[SavedSeriesRequest] = []
    for source_name, df in selected_sources:
        for col in numeric_signal_columns(df):
            requests.append(
                SavedSeriesRequest(
                    group="raw",
                    source=source_name,
                    column=str(col),
                    label=f"{Path(source_name).stem}:{col}",
                )
            )
    return requests


def raw_channel_axes(
    results: dict,
    freq_ghz: float,
    *,
    preferred_ref: str = "CH0",
) -> list[SavedAxisRequest]:
    raw_by_freq = raw_sources_by_frequency(results)
    sources = raw_by_freq.get(map_frequency_to_group(freq_ghz), [])
    if not sources:
        return [SavedAxisRequest(title=f"Rawdata ({freq_ghz:g} GHz) | None", allow_missing=True)]

    if map_frequency_to_group(freq_ghz) == 94.0:
        src_056, src_789 = _find_94_source_pair(sources)
        if src_056 is None and src_789 is None:
            selection = choose_representative_source(sources, freq_ghz)
            src_056 = selection

        axes: list[SavedAxisRequest] = []
        ref_request: SavedSeriesRequest | None = None
        if src_056 is not None:
            src056_name, src056_df = src_056
            ref_col = choose_reference_column(src056_df, preferred=preferred_ref)
            if ref_col is not None:
                ref_request = SavedSeriesRequest(
                    group="raw",
                    source=src056_name,
                    column=ref_col,
                    label=f"{Path(src056_name).stem}:{ref_col}",
                )
                axes.append(
                    SavedAxisRequest(
                        title="Rawdata (94G, ref)",
                        series=[ref_request],
                        plot_envelope=True,
                        shade_envelope=True,
                    )
                )
            for probe_col in numeric_signal_columns(src056_df):
                if probe_col == (ref_request.column if ref_request is not None else None):
                    continue
                axes.append(
                    SavedAxisRequest(
                        title=f"Rawdata (94G, {probe_col})",
                        series=[
                            SavedSeriesRequest(
                                group="raw",
                                source=src056_name,
                                column=probe_col,
                                label=f"{Path(src056_name).stem}:{probe_col}",
                            )
                        ],
                        plot_envelope=True,
                        shade_envelope=True,
                    )
                )

        if src_789 is not None:
            src789_name, src789_df = src_789
            for probe_col in numeric_signal_columns(src789_df):
                series: list[SavedSeriesRequest] = []
                if ref_request is not None:
                    series.append(ref_request)
                series.append(
                    SavedSeriesRequest(
                        group="raw",
                        source=src789_name,
                        column=probe_col,
                        label=f"{Path(src789_name).stem}:{probe_col}",
                    )
                )
                axes.append(
                    SavedAxisRequest(
                        title=f"Rawdata (94G, {probe_col} from {Path(src789_name).stem})",
                        series=series,
                        plot_envelope=True,
                        shade_envelope=True,
                    )
                )

        return axes or [SavedAxisRequest(title="Rawdata (94G) | None", allow_missing=True)]

    selection = choose_representative_source(sources, freq_ghz)
    if selection is None:
        return [SavedAxisRequest(title=f"Rawdata ({freq_ghz:g} GHz) | None", allow_missing=True)]

    source_name, df = selection
    columns = numeric_signal_columns(df)
    if not columns:
        return [SavedAxisRequest(title=f"Rawdata ({freq_ghz:g} GHz) | None", allow_missing=True)]

    ref_col = choose_reference_column(df, preferred=preferred_ref)
    probe_cols = [col for col in columns if col != ref_col]
    axes: list[SavedAxisRequest] = []

    if ref_col is not None:
        axes.append(
            SavedAxisRequest(
                title=f"Rawdata ({freq_ghz:g}G, ref)",
                series=[
                    SavedSeriesRequest(
                        group="raw",
                        source=source_name,
                        column=ref_col,
                        label=f"{Path(source_name).stem}:{ref_col}",
                    )
                ],
                plot_envelope=True,
                shade_envelope=True,
            )
        )

    for probe_col in probe_cols:
        axes.append(
            SavedAxisRequest(
                title=f"Rawdata ({freq_ghz:g}G, {probe_col})",
                series=[
                    SavedSeriesRequest(
                        group="raw",
                        source=source_name,
                        column=probe_col,
                        label=f"{Path(source_name).stem}:{probe_col}",
                    )
                ],
                plot_envelope=True,
                shade_envelope=True,
            )
        )
    return axes


def build_layout_figure(
    fig_name: str,
    section: configparser.SectionProxy,
    results: dict,
) -> SavedFigureRequest:
    shot_num = int(section.get("shot_num"))
    results_dir = section.get("results_dir", "ifi/results")
    time_unit = section.get("time_unit", "ms")
    trigger_time = float(section.get("trigger_time", 0.0))
    title = section.get("title", fig_name)
    save = section.get("save", fallback=None)
    show = parse_bool(section.get("show"), default=False)
    list_only = parse_bool(section.get("list"), default=False)
    freqs = parse_float_list(section.get("freqs", "94 280"))
    layout = section.get("layout", "").strip().lower()
    raw_plot_envelope = parse_bool(section.get("raw_plot_envelope"), default=False)
    raw_shade_envelope = parse_bool(section.get("raw_shade_envelope"), default=False)
    preferred_ref = section.get("ref_channel", "CH0")

    axes: list[SavedAxisRequest] = []
    if layout == "overview_by_frequency":
        ip_request = best_vest_series(results)
        axes.append(
            SavedAxisRequest(
                title="Ip [kA]",
                series=[ip_request] if ip_request is not None else [],
                allow_missing=True,
            )
        )
        for freq in freqs:
            axes.append(
                SavedAxisRequest(
                    title=f"rawdata({freq:g}GHz) | None",
                    series=best_raw_axis_series(results, freq),
                    plot_envelope=raw_plot_envelope,
                    shade_envelope=raw_shade_envelope,
                    allow_missing=True,
                )
            )
            axes.append(
                SavedAxisRequest(
                    title=f"density({freq:g}GHz) | None",
                    series=best_density_series(results, freq),
                    allow_missing=True,
                )
            )
    elif layout == "raw_channels_by_frequency":
        for freq in freqs:
            freq_axes = raw_channel_axes(results, freq, preferred_ref=preferred_ref)
            for axis in freq_axes:
                axis.plot_envelope = raw_plot_envelope
                axis.shade_envelope = raw_shade_envelope
            axes.extend(freq_axes)
    else:
        raise ValueError(
            f"Unsupported layout '{layout}' in section [{fig_name}.plot]. "
            "Use 'overview_by_frequency' or 'raw_channels_by_frequency'."
        )

    return SavedFigureRequest(
        shot_num=shot_num,
        results_dir=results_dir,
        title=title,
        time_unit=time_unit,
        trigger_time=trigger_time,
        save=save,
        show=show,
        list_only=list_only,
        axes=axes,
    )


def load_figure_requests(
    args: Any,
    config: configparser.ConfigParser | None,
    results: dict | None = None,
) -> list[SavedFigureRequest]:
    if config is not None:
        fig_sections = sorted(
            name for name in config.sections()
            if name.lower().endswith(".plot") and name.lower().startswith("fig")
        )
        if fig_sections:
            if results is None:
                raise ValueError("results must be loaded before building layout figures.")
            return [
                build_layout_figure(
                    section_name.rsplit(".", 1)[0],
                    config[section_name],
                    results,
                )
                for section_name in fig_sections
            ]

    if args.shot_num is None:
        raise ValueError("shot_num is required when no figure layout is defined.")
    return [
        SavedFigureRequest(
            shot_num=int(args.shot_num),
            results_dir=str(args.results_dir),
            title=args.title or f"Shot {args.shot_num} - Saved Results Stack Plot",
            time_unit=args.time_unit,
            trigger_time=float(args.trigger_time),
            save=args.save,
            show=bool(args.show),
            list_only=bool(args.list),
            axes=[SavedAxisRequest(series=parse_series_spec(spec)) for spec in args.ax],
        )
    ]


def extract_time_and_series(
    results: dict,
    request: SavedSeriesRequest,
    *,
    trigger_time: float,
) -> tuple[pd.Series, pd.Series]:
    payload = resolve_payload(results, request.group)

    if request.source == "auto":
        matched_source = next(
            (source_name for source_name, df in payload.items() if request.column in df.columns),
            None,
        )
        if matched_source is None:
            raise KeyError(
                f"Column '{request.column}' not found in any source for group '{request.group}'."
            )
    else:
        matched_source = request.source

    if matched_source not in payload:
        raise KeyError(
            f"Source '{matched_source}' not found in group '{request.group}'. "
            f"Available: {list(payload.keys())}"
        )

    df = payload[matched_source]
    if request.column not in df.columns:
        raise KeyError(
            f"Column '{request.column}' not found in {request.group}:{matched_source}. "
            f"Available: {list(df.columns)}"
        )

    if request.group == "density":
        time = pd.Series(
            resolve_density_time_data(
                df,
                density_name=matched_source,
                signals_dict=results.get("rawdata", {}),
            ),
            copy=False,
        )
    elif "TIME" in df.columns:
        time = pd.Series(df["TIME"], copy=False)
    else:
        time = pd.Series(df.index, copy=False)
    time_values = pd.to_numeric(time, errors="coerce").to_numpy(dtype=float)
    if request.group != "vest":
        time_values = time_values + float(trigger_time)
    value_values = pd.to_numeric(df[request.column], errors="coerce").to_numpy(dtype=float)
    min_len = min(len(time_values), len(value_values))
    time_values = time_values[:min_len]
    value_values = value_values[:min_len]
    mask = np.isfinite(time_values) & np.isfinite(value_values)
    return pd.Series(time_values[mask]), pd.Series(value_values[mask])


def plot_saved_axis(
    ax: plt.Axes,
    results: dict,
    axis_request: SavedAxisRequest,
    *,
    shot_num: int,
    results_dir: str,
    trigger_time: float,
    time_factor: float,
) -> tuple[float | None, float | None]:
    min_x = None
    max_x = None
    plotted_any = False

    for request in axis_request.series:
        try:
            time, values = extract_time_and_series(
                results,
                request,
                trigger_time=trigger_time,
            )
        except KeyError:
            if axis_request.allow_missing:
                continue
            raise

        if time.empty:
            continue

        x = time.to_numpy(dtype=float) * time_factor
        y = values.to_numpy(dtype=float)
        line = ax.plot(x, y, label=request.label, linewidth=1.2)[0]
        plotted_any = True

        local_min = float(x.min())
        local_max = float(x.max())
        min_x = local_min if min_x is None else min(min_x, local_min)
        max_x = local_max if max_x is None else max(max_x, local_max)

        if request.group == "raw" and (axis_request.plot_envelope or axis_request.shade_envelope):
            payload = load_envelope_payload(
                shot_num=shot_num,
                results_dir=results_dir,
                source_name=request.source,
                column_name=request.column,
            )
            if axis_request.shade_envelope:
                shade_envelope_segments(ax, payload, time_factor=time_factor)
            if axis_request.plot_envelope:
                envelope = compute_signal_envelope(
                    time.to_numpy(dtype=float),
                    values.to_numpy(dtype=float),
                    smooth_window_us=20.0,
                )
                ax.plot(
                    x,
                    envelope,
                    color=line.get_color(),
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.9,
                    label=f"{request.label} envelope",
                )

    ax.grid(True, alpha=0.3)
    if plotted_any:
        ax.legend()
    else:
        title = axis_request.title or "No data"
        ax.text(0.5, 0.5, title, transform=ax.transAxes, ha="center", va="center")
    return min_x, max_x


def render_saved_figure(results: dict, request: SavedFigureRequest) -> None:
    if request.list_only:
        print_available_series(results)
        return

    if not request.axes:
        raise ValueError("No axes configured for figure.")

    time_factor, x_label = time_scale(request.time_unit)
    fig, axes = plt.subplots(
        len(request.axes),
        1,
        sharex=True,
        figsize=(14, 2.8 * len(request.axes)),
    )
    if len(request.axes) == 1:
        axes = [axes]

    global_min = None
    global_max = None
    for ax, axis_request in zip(axes, request.axes):
        local_min, local_max = plot_saved_axis(
            ax,
            results,
            axis_request,
            shot_num=request.shot_num,
            results_dir=request.results_dir,
            trigger_time=request.trigger_time,
            time_factor=time_factor,
        )
        if axis_request.title:
            ax.set_title(axis_request.title, **FontStyle.subtitle)
        if local_min is not None:
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)

    if global_min is not None and global_max is not None:
        for ax in axes:
            ax.set_xlim(global_min, global_max)

    axes[-1].set_xlabel(x_label, **FontStyle.label)
    fig.suptitle(
        request.title or f"Shot {request.shot_num} - Saved Results Stack Plot",
        **FontStyle.suptitle,
    )
    fig.tight_layout()

    if request.save:
        output_path = Path(request.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    if request.show or not request.save:
        plt.show(block=False)
    else:
        plt.close(fig)


def run_saved_results_plot(
    *,
    figure_requests: list[SavedFigureRequest],
    results: dict | None = None,
    shot_num: int | None = None,
    results_dir: str | None = None,
) -> int:
    if results is None:
        if shot_num is None or results_dir is None:
            raise ValueError("shot_num and results_dir are required when results are not provided.")
        results = load_results_from_hdf5(int(shot_num), base_dir=results_dir)
    effective_shot = int(shot_num if shot_num is not None else figure_requests[0].shot_num)
    if not results:
        print(f"Failed to load results for shot {effective_shot}.")
        return 1
    for figure_request in figure_requests:
        render_saved_figure(results, figure_request)
    return 0


__all__ = [
    "SavedAxisRequest",
    "SavedFigureRequest",
    "SavedSeriesRequest",
    "build_layout_figure",
    "describe_mapping",
    "load_config_defaults",
    "load_figure_requests",
    "parse_bool",
    "parse_float_list",
    "parse_series_spec",
    "print_available_series",
    "render_saved_figure",
    "run_saved_results_plot",
]
