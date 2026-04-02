#!/usr/bin/env python3
"""
Waveform plotting module
=========================

Author: Jongin Wang
Date: 2025-01-16
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..utils.dsp_amplitude import compute_signal_envelope
from ..utils.func_helper import merge_kwargs, normalize_call_args
from .plot_common import apply_scaling, extract_metadata_info, prepare_time_data
from .style import FontStyle


def load_envelope_payload(
    *,
    shot_num: int,
    results_dir: str | Path,
    source_name: str,
    column_name: str,
) -> dict[str, Any] | None:
    """Load saved envelope-segment JSON for one raw signal channel if it exists."""
    json_path = (
        Path(results_dir)
        / str(shot_num)
        / "envelope"
        / f"{Path(source_name).stem}_{column_name}_envelope.json"
    )
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def shade_envelope_segments(
    ax: plt.Axes,
    payload: dict[str, Any] | None,
    *,
    time_factor: float = 1.0,
    base_color: str = "gray",
    base_alpha: float = 0.10,
    highlight_color: str = "gold",
    highlight_alpha: float = 0.18,
) -> None:
    """Shade saved low-envelope intervals and highlight the longest interval."""
    if not isinstance(payload, dict):
        return
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        return

    longest = max(
        segments,
        key=lambda item: float(item.get("duration_us", 0.0)),
        default=None,
    )
    longest_key = (
        float(longest.get("start_time", np.nan)),
        float(longest.get("end_time", np.nan)),
    ) if isinstance(longest, dict) else None

    for segment in segments:
        try:
            start = float(segment["start_time"]) * time_factor
            end = float(segment["end_time"]) * time_factor
        except (KeyError, TypeError, ValueError):
            continue
        key = (
            float(segment.get("start_time", np.nan)),
            float(segment.get("end_time", np.nan)),
        )
        if key == longest_key:
            ax.axvspan(start, end, color=highlight_color, alpha=highlight_alpha, zorder=0)
        else:
            ax.axvspan(start, end, color=base_color, alpha=base_alpha, zorder=0)


def plot_waveforms_core(
    data: pd.DataFrame | dict[str, np.ndarray] | np.ndarray,
    fs: float | None = None,
    title: str = "Waveforms",
    downsample: int = 10,
    save_path: str | None = None,
    show_plot: bool = True,
    time_scale: str = "s",
    signal_scale: str = "V",
    trigger_time: float = 0.0,
    plot_envelope: bool = False,
    subplots_kwargs: dict[str, Any] | None = None,
    plot_args: tuple[Any, ...] | list[Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    envelope_kwargs: dict[str, Any] | None = None,
    envelope_plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Core waveform plotting implementation."""
    time, signals = prepare_time_data(data, fs)
    time_scaled, signals_scaled, time_label, signal_label = apply_scaling(
        time, signals, time_scale, signal_scale
    )
    time_scaled = time_scaled + trigger_time

    if downsample > 1:
        time_scaled = time_scaled[::downsample]
        signals_scaled = {k: v[::downsample] for k, v in signals_scaled.items()}

    n_signals = len(signals_scaled)
    fig, axes = plt.subplots(
        n_signals,
        1,
        **merge_kwargs(
            {"figsize": (12, 2 * n_signals), "sharex": True},
            subplots_kwargs,
        ),
    )
    if n_signals == 1:
        axes = [axes]

    metadata_info = extract_metadata_info(data)
    if metadata_info:
        title = f"{title}\n{metadata_info}"

    line_args = normalize_call_args(plot_args)
    line_kwargs = merge_kwargs(plot_kwargs)
    envelope_options = merge_kwargs({"smooth_window_us": 20.0}, envelope_kwargs)
    envelope_line_kwargs = merge_kwargs(
        {"color": "tab:orange", "linestyle": "--", "linewidth": 1.5, "alpha": 0.9},
        envelope_plot_kwargs,
    )
    grid_options = merge_kwargs({"alpha": 0.3}, grid_kwargs)
    for i, (name, signal) in enumerate(signals_scaled.items()):
        axes[i].plot(time_scaled, signal, *line_args, label=str(name), **line_kwargs)
        if plot_envelope:
            raw_signal = np.asarray(signals[name], dtype=float)
            envelope = compute_signal_envelope(time, raw_signal, **envelope_options)
            _, scaled_envelope_map, _, _ = apply_scaling(
                time,
                {"envelope": envelope},
                time_scale,
                signal_scale,
            )
            scaled_envelope = scaled_envelope_map["envelope"]
            if downsample > 1:
                scaled_envelope = scaled_envelope[::downsample]
            axes[i].plot(
                time_scaled,
                scaled_envelope,
                label=f"{name} envelope",
                **envelope_line_kwargs,
            )
            axes[i].legend()
        axes[i].set_ylabel(f"{name} {signal_label}", **FontStyle.label)
        axes[i].grid(True, **grid_options)

    axes[-1].set_xlabel(time_label, **FontStyle.label)
    fig.suptitle(title, **FontStyle.title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, **merge_kwargs({"dpi": 300, "bbox_inches": "tight"}, savefig_kwargs))
    if show_plot:
        plt.show(block=False)
    return fig, axes


def render_overview_waveforms(
    signals_dict: dict[str, pd.DataFrame],
    title_prefix: str,
    trigger_time: float,
    downsample: int,
    warn_fn: Any,
    **kwargs: Any,
) -> None:
    """Render waveform section for analysis overview."""
    for name, df in signals_dict.items():
        if df is None or df.empty:
            continue
        try:
            plot_waveforms_core(
                df,
                title=f"{title_prefix}{name}",
                trigger_time=trigger_time,
                downsample=downsample,
                show_plot=True,
                **kwargs,
            )
        except Exception as e:
            warn_fn(name, e)


def render_signal_dict(data_dict: dict[str, pd.DataFrame], **kwargs: Any) -> None:
    """Render all signals from a name->DataFrame mapping."""
    for name, df in data_dict.items():
        plot_waveforms_core(df, title=name, **kwargs)


__all__ = [
    "load_envelope_payload",
    "plot_waveforms_core",
    "render_overview_waveforms",
    "render_signal_dict",
    "shade_envelope_segments",
]
