#!/usr/bin/env python3
"""
Density plotting module
========================
Author: Jongin Wang
Date: 2025-01-16
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..analysis.functions.power_conversion import amp2db, mag2db
from ..utils.dsp_amplitude import extract_probe_amplitudes_from_signals
from ..utils.if_utils import map_frequency_to_group, parse_frequency_group_from_signal_name
from ..utils.func_helper import merge_kwargs, normalize_call_args
from ..utils.io_process_common import parse_density_group_name
from .plot_common import apply_scaling, colored_line
from .style import FontStyle


def _convert_amplitude_values(
    amplitude: np.ndarray,
    *,
    amplitude_dbm: bool,
    amplitude_unit: str,
    amplitude_impedance: float,
) -> tuple[np.ndarray, str]:
    """Convert raw probe amplitude into colormap values and display unit."""
    amp = np.abs(np.asarray(amplitude, dtype=float))
    amp = np.where(np.isfinite(amp), amp, np.nan)
    unit = str(amplitude_unit).strip()
    scale = 1.0
    if unit == "mV":
        scale = 1e-3
    elif unit == "uV":
        scale = 1e-6
    amplitude_volts = np.maximum(amp * scale, np.finfo(float).tiny)

    if amplitude_dbm:
        return amp2db(amplitude_volts, impedance=float(amplitude_impedance), dbm=True), "dBm"

    if unit == "dB":
        return mag2db(np.maximum(amp, np.finfo(float).tiny)), "dB"

    return amp, unit or "V"


def plot_density_core(
    density_data: pd.DataFrame | dict[str, np.ndarray] | np.ndarray,
    time_data: np.ndarray | None = None,
    title: str = "Density Results",
    downsample: int = 1,
    save_path: str | None = None,
    show_plot: bool = True,
    density_scale: str = "10^18 m^-3",
    time_scale: str = "ms",
    trigger_time: float = 0.0,
    probe_amplitude: np.ndarray | dict[str, np.ndarray] | None = None,
    color_by_amplitude: bool = False,
    amplitude_colormap: str = "coolwarm",
    amplitude_impedance: float = 50,
    amplitude_dbm: bool = True,
    amplitude_unit: str = "mV",
    subplots_kwargs: dict[str, Any] | None = None,
    plot_args: tuple[Any, ...] | list[Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    line_collection_kwargs: dict[str, Any] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Core density plotting implementation."""
    downsample = max(1, int(downsample))
    if isinstance(density_data, pd.DataFrame):
        if time_data is None and hasattr(density_data, "index"):
            time_data = density_data.index.values
        elif time_data is None:
            time_data = np.arange(len(density_data))
        signals = {col: density_data[col].values for col in density_data.columns}
    elif isinstance(density_data, dict):
        if time_data is None:
            max_len = max(len(data) for data in density_data.values())
            time_data = np.arange(max_len)
        signals = density_data
    else:
        if time_data is None:
            time_data = np.arange(len(density_data))
        signals = {"Density": np.asarray(density_data)}

    time_data = np.asarray(time_data, dtype=float) + float(trigger_time)
    time_scaled, _, time_label, _ = apply_scaling(
        time_data, {"dummy": np.zeros_like(time_data)}, time_scale
    )
    _, density_scaled, _, density_label = apply_scaling(
        time_data, signals, time_scale, density_scale
    )
    if downsample > 1:
        time_scaled = time_scaled[::downsample]
        density_scaled = {name: np.asarray(data)[::downsample] for name, data in density_scaled.items()}

    amplitude_data = None
    if color_by_amplitude and probe_amplitude is not None:
        if isinstance(probe_amplitude, dict):
            amplitude_data = {}
            for name in signals.keys():
                if name in probe_amplitude:
                    amplitude_data[name] = np.asarray(probe_amplitude[name])
                elif len(probe_amplitude) == 1:
                    amplitude_data[name] = np.asarray(list(probe_amplitude.values())[0])
        else:
            amp_array = np.asarray(probe_amplitude)
            if len(signals) == 1:
                amplitude_data = {list(signals.keys())[0]: amp_array}
            else:
                amplitude_data = {name: amp_array for name in signals.keys()}

    fig, ax = plt.subplots(**merge_kwargs({"figsize": (12, 6)}, subplots_kwargs))
    line_args = normalize_call_args(plot_args)
    grid_options = merge_kwargs({"alpha": 0.3}, grid_kwargs)
    color_mappable = None
    color_label = None
    for name, data in density_scaled.items():
        line_kwargs = merge_kwargs({"label": name, "linewidth": 2}, plot_kwargs)
        min_len = min(len(time_scaled), len(data))
        time_plot = time_scaled[:min_len]
        data_plot = data[:min_len]
        if color_by_amplitude and amplitude_data is not None and name in amplitude_data:
            amp_array = amplitude_data[name]
            if downsample > 1:
                amp_array = amp_array[::downsample]
            min_amp_len = min(len(amp_array), min_len)
            amp_plot = amp_array[:min_amp_len]
            time_plot = time_plot[:min_amp_len]
            data_plot = data_plot[:min_amp_len]
            amp_color, amp_unit = _convert_amplitude_values(
                amp_plot,
                amplitude_dbm=amplitude_dbm,
                amplitude_unit=amplitude_unit,
                amplitude_impedance=amplitude_impedance,
            )
            finite_amp = amp_color[np.isfinite(amp_color)]
            if finite_amp.size == 0:
                ax.plot(time_plot, data_plot, *line_args, **line_kwargs)
                continue
            line = colored_line(
                time_plot,
                data_plot,
                amp_color,
                ax,
                autoscale_view=True,
                **merge_kwargs(
                    {
                        "cmap": plt.get_cmap(amplitude_colormap),
                        "norm": plt.Normalize(
                            vmin=float(finite_amp.min()),
                            vmax=float(finite_amp.max()),
                        ),
                        "linewidth": line_kwargs.get("linewidth", 2),
                        "label": name,
                    },
                    line_collection_kwargs,
                ),
            )
            if color_mappable is None and line is not None:
                color_mappable = line
                color_label = f"Probe Amplitude [{amp_unit}]"
        else:
            ax.plot(time_plot, data_plot, *line_args, **line_kwargs)

    ax.set_xlabel(time_label, **FontStyle.label)
    ax.set_ylabel(f"LID {density_label}", **FontStyle.label)
    ax.set_title(title, **FontStyle.title)
    ax.grid(True, **grid_options)
    if color_mappable is not None:
        cbar = plt.colorbar(color_mappable, ax=ax, **merge_kwargs(colorbar_kwargs))
        cbar.set_label(color_label or "Probe Amplitude", **FontStyle.label)
    if len(signals) > 1 and not color_by_amplitude:
        ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, **merge_kwargs({"dpi": 300, "bbox_inches": "tight"}, savefig_kwargs))
    if show_plot:
        plt.show(block=False)
    return fig, ax


def _infer_freq_from_density_df(df: pd.DataFrame) -> float:
    if any("_ALL" in str(col) for col in df.columns):
        return 280.0
    return 94.0


def _is_meaningful_time_index(df: pd.DataFrame) -> bool:
    if not hasattr(df, "index"):
        return False
    if isinstance(df.index, pd.RangeIndex):
        return False
    try:
        numeric_index = pd.to_numeric(pd.Series(df.index, copy=False), errors="coerce")
    except Exception:
        return False
    return bool(numeric_index.notna().any())


def _time_from_raw_fallback(
    density_df: pd.DataFrame,
    *,
    density_name: str | None = None,
    signals_dict: dict[str, pd.DataFrame] | None = None,
) -> np.ndarray | None:
    if not signals_dict:
        return None

    inferred_freq = None
    if density_name:
        inferred_freq = parse_density_group_name(str(density_name))
    if inferred_freq is None and "freq" in getattr(density_df, "attrs", {}):
        try:
            inferred_freq = float(density_df.attrs["freq"])
            if inferred_freq > 1.0e6:
                inferred_freq /= 1.0e9
            inferred_freq = map_frequency_to_group(inferred_freq)
        except (TypeError, ValueError):
            inferred_freq = None
    if inferred_freq is None:
        inferred_freq = _infer_freq_from_density_df(density_df)

    same_freq_sources: list[pd.DataFrame] = []
    for raw_df in signals_dict.values():
        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            continue
        raw_freq = getattr(raw_df, "attrs", {}).get("freq")
        try:
            raw_group = map_frequency_to_group(float(raw_freq))
        except (TypeError, ValueError):
            raw_group = None
        if raw_group is None:
            continue
        if raw_group == inferred_freq:
            same_freq_sources.append(raw_df)

    if not same_freq_sources:
        for source_name, raw_df in signals_dict.items():
            if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
                continue
            raw_group = parse_frequency_group_from_signal_name(str(source_name))
            if raw_group == inferred_freq:
                same_freq_sources.append(raw_df)

    if not same_freq_sources:
        for raw_df in signals_dict.values():
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                if "TIME" in raw_df.columns or _is_meaningful_time_index(raw_df):
                    same_freq_sources.append(raw_df)
                    break
    if not same_freq_sources:
        return None

    target_len = len(density_df)
    for raw_df in same_freq_sources:
        if "TIME" in raw_df.columns and len(raw_df["TIME"]) == target_len:
            return pd.to_numeric(raw_df["TIME"], errors="coerce").to_numpy()
        if _is_meaningful_time_index(raw_df) and len(raw_df.index) == target_len:
            return pd.to_numeric(pd.Series(raw_df.index, copy=False), errors="coerce").to_numpy()

    candidate = same_freq_sources[0]
    if "TIME" in candidate.columns and len(candidate["TIME"]) >= target_len:
        return pd.to_numeric(candidate["TIME"], errors="coerce").iloc[:target_len].to_numpy()
    if _is_meaningful_time_index(candidate) and len(candidate.index) >= target_len:
        numeric_index = pd.to_numeric(pd.Series(candidate.index, copy=False), errors="coerce")
        return numeric_index.iloc[:target_len].to_numpy()
    return None


def resolve_density_time_data(
    density_df: pd.DataFrame,
    *,
    density_name: str | None = None,
    signals_dict: dict[str, pd.DataFrame] | None = None,
) -> np.ndarray:
    """Resolve the best available time axis for density plotting."""
    if "TIME" in density_df.columns:
        return pd.to_numeric(density_df["TIME"], errors="coerce").to_numpy()
    if _is_meaningful_time_index(density_df):
        return pd.to_numeric(pd.Series(density_df.index, copy=False), errors="coerce").to_numpy()
    raw_time = _time_from_raw_fallback(
        density_df,
        density_name=density_name,
        signals_dict=signals_dict,
    )
    if raw_time is not None:
        return raw_time
    return np.arange(len(density_df), dtype=float)


def _auto_probe_amplitudes(
    density_df: pd.DataFrame,
    signals_dict: dict[str, pd.DataFrame] | None,
) -> dict[str, np.ndarray] | None:
    if not signals_dict or not isinstance(density_df, pd.DataFrame):
        return None
    freq_ghz = _infer_freq_from_density_df(density_df)
    for _, sig_df in signals_dict.items():
        if sig_df is None or sig_df.empty:
            continue
        amps = extract_probe_amplitudes_from_signals(density_df, sig_df, freq_ghz)
        if amps:
            return amps
    return None


def render_overview_density(
    density_dict: dict[str, pd.DataFrame],
    title_prefix: str,
    color_density_by_amplitude: bool,
    probe_amplitudes: dict[str, np.ndarray] | None,
    signals_dict: dict[str, pd.DataFrame] | None,
    amplitude_colormap: str,
    amplitude_impedance: float,
    downsample: int,
    warn_fn: Any,
    trigger_time: float = 0.0,
    **kwargs: Any,
) -> None:
    """Render density section for analysis overview."""
    for name, df in density_dict.items():
        if df is None or df.empty:
            continue
        try:
            time_data = resolve_density_time_data(
                df,
                density_name=name,
                signals_dict=signals_dict,
            )

            plot_probe_amp = None
            if color_density_by_amplitude:
                if probe_amplitudes is not None:
                    plot_probe_amp = {
                        col_name: probe_amplitudes[col_name]
                        for col_name in df.columns
                        if col_name in probe_amplitudes
                    }
                else:
                    plot_probe_amp = _auto_probe_amplitudes(df, signals_dict)

            plot_density_core(
                df,
                time_data=time_data,
                title=f"{title_prefix}{name}",
                downsample=downsample,
                trigger_time=trigger_time,
                show_plot=True,
                probe_amplitude=plot_probe_amp,
                color_by_amplitude=color_density_by_amplitude,
                amplitude_colormap=amplitude_colormap,
                amplitude_impedance=amplitude_impedance,
                **kwargs,
            )
        except Exception as e:
            warn_fn(name, e)
