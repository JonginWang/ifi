#!/usr/bin/env python3
"""
Time-frequency plotting module
=============================

Author: Jongin Wang
Date: 2025-01-16
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..analysis.functions.power_conversion import pow2db
from ..analysis.params.params_plot import FontStyle
from ..analysis.spectrum import SpectrumAnalysis
from ..utils.func_helper import merge_kwargs, normalize_call_args
from .plot_common_module import prepare_time_data

if TYPE_CHECKING:
    from ..analysis.spectrum import SpectrumAnalysis

def plot_time_frequency_core(
    data: pd.DataFrame | dict[str, np.ndarray] | np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
    method: str = "stft",
    fs: float | None = None,
    title: str = "Time-Frequency Analysis",
    save_path: str | None = None,
    show_plot: bool = True,
    time_scale: str = "s",
    freq_scale: str = "MHz",
    power_scale: str = "dB",
    trigger_time: float = 0.0,
    downsample: int = 1,
    analyzer: "SpectrumAnalysis" | None = None,
    subplots_kwargs: dict[str, Any] | None = None,
    mesh_kwargs: dict[str, Any] | None = None,
    ridge_plot_args: tuple[Any, ...] | list[Any] | None = None,
    ridge_plot_kwargs: dict[str, Any] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Core STFT/CWT plotting implementation."""
    if analyzer is None:
        analyzer = SpectrumAnalysis()

    if isinstance(data, tuple) and len(data) == 3:
        freqs, times, Sxx = data
        method = "precomputed"
        n_signals = 1
        signals = {"Precomputed": None}
        time = times
    else:
        time, signals = prepare_time_data(data, fs)
        time = time + trigger_time
        if downsample > 1:
            time = time[::downsample]
            signals = {k: v[::downsample] for k, v in signals.items()}
        n_signals = len(signals)
        if n_signals == 0:
            raise ValueError("No signals found in data.")

    if time_scale == "ms":
        time_scale_factor, time_label = 1e3, "Time [ms]"
    elif time_scale == "us":
        time_scale_factor, time_label = 1e6, "Time [us]"
    elif time_scale == "ns":
        time_scale_factor, time_label = 1e9, "Time [ns]"
    else:
        time_scale_factor, time_label = 1, "Time [s]"

    if freq_scale == "kHz":
        freq_scale_factor = 1e-3
    elif freq_scale == "MHz":
        freq_scale_factor = 1e-6
    elif freq_scale == "GHz":
        freq_scale_factor = 1e-9
    else:
        freq_scale_factor = 1

    fig, axes = plt.subplots(
        n_signals,
        1,
        **merge_kwargs(
            {"figsize": (12, 4 * n_signals), "sharex": True},
            subplots_kwargs,
        ),
    )
    if n_signals == 1:
        axes = [axes]

    mesh_options = merge_kwargs({"shading": "gouraud"}, mesh_kwargs)
    ridge_args = normalize_call_args(ridge_plot_args)
    ridge_options = merge_kwargs({"color": "r", "linewidth": 2, "label": "Frequency Ridge"}, ridge_plot_kwargs)

    for i, (name, signal) in enumerate(signals.items()):
        ax = axes[i]
        if method.lower() == "precomputed":
            freqs_scaled = freqs * freq_scale_factor
            times_scaled = times * time_scale_factor + trigger_time
            if power_scale == "dB":
                Sxx_plot = pow2db(np.abs(Sxx), dbm=False)
                power_label = "Power [dB]"
            else:
                Sxx_plot = np.abs(Sxx)
                power_label = "Power [linear]"
            im = ax.pcolormesh(times_scaled, freqs_scaled, Sxx_plot, **merge_kwargs({"cmap": "plasma"}, mesh_options))
        elif method.lower() == "stft":
            freqs, times, Sxx = analyzer.compute_stft(signal, fs, **kwargs)
            freqs_scaled = freqs * freq_scale_factor
            if power_scale == "dB":
                Sxx_plot = pow2db(np.abs(Sxx), dbm=False)
                power_label = "Power [dB]"
            else:
                Sxx_plot = np.abs(Sxx)
                power_label = "Power [linear]"
            im = ax.pcolormesh(
                times * time_scale_factor,
                freqs_scaled,
                Sxx_plot,
                **merge_kwargs({"cmap": "plasma"}, mesh_options),
            )
            try:
                ridge = analyzer.find_freq_ridge(Sxx, freqs, method="stft")
                ax.plot(
                    times * time_scale_factor,
                    ridge * freq_scale_factor,
                    *ridge_args,
                    **ridge_options,
                )
                ax.legend()
            except Exception:
                pass
        elif method.lower() == "cwt":
            freqs, Wx = analyzer.compute_cwt(signal, fs, **kwargs)
            freqs_scaled = freqs * freq_scale_factor
            cwt_matrix = np.abs(Wx)
            im = ax.pcolormesh(
                time * time_scale_factor,
                freqs_scaled,
                cwt_matrix,
                **merge_kwargs({"cmap": "hot"}, mesh_options),
            )
            power_label = "Magnitude"
        else:
            raise ValueError(f"Unknown method: {method}.")

        ax.set_ylabel(f"Frequency [{freq_scale}]", **FontStyle.label)
        ax.set_title(f"{title} - {name}", **FontStyle.title)
        cbar = fig.colorbar(im, ax=ax, **merge_kwargs(colorbar_kwargs))
        cbar.set_label(power_label, **FontStyle.label)

    axes[-1].set_xlabel(time_label, **FontStyle.label)
    fig.suptitle(f"{title} ({method.upper()})", **FontStyle.title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, **merge_kwargs({"dpi": 300, "bbox_inches": "tight"}, savefig_kwargs))
    if show_plot:
        plt.show(block=False)
    return fig, axes


def plot_response_core(
    freqs: np.ndarray,
    responses: np.ndarray,
    title: str = "Frequency Response",
    save_path: str | None = None,
    show_plot: bool = True,
    freq_scale: str = "Hz",
    subplots_kwargs: dict[str, Any] | None = None,
    plot_args: tuple[Any, ...] | list[Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    savefig_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Core response plotting implementation."""
    if freq_scale == "kHz":
        freq_scale_factor = 1e-3
    elif freq_scale == "MHz":
        freq_scale_factor = 1e-6
    elif freq_scale == "GHz":
        freq_scale_factor = 1e-9
    else:
        freq_scale_factor = 1

    fig, ax = plt.subplots(**merge_kwargs({"figsize": (10, 6)}, subplots_kwargs))
    ax.plot(
        freqs * freq_scale_factor,
        20 * np.log10(np.abs(responses)),
        *normalize_call_args(plot_args),
        **merge_kwargs(plot_kwargs),
    )
    ax.set_ylim(-100, 5)
    ax.grid(True, **merge_kwargs(grid_kwargs))
    ax.set_xlabel(f"Frequency [{freq_scale}]", **FontStyle.label)
    ax.set_ylabel("Gain [dB]", **FontStyle.label)
    ax.set_title(title, **FontStyle.title)
    if save_path:
        fig.savefig(save_path, **merge_kwargs({"dpi": 300, "bbox_inches": "tight"}, savefig_kwargs))
    if show_plot:
        plt.show(block=False)
    return fig, ax

def render_stft_results(
    stft_results: dict,
    analyzer: "SpectrumAnalysis" | None = None,
    **kwargs: Any,
) -> None:
    """Render STFT result dictionary."""
    for filename, results_by_col in stft_results.items():
        for col_name, results in results_by_col.items():
            if (
                "STFT_matrix" in results
                and "freq_STFT" in results
                and "time_STFT" in results
            ):
                freqs = results["freq_STFT"]
                times = results["time_STFT"]
                stft_matrix = results["STFT_matrix"]
                plot_time_frequency_core(
                    (freqs, times, stft_matrix),
                    method="precomputed",
                    title=kwargs.get("title_prefix", "")
                    + f"{Path(filename).name} - {col_name}",
                    analyzer=analyzer,
                    **{k: v for k, v in kwargs.items() if k != "title_prefix"},
                )
            elif "Zxx" in results and "freqs" in results and "times" in results:
                plot_time_frequency_core(
                    (results["freqs"], results["times"], results["Zxx"]),
                    method="precomputed",
                    title=kwargs.get("title_prefix", "")
                    + f"{Path(filename).name} - {col_name}",
                    analyzer=analyzer,
                    **{k: v for k, v in kwargs.items() if k != "title_prefix"},
                )


def render_cwt_results(
    cwt_results: dict,
    analyzer: "SpectrumAnalysis" | None = None,
    **kwargs: Any,
) -> None:
    """Render CWT result dictionary."""
    for filename, analysis in cwt_results.items():
        for col_name, result in analysis.items():
            if (
                "CWT_matrix" in result
                and "freq_CWT" in result
                and "time_CWT" in result
            ):
                plot_time_frequency_core(
                    (result["freq_CWT"], result["time_CWT"], result["CWT_matrix"]),
                    method="precomputed",
                    title=kwargs.get("title_prefix", "")
                    + f"{Path(filename).name} - {col_name}",
                    analyzer=analyzer,
                    **{k: v for k, v in kwargs.items() if k != "title_prefix"},
                )
            elif "cwt_matrix" in result and "freqs" in result and "times" in result:
                plot_time_frequency_core(
                    (result["freqs"], result["times"], result["cwt_matrix"]),
                    method="precomputed",
                    title=kwargs.get("title_prefix", "")
                    + f"{Path(filename).name} - {col_name}",
                    analyzer=analyzer,
                    **{k: v for k, v in kwargs.items() if k != "title_prefix"},
                )
