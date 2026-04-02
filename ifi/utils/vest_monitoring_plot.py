#!/usr/bin/env python3
"""
VEST monitoring plotting helpers
================================

Matplotlib implementation of the MATLAB shot monitoring figure set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..plots.style import FontStyle, set_plot_style


def _shot_text(shots: list[int]) -> str:
    return "-".join(map(str, shots))


def _save_fig(
    fig,
    path: Path,
    overwrite: bool,
    show: bool = False,
    auto_close_sec: float | None = None,
) -> None:
    if path.exists() and not overwrite:
        plt.close(fig)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if show:
        if auto_close_sec is not None and auto_close_sec > 0:
            plt.show(block=False)
            plt.pause(float(auto_close_sec))
        else:
            plt.show(block=True)
    plt.close(fig)


def _plot_multi_shot(
    shot_frames: dict[int, pd.DataFrame],
    title: str,
    ylabel: str,
    xlim_ms: tuple[float, float],
    output_path: Path,
    overwrite: bool = False,
    show: bool = False,
    auto_close_sec: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for shot, frame in shot_frames.items():
        if frame.empty:
            continue
        for col in frame.columns:
            label = f"{shot}" if len(frame.columns) == 1 else f"{shot}:{col}"
            ax.plot(frame.index, frame[col], linewidth=1.5, label=label)
    ax.set_title(title, **FontStyle.subtitle)
    ax.set_xlabel("Time [msec]", **FontStyle.label)
    ax.set_ylabel(ylabel, **FontStyle.label)
    ax.set_xlim(xlim_ms)
    ax.grid(True, which="both", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
    _save_fig(fig, output_path, overwrite, show=show, auto_close_sec=auto_close_sec)


def _plot_tiled_multi_shot(
    shot_frames: dict[int, pd.DataFrame],
    title: str,
    xlim_ms: tuple[float, float],
    output_path: Path,
    overwrite: bool = False,
    show: bool = False,
    auto_close_sec: float | None = None,
) -> None:
    first = next((df for df in shot_frames.values() if isinstance(df, pd.DataFrame) and not df.empty), None)
    if first is None:
        return
    nrows = len(first.columns)
    fig, axes = plt.subplots(nrows, 1, figsize=(7.0, 1.8 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    for idx, col in enumerate(first.columns):
        ax = axes[idx]
        for shot, frame in shot_frames.items():
            if col not in frame.columns:
                continue
            ax.plot(frame.index, frame[col], linewidth=1.5, label=str(shot))
        ax.set_ylabel(col, **FontStyle.default)
        ax.set_xlim(xlim_ms)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=4)
    axes[-1].set_xlabel("Time [msec]", **FontStyle.label)
    fig.suptitle(title, **FontStyle.title)
    _save_fig(fig, output_path, overwrite, show=show, auto_close_sec=auto_close_sec)


def _plot_mirnov_spectrograms(
    mirnov_specs_by_shot: dict[int, dict[str, dict[str, np.ndarray]]],
    xlim_ms: tuple[float, float],
    output_dir: Path,
    overwrite: bool = False,
    show: bool = False,
    auto_close_sec: float | None = None,
) -> None:
    for shot, spec_map in mirnov_specs_by_shot.items():
        if not spec_map:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
        items = [("Mirnov_Inboard", "Inboard Midplane"), ("Mirnov_Outboard", "Outboard Midplane")]
        for ax, (key, title) in zip(axes, items, strict=False):
            spec = spec_map.get(key)
            if not spec:
                continue
            mesh = ax.pcolormesh(
                spec["time_ms"],
                spec["freq_khz"],
                spec["power"],
                shading="auto",
            )
            ax.set_ylabel("Frequency [kHz]", **FontStyle.label)
            ax.set_ylim(0.0, 80.0)
            ax.set_xlim(max(xlim_ms[0], 290.0), min(xlim_ms[1], 310.0))
            ax.set_title(title, **FontStyle.default)
            fig.colorbar(mesh, ax=ax)
        axes[-1].set_xlabel("Time [msec]", **FontStyle.label)
        fig.suptitle(f"Mirnov Spectrogram {shot}", **FontStyle.title)
        _save_fig(
            fig,
            output_dir / f"MirnovSpectrogram_{shot}.png",
            overwrite,
            show=show,
            auto_close_sec=auto_close_sec,
        )


def save_vest_monitoring_plots(
    shots: list[int],
    monitoring_by_shot: dict[int, dict[str, pd.DataFrame]],
    mirnov_specs_by_shot: dict[int, dict[str, dict[str, np.ndarray]]],
    save_dir: str | Path,
    xrange_s: tuple[float, float] = (0.28, 0.35),
    overwrite: bool = False,
    show_plots: bool = False,
    auto_close_sec: float | None = None,
) -> None:
    set_plot_style()
    save_path = Path(save_dir)
    xlim_ms = (xrange_s[0] * 1000.0, xrange_s[1] * 1000.0)
    shot_str = _shot_text(shots)

    for key, title, ylabel, filename in [
        ("tf_current", "TF Coil Current", "I_TF [kA]", f"TF_current_{shot_str}.png"),
        ("plasma_current", "Plasma Current", "Plasma Current [kA]", f"PlasmaCurrent_{shot_str}.png"),
        ("pressure", "Pressure", "Pressure [Torr]", f"Pressure_{shot_str}.png"),
        ("flux_loop", "Flux Loop", "Flux Loop [V]", f"FluxLoop_{shot_str}.png"),
        ("diamagnetic_flux", "Diamagnetic Flux", "Diamagnetic Flux [mWb]", f"DiamagneticFlux_{shot_str}.png"),
    ]:
        frames = {shot: monitoring_by_shot.get(shot, {}).get(key, pd.DataFrame()) for shot in shots}
        _plot_multi_shot(
            frames,
            title,
            ylabel,
            xlim_ms,
            save_path / filename,
            overwrite=overwrite,
            show=show_plots,
            auto_close_sec=auto_close_sec,
        )

    for key, title, filename in [
        ("pf_current", "PF Coil Current", f"PF_Current_{shot_str}.png"),
        ("filterscope_2", "Filterscope 2", f"filterscope_2_{shot_str}.png"),
        ("filterscope_1", "Filterscope 1", f"filterscope_1_{shot_str}.png"),
        ("mirnov", "Mirnov Signals", f"Mirnov_{shot_str}.png"),
        ("limiter_current", "Limiter Current", f"LimiterCurrentMonitor_{shot_str}.png"),
    ]:
        frames = {shot: monitoring_by_shot.get(shot, {}).get(key, pd.DataFrame()) for shot in shots}
        _plot_tiled_multi_shot(
            frames,
            title,
            xlim_ms,
            save_path / filename,
            overwrite=overwrite,
            show=show_plots,
            auto_close_sec=auto_close_sec,
        )

    _plot_mirnov_spectrograms(
        mirnov_specs_by_shot=mirnov_specs_by_shot,
        xlim_ms=xlim_ms,
        output_dir=save_path,
        overwrite=overwrite,
        show=show_plots,
        auto_close_sec=auto_close_sec,
    )


__all__ = [
    "save_vest_monitoring_plots",
]
