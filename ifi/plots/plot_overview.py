#!/usr/bin/env python3
"""
Composed overview plotting helpers.
==================================

This module contains the composed overview plotting helpers.
It includes the functions for rendering the analysis overview.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .plot_density import render_overview_density
from .plot_waveform import render_overview_waveforms
from .style import FontStyle


def render_analysis_overview(
    shot_num: int,
    signals_dict: dict[str, pd.DataFrame],
    density_dict: dict[str, pd.DataFrame],
    vest_data: pd.DataFrame | None = None,
    trigger_time: float = 0.0,
    title_prefix: str = "",
    downsample: int = 100,
    plot_envelope: bool = False,
    color_density_by_amplitude: bool = False,
    probe_amplitudes: dict | None = None,
    amplitude_colormap: str = "coolwarm",
    amplitude_impedance: float = 50.0,
    wave_warn_fn: Any | None = None,
    density_warn_fn: Any | None = None,
    **kwargs: Any,
) -> None:
    """Render waveform+density+vest overview."""
    if wave_warn_fn is None:
        wave_warn_fn = lambda name, e: None
    if density_warn_fn is None:
        density_warn_fn = lambda name, e: None

    render_overview_waveforms(
        signals_dict=signals_dict,
        title_prefix=title_prefix,
        trigger_time=trigger_time,
        downsample=downsample,
        warn_fn=wave_warn_fn,
        plot_envelope=plot_envelope,
        **kwargs,
    )
    render_overview_density(
        density_dict=density_dict,
        title_prefix=title_prefix,
        color_density_by_amplitude=color_density_by_amplitude,
        probe_amplitudes=probe_amplitudes,
        signals_dict=signals_dict,
        amplitude_colormap=amplitude_colormap,
        amplitude_impedance=amplitude_impedance,
        warn_fn=density_warn_fn,
        **kwargs,
    )

    if vest_data is None or vest_data.empty:
        return

    ip_col = None
    for col in vest_data.columns:
        col_s = str(col).lower()
        if "ip" in col_s or "current" in col_s:
            ip_col = col
            break
    if ip_col is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    time_vest = vest_data.index.values + trigger_time
    ax.plot(time_vest * 1000, vest_data[ip_col].values, "r-", linewidth=2)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(f"{ip_col}")
    ax.set_title(f"{title_prefix}Plasma Current")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


def plot_shot_overview_core(
    shot_data: dict,
    vest_data: pd.DataFrame,
    shot_num: int,
    save_path: str | None = None,
    show_plot: bool = True,
):
    """Core shot overview plotting implementation."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Shot {shot_num} - Overview", **FontStyle.title)

    if shot_data:
        first_df = list(shot_data.values())[0]
        if isinstance(first_df, pd.DataFrame) and "TIME" in first_df.columns:
            signal_cols = [col for col in first_df.columns if col != "TIME"]
            if signal_cols:
                axes[0, 0].plot(first_df.index.values * 1000, first_df[signal_cols[0]].values)
                axes[0, 0].set_title("Raw Signal", **FontStyle.title)
                axes[0, 0].set_xlabel("Time [ms]", **FontStyle.label)
                axes[0, 0].set_ylabel("Amplitude", **FontStyle.label)
                axes[0, 0].grid(True, alpha=0.3)

    if vest_data is not None and not vest_data.empty and "Ip_raw ([V])" in vest_data.columns:
        axes[0, 1].plot(vest_data.index.values * 1000, vest_data["Ip_raw ([V])"].values, "r-")
        axes[0, 1].set_title("Plasma Current", **FontStyle.title)
        axes[0, 1].set_xlabel("Time [ms]", **FontStyle.label)
        axes[0, 1].set_ylabel(r"$I_{p}$ [V]", **FontStyle.label)
        axes[0, 1].grid(True, alpha=0.3)

    density_data = {
        key: df for key, df in shot_data.items() if key.startswith("ne_") and isinstance(df, pd.Series)
    }
    if density_data:
        for key, data in density_data.items():
            axes[1, 0].plot(data.index.values * 1000, data.values / 1e18, label=key)
        axes[1, 0].set_title("Density Evolution", **FontStyle.title)
        axes[1, 0].set_xlabel("Time [ms]", **FontStyle.label)
        axes[1, 0].set_ylabel(r"Density [$10^{18} m^{-3}$]", **FontStyle.label)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(0.1, 0.8, f"Shot Number: {shot_num}", fontsize=12)
    axes[1, 1].text(0.1, 0.6, f"Data Files: {len(shot_data)}", fontsize=12)
    if vest_data is not None:
        axes[1, 1].text(0.1, 0.4, "VEST Data: Available", fontsize=12)
    axes[1, 1].text(0.1, 0.2, f"Density Channels: {len(density_data)}", fontsize=12)
    axes[1, 1].set_title("Summary", **FontStyle.title)
    axes[1, 1].axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show(block=False)
    return fig, axes
