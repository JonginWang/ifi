#!/usr/bin/env python3
"""
Plotting utilities
==================

This module contains the main plotting API for the IFI package.

Key Features:
- Main plotting facade for waveform, time-frequency, and density plots
- Interactive plotting runtime/session helpers
- Shot-level plotting helpers that save figures to the results tree
- Time-frequency analysis plots (STFT/CWT)
- Density plots
- Shot overview generation
- LaTeX formatting for scientific notation
- Unified plotting interface

Author: Jongin Wang
Date: 2025-01-16
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.spectrum import SpectrumAnalysis
from ..utils.log_manager import LogManager, log_tag
from .plot_density import plot_density_core
from .plot_overview import plot_shot_overview_core, render_analysis_overview
from .plot_runtime import interactive_plotting, setup_interactive_mode
from .plot_shot import (
    plot_shot_density_evolution,
    plot_shot_spectrograms,
    plot_shot_waveforms,
)
from .plot_timefreq import (
    plot_response_core,
    plot_time_frequency_core,
    render_cwt_results,
    render_stft_results,
)
from .plot_waveform import plot_waveforms_core, render_signal_dict
from .style import set_plot_style

logger = LogManager().get_logger(__name__)


class Plotter:
    """Main plotting facade for waveform, time-frequency, and density plots."""

    def __init__(self):
        self.analyzer = SpectrumAnalysis()
        set_plot_style()

    def plot_waveforms(self, *args, **kwargs):
        return plot_waveforms_core(*args, **kwargs)

    def plot_density(self, *args, **kwargs):
        return plot_density_core(*args, **kwargs)

    def plot_response(self, *args, **kwargs):
        return plot_response_core(*args, **kwargs)

    def plot_shot_overview(self, *args, **kwargs):
        return plot_shot_overview_core(*args, **kwargs)

    def plot_time_frequency(self, *args, **kwargs):
        kwargs.setdefault("analyzer", self.analyzer)
        return plot_time_frequency_core(*args, **kwargs)

    def plot_signals(self, *args, **kwargs):
        return render_signal_dict(*args, **kwargs)

    def plot_spectrograms(self, stft_results, **kwargs):
        kwargs.setdefault("analyzer", self.analyzer)
        return render_stft_results(stft_results, **kwargs)

    def plot_cwt(self, cwt_results, **kwargs):
        kwargs.setdefault("analyzer", self.analyzer)
        return render_cwt_results(cwt_results, **kwargs)

    def plot_analysis_overview(
        self,
        shot_num: int,
        signals_dict: dict[str, pd.DataFrame],
        density_dict: dict[str, pd.DataFrame],
        vest_data: pd.DataFrame | None = None,
        trigger_time: float = 0.0,
        title_prefix: str = "",
        downsample: int = 100,
        color_density_by_amplitude: bool = False,
        probe_amplitudes: dict[str, np.ndarray] | None = None,
        amplitude_colormap: str = "coolwarm",
        amplitude_impedance: float = 50.0,
        **kwargs,
    ):
        def wave_warn(name: str, exc: Exception) -> None:
            logger.warning(f"{log_tag('PLOTS', 'OVER')} Failed to plot {name}: {exc}")

        def density_warn(name: str, exc: Exception) -> None:
            logger.warning(
                f"{log_tag('PLOTS', 'OVER')} Failed to plot density {name}: {exc}"
            )

        return render_analysis_overview(
            shot_num,
            signals_dict,
            density_dict,
            vest_data=vest_data,
            trigger_time=trigger_time,
            title_prefix=title_prefix,
            downsample=downsample,
            color_density_by_amplitude=color_density_by_amplitude,
            probe_amplitudes=probe_amplitudes,
            amplitude_colormap=amplitude_colormap,
            amplitude_impedance=amplitude_impedance,
            wave_warn_fn=wave_warn,
            density_warn_fn=density_warn,
            **kwargs,
        )


__all__ = [
    "Plotter",
    "interactive_plotting",
    "plot_shot_density_evolution",
    "plot_shot_spectrograms",
    "plot_shot_waveforms",
    "setup_interactive_mode",
]
