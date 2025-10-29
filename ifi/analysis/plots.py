#!/usr/bin/env python3
"""
Plotting Functions
==================

This module contains plotting functions for the analysis results.

Key Features:
    Unified waveform plotting function with flexible data formats
    Unified time-frequency plotting (STFT/CWT) with consistent API
    Unified density plotting function with flexible data formats
    Unified response plotting function with flexible data formats
    Unified shot overview plotting function with flexible data formats
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from contextlib import contextmanager
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ifi.utils.cache_setup import setup_project_cache
from ifi.db_controller.vest_db import VEST_DB
from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.analysis.params.params_plot import set_plot_style, FontStyle
from ifi.utils.common import LogManager, ensure_dir_exists

cache_config = setup_project_cache()

LogManager()

# ============================================================================
# Interactive Mode Setup and Management
# ============================================================================


def setup_interactive_mode(backend: str = "auto", style: str = "default"):
    """Setup matplotlib for optimal interactive use."""
    if backend == "auto":
        try:
            import tkinter  # noqa: F401

            matplotlib.use("TkAgg")
        except ImportError:
            try:
                matplotlib.use("Qt5Agg")
            except ImportError:
                matplotlib.use("Agg")
    else:
        matplotlib.use(backend)

    plt.style.use(style)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["interactive"] = True
    set_plot_style()


@contextmanager
def interactive_plotting(
    show_plots: bool = True,
    save_dir: Optional[str] = None,
    save_prefix: str = "fig_",
    save_ext: str = "png",
    dpi: int = 300,
    block: bool = True,
):
    """
    Enhanced context manager for interactive plotting.

    Args:
        show_plots: Whether to show the plots
        save_dir: Directory to save the plots
        save_prefix: Prefix for the saved plots
        save_ext: Extension for the saved plots
        dpi: DPI for the saved plots
        block: Whether to block the plots

    Returns:
        None
    """
    original_backend = matplotlib.get_backend()
    original_interactive = plt.isinteractive()

    try:
        if show_plots:
            setup_interactive_mode()
            plt.ion()
        yield
    finally:
        if save_dir:
            ensure_dir_exists(save_dir)
            for i in plt.get_fignums():
                fig = plt.figure(i)
                if fig._suptitle:
                    title = fig._suptitle.get_text()
                elif fig.axes and fig.axes[0].get_title():
                    title = fig.axes[0].get_title()
                else:
                    title = f"figure_{i}"

                filename = "".join(
                    c for c in title if c.isalnum() or c in (" ", "_", "-")
                ).rstrip()
                filename = filename.replace(" ", "_").replace("#", "")
                filepath = Path(save_dir) / f"{save_prefix}{filename}.{save_ext}"

                try:
                    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
                    logging.info(f"Saved figure to {filepath}")
                except Exception as e:
                    logging.error(f"Failed to save figure {i}: {e}")

        if show_plots:
            plt.ioff()
        if block:
            plt.show(block=True)
        else:
            plt.show(block=False)

        matplotlib.use(original_backend)
        plt.interactive(original_interactive)


# ============================================================================
# Utility Functions
# ============================================================================


def pow2db(x: np.ndarray) -> np.ndarray:
    """Convert power [W] to decibels [dB]."""
    return 10 * np.log10(x)


def db2pow(x: np.ndarray) -> np.ndarray:
    """Convert decibels [dB] to power [W]."""
    return 10 ** (x / 10)


def amp2db(x: np.ndarray, impedance: float = 50) -> np.ndarray:
    """Convert amplitude [V] to decibels [dB].
    The impedance is the characteristic impedance of the system.
    The factor of 2 is because the amplitude is the peak-to-peak value.
    """
    return 20 * np.log10(x / np.sqrt(impedance * 2))


def db2amp(x: np.ndarray, impedance: float = 50) -> np.ndarray:
    """Convert decibels [dB] to amplitude [V].
    The impedance is the characteristic impedance of the system.
    The factor of 2 is because the amplitude is the peak-to-peak value.
    """
    return 10 ** (x / 20) * np.sqrt(impedance * 2)


# ============================================================================
# Unified Plotter Class
# ============================================================================


class Plotter:
    """
    Unified plotting class with consolidated functionality.

    This class provides all plotting capabilities in a single, consistent interface:
    - Waveform plotting with flexible data formats and scaling
    - Time-frequency analysis plotting (STFT/CWT)
    - Density and response plotting
    - Shot-specific visualization
    """

    def __init__(self):
        self.db = VEST_DB()
        self.analyzer = SpectrumAnalysis()
        set_plot_style()

    def _prepare_time_data(
        self,
        data: Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray],
        fs: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Unified data preparation for temporal plotting.

        Args:
            data: Input data in various formats
            fs: Sampling frequency (optional)

        Returns:
            time: Time array
            signals: Dictionary of signal arrays
        """
        if isinstance(data, pd.DataFrame):
            # DataFrame with TIME column or index as time
            if any(col.upper() == "TIME" for col in data.columns):
                time_col = [col for col in data.columns if col.upper() == "TIME"][0]
                time = data[time_col].values
                signal_cols = [col for col in data.columns if col.upper() != "TIME"]
                signals = {col: data[col].values for col in signal_cols}
            else:
                time = (
                    data.index.values
                    if hasattr(data, "index")
                    else np.arange(len(data))
                )
                time = time / fs if fs is not None else time
                signals = {col: data[col].values for col in data.columns}

        elif isinstance(data, dict):
            # Dictionary with TIME key or arrays
            if any(k.upper() == "TIME" for k in data.keys()):
                time_key = [k for k in data.keys() if k.upper() == "TIME"][0]
                time = data[time_key]
                signals = {k: data[k] for k in data.keys() if k.upper() != "TIME"}
            else:
                max_len = max(len(data[k]) for k in data.keys())
                signals = data.copy()
                time = np.arange(max_len) / fs if fs is not None else np.arange(max_len)

        elif isinstance(data, np.ndarray):
            # Numpy array
            if data.ndim == 1 and fs is not None:
                signals = {"Signal": data}
                time = np.arange(data.shape[0]) / fs
            elif data.ndim == 2:
                if fs is not None:
                    signals = {f"Signal {i}": data[:, i] for i in range(data.shape[1])}
                    time = np.arange(data.shape[0]) / fs
                else:
                    signals = {
                        f"Signal {i}": data[:, i] for i in range(1, data.shape[1])
                    }
                    time = data[:, 0]
            else:
                logging.error(
                    f"Waveform data must be 1D or 2D numpy array. Got {data.shape}"
                )
                raise ValueError("Waveform data must be 1D or 2D numpy array")
        else:
            logging.error(
                f"Waveform data must be DataFrame, dict, or numpy array. Got {type(data)}"
            )
            raise ValueError("Waveform data must be DataFrame, dict, or numpy array")

        return time, signals

    def _apply_scaling(
        self,
        time: np.ndarray,
        signals: Dict[str, np.ndarray],
        time_scale: str = "s",
        signal_scale: str = "V",
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], str, str]:
        """
        Apply scaling to time and signals.

        Args:
            time: Time array
            signals: Signal dictionary
            time_scale: Time scale ('s', 'ms', 'us', 'ns')
            signal_scale: Signal scale ('V', 'mV', 'uV', 'a.u.', '10^18 m^-3', etc.)

        Returns:
            time_scaled: Scaled time array
            signals_scaled: Scaled signals
            time_label: Time axis label
            signal_label: Signal axis label
        """
        # Time scaling
        time_scale_factor = 1
        if time_scale == "ms":
            time_scale_factor = 1e3
            time_label = "Time [ms]"
        elif time_scale == "us":
            time_scale_factor = 1e6
            time_label = "Time [μs]"
        elif time_scale == "ns":
            time_scale_factor = 1e9
            time_label = "Time [ns]"
        else:
            time_label = "Time [s]"

        # Signal scaling with regex support for scientific notation
        signal_scale_factor = 1
        signal_label = signal_scale

        # Check for scientific notation patterns like "10^18 m^-3"
        sci_pattern = r"10\^(\d+)\s*([a-zA-Z\/\^\-0-9\s]+)"
        sci_match = re.search(sci_pattern, signal_scale)

        if sci_match:
            exponent = int(sci_match.group(1))
            unit = sci_match.group(2).strip()
            signal_scale_factor = 10**exponent

            # Convert superscript notation to LaTeX
            def replace_superscript(match):
                base = match.group(1)
                exp = match.group(2)
                return f"${base}^{{-{exp}}}$"

            unit_latex = re.sub(r"(\w+)\^-(\d+)", replace_superscript, unit)
            signal_label = f"[{unit_latex}]"
        else:
            # Standard scaling
            if signal_scale == "mV":
                signal_scale_factor = 1e3
                signal_label = "[mV]"
            elif signal_scale == "uV":
                signal_scale_factor = 1e6
                signal_label = "[μV]"
            elif signal_scale == "a.u.":
                signal_scale_factor = 1
                signal_label = "[a.u.]"
            elif signal_scale == "V":
                signal_scale_factor = 1
                signal_label = "[V]"
            elif signal_scale == "m^-2":
                signal_scale_factor = 1
                signal_label = r"[${m^{-2}}$]"
            elif signal_scale == "m^-3":
                signal_scale_factor = 1
                signal_label = r"[${m^{-3}}$]"
            else:
                # Custom scale, assume factor is 1 and use as label
                signal_scale_factor = 1
                # Convert any superscript notation to LaTeX
                if "^-" in signal_scale:

                    def replace_superscript(match):
                        base = match.group(1)
                        exp = match.group(2)
                        return f"{base}^{{-{exp}}}"

                    signal_label = (
                        r"[${"
                        + re.sub(r"(\w+)\^-(\d+)", replace_superscript, signal_scale)
                        + r"}$]"
                    )
                else:
                    signal_label = f"[{signal_scale}]"

        time_scaled = time * time_scale_factor
        signals_scaled = {k: v * signal_scale_factor for k, v in signals.items()}

        return time_scaled, signals_scaled, time_label, signal_label

    def _extract_metadata_info(
        self, data: Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]
    ) -> str:
        """
        Extract metadata information for display in plots.

        Args:
            data: Input data that may contain metadata

        Returns:
            Formatted metadata string for display
        """
        metadata_parts = []

        if isinstance(data, pd.DataFrame) and hasattr(data, "attrs") and data.attrs:
            attrs = data.attrs

            # File type and format
            if "source_file_type" in attrs:
                metadata_parts.append(f"Type: {attrs['source_file_type']}")
            if "source_file_format" in attrs:
                metadata_parts.append(f"Format: {attrs['source_file_format']}")

            # Metadata information
            if "metadata" in attrs and isinstance(attrs["metadata"], dict):
                metadata = attrs["metadata"]
                if "record_length" in metadata:
                    metadata_parts.append(f"Length: {metadata['record_length']}")
                if "time_resolution" in metadata:
                    resolution = metadata["time_resolution"]
                    if resolution < 1e-6:
                        metadata_parts.append(f"Resolution: {resolution * 1e9:.1f} ns")
                    elif resolution < 1e-3:
                        metadata_parts.append(f"Resolution: {resolution * 1e6:.1f} μs")
                    elif resolution < 1:
                        metadata_parts.append(f"Resolution: {resolution * 1e3:.1f} ms")
                    else:
                        metadata_parts.append(f"Resolution: {resolution:.3f} s")

        return " | ".join(metadata_parts) if metadata_parts else ""

    def plot_waveforms(
        self,
        data: Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray],
        fs: Optional[float] = None,
        title: str = "Waveforms",
        downsample: int = 1,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        time_scale: str = "s",
        signal_scale: str = "V",
        trigger_time: float = 0.0,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Unified waveform plotting with flexible data formats and scaling.

        Args:
            data: Input data (DataFrame, dict, or numpy array)
            fs: Sampling frequency (optional)
            title: Plot title
            downsample: Downsampling factor
            save_path: Optional path to save figure
            show_plot: Whether to display the plot
            time_scale: Time scale ('s', 'ms', 'us', 'ns')
            signal_scale: Signal scale ('V', 'mV', 'a.u.', etc.)
            trigger_time: Trigger time offset

        Returns:
            fig, axes: Matplotlib figure and axes
        """
        # Prepare data
        time, signals = self._prepare_time_data(data, fs)

        # Apply scaling
        time_scaled, signals_scaled, time_label, signal_label = self._apply_scaling(
            time, signals, time_scale, signal_scale
        )

        # Apply trigger time offset
        time_scaled = time_scaled + trigger_time

        # Downsample if needed
        if downsample > 1:
            time_scaled = time_scaled[::downsample]
            signals_scaled = {k: v[::downsample] for k, v in signals_scaled.items()}

        # Create plot
        n_signals = len(signals_scaled)
        fig, axes = plt.subplots(n_signals, 1, figsize=(12, 2 * n_signals), sharex=True)

        if n_signals == 1:
            axes = [axes]

        # Add metadata to title if available
        metadata_info = self._extract_metadata_info(data)
        if metadata_info:
            title = f"{title}\n{metadata_info}"

        for i, (name, signal) in enumerate(signals_scaled.items()):
            axes[i].plot(time_scaled, signal)
            axes[i].set_ylabel(f"{name} {signal_label}", **FontStyle.label)
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel(time_label, **FontStyle.label)
        fig.suptitle(title, **FontStyle.title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show(block=False)

        return fig, axes

    def plot_time_frequency(
        self,
        data: Union[
            pd.DataFrame,
            Dict[str, np.ndarray],
            np.ndarray,
            Tuple[np.ndarray, np.ndarray, np.ndarray],
        ],
        method: str = "stft",
        fs: Optional[float] = None,
        title: str = "Time-Frequency Analysis",
        save_path: Optional[str] = None,
        show_plot: bool = True,
        time_scale: str = "s",
        freq_scale: str = "MHz",
        power_scale: str = "dB",
        trigger_time: float = 0.0,
        downsample: int = 1,
        **kwargs,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Unified time-frequency plotting (STFT/CWT) with consistent API.

        Args:
            data: Input data or pre-computed arrays (freqs, times, Sxx/Zxx)
            method: Analysis method ('stft', 'cwt', or 'precomputed')
            fs: Sampling frequency (not needed for precomputed data)
            title: Plot title
            save_path: Optional path to save figure
            show_plot: Whether to display the plot
            time_scale: Time scale ('s', 'ms', 'us', 'ns')
            freq_scale: Frequency scale ('Hz', 'kHz', 'MHz', 'GHz')
            power_scale: Power scale ('linear', 'dB')
            trigger_time: Trigger time offset
            downsample: Downsampling factor
            **kwargs: Additional arguments for STFT/CWT computation

        Returns:
            fig, axes: Matplotlib figure and axes
        """
        # Check if data is pre-computed arrays (freqs, times, Sxx/Zxx)
        if isinstance(data, tuple) and len(data) == 3:
            freqs, times, Sxx = data
            method = "precomputed"
            n_signals = 1
            signals = {"Precomputed": None}  # Dummy for loop structure
        else:
            # Prepare data for analysis
            time, signals = self._prepare_time_data(data, fs)

            # Apply trigger time offset
            time = time + trigger_time

            # Downsample if needed
            if downsample > 1:
                time = time[::downsample]
                signals = {k: v[::downsample] for k, v in signals.items()}

            n_signals = len(signals)

        # Time scaling
        time_scale_factor = 1
        if time_scale == "ms":
            time_scale_factor = 1e3
            time_label = "Time [ms]"
        elif time_scale == "us":
            time_scale_factor = 1e6
            time_label = "Time [μs]"
        elif time_scale == "ns":
            time_scale_factor = 1e9
            time_label = "Time [ns]"
        else:
            time_label = "Time [s]"

        # Frequency scaling
        freq_scale_factor = 1
        if freq_scale == "kHz":
            freq_scale_factor = 1e-3
        elif freq_scale == "MHz":
            freq_scale_factor = 1e-6
        elif freq_scale == "GHz":
            freq_scale_factor = 1e-9

        # Create plots
        fig, axes = plt.subplots(n_signals, 1, figsize=(12, 4 * n_signals), sharex=True)

        if n_signals == 1:
            axes = [axes]

        for i, (name, signal) in enumerate(signals.items()):
            ax = axes[i]

            if method.lower() == "precomputed":
                # Use pre-computed arrays
                freqs_scaled = freqs * freq_scale_factor
                times_scaled = times * time_scale_factor + trigger_time

                if power_scale == "dB":
                    Sxx_plot = pow2db(np.abs(Sxx))
                    power_label = "Power [dB]"
                else:
                    Sxx_plot = np.abs(Sxx)
                    power_label = "Power [linear]"

                # Plot spectrogram
                im = ax.pcolormesh(
                    times_scaled,
                    freqs_scaled,
                    Sxx_plot,
                    shading="gouraud",
                    cmap="viridis",
                )

            elif method.lower() == "stft":
                # STFT analysis
                freqs, times, Sxx = self.analyzer.compute_stft(signal, fs, **kwargs)

                # Scale frequency and power
                freqs_scaled = freqs * freq_scale_factor
                if power_scale == "dB":
                    Sxx_plot = pow2db(np.abs(Sxx))
                    power_label = "Power [dB]"
                else:
                    Sxx_plot = np.abs(Sxx)
                    power_label = "Power [linear]"

                # Plot spectrogram
                im = ax.pcolormesh(
                    times * time_scale_factor,
                    freqs_scaled,
                    Sxx_plot,
                    shading="gouraud",
                    cmap="viridis",
                )

                # Add frequency ridge if available
                try:
                    ridge = self.analyzer.find_freq_ridge(Sxx, freqs, method="stft")
                    ax.plot(
                        times * time_scale_factor,
                        ridge * freq_scale_factor,
                        color="r",
                        linewidth=2,
                        label="Frequency Ridge",
                    )
                    ax.legend()
                except Exception:
                    pass

            elif method.lower() == "cwt":
                # CWT analysis
                freqs, Wx = self.analyzer.compute_cwt(signal, fs, **kwargs)

                # Scale frequency
                freqs_scaled = freqs * freq_scale_factor
                cwt_matrix = np.abs(Wx)

                # Plot scalogram
                im = ax.pcolormesh(
                    time * time_scale_factor,
                    freqs_scaled,
                    cwt_matrix,
                    shading="gouraud",
                    cmap="hot",
                )
                power_label = "Magnitude"

            else:
                raise ValueError(
                    f"Unknown method: {method}. Use 'stft', 'cwt', or 'precomputed'"
                )

            ax.set_ylabel(f"Frequency [{freq_scale}]", **FontStyle.label)
            ax.set_title(f"{title} - {name}", **FontStyle.title)

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(power_label, **FontStyle.label)

        axes[-1].set_xlabel(time_label, **FontStyle.label)
        fig.suptitle(f"{title} ({method.upper()})", **FontStyle.title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show(block=False)

        return fig, axes

    def plot_density(
        self,
        density_data: Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray],
        time_data: Optional[np.ndarray] = None,
        title: str = "Density Results",
        save_path: Optional[str] = None,
        show_plot: bool = True,
        density_scale: str = "10^18 m^-3",
        time_scale: str = "s",
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot density results with flexible data formats and scaling.

        Args:
            density_data: Density data in various formats
            time_data: Optional time array
            title: Plot title
            save_path: Optional path to save figure
            show_plot: Whether to display the plot
            density_scale: Density scale ('m^-2', 'm^-3', '10^18 m^-2', etc.)
            time_scale: Time scale ('s', 'ms', 'us', 'ns')

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        # Prepare data
        if isinstance(density_data, pd.DataFrame):
            if time_data is None and hasattr(density_data, "index"):
                time_data = density_data.index
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
            signals = {"Density": density_data}

        # Apply time and density scaling
        time_scaled, _, time_label, _ = self._apply_scaling(
            time_data, {"dummy": np.zeros_like(time_data)}, time_scale
        )
        _, density_scaled, _, density_label = self._apply_scaling(
            time_data, signals, time_scale, density_scale
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        for name, data in density_scaled.items():
            ax.plot(time_scaled, data, label=name)

        ax.set_xlabel(time_label, **FontStyle.label)
        ax.set_ylabel(f"LID {density_label}", **FontStyle.label)
        ax.set_title(title, **FontStyle.title)
        ax.grid(True, alpha=0.3)

        if len(signals) > 1:
            ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show(block=False)

        return fig, ax

    def plot_response(
        self,
        freqs: np.ndarray,
        responses: np.ndarray,
        title: str = "Frequency Response",
        save_path: Optional[str] = None,
        show_plot: bool = True,
        freq_scale: str = "Hz",
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot frequency response functions.

        Args:
            freqs: Frequency array
            responses: Response array
            title: Plot title
            save_path: Optional path to save figure
            show_plot: Whether to display the plot
            freq_scale: Frequency scale ('Hz', 'kHz', 'MHz', 'GHz')

        Returns:
            fig, ax: Matplotlib figure and axis
        """
        # Frequency scaling
        freq_scale_factor = 1
        if freq_scale == "kHz":
            freq_scale_factor = 1e-3
        elif freq_scale == "MHz":
            freq_scale_factor = 1e-6
        elif freq_scale == "GHz":
            freq_scale_factor = 1e-9

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs * freq_scale_factor, 20 * np.log10(np.abs(responses)))
        ax.set_ylim(-100, 5)
        ax.grid(True)
        ax.set_xlabel(f"Frequency [{freq_scale}]", **FontStyle.label)
        ax.set_ylabel("Gain [dB]", **FontStyle.label)
        ax.set_title(title, **FontStyle.title)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show(block=False)

        return fig, ax

    def plot_shot_overview(
        self,
        shot_data: dict,
        vest_data: pd.DataFrame,
        results_dir: Path,
        shot_num: int,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Create a comprehensive overview plot for a shot.

        Args:
            shot_data: Dictionary of DataFrames with shot data
            vest_data: VEST database data
            results_dir: Directory to save plots
            shot_num: Shot number for plot titles
            save_path: Optional path to save figure
            show_plot: Whether to display the plot

        Returns:
            fig, axes: Matplotlib figure and axes
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Shot {shot_num} - Overview", **FontStyle.title)

        # Plot 1: Raw waveforms (first available signal)
        if shot_data:
            first_df = list(shot_data.values())[0]
            if isinstance(first_df, pd.DataFrame) and "TIME" in first_df.columns:
                signal_cols = [col for col in first_df.columns if col != "TIME"]
                if signal_cols:
                    axes[0, 0].plot(
                        first_df.index.values * 1000, first_df[signal_cols[0]].values
                    )
                    axes[0, 0].set_title("Raw Signal", **FontStyle.title)
                    axes[0, 0].set_xlabel("Time [ms]", **FontStyle.label)
                    axes[0, 0].set_ylabel("Amplitude", **FontStyle.label)
                    axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: VEST data if available
        if vest_data is not None and not vest_data.empty:
            if "Ip_raw ([V])" in vest_data.columns:
                axes[0, 1].plot(
                    vest_data.index.values * 1000,
                    vest_data["Ip_raw ([V])"].values,
                    "r-",
                )
                axes[0, 1].set_title("Plasma Current", **FontStyle.title)
                axes[0, 1].set_xlabel("Time [ms]", **FontStyle.label)
                axes[0, 1].set_ylabel(r"$I_{p}$ [V]", **FontStyle.label)
                axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Density data if available
        density_data = {}
        for key, df in shot_data.items():
            if key.startswith("ne_") and isinstance(df, pd.Series):
                density_data[key] = df

        if density_data:
            for key, data in density_data.items():
                axes[1, 0].plot(data.index.values * 1000, data.values / 1e18, label=key)
            axes[1, 0].set_title("Density Evolution", **FontStyle.title)
            axes[1, 0].set_xlabel("Time [ms]", **FontStyle.label)
            axes[1, 0].set_ylabel(r"Density [$10^{18} m^{-3}$]", **FontStyle.label)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Summary statistics
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


# ============================================================================
# Shot-Specific Functions (Simplified)
# ============================================================================


def create_shot_results_directory(shot_num: int, base_dir: str = "ifi/results") -> Path:
    """Create results directory structure for a shot."""
    results_dir = Path(base_dir) / str(shot_num)
    subdirs = ["waveforms", "spectra", "density", "overview"]

    ensure_dir_exists(str(results_dir))
    for subdir in subdirs:
        ensure_dir_exists(str(results_dir / subdir))

    return results_dir


def plot_shot_waveforms(
    shot_data: dict, results_dir: Path, shot_num: int, downsample: int = 100
):
    """Generate waveform plots for all available signals in a shot."""
    logger = logging.getLogger(__name__)
    logger.info("Generating waveform plots...")

    plotter = Plotter()
    waveform_dir = results_dir / "waveforms"

    for filename, df in shot_data.items():
        if isinstance(df, pd.DataFrame) and "TIME" in df.columns:
            logger.info(f"Plotting waveforms for {filename}")

            try:
                fig, axes = plotter.plot_waveforms(
                    df,
                    title=f"Shot {shot_num} - {filename}",
                    downsample=downsample,
                    show_plot=False,
                )

                output_path = waveform_dir / f"{filename}_waveforms.png"
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                logger.info(f"Saved waveform plot: {output_path}")

            except Exception as e:
                logger.error(f"Failed to plot waveforms for {filename}: {e}")


def plot_shot_spectrograms(
    shot_data: dict, results_dir: Path, shot_num: int, max_channels: int = 2
):
    """Generate spectrogram plots using STFT analysis for a shot."""
    logger = logging.getLogger(__name__)
    logger.info("Generating spectrogram plots...")

    plotter = Plotter()
    spectra_dir = results_dir / "spectra"

    for filename, df in shot_data.items():
        if isinstance(df, pd.DataFrame) and "TIME" in df.columns:
            time_diff = (
                np.diff(df.index.values[:1000]).mean()
                if hasattr(df.index, "values")
                else np.diff(df["TIME"].values[:1000]).mean()
            )
            fs = 1.0 / time_diff

            logger.info(
                f"Processing spectrogram for {filename} (fs = {fs / 1e6:.1f} MHz)"
            )

            signal_cols = [col for col in df.columns if col != "TIME"]

            for col in signal_cols[:max_channels]:
                try:
                    signal = df[col].values[::100]
                    fs_down = fs / 100

                    fig, axes = plotter.plot_time_frequency(
                        signal,
                        method="stft",
                        fs=fs_down,
                        title=f"Shot {shot_num} - {filename} - {col}",
                        show_plot=False,
                        nperseg=1024,
                        noverlap=512,
                    )

                    output_path = spectra_dir / f"{filename}_{col}_spectrogram.png"
                    fig.savefig(output_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    logger.info(f"Saved spectrogram: {output_path}")

                except Exception as e:
                    logger.error(
                        f"Failed to generate spectrogram for {filename}_{col}: {e}"
                    )


def plot_shot_density_evolution(
    shot_data: dict, vest_data: pd.DataFrame, results_dir: Path, shot_num: int
):
    """Plot density evolution with VEST data overlay for a shot."""
    logger = logging.getLogger(__name__)
    logger.info("Generating density evolution plots...")

    plotter = Plotter()
    density_dir = results_dir / "density"

    density_data = {}
    for key, df in shot_data.items():
        if key.startswith("ne_") and isinstance(df, pd.Series):
            density_data[key] = df

    if not density_data:
        logger.warning("No density data found for plotting")
        return

    try:
        density_df = pd.DataFrame(density_data)

        fig, ax = plotter.plot_density(
            density_df, title=f"Shot {shot_num} - Density Evolution", show_plot=False
        )

        output_path = density_dir / "density_evolution.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved density evolution plot: {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate density evolution plot: {e}")


# ============================================================================
# Legacy Functions (for backward compatibility)
# ============================================================================

# Alias for backward compatibility
ifion_plotting = interactive_plotting


# Legacy functions that now use the Plotter class
def plot_waveforms(data, **kwargs):
    """Legacy function - now uses Plotter class."""
    plotter = Plotter()
    return plotter.plot_waveforms(data, **kwargs)


def plot_spectrogram(freqs, times, Sxx, **kwargs):
    """Legacy function - now uses Plotter class."""
    plotter = Plotter()
    # Create a dummy signal for the unified interface
    dummy_data = np.zeros((len(times), 1))
    return plotter.plot_time_frequency(dummy_data, method="stft", **kwargs)


def plot_density_results(density_data, **kwargs):
    """Legacy function - now uses Plotter class."""
    plotter = Plotter()
    return plotter.plot_density(density_data, **kwargs)


def plot_signals(data_dict, **kwargs):
    """Legacy function - now uses Plotter class."""
    plotter = Plotter()
    for name, df in data_dict.items():
        plotter.plot_waveforms(df, title=name, **kwargs)


def plot_spectrograms(stft_results, **kwargs):
    """Legacy function - now uses Plotter class."""
    plotter = Plotter()
    for filename, results_by_col in stft_results.items():
        for col_name, results in results_by_col.items():
            plotter.plot_time_frequency(results["Zxx"], method="stft", **kwargs)


def plot_cwt(cwt_results, **kwargs):
    """Legacy function - now uses Plotter class."""
    plotter = Plotter()
    for filename, analysis in cwt_results.items():
        for col_name, result in analysis.items():
            plotter.plot_time_frequency(result["cwt_matrix"], method="cwt", **kwargs)


def plot_response(freqs, responses, **kwargs):
    """Legacy function - now uses Plotter class."""
    plotter = Plotter()
    return plotter.plot_response(freqs, responses, **kwargs)


def plot_shot_overview(shot_data, vest_data, results_dir, shot_num, **kwargs):
    """Legacy function - now uses Plotter class."""
    plotter = Plotter()
    return plotter.plot_shot_overview(
        shot_data, vest_data, results_dir, shot_num, **kwargs
    )


# ============================================================================
# Utility Functions
# ============================================================================


def load_cached_shot_data(shot_num: int, cache_base_dir: str = "cache") -> dict:
    """Load cached shot data from HDF5 files."""
    logger = logging.getLogger(__name__)

    cache_dir = Path(cache_base_dir) / str(shot_num)
    if not cache_dir.exists():
        logger.error(f"No cached data found for shot {shot_num}")
        return None

    h5_files = list(cache_dir.glob("*.h5"))

    if not h5_files:
        logger.error(f"No HDF5 files found in {cache_dir}")
        return None

    cached_data = {}

    for h5_file in h5_files:
        logger.info(f"Loading cached data from {h5_file}")

        try:
            with pd.HDFStore(h5_file, "r") as store:
                for key in store.keys():
                    df = pd.read_hdf(h5_file, key)
                    clean_key = key.lstrip("/")
                    cached_data[clean_key] = df
                    logger.info(f"Loaded dataset '{clean_key}' with shape {df.shape}")

        except Exception as e:
            logger.error(f"Failed to load {h5_file}: {e}")

    return cached_data


if __name__ == "__main__":
    print("=" * 60)
    print("ifi.analysis.plots - Plotting Module Demo")
    print("=" * 60)

    # Create some test data
    t = np.linspace(0, 1, 1000)
    data = {
        "Signal 1": np.sin(2 * np.pi * 10 * t),
        "Signal 2": np.cos(2 * np.pi * 5 * t),
    }

    # Use the unified Plotter class
    plotter = Plotter()

    print("\nCreating demonstration plots...")
    print("(Plots will be saved to 'demo_plots' directory)")

    # Create demo directory
    demo_dir = Path("demo_plots")
    demo_dir.mkdir(exist_ok=True)

    try:
        # Example 1: Waveform plotting with different scaling
        print("\n1. Waveform plotting with different scaling...")
        fig1, axes1 = plotter.plot_waveforms(
            data,
            title="Test Signals",
            time_scale="ms",
            signal_scale="mV",
            show_plot=False,
            save_path=demo_dir / "waveforms_mV.png",
        )

        # Example 2: Time-frequency analysis
        print("2. Time-frequency analysis (STFT)...")
        signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        fig2, axes2 = plotter.plot_time_frequency(
            signal,
            method="stft",
            fs=1000,
            title="Test STFT",
            show_plot=False,
            save_path=demo_dir / "stft_analysis.png",
            nperseg=256,
            noverlap=128,
        )

        # Example 3: Pre-computed arrays
        print("3. Pre-computed time-frequency arrays...")
        freqs = np.linspace(0, 50, 100)
        times = np.linspace(0, 1, 200)
        Sxx = np.random.rand(100, 200) * np.exp(
            -((freqs[:, None] - 25) ** 2 + (times[None, :] - 0.5) ** 2) / 0.1
        )
        fig3, axes3 = plotter.plot_time_frequency(
            (freqs, times, Sxx),
            method="precomputed",
            title="Pre-computed STFT",
            show_plot=False,
            save_path=demo_dir / "precomputed_stft.png",
        )

        # Example 4: Density plotting with LaTeX formatting
        print("4. Density plotting with LaTeX formatting...")
        density_data = {
            "ne_1": np.random.rand(1000) * 1e18,
            "ne_2": np.random.rand(1000) * 2e18,
        }
        fig4, ax4 = plotter.plot_density(
            density_data,
            density_scale="10^18 m^-3",
            title="Density Evolution",
            show_plot=False,
            save_path=demo_dir / "density_evolution.png",
        )

        # Example 5: LaTeX superscript formatting
        print("5. LaTeX superscript formatting...")
        fig5, ax5 = plotter.plot_waveforms(
            data,
            signal_scale="m^-2",
            title="Test with LaTeX superscript",
            show_plot=False,
            save_path=demo_dir / "latex_superscript.png",
        )

        print("\nAll demo plots created successfully!")
        print(f"Check the '{demo_dir}' directory for saved plots")
        print("\nKey Features Demonstrated:")
        print("   • Unified Plotter class with flexible data formats")
        print("   • Time and signal scaling with different units")
        print("   • Time-frequency analysis (STFT/CWT)")
        print("   • Pre-computed array plotting")
        print("   • Scientific notation support (10^18 m^-3)")
        print("   • LaTeX formatting for superscripts (m^-2 → m⁻²)")
        print("   • Automatic figure saving and management")

    except Exception as e:
        print(f"\nError during demo: {e}")
        print("This might be due to missing dependencies or display issues.")

    print("\n" + "=" * 60)
    print("Demo completed! Use 'python -m ifi.analysis.plots' to run this demo again.")
    print("=" * 60)
