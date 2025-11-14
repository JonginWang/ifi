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

Classes:
    Plotter: Unified plotting class with consolidated functionality.

Functions:
    Interactive Mode:
        setup_interactive_mode: Setup matplotlib for optimal interactive use.
        interactive_plotting: Enhanced context manager for interactive plotting.
    Plotting Functions:
        plot_shot_waveforms: Plot the waveforms for a shot.
        plot_shot_spectrograms: Plot the spectrograms for a shot.
        plot_shot_density_evolution: Plot the density evolution for a shot.
    Legacy Plotting Functions:
        plot_waveforms: Plot the waveforms for a shot.
        plot_spectrogram: Plot the spectrogram for a shot.
        plot_density_results: Plot the density results for a shot.
        plot_signals: Plot the signals for a shot.
        plot_spectrograms: Plot the spectrograms for a shot.
        plot_cwt: Plot the CWT for a shot.
        plot_response: Plot the response for a shot.
        plot_shot_overview: Plot the shot overview for a shot.
        plot_analysis_overview: Plot comprehensive analysis overview for a shot.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from contextlib import contextmanager
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
try:
    from ..db_controller.vest_db import VEST_DB
    from .spectrum import SpectrumAnalysis
    from .params.params_plot import set_plot_style, FontStyle
    from ..utils.common import LogManager, ensure_dir_exists, log_tag
    from .functions.power_conversion import pow2db, amp2db
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.db_controller.vest_db import VEST_DB
    from ifi.analysis.spectrum import SpectrumAnalysis
    from ifi.analysis.params.params_plot import set_plot_style, FontStyle
    from ifi.utils.common import LogManager, ensure_dir_exists, log_tag
    from ifi.analysis.functions.power_conversion import pow2db, amp2db

logger = LogManager().get_logger(__name__)

"""
    Interactive Mode Setup and Management
"""


def _is_jupyter_environment() -> bool:
    """
    Detect if code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in Jupyter notebook, False otherwise
    """
    try:
        # Method 1: Try to import and get IPython instance
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None:
                # Check if running in Jupyter notebook (not just IPython)
                if hasattr(ipython, "kernel"):
                    return True

                # Check class name (Jupyter uses ZMQInteractiveShell)
                class_name = ipython.__class__.__name__
                if "ZMQ" in class_name or "Jupyter" in class_name:
                    return True
        except (ImportError, NameError):
            pass

        # Method 2: Check for Jupyter-specific environment variables
        import os

        if os.environ.get("JPY_PARENT_PID") is not None:
            return True

        # Method 3: Check if running in IPython/Jupyter by checking __main__
        import sys

        if hasattr(sys, "ps1") and sys.ps1:
            # Interactive shell, but not necessarily Jupyter
            pass
        elif "ipykernel" in sys.modules:
            # ipykernel is loaded, likely Jupyter
            return True

    except Exception:
        pass

    return False


def setup_interactive_mode(backend: str = "auto", style: str = "default"):
    """
    Setup matplotlib for optimal interactive use.

    Args:
        backend(str): The backend to use for the plots.
        style(str): The style to use for the plots.

    Returns:
        None

    Examples:
    ```python
    from .plots import setup_interactive_mode
    setup_interactive_mode(backend="TkAgg", style="default")
    ```
    """
    if backend == "auto":
        # Check if running in Jupyter notebook first
        if _is_jupyter_environment():
            try:
                # Try JupyterLab inline backend first
                matplotlib.use("module://matplotlib_inline.backend_inline")
            except (ImportError, ValueError):
                try:
                    # Fall back to classic inline backend
                    matplotlib.use("inline")
                except (ImportError, ValueError):
                    # If inline backends are not available, use Agg (non-interactive)
                    matplotlib.use("Agg")
        else:
            # Non-Jupyter environment: try interactive backends
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
        show_plots(bool): Whether to show the plots
        save_dir(Optional[str]): Directory to save the plots
        save_prefix(str): Prefix for the saved plots
        save_ext(str): Extension for the saved plots
        dpi(int): DPI for the saved plots
        block(bool): Whether to block the plots

    Returns:
        None

    Examples:
    ```python
    from .plots import interactive_plotting

    with interactive_plotting(show_plots=True, save_dir="plots"):
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("Interactive")
        plt.show()
    """
    original_backend = matplotlib.get_backend()
    original_interactive = plt.isinteractive()
    is_jupyter = _is_jupyter_environment()

    try:
        if show_plots:
            setup_interactive_mode()
            # In Jupyter, don't use plt.ion() as inline backend handles display automatically
            if not is_jupyter:
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
                    logger.info(f"{log_tag('ION', 'SAVE')} Saved figure to {filepath}")
                except Exception as e:
                    logger.error(
                        f"{log_tag('ION', 'ERROR')} Failed to save figure {i}: {e}"
                    )

        if show_plots:
            # In Jupyter, inline backend automatically displays plots
            # Only call plt.show() if not in Jupyter or if explicitly requested
            if is_jupyter:
                # In Jupyter, just display the figure (inline backend handles it)
                for i in plt.get_fignums():
                    fig = plt.figure(i)
                    plt.show(block=False)
            else:
                # Non-Jupyter: restore interactive state and show plots
                plt.ioff()
                if block:
                    plt.show(block=True)
                else:
                    plt.show(block=False)

        matplotlib.use(original_backend)
        if not is_jupyter:
            plt.interactive(original_interactive)


"""
    Unified Plotter Class
"""


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
            data(Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]):
                Input data in various formats
            fs(Optional[float]): Sampling frequency (optional)

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]:
                Tuple containing the time array and the dictionary of signal arrays
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
                logger.error(
                    f"{log_tag('PLTTR', 'PREPT')} Waveform data must be 1D or 2D numpy array. Got {data.shape}"
                )
                raise ValueError("Waveform data must be 1D or 2D numpy array")
        else:
            logger.error(
                f"{log_tag('PLTTR', 'PREPT')} Waveform data must be DataFrame, dict, or numpy array. Got {type(data)}"
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
            time(np.ndarray): Time array
            signals(Dict[str, np.ndarray]): Signal dictionary
            time_scale(str): Time scale ('s', 'ms', 'us', 'ns')
            signal_scale(str): Signal scale ('V', 'mV', 'uV', 'a.u.', '10^18 m^-3', etc.)

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray], str, str]:
                Tuple containing the scaled time array,
                the scaled signals,
                the time axis label,
                and the signal axis label
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
            signal_scale_factor = 10 ** (-exponent)

            # Convert superscript notation to LaTeX (without $ delimiters for now)
            def replace_superscript(match):
                base = match.group(1)
                exp = match.group(2)
                return f"{base}^{{-{exp}}}"

            unit_latex = re.sub(r"(\w+)\^-(\d+)", replace_superscript, unit)
            # Include the 10^exponent part in the label with proper LaTeX formatting
            signal_label = f"[$10^{{{exponent}}} {unit_latex}$]"
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
            data(Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]):
                Input data that may contain metadata

        Returns:
            str: Formatted metadata string for display
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
            data(Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]):
                Input data (DataFrame, dict, or numpy array)
            fs(Optional[float]): Sampling frequency (optional)
            title(str): Plot title
            downsample(int): Downsampling factor
            save_path(Optional[str]): Optional path to save figure
            show_plot(bool): Whether to display the plot
            time_scale(str): Time scale ('s', 'ms', 'us', 'ns')
            signal_scale(str): Signal scale ('V', 'mV', 'a.u.', etc.)
            trigger_time(float): Trigger time offset

        Returns:
            Tuple[Figure, np.ndarray]:
                Tuple containing the matplotlib figure and the axes
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
            # In Jupyter, inline backend automatically displays plots
            # plt.show() is still called to ensure display, but block=False
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
            data(Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]):
                Input data or pre-computed arrays (freqs, times, Sxx/Zxx)
            method(str): Analysis method ('stft', 'cwt', or 'precomputed')
            fs(Optional[float]): Sampling frequency (not needed for precomputed data)
            title(str): Plot title
            save_path(Optional[str]): Optional path to save figure
            show_plot(bool): Whether to display the plot
            time_scale(str): Time scale ('s', 'ms', 'us', 'ns')
            freq_scale(str): Frequency scale ('Hz', 'kHz', 'MHz', 'GHz')
            power_scale(str): Power scale ('linear', 'dB')
            trigger_time(float): Trigger time offset
            downsample(int): Downsampling factor
            **kwargs: Additional arguments for STFT/CWT computation

        Returns:
            Tuple[Figure, np.ndarray]:
                Tuple containing the matplotlib figure and the axes
        """
        # Check if data is pre-computed arrays (freqs, times, Sxx/Zxx)
        if isinstance(data, tuple) and len(data) == 3:
            freqs, times, Sxx = data
            method = "precomputed"
            n_signals = 1
            signals = {"Precomputed": None}  # Dummy for loop structure
            time = times  # Set time for consistency
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

            # Ensure n_signals is at least 1
            if n_signals == 0:
                raise ValueError(
                    "No signals found in data. Please provide data with at least one signal channel."
                )

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
                    Sxx_plot = pow2db(np.abs(Sxx), dbm=False)
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
                    cmap="plasma",  # default colormap
                )

            elif method.lower() == "stft":
                # STFT analysis
                freqs, times, Sxx = self.analyzer.compute_stft(signal, fs, **kwargs)

                # Scale frequency and power
                freqs_scaled = freqs * freq_scale_factor
                if power_scale == "dB":
                    Sxx_plot = pow2db(np.abs(Sxx), dbm=False)
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
                    cmap="plasma",  # default colormap
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
            # In Jupyter, inline backend automatically displays plots
            # plt.show() is still called to ensure display, but block=False
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
        probe_amplitude: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        color_by_amplitude: bool = False,
        amplitude_colormap: str = "coolwarm",
        amplitude_impedance: float = 50,
        amplitude_dbm: bool = True,
        amplitude_unit: str = "V",
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot density results with flexible data formats and scaling.

        This function can optionally color-code the density plot based on probe signal
        amplitude. When `color_by_amplitude=True`, each data point is colored according
        to its corresponding amplitude value, creating a visual representation of signal
        strength alongside density evolution.

        Args:
            density_data (Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]):
                Density data in various formats. If DataFrame, columns are signal names.
                If dict, keys are signal names and values are arrays.
                If array, treated as single "Density" signal.
            time_data (Optional[np.ndarray]): Optional time array. If None, inferred from
                density_data index or generated automatically.
            title (str): Plot title. Default is "Density Results".
            save_path (Optional[str]): Optional path to save figure.
            show_plot (bool): Whether to display the plot. Default is True.
            density_scale (str): Density scale ('m^-2', 'm^-3', '10^18 m^-2', etc.).
                Default is "10^18 m^-3".
            time_scale (str): Time scale ('s', 'ms', 'us', 'ns'). Default is "s".
            probe_amplitude (Optional[Union[np.ndarray, Dict[str, np.ndarray]]]):
                Probe signal amplitude in volts [V] for color-coding. If None and
                color_by_amplitude=True, amplitude-based coloring is disabled.
                - If np.ndarray: Single amplitude array (applied to all signals or first signal).
                - If Dict[str, np.ndarray]: Amplitude arrays keyed by signal name.
            color_by_amplitude (bool): If True, color-code plot lines by amplitude.
                Default is False.
            amplitude_colormap (str): Matplotlib colormap name for amplitude coloring.
                Default is "coolwarm". Options: 'viridis', 'plasma', 'inferno', 'coolwarm',
                'RdYlBu', etc.
            amplitude_impedance (float): System impedance in ohms [Ω] for amplitude-to-dB
                conversion. Default is 50 Ω.
            amplitude_dbm (bool): If True, convert amplitude to dBm [dBm] for color mapping.
                If False, convert to dB [dB]. Default is True.
            amplitude_unit (str): Unit of the amplitude. Default is "mV".
                Options: 'mV', 'V', 'dBm', 'dB'. (default is "V")

        Returns:
            Tuple[Figure, np.ndarray]:
                Tuple containing the matplotlib figure and the axis.

        Notes:
            - When `color_by_amplitude=True`, the plot uses LineCollection to create
              color-coded line segments, where each segment's color represents the amplitude
              at that point.
            - Amplitude values are converted to dB/dBm using `amp2db` from `power_conversion.py`.
            - The colormap is normalized to the amplitude range (min to max).
            - If `probe_amplitude` is provided as a single array but multiple density signals
              exist, the same amplitude is used for all signals.

        Examples:
            ```python
            import numpy as np
            import pandas as pd
            from ifi.analysis.plots import Plotter

            plotter = Plotter()

            # Basic density plot
            density = np.random.rand(1000) * 1e18
            time = np.linspace(0, 1, 1000)
            fig, ax = plotter.plot_density(density, time_data=time)

            # Density plot with amplitude-based coloring
            probe_amp = np.random.rand(1000) * 0.1  # 0-0.1 V
            fig, ax = plotter.plot_density(
                density,
                time_data=time,
                probe_amplitude=probe_amp,
                color_by_amplitude=True,
                amplitude_colormap="plasma"
            )

            # Multiple signals with different amplitudes
            density_dict = {
                "CH1": np.random.rand(1000) * 1e18,
                "CH2": np.random.rand(1000) * 1e18
            }
            amp_dict = {
                "CH1": np.random.rand(1000) * 0.1,
                "CH2": np.random.rand(1000) * 0.15
            }
            fig, ax = plotter.plot_density(
                density_dict,
                probe_amplitude=amp_dict,
                color_by_amplitude=True
            )
            ```
        """
        # Prepare data
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

        # Ensure time_data is numpy array
        time_data = np.asarray(time_data)

        # Apply time and density scaling
        time_scaled, _, time_label, _ = self._apply_scaling(
            time_data, {"dummy": np.zeros_like(time_data)}, time_scale
        )
        _, density_scaled, _, density_label = self._apply_scaling(
            time_data, signals, time_scale, density_scale
        )

        # Prepare amplitude data if color-coding is enabled
        amplitude_data = None
        if color_by_amplitude and probe_amplitude is not None:
            if isinstance(probe_amplitude, dict):
                # Match amplitude arrays to signal names
                amplitude_data = {}
                for name in signals.keys():
                    if name in probe_amplitude:
                        amplitude_data[name] = np.asarray(probe_amplitude[name])
                    elif len(probe_amplitude) == 1:
                        # Use single amplitude array for all signals
                        amplitude_data[name] = np.asarray(list(probe_amplitude.values())[0])
            else:
                # Single amplitude array - apply to all signals or first signal
                amp_array = np.asarray(probe_amplitude)
                if len(signals) == 1:
                    amplitude_data = {list(signals.keys())[0]: amp_array}
                else:
                    # Apply same amplitude to all signals
                    amplitude_data = {name: amp_array for name in signals.keys()}

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each signal
        for name, data in density_scaled.items():
            # Ensure data and time have same length
            min_len = min(len(time_scaled), len(data))
            time_plot = time_scaled[:min_len]
            data_plot = data[:min_len]

            if color_by_amplitude and amplitude_data is not None and name in amplitude_data:
                # Color-code by amplitude
                amp_array = amplitude_data[name]
                # Ensure amplitude array matches data length
                min_amp_len = min(len(amp_array), min_len)
                amp_plot = amp_array[:min_amp_len]
                time_plot = time_plot[:min_amp_len]
                data_plot = data_plot[:min_amp_len]

                # Convert amplitude to dB/dBm for color mapping
                amp_pow = amp2db(np.abs(amp_plot), impedance=amplitude_impedance, dbm=amplitude_dbm) if 1==0 else np.abs(amp_plot)

                # Create line segments for color mapping
                points = np.array([time_plot, data_plot]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create LineCollection with color mapping
                lc = LineCollection(
                    segments,
                    cmap=plt.get_cmap(amplitude_colormap),
                    norm=plt.Normalize(vmin=amp_pow.min(), vmax=amp_pow.max()),
                )
                # Set color for each segment based on amplitude
                lc.set_array(amp_pow[:-1])  # Use amplitude at start of each segment
                lc.set_linewidth(2)
                line = ax.add_collection(lc)

                # Add colorbar
                cbar = plt.colorbar(line, ax=ax)
                if amplitude_dbm:
                    amp_unit = "dBm" if amplitude_unit == "dBm" else "V"
                else:
                    amp_unit = "dB" if amplitude_unit == "dB" else "V"
                cbar.set_label(f"Probe Amplitude [{amp_unit}]", **FontStyle.label)
            else:
                # Standard line plot
                ax.plot(time_plot, data_plot, label=name, linewidth=2)

        ax.set_xlabel(time_label, **FontStyle.label)
        ax.set_ylabel(f"LID {density_label}", **FontStyle.label)
        ax.set_title(title, **FontStyle.title)
        ax.grid(True, alpha=0.3)

        if len(signals) > 1 and not color_by_amplitude:
            ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            # In Jupyter, inline backend automatically displays plots
            # plt.show() is still called to ensure display, but block=False
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
            freqs(np.ndarray): Frequency array
            responses(np.ndarray): Response array
            title(str): Plot title
            save_path(Optional[str]): Optional path to save figure
            show_plot(bool): Whether to display the plot
            freq_scale(str): Frequency scale ('Hz', 'kHz', 'MHz', 'GHz')

        Returns:
            Tuple[Figure, np.ndarray]:
                Tuple containing the matplotlib figure and the axis
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
            # In Jupyter, inline backend automatically displays plots
            # plt.show() is still called to ensure display, but block=False
            plt.show(block=False)

        return fig, ax

    def plot_shot_overview(
        self,
        shot_data: dict,
        vest_data: pd.DataFrame,
        shot_num: int,
        results_dir: Path = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Create a comprehensive overview plot for a shot.

        Args:
            shot_data(dict): Dictionary of DataFrames with shot data
            vest_data(pd.DataFrame): VEST database data
            shot_num(int): Shot number for plot titles
            results_dir(Path): Directory to save plots (auto-generated if None)
            save_path(Optional[str]): Optional path to save figure
            show_plot(bool): Whether to display the plot

        Returns:
            Tuple[Figure, np.ndarray]:
                Tuple containing the matplotlib figure and the axes
        """
        # Auto-generate results directory if not provided
        if results_dir is None:
            results_dir = Path("results") / f"{shot_num}"

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
            # In Jupyter, inline backend automatically displays plots
            # plt.show() is still called to ensure display, but block=False
            plt.show(block=False)

        return fig, axes


def plot_shot_waveforms(
    shot_data: dict, shot_num: int, downsample: int = 100, results_dir: Path = None
):
    """Generate waveform plots for all available signals in a shot.

    Args:
        shot_data(dict): Dictionary of DataFrames with shot data
        shot_num(int): Shot number
        downsample(int): Downsample factor
        results_dir(Path): Directory to save plots (auto-generated if None)

    Returns:
        None
    """
    logger.info(f"{log_tag('PLOTS', 'WFDAT')} Generating waveform plots...")

    # Auto-generate results directory if not provided
    if results_dir is None:
        results_dir = Path("results") / f"{shot_num}"

    plotter = Plotter()
    ensure_dir_exists(str(results_dir))

    for filename, df in shot_data.items():
        if isinstance(df, pd.DataFrame) and "TIME" in df.columns:
            logger.info(
                f"{log_tag('PLOTS', 'WFDAT')} Plotting waveforms for {filename}"
            )

            try:
                fig, axes = plotter.plot_waveforms(
                    df,
                    title=f"Shot {shot_num} - {filename}",
                    downsample=downsample,
                    show_plot=False,
                )

                output_path = results_dir / f"{filename}_waveforms.png"
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(
                    f"{log_tag('PLOTS', 'WFDAT')} Saved waveform plot: {output_path}"
                )

            except Exception as e:
                logger.error(
                    f"{log_tag('PLOTS', 'WFDAT')} Failed to plot waveforms for {filename}: {e}"
                )


def plot_shot_spectrograms(
    shot_data: dict, shot_num: int, max_channels: int = 2, results_dir: Path = None
):
    """Generate spectrogram plots using STFT analysis for a shot.

    Args:
        shot_data(dict): Dictionary of DataFrames with shot data
        shot_num(int): Shot number
        max_channels(int): Maximum number of channels to plot
        results_dir(Path): Directory to save plots (auto-generated if None)

    Returns:
        None
    """
    logger.info(f"{log_tag('PLOTS', 'SPCTR')} Generating spectrogram plots...")

    # Auto-generate results directory if not provided
    if results_dir is None:
        results_dir = Path("results") / f"{shot_num}"

    plotter = Plotter()
    ensure_dir_exists(str(results_dir))

    for filename, df in shot_data.items():
        if isinstance(df, pd.DataFrame) and "TIME" in df.columns:
            time_diff = (
                np.diff(df.index.values[:1000]).mean()
                if hasattr(df.index, "values")
                else np.diff(df["TIME"].values[:1000]).mean()
            )
            fs = 1.0 / time_diff

            logger.info(
                f"{log_tag('PLOTS', 'SPCTR')} Processing spectrogram for {filename} (fs = {fs / 1e6:.1f} MHz)"
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

                    output_path = results_dir / f"{filename}_{col}_spectrogram.png"
                    fig.savefig(output_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    logger.info(
                        f"{log_tag('PLOTS', 'SPCTR')} Saved spectrogram: {output_path}"
                    )

                except Exception as e:
                    logger.error(
                        f"{log_tag('PLOTS', 'SPCTR')} Failed to generate spectrogram for {filename}_{col}: {e}"
                    )


def plot_shot_density_evolution(
    shot_data: dict, vest_data: pd.DataFrame, shot_num: int, results_dir: Path = None
):
    """Plot density evolution with VEST data overlay for a shot.

    Args:
        shot_data(dict): Dictionary of DataFrames with shot data
        vest_data(pd.DataFrame): VEST database data
        shot_num(int): Shot number
        results_dir(Path): Directory to save plots (auto-generated if None)

    Returns:
        None
    """
    logger.info(f"{log_tag('PLOTS', 'DENS')} Generating density evolution plots...")

    # Auto-generate results directory if not provided
    if results_dir is None:
        results_dir = Path("results") / f"{shot_num}"

    plotter = Plotter()
    ensure_dir_exists(str(results_dir))

    density_data = {}
    for key, df in shot_data.items():
        if key.startswith("ne_") and isinstance(df, pd.Series):
            density_data[key] = df

    if not density_data:
        logger.warning(f"{log_tag('PLOTS', 'DENS')} No density data found for plotting")
        return

    try:
        density_df = pd.DataFrame(density_data)

        fig, ax = plotter.plot_density(
            density_df, title=f"Shot {shot_num} - Density Evolution", show_plot=False
        )

        output_path = results_dir / "density_evolution.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            f"{log_tag('PLOTS', 'DENS')} Saved density evolution plot: {output_path}"
        )

    except Exception as e:
        logger.error(
            f"{log_tag('PLOTS', 'DENS')} Failed to generate density evolution plot: {e}"
        )


"""
    Legacy Functions (for backward compatibility):
"""

# Alias for backward compatibility
ifion_plotting = interactive_plotting


# Legacy functions that now use the Plotter class with interactive plotting
def plot_waveforms(data, **kwargs):
    """Legacy function - now uses Plotter class with interactive plotting."""
    with interactive_plotting(show_plots=True, block=False):
        plotter = Plotter()
        return plotter.plot_waveforms(data, **kwargs)


def plot_spectrogram(freqs, times, Sxx, **kwargs):
    """Legacy function - now uses Plotter class with interactive plotting."""
    with interactive_plotting(show_plots=True, block=False):
        plotter = Plotter()
        # Pass pre-computed arrays directly as tuple
        return plotter.plot_time_frequency(
            (freqs, times, Sxx), method="precomputed", **kwargs
        )


def plot_density_results(density_data, **kwargs):
    """Legacy function - now uses Plotter class with interactive plotting."""
    with interactive_plotting(show_plots=True, block=False):
        plotter = Plotter()
        return plotter.plot_density(density_data, **kwargs)


def plot_signals(data_dict, **kwargs):
    """Legacy function - now uses Plotter class with interactive plotting."""
    with interactive_plotting(show_plots=True, block=False):
        plotter = Plotter()
        for name, df in data_dict.items():
            plotter.plot_waveforms(df, title=name, **kwargs)


def plot_spectrograms(stft_results, **kwargs):
    """Legacy function - now uses Plotter class with interactive plotting."""
    with interactive_plotting(show_plots=True, block=False):
        plotter = Plotter()
        for filename, results_by_col in stft_results.items():
            for col_name, results in results_by_col.items():
                # Handle STFT results format: {freq_STFT, time_STFT, STFT_matrix}
                if (
                    "STFT_matrix" in results
                    and "freq_STFT" in results
                    and "time_STFT" in results
                ):
                    # Pre-computed STFT: use precomputed method
                    freqs = results["freq_STFT"]
                    times = results["time_STFT"]
                    stft_matrix = results["STFT_matrix"]
                    plotter.plot_time_frequency(
                        (freqs, times, stft_matrix),
                        method="precomputed",
                        title=kwargs.get("title_prefix", "")
                        + f"{Path(filename).name} - {col_name}",
                        **{k: v for k, v in kwargs.items() if k != "title_prefix"},
                    )
                elif "Zxx" in results:
                    # Old format: try to extract freqs and times if available
                    stft_data = results["Zxx"]
                    if "freqs" in results and "times" in results:
                        plotter.plot_time_frequency(
                            (results["freqs"], results["times"], stft_data),
                            method="precomputed",
                            title=kwargs.get("title_prefix", "")
                            + f"{Path(filename).name} - {col_name}",
                            **{k: v for k, v in kwargs.items() if k != "title_prefix"},
                        )
                    else:
                        logger.warning(
                            f"STFT data format incomplete for {filename}/{col_name}, skipping"
                        )
                        continue
                else:
                    logger.warning(
                        f"Unknown STFT data format for {filename}/{col_name}"
                    )
                    continue


def plot_cwt(cwt_results, **kwargs):
    """Legacy function - now uses Plotter class with interactive plotting."""
    with interactive_plotting(show_plots=True, block=False):
        plotter = Plotter()
        for filename, analysis in cwt_results.items():
            for col_name, result in analysis.items():
                # Handle CWT results format: {freq_CWT, time_CWT, CWT_matrix}
                if (
                    "CWT_matrix" in result
                    and "freq_CWT" in result
                    and "time_CWT" in result
                ):
                    # Pre-computed CWT: use precomputed method
                    freqs = result["freq_CWT"]
                    times = result["time_CWT"]
                    cwt_matrix = result["CWT_matrix"]
                    plotter.plot_time_frequency(
                        (freqs, times, cwt_matrix),
                        method="precomputed",
                        title=kwargs.get("title_prefix", "")
                        + f"{Path(filename).name} - {col_name}",
                        **{k: v for k, v in kwargs.items() if k != "title_prefix"},
                    )
                elif "cwt_matrix" in result:
                    # Old format: try to extract freqs and times if available
                    cwt_data = result["cwt_matrix"]
                    if "freqs" in result and "times" in result:
                        plotter.plot_time_frequency(
                            (result["freqs"], result["times"], cwt_data),
                            method="precomputed",
                            title=kwargs.get("title_prefix", "")
                            + f"{Path(filename).name} - {col_name}",
                            **{k: v for k, v in kwargs.items() if k != "title_prefix"},
                        )
                    else:
                        logger.warning(
                            f"CWT data format incomplete for {filename}/{col_name}, skipping"
                        )
                        continue
                else:
                    logger.warning(f"Unknown CWT data format for {filename}/{col_name}")
                    continue


def plot_response(freqs, responses, **kwargs):
    """Legacy function - now uses Plotter class with interactive plotting."""
    with interactive_plotting(show_plots=True, block=False):
        plotter = Plotter()
        return plotter.plot_response(freqs, responses, **kwargs)


def plot_shot_overview(shot_data, vest_data, results_dir, shot_num, **kwargs):
    """Legacy function - now uses Plotter class with interactive plotting."""
    with interactive_plotting(show_plots=True, block=False):
        plotter = Plotter()
        return plotter.plot_shot_overview(
            shot_data, vest_data, results_dir, shot_num, **kwargs
        )


def plot_analysis_overview(
    shot_num: int,
    signals_dict: Dict[str, pd.DataFrame],
    density_dict: Dict[str, pd.DataFrame],
    vest_data: Optional[pd.DataFrame] = None,
    trigger_time: float = 0.0,
    title_prefix: str = "",
    downsample: int = 10,
    color_density_by_amplitude: bool = False,
    probe_amplitudes: Optional[Dict[str, np.ndarray]] = None,
    amplitude_colormap: str = "coolwarm",
    amplitude_impedance: float = 50.0,
    **kwargs,
):
    """
    Plot comprehensive analysis overview for a shot.

    This function creates an overview plot showing:
    - Processed signals (waveforms)
    - Density evolution
    - VEST data (if available)

    Args:
        shot_num(int): Shot number for plot titles
        signals_dict(Dict[str, pd.DataFrame]): Dictionary of signal DataFrames
        density_dict(Dict[str, pd.DataFrame]): Dictionary of density DataFrames
        vest_data(Optional[pd.DataFrame]): Optional VEST database data
        trigger_time(float): Trigger time offset
        title_prefix(str): Prefix for plot titles
        downsample(int): Downsampling factor for plotting
        **kwargs: Additional arguments passed to plotting functions

    Returns:
        None
    """
    plotter = Plotter()

    # Plot signals
    for name, df in signals_dict.items():
        if df is not None and not df.empty:
            try:
                plotter.plot_waveforms(
                    df,
                    title=f"{title_prefix}{name}",
                    trigger_time=trigger_time,
                    downsample=downsample,
                    show_plot=True,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"{log_tag('PLOTS', 'OVER')} Failed to plot {name}: {e}")

    # Plot density
    for name, df in density_dict.items():
        if df is not None and not df.empty:
            try:
                # Extract time from index if available
                time_data = None
                if hasattr(df, "index") and df.index.name == "TIME":
                    time_data = df.index.values
                elif "TIME" in df.columns:
                    time_data = df["TIME"].values

                # Prepare probe amplitudes for this density DataFrame
                plot_probe_amp = None
                if color_density_by_amplitude:
                    # If probe_amplitudes is provided, use it directly
                    # Otherwise, try to extract from signals_dict
                    if probe_amplitudes is not None:
                        # Match density column names to probe amplitudes
                        if isinstance(df, pd.DataFrame):
                            plot_probe_amp = {}
                            for col_name in df.columns:
                                if col_name in probe_amplitudes:
                                    plot_probe_amp[col_name] = probe_amplitudes[col_name]
                    else:
                        # Try to extract from signals_dict if available
                        # This allows loading from HDF5 files where probe_amplitudes weren't stored
                        if signals_dict and isinstance(df, pd.DataFrame):
                            # Find the corresponding signals DataFrame
                            # signals_dict format: {name: DataFrame}
                            for sig_name, sig_df in signals_dict.items():
                                if sig_df is not None and not sig_df.empty:
                                    # Import helper function from main_analysis
                                    try:
                                        from ifi.analysis.main_analysis import _extract_probe_amplitudes_from_signals
                                        # Try to infer frequency from density column names
                                        # Default to 94.0 if can't determine
                                        freq_ghz = 94.0
                                        if any("_ALL" in col for col in df.columns):
                                            freq_ghz = 280.0
                                        plot_probe_amp = _extract_probe_amplitudes_from_signals(
                                            df, sig_df, freq_ghz
                                        )
                                        if plot_probe_amp:
                                            break
                                    except ImportError:
                                        # If import fails, skip automatic extraction
                                        pass

                plotter.plot_density(
                    df,
                    time_data=time_data,
                    title=f"{title_prefix}{name}",
                    show_plot=True,
                    probe_amplitude=plot_probe_amp,
                    color_by_amplitude=color_density_by_amplitude,
                    amplitude_colormap=amplitude_colormap,
                    amplitude_impedance=amplitude_impedance,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(
                    f"{log_tag('PLOTS', 'OVER')} Failed to plot density {name}: {e}"
                )

    # Plot VEST data if available
    if vest_data is not None and not vest_data.empty:
        try:
            # Find IP column
            ip_col = None
            for col in vest_data.columns:
                if "ip" in col.lower() or "current" in col.lower():
                    ip_col = col
                    break

            if ip_col:
                fig, ax = plt.subplots(figsize=(12, 6))
                time_vest = vest_data.index.values + trigger_time
                ax.plot(time_vest * 1000, vest_data[ip_col].values, "r-", linewidth=2)
                ax.set_xlabel("Time [ms]", **FontStyle.label)
                ax.set_ylabel(f"{ip_col}", **FontStyle.label)
                ax.set_title(f"{title_prefix}Plasma Current", **FontStyle.title)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show(block=False)
        except Exception as e:
            logger.warning(f"{log_tag('PLOTS', 'OVER')} Failed to plot VEST data: {e}")


## Main-guard test code removed. See archived copy in `ifi/olds/plots_old_20251030.py`.
