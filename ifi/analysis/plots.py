import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any
import logging
from contextlib import contextmanager

from ifi.db_controller.vest_db import VEST_DB
from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.analysis.params.params_plot import set_plot_style, FontStyle
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from ifi.utils import LogManager, ensure_dir_exists

LogManager()

@contextmanager
def ifi_plotting(interactive: bool, save_dir: str = None, save_prefix: str = "fig_", save_ext: str = "png", dpi: int = 300):
    """
    A context manager to handle Matplotlib's plotting state.
    It manages interactive mode and saves all open figures upon exit if a
    directory is provided.
    
    Args:
        interactive (bool): If True, plots will be shown in a window.
        save_dir (str, optional): If provided, all figures will be saved to this
                                  directory upon exiting the context.
        save_prefix (str, optional): If provided, all figures will be saved with this prefix.
        save_ext (str, optional): If provided, all figures will be saved with this extension.
                                    Default is "png", but can be "pdf", "svg", etc.
        dpi (int, optional): If provided, all figures will be saved with this DPI.
                             Default is 300.
    """
    if interactive:
        plt.ion()
    
    try:
        yield
    finally:
        if save_dir:
            ensure_dir_exists(save_dir)
            for i in plt.get_fignums():
                fig = plt.figure(i)
                # Sanitize title for filename
                title = fig._suptitle.get_text() if fig._suptitle else f"figure_{i}"
                filename = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
                filename = filename.replace(" ", "_").replace("#", "")
                filepath = os.path.join(save_dir, f"{save_prefix}{filename}.{save_ext}")
                logging.info(f"Saving figure to {filepath}")
                fig.savefig(filepath, dpi=dpi)

        if interactive:
            plt.ioff()
            plt.show(block=True)
        
        plt.close('all')


def pow2db(x: np.ndarray) -> np.ndarray:
    """Converts power to decibels. A value of 0 is converted to -Inf."""
    return 10 * np.log10(x)

class Plotter:
    def __init__(self):
        self.db = VEST_DB()
        self.analyzer = SpectrumAnalysis()

    def _downsample(self, data: np.ndarray, factor: int) -> np.ndarray:
        """Downsamples the data by a given factor."""
        if factor <= 1:
            return data
        return data[::factor]

    def plot_multi_panel_time_series(self, time: np.ndarray, data: np.ndarray, labels: List[str], title: str, downsample: int = 1):
        """
        Plots multiple channels of time series data in subplots.
        """
        time = self._downsample(time, downsample)
        n_channels = data.shape[1]
        
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16)

        for i in range(n_channels):
            channel_data = self._downsample(data[:, i], downsample)
            axes[i].plot(time, channel_data)
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)
        
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_spectrogram_with_ridge(self, signal: np.ndarray, fs: float, title: str):
        """
        Plots a spectrogram of the signal with the frequency ridge overlaid.
        """
        f, t, Zxx = self.analyzer.compute_stft(signal, fs)
        ridge = self.analyzer.find_freq_ridg(Zxx, f)
        
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.plot(t, ridge, color='r', linewidth=2, label='Frequency Ridge')
        plt.title(title)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.legend()
        plt.colorbar(label='Magnitude')
        plt.show()

    def plot_shot_overview(self, shot_num: int, if_time: np.ndarray, if_density: np.ndarray):
        """
        Plots an overview of the shot, including interferometer data and key VEST data.
        """
        # Fetch VEST data
        with self.db as db:
            ip_time, ip_data = db.load_shot(shot_num, 109) # Ip
            h_alpha_time, h_alpha_data = db.load_shot(shot_num, 101) # H-alpha
            mirnov_time, mirnov_data = db.load_shot(shot_num, 171) # Mirnov
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Shot #{shot_num} Overview", fontsize=16)

        # Interferometer data
        axes[0].plot(if_time, if_density)
        axes[0].set_ylabel("Line Integrated Density [m^-2]")
        axes[0].grid(True)

        # Plasma Current (Ip)
        if ip_data is not None:
            axes[1].plot(ip_time, ip_data)
        axes[1].set_ylabel("Ip [A]")
        axes[1].grid(True)
        
        # H-alpha
        if h_alpha_data is not None:
            axes[2].plot(h_alpha_time, h_alpha_data)
        axes[2].set_ylabel("H-alpha [a.u.]")
        axes[2].grid(True)

        # Mirnov Coil
        if mirnov_data is not None:
            axes[3].plot(mirnov_time, mirnov_data)
        axes[3].set_ylabel("Mirnov [T/s]")
        axes[3].grid(True)
        
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == '__main__':
    # Example usage:
    # plotter = Plotter()
    # time = np.linspace(0, 1, 10000)
    # data = np.random.randn(10000, 3)
    # plotter.plot_multi_panel_time_series(time, data, ['Ch1', 'Ch2', 'Ch3'], "Test Plot", downsample=10)
    pass


def plot_signals(
    data_dict: Dict[str, pd.DataFrame],
    title_prefix: str = "",
    trigger_time: float = 0.0,
    downsample: int = 1
):
    """
    Plots signals from a dictionary of DataFrames, each in a separate figure.

    Args:
        data_dict: Dictionary mapping a name (e.g., filename) to a DataFrame.
        title_prefix: A suffix to add to the plot title (e.g., "(Offset Removed)").
        trigger_time: Time in seconds to add to the 'TIME' column.
        downsample: Factor by which to downsample the data for plotting.
    """
    if not data_dict:
        logging.warning("No data to plot.")
        return

    for name, df in data_dict.items():
        if not isinstance(df, pd.DataFrame) or 'TIME' not in df.columns:
            continue

        data_cols = [col for col in df.columns if col != 'TIME']
        if not data_cols:
            continue

        # Downsample
        if downsample > 1:
            plot_df = df.iloc[::downsample, :]
        else:
            plot_df = df

        set_plot_style()
        time_data = plot_df['TIME'] + trigger_time
        num_channels = len(data_cols)

        fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True, squeeze=False)
        axes = axes.flatten() # Ensure axes is always a flat array

        fig.suptitle(f"{os.path.basename(name)} {title_prefix}".strip(), **FontStyle.title)

        for i, col_name in enumerate(data_cols):
            axes[i].plot(time_data, plot_df[col_name])
            axes[i].set_ylabel(col_name, **FontStyle.label)
            axes[i].grid(True)
        
        axes[-1].set_xlabel(f"Time (s) [Trigger at {trigger_time}s]", **FontStyle.label)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show(block=False)


def plot_spectrograms(
    stft_results: Dict[str, Dict[str, Any]],
    title_prefix: str = "",
    trigger_time: float = 0.0,
    downsample: int = 1
):
    """
    Plots spectrograms and their frequency ridges from STFT results.

    Args:
        stft_results: Dictionary containing STFT results from SpectrumAnalysis.
                      Structure: {filename: {col_name: {'f', 't', 'Zxx'}}}
        title_prefix: A prefix to add to the plot title.
        trigger_time: Time in seconds to add to the time axis.
        downsample: Factor by which to downsample the time axis for plotting.
    """
    if not stft_results:
        return

    analyzer = SpectrumAnalysis() # Create a local instance for analysis
    for filename, results_by_col in stft_results.items():
        for col_name, results in results_by_col.items():
            f = results['f']
            t = results['t']
            Zxx = results['Zxx']
            
            ridge = analyzer.find_freq_ridge(Zxx, f)
            
            # Downsample time-dependent data for performance
            t_plot = t[::downsample]
            ridge_plot = ridge[::downsample]
            # For pcolormesh, we need to downsample Zxx along the time axis (axis 1)
            Zxx_plot = Zxx[:, ::downsample]

            set_plot_style()
            fig = plt.figure(figsize=(12, 6))
            
            plt.pcolormesh(t_plot + trigger_time, f / 1e6, pow2db(np.abs(Zxx_plot)), shading='gouraud', cmap='viridis')
            
            plt.plot(t_plot + trigger_time, ridge_plot / 1e6, color='r', linewidth=2, label='Frequency Ridge')
            
            plt.title(f"{title_prefix}Spectrogram: {os.path.basename(filename)} - {col_name}", **FontStyle.title)
            plt.ylabel("Frequency [MHz]", **FontStyle.label)
            plt.xlabel(f"Time (s) [Trigger at {trigger_time}s]", **FontStyle.label)
            plt.legend()
            cbar = plt.colorbar()
            cbar.set_label("Power [dB]", **FontStyle.label)
            
            fig.tight_layout()

    plt.show(block=False)


def plot_cwt(cwt_results, trigger_time=0.0, title_prefix=""):
    """
    Plots the results of CWT analysis.
    """
    num_plots = sum(len(cols) for cols in cwt_results.values())
    if num_plots == 0:
        return

    fig, axes = plt.subplots(
        num_plots, 1, figsize=(12, 3 * num_plots), sharex=True, squeeze=False
    )
    axes = axes.flatten()
    plot_idx = 0

    for filename, analysis in cwt_results.items():
        for col_name, result in analysis.items():
            ax = axes[plot_idx]
            t = result["t"]
            freqs = result["freqs"]
            cwt_matrix = np.abs(result["cwt_matrix"])

            # Use pcolormesh for time-frequency plotting
            im = ax.pcolormesh(
                t - trigger_time, freqs, cwt_matrix, shading="gouraud", cmap="hot"
            )
            fig.colorbar(im, ax=ax, label="Magnitude")

            ax.set_ylabel("Frequency (Hz)", **FontStyle.label)
            ax.set_title(f'CWT of {col_name} from {os.path.basename(filename)}', **FontStyle.title)
            plot_idx += 1

    axes[-1].set_xlabel("Time (s)", **FontStyle.label)
    fig.suptitle(f"{title_prefix}CWT Analysis", **FontStyle.title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_analysis_overview(
    shot_num,
    shot_data,
    density_results,
    vest_data,
    trigger_time=0.0,
    downsample=1,
    title_prefix="",
):
    """
    Plots a comprehensive overview of the analysis results, including VEST data,
    raw/processed signals, and calculated density.

    Args:
        shot_num (int): The shot number.
        shot_data (dict): Dictionary containing raw and processed signal dataframes.
        density_results (dict): Dictionary containing density dataframes.
        vest_data (pd.DataFrame): DataFrame with VEST data.
        trigger_time (float): Trigger time to align the x-axis.
        downsample (int): Factor to downsample data for plotting.
        title_prefix (str): Prefix for the plot title.
    """
    datasets = {
        "VEST": vest_data,
        **shot_data,
        **density_results,
    }

    # Filter out empty or non-DataFrame items
    datasets = {
        k: v
        for k, v in datasets.items()
        if isinstance(v, (pd.DataFrame, pd.Series)) and not v.empty
    }

    if not datasets:
        logging.warning("No data available to plot for the overview.")
        return

    set_plot_style()
    num_plots = len(datasets)
    fig, axes = plt.subplots(
        num_plots, 1, figsize=(12, 2 * num_plots), sharex=True, squeeze=False
    )
    axes = axes.flatten()

    for i, (name, data) in enumerate(datasets.items()):
        ax = axes[i]
        df_to_plot = data.iloc[::downsample] if downsample > 1 else data

        for col in df_to_plot.columns:
            ax.plot(df_to_plot.index - trigger_time, df_to_plot[col], label=col)

        ax.set_ylabel(name, **FontStyle.label)
        ax.legend(loc="upper right")
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)", **FontStyle.label)
    fig.suptitle(f"{title_prefix}Analysis Overview for Shot #{shot_num}", **FontStyle.title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_response(freqs_at_response, responses, title="Frequency Response"):
    "Utility function to plot response functions, from scipy.signal.freqz"
    set_plot_style()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freqs_at_response, 20*np.log10(np.abs(responses)))
    ax.set_ylim(-100, 5)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)', **FontStyle.label)
    ax.set_ylabel('Gain (dB)', **FontStyle.label)
    ax.set_title(title, **FontStyle.title)


def plot_waveforms(data, fs=None, title="Waveforms", downsample=1, save_path=None):
    """
    Plot multiple waveform signals.
    
    Args:
        data: DataFrame with TIME column and signal columns, or dict of arrays
        fs: Sampling frequency (optional)
        title: Plot title
        downsample: Downsampling factor
        save_path: Optional path to save figure
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """
    set_plot_style()
    
    if isinstance(data, pd.DataFrame):
        if 'TIME' in data.columns:
            time = data['TIME'].values
            signal_cols = [col for col in data.columns if col != 'TIME']
            signals = {col: data[col].values for col in signal_cols}
        else:
            # Use index as time
            time = data.index.values
            signals = {col: data[col].values for col in data.columns}
    elif isinstance(data, dict):
        # Assume dict contains signal arrays
        signals = data
        if fs is not None:
            max_len = max(len(sig) for sig in signals.values())
            time = np.arange(max_len) / fs
        else:
            max_len = max(len(sig) for sig in signals.values())
            time = np.arange(max_len)
    else:
        raise ValueError("Data must be DataFrame or dict")
    
    # Downsample if needed
    if downsample > 1:
        time = time[::downsample]
        signals = {k: v[::downsample] for k, v in signals.items()}
    
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(12, 2 * n_signals), sharex=True)
    
    if n_signals == 1:
        axes = [axes]
    
    for i, (name, signal) in enumerate(signals.items()):
        axes[i].plot(time, signal)
        axes[i].set_ylabel(name, **FontStyle.label)
        axes[i].grid(True)
    
    axes[-1].set_xlabel("Time (s)", **FontStyle.label)
    fig.suptitle(title, **FontStyle.title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_spectrogram(freqs, times, Sxx, title="Spectrogram", save_path=None):
    """
    Plot a spectrogram.
    
    Args:
        freqs: Frequency array
        times: Time array
        Sxx: Spectrogram matrix
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    
    im = ax.pcolormesh(times, freqs/1e6, Sxx_db, shading='gouraud', cmap='viridis')
    
    ax.set_ylabel("Frequency (MHz)", **FontStyle.label)
    ax.set_xlabel("Time (s)", **FontStyle.label)
    ax.set_title(title, **FontStyle.title)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power (dB)", **FontStyle.label)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_density_results(density_data, time_data=None, title="Density Results", save_path=None):
    """
    Plot density calculation results.
    
    Args:
        density_data: Dict of density arrays or DataFrame with density columns
        time_data: Optional time array
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if isinstance(density_data, pd.DataFrame):
        if time_data is None and hasattr(density_data, 'index'):
            time_data = density_data.index
        elif time_data is None:
            time_data = np.arange(len(density_data))
        
        for col in density_data.columns:
            ax.plot(time_data, density_data[col], label=col)
    
    elif isinstance(density_data, dict):
        if time_data is None:
            max_len = max(len(data) for data in density_data.values())
            time_data = np.arange(max_len)
        
        for name, data in density_data.items():
            ax.plot(time_data, data, label=name)
    
    else:
        # Single array
        if time_data is None:
            time_data = np.arange(len(density_data))
        ax.plot(time_data, density_data)
    
    ax.set_xlabel("Time (s)", **FontStyle.label)
    ax.set_ylabel("Density (m⁻³)", **FontStyle.label)
    ax.set_title(title, **FontStyle.title)
    ax.grid(True)
    
    if isinstance(density_data, (dict, pd.DataFrame)) and len(density_data) > 1:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax