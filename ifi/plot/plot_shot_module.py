#!/usr/bin/env python3
"""Shot-level plotting helpers that save figures to the results tree."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plot_density_module import plot_density_core
from .plot_timefreq_module import plot_time_frequency_core
from .plot_waveform_module import plot_waveforms_core
from ..analysis.spectrum import SpectrumAnalysis
from ..utils.log_manager import LogManager, log_tag
from ..utils.path_utils import ensure_dir_exists

logger = LogManager().get_logger(__name__)


def _resolve_results_dir(results_dir: Path | str | None, shot_num: int | None) -> Path:
    if results_dir is None:
        return Path("results") / str(shot_num or 0)
    return Path(results_dir)


def _ensure_plot_subdir(
    results_dir: Path | str | None,
    shot_num: int | None,
    subdir: str,
) -> tuple[Path, int]:
    base_dir = _resolve_results_dir(results_dir, shot_num)
    ensure_dir_exists(str(base_dir))
    target_dir = base_dir / subdir
    ensure_dir_exists(str(target_dir))
    return target_dir, int(shot_num or 0)


def _extract_raw_signal_frames(shot_data: dict) -> dict[str, pd.DataFrame]:
    if not isinstance(shot_data, dict):
        return {}
    if "rawdata" in shot_data and isinstance(shot_data["rawdata"], dict):
        return shot_data["rawdata"]
    return {
        str(name): df
        for name, df in shot_data.items()
        if isinstance(df, pd.DataFrame)
    }


def _extract_density_frames(shot_data: dict) -> dict[str, pd.DataFrame]:
    if not isinstance(shot_data, dict):
        return {}
    density_payload = shot_data.get("density")
    if isinstance(density_payload, dict):
        return {
            str(name): df
            for name, df in density_payload.items()
            if isinstance(df, pd.DataFrame) and not df.empty
        }
    density_series = {
        str(name): series
        for name, series in shot_data.items()
        if str(name).startswith("ne_") and isinstance(series, pd.Series)
    }
    if not density_series:
        return {}
    return {"density": pd.DataFrame(density_series)}


def plot_shot_waveforms(
    shot_data: dict,
    results_dir: Path | str | None = None,
    shot_num: int | None = None,
    downsample: int = 100,
    plot_envelope: bool = False,
) -> None:
    """Generate waveform plots for all available signals in a shot."""
    logger.info(f"{log_tag('PLOTS', 'WFDAT')} Generating waveform plots...")
    waveform_dir, shot_num = _ensure_plot_subdir(results_dir, shot_num, "waveforms")

    for filename, df in _extract_raw_signal_frames(shot_data).items():
        if "TIME" not in df.columns:
            continue
        logger.info(f"{log_tag('PLOTS', 'WFDAT')} Plotting waveforms for {filename}")
        try:
            fig, _ = plot_waveforms_core(
                df,
                title=f"Shot {shot_num} - {filename}",
                downsample=downsample,
                plot_envelope=plot_envelope,
                show_plot=False,
            )
            output_path = waveform_dir / f"{filename}_waveforms.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"{log_tag('PLOTS', 'WFDAT')} Saved waveform plot: {output_path}")
        except Exception as e:
            logger.error(f"{log_tag('PLOTS', 'WFDAT')} Failed to plot waveforms for {filename}: {e}")


def plot_shot_spectrograms(
    shot_data: dict,
    results_dir: Path | str | None = None,
    shot_num: int | None = None,
    max_channels: int = 2,
) -> None:
    """Generate spectrogram plots using STFT analysis for a shot."""
    logger.info(f"{log_tag('PLOTS', 'SPCTR')} Generating spectrogram plots...")
    analyzer = SpectrumAnalysis()
    spectra_dir, shot_num = _ensure_plot_subdir(results_dir, shot_num, "spectra")

    for filename, df in _extract_raw_signal_frames(shot_data).items():
        if "TIME" not in df.columns:
            continue
        time_source = df["TIME"].values
        fs = 1.0 / np.diff(time_source[:1000]).mean()
        logger.info(
            f"{log_tag('PLOTS', 'SPCTR')} Processing spectrogram for {filename} (fs = {fs / 1e6:.1f} MHz)"
        )
        signal_cols = [col for col in df.columns if col != "TIME"]
        for col in signal_cols[:max_channels]:
            try:
                fig, _ = plot_time_frequency_core(
                    df[col].values[::100],
                    method="stft",
                    fs=fs / 100,
                    title=f"Shot {shot_num} - {filename} - {col}",
                    show_plot=False,
                    analyzer=analyzer,
                    nperseg=1024,
                    noverlap=512,
                )
                output_path = spectra_dir / f"{filename}_{col}_spectrogram.png"
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"{log_tag('PLOTS', 'SPCTR')} Saved spectrogram: {output_path}")
            except Exception as e:
                logger.error(
                    f"{log_tag('PLOTS', 'SPCTR')} Failed to generate spectrogram for {filename}_{col}: {e}"
                )


def plot_shot_density_evolution(
    shot_data: dict,
    vest_data: pd.DataFrame | None,
    results_dir: Path | str | None = None,
    shot_num: int | None = None,
) -> None:
    """Plot density evolution for a shot using available density outputs."""
    del vest_data
    logger.info(f"{log_tag('PLOTS', 'DENS')} Generating density evolution plots...")
    density_dir, shot_num = _ensure_plot_subdir(results_dir, shot_num, "density")

    density_frames = _extract_density_frames(shot_data)
    if not density_frames:
        logger.warning(f"{log_tag('PLOTS', 'DENS')} No density data found for plotting")
        return

    for density_name, density_df in density_frames.items():
        try:
            title = f"Shot {shot_num} - Density Evolution"
            if density_name != "density":
                title = f"{title} - {density_name}"
            fig, _ = plot_density_core(density_df, title=title, show_plot=False)
            file_stem = "density_evolution" if density_name == "density" else f"{density_name}_density"
            output_path = density_dir / f"{file_stem}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"{log_tag('PLOTS', 'DENS')} Saved density evolution plot: {output_path}")
        except Exception as e:
            logger.error(f"{log_tag('PLOTS', 'DENS')} Failed to generate density evolution plot: {e}")


__all__ = [
    "plot_shot_density_evolution",
    "plot_shot_spectrograms",
    "plot_shot_waveforms",
]
