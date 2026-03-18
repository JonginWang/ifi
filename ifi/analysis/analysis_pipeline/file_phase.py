#!/usr/bin/env python3
"""Per-file load/process helpers for main_analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ...utils.log_manager import log_tag
from .. import processing, spectrum

if TYPE_CHECKING:
    from .. import spectrum

def resolve_allowed_frequencies(args: argparse.Namespace) -> list[float] | None:
    """Return requested frequency filters for NAS reads."""
    allowed_freqs = getattr(args, "freq", None)
    return allowed_freqs if allowed_freqs is not None else None


def load_single_file_raw_data(
    nas_instance: Any,
    file_path: str,
    args: argparse.Namespace,
) -> tuple[str, pd.DataFrame | None]:
    """Load one source file from NAS and return basename plus raw frame."""
    file_basename = Path(file_path).name
    data_dict = nas_instance.get_shot_data(
        file_path,
        force_remote=args.force_remote,
        allowed_frequencies=resolve_allowed_frequencies(args),
    )
    if not data_dict or file_basename not in data_dict:
        logging.warning(
            "\n"
            + "=" * 80
            + "\n"
            + f"  Failed to read {file_basename} at {file_path}. Skipping.  ".center(80, "!")
            + "\n"
            + "=" * 80
            + "\n"
        )
        return file_basename, None
    return file_basename, data_dict[file_basename].copy()


def refine_and_preprocess_signal(
    df_raw: pd.DataFrame,
    file_path: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Refine NaNs and optionally remove offset."""
    df_refined = processing.refine_data(df_raw)
    logging.info("\n" + f"  Data refined for {file_path}  ".center(80, "=") + "\n")
    if args.no_offset_removal:
        return df_refined
    df_processed = processing.remove_offset(df_refined, window_size=args.offset_window)
    logging.info(f"{log_tag('ANALY','LOAD')} Offset removed from {file_path}")
    return df_processed


def _resolve_selected_columns(
    df_processed: pd.DataFrame,
    requested_columns: list[int] | None,
    analysis_name: str,
) -> list[str]:
    all_data_cols = [col for col in df_processed.columns if col != "TIME"]
    if not requested_columns:
        return all_data_cols

    valid_idxs = sorted(set(range(len(all_data_cols))).intersection(requested_columns))
    if valid_idxs:
        logging.info(
            f"{log_tag('ANALY','LOAD')} Analyzing columns: {valid_idxs} vs input columns: {requested_columns}"
        )
        return [all_data_cols[idx] for idx in valid_idxs]

    logging.warning(
        f"{log_tag('ANALY','LOAD')} No columns to analyze for {analysis_name}. Skipping {analysis_name} analysis."
    )
    return []


def _resolve_sampling_rate(df_processed: pd.DataFrame) -> float:
    return 1 / df_processed["TIME"].diff().mean()


def compute_stft_results(
    file_basename: str,
    df_processed: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]] | None:
    """Compute STFT outputs for selected columns."""
    if not args.stft:
        return None

    analyzer = spectrum.SpectrumAnalysis()
    fs = _resolve_sampling_rate(df_processed)
    columns = _resolve_selected_columns(df_processed, args.stft_cols, "STFT")
    if not columns:
        return {file_basename: {}}

    stft_result_for_file: dict[str, dict[str, Any]] = {}
    for col_name in columns:
        signal = df_processed[col_name].to_numpy()
        freq_stft, time_stft, stft_matrix = analyzer.compute_stft(signal, fs)
        stft_result_for_file[col_name] = {
            "time_STFT": time_stft,
            "freq_STFT": freq_stft,
            "STFT_matrix": stft_matrix,
        }

    logging.info(f"{log_tag('ANALY','LOAD')} STFT analysis complete for {file_basename}")
    return {file_basename: stft_result_for_file}


def _compute_cwt_payload(
    analyzer: "spectrum.SpectrumAnalysis",
    signal: np.ndarray,
    time_values: np.ndarray,
    fs: float,
    col_name: str,
) -> dict[str, np.ndarray]:
    f_center = analyzer.find_center_frequency_fft(signal, fs)
    decimation_factor = 1
    if len(signal) > 100000:
        decimation_factor = max(1, len(signal) // 10000)
        logging.info(
            f"{log_tag('ANALY','LOAD')} Large signal detected ({len(signal)} samples). "
            f"Using decimation factor {decimation_factor} for CWT."
        )

    if f_center > 0:
        freq_cwt, cwt_matrix = analyzer.compute_cwt(
            signal,
            fs,
            f_center=f_center,
            f_deviation=0.1,
            decimation_factor=decimation_factor,
        )
    else:
        logging.warning(
            f"{log_tag('ANALY','LOAD')} Center frequency detection failed for {col_name}. "
            f"Using default CWT (may use more memory)."
        )
        freq_cwt, cwt_matrix = analyzer.compute_cwt(
            signal,
            fs,
            decimation_factor=decimation_factor,
        )

    time_cwt = time_values[::decimation_factor] if decimation_factor > 1 else time_values
    return {
        "time_CWT": time_cwt,
        "freq_CWT": freq_cwt,
        "CWT_matrix": cwt_matrix,
    }


def compute_cwt_results(
    file_basename: str,
    df_processed: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]] | None:
    """Compute CWT outputs for selected columns."""
    if not args.cwt:
        return None

    analyzer = spectrum.SpectrumAnalysis()
    fs = _resolve_sampling_rate(df_processed)
    columns = _resolve_selected_columns(df_processed, args.cwt_cols, "CWT")
    if not columns:
        return {file_basename: {}}

    time_values = df_processed["TIME"].values
    cwt_result_for_file: dict[str, dict[str, Any]] = {}
    for col_name in columns:
        cwt_result_for_file[col_name] = _compute_cwt_payload(
            analyzer,
            df_processed[col_name].to_numpy(),
            time_values,
            fs,
            col_name,
        )

    logging.info(f"{log_tag('ANALY','LOAD')} CWT analysis complete for {file_basename}")
    return {file_basename: cwt_result_for_file}


__all__ = [
    "compute_cwt_results",
    "compute_stft_results",
    "load_single_file_raw_data",
    "refine_and_preprocess_signal",
    "resolve_allowed_frequencies",
]
