#!/usr/bin/env python3
"""
Density workflow helpers for main_analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..utils.if_utils import build_group_signal_column_name
from ..utils.io_process_common import make_density_group_name
from ..utils.log_manager import log_tag
from ..utils.vest_utils import infer_sample_rate_from_index
from . import phi2ne, spectrum

def _sampling_frequency(freq_data: pd.DataFrame, freq_ghz: float) -> float:
    fs = infer_sample_rate_from_index(freq_data.index)
    if fs is None:
        fs = 250e6
        logging.warning(
            f"{log_tag('ANALY','RUN')} Invalid time resolution: "
            f"Detected: {freq_ghz} GHz \t Using default: fs={fs / 1e6:.1f} MHz"
        )
        return fs
    return float(fs)


def _resolve_group_reference(
    freq_data: pd.DataFrame,
    freq_ghz: float,
    freq_files: list[str],
    freq_params_list: list[dict[str, Any]],
    basename: str,
) -> tuple[pd.Series | None, str | None]:
    for other_basename, other_params in zip(freq_files, freq_params_list):
        if other_basename == basename:
            continue
        other_ref = other_params.get("ref_col")
        if not other_ref:
            continue
        ref_col = build_group_signal_column_name(freq_ghz, other_ref, other_basename)
        if ref_col in freq_data.columns:
            return freq_data[ref_col].dropna(), ref_col
    return None, None


def _apply_baseline_if_needed(
    phase_converter: phi2ne.PhaseConverter,
    freq_density_data: pd.DataFrame,
    freq_data: pd.DataFrame,
    args: Any,
    shot_num: int,
    vest_ip_data: pd.DataFrame | None,
) -> pd.DataFrame:
    if not getattr(args, "baseline", None) or vest_ip_data is None:
        return freq_density_data

    time_axis = freq_data.index.to_numpy()
    ip_column_name = None
    if args.baseline == "ip":
        for col_name in vest_ip_data.columns:
            col_lower = str(col_name).lower()
            if "ip" in col_lower or "current" in col_lower:
                ip_column_name = col_name
                break

    return phase_converter.correct_baseline(
        freq_density_data,
        time_axis,
        args.baseline,
        shot_num=shot_num,
        vest_data=vest_ip_data,
        ip_column_name=ip_column_name,
    )


def calculate_density_data_by_frequency(
    *,
    freq_combined_signals: dict[float, pd.DataFrame],
    freq_groups: dict[float, dict[str, list]],
    shot_num: int,
    args: Any,
    vest_ip_data: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """
    Calculate density results for each frequency group.

    Returns:
        dict keyed by canonical density subgroup name (e.g., `"freq_94G"`).
    """
    density_data: dict[str, pd.DataFrame] = {}

    logging.debug(f"{log_tag('ANALY','RUN')} DEBUGGING combined_signals (dict structure)")
    for freq_key, freq_df in freq_combined_signals.items():
        logging.debug(
            f"{log_tag('ANALY','RUN')} {freq_key} GHz - Shape: {freq_df.shape}, Columns: {list(freq_df.columns)}"
        )
        if len(freq_df) > 0:
            time_diff = freq_df.index.to_series().diff().mean()
            logging.debug(
                f"{log_tag('ANALY','RUN')} {freq_key} GHz - Time resolution: {time_diff:.2e} s"
            )
    logging.debug(f"{log_tag('ANALY','RUN')} END DEBUG")

    if not getattr(args, "density", False):
        return density_data

    phase_converter = phi2ne.PhaseConverter()

    for freq_ghz, freq_data in freq_combined_signals.items():
        logging.info(
            f"{log_tag('ANALY','RUN')} Processing density calculation for {freq_ghz} GHz"
        )
        freq_density_data = pd.DataFrame(index=freq_data.index)

        group_info = freq_groups.get(freq_ghz, {})
        freq_files = group_info.get("files", [])
        freq_params_list = group_info.get("params", [])
        fs = _sampling_frequency(freq_data, freq_ghz)
        logging.info(
            f"{log_tag('ANALY','RUN')} {freq_ghz} GHz - Sampling frequency: {fs / 1e6:.2f} MHz"
        )

        for basename, params in zip(freq_files, freq_params_list):
            if not params or params.get("method") == "unknown":
                continue

            method = params.get("method")
            logging.info(
                f"{log_tag('ANALY','RUN')} Processing {basename}: {method} method, {freq_ghz} GHz"
            )

            if method == "CDM":
                ref_signal = None
                ref_col_name = None
                own_ref = params.get("ref_col")

                if own_ref:
                    candidate = build_group_signal_column_name(freq_ghz, own_ref, basename)
                    if candidate in freq_data.columns:
                        ref_signal = freq_data[candidate].dropna().to_numpy()
                        ref_col_name = candidate
                        logging.info(
                            f"{log_tag('ANALY','RUN')} Using own reference {candidate} for {basename}"
                        )
                    else:
                        logging.warning(
                            f"{log_tag('ANALY','RUN')} Reference column {candidate} not found for {basename}"
                        )
                        continue
                else:
                    shared_ref, shared_col = _resolve_group_reference(
                        freq_data,
                        freq_ghz,
                        freq_files,
                        freq_params_list,
                        basename,
                    )
                    if shared_ref is None:
                        logging.warning(
                            f"{log_tag('ANALY','RUN')} No reference signal available for {basename} - skipping CDM analysis"
                        )
                        continue
                    ref_signal = shared_ref.to_numpy()
                    ref_col_name = shared_col
                    logging.info(
                        f"{log_tag('ANALY','RUN')} Using shared reference {shared_col} for {basename}"
                    )

                analyzer = spectrum.SpectrumAnalysis()
                f_center = analyzer.find_center_frequency_fft(ref_signal, fs)
                if f_center == 0.0:
                    f_center = min(fs / 8, 20e6)
                    logging.warning(
                        f"{log_tag('ANALY','RUN')} Center frequency detection failed for {basename} - using default: {f_center / 1e6:.2f} MHz"
                    )
                else:
                    logging.info(
                        f"{log_tag('ANALY','RUN')} {basename}: f_center = {f_center / 1e6:.2f} MHz"
                    )

                for probe_col in params.get("probe_cols", []):
                    probe_col_name = build_group_signal_column_name(freq_ghz, probe_col, basename)
                    if probe_col_name not in freq_data.columns:
                        logging.warning(
                            f"{log_tag('ANALY','RUN')} Probe column {probe_col_name} not found for {basename}"
                        )
                        continue

                    probe_signal = freq_data[probe_col_name].dropna().to_numpy()
                    phase, _ = phase_converter.calc_phase_cdm(
                        ref_signal, probe_signal, fs, f_center
                    )
                    density_col_name = f"ne_{probe_col}_{basename}"
                    freq_density_data[density_col_name] = phase_converter.phase_to_density(
                        phase, analysis_params=params
                    )
                    logging.info(
                        f"{log_tag('ANALY','RUN')} CDM: Calculated density for {probe_col} in {basename} ({ref_col_name})"
                    )

            elif method == "FPGA":
                ref_col = params.get("ref_col")
                if not ref_col:
                    logging.warning(
                        f"{log_tag('ANALY','RUN')} No reference signal for {basename} - skipping FPGA analysis"
                    )
                    continue
                ref_col_name = build_group_signal_column_name(freq_ghz, ref_col, basename)
                if ref_col_name not in freq_data.columns:
                    logging.warning(
                        f"{log_tag('ANALY','RUN')} Reference column {ref_col_name} not found for {basename}"
                    )
                    continue

                ref_signal = freq_data[ref_col_name].dropna().to_numpy()
                time_axis = freq_data.index.to_numpy()
                for probe_col in params.get("probe_cols", []):
                    probe_col_name = build_group_signal_column_name(freq_ghz, probe_col, basename)
                    if probe_col_name not in freq_data.columns:
                        logging.warning(
                            f"{log_tag('ANALY','RUN')} Probe column {probe_col_name} not found for {basename}"
                        )
                        continue
                    probe_signal = freq_data[probe_col_name].dropna().to_numpy()
                    phase = phase_converter.calc_phase_fpga(
                        ref_signal, probe_signal, time_axis, probe_signal, isflip=False
                    )
                    density_col_name = f"ne_{probe_col}_{basename}"
                    freq_density_data[density_col_name] = phase_converter.phase_to_density(
                        phase, analysis_params=params
                    )
                    logging.info(
                        f"{log_tag('ANALY','RUN')} FPGA: Calculated density for {probe_col} in {basename}"
                    )

            elif method == "IQ":
                probe_cols = params.get("probe_cols", [])
                if not probe_cols:
                    logging.warning(
                        f"{log_tag('ANALY','RUN')} No probe_cols defined for IQ method in {basename}"
                    )
                    continue
                probe_cols_tuple = probe_cols[0]
                if not (isinstance(probe_cols_tuple, tuple) and len(probe_cols_tuple) == 2):
                    logging.warning(
                        f"{log_tag('ANALY','RUN')} Invalid IQ probe_cols format for {basename}: {probe_cols_tuple}"
                    )
                    continue

                i_col, q_col = probe_cols_tuple
                i_col_name = build_group_signal_column_name(freq_ghz, i_col, basename)
                q_col_name = build_group_signal_column_name(freq_ghz, q_col, basename)
                if i_col_name not in freq_data.columns or q_col_name not in freq_data.columns:
                    logging.warning(
                        f"{log_tag('ANALY','RUN')} IQ columns {i_col_name}, {q_col_name} not found for {basename}"
                    )
                    continue

                i_signal = freq_data[i_col_name].dropna().to_numpy()
                q_signal = freq_data[q_col_name].dropna().to_numpy()
                phase, _ = phase_converter.calc_phase_iq(i_signal, q_signal)
                freq_density_data[f"ne_IQ_{basename}"] = phase_converter.phase_to_density(
                    phase, analysis_params=params
                )
                logging.info(
                    f"{log_tag('ANALY','RUN')} IQ: Calculated density for {basename}"
                )

            else:
                logging.warning(
                    f"{log_tag('ANALY','RUN')} Unknown interferometry method: {method} for file {basename}"
                )

        if freq_density_data.empty:
            continue

        corrected = _apply_baseline_if_needed(
            phase_converter=phase_converter,
            freq_density_data=freq_density_data,
            freq_data=freq_data,
            args=args,
            shot_num=shot_num,
            vest_ip_data=vest_ip_data,
        )
        density_data[make_density_group_name(freq_ghz)] = corrected
        logging.info(
            f"{log_tag('ANALY','RUN')} Stored density data for {freq_ghz} GHz as "
            f"{make_density_group_name(freq_ghz)} with {len(corrected.columns)} columns"
        )

    return density_data


__all__ = ["calculate_density_data_by_frequency"]
