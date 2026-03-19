#!/usr/bin/env python3
"""Amplitude/envelope DSP helpers for IFI signal analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert, medfilt

from .if_utils import map_frequency_to_group


def _as_float_array(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _odd_kernel_size(kernel_size: int) -> int:
    size = max(1, int(kernel_size))
    return size if size % 2 == 1 else size + 1


def _window_samples_from_time_axis(
    time_axis: np.ndarray,
    window_us: float | None,
) -> int:
    if window_us is None or window_us <= 0 or len(time_axis) < 2:
        return 1
    dt = float(np.nanmedian(np.diff(time_axis)))
    if not np.isfinite(dt) or dt <= 0:
        return 1
    return _odd_kernel_size(max(1, round((window_us * 1.0e-6) / dt)))


def _resolve_signal_column_name(density_col: str, signals_df: pd.DataFrame, freq_ghz: float) -> str | None:
    remaining = str(density_col)[3:]
    signal_col = remaining.split("_")[0] if map_frequency_to_group(float(freq_ghz)) == 280.0 else remaining
    return signal_col if signal_col in signals_df.columns else None


def extract_probe_amplitudes_from_signals(
    density_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    freq_ghz: float,
) -> dict[str, np.ndarray]:
    """Extract aligned absolute probe amplitudes by matching density columns to signals."""
    probe_amplitudes: dict[str, np.ndarray] = {}
    if density_df.empty or signals_df.empty:
        return probe_amplitudes

    for density_col in density_df.columns:
        if not str(density_col).startswith("ne_"):
            continue

        signal_col = _resolve_signal_column_name(str(density_col), signals_df, freq_ghz)
        if signal_col is None:
            continue

        signal_data = signals_df[signal_col].dropna()
        if len(signal_data) == 0:
            continue

        aligned_signal = signal_data.reindex(density_df.index, method="nearest", limit=1)
        probe_amplitudes[str(density_col)] = np.abs(aligned_signal.to_numpy(dtype=float))

    return probe_amplitudes


def compute_signal_envelope(
    time_axis: pd.Series | np.ndarray | list[float],
    signal: pd.Series | np.ndarray | list[float],
    *,
    spike_kernel_size: int = 5,
    smooth_window_us: float | None = None,
) -> np.ndarray:
    """Compute a spike-robust signal envelope using median filter + Hilbert magnitude."""
    time_values = _as_float_array(time_axis)
    signal_values = _as_float_array(signal)
    if len(time_values) != len(signal_values):
        raise ValueError("time_axis and signal must have the same length")
    if len(signal_values) == 0:
        return np.array([], dtype=float)

    finite_mask = np.isfinite(time_values) & np.isfinite(signal_values)
    if not np.any(finite_mask):
        return np.full_like(signal_values, np.nan, dtype=float)

    clean_signal = signal_values.copy()
    clean_signal[~finite_mask] = np.nan
    clean_signal = pd.Series(clean_signal).interpolate(
        method="linear",
        limit_direction="both",
    ).to_numpy(dtype=float)

    filtered = medfilt(clean_signal, kernel_size=_odd_kernel_size(spike_kernel_size))
    envelope = np.abs(hilbert(filtered))

    smooth_samples = _window_samples_from_time_axis(time_values, smooth_window_us)
    if smooth_samples > 1:
        envelope = (
            pd.Series(envelope)
            .rolling(window=smooth_samples, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
    return envelope


def compute_baseline_peak_mean(
    time_axis: pd.Series | np.ndarray | list[float],
    envelope: pd.Series | np.ndarray | list[float],
    *,
    baseline_fraction: float = 0.2,
    peak_distance_us: float | None = None,
) -> float:
    """Compute mean of envelope peaks over the initial baseline fraction."""
    time_values = _as_float_array(time_axis)
    envelope_values = _as_float_array(envelope)
    if len(time_values) != len(envelope_values) or len(envelope_values) == 0:
        return float("nan")

    baseline_count = max(1, int(len(envelope_values) * float(baseline_fraction)))
    baseline_env = envelope_values[:baseline_count]
    baseline_time = time_values[:baseline_count]
    if len(baseline_env) == 0:
        return float("nan")

    peak_distance = _window_samples_from_time_axis(baseline_time, peak_distance_us)
    peaks, _ = find_peaks(baseline_env, distance=max(1, peak_distance))
    peak_values = baseline_env[peaks] if len(peaks) > 0 else baseline_env[np.isfinite(baseline_env)]
    if len(peak_values) == 0:
        return float("nan")
    return float(np.nanmean(peak_values))


def find_low_envelope_segments(
    time_axis: pd.Series | np.ndarray | list[float],
    signal: pd.Series | np.ndarray | list[float],
    *,
    threshold_ratio: float = 0.7,
    min_duration_us: float = 100.0,
    baseline_fraction: float = 0.2,
    spike_kernel_size: int = 5,
    smooth_window_us: float | None = None,
    peak_distance_us: float | None = None,
    include_samples: bool = True,
) -> dict[str, Any]:
    """Detect sustained low-envelope intervals and return JSON-serializable payload."""
    time_values = _as_float_array(time_axis)
    signal_values = _as_float_array(signal)
    if len(time_values) != len(signal_values):
        raise ValueError("time_axis and signal must have the same length")

    envelope = compute_signal_envelope(
        time_values,
        signal_values,
        spike_kernel_size=spike_kernel_size,
        smooth_window_us=smooth_window_us,
    )
    baseline_peak_mean = compute_baseline_peak_mean(
        time_values,
        envelope,
        baseline_fraction=baseline_fraction,
        peak_distance_us=peak_distance_us,
    )
    threshold = float(threshold_ratio) * baseline_peak_mean if np.isfinite(baseline_peak_mean) else float("nan")

    result: dict[str, Any] = {
        "threshold_ratio": float(threshold_ratio),
        "min_duration_us": float(min_duration_us),
        "baseline_fraction": float(baseline_fraction),
        "baseline_peak_mean": baseline_peak_mean,
        "threshold": threshold,
        "segments": [],
    }

    if len(time_values) == 0 or not np.isfinite(threshold):
        return result

    below_mask = np.isfinite(envelope) & (envelope <= threshold)
    if not np.any(below_mask):
        return result

    min_duration_s = float(min_duration_us) * 1.0e-6
    start_idx: int | None = None
    for idx, is_below in enumerate(below_mask):
        if is_below and start_idx is None:
            start_idx = idx
        elif not is_below and start_idx is not None:
            end_idx = idx - 1
            duration = float(time_values[end_idx] - time_values[start_idx])
            if duration >= min_duration_s:
                segment = {
                    "start_time": float(time_values[start_idx]),
                    "end_time": float(time_values[end_idx]),
                    "duration_us": duration * 1.0e6,
                }
                if include_samples:
                    sl = slice(start_idx, end_idx + 1)
                    segment["time"] = time_values[sl].tolist()
                    segment["probe_value"] = signal_values[sl].tolist()
                    segment["envelope"] = envelope[sl].tolist()
                result["segments"].append(segment)
            start_idx = None

    if start_idx is not None:
        end_idx = len(time_values) - 1
        duration = float(time_values[end_idx] - time_values[start_idx])
        if duration >= min_duration_s:
            segment = {
                "start_time": float(time_values[start_idx]),
                "end_time": float(time_values[end_idx]),
                "duration_us": duration * 1.0e6,
            }
            if include_samples:
                sl = slice(start_idx, end_idx + 1)
                segment["time"] = time_values[sl].tolist()
                segment["probe_value"] = signal_values[sl].tolist()
                segment["envelope"] = envelope[sl].tolist()
            result["segments"].append(segment)

    return result


def export_probe_envelope_segments_json(
    output_path: str | Path,
    *,
    channel_name: str,
    time_axis: pd.Series | np.ndarray | list[float],
    signal: pd.Series | np.ndarray | list[float],
    threshold_ratio: float = 0.7,
    min_duration_us: float = 100.0,
    baseline_fraction: float = 0.2,
    spike_kernel_size: int = 5,
    smooth_window_us: float | None = None,
    peak_distance_us: float | None = None,
    include_samples: bool = True,
) -> dict[str, Any]:
    """Detect low-envelope segments for one probe channel and save them as JSON."""
    payload = find_low_envelope_segments(
        time_axis=time_axis,
        signal=signal,
        threshold_ratio=threshold_ratio,
        min_duration_us=min_duration_us,
        baseline_fraction=baseline_fraction,
        spike_kernel_size=spike_kernel_size,
        smooth_window_us=smooth_window_us,
        peak_distance_us=peak_distance_us,
        include_samples=include_samples,
    )
    payload["channel"] = str(channel_name)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = [
    "compute_baseline_peak_mean",
    "compute_signal_envelope",
    "export_probe_envelope_segments_json",
    "extract_probe_amplitudes_from_signals",
    "find_low_envelope_segments",
]
