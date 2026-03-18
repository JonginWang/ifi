#!/usr/bin/env python3
"""
Plotting Utility Helpers
========================

This module contains shared plotting utility helpers.

Key Features:
    - Normalize supported input types into (time, signals)
    - Scale time and signal values and return labels
    - Extract metadata summary for plot title suffix

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def format_unit_to_latex(unit: str) -> str:
    """Convert unit expressions like `m^2/s^2` into LaTeX-safe math text."""
    text = str(unit).strip()
    if not text:
        return text
    text = re.sub(
        r"([A-Za-z]+)\^(-?\d+)",
        lambda m: f"{m.group(1)}^{{{m.group(2)}}}",
        text,
    )
    return text


def format_signal_scale_label(signal_scale: str) -> tuple[float, str]:
    """Build a display label and scale factor for signal units."""
    text = str(signal_scale).strip()
    if not text:
        return 1.0, "[]"
    if text == "mV":
        return 1e3, "[mV]"
    if text == "uV":
        return 1e6, "[uV]"
    if text == "a.u.":
        return 1.0, "[a.u.]"
    if text == "V":
        return 1.0, "[V]"

    sci_match = re.fullmatch(r"10\^(-?\d+)\s*(.+)", text)
    if sci_match:
        exponent = int(sci_match.group(1))
        unit_latex = format_unit_to_latex(sci_match.group(2))
        return 10 ** (-exponent), f"[$10^{{{exponent}}} {unit_latex}$]"

    if "^" in text:
        return 1.0, f"[${format_unit_to_latex(text)}$]"

    return 1.0, f"[{text}]"


def prepare_time_data(
    data: pd.DataFrame | dict[str, np.ndarray] | np.ndarray,
    fs: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Normalize supported input types into (time, signals)."""
    if isinstance(data, pd.DataFrame):
        if any(col.upper() == "TIME" for col in data.columns):
            time_col = [col for col in data.columns if col.upper() == "TIME"][0]
            time = data[time_col].values
            signal_cols = [col for col in data.columns if col.upper() != "TIME"]
            signals = {col: data[col].values for col in signal_cols}
        else:
            time = data.index.values if hasattr(data, "index") else np.arange(len(data))
            time = time / fs if fs is not None else time
            signals = {col: data[col].values for col in data.columns}
        return time, signals

    if isinstance(data, dict):
        if any(k.upper() == "TIME" for k in data.keys()):
            time_key = [k for k in data.keys() if k.upper() == "TIME"][0]
            time = data[time_key]
            signals = {k: data[k] for k in data.keys() if k.upper() != "TIME"}
        else:
            max_len = max(len(data[k]) for k in data.keys())
            signals = data.copy()
            time = np.arange(max_len) / fs if fs is not None else np.arange(max_len)
        return time, signals

    if isinstance(data, np.ndarray):
        if data.ndim == 1 and fs is not None:
            return np.arange(data.shape[0]) / fs, {"Signal": data}
        if data.ndim == 2:
            if fs is not None:
                signals = {f"Signal {i}": data[:, i] for i in range(data.shape[1])}
                return np.arange(data.shape[0]) / fs, signals
            signals = {f"Signal {i}": data[:, i] for i in range(1, data.shape[1])}
            return data[:, 0], signals
        raise ValueError("Waveform data must be 1D or 2D numpy array")

    raise ValueError("Waveform data must be DataFrame, dict, or numpy array")


def apply_scaling(
    time: np.ndarray,
    signals: dict[str, np.ndarray],
    time_scale: str = "s",
    signal_scale: str = "V",
) -> tuple[np.ndarray, dict[str, np.ndarray], str, str]:
    """Scale time and signal values and return labels."""
    time_scale_factor = 1.0
    if time_scale == "ms":
        time_scale_factor, time_label = 1e3, "Time [ms]"
    elif time_scale == "us":
        time_scale_factor, time_label = 1e6, "Time [us]"
    elif time_scale == "ns":
        time_scale_factor, time_label = 1e9, "Time [ns]"
    else:
        time_label = "Time [s]"

    signal_scale_factor, signal_label = format_signal_scale_label(signal_scale)

    time_scaled = time * time_scale_factor
    signals_scaled = {k: v * signal_scale_factor for k, v in signals.items()}
    return time_scaled, signals_scaled, time_label, signal_label


def extract_metadata_info(data: pd.DataFrame | dict[str, np.ndarray] | np.ndarray) -> str:
    """Extract metadata summary for plot title suffix."""
    metadata_parts = []
    if isinstance(data, pd.DataFrame) and hasattr(data, "attrs") and data.attrs:
        attrs = data.attrs
        if "source_file_type" in attrs:
            metadata_parts.append(f"Type: {attrs['source_file_type']}")
        if "source_file_format" in attrs:
            metadata_parts.append(f"Format: {attrs['source_file_format']}")
        if "metadata" in attrs and isinstance(attrs["metadata"], dict):
            metadata = attrs["metadata"]
            if "record_length" in metadata:
                metadata_parts.append(f"Length: {metadata['record_length']}")
            if "time_resolution" in metadata:
                resolution = metadata["time_resolution"]
                if resolution < 1e-6:
                    metadata_parts.append(f"Resolution: {resolution * 1e9:.1f} ns")
                elif resolution < 1e-3:
                    metadata_parts.append(f"Resolution: {resolution * 1e6:.1f} us")
                elif resolution < 1:
                    metadata_parts.append(f"Resolution: {resolution * 1e3:.1f} ms")
                else:
                    metadata_parts.append(f"Resolution: {resolution:.3f} s")
    return " | ".join(metadata_parts) if metadata_parts else ""
