#!/usr/bin/env python3
"""
Plotting Common Helpers
========================

This module contains shared plotting common helpers.
It includes the functions for formatting unit to LaTeX,
formatting signal scale label, preparing time data,
and applying scaling.

Key Features:
    - Normalize supported input types into (time, signals)
    - Scale time and signal values and return labels
    - Extract metadata summary for plot title suffix

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes as mplAxes
from matplotlib.collections import LineCollection

from ..utils.log_manager import LogManager

logger = LogManager.get_logger(__name__)


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


def colored_line(
    x: list | np.ndarray,
    y: list | np.ndarray,
    c: list | np.ndarray,
    ax: mplAxes,
    autoscale_view: bool = True,
    **lc_kwargs: dict[str, Any],
) -> LineCollection | None:
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Args:
        x (list | np.ndarray): data points along the x-axis
        y (list | np.ndarray): data points along the y-axis
        c (list | np.ndarray): color values
        ax (mplAxes): the axis to plot on
        autoscale_view (bool): if True, autoscale the axis after adding the artist.
        lc_kwargs (dict): additional arguments to pass to 
            matplotlib.collections.LineCollection constructor. 
            This should not include the array keyword argument because
            that is set to the color argument. If provided, it will be overridden.
    """
    if "array" in lc_kwargs:
        logger.warning('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)

    if not (len(x) == len(y) == len(c)):
        raise ValueError(
            "colored_line requires x, y, and c to have the same length "
            f"(got {len(x)}, {len(y)}, {len(c)})"
        )

    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    removed_count = int(len(x) - np.count_nonzero(finite_mask))
    if removed_count > 0:
        logger.warning(
            "colored_line removed %d non-finite point(s) from x/y/c inputs.",
            removed_count,
        )
    x = x[finite_mask]
    y = y[finite_mask]
    c = c[finite_mask]

    if len(x) == 0:
        logger.warning("colored_line received no finite points after filtering.")
        return None
    if len(x) == 1:
        logger.warning("colored_line received 1 finite point. Falling back to a single-color marker.")
        ax.plot(x, y, linestyle="None", marker="o")
        if autoscale_view:
            ax.autoscale_view()
        return None
    if len(x) == 2:
        logger.warning("colored_line received 2 finite points. Falling back to a single-color line.")
        ax.plot(x, y)
        if autoscale_view:
            ax.autoscale_view()
        return None

    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment
    line = ax.add_collection(lc)
    if autoscale_view:
        ax.autoscale_view()
    return line


__all__ = [
    "format_unit_to_latex",
    "format_signal_scale_label",
    "prepare_time_data",
    "apply_scaling",
    "extract_metadata_info",
    "colored_line",
]
