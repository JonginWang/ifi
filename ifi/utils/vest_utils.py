#!/usr/bin/env python3
"""
Vest Utilities
===============

This module contains the utilities for flattening the shot list for VEST query
and loading the VEST field maps to match the field names.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .vest_fieldcode import (
    default_vest_field_csv,
    format_vest_field_label,
    infer_field_meta,
    load_vest_field_maps,
)
from .vest_postprocess import FlatShotList


def infer_sample_rate_from_key(rate_key: str) -> float | None:
    """
    Parse sample-rate key (e.g. `SR_25k`, `25k`) into Hz.
    In the case of `A.U.`, return 1.0 for unit-less rate.
    """
    text = str(rate_key).strip()
    if text.upper().startswith("SR_"):
        text = text[3:]
    match = re.match(r"^(\d+(?:\.\d+)?)([gGkKmM]?)$", text)
    if not match:
        return None if not text.upper().endswith("A.U.") else 1.0

    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "k":
        return value * 1_000.0
    if unit == "m":
        return value * 1_000_000.0
    if unit == "g":
        return value * 1_000_000_000.0
    return value


def infer_sample_rate_from_index(index: pd.Index) -> float | None:
    """Infer sample rate from numeric index spacing."""
    if len(index) < 2:
        return None
    try:
        values = index.to_numpy(dtype=float)
        dt = float(np.nanmean(np.diff(values)))
    except Exception:
        return None
    if dt <= 0:
        return None
    return 1.0 / dt


def normalize_sr_group_name(rate_key: str) -> str:
    """Normalize sample-rate group name to `SR_*` format."""
    text = str(rate_key).strip()
    if text.upper().startswith("SR_"):
        return text
    return f"SR_{text}" if text and text[0].isdigit() else "SR_001A.U."


def extract_analysis_attrs(params_in_attrs: dict[str, Any] | None) -> dict[str, Any]:
    """Extract canonical HDF5 analysis attrs from interferometry params in attributes."""
    if not isinstance(params_in_attrs, dict):
        return {}
    attrs: dict[str, Any] = {}
    for key in ("freq", "n_ch", "n_path", "meas_name"):
        if key in params_in_attrs and params_in_attrs[key] is not None:
            attrs[key] = params_in_attrs[key]
    return attrs


__all__ = [
    "FlatShotList",
    "default_vest_field_csv",
    "format_vest_field_label",
    "load_vest_field_maps",
    "infer_field_meta",
    "infer_sample_rate_from_key",
    "infer_sample_rate_from_index",
    "normalize_sr_group_name",
    "extract_analysis_attrs",
]
