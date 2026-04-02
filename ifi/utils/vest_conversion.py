#!/usr/bin/env python3
"""
VEST conversion helpers
=======================

Non-destructive conversion of VEST raw waveforms into derived engineering units.
Converted results are intended for local HDF5 storage only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .io_h5 import H5_GROUP_VEST, h5_safe_name
from .io_process_write import write_structured_vest_groups
from .vest_fieldcode import load_vest_field_maps


@dataclass(frozen=True)
class VestConversionSpec:
    raw_field: int
    converted_field: int
    label: str
    unit: str


VEST_CONVERSION_SPECS: tuple[VestConversionSpec, ...] = (
    VestConversionSpec(109, 102, "Plasma Current", "A"),
    VestConversionSpec(27, 103, "6kW ECH Forward Power at Main Chamber", "W"),
    VestConversionSpec(28, 104, "6kW ECH Reverse Power at Main Chamber", "W"),
    VestConversionSpec(12, 13, "Main Chamber Pressure", "Torr"),
    VestConversionSpec(219, 220, "MCP Chamber Pressure", "Torr"),
    VestConversionSpec(159, 164, "ECH Power Diagnostics Forward #1", "W"),
    VestConversionSpec(160, 165, "ECH Power Diagnostics Reflect #1", "W"),
    VestConversionSpec(161, 166, "ECH Power Diagnostics Forward #2", "W"),
    VestConversionSpec(162, 167, "ECH Power Diagnostics Reflect #2", "W"),
    VestConversionSpec(96, 249, "7.875 GHz Klystron Oscillator out power", "W"),
    VestConversionSpec(97, 250, "7.875 GHz Klystron Forward power", "W"),
    VestConversionSpec(98, 251, "7.875 GHz Klystron Reflected out power", "W"),
    VestConversionSpec(228, 232, "10kW ECH Forward Power 1", "W"),
    VestConversionSpec(230, 234, "10kW ECH Reverse Power 1", "W"),
    VestConversionSpec(229, 233, "10kW ECH Forward Power 2", "W"),
    VestConversionSpec(231, 235, "10kW ECH Reverse Power 2", "W"),
)

def _normalize_unit_text(unit: str) -> str:
    text = str(unit).strip()
    return text[1:-1] if text.startswith("[") and text.endswith("]") else text


def _convert_values(shot_num: int, spec: VestConversionSpec, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if spec.converted_field == 102:
        if shot_num <= 42850:
            return raw / (2.0e-8 * 1000.0)
        return raw / (1.0e-8 * 1000.0)
    if spec.converted_field in {103, 104, 164, 165, 166, 167}:
        return 1000.0 * np.power(10.0, (raw - 1.01745) / (-0.2763))
    if spec.converted_field in {13, 220}:
        return 2.4 * np.power(10.0, 1.667 * raw - 11.46)
    if spec.converted_field == 249:
        return np.power(10.0, 5.0144 - 4.2689 * raw)
    if spec.converted_field == 250:
        return np.power(10.0, 7.7180 - 4.2689 * raw)
    if spec.converted_field == 251:
        return np.power(10.0, 7.7144 - 4.2689 * raw)
    if spec.converted_field in {232, 233, 234, 235}:
        return np.power(10.0, (raw - 0.9768) / (-0.249) + 6.3)
    raise KeyError(f"Unsupported converted field: {spec.converted_field}")


def compute_vest_converted_fields(
    shot_num: int,
    raw_by_field: dict[int, dict[str, np.ndarray | float | str]],
) -> dict[str, pd.DataFrame]:
    """
    Build converted VEST DataFrames grouped by sample-rate key.

    Args:
        shot_num: Target shot number.
        raw_by_field: Mapping such as
            {field_id: {"time": np.ndarray, "data": np.ndarray, "rate_key": str}}.
    """
    grouped_series: dict[str, list[pd.Series]] = {}
    field_meta, _ = load_vest_field_maps()

    for spec in VEST_CONVERSION_SPECS:
        raw_info = raw_by_field.get(spec.raw_field)
        if not raw_info:
            continue

        time_axis = np.asarray(raw_info["time"], dtype=float)
        raw_values = np.asarray(raw_info["data"], dtype=float)
        rate_key = str(raw_info["rate_key"])
        converted = _convert_values(shot_num, spec, raw_values)

        meta = field_meta.get(spec.converted_field, {})
        field_name = meta.get("field_name", h5_safe_name(spec.label))
        field_unit = _normalize_unit_text(str(meta.get("field_unit", spec.unit)))
        column_name = f"{field_name} [{field_unit}]" if field_unit else str(field_name)
        if not column_name.endswith("]") and field_unit:
            column_name = f"{field_name} [{field_unit}]"
        series = pd.Series(converted, index=time_axis, name=column_name)
        grouped_series.setdefault(rate_key, []).append(series)

    return {
        rate_key: pd.concat(series_list, axis=1)
        for rate_key, series_list in grouped_series.items()
        if series_list
    }


def append_vest_frames_to_results_h5(
    h5_path: str | Path,
    shot_num: int,
    vest_by_rate: dict[str, pd.DataFrame],
    overwrite: bool = False,
) -> int:
    """Append VEST waveforms into canonical `/vestdata` without clobbering other groups."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, "a") as hf:
        vest_root = hf.require_group(H5_GROUP_VEST)
        return write_structured_vest_groups(
            vest_root,
            shot_num,
            vest_by_rate,
            replace_rate_groups=False,
            overwrite_datasets=overwrite,
        )


__all__ = [
    "VEST_CONVERSION_SPECS",
    "VestConversionSpec",
    "append_vest_frames_to_results_h5",
    "compute_vest_converted_fields",
]
