#!/usr/bin/env python3
"""
Vest Codes
===========

This module contains the utilities for Vest codes.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .. import IFI_ROOT


def default_vest_field_csv() -> Path:
    """Return the default VEST field metadata CSV path in this codebase."""
    if IFI_ROOT is not None and isinstance(IFI_ROOT, Path):  # noqa: F823
        repo_candidate = IFI_ROOT / "ifi" / "db_controller" / "shotdata_field_260101.csv"
        if repo_candidate.exists():
            return repo_candidate
        package_candidate = IFI_ROOT / "db_controller" / "shotdata_field_260101.csv"
        if package_candidate.exists():
            return package_candidate
    fallback_root = Path(__file__).resolve().parent.parent
    return fallback_root / "db_controller" / "shotdata_field_260101.csv"


def format_vest_field_label(field_name: str, field_unit: str) -> str:
    """Format display label such as 'Ip', 'A' -> 'Ip [A]'."""
    name = str(field_name).strip()
    unit = str(field_unit).strip()
    return f"{name} [{unit}]" if unit else name


def load_vest_field_maps(
    csv_path: str | Path | None = None,
) -> tuple[dict[int, dict[str, str]], dict[str, int]]:
    """
    Load VEST field metadata maps from CSV.

    Returns:
        - by_id: {field_id: {"field_name": ..., "field_unit": ...}}
        - label_to_id: {"field_name [unit]": field_id, "field_name (unit)": field_id, ...}
    """
    path = Path(csv_path) if csv_path is not None else default_vest_field_csv()
    by_id: dict[int, dict[str, str]] = {}
    label_to_id: dict[str, int] = {}
    if not path.exists():
        return by_id, label_to_id

    try:
        df = pd.read_csv(path)
    except Exception:
        return by_id, label_to_id

    for _, row in df.iterrows():
        try:
            field_id = int(row["field_id"])
        except Exception:
            continue

        field_name = str(row.get("field_name", "")).strip()
        field_unit = str(row.get("field_unit", "")).strip()
        by_id[field_id] = {"field_name": field_name, "field_unit": field_unit}

        # Canonical format for 0.1.1.
        label_to_id[format_vest_field_label(field_name, field_unit)] = field_id
        # Tolerate previous formatting variants for robust matching.
        label_to_id[f"{field_name} ({field_unit})"] = field_id
        label_to_id[field_name] = field_id

    return by_id, label_to_id


def infer_field_meta(
    column_name: str,
    by_id: dict[int, dict[str, str]],
    label_to_id: dict[str, int],
) -> tuple[int | None, str, str]:
    """Infer field metadata from a VEST column name."""
    text = str(column_name).strip()

    field_id: int | None = None
    if text.isdigit():
        field_id = int(text)
    elif text in label_to_id:
        field_id = label_to_id[text]
    else:
        match = re.match(r"^field[_\s]*(\d+)$", text, flags=re.IGNORECASE)
        if match:
            field_id = int(match.group(1))

    if field_id is not None and field_id in by_id:
        meta = by_id[field_id]
        return field_id, meta.get("field_name", text), meta.get("field_unit", "")

    return field_id, text, ""


__all__ = [
    "default_vest_field_csv",
    "format_vest_field_label",
    "load_vest_field_maps",
    "infer_field_meta",
]

