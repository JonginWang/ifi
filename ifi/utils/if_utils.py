#!/usr/bin/env python3
"""
Interferometry parameter helpers
=================================

This module contains the functions for the IFI project.
It includes the interferometry parameter helpers.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import ast
import configparser
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_process_common import append_source_stem_to_columns, ensure_time_indexed_df


def get_default_interferometry_config_path() -> Path:
    """Return the canonical interferometry config path."""
    return Path(__file__).resolve().parent.parent / "analysis" / "config_if.ini"


@lru_cache(maxsize=None)
def _load_interferometry_config(config_path_str: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path_str, encoding="utf-8")
    return config


def _get_interferometry_config(
    config_path: str | Path | None
) -> configparser.ConfigParser:
    resolved_path = Path(config_path) if config_path is not None else get_default_interferometry_config_path()
    return _load_interferometry_config(str(resolved_path))


def _parse_rule_value(raw_value: str) -> Any:
    text = str(raw_value).strip()
    if text == "":
        return ""
    if text.lower() in {"none", "null"}:
        return None
    if text.startswith(("[", "(", "{", '"', "'")):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
    return text


def _iter_assignment_rules(config: configparser.ConfigParser) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for section in config.sections():
        if not section.lower().startswith("channel_assign."):
            continue
        rule = {"section": section}
        for key, value in config.items(section):
            rule[key] = _parse_rule_value(value)
        rules.append(rule)
    return rules


def _resolve_profile_section_name(
    config: configparser.ConfigParser,
    section: str,
) -> str:
    candidates = [str(section).strip()]
    if "." not in str(section):
        candidates.append(f"freq_feature.{section}")
    for candidate in candidates:
        if config.has_section(candidate):
            return candidate
    raise ValueError(f"Section {section} not found in config file.")


def _rule_matches(rule: dict[str, Any], shot_num: int, basename: str) -> bool:
    shot_min = rule.get("shot_min")
    if shot_min is not None and shot_num < int(shot_min):
        return False
    shot_max = rule.get("shot_max")
    if shot_max is not None and shot_num > int(shot_max):
        return False
    filename_contains = rule.get("filename_contains")
    if filename_contains and str(filename_contains) not in basename:
        return False
    filename_regex = rule.get("filename_regex")
    if filename_regex and re.search(str(filename_regex), basename) is None:
        return False
    return True


def _build_assignment_params(
    rule: dict[str, Any],
    base_params: dict[str, Any],
) -> dict[str, Any]:
    params = {
        "method": rule.get("method", "unknown"),
        **base_params,
        "ref_col": rule.get("ref_col"),
        "probe_cols": rule.get("probe_cols", []),
    }
    if "amp_ref_col" in rule:
        params["amp_ref_col"] = rule.get("amp_ref_col")
    if "amp_probe_cols" in rule:
        params["amp_probe_cols"] = rule.get("amp_probe_cols", [])
    return params


def get_interferometry_params_by_section(
    config_path: str | Path | None,
    section: str,
) -> dict[str, Any]:
    """Read one interferometer section from config."""
    config = _get_interferometry_config(config_path)
    resolved_section = _resolve_profile_section_name(config, section)
    return {
        "freq": float(config.get(resolved_section, "freq")),
        "freq_ghz": float(config.get(resolved_section, "freq")) / 1.0e9,
        "n_ch": int(config.get(resolved_section, "n_ch")),
        "n_path": int(config.get(resolved_section, "n_path")),
        "meas_name": config.get(resolved_section, "name").strip().strip('"'),
    }


def assign_interferometry_params_to_shot(
    shot_num: int,
    filename: str,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve interferometry parameters from shot number and filename."""
    basename = Path(filename).name
    config = _get_interferometry_config(config_path)
    base_params_by_profile = {
        section: get_interferometry_params_by_section(config_path, section)
        for section in config.sections()
        if section.startswith("freq_feature.")
    }

    for rule in _iter_assignment_rules(config):
        if not _rule_matches(rule, shot_num, basename):
            continue
        profile = _resolve_profile_section_name(
            config,
            str(rule.get("profile", "freq_feature.94GHz")),
        )
        if profile not in base_params_by_profile:
            raise ValueError(f"Profile {profile} not found in config file.")
        return _build_assignment_params(rule, base_params_by_profile[profile])

    raise ValueError("No interferometry assignment rule matched and no default rule was configured.")


def map_frequency_to_group(freq_ghz: float) -> float:
    """Map measured frequency to standard interferometry groups."""
    if 93.0 <= freq_ghz <= 95.0:
        return 94.0
    if 275.0 <= freq_ghz <= 285.0:
        return 280.0
    return float(freq_ghz)


def parse_frequency_group_from_signal_name(signal_name: str) -> float | None:
    """
    Infer frequency group from a signal identifier.

    Supports:
    - `freq_94.0_GHz`, `freq_94_0_GHz`
    - stem/suffix patterns such as `<shot>_ALL`, `<shot>_056`, `<shot>_789`
    """
    text = str(signal_name).strip()
    if not text:
        return None

    lower_text = text.lower()
    freq_match = re.search(r"freq_([0-9]+(?:[._][0-9]+)?)_ghz", lower_text)
    if freq_match:
        raw = freq_match.group(1).replace("_", ".")
        try:
            return map_frequency_to_group(float(raw))
        except ValueError:
            return None

    stem_upper = Path(text).stem.upper()
    if stem_upper.endswith("_ALL") or "_ALL_" in stem_upper:
        return 280.0
    if re.search(r"_\d{3}$", stem_upper):
        return 94.0
    return None


def extract_available_frequency_groups(signal_names: list[str]) -> set[float]:
    """Extract available frequency groups from signal name list."""
    groups: set[float] = set()
    for name in signal_names:
        freq = parse_frequency_group_from_signal_name(name)
        if freq is not None:
            groups.add(freq)
    return groups


def build_frequency_groups_from_params(
    shot_interferometry_params: dict[str, dict],
) -> tuple[dict[float, dict[str, list]], list[tuple[str, float, float]], list[tuple[str, float]]]:
    """
    Build frequency groups from interferometry parameter mapping.

    Returns:
        - freq_groups: `{group_freq: {"files": [...], "params": [...]}}`
        - mapped_infos: list of `(basename, freq_ghz, group_freq)` where mapping changed
        - out_of_standard_infos: list of `(basename, freq_ghz)` outside 93-95/275-285
    """
    freq_groups: dict[float, dict[str, list]] = {}
    mapped_infos: list[tuple[str, float, float]] = []
    out_of_standard_infos: list[tuple[str, float]] = []

    for basename, params in shot_interferometry_params.items():
        freq_ghz = float(params.get("freq_ghz", 94.0))
        group_freq = map_frequency_to_group(freq_ghz)

        if group_freq not in freq_groups:
            freq_groups[group_freq] = {"files": [], "params": []}
        freq_groups[group_freq]["files"].append(basename)
        freq_groups[group_freq]["params"].append(params)

        if freq_ghz != group_freq:
            mapped_infos.append((basename, freq_ghz, group_freq))
        if not (93.0 <= freq_ghz <= 95.0 or 275.0 <= freq_ghz <= 285.0):
            out_of_standard_infos.append((basename, freq_ghz))

    return freq_groups, mapped_infos, out_of_standard_infos


def filter_frequency_groups(
    freq_groups: dict[float, dict[str, list]],
    requested_freqs: list[float] | None,
) -> tuple[dict[float, dict[str, list]], bool]:
    """Filter frequency groups by requested frequencies."""
    if not requested_freqs:
        return freq_groups, True
    allowed = set(requested_freqs)
    filtered = {freq: info for freq, info in freq_groups.items() if freq in allowed}
    return filtered, bool(filtered)


def get_frequency_df(
    freq_df_dict: dict,
    freq_ghz: float | None,
    default: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Get frequency DataFrame from dict that may use float or string keys."""
    if default is None:
        default = pd.DataFrame()
    if freq_ghz is None or not isinstance(freq_df_dict, dict):
        return default

    candidates = [
        freq_ghz,
        str(freq_ghz),
        f"{freq_ghz:g}",
    ]
    for key in candidates:
        if key in freq_df_dict:
            df = freq_df_dict[key]
            if isinstance(df, pd.DataFrame):
                return df
    return default

def build_group_signal_column_name(
    freq_ghz: float,
    signal_name: str,
    source_name: str,
) -> str:
    """Build merged signal column name for a frequency group and source."""
    if map_frequency_to_group(float(freq_ghz)) == 280.0:
        return str(signal_name)
    return f"{signal_name}_{Path(source_name).stem}"


def build_combined_signals_by_frequency(
    shot_nas_data: dict[str, pd.DataFrame],
    freq_groups: dict[float, dict[str, list]],
) -> dict[float, pd.DataFrame]:
    """
    Build frequency-keyed combined signal DataFrames for analysis.

    Behavior:
    - `280.0` GHz: use `_ALL` source directly.
    - `94.0` GHz: use `_0xx` source as TIME reference and merge others with nearest reindex.
    """
    combined: dict[float, pd.DataFrame] = {}

    for freq_ghz, group_info in freq_groups.items():
        files_in_group = group_info.get("files", [])

        if freq_ghz == 280.0:
            all_file = next(
                (
                    basename
                    for basename in files_in_group
                    if "_ALL" in basename and basename in shot_nas_data
                ),
                None,
            )
            if all_file is None:
                continue
            all_df, _ = ensure_time_indexed_df(shot_nas_data[all_file])
            combined[freq_ghz] = all_df
            continue

        if freq_ghz != 94.0:
            continue

        ref_file = None
        other_files: list[str] = []
        for basename in files_in_group:
            if basename not in shot_nas_data:
                continue
            if "_0" in basename:
                ref_file = basename
            else:
                other_files.append(basename)

        if ref_file is None:
            continue

        ref_df, ref_time_axis = ensure_time_indexed_df(shot_nas_data[ref_file])
        combined_dfs: list[pd.DataFrame] = [
            append_source_stem_to_columns(ref_df, ref_file)
        ]

        for other_file in other_files:
            other_df, _ = ensure_time_indexed_df(shot_nas_data[other_file])
            other_df = other_df.reindex(ref_time_axis, method="nearest", limit=1)
            combined_dfs.append(append_source_stem_to_columns(other_df, other_file))

        merged_df = pd.concat(combined_dfs, axis=1)
        merged_df.index.name = "TIME"
        combined[freq_ghz] = merged_df

    return combined


__all__ = [
    "get_default_interferometry_config_path",
    "get_interferometry_params_by_section",
    "assign_interferometry_params_to_shot",
    "build_frequency_groups_from_params",
    "extract_available_frequency_groups",
    "build_combined_signals_by_frequency",
    "build_group_signal_column_name",
    "filter_frequency_groups",
    "get_frequency_df",
    "map_frequency_to_group",
    "parse_frequency_group_from_signal_name",
]
