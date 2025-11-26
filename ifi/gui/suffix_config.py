#!/usr/bin/env python3
"""
Suffix Configuration
====================

This module provides helpers for loading and accessing suffix and channel
configuration used by the Tektronix data automator GUI. Configuration is
read from ``ifi/gui/suffix.ini`` using :mod:`configparser`.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SuffixProfile:
    """
    Simple container for suffix and associated channel names.

    Attributes:
        suffix: The configured suffix string, including the leading underscore
            (for example ``\"_056\"`` or ``\"_ALL\"``).
        channels: A list of channel column names in the order they should
            appear in saved data (e.g. ``[\"TIME\", \"CH1\", \"CH2\"]``).
    """

    suffix: str
    channels: List[str]


@dataclass
class MachineSuffixConfig:
    """
    Resolved suffix configuration for a specific machine type.

    Attributes:
        machine_key: Key used to look up the default suffix in the
            ``[defaults]`` section (for example ``\"94G\"`` or ``\"280G\"``).
        profile: The resolved :class:`SuffixProfile` for this machine.
    """

    machine_key: str
    profile: SuffixProfile


def load_suffix_config(
    config_path: Path, machine_key: str
) -> Optional[MachineSuffixConfig]:
    """
    Load suffix configuration for a specific machine.

    This reads ``suffix.ini`` and resolves:

    - The default suffix for the given ``machine_key`` from the ``[defaults]``
      section.
    - The channel list for that suffix from the ``[channels]`` section.

    Args:
        config_path: Path to the ``suffix.ini`` file.
        machine_key: Key under the ``[defaults]`` section that identifies the
            current machine (for example ``\"94G\"`` or ``\"280G\"``).

    Returns:
        MachineSuffixConfig | None: The resolved configuration, or ``None`` if
        the configuration file is missing or incomplete.
    """
    if not config_path.exists():
        return None

    parser = configparser.ConfigParser()
    parser.read(config_path)

    if "defaults" not in parser or "channels" not in parser:
        return None

    defaults = parser["defaults"]
    channels_section = parser["channels"]

    if machine_key not in defaults:
        return None

    suffix = defaults.get(machine_key, "").strip()
    if not suffix:
        return None

    channel_csv = channels_section.get(suffix, "").strip()
    if not channel_csv:
        return None

    channel_names = [name.strip() for name in channel_csv.split(",") if name.strip()]
    if not channel_names:
        return None

    profile = SuffixProfile(suffix=suffix, channels=channel_names)
    return MachineSuffixConfig(machine_key=machine_key, profile=profile)


def build_data_dict_from_channels(
    channels: List[str], time_array, channel_arrays: Dict[str, object]
) -> Dict[str, object]:
    """
    Build a data dictionary suitable for TekScopeController.save_data.

    Args:
        channels: Ordered list of column names, typically from
            :class:`SuffixProfile.channels`.
        time_array: NumPy array representing the time axis.
        channel_arrays: Mapping from channel name (e.g. ``\"CH1\"``) to its
            NumPy data array.

    Returns:
        dict[str, object]: Mapping suitable for constructing a pandas
        DataFrame, where ``\"TIME\"`` is mapped to ``time_array`` and each
        other channel name is mapped from ``channel_arrays`` when available.
        Missing channels are skipped.
    """
    data: Dict[str, object] = {}
    for name in channels:
        if name.upper() == "TIME":
            data[name] = time_array
        else:
            if name in channel_arrays:
                data[name] = channel_arrays[name]
    return data


