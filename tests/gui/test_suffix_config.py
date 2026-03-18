#!/usr/bin/env python3
"""
Tests for suffix and channel configuration helpers.

These tests exercise the pure-Python logic in ``ifi.gui.suffix_config`` using
temporary configuration files. No GUI or hardware components are required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ifi.gui.suffix_config import (
    MachineSuffixConfig,
    SuffixProfile,
    build_data_dict_from_channels,
    load_suffix_config,
)


def test_load_suffix_config_parses_defaults_and_channels(tmp_path: Path) -> None:
    """load_suffix_config should resolve suffix and channels for a machine key."""
    ini = tmp_path / "suffix.ini"
    ini.write_text(
        "[defaults]\n"
        "94G=_056\n"
        "\n"
        "[channels]\n"
        "_056=TIME,CH1,CH2\n",
        encoding="utf-8",
    )

    cfg = load_suffix_config(ini, machine_key="94G")
    assert isinstance(cfg, MachineSuffixConfig)
    assert cfg.machine_key == "94G"
    assert cfg.profile.suffix == "_056"
    assert cfg.profile.channels == ["TIME", "CH1", "CH2"]


def test_build_data_dict_from_channels_uses_time_and_known_channels() -> None:
    """build_data_dict_from_channels should map TIME and known channels only."""
    time_arr = np.array([0.0, 1.0], dtype=float)
    ch1 = np.array([0.1, 0.2], dtype=float)
    ch2 = np.array([0.3, 0.4], dtype=float)

    channels = ["TIME", "CH1", "CH2", "CH3"]
    channel_arrays = {"CH1": ch1, "CH2": ch2}

    data = build_data_dict_from_channels(channels, time_arr, channel_arrays)

    assert list(data.keys()) == ["TIME", "CH1", "CH2"]
    assert data["TIME"] is time_arr
    assert data["CH1"] is ch1
    assert data["CH2"] is ch2


