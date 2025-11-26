#!/usr/bin/env python3
"""
Tests for CSV to HDF5 conversion utilities.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ifi.utils.file_io import convert_to_hdf5


def test_convert_to_hdf5_creates_waveform_group_and_datasets(tmp_path: Path) -> None:
    """convert_to_hdf5 should create /waveform group with one dataset per CSV column."""
    src = tmp_path / "test.csv"
    dst = tmp_path / "test.h5"

    df = pd.DataFrame(
        {
            "TIME": np.array([0.0, 1.0], dtype=float),
            "CH1": np.array([0.1, 0.2], dtype=float),
        }
    )
    df.to_csv(src, index=False)

    convert_to_hdf5(src, dst)

    assert dst.exists()
    with h5py.File(dst, "r") as h5f:
        assert "waveform" in h5f
        group = h5f["waveform"]
        assert "TIME" in group
        assert "CH1" in group


