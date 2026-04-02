#!/usr/bin/env python3
"""
Tests for multi-frequency density save/load behavior.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ifi.utils.io_utils import load_results_from_hdf5, save_results_to_hdf5
from ifi.utils.io_h5 import H5_GROUP_DENSITY


def test_save_load_multifrequency_density_dict(tmp_path):
    """Density dict should persist as per-frequency subgroups and reload as dict."""
    shot_num = 45821
    results_dir = tmp_path / "results" / str(shot_num)

    signal_df = pd.DataFrame(
        {
            "TIME": np.linspace(0.0, 1.0e-3, 64),
            "CH0": np.sin(np.linspace(0, 2 * np.pi, 64)),
        }
    )
    signal_df.attrs["freq"] = 94.0e9
    signal_df.attrs["n_ch"] = 5
    signal_df.attrs["n_path"] = 2
    signal_df.attrs["meas_name"] = "94GHz Interferometer"
    density_data = {
        "freq_94G": pd.DataFrame(
            {"ne_CH1_45821_056": np.random.randn(64)},
            index=signal_df["TIME"].to_numpy(),
        ),
        "freq_280G": pd.DataFrame(
            {"ne_CH1_45821_ALL": np.random.randn(64)},
            index=signal_df["TIME"].to_numpy(),
        ),
    }
    density_meta = {
        94.0: {
            "freq": 94.0e9,
            "n_ch": 5,
            "n_path": 2,
            "meas_name": "94GHz Interferometer",
        },
        280.0: {
            "freq": 282.0e9,
            "n_ch": 1,
            "n_path": 1,
            "meas_name": "280GHz Interferometer",
        },
    }

    saved_path = save_results_to_hdf5(
        output_dir=str(results_dir),
        shot_num=shot_num,
        signals={"45821_056.csv": signal_df},
        stft_results={},
        cwt_results={},
        density_data=density_data,
        vest_data=None,
        density_meta_by_freq=density_meta,
    )

    assert saved_path is not None
    h5_path = Path(saved_path)
    assert h5_path.exists()

    with h5py.File(h5_path, "r") as hf:
        assert H5_GROUP_DENSITY in hf
        density_group = hf[H5_GROUP_DENSITY]
        assert "freq_94G" in density_group
        assert "freq_280G" in density_group
        assert "ne_CH1_45821_056" in density_group["freq_94G"]
        assert "ne_CH1_45821_ALL" in density_group["freq_280G"]
        assert density_group["freq_94G"].attrs["freq"] == 94.0e9
        assert density_group["freq_280G"].attrs["freq"] == 282.0e9

    loaded = load_results_from_hdf5(shot_num=shot_num, base_dir=str(tmp_path / "results"))
    assert loaded is not None
    assert "density" in loaded
    assert isinstance(loaded["density"], dict)
    assert "freq_94G" in loaded["density"]
    assert "freq_280G" in loaded["density"]
    assert "ne_CH1_45821_056" in loaded["density"]["freq_94G"].columns
    assert "ne_CH1_45821_ALL" in loaded["density"]["freq_280G"].columns
    assert "TIME" in loaded["density"]["freq_94G"].columns
    assert "TIME" in loaded["density"]["freq_280G"].columns
    np.testing.assert_allclose(
        loaded["density"]["freq_94G"]["TIME"].to_numpy(),
        signal_df["TIME"].to_numpy(),
    )
    np.testing.assert_allclose(
        loaded["density"]["freq_280G"]["TIME"].to_numpy(),
        signal_df["TIME"].to_numpy(),
    )
