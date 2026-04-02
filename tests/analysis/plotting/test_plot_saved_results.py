#!/usr/bin/env python3
"""
Tests for saved-results plotting helpers.
"""

import numpy as np
import pandas as pd

from ifi.plots.plot_density import resolve_density_time_data
from ifi.plots.plot_saved_results import (
    SavedSeriesRequest,
    best_vest_series,
    extract_time_and_series,
    raw_sources_by_frequency,
)


def test_extract_time_and_series_prefers_density_time_column():
    density_df = pd.DataFrame(
        {
            "TIME": np.array([0.1, 0.2, 0.3]),
            "ne_CH1_00001_056": np.array([1.0, 2.0, 3.0]),
        }
    )
    results = {"density": {"freq_94G": density_df}}

    time, values = extract_time_and_series(
        results,
        SavedSeriesRequest(
            group="density",
            source="freq_94G",
            column="ne_CH1_00001_056",
            label="density",
        ),
        trigger_time=0.0,
    )

    np.testing.assert_allclose(time.to_numpy(), np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(values.to_numpy(), np.array([1.0, 2.0, 3.0]))


def test_extract_time_and_series_falls_back_to_raw_time_for_density():
    density_df = pd.DataFrame({"ne_CH1_00001_056": np.array([1.0, 2.0, 3.0])})
    raw_df = pd.DataFrame(
        {
            "TIME": np.array([0.01, 0.02, 0.03]),
            "CH0": np.array([0.0, 0.0, 0.0]),
            "CH1": np.array([1.0, 2.0, 3.0]),
        }
    )
    raw_df.attrs["freq"] = 94.0

    results = {
        "density": {"freq_94G": density_df},
        "rawdata": {"00001_056.csv": raw_df},
    }

    time, values = extract_time_and_series(
        results,
        SavedSeriesRequest(
            group="density",
            source="freq_94G",
            column="ne_CH1_00001_056",
            label="density",
        ),
        trigger_time=0.0,
    )

    np.testing.assert_allclose(time.to_numpy(), np.array([0.01, 0.02, 0.03]))
    np.testing.assert_allclose(values.to_numpy(), np.array([1.0, 2.0, 3.0]))


def test_resolve_density_time_data_falls_back_from_range_index_to_raw_time():
    density_df = pd.DataFrame({"ne_CH1_00001_056": np.array([1.0, 2.0, 3.0])})
    raw_df = pd.DataFrame(
        {
            "TIME": np.array([0.01, 0.02, 0.03]),
            "CH0": np.array([0.0, 0.0, 0.0]),
        }
    )
    raw_df.attrs["freq"] = 94.0

    resolved = resolve_density_time_data(
        density_df,
        density_name="freq_94G",
        signals_dict={"00001_056.csv": raw_df},
    )

    np.testing.assert_allclose(resolved, np.array([0.01, 0.02, 0.03]))


def test_raw_sources_by_frequency_can_infer_from_source_name_without_attrs():
    raw_df_94 = pd.DataFrame({"TIME": np.array([0.01, 0.02]), "CH0": np.array([1.0, 2.0])})
    raw_df_280 = pd.DataFrame({"TIME": np.array([0.01, 0.02]), "CH0": np.array([1.0, 2.0])})
    results = {
        "rawdata": {
            "47807_056.csv": raw_df_94,
            "47807_ALL.csv": raw_df_280,
        }
    }

    grouped = raw_sources_by_frequency(results)
    assert 94.0 in grouped
    assert 280.0 in grouped


def test_best_vest_series_accepts_bracket_variants():
    vest_df = pd.DataFrame({"Ip_raw [[V]]": np.array([1.0, 2.0, 3.0])})
    results = {"vestdata": {"25k": vest_df}}

    selected = best_vest_series(results)

    assert selected is not None
    assert selected.group == "vest"
    assert selected.source == "25k"
    assert selected.column == "Ip_raw [[V]]"
