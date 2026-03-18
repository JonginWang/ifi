#!/usr/bin/env python3
"""Unit tests for analysis workflow utilities."""

import numpy as np
import pandas as pd

from ifi.analysis.workflow import (
    build_combined_signals_by_frequency,
    build_plot_overview_maps,
    build_signals_dict_for_hdf5,
    evaluate_cached_results_summary,
    has_density_data,
    merge_cached_results_into_analysis_bundle,
)


def _make_raw_df(n: int = 32) -> pd.DataFrame:
    t = np.linspace(0.0, 1.0e-6, n)
    return pd.DataFrame({"TIME": t, "CH0": np.sin(np.linspace(0, 2 * np.pi, n))})


def test_has_density_data_for_dataframe_and_dict():
    assert has_density_data(pd.DataFrame({"a": [1, 2, 3]})) is True
    assert has_density_data(pd.DataFrame()) is False
    assert has_density_data({"freq_94G": pd.DataFrame({"a": [1]})}) is True
    assert has_density_data({"freq_94G": pd.DataFrame()}) is False


def test_build_signals_dict_for_hdf5_uses_source_stem_across_extensions():
    shot_raw_data = {
        "45821_056.mat": _make_raw_df(),
        "45821_789.dat": _make_raw_df(),
        "45821_ALL.csv": _make_raw_df(),
    }
    shot_interferometry_params = {
        "45821_056.mat": {
            "freq": 94.0e9,
            "n_ch": 5,
            "n_path": 2,
            "meas_name": "94GHz Interferometer",
        },
        "45821_789.dat": {
            "freq": 94.0e9,
            "n_ch": 5,
            "n_path": 2,
            "meas_name": "94GHz Interferometer",
        },
        "45821_ALL.csv": {
            "freq": 282.0e9,
            "n_ch": 1,
            "n_path": 1,
            "meas_name": "280GHz Interferometer",
        },
    }

    signals = build_signals_dict_for_hdf5(
        shot_raw_data,
        shot_interferometry_params,
    )

    assert "45821_056.mat" in signals
    assert "45821_789.dat" in signals
    assert "45821_ALL.csv" in signals

    df_94 = signals["45821_056.mat"]
    df_280 = signals["45821_ALL.csv"]

    assert "TIME" in df_94.columns
    assert "TIME" in df_280.columns
    assert "CH0" in df_94.columns
    assert "CH0" in df_280.columns
    assert df_94.attrs["freq"] == 94.0e9
    assert df_94.attrs["n_ch"] == 5
    assert df_94.attrs["n_path"] == 2
    assert df_94.attrs["meas_name"] == "94GHz Interferometer"
    assert df_280.attrs["freq"] == 282.0e9


def test_evaluate_cached_results_summary_respects_requested_freqs():
    signal_df = pd.DataFrame({"TIME": [0.0, 1.0], "CH0": [0.1, 0.2]})
    signal_df.attrs["freq"] = 94.0e9
    results = {
        "rawdata": {"45821_056.csv": signal_df},
        "stft": {"x": {}},
        "cwt": {"x": {}},
        "density": {"freq_94G": pd.DataFrame({"ne_CH1": [1.0, 2.0]})},
    }
    summary = evaluate_cached_results_summary(
        results=results,
        requested_freqs=[94.0, 280.0],
        need_stft=True,
        need_cwt=True,
        need_density=True,
    )

    assert summary["has_signals"] is False
    assert summary["has_stft"] is True
    assert summary["has_cwt"] is True
    assert summary["has_density"] is True
    assert summary["available_freqs"] == {94.0}
    assert summary["missing_requested_freqs"] == {280.0}


def test_build_combined_signals_by_frequency_for_94_and_280():
    shot_nas_data = {
        "45821_056.csv": _make_raw_df(),
        "45821_789.csv": _make_raw_df(),
        "45821_ALL.csv": _make_raw_df(),
    }
    freq_groups = {
        94.0: {"files": ["45821_056.csv", "45821_789.csv"], "params": []},
        280.0: {"files": ["45821_ALL.csv"], "params": []},
    }

    combined = build_combined_signals_by_frequency(shot_nas_data, freq_groups)

    assert set(combined.keys()) == {94.0, 280.0}
    assert combined[94.0].index.name == "TIME"
    assert "CH0_45821_056" in combined[94.0].columns
    assert "CH0_45821_789" in combined[94.0].columns
    # 280GHz keeps original channel names.
    assert "CH0" in combined[280.0].columns


def test_merge_cached_results_into_analysis_bundle_fills_missing_only():
    analyzed_bundle = {
        "processed_data": {"signals": {"94.0": pd.DataFrame({"a": [1]})}, "density": {}},
        "analysis_results": {"stft": {}, "cwt": {"new": {}}},
        "raw_data": {},
    }
    cached_results = {
        "rawdata": {"45821_ALL.csv": pd.DataFrame({"b": [2]})},
        "density": {"freq_94G": pd.DataFrame({"ne_x": [1.0]})},
        "stft": {"cached_stft": {}},
        "cwt": {"cached_cwt": {}},
        "vestdata": {"25k": pd.DataFrame({"Ip": [0.0]})},
    }

    merged = merge_cached_results_into_analysis_bundle(analyzed_bundle, cached_results)

    # Keep newly analyzed signals.
    assert "94.0" in merged["processed_data"]["signals"]
    # Fill missing density/stft/vest from cache.
    assert has_density_data(merged["processed_data"]["density"]) is True
    assert merged["analysis_results"]["stft"] == {"cached_stft": {}}
    # Keep newly analyzed cwt.
    assert merged["analysis_results"]["cwt"] == {"new": {}}
    assert "vest" in merged["raw_data"]


def test_build_plot_overview_maps_handles_multifrequency_dicts():
    combined_signals = {
        94.0: pd.DataFrame({"CH0": [1.0, 2.0]}),
        280.0: pd.DataFrame({"CH0": [3.0, 4.0]}),
    }
    density_data = {
        "freq_94G": pd.DataFrame({"ne_A": [10.0, 11.0]}),
        "freq_280G": pd.DataFrame({"ne_B": [12.0, 13.0]}),
    }

    signal_map, density_map = build_plot_overview_maps(combined_signals, density_data)

    assert set(signal_map.keys()) == {
        "Processed Signals (94 GHz)",
        "Processed Signals (280 GHz)",
    }
    assert set(density_map.keys()) == {"Density (94 GHz)", "Density (280 GHz)"}
