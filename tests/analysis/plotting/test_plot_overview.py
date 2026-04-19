from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ifi.plots.plot import Plotter
from ifi.plots import plot_overview
from ifi.plots import plot_density
from ifi.plots.plot_density import plot_density_core
from ifi.plots.plot_saved_results import SavedSeriesRequest, extract_time_and_series


def test_plot_analysis_overview_does_not_forward_plot_envelope_to_density():
    plotter = Plotter()

    time_axis = np.linspace(0.0, 1.0e-3, 400)
    raw_df = pd.DataFrame(
        {
            "TIME": time_axis,
            "CH0": np.sin(2 * np.pi * 1.0e5 * time_axis),
            "CH1": np.cos(2 * np.pi * 1.0e5 * time_axis),
        }
    )
    raw_df.attrs["freq"] = 94.0

    density_df = pd.DataFrame(
        {
            "ne_CH1": np.linspace(1.0, 2.0, len(time_axis)),
        },
        index=time_axis,
    )
    density_df.attrs["freq"] = 94.0

    vest_df = pd.DataFrame(
        {
            "Ip_raw ([V])": np.linspace(0.0, 1.0, len(time_axis)),
        },
        index=time_axis,
    )

    plotter.plot_analysis_overview(
        shot_num=45821,
        signals_dict={"45821_056.csv": raw_df},
        density_dict={"freq_94G": density_df},
        vest_data=vest_df,
        trigger_time=0.0,
        downsample=10,
        plot_envelope=True,
    )

    assert len(plt.get_fignums()) >= 2
    plt.close("all")


def test_plot_density_core_applies_downsample():
    time_axis = np.linspace(0.0, 1.0e-3, 400)
    density_df = pd.DataFrame({"ne_CH1": np.linspace(1.0, 2.0, len(time_axis))})

    fig, ax = plot_density_core(
        density_df,
        time_data=time_axis,
        downsample=20,
        show_plot=False,
    )

    assert len(ax.lines) == 1
    assert len(ax.lines[0].get_xdata()) == 20
    plt.close(fig)


def test_plot_density_core_applies_trigger_time():
    time_axis = np.linspace(0.0, 1.0e-3, 100)
    density_df = pd.DataFrame({"ne_CH1": np.linspace(1.0, 2.0, len(time_axis))})

    fig, ax = plot_density_core(
        density_df,
        time_data=time_axis,
        trigger_time=0.290,
        show_plot=False,
    )

    x_data = ax.lines[0].get_xdata()
    assert np.isclose(x_data[0], 290.0)
    assert np.isclose(x_data[-1], 291.0)
    plt.close(fig)


def test_analysis_overview_forwards_trigger_time_to_density(monkeypatch):
    captured = {}

    def fake_waveforms(**kwargs):
        return None

    def fake_density_core(*args, **kwargs):
        captured["trigger_time"] = kwargs.get("trigger_time")
        fig, ax = plt.subplots()
        return fig, ax

    monkeypatch.setattr(plot_overview, "render_overview_waveforms", fake_waveforms)
    monkeypatch.setattr(plot_density, "plot_density_core", fake_density_core)

    time_axis = np.linspace(0.0, 1.0e-3, 10)
    density_df = pd.DataFrame({"ne_CH1": np.linspace(1.0, 2.0, len(time_axis))}, index=time_axis)

    plot_overview.render_analysis_overview(
        shot_num=47813,
        signals_dict={},
        density_dict={"freq_94G": density_df},
        trigger_time=0.290,
        downsample=1,
    )

    assert captured["trigger_time"] == 0.290
    plt.close("all")


def test_saved_results_trigger_time_applies_to_raw_and_density_not_vest():
    time_axis = np.array([0.0, 0.1, 0.2])
    results = {
        "rawdata": {
            "raw_source": pd.DataFrame({"TIME": time_axis, "CH1": [1.0, 2.0, 3.0]})
        },
        "density": {
            "freq_94G": pd.DataFrame(
                {"TIME": time_axis, "ne_CH1": [4.0, 5.0, 6.0]},
                index=time_axis,
            )
        },
        "vestdata": {
            "25k": pd.DataFrame({"Ip": [7.0, 8.0, 9.0]}, index=time_axis)
        },
    }

    raw_time, _ = extract_time_and_series(
        results,
        SavedSeriesRequest("raw", "raw_source", "CH1", "CH1"),
        trigger_time=0.290,
    )
    density_time, _ = extract_time_and_series(
        results,
        SavedSeriesRequest("density", "freq_94G", "ne_CH1", "ne_CH1"),
        trigger_time=0.290,
    )
    vest_time, _ = extract_time_and_series(
        results,
        SavedSeriesRequest("vest", "25k", "Ip", "Ip"),
        trigger_time=0.290,
    )

    assert np.allclose(raw_time.to_numpy(), time_axis + 0.290)
    assert np.allclose(density_time.to_numpy(), time_axis + 0.290)
    assert np.allclose(vest_time.to_numpy(), time_axis)
