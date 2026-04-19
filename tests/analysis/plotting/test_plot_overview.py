from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ifi.plots.plot import Plotter
from ifi.plots.plot_density import plot_density_core


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
