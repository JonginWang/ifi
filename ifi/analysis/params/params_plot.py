#!/usr/bin/env python3
"""
Params Plot
===========

This module contains functions and classes for setting up the plot style for the IFI package.

Functions:
    set_plot_style: Set the plot style for the IFI package.
    FontStyle: A collection of predefined font style dictionaries for plot elements.
"""

import matplotlib as mpl


def set_plot_style(font="Arial", size=8.5):
    """
    Apply a consistent plot style, suitable for scientific publications (e.g., AIP).

    This function sets global matplotlib parameters for font family, sizes,
    and other common style elements.

    Args:
        font (str): The font family to use for all text elements. Default is "Arial".
        size (float): The base font size for labels and ticks. Default is 8.5.

    Examples:
    ```python
    from .params_plot import set_plot_style, FontStyle
    set_plot_style()
    fig, ax = plt.subplots()
    ax.set_title("My Plot Title", **FontStyle.title)
    ax.set_xlabel("X-axis Label", **FontStyle.label)
    ax.set_ylabel("Y-axis Label", **FontStyle.label)
    ```
    """
    mpl.rcParams["font.family"] = font
    mpl.rcParams["font.size"] = size
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["axes.labelsize"] = size
    mpl.rcParams["legend.fontsize"] = size
    mpl.rcParams["xtick.labelsize"] = size
    mpl.rcParams["ytick.labelsize"] = size
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["savefig.dpi"] = 300


class FontStyle:
    """
    A collection of predefined font style dictionaries for plot elements.

    Attributes:
        default (dict): The default font style dictionary.
        title (dict): The title font style dictionary.
        suptitle (dict): The suptitle font style dictionary.
        subtitle (dict): The subtitle font style dictionary.
        label (dict): The label font style dictionary.
        custom10 (dict): The custom 10 font style dictionary.
        custom12 (dict): The custom 12 font style dictionary.
        custom14 (dict): The custom 14 font style dictionary.
        custom16 (dict): The custom 16 font style dictionary.
    """

    default = {"fontname": "Arial", "fontsize": 8.5}
    title = {"fontname": "Arial", "fontsize": 15, "fontweight": "bold"}
    suptitle = {"fontname": "Arial", "fontsize": 15, "fontweight": "bold"}
    subtitle = {"fontname": "Arial", "fontsize": 12, "fontweight": "bold"}
    label = {"fontname": "Arial", "fontsize": 8.5, "fontweight": "bold"}
    custom10 = {"fontname": "Arial", "fontsize": 10}
    custom12 = {"fontname": "Arial", "fontsize": 12}
    custom14 = {"fontname": "Arial", "fontsize": 14}
    custom16 = {"fontname": "Arial", "fontsize": 16}
