#!/usr/bin/env python3
"""
Plot style
===========

This module contains functions and classes for setting up the plot style.

- TeX convention for math: use LaTeX-style strings (e.g. r'$\\omega$') in labels;
  with tex=False (default), matplotlib's mathtext renders these quickly without
  calling an external LaTeX process.
- Full LaTeX (tex=True) gives publication-quality rendering but is slow when many
  text elements or points are drawn; use only when exporting figures (e.g. via
  use_tex_context() or savefig_publication()).

Classes:
    FontStyle: A collection of predefined font style dictionaries for plot elements.

Functions:
    set_plot_style: Set the plot style for the package.
    use_tex_context: Context manager to enable TeX only within a block (e.g. for one save).
    savefig_publication: Save a figure with TeX enabled only for the save (reduces slowdown).

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl

# Keys we touch when enabling TeX; restored by use_tex_context on exit.
_TEX_RC_KEYS = ("text.usetex", "text.latex.preamble", "font.family")

_LATEX_PREAMBLE = "\n".join([
    r"\usepackage{siunitx}",
    r"\sisetup{detect-all}",
    r"\usepackage{helvet}",
    r"\usepackage{sansmath}",
    r"\sansmath",
])


def set_plot_style(font: str = "sans-serif", size: float = 8.5, tex: bool = False) -> None:
    """
    Apply a consistent plot style, suitable for scientific publications (e.g., AIP).

    Sets global matplotlib parameters for font family, sizes, and common style elements.
    Use tex=False (default) for fast plotting; math in labels (e.g. r'$\\omega$') is
    still rendered via mathtext. Use tex=True only when you need full LaTeX rendering,
    or prefer use_tex_context() / savefig_publication() so TeX runs only when saving.

    Args:
        font: Font family for all text. Default "sans-serif".
        size: Base font size for labels and ticks. Default 8.5.
        tex: If True, use LaTeX for all text (slow with many points/labels). Default False.
    """
    mpl.rcParams["text.usetex"] = tex
    mpl.rcParams["font.family"] = font
    mpl.rcParams["font.size"] = size
    if tex:
        mpl.rcParams["text.latex.preamble"] = _LATEX_PREAMBLE
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["axes.labelsize"] = size
    mpl.rcParams["legend.fontsize"] = size
    mpl.rcParams["xtick.labelsize"] = size
    mpl.rcParams["ytick.labelsize"] = size
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["savefig.dpi"] = 300


@contextmanager
def use_tex_context(font: str = "sans-serif"):
    """
    Context manager that enables LaTeX rendering only inside the block.

    Use this to limit TeX overhead to a short scope (e.g. building one figure and
    saving it), instead of setting set_plot_style(tex=True) globally. Previous
    rcParams for text.usetex, text.latex.preamble, and font.family are restored on exit.

    Example:
        set_plot_style(tex=False)   # fast for interactive / many points
        fig, ax = plt.subplots()
        ax.plot(x, y)
        with use_tex_context():
            fig.savefig("paper_fig.pdf")
    """
    backup = {k: mpl.rcParams[k] for k in _TEX_RC_KEYS}
    try:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = _LATEX_PREAMBLE
        mpl.rcParams["font.family"] = font
        yield
    finally:
        for k, v in backup.items():
            mpl.rcParams[k] = v


def savefig_publication(
    fig: mpl.figure.Figure,
    path: str | Path,
    *,
    tex: bool = True,
    **kwargs: Any,
) -> None:
    """
    Save a figure with optional LaTeX rendering applied only for this save.

    When tex=True (default), TeX is enabled temporarily and the figure is re-drawn
    and saved, then previous rcParams are restored. This avoids the slowdown of
    global text.usetex=True during interactive plotting or when plotting many points.

    Args:
        fig: Matplotlib figure to save.
        path: Output path (e.g. "figure.pdf").
        tex: If True, use TeX for this save only. If False, save with current style.
        **kwargs: Passed through to fig.savefig() (e.g. bbox_inches="tight", dpi=300).
    """
    if not tex:
        fig.savefig(path, **kwargs)
        return
    with use_tex_context():
        fig.savefig(path, **kwargs)


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
