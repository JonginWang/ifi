#!/usr/bin/env python3
"""
Runtime/session environment for interactive mode
================================================

This module contains the runtime/session helpers for interactive mode.

Author: Jongin Wang
Date: 2026-03-19
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from ..analysis.params.params_plot import set_plot_style
from ..utils.log_manager import LogManager, log_tag
from ..utils.path_utils import ensure_dir_exists

logger = LogManager().get_logger(__name__)


def _is_jupyter_environment() -> bool:
    """Detect if code is running in a Jupyter notebook environment."""
    try:
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None:
                if hasattr(ipython, "kernel"):
                    return True
                class_name = ipython.__class__.__name__
                if "ZMQ" in class_name or "Jupyter" in class_name:
                    return True
        except (ImportError, NameError):
            pass

        import os
        if os.environ.get("JPY_PARENT_PID") is not None:
            return True

        import sys
        if "ipykernel" in sys.modules:
            return True
    except Exception:
        pass

    return False


def setup_interactive_mode(backend: str = "auto", style: str = "default") -> None:
    """Setup matplotlib for interactive or notebook-aware plotting."""
    if backend == "auto":
        if _is_jupyter_environment():
            try:
                matplotlib.use("module://matplotlib_inline.backend_inline")
            except (ImportError, ValueError):
                try:
                    matplotlib.use("inline")
                except (ImportError, ValueError):
                    matplotlib.use("Agg")
        else:
            try:
                import tkinter  # noqa: F401

                matplotlib.use("TkAgg")
            except ImportError:
                try:
                    matplotlib.use("Qt5Agg")
                except ImportError:
                    matplotlib.use("Agg")
    else:
        matplotlib.use(backend)

    plt.style.use(style)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["interactive"] = True
    set_plot_style()


@contextmanager
def interactive_plotting(
    show_plots: bool = True,
    save_dir: str | None = None,
    save_prefix: str = "fig_",
    save_ext: str = "png",
    dpi: int = 300,
    block: bool = True,
):
    """Manage non-blocking plotting, optional figure saving, and backend restore."""
    original_backend = matplotlib.get_backend()
    original_interactive = plt.isinteractive()
    is_jupyter = _is_jupyter_environment()

    try:
        if show_plots:
            setup_interactive_mode()
            if not is_jupyter:
                plt.ion()
        yield
    finally:
        if save_dir:
            ensure_dir_exists(save_dir)
            for i in plt.get_fignums():
                fig = plt.figure(i)
                if fig._suptitle:
                    title = fig._suptitle.get_text()
                elif fig.axes and fig.axes[0].get_title():
                    title = fig.axes[0].get_title()
                else:
                    title = f"figure_{i}"

                filename = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
                filename = filename.replace(" ", "_").replace("#", "")
                filepath = Path(save_dir) / f"{save_prefix}{filename}.{save_ext}"

                try:
                    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
                    logger.info(f"{log_tag('ION', 'SAVE')} Saved figure to {filepath}")
                except Exception as e:
                    logger.error(f"{log_tag('ION', 'ERROR')} Failed to save figure {i}: {e}")

        if show_plots:
            if is_jupyter:
                for _ in plt.get_fignums():
                    plt.show(block=False)
            else:
                plt.ioff()
                plt.show(block=block)

        plt.close("all")
        matplotlib.use(original_backend)
        if not is_jupyter:
            plt.interactive(original_interactive)


__all__ = [
    "interactive_plotting",
    "setup_interactive_mode",
]
