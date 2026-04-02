#!/usr/bin/env python3
"""
VEST monitoring runner
=====================

High-level orchestration for local VEST monitoring post-processing.
"""

from __future__ import annotations

import h5py
import pandas as pd
from pathlib import Path

from .. import __version__
from .io_h5 import H5_GROUP_RAWDATA
from .vest_monitoring_data import run_vest_monitoring_data_pipeline
from .vest_monitoring_plot import save_vest_monitoring_plots


def _ensure_results_h5(h5_path: Path, shot_num: int) -> None:
    if h5_path.exists():
        return
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as hf:
        hf.attrs["shot_number"] = int(shot_num)
        hf.attrs["created_at"] = pd.Timestamp.now().isoformat()
        hf.attrs["ifi_version"] = __version__
        raw_group = hf.create_group(H5_GROUP_RAWDATA)
        raw_group.attrs["empty"] = True


def run_vest_shot_monitoring(
    vest_db,
    shots: list[int],
    *,
    results_dir: str | Path = "ifi/results",
    xcoil: list[int] | None = None,
    gas: str = "H2",
    xrange_s: tuple[float, float] = (0.28, 0.35),
    overwrite_local: bool = False,
    save_plots: bool = True,
    plot_each: bool = False,
) -> dict[str, object]:
    xcoil = xcoil or [1, 5, 6, 10]
    results_root = Path(results_dir)
    monitoring_by_shot = {}
    spectrograms_by_shot = {}
    write_summary = {}

    for shot_num in shots:
        shot_dir = results_root / str(shot_num)
        shot_dir.mkdir(parents=True, exist_ok=True)
        h5_path = shot_dir / f"{shot_num}.h5"
        _ensure_results_h5(h5_path, shot_num)

        payload = run_vest_monitoring_data_pipeline(
            vest_db=vest_db,
            shot_num=shot_num,
            h5_path=h5_path,
            xcoil=xcoil,
            gas=gas,
            overwrite_local=overwrite_local,
        )
        monitoring_by_shot[shot_num] = payload["monitoring_frames"]
        spectrograms_by_shot[shot_num] = payload["mirnov_spectrograms"]
        write_summary[shot_num] = payload["written"]

    if save_plots and shots:
        if plot_each:
            for shot_num in shots:
                save_vest_monitoring_plots(
                    shots=[shot_num],
                    monitoring_by_shot={shot_num: monitoring_by_shot.get(shot_num, {})},
                    mirnov_specs_by_shot={shot_num: spectrograms_by_shot.get(shot_num, {})},
                    save_dir=results_root / str(shot_num) / "monitoring",
                    xrange_s=xrange_s,
                    overwrite=overwrite_local,
                    show_plots=True,
                )
        else:
            save_vest_monitoring_plots(
                shots=shots,
                monitoring_by_shot=monitoring_by_shot,
                mirnov_specs_by_shot=spectrograms_by_shot,
                save_dir=results_root / str(shots[-1]) / "monitoring",
                xrange_s=xrange_s,
                overwrite=overwrite_local,
                show_plots=False,
            )

    return {
        "shots": shots,
        "monitoring_by_shot": monitoring_by_shot,
        "spectrograms_by_shot": spectrograms_by_shot,
        "written": write_summary,
    }


__all__ = [
    "run_vest_shot_monitoring",
]
