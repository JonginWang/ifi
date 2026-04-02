from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.run_vest_monitoring import _normalize_query_items, _resolve_query_text
from ifi.utils.vest_fieldcode import load_vest_field_maps
from ifi.utils.vest_monitoring import run_vest_shot_monitoring


class _FakeVestDB:
    def load_shot(self, shot: int, fields: list[int]) -> dict[str, pd.DataFrame]:
        t25 = np.linspace(0.0, 0.4, 10000, endpoint=False)
        names = {
            1: "I_TF [[kA]]",
            12: "Press_raw [[V]]",
            25: "V_Loop_#10 [[V]]",
            101: "H_alpha [[a.u.]]",
            109: "Ip_raw [[V]]",
            138: "H_gamma_434p05_nm [[a.u.]]",
            139: "O_II_441p5_nm [[a.u.]]",
            140: "C_III_465p0_nm [[a.u.]]",
            141: "H_beta_486p13_nm [[a.u.]]",
            142: "C_II_514p0_nm [[a.u.]]",
            143: "O_650p0_nm [[a.u.]]",
            144: "H_alpha_656p28_nm [[a.u.]]",
            171: "Mirnov_Outboard [[T/s]]",
            192: "Mirnov_Inboard [[T/s]]",
            214: "OI_line_777_nm [[a.u.]]",
            216: "Limiter_LC [[V]]",
            217: "Limiter_UC [[V]]",
            218: "Limiter_MM [[V]]",
            257: "Diamagnetic_Flux_raw_Rogowski [[V]]",
            59: "PF1 [[kA]]",
            5: "PF5 [[kA]]",
            62: "PF6 [[kA]]",
            65: "PF10 [[kA]]",
            27: "ECH6kW_Forward_raw [[V]]",
            28: "ECH6kW_Reverse_raw [[V]]",
        }
        data = {}
        for field_id in fields:
            if field_id not in names:
                continue
            signal = np.sin(2 * np.pi * 50 * t25) * 0.1
            if field_id == 101:
                signal += np.exp(-((t25 - 0.31) ** 2) / (2 * 0.003**2))
            elif field_id == 109:
                signal += -2.0 * np.exp(-((t25 - 0.31) ** 2) / (2 * 0.01**2))
            elif field_id in {171, 192}:
                signal += np.sin(2 * np.pi * 12000 * t25)
            else:
                signal += 0.2 * np.cos(2 * np.pi * 20 * t25)
            data[names[field_id]] = signal + field_id * 1e-4
        return {"25k": pd.DataFrame(data, index=t25)}

    def disconnect(self) -> None:
        return None


def test_vest_monitoring_writes_local_h5_and_plots(tmp_path: Path):
    shot_num = 99990
    results_dir = tmp_path / "results"
    payload = run_vest_shot_monitoring(
        vest_db=_FakeVestDB(),
        shots=[shot_num],
        results_dir=results_dir,
        overwrite_local=True,
    )

    assert payload["written"][shot_num]["monitoring"] > 0
    assert payload["written"][shot_num]["mirnov_spectrogram"] > 0
    assert (results_dir / str(shot_num) / f"{shot_num}.h5").exists()
    assert (results_dir / str(shot_num) / "monitoring" / f"TF_current_{shot_num}.png").exists()


def test_vest_field_map_260101_contains_monitoring_codes():
    by_id, _ = load_vest_field_maps()
    for field_id in (12, 27, 28, 103, 104, 219, 220, 232, 233, 234, 235, 259, 260, 261, 262):
        assert field_id in by_id


def test_vest_monitoring_plot_each_saves_under_each_shot(tmp_path: Path, monkeypatch):
    shots = [99991, 99992]
    results_dir = tmp_path / "results"
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    payload = run_vest_shot_monitoring(
        vest_db=_FakeVestDB(),
        shots=shots,
        results_dir=results_dir,
        overwrite_local=True,
        plot_each=True,
    )

    assert payload["shots"] == shots
    for shot_num in shots:
        shot_monitor_dir = results_dir / str(shot_num) / "monitoring"
        assert shot_monitor_dir.exists()
        assert (shot_monitor_dir / f"TF_current_{shot_num}.png").exists()
        assert (shot_monitor_dir / f"MirnovSpectrogram_{shot_num}.png").exists()


def test_run_vest_monitoring_query_normalization():
    assert _resolve_query_text(["48710", "48720"], None) == "48710 48720"
    assert _resolve_query_text([], "48710:48720") == "48710:48720"
    assert _normalize_query_items("48710 48720") == ["48710", "48720"]
    assert _normalize_query_items("48710,48720") == ["48710", "48720"]
    assert _normalize_query_items("48710:48720") == ["48710:48720"]
