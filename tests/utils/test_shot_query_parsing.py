from __future__ import annotations

import sys

from ifi.analysis.ifi_analyzer import normalize_cli_argv
from ifi.analysis.main_analysis import build_argument_parser
from ifi.utils.vest_postprocess import FlatShotList, normalize_shot_query_items
import scripts.plot_saved_results as plot_saved_results_script
import scripts.run_analysis_smart as run_analysis_smart_script
import scripts.run_vest_monitoring as run_vest_monitoring_script


def test_flat_shot_list_supports_all_requested_query_forms():
    assert FlatShotList(["12345"]).nums == [12345]
    assert FlatShotList(["12345, 23456"]).nums == [12345, 23456]
    assert FlatShotList(["12345,23456"]).nums == [12345, 23456]
    assert FlatShotList(["12345 23456"]).nums == [12345, 23456]
    assert FlatShotList(["12345:12347"]).nums == [12345, 12346, 12347]
    assert FlatShotList(["12345:12347:2"]).nums == [12345, 12347]


def test_normalize_shot_query_items_supports_all_requested_query_forms():
    assert normalize_shot_query_items("12345") == ["12345"]
    assert normalize_shot_query_items("12345, 23456") == ["12345", "23456"]
    assert normalize_shot_query_items("12345,23456") == ["12345", "23456"]
    assert normalize_shot_query_items("12345 23456") == ["12345", "23456"]
    assert normalize_shot_query_items("12345:12347") == ["12345:12347"]
    assert normalize_shot_query_items("12345:12347:2") == ["12345:12347:2"]


def test_ifi_analyzer_query_normalization_and_main_parser():
    parser = build_argument_parser()

    for raw_query, expected in [
        ("12345", [12345]),
        ("12345, 23456", [12345, 23456]),
        ("12345,23456", [12345, 23456]),
        ("12345 23456", [12345, 23456]),
        ("12345:12347", [12345, 12346, 12347]),
        ("12345:12347:2", [12345, 12347]),
    ]:
        normalized = normalize_cli_argv(["--query", raw_query, "--density"])
        args = parser.parse_args(normalized)
        assert FlatShotList(args.query).nums == expected


def test_plot_saved_results_accepts_multi_shot_query_forms(monkeypatch):
    captured_shots: list[int] = []

    monkeypatch.setattr(plot_saved_results_script, "load_config_defaults", lambda cfg: ({}, None))
    monkeypatch.setattr(plot_saved_results_script, "set_plot_style", lambda: None)
    monkeypatch.setattr(
        plot_saved_results_script,
        "load_results_from_hdf5",
        lambda shot_num, base_dir: {"rawdata": {}, "density": {}, "vestdata": {}, "shot_num": shot_num},
    )
    monkeypatch.setattr(plot_saved_results_script, "print_available_series", lambda results: None)

    assert (
        plot_saved_results_script.main(["--query", "12345, 23456", "--list"]) == 0
    )

    def _capture(shot_num, base_dir):
        captured_shots.append(int(shot_num))
        return {"rawdata": {}, "density": {}, "vestdata": {}, "shot_num": shot_num}

    monkeypatch.setattr(plot_saved_results_script, "load_results_from_hdf5", _capture)
    assert plot_saved_results_script.main(["12345:12347:2", "--list"]) == 0
    assert captured_shots == [12345, 12347]


def test_run_vest_monitoring_accepts_multi_shot_query_forms(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeVestDB:
        def __init__(self, config_path):
            self.config_path = config_path

        def disconnect(self):
            return None

    def _capture(**kwargs):
        captured.update(kwargs)
        return {"written": {}}

    monkeypatch.setattr(run_vest_monitoring_script, "VestDB", _FakeVestDB)
    monkeypatch.setattr(run_vest_monitoring_script, "run_vest_shot_monitoring", _capture)

    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "run_vest_monitoring.py",
            "--query",
            "12345 23456",
            "--no_save_plots",
        ]
        assert run_vest_monitoring_script.main() == 0
        assert captured["shots"] == [12345, 23456]

        sys.argv = [
            "run_vest_monitoring.py",
            "--query",
            "12345:12347:2",
            "--no_save_plots",
        ]
        assert run_vest_monitoring_script.main() == 0
        assert captured["shots"] == [12345, 12347]
    finally:
        sys.argv = argv_backup


def test_run_analysis_smart_parse_query_supports_requested_forms():
    assert run_analysis_smart_script.parse_shot_query("12345") == [12345]
    assert run_analysis_smart_script.parse_shot_query("12345, 23456") == [12345, 23456]
    assert run_analysis_smart_script.parse_shot_query("12345,23456") == [12345, 23456]
    assert run_analysis_smart_script.parse_shot_query("12345 23456") == [12345, 23456]
    assert run_analysis_smart_script.parse_shot_query("12345:12347") == [12345, 12346, 12347]
    assert run_analysis_smart_script.parse_shot_query("12345:12347:2") == [12345, 12347]
