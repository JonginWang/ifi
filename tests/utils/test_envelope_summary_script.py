from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.summarize_envelope_segments import _parse_shots, summarize_envelope_segments


def test_parse_shots_supports_space_comma_and_range():
    assert _parse_shots("47807 47808") == [47807, 47808]
    assert _parse_shots("47807,47808") == [47807, 47808]
    assert _parse_shots("47807:47809") == [47807, 47808, 47809]


def test_summarize_envelope_segments_nan_zero_and_max(tmp_path: Path):
    results_dir = tmp_path / "results"

    missing_shot = 10001
    empty_shot = 10002
    full_shot = 10003

    (results_dir / str(empty_shot) / "envelope").mkdir(parents=True, exist_ok=True)

    full_dir = results_dir / str(full_shot) / "envelope"
    full_dir.mkdir(parents=True, exist_ok=True)
    (full_dir / "a.json").write_text(
        json.dumps({"segments": [{"duration_us": 120.0}, {"duration_us": 80.0}]}),
        encoding="utf-8",
    )
    (full_dir / "b.json").write_text(
        json.dumps({"segments": [{"duration_us": 250.0}]}),
        encoding="utf-8",
    )

    summary = summarize_envelope_segments(
        [missing_shot, empty_shot, full_shot],
        results_dir=results_dir,
    )

    by_shot = {int(row.shot_num): float(row.max_duration_us) for row in summary.itertuples()}
    assert math.isnan(by_shot[missing_shot])
    assert by_shot[empty_shot] == 0.0
    assert by_shot[full_shot] == 250.0
