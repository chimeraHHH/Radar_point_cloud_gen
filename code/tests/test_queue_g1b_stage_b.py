import json
from pathlib import Path

from scripts.queue_g1b_stage_b import atomic_json, sha256, wait_for_screen


def test_stage_b_wait_requires_final_screen_decision(tmp_path: Path) -> None:
    path = tmp_path / "screen.json"
    report = {
        "stage_b_authorized": True,
        "selected_candidate": "rae_circular_harmonics",
    }
    atomic_json(path, report)

    assert wait_for_screen(path, poll_seconds=0) == report
    assert len(sha256(path)) == 64
    assert json.loads(path.read_text(encoding="utf-8")) == report
