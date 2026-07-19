import json
from pathlib import Path

from scripts.queue_rald_anchor_rh1 import atomic_json, wait_for_json


def test_queue_json_helpers_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    expected = {"decision": {"g1_passed": False}}

    atomic_json(path, expected)

    assert wait_for_json(path, poll_seconds=0) == expected
    assert json.loads(path.read_text(encoding="utf-8")) == expected
