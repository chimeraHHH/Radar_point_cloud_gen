import json
from pathlib import Path

from scripts.queue_rald_anchor_rh1 import atomic_json, select_parent, wait_for_json


def test_queue_json_helpers_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    expected = {"decision": {"g1_passed": False}}

    atomic_json(path, expected)

    assert wait_for_json(path, poll_seconds=0) == expected
    assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_select_parent_preserves_g1_and_uses_rae_only_as_named_recovery(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full"
    rae = tmp_path / "rae"

    assert select_parent({"g1_passed": True}, full, rae) == (
        "full_raed",
        full,
        "formal_g1_passed",
    )
    assert select_parent(
        {"g1_passed": False, "rae_max_beats_cfar": True}, full, rae
    ) == ("rae_max", rae, "late_fusion_recovery_after_g1_failure")
    assert select_parent(
        {"g1_passed": False, "rae_max_beats_cfar": False}, full, rae
    ) is None
