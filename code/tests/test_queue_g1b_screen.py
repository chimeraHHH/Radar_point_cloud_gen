import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from queue_g1b_screen import wait_for_failed_g1  # noqa: E402


def test_g1b_gate_accepts_only_a_final_failed_original_g1(tmp_path) -> None:
    report = tmp_path / "g1.json"
    report.write_text(
        json.dumps({"decision": {"g1_passed": False}}), encoding="utf-8"
    )

    loaded = wait_for_failed_g1(report, poll_seconds=0)

    assert loaded["decision"]["g1_passed"] is False


def test_g1b_gate_exits_without_training_when_original_g1_passes(tmp_path) -> None:
    report = tmp_path / "g1.json"
    report.write_text(
        json.dumps({"decision": {"g1_passed": True}}), encoding="utf-8"
    )

    with pytest.raises(SystemExit) as error:
        wait_for_failed_g1(report, poll_seconds=0)

    assert error.value.code == 0
