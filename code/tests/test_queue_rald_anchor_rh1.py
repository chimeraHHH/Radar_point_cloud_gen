import json
from pathlib import Path

import pytest

from g1b_contract import sha256
from scripts.queue_rald_anchor_rh1 import (
    atomic_json,
    select_g1b_parent,
    select_parent,
    wait_for_json,
)


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
    with pytest.raises(ValueError, match="boolean g1_passed"):
        select_parent({}, full, rae)
    with pytest.raises(ValueError, match="boolean rae_max_beats_cfar"):
        select_parent({"g1_passed": False}, full, rae)


def test_select_g1b_parent_requires_exact_authorized_run(tmp_path: Path) -> None:
    source = "3fa7ae88f2445e5f610bd421f4b3044975267b89"
    decision_source = "a8f3432" * 5
    run_root = tmp_path / "runs"
    expected = (
        run_root
        / f"g1b_stage_b_rae_circular_harmonics_seed20260716_{source[:8]}"
    )
    screen = tmp_path / "screen.json"
    launch = tmp_path / "launch.json"
    comparison = tmp_path / "comparison.json"
    for path in (screen, launch, comparison):
        path.write_text(path.name, encoding="utf-8")
    seeds = [20260716, 20260717, 20260718]
    candidate_runs = [
        run_root / f"g1b_stage_b_rae_circular_harmonics_seed{seed}_{source[:8]}"
        for seed in seeds
    ]
    baseline_runs = [
        run_root / f"g1b_stage_b_rae_max_seed{seed}_{source[:8]}" for seed in seeds
    ]
    summary = {
        "status": "g1b_passed",
        "candidate_mode": "rae_circular_harmonics",
        "training_source_commit": source,
        "decision_source_commit": decision_source,
        "seeds": seeds,
        "epochs": 50,
        "screen_report": str(screen),
        "screen_report_sha256": sha256(screen),
        "launch_decision": str(launch),
        "launch_decision_sha256": sha256(launch),
        "comparison": str(comparison),
        "comparison_sha256": sha256(comparison),
        "candidate_runs": [str(path) for path in candidate_runs],
        "baseline_runs": [str(path) for path in baseline_runs],
        "runs": [str(path) for path in baseline_runs + candidate_runs],
    }

    assert select_g1b_parent(
        summary, 20260716, run_root, source, decision_source
    ) == (
        "rae_circular_harmonics",
        expected,
    )
    with pytest.raises(ValueError, match="exact candidate"):
        select_g1b_parent(
            {**summary, "candidate_runs": []},
            20260716,
            run_root,
            source,
            decision_source,
        )
    with pytest.raises(ValueError, match="decision source commit"):
        select_g1b_parent(
            summary,
            20260716,
            run_root,
            source,
            "wrong-decision-source",
        )
