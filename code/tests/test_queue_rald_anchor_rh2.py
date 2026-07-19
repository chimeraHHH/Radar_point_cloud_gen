import json
from pathlib import Path

import pytest

from g1b_contract import sha256
from scripts.queue_rald_anchor_rh2 import (
    Job,
    completed,
    g1b_parent_runs,
    validate_rh1_summary_contract,
)


def test_completed_rh2_requires_matching_source_and_epoch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    parent = tmp_path / "parent"
    parent.mkdir()
    checkpoint = parent / "best.pt"
    checkpoint.write_bytes(b"parent")
    (run / "best_validation_metrics.json").write_text("{}", encoding="utf-8")
    (run / "config.json").write_text(
        json.dumps(
            {
                "config": {"parent_route": "independent_g1b_parent"},
                "provenance": {
                    "git_commit": "abc",
                    "parent_g1_checkpoint": str(checkpoint),
                    "parent_g1_checkpoint_sha256": sha256(checkpoint),
                    "parent_g1_git_commit": "parent-source",
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "train_log.jsonl").write_text(
        json.dumps({"epoch": 20}) + "\n", encoding="utf-8"
    )
    job = Job(seed=1, parent=parent, run=run, log=tmp_path / "log")

    assert completed(
        job, 20, "abc", "independent_g1b_parent", "parent-source"
    ) is True
    assert (
        completed(job, 19, "abc", "independent_g1b_parent", "parent-source")
        is False
    )
    assert (
        completed(job, 20, "def", "independent_g1b_parent", "parent-source")
        is False
    )
    checkpoint.write_bytes(b"changed")
    assert completed(
        job, 20, "abc", "independent_g1b_parent", "parent-source"
    ) is False


def test_g1b_parent_runs_are_taken_from_authoritative_summary(tmp_path: Path) -> None:
    source = "3fa7ae88f2445e5f610bd421f4b3044975267b89"
    decision_source = "a8f3432" * 5
    seeds = [20260716, 20260717, 20260718]
    runs = [
        tmp_path / f"g1b_stage_b_rae_moments_seed{seed}_{source[:8]}"
        for seed in seeds
    ]
    summary = {
        "status": "g1b_passed",
        "candidate_mode": "rae_moments",
        "training_source_commit": source,
        "decision_source_commit": decision_source,
        "seeds": seeds,
        "epochs": 50,
        "candidate_runs": [str(path) for path in runs],
    }

    screen = tmp_path / "screen.json"
    launch = tmp_path / "launch.json"
    comparison = tmp_path / "comparison.json"
    for path in (screen, launch, comparison):
        path.write_text(path.name, encoding="utf-8")
    summary.update(
        {
            "screen_report": str(screen),
            "screen_report_sha256": sha256(screen),
            "launch_decision": str(launch),
            "launch_decision_sha256": sha256(launch),
            "comparison": str(comparison),
            "comparison_sha256": sha256(comparison),
            "baseline_runs": [
                str(tmp_path / f"g1b_stage_b_rae_max_seed{seed}_{source[:8]}")
                for seed in seeds
            ],
        }
    )
    summary["runs"] = summary["baseline_runs"] + summary["candidate_runs"]

    assert g1b_parent_runs(
        summary, seeds, tmp_path, source, decision_source
    ) == dict(
        zip(seeds, runs, strict=True)
    )
    with pytest.raises(ValueError, match="exact candidate"):
        g1b_parent_runs(
            {**summary, "candidate_runs": [str(runs[0])]},
            seeds,
            tmp_path,
            source,
            decision_source,
        )
    with pytest.raises(ValueError, match="frozen three"):
        g1b_parent_runs(
            summary,
            [20260716],
            tmp_path,
            source,
            decision_source,
        )


def test_rh1_summary_contract_rejects_old_source() -> None:
    expected = {"source_commit": "new", "parent_checkpoint_sha256": "hash"}
    validate_rh1_summary_contract(expected, expected)
    with pytest.raises(ValueError, match="differs"):
        validate_rh1_summary_contract(
            {**expected, "source_commit": "old"}, expected
        )
