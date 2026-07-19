import json
from pathlib import Path

import pytest

from scripts.queue_rald_anchor_rh2 import Job, completed, g1b_parent_runs


def test_completed_rh2_requires_matching_source_and_epoch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "best_validation_metrics.json").write_text("{}", encoding="utf-8")
    (run / "config.json").write_text(
        json.dumps({"provenance": {"git_commit": "abc"}}), encoding="utf-8"
    )
    (run / "train_log.jsonl").write_text(
        json.dumps({"epoch": 20}) + "\n", encoding="utf-8"
    )
    job = Job(seed=1, parent=tmp_path / "parent", run=run, log=tmp_path / "log")

    assert completed(job, epochs=20, source_commit="abc") is True
    assert completed(job, epochs=19, source_commit="abc") is False
    assert completed(job, epochs=20, source_commit="def") is False


def test_g1b_parent_runs_are_taken_from_authoritative_summary(tmp_path: Path) -> None:
    source = "3fa7ae88f2445e5f610bd421f4b3044975267b89"
    seeds = [20260716, 20260717, 20260718]
    runs = [
        tmp_path / f"g1b_stage_b_rae_moments_seed{seed}_{source[:8]}"
        for seed in seeds
    ]
    summary = {
        "status": "g1b_passed",
        "candidate_mode": "rae_moments",
        "training_source_commit": source,
        "seeds": seeds,
        "candidate_runs": [str(path) for path in runs],
    }

    assert g1b_parent_runs(summary, seeds, tmp_path, source) == dict(
        zip(seeds, runs, strict=True)
    )
    with pytest.raises(ValueError, match="authorize all"):
        g1b_parent_runs(
            {**summary, "candidate_runs": [str(runs[0])]}, seeds, tmp_path, source
        )
