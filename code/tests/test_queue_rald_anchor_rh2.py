import json
from pathlib import Path

from scripts.queue_rald_anchor_rh2 import Job, completed


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
