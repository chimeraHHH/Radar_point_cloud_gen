import json

import pytest

from cube_dense.training_io import checkpoint_due, truncate_resume_artifacts


def test_checkpoint_cadence_includes_evaluation_interval_and_final_epoch() -> None:
    assert checkpoint_due(5, 12, 5, evaluated=False) is True
    assert checkpoint_due(7, 12, 5, evaluated=True) is True
    assert checkpoint_due(12, 12, 5, evaluated=False) is True
    assert checkpoint_due(3, 12, 5, evaluated=False) is False
    with pytest.raises(ValueError, match="positive"):
        checkpoint_due(1, 2, 0, evaluated=False)


def test_resume_truncates_log_and_metrics_after_checkpoint(tmp_path) -> None:
    log = tmp_path / "train_log.jsonl"
    log.write_text(
        "".join(json.dumps({"epoch": epoch}) + "\n" for epoch in range(1, 8)),
        encoding="utf-8",
    )
    for epoch in (1, 5, 7):
        (tmp_path / f"metrics_epoch_{epoch:04d}.json").write_text(
            "{}", encoding="utf-8"
        )

    retained = truncate_resume_artifacts(tmp_path, checkpoint_epoch=5)

    assert [record["epoch"] for record in retained] == [1, 2, 3, 4, 5]
    assert not (tmp_path / "metrics_epoch_0007.json").exists()
    assert (tmp_path / "metrics_epoch_0005.json").exists()
