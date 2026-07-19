import json
from pathlib import Path
from types import SimpleNamespace

from g1b_contract import sha256
from scripts.queue_rald_anchor_g2r_g3r import Job, completed


def test_completed_g3r_binds_initial_and_parent_checkpoints(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "best.pt").write_bytes(b"parent")
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "best.pt").write_bytes(b"initial")
    run = tmp_path / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"result")
    (run / "best_validation_metrics.json").write_text("{}", encoding="utf-8")
    (run / "train_log.jsonl").write_text(
        json.dumps({"epoch": 20}) + "\n", encoding="utf-8"
    )
    (run / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "seed": 20260716,
                    "epochs": 20,
                    "doppler_head_mode": "distribution",
                    "cycle_variant": "full",
                    "physical_head_warmup_epochs": 0,
                    "initial_refiner_run": str(initial),
                },
                "provenance": {
                    "git_commit": "source",
                    "parent_g1_checkpoint": str(parent / "best.pt"),
                    "parent_g1_checkpoint_sha256": sha256(parent / "best.pt"),
                    "initial_refiner_checkpoint_sha256": sha256(
                        initial / "best.pt"
                    ),
                    "initial_refiner_source_commit": "source",
                },
            }
        ),
        encoding="utf-8",
    )
    job = Job(
        stage="g3r",
        seed=20260716,
        arm="full",
        parent=parent,
        run=run,
        log=tmp_path / "run.log",
        initial_refiner=initial,
    )
    args = SimpleNamespace(
        source_commit="source",
        g2r_epochs=30,
        g2r_warmup_epochs=5,
        g3r_epochs=20,
    )

    assert completed(job, args) is True
    (initial / "best.pt").write_bytes(b"changed")
    assert completed(job, args) is False
