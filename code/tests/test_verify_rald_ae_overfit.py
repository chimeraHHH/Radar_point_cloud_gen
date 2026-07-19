import json

from scripts.verify_rald_ae_overfit import verify


def write_run(tmp_path, *, chamfer: float, final_loss: float) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "overfit_one_frame": True,
                    "train_limit": 1,
                    "validation_limit": 1,
                },
                "provenance": {
                    "external_pretraining": False,
                    "git_commit": "a" * 40,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "train_log.jsonl").write_text(
        json.dumps({"epoch": 1, "train_loss_mean": 1.0})
        + "\n"
        + json.dumps({"epoch": 100, "train_loss_mean": final_loss})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "best_validation_metrics.json").write_text(
        json.dumps(
            {
                "validation": {
                    "generated": {
                        "chamfer_m": {"median": chamfer},
                        "outlier_fraction_2m": {"mean": 0.3},
                        "fscore_1p0m": {"mean": 0.4},
                    },
                    "frames": [{"mean_output_confidence": 0.2}],
                }
            }
        ),
        encoding="utf-8",
    )


def test_overfit_gate_accepts_geometry_and_learning(tmp_path) -> None:
    write_run(tmp_path, chamfer=2.0, final_loss=0.5)

    report = verify(tmp_path, required_epoch=100)

    assert report["passed"] is True


def test_overfit_gate_rejects_low_loss_without_geometry(tmp_path) -> None:
    write_run(tmp_path, chamfer=8.0, final_loss=0.1)

    report = verify(tmp_path, required_epoch=100)

    assert report["checks"]["loss_reduction"] is True
    assert report["checks"]["chamfer"] is False
    assert report["passed"] is False
