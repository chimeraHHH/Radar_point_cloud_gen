import json
import sys
from pathlib import Path

from scripts.g1b_contract import sha256
from scripts.queue_rald_query_field import (
    Job,
    completed_run,
    resolve_python,
    validate_preflight,
)
from scripts.train_rald_query_field import PROTOCOL
from scripts.train_rald_query_field import cross_scene_condition_indices


SOURCE_COMMIT = "1" * 40


def test_resolve_python_uses_current_interpreter_without_path_lookup(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PYTHON", raising=False)

    assert resolve_python() == Path(sys.executable).resolve()


def test_resolve_python_rejects_missing_configured_executable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYTHON", "definitely-not-a-python-executable")

    try:
        resolve_python()
    except FileNotFoundError as error:
        assert "Configured Python executable not found" in str(error)
    else:
        raise AssertionError("Missing configured interpreter was accepted")


def preflight_gradient_steps() -> list[dict]:
    return [
        {"update": 1, "gradients": {"output_heads": 1.0}},
        {
            "update": 2,
            "gradients": {
                "mixed_latent": 1.0,
                "query_decoder": 1.0,
                "full_raed_radar_encoder": 1.0,
                "cube_input_channel_norms": [1.0] * 64,
                "local_spectrum_input_column_norms": [1.0] * 64,
                "absolute_energy_input_column_norms": [1.0],
                "normalized_range_input_column_norms": [1.0],
                "radar_projection_input_column_norms": [1.0] * 64,
                "condition_block_gradient_norms": [1.0] * 24,
            },
        },
    ]


def write_preflight_run(tmp_path: Path) -> Job:
    run_path = tmp_path / "run"
    run_path.mkdir()
    checkpoint = run_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "config": {
            "protocol": PROTOCOL,
            "seed": 20260716,
            "epochs": 1,
            "depth": 24,
            "model_dim": 512,
            "latent_count": 512,
            "occupancy_query_count": 10_000,
            "positive_query_ratio": 0.0625,
            "positive_occupancy_weight": 0.1,
            "negative_occupancy_weight": 1.0,
        },
        "provenance": {"git_commit": SOURCE_COMMIT},
    }
    frame = {
        "generated": {"prediction_count": 10_000},
        "radar_token_count": 336,
        "coarse_query_count": 32_000,
        "selected_coarse_count": 2_500,
        "occupancy_query_count": 10_000,
        "positive_occupancy_query_count": 625,
        "empty_occupancy_query_count": 9_375,
        "positive_fractional_coordinate_rate": 0.999,
        "empty_fractional_coordinate_rate": 0.999,
        "sequence": 1,
        "shuffled_condition_sequence": 2,
        "normalized_log_energy_abs_max": 2.0,
        "proposal_cache_used": True,
    }
    metrics = {
        "completed": True,
        "test_accessed": False,
        "best_checkpoint_sha256": sha256(checkpoint),
        "gradient_steps": preflight_gradient_steps(),
        "validation": {
            "frames": [
                frame,
                {
                    **frame,
                    "sequence": 2,
                    "shuffled_condition_sequence": 1,
                },
            ]
        },
    }
    (run_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_path / "best_validation_metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    return Job(20260716, run_path, tmp_path / "run.log")


def test_preflight_validates_full_rald_structure_and_counts(tmp_path: Path) -> None:
    job = write_preflight_run(tmp_path)
    report_path = tmp_path / "preflight.json"

    assert completed_run(job, SOURCE_COMMIT, 1) is True
    report = validate_preflight(job, SOURCE_COMMIT, report_path)

    assert report["passed"] is True
    assert all(report["checks"].values())


def test_condition_shuffle_is_deterministic_and_cross_scene() -> None:
    records = [
        {"sequence": 1},
        {"sequence": 1},
        {"sequence": 2},
        {"sequence": 2},
        {"sequence": 3},
        {"sequence": 3},
    ]

    shuffled = cross_scene_condition_indices(records)

    assert sorted(shuffled) == list(range(len(records)))
    assert all(
        records[index]["sequence"] != records[other]["sequence"]
        for index, other in enumerate(shuffled)
    )


def test_preflight_rejects_missing_rald_condition_block_gradient(
    tmp_path: Path,
) -> None:
    job = write_preflight_run(tmp_path)
    metrics_path = job.run_path / "best_validation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["gradient_steps"][1]["gradients"][
        "condition_block_gradient_norms"
    ].pop()
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    try:
        validate_preflight(job, SOURCE_COMMIT, tmp_path / "failed.json")
    except ValueError as error:
        assert "preflight failed" in str(error)
    else:
        raise AssertionError("Incomplete per-layer condition gradient was accepted")
