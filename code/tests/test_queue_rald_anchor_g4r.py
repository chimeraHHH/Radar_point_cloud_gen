from pathlib import Path
from types import SimpleNamespace
import json
import hashlib

import pytest

from scripts.queue_rald_anchor_g4r import (
    train_command,
    validate_formal_temporal_manifest,
)
from scripts.queue_g4_temporal import training_complete


def test_g4r_train_command_uses_rald_native_trainer() -> None:
    args = SimpleNamespace(
        repo_root=Path("/repo"),
        data_root=Path("/data"),
        cache_root=Path("/cache"),
        manifest=Path("/manifest.json"),
        scene_split=Path("/split.json"),
        normalization=Path("/normalization.json"),
        dense_cache_report=Path("/dense.json"),
        g3r_summary=Path("/g3r.json"),
        g3r_source_commit="g3r-source",
        source_commit="source",
    )

    command = train_command(
        Path("/python"),
        args,
        Path("/parent"),
        Path("/parent-cache"),
        Path("/output"),
        "latent",
        20260716,
        20,
    )

    assert command[2] == "/repo/code/scripts/train_rald_anchor_temporal.py"
    assert command[command.index("--fusion-mode") + 1] == "latent"
    assert command[command.index("--g3r-source-commit") + 1] == "g3r-source"
    assert command[command.index("--temporal-warmup-epochs") + 1] == "5"
    assert "static" not in " ".join(command).lower()


def formal_manifest() -> dict:
    windows = [
        {
            "window_id": f"seq{sequence:02d}_w00",
            "sequence": sequence,
            "frame_count": 48,
        }
        for sequence in range(1, 46)
    ]
    frames = [
        {
            "window_id": window["window_id"],
            "sequence": window["sequence"],
            "frame_in_window": position,
            "current_radar_from_previous_radar": list(range(16)),
        }
        for window in windows
        for position in range(48)
    ]
    return {
        "source_commit": "a" * 40,
        "gate_pass": True,
        "selection": {"window_length": 48, "windows_per_sequence": 1},
        "summary": {"frame_count": 2160, "window_count": 45, "sequence_count": 45},
        "checks": {"radar_frame_ego_transforms_present": True},
        "windows": windows,
        "frames": frames,
    }


def test_formal_manifest_freezes_complete_48_step_windows() -> None:
    manifest = formal_manifest()
    assert len(validate_formal_temporal_manifest(manifest, "a" * 40)) == 45


def test_formal_manifest_rejects_split_short_windows() -> None:
    manifest = formal_manifest()
    manifest["selection"]["window_length"] = 24
    with pytest.raises(ValueError, match="48-frame"):
        validate_formal_temporal_manifest(manifest, "a" * 40)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_completion_binds_config_and_checkpoint_hashes(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config = {
        "config": {"epochs": 20, "fusion_mode": "latent", "seed": 20260716},
        "provenance": {"git_commit": "a" * 40},
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "best.pt").write_bytes(b"best")
    (run / "last.pt").write_bytes(b"last")
    (run / "train_log.jsonl").write_text(
        json.dumps({"epoch": 20}) + "\n", encoding="utf-8"
    )
    metrics = {
        "completed": True,
        "config_sha256": digest(run / "config.json"),
        "best_checkpoint_sha256": digest(run / "best.pt"),
        "last_checkpoint_sha256": digest(run / "last.pt"),
    }
    (run / "best_validation_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )

    assert training_complete(
        run,
        20,
        "a" * 40,
        {"fusion_mode": "latent", "seed": 20260716},
    )
    assert not training_complete(
        run,
        20,
        "a" * 40,
        {"fusion_mode": "query", "seed": 20260716},
    )
    (run / "best.pt").write_bytes(b"changed")
    assert not training_complete(run, 20, "a" * 40)
