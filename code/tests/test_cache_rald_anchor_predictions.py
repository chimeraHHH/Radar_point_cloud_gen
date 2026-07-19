import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cache_rald_anchor_predictions import (
    PREDICTION_KEYS,
    prediction_metadata,
    sha256,
    validate_cache_inputs,
    validate_frame_record,
    validate_prediction,
    write_prediction,
)
from g1b_contract import FROZEN_G1B_SEEDS


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def build_contract(tmp_path: Path) -> dict:
    source = "g3r-source"
    cache_source = "cache-source"
    split = tmp_path / "split.json"
    normalization = tmp_path / "normalization.json"
    temporal = tmp_path / "temporal.json"
    write_json(split, {"gate_pass": True, "splits": {}})
    write_json(normalization, {"log_center": 1.0, "log_scale": 2.0})
    split_hash = sha256(split)
    normalization_hash = sha256(normalization)
    write_json(
        temporal,
        {
            "gate_pass": True,
            "source_commit": "temporal-source",
            "source_split_sha256": split_hash,
            "frames": [
                {"sequence": 1, "radar_index": 2, "partition": "train"},
                {"sequence": 1, "radar_index": 3, "partition": "validation"},
            ],
        },
    )

    parent = tmp_path / "parent"
    parent.mkdir()
    parent_checkpoint = parent / "best.pt"
    parent_checkpoint.write_bytes(b"parent-checkpoint")
    write_json(
        parent / "config.json",
        {
            "config": {
                "mode": "candidate",
                "base_channels": 8,
                "log_center": 1.0,
                "log_scale": 2.0,
            },
            "provenance": {
                "git_commit": "parent-source",
                "scene_split_sha256": split_hash,
                "normalization_sha256": normalization_hash,
            },
        },
    )

    runs = {}
    hashes = {}
    for seed in FROZEN_G1B_SEEDS:
        run = tmp_path / f"seed{seed}"
        run.mkdir()
        write_json(
            run / "config.json",
            {
                "config": {
                    "seed": seed,
                    "cycle_variant": "full",
                    "doppler_head_mode": "distribution",
                    "point_count": 4,
                    "latent_count": 2,
                    "model_dim": 8,
                    "depth": 1,
                    "heads": 1,
                    "head_dim": 8,
                    "radar_base_channels": 4,
                    "radar_spectral_channels": 2,
                },
                "provenance": {
                    "git_commit": source,
                    "manifest_sha256": "training-manifest-hash",
                    "scene_split_sha256": split_hash,
                    "normalization_sha256": normalization_hash,
                    "parent_g1_checkpoint": str(parent_checkpoint),
                    "parent_g1_checkpoint_sha256": sha256(parent_checkpoint),
                    "parent_g1_git_commit": "parent-source",
                },
            },
        )
        (run / "best.pt").write_bytes(f"checkpoint-{seed}".encode())
        runs[str(seed)] = str(run)
        hashes[str(seed)] = {
            "config_sha256": sha256(run / "config.json"),
            "best_checkpoint_sha256": sha256(run / "best.pt"),
        }
    comparison = tmp_path / "g3r_comparison.json"
    write_json(
        comparison,
        {
            "decision": {"g3r_passed": True},
            "runs": {"full": runs},
            "run_hashes": {"full": hashes},
        },
    )
    summary = tmp_path / "g3r_summary.json"
    write_json(
        summary,
        {
            "status": "g3r_passed",
            "source_commit": source,
            "seeds": list(FROZEN_G1B_SEEDS),
            "selected_arm": "full",
            "selected_runs": runs,
            "selected_run_hashes": hashes,
            "g3r_comparison": str(comparison),
            "g3r_comparison_sha256": sha256(comparison),
        },
    )
    return {
        "source": source,
        "cache_source": cache_source,
        "split": split,
        "normalization": normalization,
        "temporal": temporal,
        "summary": summary,
        "runs": runs,
    }


def load_contract(paths: dict) -> dict:
    return validate_cache_inputs(
        paths["temporal"],
        paths["split"],
        paths["normalization"],
        paths["summary"],
        g3r_source_commit=paths["source"],
        cache_source_commit=paths["cache_source"],
        seed=FROZEN_G1B_SEEDS[0],
    )


def test_contract_binds_temporal_and_selected_g3r_hashes(tmp_path: Path) -> None:
    paths = build_contract(tmp_path)
    contract = load_contract(paths)
    configuration = contract["configuration"]

    assert configuration["selected_arm"] == "full"
    assert configuration["doppler_head_mode"] == "distribution"
    assert configuration["g3r_seed"] == FROZEN_G1B_SEEDS[0]
    assert configuration["temporal_manifest_sha256"] == sha256(paths["temporal"])
    assert configuration["scene_split_sha256"] == sha256(paths["split"])
    assert configuration["normalization_sha256"] == sha256(paths["normalization"])
    assert configuration["g3r_config_sha256"] == sha256(
        Path(paths["runs"][str(FROZEN_G1B_SEEDS[0])]) / "config.json"
    )
    assert configuration["g3r_checkpoint_sha256"] == sha256(
        Path(paths["runs"][str(FROZEN_G1B_SEEDS[0])]) / "best.pt"
    )
    assert configuration["g3r_source_commit"] == paths["source"]
    assert configuration["required_frames"] == 2

    selected_checkpoint = (
        Path(paths["runs"][str(FROZEN_G1B_SEEDS[0])]) / "best.pt"
    )
    selected_checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_contract(paths)


def test_prediction_schema_metadata_and_shapes_are_strict(tmp_path: Path) -> None:
    contract = load_contract(build_contract(tmp_path))
    metadata = prediction_metadata(contract["configuration"])
    point_count = int(contract["configuration"]["point_count"])
    probability = torch.full((point_count, 64), 1.0 / 64.0)
    output = {
        "xyz_m": torch.arange(point_count * 3, dtype=torch.float32).reshape(
            point_count, 3
        ),
        "coordinates_rae": torch.zeros(point_count, 3),
        "doppler_probability": probability,
        "confidence": torch.linspace(0.1, 0.9, point_count),
    }
    prediction = tmp_path / "prediction.npz"
    write_prediction(
        prediction,
        output,
        metadata,
        sequence=1,
        radar_index=2,
    )
    validation = validate_prediction(
        prediction,
        metadata,
        point_count,
        sequence=1,
        radar_index=2,
    )
    assert validation["point_count"] == point_count
    with np.load(prediction, allow_pickle=False) as cache:
        assert set(cache.files) == PREDICTION_KEYS
        assert "static_center_mps" not in cache.files
        assert not any("pce" in key.lower() for key in cache.files)

    wrong_metadata = {**metadata, "normalization_sha256": "wrong"}
    with pytest.raises(ValueError, match="normalization_sha256"):
        validate_prediction(prediction, wrong_metadata, point_count)
    with pytest.raises(ValueError, match="expected"):
        validate_prediction(prediction, metadata, point_count + 1)


def test_cached_frame_hash_and_probability_validation(tmp_path: Path) -> None:
    contract = load_contract(build_contract(tmp_path))
    metadata = prediction_metadata(contract["configuration"])
    point_count = int(contract["configuration"]["point_count"])
    prediction = tmp_path / "prediction.npz"
    output = {
        "xyz_m": torch.zeros(point_count, 3),
        "coordinates_rae": torch.zeros(point_count, 3),
        "doppler_probability": torch.full((point_count, 64), 1.0 / 64.0),
        "confidence": torch.full((point_count,), 0.5),
    }
    write_prediction(
        prediction,
        output,
        metadata,
        sequence=1,
        radar_index=2,
    )
    frame = {
        "sequence": 1,
        "radar_index": 2,
        "prediction": str(prediction),
        "prediction_sha256": sha256(prediction),
    }
    assert validate_frame_record(frame, metadata, point_count)["radar_index"] == 2

    prediction.write_bytes(prediction.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        validate_frame_record(frame, metadata, point_count)

    bad_probability = torch.zeros(point_count, 64)
    output["doppler_probability"] = bad_probability
    write_prediction(
        prediction,
        output,
        metadata,
        sequence=1,
        radar_index=2,
    )
    with pytest.raises(ValueError, match="do not sum to one"):
        validate_prediction(prediction, metadata, point_count)
