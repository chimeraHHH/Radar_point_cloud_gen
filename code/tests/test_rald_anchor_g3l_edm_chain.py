import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.rald_matched import FullRAEDRadarTokenEncoder, edm_loss
from scripts.cache_rald_anchor_g3l_latents import (
    G3L1_GATE_PROTOCOL,
    LATENT_SHAPE,
    PROTOCOL as CACHE_PROTOCOL,
    latent_metadata,
    sha256,
    validate_g3l1_run,
    validate_latent_cache_manifest,
    validate_latent_record,
    write_latent_file,
)
from scripts.train_rald_anchor_g3l_edm import (
    FORMAL_SEEDS,
    OFFICIAL_DENOISER_DEPTH,
    OFFICIAL_EDM_STEPS,
    OFFICIAL_EPOCHS,
    OFFICIAL_P_MEAN,
    OFFICIAL_P_STD,
    OFFICIAL_RHO,
    OFFICIAL_SIGMA_DATA,
    OFFICIAL_SIGMA_MAX,
    OFFICIAL_SIGMA_MIN,
    TrainConfig,
    build_edm,
    gradient_audit,
    require_cache_source,
    validate_two_step_gradient_audit,
)


def write_json(path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def metadata_configuration() -> dict:
    return {
        "cache_source_commit": "cache-source",
        "manifest_sha256": "manifest-hash",
        "scene_split_sha256": "split-hash",
        "normalization_sha256": "normalization-hash",
        "g3l1_report_sha256": "report-hash",
        "g3l1_config_sha256": "config-hash",
        "g3l1_checkpoint_sha256": "checkpoint-hash",
        "g3l1_source_commit": "g3l1-source",
        "parent_config_sha256": "parent-config-hash",
        "parent_checkpoint_sha256": "parent-checkpoint-hash",
    }


def test_cache_record_detects_file_tampering(tmp_path) -> None:
    configuration = metadata_configuration()
    metadata = latent_metadata(configuration)
    path = tmp_path / "latents" / "seq01_radar_00002.npz"
    write_latent_file(
        path,
        np.zeros(LATENT_SHAPE, dtype=np.float32),
        metadata,
        sequence=1,
        radar_index=2,
    )
    record = {
        "sequence": 1,
        "radar_index": 2,
        "partition": "train",
        "path": str(path.relative_to(tmp_path)),
        "sha256": sha256(path),
    }
    manifest = {
        "protocol": CACHE_PROTOCOL,
        "schema_version": 1,
        "configuration": configuration,
        "partitions": ["train"],
        "frame_count": 1,
        "latent_shape": list(LATENT_SHAPE),
        "test_accessed": False,
        "records": [record],
    }
    manifest_path = tmp_path / "g3l2_latent_cache_manifest.json"
    write_json(manifest_path, manifest)

    validated = validate_latent_cache_manifest(manifest_path)
    assert validated["frame_count"] == 1

    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="file hash differs"):
        validate_latent_record(tmp_path, record, metadata)


def test_cache_rejects_nonfinite_latent_and_nontrain_manifest(tmp_path) -> None:
    configuration = metadata_configuration()
    with pytest.raises(ValueError, match="finite"):
        write_latent_file(
            tmp_path / "bad.npz",
            np.full(LATENT_SHAPE, np.nan, dtype=np.float32),
            latent_metadata(configuration),
            sequence=1,
            radar_index=2,
        )

    manifest_path = tmp_path / "g3l2_latent_cache_manifest.json"
    write_json(
        manifest_path,
        {
            "protocol": CACHE_PROTOCOL,
            "schema_version": 1,
            "configuration": configuration,
            "partitions": ["train", "validation"],
            "frame_count": 0,
            "latent_shape": list(LATENT_SHAPE),
            "test_accessed": False,
            "records": [],
        },
    )
    with pytest.raises(ValueError, match="train only"):
        validate_latent_cache_manifest(manifest_path, validate_files=False)


def test_g3l1_run_rejects_source_mismatch(tmp_path) -> None:
    parent_config = tmp_path / "parent_config.json"
    parent_checkpoint = tmp_path / "parent.pt"
    parent_config.write_bytes(b"parent-config")
    parent_checkpoint.write_bytes(b"parent-checkpoint")
    config = {
        "protocol": "rald_anchor_g3l_physical_vae_v1",
        "latent_count": 512,
        "latent_dim": 32,
        "model_dim": 32,
        "seed": 20260716,
        "decoder_depth": 24,
        "denoiser_depth": 24,
        "edm_steps": 18,
    }
    provenance = {
        "git_commit": "expected-g3l1-source",
        "manifest_sha256": "manifest",
        "scene_split_sha256": "split",
        "normalization_sha256": "normalization",
        "g3r_selected_config": str(parent_config),
        "g3r_selected_config_sha256": hashlib.sha256(
            parent_config.read_bytes()
        ).hexdigest(),
        "g3r_selected_checkpoint": str(parent_checkpoint),
        "g3r_selected_checkpoint_sha256": hashlib.sha256(
            parent_checkpoint.read_bytes()
        ).hexdigest(),
        "g3r_geometry_parent_checkpoint": str(parent_checkpoint),
        "g3r_geometry_parent_checkpoint_sha256": hashlib.sha256(
            parent_checkpoint.read_bytes()
        ).hexdigest(),
    }
    run = tmp_path / "g3l1"
    run.mkdir()
    write_json(run / "config.json", {"config": config, "provenance": provenance})
    torch.save(
        {
            "config": config,
            "provenance": provenance,
            "g3l_vae": {
                "posterior_encoder": {},
                "anchor_decoder": {},
                "physical_head": {},
            },
        },
        run / "best.pt",
    )
    report = {
        "protocol": G3L1_GATE_PROTOCOL,
        "decision": {"g3l1_passed": True},
        "best_of_k": False,
        "selected_runs": {"20260716": str(run)},
        "selected_run_hashes": {
            "20260716": {
                "config_sha256": sha256(run / "config.json"),
                "best_checkpoint_sha256": sha256(run / "best.pt"),
            }
        },
    }
    report_path = tmp_path / "g3l1_gate.json"
    write_json(report_path, report)

    with pytest.raises(ValueError, match="source commit mismatch"):
        validate_g3l1_run(
            run,
            report_path,
            manifest_hash="manifest",
            scene_split_hash="split",
            normalization_hash="normalization",
            g3l1_source_commit="wrong-source",
        )
    with pytest.raises(ValueError, match="cache source mismatch"):
        require_cache_source(configuration=metadata_configuration(), expected_source_commit="wrong")


def test_default_schedule_matches_official_rald_contract() -> None:
    config = TrainConfig()

    assert config.epochs == OFFICIAL_EPOCHS == 100
    assert config.depth == OFFICIAL_DENOISER_DEPTH == 24
    assert config.p_mean == OFFICIAL_P_MEAN == -1.2
    assert config.p_std == OFFICIAL_P_STD == 1.2
    assert config.sigma_data == OFFICIAL_SIGMA_DATA == 1.0
    assert config.edm_steps == OFFICIAL_EDM_STEPS == 18
    assert config.sigma_min == OFFICIAL_SIGMA_MIN == 0.002
    assert config.sigma_max == OFFICIAL_SIGMA_MAX == 80.0
    assert config.rho == OFFICIAL_RHO == 7.0
    assert config.condition_mode == "full_raed"
    assert config.inference_sampler == "heun"
    assert config.checkpoint_selection == "final_epoch_no_validation_selection"
    assert len(FORMAL_SEEDS) == 3 and len(set(FORMAL_SEEDS)) == 3


def tiny_full_raed_encoder() -> FullRAEDRadarTokenEncoder:
    return FullRAEDRadarTokenEncoder(
        log_center=2.0,
        log_scale=0.5,
        spectral_channels=4,
        encoded_shape=(2, 2, 2),
        encoded_channels=4,
        token_dim=32,
        base_channels=4,
        channel_multipliers=(1, 1, 1),
        blocks_per_level=1,
    )


def test_first_and_second_edm_gradients_reach_full_raed_condition() -> None:
    torch.manual_seed(41)
    config = replace(
        TrainConfig(),
        model_dim=32,
        depth=2,
        heads=4,
        head_dim=8,
        radar_base_channels=4,
        radar_encoded_channels=4,
        radar_blocks_per_level=1,
        radar_spectral_channels=4,
    )
    model = build_edm(config, radar_encoder=tiny_full_raed_encoder())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    latent = torch.randn(1, *LATENT_SHAPE)
    cube = torch.rand(1, 64, 8, 8, 8)
    records = []

    for update in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss = edm_loss(
            model,
            latent,
            cube,
            p_mean=config.p_mean,
            p_std=config.p_std,
        )
        loss.backward()
        records.append({"update": update, "gradients": gradient_audit(model)})
        optimizer.step()

    validate_two_step_gradient_audit(records)
    assert records[0]["gradients"]["denoiser_output"] > 0.0
    assert records[1]["gradients"]["full_raed_condition_encoder"] > 0.0
