import hashlib
import json

import pytest

from scripts.train_rald_matched_edm import (
    TrainConfig,
    build_edm,
    frame_seed,
    validate_latent_cache,
    validate_normalization,
)


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_normalization_requires_full_train_only_rae_sum(tmp_path) -> None:
    path = tmp_path / "normalization.json"
    document = {
        "representation": "log10_sum_doppler_power_plus_one",
        "manifest_sha256": "manifest",
        "scene_split_sha256": "split",
        "partitions": ["train"],
        "frame_limit": None,
        "frame_count": 76,
    }
    write_json(path, document)

    assert validate_normalization(path, "manifest", "split") == document

    document["partitions"] = ["train", "validation"]
    write_json(path, document)
    with pytest.raises(ValueError, match="all train frames"):
        validate_normalization(path, "manifest", "split")


def test_latent_cache_is_bound_to_ae_checkpoint_and_development_split(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen-ae")
    path = tmp_path / "latent_cache_manifest.json"
    document = {
        "manifest_sha256": "manifest",
        "scene_split_sha256": "split",
        "partitions": ["train", "validation"],
        "frame_count": 100,
        "ae_checkpoint_sha256": digest(checkpoint),
    }
    write_json(path, document)

    assert validate_latent_cache(
        path, "manifest", "split", checkpoint
    ) == document

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_latent_cache(path, "manifest", "split", checkpoint)


def test_small_edm_builder_preserves_ae_latent_shape() -> None:
    config = TrainConfig(
        epochs=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        seed=17,
        model_dim=32,
        depth=2,
        heads=4,
        head_dim=8,
        radar_base_channels=4,
        radar_encoded_channels=4,
        radar_blocks_per_level=1,
        edm_steps=3,
        output_point_count=10,
        query_chunk_size=8,
        eval_every=1,
        max_eval_frames=1,
        train_limit=1,
        validation_limit=1,
        overfit_one_frame=True,
        normalization_center=2.0,
        normalization_scale=0.5,
    )

    model = build_edm(config, {"latent_count": 8, "latent_dim": 4})

    assert model.latent_count == 8
    assert model.latent_dim == 4
    assert frame_seed(17, 3, 41) != frame_seed(17, 3, 42)
