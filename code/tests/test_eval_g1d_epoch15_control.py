from pathlib import Path

import pytest

from scripts.eval_g1d_epoch15_control import (
    build_document,
    validate_checkpoint,
)
from scripts.train_rald_query_field import TrainConfig


def checkpoint_document() -> dict:
    config = TrainConfig(
        protocol="g1d_rald_query_field_geometry_v2",
        epochs=150,
        learning_rate=1e-4,
        minimum_learning_rate=1e-6,
        warmup_epochs=5,
        weight_decay=0.05,
        gradient_clip_norm=10.0,
        ema_decay=0.999,
        seed=20260716,
        occupancy_query_count=10_000,
        positive_query_ratio=0.0625,
        base_seed_count=1_000,
        coarse_templates_per_seed=32,
        coarse_query_count=32_000,
        selected_coarse_count=2_500,
        local_templates_per_selection=4,
        point_count=10_000,
        latent_count=512,
        model_dim=512,
        depth=24,
        heads=8,
        head_dim=64,
        radar_base_channels=64,
        radar_spectral_channels=16,
        radar_token_count=336,
        nms_kernel=(5, 5, 3),
        offset_bounds_bins=(8.0, 4.0, 2.0),
        decode_chunk_size=4_096,
        positive_occupancy_weight=0.1,
        negative_occupancy_weight=1.0,
        geometry_weight=1.0,
        outlier_weight=0.25,
        existence_weight=0.10,
        offset_weight=0.02,
        repulsion_weight=0.02,
        outlier_threshold_m=2.0,
        existence_radius_m=1.0,
        repulsion_distance_m=0.10,
        eval_every=5,
        train_limit=None,
        validation_limit=None,
        selection_metric="test",
        test_accessed=False,
    )
    return {
        "epoch": 15,
        "provenance": {
            "git_commit": "4c6150cdd86ec1298f4b056569e3780020b4d8af"
        },
        "config": config.__dict__,
        "ema_model": {},
    }


def test_checkpoint_validation_requires_frozen_epoch_source_and_ema() -> None:
    checkpoint = checkpoint_document()
    config = validate_checkpoint(checkpoint)
    assert config.seed == 20260716
    assert config.point_count == 10_000

    for key, value, message in (
        ("epoch", 14, "epoch-15"),
        ("ema_model", None, "EMA"),
    ):
        invalid = dict(checkpoint)
        if value is None:
            del invalid[key]
        else:
            invalid[key] = value
        with pytest.raises(ValueError, match=message):
            validate_checkpoint(invalid)


def test_document_rejects_censored_far_metrics(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    source_script = tmp_path / "scripts/eval.py"
    source_script.parent.mkdir()
    source_script.write_text("pass\n", encoding="utf-8")
    evaluator = tmp_path / "eval/dense_geometry.py"
    evaluator.parent.mkdir()
    evaluator.write_text("pass\n", encoding="utf-8")
    metrics = {
        "frame_count": 24,
        "generated": {
            "range_60_120m_completeness_mean_distance_m": {
                "sample_count": 23,
            }
        },
    }
    with pytest.raises(ValueError, match="far-target"):
        build_document(
            source_commit="c" * 40,
            source_script=source_script,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint_document(),
            data_contract={},
            validation_records=[],
            metrics=metrics,
            device_name="NVIDIA H200",
        )
