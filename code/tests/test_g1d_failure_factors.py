from dataclasses import asdict
import json

import pytest
import torch

from eval.g1d_failure_factors import (
    CURRENT_RANKING_SOURCE,
    OFFSET_ALPHAS,
    RANKING_SOURCES,
    aggregate_arm,
    alpha_label,
    current_ranking_score,
    generate_selected_points,
    paired_scene_bootstrap,
    prepare_query_field,
    ranking_capability,
    ranking_scores,
    tensor_sha256,
)
from models.rald_query_field import RaLDQueryField
from scripts.diagnose_g1d_failure_factors import (
    EXPECTED_CHECKPOINT_EPOCH,
    EXPECTED_FAR_FRAME_COUNT,
    EXPECTED_POINT_COUNT,
    EXPECTED_SEED,
    validate_checkpoint_metadata,
    validate_corrected_reference,
)
from scripts.g1b_contract import sha256
from scripts.train_g1g_hierarchy import G1D_CONTROL_PROTOCOL
from scripts.train_rald_query_field import PROTOCOL, TrainConfig


SPATIAL_SHAPE = (5, 5, 3)


def tiny_model() -> RaLDQueryField:
    return RaLDQueryField(
        torch.linspace(0.0, 8.0, SPATIAL_SHAPE[0]),
        torch.linspace(-0.5, 0.5, SPATIAL_SHAPE[1]),
        torch.linspace(-0.2, 0.2, SPATIAL_SHAPE[2]),
        log_center=0.0,
        log_scale=1.0,
        base_seed_count=2,
        selected_coarse_count=3,
        latent_count=4,
        model_dim=12,
        depth=1,
        heads=3,
        head_dim=4,
        fourier_frequency_dim=12,
        radar_base_channels=4,
        radar_spectral_channels=4,
        radar_encoded_shape=SPATIAL_SHAPE,
        radar_encoded_channels=4,
        radar_channel_multipliers=(1,),
        radar_blocks_per_level=1,
        offset_bounds_bins=(1.0, 0.5, 0.5),
        nms_kernel=(3, 3, 3),
        decode_chunk_size=7,
    )


def cube(
    primary: tuple[int, int, int] = (1, 1, 1),
    secondary: tuple[int, int, int] = (3, 3, 1),
) -> torch.Tensor:
    measured = torch.full((1, 64, *SPATIAL_SHAPE), 1e-3)
    measured[:, :, primary[0], primary[1], primary[2]] = 2.0
    measured[:, :, secondary[0], secondary[1], secondary[2]] = 1.0
    measured[:, 7, primary[0], primary[1], primary[2]] = 4.0
    measured[:, 41, secondary[0], secondary[1], secondary[2]] = 3.0
    return measured


def formal_config() -> TrainConfig:
    return TrainConfig(
        protocol=PROTOCOL,
        epochs=EXPECTED_CHECKPOINT_EPOCH,
        learning_rate=1e-4,
        minimum_learning_rate=1e-6,
        warmup_epochs=5,
        weight_decay=0.05,
        gradient_clip_norm=10.0,
        ema_decay=0.999,
        seed=EXPECTED_SEED,
        occupancy_query_count=10_000,
        positive_query_ratio=0.0625,
        base_seed_count=1_000,
        coarse_templates_per_seed=32,
        coarse_query_count=32_000,
        selected_coarse_count=2_500,
        local_templates_per_selection=4,
        point_count=EXPECTED_POINT_COUNT,
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
        existence_weight=0.1,
        offset_weight=0.02,
        repulsion_weight=0.02,
        outlier_threshold_m=2.0,
        existence_radius_m=1.0,
        repulsion_distance_m=0.1,
        eval_every=5,
        train_limit=None,
        validation_limit=None,
        selection_metric="frozen",
        test_accessed=False,
    )


def test_joint_alpha_endpoints_match_checkpoint_forward_controls() -> None:
    torch.manual_seed(7)
    model = tiny_model().eval()
    with torch.no_grad():
        model.offset_head.weight.normal_(mean=0.0, std=0.05)
        model.offset_head.bias.copy_(torch.tensor([0.4, -0.2, 0.1]))
    measured = cube()
    reference = model(measured)
    prepared = prepare_query_field(
        model,
        measured,
        measured,
        proposal_flat_index=reference["proposal_flat_index"],
    )
    score = current_ranking_score(prepared)
    zero = generate_selected_points(
        model,
        prepared,
        score,
        residual_alpha=0.0,
    )
    full = generate_selected_points(
        model,
        prepared,
        score,
        residual_alpha=1.0,
    )

    torch.testing.assert_close(zero["xyz_m"], reference["zero_offset_xyz_m"])
    torch.testing.assert_close(full["xyz_m"], reference["xyz_m"])
    torch.testing.assert_close(
        full["selected_index"],
        reference["coarse_selection_provenance"]["selected_coarse_index"],
    )


def test_d1_freezes_topk_for_every_registered_alpha() -> None:
    torch.manual_seed(11)
    model = tiny_model().eval()
    prepared = prepare_query_field(model, cube(), cube())
    score = current_ranking_score(prepared)
    selections = {
        generate_selected_points(
            model,
            prepared,
            score,
            residual_alpha=alpha,
        )["selected_index_sha256"]
        for alpha in OFFSET_ALPHAS
    }

    assert len(selections) == 1
    assert [alpha_label(alpha) for alpha in OFFSET_ALPHAS] == [
        "alpha_0p0",
        "alpha_0p25",
        "alpha_0p5",
        "alpha_0p75",
        "alpha_1p0",
    ]


def test_tensor_hash_supports_bfloat16_outputs() -> None:
    value = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    assert tensor_sha256(value) == tensor_sha256(value.clone())


def test_d3_exposes_five_scores_on_the_same_32k_analogue_pool() -> None:
    torch.manual_seed(13)
    model = tiny_model().eval()
    prepared = prepare_query_field(model, cube(), cube())
    target = torch.tensor([[2.0, 0.0, 0.0], [5.0, 1.0, 0.2]])
    scores = ranking_scores(
        model,
        prepared,
        validation_target_xyz=target,
    )

    assert tuple(scores) == RANKING_SOURCES
    assert ranking_capability(model, prepared.coarse_fields)["supported"] is True
    assert torch.equal(
        scores[CURRENT_RANKING_SOURCE],
        prepared.coarse_fields["occupancy_logit"]
        + prepared.coarse_fields["confidence_logit"],
    )
    assert torch.equal(
        scores["integrated_energy"],
        prepared.coarse_fields["query_absolute_log_energy"].squeeze(-1),
    )
    assert all(
        score.shape == (1, model.coarse_query_count)
        for score in scores.values()
    )


def test_d3_reports_machine_readable_unsupported_capability() -> None:
    class InseparableModel:
        coarse_query_count = 32_000
        selected_coarse_count = 2_500
        occupancy_head = object()

    capability = ranking_capability(InseparableModel())

    assert capability["status"] == "unsupported"
    assert capability["supported"] is False
    assert capability["missing_model_attributes"] == ["confidence_head"]
    assert capability["available_sources"] == []


def test_d2_separates_measured_and_global_condition_paths() -> None:
    torch.manual_seed(17)
    model = tiny_model().eval()
    current = cube()
    shuffled = cube(primary=(3, 1, 1), secondary=(1, 3, 1))
    matched = prepare_query_field(model, current, current)
    condition_wrong = prepare_query_field(
        model,
        current,
        shuffled,
        proposal_flat_index=matched.proposal_flat_index,
    )
    measured_wrong = prepare_query_field(model, shuffled, current)

    torch.testing.assert_close(
        matched.proposal_flat_index,
        condition_wrong.proposal_flat_index,
    )
    torch.testing.assert_close(
        matched.coarse_fields["query_absolute_log_energy"],
        condition_wrong.coarse_fields["query_absolute_log_energy"],
    )
    assert not torch.allclose(matched.latent, condition_wrong.latent)
    assert not torch.equal(
        matched.proposal_flat_index,
        measured_wrong.proposal_flat_index,
    )


def test_aggregate_arm_requires_exact_points_and_corrected_far_count() -> None:
    base = {
        "precision_mean_distance_m": 1.0,
        "completeness_mean_distance_m": 2.0,
        "chamfer_m": 3.0,
        "outlier_fraction_2m": 0.2,
        "prediction_count": 10,
        "target_count": 4,
        "target_effective_count": 3.0,
    }
    frames = [
        {
            "sequence": 1,
            "radar_index": 10,
            "geometry": {
                **base,
                "range_60_120m_completeness_mean_distance_m": 8.0,
            },
        },
        {
            "sequence": 2,
            "radar_index": 20,
            "geometry": base,
        },
    ]
    report = aggregate_arm(
        frames,
        expected_frame_count=2,
        expected_far_count=1,
        expected_point_count=10,
    )

    assert report["frame_count"] == 2
    assert report["geometry"][
        "range_60_120m_completeness_mean_distance_m"
    ]["sample_count"] == 1
    with pytest.raises(ValueError, match="far completeness"):
        aggregate_arm(
            frames,
            expected_frame_count=2,
            expected_far_count=2,
            expected_point_count=10,
        )


def test_paired_bootstrap_is_scene_first_and_candidate_minus_reference() -> None:
    reference = [
        {
            "sequence": sequence,
            "radar_index": frame,
            "geometry": {"chamfer_m": 2.0},
        }
        for sequence, frame in ((1, 10), (1, 20), (2, 30))
    ]
    candidate = [
        {
            **frame,
            "geometry": {"chamfer_m": frame["geometry"]["chamfer_m"] - 0.5},
        }
        for frame in reference
    ]
    report = paired_scene_bootstrap(
        reference,
        candidate,
        group="geometry",
        metric="chamfer_m",
        samples=100,
        seed=3,
    )

    assert report["scene_count"] == 2
    assert report["frame_count"] == 3
    assert report["mean_delta"] == pytest.approx(-0.5)
    assert report["delta_ci95"] == pytest.approx([-0.5, -0.5])


def test_checkpoint_validator_requires_epoch150_ema_and_no_test_access() -> None:
    config = formal_config()
    checkpoint = {
        "epoch": EXPECTED_CHECKPOINT_EPOCH,
        "config": asdict(config),
        "provenance": {
            "git_commit": "a" * 40,
            "test_accessed": False,
            "partitions": ["train", "validation"],
        },
        "ema_model": {"weight": torch.ones(1)},
        "record": {"epoch": EXPECTED_CHECKPOINT_EPOCH},
    }

    assert validate_checkpoint_metadata(checkpoint) == config
    checkpoint["epoch"] = EXPECTED_CHECKPOINT_EPOCH - 1
    with pytest.raises(ValueError, match="requires epoch"):
        validate_checkpoint_metadata(checkpoint)


def test_corrected_reference_binds_evaluator_and_frozen_24_23(
    tmp_path,
) -> None:
    evaluator = tmp_path / "dense_geometry.py"
    evaluator.write_text("# corrected\n", encoding="utf-8")
    reference_path = tmp_path / "control.json"
    reference_path.write_text(json.dumps({"placeholder": True}), encoding="utf-8")
    identities = [
        {"sequence": index, "radar_index": index + 100}
        for index in range(24)
    ]
    far = identities[:EXPECTED_FAR_FRAME_COUNT]
    data_contract = {"manifest_sha256": "a" * 64}
    corrected = {
        "protocol": G1D_CONTROL_PROTOCOL,
        "test_accessed": False,
        "data_contract": data_contract,
        "validation_frame_identities": identities,
        "far_target_frame_identities": far,
        "evaluator": {
            "dense_geometry_sha256": sha256(evaluator),
            "far_target_frame_censoring_fixed": True,
        },
        "metrics": {
            "frame_count": 24,
            "generated": {
                "range_60_120m_completeness_mean_distance_m": {
                    "sample_count": EXPECTED_FAR_FRAME_COUNT,
                }
            },
        },
    }
    result = validate_corrected_reference(
        corrected,
        corrected_path=reference_path,
        data_contract=data_contract,
        validation_identities=identities,
        far_identities=far,
        evaluator_path=evaluator,
    )

    assert result["evaluator_sha256"] == sha256(evaluator)
    assert all(result["checks"].values())
