import pytest
import torch

from eval.rald_wce_stage0 import (
    ExactExportCapacityError,
    WideInferenceConfig,
    exact_capacity_export,
    global_exact_capacity_export,
    fixed_wide_q0,
    occupancy_dependent_q1,
    range_stratum_codes,
)
from losses.rald_wce import sample_bounded_occupancy_queries


def axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(0.0, 119.0, 120),
        torch.linspace(-1.0, 1.0, 41),
        torch.linspace(-0.4, 0.4, 21),
    )


def test_formal_inference_config_is_exact_and_doppler_locked() -> None:
    config = WideInferenceConfig()
    metadata = config.metadata()

    assert metadata["q0_query_count"] == 500_000
    assert metadata["q1_query_count"] == 200_000
    assert metadata["export_count"] == 10_000
    assert sum(metadata["output_range_quotas"].values()) == 10_000
    assert metadata["same_query_wrong_condition"] is True
    assert metadata["copy_padding_jitter_duplicate"] is False
    assert metadata["doppler_head"] is False


def test_fixed_q0_is_deterministic_stratified_and_frame_invariant() -> None:
    range_m, azimuth, elevation = axes()
    first = fixed_wide_q0(
        range_m,
        azimuth,
        elevation,
        quotas=(9, 8, 7),
        seed=37,
    )
    second = fixed_wide_q0(
        range_m,
        azimuth,
        elevation,
        quotas=(9, 8, 7),
        seed=37,
    )

    assert first.shape == (1, 24, 3)
    assert torch.all(first.abs() <= 1.0)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_q1_depends_on_matched_occupancy_and_is_deterministic() -> None:
    range_m, azimuth, elevation = axes()
    q0 = fixed_wide_q0(
        range_m,
        azimuth,
        elevation,
        quotas=(12, 12, 12),
        seed=41,
    )
    confidence = torch.linspace(0.0, 1.0, q0.shape[1]).unsqueeze(0)
    output = {
        "refined_normalized_rae": q0,
        "confidence": confidence,
    }
    first, first_report = occupancy_dependent_q1(
        output,
        range_m,
        azimuth,
        elevation,
        anchor_quotas=(2, 2, 2),
        samples_per_anchor=3,
        seed=43,
    )
    second, second_report = occupancy_dependent_q1(
        output,
        range_m,
        azimuth,
        elevation,
        anchor_quotas=(2, 2, 2),
        samples_per_anchor=3,
        seed=43,
    )

    assert first.shape == (1, 18, 3)
    assert first_report["source"] == "matched_condition_q0_occupancy_only"
    assert first_report["query_sha256"] == second_report["query_sha256"]
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def _point_grid(
    count_xyz: tuple[int, int, int],
    origin: tuple[float, float, float],
) -> torch.Tensor:
    x = origin[0] + 0.06 * torch.arange(count_xyz[0])
    y = origin[1] + 0.06 * torch.arange(count_xyz[1])
    z = origin[2] + 0.06 * torch.arange(count_xyz[2])
    return torch.cartesian_prod(x, y, z)


def exact_candidate_set() -> tuple[torch.Tensor, torch.Tensor]:
    near = _point_grid((20, 20, 20), (10.0, -0.6, -0.6))
    middle = _point_grid((17, 10, 10), (40.0, -0.3, -0.3))
    far = _point_grid((3, 10, 10), (70.0, -0.3, -0.3))
    xyz = torch.cat((near, middle, far), dim=0)
    confidence = torch.linspace(1.0, 0.0, xyz.shape[0])
    return xyz, confidence


def test_exact_export_has_true_5cm_capacity_and_frozen_quotas() -> None:
    xyz, confidence = exact_candidate_set()
    export = exact_capacity_export(
        xyz,
        confidence,
        output_quotas=(8_000, 1_700, 300),
        minimum_distance_m=0.05,
    )

    assert export.xyz_m.shape == (10_000, 3)
    assert export.confidence.shape == (10_000,)
    assert torch.unique(export.selected_candidate_rows).numel() == 10_000
    assert export.report["selected_by_range"] == {
        "range_0_30m": 8_000,
        "range_30_60m": 1_700,
        "range_60_120m": 300,
    }
    assert export.report["observed_minimum_pair_distance_m"] >= 0.05 - 1e-6
    assert export.report["ground_truth_accessed"] is False
    codes = range_stratum_codes(export.xyz_m)
    assert torch.bincount(codes, minlength=3).tolist() == [8_000, 1_700, 300]


def test_exact_export_refuses_capacity_failure_without_fill() -> None:
    xyz, confidence = exact_candidate_set()
    keep = torch.arange(xyz.shape[0] - 101)
    with pytest.raises(ExactExportCapacityError) as caught:
        exact_capacity_export(
            xyz[keep],
            confidence[keep],
            output_quotas=(8_000, 1_700, 300),
            minimum_distance_m=0.05,
        )

    report = caught.value.report
    assert report["selected_by_range"]["range_60_120m"] == 199
    assert report["copy_padding_jitter_duplicate"] is False


def test_global_export_has_exact_count_without_range_quotas() -> None:
    xyz, confidence = exact_candidate_set()
    confidence[:300] = 2.0
    export = global_exact_capacity_export(
        xyz,
        confidence,
        minimum_distance_m=0.05,
    )

    assert export.xyz_m.shape == (10_000, 3)
    assert torch.unique(export.selected_candidate_rows).numel() == 10_000
    assert export.report["range_quotas_enforced"] is False
    assert sum(export.report["selected_by_range"].values()) == 10_000
    assert export.report["observed_minimum_pair_distance_m"] >= 0.05 - 1e-6
    assert export.report["ground_truth_accessed"] is False


def test_global_export_refuses_true_capacity_failure() -> None:
    xyz = torch.zeros(10_000, 3, dtype=torch.float32)
    xyz[:, 0] = 10.0 + 0.001 * torch.arange(10_000)
    confidence = torch.linspace(1.0, 0.0, xyz.shape[0])

    with pytest.raises(ExactExportCapacityError) as caught:
        global_exact_capacity_export(
            xyz,
            confidence,
            minimum_distance_m=0.05,
        )

    assert caught.value.report["range_quotas_enforced"] is False
    assert caught.value.report["copy_padding_jitter_duplicate"] is False


def test_bounded_training_queries_match_rald_positive_ratio() -> None:
    target = torch.cartesian_prod(
        torch.arange(16),
        torch.arange(8),
        torch.arange(2),
    )
    sampled = sample_bounded_occupancy_queries(
        target,
        spatial_shape=(32, 16, 4),
        query_count=1_024,
        positive_query_ratio=0.0625,
        seed=47,
    )

    assert sampled["positive_count"] == 64
    assert sampled["negative_count"] == 960
    assert sampled["requested_positive_count"] == 64
    assert sampled["normalized_rae"].shape == (1, 1_024, 3)
    assert torch.count_nonzero(sampled["occupancy_target"]) == 64
