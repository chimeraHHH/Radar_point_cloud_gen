import inspect
import math

import torch

from eval.g1a_wide_support import (
    DIAGNOSTIC_ARTIFACT_LABEL,
    EXPORT_COUNT,
    RADAR_QUERY_COUNT,
    RANDOM_QUERY_COUNT,
    RANGE_STRATA_M,
    RAW_QUERY_COUNT,
    SOURCE_RADAR_HELPER_AUGMENTED,
    SOURCE_RADAR_LOW_THRESHOLD,
    SOURCE_RANDOM_FRUSTUM,
    WideQueryDomain,
    build_wide_query_domain,
    capacity_aware_topk_greedy,
    capacity_cell_keys,
    diagnostic_artifact_label,
    low_threshold_radar_queries,
    select_wide_support_diagnostic,
    stable_capacity_one_indices,
    stratified_random_queries,
    streaming_bidirectional_nearest,
)
from models.cube_cycle import continuous_rae_to_xyz


def axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(0.1, 119.9, 121),
        torch.linspace(-0.8, 0.8, 17),
        torch.linspace(-0.2, 0.2, 9),
    )


def test_frozen_ra0_counts_and_gt_free_domain_signature() -> None:
    assert RANDOM_QUERY_COUNT == 500_000
    assert RADAR_QUERY_COUNT == 700_000
    assert RAW_QUERY_COUNT == 1_200_000
    assert EXPORT_COUNT == 10_000
    assert RANGE_STRATA_M == (
        ("range_0_30m", 0.0, 30.0),
        ("range_30_60m", 30.0, 60.0),
        ("range_60_120m", 60.0, 120.0),
    )

    signature = inspect.signature(build_wide_query_domain)
    assert "target_xyz_m" not in signature.parameters
    assert "target_weight" not in signature.parameters
    assert "random_count" not in signature.parameters
    assert "radar_count" not in signature.parameters
    assert "threshold_quantile" not in signature.parameters


def test_stratified_sobol_queries_keep_exact_requested_ranges() -> None:
    range_m, azimuth, elevation = axes()
    quotas = (7, 5, 9)
    coordinates, strata = stratified_random_queries(
        range_m,
        azimuth,
        elevation,
        quotas=quotas,
        base_seed=20260716,
        sequence=6,
        radar_index=183,
    )
    xyz = continuous_rae_to_xyz(
        coordinates,
        range_m,
        azimuth,
        elevation,
    )
    radius = torch.linalg.vector_norm(xyz, dim=1)

    assert coordinates.shape == (sum(quotas), 3)
    assert torch.equal(
        torch.bincount(strata.to(torch.long), minlength=3),
        torch.tensor(quotas),
    )
    assert bool(((radius[strata == 0] >= 0.0) & (radius[strata == 0] < 30.0)).all())
    assert bool(
        ((radius[strata == 1] >= 30.0) & (radius[strata == 1] < 60.0)).all()
    )
    assert bool(
        ((radius[strata == 2] >= 60.0) & (radius[strata == 2] < 120.0)).all()
    )


def test_low_threshold_radar_queries_are_per_stratum_and_helper_augmented() -> None:
    range_m = torch.tensor(
        [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0, 115.0]
    )
    cube = torch.arange(64 * 12 * 4 * 3, dtype=torch.float32).reshape(
        64,
        12,
        4,
        3,
    )
    quotas = (20, 15, 25)
    coordinates, source, strata, reports = low_threshold_radar_queries(
        cube,
        range_m,
        quotas=quotas,
        base_seed=20260716,
        sequence=55,
        radar_index=376,
        threshold_quantile=0.75,
    )

    assert coordinates.shape == (sum(quotas), 3)
    assert torch.equal(
        torch.bincount(strata.to(torch.long), minlength=3),
        torch.tensor(quotas),
    )
    assert bool((source == SOURCE_RADAR_LOW_THRESHOLD).any())
    assert bool((source == SOURCE_RADAR_HELPER_AUGMENTED).any())
    assert sum(report["helper_augmentation_count"] for report in reports) > 0
    assert all(report["occupancy_dependent_second_pass"] is False for report in reports)
    assert all(report["threshold_quantile"] == 0.75 for report in reports)
    assert bool(
        ((coordinates[strata == 0, 0] >= 0) & (coordinates[strata == 0, 0] <= 2)).all()
    )
    assert bool(
        ((coordinates[strata == 1, 0] >= 3) & (coordinates[strata == 1, 0] <= 5)).all()
    )
    assert bool(
        ((coordinates[strata == 2, 0] >= 6) & (coordinates[strata == 2, 0] <= 11)).all()
    )


def test_capacity_cell_representative_prefers_radar_evidence() -> None:
    xyz = torch.tensor(
        [
            [1.001, 2.001, 3.001],
            [1.019, 2.019, 3.019],
            [1.101, 2.101, 3.101],
        ]
    )
    source = torch.tensor(
        [
            SOURCE_RANDOM_FRUSTUM,
            SOURCE_RADAR_LOW_THRESHOLD,
            SOURCE_RADAR_HELPER_AUGMENTED,
        ],
        dtype=torch.int8,
    )
    kept = stable_capacity_one_indices(xyz, source_codes=source)

    assert kept.tolist() == [1, 2]
    keys = capacity_cell_keys(xyz[kept])
    assert torch.unique(keys).numel() == 2


def test_streaming_nearest_matches_full_matrix_and_uses_id_ties() -> None:
    candidates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ]
    )
    candidate_ids = torch.tensor([9, 3, 11])
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [7.5, 0.0, 0.0],
        ]
    )
    result = streaming_bidirectional_nearest(
        candidates,
        targets,
        candidate_ids,
        candidate_chunk_size=2,
        target_chunk_size=1,
        target_topk_count=2,
    )
    full = torch.cdist(candidates, targets)

    torch.testing.assert_close(
        result.candidate_to_target_m,
        full.amin(dim=1),
    )
    torch.testing.assert_close(
        result.target_to_candidate_m,
        full.amin(dim=0),
    )
    assert result.target_nearest_candidate_row.tolist() == [1, 2]
    assert result.target_topk_candidate_rows.tolist() == [[1, 0], [2, 1]]


def test_capacity_aware_topk_reassigns_after_true_euclidean_rejection() -> None:
    candidate_xyz = torch.tensor(
        [
            [10.00, 0.0, 0.0],
            [10.02, 0.0, 0.0],
            [10.10, 0.0, 0.0],
        ]
    )
    selected, report = capacity_aware_topk_greedy(
        candidate_xyz,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0, 0], dtype=torch.int8),
        torch.tensor([0.01, 0.01, 0.08]),
        torch.tensor([[0, 2], [1, 2]]),
        torch.tensor([[0.01, 0.10], [0.01, 0.08]]),
        torch.tensor([1.0, 0.9]),
        output_quotas=[2, 0, 0],
    )

    assert selected.tolist() == [0, 2]
    assert report["topk_assignment_count"] == 2
    assert report["rejected_euclidean_duplicate_count"] == 1
    distance = torch.cdist(
        candidate_xyz[selected],
        candidate_xyz[selected],
    )
    distance.fill_diagonal_(float("inf"))
    assert float(distance.amin().item()) >= 0.05


def synthetic_capacity_one_domain() -> WideQueryDomain:
    parts = []
    strata = []
    for stratum, radius in enumerate((20.0, 50.0, 90.0)):
        radius_values = torch.linspace(radius - 1.5, radius + 1.5, 4)
        azimuth_values = torch.linspace(-0.5, 0.5, 50)
        elevation_values = torch.linspace(-0.1, 0.1, 20)
        grid = torch.cartesian_prod(
            radius_values,
            azimuth_values,
            elevation_values,
        )
        r, a, e = grid.unbind(dim=1)
        xyz = torch.stack(
            (
                r * torch.cos(e) * torch.cos(a),
                r * torch.cos(e) * torch.sin(a),
                r * torch.sin(e),
            ),
            dim=1,
        )
        assert xyz.shape == (4_000, 3)
        parts.append(xyz)
        strata.append(torch.full((4_000,), stratum, dtype=torch.int8))
    xyz = torch.cat(parts)
    assert torch.unique(capacity_cell_keys(xyz)).numel() == xyz.shape[0]
    count = xyz.shape[0]
    return WideQueryDomain(
        coordinates_rae=torch.zeros(count, 3),
        xyz_m=xyz,
        candidate_ids=torch.arange(count, dtype=torch.long) * 2 + 1,
        source_codes=(torch.arange(count) % 3).to(torch.int8),
        range_stratum_codes=torch.cat(strata),
        report={"ground_truth_used_for_proposal_generation": False},
    )


def test_wide_diagnostic_exports_exact_10k_without_candidate_reuse() -> None:
    domain = synthetic_capacity_one_domain()
    target_rows = torch.arange(0, domain.xyz_m.shape[0], 400)
    target = domain.xyz_m[target_rows] + 0.01
    weight = torch.linspace(0.5, 1.5, target.shape[0])
    selection = select_wide_support_diagnostic(
        domain,
        target,
        target_weight=weight,
        candidate_chunk_size=256,
        target_chunk_size=16,
    )

    assert selection.selected_xyz_m.shape == (EXPORT_COUNT, 3)
    assert selection.selected_candidate_ids.shape == (EXPORT_COUNT,)
    assert torch.unique(selection.selected_candidate_ids).numel() == EXPORT_COUNT
    assert (
        sum(
            int(report["selected_count"])
            for report in selection.per_range_support.values()
        )
        == EXPORT_COUNT
    )
    assert selection.overall_support["selected_count"] == EXPORT_COUNT
    assert selection.overall_support["pool_support_scope"] == (
        "global_cross_range_nearest"
    )
    assert selection.selection_report["heuristic"] is True
    assert selection.selection_report["strict_upper_bound"] is False
    assert selection.selection_report[
        "range_capacity_validated_after_unique_pool"
    ] is True
    assert set(selection.selection_report["output_quotas"]) == {
        "range_0_30m",
        "range_30_60m",
        "range_60_120m",
    }
    assert math.isfinite(
        float(selection.overall_support["gt_recall_from_pool_2p0m"])
    )


def test_full_pool_support_crosses_range_boundaries() -> None:
    domain = synthetic_capacity_one_domain()
    xyz = domain.xyz_m.clone()
    xyz[4_000] = torch.tensor([30.01, 0.0, 0.0])
    assert torch.unique(capacity_cell_keys(xyz)).numel() == xyz.shape[0]
    physical_range_codes = torch.zeros_like(domain.range_stratum_codes)
    radius = torch.linalg.vector_norm(xyz, dim=1)
    physical_range_codes[(radius >= 30.0) & (radius < 60.0)] = 1
    physical_range_codes[(radius >= 60.0) & (radius < 120.0)] = 2
    cross_boundary_domain = WideQueryDomain(
        coordinates_rae=domain.coordinates_rae,
        xyz_m=xyz,
        candidate_ids=domain.candidate_ids,
        source_codes=domain.source_codes,
        range_stratum_codes=physical_range_codes,
        report=domain.report,
    )
    selection = select_wide_support_diagnostic(
        cross_boundary_domain,
        torch.tensor([[29.99, 0.0, 0.0]]),
        candidate_chunk_size=256,
        target_chunk_size=1,
    )

    near_report = selection.per_range_support["range_0_30m"]
    assert near_report["target_count"] == 1
    assert near_report["gt_recall_from_pool_0p5m"] == 1.0
    assert near_report["pool_support_scope"] == "global_cross_range_nearest"


def test_diagnostic_label_cannot_be_mistaken_for_a_strict_upper_bound() -> None:
    label = diagnostic_artifact_label()

    assert label["label"] == DIAGNOSTIC_ARTIFACT_LABEL
    assert label["diagnostic"] is True
    assert label["unattainable"] is True
    assert label["strict_upper_bound"] is False
    assert label["selection_is_heuristic"] is True
    assert label["rald_scope"] == "inspired_initial_query_domain_only"
    assert label["full_rald_wide_family_closure_eligible"] is False
    assert label["ground_truth_used_for_selection"] is True
    assert label["ground_truth_used_for_proposal_generation"] is False
    assert label["eligible_as_method_result"] is False
