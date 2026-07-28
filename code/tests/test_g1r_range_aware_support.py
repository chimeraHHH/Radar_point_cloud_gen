import torch

from eval.g1f_candidate_support import PROPOSAL_COUNT, RANGE_BINS_M
from eval.g1r_range_aware_support import (
    CANDIDATE_PARENT_QUOTAS,
    EXPORT_QUOTAS,
    ExpandedCandidatePool,
    calibrated_integrated_energy,
    fit_range_energy_calibration,
    physical_angular_suppression_mask,
    range_aware_proposal_indices,
    select_fixed_quota_support_oracle,
    stable_score_proposal_indices,
    template_safe_range_mask,
)
from models.rald_query_field import (
    integrated_log_energy,
    stable_radar_proposals,
)


def test_exact_per_range_median_mad_calibration() -> None:
    first = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[10.0, 12.0], [14.0, 16.0]],
        ]
    )
    second = first + torch.tensor([[[2.0]], [[4.0]]])

    profile = fit_range_energy_calibration([first, second])
    calibrated = calibrated_integrated_energy(
        torch.stack((first, second)),
        profile,
    )

    assert profile.training_frame_count == 2
    assert profile.sample_count.tolist() == [8, 8]
    torch.testing.assert_close(profile.median, torch.tensor([5.0, 14.0]))
    torch.testing.assert_close(profile.mad, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(
        calibrated[0, :, 0, 0],
        torch.tensor([-2.0, -2.0]),
    )


def test_external_score_nms_matches_g1d_stable_proposals() -> None:
    cube = torch.arange(
        64 * 8 * 7 * 5,
        dtype=torch.float32,
    ).reshape(1, 64, 8, 7, 5)
    energy = integrated_log_energy(cube)

    expected = stable_radar_proposals(
        cube,
        seed_count=8,
        nms_kernel=(3, 3, 3),
    ).flat_index
    actual = stable_score_proposal_indices(
        energy,
        seed_count=8,
        nms_kernel=(3, 3, 3),
    )

    torch.testing.assert_close(actual, expected)


def test_physical_angular_nms_shrinks_with_range() -> None:
    range_m = torch.tensor([10.0, 100.0])
    azimuth = torch.linspace(-0.2, 0.2, 41)
    elevation = torch.linspace(-0.1, 0.1, 21)

    near = physical_angular_suppression_mask(
        range_m,
        azimuth,
        elevation,
        range_index=0,
        azimuth_index=20,
        elevation_index=10,
        lateral_radius_m=1.0,
    )
    far = physical_angular_suppression_mask(
        range_m,
        azimuth,
        elevation,
        range_index=1,
        azimuth_index=20,
        elevation_index=10,
        lateral_radius_m=1.0,
    )

    assert int(near.sum()) > int(far.sum())
    assert bool(near[20, 10])
    assert bool(far[20, 10])


def test_range_aware_selector_obeys_supplied_bin_quotas() -> None:
    range_m = torch.arange(0.5, 120.0, 0.5)
    azimuth = torch.linspace(-0.4, 0.4, 17)
    elevation = torch.linspace(-0.2, 0.2, 9)
    score = torch.arange(
        range_m.numel() * azimuth.numel() * elevation.numel(),
        dtype=torch.float32,
    ).reshape(1, range_m.numel(), azimuth.numel(), elevation.numel())

    selected = range_aware_proposal_indices(
        score,
        range_m,
        azimuth,
        elevation,
        seed_quotas=(3, 2, 1),
        lateral_radius_m=0.05,
        radial_radius_m=0.05,
    )
    plane = azimuth.numel() * elevation.numel()
    radial_index = selected[0] // plane
    selected_range = range_m[radial_index]
    counts = tuple(
        int(((selected_range >= lower) & (selected_range < upper)).sum())
        for _, lower, upper in RANGE_BINS_M
    )

    assert selected.shape == (1, 6)
    assert torch.unique(selected).numel() == 6
    assert counts == (3, 2, 1)
    safe = template_safe_range_mask(range_m)
    assert bool(safe[radial_index].all())


def test_template_safe_mask_excludes_range_quota_boundaries() -> None:
    range_m = torch.arange(0.5, 120.0, 0.5)
    safe = template_safe_range_mask(range_m)
    index_30m = int(torch.argmin((range_m - 30.0).abs()))
    index_60m = int(torch.argmin((range_m - 60.0).abs()))

    assert not bool(safe[index_30m])
    assert not bool(safe[index_30m - 1])
    assert not bool(safe[index_60m])
    assert not bool(safe[index_60m - 1])
    assert bool(safe[int(torch.argmin((range_m - 20.0).abs()))])
    assert bool(safe[int(torch.argmin((range_m - 40.0).abs()))])
    assert bool(safe[int(torch.argmin((range_m - 80.0).abs()))])


def _full_candidate_pool() -> ExpandedCandidatePool:
    counts = (24_000, 6_400, 1_600)
    ranges = []
    parent_bins = []
    for bin_index, (count, radius) in enumerate(
        zip(counts, (10.0, 40.0, 80.0))
    ):
        ranges.append(
            torch.stack(
                (
                    torch.full((count,), radius),
                    torch.linspace(-0.2, 0.2, count),
                    torch.zeros(count),
                ),
                dim=1,
            )
        )
        parent_bins.append(torch.full((count,), bin_index, dtype=torch.long))
    xyz = torch.cat(ranges)
    return ExpandedCandidatePool(
        xyz_m=xyz,
        candidate_indices=torch.arange(PROPOSAL_COUNT),
        parent_range_bin=torch.cat(parent_bins),
        proposal_flat_index=torch.arange(1_000),
    )


def test_gt_aided_heuristic_locks_explicit_export_quotas() -> None:
    target_xyz = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
        ]
    )
    target_weight = torch.tensor([1.0, 3.0, 2.0, 4.0])

    result = select_fixed_quota_support_oracle(
        _full_candidate_pool(),
        target_xyz,
        target_weight,
        distance_chunk_size=32,
    )
    selected_counts = tuple(
        int(result.per_range_support[label]["selected_count"])
        for label, _, _ in RANGE_BINS_M
    )
    assert result.selected_xyz_m.shape == (10_000, 3)
    assert selected_counts == EXPORT_QUOTAS
    assert CANDIDATE_PARENT_QUOTAS == (24_000, 6_400, 1_600)


def test_seq51_radar305_equivalent_empty_mid_bin_uses_gt_free_fill() -> None:
    target_xyz = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
        ]
    )
    result = select_fixed_quota_support_oracle(
        _full_candidate_pool(),
        target_xyz,
        torch.ones(2),
        distance_chunk_size=32,
    )

    mid = result.per_range_support["range_30_60m"]
    assert mid["target_count"] == 0
    assert mid["ground_truth_used_for_selection"] is False
    assert (
        mid["selection_mode"]
        == "deterministic_candidate_index_gt_free_fill"
    )
    torch.testing.assert_close(
        result.selected_candidate_indices[7_500:9_500],
        torch.arange(24_000, 26_000),
    )


def test_seq51_radar454_equivalent_empty_mid_and_far_bins_are_deterministic() -> None:
    target_xyz = torch.tensor([[10.0, 0.0, 0.0]])
    pool = _full_candidate_pool()
    first = select_fixed_quota_support_oracle(
        pool,
        target_xyz,
        torch.ones(1),
        distance_chunk_size=32,
    )
    second = select_fixed_quota_support_oracle(
        pool,
        target_xyz + torch.tensor([[1.0, 0.0, 0.0]]),
        torch.ones(1),
        distance_chunk_size=32,
    )

    for label in ("range_30_60m", "range_60_120m"):
        report = first.per_range_support[label]
        assert report["target_count"] == 0
        assert report["ground_truth_used_for_selection"] is False
        assert (
            report["selection_mode"]
            == "deterministic_candidate_index_gt_free_fill"
        )
    torch.testing.assert_close(
        first.selected_candidate_indices[7_500:],
        second.selected_candidate_indices[7_500:],
    )
    torch.testing.assert_close(
        first.selected_candidate_indices[9_500:],
        torch.arange(30_400, 30_900),
    )
