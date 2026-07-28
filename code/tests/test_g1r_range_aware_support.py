import torch

from eval.g1f_candidate_support import PROPOSAL_COUNT, RANGE_BINS_M
from eval.g1r_range_aware_support import (
    CANDIDATE_PARENT_QUOTAS,
    EXPORT_QUOTAS,
    ExpandedCandidatePool,
    calibrated_integrated_energy,
    fit_range_energy_calibration,
    fixed_quota_oracle_weights,
    physical_angular_suppression_mask,
    range_aware_proposal_indices,
    select_fixed_quota_support_oracle,
    stable_score_proposal_indices,
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
    range_m = torch.tensor([10.0, 20.0, 40.0, 50.0, 80.0, 100.0])
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


def test_fixed_weight_rescaling_and_reused_g1f_oracle_lock_export_quotas() -> None:
    target_xyz = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
        ]
    )
    target_weight = torch.tensor([1.0, 3.0, 2.0, 4.0])
    selection_weight = fixed_quota_oracle_weights(
        target_xyz,
        target_weight,
    )

    radius = torch.linalg.vector_norm(target_xyz, dim=1)
    masses = tuple(
        float(
            selection_weight[
                (radius >= lower) & (radius < upper)
            ].sum()
        )
        for _, lower, upper in RANGE_BINS_M
    )
    assert masses == tuple(float(value) for value in EXPORT_QUOTAS)

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

