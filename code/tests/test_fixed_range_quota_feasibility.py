from __future__ import annotations

import numpy as np
import pytest

from eval.fixed_range_quota_feasibility import fixed_quota_feasibility


def _xyz_from_ranges(ranges: list[float]) -> np.ndarray:
    xyz = np.zeros((len(ranges), 3), dtype=np.float64)
    xyz[:, 0] = np.asarray(ranges)
    return xyz


def test_near_only_frame_forces_twenty_percent_outliers() -> None:
    report = fixed_quota_feasibility(_xyz_from_ranges([3.0, 27.877]))

    assert report.target_counts_by_stratum == (2, 0, 0)
    assert report.radially_unreachable == (False, True, True)
    expected_lower_bound = (1_700 * 2.123 + 300 * 32.123) / 10_000
    assert report.forced_chamfer_lower_bound_m == pytest.approx(
        expected_lower_bound
    )
    assert report.forced_outlier_count == 2_000
    assert report.forced_outlier_fraction == pytest.approx(0.20)


def test_boundary_support_makes_middle_stratum_reachable() -> None:
    report = fixed_quota_feasibility(_xyz_from_ranges([28.0]))

    assert report.minimum_radial_gaps_m[1] == pytest.approx(2.0)
    assert report.radially_unreachable == (False, False, True)
    assert report.forced_chamfer_lower_bound_m == pytest.approx(1.30)
    assert report.forced_outlier_fraction == pytest.approx(0.03)


def test_far_target_can_make_all_strata_radially_reachable() -> None:
    report = fixed_quota_feasibility(_xyz_from_ranges([29.0, 59.0]))

    assert report.radially_unreachable == (False, False, False)
    assert report.forced_chamfer_lower_bound_m == pytest.approx(0.03)
    assert report.forced_outlier_fraction == 0.0


def test_sparse_middle_support_leaves_almost_no_chamfer_budget() -> None:
    report = fixed_quota_feasibility(_xyz_from_ranges([7.4, 34.5373]))

    assert report.radially_unreachable == (False, False, True)
    assert report.forced_outlier_fraction == pytest.approx(0.03)
    assert report.forced_chamfer_lower_bound_m == pytest.approx(0.763881)


@pytest.mark.parametrize(
    "target",
    [
        np.empty((0, 3)),
        np.zeros((2, 4)),
        np.asarray([[np.nan, 0.0, 0.0]]),
    ],
)
def test_invalid_target_is_rejected(target: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fixed_quota_feasibility(target)


def test_custom_quotas_preserve_strict_lower_bound() -> None:
    report = fixed_quota_feasibility(
        _xyz_from_ranges([10.0]),
        intervals_m=((0.0, 20.0), (20.0, 40.0)),
        quotas=(90, 10),
        outlier_distance_m=1.0,
    )

    assert report.radially_unreachable == (False, True)
    assert report.forced_outlier_fraction == pytest.approx(0.10)
