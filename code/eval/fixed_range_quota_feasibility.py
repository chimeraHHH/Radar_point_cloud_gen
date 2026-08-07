"""Strict radial lower bounds for fixed per-frame range quotas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RANGE_INTERVALS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
OUTPUT_QUOTAS = (8_000, 1_700, 300)
OUTLIER_DISTANCE_M = 2.0


@dataclass(frozen=True)
class FixedQuotaFeasibility:
    target_count: int
    target_range_min_m: float
    target_range_max_m: float
    target_counts_by_stratum: tuple[int, int, int]
    minimum_radial_gaps_m: tuple[float, float, float]
    radially_unreachable: tuple[bool, bool, bool]
    forced_outlier_count: int
    forced_outlier_fraction: float


def _minimum_distance_to_half_open_interval(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    """Return the infimum radial gap from values to ``[lower, upper)``."""

    below = np.maximum(lower - values, 0.0)
    above = np.maximum(values - upper, 0.0)
    gaps = np.maximum(below, above)
    return float(gaps.min())


def fixed_quota_feasibility(
    target_xyz_m: np.ndarray,
    *,
    intervals_m: tuple[tuple[float, float], ...] = RANGE_INTERVALS_M,
    quotas: tuple[int, ...] = OUTPUT_QUOTAS,
    outlier_distance_m: float = OUTLIER_DISTANCE_M,
) -> FixedQuotaFeasibility:
    target = np.asarray(target_xyz_m, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] == 0:
        raise ValueError("fixed-quota target must have shape [N,3] with N>0")
    if not np.isfinite(target).all():
        raise ValueError("fixed-quota target must be finite")
    if len(intervals_m) != len(quotas) or not intervals_m:
        raise ValueError("fixed-quota intervals and quotas must align")
    if any(quota < 0 for quota in quotas) or sum(quotas) <= 0:
        raise ValueError("fixed-quota output quotas must be nonnegative")
    if outlier_distance_m <= 0.0:
        raise ValueError("fixed-quota outlier distance must be positive")
    for lower, upper in intervals_m:
        if lower < 0.0 or upper <= lower:
            raise ValueError("fixed-quota intervals must be positive and ordered")

    ranges = np.linalg.norm(target, axis=1)
    counts = tuple(
        int(((ranges >= lower) & (ranges < upper)).sum())
        for lower, upper in intervals_m
    )
    gaps = tuple(
        _minimum_distance_to_half_open_interval(ranges, lower, upper)
        for lower, upper in intervals_m
    )
    unreachable = tuple(gap > outlier_distance_m for gap in gaps)
    forced_count = int(
        sum(quota for quota, blocked in zip(quotas, unreachable) if blocked)
    )
    total = int(sum(quotas))
    return FixedQuotaFeasibility(
        target_count=int(target.shape[0]),
        target_range_min_m=float(ranges.min()),
        target_range_max_m=float(ranges.max()),
        target_counts_by_stratum=counts,
        minimum_radial_gaps_m=gaps,
        radially_unreachable=unreachable,
        forced_outlier_count=forced_count,
        forced_outlier_fraction=float(forced_count / total),
    )
