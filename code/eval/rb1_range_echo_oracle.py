"""GT-aided structural diagnostics for the R-B1 range-echo representation.

This module never reads a Radar Cube.  It projects target geometry onto fixed
azimuth/elevation rays, fits at most K continuous range peaks per ray, and
checks whether the resulting representation can satisfy the frozen exact-count,
range-quota, and Euclidean-spacing contract.

The construction uses ground truth at inference time.  It is therefore an
unattainable diagnostic, not a deployable method or a mathematical upper bound.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


AZIMUTH_RAY_COUNT = 107
ELEVATION_RAY_COUNT = 37
RANGE_BIN_COUNT = 256
TOTAL_RAY_COUNT = AZIMUTH_RAY_COUNT * ELEVATION_RAY_COUNT
DEFAULT_K_VALUES = (4, 6)
EXPORT_COUNT = 10_000
EXPORT_RANGE_QUOTAS = (8_000, 1_700, 300)
RANGE_BOUNDS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
RANGE_LABELS = ("range_0_30m", "range_30_60m", "range_60_120m")
MINIMUM_DISTANCE_M = 0.05
ORACLE_LABEL = "unattainable_gt_aided_structural_diagnostic"
ORACLE_DISCLAIMER = (
    "Uses target geometry to fit and rank peaks; not deployable and not a "
    "strict mathematical upper bound."
)


@dataclass(frozen=True)
class RangeEchoOracleConfig:
    """Frozen structural parameters for one audited K value."""

    k: int
    exact_count: int = EXPORT_COUNT
    range_quotas: tuple[int, int, int] = EXPORT_RANGE_QUOTAS
    minimum_distance_m: float = MINIMUM_DISTANCE_M
    radial_merge_distance_m: float = MINIMUM_DISTANCE_M

    def __post_init__(self) -> None:
        if self.k < 4:
            raise ValueError("R-B1 requires at least four range peaks per ray")
        if self.exact_count <= 0:
            raise ValueError("R-B1 exact count must be positive")
        if len(self.range_quotas) != 3:
            raise ValueError("R-B1 requires three frozen range quotas")
        if any(value < 0 for value in self.range_quotas):
            raise ValueError("R-B1 range quotas must be nonnegative")
        if sum(self.range_quotas) != self.exact_count:
            raise ValueError("R-B1 range quotas must sum to the exact count")
        if self.minimum_distance_m <= 0.0:
            raise ValueError("R-B1 minimum distance must be positive")
        if self.radial_merge_distance_m <= 0.0:
            raise ValueError("R-B1 radial merge distance must be positive")


@dataclass(frozen=True)
class RangeEchoCandidates:
    """One target-aided set of capped continuous peaks."""

    xyz_m: np.ndarray
    continuous_range_m: np.ndarray
    ray_indices_ae: np.ndarray
    range_bin_indices: np.ndarray
    sub_bin_offsets_m: np.ndarray
    sub_bin_lower_bounds_m: np.ndarray
    sub_bin_upper_bounds_m: np.ndarray
    confidence: np.ndarray
    support_weight: np.ndarray
    support_count: np.ndarray
    nearest_target_distance_m: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class ExactRangeEchoSelection:
    """One exact fixed-count selection from an R-B1 candidate set."""

    xyz_m: np.ndarray
    confidence: np.ndarray
    selected_candidate_indices: np.ndarray
    report: dict[str, Any]


class RangeEchoCapacityError(RuntimeError):
    """Raised when no audited deterministic order can fill the frozen quotas."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def array_sha256(array: np.ndarray) -> str:
    """Hash a canonical contiguous array, including dtype and shape."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_axes(
    range_axis_m: np.ndarray,
    azimuth_axis_rad: np.ndarray,
    elevation_axis_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = tuple(
        np.asarray(axis, dtype=np.float64).reshape(-1)
        for axis in (range_axis_m, azimuth_axis_rad, elevation_axis_rad)
    )
    expected = (RANGE_BIN_COUNT, AZIMUTH_RAY_COUNT, ELEVATION_RAY_COUNT)
    actual = tuple(len(axis) for axis in axes)
    if actual != expected:
        raise ValueError(f"R-B1 requires axes {expected}, got {actual}")
    if any(not np.isfinite(axis).all() for axis in axes):
        raise ValueError("R-B1 axes must be finite")
    if any(not np.all(np.diff(axis) > 0.0) for axis in axes):
        raise ValueError("R-B1 axes must be strictly increasing")
    return axes


def _validate_target(target_xyz_confidence: np.ndarray) -> np.ndarray:
    target = np.asarray(target_xyz_confidence, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError("R-B1 target must be a non-empty (N,4) array")
    if not np.isfinite(target).all():
        raise ValueError("R-B1 target must be finite")
    if np.any(target[:, 3] < 0.0):
        raise ValueError("R-B1 target confidence must be nonnegative")
    if np.any(np.linalg.norm(target[:, :3], axis=1) <= 0.0):
        raise ValueError("R-B1 target points must have positive range")
    return target


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    right = np.searchsorted(axis, values, side="left")
    right = np.clip(right, 0, len(axis) - 1)
    left = np.clip(right - 1, 0, len(axis) - 1)
    choose_left = np.abs(values - axis[left]) <= np.abs(values - axis[right])
    return np.where(choose_left, left, right).astype(np.int64)


def _polar_to_cartesian(
    radius: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    cosine = np.cos(elevation)
    return np.column_stack(
        (
            radius * cosine * np.cos(azimuth),
            radius * cosine * np.sin(azimuth),
            radius * np.sin(elevation),
        )
    )


def _range_codes(range_m: np.ndarray) -> np.ndarray:
    value = np.asarray(range_m)
    result = np.full(value.shape, -1, dtype=np.int64)
    for code, (lower, upper) in enumerate(RANGE_BOUNDS_M):
        mask = (value >= lower) & (value < upper)
        result[mask] = code
    return result


def _quota_dict(values: tuple[int, int, int] | list[int]) -> dict[str, int]:
    return {name: int(value) for name, value in zip(RANGE_LABELS, values)}


def _weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    if values.size == 0:
        return 0.0
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = percentile * cumulative[-1]
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _radial_groups(
    ranges: np.ndarray,
    confidence: np.ndarray,
    merge_distance_m: float,
) -> list[dict[str, float | int]]:
    """Merge target samples that cannot occupy separate 5 cm radial peaks."""

    order = np.lexsort((confidence, ranges))
    sorted_range = ranges[order]
    sorted_confidence = confidence[order]
    groups: list[list[int]] = []
    for index, value in enumerate(sorted_range):
        if (
            not groups
            or value - sorted_range[groups[-1][-1]] >= merge_distance_m
        ):
            groups.append([index])
        else:
            groups[-1].append(index)

    result: list[dict[str, float | int]] = []
    for group in groups:
        group_indices = np.asarray(group, dtype=np.int64)
        group_range = sorted_range[group_indices]
        group_confidence = sorted_confidence[group_indices]
        weights = np.maximum(group_confidence, 1e-6)
        result.append(
            {
                "range_m": float(np.average(group_range, weights=weights)),
                "confidence": float(np.average(group_confidence, weights=weights)),
                "support_weight": float(weights.sum()),
                "support_count": int(len(group)),
            }
        )
    return result


def _fit_capped_peaks(
    groups: list[dict[str, float | int]],
    k: int,
) -> list[dict[str, float | int]]:
    """Fit a deterministic weighted 1-D Lloyd heuristic with at most K peaks."""

    if len(groups) <= k:
        return groups
    values = np.asarray([group["range_m"] for group in groups], dtype=np.float64)
    weights = np.asarray(
        [group["support_weight"] for group in groups],
        dtype=np.float64,
    )
    confidence = np.asarray(
        [group["confidence"] for group in groups],
        dtype=np.float64,
    )
    counts = np.asarray(
        [group["support_count"] for group in groups],
        dtype=np.int64,
    )
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    confidence = confidence[order]
    counts = counts[order]

    initial_parts = np.array_split(np.arange(len(values)), k)
    centers = np.asarray(
        [
            np.average(values[part], weights=weights[part])
            for part in initial_parts
        ],
        dtype=np.float64,
    )
    assignments = np.zeros(len(values), dtype=np.int64)
    for _ in range(32):
        distances = np.abs(values[:, None] - centers[None, :])
        updated_assignments = np.argmin(distances, axis=1)
        if len(np.unique(updated_assignments)) != k:
            updated_assignments = np.concatenate(
                [
                    np.full(len(part), index, dtype=np.int64)
                    for index, part in enumerate(initial_parts)
                ]
            )
        updated_centers = np.asarray(
            [
                np.average(
                    values[updated_assignments == index],
                    weights=weights[updated_assignments == index],
                )
                for index in range(k)
            ],
            dtype=np.float64,
        )
        if np.array_equal(assignments, updated_assignments) and np.allclose(
            centers,
            updated_centers,
            rtol=0.0,
            atol=1e-12,
        ):
            assignments = updated_assignments
            centers = updated_centers
            break
        assignments = updated_assignments
        centers = updated_centers

    peaks: list[dict[str, float | int]] = []
    for index in range(k):
        mask = assignments == index
        cluster_weight = weights[mask]
        peaks.append(
            {
                "range_m": float(
                    np.average(values[mask], weights=cluster_weight)
                ),
                "confidence": float(
                    np.average(confidence[mask], weights=cluster_weight)
                ),
                "support_weight": float(cluster_weight.sum()),
                "support_count": int(counts[mask].sum()),
            }
        )
    return sorted(peaks, key=lambda item: float(item["range_m"]))


def _sub_bin_encoding(
    continuous_range_m: np.ndarray,
    range_axis_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = _nearest_axis_indices(range_axis_m, continuous_range_m)
    center = range_axis_m[indices]
    lower_edges = np.empty_like(center)
    upper_edges = np.empty_like(center)
    for row, index in enumerate(indices):
        lower_edges[row] = (
            0.5 * (range_axis_m[index - 1] + range_axis_m[index])
            if index > 0
            else range_axis_m[0]
            - 0.5 * (range_axis_m[1] - range_axis_m[0])
        )
        upper_edges[row] = (
            0.5 * (range_axis_m[index] + range_axis_m[index + 1])
            if index + 1 < len(range_axis_m)
            else range_axis_m[-1]
            + 0.5 * (range_axis_m[-1] - range_axis_m[-2])
        )
    offsets = continuous_range_m - center
    lower_bounds = lower_edges - center
    upper_bounds = upper_edges - center
    if np.any(offsets < lower_bounds - 1e-10) or np.any(
        offsets > upper_bounds + 1e-10
    ):
        raise AssertionError("R-B1 sub-bin offset exceeded its Voronoi bound")
    return indices, offsets, lower_bounds, upper_bounds


def _echo_distribution(echo_counts: np.ndarray, k: int) -> dict[str, Any]:
    occupied = echo_counts[echo_counts > 0]
    histogram = Counter(int(value) for value in echo_counts)
    return {
        "all_rays_including_zero": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
        "occupied_ray_count": int(occupied.size),
        "empty_ray_count": int((echo_counts == 0).sum()),
        "maximum_echo_count": int(occupied.max()) if occupied.size else 0,
        "median_echo_count_on_occupied_rays": (
            float(np.median(occupied)) if occupied.size else 0.0
        ),
        "p95_echo_count_on_occupied_rays": (
            float(np.percentile(occupied, 95.0)) if occupied.size else 0.0
        ),
        "rays_exceeding_k": int((echo_counts > k).sum()),
    }


def build_gt_aided_range_echo_candidates(
    target_xyz_confidence: np.ndarray,
    *,
    range_axis_m: np.ndarray,
    azimuth_axis_rad: np.ndarray,
    elevation_axis_rad: np.ndarray,
    config: RangeEchoOracleConfig,
) -> RangeEchoCandidates:
    """Construct the best-found deterministic GT-aided R-B1 peak set."""

    target = _validate_target(target_xyz_confidence)
    range_axis, azimuth_axis, elevation_axis = _validate_axes(
        range_axis_m,
        azimuth_axis_rad,
        elevation_axis_rad,
    )
    xyz = target[:, :3]
    target_range = np.linalg.norm(xyz, axis=1)
    target_azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    target_elevation = np.arcsin(
        np.divide(
            xyz[:, 2],
            target_range,
            out=np.zeros_like(target_range),
            where=target_range > 0.0,
        )
    )
    azimuth_indices = _nearest_axis_indices(azimuth_axis, target_azimuth)
    elevation_indices = _nearest_axis_indices(
        elevation_axis,
        target_elevation,
    )
    ray_linear = (
        azimuth_indices * ELEVATION_RAY_COUNT + elevation_indices
    )

    target_order = np.lexsort(
        (
            target[:, 3],
            xyz[:, 2],
            xyz[:, 1],
            xyz[:, 0],
            target_range,
            ray_linear,
        )
    )
    ray_linear = ray_linear[target_order]
    target_range = target_range[target_order]
    target_confidence = target[target_order, 3]

    peak_rows: list[tuple[int, int, float, float, float, int]] = []
    raw_echo_counts = np.zeros(TOTAL_RAY_COUNT, dtype=np.int64)
    raw_echo_ranges: list[float] = []
    raw_echo_weights: list[float] = []
    represented_echo_ranges: list[float] = []
    represented_echo_weights: list[float] = []
    start = 0
    while start < len(ray_linear):
        stop = start + 1
        while stop < len(ray_linear) and ray_linear[stop] == ray_linear[start]:
            stop += 1
        ray = int(ray_linear[start])
        groups = _radial_groups(
            target_range[start:stop],
            target_confidence[start:stop],
            config.radial_merge_distance_m,
        )
        raw_echo_counts[ray] = len(groups)
        peaks = _fit_capped_peaks(groups, config.k)
        azimuth_index = ray // ELEVATION_RAY_COUNT
        elevation_index = ray % ELEVATION_RAY_COUNT
        for group in groups:
            raw_echo_ranges.append(float(group["range_m"]))
            raw_echo_weights.append(float(group["support_weight"]))
        for peak in peaks:
            peak_range = float(peak["range_m"])
            represented_echo_ranges.append(peak_range)
            represented_echo_weights.append(float(peak["support_weight"]))
            peak_rows.append(
                (
                    azimuth_index,
                    elevation_index,
                    peak_range,
                    float(peak["confidence"]),
                    float(peak["support_weight"]),
                    int(peak["support_count"]),
                )
            )
        start = stop

    if not peak_rows:
        raise ValueError("R-B1 target produced no representable peaks")
    peak_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    ray_indices = np.asarray(
        [[row[0], row[1]] for row in peak_rows],
        dtype=np.int64,
    )
    continuous_range = np.asarray(
        [row[2] for row in peak_rows],
        dtype=np.float64,
    )
    confidence = np.asarray(
        [row[3] for row in peak_rows],
        dtype=np.float64,
    )
    support_weight = np.asarray(
        [row[4] for row in peak_rows],
        dtype=np.float64,
    )
    support_count = np.asarray(
        [row[5] for row in peak_rows],
        dtype=np.int64,
    )
    candidate_xyz = _polar_to_cartesian(
        continuous_range,
        azimuth_axis[ray_indices[:, 0]],
        elevation_axis[ray_indices[:, 1]],
    )
    (
        range_bin_indices,
        sub_bin_offsets,
        sub_bin_lower,
        sub_bin_upper,
    ) = _sub_bin_encoding(continuous_range, range_axis)
    nearest_target = cKDTree(xyz).query(candidate_xyz, k=1, workers=1)[0]

    raw_ranges = np.asarray(raw_echo_ranges, dtype=np.float64)
    raw_weights = np.asarray(raw_echo_weights, dtype=np.float64)
    represented_ranges = np.asarray(
        represented_echo_ranges,
        dtype=np.float64,
    )
    represented_weights = np.asarray(
        represented_echo_weights,
        dtype=np.float64,
    )
    radial_residuals: list[np.ndarray] = []
    radial_weights: list[np.ndarray] = []
    start = 0
    peak_start = 0
    occupied_rays = np.flatnonzero(raw_echo_counts)
    for ray in occupied_rays:
        raw_count = int(raw_echo_counts[ray])
        peak_count = min(raw_count, config.k)
        ray_raw = raw_ranges[start : start + raw_count]
        ray_weight = raw_weights[start : start + raw_count]
        ray_peaks = represented_ranges[peak_start : peak_start + peak_count]
        radial_residuals.append(
            np.min(np.abs(ray_raw[:, None] - ray_peaks[None, :]), axis=1)
        )
        radial_weights.append(ray_weight)
        start += raw_count
        peak_start += peak_count
    residual = np.concatenate(radial_residuals)
    residual_weight = np.concatenate(radial_weights)
    candidate_codes = _range_codes(continuous_range)
    candidate_range_counts = [
        int((candidate_codes == code).sum()) for code in range(3)
    ]
    outside_count = int((candidate_codes < 0).sum())
    weighted_mae = float(
        np.average(residual, weights=np.maximum(residual_weight, 1e-6))
    )
    report = {
        "oracle_label": ORACLE_LABEL,
        "oracle_disclaimer": ORACLE_DISCLAIMER,
        "deployable_method": False,
        "strict_mathematical_upper_bound": False,
        "representation": {
            "azimuth_ray_count": AZIMUTH_RAY_COUNT,
            "elevation_ray_count": ELEVATION_RAY_COUNT,
            "total_ray_count": TOTAL_RAY_COUNT,
            "maximum_peaks_per_ray": config.k,
            "continuous_peak_parameterization": (
                "range_bin_plus_bounded_sub_bin_offset_and_confidence"
            ),
            "maximum_raw_slot_capacity": TOTAL_RAY_COUNT * config.k,
        },
        "target_count": int(target.shape[0]),
        "occupied_ray_count": int(len(occupied_rays)),
        "echo_count_distribution": _echo_distribution(
            raw_echo_counts,
            config.k,
        ),
        "k_truncation": {
            "raw_gt_aided_echo_group_count": int(raw_echo_counts.sum()),
            "represented_peak_count": int(len(peak_rows)),
            "slot_count_fraction": float(
                len(peak_rows) / max(int(raw_echo_counts.sum()), 1)
            ),
            "rays_truncated": int((raw_echo_counts > config.k).sum()),
            "weighted_radial_mae_m": weighted_mae,
            "weighted_radial_p95_m": _weighted_percentile(
                residual,
                np.maximum(residual_weight, 1e-6),
                0.95,
            ),
            "raw_echo_fraction_within_0p05m": float(
                np.average(
                    residual <= 0.05,
                    weights=np.maximum(residual_weight, 1e-6),
                )
            ),
            "raw_echo_fraction_within_0p5m": float(
                np.average(
                    residual <= 0.5,
                    weights=np.maximum(residual_weight, 1e-6),
                )
            ),
        },
        "candidate_capacity": {
            "total": int(len(peak_rows)),
            "by_range": _quota_dict(candidate_range_counts),
            "outside_frozen_range_count": outside_count,
            "raw_quota_capacity_satisfied": all(
                actual >= required
                for actual, required in zip(
                    candidate_range_counts,
                    config.range_quotas,
                )
            ),
        },
        "sub_bin_offset": {
            "all_offsets_within_bounds": True,
            "minimum_offset_m": float(sub_bin_offsets.min()),
            "maximum_offset_m": float(sub_bin_offsets.max()),
        },
        "hashes": {
            "xyz_sha256": array_sha256(candidate_xyz.astype(np.float32)),
            "ray_indices_sha256": array_sha256(ray_indices),
            "range_bin_indices_sha256": array_sha256(range_bin_indices),
            "sub_bin_offsets_sha256": array_sha256(
                sub_bin_offsets.astype(np.float32)
            ),
        },
    }
    return RangeEchoCandidates(
        xyz_m=candidate_xyz.astype(np.float32),
        continuous_range_m=continuous_range.astype(np.float32),
        ray_indices_ae=ray_indices,
        range_bin_indices=range_bin_indices,
        sub_bin_offsets_m=sub_bin_offsets.astype(np.float32),
        sub_bin_lower_bounds_m=sub_bin_lower.astype(np.float32),
        sub_bin_upper_bounds_m=sub_bin_upper.astype(np.float32),
        confidence=confidence.astype(np.float32),
        support_weight=represented_weights.astype(np.float32),
        support_count=support_count,
        nearest_target_distance_m=nearest_target.astype(np.float32),
        report=report,
    )


def _stable_candidate_order(
    candidates: RangeEchoCandidates,
    indices: np.ndarray,
) -> np.ndarray:
    return indices[
        np.lexsort(
            (
                candidates.continuous_range_m[indices],
                candidates.ray_indices_ae[indices, 1],
                candidates.ray_indices_ae[indices, 0],
                -candidates.support_weight[indices],
                -candidates.confidence[indices],
                candidates.nearest_target_distance_m[indices],
            )
        )
    ]


class _SpatialExclusion:
    def __init__(self, minimum_distance_m: float) -> None:
        self.minimum_distance_m = minimum_distance_m
        self.minimum_squared = minimum_distance_m**2
        self.cells: dict[tuple[int, int, int], list[np.ndarray]] = {}

    def _cell(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(
            math.floor(float(value) / self.minimum_distance_m)
            for value in point
        )

    def nearest_conflict_distance(self, point: np.ndarray) -> float | None:
        cell = self._cell(point)
        nearest_squared = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in self.cells.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz),
                        (),
                    ):
                        distance_squared = float(np.sum((point - other) ** 2))
                        nearest_squared = min(nearest_squared, distance_squared)
        if nearest_squared < self.minimum_squared - 1e-12:
            return math.sqrt(nearest_squared)
        return None

    def add(self, point: np.ndarray) -> None:
        self.cells.setdefault(self._cell(point), []).append(point)


def _attempt_selection(
    candidates: RangeEchoCandidates,
    config: RangeEchoOracleConfig,
    range_order: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    codes = _range_codes(candidates.continuous_range_m)
    selected: list[int] = []
    selected_by_range = [0, 0, 0]
    rejected_distance = [0, 0, 0]
    rejected_values: list[list[float]] = [[], [], []]
    considered = [0, 0, 0]
    exclusion = _SpatialExclusion(config.minimum_distance_m)
    for code in range_order:
        indices = np.flatnonzero(codes == code)
        ordered = _stable_candidate_order(candidates, indices)
        quota = config.range_quotas[code]
        for candidate_index in ordered:
            if selected_by_range[code] >= quota:
                break
            considered[code] += 1
            point = candidates.xyz_m[candidate_index].astype(np.float64)
            conflict_distance = exclusion.nearest_conflict_distance(point)
            if conflict_distance is not None:
                rejected_distance[code] += 1
                rejected_values[code].append(conflict_distance)
                continue
            exclusion.add(point)
            selected.append(int(candidate_index))
            selected_by_range[code] += 1

    raw_counts = [int((codes == code).sum()) for code in range(3)]
    report = {
        "range_order": [RANGE_LABELS[code] for code in range_order],
        "selected_by_range": _quota_dict(selected_by_range),
        "raw_candidate_by_range": _quota_dict(raw_counts),
        "considered_by_range": _quota_dict(considered),
        "rejected_minimum_distance_by_range": _quota_dict(
            rejected_distance
        ),
        "rejected_distance_m_by_range": {
            RANGE_LABELS[code]: {
                "count": len(rejected_values[code]),
                "minimum": (
                    float(min(rejected_values[code]))
                    if rejected_values[code]
                    else None
                ),
                "mean": (
                    float(np.mean(rejected_values[code]))
                    if rejected_values[code]
                    else None
                ),
                "maximum": (
                    float(max(rejected_values[code]))
                    if rejected_values[code]
                    else None
                ),
            }
            for code in range(3)
        },
        "quota_deficit_by_range": _quota_dict(
            [
                max(required - actual, 0)
                for required, actual in zip(
                    config.range_quotas,
                    selected_by_range,
                )
            ]
        ),
        "selected_count": len(selected),
        "exact_count_reached": len(selected) == config.exact_count,
    }
    return np.asarray(selected, dtype=np.int64), report


def select_exact_range_echoes(
    candidates: RangeEchoCandidates,
    *,
    config: RangeEchoOracleConfig,
) -> ExactRangeEchoSelection:
    """Select exact quotas with true 5 cm spacing and no fill fallback."""

    attempts: list[tuple[np.ndarray, dict[str, Any]]] = []
    for range_order in itertools.permutations(range(3)):
        attempts.append(
            _attempt_selection(candidates, config, range_order)
        )
    successful = [
        item for item in attempts if item[1]["exact_count_reached"]
    ]
    if not successful:
        best_indices, best_report = min(
            attempts,
            key=lambda item: (
                config.exact_count - item[1]["selected_count"],
                sum(
                    item[1]["quota_deficit_by_range"].values()
                ),
                item[1]["range_order"],
            ),
        )
        failure_report = {
            "oracle_label": ORACLE_LABEL,
            "hard_capacity_failure": True,
            "copy_padding_jitter_duplicate": False,
            "required_exact_count": config.exact_count,
            "required_range_quotas": _quota_dict(config.range_quotas),
            "minimum_euclidean_distance_m": config.minimum_distance_m,
            "best_failed_attempt": best_report,
            "attempts": [report for _, report in attempts],
            "best_failed_selected_indices_sha256": array_sha256(best_indices),
        }
        raise RangeEchoCapacityError(
            "R-B1 range-echo candidates cannot fill exact quotas at 5 cm",
            failure_report,
        )

    selected_indices, chosen_report = min(
        successful,
        key=lambda item: (
            float(
                candidates.nearest_target_distance_m[item[0]].sum()
            ),
            item[1]["range_order"],
        ),
    )
    selected_xyz = candidates.xyz_m[selected_indices]
    selected_confidence = candidates.confidence[selected_indices]
    nearest_other = cKDTree(selected_xyz).query(
        selected_xyz,
        k=2,
        workers=1,
    )[0][:, 1]
    minimum_observed = float(nearest_other.min())
    if minimum_observed < config.minimum_distance_m - 1e-6:
        raise AssertionError("R-B1 exact export violated the 5 cm constraint")
    report = {
        "oracle_label": ORACLE_LABEL,
        "oracle_disclaimer": ORACLE_DISCLAIMER,
        "deployable_method": False,
        "strict_mathematical_upper_bound": False,
        "method": "gt_ranked_six_order_greedy_true_5cm_range_quota",
        "exact_point_count": int(selected_xyz.shape[0]),
        "required_range_quotas": _quota_dict(config.range_quotas),
        "selected_by_range": chosen_report["selected_by_range"],
        "minimum_euclidean_distance_m": config.minimum_distance_m,
        "observed_minimum_pair_distance_m": minimum_observed,
        "copy_padding_jitter_duplicate": False,
        "chosen_attempt": chosen_report,
        "successful_order_count": len(successful),
        "all_attempts": [report for _, report in attempts],
        "hashes": {
            "xyz_sha256": array_sha256(selected_xyz),
            "selected_candidate_indices_sha256": array_sha256(
                selected_indices
            ),
        },
    }
    return ExactRangeEchoSelection(
        xyz_m=selected_xyz,
        confidence=selected_confidence,
        selected_candidate_indices=selected_indices,
        report=report,
    )


def geometry_endpoints(
    prediction_xyz_m: np.ndarray,
    target_xyz_confidence: np.ndarray,
) -> dict[str, Any]:
    """Compute CPU KD-tree geometry metrics with frozen range semantics."""

    prediction = np.asarray(prediction_xyz_m, dtype=np.float64)
    target = _validate_target(target_xyz_confidence)
    if prediction.ndim != 2 or prediction.shape[1] != 3 or len(prediction) == 0:
        raise ValueError("R-B1 prediction must be a non-empty (N,3) array")
    if not np.isfinite(prediction).all():
        raise ValueError("R-B1 prediction must be finite")
    target_xyz = target[:, :3]
    target_weight = np.maximum(target[:, 3], 0.0)
    if float(target_weight.sum()) <= 0.0:
        target_weight = np.ones_like(target_weight)
    prediction_to_target = cKDTree(target_xyz).query(
        prediction,
        k=1,
        workers=1,
    )[0]
    target_to_prediction = cKDTree(prediction).query(
        target_xyz,
        k=1,
        workers=1,
    )[0]
    weighted_completeness = float(
        np.average(target_to_prediction, weights=target_weight)
    )
    report: dict[str, Any] = {
        "chamfer_m": float(
            prediction_to_target.mean() + weighted_completeness
        ),
        "precision_mean_distance_m": float(prediction_to_target.mean()),
        "completeness_mean_distance_m": weighted_completeness,
        "outlier_fraction_2m": float(
            np.mean(prediction_to_target > 2.0)
        ),
        "prediction_count": int(len(prediction)),
        "target_count": int(len(target)),
        "target_effective_count": float(target_weight.sum()),
    }
    for threshold in (0.5, 1.0, 2.0):
        precision = float(np.mean(prediction_to_target <= threshold))
        recall = float(
            np.average(
                target_to_prediction <= threshold,
                weights=target_weight,
            )
        )
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        suffix = str(threshold).replace(".", "p")
        report[f"precision_{suffix}m"] = precision
        report[f"recall_{suffix}m"] = recall
        report[f"fscore_{suffix}m"] = fscore

    prediction_range = np.linalg.norm(prediction, axis=1)
    target_range = np.linalg.norm(target_xyz, axis=1)
    for label, (lower, upper) in zip(RANGE_LABELS, RANGE_BOUNDS_M):
        prediction_mask = (
            (prediction_range >= lower) & (prediction_range < upper)
        )
        target_mask = (target_range >= lower) & (target_range < upper)
        if not target_mask.any():
            continue
        bin_weight = target_weight[target_mask]
        bin_target_distance = target_to_prediction[target_mask]
        report[f"{label}_target_count"] = int(target_mask.sum())
        report[f"{label}_completeness_mean_distance_m"] = float(
            np.average(bin_target_distance, weights=bin_weight)
        )
        recall = float(
            np.average(
                bin_target_distance <= 1.0,
                weights=bin_weight,
            )
        )
        if prediction_mask.any():
            bin_prediction_distance = prediction_to_target[prediction_mask]
            precision = float(np.mean(bin_prediction_distance <= 1.0))
            report[f"{label}_precision_mean_distance_m"] = float(
                bin_prediction_distance.mean()
            )
        else:
            precision = 0.0
        report[f"{label}_fscore_1m"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
    report["far_target_frame"] = bool(
        ((target_range >= 60.0) & (target_range < 120.0)).any()
    )
    return report


def aggregate_numeric_reports(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate scalar frame metrics while preserving missing-bin semantics."""

    if not reports:
        raise ValueError("R-B1 cannot aggregate an empty report list")
    keys = sorted(
        {
            key
            for report in reports
            for key, value in report.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not key.endswith("_count")
        }
    )
    result: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = np.asarray(
            [
                float(report[key])
                for report in reports
                if key in report
            ],
            dtype=np.float64,
        )
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "sample_count": int(len(values)),
        }
    return result
