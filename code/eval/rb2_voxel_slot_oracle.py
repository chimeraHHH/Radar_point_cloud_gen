"""GT-aided structural oracle for the R-B2 Cartesian voxel-slot route.

The oracle is deliberately unattainable by a deployable model: ground truth
chooses active support voxels, bounded slot offsets, and the final ranking. It
only tests whether the fixed Cartesian lattice and per-voxel slot capacity can
represent an exact, separated point set under the frozen range quotas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


PROTOCOL = "rb2_cartesian_voxel_slot_gt_aided_oracle_v1"
EXPORT_COUNT = 10_000
MINIMUM_DISTANCE_M = 0.05
RANGE_STRATA_M = (
    ("range_0_30m", 0.0, 30.0),
    ("range_30_60m", 30.0, 60.0),
    ("range_60_120m", 60.0, 120.0),
)
OUTPUT_RANGE_QUOTAS = (8_000, 1_700, 300)


class VoxelSlotCapacityError(RuntimeError):
    """Raised when a fixed lattice cannot satisfy the frozen export contract."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class KRadarFOV:
    """Frozen K-Radar spatial support without loading a Cube or axis file."""

    range_min_m: float = 0.0
    range_max_m: float = 118.037109
    azimuth_min_deg: float = -53.0
    azimuth_max_deg: float = 53.0
    elevation_min_deg: float = -18.0
    elevation_max_deg: float = 18.0

    def metadata(self) -> dict[str, float]:
        return {
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "azimuth_min_deg": self.azimuth_min_deg,
            "azimuth_max_deg": self.azimuth_max_deg,
            "elevation_min_deg": self.elevation_min_deg,
            "elevation_max_deg": self.elevation_max_deg,
        }


@dataclass(frozen=True)
class VoxelSlotConfig:
    """One fixed Cartesian lattice and bounded slot-capacity hypothesis."""

    name: str
    voxel_size_xyz_m: tuple[float, float, float]
    slots_per_voxel: int
    template_resolution: int
    minimum_distance_m: float = MINIMUM_DISTANCE_M

    def validate(self) -> None:
        if not self.name:
            raise ValueError("R-B2 config name must be non-empty")
        if len(self.voxel_size_xyz_m) != 3 or any(
            value <= 0.0 for value in self.voxel_size_xyz_m
        ):
            raise ValueError("R-B2 voxel sizes must be three positive values")
        if self.slots_per_voxel <= 0:
            raise ValueError("R-B2 slots per voxel must be positive")
        if self.template_resolution < 2:
            raise ValueError("R-B2 slot template resolution must be at least two")
        if self.minimum_distance_m <= 0.0:
            raise ValueError("R-B2 minimum distance must be positive")
        usable = min(self.voxel_size_xyz_m) - 1.02 * self.minimum_distance_m
        if usable <= 0.0:
            raise ValueError("R-B2 voxel is too small for the boundary margin")
        if self.template_resolution**3 < self.slots_per_voxel:
            raise ValueError("R-B2 slot template has fewer points than slots")

    def metadata(self) -> dict[str, Any]:
        self.validate()
        volume = float(np.prod(np.asarray(self.voxel_size_xyz_m)))
        return {
            "name": self.name,
            "voxel_size_xyz_m": list(self.voxel_size_xyz_m),
            "voxel_volume_m3": volume,
            "slots_per_voxel": self.slots_per_voxel,
            "slot_density_per_m3": self.slots_per_voxel / volume,
            "template_resolution": self.template_resolution,
            "template_capacity": self.template_resolution**3,
            "minimum_distance_m": self.minimum_distance_m,
            "slot_boundary_margin_m": _boundary_margin(self),
            "rationale": (
                "fixed Cartesian support with bounded mutually exclusive slots; "
                "cell identity cannot move or collapse into a free center"
            ),
        }


DEFAULT_CONFIGS = (
    VoxelSlotConfig(
        name="cartesian_0p40m_s4",
        voxel_size_xyz_m=(0.40, 0.40, 0.40),
        slots_per_voxel=4,
        template_resolution=4,
    ),
    VoxelSlotConfig(
        name="cartesian_0p60m_s8",
        voxel_size_xyz_m=(0.60, 0.60, 0.60),
        slots_per_voxel=8,
        template_resolution=5,
    ),
)


@dataclass(frozen=True)
class CartesianLattice:
    """Axis-aligned lattice enclosing the frozen spherical radar FOV."""

    origin_xyz_m: np.ndarray
    upper_xyz_m: np.ndarray
    voxel_size_xyz_m: np.ndarray
    shape_xyz: tuple[int, int, int]
    fov: KRadarFOV

    def point_indices(self, xyz_m: np.ndarray) -> np.ndarray:
        points = _validate_xyz(xyz_m)
        if not bool(np.all(points_in_fov(points, self.fov))):
            raise ValueError("R-B2 target contains a point outside the K-Radar FOV")
        raw = np.floor(
            (points - self.origin_xyz_m[None]) / self.voxel_size_xyz_m[None]
        ).astype(np.int64)
        maximum = np.asarray(self.shape_xyz, dtype=np.int64) - 1
        return np.minimum(np.maximum(raw, 0), maximum)

    def valid_index(self, index_xyz: tuple[int, int, int]) -> bool:
        return all(
            0 <= int(index) < int(size)
            for index, size in zip(index_xyz, self.shape_xyz, strict=True)
        )

    def cell_bounds(
        self,
        index_xyz: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.valid_index(index_xyz):
            raise ValueError(f"R-B2 voxel index is outside the lattice: {index_xyz}")
        index = np.asarray(index_xyz, dtype=np.float64)
        lower = self.origin_xyz_m + index * self.voxel_size_xyz_m
        upper = np.minimum(lower + self.voxel_size_xyz_m, self.upper_xyz_m)
        return lower, upper

    def linear_id(self, index_xyz: tuple[int, int, int]) -> int:
        if not self.valid_index(index_xyz):
            raise ValueError(f"R-B2 voxel index is outside the lattice: {index_xyz}")
        ix, iy, iz = (int(value) for value in index_xyz)
        _, ny, nz = self.shape_xyz
        return (ix * ny + iy) * nz + iz

    def metadata(self) -> dict[str, Any]:
        return {
            "origin_xyz_m": self.origin_xyz_m.tolist(),
            "upper_xyz_m": self.upper_xyz_m.tolist(),
            "voxel_size_xyz_m": self.voxel_size_xyz_m.tolist(),
            "shape_xyz": list(self.shape_xyz),
            "total_lattice_voxel_count": int(np.prod(self.shape_xyz)),
            "support": "fixed_cartesian_lattice_intersected_with_kradar_fov",
        }


@dataclass(frozen=True)
class SlotPlacement:
    xyz_m: np.ndarray
    nearest_target_distance_m: np.ndarray
    target_derived: np.ndarray


@dataclass(frozen=True)
class VoxelSlotOracleResult:
    selected_xyz_m: np.ndarray
    selected_candidate_ids: np.ndarray
    report: dict[str, Any]


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def make_lattice(
    config: VoxelSlotConfig,
    fov: KRadarFOV = KRadarFOV(),
) -> CartesianLattice:
    config.validate()
    lateral = fov.range_max_m * math.sin(math.radians(fov.azimuth_max_deg))
    vertical = fov.range_max_m * math.sin(math.radians(fov.elevation_max_deg))
    origin = np.asarray(
        [0.0, -lateral, -vertical],
        dtype=np.float64,
    )
    upper = np.asarray(
        [fov.range_max_m, lateral, vertical],
        dtype=np.float64,
    )
    voxel_size = np.asarray(config.voxel_size_xyz_m, dtype=np.float64)
    shape = tuple(int(value) for value in np.ceil((upper - origin) / voxel_size))
    return CartesianLattice(
        origin_xyz_m=origin,
        upper_xyz_m=upper,
        voxel_size_xyz_m=voxel_size,
        shape_xyz=shape,
        fov=fov,
    )


def points_in_fov(
    xyz_m: np.ndarray,
    fov: KRadarFOV = KRadarFOV(),
    tolerance: float = 1e-8,
) -> np.ndarray:
    points = _validate_xyz(xyz_m)
    radius = np.linalg.norm(points, axis=1)
    horizontal = np.linalg.norm(points[:, :2], axis=1)
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    elevation = np.degrees(np.arctan2(points[:, 2], horizontal))
    return (
        (radius >= fov.range_min_m - tolerance)
        & (radius <= fov.range_max_m + tolerance)
        & (azimuth >= fov.azimuth_min_deg - tolerance)
        & (azimuth <= fov.azimuth_max_deg + tolerance)
        & (elevation >= fov.elevation_min_deg - tolerance)
        & (elevation <= fov.elevation_max_deg + tolerance)
    )


def range_stratum_codes(xyz_m: np.ndarray) -> np.ndarray:
    points = _validate_xyz(xyz_m)
    radius = np.linalg.norm(points, axis=1)
    codes = np.full(points.shape[0], -1, dtype=np.int8)
    for index, (_, lower, upper) in enumerate(RANGE_STRATA_M):
        codes[(radius >= lower) & (radius < upper)] = index
    return codes


def construct_gt_aided_slots(
    lattice: CartesianLattice,
    index_xyz: tuple[int, int, int],
    target_xyz_m: np.ndarray,
    config: VoxelSlotConfig,
    *,
    local_target_xyz_m: np.ndarray | None = None,
    required_stratum: int | None = None,
    target_tree: cKDTree | None = None,
) -> SlotPlacement:
    """Place exactly S bounded slots using an explicitly GT-aided heuristic."""

    config.validate()
    target = _validate_xyz(target_xyz_m)
    if target_tree is None:
        target_tree = cKDTree(target)
    lower, upper = lattice.cell_bounds(index_xyz)
    margin = _boundary_margin(config)
    interior_lower = lower + margin
    interior_upper = upper - margin
    if bool(np.any(interior_upper <= interior_lower)):
        raise VoxelSlotCapacityError(
            "R-B2 boundary margin leaves no slot interior",
            {"voxel_index_xyz": list(index_xyz)},
        )

    axes = [
        np.linspace(
            interior_lower[axis],
            interior_upper[axis],
            config.template_resolution,
            dtype=np.float64,
        )
        for axis in range(3)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    template = np.stack(mesh, axis=-1).reshape(-1, 3)
    template_source = np.zeros(template.shape[0], dtype=bool)

    local = (
        np.empty((0, 3), dtype=np.float64)
        if local_target_xyz_m is None
        else _validate_xyz(local_target_xyz_m)
    )
    if local.shape[0]:
        clipped = np.minimum(
            np.maximum(local, interior_lower[None]),
            interior_upper[None],
        )
        candidates = np.concatenate((clipped, template), axis=0)
        target_source = np.concatenate(
            (np.ones(clipped.shape[0], dtype=bool), template_source),
            axis=0,
        )
    else:
        candidates = template
        target_source = template_source

    valid = points_in_fov(candidates, lattice.fov)
    if required_stratum is not None:
        if not 0 <= required_stratum < len(RANGE_STRATA_M):
            raise ValueError("R-B2 required stratum index is invalid")
        valid &= range_stratum_codes(candidates) == required_stratum
    candidates = candidates[valid]
    target_source = target_source[valid]
    if candidates.shape[0] < config.slots_per_voxel:
        raise VoxelSlotCapacityError(
            "R-B2 voxel has fewer valid bounded positions than fixed slots",
            {
                "voxel_index_xyz": list(index_xyz),
                "valid_position_count": int(candidates.shape[0]),
                "required_slot_count": config.slots_per_voxel,
                "required_stratum": required_stratum,
            },
        )

    unique: dict[tuple[float, float, float], tuple[np.ndarray, bool]] = {}
    for point, source in zip(candidates, target_source, strict=True):
        key = tuple(float(value) for value in np.round(point, decimals=10))
        previous = unique.get(key)
        if previous is None or (bool(source) and not previous[1]):
            unique[key] = (point, bool(source))
    candidates = np.asarray([item[0] for item in unique.values()], dtype=np.float64)
    target_source = np.asarray([item[1] for item in unique.values()], dtype=bool)
    distances = np.asarray(
        target_tree.query(candidates, k=1, workers=1)[0],
        dtype=np.float64,
    )
    order = np.lexsort(
        (
            candidates[:, 2],
            candidates[:, 1],
            candidates[:, 0],
            (~target_source).astype(np.int8),
            distances,
        )
    )
    selected_rows: list[int] = []
    minimum_squared = config.minimum_distance_m**2
    for row in order:
        point = candidates[int(row)]
        if selected_rows:
            chosen = candidates[np.asarray(selected_rows, dtype=np.int64)]
            if bool(
                np.any(
                    np.sum((chosen - point[None]) ** 2, axis=1)
                    < minimum_squared - 1e-12
                )
            ):
                continue
        selected_rows.append(int(row))
        if len(selected_rows) == config.slots_per_voxel:
            break
    if len(selected_rows) != config.slots_per_voxel:
        raise VoxelSlotCapacityError(
            "R-B2 voxel cannot fit its fixed mutually exclusive slots",
            {
                "voxel_index_xyz": list(index_xyz),
                "valid_position_count": int(candidates.shape[0]),
                "selected_slot_count": len(selected_rows),
                "required_slot_count": config.slots_per_voxel,
                "minimum_distance_m": config.minimum_distance_m,
                "required_stratum": required_stratum,
            },
        )

    rows = np.asarray(selected_rows, dtype=np.int64)
    xyz = candidates[rows]
    _validate_slot_placement(lattice, index_xyz, xyz, config)
    return SlotPlacement(
        xyz_m=xyz,
        nearest_target_distance_m=distances[rows],
        target_derived=target_source[rows],
    )


def run_gt_aided_voxel_slot_oracle(
    target_xyz_m: np.ndarray,
    target_weight: np.ndarray,
    config: VoxelSlotConfig,
    *,
    fov: KRadarFOV = KRadarFOV(),
    output_quotas: tuple[int, int, int] = OUTPUT_RANGE_QUOTAS,
) -> VoxelSlotOracleResult:
    """Construct and score one deterministic, unattainable GT-aided oracle."""

    config.validate()
    target = _validate_xyz(target_xyz_m)
    weight = np.asarray(target_weight, dtype=np.float64)
    if weight.shape != (target.shape[0],):
        raise ValueError("R-B2 target weight must match target XYZ")
    if not bool(np.all(np.isfinite(weight))) or bool(np.any(weight < 0.0)):
        raise ValueError("R-B2 target weight must be finite and non-negative")
    if float(weight.sum()) <= 0.0:
        raise ValueError("R-B2 target weight must contain positive mass")
    if len(output_quotas) != len(RANGE_STRATA_M) or any(
        int(value) <= 0 for value in output_quotas
    ):
        raise ValueError("R-B2 output quotas must contain three positives")
    if sum(output_quotas) != EXPORT_COUNT:
        raise ValueError("R-B2 output quotas must sum to exact 10k")
    if not bool(np.all(points_in_fov(target, fov))):
        raise ValueError("R-B2 target contains points outside the frozen FOV")

    lattice = make_lattice(config, fov)
    target_tree = cKDTree(target)
    target_indices = lattice.point_indices(target)
    target_codes = range_stratum_codes(target)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for row, index in enumerate(target_indices):
        key = tuple(int(value) for value in index)
        groups.setdefault(key, []).append(row)

    placements: dict[tuple[int, int, int], tuple[SlotPlacement, str]] = {}
    unavailable_target_voxels: list[dict[str, Any]] = []
    for index_xyz in sorted(groups):
        rows = np.asarray(groups[index_xyz], dtype=np.int64)
        try:
            placement = construct_gt_aided_slots(
                lattice,
                index_xyz,
                target,
                config,
                local_target_xyz_m=target[rows],
                target_tree=target_tree,
            )
        except VoxelSlotCapacityError as error:
            unavailable_target_voxels.append(
                {
                    "voxel_index_xyz": list(index_xyz),
                    "target_point_count": int(rows.size),
                    "reason": str(error),
                    "detail": error.report,
                }
            )
            continue
        placements[index_xyz] = (placement, "target_occupied")

    occupied_indices = set(groups)
    for stratum, quota in enumerate(output_quotas):
        current = _placement_capacity_by_range(placements)[stratum]
        if current >= quota:
            continue
        seed_indices = sorted(
            {
                tuple(int(value) for value in target_indices[row])
                for row in np.flatnonzero(target_codes == stratum)
            }
        )
        if not seed_indices:
            _, lower, upper = RANGE_STRATA_M[stratum]
            safe_upper = min(upper, fov.range_max_m)
            radius = (lower + safe_upper) / 2.0
            canonical = np.asarray([[radius, 0.0, 0.0]], dtype=np.float64)
            seed_indices = [
                tuple(int(value) for value in lattice.point_indices(canonical)[0])
            ]
        seen = set(placements) | occupied_indices
        maximum_shell = max(lattice.shape_xyz)
        for shell in range(maximum_shell):
            for offset in _shell_offsets(shell):
                for seed in seed_indices:
                    candidate_index = tuple(
                        int(seed[axis] + offset[axis]) for axis in range(3)
                    )
                    if candidate_index in seen:
                        continue
                    seen.add(candidate_index)
                    if not lattice.valid_index(candidate_index):
                        continue
                    try:
                        placement = construct_gt_aided_slots(
                            lattice,
                            candidate_index,
                            target,
                            config,
                            required_stratum=stratum,
                            target_tree=target_tree,
                        )
                    except VoxelSlotCapacityError:
                        continue
                    placements[candidate_index] = (
                        placement,
                        f"quota_support_{RANGE_STRATA_M[stratum][0]}",
                    )
                    current += config.slots_per_voxel
                    if current >= quota:
                        break
                if current >= quota:
                    break
            if current >= quota:
                break
        if current < quota:
            raise VoxelSlotCapacityError(
                "R-B2 fixed lattice has insufficient range capacity",
                {
                    "config": config.metadata(),
                    "failed_stratum": RANGE_STRATA_M[stratum][0],
                    "candidate_capacity_by_range": _quota_dict(
                        _placement_capacity_by_range(placements)
                    ),
                    "required_output_quotas": _quota_dict(output_quotas),
                    "copy_padding_jitter_duplicate": False,
                },
            )

    candidate = _flatten_placements(placements, lattice, config)
    selected_rows, selection = _select_exact_capacity(
        candidate["xyz_m"],
        candidate["nearest_target_distance_m"],
        candidate["candidate_ids"],
        output_quotas,
        config.minimum_distance_m,
    )
    selected_xyz = candidate["xyz_m"][selected_rows]
    selected_ids = candidate["candidate_ids"][selected_rows]
    geometry = geometry_report(
        selected_xyz,
        target,
        target_weight=weight,
    )
    selected_activation_sources = candidate["activation_source"][selected_rows]
    selected_target_derived = candidate["target_derived"][selected_rows]
    occupied_counts = np.asarray(
        [len(groups[index]) for index in groups],
        dtype=np.int64,
    )
    activated_target_count = sum(
        source == "target_occupied" for _, source in placements.values()
    )
    support_count = len(placements) - activated_target_count
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_label": {
            "label": "unattainable_gt_aided_voxel_slot_heuristic",
            "unattainable_gt_aided_heuristic": True,
            "strict_upper_bound": False,
            "eligible_as_model_result": False,
            "ground_truth_used_for_activation": True,
            "ground_truth_used_for_slot_offsets": True,
            "ground_truth_used_for_candidate_ranking": True,
            "ground_truth_used_for_selection": True,
        },
        "forbidden_operations": {
            "copy": False,
            "padding": False,
            "jitter": False,
            "free_centers": False,
        },
        "fov": fov.metadata(),
        "config": config.metadata(),
        "lattice": lattice.metadata(),
        "target": {
            "point_count": int(target.shape[0]),
            "effective_count": float(weight.sum()),
            "xyz_sha256": array_sha256(target),
            "weight_sha256": array_sha256(weight),
            "has_range_60_120m_target": bool(np.any(target_codes == 2)),
        },
        "voxel_capacity": {
            "target_occupied_voxel_count": len(groups),
            "activated_target_occupied_voxel_count": activated_target_count,
            "unavailable_target_occupied_voxel_count": len(
                unavailable_target_voxels
            ),
            "unavailable_target_occupied_voxels": unavailable_target_voxels,
            "quota_support_voxel_count": support_count,
            "total_activated_voxel_count": len(placements),
            "fixed_candidate_count": int(candidate["xyz_m"].shape[0]),
            "candidate_capacity_by_range": _quota_dict(
                [
                    int(np.sum(candidate["range_codes"] == index))
                    for index in range(len(RANGE_STRATA_M))
                ]
            ),
            "occupied_voxel_target_count_mean": float(occupied_counts.mean()),
            "occupied_voxel_target_count_max": int(occupied_counts.max()),
            "slot_saturated_occupied_voxel_count": int(
                np.sum(occupied_counts >= config.slots_per_voxel)
            ),
            "slot_saturated_occupied_voxel_fraction": float(
                np.mean(occupied_counts >= config.slots_per_voxel)
            ),
            "target_derived_candidate_slot_count": int(
                candidate["target_derived"].sum()
            ),
            "selected_slot_fraction": (
                EXPORT_COUNT / float(candidate["xyz_m"].shape[0])
            ),
        },
        "selection": {
            **selection,
            "selected_from_target_occupied_voxels": int(
                np.sum(selected_activation_sources == "target_occupied")
            ),
            "selected_from_quota_support_voxels": int(
                np.sum(selected_activation_sources != "target_occupied")
            ),
            "selected_target_derived_slot_count": int(
                selected_target_derived.sum()
            ),
        },
        "geometry": geometry,
        "hashes": {
            "candidate_xyz_sha256": array_sha256(candidate["xyz_m"]),
            "candidate_ids_sha256": array_sha256(candidate["candidate_ids"]),
            "selected_xyz_sha256": array_sha256(selected_xyz),
            "selected_candidate_ids_sha256": array_sha256(selected_ids),
        },
        "checks": {
            "fixed_slots_per_activated_voxel": (
                candidate["xyz_m"].shape[0]
                == len(placements) * config.slots_per_voxel
            ),
            "unique_voxel_slot_candidate_ids": (
                np.unique(candidate["candidate_ids"]).size
                == candidate["candidate_ids"].size
            ),
            "all_candidates_inside_fixed_cells": bool(
                candidate["inside_fixed_cell"].all()
            ),
            "all_candidates_inside_kradar_fov": bool(
                points_in_fov(candidate["xyz_m"], fov).all()
            ),
            "exact_10000": selected_xyz.shape == (EXPORT_COUNT, 3),
            "range_quotas_exact": selection["selected_by_range"]
            == _quota_dict(output_quotas),
            "minimum_pair_distance_at_least_5cm": (
                selection["observed_minimum_pair_distance_m"]
                >= config.minimum_distance_m - 1e-9
            ),
            "capacity_sufficient": True,
            "gt_aided_label_explicit": True,
            "no_copy_padding_jitter_or_free_centers": True,
        },
    }
    report["passed"] = all(report["checks"].values())
    return VoxelSlotOracleResult(
        selected_xyz_m=selected_xyz,
        selected_candidate_ids=selected_ids,
        report=report,
    )


def geometry_report(
    prediction_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    *,
    target_weight: np.ndarray,
) -> dict[str, float | int]:
    prediction = _validate_xyz(prediction_xyz_m)
    target = _validate_xyz(target_xyz_m)
    weight = np.asarray(target_weight, dtype=np.float64)
    if weight.shape != (target.shape[0],):
        raise ValueError("R-B2 geometry target weight shape mismatch")
    prediction_to_target = cKDTree(target).query(
        prediction, k=1, workers=1
    )[0]
    target_to_prediction = cKDTree(prediction).query(
        target, k=1, workers=1
    )[0]
    weight_sum = float(weight.sum())
    if weight_sum <= 0.0:
        raise ValueError("R-B2 geometry target weight must contain positive mass")
    completeness = float(np.sum(target_to_prediction * weight) / weight_sum)
    report: dict[str, float | int] = {
        "chamfer_m": float(prediction_to_target.mean() + completeness),
        "precision_mean_distance_m": float(prediction_to_target.mean()),
        "completeness_mean_distance_m": completeness,
        "outlier_fraction_2m": float(np.mean(prediction_to_target > 2.0)),
        "prediction_count": int(prediction.shape[0]),
        "target_count": int(target.shape[0]),
        "target_effective_count": weight_sum,
    }
    for threshold in (0.5, 1.0, 2.0):
        precision = float(np.mean(prediction_to_target <= threshold))
        recall = float(
            np.sum((target_to_prediction <= threshold) * weight) / weight_sum
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
    target_range = np.linalg.norm(target, axis=1)
    for label, lower, upper in RANGE_STRATA_M:
        prediction_mask = (
            (prediction_range >= lower) & (prediction_range < upper)
        )
        target_mask = (target_range >= lower) & (target_range < upper)
        if not bool(target_mask.any()):
            continue
        bin_target_distance = target_to_prediction[target_mask]
        bin_weight = weight[target_mask]
        bin_weight_sum = float(bin_weight.sum())
        if bin_weight_sum <= 0.0:
            continue
        precision = (
            float(np.mean(prediction_to_target[prediction_mask] <= 1.0))
            if bool(prediction_mask.any())
            else 0.0
        )
        recall = float(
            np.sum((bin_target_distance <= 1.0) * bin_weight) / bin_weight_sum
        )
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        if bool(prediction_mask.any()):
            report[f"{label}_precision_mean_distance_m"] = float(
                prediction_to_target[prediction_mask].mean()
            )
        report[f"{label}_completeness_mean_distance_m"] = float(
            np.sum(bin_target_distance * bin_weight) / bin_weight_sum
        )
        report[f"{label}_fscore_1m"] = fscore
    return report


def aggregate_geometry_reports(
    reports: list[dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    if not reports:
        raise ValueError("R-B2 cannot aggregate an empty geometry report list")
    keys = sorted(
        {
            key
            for report in reports
            for key, value in report.items()
            if isinstance(value, (int, float)) and not key.endswith("_count")
        }
    )
    aggregate: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = np.asarray(
            [float(report[key]) for report in reports if key in report],
            dtype=np.float64,
        )
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "sample_count": int(values.size),
        }
    return aggregate


def _validate_xyz(xyz_m: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"R-B2 XYZ must have non-empty shape (N,3), got {points.shape}")
    if not bool(np.all(np.isfinite(points))):
        raise ValueError("R-B2 XYZ must be finite")
    return points


def _boundary_margin(config: VoxelSlotConfig) -> float:
    return 0.51 * config.minimum_distance_m


def _validate_slot_placement(
    lattice: CartesianLattice,
    index_xyz: tuple[int, int, int],
    xyz_m: np.ndarray,
    config: VoxelSlotConfig,
) -> None:
    points = _validate_xyz(xyz_m)
    if points.shape[0] != config.slots_per_voxel:
        raise AssertionError("R-B2 activated voxel changed fixed slot cardinality")
    lower, upper = lattice.cell_bounds(index_xyz)
    margin = _boundary_margin(config)
    if not bool(
        np.all(points >= lower[None] + margin - 1e-10)
        and np.all(points <= upper[None] - margin + 1e-10)
    ):
        raise AssertionError("R-B2 slot escaped its fixed bounded voxel")
    if points.shape[0] > 1:
        distances = np.linalg.norm(
            points[:, None, :] - points[None, :, :],
            axis=2,
        )
        distances[np.diag_indices_from(distances)] = np.inf
        if float(distances.min()) < config.minimum_distance_m - 1e-9:
            raise AssertionError("R-B2 slots inside one voxel are not mutually exclusive")


def _placement_capacity_by_range(
    placements: dict[tuple[int, int, int], tuple[SlotPlacement, str]],
) -> list[int]:
    if not placements:
        return [0 for _ in RANGE_STRATA_M]
    xyz = np.concatenate(
        [placement.xyz_m for placement, _ in placements.values()],
        axis=0,
    )
    codes = range_stratum_codes(xyz)
    return [
        int(np.sum(codes == index)) for index in range(len(RANGE_STRATA_M))
    ]


def _flatten_placements(
    placements: dict[tuple[int, int, int], tuple[SlotPlacement, str]],
    lattice: CartesianLattice,
    config: VoxelSlotConfig,
) -> dict[str, np.ndarray]:
    xyz_parts = []
    distance_parts = []
    target_derived_parts = []
    candidate_id_parts = []
    activation_source_parts = []
    inside_parts = []
    margin = _boundary_margin(config)
    for index_xyz in sorted(placements):
        placement, source = placements[index_xyz]
        lower, upper = lattice.cell_bounds(index_xyz)
        xyz_parts.append(placement.xyz_m)
        distance_parts.append(placement.nearest_target_distance_m)
        target_derived_parts.append(placement.target_derived)
        base = lattice.linear_id(index_xyz) * config.slots_per_voxel
        candidate_id_parts.append(
            base + np.arange(config.slots_per_voxel, dtype=np.int64)
        )
        activation_source_parts.append(
            np.full(config.slots_per_voxel, source, dtype=object)
        )
        inside_parts.append(
            np.all(placement.xyz_m >= lower[None] + margin - 1e-10, axis=1)
            & np.all(placement.xyz_m <= upper[None] - margin + 1e-10, axis=1)
        )
    xyz = np.concatenate(xyz_parts, axis=0)
    return {
        "xyz_m": xyz,
        "nearest_target_distance_m": np.concatenate(distance_parts),
        "target_derived": np.concatenate(target_derived_parts),
        "candidate_ids": np.concatenate(candidate_id_parts),
        "activation_source": np.concatenate(activation_source_parts),
        "inside_fixed_cell": np.concatenate(inside_parts),
        "range_codes": range_stratum_codes(xyz),
    }


def _select_exact_capacity(
    candidate_xyz_m: np.ndarray,
    nearest_target_distance_m: np.ndarray,
    candidate_ids: np.ndarray,
    output_quotas: tuple[int, int, int],
    minimum_distance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    codes = range_stratum_codes(candidate_xyz_m)
    selected: list[int] = []
    selected_by_range = [0, 0, 0]
    rejected_by_range = [0, 0, 0]
    cells: dict[tuple[int, int, int], list[np.ndarray]] = {}
    observed_minimum_squared = math.inf
    threshold_squared = minimum_distance_m**2

    for stratum, quota in enumerate(output_quotas):
        eligible = np.flatnonzero(codes == stratum)
        order = np.lexsort(
            (
                candidate_ids[eligible],
                nearest_target_distance_m[eligible],
            )
        )
        for row in eligible[order]:
            point = candidate_xyz_m[int(row)]
            cell = tuple(
                int(math.floor(float(value) / minimum_distance_m))
                for value in point
            )
            local_minimum = math.inf
            rejected = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for neighbour in cells.get(
                            (cell[0] + dx, cell[1] + dy, cell[2] + dz),
                            (),
                        ):
                            distance_squared = float(np.sum((point - neighbour) ** 2))
                            local_minimum = min(local_minimum, distance_squared)
                            if distance_squared < threshold_squared - 1e-12:
                                rejected = True
                                break
                        if rejected:
                            break
                    if rejected:
                        break
                if rejected:
                    break
            if rejected:
                rejected_by_range[stratum] += 1
                continue
            observed_minimum_squared = min(
                observed_minimum_squared,
                local_minimum,
            )
            cells.setdefault(cell, []).append(point)
            selected.append(int(row))
            selected_by_range[stratum] += 1
            if selected_by_range[stratum] == quota:
                break
        if selected_by_range[stratum] != quota:
            raise VoxelSlotCapacityError(
                "R-B2 true 5 cm selection cannot satisfy frozen quotas",
                {
                    "candidate_capacity_by_range": _quota_dict(
                        [
                            int(np.sum(codes == index))
                            for index in range(len(RANGE_STRATA_M))
                        ]
                    ),
                    "required_output_quotas": _quota_dict(output_quotas),
                    "selected_by_range": _quota_dict(selected_by_range),
                    "rejected_5cm_by_range": _quota_dict(rejected_by_range),
                    "copy_padding_jitter_duplicate": False,
                },
            )

    rows = np.asarray(selected, dtype=np.int64)
    minimum_observed = (
        math.sqrt(observed_minimum_squared)
        if math.isfinite(observed_minimum_squared)
        else minimum_distance_m
    )
    return rows, {
        "method": "gt_distance_stable_true_5cm_range_quota",
        "candidate_count": int(candidate_xyz_m.shape[0]),
        "candidate_capacity_by_range": _quota_dict(
            [
                int(np.sum(codes == index))
                for index in range(len(RANGE_STRATA_M))
            ]
        ),
        "required_output_quotas": _quota_dict(output_quotas),
        "selected_by_range": _quota_dict(selected_by_range),
        "rejected_5cm_by_range": _quota_dict(rejected_by_range),
        "rejected_5cm_total": int(sum(rejected_by_range)),
        "minimum_euclidean_distance_m": minimum_distance_m,
        "observed_minimum_pair_distance_m": minimum_observed,
        "exact_point_count": int(rows.size),
        "unique_selected_candidate_count": int(np.unique(candidate_ids[rows]).size),
        "copy_padding_jitter_duplicate": False,
    }


def _quota_dict(values: tuple[int, int, int] | list[int]) -> dict[str, int]:
    return {
        label: int(value)
        for (label, _, _), value in zip(RANGE_STRATA_M, values, strict=True)
    }


@lru_cache(maxsize=None)
def _shell_offsets(shell: int) -> tuple[tuple[int, int, int], ...]:
    if shell < 0:
        raise ValueError("R-B2 shell must be non-negative")
    offsets = [
        (dx, dy, dz)
        for dx in range(-shell, shell + 1)
        for dy in range(-shell, shell + 1)
        for dz in range(-shell, shell + 1)
        if max(abs(dx), abs(dy), abs(dz)) == shell
    ]
    return tuple(
        sorted(
            offsets,
            key=lambda item: (
                item[0] ** 2 + item[1] ** 2 + item[2] ** 2,
                item,
            ),
        )
    )
