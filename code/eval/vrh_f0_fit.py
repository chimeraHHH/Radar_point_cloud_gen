"""The only GT-visible mark fitter for the frozen VRH-F0 capacity oracle."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from cube_dense.kradar import cartesian_to_polar, polar_to_cartesian
from eval.vrh_f0_support import (
    HAZARD_EPSILON,
    HAZARD_TAU_M,
    ModelMarks,
    SupportCommit,
    VRHSupport,
    assign_polar_to_support,
    canonical_columns_digest,
    canonicalize_azimuth,
    model_marks_interior_report,
    model_marks_digest,
    validate_support_commit,
    write_canonical_columns,
)


NEAREST_TIE_TOLERANCE_M2 = 1e-12
FIT_SIDECAR_SCHEMA = (
    ("nearest_target_id", "<i8"),
    ("fitted_distance_m", "<f8"),
    ("exact_target_coordinate_match", "<u1"),
)
CANONICAL_TARGET_SCHEMA = (
    ("canonical_target_id", "<i8"),
    ("r", "<f8"),
    ("a", "<f8"),
    ("e", "<f8"),
    ("x", "<f8"),
    ("y", "<f8"),
    ("z", "<f8"),
    ("confidence", "<f8"),
)


@dataclass(frozen=True)
class CanonicalTargets:
    xyz: np.ndarray
    confidence: np.ndarray
    polar_rae: np.ndarray
    canonical_target_id: np.ndarray
    digest_sha256: str

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])


@dataclass(frozen=True)
class FitSidecar:
    nearest_target_id: np.ndarray
    fitted_distance_m: np.ndarray
    exact_target_coordinate_match: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.nearest_target_id.shape)  # type: ignore[return-value]

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "nearest_target_id": self.nearest_target_id.reshape(-1),
            "fitted_distance_m": self.fitted_distance_m.reshape(-1),
            "exact_target_coordinate_match": self.exact_target_coordinate_match.reshape(-1),
        }

    def validate(self, support: VRHSupport) -> None:
        expected = (support.ray_count, support.range_axis.count)
        expected_dtypes = dict(FIT_SIDECAR_SCHEMA)
        for name in expected_dtypes:
            values = getattr(self, name)
            if values.shape != expected:
                raise ValueError(f"FitSidecar.{name} shape {values.shape} != {expected}")
            if np.dtype(values.dtype) != np.dtype(expected_dtypes[name]):
                raise ValueError(
                    f"FitSidecar.{name} dtype {values.dtype} != {expected_dtypes[name]}"
                )
        if np.any(self.nearest_target_id < 0):
            raise ValueError("FitSidecar contains a negative target ID")
        if not np.isfinite(self.fitted_distance_m).all():
            raise ValueError("FitSidecar fitted distance is non-finite")
        if np.any(self.fitted_distance_m < 0.0):
            raise ValueError("FitSidecar fitted distance is negative")


@dataclass(frozen=True)
class OracleFitResult:
    model_marks: ModelMarks
    fit_sidecar: FitSidecar
    canonical_targets: CanonicalTargets
    report: dict[str, Any]


def load_target_only(
    path: Path,
    *,
    support_commit: SupportCommit,
) -> np.ndarray:
    """Load the sole allowed target tensor after support commitment."""

    validate_support_commit(support_commit)
    with np.load(path) as cache:
        target = cache["target_xyz_confidence"].astype(np.float64)
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError(f"VRH target must be non-empty (N,4), got {target.shape}")
    if not np.isfinite(target).all():
        raise ValueError("VRH target contains a non-finite value")
    return np.ascontiguousarray(target)


def _target_columns(targets: CanonicalTargets) -> dict[str, np.ndarray]:
    return {
        "canonical_target_id": targets.canonical_target_id,
        "r": targets.polar_rae[:, 0],
        "a": targets.polar_rae[:, 1],
        "e": targets.polar_rae[:, 2],
        "x": targets.xyz[:, 0],
        "y": targets.xyz[:, 1],
        "z": targets.xyz[:, 2],
        "confidence": targets.confidence,
    }


def canonicalize_targets(target_xyz_confidence: np.ndarray) -> CanonicalTargets:
    target = np.asarray(target_xyz_confidence, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError("VRH target canonicalization requires non-empty (N,4)")
    if not np.isfinite(target).all():
        raise ValueError("VRH target canonicalization received non-finite values")
    xyz = target[:, :3]
    confidence = target[:, 3]
    polar = cartesian_to_polar(xyz)
    polar[:, 1] = canonicalize_azimuth(polar[:, 1])
    order = np.lexsort(
        (
            confidence,
            xyz[:, 2],
            xyz[:, 1],
            xyz[:, 0],
            polar[:, 2],
            polar[:, 1],
            polar[:, 0],
        )
    )
    xyz = np.ascontiguousarray(xyz[order], dtype=np.float64)
    confidence = np.ascontiguousarray(confidence[order], dtype=np.float64)
    polar = np.ascontiguousarray(polar[order], dtype=np.float64)
    ids = np.arange(target.shape[0], dtype="<i8")
    provisional = CanonicalTargets(
        xyz=xyz,
        confidence=confidence,
        polar_rae=polar,
        canonical_target_id=ids,
        digest_sha256="",
    )
    digest = canonical_columns_digest(
        kind="vrh_f0_canonical_targets_v1",
        schema=CANONICAL_TARGET_SCHEMA,
        columns=_target_columns(provisional),
    )
    return CanonicalTargets(
        xyz=xyz,
        confidence=confidence,
        polar_rae=polar,
        canonical_target_id=ids,
        digest_sha256=digest,
    )


def _unique_target_coordinates(targets: CanonicalTargets) -> tuple[np.ndarray, np.ndarray]:
    unique_xyz, first = np.unique(targets.xyz, axis=0, return_index=True)
    canonical_ids = targets.canonical_target_id[first]
    order = np.argsort(canonical_ids, kind="stable")
    return (
        np.ascontiguousarray(unique_xyz[order], dtype=np.float64),
        np.ascontiguousarray(canonical_ids[order], dtype=np.int64),
    )


def _resolve_nearest_targets(
    tree: cKDTree,
    unique_xyz: np.ndarray,
    unique_canonical_ids: np.ndarray,
    query_xyz: np.ndarray,
) -> np.ndarray:
    if unique_xyz.shape[0] == 1:
        return np.full(query_xyz.shape[0], unique_canonical_ids[0], dtype=np.int64)
    distance, index = tree.query(query_xyz, k=2, workers=1)
    nearest = unique_canonical_ids[index[:, 0]].astype(np.int64, copy=True)
    distance_squared = distance * distance
    tie_rows = np.flatnonzero(
        distance_squared[:, 1] - distance_squared[:, 0]
        <= NEAREST_TIE_TOLERANCE_M2
    )
    for row in tie_rows:
        radius = math.sqrt(
            max(float(distance_squared[row, 0]) + NEAREST_TIE_TOLERANCE_M2, 0.0)
        )
        candidates = np.asarray(
            tree.query_ball_point(query_xyz[row], np.nextafter(radius, np.inf)),
            dtype=np.int64,
        )
        differences = unique_xyz[candidates] - query_xyz[row]
        squared = np.einsum("ij,ij->i", differences, differences)
        minimum = float(squared.min())
        eligible = candidates[squared <= minimum + NEAREST_TIE_TOLERANCE_M2]
        nearest[row] = int(unique_canonical_ids[eligible].min())
    return nearest


def _cell_coordinates(
    support: VRHSupport,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell_ids = np.arange(start, stop, dtype=np.uint64)
    range_count = support.range_axis.count
    elevation_count = support.elevation_axis.count
    r2 = (cell_ids % range_count).astype(np.int64)
    ray = cell_ids // range_count
    e2 = (ray % elevation_count).astype(np.int64)
    a2 = (ray // elevation_count).astype(np.int64)
    radius = support.range_axis.centers[r2]
    azimuth = support.azimuth_axis.centers[a2]
    elevation = support.elevation_axis.centers[e2]
    return r2, a2, e2, radius, azimuth, elevation


def _representable_delta(
    desired: np.ndarray,
    center: np.ndarray,
    lower_interior: np.ndarray,
    upper_interior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a delta whose float64 reconstruction remains in the cell."""

    delta = np.asarray(desired - center, dtype=np.float64)
    for _ in range(4):
        reconstructed = center + delta
        below = reconstructed < lower_interior
        above = reconstructed > upper_interior
        if not bool(below.any() or above.any()):
            return delta, reconstructed
        delta[below] = np.nextafter(delta[below], np.inf)
        delta[above] = np.nextafter(delta[above], -np.inf)
    reconstructed = center + delta
    if np.any(
        (reconstructed < lower_interior) | (reconstructed > upper_interior)
    ):
        raise ValueError("VRH fitted delta cannot reconstruct inside its source cell")
    return delta, reconstructed


def fit_gt_oracle_marks(
    support: VRHSupport,
    target_xyz_confidence: np.ndarray,
    *,
    query_chunk_size: int = 131_072,
) -> OracleFitResult:
    """Fit the sole frozen nearest-target mark field without export state."""

    if query_chunk_size <= 0:
        raise ValueError("VRH fitter query chunk size must be positive")
    targets = canonicalize_targets(target_xyz_confidence)
    unique_xyz, unique_ids = _unique_target_coordinates(targets)
    tree = cKDTree(unique_xyz)
    shape = (support.ray_count, support.range_axis.count)
    base_hazard = np.empty(shape, dtype="<f8")
    delta_r = np.empty(shape, dtype="<f8")
    delta_a = np.empty(shape, dtype="<f8")
    delta_e = np.empty(shape, dtype="<f8")
    priority = np.empty(shape, dtype="<i8")
    confidence = np.empty(shape, dtype="<f8")
    nearest_target_id = np.empty(shape, dtype="<i8")
    fitted_distance = np.empty(shape, dtype="<f8")
    exact_match = np.empty(shape, dtype="<u1")

    flat_hazard = base_hazard.reshape(-1)
    flat_delta_r = delta_r.reshape(-1)
    flat_delta_a = delta_a.reshape(-1)
    flat_delta_e = delta_e.reshape(-1)
    flat_priority = priority.reshape(-1)
    flat_confidence = confidence.reshape(-1)
    flat_nearest = nearest_target_id.reshape(-1)
    flat_distance = fitted_distance.reshape(-1)
    flat_exact = exact_match.reshape(-1)

    for start in range(0, support.cell_count, query_chunk_size):
        stop = min(start + query_chunk_size, support.cell_count)
        r2, a2, e2, radius, azimuth, elevation = _cell_coordinates(
            support, start, stop
        )
        center_xyz = polar_to_cartesian(radius, azimuth, elevation)
        chosen_ids = _resolve_nearest_targets(
            tree,
            unique_xyz,
            unique_ids,
            center_xyz,
        )
        chosen_polar = targets.polar_rae[chosen_ids]
        fitted_r = np.clip(
            chosen_polar[:, 0],
            support.range_axis.lower_interior[r2],
            support.range_axis.upper_interior[r2],
        )
        fitted_a = np.clip(
            chosen_polar[:, 1],
            support.azimuth_axis.lower_interior[a2],
            support.azimuth_axis.upper_interior[a2],
        )
        fitted_e = np.clip(
            chosen_polar[:, 2],
            support.elevation_axis.lower_interior[e2],
            support.elevation_axis.upper_interior[e2],
        )
        reconstructed_delta_r, fitted_r = _representable_delta(
            fitted_r,
            radius,
            support.range_axis.lower_interior[r2],
            support.range_axis.upper_interior[r2],
        )
        reconstructed_delta_a, fitted_a = _representable_delta(
            fitted_a,
            azimuth,
            support.azimuth_axis.lower_interior[a2],
            support.azimuth_axis.upper_interior[a2],
        )
        reconstructed_delta_e, fitted_e = _representable_delta(
            fitted_e,
            elevation,
            support.elevation_axis.lower_interior[e2],
            support.elevation_axis.upper_interior[e2],
        )
        fitted_xyz = polar_to_cartesian(fitted_r, fitted_a, fitted_e)
        difference = fitted_xyz - targets.xyz[chosen_ids]
        distance = np.sqrt(np.einsum("ij,ij->i", difference, difference))
        scaled_priority = np.rint(1.0e9 * distance)
        if not np.isfinite(scaled_priority).all() or np.any(
            scaled_priority > np.iinfo(np.int64).max
        ):
            raise ValueError("VRH fitted priority exceeds signed int64")
        flat_hazard[start:stop] = np.clip(
            np.exp(-distance / HAZARD_TAU_M),
            HAZARD_EPSILON,
            1.0 - HAZARD_EPSILON,
        )
        flat_delta_r[start:stop] = reconstructed_delta_r
        flat_delta_a[start:stop] = reconstructed_delta_a
        flat_delta_e[start:stop] = reconstructed_delta_e
        flat_priority[start:stop] = scaled_priority.astype(np.int64)
        flat_confidence[start:stop] = np.maximum(targets.confidence[chosen_ids], 0.0)
        flat_nearest[start:stop] = chosen_ids
        flat_distance[start:stop] = distance
        flat_exact[start:stop] = np.all(
            np.ascontiguousarray(fitted_xyz).view(np.uint64)
            == np.ascontiguousarray(targets.xyz[chosen_ids]).view(np.uint64),
            axis=1,
        ).astype(np.uint8)

    marks = ModelMarks(
        base_hazard=base_hazard,
        delta_r=delta_r,
        delta_a=delta_a,
        delta_e=delta_e,
        opaque_distance_priority_key=priority,
        confidence=confidence,
    )
    marks.validate(support)
    interior_report = model_marks_interior_report(support, marks)
    if not interior_report["all_marks_inside_source_cell_interiors"]:
        raise AssertionError("VRH fitter produced an out-of-cell mark")
    sidecar = FitSidecar(
        nearest_target_id=nearest_target_id,
        fitted_distance_m=fitted_distance,
        exact_target_coordinate_match=exact_match,
    )
    sidecar.validate(support)
    _, _, _, in_support = assign_polar_to_support(support, targets.polar_rae)
    positive_weight = np.maximum(targets.confidence, 0.0)
    outside_positive = (~in_support) & (positive_weight > 0.0)
    fanout = np.bincount(flat_nearest, minlength=targets.count)
    sidecar_digest = fit_sidecar_digest(sidecar, support)
    report = {
        "target_count": targets.count,
        "unique_target_coordinate_count": int(unique_xyz.shape[0]),
        "canonical_target_sha256": targets.digest_sha256,
        "model_marks_sha256": model_marks_digest(marks, support),
        "fit_sidecar_sha256": sidecar_digest,
        "mark_interior_verification": interior_report,
        "nearest_tie_tolerance_m2": NEAREST_TIE_TOLERANCE_M2,
        "tau_m": HAZARD_TAU_M,
        "epsilon": HAZARD_EPSILON,
        "exact_target_coordinate_match_count": int(flat_exact.sum()),
        "lifting_fanout": {
            "minimum": int(fanout.min()),
            "median": float(np.median(fanout)),
            "maximum": int(fanout.max()),
            "sum": int(fanout.sum()),
        },
        "target_support": {
            "in_support_count": int(in_support.sum()),
            "out_of_support_count": int((~in_support).sum()),
            "positive_confidence_out_of_support_count": int(outside_positive.sum()),
            "positive_confidence_out_of_support_weight": float(
                positive_weight[outside_positive].sum()
            ),
        },
        "target_used_for_mark_hazard_fit": True,
        "target_used_for_mark_offsets": True,
        "target_used_for_mark_priority": True,
        "target_used_for_seed_reservation": False,
    }
    return OracleFitResult(
        model_marks=marks,
        fit_sidecar=sidecar,
        canonical_targets=targets,
        report=report,
    )


def fit_sidecar_digest(sidecar: FitSidecar, support: VRHSupport) -> str:
    sidecar.validate(support)
    return canonical_columns_digest(
        kind="vrh_f0_fit_sidecar_v1",
        schema=FIT_SIDECAR_SCHEMA,
        columns=sidecar.columns(),
        extra_header={"shape": list(sidecar.shape), "support_sha256": support.digest_sha256},
    )


def write_fit_sidecar(path: Path, sidecar: FitSidecar, support: VRHSupport) -> str:
    sidecar.validate(support)
    return write_canonical_columns(
        path,
        kind="vrh_f0_fit_sidecar_v1",
        schema=FIT_SIDECAR_SCHEMA,
        columns=sidecar.columns(),
        extra_header={"shape": list(sidecar.shape), "support_sha256": support.digest_sha256},
    )
