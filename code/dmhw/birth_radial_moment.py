"""Conservative label-only targets for D-MHW birth radial moments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROTOCOL = "dmhw_birth_geometry_track_radial_moment_label_audit_v1"
SIGN_HYPOTHESES = ("positive_ego", "negative_ego")
PROVENANCE_DYNAMIC = "tracked_rigid_dynamic"
PROVENANCE_STATIC_LIKE = "tracked_rigid_static_like"


@dataclass(frozen=True)
class TrackedBox:
    object_index: str
    track_id: str
    class_name: str
    center_xyz_m: np.ndarray
    yaw_rad: float
    half_size_xyz_m: np.ndarray


@dataclass
class FrameRadialMomentLabels:
    eligible_point_count: int
    valid_point_count: int
    invalid_reason_counts: dict[str, int]
    records: list[dict]
    box_comparisons: list[dict]
    dynamic_track_ids: list[str]
    pose_residual_max_abs: float
    pose_hash: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wrap_scalar(value: np.ndarray | float, lower: float, period: float):
    if not np.isfinite(lower) or not np.isfinite(period) or period <= 0.0:
        raise ValueError("Circular Doppler interval must be finite and positive")
    wrapped = np.remainder(np.asarray(value) - lower, period) + lower
    if np.ndim(value) == 0:
        return float(wrapped)
    return wrapped


def circular_error(
    observed: np.ndarray | float,
    target: np.ndarray | float,
    period: float,
):
    error = (
        np.remainder(
            np.asarray(observed) - np.asarray(target) + period / 2.0,
            period,
        )
        - period / 2.0
    )
    if np.ndim(observed) == 0 and np.ndim(target) == 0:
        return float(error)
    return error


def parse_label_boxes(
    label_text: str,
    calibration_xyz_m: np.ndarray,
) -> list[TrackedBox]:
    calibration = np.asarray(calibration_xyz_m, dtype=np.float64)
    if calibration.shape != (3,) or not np.isfinite(calibration).all():
        raise ValueError("Radar/LiDAR calibration translation must be finite XYZ")
    boxes = []
    for line in label_text.splitlines()[1:]:
        values = [value.strip() for value in line.split(",")]
        if len(values) < 11 or values[0] != "*":
            continue
        center = np.asarray(values[4:7], dtype=np.float64) + calibration
        half_size = np.asarray(values[8:11], dtype=np.float64)
        yaw = math.radians(float(values[7]))
        if (
            not np.isfinite(center).all()
            or not np.isfinite(half_size).all()
            or not np.isfinite(yaw)
            or np.any(half_size <= 0.0)
        ):
            raise ValueError("K-Radar label contains a malformed box")
        boxes.append(
            TrackedBox(
                object_index=values[1],
                track_id=values[2],
                class_name=values[3],
                center_xyz_m=center,
                yaw_rad=yaw,
                half_size_xyz_m=half_size,
            )
        )
    return boxes


def unique_tracks(
    boxes: Sequence[TrackedBox],
) -> tuple[dict[str, TrackedBox], set[str]]:
    grouped: dict[str, list[TrackedBox]] = defaultdict(list)
    for box in boxes:
        grouped[box.track_id].append(box)
    duplicate_ids = {
        track_id for track_id, values in grouped.items() if len(values) != 1
    }
    return (
        {
            track_id: values[0]
            for track_id, values in grouped.items()
            if track_id not in duplicate_ids
        },
        duplicate_ids,
    )


def _rotation_z(yaw_rad: float) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def box_local_coordinates(
    points_xyz_m: np.ndarray,
    box: TrackedBox,
) -> np.ndarray:
    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point geometry must have shape Nx3")
    return (points - box.center_xyz_m) @ _rotation_z(box.yaw_rad)


def points_in_box(
    points_xyz_m: np.ndarray,
    box: TrackedBox,
    *,
    inward_margin_m: float = 0.0,
) -> np.ndarray:
    if inward_margin_m < 0.0:
        raise ValueError("Box inward margin cannot be negative")
    interior = box.half_size_xyz_m - inward_margin_m
    if np.any(interior <= 0.0):
        return np.zeros(len(points_xyz_m), dtype=bool)
    local = box_local_coordinates(points_xyz_m, box)
    return np.all(np.abs(local) <= interior, axis=1)


def radar_roi_mask(
    points_xyz_m: np.ndarray,
    *,
    range_bounds_m: tuple[float, float],
    azimuth_bounds_rad: tuple[float, float],
    elevation_bounds_rad: tuple[float, float],
) -> np.ndarray:
    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Radar ROI points must have shape Nx3")
    finite = np.isfinite(points).all(axis=1)
    radius = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    elevation = np.arcsin(
        np.divide(
            points[:, 2],
            radius,
            out=np.zeros_like(radius),
            where=radius > 0.0,
        )
    )
    return (
        finite
        & (radius >= range_bounds_m[0])
        & (radius <= range_bounds_m[1])
        & (azimuth >= azimuth_bounds_rad[0])
        & (azimuth <= azimuth_bounds_rad[1])
        & (elevation >= elevation_bounds_rad[0])
        & (elevation <= elevation_bounds_rad[1])
    )


def _as_transform(frame: Mapping[str, object], name: str) -> np.ndarray:
    raw = frame.get(name)
    if not isinstance(raw, list) or len(raw) != 16:
        raise ValueError(f"Frame is missing a 4x4 {name}")
    transform = np.asarray(raw, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(transform).all():
        raise ValueError(f"Frame {name} contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"Frame {name} is not homogeneous")
    return transform


def adjacent_pose_audit(
    source_frame: Mapping[str, object],
    target_frame: Mapping[str, object],
) -> tuple[np.ndarray, float, str]:
    identity_fields = ("sequence", "partition", "window_id")
    if any(source_frame.get(key) != target_frame.get(key) for key in identity_fields):
        raise ValueError("Radial target frames cross a sequence, partition, or window")
    source_index = int(source_frame["frame_in_window"])
    target_index = int(target_frame["frame_in_window"])
    if target_index != source_index + 1:
        raise ValueError("Radial target requires immediately adjacent labels")

    delta_seconds = float(target_frame["delta_seconds_from_previous"])
    timestamp_delta = float(target_frame["timestamp"]) - float(source_frame["timestamp"])
    if delta_seconds <= 0.0 or not np.isclose(
        delta_seconds,
        timestamp_delta,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("Adjacent-frame timestamp delta is inconsistent")

    source_world_from_lidar = _as_transform(
        source_frame, "world_from_lidar64"
    )
    target_world_from_lidar = _as_transform(
        target_frame, "world_from_lidar64"
    )
    calibration = _as_transform(target_frame, "radar_from_lidar64")
    source_calibration = _as_transform(source_frame, "radar_from_lidar64")
    if not np.allclose(calibration, source_calibration, atol=1e-9, rtol=0.0):
        raise ValueError("Adjacent frames use different radar/LiDAR calibration")

    expected_lidar_motion = (
        np.linalg.inv(target_world_from_lidar) @ source_world_from_lidar
    )
    expected_radar_motion = (
        calibration
        @ expected_lidar_motion
        @ np.linalg.inv(calibration)
    )
    stored_radar_motion = _as_transform(
        target_frame, "current_radar_from_previous_radar"
    )
    residual = float(np.max(np.abs(expected_radar_motion - stored_radar_motion)))
    pose_payload = {
        "source_world_from_lidar64": source_frame["world_from_lidar64"],
        "target_world_from_lidar64": target_frame["world_from_lidar64"],
        "radar_from_lidar64": target_frame["radar_from_lidar64"],
        "stored_current_radar_from_previous_radar": target_frame[
            "current_radar_from_previous_radar"
        ],
        "delta_seconds": delta_seconds,
    }
    return stored_radar_motion, residual, canonical_hash(pose_payload)


def validate_static_sign_audit(document: Mapping[str, object]) -> dict:
    protocol = document.get("protocol")
    train = document.get("train")
    checks = document.get("checks")
    if not all(isinstance(value, Mapping) for value in (protocol, train, checks)):
        raise ValueError("Static Doppler audit is incomplete")
    hypothesis = document.get("frozen_hypothesis")
    if hypothesis not in SIGN_HYPOTHESES:
        raise ValueError(f"Unsupported P5 sign hypothesis: {hypothesis}")
    if protocol.get("selection_partition") != "train":
        raise ValueError("P5 sign convention was not selected on train")
    if train.get("selected_hypothesis") != hypothesis:
        raise ValueError("Frozen and train-selected P5 signs differ")
    if float(train["selected_margin_to_second_mps"]) < float(
        protocol["minimum_selection_margin_mps"]
    ):
        raise ValueError("P5 train-only sign margin did not pass")
    for name in (
        "required_frame_count",
        "no_frame_errors",
        "train_hypothesis_meets_selection_margin",
    ):
        if checks.get(name) is not True:
            raise ValueError(f"P5 sign-only audit failed required check {name}")
    return {
        "hypothesis": hypothesis,
        "selection_partition": "train",
        "sign_only_calibration": document.get("passed") is not True,
        "physics_prior_claim_enabled": document.get("passed") is True,
        "doppler_period_mps": float(protocol["doppler_period_mps"]),
        "train_selection_margin_mps": float(
            train["selected_margin_to_second_mps"]
        ),
    }


def _box_membership(
    points_xyz_m: np.ndarray,
    boxes: Sequence[TrackedBox],
    boundary_margin_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inclusive_count = np.zeros(len(points_xyz_m), dtype=np.int16)
    strict_count = np.zeros(len(points_xyz_m), dtype=np.int16)
    strict_owner = np.full(len(points_xyz_m), -1, dtype=np.int32)
    for index, box in enumerate(boxes):
        inclusive = points_in_box(points_xyz_m, box)
        strict = points_in_box(
            points_xyz_m,
            box,
            inward_margin_m=boundary_margin_m,
        )
        inclusive_count += inclusive.astype(np.int16)
        strict_count += strict.astype(np.int16)
        strict_owner[strict] = index
    return inclusive_count, strict_count, strict_owner


def _rigid_previous_points(
    target_points_xyz_m: np.ndarray,
    target_box: TrackedBox,
    source_box: TrackedBox,
) -> np.ndarray:
    local = box_local_coordinates(target_points_xyz_m, target_box)
    return local @ _rotation_z(source_box.yaw_rad).T + source_box.center_xyz_m


def _p5_sign_multiplier(hypothesis: str) -> float:
    if hypothesis == "positive_ego":
        return -1.0
    if hypothesis == "negative_ego":
        return 1.0
    raise ValueError(f"Unsupported P5 sign hypothesis: {hypothesis}")


def build_frame_radial_moment_labels(
    *,
    points_xyz_m: np.ndarray,
    original_point_indices: np.ndarray,
    source_frame: Mapping[str, object],
    target_frame: Mapping[str, object],
    source_boxes: Sequence[TrackedBox],
    target_boxes: Sequence[TrackedBox],
    doppler_lower_mps: float,
    doppler_period_mps: float,
    p5_sign_hypothesis: str,
    source_label_sha256: str,
    target_label_sha256: str,
    target_lidar_sha256: str,
    calibration_sha256: str,
    boundary_margin_m: float = 0.1,
    dynamic_threshold_mps: float = 0.5,
) -> FrameRadialMomentLabels:
    """Build valid point labels while preserving every invalid point."""

    points = np.asarray(points_xyz_m, dtype=np.float64)
    original_indices = np.asarray(original_point_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Eligible future LiDAR geometry must be non-empty Nx3")
    if original_indices.shape != (len(points),):
        raise ValueError("Original future LiDAR point indices are misaligned")
    if not np.isfinite(points).all():
        raise ValueError("Eligible future LiDAR geometry must be finite")
    if boundary_margin_m <= 0.0:
        raise ValueError("G-RM requires a positive box-boundary exclusion")
    if dynamic_threshold_mps <= 0.0:
        raise ValueError("Dynamic radial threshold must be positive")

    radar_motion, pose_residual, pose_hash = adjacent_pose_audit(
        source_frame,
        target_frame,
    )
    if pose_residual > 1e-8:
        raise ValueError(
            "Stored radar motion differs from world-pose composition: "
            f"{pose_residual:.3e}"
        )
    rotation_target_from_source = radar_motion[:3, :3]
    delta_seconds = float(target_frame["delta_seconds_from_previous"])
    sign = _p5_sign_multiplier(p5_sign_hypothesis)

    target_unique, target_duplicates = unique_tracks(target_boxes)
    source_unique, source_duplicates = unique_tracks(source_boxes)
    inclusive_count, strict_count, strict_owner = _box_membership(
        points,
        target_boxes,
        boundary_margin_m,
    )
    reason = np.full(len(points), "", dtype=object)
    reason[inclusive_count == 0] = "unverified_background"
    reason[inclusive_count > 1] = "overlapping_target_boxes"
    boundary = (inclusive_count == 1) & (strict_count == 0)
    reason[boundary] = "target_box_boundary"
    malformed_strict = (inclusive_count == 1) & (strict_count > 1)
    reason[malformed_strict] = "overlapping_target_boxes"

    records: list[dict] = []
    box_comparisons: list[dict] = []
    dynamic_track_ids: set[str] = set()
    label_pair_hash = canonical_hash(
        {
            "source_label_sha256": source_label_sha256,
            "target_label_sha256": target_label_sha256,
        }
    )
    for target_index, target_box in enumerate(target_boxes):
        selected = np.flatnonzero(
            (reason == "") & (strict_owner == target_index)
        )
        if selected.size == 0:
            continue
        track_id = target_box.track_id
        if track_id in target_duplicates:
            reason[selected] = "duplicate_target_track_id"
            continue
        if target_unique.get(track_id) is not target_box:
            reason[selected] = "target_track_not_unique"
            continue
        if track_id in source_duplicates:
            reason[selected] = "duplicate_source_track_id"
            continue
        source_box = source_unique.get(track_id)
        if source_box is None:
            reason[selected] = "missing_source_track"
            continue
        if source_box.class_name != target_box.class_name:
            reason[selected] = "track_class_changed"
            continue

        target_points = points[selected]
        source_points = _rigid_previous_points(
            target_points,
            target_box,
            source_box,
        )
        source_inclusive = np.zeros(len(selected), dtype=np.int16)
        for other_box in source_boxes:
            source_inclusive += points_in_box(
                source_points,
                other_box,
            ).astype(np.int16)
        source_strict = points_in_box(
            source_points,
            source_box,
            inward_margin_m=boundary_margin_m,
        )
        source_overlap = source_inclusive > 1
        source_boundary = (source_inclusive == 1) & ~source_strict
        source_missing = source_inclusive == 0
        reason[selected[source_overlap]] = "overlapping_source_boxes"
        reason[selected[source_boundary]] = "source_box_boundary"
        reason[selected[source_missing]] = "source_rigid_point_outside_box"
        valid_local = ~(source_overlap | source_boundary | source_missing)
        valid_indices = selected[valid_local]
        if valid_indices.size == 0:
            continue
        valid_target_points = points[valid_indices]
        valid_source_points = source_points[valid_local]

        target_radius = np.linalg.norm(valid_target_points, axis=1)
        nonzero_ray = target_radius > 1e-6
        reason[valid_indices[~nonzero_ray]] = "degenerate_target_ray"
        valid_indices = valid_indices[nonzero_ray]
        valid_target_points = valid_target_points[nonzero_ray]
        valid_source_points = valid_source_points[nonzero_ray]
        target_radius = target_radius[nonzero_ray]
        if valid_indices.size == 0:
            continue

        aligned_source_points = (
            valid_source_points @ rotation_target_from_source.T
        )
        relative_displacement = valid_target_points - aligned_source_points
        ray = valid_target_points / target_radius[:, None]
        projected_range_rate = np.sum(relative_displacement * ray, axis=1) / (
            delta_seconds
        )
        radial_target = sign * projected_range_rate
        wrapped_target = wrap_scalar(
            radial_target,
            doppler_lower_mps,
            doppler_period_mps,
        )

        source_range = float(np.linalg.norm(source_box.center_xyz_m))
        target_range = float(np.linalg.norm(target_box.center_xyz_m))
        p5_range_rate = (target_range - source_range) / delta_seconds
        p5_target = sign * p5_range_rate
        p5_wrapped = wrap_scalar(
            p5_target,
            doppler_lower_mps,
            doppler_period_mps,
        )
        point_box_error = np.abs(
            circular_error(
                wrapped_target,
                p5_wrapped,
                doppler_period_mps,
            )
        )
        dynamic = abs(p5_target) >= dynamic_threshold_mps
        provenance = (
            PROVENANCE_DYNAMIC if dynamic else PROVENANCE_STATIC_LIKE
        )
        if dynamic:
            dynamic_track_ids.add(track_id)

        box_comparisons.append(
            {
                "track_id": track_id,
                "class": target_box.class_name,
                "valid_point_count": int(valid_indices.size),
                "p5_box_center_target_unwrapped_mps": p5_target,
                "point_box_abs_circular_error_mean_mps": float(
                    np.mean(point_box_error)
                ),
                "point_box_abs_circular_error_median_mps": float(
                    np.median(point_box_error)
                ),
                "dynamic": dynamic,
            }
        )
        for local_index, point_index in enumerate(valid_indices.tolist()):
            records.append(
                {
                    "sequence": int(target_frame["sequence"]),
                    "partition": str(target_frame["partition"]),
                    "window_id": str(target_frame["window_id"]),
                    "source_frame": {
                        "label": str(source_frame["label"]),
                        "radar_index": int(source_frame["radar_index"]),
                        "lidar64_index": int(source_frame["lidar64_index"]),
                        "frame_in_window": int(source_frame["frame_in_window"]),
                    },
                    "target_frame": {
                        "label": str(target_frame["label"]),
                        "radar_index": int(target_frame["radar_index"]),
                        "lidar64_index": int(target_frame["lidar64_index"]),
                        "frame_in_window": int(target_frame["frame_in_window"]),
                    },
                    "target_point_index": int(original_indices[point_index]),
                    "target_xyz_m": valid_target_points[local_index].tolist(),
                    "delta_seconds": delta_seconds,
                    "track_id": track_id,
                    "class": target_box.class_name,
                    "provenance": provenance,
                    "radial_velocity_unwrapped_mps": float(
                        radial_target[local_index]
                    ),
                    "radial_velocity_wrapped_mps": float(
                        wrapped_target[local_index]
                    ),
                    "p5_box_center_target_unwrapped_mps": p5_target,
                    "point_box_abs_circular_error_mps": float(
                        point_box_error[local_index]
                    ),
                    "pose_sha256": pose_hash,
                    "source_label_sha256": source_label_sha256,
                    "target_label_sha256": target_label_sha256,
                    "label_pair_sha256": label_pair_hash,
                    "target_lidar_sha256": target_lidar_sha256,
                    "radar_lidar_calibration_sha256": calibration_sha256,
                    "future_cube_accessed": False,
                    "deployment_eligible": False,
                }
            )
        reason[valid_indices] = "__valid__"

    unresolved = reason == ""
    reason[unresolved] = "unresolved_conservative_invalid"
    invalid_counts = Counter(
        str(value)
        for value in reason
        if value and value != "__valid__"
    )
    valid_count = len(records)
    if sum(invalid_counts.values()) + valid_count != len(points):
        raise AssertionError("G-RM valid and invalid labels do not cover the denominator")
    return FrameRadialMomentLabels(
        eligible_point_count=len(points),
        valid_point_count=valid_count,
        invalid_reason_counts=dict(sorted(invalid_counts.items())),
        records=records,
        box_comparisons=box_comparisons,
        dynamic_track_ids=sorted(dynamic_track_ids),
        pose_residual_max_abs=pose_residual,
        pose_hash=pose_hash,
    )
