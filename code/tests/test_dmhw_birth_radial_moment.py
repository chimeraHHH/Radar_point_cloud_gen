from __future__ import annotations

import numpy as np
import pytest

from dmhw.birth_radial_moment import (
    PROVENANCE_DYNAMIC,
    TrackedBox,
    adjacent_pose_audit,
    build_frame_radial_moment_labels,
    parse_label_boxes,
    radar_roi_mask,
    validate_static_sign_audit,
    wrap_scalar,
)


def frame(
    index: int,
    *,
    timestamp: float,
    radar_index: int,
    world_translation_x: float,
) -> dict:
    world = np.eye(4)
    world[0, 3] = world_translation_x
    calibration = np.eye(4)
    calibration[:3, 3] = [-2.54, 0.3, 0.7]
    if index == 0:
        radar_motion = np.eye(4)
        delta = None
    else:
        source_world = np.eye(4)
        source_world[0, 3] = world_translation_x - 1.0
        lidar_motion = np.linalg.inv(world) @ source_world
        radar_motion = calibration @ lidar_motion @ np.linalg.inv(calibration)
        delta = 1.0
    return {
        "sequence": 1,
        "partition": "train",
        "window_id": "seq01_w00",
        "frame_in_window": index,
        "label": f"{radar_index:05d}_{radar_index:05d}.txt",
        "radar_index": radar_index,
        "lidar64_index": radar_index,
        "timestamp": timestamp,
        "delta_seconds_from_previous": delta,
        "world_from_lidar64": world.reshape(-1).tolist(),
        "radar_from_lidar64": calibration.reshape(-1).tolist(),
        "current_radar_from_previous_radar": radar_motion.reshape(-1).tolist(),
    }


def box(
    track_id: str,
    center: tuple[float, float, float],
    *,
    class_name: str = "Sedan",
    half_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> TrackedBox:
    return TrackedBox(
        object_index=track_id,
        track_id=track_id,
        class_name=class_name,
        center_xyz_m=np.asarray(center, dtype=np.float64),
        yaw_rad=0.0,
        half_size_xyz_m=np.asarray(half_size, dtype=np.float64),
    )


def build(
    points: np.ndarray,
    source_boxes: list[TrackedBox],
    target_boxes: list[TrackedBox],
):
    return build_frame_radial_moment_labels(
        points_xyz_m=points,
        original_point_indices=np.arange(len(points)),
        source_frame=frame(
            0,
            timestamp=0.0,
            radar_index=10,
            world_translation_x=0.0,
        ),
        target_frame=frame(
            1,
            timestamp=1.0,
            radar_index=11,
            world_translation_x=1.0,
        ),
        source_boxes=source_boxes,
        target_boxes=target_boxes,
        doppler_lower_mps=-2.0,
        doppler_period_mps=4.0,
        p5_sign_hypothesis="positive_ego",
        source_label_sha256="a" * 64,
        target_label_sha256="b" * 64,
        target_lidar_sha256="c" * 64,
        calibration_sha256="d" * 64,
        boundary_margin_m=0.1,
        dynamic_threshold_mps=0.5,
    )


def test_label_parser_matches_p5_radar_translation() -> None:
    text = (
        "* header\n"
        "*, 0, 7, Sedan, 1, 2, 3, 90, 2, 1, 0.5\n"
    )
    boxes = parse_label_boxes(text, np.asarray([0.5, -0.5, 0.7]))
    assert boxes[0].track_id == "7"
    np.testing.assert_allclose(boxes[0].center_xyz_m, [1.5, 1.5, 3.7])


def test_pose_composition_reproduces_stored_radar_motion() -> None:
    source = frame(
        0,
        timestamp=0.0,
        radar_index=10,
        world_translation_x=0.0,
    )
    target = frame(
        1,
        timestamp=1.0,
        radar_index=11,
        world_translation_x=1.0,
    )
    motion, residual, pose_hash = adjacent_pose_audit(source, target)
    assert residual < 1e-12
    assert motion[0, 3] == pytest.approx(-1.0)
    assert len(pose_hash) == 64


def test_rigid_point_target_agrees_with_p5_box_center() -> None:
    result = build(
        np.asarray([[8.0, 0.0, 0.0]]),
        [box("7", (9.0, 0.0, 0.0))],
        [box("7", (8.0, 0.0, 0.0))],
    )
    assert result.valid_point_count == 1
    record = result.records[0]
    assert record["radial_velocity_unwrapped_mps"] == pytest.approx(1.0)
    assert record["p5_box_center_target_unwrapped_mps"] == pytest.approx(1.0)
    assert record["point_box_abs_circular_error_mps"] == pytest.approx(0.0)
    assert record["provenance"] == PROVENANCE_DYNAMIC
    assert record["future_cube_accessed"] is False


def test_target_box_boundary_is_invalid_and_kept_in_denominator() -> None:
    result = build(
        np.asarray([[9.0, 0.0, 0.0]]),
        [box("7", (9.0, 0.0, 0.0))],
        [box("7", (8.0, 0.0, 0.0))],
    )
    assert result.eligible_point_count == 1
    assert result.valid_point_count == 0
    assert result.invalid_reason_counts == {"target_box_boundary": 1}


def test_overlapping_target_boxes_are_invalid() -> None:
    result = build(
        np.asarray([[8.0, 0.0, 0.0]]),
        [box("7", (9.0, 0.0, 0.0)), box("8", (9.2, 0.0, 0.0))],
        [box("7", (8.0, 0.0, 0.0)), box("8", (8.2, 0.0, 0.0))],
    )
    assert result.valid_point_count == 0
    assert result.invalid_reason_counts == {"overlapping_target_boxes": 1}


def test_missing_track_and_class_change_are_invalid() -> None:
    missing = build(
        np.asarray([[8.0, 0.0, 0.0]]),
        [],
        [box("7", (8.0, 0.0, 0.0))],
    )
    changed = build(
        np.asarray([[8.0, 0.0, 0.0]]),
        [box("7", (9.0, 0.0, 0.0), class_name="Bus")],
        [box("7", (8.0, 0.0, 0.0), class_name="Sedan")],
    )
    assert missing.invalid_reason_counts == {"missing_source_track": 1}
    assert changed.invalid_reason_counts == {"track_class_changed": 1}


def test_unverified_background_is_never_filled_from_ego_pose() -> None:
    result = build(
        np.asarray([[30.0, 20.0, 0.0]]),
        [box("7", (9.0, 0.0, 0.0))],
        [box("7", (8.0, 0.0, 0.0))],
    )
    assert result.valid_point_count == 0
    assert result.invalid_reason_counts == {"unverified_background": 1}


def test_radar_roi_excludes_zero_and_out_of_fov_points() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 20.0, 0.0],
            [10.0, 0.0, 8.0],
        ]
    )
    mask = radar_roi_mask(
        points,
        range_bounds_m=(1.0, 20.0),
        azimuth_bounds_rad=(-0.5, 0.5),
        elevation_bounds_rad=(-0.2, 0.2),
    )
    np.testing.assert_array_equal(mask, [False, True, False, False])


def test_wrap_scalar_uses_frozen_circular_interval() -> None:
    assert wrap_scalar(3.0, -2.0, 4.0) == pytest.approx(-1.0)


def test_static_sign_audit_is_sign_only_when_validation_failed() -> None:
    document = {
        "protocol": {
            "selection_partition": "train",
            "minimum_selection_margin_mps": 0.05,
            "doppler_period_mps": 4.0,
        },
        "train": {
            "selected_hypothesis": "positive_ego",
            "selected_margin_to_second_mps": 0.1,
        },
        "checks": {
            "required_frame_count": True,
            "no_frame_errors": True,
            "train_hypothesis_meets_selection_margin": True,
            "frozen_hypothesis_beats_random_on_validation": False,
        },
        "frozen_hypothesis": "positive_ego",
        "passed": False,
    }
    result = validate_static_sign_audit(document)
    assert result["hypothesis"] == "positive_ego"
    assert result["sign_only_calibration"] is True
    assert result["physics_prior_claim_enabled"] is False
