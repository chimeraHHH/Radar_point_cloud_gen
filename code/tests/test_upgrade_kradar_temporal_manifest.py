import numpy as np
import pytest

from upgrade_kradar_temporal_manifest import upgrade_manifest_document


def legacy_document() -> dict:
    return {
        "gate_pass": True,
        "checks": {
            "scene_split_gate_passed": True,
            "test_partition_untouched": True,
        },
        "windows": [
            {
                "window_id": "seq06_w00",
                "sequence": 6,
                "partition": "validation",
            }
        ],
        "frames": [
            {
                "window_id": "seq06_w00",
                "sequence": 6,
                "partition": "validation",
                "radar_index": 10,
                "current_lidar64_from_previous_lidar64": (
                    np.eye(4).reshape(-1).tolist()
                ),
            }
        ],
    }


def test_upgrade_conjugates_lidar_motion_without_changing_membership() -> None:
    legacy = legacy_document()
    calibration = np.eye(4)
    calibration[:3, 3] = [1.5, -0.25, 0.7]
    lidar_motion = np.eye(4)
    lidar_motion[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    lidar_motion[0, 3] = 2.0
    legacy["frames"][0][
        "current_lidar64_from_previous_lidar64"
    ] = lidar_motion.reshape(-1).tolist()

    upgraded = upgrade_manifest_document(legacy, {6: calibration})

    expected = calibration @ lidar_motion @ np.linalg.inv(calibration)
    actual = np.asarray(
        upgraded["frames"][0]["current_radar_from_previous_radar"]
    ).reshape(4, 4)
    np.testing.assert_allclose(actual, expected)
    assert len(upgraded["frames"]) == len(legacy["frames"])
    assert len(upgraded["windows"]) == len(legacy["windows"])
    assert upgraded["checks"]["radar_frame_ego_transforms_present"]
    assert upgraded["checks"]["frame_membership_unchanged"]
    assert upgraded["gate_pass"]


def test_upgrade_rejects_test_partition() -> None:
    legacy = legacy_document()
    legacy["frames"][0]["partition"] = "test"

    with pytest.raises(ValueError, match="cannot access test"):
        upgrade_manifest_document(legacy, {6: np.eye(4)})


def test_upgrade_requires_every_sequence_calibration() -> None:
    with pytest.raises(ValueError, match="sequence 6"):
        upgrade_manifest_document(legacy_document(), {})
