from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_kradar_temporal_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("build_kradar_temporal_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TemporalManifestReleaseTest(unittest.TestCase):
    def test_lidar_motion_is_conjugated_into_radar_frame(self) -> None:
        radar_from_lidar = np.eye(4)
        radar_from_lidar[:3, 3] = [1.5, -0.25, 0.7]
        current_lidar_from_previous = np.eye(4)
        angle = np.deg2rad(30.0)
        current_lidar_from_previous[:2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        current_lidar_from_previous[:3, 3] = [2.0, 1.0, 0.1]

        actual = MODULE.conjugate_lidar_motion_to_radar(
            current_lidar_from_previous, radar_from_lidar
        )
        expected = (
            radar_from_lidar
            @ current_lidar_from_previous
            @ np.linalg.inv(radar_from_lidar)
        )

        np.testing.assert_allclose(actual, expected)
        self.assertFalse(np.allclose(actual, current_lidar_from_previous))

    def test_identity_lidar_motion_remains_identity_in_radar_frame(self) -> None:
        radar_from_lidar = np.eye(4)
        radar_from_lidar[:3, 3] = [0.5, 0.1, 0.7]
        actual = MODULE.conjugate_lidar_motion_to_radar(
            np.eye(4), radar_from_lidar
        )
        np.testing.assert_allclose(actual, np.eye(4))

    def test_development_manifest_needs_no_release(self) -> None:
        self.assertIsNone(
            MODULE.validate_test_release(["train", "validation"], None)
        )

    def test_test_manifest_requires_completed_g4_summary(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            MODULE.validate_test_release(["test"], None)
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "g4.json"
            summary.write_text(json.dumps({"completed": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "remains sealed"):
                MODULE.validate_test_release(["test"], summary)

    def test_test_manifest_rejects_mixed_partitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "test-only"):
            MODULE.validate_test_release(["validation", "test"], None)

    def test_completed_g4_summary_releases_test_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "g4.json"
            summary.write_text(
                json.dumps(
                    {
                        "completed": True,
                        "source_commit": "abc",
                        "g4_passed": False,
                        "selected_arm": "T5",
                        "selected_fusion_mode": "cross_attention",
                        "comparison": "/runs/g4_comparison.json",
                    }
                ),
                encoding="utf-8",
            )
            release = MODULE.validate_test_release(["test"], summary)
        self.assertEqual(release["selected_arm"], "T5")
        self.assertFalse(release["g4_passed"])
        self.assertEqual(len(release["g4_summary_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
