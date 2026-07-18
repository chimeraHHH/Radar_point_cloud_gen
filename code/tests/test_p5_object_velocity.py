from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_p5_object_velocity.py"
SPEC = importlib.util.spec_from_file_location("eval_p5_object_velocity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class P5ObjectVelocityTest(unittest.TestCase):
    @staticmethod
    def static_audit(
        hypothesis: str = "positive_ego",
        train_margin: float = 0.1,
        minimum_margin: float = 0.05,
        passed: bool = False,
    ) -> dict:
        return {
            "protocol": {
                "selection_partition": "train",
                "minimum_selection_margin_mps": minimum_margin,
            },
            "train": {
                "selected_hypothesis": hypothesis,
                "selected_margin_to_second_mps": train_margin,
            },
            "frozen_hypothesis": hypothesis,
            "checks": {
                "required_frame_count": True,
                "no_frame_errors": True,
                "train_hypothesis_meets_selection_margin": train_margin
                >= minimum_margin,
                "frozen_hypothesis_beats_random_on_validation": passed,
            },
            "passed": passed,
        }

    def test_failed_static_prior_can_supply_train_frozen_sign(self) -> None:
        calibration = MODULE.frozen_sign_calibration(self.static_audit())
        self.assertEqual(calibration["hypothesis"], "positive_ego")
        self.assertTrue(calibration["sign_only_calibration"])
        self.assertFalse(calibration["physics_prior_claim_enabled"])

    def test_sign_calibration_rejects_insufficient_train_margin(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection margin"):
            MODULE.frozen_sign_calibration(self.static_audit(train_margin=0.01))

    def test_sign_calibration_rejects_zero_centered_hypothesis(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            MODULE.frozen_sign_calibration(
                self.static_audit(hypothesis="zero_centered")
            )

    def test_alias_boundary_error_is_circular(self) -> None:
        error = MODULE.circular_error(-1.9, 1.9, 4.0)
        self.assertAlmostEqual(error, 0.2)

    def test_zero_margin_box_includes_boundary_only(self) -> None:
        box = {
            "center_xyz_m": np.zeros(3),
            "yaw_rad": 0.0,
            "half_size_xyz_m": np.asarray([1.0, 2.0, 3.0]),
        }
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0001, 0.0, 0.0]]
        )
        np.testing.assert_array_equal(
            MODULE.points_in_box(points, box), np.asarray([True, True, False])
        )

    def test_confidence_weighted_circular_distribution(self) -> None:
        doppler = np.asarray([-2.0, -1.0, 0.0, 1.0])
        probability = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        estimate, strength = MODULE.circular_object_estimate(
            probability, np.asarray([0.2, 0.8]), doppler, -2.0, 4.0
        )
        self.assertAlmostEqual(estimate, -2.0)
        self.assertAlmostEqual(strength, 1.0)

    def test_labels_are_translated_to_radar_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "label.txt"
            label.write_text(
                "* header\n*, 0, 7, Sedan, 1, 2, 3, 90, 2, 1, 0.5\n",
                encoding="utf-8",
            )
            boxes = MODULE.parse_boxes(label, np.asarray([0.5, -0.5, 0.7]))
        self.assertEqual(boxes[0]["track_id"], "7")
        np.testing.assert_allclose(boxes[0]["center_xyz_m"], [1.5, 1.5, 3.7])

    def test_object_geometry_is_exact_for_identical_points(self) -> None:
        xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target = np.column_stack((xyz, np.asarray([0.25, 1.0])))
        report = MODULE.object_geometry(xyz, target)
        self.assertIsNotNone(report)
        self.assertAlmostEqual(report["chamfer_m"], 0.0)
        self.assertAlmostEqual(report["fscore_1m"], 1.0)

    def test_scene_first_bootstrap_preserves_paired_improvement(self) -> None:
        observations = []
        for sequence in (10, 20):
            for seed in MODULE.FORMAL_SEEDS:
                common = {
                    "seed": seed,
                    "sequence": sequence,
                    "radar_index": 3,
                    "track_id": "1",
                    "prediction_mps": 0.0,
                }
                observations.append(
                    common
                    | {
                        "method": "T0",
                        "absolute_error_mps": 1.0,
                        "point_count": 5,
                    }
                )
                observations.append(
                    common
                    | {
                        "method": "T3",
                        "absolute_error_mps": 0.5,
                        "point_count": 15,
                    }
                )
        report = MODULE.paired_scene_bootstrap(
            observations, "T0", "T3", samples=100, random_seed=8
        )
        mae = report["mae_difference_mps_candidate_minus_reference"]
        support = report["support10_difference_candidate_minus_reference"]
        self.assertAlmostEqual(mae["estimate"], -0.5)
        self.assertEqual(mae["probability_candidate_better"], 1.0)
        self.assertAlmostEqual(support["estimate"], 1.0)


if __name__ == "__main__":
    unittest.main()
