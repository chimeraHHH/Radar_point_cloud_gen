from __future__ import annotations

import importlib.util
import unittest
from collections import defaultdict
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_p5_final.py"
SPEC = importlib.util.spec_from_file_location("compare_p5_final", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CompareP5FinalTest(unittest.TestCase):
    def test_lower_is_better_bootstrap_direction(self) -> None:
        grouped = defaultdict(lambda: defaultdict(list))
        for seed in MODULE.FORMAL_SEEDS:
            for scene in range(1, 9):
                grouped[seed][scene].append((1.0, 0.5))
        report = MODULE.summarize_groups(
            grouped,
            "lower",
            bootstrap_samples=100,
            rng=MODULE.np.random.default_rng(4),
        )
        self.assertAlmostEqual(report["improvement"], 0.5)
        self.assertEqual(report["probability_candidate_better"], 1.0)

    def test_failure_taxonomy_uses_correct_tail(self) -> None:
        methods = {}
        for seed in MODULE.FORMAL_SEEDS:
            frames = {}
            for index in range(6):
                frames[(1, index)] = {
                    "sequence": 1,
                    "radar_index": index,
                    "window_id": "seq01_w00",
                    "rollout_step": 10 + index,
                    "prediction": {"path": f"{index}.npz", "sha256": str(index)},
                    "current": {
                        "generated_geometry": {"chamfer_m": float(index)},
                        "cycle": {
                            "local_spectrum_kl": float(index),
                            "confidence_mean": float(index),
                            "covered_cell_count": float(index),
                        },
                        "doppler": {"static_pce_median_mps": float(index)},
                    },
                    "temporal": {"temporal_radial_error_mean_m": float(index)},
                }
            methods[seed] = {"T0": frames}
        report = MODULE.failure_taxonomy(methods)
        geometry = report[str(MODULE.FORMAL_SEEDS[0])]["T0"]["geometry_outlier"]
        confidence = report[str(MODULE.FORMAL_SEEDS[0])]["T0"]["confidence_collapse"]
        self.assertEqual(geometry[0]["value"], 5.0)
        self.assertEqual(confidence[0]["value"], 0.0)


if __name__ == "__main__":
    unittest.main()
