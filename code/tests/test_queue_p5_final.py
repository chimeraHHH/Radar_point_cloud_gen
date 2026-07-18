from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "queue_p5_final.py"
SPEC = importlib.util.spec_from_file_location("queue_p5_final", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QueueP5FinalTest(unittest.TestCase):
    @staticmethod
    def summary(completed: bool = True) -> dict:
        seeds = MODULE.FORMAL_SEEDS
        return {
            "completed": completed,
            "source_commit": "abc",
            "g4_passed": False,
            "selected_arm": "T5",
            "selected_fusion_mode": "cross_attention",
            "parent_summary": "/runs/g2_g3.json",
            "preflight_selection": "/runs/preflight.json",
            "comparison": "/runs/g4_compare.json",
            "formal_runs": {str(seed): f"/runs/t5_{seed}" for seed in seeds},
            "baseline_reports": {
                str(seed): f"/runs/baseline_{seed}.json" for seed in seeds
            },
            "rollout_reports": {
                str(seed): f"/runs/rollout_{seed}.json" for seed in seeds
            },
        }

    def test_failed_but_completed_g4_releases_frozen_family(self) -> None:
        release = MODULE.validate_g4_release(self.summary(), MODULE.FORMAL_SEEDS)
        self.assertFalse(release["g4_passed"])
        self.assertEqual(release["selected_arm"], "T5")
        self.assertEqual(set(release["formal_runs"]), set(MODULE.FORMAL_SEEDS))

    def test_incomplete_g4_keeps_test_sealed(self) -> None:
        with self.assertRaisesRegex(ValueError, "remains sealed"):
            MODULE.validate_g4_release(
                self.summary(completed=False), MODULE.FORMAL_SEEDS
            )

    def test_missing_seed_is_rejected(self) -> None:
        summary = self.summary()
        summary["formal_runs"].pop(str(MODULE.FORMAL_SEEDS[-1]))
        with self.assertRaisesRegex(ValueError, "three formal seeds"):
            MODULE.validate_g4_release(summary, MODULE.FORMAL_SEEDS)

    def test_incomplete_directory_is_archived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "object"
            root.mkdir()
            (root / "observations.jsonl").write_text("partial\n", encoding="utf-8")
            MODULE.archive_incomplete_directory(root, root / "report.json", "abc")
            archived = list(Path(directory).glob("object.incomplete.*"))
            self.assertFalse(root.exists())
            self.assertEqual(len(archived), 1)


if __name__ == "__main__":
    unittest.main()
