from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


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
