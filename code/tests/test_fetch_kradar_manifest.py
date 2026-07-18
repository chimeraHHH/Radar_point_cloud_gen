from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_kradar_g0_manifest.py"
try:
    import remotezip  # noqa: F401
except ModuleNotFoundError:
    sys.modules["remotezip"] = types.SimpleNamespace(RemoteZip=object)
SPEC = importlib.util.spec_from_file_location("fetch_kradar_g0_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FetchKradarManifestTest(unittest.TestCase):
    def test_summary_is_atomic_and_keeps_global_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            report = MODULE.write_summary(
                path,
                requested={1, 2, 3},
                completed={1},
                failures={2: "network"},
                active={3},
                round_index=4,
            )
            stored = json.loads(path.read_text(encoding="utf-8"))
            temporary_exists = path.with_suffix(".json.tmp").exists()
        self.assertEqual(report, stored)
        self.assertEqual(stored["completed_sequences"], [1])
        self.assertEqual(stored["pending_sequences"], [2, 3])
        self.assertEqual(stored["active_sequences"], [3])
        self.assertEqual(stored["failures"], [{"sequence": 2, "error": "network"}])
        self.assertFalse(temporary_exists)

    def test_completed_sequence_cannot_remain_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = MODULE.write_summary(
                Path(directory) / "summary.json",
                requested={1},
                completed={1},
                failures={1: "stale"},
                active=set(),
                round_index=2,
            )
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["pending_sequences"], [])


if __name__ == "__main__":
    unittest.main()
