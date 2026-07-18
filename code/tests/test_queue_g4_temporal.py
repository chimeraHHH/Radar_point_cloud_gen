from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "queue_g4_temporal.py"
SPEC = importlib.util.spec_from_file_location("queue_g4_temporal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QueueG4TemporalTest(unittest.TestCase):
    def test_failed_download_summary_waits_for_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "completed_sequences": [1],
                        "failures": [{"sequence": 2, "error": "network"}],
                    }
                ),
                encoding="utf-8",
            )
            sleeps = []

            def recover(_seconds: int) -> None:
                sleeps.append(1)
                summary.write_text(
                    json.dumps(
                        {"completed_sequences": [1, 2], "failures": []}
                    ),
                    encoding="utf-8",
                )

            original_sleep = MODULE.time.sleep
            original_emit = MODULE.emit
            MODULE.time.sleep = recover
            MODULE.emit = lambda *_args, **_kwargs: None
            try:
                result = MODULE.wait_for_download_completion(summary, {1, 2}, 1)
            finally:
                MODULE.time.sleep = original_sleep
                MODULE.emit = original_emit
        self.assertEqual(sleeps, [1])
        self.assertEqual(result["completed_sequences"], [1, 2])


if __name__ == "__main__":
    unittest.main()
