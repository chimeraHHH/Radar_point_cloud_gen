#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from queue_g1_preflight import wait_for_g0  # noqa: E402


class QueueG1PreflightTest(unittest.TestCase):
    def test_partial_report_waits_for_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "g0_audit.json"
            report_path.write_text(
                json.dumps({"frames": [{"sequence": 1}]}), encoding="utf-8"
            )

            def finish_report(_seconds: int) -> None:
                report_path.write_text(
                    json.dumps(
                        {
                            "frames": [{"sequence": 1}] * 100,
                            "aggregate": {
                                "successful_frames": 100,
                                "failed_frames": 0,
                                "gate_pass": True,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            with patch("queue_g1_preflight.time.sleep", side_effect=finish_report):
                report = wait_for_g0(report_path, required_frames=100, poll_seconds=1)

            self.assertTrue(report["aggregate"]["gate_pass"])


if __name__ == "__main__":
    unittest.main()
