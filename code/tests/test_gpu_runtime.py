#!/usr/bin/env python3

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from gpu_runtime import cuda_environment, validate_gpu_candidates  # noqa: E402


class GpuRuntimeTest(unittest.TestCase):
    def test_cuda_environment_uses_physical_pci_order(self) -> None:
        environment = cuda_environment(2, {"PATH": os.environ.get("PATH", "")})
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")

    @patch("gpu_runtime.gpu_name", side_effect=["NVIDIA H200 NVL", "NVIDIA H200 NVL"])
    def test_required_model_accepts_h200_candidates(self, _gpu_name) -> None:
        validate_gpu_candidates([0, 2], "NVIDIA H200 NVL")

    @patch("gpu_runtime.gpu_name", return_value="NVIDIA RTX PRO 5000 Blackwell")
    def test_required_model_rejects_other_gpu(self, _gpu_name) -> None:
        with self.assertRaisesRegex(ValueError, "NVIDIA H200 NVL"):
            validate_gpu_candidates([1], "NVIDIA H200 NVL")


if __name__ == "__main__":
    unittest.main()
