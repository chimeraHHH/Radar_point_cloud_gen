"""Read-only frozen point predictions used as temporal teacher history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class PointPrediction:
    xyz_m: torch.Tensor
    coordinates_rae: torch.Tensor
    probability: torch.Tensor
    confidence: torch.Tensor
    static_center_mps: torch.Tensor

    def detached(self) -> "PointPrediction":
        return PointPrediction(
            xyz_m=self.xyz_m.detach(),
            coordinates_rae=self.coordinates_rae.detach(),
            probability=self.probability.detach(),
            confidence=self.confidence.detach(),
            static_center_mps=self.static_center_mps.detach(),
        )


class FrozenPredictionCache:
    def __init__(self, root: Path, expected_frames: int | None = None) -> None:
        self.root = root
        self.manifest_path = root / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("completed") is not True:
            raise ValueError("Frozen parent prediction cache is incomplete")
        if expected_frames is not None and len(manifest["frames"]) != expected_frames:
            raise ValueError(
                f"Parent prediction cache has {len(manifest['frames'])} != "
                f"{expected_frames} frames"
            )
        self.configuration = manifest["configuration"]
        self.records = {
            (int(frame["sequence"]), int(frame["radar_index"])): frame
            for frame in manifest["frames"]
        }
        if len(self.records) != len(manifest["frames"]):
            raise ValueError("Duplicate parent prediction frame keys")

    def path(self, sequence: int, radar_index: int) -> Path:
        try:
            record = self.records[(sequence, radar_index)]
        except KeyError as error:
            raise KeyError(
                f"Missing frozen prediction seq={sequence} radar={radar_index}"
            ) from error
        path = Path(record["prediction"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def load(
        self,
        sequence: int,
        radar_index: int,
        device: torch.device,
    ) -> PointPrediction:
        path = self.path(sequence, radar_index)
        with np.load(path) as cache:
            probability = torch.from_numpy(
                cache["doppler_probability"].astype(np.float32)
            ).to(device)
            probability = probability / probability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            return PointPrediction(
                xyz_m=torch.from_numpy(cache["xyz_m"].astype(np.float32)).to(device),
                coordinates_rae=torch.from_numpy(
                    cache["coordinates_rae"].astype(np.float32)
                ).to(device),
                probability=probability,
                confidence=torch.from_numpy(
                    cache["confidence"].astype(np.float32)
                ).to(device),
                static_center_mps=torch.from_numpy(
                    cache["static_center_mps"].astype(np.float32)
                ).to(device),
            )


def prediction_from_output(
    prediction: dict[str, torch.Tensor],
    confidence: torch.Tensor,
    static_center_mps: torch.Tensor,
) -> PointPrediction:
    return PointPrediction(
        xyz_m=prediction["xyz_m"],
        coordinates_rae=prediction["coordinates_rae"],
        probability=prediction["probability"],
        confidence=confidence,
        static_center_mps=static_center_mps,
    )
