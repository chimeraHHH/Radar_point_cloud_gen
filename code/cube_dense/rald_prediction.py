"""Frozen RaLD point predictions used as convention-free temporal history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class RaLDPointPrediction:
    xyz_m: torch.Tensor
    coordinates_rae: torch.Tensor
    probability: torch.Tensor
    confidence: torch.Tensor

    def detached(self) -> "RaLDPointPrediction":
        return RaLDPointPrediction(
            xyz_m=self.xyz_m.detach(),
            coordinates_rae=self.coordinates_rae.detach(),
            probability=self.probability.detach(),
            confidence=self.confidence.detach(),
        )


class FrozenRaLDPredictionCache:
    def __init__(self, root: Path, expected_frames: int | None = None) -> None:
        self.root = root.resolve()
        self.manifest_path = self.root / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("completed") is not True:
            raise ValueError("Frozen RaLD prediction cache is incomplete")
        if expected_frames is not None and len(manifest["frames"]) != expected_frames:
            raise ValueError(
                f"RaLD prediction cache has {len(manifest['frames'])} != "
                f"{expected_frames} frames"
            )
        self.configuration = manifest["configuration"]
        self.records = {
            (int(frame["sequence"]), int(frame["radar_index"])): frame
            for frame in manifest["frames"]
        }
        if len(self.records) != len(manifest["frames"]):
            raise ValueError("Duplicate RaLD prediction frame keys")

    def path(self, sequence: int, radar_index: int) -> Path:
        try:
            record = self.records[(sequence, radar_index)]
        except KeyError as error:
            raise KeyError(
                f"Missing RaLD prediction seq={sequence} radar={radar_index}"
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
    ) -> RaLDPointPrediction:
        path = self.path(sequence, radar_index)
        with np.load(path) as cache:
            probability = torch.from_numpy(
                cache["doppler_probability"].astype(np.float32)
            ).to(device)
            probability = probability / probability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            return RaLDPointPrediction(
                xyz_m=torch.from_numpy(cache["xyz_m"].astype(np.float32)).to(device),
                coordinates_rae=torch.from_numpy(
                    cache["coordinates_rae"].astype(np.float32)
                ).to(device),
                probability=probability,
                confidence=torch.from_numpy(
                    cache["confidence"].astype(np.float32)
                ).to(device),
            )


def rald_prediction_from_output(output: dict[str, torch.Tensor]) -> RaLDPointPrediction:
    return RaLDPointPrediction(
        xyz_m=output["xyz_m"][0],
        coordinates_rae=output["coordinates_rae"][0],
        probability=output["doppler_probability"][0],
        confidence=output["confidence"][0],
    )
