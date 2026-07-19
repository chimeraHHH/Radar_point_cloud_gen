"""Frozen RaLD point predictions used as convention-free temporal history."""

from __future__ import annotations

import hashlib
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
        self.point_count = int(self.configuration["point_count"])
        if self.point_count <= 0:
            raise ValueError("Frozen RaLD prediction point count must be positive")
        self.records = {
            (int(frame["sequence"]), int(frame["radar_index"])): frame
            for frame in manifest["frames"]
        }
        if len(self.records) != len(manifest["frames"]):
            raise ValueError("Duplicate RaLD prediction frame keys")
        self._verified_paths: set[Path] = set()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
        expected_hash = record.get("prediction_sha256")
        if (
            path not in self._verified_paths
            and (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or self._sha256(path) != expected_hash
            )
        ):
            raise ValueError(f"Frozen RaLD prediction hash differs: {path}")
        self._verified_paths.add(path)
        return path

    def load(
        self,
        sequence: int,
        radar_index: int,
        device: torch.device,
    ) -> RaLDPointPrediction:
        path = self.path(sequence, radar_index)
        with np.load(path, allow_pickle=False) as cache:
            required = {"xyz_m", "coordinates_rae", "doppler_probability", "confidence"}
            if not required.issubset(cache.files):
                raise ValueError(f"Frozen RaLD prediction schema differs: {path}")
            arrays = {name: np.asarray(cache[name]) for name in required}
            expected_shapes = {
                "xyz_m": (self.point_count, 3),
                "coordinates_rae": (self.point_count, 3),
                "doppler_probability": (self.point_count, 64),
                "confidence": (self.point_count,),
            }
            if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
                raise ValueError(f"Frozen RaLD prediction shapes differ: {path}")
            if any(not np.isfinite(array).all() for array in arrays.values()):
                raise ValueError(f"Frozen RaLD prediction contains non-finite values: {path}")
            if (
                np.any(arrays["doppler_probability"] < 0.0)
                or np.any(arrays["doppler_probability"].sum(axis=1) <= 0.0)
            ):
                raise ValueError(f"Frozen RaLD prediction probability is invalid: {path}")
            if np.any(arrays["confidence"] < 0.0) or np.any(arrays["confidence"] > 1.0):
                raise ValueError(f"Frozen RaLD prediction confidence is invalid: {path}")
            probability = torch.from_numpy(
                arrays["doppler_probability"].astype(np.float32)
            ).to(device)
            probability = probability / probability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            return RaLDPointPrediction(
                xyz_m=torch.from_numpy(arrays["xyz_m"].astype(np.float32)).to(device),
                coordinates_rae=torch.from_numpy(
                    arrays["coordinates_rae"].astype(np.float32)
                ).to(device),
                probability=probability,
                confidence=torch.from_numpy(
                    arrays["confidence"].astype(np.float32)
                ).to(device),
            )


def rald_prediction_from_output(output: dict[str, torch.Tensor]) -> RaLDPointPrediction:
    return RaLDPointPrediction(
        xyz_m=output["xyz_m"][0],
        coordinates_rae=output["coordinates_rae"][0],
        probability=output["doppler_probability"][0],
        confidence=output["confidence"][0],
    )
