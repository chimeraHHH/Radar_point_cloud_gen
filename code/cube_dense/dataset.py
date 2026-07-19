"""K-Radar Cube and cached radar-observable target dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cube_dense.kradar import load_tesseract


def _manifest_records(
    audit_manifest: Path, partitions: tuple[str, ...]
) -> list[dict]:
    manifest = json.loads(audit_manifest.read_text(encoding="utf-8"))
    records = [
        record for record in manifest["frames"] if record["partition"] in partitions
    ]
    if not records:
        raise ValueError(f"No records found for partitions {partitions}")
    return records


def _cache_path(cache_root: Path, sequence: int, radar_index: int) -> Path:
    return cache_root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


class KRadarCubeDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cache_root: Path,
        audit_manifest: Path,
        partitions: tuple[str, ...],
    ) -> None:
        self.data_root = data_root
        self.cache_root = cache_root
        self.records = _manifest_records(audit_manifest, partitions)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sequence = int(record["sequence"])
        radar_index = int(record["radar_index"])
        cube = load_tesseract(
            self.data_root
            / str(sequence)
            / "radar_tesseract"
            / f"tesseract_{radar_index:05d}.mat"
        ).astype(np.float32, copy=False)
        cache_path = _cache_path(self.cache_root, sequence, radar_index)
        with np.load(cache_path) as cache:
            target = cache["target_xyz_confidence"].astype(np.float32)
            target_index = cache["target_rae_index"].astype(np.int64)
            cfar = cache["cfar_xyzd_power_snr"].astype(np.float32)
            ego_velocity = cache["ego_velocity_xyz_mps"].astype(np.float32)
            ego_speed = cache["ego_speed_mps"].astype(np.float32)
            ego_yaw_rate = cache["ego_yaw_rate_radps"].astype(np.float32)
        occupancy = np.zeros((256, 107, 37), dtype=np.float32)
        np.maximum.at(
            occupancy,
            (target_index[:, 0], target_index[:, 1], target_index[:, 2]),
            target[:, 3],
        )
        return {
            "cube_drae": torch.from_numpy(cube),
            "occupancy": torch.from_numpy(occupancy),
            "target_xyz_confidence": torch.from_numpy(target),
            "target_rae_index": torch.from_numpy(target_index),
            "cfar_xyzd_power_snr": torch.from_numpy(cfar),
            "ego_velocity_xyz_mps": torch.from_numpy(ego_velocity),
            "ego_speed_mps": torch.from_numpy(ego_speed),
            "ego_yaw_rate_radps": torch.from_numpy(ego_yaw_rate),
            "sequence": sequence,
            "radar_index": radar_index,
            "partition": record["partition"],
        }


class KRadarDenseTargetDataset(Dataset):
    """Read cached radar-observable targets without loading the full Cube."""

    def __init__(
        self,
        cache_root: Path,
        audit_manifest: Path,
        partitions: tuple[str, ...],
    ) -> None:
        self.cache_root = cache_root
        self.records = _manifest_records(audit_manifest, partitions)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sequence = int(record["sequence"])
        radar_index = int(record["radar_index"])
        with np.load(_cache_path(self.cache_root, sequence, radar_index)) as cache:
            target = cache["target_xyz_confidence"].astype(np.float32)
            target_index = cache["target_rae_index"].astype(np.int64)
        return {
            "target_xyz_confidence": torch.from_numpy(target),
            "target_rae_index": torch.from_numpy(target_index),
            "sequence": sequence,
            "radar_index": radar_index,
            "partition": record["partition"],
        }
