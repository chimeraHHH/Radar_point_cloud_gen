"""Sequence-aware K-Radar temporal pairs over cached dense targets."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from cube_dense.dataset import KRadarCubeDataset


class KRadarTemporalDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cache_root: Path,
        temporal_manifest: Path,
        partitions: tuple[str, ...],
    ) -> None:
        manifest = json.loads(temporal_manifest.read_text(encoding="utf-8"))
        if manifest.get("gate_pass") is not True:
            raise ValueError("Temporal manifest did not pass its data gate")
        if manifest.get("checks", {}).get("radar_frame_ego_transforms_present") is not True:
            raise ValueError(
                "Temporal manifest lacks calibrated radar-frame ego transforms"
            )
        if len(set(partitions)) != len(partitions):
            raise ValueError("Duplicate temporal partitions")
        self.frame_dataset = KRadarCubeDataset(
            data_root, cache_root, temporal_manifest, partitions
        )
        frame_index = {
            (int(record["sequence"]), int(record["radar_index"])): index
            for index, record in enumerate(self.frame_dataset.records)
        }
        if len(frame_index) != len(self.frame_dataset.records):
            raise ValueError("Duplicate sequence/radar keys in temporal manifest")

        selected_windows = [
            window
            for window in manifest["windows"]
            if window["partition"] in partitions
        ]
        records_by_window: dict[str, list[dict]] = {
            window["window_id"]: [] for window in selected_windows
        }
        for record in manifest["frames"]:
            if record["partition"] in partitions:
                records_by_window[record["window_id"]].append(record)

        self.windows = []
        self.pairs = []
        for window in selected_windows:
            records = sorted(
                records_by_window[window["window_id"]],
                key=lambda record: int(record["frame_in_window"]),
            )
            expected_positions = list(range(int(window["frame_count"])))
            positions = [int(record["frame_in_window"]) for record in records]
            if positions != expected_positions:
                raise ValueError(f"Non-contiguous temporal window {window['window_id']}")
            keys = [
                (int(record["sequence"]), int(record["radar_index"]))
                for record in records
            ]
            try:
                dataset_indices = [frame_index[key] for key in keys]
            except KeyError as error:
                raise ValueError(
                    f"Window {window['window_id']} references an absent frame"
                ) from error
            self.windows.append(
                {
                    **window,
                    "dataset_indices": dataset_indices,
                    "frame_keys": keys,
                }
            )
            for previous, current in zip(records, records[1:]):
                delta_seconds = current["delta_seconds_from_previous"]
                if delta_seconds is None or float(delta_seconds) <= 0.0:
                    raise ValueError(
                        f"Invalid pair delta in window {window['window_id']}"
                    )
                self.pairs.append(
                    {
                        "window_id": window["window_id"],
                        "sequence": int(window["sequence"]),
                        "partition": window["partition"],
                        "previous_frame_in_window": int(previous["frame_in_window"]),
                        "current_frame_in_window": int(current["frame_in_window"]),
                        "previous_dataset_index": frame_index[
                            (int(previous["sequence"]), int(previous["radar_index"]))
                        ],
                        "current_dataset_index": frame_index[
                            (int(current["sequence"]), int(current["radar_index"]))
                        ],
                        "delta_seconds": float(delta_seconds),
                        "current_from_previous": current[
                            "current_radar_from_previous_radar"
                        ],
                    }
                )
        if not self.pairs:
            raise ValueError(f"No temporal pairs found for partitions {partitions}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        return {
            "previous": self.frame_dataset[pair["previous_dataset_index"]],
            "current": self.frame_dataset[pair["current_dataset_index"]],
            "current_from_previous": torch.tensor(
                pair["current_from_previous"], dtype=torch.float32
            ).reshape(4, 4),
            "delta_seconds": torch.tensor(pair["delta_seconds"], dtype=torch.float32),
            "window_id": pair["window_id"],
            "sequence": pair["sequence"],
            "partition": pair["partition"],
            "previous_frame_in_window": pair["previous_frame_in_window"],
            "current_frame_in_window": pair["current_frame_in_window"],
        }
