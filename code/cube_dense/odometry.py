"""Official K-Radar pose trajectory utilities for sequence-level protocols."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def label_lidar_index(label: str) -> int:
    return int(label.removesuffix(".txt").split("_", maxsplit=1)[1])


def read_os2_times(path: Path) -> dict[int, float]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for name, timestamp, *_ in csv.reader(handle):
            index = int(name.rsplit("_", maxsplit=1)[1].split(".", maxsplit=1)[0])
            if index in result:
                raise ValueError(f"Duplicate OS2 timestamp index {index} in {path}")
            result[index] = float(timestamp)
    return result


def load_pose_trajectory(
    path: Path,
    labels: list[str],
    os2_times: dict[int, float],
) -> dict[str, np.ndarray]:
    ordered_labels = sorted(labels, key=label_lidar_index)
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    if values.shape != (len(ordered_labels), 12):
        raise ValueError(
            f"Odometry/label mismatch in {path}: {values.shape} vs "
            f"{len(ordered_labels)} labels"
        )
    poses = values.reshape(-1, 3, 4)
    timestamp = np.asarray(
        [os2_times[label_lidar_index(label)] for label in ordered_labels],
        dtype=np.float64,
    )
    if np.any(np.diff(timestamp) <= 0.0):
        raise ValueError(f"Non-increasing pose timestamps in {path}")
    position = poses[:, :, 3]
    heading = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    velocity = np.column_stack(
        [np.gradient(position[:, axis], timestamp) for axis in range(3)]
    )
    yaw_rate = np.gradient(heading, timestamp)
    return {
        "timestamp": timestamp,
        "position": position,
        "velocity": velocity,
        "heading": heading,
        "yaw_rate": yaw_rate,
    }


def interpolate_motion(trajectory: dict[str, np.ndarray], timestamp: float) -> dict:
    time_axis = trajectory["timestamp"]
    nearest = int(np.argmin(np.abs(time_axis - timestamp)))
    velocity = np.asarray(
        [
            np.interp(timestamp, time_axis, trajectory["velocity"][:, axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    return {
        "velocity_xyz_mps": velocity.tolist(),
        "speed_mps": float(np.linalg.norm(velocity)),
        "yaw_rate_radps": float(
            np.interp(timestamp, time_axis, trajectory["yaw_rate"])
        ),
        "nearest_timestamp_delta_ms": float(
            abs(time_axis[nearest] - timestamp) * 1e3
        ),
    }
