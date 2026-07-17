#!/usr/bin/env python3
"""Build deterministic, sequence-isolated K-Radar temporal windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_indices(label: str) -> tuple[int, int]:
    radar, lidar = label.removesuffix(".txt").split("_", maxsplit=1)
    return int(radar), int(lidar)


def read_sensor_times(path: Path) -> dict[int, float]:
    times = {}
    with path.open("r", encoding="utf-8") as handle:
        for name, timestamp, *_ in csv.reader(handle):
            index = int(name.rsplit("_", maxsplit=1)[1].split(".", maxsplit=1)[0])
            if index in times:
                raise ValueError(f"Duplicate sensor timestamp index {index} in {path}")
            times[index] = float(timestamp)
    return times


def homogeneous_pose(flat_pose: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3] = flat_pose.reshape(3, 4)
    return pose


def centered_window_starts(
    frame_count: int,
    window_length: int,
    windows_per_sequence: int,
) -> list[int]:
    occupied = window_length * windows_per_sequence
    if occupied > frame_count:
        raise ValueError(
            f"Cannot fit {windows_per_sequence} windows of {window_length} "
            f"frames into {frame_count} frames"
        )
    free = frame_count - occupied
    gaps = np.linspace(0.0, float(free), windows_per_sequence + 2)[1:-1]
    starts = [
        int(round(gap)) + window_index * window_length
        for window_index, gap in enumerate(gaps)
    ]
    for first, second in zip(starts, starts[1:]):
        if first + window_length > second:
            raise RuntimeError(f"Generated overlapping temporal windows: {starts}")
    return starts


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--odometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=48)
    parser.add_argument("--windows-per-sequence", type=int, default=1)
    parser.add_argument(
        "--partitions", nargs="+", default=["train", "validation"]
    )
    parser.add_argument("--minimum-delta-seconds", type=float, default=0.05)
    parser.add_argument("--maximum-delta-seconds", type=float, default=0.15)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--estimated-cube-bytes-per-frame", type=int, default=292_773_000
    )
    parser.add_argument(
        "--estimated-lidar-bytes-per-frame", type=int, default=5_872_866
    )
    args = parser.parse_args()
    if args.window_length < 2:
        raise ValueError("Temporal windows require at least two frames")
    if args.windows_per_sequence < 1:
        raise ValueError("--windows-per-sequence must be positive")
    if len(set(args.partitions)) != len(args.partitions):
        raise ValueError("Duplicate requested partitions")
    if not 0.0 < args.minimum_delta_seconds < args.maximum_delta_seconds:
        raise ValueError("Invalid timestamp-delta interval")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    if split.get("gate_pass") is not True:
        raise ValueError("Input scene split did not pass its leakage gate")
    unknown_partitions = sorted(set(args.partitions) - set(split["splits"]))
    if unknown_partitions:
        raise ValueError(f"Unknown partitions: {unknown_partitions}")

    sequence_partition = {}
    for partition in args.partitions:
        for sequence in split["splits"][partition]["sequences"]:
            if sequence in sequence_partition:
                raise ValueError(f"Sequence {sequence} appears in multiple partitions")
            sequence_partition[int(sequence)] = partition

    records = []
    windows = []
    all_delta_seconds = []
    input_file_hashes = {}
    maximum_rotation_orthogonality_error = 0.0
    minimum_rotation_determinant = float("inf")
    for sequence, partition in sorted(sequence_partition.items()):
        labels = sorted(
            split["splits"][partition]["labels"][str(sequence)],
            key=lambda label: label_indices(label)[1],
        )
        lidar_indices = [label_indices(label)[1] for label in labels]
        radar_indices = [label_indices(label)[0] for label in labels]
        if any(second <= first for first, second in zip(lidar_indices, lidar_indices[1:])):
            raise ValueError(f"Sequence {sequence} LiDAR indices are not increasing")
        if any(second <= first for first, second in zip(radar_indices, radar_indices[1:])):
            raise ValueError(f"Sequence {sequence} radar indices are not increasing")

        time_path = args.metadata_root / str(sequence) / "time_info" / "os2-64.txt"
        odometry_path = args.odometry_root / f"gt_{sequence:02d}.txt"
        times = read_sensor_times(time_path)
        flat_poses = np.loadtxt(odometry_path)
        input_file_hashes[str(sequence)] = {
            "os2_time_sha256": sha256(time_path),
            "odometry_sha256": sha256(odometry_path),
        }
        if flat_poses.ndim == 1:
            flat_poses = flat_poses[None]
        if flat_poses.shape != (len(labels), 12):
            raise ValueError(
                f"Sequence {sequence}: {flat_poses.shape} poses for {len(labels)} labels"
            )
        poses = np.stack([homogeneous_pose(row) for row in flat_poses])
        for pose in poses:
            rotation = pose[:3, :3]
            error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
            determinant = float(np.linalg.det(rotation))
            maximum_rotation_orthogonality_error = max(
                maximum_rotation_orthogonality_error, error
            )
            minimum_rotation_determinant = min(minimum_rotation_determinant, determinant)

        starts = centered_window_starts(
            len(labels), args.window_length, args.windows_per_sequence
        )
        for window_index, start in enumerate(starts):
            stop = start + args.window_length
            window_labels = labels[start:stop]
            window_times = np.asarray(
                [times[label_indices(label)[1]] for label in window_labels],
                dtype=np.float64,
            )
            delta_seconds = np.diff(window_times)
            all_delta_seconds.extend(delta_seconds.tolist())
            window_poses = poses[start:stop]
            relative_window = np.linalg.inv(window_poses[0]) @ window_poses[-1]
            window_id = f"seq{sequence:02d}_w{window_index:02d}"
            windows.append(
                {
                    "window_id": window_id,
                    "sequence": sequence,
                    "partition": partition,
                    "start_pose_index": start,
                    "stop_pose_index_exclusive": stop,
                    "frame_count": len(window_labels),
                    "start_label": window_labels[0],
                    "end_label": window_labels[-1],
                    "duration_seconds": float(window_times[-1] - window_times[0]),
                    "ego_translation_m": float(
                        np.linalg.norm(relative_window[:3, 3])
                    ),
                    "ego_rotation_rad": rotation_angle(relative_window[:3, :3]),
                }
            )
            for frame_in_window, (pose_index, label, timestamp, pose) in enumerate(
                zip(
                    range(start, stop),
                    window_labels,
                    window_times,
                    window_poses,
                )
            ):
                radar_index, lidar_index = label_indices(label)
                if frame_in_window == 0:
                    previous_to_current = np.eye(4, dtype=np.float64)
                    delta_from_previous = None
                else:
                    previous_to_current = np.linalg.inv(pose) @ window_poses[
                        frame_in_window - 1
                    ]
                    delta_from_previous = float(delta_seconds[frame_in_window - 1])
                records.append(
                    {
                        "sequence": sequence,
                        "partition": partition,
                        "description": split["sequence_descriptions"][str(sequence)],
                        "window_id": window_id,
                        "window_index": window_index,
                        "frame_in_window": frame_in_window,
                        "label": label,
                        "radar_index": radar_index,
                        "lidar64_index": lidar_index,
                        "timestamp": float(timestamp),
                        "odometry_pose_index": pose_index,
                        "world_from_lidar64": pose.reshape(-1).tolist(),
                        "current_lidar64_from_previous_lidar64": (
                            previous_to_current.reshape(-1).tolist()
                        ),
                        "delta_seconds_from_previous": delta_from_previous,
                    }
                )

    frame_keys = [(record["sequence"], record["label"]) for record in records]
    member_keys = [
        (record["sequence"], record["radar_index"], record["lidar64_index"])
        for record in records
    ]
    windows_by_partition = Counter(window["partition"] for window in windows)
    frames_by_partition = Counter(record["partition"] for record in records)
    frames_by_sequence = Counter(record["sequence"] for record in records)
    expected_frames = (
        len(sequence_partition) * args.windows_per_sequence * args.window_length
    )
    all_delta = np.asarray(all_delta_seconds, dtype=np.float64)
    partition_sequences = {
        partition: {
            record["sequence"]
            for record in records
            if record["partition"] == partition
        }
        for partition in args.partitions
    }
    overlaps = {
        f"{first}_{second}": sorted(partition_sequences[first] & partition_sequences[second])
        for first_index, first in enumerate(args.partitions)
        for second in args.partitions[first_index + 1 :]
    }
    checks = {
        "scene_split_gate_passed": split["gate_pass"] is True,
        "required_frame_count": len(records) == expected_frames,
        "required_window_count": len(windows)
        == len(sequence_partition) * args.windows_per_sequence,
        "every_sequence_has_required_frames": all(
            count == args.window_length * args.windows_per_sequence
            for count in frames_by_sequence.values()
        ),
        "unique_frames": len(set(frame_keys)) == len(frame_keys),
        "unique_sensor_triplets": len(set(member_keys)) == len(member_keys),
        "timestamp_delta_in_range": bool(
            all_delta.size
            and all_delta.min() >= args.minimum_delta_seconds
            and all_delta.max() <= args.maximum_delta_seconds
        ),
        "rotation_matrices_valid": maximum_rotation_orthogonality_error <= 1e-3
        and minimum_rotation_determinant >= 0.999,
        "zero_sequence_overlap": not any(overlaps.values()),
        "test_partition_untouched": "test" not in args.partitions,
    }
    payload = {
        "protocol": "centered continuous K-Radar windows with sequence-isolated partitions",
        "source_commit": args.source_commit,
        "source_split": str(args.split),
        "source_split_sha256": sha256(args.split),
        "input_file_hashes": input_file_hashes,
        "selection": {
            "method": "centered non-overlapping windows; no model or target metric used",
            "partitions": args.partitions,
            "window_length": args.window_length,
            "windows_per_sequence": args.windows_per_sequence,
            "minimum_delta_seconds": args.minimum_delta_seconds,
            "maximum_delta_seconds": args.maximum_delta_seconds,
        },
        "summary": {
            "frame_count": len(records),
            "window_count": len(windows),
            "sequence_count": len(frames_by_sequence),
            "frame_count_by_partition": dict(sorted(frames_by_partition.items())),
            "window_count_by_partition": dict(sorted(windows_by_partition.items())),
            "frames_per_sequence": {
                str(sequence): frames_by_sequence[sequence]
                for sequence in sorted(frames_by_sequence)
            },
            "delta_seconds_min": float(all_delta.min()),
            "delta_seconds_median": float(np.median(all_delta)),
            "delta_seconds_max": float(all_delta.max()),
            "estimated_cube_bytes": len(records) * args.estimated_cube_bytes_per_frame,
            "estimated_lidar_bytes": len(records) * args.estimated_lidar_bytes_per_frame,
            "estimated_total_gib": (
                len(records)
                * (
                    args.estimated_cube_bytes_per_frame
                    + args.estimated_lidar_bytes_per_frame
                )
                / (1024**3)
            ),
        },
        "leakage_audit": {
            "sequence_overlap": overlaps,
            "adjacent_frame_cross_partition_possible": False,
        },
        "pose_audit": {
            "maximum_rotation_orthogonality_error": maximum_rotation_orthogonality_error,
            "minimum_rotation_determinant": minimum_rotation_determinant,
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
        "windows": windows,
        "frames": records,
    }
    if not payload["gate_pass"]:
        raise RuntimeError(f"Temporal manifest checks failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
