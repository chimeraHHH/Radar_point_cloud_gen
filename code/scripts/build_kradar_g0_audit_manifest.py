#!/usr/bin/env python3
"""Select a deterministic cross-scene K-Radar G0 audit cohort."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_sensor_times(path: Path) -> dict[int, float]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for name, timestamp, *_ in csv.reader(handle):
            index = int(name.rsplit("_", maxsplit=1)[1].split(".", maxsplit=1)[0])
            result[index] = float(timestamp)
    return result


def label_indices(label: str) -> tuple[int, int]:
    radar, lidar = label.removesuffix(".txt").split("_", maxsplit=1)
    return int(radar), int(lidar)


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count > length:
        raise ValueError(f"Cannot select {count} entries from a sequence of {length}")
    values = np.linspace(0, length - 1, count + 2)[1:-1]
    indices = [int(round(value)) for value in values]
    if len(set(indices)) != count:
        raise RuntimeError(f"Even selection produced duplicate indices: {indices}")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--odometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--partitions", nargs="+", default=["train", "validation"]
    )
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    sequence_partition = {}
    for partition in args.partitions:
        for sequence in split["splits"][partition]["sequences"]:
            if sequence in sequence_partition:
                raise RuntimeError(f"Sequence {sequence} appears in multiple partitions")
            sequence_partition[sequence] = partition
    sequences = sorted(sequence_partition)
    if args.frames < len(sequences):
        raise ValueError("Audit frame count must cover every selected sequence")

    base_count, extra_count = divmod(args.frames, len(sequences))
    priority = sorted(
        sequences,
        key=lambda sequence: (
            sequence_partition[sequence] != "validation",
            -len(
                split["splits"][sequence_partition[sequence]]["labels"][str(sequence)]
            ),
            sequence,
        ),
    )
    requested = {sequence: base_count for sequence in sequences}
    for sequence in priority[:extra_count]:
        requested[sequence] += 1

    records = []
    odometry_checks = {}
    for sequence in sequences:
        partition = sequence_partition[sequence]
        labels = split["splits"][partition]["labels"][str(sequence)]
        labels = sorted(labels, key=lambda label: label_indices(label)[1])
        pose_path = args.odometry_root / f"gt_{sequence:02d}.txt"
        poses = np.loadtxt(pose_path)
        if poses.ndim == 1:
            poses = poses[None, :]
        if poses.shape != (len(labels), 12):
            raise ValueError(
                f"Sequence {sequence}: {poses.shape[0]} poses for {len(labels)} labels"
            )
        times = read_sensor_times(
            args.metadata_root / str(sequence) / "time_info" / "os2-64.txt"
        )
        odometry_checks[str(sequence)] = {
            "label_count": len(labels),
            "pose_count": int(poses.shape[0]),
            "one_to_one": True,
        }
        for pose_index in evenly_spaced_indices(len(labels), requested[sequence]):
            label = labels[pose_index]
            radar_index, lidar_index = label_indices(label)
            records.append(
                {
                    "sequence": sequence,
                    "partition": partition,
                    "description": split["sequence_descriptions"][str(sequence)],
                    "label": label,
                    "radar_index": radar_index,
                    "lidar64_index": lidar_index,
                    "timestamp": times[lidar_index],
                    "odometry_pose_index": pose_index,
                }
            )

    tag_counts = Counter(
        tag for record in records for tag in record["description"]
    )
    partition_counts = Counter(record["partition"] for record in records)
    sequence_counts = Counter(record["sequence"] for record in records)
    unique_keys = {(record["sequence"], record["label"]) for record in records}
    checks = {
        "required_frame_count": len(records) == args.frames,
        "all_selected_sequences_covered": set(sequence_counts) == set(sequences),
        "unique_frames": len(unique_keys) == len(records),
        "test_partition_untouched": "test" not in partition_counts,
        "odometry_one_to_one": all(
            item["one_to_one"] for item in odometry_checks.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Audit manifest checks failed: {checks}")
    payload = {
        "protocol": "cross-scene train/validation G0 audit; test remains untouched",
        "selection": {
            "method": "evenly spaced interior labels per sequence",
            "requested_frames": args.frames,
            "partitions": args.partitions,
            "base_frames_per_sequence": base_count,
            "extra_sequence_priority": "validation first, then sequence frame count",
        },
        "summary": {
            "frame_count": len(records),
            "sequence_count": len(sequence_counts),
            "frame_count_by_partition": dict(sorted(partition_counts.items())),
            "frame_count_by_tag": dict(sorted(tag_counts.items())),
            "frames_per_sequence": {
                str(sequence): sequence_counts[sequence]
                for sequence in sorted(sequence_counts)
            },
        },
        "checks": checks,
        "odometry_checks": odometry_checks,
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
