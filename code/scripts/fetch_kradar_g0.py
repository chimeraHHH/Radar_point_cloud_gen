#!/usr/bin/env python3
"""Fetch a minimal, synchronized K-Radar G0 audit set from the official NAS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar_archive import (  # noqa: E402
    SynologySession,
    credentials_from_environment,
    fetch_members,
)
from cube_dense.kradar import parse_label_header  # noqa: E402


DEFAULT_LABELS = [
    f"{radar:05d}_{lidar:05d}.txt"
    for radar, lidar in zip(range(33, 41), range(1, 9))
]


def build_metadata_members(sequence: int, labels: list[str]) -> list[str]:
    prefix = f"{sequence}/"
    return [
        f"{prefix}description.txt",
        f"{prefix}info_calib/calib_radar_lidar.txt",
        f"{prefix}time_info/os1-128.txt",
        f"{prefix}time_info/os2-64.txt",
        *(f"{prefix}info_label/{label}" for label in labels),
    ]


def build_sensor_members(
    sequence: int, labels: list[str], output_root: Path
) -> list[str]:
    prefix = f"{sequence}/"
    members: list[str] = []
    for label in labels:
        indices = parse_label_header(output_root / prefix / "info_label" / label)
        members.extend(
            [
                f"{prefix}radar_tesseract/tesseract_{indices.radar:05d}.mat",
                f"{prefix}os1-128/os1-128_{indices.lidar128:05d}.pcd",
                f"{prefix}os2-64/os2-64_{indices.lidar64:05d}.pcd",
            ]
        )
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proxy", default=None)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Label file names whose indices define synchronized sensor members.",
    )
    parser.add_argument(
        "--base-url",
        default="https://kaistavelab.tw5.quickconnect.to",
    )
    args = parser.parse_args()

    account, password = credentials_from_environment()
    metadata_members = build_metadata_members(args.sequence, args.labels)
    with SynologySession(args.base_url, account, password, proxy=args.proxy) as client:
        fetch_members(
            client,
            args.sequence,
            metadata_members,
            args.output,
            args.manifest,
        )
        sensor_members = build_sensor_members(args.sequence, args.labels, args.output)
        members = metadata_members + sensor_members
        print(
            json.dumps({"sequence": args.sequence, "members": members}, indent=2),
            flush=True,
        )
        records = fetch_members(client, args.sequence, members, args.output, args.manifest)
    print(json.dumps([record.__dict__ for record in records], indent=2), flush=True)


if __name__ == "__main__":
    main()
