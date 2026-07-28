#!/usr/bin/env python3
"""Upgrade the frozen K-Radar manifest with radar-frame ego transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_kradar_temporal_manifest import (  # noqa: E402
    conjugate_lidar_motion_to_radar,
    radar_from_lidar64,
)


PROTOCOL = "kradar_temporal_manifest_radar_frame_upgrade_v1"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if not SOURCE_PATTERN.fullmatch(commit):
        raise RuntimeError("Temporal manifest upgrade requires a full commit")
    return commit


def upgrade_manifest_document(
    legacy: dict,
    radar_from_lidar_by_sequence: dict[int, np.ndarray],
) -> dict:
    """Add calibrated radar-frame transforms without changing membership."""

    if legacy.get("gate_pass") is not True:
        raise ValueError("Legacy temporal manifest did not pass its gate")
    if any(
        frame.get("partition") == "test"
        for frame in legacy.get("frames", [])
    ):
        raise ValueError("Development manifest upgrade cannot access test")
    if not legacy.get("frames") or not legacy.get("windows"):
        raise ValueError("Legacy temporal manifest is empty")

    upgraded = json.loads(json.dumps(legacy))
    for frame in upgraded["frames"]:
        sequence = int(frame["sequence"])
        try:
            calibration = radar_from_lidar_by_sequence[sequence]
        except KeyError as error:
            raise ValueError(
                f"Missing radar/LiDAR calibration for sequence {sequence}"
            ) from error
        if calibration.shape != (4, 4):
            raise ValueError("Radar/LiDAR calibration must be 4x4")
        raw_lidar_motion = frame.get(
            "current_lidar64_from_previous_lidar64"
        )
        if not isinstance(raw_lidar_motion, list) or len(
            raw_lidar_motion
        ) != 16:
            raise ValueError(
                "Legacy frame lacks a 4x4 LiDAR ego transform"
            )
        lidar_motion = np.asarray(
            raw_lidar_motion,
            dtype=np.float64,
        ).reshape(4, 4)
        radar_motion = conjugate_lidar_motion_to_radar(
            lidar_motion,
            calibration,
        )
        frame["radar_from_lidar64"] = calibration.reshape(-1).tolist()
        frame["current_radar_from_previous_radar"] = (
            radar_motion.reshape(-1).tolist()
        )

    checks = dict(upgraded.get("checks", {}))
    checks["radar_frame_ego_transforms_present"] = all(
        len(frame.get("radar_from_lidar64", [])) == 16
        and len(frame.get("current_radar_from_previous_radar", [])) == 16
        for frame in upgraded["frames"]
    )
    checks["frame_membership_unchanged"] = len(
        upgraded["frames"]
    ) == len(legacy["frames"])
    checks["window_membership_unchanged"] = len(
        upgraded["windows"]
    ) == len(legacy["windows"])
    checks["test_partition_untouched"] = all(
        frame["partition"] != "test" for frame in upgraded["frames"]
    )
    upgraded["checks"] = checks
    upgraded["gate_pass"] = all(checks.values())
    if not upgraded["gate_pass"]:
        failed = [key for key, value in checks.items() if value is not True]
        raise RuntimeError(f"Upgraded temporal manifest failed: {failed}")
    return upgraded


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    current_commit = git_commit(repo)
    if args.source_commit != current_commit:
        raise ValueError(
            "Temporal manifest upgrade source differs from Git snapshot"
        )
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    legacy = json.loads(
        args.legacy_manifest.read_text(encoding="utf-8")
    )
    sequences = sorted(
        {int(frame["sequence"]) for frame in legacy.get("frames", [])}
    )
    calibrations = {}
    calibration_hashes = {}
    for sequence in sequences:
        path = (
            args.metadata_root
            / str(sequence)
            / "info_calib"
            / "calib_radar_lidar.txt"
        )
        calibrations[sequence] = radar_from_lidar64(path)
        calibration_hashes[str(sequence)] = sha256(path)
    upgraded = upgrade_manifest_document(legacy, calibrations)
    upgraded["upgrade"] = {
        "protocol": PROTOCOL,
        "source_commit": current_commit,
        "legacy_manifest": str(args.legacy_manifest.resolve()),
        "legacy_manifest_sha256": sha256(args.legacy_manifest),
        "metadata_root": str(args.metadata_root.resolve()),
        "calibration_sha256_by_sequence": calibration_hashes,
        "transform": (
            "radar_from_lidar64 @ current_lidar64_from_previous_lidar64 "
            "@ inverse(radar_from_lidar64)"
        ),
        "frame_count": len(upgraded["frames"]),
        "window_count": len(upgraded["windows"]),
        "partitions": sorted(
            {frame["partition"] for frame in upgraded["frames"]}
        ),
        "test_accessed": False,
    }
    atomic_json(args.output, upgraded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "upgrade": upgraded["upgrade"],
                "checks": upgraded["checks"],
                "gate_pass": upgraded["gate_pass"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
