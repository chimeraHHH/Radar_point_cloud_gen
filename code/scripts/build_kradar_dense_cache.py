#!/usr/bin/env python3
"""Build resumable dense-target caches after the full G0 audit has passed."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_frame  # noqa: E402
from cube_dense.odometry import (  # noqa: E402
    interpolate_motion,
    load_pose_trajectory,
    read_os2_times,
)
from cube_dense.observability import (  # noqa: E402
    CFARConfig,
    ca_cfar_points,
    deskew_lidar_to_reference,
    observable_lidar_target,
)


CACHE_SCHEMA_VERSION = 1
LIDAR_TIME_REFERENCES = ("none", "start", "center", "end")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_g0_lidar_time_reference(report: dict, expected: str) -> None:
    successful = [
        frame for frame in report.get("frames", []) if "error" not in frame
    ]
    if not successful:
        raise ValueError("G0 report has no successful frames")
    references = {
        frame.get("lidar_scan_timing", {}).get("selected_reference")
        for frame in successful
    }
    if None in references:
        raise ValueError("G0 report is missing the selected LiDAR time reference")
    if references != {expected}:
        raise ValueError(
            "Dense-cache LiDAR time reference differs from G0: "
            f"expected {expected}, found {sorted(references)}"
        )


def lidar_time_origin_shift(
    reference: str, scan_duration_seconds: float
) -> float | None:
    if reference not in LIDAR_TIME_REFERENCES:
        raise ValueError(f"Unknown LiDAR time reference: {reference}")
    if scan_duration_seconds < 0.0:
        raise ValueError("LiDAR scan duration cannot be negative")
    if reference == "none":
        return None
    return {
        "start": 0.0,
        "center": -scan_duration_seconds / 2.0,
        "end": -scan_duration_seconds,
    }[reference]


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cache_path(cache_root: Path, sequence: int, radar_index: int) -> Path:
    return cache_root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


def write_cache(
    path: Path,
    cfar,
    target,
    motion: dict,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    surface = target.surface_mask
    target_values = torch.cat(
        (target.points_xyz[surface], target.confidence[surface, None]), dim=1
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
            source_manifest_sha256=np.asarray(metadata["source_manifest_sha256"]),
            source_commit=np.asarray(metadata["source_commit"]),
            cfar_false_alarm_rate=np.asarray(
                metadata["cfar_false_alarm_rate"], dtype=np.float64
            ),
            cfar_max_points=np.asarray(metadata["cfar_max_points"], dtype=np.int32),
            lidar_time_reference=np.asarray(metadata["lidar_time_reference"]),
            cfar_xyzd_power_snr=(
                cfar.points_xyzd_power_snr.detach().cpu().numpy().astype(np.float32)
            ),
            cfar_drae_index=(
                cfar.indices_drae.detach().cpu().numpy().astype(np.int16)
            ),
            target_xyz_confidence=(
                target_values.detach().cpu().numpy().astype(np.float32)
            ),
            target_rae_index=(
                target.indices_rae[surface].detach().cpu().numpy().astype(np.int16)
            ),
            ego_velocity_xyz_mps=np.asarray(
                motion["velocity_xyz_mps"], dtype=np.float32
            ),
            ego_speed_mps=np.asarray(motion["speed_mps"], dtype=np.float32),
            ego_yaw_rate_radps=np.asarray(
                motion["yaw_rate_radps"], dtype=np.float32
            ),
        )
    temporary.replace(path)


def validate_cache(path: Path, metadata: dict) -> dict:
    with np.load(path) as cache:
        required = {
            "cache_schema_version",
            "source_manifest_sha256",
            "source_commit",
            "cfar_false_alarm_rate",
            "cfar_max_points",
            "lidar_time_reference",
            "cfar_xyzd_power_snr",
            "cfar_drae_index",
            "target_xyz_confidence",
            "target_rae_index",
            "ego_velocity_xyz_mps",
            "ego_speed_mps",
            "ego_yaw_rate_radps",
        }
        if not required.issubset(cache.files):
            raise ValueError(f"Cache keys missing in {path}")
        if int(cache["cache_schema_version"]) != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Cache schema mismatch in {path}")
        comparisons = {
            "source_manifest_sha256": metadata["source_manifest_sha256"],
            "source_commit": metadata["source_commit"],
            "lidar_time_reference": metadata["lidar_time_reference"],
        }
        for key, expected in comparisons.items():
            if str(cache[key]) != str(expected):
                raise ValueError(f"Cache metadata {key} differs in {path}")
        if not np.isclose(
            float(cache["cfar_false_alarm_rate"]),
            metadata["cfar_false_alarm_rate"],
        ) or int(cache["cfar_max_points"]) != metadata["cfar_max_points"]:
            raise ValueError(f"Cache CFAR configuration differs in {path}")
        cfar = cache["cfar_xyzd_power_snr"]
        cfar_index = cache["cfar_drae_index"]
        target = cache["target_xyz_confidence"]
        target_index = cache["target_rae_index"]
        if cfar.ndim != 2 or cfar.shape[1] != 6 or cfar.shape[0] == 0:
            raise ValueError(f"Invalid CFAR array in {path}: {cfar.shape}")
        if cfar_index.shape != (cfar.shape[0], 4):
            raise ValueError(f"Invalid CFAR indices in {path}")
        if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
            raise ValueError(f"Invalid target array in {path}: {target.shape}")
        if target_index.shape != (target.shape[0], 3):
            raise ValueError(f"Invalid target indices in {path}")
        if not np.isfinite(cfar).all() or not np.isfinite(target).all():
            raise ValueError(f"Non-finite cache values in {path}")
        return {
            "cfar_point_count": int(cfar.shape[0]),
            "target_point_count": int(target.shape[0]),
            "ego_speed_mps": float(cache["ego_speed_mps"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--odometry-root", type=Path, required=True)
    parser.add_argument("--g0-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--false-alarm-rate", type=float, default=1e-3)
    parser.add_argument("--max-cfar-points", type=int, default=10_000)
    parser.add_argument(
        "--lidar-time-reference", choices=LIDAR_TIME_REFERENCES, required=True
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--required-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Dense-cache construction requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.overwrite:
        args.output.unlink(missing_ok=True)

    g0_report = json.loads(args.g0_report.read_text(encoding="utf-8"))
    if g0_report.get("aggregate", {}).get("gate_pass") is not True:
        raise ValueError("Dense cache requires a passed full G0 audit")
    validate_g0_lidar_time_reference(g0_report, args.lidar_time_reference)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("gate_pass") is not True:
        raise ValueError("Dense-cache manifest did not pass its data gate")
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    g0_split_hash = g0_report["provenance"]["scene_split_sha256"]
    if g0_split_hash != scene_split_hash:
        raise ValueError("G0 and dense-cache scene splits differ")
    if manifest.get("source_split_sha256") != scene_split_hash:
        raise ValueError("Temporal manifest and scene split differ")
    required_frames = args.required_frames or len(manifest["frames"])
    if required_frames != len(manifest["frames"]):
        raise ValueError("Formal cache must cover every manifest frame")

    metadata = {
        "source_manifest_sha256": manifest_hash,
        "source_commit": args.source_commit,
        "cfar_false_alarm_rate": args.false_alarm_rate,
        "cfar_max_points": args.max_cfar_points,
        "lidar_time_reference": args.lidar_time_reference,
    }
    configuration = {
        **metadata,
        "scene_split_sha256": scene_split_hash,
        "g0_report_sha256": sha256(args.g0_report),
        "required_frames": required_frames,
        "device": args.device,
    }
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report["configuration"] != configuration:
            raise ValueError("Dense-cache resume configuration differs")
    else:
        report = {
            "schema_version": 1,
            "configuration": configuration,
            "frames": [],
            "completed": False,
        }
        atomic_json(args.output, report)
    completed = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in report["frames"]
    }

    axes = load_axes(args.data_root / "resources")
    cfar_config = CFARConfig(
        false_alarm_rate=args.false_alarm_rate,
        max_points=args.max_cfar_points,
    )
    contexts = {}

    def sequence_context(sequence: int, partition: str) -> dict:
        if sequence in contexts:
            return contexts[sequence]
        times = read_os2_times(
            args.data_root / str(sequence) / "time_info" / "os2-64.txt"
        )
        labels = scene_split["splits"][partition]["labels"][str(sequence)]
        trajectory = load_pose_trajectory(
            args.odometry_root / f"gt_{sequence:02d}.txt", labels, times
        )
        cuda_trajectory = {
            key: torch.as_tensor(
                value,
                dtype=torch.float64 if key == "timestamp" else torch.float32,
                device=args.device,
            )
            for key, value in trajectory.items()
            if key in {"timestamp", "position", "heading"}
        }
        contexts[sequence] = {
            "trajectory": trajectory,
            "cuda_trajectory": cuda_trajectory,
        }
        return contexts[sequence]

    for specification in manifest["frames"]:
        sequence = int(specification["sequence"])
        radar_index = int(specification["radar_index"])
        key = (sequence, radar_index)
        path = cache_path(args.cache_root, sequence, radar_index)
        prior = completed.get(key)
        if prior and path.is_file() and sha256(path) == prior["cache_sha256"]:
            print(json.dumps({"sequence": sequence, "radar_index": radar_index, "status": "cached"}), flush=True)
            continue
        started = time.monotonic()
        context = sequence_context(sequence, specification["partition"])
        sequence_root = args.data_root / str(sequence)
        frame = load_frame(
            sequence_root,
            sequence_root / "info_label" / specification["label"],
            args.data_root / "resources",
        )
        cube = torch.as_tensor(frame.cube_drae, dtype=torch.float32, device=args.device)
        cfar = ca_cfar_points(cube, axes, cfar_config)
        lidar = torch.as_tensor(frame.lidar64, dtype=torch.float32, device=args.device)
        calibration = torch.as_tensor(
            frame.calibration.translation_xyz_m,
            dtype=lidar.dtype,
            device=lidar.device,
        )
        point_offsets_s = lidar[:, frame.lidar64_fields.index("t")] * 1e-9
        origin_shift = lidar_time_origin_shift(
            args.lidar_time_reference, float(point_offsets_s.max().item())
        )
        if origin_shift is None:
            aligned_lidar = lidar[:, :3] + calibration
        else:
            aligned_lidar = deskew_lidar_to_reference(
                points_xyz=lidar[:, :3],
                point_offsets_s=point_offsets_s,
                reference_timestamp=frame.indices.timestamp,
                timestamp_origin_shift_s=origin_shift,
                calibration_xyz_m=calibration,
                odometry_timestamps=context["cuda_trajectory"]["timestamp"],
                odometry_positions=context["cuda_trajectory"]["position"],
                odometry_headings=context["cuda_trajectory"]["heading"],
            )
        target = observable_lidar_target(aligned_lidar, axes, cfar)
        motion = interpolate_motion(
            context["trajectory"], frame.indices.timestamp
        )
        write_cache(path, cfar, target, motion, metadata)
        validation = validate_cache(path, metadata)
        record = {
            "sequence": sequence,
            "partition": specification["partition"],
            "window_id": specification.get("window_id"),
            "radar_index": radar_index,
            "lidar64_index": int(specification["lidar64_index"]),
            "cache": str(path),
            "cache_sha256": sha256(path),
            "cache_size": path.stat().st_size,
            **validation,
            "odometry_nearest_delta_ms": motion["nearest_timestamp_delta_ms"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        completed[key] = record
        report["frames"] = sorted(
            completed.values(),
            key=lambda value: (value["sequence"], value["radar_index"]),
        )
        atomic_json(args.output, report)
        print(json.dumps(record), flush=True)
        del frame, cube, cfar, lidar, aligned_lidar, target
        torch.cuda.empty_cache()

    frame_records = report["frames"]
    checks = {
        "required_frame_count": len(frame_records) == required_frames,
        "all_cache_files_present": all(
            Path(frame["cache"]).is_file() for frame in frame_records
        ),
        "all_targets_nonempty": all(
            frame["target_point_count"] > 0 for frame in frame_records
        ),
        "all_cfar_nonempty": all(
            frame["cfar_point_count"] > 0 for frame in frame_records
        ),
        "odometry_support_le_60ms": max(
            frame["odometry_nearest_delta_ms"] for frame in frame_records
        )
        <= 60.0,
    }
    report.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "device_name": torch.cuda.get_device_name(args.device),
            "torch_version": torch.__version__,
            "checks": checks,
            "completed": all(checks.values()),
        }
    )
    atomic_json(args.output, report)
    print(json.dumps({"checks": checks, "completed": report["completed"]}, indent=2))
    if not report["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
