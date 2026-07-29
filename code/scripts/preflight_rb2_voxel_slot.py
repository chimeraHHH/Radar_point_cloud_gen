#!/usr/bin/env python3
"""Run source-bound CPU preflights for the R-B2 voxel-slot representation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.rb2_voxel_slot_oracle import (  # noqa: E402
    DEFAULT_CONFIGS,
    EXPORT_COUNT,
    OUTPUT_RANGE_QUOTAS,
    VoxelSlotCapacityError,
    VoxelSlotConfig,
    aggregate_geometry_reports,
    array_sha256,
    run_gt_aided_voxel_slot_oracle,
)


PROTOCOL = "rb2_cartesian_voxel_slot_source_bound_preflight_v1"
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FORMAL_TRAIN_COUNT = 76
FORMAL_VALIDATION_COUNT = 24
FORMAL_FAR_VALIDATION_FRAME_COUNT = 23
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MODE_FRAME_COUNTS = {
    "preflight": 2,
    "capacity": FORMAL_TRAIN_COUNT + FORMAL_VALIDATION_COUNT,
    "geometry": FORMAL_VALIDATION_COUNT,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_exclusive(path: Path, document: dict[str, Any]) -> None:
    """Publish one immutable JSON result without overwrite races."""

    if path.exists():
        raise FileExistsError(f"R-B2 refuses to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"R-B2 refuses to overwrite output: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> dict[str, Any]:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("R-B2 source commit must be a full lowercase Git SHA")
    head = git_output(repo, "rev-parse", "HEAD")
    if head != source_commit:
        raise ValueError("R-B2 source commit differs from checked-out HEAD")
    tracked_dirty = git_output(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_dirty:
        raise ValueError(f"R-B2 tracked source tree is dirty: {tracked_dirty}")
    return {
        "source_commit": head,
        "tracked_worktree_clean": True,
        "untracked_files_ignored_by_source_cleanliness_check": True,
    }


def validate_manifest_contract(document: dict[str, Any]) -> list[dict[str, Any]]:
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise ValueError("R-B2 manifest must contain a frame list")
    counts = Counter(str(frame.get("partition")) for frame in frames)
    expected = {
        "train": FORMAL_TRAIN_COUNT,
        "validation": FORMAL_VALIDATION_COUNT,
    }
    if dict(counts) != expected:
        raise ValueError(
            f"R-B2 requires frozen 76/24 development frames, got {dict(counts)}"
        )
    if any(str(frame.get("partition")) == "test" for frame in frames):
        raise ValueError("R-B2 refuses every manifest containing test records")
    identities = [
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in frames
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("R-B2 manifest contains duplicate frame identities")
    return sorted(
        frames,
        key=lambda frame: (
            str(frame["partition"]),
            int(frame["sequence"]),
            int(frame["radar_index"]),
        ),
    )


def select_mode_records(
    frames: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "preflight":
        train = [frame for frame in frames if frame["partition"] == "train"]
        validation = [
            frame for frame in frames if frame["partition"] == "validation"
        ]
        selected = [train[0], validation[0]]
    elif mode == "capacity":
        selected = frames
    elif mode == "geometry":
        selected = [
            frame for frame in frames if frame["partition"] == "validation"
        ]
    else:
        raise ValueError(f"Unknown R-B2 preflight mode: {mode}")
    if len(selected) != MODE_FRAME_COUNTS[mode]:
        raise ValueError(
            f"R-B2 {mode} selected {len(selected)} unexpected records"
        )
    if any(frame["partition"] == "test" for frame in selected):
        raise ValueError("R-B2 mode selection touched test")
    return selected


def cache_path(cache_root: Path, frame: dict[str, Any]) -> Path:
    return cache_root / (
        f"seq{int(frame['sequence']):02d}_"
        f"radar_{int(frame['radar_index']):05d}.npz"
    )


def load_target_only(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read only the target array; Cube and CFAR arrays remain unopened."""

    if not path.is_file():
        raise FileNotFoundError(f"R-B2 target cache is missing: {path}")
    with np.load(path, allow_pickle=False) as cache:
        if "target_xyz_confidence" not in cache.files:
            raise ValueError(f"R-B2 target cache lacks target array: {path}")
        target = np.asarray(cache["target_xyz_confidence"], dtype=np.float64)
    if target.ndim != 2 or target.shape[1] < 4 or target.shape[0] == 0:
        raise ValueError(
            f"R-B2 target must have non-empty (N,>=4) shape, got {target.shape}"
        )
    if not bool(np.all(np.isfinite(target[:, :4]))):
        raise ValueError("R-B2 target XYZ/confidence contains non-finite values")
    return (
        target[:, :3].copy(),
        target[:, 3].copy(),
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "target_xyz_confidence_sha256": array_sha256(target[:, :4]),
            "target_count": int(target.shape[0]),
            "npz_arrays_accessed": ["target_xyz_confidence"],
            "cube_accessed": False,
            "cfar_accessed": False,
        },
    )


def source_hashes(repo: Path) -> dict[str, dict[str, str]]:
    relative_paths = (
        "code/eval/rb2_voxel_slot_oracle.py",
        "code/scripts/preflight_rb2_voxel_slot.py",
        "docs/rb2_voxel_slot_stage0_protocol.md",
    )
    result = {}
    for relative in relative_paths:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"R-B2 source-bound file is missing: {path}")
        result[relative] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    return result


def _aggregate_numeric(values: list[float | int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "sample_count": int(array.size),
    }


def aggregate_config(
    frames: list[dict[str, Any]],
    config: VoxelSlotConfig,
) -> dict[str, Any]:
    successful = [
        frame["configs"][config.name]
        for frame in frames
        if frame["configs"][config.name]["status"] == "passed"
    ]
    failures = [
        frame["configs"][config.name]
        for frame in frames
        if frame["configs"][config.name]["status"] != "passed"
    ]
    result: dict[str, Any] = {
        "config": config.metadata(),
        "requested_frame_count": len(frames),
        "successful_frame_count": len(successful),
        "failed_frame_count": len(failures),
        "capacity_failures": failures,
    }
    if not successful:
        result["passed"] = False
        return result
    reports = [item["report"] for item in successful]
    result["geometry"] = aggregate_geometry_reports(
        [report["geometry"] for report in reports]
    )
    result["structure"] = {
        key: _aggregate_numeric(
            [report["voxel_capacity"][key] for report in reports]
        )
        for key in (
            "target_occupied_voxel_count",
            "activated_target_occupied_voxel_count",
            "unavailable_target_occupied_voxel_count",
            "quota_support_voxel_count",
            "total_activated_voxel_count",
            "fixed_candidate_count",
            "slot_saturated_occupied_voxel_fraction",
            "selected_slot_fraction",
        )
    }
    result["selection"] = {
        "rejected_5cm_total": _aggregate_numeric(
            [
                report["selection"]["rejected_5cm_total"]
                for report in reports
            ]
        ),
        "observed_minimum_pair_distance_m": _aggregate_numeric(
            [
                report["selection"]["observed_minimum_pair_distance_m"]
                for report in reports
            ]
        ),
        "candidate_capacity_minimum_by_range": {
            label: min(
                report["selection"]["candidate_capacity_by_range"][label]
                for report in reports
            )
            for label in reports[0]["selection"][
                "candidate_capacity_by_range"
            ]
        },
        "selected_by_range_all_frames": {
            label: sorted(
                {
                    report["selection"]["selected_by_range"][label]
                    for report in reports
                }
            )
            for label in reports[0]["selection"]["selected_by_range"]
        },
    }
    result["checks"] = {
        "all_requested_frames_succeeded": len(successful) == len(frames),
        "all_exact_10000": all(
            report["checks"]["exact_10000"] for report in reports
        ),
        "all_range_quotas_exact": all(
            report["checks"]["range_quotas_exact"] for report in reports
        ),
        "all_minimum_pair_distance_at_least_5cm": all(
            report["checks"]["minimum_pair_distance_at_least_5cm"]
            for report in reports
        ),
        "all_fixed_slots_and_cells_valid": all(
            report["checks"]["fixed_slots_per_activated_voxel"]
            and report["checks"]["unique_voxel_slot_candidate_ids"]
            and report["checks"]["all_candidates_inside_fixed_cells"]
            for report in reports
        ),
        "all_gt_aided_labels_explicit": all(
            report["artifact_label"]["unattainable_gt_aided_heuristic"]
            and not report["artifact_label"]["strict_upper_bound"]
            and not report["artifact_label"]["eligible_as_model_result"]
            for report in reports
        ),
        "no_copy_padding_jitter_or_free_centers": all(
            report["checks"]["no_copy_padding_jitter_or_free_centers"]
            for report in reports
        ),
    }
    result["passed"] = all(result["checks"].values())
    return result


def _config_by_name(name: str) -> VoxelSlotConfig:
    matches = [config for config in DEFAULT_CONFIGS if config.name == name]
    if not matches:
        choices = ", ".join(config.name for config in DEFAULT_CONFIGS)
        raise ValueError(f"Unknown R-B2 config {name}; choose from {choices}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_FRAME_COUNTS),
        required=True,
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        choices=tuple(config.name for config in DEFAULT_CONFIGS),
        help="Repeat to select configs; defaults to both frozen configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"R-B2 output already exists: {args.output}")
    repo = Path(__file__).resolve().parents[2]
    source = verify_source_tree(repo, args.source_commit)
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            "R-B2 manifest differs from the frozen 100-frame data contract: "
            f"{manifest_hash}"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_frames = validate_manifest_contract(manifest)
    selected_records = select_mode_records(all_frames, args.mode)
    configs = (
        tuple(_config_by_name(name) for name in args.configs)
        if args.configs
        else DEFAULT_CONFIGS
    )
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("R-B2 config list contains duplicates")

    started = time.perf_counter()
    frame_reports: list[dict[str, Any]] = []
    for record in selected_records:
        target_xyz, target_weight, target_source = load_target_only(
            cache_path(args.cache_root, record)
        )
        per_config: dict[str, Any] = {}
        for config in configs:
            frame_started = time.perf_counter()
            try:
                result = run_gt_aided_voxel_slot_oracle(
                    target_xyz,
                    target_weight,
                    config,
                    output_quotas=OUTPUT_RANGE_QUOTAS,
                )
            except VoxelSlotCapacityError as error:
                per_config[config.name] = {
                    "status": "capacity_failed",
                    "reason": str(error),
                    "capacity_report": error.report,
                    "elapsed_seconds": time.perf_counter() - frame_started,
                }
            else:
                per_config[config.name] = {
                    "status": "passed" if result.report["passed"] else "failed",
                    "report": result.report,
                    "elapsed_seconds": time.perf_counter() - frame_started,
                }
        frame_reports.append(
            {
                "partition": str(record["partition"]),
                "sequence": int(record["sequence"]),
                "radar_index": int(record["radar_index"]),
                "target_source": target_source,
                "configs": per_config,
                "test_accessed": False,
                "cube_accessed": False,
                "cfar_accessed": False,
            }
        )

    aggregates = {
        config.name: aggregate_config(frame_reports, config)
        for config in configs
    }
    far_frame_count = sum(
        bool(
            any(
                frame["configs"][config.name].get("report", {})
                .get("target", {})
                .get("has_range_60_120m_target", False)
                for config in configs
            )
        )
        for frame in frame_reports
        if frame["partition"] == "validation"
    )
    checks = {
        "mode_frame_count_exact": len(frame_reports)
        == MODE_FRAME_COUNTS[args.mode],
        "manifest_frozen_76_train_24_validation": True,
        "test_untouched": all(not frame["test_accessed"] for frame in frame_reports),
        "cube_untouched": all(not frame["cube_accessed"] for frame in frame_reports),
        "cfar_untouched": all(not frame["cfar_accessed"] for frame in frame_reports),
        "only_target_array_accessed": all(
            frame["target_source"]["npz_arrays_accessed"]
            == ["target_xyz_confidence"]
            for frame in frame_reports
        ),
        "all_configs_passed": all(
            aggregate["passed"] for aggregate in aggregates.values()
        ),
        "geometry_mode_has_23_far_target_validation_frames": (
            args.mode != "geometry"
            or far_frame_count == FORMAL_FAR_VALIDATION_FRAME_COUNT
        ),
    }
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "source": source,
        "source_hashes": source_hashes(repo),
        "data": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_hash,
            "cache_root": str(args.cache_root.resolve()),
            "full_manifest_frame_count": len(all_frames),
            "selected_frame_count": len(selected_records),
            "selected_partitions": dict(
                Counter(str(frame["partition"]) for frame in selected_records)
            ),
            "frame_source_aggregate_sha256": hashlib.sha256(
                json.dumps(
                    [
                        {
                            "partition": frame["partition"],
                            "sequence": frame["sequence"],
                            "radar_index": frame["radar_index"],
                            "cache_sha256": frame["target_source"]["sha256"],
                            "target_sha256": frame["target_source"][
                                "target_xyz_confidence_sha256"
                            ],
                        }
                        for frame in frame_reports
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "arrays_accessed": ["target_xyz_confidence"],
            "test_accessed": False,
            "cube_accessed": False,
            "cfar_accessed": False,
        },
        "contract": {
            "artifact_label": "unattainable_gt_aided_voxel_slot_heuristic",
            "strict_upper_bound": False,
            "eligible_as_model_result": False,
            "export_count": EXPORT_COUNT,
            "output_range_quotas": {
                "range_0_30m": OUTPUT_RANGE_QUOTAS[0],
                "range_30_60m": OUTPUT_RANGE_QUOTAS[1],
                "range_60_120m": OUTPUT_RANGE_QUOTAS[2],
            },
            "minimum_pair_distance_m": 0.05,
            "copy_padding_jitter_free_centers": False,
            "capacity_shortfall_is_hard_failure": True,
        },
        "far_frame_semantics": {
            "far_range_m": [60.0, 120.0],
            "selected_validation_frame_count": sum(
                frame["partition"] == "validation" for frame in frame_reports
            ),
            "far_target_validation_frame_count": far_frame_count,
            "formal_geometry_expected_far_target_frame_count": (
                FORMAL_FAR_VALIDATION_FRAME_COUNT
            ),
            "geometry_aggregate_sample_count_rule": (
                "include every validation frame with far-range GT, including "
                "frames with zero same-bin predictions"
            ),
        },
        "frames": frame_reports,
        "aggregates": aggregates,
        "runtime": {
            "device": "cpu",
            "gpu_started": False,
            "elapsed_seconds": time.perf_counter() - started,
            "numpy_version": np.__version__,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    atomic_json_exclusive(args.output, document)
    print(json.dumps(document, indent=2), flush=True)
    if not document["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
