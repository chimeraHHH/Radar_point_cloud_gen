#!/usr/bin/env python3
"""Run the source-bound, CPU-only R-B1 range-echo structural diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from eval.rb1_range_echo_oracle import (  # noqa: E402
    DEFAULT_K_VALUES,
    ORACLE_DISCLAIMER,
    ORACLE_LABEL,
    RANGE_LABELS,
    RangeEchoCapacityError,
    RangeEchoOracleConfig,
    aggregate_numeric_reports,
    array_sha256,
    build_gt_aided_range_echo_candidates,
    geometry_endpoints,
    select_exact_range_echoes,
)


PROTOCOL = "rb1_range_echo_gt_aided_stage0_preflight_v1"
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
FROZEN_TRAIN_FRAME_COUNT = 76
FROZEN_VALIDATION_FRAME_COUNT = 24
FROZEN_DEVELOPMENT_FRAME_COUNT = 100
FROZEN_FAR_VALIDATION_FRAME_COUNT = 23
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MODES = ("preflight", "capacity-audit", "validation-geometry")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_document(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("R-B1 source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("R-B1 source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"R-B1 tracked source tree is dirty: {dirty}")


def validate_development_contract(
    manifest_path: Path,
    scene_split_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Validate the frozen 76/24 development data without selecting test."""

    hashes = {
        "manifest_sha256": sha256_file(manifest_path),
        "scene_split_sha256": sha256_file(scene_split_path),
    }
    expected = {
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "scene_split_sha256": FROZEN_SCENE_SPLIT_SHA256,
    }
    if hashes != expected:
        raise ValueError(
            f"R-B1 requires the frozen 76/24 contract, got {hashes}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_split = json.loads(scene_split_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("R-B1 manifest does not contain a frame list")
    counts = Counter(str(frame.get("partition")) for frame in frames)
    expected_counts = {
        "train": FROZEN_TRAIN_FRAME_COUNT,
        "validation": FROZEN_VALIDATION_FRAME_COUNT,
    }
    if dict(counts) != expected_counts:
        raise ValueError(
            f"R-B1 requires frozen 76/24 frames, got {dict(counts)}"
        )
    if any(str(frame.get("partition")) == "test" for frame in frames):
        raise ValueError("R-B1 development manifest must not contain test")
    identities = [
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in frames
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("R-B1 manifest contains duplicate frame identities")

    split_sequences = {
        partition: {
            int(value)
            for value in scene_split["splits"][partition]["sequences"]
        }
        for partition in ("train", "validation", "test")
    }
    if any(
        split_sequences[left] & split_sequences[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("R-B1 scene split partitions overlap")
    for frame in frames:
        partition = str(frame["partition"])
        sequence = int(frame["sequence"])
        if sequence not in split_sequences[partition]:
            raise ValueError("R-B1 manifest contradicts the scene split")
        if sequence in split_sequences["test"]:
            raise ValueError("R-B1 selected a test sequence")
    return frames, hashes


def _two_validation_records(
    validation_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for first, record in enumerate(validation_records):
        for other in validation_records[first + 1 :]:
            if int(record["sequence"]) != int(other["sequence"]):
                return [record, other]
    raise ValueError("R-B1 two-frame preflight needs two validation scenes")


def select_mode_records(
    frames: list[dict[str, Any]],
    mode: str,
) -> tuple[list[dict[str, Any]], bool]:
    if mode not in MODES:
        raise ValueError(f"Unknown R-B1 mode: {mode}")
    validation = [
        frame for frame in frames if frame["partition"] == "validation"
    ]
    if mode == "preflight":
        return _two_validation_records(validation), True
    if mode == "capacity-audit":
        if len(frames) != FROZEN_DEVELOPMENT_FRAME_COUNT:
            raise ValueError("R-B1 capacity audit requires all 100 frames")
        return list(frames), False
    if len(validation) != FROZEN_VALIDATION_FRAME_COUNT:
        raise ValueError("R-B1 geometry mode requires all 24 validation frames")
    return validation, True


def target_cache_path(cache_root: Path, record: dict[str, Any]) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_"
        f"radar_{int(record['radar_index']):05d}.npz"
    )


def load_target_only(path: Path) -> np.ndarray:
    """Read only the target array; Cube and CFAR arrays are never accessed."""

    with np.load(path, allow_pickle=False) as cache:
        if "target_xyz_confidence" not in cache.files:
            raise ValueError(f"R-B1 target cache lacks target geometry: {path}")
        target = cache["target_xyz_confidence"].astype(
            np.float32,
            copy=True,
        )
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError(f"R-B1 target cache has invalid shape: {path}")
    if not np.isfinite(target).all():
        raise ValueError(f"R-B1 target cache is non-finite: {path}")
    return target


def _target_is_far_frame(target: np.ndarray) -> bool:
    ranges = np.linalg.norm(target[:, :3], axis=1)
    return bool(((ranges >= 60.0) & (ranges < 120.0)).any())


def _mean_summary(values: list[float | int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "sample_count": int(len(array)),
    }


def evaluate_k(
    records: list[dict[str, Any]],
    *,
    cache_root: Path,
    range_axis_m: np.ndarray,
    azimuth_axis_rad: np.ndarray,
    elevation_axis_rad: np.ndarray,
    config: RangeEchoOracleConfig,
    include_geometry: bool,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    capacity_failure_count = 0
    pool_geometry: list[dict[str, Any]] = []
    selected_geometry: list[dict[str, Any]] = []
    for record in records:
        cache_path = target_cache_path(cache_root, record)
        target = load_target_only(cache_path)
        candidates = build_gt_aided_range_echo_candidates(
            target,
            range_axis_m=range_axis_m,
            azimuth_axis_rad=azimuth_axis_rad,
            elevation_axis_rad=elevation_axis_rad,
            config=config,
        )
        frame: dict[str, Any] = {
            "sequence": int(record["sequence"]),
            "radar_index": int(record["radar_index"]),
            "partition": str(record["partition"]),
            "target_count": int(target.shape[0]),
            "target_array_sha256": array_sha256(target),
            "far_target_frame": _target_is_far_frame(target),
            "occupied_ray_count": candidates.report["occupied_ray_count"],
            "echo_count_distribution": candidates.report[
                "echo_count_distribution"
            ],
            "k_truncation": candidates.report["k_truncation"],
            "candidate_capacity": candidates.report["candidate_capacity"],
            "sub_bin_offset": candidates.report["sub_bin_offset"],
            "candidate_hashes": candidates.report["hashes"],
        }
        if include_geometry:
            pool_endpoint = geometry_endpoints(candidates.xyz_m, target)
            frame.setdefault("geometry_endpoints", {})[
                "capped_ray_peak_pool"
            ] = pool_endpoint
            pool_geometry.append(pool_endpoint)
        try:
            selection = select_exact_range_echoes(
                candidates,
                config=config,
            )
        except RangeEchoCapacityError as error:
            capacity_failure_count += 1
            frame["capacity_reachable"] = False
            frame["exact_selection"] = error.report
        else:
            frame["capacity_reachable"] = True
            frame["exact_selection"] = selection.report
            if include_geometry:
                endpoint = geometry_endpoints(selection.xyz_m, target)
                frame["geometry_endpoints"][
                    "exact_10k_quota_5cm_selection"
                ] = endpoint
                selected_geometry.append(endpoint)
        frames.append(frame)

    far_frame_count = sum(frame["far_target_frame"] for frame in frames)
    aggregate: dict[str, Any] = {
        "frame_count": len(frames),
        "far_target_frame_count": far_frame_count,
        "capacity_failure_count": capacity_failure_count,
        "all_frames_capacity_reachable": capacity_failure_count == 0,
        "occupied_ray_count": _mean_summary(
            [frame["occupied_ray_count"] for frame in frames]
        ),
        "raw_gt_aided_echo_group_count": _mean_summary(
            [
                frame["k_truncation"]["raw_gt_aided_echo_group_count"]
                for frame in frames
            ]
        ),
        "candidate_peak_count": _mean_summary(
            [
                frame["candidate_capacity"]["total"]
                for frame in frames
            ]
        ),
        "rays_truncated": _mean_summary(
            [frame["k_truncation"]["rays_truncated"] for frame in frames]
        ),
        "weighted_radial_mae_m": _mean_summary(
            [
                frame["k_truncation"]["weighted_radial_mae_m"]
                for frame in frames
            ]
        ),
        "candidate_capacity_by_range": {
            label: _mean_summary(
                [
                    frame["candidate_capacity"]["by_range"][label]
                    for frame in frames
                ]
            )
            for label in RANGE_LABELS
        },
    }
    if include_geometry:
        aggregate["geometry_endpoints"] = {
            "capped_ray_peak_pool": aggregate_numeric_reports(
                pool_geometry
            ),
            "exact_10k_quota_5cm_selection": (
                aggregate_numeric_reports(selected_geometry)
                if selected_geometry
                else None
            ),
        }
        aggregate["far_frame_semantics"] = {
            "definition": (
                "A frame contributes far-range geometry iff its target has at "
                "least one point with 60m <= range < 120m."
            ),
            "missing_far_frames_are_not_imputed_as_zero": True,
            "observed_far_target_frame_count": far_frame_count,
            "expected_full_validation_far_target_frame_count": (
                FROZEN_FAR_VALIDATION_FRAME_COUNT
            ),
            "far_metric_sample_count": (
                aggregate["geometry_endpoints"][
                    "capped_ray_peak_pool"
                ]
                .get(
                    "range_60_120m_completeness_mean_distance_m",
                    {},
                )
                .get("sample_count", 0)
            ),
        }
    return {
        "k": config.k,
        "config": {
            "exact_count": config.exact_count,
            "range_quotas": {
                label: value
                for label, value in zip(RANGE_LABELS, config.range_quotas)
            },
            "minimum_distance_m": config.minimum_distance_m,
            "radial_merge_distance_m": config.radial_merge_distance_m,
        },
        "frames": frames,
        "aggregate": aggregate,
    }


def source_hashes(repo: Path) -> dict[str, str]:
    relative_paths = (
        "code/eval/rb1_range_echo_oracle.py",
        "code/scripts/preflight_rb1_range_echo.py",
        "code/tests/test_rb1_range_echo_oracle.py",
        "docs/rb1_range_echo_stage0_protocol.md",
    )
    return {
        relative: sha256_file(repo / relative)
        for relative in relative_paths
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"R-B1 output already exists: {args.output}")
    if len(set(args.k_values)) != len(args.k_values):
        raise ValueError("R-B1 K values must be unique")
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    frames, frozen_hashes = validate_development_contract(
        args.manifest,
        args.scene_split,
    )
    selected_records, include_geometry = select_mode_records(
        frames,
        args.mode,
    )
    if any(record["partition"] == "test" for record in selected_records):
        raise ValueError("R-B1 refuses test records")

    axes = load_axes(args.resources)
    cache_provenance = []
    for record in selected_records:
        path = target_cache_path(args.cache_root, record)
        cache_provenance.append(
            {
                "sequence": int(record["sequence"]),
                "radar_index": int(record["radar_index"]),
                "partition": str(record["partition"]),
                "relative_name": path.name,
                "sha256": sha256_file(path),
            }
        )

    results = [
        evaluate_k(
            selected_records,
            cache_root=args.cache_root,
            range_axis_m=axes.range_m,
            azimuth_axis_rad=axes.azimuth_rad,
            elevation_axis_rad=axes.elevation_rad,
            config=RangeEchoOracleConfig(k=k),
            include_geometry=include_geometry,
        )
        for k in args.k_values
    ]
    per_k_checks = {}
    for result in results:
        checks = {
            "all_frames_capacity_reachable": result["aggregate"][
                "all_frames_capacity_reachable"
            ],
            "no_copy_padding_or_jitter": all(
                (
                    frame["exact_selection"][
                        "copy_padding_jitter_duplicate"
                    ]
                    is False
                )
                for frame in result["frames"]
            ),
        }
        if args.mode == "validation-geometry":
            checks["exact_24_validation_frames"] = (
                result["aggregate"]["frame_count"]
                == FROZEN_VALIDATION_FRAME_COUNT
            )
            checks["exact_23_far_target_frames"] = (
                result["aggregate"]["far_target_frame_count"]
                == FROZEN_FAR_VALIDATION_FRAME_COUNT
            )
        per_k_checks[str(result["k"])] = {
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(value["passed"] for value in per_k_checks.values())
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "oracle_label": ORACLE_LABEL,
        "oracle_disclaimer": ORACLE_DISCLAIMER,
        "deployable_method": False,
        "strict_mathematical_upper_bound": False,
        "representation_family_closure_eligible": False,
        "training_authorized": False,
        "scientific_model_result": False,
        "source_commit": args.source_commit,
        "source_hashes": source_hashes(repo),
        "inputs": {
            **frozen_hashes,
            "resources": str(args.resources.resolve()),
            "axis_hashes": {
                "info_arr.mat": sha256_file(args.resources / "info_arr.mat"),
                "arr_doppler.mat": sha256_file(
                    args.resources / "arr_doppler.mat"
                ),
            },
            "target_cache_accessed_keys": ["target_xyz_confidence"],
            "cube_accessed": False,
            "cfar_accessed": False,
            "test_accessed": False,
            "selected_frame_count": len(selected_records),
            "selected_partitions": sorted(
                {str(record["partition"]) for record in selected_records}
            ),
            "target_cache_aggregate_sha256": sha256_document(
                cache_provenance
            ),
            "target_cache_files": cache_provenance,
        },
        "runtime": {
            "device": "cpu",
            "numpy_version": np.__version__,
        },
        "results_by_k": {
            str(result["k"]): result for result in results
        },
        "decision": {
            "per_k": per_k_checks,
            "passed": passed,
            "hard_capacity_failure": not passed,
            "failure_scope": (
                "direct_gt_supported_peak_construction_under_frozen_quotas"
            ),
            "sparse_target_lifting_implemented": False,
            "representation_family_closure_eligible": False,
            "training_authorized": False,
        },
    }
    atomic_json(args.output, document)
    print(json.dumps(document["decision"], indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
