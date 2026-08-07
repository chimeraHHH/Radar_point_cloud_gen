#!/usr/bin/env python3
"""Read-only bounded sweep of R-B2 Cube-only candidate activation.

The sweep changes only the current-Cube score aggregation and candidate-bank
size. Ground truth is loaded after candidate construction and is used only to
report occupied-voxel recall and confidence-weighted coverage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_tesseract  # noqa: E402
from models.rb2_voxel_slot_model import (  # noqa: E402
    CandidateCapacityError,
    LATTICE_ORIGIN_XYZ_M,
    LATTICE_SHAPE_XYZ,
    RANGE_STRATA_M,
    VOXEL_SIZE_M,
    _candidate_voxels_one_frame,
    candidate_voxels_from_cube,
    lattice_indices_to_centers,
    range_stratum_codes,
)


PROTOCOL = "rb2_cube_only_candidate_support_bounded_sweep_v1"
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
FROZEN_NORMALIZATION_SHA256 = (
    "4d0bca7d027a1a9f457c526f21a034a406ce4973e2dccee55ddd2d7019b41b77"
)
DIAGNOSED_PARENT_SOURCE_COMMIT = "8fe928069169624f2db40fad3d5daba86cb6740f"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FROZEN_CURRENT_PREFLIGHT_IDENTITY = (1, 232)
FROZEN_CURRENT_CANDIDATE_SHA256 = (
    "247a3ea5bf3ce5175a40818aae1f8f47d72d8d647bddd39ed8210bfe660b0544"
)
EXPECTED_PARTITION_COUNTS = {"train": 76, "validation": 24}
TARGET_ARRAY = "target_xyz_confidence"
FORBIDDEN_RECORD_KEY_FRAGMENTS = ("future", "next_cube", "test", "cfar")

BASE_CANDIDATE_QUOTAS = (16_000, 3_400, 600)
BANK_SCALES = (1, 2, 4)
SCORE_MODES = ("max_d", "sum_d", "log_sum_d")
SCORE_PRIORITY = {name: index for index, name in enumerate(SCORE_MODES)}
SEED_MULTIPLIER = 8
NEIGHBORHOOD_RADIUS = 4
MINIMUM_OCCUPIED_VOXEL_RECALL = 0.20
MINIMUM_CONFIDENCE_COVERAGE = 0.30


@dataclass(frozen=True)
class CandidateArm:
    score_mode: str
    bank_scale: int
    candidate_quotas: tuple[int, int, int]
    candidate_count: int
    seed_multiplier: int = SEED_MULTIPLIER
    neighborhood_radius: int = NEIGHBORHOOD_RADIUS

    @property
    def name(self) -> str:
        return (
            f"{self.score_mode}_bank{self.candidate_count // 1000}k_"
            f"r{self.neighborhood_radius}"
        )


def frozen_sweep_grid() -> tuple[CandidateArm, ...]:
    """Return the pre-registered score x bank grid in deterministic order."""

    arms = []
    for scale in BANK_SCALES:
        quotas = tuple(value * scale for value in BASE_CANDIDATE_QUOTAS)
        for score_mode in SCORE_MODES:
            arms.append(
                CandidateArm(
                    score_mode=score_mode,
                    bank_scale=scale,
                    candidate_quotas=quotas,
                    candidate_count=sum(quotas),
                )
            )
    return tuple(arms)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R-B2 diagnostic output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def aggregate_doppler_score(
    cube_drae: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    """Collapse current-Cube Doppler bins without target-dependent inputs."""

    if cube_drae.ndim != 4:
        raise ValueError("R-B2 diagnostic Cube must have shape (D,R,A,E)")
    cube = cube_drae.float().clamp_min(0.0)
    if score_mode == "max_d":
        return torch.log1p(cube).amax(dim=0)
    if score_mode == "sum_d":
        return cube.sum(dim=0)
    if score_mode == "log_sum_d":
        return torch.log1p(cube).sum(dim=0)
    raise ValueError(f"Unknown frozen R-B2 score mode: {score_mode}")


@torch.no_grad()
def build_cube_only_candidates(
    cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    arm: CandidateArm,
) -> torch.Tensor:
    """Construct candidates from the current Cube and frozen axes only."""

    if cube_drae.ndim != 4:
        raise ValueError("R-B2 diagnostic Cube must have shape (D,R,A,E)")
    if arm.score_mode == "max_d":
        candidates = candidate_voxels_from_cube(
            cube_drae.unsqueeze(0),
            range_m,
            azimuth_rad,
            elevation_rad,
            candidate_voxel_quotas=arm.candidate_quotas,
            seed_multiplier=arm.seed_multiplier,
            maximum_neighbor_radius=arm.neighborhood_radius,
        )[0]
    else:
        score = aggregate_doppler_score(cube_drae, arm.score_mode)
        candidates = _candidate_voxels_one_frame(
            score,
            range_m.float(),
            azimuth_rad.float(),
            elevation_rad.float(),
            arm.candidate_quotas,
            seed_multiplier=arm.seed_multiplier,
            maximum_neighbor_radius=arm.neighborhood_radius,
        )
    return candidates


def _linear_ids(indices_xyz: torch.Tensor) -> torch.Tensor:
    _, ny, nz = LATTICE_SHAPE_XYZ
    return (indices_xyz[..., 0] * ny + indices_xyz[..., 1]) * nz + indices_xyz[..., 2]


@torch.no_grad()
def candidate_support_metrics(
    candidate_indices_xyz: torch.Tensor,
    target_xyz_confidence: torch.Tensor,
) -> dict[str, Any]:
    """Report target support without changing candidates or selecting an arm."""

    if candidate_indices_xyz.ndim != 2 or candidate_indices_xyz.shape[1] != 3:
        raise ValueError("Candidate indices must have shape (N,3)")
    if target_xyz_confidence.ndim != 2 or target_xyz_confidence.shape[1] < 4:
        raise ValueError("Target must have shape (M,>=4)")
    if not bool(torch.isfinite(target_xyz_confidence[:, :4]).all()):
        raise ValueError("Target contains non-finite XYZ/confidence")
    candidate_ids = _linear_ids(candidate_indices_xyz.long())
    if torch.unique(candidate_ids).numel() != candidate_ids.numel():
        raise ValueError("Candidate bank contains duplicate voxel IDs")

    origin = torch.tensor(
        LATTICE_ORIGIN_XYZ_M,
        device=target_xyz_confidence.device,
        dtype=target_xyz_confidence.dtype,
    )
    target_indices = torch.floor(
        (target_xyz_confidence[:, :3] - origin) / VOXEL_SIZE_M
    ).long()
    maximum = torch.tensor(
        LATTICE_SHAPE_XYZ,
        device=target_indices.device,
        dtype=target_indices.dtype,
    )
    if bool(((target_indices < 0) | (target_indices >= maximum)).any()):
        raise ValueError("Target lies outside the frozen R-B2 lattice")
    target_ids = _linear_ids(target_indices)
    target_occupied_ids = torch.unique(target_ids)
    represented_occupied = torch.isin(target_occupied_ids, candidate_ids)
    represented_points = torch.isin(target_ids, candidate_ids)
    confidence = target_xyz_confidence[:, 3].clamp_min(0.0)
    confidence_total = confidence.sum()
    confidence_covered = confidence[represented_points].sum()
    recall = represented_occupied.float().mean()
    coverage = (
        torch.zeros((), device=confidence.device)
        if float(confidence_total.item()) <= 0.0
        else confidence_covered / confidence_total
    )
    return {
        "target_point_count": int(target_ids.numel()),
        "target_occupied_voxel_count": int(target_occupied_ids.numel()),
        "represented_target_occupied_voxel_count": int(
            represented_occupied.sum().item()
        ),
        "target_occupied_voxel_recall": float(recall.item()),
        "target_confidence_total": float(confidence_total.item()),
        "target_confidence_covered": float(confidence_covered.item()),
        "target_confidence_coverage": float(coverage.item()),
    }


def arm_gate(
    frame_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not frame_metrics:
        raise ValueError("R-B2 arm gate requires at least one frame")
    failures = [row for row in frame_metrics if row.get("construction_error")]
    valid = [row for row in frame_metrics if not row.get("construction_error")]
    if not valid:
        return {
            "checks": {
                "candidate_construction_passed_on_every_frame": False,
                "minimum_frame_occupied_voxel_recall_at_least_20pct": False,
                "minimum_frame_confidence_coverage_at_least_30pct": False,
            },
            "passed": False,
            "construction_failure_count": len(failures),
            "occupied_voxel_recall": None,
            "confidence_coverage": None,
        }
    recalls = np.asarray(
        [row["target_occupied_voxel_recall"] for row in valid],
        dtype=np.float64,
    )
    coverages = np.asarray(
        [row["target_confidence_coverage"] for row in valid],
        dtype=np.float64,
    )
    checks = {
        "candidate_construction_passed_on_every_frame": not failures,
        "minimum_frame_occupied_voxel_recall_at_least_20pct": bool(
            recalls.min() >= MINIMUM_OCCUPIED_VOXEL_RECALL
        ),
        "minimum_frame_confidence_coverage_at_least_30pct": bool(
            coverages.min() >= MINIMUM_CONFIDENCE_COVERAGE
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "construction_failure_count": len(failures),
        "occupied_voxel_recall": {
            "minimum": float(recalls.min()),
            "median": float(np.median(recalls)),
            "mean": float(recalls.mean()),
            "maximum": float(recalls.max()),
        },
        "confidence_coverage": {
            "minimum": float(coverages.min()),
            "median": float(np.median(coverages)),
            "mean": float(coverages.mean()),
            "maximum": float(coverages.max()),
        },
    }


def sweep_decision(
    arm_summaries: Sequence[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    passing = [row for row in arm_summaries if bool(row["gate"]["passed"])]
    if not passing:
        return {
            "decision": "close_current_cube_activation_family",
            "mode": mode,
            "minimum_passing_candidate_count": None,
            "eligible_arms_at_minimum_bank": [],
            "training_authorized": False,
            "claim_boundary": (
                "No frozen arm met both support thresholds. This closes the "
                "current score-plus-fixed-neighborhood activation family; it "
                "does not close all Cube-only representations."
            ),
        }
    minimum = min(int(row["arm"]["candidate_count"]) for row in passing)
    eligible = sorted(
        [
            str(row["arm"]["name"])
            for row in passing
            if int(row["arm"]["candidate_count"]) == minimum
        ],
        key=lambda name: SCORE_PRIORITY[name.split("_bank", 1)[0]],
    )
    return {
        "decision": "cube_activation_support_gate_pass",
        "mode": mode,
        "minimum_passing_candidate_count": minimum,
        "eligible_arms_at_minimum_bank": eligible,
        "training_authorized": False,
        "claim_boundary": (
            "Ground truth selected a diagnostic support arm. This is not a "
            "trained-model result, validation score, or generalization claim."
        ),
    }


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("R-B2 execution source must be a full lowercase Git SHA")
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        text=True,
    ).strip()
    if head != source_commit:
        raise ValueError(f"Source commit {source_commit} differs from HEAD {head}")
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repo,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("R-B2 diagnosis requires a clean source tree")


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("R-B2 candidate diagnosis requires H200 CUDA")
    device = torch.device(device_name)
    hardware = torch.cuda.get_device_name(device)
    if "H200" not in hardware.upper():
        raise RuntimeError(f"Expected H200, found {hardware}")
    return device, hardware


def validate_inputs(
    manifest_path: Path,
    scene_split_path: Path,
    normalization_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    hashes = {
        "manifest_sha256": sha256_file(manifest_path),
        "scene_split_sha256": sha256_file(scene_split_path),
        "normalization_sha256": sha256_file(normalization_path),
    }
    expected = {
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "scene_split_sha256": FROZEN_SCENE_SPLIT_SHA256,
        "normalization_sha256": FROZEN_NORMALIZATION_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"R-B2 frozen input hash mismatch: {hashes}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_split = json.loads(scene_split_path.read_text(encoding="utf-8"))
    records = manifest.get("frames")
    if not isinstance(records, list):
        raise ValueError("R-B2 manifest has no frame list")
    counts = {
        partition: sum(record.get("partition") == partition for record in records)
        for partition in ("train", "validation")
    }
    if counts != EXPECTED_PARTITION_COUNTS or len(records) != sum(counts.values()):
        raise ValueError(f"R-B2 frozen 76/24 split differs: {counts}")
    identities = [
        (int(record["sequence"]), int(record["radar_index"])) for record in records
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("R-B2 manifest contains duplicate frame identities")
    for record in records:
        lowered = [str(key).lower() for key in record]
        if any(
            fragment in key
            for key in lowered
            for fragment in FORBIDDEN_RECORD_KEY_FRAGMENTS
        ):
            raise ValueError("R-B2 manifest exposes a forbidden input field")
    if scene_split.get("gate_pass") is not True:
        raise ValueError("R-B2 scene split gate did not pass")
    train_sequences = {
        int(value)
        for value in scene_split.get("splits", {})
        .get("train", {})
        .get("sequences", [])
    }
    train = sorted(
        (record for record in records if record["partition"] == "train"),
        key=lambda row: (int(row["sequence"]), int(row["radar_index"])),
    )
    if any(int(record["sequence"]) not in train_sequences for record in train):
        raise ValueError("R-B2 train record violates the frozen scene split")
    return train, hashes


def records_for_mode(
    train_records: Sequence[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "one-frame":
        selected = [
            record
            for record in train_records
            if (int(record["sequence"]), int(record["radar_index"]))
            == FROZEN_CURRENT_PREFLIGHT_IDENTITY
        ]
        if len(selected) != 1:
            raise ValueError("Frozen R-B2 one-frame identity is unavailable")
        return selected
    if mode == "full-train":
        if len(train_records) != EXPECTED_PARTITION_COUNTS["train"]:
            raise ValueError("R-B2 full audit requires all 76 train frames")
        return list(train_records)
    raise ValueError(f"Unsupported R-B2 audit mode: {mode}")


def _cube_path(data_root: Path, record: dict[str, Any]) -> Path:
    return (
        data_root
        / str(int(record["sequence"]))
        / "radar_tesseract"
        / f"tesseract_{int(record['radar_index']):05d}.mat"
    )


def _cache_path(cache_root: Path, record: dict[str, Any]) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_"
        f"radar_{int(record['radar_index']):05d}.npz"
    )


def candidate_structure(
    candidates: torch.Tensor,
    arm: CandidateArm,
) -> dict[str, Any]:
    strata = range_stratum_codes(lattice_indices_to_centers(candidates))
    counts = [int((strata == code).sum().item()) for code in range(len(RANGE_STRATA_M))]
    unique = int(torch.unique(_linear_ids(candidates)).numel())
    checks = {
        "candidate_count_exact": int(candidates.shape[0]) == arm.candidate_count,
        "range_quotas_exact": tuple(counts) == arm.candidate_quotas,
        "voxel_ids_unique": unique == int(candidates.shape[0]),
    }
    return {
        "candidate_count": int(candidates.shape[0]),
        "candidate_count_by_range": counts,
        "unique_candidate_voxel_count": unique,
        "checks": checks,
        "passed": all(checks.values()),
    }


def measure_arm(
    cube: torch.Tensor,
    target: torch.Tensor,
    axes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    arm: CandidateArm,
    *,
    identity: tuple[int, int],
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize(cube.device)
    baseline_allocated = torch.cuda.memory_allocated(cube.device)
    baseline_reserved = torch.cuda.memory_reserved(cube.device)
    torch.cuda.reset_peak_memory_stats(cube.device)
    started = time.perf_counter()
    try:
        candidates = build_cube_only_candidates(cube, *axes, arm)
    except CandidateCapacityError as error:
        torch.cuda.synchronize(cube.device)
        peak_allocated = torch.cuda.max_memory_allocated(cube.device)
        peak_reserved = torch.cuda.max_memory_reserved(cube.device)
        return {
            "construction_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "memory": {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": max(
                    0, peak_allocated - baseline_allocated
                ),
                "incremental_peak_reserved_bytes": max(
                    0, peak_reserved - baseline_reserved
                ),
                "candidate_tensor_bytes": 0,
                "process_max_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            },
        }
    torch.cuda.synchronize(cube.device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(cube.device)
    peak_reserved = torch.cuda.max_memory_reserved(cube.device)
    structure = candidate_structure(candidates, arm)
    if not structure["passed"]:
        raise RuntimeError(f"R-B2 candidate structure failed for {arm.name}")
    if (
        identity == FROZEN_CURRENT_PREFLIGHT_IDENTITY
        and arm.name == "max_d_bank20k_r4"
        and tensor_sha256(candidates) != FROZEN_CURRENT_CANDIDATE_SHA256
    ):
        raise RuntimeError("Frozen current R-B2 candidate hash was not reproduced")
    support = candidate_support_metrics(candidates, target)
    result = {
        **support,
        "candidate_structure": structure,
        "candidate_indices_sha256": tensor_sha256(candidates),
        "elapsed_seconds": elapsed,
        "memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": max(
                0, peak_allocated - baseline_allocated
            ),
            "incremental_peak_reserved_bytes": max(
                0, peak_reserved - baseline_reserved
            ),
            "candidate_tensor_bytes": candidates.numel() * candidates.element_size(),
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    del candidates
    return result


def summarize_arm(
    arm: CandidateArm,
    frames: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    gate = arm_gate(frames)
    elapsed = np.asarray([row["elapsed_seconds"] for row in frames], dtype=np.float64)
    incremental_allocated = [
        int(row["memory"]["incremental_peak_allocated_bytes"]) for row in frames
    ]
    incremental_reserved = [
        int(row["memory"]["incremental_peak_reserved_bytes"]) for row in frames
    ]
    return {
        "arm": {"name": arm.name, **asdict(arm)},
        "gate": gate,
        "runtime": {
            "total_seconds": float(elapsed.sum()),
            "mean_seconds_per_frame": float(elapsed.mean()),
            "maximum_seconds_per_frame": float(elapsed.max()),
        },
        "memory": {
            "maximum_incremental_peak_allocated_bytes": max(incremental_allocated),
            "maximum_incremental_peak_reserved_bytes": max(incremental_reserved),
        },
        "frames": list(frames),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("one-frame", "full-train"),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"R-B2 diagnostic output exists: {args.output}")
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    train, input_hashes = validate_inputs(
        args.manifest,
        args.scene_split,
        args.normalization_stats,
    )
    records = records_for_mode(train, args.mode)
    device, hardware = require_h200(args.device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    axes_np = load_axes(args.data_root / "resources")
    axes = (
        torch.as_tensor(axes_np.range_m, dtype=torch.float32, device=device),
        torch.as_tensor(axes_np.azimuth_rad, dtype=torch.float32, device=device),
        torch.as_tensor(axes_np.elevation_rad, dtype=torch.float32, device=device),
    )
    grid = frozen_sweep_grid()
    per_arm: dict[str, list[dict[str, Any]]] = {arm.name: [] for arm in grid}
    frame_provenance = []
    sweep_started = time.perf_counter()
    for record in records:
        identity = (int(record["sequence"]), int(record["radar_index"]))
        cube_path = _cube_path(args.data_root, record)
        cache_path = _cache_path(args.cache_root, record)
        cube = torch.from_numpy(
            load_tesseract(cube_path).astype(np.float32, copy=False)
        ).to(device)
        with np.load(cache_path) as cache:
            if TARGET_ARRAY not in cache.files:
                raise ValueError(f"R-B2 target array is missing: {cache_path}")
            target_np = cache[TARGET_ARRAY].astype(np.float32)
            arrays_read = [TARGET_ARRAY]
        target = torch.from_numpy(target_np).to(device)
        frame_provenance.append(
            {
                "sequence": identity[0],
                "radar_index": identity[1],
                "partition": "train",
                "cube_path": str(cube_path),
                "cube_sha256": sha256_file(cube_path),
                "cache_path": str(cache_path),
                "cache_sha256": sha256_file(cache_path),
                "cache_arrays_read": arrays_read,
            }
        )
        for arm in grid:
            row = measure_arm(cube, target, axes, arm, identity=identity)
            row["sequence"] = identity[0]
            row["radar_index"] = identity[1]
            per_arm[arm.name].append(row)
        del cube, target
        torch.cuda.empty_cache()
    summaries = [summarize_arm(arm, per_arm[arm.name]) for arm in grid]
    decision = sweep_decision(summaries, mode=args.mode)
    document = {
        "protocol": PROTOCOL,
        "mode": args.mode,
        "frozen_grid": [asdict(arm) | {"name": arm.name} for arm in grid],
        "frozen_gate": {
            "scope": "all selected train frames independently",
            "minimum_target_occupied_voxel_recall": (
                MINIMUM_OCCUPIED_VOXEL_RECALL
            ),
            "minimum_target_confidence_coverage": MINIMUM_CONFIDENCE_COVERAGE,
            "selection_rule": (
                "report every passing arm at the smallest candidate count; "
                "do not choose one score using target metrics"
            ),
            "no_passing_arm_decision": "close_current_cube_activation_family",
        },
        "provenance": {
            "source_commit": args.source_commit,
            "diagnosed_parent_source_commit": DIAGNOSED_PARENT_SOURCE_COMMIT,
            **input_hashes,
            "device": hardware,
            "torch_version": torch.__version__,
            "candidate_inputs": [
                "current_cube_drae",
                "frozen_range_axis",
                "frozen_azimuth_axis",
                "frozen_elevation_axis",
            ],
            "report_only_inputs": [TARGET_ARRAY],
            "forbidden_inputs": [
                "test_partition",
                "validation_partition",
                "cfar",
                "future_cube",
                "target_for_candidate_generation",
            ],
            "frame_count": len(records),
            "frames": frame_provenance,
        },
        "arms": summaries,
        "decision": decision,
        "total_elapsed_seconds": time.perf_counter() - sweep_started,
        "evidence_boundary": (
            "This read-only GT-aided support diagnosis measures whether frozen "
            "Cube-only activation covers target voxels. It is not a model, "
            "training, validation, test, or point-cloud geometry result."
        ),
    }
    if not math.isfinite(document["total_elapsed_seconds"]):
        raise FloatingPointError("R-B2 diagnostic runtime is non-finite")
    atomic_json(args.output, document)
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
