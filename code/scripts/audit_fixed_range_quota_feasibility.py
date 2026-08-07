#!/usr/bin/env python3
"""Audit whether fixed per-frame radial quotas imply unavoidable outliers."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.fixed_range_quota_feasibility import (  # noqa: E402
    OUTLIER_DISTANCE_M,
    OUTPUT_QUOTAS,
    RANGE_INTERVALS_M,
    fixed_quota_feasibility,
)


PROTOCOL = "fixed_per_frame_range_quota_feasibility_v1"
EXPECTED_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
EXPECTED_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
TRAIN_FRAME_COUNT = 76
OUTLIER_GATE = 0.05
CHAMFER_GATE_M = 0.8
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("quota audit source commit must be a full lowercase SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("quota audit source commit does not match HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"quota audit source worktree is dirty: {dirty}")


def load_contract(
    manifest_path: Path,
    scene_split_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    hashes = {
        "manifest_sha256": sha256_file(manifest_path),
        "scene_split_sha256": sha256_file(scene_split_path),
    }
    expected = {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "scene_split_sha256": EXPECTED_SCENE_SPLIT_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"quota audit frozen inputs changed: {hashes}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(scene_split_path.read_text(encoding="utf-8"))
    if split.get("gate_pass") is not True:
        raise ValueError("quota audit scene split did not pass its gate")
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("quota audit manifest has no frame list")
    if any(str(frame.get("partition")) == "test" for frame in frames):
        raise ValueError("quota audit manifest contains a test record")
    train = [frame for frame in frames if frame.get("partition") == "train"]
    if len(train) != TRAIN_FRAME_COUNT:
        raise ValueError(f"quota audit requires 76 train frames, got {len(train)}")
    identities = [
        (int(frame["sequence"]), int(frame["radar_index"])) for frame in train
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("quota audit train identities are not unique")
    train_sequences = {
        int(value) for value in split["splits"]["train"]["sequences"]
    }
    if any(int(frame["sequence"]) not in train_sequences for frame in train):
        raise ValueError("quota audit train manifest contradicts scene split")
    return train, hashes


def cache_path(cache_root: Path, frame: dict[str, Any]) -> Path:
    return cache_root / (
        f"seq{int(frame['sequence']):02d}_"
        f"radar_{int(frame['radar_index']):05d}.npz"
    )


def load_target(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        if "target_xyz_confidence" not in cache.files:
            raise ValueError(f"quota audit target is absent: {path}")
        target = np.asarray(cache["target_xyz_confidence"], dtype=np.float32)
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError(f"quota audit target shape is invalid: {path}")
    if not np.isfinite(target).all():
        raise ValueError(f"quota audit target is non-finite: {path}")
    return target


def atomic_write_json(output: Path, document: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        staged = staging / output.name
        staged.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    frames, frozen_hashes = load_contract(args.manifest, args.scene_split)

    reports: list[dict[str, Any]] = []
    for position, frame in enumerate(frames):
        path = cache_path(args.cache_root, frame)
        target = load_target(path)
        feasibility = fixed_quota_feasibility(target[:, :3])
        reports.append(
            {
                "cohort_position": position,
                "sequence": int(frame["sequence"]),
                "radar_index": int(frame["radar_index"]),
                "partition": "train",
                "target_cache_path": str(path.resolve()),
                "target_cache_sha256": sha256_file(path),
                "target_tensor_sha256": array_sha256(target),
                **asdict(feasibility),
                "contract_impossible_at_5pct": (
                    feasibility.forced_outlier_fraction > OUTLIER_GATE
                ),
                "contract_impossible_at_0p8m_chamfer": (
                    feasibility.forced_chamfer_lower_bound_m > CHAMFER_GATE_M
                ),
            }
        )

    impossible = [
        report
        for report in reports
        if report["contract_impossible_at_5pct"]
        or report["contract_impossible_at_0p8m_chamfer"]
    ]
    histogram = Counter(
        str(report["forced_outlier_fraction"]) for report in reports
    )
    passed = not impossible
    status = (
        "fixed_range_quota_contract_radially_feasible"
        if passed
        else "fixed_range_quota_contract_no_go"
    )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": status,
        "passed": passed,
        "source_commit": args.source_commit,
        "runtime": {
            "host": subprocess.check_output(("hostname",), text=True).strip(),
            "elapsed_seconds": time.monotonic() - started,
        },
        "input_binding": {
            **frozen_hashes,
            "cache_root": str(args.cache_root.resolve()),
            "train_frame_count": len(reports),
            "validation_cache_accessed": False,
            "test_accessed": False,
        },
        "contract": {
            "range_intervals_m": [list(interval) for interval in RANGE_INTERVALS_M],
            "output_quotas": list(OUTPUT_QUOTAS),
            "output_count": sum(OUTPUT_QUOTAS),
            "outlier_distance_m": OUTLIER_DISTANCE_M,
            "maximum_outlier_fraction": OUTLIER_GATE,
            "maximum_chamfer_m": CHAMFER_GATE_M,
            "completeness_lower_bound_m": 0.0,
            "lower_bound_uses_only_reverse_triangle_inequality": True,
        },
        "summary": {
            "impossible_frame_count": len(impossible),
            "outlier_impossible_frame_count": sum(
                report["contract_impossible_at_5pct"] for report in reports
            ),
            "chamfer_impossible_frame_count": sum(
                report["contract_impossible_at_0p8m_chamfer"]
                for report in reports
            ),
            "impossible_identities": [
                [report["sequence"], report["radar_index"]]
                for report in impossible
            ],
            "maximum_forced_outlier_fraction": max(
                report["forced_outlier_fraction"] for report in reports
            ),
            "maximum_forced_chamfer_lower_bound_m": max(
                report["forced_chamfer_lower_bound_m"] for report in reports
            ),
            "forced_outlier_fraction_histogram": dict(sorted(histogram.items())),
        },
        "frames": reports,
        "decision": {
            "hard_per_frame_quotas_reusable": passed,
            "qlocal_training_authorized": False,
            "ray_hazard_with_same_hard_quotas_authorized": passed,
            "adaptive_range_mass_or_variable_cardinality_requires_new_protocol": (
                not passed
            ),
        },
        "evidence_boundary": {
            "model_evaluated": False,
            "candidate_field_evaluated": False,
            "cube_accessed": False,
            "cfar_accessed": False,
            "validation_accessed": False,
            "test_accessed": False,
            "deployable_allocation_policy_established": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = run(args)
    atomic_write_json(args.output, document)
    print(json.dumps({"status": document["status"], "output": str(args.output)}))
    return 0 if document["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
