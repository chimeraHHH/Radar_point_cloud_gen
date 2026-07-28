#!/usr/bin/env python3
"""Run one source-bound G1G forward/backward on the largest target frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from losses.g1g_hierarchy import g1g_hierarchy_loss  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from scripts.g1b_contract import sha256  # noqa: E402
from scripts.train_g1g_hierarchy import (  # noqa: E402
    FORMAL_TRAIN_COUNT,
    FORMAL_VALIDATION_COUNT,
    attach_center_xyz,
    build_model,
    frozen_config,
    move_frame,
    require_h200,
    verify_source_tree,
)


PROTOCOL = "g1g_max_target_h200_preflight_v1"


def cache_path(cache_root: Path, record: dict) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_"
        f"radar_{int(record['radar_index']):05d}.npz"
    )


def target_count(cache_root: Path, record: dict) -> int:
    with np.load(cache_path(cache_root, record)) as cache:
        return int(cache["target_xyz_confidence"].shape[0])


def select_max_target_index(counts: list[int]) -> int:
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("G1G preflight target counts must be non-empty and positive")
    return max(range(len(counts)), key=lambda index: (counts[index], -index))


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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"G1G preflight output exists: {args.output}")
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)
    config = frozen_config(smoke=False)
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train", "validation"),
    )
    if len(dataset) != FORMAL_TRAIN_COUNT + FORMAL_VALIDATION_COUNT:
        raise ValueError("G1G max-target preflight requires the frozen 100 frames")
    counts = [target_count(args.cache_root, record) for record in dataset.records]
    selected_index = select_max_target_index(counts)
    selected_record = dataset.records[selected_index]

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    axes = load_axes(args.data_root / "resources")
    model = build_model(config, axes, normalization).to(device).train()
    item = dataset[selected_index]
    cube = move_frame(item, device)
    target = item["target_xyz_confidence"].to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = attach_center_xyz(model, model(cube))
    loss = g1g_hierarchy_loss(
        output,
        target,
        expected_point_count=config.point_count,
        expected_center_count=config.center_count,
        expected_children_per_center=config.children_per_center,
        geometry_weight=config.geometry_weight,
        outlier_weight=config.outlier_weight,
        child_existence_weight=config.child_existence_weight,
        center_coverage_weight=config.center_coverage_weight,
        center_existence_weight=config.center_existence_weight,
        center_repulsion_weight=config.center_repulsion_weight,
        final_repulsion_weight=config.final_repulsion_weight,
        child_diversity_weight=config.child_diversity_weight,
        child_bound_weight=config.child_bound_weight,
        outlier_threshold_m=config.outlier_threshold_m,
        existence_radius_m=config.existence_radius_m,
        center_repulsion_distance_m=config.center_repulsion_distance_m,
        final_repulsion_distance_m=config.final_repulsion_distance_m,
        child_diversity_diagonal_fraction=(
            config.child_diversity_diagonal_fraction
        ),
    )
    loss.total.backward()
    torch.cuda.synchronize(device)
    elapsed_seconds = time.monotonic() - started
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    checks = {
        "largest_target_frame_selected": (
            target.shape[0] == max(counts)
        ),
        "exact_10000_output": (
            tuple(output["xyz_m"].shape) == (1, config.point_count, 3)
        ),
        "finite_loss": bool(torch.isfinite(loss.total).item()),
        "finite_nonzero_gradients": (
            bool(gradients)
            and all(bool(torch.isfinite(value).all().item()) for value in gradients)
            and any(bool(torch.count_nonzero(value).item()) for value in gradients)
        ),
        "no_test_partition_accessed": True,
    }
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "source_hashes": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str((repo / "code/models/g1g_hierarchical_allocator.py").resolve()): (
                sha256(repo / "code/models/g1g_hierarchical_allocator.py")
            ),
            str((repo / "code/losses/g1g_hierarchy.py").resolve()): sha256(
                repo / "code/losses/g1g_hierarchy.py"
            ),
            str((repo / "code/scripts/train_g1g_hierarchy.py").resolve()): (
                sha256(repo / "code/scripts/train_g1g_hierarchy.py")
            ),
        },
        "data": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "cache_file": str(
                cache_path(args.cache_root, selected_record).resolve()
            ),
            "cache_sha256": sha256(
                cache_path(args.cache_root, selected_record)
            ),
            "sequence": int(item["sequence"]),
            "radar_index": int(item["radar_index"]),
            "partition": str(item["partition"]),
            "target_count": int(target.shape[0]),
            "frame_count_screened": len(dataset),
        },
        "runtime": {
            "device": args.device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "model_parameter_count": parameter_count(model),
            "elapsed_seconds": elapsed_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "loss": {
            "total": float(loss.total.detach().item()),
            "components": {
                name: float(value.item())
                for name, value in loss.components.items()
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    atomic_json(args.output, document)
    print(json.dumps(document, indent=2), flush=True)
    if not document["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
