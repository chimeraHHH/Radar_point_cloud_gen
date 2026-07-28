#!/usr/bin/env python3
"""Run the H200-only structural preflight for the G1G Stage-0 scaffold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_tesseract  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from models.g1g_hierarchical_allocator import (  # noqa: E402
    G1GConditionExclusiveHierarchy,
)

SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def finite_nonzero(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    return bool(torch.isfinite(tensor).all() and torch.count_nonzero(tensor) > 0)


def gradient_norm(module: torch.nn.Module) -> float | None:
    gradients = [
        parameter.grad.detach().float()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return None
    value = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
    return float(value.item())


def finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def load_normalization(path: Path) -> tuple[float, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    normalization = document.get("normalization", document)
    return float(normalization["center"]), float(normalization["scale"])


def git_commit(repo: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if not SOURCE_PATTERN.fullmatch(commit):
        raise RuntimeError("G1G requires a full Git source commit")
    return commit


def load_cube(
    data_root: Path,
    sequence: int,
    radar_index: int,
    device: torch.device,
) -> torch.Tensor:
    path = (
        data_root
        / str(sequence)
        / "radar_tesseract"
        / f"tesseract_{radar_index:05d}.mat"
    )
    return (
        torch.from_numpy(load_tesseract(path))
        .float()
        .unsqueeze(0)
        .to(device)
    )


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
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sequence", type=int, default=6)
    parser.add_argument("--radar-index", type=int, default=183)
    parser.add_argument("--shuffle-sequence", type=int, default=55)
    parser.add_argument("--shuffle-radar-index", type=int, default=376)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpu-name", default="NVIDIA H200 NVL")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    current_commit = git_commit(repo)
    if args.source_commit != current_commit:
        raise ValueError("G1G requested source differs from Git snapshot")
    if args.shuffle_sequence == args.sequence:
        raise ValueError("G1G preflight condition shuffle must cross scenes")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("G1G preflight requires CUDA on an H200")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    if device_name != args.required_gpu_name or "H200" not in device_name.upper():
        raise RuntimeError(
            f"G1G preflight requires {args.required_gpu_name}, got {device_name}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    axes = load_axes(args.data_root / "resources")
    log_center, log_scale = load_normalization(args.normalization)
    measured = load_cube(
        args.data_root,
        args.sequence,
        args.radar_index,
        device,
    )
    shuffled = load_cube(
        args.data_root,
        args.shuffle_sequence,
        args.shuffle_radar_index,
        device,
    )
    model = G1GConditionExclusiveHierarchy(
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        log_center=log_center,
        log_scale=log_scale,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        head_dim=args.head_dim,
    ).to(device)

    model.eval()
    with torch.inference_mode():
        current_allocation = model.allocate_centers(measured)
        shuffled_allocation = model.allocate_centers(shuffled)
        center_shuffle_delta = (
            current_allocation["center_coordinates_rae"]
            - shuffled_allocation["center_coordinates_rae"]
        ).abs()
        feature_shuffle_delta = (
            current_allocation["center_query_features"]
            - shuffled_allocation["center_query_features"]
        ).abs()

    model.train()
    measured_probe = measured.detach().requires_grad_(True)
    condition_probe = shuffled.detach().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats(device)
    output = model(
        measured_probe,
        condition_cube_drae=condition_probe,
    )
    allocation_loss = (
        output["center_coordinates_rae"].float().square().mean()
        + output["center_score_logit"].float().square().mean()
    )
    measured_allocation_gradient, condition_allocation_gradient = (
        torch.autograd.grad(
            allocation_loss,
            (measured_probe, condition_probe),
            allow_unused=True,
            retain_graph=True,
        )
    )
    total_loss = (
        allocation_loss
        + output["coordinates_rae"].float().square().mean()
        + output["confidence_logit"].float().square().mean()
        + output["center_cube_spectrum"].float().square().mean()
        + output["point_cube_spectrum"].float().square().mean()
    )
    total_loss.backward()
    torch.cuda.synchronize(device)

    child_offset_norm = torch.linalg.vector_norm(
        output["child_physical_offset_m"].float(),
        dim=-1,
    )
    physical_radius = output["center_cell_diagonal_m"].float()
    physical_bound_excess = (
        child_offset_norm - physical_radius
    ).clamp_min(0.0)
    maximum_bin_excess = (
        output["child_offset_bins"].detach().float().abs()
        - model.child_offset_bound_bins.detach().float()
    ).clamp_min(0.0)
    parent_counts = torch.bincount(
        output["point_parent_center_index"],
        minlength=model.CENTER_COUNT,
    )
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    metadata = model.architecture_metadata()
    group_gradients = {
        "radar_encoder": gradient_norm(model.radar_encoder),
        "center_queries": gradient_norm(model.center_queries),
        "allocation_blocks": gradient_norm(model.allocation_blocks),
        "center_coordinate_head": gradient_norm(
            model.center_coordinate_head
        ),
        "center_spectrum_projection": gradient_norm(
            model.center_spectrum_projection
        ),
        "patch_decoder": gradient_norm(model.patch_decoder),
        "child_offset_head": gradient_norm(model.child_offset_head),
    }
    checks = {
        "h200_identity": device_name == args.required_gpu_name,
        "global_token_shape": list(output["radar_tokens"].shape)
        == [1, 336, args.model_dim],
        "exact_center_shape": list(output["center_coordinates_rae"].shape)
        == [1, 2_500, 3],
        "exact_point_shape": list(output["coordinates_rae"].shape)
        == [1, 10_000, 3],
        "exact_four_children_per_center": bool(
            torch.all(parent_counts == 4)
        ),
        "pre_allocation_source_allowlist": metadata[
            "pre_allocation_sources"
        ]
        == [
            "learned_center_queries",
            "normalized_center_templates",
            "global_full_raed_tokens",
        ],
        "local_sampling_after_allocation": metadata[
            "local_cube_sampling_stage"
        ]
        == "after_center_allocation",
        "measured_cube_has_no_allocation_gradient": (
            measured_allocation_gradient is None
        ),
        "condition_cube_drives_allocation": finite_nonzero(
            condition_allocation_gradient
        ),
        "condition_shuffle_changes_center_features": bool(
            feature_shuffle_delta.max() > 0.0
        ),
        "condition_shuffle_changes_center_coordinates": bool(
            center_shuffle_delta.max() > 0.0
        ),
        "child_bin_offsets_bounded": bool(maximum_bin_excess.max() <= 1e-6),
        "child_physical_offsets_bounded": bool(
            physical_bound_excess.max() <= 1e-5
        ),
        "all_trainable_gradients_present": all(
            gradient is not None for gradient in trainable_gradients
        ),
        "all_trainable_gradients_finite": all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in trainable_gradients
        ),
        "required_gradient_groups_nonzero": all(
            finite_positive(value) for value in group_gradients.values()
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": "g1g_condition_exclusive_hierarchy_stage0_preflight",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": current_commit,
        "device": device_name,
        "seed": args.seed,
        "input": {
            "sequence": args.sequence,
            "radar_index": args.radar_index,
            "shuffle_sequence": args.shuffle_sequence,
            "shuffle_radar_index": args.shuffle_radar_index,
            "cube_shape": list(measured.shape),
        },
        "architecture": metadata,
        "parameter_count": parameter_count(model),
        "tensor_shapes": {
            "radar_tokens": list(output["radar_tokens"].shape),
            "centers": list(output["center_coordinates_rae"].shape),
            "center_spectrum": list(output["center_cube_spectrum"].shape),
            "child_offsets": list(output["child_offset_bins"].shape),
            "points": list(output["coordinates_rae"].shape),
            "point_spectrum": list(output["point_cube_spectrum"].shape),
        },
        "dependency_audit": {
            "measured_cube_allocation_gradient_is_none": (
                measured_allocation_gradient is None
            ),
            "condition_cube_allocation_gradient_nonzero": finite_nonzero(
                condition_allocation_gradient
            ),
            "condition_shuffle_center_coordinate_max_abs": float(
                center_shuffle_delta.max().item()
            ),
            "condition_shuffle_center_feature_mean_abs": float(
                feature_shuffle_delta.mean().item()
            ),
        },
        "bounds": {
            "child_offset_bound_bins": [
                float(value)
                for value in model.child_offset_bound_bins.detach().cpu()
            ],
            "maximum_bin_bound_excess": float(
                maximum_bin_excess.max().item()
            ),
            "maximum_physical_bound_excess_m": float(
                physical_bound_excess.max().item()
            ),
            "maximum_cell_diagonal_m": float(physical_radius.max().item()),
        },
        "gradient_norms": group_gradients,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "checks": checks,
        "passed": all(checks.values()),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
