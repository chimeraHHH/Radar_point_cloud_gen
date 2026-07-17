#!/usr/bin/env python3
"""Produce a reproducible GPU unit-test artifact for point-to-Cube splatting."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.cube_cycle import cube_cycle_loss  # noqa: E402
from models.point_to_cube import soft_splat_raed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--point-count", type=int, default=256)
    parser.add_argument("--spatial-shape", type=int, nargs=3, default=[16, 12, 8])
    parser.add_argument("--doppler-bins", type=int, default=64)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Renderer verification requires CUDA")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.point_count < 8 or args.doppler_bins < 2:
        raise ValueError("Renderer verification dimensions are too small")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    shape = tuple(args.spatial_shape)
    coordinates = torch.rand(args.point_count, 3, device=device)
    scale = torch.tensor(shape, dtype=torch.float32, device=device) - 1.0
    coordinates = (coordinates * scale).requires_grad_(True)
    logits = torch.randn(
        args.point_count, args.doppler_bins, device=device, requires_grad=True
    )
    confidence_logits = torch.randn(
        args.point_count, device=device, requires_grad=True
    )
    probability = torch.softmax(logits, dim=1)
    confidence = torch.sigmoid(confidence_logits)
    rendered = soft_splat_raed(coordinates, probability, confidence, shape)
    conservation_error = float(
        (rendered.energy_drae.sum() - confidence.sum()).abs().item()
    )

    cube = torch.rand(
        args.doppler_bins, *shape, dtype=torch.float32, device=device
    )
    loss, components = cube_cycle_loss(
        rendered, cube, confidence, variant="full"
    )
    loss.backward()
    gradient_norms = {
        "coordinates": float(coordinates.grad.norm().item()),
        "doppler_logits": float(logits.grad.norm().item()),
        "confidence_logits": float(confidence_logits.grad.norm().item()),
    }

    boundary_coordinates = torch.tensor(
        [
            [-0.49, 0.0, 0.0],
            [shape[0] - 0.51, shape[1] - 1.0, shape[2] - 1.0],
            [0.0, -0.49, shape[2] - 1.0],
            [shape[0] - 1.0, shape[1] - 0.51, -0.49],
        ],
        dtype=torch.float32,
        device=device,
    )
    boundary_probability = torch.full(
        (4, args.doppler_bins),
        1.0 / args.doppler_bins,
        dtype=torch.float32,
        device=device,
    )
    boundary_confidence = torch.tensor(
        [0.2, 0.4, 0.6, 0.8], dtype=torch.float32, device=device
    )
    boundary_rendered = soft_splat_raed(
        boundary_coordinates,
        boundary_probability,
        boundary_confidence,
        shape,
    )
    boundary_error = float(
        (
            boundary_rendered.energy_drae.sum()
            - boundary_confidence.sum()
        )
        .abs()
        .item()
    )

    permutation = torch.randperm(args.point_count, device=device)
    permuted = soft_splat_raed(
        coordinates.detach()[permutation],
        probability.detach()[permutation],
        confidence.detach()[permutation],
        shape,
    )
    permutation_error = float(
        (permuted.energy_drae - rendered.energy_drae.detach()).abs().max().item()
    )
    checks = {
        "energy_conserved": conservation_error <= args.absolute_tolerance,
        "boundary_energy_conserved": boundary_error <= args.absolute_tolerance,
        "point_permutation_invariant": permutation_error <= args.absolute_tolerance,
        "coordinate_gradient_finite_nonzero": np.isfinite(
            gradient_norms["coordinates"]
        )
        and gradient_norms["coordinates"] > 0.0,
        "doppler_gradient_finite_nonzero": np.isfinite(
            gradient_norms["doppler_logits"]
        )
        and gradient_norms["doppler_logits"] > 0.0,
        "confidence_gradient_finite_nonzero": np.isfinite(
            gradient_norms["confidence_logits"]
        )
        and gradient_norms["confidence_logits"] > 0.0,
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "point_count": args.point_count,
        "spatial_shape": list(shape),
        "doppler_bins": args.doppler_bins,
        "absolute_tolerance": args.absolute_tolerance,
        "conservation_error": conservation_error,
        "boundary_conservation_error": boundary_error,
        "permutation_max_error": permutation_error,
        "gradient_norms": gradient_norms,
        "cycle_loss": float(loss.detach().item()),
        "cycle_components": {
            key: float(value.item()) for key, value in components.items()
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
