#!/usr/bin/env python3
"""Produce a reproducible CUDA verification artifact for the temporal prior."""

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

from models.cube_cycle import continuous_rae_to_xyz  # noqa: E402
from models.point_to_cube import soft_splat_features  # noqa: E402
from models.temporal_prior import (  # noqa: E402
    gated_doppler_warp,
    rasterize_temporal_prior,
    transform_points,
    xyz_to_continuous_rae,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=2e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Temporal-prior verification requires CUDA")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    range_m = torch.linspace(1.0, 100.0, 256, device=device)
    azimuth_rad = torch.linspace(-0.8, 0.8, 107, device=device)
    elevation_rad = torch.linspace(-0.3, 0.3, 37, device=device)
    doppler_mps = torch.linspace(-10.0, 10.0 - 20.0 / 64.0, 64, device=device)
    doppler_step = doppler_mps[1] - doppler_mps[0]
    doppler_period = doppler_step * doppler_mps.numel()
    doppler_lower = doppler_mps[0]

    coordinates = torch.tensor(
        [[32.25, 41.5, 17.25], [100.5, 70.25, 8.75], [210.1, 52.0, 24.0]],
        dtype=torch.float32,
        device=device,
    )
    xyz = continuous_rae_to_xyz(
        coordinates, range_m, azimuth_rad, elevation_rad
    )
    recovered, valid = xyz_to_continuous_rae(
        xyz, range_m, azimuth_rad, elevation_rad
    )
    roundtrip_error = float((recovered - coordinates).abs().max().item())
    descending, descending_valid = xyz_to_continuous_rae(
        xyz, range_m, azimuth_rad.flip(0), elevation_rad.flip(0)
    )
    descending_expected = torch.stack(
        (
            coordinates[:, 0],
            azimuth_rad.numel() - 1 - coordinates[:, 1],
            elevation_rad.numel() - 1 - coordinates[:, 2],
        ),
        dim=1,
    )
    descending_error = float((descending - descending_expected).abs().max().item())

    probability = torch.zeros(3, 64, dtype=torch.float32, device=device)
    selected_bins = torch.tensor([38, 32, 26], device=device)
    probability.scatter_(1, selected_bins[:, None], 1.0)
    confidence = torch.tensor([0.8, 0.6, 0.4], device=device)
    transform = torch.eye(4, dtype=torch.float32, device=device)
    delta_seconds = 0.1
    warped = gated_doppler_warp(
        xyz,
        probability,
        confidence,
        transform,
        delta_seconds,
        doppler_mps,
        doppler_lower,
        doppler_period,
        range_m,
        azimuth_rad,
        elevation_rad,
        previous_static_center_mps=torch.zeros(3, device=device),
        dynamic_threshold_mps=0.1,
    )
    scalar = doppler_mps[selected_bins]
    radial_direction = xyz / torch.linalg.vector_norm(xyz, dim=1, keepdim=True)
    expected_warped = xyz + scalar[:, None] * delta_seconds * radial_direction
    doppler_warp_error = float((warped.xyz_m - expected_warped).abs().max().item())
    ego_only = gated_doppler_warp(
        xyz,
        probability,
        confidence,
        transform,
        delta_seconds,
        doppler_mps,
        doppler_lower,
        doppler_period,
        range_m,
        azimuth_rad,
        elevation_rad,
        apply_doppler_displacement=False,
    )
    ego_only_error = float((ego_only.xyz_m - xyz).abs().max().item())

    translated_transform = torch.eye(4, dtype=torch.float32, device=device)
    translated_transform[:3, 3] = torch.tensor([1.0, -2.0, 0.5], device=device)
    transformed = transform_points(xyz, translated_transform)
    transform_error = float(
        (transformed - xyz - translated_transform[:3, 3]).abs().max().item()
    )
    raster = rasterize_temporal_prior(
        warped,
        doppler_mps,
        doppler_lower,
        doppler_period,
    )

    gradient_coordinates = coordinates.detach().clone().requires_grad_(True)
    gradient_features = torch.randn(3, 4, device=device, requires_grad=True)
    confidence_logits = torch.randn(3, device=device, requires_grad=True)
    splat = soft_splat_features(
        gradient_coordinates,
        gradient_features,
        torch.sigmoid(confidence_logits),
        spatial_shape=(256, 107, 37),
    )
    conservation_error = float(
        (splat.weight_rae.sum() - torch.sigmoid(confidence_logits).sum())
        .abs()
        .item()
    )
    loss = splat.feature_crae.square().sum() + splat.weight_rae.square().sum()
    loss.backward()
    gradient_norms = {
        "coordinates": float(gradient_coordinates.grad.norm().item()),
        "features": float(gradient_features.grad.norm().item()),
        "confidence_logits": float(confidence_logits.grad.norm().item()),
    }
    checks = {
        "xyz_rae_roundtrip": bool(valid.all())
        and roundtrip_error <= args.absolute_tolerance,
        "descending_axes_supported": bool(descending_valid.all())
        and descending_error <= args.absolute_tolerance,
        "ego_only_identity": ego_only_error <= args.absolute_tolerance,
        "doppler_radial_displacement": doppler_warp_error <= args.absolute_tolerance,
        "rigid_transform_exact": transform_error <= args.absolute_tolerance,
        "prior_raster_shape": list(raster.shape) == [5, 256, 107, 37],
        "prior_raster_finite_nonzero": bool(torch.isfinite(raster).all())
        and float(raster.abs().sum().item()) > 0.0,
        "feature_splat_energy_conserved": conservation_error
        <= args.absolute_tolerance,
        "coordinate_gradient_finite_nonzero": np.isfinite(
            gradient_norms["coordinates"]
        )
        and gradient_norms["coordinates"] > 0.0,
        "feature_gradient_finite_nonzero": np.isfinite(gradient_norms["features"])
        and gradient_norms["features"] > 0.0,
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
        "absolute_tolerance": args.absolute_tolerance,
        "roundtrip_max_error_bins": roundtrip_error,
        "descending_axis_max_error_bins": descending_error,
        "ego_only_max_error_m": ego_only_error,
        "doppler_warp_max_error_m": doppler_warp_error,
        "rigid_transform_max_error_m": transform_error,
        "feature_splat_conservation_error": conservation_error,
        "gradient_norms": gradient_norms,
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
