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

from models.cube_cycle import CubeCycleNet, continuous_rae_to_xyz  # noqa: E402
from models.cube_temporal import CubeTemporalNet, FUSION_MODES  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from models.point_to_cube import soft_splat_features  # noqa: E402
from losses.temporal_consistency import temporal_match, temporal_radial_loss  # noqa: E402
from eval.temporal_cube import temporal_consistency_report  # noqa: E402
from models.temporal_prior import (  # noqa: E402
    WarpedPrior,
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

    consistent_current_xyz = warped.xyz_m.detach().clone()
    consistent_match = temporal_match(
        xyz,
        probability,
        confidence,
        consistent_current_xyz,
        probability,
        confidence,
        transform,
        delta_seconds,
        doppler_mps,
        doppler_lower,
        doppler_period,
        dynamic_threshold_mps=0.1,
    )
    consistent_report = temporal_consistency_report(
        consistent_match,
        confidence,
        consistent_current_xyz.detach(),
        confidence,
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    perturbed_xyz = (
        warped.xyz_m + 0.5 * radial_direction
    ).detach().requires_grad_(True)
    perturbed_match = temporal_match(
        xyz,
        probability,
        confidence,
        perturbed_xyz,
        probability,
        confidence,
        transform,
        delta_seconds,
        doppler_mps,
        doppler_lower,
        doppler_period,
        dynamic_threshold_mps=0.1,
    )
    perturbed_temporal_loss = temporal_radial_loss(perturbed_match)
    perturbed_temporal_loss.backward()
    perturbed_temporal_error = float(perturbed_match.radial_error_m.mean().item())
    temporal_xyz_gradient = float(perturbed_xyz.grad.norm().item())

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
    model_kwargs = {
        "head_mode": "physics_distribution",
        "doppler_mps": doppler_mps,
        "range_m": range_m,
        "azimuth_rad": azimuth_rad,
        "elevation_rad": elevation_rad,
        "base_channels": 8,
        "static_hypothesis": "zero_centered",
    }
    parent = CubeCycleNet(**model_kwargs).to(device).eval()
    small_cube = torch.rand(1, 64, 16, 12, 8, device=device)
    query_indices = torch.stack(
        (
            torch.randint(0, 16, (32,), device=device),
            torch.randint(0, 12, (32,), device=device),
            torch.randint(0, 8, (32,), device=device),
        ),
        dim=1,
    )
    query_xyz = continuous_rae_to_xyz(
        query_indices.float(), range_m, azimuth_rad, elevation_rad
    )
    ego_speed = torch.tensor([5.0], device=device)
    with torch.no_grad():
        parent_occupancy, parent_features = parent(small_cube)
        parent_query = parent.query_cycle(
            parent_features, query_indices, ego_speed
        )
    fusion_coordinates = torch.stack(
        (
            torch.rand(48, device=device) * 15.0,
            torch.rand(48, device=device) * 11.0,
            torch.rand(48, device=device) * 7.0,
        ),
        dim=1,
    )
    fusion_xyz = continuous_rae_to_xyz(
        fusion_coordinates, range_m, azimuth_rad, elevation_rad
    )
    fusion_probability = torch.softmax(torch.randn(48, 64, device=device), dim=1)
    fusion_confidence = torch.sigmoid(torch.randn(48, device=device))
    fusion_prior = WarpedPrior(
        xyz_m=fusion_xyz,
        coordinates_rae=fusion_coordinates,
        valid=torch.ones(48, dtype=torch.bool, device=device),
        dynamic_gate=torch.rand(48, device=device) > 0.5,
        residual_doppler_mps=torch.randn(48, device=device),
        probability=fusion_probability,
        confidence=fusion_confidence,
    )
    small_raster = rasterize_temporal_prior(
        fusion_prior,
        doppler_mps,
        doppler_lower,
        doppler_period,
        spatial_shape=(16, 12, 8),
    )
    fusion_reports = {}
    maximum_fallback_error = 0.0
    maximum_parameter_increase = 0.0
    all_temporal_gradients_nonzero = True
    parent_parameters = parameter_count(parent)
    for fusion_mode in FUSION_MODES:
        model = CubeTemporalNet(fusion_mode=fusion_mode, **model_kwargs).to(device)
        missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
        if not missing or unexpected:
            raise RuntimeError(
                f"Unexpected temporal initialization for {fusion_mode}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        with torch.no_grad():
            fallback_occupancy, fallback_features = model.forward_temporal(
                small_cube, None
            )
            fallback_query = model.query_temporal(
                fallback_features,
                query_indices,
                query_xyz,
                ego_speed,
                None,
            )
        fallback_error = max(
            float((fallback_occupancy - parent_occupancy).abs().max().item()),
            *(
                float((fallback_query[key] - parent_query[key]).abs().max().item())
                for key in ("probability", "offset_rae_bins", "xyz_m")
            ),
        )
        maximum_fallback_error = max(maximum_fallback_error, fallback_error)
        model.train()
        model.zero_grad(set_to_none=True)
        condition_raster = small_raster if fusion_mode == "concat" else None
        occupancy, features = model.forward_temporal(small_cube, condition_raster)
        prediction = model.query_temporal(
            features,
            query_indices,
            query_xyz,
            ego_speed,
            fusion_prior,
        )
        model_loss = (
            occupancy.square().mean()
            + prediction["offset_rae_bins"].square().mean()
            + prediction["logits"].square().mean()
        )
        model_loss.backward()
        temporal_gradient_norms = {
            name: float(parameter.grad.norm().item())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and any(
                prefix in name
                for prefix in (
                    "prior_",
                    "concat_",
                    "relative_",
                    "temporal_",
                    "draft_",
                )
            )
        }
        temporal_gradient_max = max(temporal_gradient_norms.values(), default=0.0)
        temporal_gradients_ok = bool(
            np.isfinite(temporal_gradient_max) and temporal_gradient_max > 0.0
        )
        all_temporal_gradients_nonzero &= temporal_gradients_ok
        parameters = parameter_count(model)
        relative_increase = (parameters - parent_parameters) / parent_parameters
        maximum_parameter_increase = max(maximum_parameter_increase, relative_increase)
        fusion_reports[fusion_mode] = {
            "parameters": parameters,
            "relative_parameter_increase": relative_increase,
            "missing_parent_key_count": len(missing),
            "fallback_max_error": fallback_error,
            "temporal_gradient_max": temporal_gradient_max,
            "temporal_gradient_parameter_count": len(temporal_gradient_norms),
        }
        del model, occupancy, features, prediction, model_loss
        torch.cuda.empty_cache()
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
        "consistent_temporal_radial_error_near_zero": consistent_report[
            "temporal_radial_error_mean_m"
        ]
        <= args.absolute_tolerance,
        "consistent_temporal_flicker_near_zero": consistent_report[
            "occupancy_flicker"
        ]
        <= 10.0 * args.absolute_tolerance,
        "temporal_metric_detects_radial_perturbation": perturbed_temporal_error
        >= consistent_report["temporal_radial_error_mean_m"] + 0.1,
        "temporal_loss_xyz_gradient_finite_nonzero": np.isfinite(
            temporal_xyz_gradient
        )
        and temporal_xyz_gradient > 0.0,
        "all_fusion_fallbacks_exact": maximum_fallback_error
        <= args.absolute_tolerance,
        "all_fusion_temporal_gradients_nonzero": all_temporal_gradients_nonzero,
        "fusion_parameter_increase_le_5pct": maximum_parameter_increase <= 0.05,
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
        "consistent_temporal_report": consistent_report,
        "perturbed_temporal_error_m": perturbed_temporal_error,
        "temporal_xyz_gradient_norm": temporal_xyz_gradient,
        "parent_model_parameters": parent_parameters,
        "maximum_fusion_parameter_increase": maximum_parameter_increase,
        "maximum_fusion_fallback_error": maximum_fallback_error,
        "fusion_reports": fusion_reports,
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
