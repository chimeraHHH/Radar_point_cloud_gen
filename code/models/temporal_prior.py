"""Physical Doppler warp and differentiable temporal-prior rasterization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.cube_doppler import circular_mean, wrapped_delta
from models.point_to_cube import soft_splat_features


@dataclass(frozen=True)
class WarpedPrior:
    xyz_m: torch.Tensor
    coordinates_rae: torch.Tensor
    valid: torch.Tensor
    dynamic_gate: torch.Tensor
    residual_doppler_mps: torch.Tensor
    probability: torch.Tensor
    confidence: torch.Tensor


def transform_points(points_xyz: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got {points_xyz.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"Expected one 4x4 transform, got {transform.shape}")
    rotation = transform[:3, :3].to(points_xyz)
    translation = transform[:3, 3].to(points_xyz)
    return points_xyz @ rotation.T + translation


def fractional_axis_index(axis: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if axis.ndim != 1 or axis.numel() < 2:
        raise ValueError("Axis must contain at least two values")
    axis = axis.to(values)
    if axis[-1] < axis[0]:
        reversed_index = fractional_axis_index(axis.flip(0), values)
        return axis.numel() - 1 - reversed_index
    if not torch.all(axis[1:] > axis[:-1]):
        raise ValueError("Axis must be strictly monotonic")
    upper = torch.searchsorted(axis, values.contiguous(), right=False)
    upper = upper.clamp(1, axis.numel() - 1)
    lower = upper - 1
    lower_value = axis[lower]
    upper_value = axis[upper]
    fraction = (values - lower_value) / (upper_value - lower_value).clamp_min(1e-12)
    return lower.to(values) + fraction


def xyz_to_continuous_rae(
    xyz_m: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = torch.linalg.vector_norm(xyz_m, dim=1)
    azimuth = torch.atan2(xyz_m[:, 1], xyz_m[:, 0])
    horizontal = torch.linalg.vector_norm(xyz_m[:, :2], dim=1)
    elevation = torch.atan2(xyz_m[:, 2], horizontal)
    valid = (
        (radius >= range_m.min())
        & (radius <= range_m.max())
        & (azimuth >= azimuth_rad.min())
        & (azimuth <= azimuth_rad.max())
        & (elevation >= elevation_rad.min())
        & (elevation <= elevation_rad.max())
    )
    coordinates = torch.stack(
        (
            fractional_axis_index(range_m, radius),
            fractional_axis_index(azimuth_rad, azimuth),
            fractional_axis_index(elevation_rad, elevation),
        ),
        dim=1,
    )
    return coordinates, valid


def gated_doppler_warp(
    previous_xyz_m: torch.Tensor,
    previous_probability: torch.Tensor,
    previous_confidence: torch.Tensor,
    current_from_previous: torch.Tensor,
    delta_seconds: torch.Tensor | float,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    previous_static_center_mps: torch.Tensor | None = None,
    dynamic_threshold_mps: float = 1.0,
    apply_doppler_displacement: bool = True,
) -> WarpedPrior:
    point_count = previous_xyz_m.shape[0]
    if previous_probability.ndim != 2 or previous_probability.shape[0] != point_count:
        raise ValueError("Previous Doppler distributions do not match points")
    if previous_confidence.shape != (point_count,):
        raise ValueError("Previous confidence does not match points")
    scalar = circular_mean(
        previous_probability,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
    )
    if previous_static_center_mps is None:
        previous_static_center_mps = torch.zeros_like(scalar)
    if previous_static_center_mps.shape != scalar.shape:
        raise ValueError("Previous static centers do not match points")
    residual = wrapped_delta(
        scalar, previous_static_center_mps, doppler_period_mps
    )
    dynamic_gate = residual.abs() > dynamic_threshold_mps
    displacement_velocity = (
        residual * dynamic_gate.to(residual)
        if apply_doppler_displacement
        else torch.zeros_like(residual)
    )
    radial_direction = previous_xyz_m / torch.linalg.vector_norm(
        previous_xyz_m, dim=1, keepdim=True
    ).clamp_min(1e-8)
    delta = torch.as_tensor(delta_seconds, dtype=previous_xyz_m.dtype, device=previous_xyz_m.device)
    advanced = previous_xyz_m + displacement_velocity[:, None] * delta * radial_direction
    warped_xyz = transform_points(advanced, current_from_previous)
    coordinates, valid = xyz_to_continuous_rae(
        warped_xyz, range_m, azimuth_rad, elevation_rad
    )
    return WarpedPrior(
        xyz_m=warped_xyz,
        coordinates_rae=coordinates,
        valid=valid,
        dynamic_gate=dynamic_gate,
        residual_doppler_mps=residual,
        probability=previous_probability,
        confidence=previous_confidence,
    )


def prior_point_features(
    prior: WarpedPrior,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    probability = prior.probability / prior.probability.sum(
        dim=1, keepdim=True
    ).clamp_min(epsilon)
    angle = 2.0 * torch.pi * (
        doppler_mps - doppler_lower_mps
    ) / doppler_period_mps
    sine = (probability * torch.sin(angle)[None]).sum(dim=1)
    cosine = (probability * torch.cos(angle)[None]).sum(dim=1)
    entropy = -(
        probability * probability.clamp_min(epsilon).log()
    ).sum(dim=1) / torch.log(probability.new_tensor(probability.shape[1], dtype=torch.float32))
    return torch.stack(
        (
            sine,
            cosine,
            entropy,
            prior.dynamic_gate.to(probability),
        ),
        dim=1,
    )


def rasterize_temporal_prior(
    prior: WarpedPrior,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    spatial_shape: tuple[int, int, int] = (256, 107, 37),
) -> torch.Tensor:
    valid = prior.valid & torch.isfinite(prior.coordinates_rae).all(dim=1)
    if not valid.any():
        return prior.xyz_m.new_zeros((5, *spatial_shape))
    features = prior_point_features(
        prior, doppler_mps, doppler_lower_mps, doppler_period_mps
    )
    splat = soft_splat_features(
        prior.coordinates_rae[valid],
        features[valid],
        prior.confidence[valid],
        spatial_shape=spatial_shape,
    )
    energy = torch.log1p(splat.weight_rae)[None]
    return torch.cat((energy, splat.feature_crae), dim=0)
