"""Matched radial-motion consistency for recurrent dense radar points."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.cube_doppler import circular_mean, wrapped_delta
from models.point_to_cube import soft_splat_features
from models.temporal_prior import (
    nearest_point_indices,
    transform_points,
    xyz_to_continuous_rae,
)


@dataclass(frozen=True)
class TemporalMatch:
    radial_error_m: torch.Tensor
    match_distance_m: torch.Tensor
    weight: torch.Tensor
    previous_to_current_index: torch.Tensor
    expected_prior_xyz_m: torch.Tensor
    ego_only_prior_xyz_m: torch.Tensor


def temporal_match(
    previous_xyz_m: torch.Tensor,
    previous_probability: torch.Tensor,
    previous_confidence: torch.Tensor,
    current_xyz_m: torch.Tensor,
    current_probability: torch.Tensor,
    current_confidence: torch.Tensor,
    current_from_previous: torch.Tensor,
    delta_seconds: torch.Tensor | float,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    previous_static_center_mps: torch.Tensor | None = None,
    current_static_center_mps: torch.Tensor | None = None,
    dynamic_threshold_mps: float = 1.0,
    matching_scale_m: float = 2.0,
) -> TemporalMatch:
    previous_scalar = circular_mean(
        previous_probability,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
    )
    current_scalar = circular_mean(
        current_probability,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
    )
    if previous_static_center_mps is None:
        previous_static_center_mps = torch.zeros_like(previous_scalar)
    if current_static_center_mps is None:
        current_static_center_mps = torch.zeros_like(current_scalar)
    if previous_static_center_mps.shape != previous_scalar.shape:
        raise ValueError("Previous static center does not match points")
    if current_static_center_mps.shape != current_scalar.shape:
        raise ValueError("Current static center does not match points")
    previous_residual = wrapped_delta(
        previous_scalar, previous_static_center_mps, doppler_period_mps
    )
    current_residual = wrapped_delta(
        current_scalar, current_static_center_mps, doppler_period_mps
    )
    gate = previous_residual.abs() > dynamic_threshold_mps
    delta = torch.as_tensor(
        delta_seconds,
        dtype=previous_xyz_m.dtype,
        device=previous_xyz_m.device,
    )
    radial_direction = previous_xyz_m / torch.linalg.vector_norm(
        previous_xyz_m, dim=1, keepdim=True
    ).clamp_min(1e-8)
    advanced = previous_xyz_m + (
        previous_residual * gate.to(previous_residual)
    )[:, None] * delta * radial_direction
    ego_only = transform_points(previous_xyz_m, current_from_previous)
    expected = transform_points(advanced, current_from_previous)
    current_index, distance = nearest_point_indices(
        expected, current_xyz_m, neighbor_count=1
    )
    current_index = current_index[:, 0]
    distance = distance[:, 0]
    matched_current_xyz = current_xyz_m[current_index]
    matched_current_residual = current_residual[current_index]
    delta_range = (
        torch.linalg.vector_norm(matched_current_xyz, dim=1)
        - torch.linalg.vector_norm(ego_only, dim=1)
    )
    expected_delta_range = 0.5 * (
        previous_residual + matched_current_residual
    ) * delta
    radial_error = (delta_range - expected_delta_range).abs()
    weight = (
        previous_confidence.clamp_min(0.0)
        * current_confidence[current_index].clamp_min(0.0)
        * torch.exp(-0.5 * (distance / matching_scale_m).square())
    )
    return TemporalMatch(
        radial_error_m=radial_error,
        match_distance_m=distance,
        weight=weight,
        previous_to_current_index=current_index,
        expected_prior_xyz_m=expected,
        ego_only_prior_xyz_m=ego_only,
    )


def temporal_radial_loss(
    match: TemporalMatch,
    huber_delta_m: float = 0.5,
) -> torch.Tensor:
    per_point = F.huber_loss(
        match.radial_error_m,
        torch.zeros_like(match.radial_error_m),
        delta=huber_delta_m,
        reduction="none",
    )
    return (per_point * match.weight).sum() / match.weight.sum().clamp_min(1e-8)


def occupancy_flicker(
    expected_prior_xyz_m: torch.Tensor,
    previous_confidence: torch.Tensor,
    current_xyz_m: torch.Tensor,
    current_confidence: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    spatial_shape: tuple[int, int, int] = (256, 107, 37),
) -> torch.Tensor:
    prior_coordinates, prior_valid = xyz_to_continuous_rae(
        expected_prior_xyz_m, range_m, azimuth_rad, elevation_rad
    )
    current_coordinates, current_valid = xyz_to_continuous_rae(
        current_xyz_m, range_m, azimuth_rad, elevation_rad
    )
    if not prior_valid.any() or not current_valid.any():
        return current_xyz_m.new_tensor(1.0)
    prior = soft_splat_features(
        prior_coordinates[prior_valid],
        torch.ones(
            int(prior_valid.sum()), 1, device=current_xyz_m.device, dtype=current_xyz_m.dtype
        ),
        previous_confidence[prior_valid],
        spatial_shape=spatial_shape,
    ).weight_rae.clamp_max(1.0)
    current = soft_splat_features(
        current_coordinates[current_valid],
        torch.ones(
            int(current_valid.sum()), 1, device=current_xyz_m.device, dtype=current_xyz_m.dtype
        ),
        current_confidence[current_valid],
        spatial_shape=spatial_shape,
    ).weight_rae.clamp_max(1.0)
    prior = prior / prior.sum().clamp_min(1e-8)
    current = current / current.sum().clamp_min(1e-8)
    overlap = torch.minimum(prior, current).sum()
    return 1.0 - overlap
