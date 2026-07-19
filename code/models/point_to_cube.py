"""Differentiable trilinear point-to-RAED soft splatting."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SoftSplatResult:
    energy_drae: torch.Tensor
    spatial_energy_rae: torch.Tensor
    normalized_spectrum_drae: torch.Tensor
    covered_rae: torch.Tensor


@dataclass(frozen=True)
class FeatureSplatResult:
    feature_crae: torch.Tensor
    weight_rae: torch.Tensor
    covered_rae: torch.Tensor


def trilinear_query_features(
    features_bcrae: torch.Tensor,
    coordinates_rae: torch.Tensor,
    batch_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a spatial feature grid at continuous RAE coordinates."""

    if features_bcrae.ndim != 5:
        raise ValueError(
            f"Expected feature grid (B,C,R,A,E), got {features_bcrae.shape}"
        )
    if coordinates_rae.ndim != 2 or coordinates_rae.shape[1] != 3:
        raise ValueError(
            f"Expected continuous (N,3) RAE coordinates, got {coordinates_rae.shape}"
        )
    point_count = coordinates_rae.shape[0]
    if batch_index is None:
        if features_bcrae.shape[0] != 1:
            raise ValueError("Batch indices are required for multi-batch queries")
        batch_index = torch.zeros(
            point_count, dtype=torch.long, device=coordinates_rae.device
        )
    if batch_index.shape != (point_count,):
        raise ValueError("Batch indices must contain one value per query")
    batch_index = batch_index.long()
    if (batch_index < 0).any() or (batch_index >= features_bcrae.shape[0]).any():
        raise IndexError("Query batch index is out of bounds")

    spatial_shape = tuple(int(size) for size in features_bcrae.shape[2:])
    neighbors, weights = trilinear_neighbors(coordinates_rae, spatial_shape)
    range_count, azimuth_count, elevation_count = spatial_shape
    flat_index = (
        neighbors[:, :, 0] * azimuth_count * elevation_count
        + neighbors[:, :, 1] * elevation_count
        + neighbors[:, :, 2]
    )
    flattened = features_bcrae.flatten(start_dim=2).transpose(1, 2)
    sampled = flattened[batch_index[:, None], flat_index]
    return (sampled * weights[:, :, None].to(sampled)).sum(dim=1)


def trilinear_neighbors(
    coordinates_rae: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if coordinates_rae.ndim != 2 or coordinates_rae.shape[1] != 3:
        raise ValueError(f"Expected continuous (N,3) RAE coordinates, got {coordinates_rae.shape}")
    lower = torch.floor(coordinates_rae).long()
    fraction = coordinates_rae - lower.to(coordinates_rae)
    neighbor_indices = []
    neighbor_weights = []
    for range_upper in (0, 1):
        for azimuth_upper in (0, 1):
            for elevation_upper in (0, 1):
                offset = torch.tensor(
                    [range_upper, azimuth_upper, elevation_upper],
                    dtype=torch.long,
                    device=coordinates_rae.device,
                )
                index = lower + offset
                choose_upper = offset.to(fraction).view(1, 3)
                weight = torch.where(
                    choose_upper.bool(), fraction, 1.0 - fraction
                ).prod(dim=1)
                valid = torch.ones_like(weight, dtype=torch.bool)
                for axis, size in enumerate(spatial_shape):
                    valid &= (index[:, axis] >= 0) & (index[:, axis] < size)
                clamped = torch.stack(
                    [
                        index[:, axis].clamp(0, spatial_shape[axis] - 1)
                        for axis in range(3)
                    ],
                    dim=1,
                )
                neighbor_indices.append(clamped)
                neighbor_weights.append(weight * valid.to(weight))
    indices = torch.stack(neighbor_indices, dim=1)
    weights = torch.stack(neighbor_weights, dim=1)
    weight_sum = weights.sum(dim=1, keepdim=True)
    weights = torch.where(
        weight_sum > 0, weights / weight_sum.clamp_min(1e-12), weights
    )
    return indices, weights


def soft_splat_raed(
    coordinates_rae: torch.Tensor,
    doppler_probability: torch.Tensor,
    confidence: torch.Tensor,
    spatial_shape: tuple[int, int, int] = (256, 107, 37),
    epsilon: float = 1e-8,
) -> SoftSplatResult:
    if doppler_probability.ndim != 2 or doppler_probability.shape[0] != coordinates_rae.shape[0]:
        raise ValueError("Doppler distributions must have one row per point")
    if confidence.shape != (coordinates_rae.shape[0],):
        raise ValueError("Confidence must have one value per point")
    if any(size <= 0 for size in spatial_shape):
        raise ValueError(f"Invalid spatial shape {spatial_shape}")
    probability = doppler_probability.clamp_min(0.0)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(epsilon)
    confidence = confidence.clamp(0.0, 1.0)
    neighbors, weights = trilinear_neighbors(coordinates_rae, spatial_shape)
    range_count, azimuth_count, elevation_count = spatial_shape
    flat_index = (
        neighbors[:, :, 0] * azimuth_count * elevation_count
        + neighbors[:, :, 1] * elevation_count
        + neighbors[:, :, 2]
    )
    contribution = (
        confidence[:, None, None]
        * weights[:, :, None]
        * probability[:, None, :]
    )
    spatial_count = range_count * azimuth_count * elevation_count
    doppler_count = probability.shape[1]
    flat_energy = torch.zeros(
        spatial_count,
        doppler_count,
        dtype=contribution.dtype,
        device=contribution.device,
    )
    flat_energy.scatter_add_(
        0,
        flat_index.reshape(-1, 1).expand(-1, doppler_count),
        contribution.reshape(-1, doppler_count),
    )
    energy_raed = flat_energy.reshape(
        range_count, azimuth_count, elevation_count, doppler_count
    )
    spatial_energy = energy_raed.sum(dim=3)
    normalized = energy_raed / spatial_energy[..., None].clamp_min(epsilon)
    covered = spatial_energy > epsilon
    return SoftSplatResult(
        energy_drae=energy_raed.permute(3, 0, 1, 2).contiguous(),
        spatial_energy_rae=spatial_energy,
        normalized_spectrum_drae=normalized.permute(3, 0, 1, 2).contiguous(),
        covered_rae=covered,
    )


def soft_splat_features(
    coordinates_rae: torch.Tensor,
    point_features: torch.Tensor,
    point_weight: torch.Tensor,
    spatial_shape: tuple[int, int, int] = (256, 107, 37),
    epsilon: float = 1e-8,
) -> FeatureSplatResult:
    """Confidence-weighted trilinear mean of point features on an RAE grid."""

    if point_features.ndim != 2 or point_features.shape[0] != coordinates_rae.shape[0]:
        raise ValueError("Point features must have one row per coordinate")
    if point_weight.shape != (coordinates_rae.shape[0],):
        raise ValueError("Point weights must have one value per coordinate")
    if any(size <= 0 for size in spatial_shape):
        raise ValueError(f"Invalid spatial shape {spatial_shape}")
    neighbors, interpolation_weight = trilinear_neighbors(
        coordinates_rae, spatial_shape
    )
    range_count, azimuth_count, elevation_count = spatial_shape
    flat_index = (
        neighbors[:, :, 0] * azimuth_count * elevation_count
        + neighbors[:, :, 1] * elevation_count
        + neighbors[:, :, 2]
    )
    weighted_interpolation = interpolation_weight * point_weight.clamp_min(0.0)[:, None]
    spatial_count = range_count * azimuth_count * elevation_count
    channel_count = point_features.shape[1]
    flat_weight = torch.zeros(
        spatial_count,
        dtype=point_features.dtype,
        device=point_features.device,
    )
    flat_weight.scatter_add_(0, flat_index.reshape(-1), weighted_interpolation.reshape(-1))
    flat_features = torch.zeros(
        spatial_count,
        channel_count,
        dtype=point_features.dtype,
        device=point_features.device,
    )
    contribution = (
        weighted_interpolation[:, :, None] * point_features[:, None, :]
    )
    flat_features.scatter_add_(
        0,
        flat_index.reshape(-1, 1).expand(-1, channel_count),
        contribution.reshape(-1, channel_count),
    )
    normalized = flat_features / flat_weight[:, None].clamp_min(epsilon)
    feature_crae = normalized.reshape(
        range_count, azimuth_count, elevation_count, channel_count
    ).permute(3, 0, 1, 2).contiguous()
    weight_rae = flat_weight.reshape(spatial_shape)
    return FeatureSplatResult(
        feature_crae=feature_crae,
        weight_rae=weight_rae,
        covered_rae=weight_rae > epsilon,
    )
