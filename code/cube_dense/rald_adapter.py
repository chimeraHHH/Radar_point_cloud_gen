"""K-Radar target and query adapters for the matched RaLD-style baseline."""

from __future__ import annotations

import math

import numpy as np
import torch

from cube_dense.kradar import KRadarAxes


def _axis_tensor(
    values: np.ndarray, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def normalize_axis(values: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    lower = axis.amin()
    upper = axis.amax()
    span = upper - lower
    if not torch.isfinite(span) or span <= 0:
        raise ValueError("Physical axis must have a finite nonzero span")
    return 2.0 * (values - lower) / span - 1.0


def xyz_to_normalized_rae(points_xyz: torch.Tensor, axes: KRadarAxes) -> torch.Tensor:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected target XYZ shape (N,3), got {points_xyz.shape}")
    radius = torch.linalg.vector_norm(points_xyz, dim=1)
    azimuth = torch.atan2(points_xyz[:, 1], points_xyz[:, 0])
    elevation = torch.asin(
        (points_xyz[:, 2] / radius.clamp_min(1e-8)).clamp(-1.0, 1.0)
    )
    range_axis = _axis_tensor(
        axes.range_m, device=points_xyz.device, dtype=points_xyz.dtype
    )
    azimuth_axis = _axis_tensor(
        axes.azimuth_rad, device=points_xyz.device, dtype=points_xyz.dtype
    )
    elevation_axis = _axis_tensor(
        axes.elevation_rad, device=points_xyz.device, dtype=points_xyz.dtype
    )
    return torch.stack(
        (
            normalize_axis(radius, range_axis),
            normalize_axis(azimuth, azimuth_axis),
            normalize_axis(elevation, elevation_axis),
        ),
        dim=1,
    ).clamp(-1.0, 1.0)


def indices_to_normalized_rae(
    indices_rae: torch.Tensor, axes: KRadarAxes
) -> torch.Tensor:
    if indices_rae.ndim != 2 or indices_rae.shape[1] != 3:
        raise ValueError(f"Expected RAE indices (N,3), got {indices_rae.shape}")
    device = indices_rae.device
    dtype = torch.float32
    physical_axes = (
        _axis_tensor(axes.range_m, device=device, dtype=dtype),
        _axis_tensor(axes.azimuth_rad, device=device, dtype=dtype),
        _axis_tensor(axes.elevation_rad, device=device, dtype=dtype),
    )
    return torch.stack(
        tuple(
            normalize_axis(axis[indices_rae[:, dimension]], axis)
            for dimension, axis in enumerate(physical_axes)
        ),
        dim=1,
    )


def rae_sum_condition(
    cube_drae: torch.Tensor, center: float, scale: float
) -> torch.Tensor:
    if cube_drae.ndim != 5 or cube_drae.shape[1] != 64:
        raise ValueError(f"Expected Cube shape (B,64,R,A,E), got {cube_drae.shape}")
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("RAE-Sum normalization must be finite with positive scale")
    log_power = torch.log10(cube_drae.clamp_min(0.0).sum(dim=1, keepdim=True) + 1.0)
    return ((log_power - center) / scale).clamp(-4.0, 4.0)


def _weighted_sample_indices(
    weights: torch.Tensor, count: int, generator: torch.Generator
) -> torch.Tensor:
    if weights.ndim != 1 or not weights.numel():
        raise ValueError("Sampling weights must be a nonempty vector")
    if count <= 0:
        raise ValueError("Sample count must be positive")
    probabilities = weights.float().clamp_min(0.0)
    if probabilities.sum() <= 0.0:
        probabilities = torch.ones_like(probabilities)
    replacement = count > probabilities.numel()
    return torch.multinomial(
        probabilities,
        count,
        replacement=replacement,
        generator=generator,
    )


def sample_target_points(
    target_xyz_confidence: torch.Tensor,
    axes: KRadarAxes,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if target_xyz_confidence.ndim != 2 or target_xyz_confidence.shape[1] != 4:
        raise ValueError(
            f"Expected radar-observable target (N,4), got {target_xyz_confidence.shape}"
        )
    selected = _weighted_sample_indices(
        target_xyz_confidence[:, 3], count, generator
    )
    return xyz_to_normalized_rae(target_xyz_confidence[selected, :3], axes)


def _flat_rae_indices(indices: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return (
        indices[:, 0] * shape[1] * shape[2]
        + indices[:, 1] * shape[2]
        + indices[:, 2]
    )


def _unflatten_rae_indices(
    flat: torch.Tensor, shape: tuple[int, int, int]
) -> torch.Tensor:
    range_index = flat // (shape[1] * shape[2])
    remainder = flat % (shape[1] * shape[2])
    azimuth_index = remainder // shape[2]
    elevation_index = remainder % shape[2]
    return torch.stack((range_index, azimuth_index, elevation_index), dim=1)


def sample_empty_indices(
    occupied_indices: torch.Tensor,
    shape: tuple[int, int, int],
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("Empty-query count must be positive")
    cell_count = math.prod(shape)
    occupied = torch.zeros(cell_count, dtype=torch.bool, device=occupied_indices.device)
    occupied[_flat_rae_indices(occupied_indices.long(), shape)] = True
    available = (~occupied).nonzero(as_tuple=False).flatten()
    if count > available.numel():
        raise ValueError("Requested more unique empty queries than available cells")
    order = torch.randperm(
        available.numel(), device=available.device, generator=generator
    )[:count]
    return _unflatten_rae_indices(available[order], shape)


def sample_occupancy_queries(
    target_xyz_confidence: torch.Tensor,
    target_rae_index: torch.Tensor,
    axes: KRadarAxes,
    positive_count: int,
    negative_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = _weighted_sample_indices(
        target_xyz_confidence[:, 3], positive_count, generator
    )
    positive = xyz_to_normalized_rae(
        target_xyz_confidence[selected, :3], axes
    )
    positive_labels = target_xyz_confidence[selected, 3].float().clamp(0.0, 1.0)
    shape = (
        len(axes.range_m),
        len(axes.azimuth_rad),
        len(axes.elevation_rad),
    )
    negative_indices = sample_empty_indices(
        target_rae_index, shape, negative_count, generator
    )
    negative = indices_to_normalized_rae(negative_indices, axes)
    queries = torch.cat((positive, negative), dim=0)
    labels = torch.cat(
        (positive_labels, torch.zeros(negative_count, device=queries.device)), dim=0
    )
    permutation = torch.randperm(
        queries.shape[0], device=queries.device, generator=generator
    )
    return queries[permutation], labels[permutation]


def rald_occupancy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    posterior_kl: torch.Tensor,
    positive_weight: float = 0.1,
    negative_weight: float = 1.0,
    kl_weight: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.shape != labels.shape:
        raise ValueError(f"Logit/label mismatch: {logits.shape} vs {labels.shape}")
    positive = labels > 0.0
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("RaLD occupancy loss requires positive and negative queries")
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels.to(logits), reduction="none"
    )
    positive_loss = elementwise[positive].mean()
    negative_loss = elementwise[negative].mean()
    kl_loss = posterior_kl.mean()
    total = (
        positive_weight * positive_loss
        + negative_weight * negative_loss
        + kl_weight * kl_loss
    )
    return total, {
        "positive_bce": positive_loss,
        "negative_bce": negative_loss,
        "kl": kl_loss,
    }


@torch.inference_mode()
def decode_grid_topk(
    autoencoder,
    latent: torch.Tensor,
    axes: KRadarAxes,
    point_count: int = 10_000,
    query_chunk_size: int = 8_192,
) -> tuple[torch.Tensor, torch.Tensor]:
    if latent.ndim != 3 or latent.shape[0] != 1:
        raise ValueError("Grid decoding currently requires one latent sample")
    shape = (
        len(axes.range_m),
        len(axes.azimuth_rad),
        len(axes.elevation_rad),
    )
    cell_count = math.prod(shape)
    if not 0 < point_count <= cell_count:
        raise ValueError("Point count must fit inside the RAE grid")
    if query_chunk_size <= 0:
        raise ValueError("Query chunk size must be positive")
    best_logits = torch.empty(0, dtype=torch.float32, device=latent.device)
    best_flat = torch.empty(0, dtype=torch.long, device=latent.device)
    for start in range(0, cell_count, query_chunk_size):
        flat = torch.arange(
            start,
            min(start + query_chunk_size, cell_count),
            device=latent.device,
        )
        indices = _unflatten_rae_indices(flat, shape)
        queries = indices_to_normalized_rae(indices, axes).to(latent)
        logits = autoencoder.decode(latent, queries.unsqueeze(0))[0].float()
        candidate_logits = torch.cat((best_logits, logits), dim=0)
        candidate_flat = torch.cat((best_flat, flat), dim=0)
        keep = min(point_count, candidate_logits.numel())
        best_logits, selected = torch.topk(candidate_logits, keep, sorted=True)
        best_flat = candidate_flat[selected]
    return _unflatten_rae_indices(best_flat, shape), best_logits.sigmoid()
