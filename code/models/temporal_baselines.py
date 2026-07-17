"""Deterministic recurrent-copy and DoppDrive-style temporal baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cube_dense.parent_prediction import PointPrediction
from models.temporal_prior import gated_doppler_warp


@dataclass(frozen=True)
class AggregationDiagnostics:
    candidate_count: int
    unique_voxel_count: int
    suppressed_count: int
    fill_count: int
    history_count: int


def analytic_static_center(
    xyz_m: torch.Tensor,
    ego_speed_mps: torch.Tensor | float,
    static_hypothesis: str,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
) -> torch.Tensor:
    """Match the parent head's static-center convention at continuous XYZ."""

    if static_hypothesis not in ("negative_ego", "positive_ego", "zero_centered"):
        raise ValueError(f"Unsupported static hypothesis {static_hypothesis}")
    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
        raise ValueError("Static-center points must have shape (N,3)")
    speed = torch.as_tensor(
        ego_speed_mps, dtype=xyz_m.dtype, device=xyz_m.device
    ).reshape(-1)
    if speed.numel() != 1:
        raise ValueError("Temporal baselines require one ego speed per frame")
    radius = torch.linalg.vector_norm(xyz_m, dim=1).clamp_min(1e-8)
    radial_projection = speed[0] * xyz_m[:, 0] / radius
    if static_hypothesis == "negative_ego":
        center = -radial_projection
    elif static_hypothesis == "positive_ego":
        center = radial_projection
    else:
        center = torch.zeros_like(radial_projection)
    return torch.remainder(
        center - doppler_lower_mps, doppler_period_mps
    ) + doppler_lower_mps


def warp_prediction(
    state: PointPrediction,
    current_from_previous: torch.Tensor,
    delta_seconds: torch.Tensor | float,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    current_ego_speed_mps: torch.Tensor | float,
    static_hypothesis: str,
    apply_doppler_displacement: bool,
    dynamic_threshold_mps: float = 1.0,
) -> PointPrediction:
    prior = gated_doppler_warp(
        state.xyz_m,
        state.probability,
        state.confidence,
        current_from_previous,
        delta_seconds,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
        range_m,
        azimuth_rad,
        elevation_rad,
        previous_static_center_mps=state.static_center_mps,
        dynamic_threshold_mps=dynamic_threshold_mps,
        apply_doppler_displacement=apply_doppler_displacement,
    )
    return PointPrediction(
        xyz_m=prior.xyz_m,
        coordinates_rae=prior.coordinates_rae,
        probability=prior.probability,
        confidence=prior.confidence,
        static_center_mps=analytic_static_center(
            prior.xyz_m,
            current_ego_speed_mps,
            static_hypothesis,
            doppler_lower_mps,
            doppler_period_mps,
        ),
    )


def in_spatial_grid(
    coordinates_rae: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    valid = torch.isfinite(coordinates_rae).all(dim=1)
    for axis, size in enumerate(spatial_shape):
        valid &= (coordinates_rae[:, axis] >= 0.0) & (
            coordinates_rae[:, axis] <= size - 1
        )
    return valid


def doppdrive_aggregate(
    current: PointPrediction,
    warped_history_newest_first: list[PointPrediction],
    point_count: int,
    current_ego_speed_mps: torch.Tensor | float,
    static_hypothesis: str,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    spatial_shape: tuple[int, int, int] = (256, 107, 37),
    recency_time_constant_frames: float = 4.0,
) -> tuple[PointPrediction, AggregationDiagnostics]:
    """Suppress confidence-ranked RAE voxels and return an exact-size cloud."""

    if point_count < 1:
        raise ValueError("DoppDrive point count must be positive")
    if recency_time_constant_frames <= 0.0:
        raise ValueError("DoppDrive recency time constant must be positive")
    aged_states = [(0, current)]
    for age, history in enumerate(warped_history_newest_first, start=1):
        valid = in_spatial_grid(history.coordinates_rae, spatial_shape)
        if valid.any():
            aged_states.append(
                (
                    age,
                    PointPrediction(
                        xyz_m=history.xyz_m[valid],
                        coordinates_rae=history.coordinates_rae[valid],
                        probability=history.probability[valid],
                        confidence=history.confidence[valid],
                        static_center_mps=history.static_center_mps[valid],
                    ),
                )
            )
    states = [state for _, state in aged_states]
    xyz = torch.cat([state.xyz_m for state in states])
    coordinates = torch.cat([state.coordinates_rae for state in states])
    probability = torch.cat([state.probability for state in states])
    confidence = torch.cat([state.confidence for state in states])
    if xyz.shape[0] < point_count:
        raise ValueError(
            f"DoppDrive has {xyz.shape[0]} candidates for {point_count} outputs"
        )
    ages = torch.cat(
        [
            torch.full(
                (state.xyz_m.shape[0],),
                age,
                dtype=confidence.dtype,
                device=confidence.device,
            )
            for age, state in aged_states
        ]
    )
    recency = torch.exp(-ages / recency_time_constant_frames)
    base_score = confidence.clamp_min(0.0).double() * recency.double()
    candidate_index = torch.arange(xyz.shape[0], device=xyz.device)
    score = base_score + (xyz.shape[0] - candidate_index).double() * 1e-12

    quantized = torch.floor(coordinates).long()
    quantized = torch.stack(
        [
            quantized[:, axis].clamp(0, spatial_shape[axis] - 1)
            for axis in range(3)
        ],
        dim=1,
    )
    flat_voxel = (
        quantized[:, 0] * spatial_shape[1] * spatial_shape[2]
        + quantized[:, 1] * spatial_shape[2]
        + quantized[:, 2]
    )
    voxel_max = torch.full(
        (spatial_shape[0] * spatial_shape[1] * spatial_shape[2],),
        -torch.inf,
        dtype=score.dtype,
        device=score.device,
    )
    voxel_max.scatter_reduce_(
        0, flat_voxel, score, reduce="amax", include_self=True
    )
    winner = score == voxel_max[flat_voxel]
    winner_index = torch.nonzero(winner, as_tuple=False).squeeze(1)
    unique_voxel_count = int(winner_index.numel())
    if winner_index.numel() >= point_count:
        selected_score = score[winner_index]
        selected = winner_index[
            torch.topk(selected_score, point_count, sorted=True).indices
        ]
        fill_count = 0
    else:
        remaining = torch.nonzero(~winner, as_tuple=False).squeeze(1)
        needed = point_count - winner_index.numel()
        fill = remaining[torch.topk(score[remaining], needed, sorted=True).indices]
        selected = torch.cat((winner_index, fill))
        fill_count = int(needed)
    result = PointPrediction(
        xyz_m=xyz[selected],
        coordinates_rae=coordinates[selected],
        probability=probability[selected],
        confidence=confidence[selected],
        static_center_mps=analytic_static_center(
            xyz[selected],
            current_ego_speed_mps,
            static_hypothesis,
            doppler_lower_mps,
            doppler_period_mps,
        ),
    )
    diagnostics = AggregationDiagnostics(
        candidate_count=int(xyz.shape[0]),
        unique_voxel_count=unique_voxel_count,
        suppressed_count=int(xyz.shape[0] - unique_voxel_count),
        fill_count=fill_count,
        history_count=len(states) - 1,
    )
    return result, diagnostics
