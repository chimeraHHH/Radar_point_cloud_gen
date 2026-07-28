"""Deterministic synthetic contract data for D-MHW mechanism preflight."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dmhw.contracts import DIRECT_HORIZONS_SECONDS
from models.cube_cycle import continuous_rae_to_xyz
from models.temporal_prior import transform_points


def dmhw_axes(
    device: torch.device,
    *,
    full_raed: bool,
) -> tuple[torch.Tensor, ...]:
    if full_raed:
        range_count, azimuth_count, elevation_count = 256, 107, 37
    else:
        range_count, azimuth_count, elevation_count = 16, 11, 7
    return (
        torch.linspace(0.5, 120.0, range_count, device=device),
        torch.linspace(-1.15, 1.15, azimuth_count, device=device),
        torch.linspace(-0.35, 0.35, elevation_count, device=device),
        torch.linspace(-15.75, 15.75, 64, device=device),
    )


def _transform_batched(
    xyz_m: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            transform_points(xyz_m[index], transform[index])
            for index in range(xyz_m.shape[0])
        ]
    )


def make_dmhw_synthetic_contract(
    device: torch.device,
    *,
    point_count: int,
    persistent_fraction: float,
    batch_size: int = 2,
    history_frame_count: int = 3,
    full_raed: bool = False,
    seed: int = 20260716,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    tuple[torch.Tensor, ...],
]:
    """Create exact-count causal inputs and independent three-horizon targets."""

    if batch_size < 2:
        raise ValueError("D-MHW interventions require batch size at least two")
    if history_frame_count < 2:
        raise ValueError("D-MHW requires at least two history Cubes")
    persistent_count = round(point_count * persistent_fraction)
    persistent_count = min(max(persistent_count, 1), point_count - 1)
    birth_count = point_count - persistent_count
    observed_count = point_count
    axes = dmhw_axes(device, full_raed=full_raed)
    range_m, azimuth_rad, elevation_rad, doppler_mps = axes
    generator = torch.Generator(device=device).manual_seed(seed)
    cube_shape = (
        batch_size,
        doppler_mps.numel(),
        range_m.numel(),
        azimuth_rad.numel(),
        elevation_rad.numel(),
    )
    current_cube = torch.rand(
        cube_shape, generator=generator, device=device
    )
    history_cube = torch.rand(
        (batch_size, history_frame_count, *cube_shape[1:]),
        generator=generator,
        device=device,
    )
    batch_bias = torch.linspace(
        0.0, 0.2, batch_size, device=device
    ).view(batch_size, 1, 1, 1, 1)
    current_cube = current_cube + batch_bias
    time_bias = torch.linspace(
        0.0, 0.15, history_frame_count, device=device
    ).view(1, history_frame_count, 1, 1, 1, 1)
    history_cube = history_cube + time_bias + batch_bias[:, None]

    fraction = torch.linspace(
        0.0, 1.0, observed_count, device=device
    )
    radius = 8.0 + 86.0 * fraction
    azimuth = -0.95 + 1.90 * fraction
    elevation = 0.16 * torch.sin(8.0 * torch.pi * fraction)
    base_xyz = torch.stack(
        (
            radius * torch.cos(elevation) * torch.cos(azimuth),
            radius * torch.cos(elevation) * torch.sin(azimuth),
            radius * torch.sin(elevation),
        ),
        dim=1,
    )
    observed_xyz = torch.stack(
        [
            base_xyz
            + base_xyz.new_tensor([0.0, 0.35 * batch_index, 0.0])
            for batch_index in range(batch_size)
        ]
    )
    doppler_index = (
        torch.arange(observed_count, device=device) * 7 + 13
    ) % doppler_mps.numel()
    observed_probability = F.one_hot(
        doppler_index, doppler_mps.numel()
    ).to(current_cube)
    observed_probability = observed_probability[None].expand(
        batch_size, -1, -1
    ).clone()
    observed_probability[1] = torch.roll(
        observed_probability[1], shifts=9, dims=-1
    )
    observed_confidence = torch.linspace(
        0.99, 0.55, observed_count, device=device
    )[None].expand(batch_size, -1).clone()
    source_id = torch.arange(
        observed_count, dtype=torch.long, device=device
    )[None].expand(batch_size, -1).clone()
    source_id += (
        torch.arange(batch_size, device=device)[:, None] * point_count
    )
    history_time = torch.linspace(
        -0.3,
        -0.1,
        history_frame_count,
        device=device,
    )[None].expand(batch_size, -1).clone()
    horizons = current_cube.new_tensor(DIRECT_HORIZONS_SECONDS)
    target_from_current = torch.eye(
        4, device=device
    ).reshape(1, 1, 4, 4).repeat(
        batch_size, len(DIRECT_HORIZONS_SECONDS), 1, 1
    )
    target_from_current[:, :, 0, 3] = -1.5 * horizons[None]
    target_from_current[:, :, 1, 3] = (
        0.1 * torch.arange(batch_size, device=device)[:, None]
    )

    persistent_xyz = observed_xyz[:, :persistent_count]
    persistent_probability = observed_probability[:, :persistent_count]
    persistent_confidence = observed_confidence[:, :persistent_count]
    radial_direction = persistent_xyz / torch.linalg.vector_norm(
        persistent_xyz, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    radial_velocity = (
        persistent_probability * doppler_mps
    ).sum(dim=-1)
    tangent = torch.stack(
        (
            -radial_direction[:, :, 1],
            radial_direction[:, :, 0],
            torch.zeros_like(radial_direction[:, :, 0]),
        ),
        dim=-1,
    )
    tangent = tangent / torch.linalg.vector_norm(
        tangent, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    tangent = 0.4 * tangent
    persistent_targets = []
    for horizon_index, horizon in enumerate(horizons):
        moved = (
            persistent_xyz
            + horizon
            * (
                radial_velocity[:, :, None] * radial_direction
                + tangent
                + 0.1 * radial_direction
            )
        )
        persistent_targets.append(
            _transform_batched(
                moved, target_from_current[:, horizon_index]
            )
        )
    persistent_target_xyz = torch.stack(persistent_targets, dim=1)

    birth_fraction = torch.linspace(
        0.0, 1.0, birth_count, device=device
    )
    birth_targets = []
    birth_probabilities = []
    for horizon_index, horizon in enumerate(horizons):
        range_coordinate = (
            0.15 + 0.75 * birth_fraction
        ) * (range_m.numel() - 1)
        azimuth_coordinate = (
            0.1
            + 0.8
            * torch.remainder(
                birth_fraction + 0.09 * horizon_index, 1.0
            )
        ) * (azimuth_rad.numel() - 1)
        elevation_coordinate = (
            0.5
            + 0.35 * torch.sin(6.0 * torch.pi * birth_fraction)
        ) * (elevation_rad.numel() - 1)
        coordinate = torch.stack(
            (
                range_coordinate,
                azimuth_coordinate,
                elevation_coordinate,
            ),
            dim=-1,
        )
        xyz = continuous_rae_to_xyz(
            coordinate, range_m, azimuth_rad, elevation_rad
        )
        xyz = xyz[None].expand(batch_size, -1, -1).clone()
        xyz[:, :, 1] += (
            0.2 * torch.arange(batch_size, device=device)[:, None]
        )
        birth_targets.append(xyz)
        birth_index = (
            torch.arange(birth_count, device=device)
            + 5 * horizon_index
        ) % doppler_mps.numel()
        birth_probability = F.one_hot(
            birth_index, doppler_mps.numel()
        ).to(current_cube)
        birth_probabilities.append(
            birth_probability[None].expand(batch_size, -1, -1)
        )
    birth_target_xyz = torch.stack(birth_targets, dim=1)
    target_xyz = torch.cat(
        (persistent_target_xyz, birth_target_xyz), dim=2
    )
    persistent_target_probability = persistent_probability[
        :, None
    ].expand(-1, len(DIRECT_HORIZONS_SECONDS), -1, -1)
    birth_target_probability = torch.stack(
        birth_probabilities, dim=1
    )
    target_probability = torch.cat(
        (
            persistent_target_probability,
            birth_target_probability,
        ),
        dim=2,
    )
    target_confidence = torch.cat(
        (
            persistent_confidence[:, None].expand(
                -1, len(DIRECT_HORIZONS_SECONDS), -1
            ),
            torch.full(
                (
                    batch_size,
                    len(DIRECT_HORIZONS_SECONDS),
                    birth_count,
                ),
                0.8,
                dtype=current_cube.dtype,
                device=device,
            ),
        ),
        dim=2,
    )
    inputs = {
        "current_cube_drae": current_cube,
        "history_cube_drae": history_cube,
        "history_time_seconds": history_time,
        "observed_xyz_m": observed_xyz,
        "observed_doppler_probability": observed_probability,
        "observed_confidence": observed_confidence,
        "observed_source_id": source_id,
        "target_from_current": target_from_current,
    }
    targets = {
        "target_xyz_m": target_xyz,
        "target_doppler_probability": target_probability,
        "target_confidence": target_confidence,
    }
    return inputs, targets, axes
