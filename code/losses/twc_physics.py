"""Physical motion primitives and losses for forced-temporal T-WC models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.cube_doppler import circular_mean, wrapped_delta
from models.temporal_prior import transform_points


@dataclass(frozen=True)
class MandatoryRadialWarp:
    """History points after mandatory Doppler displacement and ego alignment."""

    xyz_m: torch.Tensor
    pre_ego_xyz_m: torch.Tensor
    ego_only_xyz_m: torch.Tensor
    radial_direction: torch.Tensor
    radial_velocity_mps: torch.Tensor
    radial_displacement_m: torch.Tensor


@dataclass(frozen=True)
class TWCMotion:
    """Persistent-point motion with explicit analytic and learned components."""

    xyz_m: torch.Tensor
    tangential_velocity_mps: torch.Tensor
    radial_correction_mps: torch.Tensor
    learned_displacement_m: torch.Tensor


@dataclass(frozen=True)
class TWCPhysicsLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def _batch_scalar(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    scalar = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if scalar.ndim == 0:
        scalar = scalar.expand(batch_size)
    if scalar.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape (B,)")
    if not torch.isfinite(scalar).all():
        raise ValueError(f"{name} must be finite")
    return scalar


def _batched_transform(
    points_xyz_m: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    if points_xyz_m.ndim != 3 or points_xyz_m.shape[-1] != 3:
        raise ValueError("Batched points must have shape (B,N,3)")
    batch_size = points_xyz_m.shape[0]
    if transform.shape != (batch_size, 4, 4):
        raise ValueError("Batched transforms must have shape (B,4,4)")
    return torch.stack(
        [
            transform_points(points_xyz_m[index], transform[index])
            for index in range(batch_size)
        ],
        dim=0,
    )


def mandatory_radial_doppler_warp(
    history_xyz_m: torch.Tensor,
    history_doppler_probability: torch.Tensor,
    current_from_history: torch.Tensor,
    delta_seconds: torch.Tensor | float,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
) -> MandatoryRadialWarp:
    """Apply radial Doppler displacement to every point before ego alignment.

    There is deliberately no dynamic threshold, learned gate, or switch that can
    disable this displacement. A zero displacement is possible only when the
    supplied circular Doppler mean or time interval is itself zero.
    """

    if history_xyz_m.ndim != 3 or history_xyz_m.shape[-1] != 3:
        raise ValueError("History XYZ must have shape (B,N,3)")
    batch_size, point_count, _ = history_xyz_m.shape
    expected_probability_shape = (
        batch_size,
        point_count,
        doppler_mps.numel(),
    )
    if history_doppler_probability.shape != expected_probability_shape:
        raise ValueError(
            "History Doppler probability must have shape "
            f"{expected_probability_shape}"
        )
    if current_from_history.shape != (batch_size, 4, 4):
        raise ValueError("current_from_history must have shape (B,4,4)")
    probability = history_doppler_probability.clamp_min(0.0)
    probability = probability / probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    radial_velocity = circular_mean(
        probability,
        doppler_mps.to(probability),
        doppler_lower_mps.to(probability),
        doppler_period_mps.to(probability),
    )
    radial_direction = history_xyz_m / torch.linalg.vector_norm(
        history_xyz_m, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    delta = _batch_scalar(
        delta_seconds,
        history_xyz_m,
        batch_size,
        "delta_seconds",
    )
    radial_displacement = radial_velocity * delta[:, None]
    pre_ego = (
        history_xyz_m
        + radial_displacement[:, :, None] * radial_direction
    )
    return MandatoryRadialWarp(
        xyz_m=_batched_transform(pre_ego, current_from_history),
        pre_ego_xyz_m=pre_ego,
        ego_only_xyz_m=_batched_transform(
            history_xyz_m, current_from_history
        ),
        radial_direction=radial_direction,
        radial_velocity_mps=radial_velocity,
        radial_displacement_m=radial_displacement,
    )


def tangential_component(
    vector_mps: torch.Tensor,
    radial_direction: torch.Tensor,
) -> torch.Tensor:
    """Project a predicted velocity onto the plane orthogonal to the ray."""

    if vector_mps.shape != radial_direction.shape:
        raise ValueError("Velocity and radial direction must have matching shape")
    projection = (vector_mps * radial_direction).sum(
        dim=-1, keepdim=True
    )
    return vector_mps - projection * radial_direction


def compose_persistent_motion(
    warp: MandatoryRadialWarp,
    raw_tangential_velocity_mps: torch.Tensor,
    raw_radial_correction: torch.Tensor,
    current_from_history: torch.Tensor,
    delta_seconds: torch.Tensor | float,
    *,
    maximum_tangential_speed_mps: float,
    maximum_radial_correction_mps: float,
) -> TWCMotion:
    """Add learned tangential motion and bounded radial calibration."""

    if maximum_tangential_speed_mps <= 0.0:
        raise ValueError("Maximum tangential speed must be positive")
    if maximum_radial_correction_mps <= 0.0:
        raise ValueError("Maximum radial correction must be positive")
    if raw_tangential_velocity_mps.shape != warp.radial_direction.shape:
        raise ValueError("Raw tangential velocity does not match persistent points")
    if raw_radial_correction.shape not in (
        warp.radial_velocity_mps.shape,
        (*warp.radial_velocity_mps.shape, 1),
    ):
        raise ValueError("Raw radial correction does not match persistent points")
    raw_radial_correction = raw_radial_correction.reshape(
        warp.radial_velocity_mps.shape
    )
    batch_size = warp.xyz_m.shape[0]
    delta = _batch_scalar(
        delta_seconds,
        warp.xyz_m,
        batch_size,
        "delta_seconds",
    )
    raw_tangential = torch.tanh(raw_tangential_velocity_mps)
    tangential = tangential_component(
        raw_tangential, warp.radial_direction
    )
    tangential_norm = torch.linalg.vector_norm(
        tangential, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    tangential = tangential / tangential_norm
    requested_norm = torch.linalg.vector_norm(
        raw_tangential, dim=-1, keepdim=True
    ).clamp_max(1.0)
    tangential = (
        tangential
        * requested_norm
        * maximum_tangential_speed_mps
    )
    radial_correction = (
        torch.tanh(raw_radial_correction)
        * maximum_radial_correction_mps
    )
    learned_velocity = (
        tangential
        + radial_correction[:, :, None] * warp.radial_direction
    )
    learned_displacement = learned_velocity * delta[:, None, None]
    corrected_pre_ego = warp.pre_ego_xyz_m + learned_displacement
    return TWCMotion(
        xyz_m=_batched_transform(
            corrected_pre_ego, current_from_history
        ),
        tangential_velocity_mps=tangential,
        radial_correction_mps=radial_correction,
        learned_displacement_m=learned_displacement,
    )


def inverse_radial_loss(
    history_xyz_m: torch.Tensor,
    current_xyz_m: torch.Tensor,
    current_from_history: torch.Tensor,
    delta_seconds: torch.Tensor | float,
    predicted_doppler_probability: torch.Tensor,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    *,
    huber_delta_mps: float = 0.5,
) -> torch.Tensor:
    """Match position-inferred radial speed to generated circular Doppler."""

    if history_xyz_m.shape != current_xyz_m.shape:
        raise ValueError("Inverse radial point sets must have matching shape")
    if history_xyz_m.ndim != 3 or history_xyz_m.shape[-1] != 3:
        raise ValueError("Inverse radial point sets must have shape (B,N,3)")
    batch_size, point_count, _ = history_xyz_m.shape
    if predicted_doppler_probability.shape != (
        batch_size,
        point_count,
        doppler_mps.numel(),
    ):
        raise ValueError("Predicted Doppler does not match inverse-radial points")
    delta = _batch_scalar(
        delta_seconds,
        history_xyz_m,
        batch_size,
        "delta_seconds",
    )
    if (delta.abs() < 1e-8).any():
        raise ValueError("Inverse radial loss requires non-zero time intervals")
    ego_only = _batched_transform(
        history_xyz_m, current_from_history
    )
    inferred = (
        torch.linalg.vector_norm(current_xyz_m, dim=-1)
        - torch.linalg.vector_norm(ego_only, dim=-1)
    ) / delta[:, None]
    probability = predicted_doppler_probability.clamp_min(0.0)
    probability = probability / probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    predicted = circular_mean(
        probability,
        doppler_mps.to(probability),
        doppler_lower_mps.to(probability),
        doppler_period_mps.to(probability),
    )
    error = wrapped_delta(
        inferred,
        predicted,
        doppler_period_mps.to(inferred),
    )
    return F.huber_loss(
        error,
        torch.zeros_like(error),
        delta=huber_delta_mps,
    )


def motion_cycle_loss(
    history_xyz_m: torch.Tensor,
    current_xyz_m: torch.Tensor,
    current_from_history: torch.Tensor,
    delta_seconds: torch.Tensor | float,
    radial_velocity_mps: torch.Tensor,
    tangential_velocity_mps: torch.Tensor,
    radial_correction_mps: torch.Tensor,
    *,
    huber_delta_m: float = 0.25,
) -> torch.Tensor:
    """Invert the explicit motion decomposition and recover history points."""

    if history_xyz_m.shape != current_xyz_m.shape:
        raise ValueError("Cycle point sets must have matching shape")
    if history_xyz_m.ndim != 3 or history_xyz_m.shape[-1] != 3:
        raise ValueError("Cycle point sets must have shape (B,N,3)")
    batch_size, point_count, _ = history_xyz_m.shape
    if radial_velocity_mps.shape != (batch_size, point_count):
        raise ValueError("Cycle radial velocity does not match points")
    if tangential_velocity_mps.shape != history_xyz_m.shape:
        raise ValueError("Cycle tangential velocity does not match points")
    if radial_correction_mps.shape != (batch_size, point_count):
        raise ValueError("Cycle radial correction does not match points")
    delta = _batch_scalar(
        delta_seconds,
        history_xyz_m,
        batch_size,
        "delta_seconds",
    )
    inverse_transform = torch.linalg.inv(current_from_history)
    pre_ego_current = _batched_transform(
        current_xyz_m, inverse_transform
    )
    radial_direction = history_xyz_m / torch.linalg.vector_norm(
        history_xyz_m, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    velocity = (
        (radial_velocity_mps + radial_correction_mps)[:, :, None]
        * radial_direction
        + tangential_velocity_mps
    )
    reconstructed = pre_ego_current - velocity * delta[:, None, None]
    return F.huber_loss(
        reconstructed,
        history_xyz_m,
        delta=huber_delta_m,
    )


def twc_physics_loss(
    output: dict[str, torch.Tensor],
    *,
    inverse_weight: float = 1.0,
    cycle_weight: float = 1.0,
) -> TWCPhysicsLoss:
    """Compute inverse-radial and cycle losses from a T-WC model output."""

    required = (
        "selected_history_xyz_m",
        "persistent_xyz_m",
        "selected_history_doppler_probability",
        "persistent_doppler_probability",
        "current_from_history",
        "delta_seconds",
        "doppler_mps",
        "doppler_lower_mps",
        "doppler_period_mps",
        "analytic_radial_velocity_mps",
        "tangential_velocity_mps",
        "radial_correction_mps",
    )
    missing = [name for name in required if name not in output]
    if missing:
        raise KeyError(f"T-WC physics output is missing {missing}")
    inverse = inverse_radial_loss(
        output["selected_history_xyz_m"],
        output["persistent_xyz_m"],
        output["current_from_history"],
        output["delta_seconds"],
        output["persistent_doppler_probability"],
        output["doppler_mps"],
        output["doppler_lower_mps"],
        output["doppler_period_mps"],
    )
    cycle = motion_cycle_loss(
        output["selected_history_xyz_m"],
        output["persistent_xyz_m"],
        output["current_from_history"],
        output["delta_seconds"],
        output["analytic_radial_velocity_mps"],
        output["tangential_velocity_mps"],
        output["radial_correction_mps"],
    )
    components = {
        "inverse_radial": inverse,
        "motion_cycle": cycle,
    }
    total = inverse_weight * inverse + cycle_weight * cycle
    return TWCPhysicsLoss(total=total, components=components)
