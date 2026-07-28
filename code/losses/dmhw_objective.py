"""Stage-0 objectives for direct multi-horizon Doppler-world generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from losses.twc_physics import inverse_radial_loss, motion_cycle_loss


@dataclass(frozen=True)
class DMHWLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    attribute_supervision: str


def _chunked_nearest_mean(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    values = []
    for start in range(0, source_xyz.shape[0], chunk_size):
        distance = torch.cdist(
            source_xyz[start : start + chunk_size][None],
            target_xyz[None],
        )[0]
        values.append(distance.min(dim=1).values)
    return torch.cat(values).mean()


def direct_horizon_chamfer(
    prediction_xyz_m: torch.Tensor,
    target_xyz_m: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Symmetric set distance averaged across batch and direct horizons."""

    if prediction_xyz_m.ndim != 4 or prediction_xyz_m.shape[-1] != 3:
        raise ValueError("D-MHW prediction must have shape (B,H,N,3)")
    if target_xyz_m.ndim != 4 or target_xyz_m.shape[-1] != 3:
        raise ValueError("D-MHW target must have shape (B,H,M,3)")
    if prediction_xyz_m.shape[:2] != target_xyz_m.shape[:2]:
        raise ValueError("D-MHW prediction/target horizons do not align")
    if not torch.isfinite(target_xyz_m).all():
        raise ValueError("D-MHW target geometry must be finite")
    distances = []
    for batch_index in range(prediction_xyz_m.shape[0]):
        for horizon_index in range(prediction_xyz_m.shape[1]):
            prediction = prediction_xyz_m[batch_index, horizon_index]
            target = target_xyz_m[batch_index, horizon_index]
            distances.append(
                _chunked_nearest_mean(
                    prediction, target, chunk_size=chunk_size
                )
                + _chunked_nearest_mean(
                    target, prediction, chunk_size=chunk_size
                )
            )
    return torch.stack(distances).mean()


def _flatten_persistent(
    output: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    persistent = output["persistent_xyz_m"]
    batch_size, horizon_count, point_count, _ = persistent.shape
    selected_xyz = output["selected_observed_xyz_m"][:, None].expand(
        -1, horizon_count, -1, -1
    )
    selected_probability = output[
        "selected_observed_doppler_probability"
    ][:, None].expand(-1, horizon_count, -1, -1)
    delta = output["horizons_seconds"][None].expand(
        batch_size, -1
    ).reshape(-1)
    return (
        selected_xyz.reshape(-1, point_count, 3),
        selected_probability.reshape(
            -1, point_count, selected_probability.shape[-1]
        ),
        persistent.reshape(-1, point_count, 3),
        output["persistent_doppler_probability"].reshape(
            -1, point_count, selected_probability.shape[-1]
        ),
        output["target_from_current"].reshape(-1, 4, 4),
        delta,
        output["analytic_radial_velocity_mps"].reshape(
            -1, point_count
        ),
        output["tangential_velocity_mps"].reshape(
            -1, point_count, 3
        ),
        output["radial_correction_mps"].reshape(-1, point_count),
    )


def dmhw_stage0_loss(
    output: dict[str, torch.Tensor],
    *,
    target_xyz_m: torch.Tensor,
    target_doppler_probability: torch.Tensor | None = None,
    target_confidence: torch.Tensor | None = None,
    aligned_synthetic_attributes: bool = False,
    geometry_weight: float = 1.0,
    inverse_radial_weight: float = 0.25,
    cycle_weight: float = 0.1,
    residual_weight: float = 0.01,
    doppler_weight: float = 0.25,
    confidence_weight: float = 0.1,
) -> DMHWLoss:
    """Compute a geometry/physics objective with explicit supervision bounds.

    K-Radar future dense targets do not provide persistent IDs or generated
    birth-point Doppler labels. Point-aligned Doppler/confidence targets are
    therefore accepted only for the synthetic contract preflight.
    """

    geometry = direct_horizon_chamfer(
        output["xyz_m"], target_xyz_m
    )
    (
        selected_xyz,
        _,
        persistent_xyz,
        persistent_probability,
        transform,
        delta,
        analytic_radial_velocity,
        tangential_velocity,
        radial_correction,
    ) = _flatten_persistent(output)
    inverse = inverse_radial_loss(
        selected_xyz,
        persistent_xyz,
        transform,
        delta,
        persistent_probability,
        output["doppler_mps"],
        output["doppler_lower_mps"],
        output["doppler_period_mps"],
    )
    cycle = motion_cycle_loss(
        selected_xyz,
        persistent_xyz,
        transform,
        delta,
        analytic_radial_velocity,
        tangential_velocity,
        radial_correction,
    )
    residual = (
        output["tangential_velocity_mps"].square().mean()
        + output["radial_correction_mps"].square().mean()
    )
    zero = geometry.new_zeros(())
    doppler = zero
    confidence = zero
    if target_doppler_probability is not None:
        if not aligned_synthetic_attributes:
            raise ValueError(
                "Aligned D-MHW Doppler targets are synthetic-preflight only"
            )
        if target_doppler_probability.shape != output[
            "doppler_probability"
        ].shape:
            raise ValueError("D-MHW Doppler target shape mismatch")
        target_probability = target_doppler_probability.clamp_min(0.0)
        target_probability = target_probability / target_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        doppler = -(
            target_probability
            * F.log_softmax(output["doppler_logit"], dim=-1)
        ).sum(dim=-1).mean()
    if target_confidence is not None:
        if not aligned_synthetic_attributes:
            raise ValueError(
                "Aligned D-MHW confidence targets are synthetic-preflight only"
            )
        if target_confidence.shape != output["confidence"].shape:
            raise ValueError("D-MHW confidence target shape mismatch")
        confidence = F.binary_cross_entropy_with_logits(
            output["confidence_logit"],
            target_confidence,
        )
    components = {
        "geometry_chamfer": geometry,
        "inverse_radial": inverse,
        "motion_cycle": cycle,
        "bounded_residual_regularizer": residual,
        "synthetic_doppler_cross_entropy": doppler,
        "synthetic_confidence_bce": confidence,
    }
    total = (
        geometry_weight * geometry
        + inverse_radial_weight * inverse
        + cycle_weight * cycle
        + residual_weight * residual
        + doppler_weight * doppler
        + confidence_weight * confidence
    )
    return DMHWLoss(
        total=total,
        components=components,
        attribute_supervision=(
            "synthetic_point_aligned"
            if aligned_synthetic_attributes
            else "geometry_plus_persistent_inverse_physics_only"
        ),
    )
