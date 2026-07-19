"""Supervision for RaLD refinement of frozen occupancy anchors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from losses.cube_cycle import cube_cycle_loss, existence_confidence_loss
from losses.doppler_distribution import (
    circular_scalar_target,
    circular_smooth_l1,
    distribution_cross_entropy,
)
from models.cube_doppler import query_cube_spectrum
from models.point_to_cube import soft_splat_raed


@dataclass(frozen=True)
class AnchorRefinementLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    prediction_to_target_m: torch.Tensor
    matched_target_spectrum: torch.Tensor
    direct_matched_target_spectrum: torch.Tensor
    existence_target: torch.Tensor


def nearest_target_assignment(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest-target distance and index without materializing the full matrix."""

    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("Source points must have shape (N,3)")
    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError("Target points must have shape (M,3)")
    if source_xyz.shape[0] == 0 or target_xyz.shape[0] == 0:
        raise ValueError("Nearest-target assignment requires non-empty point sets")
    if chunk_size <= 0:
        raise ValueError("Nearest-target chunk size must be positive")
    distances = []
    indices = []
    for start in range(0, source_xyz.shape[0], chunk_size):
        pairwise = torch.cdist(source_xyz[start : start + chunk_size], target_xyz)
        distance, index = pairwise.min(dim=1)
        distances.append(distance)
        indices.append(index)
    return torch.cat(distances), torch.cat(indices)


def anchor_refinement_loss(
    output: dict[str, torch.Tensor],
    cube_drae: torch.Tensor,
    target_xyz_confidence: torch.Tensor,
    target_rae_index: torch.Tensor,
    *,
    geometry_weight: float = 1.0,
    doppler_weight: float = 1.0,
    existence_weight: float = 0.1,
    cycle_weight: float = 0.1,
    offset_weight: float = 0.01,
    cycle_variant: str = "full",
) -> AnchorRefinementLoss:
    """Joint geometry, physical-attribute, confidence, and Cube-cycle objective."""

    if output["xyz_m"].shape[0] != 1 or cube_drae.shape[0] != 1:
        raise ValueError("Anchor refinement currently trains one Cube per step")
    if target_xyz_confidence.ndim != 2 or target_xyz_confidence.shape[1] != 4:
        raise ValueError("Dense target must have shape (M,4)")
    if target_rae_index.shape != (target_xyz_confidence.shape[0], 3):
        raise ValueError("Target RAE indices must align with dense target points")

    prediction_xyz = output["xyz_m"][0].float()
    target_xyz = target_xyz_confidence[:, :3].float()
    target_weight = target_xyz_confidence[:, 3].float().clamp_min(0.0)
    prediction_to_target, matched_index = nearest_target_assignment(
        prediction_xyz, target_xyz
    )
    target_to_prediction, _ = nearest_target_assignment(target_xyz, prediction_xyz)
    completeness = (target_to_prediction * target_weight).sum()
    completeness = completeness / target_weight.sum().clamp_min(1e-8)
    geometry = prediction_to_target.mean() + completeness

    all_target_spectrum = query_cube_spectrum(cube_drae, target_rae_index)
    matched_target_spectrum = all_target_spectrum[matched_index]
    fixed_doppler_weight = output["anchor_parent_confidence"][0].float().detach()
    if "doppler_scalar_bin" in output:
        bin_axis = torch.arange(
            matched_target_spectrum.shape[-1],
            device=matched_target_spectrum.device,
            dtype=matched_target_spectrum.dtype,
        )
        target_scalar_bin = circular_scalar_target(
            matched_target_spectrum,
            bin_axis,
            bin_axis.new_zeros(()),
            bin_axis.new_tensor(float(bin_axis.numel())),
        )
        doppler = circular_smooth_l1(
            output["doppler_scalar_bin"][0].float(),
            target_scalar_bin,
            bin_axis.new_tensor(float(bin_axis.numel())),
            beta=1.0,
            weight=fixed_doppler_weight,
        )
        doppler_name = "doppler_scalar_smooth_l1_bins"
    else:
        doppler = distribution_cross_entropy(
            output["doppler_probability"][0].float(),
            matched_target_spectrum,
            weight=fixed_doppler_weight,
        )
        doppler_name = "doppler_cross_entropy"

    direct_matched_target_spectrum = matched_target_spectrum
    direct_doppler = distribution_cross_entropy(
        output["point_cube_spectrum"][0].float(),
        direct_matched_target_spectrum,
        weight=fixed_doppler_weight,
    )

    confidence = output["confidence"][0].float()
    existence, existence_target = existence_confidence_loss(
        confidence, prediction_to_target
    )
    if cycle_variant == "none":
        cycle = confidence.new_zeros(())
        cycle_components = {
            "confidence_mean": confidence.mean().detach(),
            "total": cycle.detach(),
        }
    else:
        rendered = soft_splat_raed(
            output["coordinates_rae"][0].float(),
            output["doppler_probability"][0].float(),
            confidence,
        )
        cycle, cycle_components = cube_cycle_loss(
            rendered,
            cube_drae[0].float(),
            confidence,
            cycle_variant,
        )
    offset = output["offset_bins"][0].float().square().mean()
    total = (
        geometry_weight * geometry
        + doppler_weight * doppler
        + existence_weight * existence
        + cycle_weight * cycle
        + offset_weight * offset
    )
    components = {
        "geometry_chamfer": geometry.detach(),
        doppler_name: doppler.detach(),
        "direct_cube_cross_entropy": direct_doppler.detach(),
        "existence_confidence": existence.detach(),
        "cycle": cycle.detach(),
        "offset_square": offset.detach(),
        "confidence_mean": confidence.mean().detach(),
        "total": total.detach(),
    }
    components.update(
        {f"cycle_{name}": value.detach() for name, value in cycle_components.items()}
    )
    return AnchorRefinementLoss(
        total=total,
        components=components,
        prediction_to_target_m=prediction_to_target,
        matched_target_spectrum=matched_target_spectrum,
        direct_matched_target_spectrum=direct_matched_target_spectrum,
        existence_target=existence_target,
    )
