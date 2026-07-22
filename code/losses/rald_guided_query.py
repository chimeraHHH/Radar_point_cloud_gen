"""Frozen G1C geometry objective for RaLD-guided dense queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from losses.cube_cycle import existence_confidence_loss
from losses.rald_anchor import nearest_target_assignment


@dataclass(frozen=True)
class RaLDGuidedGeometryLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    prediction_to_target_m: torch.Tensor
    target_to_prediction_m: torch.Tensor


def within_seed_repulsion(
    xyz_m: torch.Tensor,
    *,
    queries_per_seed: int,
    minimum_distance_m: float = 0.10,
) -> torch.Tensor:
    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
        raise ValueError("G1C repulsion expects points with shape (N,3)")
    if queries_per_seed <= 1 or xyz_m.shape[0] % queries_per_seed != 0:
        raise ValueError("G1C points do not match the frozen seed expansion")
    groups = xyz_m.reshape(-1, queries_per_seed, 3)
    distance = torch.cdist(groups, groups)
    mask = torch.triu(
        torch.ones(
            queries_per_seed,
            queries_per_seed,
            device=xyz_m.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    pairwise = distance[:, mask]
    return torch.relu(pairwise.new_tensor(minimum_distance_m) - pairwise).square().mean()


def rald_guided_geometry_loss(
    output: dict[str, torch.Tensor],
    target_xyz_confidence: torch.Tensor,
    *,
    queries_per_seed: int = 10,
    geometry_weight: float = 1.0,
    outlier_weight: float = 0.25,
    existence_weight: float = 0.10,
    offset_weight: float = 0.02,
    repulsion_weight: float = 0.02,
    outlier_threshold_m: float = 2.0,
    repulsion_distance_m: float = 0.10,
) -> RaLDGuidedGeometryLoss:
    if output["xyz_m"].shape[0] != 1:
        raise ValueError("G1C geometry training uses one Cube per optimizer step")
    if target_xyz_confidence.ndim != 2 or target_xyz_confidence.shape[1] != 4:
        raise ValueError("G1C dense target must have shape (M,4)")
    prediction = output["xyz_m"][0].float()
    target = target_xyz_confidence[:, :3].float()
    target_weight = target_xyz_confidence[:, 3].float().clamp_min(0.0)
    prediction_distance, _ = nearest_target_assignment(prediction, target)
    target_distance, _ = nearest_target_assignment(target, prediction)
    completeness = (target_distance * target_weight).sum()
    completeness = completeness / target_weight.sum().clamp_min(1e-8)
    chamfer = prediction_distance.mean() + completeness
    outlier_hinge = torch.relu(
        prediction_distance - prediction_distance.new_tensor(outlier_threshold_m)
    ).square().mean()
    existence, _ = existence_confidence_loss(
        output["confidence"][0].float(), prediction_distance
    )
    normalized_offset = output["raw_offset_bins"][0].float().square().mean()
    repulsion = within_seed_repulsion(
        prediction,
        queries_per_seed=queries_per_seed,
        minimum_distance_m=repulsion_distance_m,
    )
    total = (
        geometry_weight * chamfer
        + outlier_weight * outlier_hinge
        + existence_weight * existence
        + offset_weight * normalized_offset
        + repulsion_weight * repulsion
    )
    components = {
        "geometry_chamfer": chamfer.detach(),
        "outlier_hinge_2m": outlier_hinge.detach(),
        "existence_confidence": existence.detach(),
        "normalized_offset_square": normalized_offset.detach(),
        "within_seed_repulsion": repulsion.detach(),
        "confidence_mean": output["confidence"][0].float().mean().detach(),
        "total": total.detach(),
    }
    return RaLDGuidedGeometryLoss(
        total=total,
        components=components,
        prediction_to_target_m=prediction_distance,
        target_to_prediction_m=target_distance,
    )
