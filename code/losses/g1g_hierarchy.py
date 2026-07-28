"""Frozen geometry objective for the G1G condition-exclusive hierarchy.

The objective supervises the exported 10,000-point set directly. Coarse
centers receive coverage, existence, and repulsion supervision, while each
four-child patch receives a scale-normalized diversity term inside the model's
hard physical cell bound. No occupancy-query objective is present.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from losses.rald_anchor import nearest_target_assignment


@dataclass(frozen=True)
class G1GHierarchyLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    distances: dict[str, torch.Tensor]
    existence_targets: dict[str, torch.Tensor]


def nearest_other_distance(
    xyz_m: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return differentiable nearest-other distances for one point set."""

    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3 or xyz_m.shape[0] < 2:
        raise ValueError("G1G nearest-other distance expects at least two XYZ points")
    if chunk_size <= 0:
        raise ValueError("G1G nearest-other chunk size must be positive")
    point_count = xyz_m.shape[0]
    all_indices = torch.arange(point_count, device=xyz_m.device)
    nearest = []
    for start in range(0, point_count, chunk_size):
        stop = min(start + chunk_size, point_count)
        distance = torch.cdist(xyz_m[start:stop], xyz_m)
        local_indices = torch.arange(start, stop, device=xyz_m.device)
        distance = distance.masked_fill(
            local_indices[:, None] == all_indices[None],
            float("inf"),
        )
        nearest.append(distance.amin(dim=1))
    return torch.cat(nearest)


def center_repulsion_loss(
    center_xyz_m: torch.Tensor,
    *,
    minimum_distance_m: float = 0.10,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    if minimum_distance_m <= 0.0:
        raise ValueError("G1G center repulsion distance must be positive")
    nearest = nearest_other_distance(center_xyz_m, chunk_size=chunk_size)
    repulsion = torch.relu(
        nearest.new_tensor(minimum_distance_m) - nearest
    ).square().mean()
    return repulsion, nearest


def bounded_child_diversity_loss(
    child_xyz_m: torch.Tensor,
    center_cell_diagonal_m: torch.Tensor,
    *,
    minimum_diagonal_fraction: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize collapsed siblings relative to their physical Cube-cell size."""

    if child_xyz_m.ndim != 3 or child_xyz_m.shape[-1] != 3:
        raise ValueError("G1G child XYZ must have shape (C,K,3)")
    center_count, children_per_center, _ = child_xyz_m.shape
    if center_count == 0 or children_per_center < 2:
        raise ValueError("G1G child diversity requires at least two children")
    if center_cell_diagonal_m.shape not in (
        (center_count,),
        (center_count, 1),
    ):
        raise ValueError("G1G cell diagonals must align with centers")
    if not 0.0 < minimum_diagonal_fraction < 1.0:
        raise ValueError("G1G child diversity fraction must lie in (0,1)")

    pairwise = torch.cdist(child_xyz_m, child_xyz_m)
    upper = torch.triu(
        torch.ones(
            children_per_center,
            children_per_center,
            dtype=torch.bool,
            device=child_xyz_m.device,
        ),
        diagonal=1,
    )
    sibling_distance = pairwise[:, upper]
    diagonal = center_cell_diagonal_m.reshape(center_count, 1).clamp_min(1e-6)
    normalized_distance = sibling_distance / diagonal
    diversity = torch.relu(
        normalized_distance.new_tensor(minimum_diagonal_fraction)
        - normalized_distance
    ).square().mean()
    return diversity, normalized_distance


def physical_child_bound_loss(
    child_physical_offset_m: torch.Tensor,
    center_cell_diagonal_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if child_physical_offset_m.ndim != 3 or child_physical_offset_m.shape[-1] != 3:
        raise ValueError("G1G child physical offsets must have shape (C,K,3)")
    center_count = child_physical_offset_m.shape[0]
    if center_cell_diagonal_m.shape not in (
        (center_count,),
        (center_count, 1),
    ):
        raise ValueError("G1G physical bounds must align with centers")
    radius = center_cell_diagonal_m.reshape(center_count, 1)
    norm = torch.linalg.vector_norm(child_physical_offset_m, dim=-1)
    excess = torch.relu(norm - radius)
    return excess.square().mean(), excess


def g1g_hierarchy_loss(
    output: dict[str, torch.Tensor],
    target_xyz_confidence: torch.Tensor,
    *,
    expected_point_count: int = 10_000,
    expected_center_count: int = 2_500,
    expected_children_per_center: int = 4,
    geometry_weight: float = 1.0,
    outlier_weight: float = 0.25,
    child_existence_weight: float = 0.10,
    center_coverage_weight: float = 0.25,
    center_existence_weight: float = 0.05,
    center_repulsion_weight: float = 0.05,
    final_repulsion_weight: float = 0.02,
    child_diversity_weight: float = 0.05,
    child_bound_weight: float = 1.0,
    outlier_threshold_m: float = 2.0,
    existence_radius_m: float = 1.0,
    center_repulsion_distance_m: float = 0.10,
    final_repulsion_distance_m: float = 0.10,
    child_diversity_diagonal_fraction: float = 0.20,
    nearest_other_chunk_size: int = 512,
) -> G1GHierarchyLoss:
    """Compute the preregistered G1G Stage-0 geometry objective."""

    if target_xyz_confidence.ndim != 2 or target_xyz_confidence.shape[1] != 4:
        raise ValueError("G1G dense target must have shape (M,4)")
    if target_xyz_confidence.shape[0] == 0:
        raise ValueError("G1G dense target cannot be empty")
    if expected_point_count != expected_center_count * expected_children_per_center:
        raise ValueError("G1G formal cardinalities are inconsistent")

    prediction = output["xyz_m"]
    centers = output["center_coordinates_rae"]
    child_xyz = output["child_xyz_m"]
    child_offset = output["child_physical_offset_m"]
    diagonal = output["center_cell_diagonal_m"]
    confidence = output["confidence"]
    confidence_logit = output["confidence_logit"]
    center_score_logit = output["center_score_logit"]
    if prediction.shape != (1, expected_point_count, 3):
        raise ValueError("G1G output must contain the frozen point count")
    if centers.shape != (1, expected_center_count, 3):
        raise ValueError("G1G output must contain the frozen center count")
    if child_xyz.shape != (
        1,
        expected_center_count,
        expected_children_per_center,
        3,
    ):
        raise ValueError("G1G children do not match the frozen hierarchy")
    if child_offset.shape != child_xyz.shape:
        raise ValueError("G1G child offsets must align with child XYZ")
    if diagonal.shape not in (
        (1, expected_center_count),
        (1, expected_center_count, 1),
    ):
        raise ValueError("G1G center cell diagonals have the wrong shape")
    if confidence.shape != (1, expected_point_count):
        raise ValueError("G1G child confidence must align with exported points")
    if confidence_logit.shape != (1, expected_point_count):
        raise ValueError(
            "G1G child confidence logits must align with exported points"
        )
    if center_score_logit.shape != (1, expected_center_count):
        raise ValueError("G1G center scores must align with allocated centers")

    prediction_xyz = prediction[0].float()
    center_xyz = output.get("center_xyz_m")
    if center_xyz is None:
        raise ValueError("G1G loss requires explicit center_xyz_m")
    if center_xyz.shape != (1, expected_center_count, 3):
        raise ValueError("G1G center XYZ must align with allocated centers")
    center_xyz = center_xyz[0].float()
    target_xyz = target_xyz_confidence[:, :3].float()
    target_weight = target_xyz_confidence[:, 3].float().clamp_min(0.0)
    target_weight_sum = target_weight.sum().clamp_min(1e-8)

    prediction_to_target, _ = nearest_target_assignment(
        prediction_xyz,
        target_xyz,
    )
    target_to_prediction, _ = nearest_target_assignment(
        target_xyz,
        prediction_xyz,
    )
    completeness = (target_to_prediction * target_weight).sum()
    completeness = completeness / target_weight_sum
    chamfer = prediction_to_target.mean() + completeness
    outlier_hinge = torch.relu(
        prediction_to_target
        - prediction_to_target.new_tensor(outlier_threshold_m)
    ).square().mean()
    child_existence_target = (
        prediction_to_target.detach() <= existence_radius_m
    ).to(confidence_logit)
    child_existence = F.binary_cross_entropy_with_logits(
        confidence_logit[0].float(),
        child_existence_target.float(),
    )

    center_to_target, _ = nearest_target_assignment(center_xyz, target_xyz)
    target_to_center, _ = nearest_target_assignment(target_xyz, center_xyz)
    center_coverage = (target_to_center * target_weight).sum()
    center_coverage = center_coverage / target_weight_sum
    center_existence_target = (
        center_to_target.detach() <= existence_radius_m
    ).to(center_score_logit)
    center_existence = F.binary_cross_entropy_with_logits(
        center_score_logit[0].float(),
        center_existence_target.float(),
    )
    center_repulsion, center_nearest_other = center_repulsion_loss(
        center_xyz,
        minimum_distance_m=center_repulsion_distance_m,
        chunk_size=nearest_other_chunk_size,
    )
    final_repulsion, prediction_nearest_other = center_repulsion_loss(
        prediction_xyz,
        minimum_distance_m=final_repulsion_distance_m,
        chunk_size=nearest_other_chunk_size,
    )
    child_diversity, child_pair_fraction = bounded_child_diversity_loss(
        child_xyz[0].float(),
        diagonal[0].float(),
        minimum_diagonal_fraction=child_diversity_diagonal_fraction,
    )
    child_bound, child_bound_excess = physical_child_bound_loss(
        child_offset[0].float(),
        diagonal[0].float(),
    )

    total = (
        geometry_weight * chamfer
        + outlier_weight * outlier_hinge
        + child_existence_weight * child_existence
        + center_coverage_weight * center_coverage
        + center_existence_weight * center_existence
        + center_repulsion_weight * center_repulsion
        + final_repulsion_weight * final_repulsion
        + child_diversity_weight * child_diversity
        + child_bound_weight * child_bound
    )
    components = {
        "geometry_chamfer": chamfer.detach(),
        "precision_mean_distance_m": prediction_to_target.mean().detach(),
        "completeness_mean_distance_m": completeness.detach(),
        "outlier_hinge_2m": outlier_hinge.detach(),
        "child_existence_confidence": child_existence.detach(),
        "center_coverage_mean_distance_m": center_coverage.detach(),
        "center_existence_confidence": center_existence.detach(),
        "center_repulsion": center_repulsion.detach(),
        "final_point_repulsion": final_repulsion.detach(),
        "bounded_child_diversity": child_diversity.detach(),
        "child_physical_bound_violation": child_bound.detach(),
        "child_confidence_mean": confidence[0].float().mean().detach(),
        "center_confidence_mean": torch.sigmoid(
            center_score_logit[0].float()
        ).mean().detach(),
        "total": total.detach(),
    }
    return G1GHierarchyLoss(
        total=total,
        components=components,
        distances={
            "prediction_to_target_m": prediction_to_target,
            "target_to_prediction_m": target_to_prediction,
            "center_to_target_m": center_to_target,
            "target_to_center_m": target_to_center,
            "center_nearest_other_m": center_nearest_other,
            "prediction_nearest_other_m": prediction_nearest_other,
            "child_pair_diagonal_fraction": child_pair_fraction,
            "child_physical_bound_excess_m": child_bound_excess,
        },
        existence_targets={
            "child": child_existence_target,
            "center": center_existence_target,
        },
    )
