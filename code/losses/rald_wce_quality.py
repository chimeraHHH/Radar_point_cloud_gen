"""Continuous geometry-quality targets and ranking loss for Q1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RaLDWCEQualityLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


@torch.no_grad()
def continuous_geometry_quality_target(
    candidate_xyz_m: torch.Tensor,
    target_xyz_confidence: torch.Tensor,
    *,
    distance_temperature_m: float = 1.0,
    candidate_chunk_size: int = 4_096,
) -> dict[str, torch.Tensor]:
    """Build stop-gradient quality from nearest-target distance and confidence.

    The target is ``normalized_confidence * exp(-distance / temperature)``.
    Candidate coordinates and target geometry are detached before the nearest
    neighbour calculation, so this target cannot move the frozen residual.
    """

    if candidate_xyz_m.ndim != 2 or candidate_xyz_m.shape[1] != 3:
        raise ValueError("Q1 candidate XYZ must have shape (N,3)")
    if candidate_xyz_m.shape[0] <= 0:
        raise ValueError("Q1 candidate set cannot be empty")
    if (
        target_xyz_confidence.ndim != 2
        or target_xyz_confidence.shape[1] != 4
        or target_xyz_confidence.shape[0] <= 0
    ):
        raise ValueError("Q1 target must have nonempty shape (M,4)")
    if not math.isfinite(distance_temperature_m) or distance_temperature_m <= 0:
        raise ValueError("Q1 distance temperature must be positive")
    if candidate_chunk_size <= 0:
        raise ValueError("Q1 candidate chunk size must be positive")
    for value, name in (
        (candidate_xyz_m, "candidate XYZ"),
        (target_xyz_confidence, "target"),
    ):
        if not torch.is_floating_point(value):
            raise TypeError(f"Q1 {name} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Q1 {name} must be finite")
    if candidate_xyz_m.device != target_xyz_confidence.device:
        raise ValueError("Q1 candidates and target must share a device")

    candidate = candidate_xyz_m.detach().float()
    target = target_xyz_confidence.detach().float()
    target_xyz = target[:, :3]
    target_confidence = target[:, 3].clamp_min(0.0)
    maximum_confidence = target_confidence.max()
    if float(maximum_confidence.item()) <= 0.0:
        raise ValueError("Q1 target confidence must have positive mass")
    normalized_confidence = target_confidence / maximum_confidence

    distance_parts: list[torch.Tensor] = []
    confidence_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    for start in range(0, candidate.shape[0], candidate_chunk_size):
        stop = min(start + candidate_chunk_size, candidate.shape[0])
        pairwise = torch.cdist(candidate[start:stop], target_xyz)
        distance, index = pairwise.min(dim=1)
        distance_parts.append(distance)
        confidence_parts.append(normalized_confidence[index])
        index_parts.append(index)
    nearest_distance = torch.cat(distance_parts, dim=0)
    nearest_confidence = torch.cat(confidence_parts, dim=0)
    nearest_target_index = torch.cat(index_parts, dim=0)
    quality = nearest_confidence * torch.exp(
        -nearest_distance / distance_temperature_m
    )
    if not bool(torch.isfinite(quality).all()):
        raise FloatingPointError("Q1 geometry-quality target became non-finite")
    return {
        "quality_target": quality.clamp(0.0, 1.0).detach(),
        "nearest_distance_m": nearest_distance.detach(),
        "nearest_target_confidence": nearest_confidence.detach(),
        "nearest_target_index": nearest_target_index.detach(),
    }


def _validate_quality_loss_inputs(
    quality_logit: torch.Tensor,
    quality_target: torch.Tensor,
    range_class: torch.Tensor,
) -> None:
    if quality_logit.ndim != 2:
        raise ValueError("Q1 quality logits must have shape (B,N)")
    if quality_target.shape != quality_logit.shape:
        raise ValueError("Q1 quality target must align with logits")
    if range_class.shape != quality_logit.shape:
        raise ValueError("Q1 range class must align with logits")
    for value, name in (
        (quality_logit, "quality logits"),
        (quality_target, "quality target"),
    ):
        if not torch.is_floating_point(value):
            raise TypeError(f"Q1 {name} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Q1 {name} must be finite")
    if range_class.dtype != torch.long:
        raise TypeError("Q1 range class must be torch.long")
    if bool(((quality_target < 0.0) | (quality_target > 1.0)).any()):
        raise ValueError("Q1 quality target must lie in [0,1]")
    if bool(((range_class < 0) | (range_class > 2)).any()):
        raise ValueError("Q1 range class must lie in {0,1,2}")
    for code in range(3):
        if not bool((range_class == code).any()):
            raise ValueError(f"Q1 sampled batch misses range class {code}")


def _range_balanced_soft_bce(
    quality_logit: torch.Tensor,
    quality_target: torch.Tensor,
    range_class: torch.Tensor,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for code in range(3):
        mask = range_class == code
        values.append(
            F.binary_cross_entropy_with_logits(
                quality_logit[mask].float(),
                quality_target[mask].detach().float(),
            )
        )
    return torch.stack(values).mean()


def _range_balanced_pairwise_ranking(
    quality_logit: torch.Tensor,
    quality_target: torch.Tensor,
    range_class: torch.Tensor,
    *,
    minimum_target_gap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses: list[torch.Tensor] = []
    pair_counts: list[int] = []
    for batch in range(quality_logit.shape[0]):
        for code in range(3):
            rows = torch.nonzero(
                range_class[batch] == code,
                as_tuple=False,
            ).flatten()
            target = quality_target[batch, rows].detach()
            order = torch.argsort(target, stable=True)
            pair_count = rows.numel() // 2
            if pair_count <= 0:
                continue
            low_rows = rows[order[:pair_count]]
            high_rows = rows[order[-pair_count:]]
            gap = (
                quality_target[batch, high_rows].detach()
                - quality_target[batch, low_rows].detach()
            )
            eligible = gap >= minimum_target_gap
            if not bool(eligible.any()):
                continue
            logit_gap = (
                quality_logit[batch, high_rows[eligible]]
                - quality_logit[batch, low_rows[eligible]]
            )
            pair_loss = F.softplus(-logit_gap.float())
            weight = gap[eligible].float().clamp_min(minimum_target_gap)
            losses.append((pair_loss * weight).sum() / weight.sum())
            pair_counts.append(int(eligible.sum().item()))
    if not losses:
        return quality_logit.sum() * 0.0, quality_logit.new_tensor(0.0)
    return (
        torch.stack(losses).mean(),
        quality_logit.new_tensor(float(sum(pair_counts))),
    )


def rald_wce_quality_loss(
    quality_logit: torch.Tensor,
    quality_target: torch.Tensor,
    range_class: torch.Tensor,
    *,
    ranking_weight: float = 0.5,
    minimum_target_gap: float = 0.05,
) -> RaLDWCEQualityLoss:
    """Align score with continuous geometry quality within frozen range quotas."""

    _validate_quality_loss_inputs(
        quality_logit,
        quality_target,
        range_class,
    )
    if ranking_weight < 0.0 or not math.isfinite(ranking_weight):
        raise ValueError("Q1 ranking weight must be finite and nonnegative")
    if not 0.0 < minimum_target_gap < 1.0:
        raise ValueError("Q1 minimum target gap must lie in (0,1)")
    calibration = _range_balanced_soft_bce(
        quality_logit,
        quality_target,
        range_class,
    )
    ranking, pair_count = _range_balanced_pairwise_ranking(
        quality_logit,
        quality_target,
        range_class,
        minimum_target_gap=minimum_target_gap,
    )
    total = calibration + ranking_weight * ranking
    return RaLDWCEQualityLoss(
        total=total,
        components={
            "quality_calibration": calibration,
            "quality_pairwise_ranking": ranking,
            "quality_pair_count": pair_count,
        },
    )
