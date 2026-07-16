"""Losses for sparse radar-observable frustum occupancy."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Normalize positive and negative terms separately before combining them."""

    if logits.shape != target.shape:
        raise ValueError(f"Shape mismatch: {logits.shape} vs {target.shape}")
    target = target.clamp(0.0, 1.0)
    probability = torch.sigmoid(logits)
    positive = -target * (1.0 - probability).pow(gamma) * F.logsigmoid(logits)
    negative_weight = 1.0 - target
    negative = (
        -negative_weight
        * probability.pow(gamma)
        * F.logsigmoid(-logits)
    )
    positive = positive.sum() / target.sum().clamp_min(1.0)
    negative = negative.sum() / negative_weight.sum().clamp_min(1.0)
    return positive_weight * positive + (1.0 - positive_weight) * negative


def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    dimensions = tuple(range(1, logits.ndim))
    overlap = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    return (1.0 - (2.0 * overlap + epsilon) / (denominator + epsilon)).mean()


def occupancy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    focal = balanced_focal_loss(logits, target)
    dice = soft_dice_loss(logits, target)
    total = focal + dice_weight * dice
    return total, {"total": total.detach(), "focal": focal.detach(), "dice": dice.detach()}
