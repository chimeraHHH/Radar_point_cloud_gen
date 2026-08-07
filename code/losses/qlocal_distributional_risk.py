"""Distributional distance supervision and listwise risk loss for Q-Local."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QLocalRiskLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


@torch.no_grad()
def distance_distribution_target(
    distance_m: torch.Tensor,
    centers_m: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate each distance onto adjacent ordered centers."""

    if distance_m.ndim != 2:
        raise ValueError("Q-Local distances must have shape (B,N)")
    if centers_m.ndim != 1 or centers_m.numel() < 2:
        raise ValueError("Q-Local distance centers must be a vector")
    if not torch.is_floating_point(distance_m) or not torch.is_floating_point(
        centers_m
    ):
        raise TypeError("Q-Local distance targets must be floating point")
    if not bool(torch.isfinite(distance_m).all()) or not bool(
        torch.isfinite(centers_m).all()
    ):
        raise ValueError("Q-Local distance targets must be finite")
    if bool((distance_m < 0.0).any()):
        raise ValueError("Q-Local nearest distances cannot be negative")
    if not bool((centers_m[1:] > centers_m[:-1]).all()):
        raise ValueError("Q-Local distance centers must be strictly ordered")

    values = distance_m.clamp(centers_m[0], centers_m[-1])
    upper = torch.searchsorted(centers_m, values, right=False).clamp(
        1, centers_m.numel() - 1
    )
    lower = upper - 1
    lower_center = centers_m[lower]
    upper_center = centers_m[upper]
    upper_weight = (values - lower_center) / (
        upper_center - lower_center
    ).clamp_min(torch.finfo(values.dtype).eps)
    upper_weight = torch.where(
        values <= centers_m[0],
        torch.zeros_like(upper_weight),
        upper_weight,
    )
    upper_weight = torch.where(
        values >= centers_m[-1],
        torch.ones_like(upper_weight),
        upper_weight,
    )
    target = torch.zeros(
        *distance_m.shape,
        centers_m.numel(),
        dtype=distance_m.dtype,
        device=distance_m.device,
    )
    target.scatter_add_(-1, lower[..., None], (1.0 - upper_weight)[..., None])
    target.scatter_add_(-1, upper[..., None], upper_weight[..., None])
    if not torch.allclose(
        target.sum(dim=-1),
        torch.ones_like(distance_m),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("Q-Local distance target lost probability mass")
    return target.detach()


def _validate_loss_inputs(
    distance_logits: torch.Tensor,
    target_distribution: torch.Tensor,
    target_distance_m: torch.Tensor,
    range_class: torch.Tensor,
    centers_m: torch.Tensor,
) -> None:
    if distance_logits.ndim != 3:
        raise ValueError("Q-Local distance logits must have shape (B,N,K)")
    if target_distribution.shape != distance_logits.shape:
        raise ValueError("Q-Local target distribution must align with logits")
    if target_distance_m.shape != distance_logits.shape[:2]:
        raise ValueError("Q-Local target distances must align with logits")
    if range_class.shape != distance_logits.shape[:2]:
        raise ValueError("Q-Local range classes must align with logits")
    if centers_m.shape != (distance_logits.shape[-1],):
        raise ValueError("Q-Local center count must match logits")
    for value in (
        distance_logits,
        target_distribution,
        target_distance_m,
        centers_m,
    ):
        if not torch.is_floating_point(value):
            raise TypeError("Q-Local loss tensors must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Q-Local loss tensors must be finite")
    if range_class.dtype != torch.long:
        raise TypeError("Q-Local range class must be torch.long")
    if bool(((range_class < 0) | (range_class > 2)).any()):
        raise ValueError("Q-Local range class must lie in {0,1,2}")
    for code in range(3):
        if not bool((range_class == code).any()):
            raise ValueError(f"Q-Local batch misses range class {code}")


def qlocal_distributional_risk_loss(
    distance_logits: torch.Tensor,
    target_distribution: torch.Tensor,
    target_distance_m: torch.Tensor,
    range_class: torch.Tensor,
    centers_m: torch.Tensor,
    *,
    listwise_weight: float = 0.5,
    listwise_temperature_m: float = 0.25,
) -> QLocalRiskLoss:
    """Balance distributional calibration and ListNet risk ranking by range."""

    _validate_loss_inputs(
        distance_logits,
        target_distribution,
        target_distance_m,
        range_class,
        centers_m,
    )
    if listwise_weight < 0.0 or not math.isfinite(listwise_weight):
        raise ValueError("Q-Local listwise weight must be finite and nonnegative")
    if listwise_temperature_m <= 0.0 or not math.isfinite(
        listwise_temperature_m
    ):
        raise ValueError("Q-Local listwise temperature must be positive")

    log_probability = F.log_softmax(distance_logits.float(), dim=-1)
    expected_distance = (
        torch.softmax(distance_logits.float(), dim=-1)
        * centers_m.float().view(1, 1, -1)
    ).sum(dim=-1)
    distribution_terms: list[torch.Tensor] = []
    listwise_terms: list[torch.Tensor] = []
    for batch in range(distance_logits.shape[0]):
        for code in range(3):
            mask = range_class[batch] == code
            distribution_terms.append(
                -(
                    target_distribution[batch, mask].detach().float()
                    * log_probability[batch, mask]
                ).sum(dim=-1).mean()
            )
            target_log_mass = F.log_softmax(
                -target_distance_m[batch, mask].detach().float()
                / listwise_temperature_m,
                dim=0,
            )
            predicted_log_mass = F.log_softmax(
                -expected_distance[batch, mask] / listwise_temperature_m,
                dim=0,
            )
            target_mass = target_log_mass.exp()
            listwise_terms.append(-(target_mass * predicted_log_mass).sum())
    distribution = torch.stack(distribution_terms).mean()
    listwise = torch.stack(listwise_terms).mean()
    total = distribution + listwise_weight * listwise
    return QLocalRiskLoss(
        total=total,
        components={
            "distribution_cross_entropy": distribution,
            "listwise_risk_ranking": listwise,
            "expected_distance_mean_m": expected_distance.mean(),
        },
    )
