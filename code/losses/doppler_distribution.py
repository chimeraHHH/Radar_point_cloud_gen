"""Circular Doppler distribution losses and physics-mixture supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.cube_doppler import circular_mean, wrapped_delta


def distribution_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"Distribution shape mismatch: {prediction.shape}, {target.shape}")
    per_point = -(target * prediction.clamp_min(epsilon).log()).sum(dim=1)
    if weight is None:
        return per_point.mean()
    if weight.shape != per_point.shape:
        raise ValueError(f"Weight shape mismatch: {weight.shape}, {per_point.shape}")
    weight = weight.to(per_point).clamp_min(0.0)
    return (per_point * weight).sum() / weight.sum().clamp_min(epsilon)


def distribution_kl(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"Distribution shape mismatch: {prediction.shape}, {target.shape}")
    target_safe = target.clamp_min(epsilon)
    prediction_safe = prediction.clamp_min(epsilon)
    return (target * (target_safe.log() - prediction_safe.log())).sum(dim=1)


def circular_wasserstein1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    bin_step: torch.Tensor,
) -> torch.Tensor:
    """Exact circular 1-Wasserstein distance on an equally spaced axis."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"Distribution shape mismatch: {prediction.shape}, {target.shape}")
    cumulative = torch.cumsum(prediction - target, dim=1)
    offset = cumulative.median(dim=1, keepdim=True).values
    return (cumulative - offset).abs().sum(dim=1) * bin_step


def circular_scalar_target(
    target: torch.Tensor,
    axis: torch.Tensor,
    lower: torch.Tensor,
    period: torch.Tensor,
) -> torch.Tensor:
    return circular_mean(target, axis, lower, period)


def circular_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    period: torch.Tensor,
    beta: float = 0.25,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    error = wrapped_delta(prediction, target, period)
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), beta=beta, reduction="none")
    if weight is None:
        return loss.mean()
    if weight.shape != loss.shape:
        raise ValueError(f"Weight shape mismatch: {weight.shape}, {loss.shape}")
    weight = weight.to(loss).clamp_min(0.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def soft_static_target(
    observed_scalar_mps: torch.Tensor,
    static_center_mps: torch.Tensor,
    period_mps: torch.Tensor,
    static_threshold_mps: float = 1.0,
    dynamic_threshold_mps: float = 2.0,
) -> torch.Tensor:
    if dynamic_threshold_mps <= static_threshold_mps:
        raise ValueError("Dynamic threshold must exceed static threshold")
    residual = wrapped_delta(
        observed_scalar_mps, static_center_mps, period_mps
    ).abs()
    return (
        (dynamic_threshold_mps - residual)
        / (dynamic_threshold_mps - static_threshold_mps)
    ).clamp(0.0, 1.0)


def doppler_head_loss(
    prediction: dict[str, torch.Tensor],
    target_distribution: torch.Tensor,
    axis: torch.Tensor,
    lower: torch.Tensor,
    period: torch.Tensor,
    confidence: torch.Tensor | None = None,
    static_gate_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_scalar = circular_scalar_target(
        target_distribution, axis, lower, period
    )
    if "logits" not in prediction:
        scalar = circular_smooth_l1(
            prediction["scalar_mps"],
            target_scalar,
            period,
            weight=confidence,
        )
        return scalar, {"total": scalar.detach(), "scalar": scalar.detach()}

    spectrum = distribution_cross_entropy(
        prediction["probability"], target_distribution, weight=confidence
    )
    total = spectrum
    components = {"spectrum": spectrum.detach()}
    if "static_probability" in prediction:
        static_target = soft_static_target(
            target_scalar,
            prediction["static_center_mps"].detach(),
            period,
        )
        gate_per_point = F.binary_cross_entropy(
            prediction["static_probability"], static_target, reduction="none"
        )
        if confidence is None:
            gate = gate_per_point.mean()
        else:
            confidence = confidence.to(gate_per_point).clamp_min(0.0)
            gate = (gate_per_point * confidence).sum() / confidence.sum().clamp_min(
                1e-8
            )
        total = total + static_gate_weight * gate
        components["static_gate"] = gate.detach()
        components["target_static_fraction"] = static_target.mean().detach()
        components["predicted_static_fraction"] = prediction[
            "static_probability"
        ].mean().detach()
    components["total"] = total.detach()
    return total, components
