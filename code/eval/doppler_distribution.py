"""Metrics for circular Doppler distributions attached to generated points."""

from __future__ import annotations

import numpy as np
import torch

from eval.dense_geometry import nearest_distance
from losses.doppler_distribution import (
    circular_scalar_target,
    circular_wasserstein1,
    distribution_kl,
    soft_static_target,
)
from models.cube_doppler import wrapped_delta


def weighted_mean(values: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return values.mean()
    if weight.shape != values.shape:
        raise ValueError(f"Weight shape mismatch: {weight.shape}, {values.shape}")
    weight = weight.to(values).clamp_min(0.0)
    return (values * weight).sum() / weight.sum().clamp_min(1e-8)


def soft_ece(
    prediction: torch.Tensor,
    target: torch.Tensor,
    bins: int = 10,
) -> torch.Tensor:
    confidence, predicted_mode = prediction.max(dim=1)
    target_mass = target.gather(1, predicted_mode[:, None]).squeeze(1)
    boundaries = torch.linspace(
        0.0, 1.0, bins + 1, dtype=confidence.dtype, device=confidence.device
    )
    error = confidence.new_zeros(())
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        else:
            mask = (confidence >= boundaries[index]) & (
                confidence < boundaries[index + 1]
            )
        if mask.any():
            error = error + mask.float().mean() * (
                confidence[mask].mean() - target_mass[mask].mean()
            ).abs()
    return error


def doppler_distribution_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    axis: torch.Tensor,
    lower: torch.Tensor,
    period: torch.Tensor,
    bin_step: torch.Tensor,
    confidence: torch.Tensor | None = None,
    static_center_mps: torch.Tensor | None = None,
    predicted_static_probability: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"Distribution shape mismatch: {prediction.shape}, {target.shape}")
    prediction = prediction / prediction.sum(dim=1, keepdim=True).clamp_min(epsilon)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(epsilon)
    nll = -(target * prediction.clamp_min(epsilon).log()).sum(dim=1)
    kl = distribution_kl(prediction, target, epsilon=epsilon)
    wasserstein = circular_wasserstein1(prediction, target, bin_step)
    prediction_scalar = circular_scalar_target(prediction, axis, lower, period)
    target_scalar = circular_scalar_target(target, axis, lower, period)
    scalar_error = wrapped_delta(prediction_scalar, target_scalar, period).abs()
    predicted_mode = prediction.argmax(dim=1)
    target_mode = target.argmax(dim=1)
    mode_delta = torch.remainder(
        predicted_mode - target_mode + prediction.shape[1] // 2,
        prediction.shape[1],
    ) - prediction.shape[1] // 2
    report = {
        "spectrum_nll": float(weighted_mean(nll, confidence).item()),
        "spectrum_kl": float(weighted_mean(kl, confidence).item()),
        "circular_w1_mps": float(weighted_mean(wasserstein, confidence).item()),
        "circular_scalar_mae_mps": float(
            weighted_mean(scalar_error, confidence).item()
        ),
        "mode_accuracy_exact": float((mode_delta == 0).float().mean().item()),
        "mode_accuracy_within_one_bin": float(
            (mode_delta.abs() <= 1).float().mean().item()
        ),
        "soft_ece_10bin": float(soft_ece(prediction, target, bins=10).item()),
        "prediction_entropy": float(
            (-(prediction * prediction.clamp_min(epsilon).log()).sum(dim=1))
            .mean()
            .item()
        ),
        "target_entropy": float(
            (-(target * target.clamp_min(epsilon).log()).sum(dim=1)).mean().item()
        ),
        "point_count": int(prediction.shape[0]),
    }
    if static_center_mps is not None:
        if static_center_mps.shape != prediction_scalar.shape:
            raise ValueError("Static-center shape does not match queried points")
        target_static = soft_static_target(
            target_scalar, static_center_mps, period
        )
        static_mask = target_static >= 0.5
        dynamic_mask = target_static < 0.5
        pce = wrapped_delta(
            prediction_scalar, static_center_mps, period
        ).abs()
        report["target_dynamic_fraction"] = float(dynamic_mask.float().mean().item())
        report["target_static_fraction"] = float(static_mask.float().mean().item())
        if static_mask.any():
            static_pce = pce[static_mask]
            report["static_pce_median_mps"] = float(static_pce.median().item())
            for threshold in (0.25, 0.5, 1.0):
                suffix = str(threshold).replace(".", "p")
                report[f"static_pce_fraction_{suffix}mps"] = float(
                    (static_pce <= threshold).float().mean().item()
                )
        if dynamic_mask.any():
            report["dynamic_scalar_mae_mps"] = float(
                scalar_error[dynamic_mask].mean().item()
            )
        if predicted_static_probability is None:
            predicted_dynamic = pce >= 1.5
        else:
            if predicted_static_probability.shape != prediction_scalar.shape:
                raise ValueError("Predicted static probability has the wrong shape")
            predicted_dynamic = predicted_static_probability < 0.5
        report["predicted_dynamic_fraction"] = float(
            predicted_dynamic.float().mean().item()
        )
    return report


def cd_doppler_report(
    prediction_xyz: torch.Tensor,
    prediction_vr: torch.Tensor,
    target_xyz: torch.Tensor,
    target_vr: torch.Tensor,
    target_weight: torch.Tensor | None = None,
    velocity_scale: float = 1.0,
    chunk_size: int = 1024,
) -> dict[str, float]:
    prediction = torch.cat(
        (prediction_xyz, prediction_vr[:, None] * velocity_scale), dim=1
    )
    target = torch.cat((target_xyz, target_vr[:, None] * velocity_scale), dim=1)
    prediction_to_target = nearest_distance(prediction, target, chunk_size=chunk_size)
    target_to_prediction = nearest_distance(target, prediction, chunk_size=chunk_size)
    if target_weight is None:
        target_weight = torch.ones_like(target_to_prediction)
    target_weight = target_weight.to(target_to_prediction).clamp_min(0.0)
    completeness = (target_to_prediction * target_weight).sum()
    completeness = completeness / target_weight.sum().clamp_min(1e-8)
    return {
        "cd_doppler": float((prediction_to_target.mean() + completeness).item()),
        "precision_mean_distance": float(prediction_to_target.mean().item()),
        "completeness_mean_distance": float(completeness.item()),
        "prediction_count": int(prediction.shape[0]),
        "target_count": int(target.shape[0]),
    }


def aggregate_doppler_reports(
    reports: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not reports:
        raise ValueError("Cannot aggregate an empty Doppler report list")
    keys = sorted(
        {
            key
            for report in reports
            for key, value in report.items()
            if isinstance(value, (int, float)) and not key.endswith("_count")
        }
    )
    aggregate = {}
    for key in keys:
        values = np.asarray(
            [report[key] for report in reports if key in report], dtype=np.float64
        )
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "sample_count": int(values.size),
        }
    return aggregate
