"""Evaluation metrics for differentiable point-to-Cube reconstruction."""

from __future__ import annotations

import numpy as np
import torch

from losses.cube_cycle import (
    covered_spectrum_kl,
    doppler_marginal_kl,
    normalized_cube_spectrum,
    spatial_energy_loss,
    target_peak_support,
)
from models.point_to_cube import SoftSplatResult


def tensor_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().flatten()
    second = second.float().flatten()
    if first.numel() < 2 or first.std() <= 1e-12 or second.std() <= 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack((first, second)))[0, 1].item())


def binary_ece(
    confidence: torch.Tensor,
    target: torch.Tensor,
    bins: int = 10,
) -> torch.Tensor:
    if confidence.ndim != 1 or target.shape != confidence.shape:
        raise ValueError("Binary confidence and target must be aligned")
    boundaries = torch.linspace(
        0.0, 1.0, bins + 1, dtype=confidence.dtype, device=confidence.device
    )
    error = confidence.new_zeros(())
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= boundaries[index]) & (
            confidence <= boundaries[index + 1]
            if upper_inclusive
            else confidence < boundaries[index + 1]
        )
        if mask.any():
            error = error + mask.float().mean() * (
                confidence[mask].mean() - target[mask].mean()
            ).abs()
    return error


def cube_cycle_report(
    rendered: SoftSplatResult,
    cube_drae: torch.Tensor,
    confidence: torch.Tensor,
    existence_target: torch.Tensor | None = None,
) -> dict[str, float]:
    target_probability, target_energy = normalized_cube_spectrum(cube_drae)
    local = covered_spectrum_kl(rendered, target_probability, target_energy)
    marginal = doppler_marginal_kl(rendered, target_probability, target_energy)
    spatial = spatial_energy_loss(rendered, target_energy)
    covered = rendered.covered_rae
    target_support = target_peak_support(target_energy)
    confidence_quantiles = torch.quantile(
        confidence.float(),
        torch.tensor(
            [0.05, 0.25, 0.5, 0.75, 0.95],
            dtype=torch.float32,
            device=confidence.device,
        ),
    )
    report = {
        "local_spectrum_kl": float(local.item()),
        "doppler_marginal_kl": float(marginal.item()),
        "spatial_energy_loss": float(spatial.item()),
        "confidence_mean": float(confidence.mean().item()),
        "confidence_q05": float(confidence_quantiles[0].item()),
        "confidence_q25": float(confidence_quantiles[1].item()),
        "confidence_median": float(confidence_quantiles[2].item()),
        "confidence_q75": float(confidence_quantiles[3].item()),
        "confidence_q95": float(confidence_quantiles[4].item()),
        "rendered_total_energy": float(rendered.energy_drae.sum().item()),
        "covered_cell_count": int(covered.sum().item()),
        "covered_cell_fraction": float(covered.float().mean().item()),
        "target_support_cell_count": int(target_support.sum().item()),
        "target_support_recall": float(
            (covered & target_support).sum().float()
            / target_support.sum().clamp_min(1)
        ),
        "covered_energy_correlation": tensor_correlation(
            rendered.spatial_energy_rae[covered], target_energy[covered]
        ),
    }
    if existence_target is not None:
        if existence_target.shape != confidence.shape:
            raise ValueError("Existence target does not match point confidence")
        existence_target = existence_target.to(confidence)
        report["existence_ece_10bin"] = float(
            binary_ece(confidence.float(), existence_target.float()).item()
        )
        report["existence_nll"] = float(
            -(
                existence_target * confidence.clamp_min(1e-8).log()
                + (1.0 - existence_target)
                * (1.0 - confidence).clamp_min(1e-8).log()
            )
            .mean()
            .item()
        )
        report["existence_target_fraction"] = float(existence_target.mean().item())
    return report


def aggregate_cycle_reports(
    reports: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not reports:
        raise ValueError("Cannot aggregate an empty Cube-cycle report list")
    keys = sorted(
        {
            key
            for report in reports
            for key, value in report.items()
            if isinstance(value, (int, float))
        }
    )
    aggregate = {}
    for key in keys:
        values = np.asarray(
            [report[key] for report in reports if key in report and np.isfinite(report[key])],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "sample_count": int(values.size),
        }
    return aggregate
