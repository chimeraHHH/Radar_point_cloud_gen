"""Evaluation metrics for differentiable point-to-Cube reconstruction."""

from __future__ import annotations

import numpy as np
import torch

from losses.cube_cycle import (
    covered_spectrum_kl,
    doppler_marginal_kl,
    normalized_cube_spectrum,
    spatial_energy_loss,
)
from models.point_to_cube import SoftSplatResult


def tensor_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().flatten()
    second = second.float().flatten()
    if first.numel() < 2 or first.std() <= 1e-12 or second.std() <= 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack((first, second)))[0, 1].item())


def cube_cycle_report(
    rendered: SoftSplatResult,
    cube_drae: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float]:
    target_probability, target_energy = normalized_cube_spectrum(cube_drae)
    local = covered_spectrum_kl(rendered, target_probability, target_energy)
    marginal = doppler_marginal_kl(rendered, target_probability, target_energy)
    spatial = spatial_energy_loss(rendered, target_energy)
    covered = rendered.covered_rae
    confidence_quantiles = torch.quantile(
        confidence.float(),
        torch.tensor(
            [0.05, 0.25, 0.5, 0.75, 0.95],
            dtype=torch.float32,
            device=confidence.device,
        ),
    )
    return {
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
        "covered_energy_correlation": tensor_correlation(
            rendered.spatial_energy_rae[covered], target_energy[covered]
        ),
    }


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
