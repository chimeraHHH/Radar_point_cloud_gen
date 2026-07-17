"""Cycle losses between soft-splatted points and an observed RAED Cube."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.point_to_cube import SoftSplatResult


def normalized_cube_spectrum(
    cube_drae: torch.Tensor,
    smoothing: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cube_drae.ndim != 4:
        raise ValueError(f"Expected one DRAE Cube, got {cube_drae.shape}")
    evidence = torch.log1p(cube_drae.clamp_min(0.0))
    spatial_energy = evidence.sum(dim=0)
    probability = (evidence + smoothing) / (
        evidence + smoothing
    ).sum(dim=0, keepdim=True).clamp_min(1e-12)
    return probability, spatial_energy


def covered_spectrum_kl(
    rendered: SoftSplatResult,
    target_probability_drae: torch.Tensor,
    target_spatial_energy_rae: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if rendered.normalized_spectrum_drae.shape != target_probability_drae.shape:
        raise ValueError("Rendered and target spectrum shapes differ")
    mask = rendered.covered_rae & (target_spatial_energy_rae > 0)
    if not mask.any():
        raise ValueError("No covered Cube cells for spectrum cycle loss")
    prediction = rendered.normalized_spectrum_drae[:, mask].transpose(0, 1)
    target = target_probability_drae[:, mask].transpose(0, 1)
    per_cell = (
        target
        * (
            target.clamp_min(epsilon).log()
            - prediction.clamp_min(epsilon).log()
        )
    ).sum(dim=1)
    weight = rendered.spatial_energy_rae[mask].detach()
    return (per_cell * weight).sum() / weight.sum().clamp_min(epsilon)


def doppler_marginal_kl(
    rendered: SoftSplatResult,
    target_probability_drae: torch.Tensor,
    target_spatial_energy_rae: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    prediction = rendered.energy_drae.sum(dim=(1, 2, 3))
    target = (
        target_probability_drae * target_spatial_energy_rae[None]
    ).sum(dim=(1, 2, 3))
    prediction = prediction / prediction.sum().clamp_min(epsilon)
    target = target / target.sum().clamp_min(epsilon)
    return (
        target
        * (target.clamp_min(epsilon).log() - prediction.clamp_min(epsilon).log())
    ).sum()


def spatial_energy_loss(
    rendered: SoftSplatResult,
    target_spatial_energy_rae: torch.Tensor,
    target_peak_count: int = 10_000,
) -> torch.Tensor:
    count = min(target_peak_count, target_spatial_energy_rae.numel())
    peak_flat = torch.topk(target_spatial_energy_rae.flatten(), count).indices
    target_peak = torch.zeros_like(target_spatial_energy_rae, dtype=torch.bool)
    target_peak.view(-1)[peak_flat] = True
    mask = rendered.covered_rae | target_peak
    if not mask.any():
        raise ValueError("No occupied cells for Cube energy cycle loss")
    prediction = torch.log1p(rendered.spatial_energy_rae[mask])
    target = torch.log1p(target_spatial_energy_rae[mask])
    prediction = prediction / prediction.mean().clamp_min(1e-8)
    target = target / target.mean().clamp_min(1e-8)
    return F.smooth_l1_loss(prediction, target)


def confidence_floor_loss(
    confidence: torch.Tensor,
    minimum_mean: float = 0.1,
) -> torch.Tensor:
    if confidence.ndim != 1:
        raise ValueError("Point confidence must be one-dimensional")
    return F.relu(confidence.new_tensor(minimum_mean) - confidence.mean())


def cube_cycle_loss(
    rendered: SoftSplatResult,
    cube_drae: torch.Tensor,
    confidence: torch.Tensor,
    variant: str,
    local_weight: float = 1.0,
    marginal_weight: float = 0.25,
    energy_weight: float = 0.25,
    confidence_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if variant not in ("local_peak", "marginal", "full"):
        raise ValueError(f"Unsupported Cube cycle variant {variant}")
    target_probability, target_energy = normalized_cube_spectrum(cube_drae)
    local = covered_spectrum_kl(rendered, target_probability, target_energy)
    total = local_weight * local
    components = {"local_spectrum_kl": local.detach()}
    if variant in ("marginal", "full"):
        marginal = doppler_marginal_kl(
            rendered, target_probability, target_energy
        )
        total = total + marginal_weight * marginal
        components["doppler_marginal_kl"] = marginal.detach()
    if variant == "full":
        energy = spatial_energy_loss(rendered, target_energy)
        total = total + energy_weight * energy
        components["spatial_energy"] = energy.detach()
    confidence_floor = confidence_floor_loss(confidence)
    total = total + confidence_weight * confidence_floor
    components["confidence_floor"] = confidence_floor.detach()
    components["confidence_mean"] = confidence.mean().detach()
    components["total"] = total.detach()
    return total, components
