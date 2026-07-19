"""Evaluation and aggregation for current-Cube temporal predictions."""

from __future__ import annotations

import numpy as np
import torch

from losses.temporal_consistency import (
    EgoAlignedMatch,
    TemporalMatch,
    occupancy_flicker,
)


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> float:
    return float(
        ((values * weight).sum() / weight.sum().clamp_min(1e-8)).item()
    )


def weighted_quantile(
    values: torch.Tensor, weight: torch.Tensor, quantile: float
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must lie in [0, 1]")
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weight = weight[order]
    cumulative = torch.cumsum(sorted_weight, dim=0)
    threshold = cumulative[-1] * quantile
    index = torch.searchsorted(cumulative, threshold).clamp_max(
        sorted_values.numel() - 1
    )
    return float(sorted_values[index].item())


def temporal_consistency_report(
    match: TemporalMatch,
    previous_confidence: torch.Tensor,
    current_xyz_m: torch.Tensor,
    current_confidence: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> dict[str, float]:
    error = match.radial_error_m.detach().float()
    distance = match.match_distance_m.detach().float()
    weight = match.weight.detach().float()
    if float(weight.sum().item()) <= 1e-8:
        raise ValueError("Temporal report has zero effective correspondence weight")
    flicker = occupancy_flicker(
        match.expected_prior_xyz_m.detach(),
        previous_confidence.detach().float(),
        current_xyz_m.detach().float(),
        current_confidence.detach().float(),
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    report = {
        "temporal_radial_error_mean_m": weighted_mean(error, weight),
        "temporal_radial_error_median_m": weighted_quantile(error, weight, 0.5),
        "temporal_radial_error_q90_m": weighted_quantile(error, weight, 0.9),
        "matched_distance_mean_m": weighted_mean(distance, weight),
        "matched_distance_median_m": weighted_quantile(distance, weight, 0.5),
        "occupancy_flicker": float(flicker.item()),
        "effective_match_weight": float(weight.sum().item()),
        "point_count": int(error.numel()),
    }
    for threshold in (0.25, 0.5, 1.0):
        suffix = str(threshold).replace(".", "p")
        report[f"temporal_radial_fraction_{suffix}m"] = weighted_mean(
            (error <= threshold).to(weight), weight
        )
    return report


def ego_aligned_consistency_report(
    match: EgoAlignedMatch,
    previous_confidence: torch.Tensor,
    current_xyz_m: torch.Tensor,
    current_confidence: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> dict[str, float]:
    distance = match.match_distance_m.detach().float()
    weight = match.weight.detach().float()
    if float(weight.sum().item()) <= 1e-8:
        raise ValueError("Ego-aligned report has zero correspondence weight")
    flicker = occupancy_flicker(
        match.expected_prior_xyz_m.detach(),
        previous_confidence.detach().float(),
        current_xyz_m.detach().float(),
        current_confidence.detach().float(),
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    return {
        "ego_aligned_matched_distance_mean_m": weighted_mean(distance, weight),
        "ego_aligned_matched_distance_median_m": weighted_quantile(
            distance, weight, 0.5
        ),
        "ego_aligned_matched_distance_q90_m": weighted_quantile(
            distance, weight, 0.9
        ),
        "occupancy_flicker": float(flicker.item()),
        "effective_match_weight": float(weight.sum().item()),
        "point_count": int(distance.numel()),
    }


def aggregate_temporal_reports(
    reports: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not reports:
        raise ValueError("Cannot aggregate empty temporal reports")
    keys = sorted(
        {
            key
            for report in reports
            for key, value in report.items()
            if isinstance(value, (int, float)) and key != "point_count"
        }
    )
    result = {}
    for key in keys:
        values = np.asarray(
            [report[key] for report in reports if key in report], dtype=np.float64
        )
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "sample_count": int(values.size),
        }
    return result
