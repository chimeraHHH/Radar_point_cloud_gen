"""Dense Cube-to-point geometry decoding and metrics."""

from __future__ import annotations

import numpy as np
import torch

from cube_dense.kradar import KRadarAxes


def rae_indices_to_xyz(
    indices_rae: torch.Tensor,
    axes: KRadarAxes,
) -> torch.Tensor:
    if indices_rae.ndim != 2 or indices_rae.shape[1] != 3:
        raise ValueError(f"Expected (N,3) RAE indices, got {indices_rae.shape}")
    device = indices_rae.device
    dtype = torch.float32
    range_axis = torch.as_tensor(axes.range_m, dtype=dtype, device=device)
    azimuth_axis = torch.as_tensor(axes.azimuth_rad, dtype=dtype, device=device)
    elevation_axis = torch.as_tensor(axes.elevation_rad, dtype=dtype, device=device)
    radius = range_axis[indices_rae[:, 0]]
    azimuth = azimuth_axis[indices_rae[:, 1]]
    elevation = elevation_axis[indices_rae[:, 2]]
    cosine = torch.cos(elevation)
    return torch.stack(
        (
            radius * cosine * torch.cos(azimuth),
            radius * cosine * torch.sin(azimuth),
            radius * torch.sin(elevation),
        ),
        dim=1,
    )


def occupancy_to_points(
    logits_rae: torch.Tensor,
    axes: KRadarAxes,
    point_count: int = 10_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits_rae.ndim != 3:
        raise ValueError(f"Expected (R,A,E) occupancy logits, got {logits_rae.shape}")
    flat = torch.sigmoid(logits_rae).flatten()
    count = min(point_count, flat.numel())
    confidence, flat_index = torch.topk(flat, count, sorted=True)
    elevation_count = logits_rae.shape[2]
    azimuth_count = logits_rae.shape[1]
    range_index = flat_index // (azimuth_count * elevation_count)
    remainder = flat_index % (azimuth_count * elevation_count)
    azimuth_index = remainder // elevation_count
    elevation_index = remainder % elevation_count
    indices = torch.stack((range_index, azimuth_index, elevation_index), dim=1)
    return rae_indices_to_xyz(indices, axes), confidence, indices


def nearest_distance(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    if source_xyz.numel() == 0 or target_xyz.numel() == 0:
        raise ValueError("Nearest-neighbour metrics require non-empty point sets")
    distances = []
    for start in range(0, source_xyz.shape[0], chunk_size):
        chunk = source_xyz[start : start + chunk_size]
        distances.append(torch.cdist(chunk, target_xyz).amin(dim=1))
    return torch.cat(distances)


def geometry_report(
    prediction_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_weight: torch.Tensor | None = None,
    thresholds_m: tuple[float, ...] = (0.5, 1.0, 2.0),
    distance_bins_m: tuple[tuple[float, float], ...] = (
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 120.0),
    ),
    chunk_size: int = 1024,
) -> dict[str, float]:
    prediction_to_target = nearest_distance(
        prediction_xyz, target_xyz, chunk_size=chunk_size
    )
    target_to_prediction = nearest_distance(
        target_xyz, prediction_xyz, chunk_size=chunk_size
    )
    if target_weight is None:
        target_weight = torch.ones_like(target_to_prediction)
    if target_weight.shape != target_to_prediction.shape:
        raise ValueError(
            f"Target-weight mismatch: {target_weight.shape} vs {target_to_prediction.shape}"
        )
    target_weight = target_weight.to(target_to_prediction).clamp_min(0.0)
    weight_sum = target_weight.sum().clamp_min(1e-8)
    weighted_completeness = (target_to_prediction * target_weight).sum() / weight_sum
    report = {
        "chamfer_m": float(
            (prediction_to_target.mean() + weighted_completeness).item()
        ),
        "precision_mean_distance_m": float(prediction_to_target.mean().item()),
        "completeness_mean_distance_m": float(weighted_completeness.item()),
        "outlier_fraction_2m": float((prediction_to_target > 2.0).float().mean().item()),
        "prediction_count": int(prediction_xyz.shape[0]),
        "target_count": int(target_xyz.shape[0]),
        "target_effective_count": float(target_weight.sum().item()),
    }
    for threshold in thresholds_m:
        precision = (prediction_to_target <= threshold).float().mean()
        recall = (
            (target_to_prediction <= threshold).to(target_weight) * target_weight
        ).sum() / weight_sum
        fscore = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        suffix = str(threshold).replace(".", "p")
        report[f"precision_{suffix}m"] = float(precision.item())
        report[f"recall_{suffix}m"] = float(recall.item())
        report[f"fscore_{suffix}m"] = float(fscore.item())
    prediction_range = torch.linalg.vector_norm(prediction_xyz, dim=1)
    target_range = torch.linalg.vector_norm(target_xyz, dim=1)
    for lower, upper in distance_bins_m:
        prediction_mask = (prediction_range >= lower) & (prediction_range < upper)
        target_mask = (target_range >= lower) & (target_range < upper)
        if not prediction_mask.any() or not target_mask.any():
            continue
        bin_prediction = prediction_to_target[prediction_mask]
        bin_target = target_to_prediction[target_mask]
        bin_weight = target_weight[target_mask]
        bin_weight_sum = bin_weight.sum().clamp_min(1e-8)
        precision = (bin_prediction <= 1.0).float().mean()
        recall = ((bin_target <= 1.0).to(bin_weight) * bin_weight).sum()
        recall = recall / bin_weight_sum
        fscore = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        prefix = f"range_{int(lower)}_{int(upper)}m"
        report[f"{prefix}_precision_mean_distance_m"] = float(
            bin_prediction.mean().item()
        )
        report[f"{prefix}_completeness_mean_distance_m"] = float(
            ((bin_target * bin_weight).sum() / bin_weight_sum).item()
        )
        report[f"{prefix}_fscore_1m"] = float(fscore.item())
    return report


def aggregate_geometry_reports(reports: list[dict[str, float]]) -> dict[str, dict]:
    if not reports:
        raise ValueError("Cannot aggregate an empty report list")
    numeric_keys = sorted(
        {
            key
            for report in reports
            for key, value in report.items()
            if isinstance(value, (int, float)) and not key.endswith("_count")
        }
    )
    aggregate = {}
    for key in numeric_keys:
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
