"""Evaluation helpers specific to the G1C radar-guided query family."""

from __future__ import annotations

import torch


def nearest_other_distance(
    xyz_m: torch.Tensor, *, chunk_size: int = 512
) -> torch.Tensor:
    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3 or xyz_m.shape[0] < 2:
        raise ValueError("Nearest-other distance requires at least two XYZ points")
    if chunk_size <= 0:
        raise ValueError("Nearest-other chunk size must be positive")
    nearest = []
    point_count = xyz_m.shape[0]
    all_indices = torch.arange(point_count, device=xyz_m.device)
    for start in range(0, point_count, chunk_size):
        stop = min(start + chunk_size, point_count)
        distance = torch.cdist(xyz_m[start:stop], xyz_m)
        local_indices = torch.arange(start, stop, device=xyz_m.device)
        self_mask = local_indices[:, None] == all_indices[None]
        distance = distance.masked_fill(self_mask, float("inf"))
        nearest.append(distance.amin(dim=1))
    return torch.cat(nearest)


def duplicate_report(
    xyz_m: torch.Tensor, *, threshold_m: float = 0.05
) -> dict[str, float]:
    distance = nearest_other_distance(xyz_m)
    return {
        "duplicate_fraction_0p05m": float(
            (distance < threshold_m).float().mean().item()
        ),
        "nearest_other_median_m": float(distance.median().item()),
        "nearest_other_mean_m": float(distance.mean().item()),
    }
