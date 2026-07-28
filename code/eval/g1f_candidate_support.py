"""Diagnostic candidate-support oracle for G1F-F0.

This module deliberately uses ground-truth geometry during selection. Its
outputs are unattainable diagnostics and are not eligible method results.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from eval.dense_geometry import nearest_distance


PROPOSAL_COUNT = 32_000
EXPORT_COUNT = 10_000
ORACLE_ARTIFACT_LABEL = "g1f_f0_diagnostic_unattainable_gt_oracle"
RANGE_BINS_M = (
    ("range_0_30m", 0.0, 30.0),
    ("range_30_60m", 30.0, 60.0),
    ("range_60_120m", 60.0, 120.0),
)
SUPPORT_THRESHOLDS_M = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class CandidateSupportSelection:
    """One exact, index-capacity-constrained oracle export."""

    selected_pool_indices: torch.Tensor
    selected_candidate_indices: torch.Tensor
    selected_xyz_m: torch.Tensor
    per_range_support: dict[str, dict[str, float | int | None]]
    artifact_label: dict[str, bool | str]


def diagnostic_artifact_label() -> dict[str, bool | str]:
    """Return the mandatory label carried by every G1F-F0 artifact."""

    return {
        "label": ORACLE_ARTIFACT_LABEL,
        "diagnostic": True,
        "unattainable": True,
        "ground_truth_used_for_selection": True,
        "eligible_as_method_result": False,
    }


def _validate_inputs(
    candidate_xyz_m: torch.Tensor,
    candidate_indices: torch.Tensor,
    target_xyz_m: torch.Tensor,
    target_weight: torch.Tensor | None,
) -> torch.Tensor:
    if candidate_xyz_m.shape != (PROPOSAL_COUNT, 3):
        raise ValueError(
            "G1F-F0 requires exactly 32,000 candidate XYZ coordinates"
        )
    if not torch.is_floating_point(candidate_xyz_m):
        raise TypeError("G1F-F0 candidate XYZ coordinates must be floating point")
    if not torch.isfinite(candidate_xyz_m).all():
        raise ValueError("G1F-F0 candidate XYZ coordinates must be finite")
    if candidate_indices.shape != (PROPOSAL_COUNT,):
        raise ValueError("G1F-F0 requires one index for every candidate")
    if candidate_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("G1F-F0 candidate indices must be integers")
    if torch.unique(candidate_indices).numel() != PROPOSAL_COUNT:
        raise ValueError("G1F-F0 candidate indices must be unique")
    if (
        target_xyz_m.ndim != 2
        or target_xyz_m.shape[1] != 3
        or target_xyz_m.shape[0] == 0
    ):
        raise ValueError("G1F-F0 target XYZ must have non-empty shape (N,3)")
    if not torch.is_floating_point(target_xyz_m):
        raise TypeError("G1F-F0 target XYZ coordinates must be floating point")
    if not torch.isfinite(target_xyz_m).all():
        raise ValueError("G1F-F0 target XYZ coordinates must be finite")
    if target_weight is None:
        target_weight = torch.ones(
            target_xyz_m.shape[0],
            dtype=target_xyz_m.dtype,
            device=target_xyz_m.device,
        )
    if target_weight.shape != (target_xyz_m.shape[0],):
        raise ValueError("G1F-F0 target weights must have shape (N,)")
    target_weight = target_weight.to(target_xyz_m)
    if not torch.isfinite(target_weight).all() or bool((target_weight < 0).any()):
        raise ValueError("G1F-F0 target weights must be finite and nonnegative")
    if float(target_weight.sum().item()) <= 0.0:
        raise ValueError("G1F-F0 target weights must have positive total mass")
    return target_weight


def _range_masks(xyz_m: torch.Tensor) -> list[torch.Tensor]:
    radius = torch.linalg.vector_norm(xyz_m, dim=1)
    masks = [
        (radius >= lower) & (radius < upper)
        for _, lower, upper in RANGE_BINS_M
    ]
    covered = torch.stack(masks).any(dim=0)
    if not bool(covered.all()):
        raise ValueError("G1F-F0 coordinates must lie in the frozen [0,120) m bins")
    return masks


def _capacity_constrained_quotas(
    target_mass: list[float],
    candidate_capacity: list[int],
) -> list[int]:
    if sum(candidate_capacity) != PROPOSAL_COUNT:
        raise ValueError("G1F-F0 range capacities must cover all proposals")
    if sum(candidate_capacity) < EXPORT_COUNT:
        raise ValueError("G1F-F0 candidate capacity cannot fill the export")
    quotas = [0 for _ in candidate_capacity]
    for _ in range(EXPORT_COUNT):
        eligible = [
            index
            for index, capacity in enumerate(candidate_capacity)
            if quotas[index] < capacity
        ]
        if not eligible:
            raise RuntimeError("G1F-F0 exhausted candidate capacity")
        supported = [index for index in eligible if target_mass[index] > 0.0]
        allocation = supported or eligible
        if supported:
            chosen = min(
                allocation,
                key=lambda index: (
                    (quotas[index] + 1) / target_mass[index],
                    index,
                ),
            )
        else:
            chosen = min(
                allocation,
                key=lambda index: (
                    (quotas[index] + 1) / candidate_capacity[index],
                    index,
                ),
            )
        quotas[chosen] += 1
    if sum(quotas) != EXPORT_COUNT:
        raise AssertionError("G1F-F0 quota allocation changed export cardinality")
    return quotas


def _stable_distance_order(
    distance_m: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    by_index = torch.argsort(candidate_indices, stable=True)
    by_distance = torch.argsort(distance_m[by_index], stable=True)
    return by_index[by_distance]


def nearest_candidate_assignment(
    target_xyz_m: torch.Tensor,
    candidate_xyz_m: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map every target to one nearest candidate with deterministic ties.

    Returned candidate indices are local row positions in ``candidate_xyz_m``.
    Candidates are ordered by their stable external index before distance
    minimization, so an exact distance tie resolves to the smaller index.
    """

    if (
        target_xyz_m.ndim != 2
        or candidate_xyz_m.ndim != 2
        or target_xyz_m.shape[1] != 3
        or candidate_xyz_m.shape[1] != 3
    ):
        raise ValueError("Nearest-candidate assignment requires XYZ matrices")
    if target_xyz_m.shape[0] == 0 or candidate_xyz_m.shape[0] == 0:
        raise ValueError("Nearest-candidate assignment requires non-empty sets")
    if candidate_indices.shape != (candidate_xyz_m.shape[0],):
        raise ValueError("Nearest-candidate indices must match candidate rows")
    if chunk_size <= 0:
        raise ValueError("Nearest-candidate chunk size must be positive")

    candidate_order = torch.argsort(candidate_indices, stable=True)
    ordered_candidates = candidate_xyz_m[candidate_order]
    nearest_distance_parts = []
    nearest_index_parts = []
    for start in range(0, target_xyz_m.shape[0], chunk_size):
        distance = torch.cdist(
            target_xyz_m[start : start + chunk_size],
            ordered_candidates,
        )
        nearest_distance_m, ordered_index = distance.min(dim=1)
        nearest_distance_parts.append(nearest_distance_m)
        nearest_index_parts.append(candidate_order[ordered_index])
    return (
        torch.cat(nearest_distance_parts),
        torch.cat(nearest_index_parts),
    )


def _segment_sum(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    prefix = torch.cat((values.new_zeros(1), values.cumsum(dim=0)))
    ends = counts.cumsum(dim=0)
    starts = ends - counts
    return prefix[ends] - prefix[starts]


def _covered_candidate_statistics(
    assigned_candidate: torch.Tensor,
    assignment_distance_m: torch.Tensor,
    target_weight: torch.Tensor,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate target confidence and weighted distance without index reuse."""

    order = torch.argsort(assigned_candidate, stable=True)
    assigned = assigned_candidate[order]
    weight = target_weight[order]
    weighted_distance = assignment_distance_m[order] * weight
    unique_candidate, counts = torch.unique_consecutive(
        assigned,
        return_counts=True,
    )
    mass_values = _segment_sum(weight, counts)
    distance_values = _segment_sum(weighted_distance, counts)
    covered_mass = weight.new_zeros(candidate_count)
    covered_distance = weight.new_full((candidate_count,), float("inf"))
    covered_mass[unique_candidate] = mass_values
    positive = mass_values > 0.0
    covered_distance[unique_candidate[positive]] = (
        distance_values[positive] / mass_values[positive]
    )
    return covered_mass, covered_distance


def _stable_covered_mass_order(
    covered_mass: torch.Tensor,
    covered_distance_m: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    assigned = torch.nonzero(covered_mass > 0.0, as_tuple=False).flatten()
    by_index = torch.argsort(candidate_indices[assigned], stable=True)
    ordered = assigned[by_index]
    by_distance = torch.argsort(
        covered_distance_m[ordered],
        stable=True,
    )
    ordered = ordered[by_distance]
    by_mass = torch.argsort(
        covered_mass[ordered],
        descending=True,
        stable=True,
    )
    return ordered[by_mass]


def _weighted_support_fraction(
    distance_m: torch.Tensor,
    weight: torch.Tensor,
    threshold_m: float,
) -> float:
    supported = (distance_m <= threshold_m).to(weight)
    return float(
        ((supported * weight).sum() / weight.sum().clamp_min(1e-8)).item()
    )


def select_candidate_support_oracle(
    candidate_xyz_m: torch.Tensor,
    candidate_indices: torch.Tensor,
    target_xyz_m: torch.Tensor,
    *,
    target_weight: torch.Tensor | None = None,
    distance_chunk_size: int = 1024,
) -> CandidateSupportSelection:
    """Select the frozen 10k unattainable upper bound from a 32k pool.

    Candidate index capacity is exactly one. Output mass is allocated across
    the frozen range bins in proportion to target confidence mass, subject to
    candidate capacity. Within each bin, every GT target first votes for its
    nearest candidate. Unique candidates are ranked by covered confidence mass,
    weighted assignment distance, and candidate index. Nearest-GT candidate
    ranking is used only to fill unused quota.
    """

    if distance_chunk_size <= 0:
        raise ValueError("G1F-F0 distance chunk size must be positive")
    target_weight = _validate_inputs(
        candidate_xyz_m,
        candidate_indices,
        target_xyz_m,
        target_weight,
    )
    candidate_masks = _range_masks(candidate_xyz_m)
    target_masks = _range_masks(target_xyz_m)
    candidate_capacity = [
        int(mask.sum().item()) for mask in candidate_masks
    ]
    target_mass = [
        float(target_weight[mask].sum().item()) for mask in target_masks
    ]
    quotas = _capacity_constrained_quotas(target_mass, candidate_capacity)

    selected_pool_parts = []
    support: dict[str, dict[str, float | int | None]] = {}
    all_target_indices = torch.arange(
        target_xyz_m.shape[0], device=target_xyz_m.device
    )
    for bin_index, (label, lower, upper) in enumerate(RANGE_BINS_M):
        candidate_pool_indices = torch.nonzero(
            candidate_masks[bin_index], as_tuple=False
        ).flatten()
        target_indices = torch.nonzero(
            target_masks[bin_index], as_tuple=False
        ).flatten()
        width = upper - lower
        report: dict[str, float | int | None] = {
            "candidate_count": int(candidate_pool_indices.numel()),
            "candidate_density_per_radial_m": float(
                candidate_pool_indices.numel() / width
            ),
            "target_count": int(target_indices.numel()),
            "target_effective_count": float(
                target_weight[target_indices].sum().item()
            ),
            "output_quota": int(quotas[bin_index]),
            "selected_count": 0,
            "assigned_candidate_count": 0,
            "selected_assigned_candidate_count": 0,
            "fill_candidate_count": 0,
            "covered_target_effective_mass": 0.0,
            "selected_covered_target_effective_mass": 0.0,
            "selected_covered_target_mass_fraction": (
                0.0 if target_indices.numel() else None
            ),
        }
        if candidate_pool_indices.numel() == 0:
            if quotas[bin_index] != 0:
                raise AssertionError("G1F-F0 assigned mass to an empty range bin")
            for threshold in SUPPORT_THRESHOLDS_M:
                suffix = str(threshold).replace(".", "p")
                report[f"proposal_to_gt_support_fraction_{suffix}m"] = None
                report[f"selected_to_gt_support_fraction_{suffix}m"] = None
                report[f"gt_recall_from_proposals_{suffix}m"] = (
                    0.0 if target_indices.numel() else None
                )
                report[f"gt_recall_from_selected_{suffix}m"] = (
                    0.0 if target_indices.numel() else None
                )
            support[label] = report
            selected_pool_parts.append(candidate_pool_indices)
            continue
        support_target_indices = (
            target_indices if target_indices.numel() else all_target_indices
        )
        candidate_bin = candidate_xyz_m[candidate_pool_indices]
        support_target_bin = target_xyz_m[support_target_indices]
        candidate_to_target = nearest_distance(
            candidate_bin,
            support_target_bin,
            chunk_size=distance_chunk_size,
        )
        candidate_bin_indices = candidate_indices[candidate_pool_indices]
        selected_assigned = candidate_pool_indices.new_empty(0)
        covered_mass = candidate_bin.new_zeros(candidate_bin.shape[0])
        target_to_candidate = None
        if target_indices.numel():
            target_bin = target_xyz_m[target_indices]
            target_bin_weight = target_weight[target_indices]
            target_to_candidate, assigned_candidate = (
                nearest_candidate_assignment(
                    target_bin,
                    candidate_bin,
                    candidate_bin_indices,
                    chunk_size=distance_chunk_size,
                )
            )
            covered_mass, covered_distance = _covered_candidate_statistics(
                assigned_candidate,
                target_to_candidate,
                target_bin_weight,
                candidate_bin.shape[0],
            )
            covered_order = _stable_covered_mass_order(
                covered_mass,
                covered_distance,
                candidate_bin_indices,
            )
            selected_assigned = covered_order[: quotas[bin_index]]

        selected_mask = torch.zeros(
            candidate_bin.shape[0],
            dtype=torch.bool,
            device=candidate_bin.device,
        )
        selected_mask[selected_assigned] = True
        fill_count = quotas[bin_index] - int(selected_assigned.numel())
        available = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        fill_order = _stable_distance_order(
            candidate_to_target[available],
            candidate_bin_indices[available],
        )
        selected_fill = available[fill_order[:fill_count]]
        selected_local = torch.cat((selected_assigned, selected_fill))
        if selected_local.numel() != quotas[bin_index]:
            raise AssertionError("G1F-F0 could not fill its range quota")
        selected_pool = candidate_pool_indices[selected_local]
        selected_pool_parts.append(selected_pool)

        selected_bin = candidate_xyz_m[selected_pool]
        target_to_selected = (
            nearest_distance(
                target_xyz_m[target_indices],
                selected_bin,
                chunk_size=distance_chunk_size,
            )
            if target_indices.numel() and selected_bin.shape[0]
            else None
        )
        selected_covered_mass = float(
            covered_mass[selected_assigned].sum().item()
        )
        total_covered_mass = float(covered_mass.sum().item())
        report["selected_count"] = int(selected_pool.numel())
        report["assigned_candidate_count"] = int(
            (covered_mass > 0.0).sum().item()
        )
        report["selected_assigned_candidate_count"] = int(
            selected_assigned.numel()
        )
        report["fill_candidate_count"] = int(selected_fill.numel())
        report["covered_target_effective_mass"] = total_covered_mass
        report["selected_covered_target_effective_mass"] = (
            selected_covered_mass
        )
        report["selected_covered_target_mass_fraction"] = (
            selected_covered_mass / max(total_covered_mass, 1e-8)
            if target_indices.numel()
            else None
        )
        for threshold in SUPPORT_THRESHOLDS_M:
            suffix = str(threshold).replace(".", "p")
            report[f"proposal_to_gt_support_fraction_{suffix}m"] = float(
                (candidate_to_target <= threshold).float().mean().item()
            )
            report[f"selected_to_gt_support_fraction_{suffix}m"] = (
                float(
                    (candidate_to_target[selected_local] <= threshold)
                    .float()
                    .mean()
                    .item()
                )
                if selected_local.numel()
                else None
            )
            report[f"gt_recall_from_proposals_{suffix}m"] = (
                _weighted_support_fraction(
                    target_to_candidate,
                    target_weight[target_indices],
                    threshold,
                )
                if target_to_candidate is not None
                else None
            )
            report[f"gt_recall_from_selected_{suffix}m"] = (
                _weighted_support_fraction(
                    target_to_selected,
                    target_bin_weight,
                    threshold,
                )
                if target_to_selected is not None
                else None
            )
        support[label] = report

    selected_pool_indices = torch.cat(selected_pool_parts)
    if selected_pool_indices.shape != (EXPORT_COUNT,):
        raise AssertionError("G1F-F0 must export exactly 10,000 candidates")
    selected_candidate_indices = candidate_indices[selected_pool_indices]
    if torch.unique(selected_candidate_indices).numel() != EXPORT_COUNT:
        raise AssertionError("G1F-F0 selected a candidate index more than once")
    return CandidateSupportSelection(
        selected_pool_indices=selected_pool_indices,
        selected_candidate_indices=selected_candidate_indices,
        selected_xyz_m=candidate_xyz_m[selected_pool_indices],
        per_range_support=support,
        artifact_label=diagnostic_artifact_label(),
    )
