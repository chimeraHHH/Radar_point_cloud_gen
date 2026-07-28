"""Stage-0 objectives for the RaLD-WCE condition-exclusive field."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RaLDWCEStage0Loss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def _flat_target_indices(
    target_rae_index: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    if target_rae_index.ndim != 2 or target_rae_index.shape[1] != 3:
        raise ValueError("RaLD-WCE target RAE indices must have shape (M,3)")
    if target_rae_index.shape[0] <= 0:
        raise ValueError("RaLD-WCE target RAE indices cannot be empty")
    target = target_rae_index.detach().to(device="cpu", dtype=torch.long)
    maximum = torch.tensor(spatial_shape, dtype=torch.long) - 1
    if bool(((target < 0) | (target > maximum)).any()):
        raise ValueError("RaLD-WCE target RAE index is outside the Cube")
    return (
        target[:, 0] * spatial_shape[1] * spatial_shape[2]
        + target[:, 1] * spatial_shape[2]
        + target[:, 2]
    )


def sample_bounded_occupancy_queries(
    target_rae_index: torch.Tensor,
    *,
    spatial_shape: tuple[int, int, int],
    query_count: int,
    positive_query_ratio: float,
    seed: int,
) -> dict[str, torch.Tensor | int]:
    """Sample bounded occupied and empty coordinates without Cube-side hints.

    The positive/negative ratio is explicit so formal R-A1 training can match
    RaLD's 6.25% occupied-query protocol while legacy preflight coverage can
    retain its balanced stress configuration.
    """

    if (
        len(spatial_shape) != 3
        or any(int(size) <= 1 for size in spatial_shape)
    ):
        raise ValueError("RaLD-WCE spatial shape must contain three sizes > 1")
    spatial_count = math.prod(spatial_shape)
    if query_count < 2 or query_count > spatial_count:
        raise ValueError("RaLD-WCE query count is outside spatial capacity")
    if not 0.0 < positive_query_ratio < 1.0:
        raise ValueError("RaLD-WCE positive query ratio must lie in (0,1)")

    target_flat = torch.unique(
        _flat_target_indices(target_rae_index, spatial_shape),
        sorted=True,
    )
    requested_positive = max(1, int(round(query_count * positive_query_ratio)))
    positive_count = min(requested_positive, int(target_flat.numel()))
    if positive_count <= 0:
        raise ValueError("RaLD-WCE training requires positive target cells")
    selected_position = torch.linspace(
        0,
        target_flat.numel() - 1,
        positive_count,
    ).round().long()
    selected_flat = target_flat[selected_position]
    range_index = selected_flat // (spatial_shape[1] * spatial_shape[2])
    remainder = selected_flat % (spatial_shape[1] * spatial_shape[2])
    azimuth_index = remainder // spatial_shape[2]
    elevation_index = remainder % spatial_shape[2]
    selected_target = torch.stack(
        (range_index, azimuth_index, elevation_index),
        dim=-1,
    ).float()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    jitter = (
        torch.rand((positive_count, 3), generator=generator) - 0.5
    ) * 0.5
    maximum = torch.tensor(spatial_shape, dtype=torch.float32) - 1.0
    positive_query_bins = (selected_target + jitter).clamp(
        min=torch.zeros(3),
        max=maximum,
    )
    positive_residual = selected_target - positive_query_bins

    negative_count = query_count - positive_count
    target_cells = {int(value) for value in target_flat.tolist()}
    selected_negative_cells: set[int] = set()
    negative_queries: list[torch.Tensor] = []
    sobol = torch.quasirandom.SobolEngine(
        dimension=3,
        scramble=True,
        seed=seed,
    )
    while len(negative_queries) < negative_count:
        remaining = negative_count - len(negative_queries)
        candidates = sobol.draw(max(remaining * 2, 256))
        candidate_bins = candidates * maximum
        rounded = candidate_bins.round().long()
        flat = (
            rounded[:, 0] * spatial_shape[1] * spatial_shape[2]
            + rounded[:, 1] * spatial_shape[2]
            + rounded[:, 2]
        )
        for candidate, cell in zip(candidate_bins, flat.tolist()):
            cell = int(cell)
            if cell in target_cells or cell in selected_negative_cells:
                continue
            selected_negative_cells.add(cell)
            negative_queries.append(candidate)
            if len(negative_queries) == negative_count:
                break
        if len(selected_negative_cells) + len(target_cells) >= spatial_count:
            break
    if len(negative_queries) != negative_count:
        raise RuntimeError(
            "RaLD-WCE cannot construct the requested unique empty-cell queries"
        )

    negative_query_bins = torch.stack(negative_queries)
    query_bins = torch.cat((positive_query_bins, negative_query_bins), dim=0)
    occupancy_target = torch.cat(
        (torch.ones(positive_count), torch.zeros(negative_count))
    )
    residual_target = torch.cat(
        (positive_residual, torch.zeros(negative_count, 3)),
        dim=0,
    )
    normalized_rae = 2.0 * query_bins / maximum - 1.0
    order = torch.randperm(query_count, generator=generator)
    return {
        "normalized_rae": normalized_rae[order].unsqueeze(0),
        "occupancy_target": occupancy_target[order].unsqueeze(0),
        "residual_target_bins": residual_target[order].unsqueeze(0),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "requested_positive_count": requested_positive,
        "unique_target_cell_count": int(target_flat.numel()),
    }


def _validate_field_output(
    output: dict[str, torch.Tensor],
    expected_shape: tuple[int, int],
) -> None:
    for key in ("occupancy_logit", "confidence"):
        if key not in output or tuple(output[key].shape) != expected_shape:
            raise ValueError(f"RaLD-WCE {key} must have shape {expected_shape}")
    if "residual_bins" not in output or tuple(output["residual_bins"].shape) != (
        *expected_shape,
        3,
    ):
        raise ValueError("RaLD-WCE residual_bins must align with field queries")


def _range_stratified_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    normalized_range: torch.Tensor,
) -> torch.Tensor:
    boundaries = (-1.0, -0.5, 0.0, 1.0 + 1e-6)
    strata: list[torch.Tensor] = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (normalized_range >= lower) & (normalized_range < upper)
        if bool(mask.any()):
            strata.append(
                F.binary_cross_entropy_with_logits(
                    logits[mask].float(),
                    target[mask].float(),
                )
            )
    if not strata:
        raise ValueError("RaLD-WCE occupancy queries do not cover any range stratum")
    return torch.stack(strata).mean()


def rald_wce_stage0_loss(
    matched: dict[str, torch.Tensor],
    wrong: dict[str, torch.Tensor],
    normalized_rae: torch.Tensor,
    occupancy_target: torch.Tensor,
    residual_target_bins: torch.Tensor,
    *,
    wrong_condition_margin: float = 0.05,
    wrong_condition_weight: float = 0.5,
    residual_weight: float = 0.05,
    brier_weight: float = 0.05,
    range_stratified_occupancy: bool = False,
) -> RaLDWCEStage0Loss:
    """Compute occupancy, same-query wrong-condition, residual, and calibration."""

    if normalized_rae.ndim != 3 or normalized_rae.shape[-1] != 3:
        raise ValueError("RaLD-WCE normalized queries must have shape (B,N,3)")
    expected_shape = tuple(normalized_rae.shape[:2])
    if tuple(occupancy_target.shape) != expected_shape:
        raise ValueError("RaLD-WCE occupancy target must align with queries")
    if tuple(residual_target_bins.shape) != (*expected_shape, 3):
        raise ValueError("RaLD-WCE residual target must align with queries")
    if not bool(torch.isfinite(normalized_rae).all()):
        raise ValueError("RaLD-WCE normalized queries must be finite")
    if not bool(torch.isfinite(occupancy_target).all()):
        raise ValueError("RaLD-WCE occupancy target must be finite")
    if bool(((occupancy_target < 0.0) | (occupancy_target > 1.0)).any()):
        raise ValueError("RaLD-WCE occupancy target must lie in [0,1]")
    if not bool(torch.isfinite(residual_target_bins).all()):
        raise ValueError("RaLD-WCE residual target must be finite")
    if wrong_condition_margin <= 0.0:
        raise ValueError("RaLD-WCE wrong-condition margin must be positive")
    if min(wrong_condition_weight, residual_weight, brier_weight) < 0.0:
        raise ValueError("RaLD-WCE loss weights must be nonnegative")
    _validate_field_output(matched, expected_shape)
    _validate_field_output(wrong, expected_shape)
    if "query_normalized_rae" in matched and not torch.equal(
        matched["query_normalized_rae"],
        normalized_rae,
    ):
        raise ValueError("RaLD-WCE matched branch changed the query set")
    if "query_normalized_rae" in wrong and not torch.equal(
        wrong["query_normalized_rae"],
        normalized_rae,
    ):
        raise ValueError("RaLD-WCE wrong-condition branch changed the query set")

    normalized_range = normalized_rae[..., 0]
    if range_stratified_occupancy:
        matched_occupancy = _range_stratified_bce(
            matched["occupancy_logit"],
            occupancy_target,
            normalized_range,
        )
        wrong_occupancy = _range_stratified_bce(
            wrong["occupancy_logit"],
            occupancy_target,
            normalized_range,
        )
    else:
        matched_occupancy = F.binary_cross_entropy_with_logits(
            matched["occupancy_logit"].float(),
            occupancy_target.float(),
        )
        wrong_occupancy = F.binary_cross_entropy_with_logits(
            wrong["occupancy_logit"].float(),
            occupancy_target.float(),
        )
    wrong_condition = torch.relu(
        matched_occupancy.new_tensor(wrong_condition_margin)
        + matched_occupancy
        - wrong_occupancy
    )

    positive = occupancy_target > 0.5
    if not bool(positive.any()):
        raise ValueError("RaLD-WCE residual objective requires positive queries")
    residual = F.smooth_l1_loss(
        matched["residual_bins"][positive].float(),
        residual_target_bins[positive].float(),
    )
    brier = (
        matched["confidence"].float() - occupancy_target.float()
    ).square().mean()
    total = (
        matched_occupancy
        + wrong_condition_weight * wrong_condition
        + residual_weight * residual
        + brier_weight * brier
    )
    return RaLDWCEStage0Loss(
        total=total,
        components={
            "occupancy": matched_occupancy,
            "wrong_condition_occupancy": wrong_occupancy,
            "wrong_condition_margin": wrong_condition,
            "positive_residual": residual,
            "confidence_brier": brier,
        },
    )
