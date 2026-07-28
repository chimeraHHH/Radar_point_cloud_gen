"""Stage-0 objectives for the RaLD-WCE condition-exclusive field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RaLDWCEStage0Loss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


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
