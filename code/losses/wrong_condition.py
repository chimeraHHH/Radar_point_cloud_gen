"""Wrong-condition interventions and ranking loss for forced-temporal T-WC."""

from __future__ import annotations

from dataclasses import dataclass

import torch


WRONG_CONDITIONS = (
    "history_replacement",
    "time_reversal",
    "doppler_zero",
    "doppler_sign",
    "doppler_bin_roll",
)


@dataclass(frozen=True)
class WrongConditionLoss:
    total: torch.Tensor
    per_example: torch.Tensor
    matched_discrepancy: torch.Tensor
    wrong_discrepancy: torch.Tensor


def _validate_history(
    history_xyz_m: torch.Tensor,
    history_doppler_probability: torch.Tensor,
    history_confidence: torch.Tensor,
    history_source_id: torch.Tensor,
) -> tuple[int, int, int, int]:
    if history_xyz_m.ndim != 4 or history_xyz_m.shape[-1] != 3:
        raise ValueError("History XYZ must have shape (B,K,N,3)")
    batch, frame_count, point_count, _ = history_xyz_m.shape
    if history_doppler_probability.ndim != 4:
        raise ValueError("History Doppler must have shape (B,K,N,D)")
    doppler_count = history_doppler_probability.shape[-1]
    if history_doppler_probability.shape[:3] != (
        batch,
        frame_count,
        point_count,
    ):
        raise ValueError("History Doppler does not align with history XYZ")
    if history_confidence.shape != (batch, frame_count, point_count):
        raise ValueError("History confidence does not align with history XYZ")
    if history_source_id.shape != (batch, frame_count, point_count):
        raise ValueError("History source IDs do not align with history XYZ")
    return batch, frame_count, point_count, doppler_count


def _signed_bin_permutation(doppler_mps: torch.Tensor) -> torch.Tensor:
    distance = (
        doppler_mps[:, None].to(torch.float64)
        + doppler_mps[None, :].to(torch.float64)
    ).abs()
    return distance.argmin(dim=1)


def _zero_doppler_probability(
    reference: torch.Tensor,
    doppler_mps: torch.Tensor,
) -> torch.Tensor:
    if doppler_mps.min() > 0.0 or doppler_mps.max() < 0.0:
        raise ValueError("Doppler zero intervention requires an axis spanning zero")
    probability = torch.zeros_like(reference)
    exact = torch.nonzero(doppler_mps == 0.0, as_tuple=False).flatten()
    if exact.numel():
        probability[..., int(exact[0].item())] = 1.0
        return probability
    upper = int(torch.searchsorted(doppler_mps, doppler_mps.new_tensor(0.0)).item())
    lower = upper - 1
    lower_value = doppler_mps[lower]
    upper_value = doppler_mps[upper]
    upper_weight = -lower_value / (upper_value - lower_value)
    probability[..., lower] = 1.0 - upper_weight
    probability[..., upper] = upper_weight
    return probability


def apply_wrong_condition(
    *,
    history_xyz_m: torch.Tensor,
    history_doppler_probability: torch.Tensor,
    history_confidence: torch.Tensor,
    history_source_id: torch.Tensor,
    condition: str,
    doppler_mps: torch.Tensor,
    bin_roll: int = 8,
    replacement_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return a modified causal history while leaving the current Cube fixed."""

    batch, _, _, doppler_count = _validate_history(
        history_xyz_m,
        history_doppler_probability,
        history_confidence,
        history_source_id,
    )
    if condition not in WRONG_CONDITIONS:
        raise ValueError(f"Unsupported T-WC wrong condition {condition}")
    if doppler_mps.shape != (doppler_count,):
        raise ValueError("Doppler axis does not match history distributions")
    result = {
        "history_xyz_m": history_xyz_m.clone(),
        "history_doppler_probability": (
            history_doppler_probability.clone()
        ),
        "history_confidence": history_confidence.clone(),
        "history_source_id": history_source_id.clone(),
    }
    if condition == "history_replacement":
        if batch < 2 and replacement_index is None:
            raise ValueError(
                "History replacement requires batch size at least two "
                "or explicit replacement indices"
            )
        if replacement_index is None:
            replacement_index = torch.roll(
                torch.arange(batch, device=history_xyz_m.device),
                shifts=1,
            )
        if replacement_index.shape != (batch,):
            raise ValueError("Replacement index must have shape (B,)")
        if (replacement_index < 0).any() or (
            replacement_index >= batch
        ).any():
            raise IndexError("Replacement index is out of bounds")
        for name in result:
            result[name] = result[name][replacement_index]
    elif condition == "time_reversal":
        for name in result:
            result[name] = torch.flip(result[name], dims=(1,))
    elif condition == "doppler_zero":
        result["history_doppler_probability"] = (
            _zero_doppler_probability(
                history_doppler_probability,
                doppler_mps.to(history_doppler_probability),
            )
        )
    elif condition == "doppler_sign":
        permutation = _signed_bin_permutation(
            doppler_mps.to(history_doppler_probability.device)
        )
        result["history_doppler_probability"] = (
            history_doppler_probability[..., permutation]
        )
    else:
        if bin_roll == 0 or abs(bin_roll) >= doppler_count:
            raise ValueError(
                "Doppler bin roll must be non-zero and smaller than bin count"
            )
        result["history_doppler_probability"] = torch.roll(
            history_doppler_probability,
            shifts=bin_roll,
            dims=-1,
        )
    return result


def wrong_condition_margin_loss(
    matched_discrepancy: torch.Tensor,
    wrong_discrepancy: torch.Tensor,
    *,
    margin: float = 0.05,
) -> WrongConditionLoss:
    """Require a wrong temporal condition to be worse than the matched one."""

    if matched_discrepancy.shape != wrong_discrepancy.shape:
        raise ValueError("Matched and wrong discrepancies must have equal shape")
    if margin <= 0.0:
        raise ValueError("Wrong-condition margin must be positive")
    if not torch.isfinite(matched_discrepancy).all() or not torch.isfinite(
        wrong_discrepancy
    ).all():
        raise ValueError("Wrong-condition discrepancies must be finite")
    per_example = torch.relu(
        matched_discrepancy.new_tensor(margin)
        + matched_discrepancy
        - wrong_discrepancy
    )
    return WrongConditionLoss(
        total=per_example.mean(),
        per_example=per_example,
        matched_discrepancy=matched_discrepancy,
        wrong_discrepancy=wrong_discrepancy,
    )
