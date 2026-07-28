"""Training objectives for independent polar rectified flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from losses.doppler_distribution import (
    circular_wasserstein1,
    distribution_cross_entropy,
)


@dataclass(frozen=True)
class PolarRectifiedFlowLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def validate_source_to_target(
    source_to_target: torch.Tensor,
    point_count: int,
) -> None:
    if source_to_target.shape != (point_count,):
        raise ValueError("Source-to-target assignment must have shape (N,)")
    if source_to_target.dtype != torch.long:
        raise ValueError("Source-to-target assignment must use torch.long")
    expected = torch.arange(
        point_count,
        device=source_to_target.device,
        dtype=torch.long,
    )
    if not torch.equal(torch.sort(source_to_target).values, expected):
        raise ValueError("Source-to-target assignment must be one-to-one")


def paired_target(
    target_state: torch.Tensor,
    source_to_target: torch.Tensor,
) -> torch.Tensor:
    """Reorder target endpoints into source-particle order."""

    if target_state.ndim == 2:
        target_state = target_state.unsqueeze(0)
    if target_state.ndim != 3 or target_state.shape[-1] != 3:
        raise ValueError("Target flow state must have shape (N,3) or (B,N,3)")
    validate_source_to_target(source_to_target, target_state.shape[1])
    return target_state[:, source_to_target.to(target_state.device)]


def rectified_flow_interpolant(
    source_state: torch.Tensor,
    target_state_paired: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_state.shape != target_state_paired.shape or source_state.ndim != 3:
        raise ValueError("Source and paired target must align as (B,N,3)")
    if time.shape != (source_state.shape[0],):
        raise ValueError("Rectified-flow time must have shape (B,)")
    weight = time[:, None, None].to(source_state)
    state = (1.0 - weight) * source_state + weight * target_state_paired
    velocity = target_state_paired - source_state
    return state, velocity


def _flatten_distribution(values: torch.Tensor, point_count: int) -> torch.Tensor:
    if values.ndim == 2 and values.shape == (point_count, 64):
        values = values.unsqueeze(0)
    if values.ndim != 3 or values.shape[1:] != (point_count, 64):
        raise ValueError("Doppler distribution must have shape (B,N,64)")
    return values.reshape(-1, 64)


def _flatten_confidence(values: torch.Tensor, point_count: int) -> torch.Tensor:
    if values.ndim == 1 and values.shape == (point_count,):
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] != point_count:
        raise ValueError("Confidence target must have shape (B,N)")
    return values.reshape(-1)


def _pair_point_attributes(
    values: torch.Tensor,
    source_to_target: torch.Tensor,
    point_count: int,
) -> torch.Tensor:
    if values.ndim == 1:
        if values.shape[0] != point_count:
            raise ValueError("Point attributes must align with the target endpoint")
        values = values.unsqueeze(0)
    elif values.shape[0] == point_count:
        values = values.unsqueeze(0)
    if values.shape[1] != point_count:
        raise ValueError("Point attributes must align with the target endpoint")
    validate_source_to_target(source_to_target, point_count)
    return values[:, source_to_target.to(values.device)]


def polar_rectified_flow_loss(
    model,
    cube_drae: torch.Tensor,
    source_state: torch.Tensor,
    target_state: torch.Tensor,
    source_to_target: torch.Tensor,
    *,
    time: torch.Tensor,
    target_doppler_distribution: torch.Tensor | None = None,
    target_confidence: torch.Tensor | None = None,
    wrong_cube_drae: torch.Tensor | None = None,
    chunk_size: int = 2_048,
    endpoint_geometry_weight: float = 0.25,
    wrong_condition_weight: float = 0.1,
    wrong_condition_margin: float = 0.05,
    doppler_weight: float = 0.25,
    circular_weight: float = 0.25,
    confidence_weight: float = 0.05,
) -> PolarRectifiedFlowLoss:
    """Compute flow, endpoint, attribute, and wrong-condition objectives."""

    if source_state.ndim == 2:
        source_state = source_state.unsqueeze(0)
    batch_size, point_count, coordinate_dim = source_state.shape
    if coordinate_dim != 3:
        raise ValueError("Source flow state must end in three coordinates")
    paired = paired_target(target_state, source_to_target)
    if paired.shape[0] == 1 and batch_size > 1:
        paired = paired.expand(batch_size, -1, -1)
    if paired.shape != source_state.shape:
        raise ValueError("Paired target and source batches must align")
    if time.shape != (batch_size,):
        raise ValueError("Flow time must have shape (B,)")

    state_t, target_velocity = rectified_flow_interpolant(
        source_state,
        paired,
        time,
    )
    condition_latents = model.encode_condition(cube_drae)
    prediction = model.velocity(
        state_t,
        time,
        condition_latents,
        chunk_size=chunk_size,
    )
    flow = F.smooth_l1_loss(prediction.float(), target_velocity.float())
    remaining = 1.0 - time[:, None, None].to(state_t)
    endpoint = state_t + remaining * prediction
    endpoint_geometry = F.smooth_l1_loss(endpoint.float(), paired.float())
    total = flow + endpoint_geometry_weight * endpoint_geometry
    components = {
        "flow_matching": flow.detach(),
        "endpoint_geometry": endpoint_geometry.detach(),
    }

    if wrong_cube_drae is not None:
        wrong_condition = model.encode_condition(wrong_cube_drae)
        wrong_prediction = model.velocity(
            state_t,
            time,
            wrong_condition,
            chunk_size=chunk_size,
        )
        matched_error = F.smooth_l1_loss(
            prediction.float(),
            target_velocity.float(),
            reduction="none",
        ).mean(dim=(1, 2))
        wrong_error = F.smooth_l1_loss(
            wrong_prediction.float(),
            target_velocity.float(),
            reduction="none",
        ).mean(dim=(1, 2))
        wrong_condition_loss = F.relu(
            wrong_condition_margin + matched_error - wrong_error
        ).mean()
        total = total + wrong_condition_weight * wrong_condition_loss
        components["wrong_condition_margin"] = wrong_condition_loss.detach()
        components["wrong_condition_error_delta"] = (
            wrong_error - matched_error
        ).mean().detach()

    if target_doppler_distribution is not None or target_confidence is not None:
        attributes = model.endpoint_attributes(
            paired,
            condition_latents,
            chunk_size=chunk_size,
        )
    else:
        attributes = None

    if target_doppler_distribution is not None:
        target_distribution = _flatten_distribution(
            _pair_point_attributes(
                target_doppler_distribution,
                source_to_target,
                point_count,
            ),
            point_count,
        )
        prediction_distribution = attributes["doppler_probability"].reshape(-1, 64)
        spectrum = distribution_cross_entropy(
            prediction_distribution,
            target_distribution.to(prediction_distribution),
        )
        bin_step = torch.diff(model.doppler_axis_mps).median().to(
            prediction_distribution
        )
        circular = circular_wasserstein1(
            prediction_distribution,
            target_distribution.to(prediction_distribution),
            bin_step,
        ).mean()
        total = total + doppler_weight * spectrum + circular_weight * circular
        components["doppler_cross_entropy"] = spectrum.detach()
        components["doppler_circular_w1"] = circular.detach()

    if target_confidence is not None:
        confidence_target = _flatten_confidence(
            _pair_point_attributes(
                target_confidence,
                source_to_target,
                point_count,
            ),
            point_count,
        ).to(attributes["confidence_logit"])
        confidence = F.binary_cross_entropy_with_logits(
            attributes["confidence_logit"].reshape(-1),
            confidence_target.clamp(0.0, 1.0),
        )
        total = total + confidence_weight * confidence
        components["confidence_bce"] = confidence.detach()

    components["total"] = total.detach()
    return PolarRectifiedFlowLoss(total=total, components=components)
