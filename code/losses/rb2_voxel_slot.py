"""Supervision and losses for the fixed R-B2 Cartesian voxel-slot model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from models.rb2_voxel_slot_model import (
    LATTICE_ORIGIN_XYZ_M,
    LATTICE_SHAPE_XYZ,
    SLOT_ANCHORS_XYZ_M,
    SLOT_RESIDUAL_LIMIT_XYZ_M,
    SLOTS_PER_VOXEL,
    VOXEL_SIZE_M,
    VoxelSlotPrediction,
    points_in_fov,
)


@dataclass(frozen=True)
class VoxelSlotTargets:
    occupancy: torch.Tensor
    slot_confidence: torch.Tensor
    local_offsets_xyz_m: torch.Tensor
    offset_mask: torch.Tensor
    offset_weight: torch.Tensor
    target_point_count: tuple[int, ...]
    target_occupied_voxel_count: tuple[int, ...]
    represented_target_occupied_voxel_count: tuple[int, ...]
    target_confidence_coverage: tuple[float, ...]


def _linear_ids(indices_xyz: torch.Tensor) -> torch.Tensor:
    _, ny, nz = LATTICE_SHAPE_XYZ
    return (indices_xyz[..., 0] * ny + indices_xyz[..., 1]) * nz + indices_xyz[..., 2]


def _target_list(
    target_xyz_confidence: torch.Tensor | Sequence[torch.Tensor],
    batch_size: int,
) -> list[torch.Tensor]:
    if isinstance(target_xyz_confidence, torch.Tensor):
        if target_xyz_confidence.ndim == 2:
            if batch_size != 1:
                raise ValueError("R-B2 unbatched targets require batch size one")
            result = [target_xyz_confidence]
        elif target_xyz_confidence.ndim == 3:
            if target_xyz_confidence.shape[0] != batch_size:
                raise ValueError("R-B2 target and candidate batch sizes differ")
            result = [target_xyz_confidence[index] for index in range(batch_size)]
        else:
            raise ValueError("R-B2 targets must have shape (T,4) or (B,T,4)")
    else:
        result = list(target_xyz_confidence)
        if len(result) != batch_size:
            raise ValueError("R-B2 target sequence and candidate batch sizes differ")
    if any(value.ndim != 2 or value.shape[1] < 4 for value in result):
        raise ValueError("R-B2 each target must contain XYZ and confidence")
    return result


def build_voxel_slot_targets(
    candidate_indices_xyz: torch.Tensor,
    target_xyz_confidence: torch.Tensor | Sequence[torch.Tensor],
) -> VoxelSlotTargets:
    """Assign target points to immutable candidate voxels and four XY slots."""

    if candidate_indices_xyz.ndim != 3 or candidate_indices_xyz.shape[-1] != 3:
        raise ValueError("R-B2 candidates must have shape (B,N,3)")
    if candidate_indices_xyz.dtype not in (torch.int32, torch.int64):
        raise ValueError("R-B2 candidate indices must be integer")
    targets = _target_list(target_xyz_confidence, candidate_indices_xyz.shape[0])
    device = candidate_indices_xyz.device
    dtype = targets[0].dtype
    batch_size, candidate_count, _ = candidate_indices_xyz.shape
    occupancy = torch.zeros((batch_size, candidate_count), device=device, dtype=dtype)
    slot_confidence = torch.zeros(
        (batch_size, candidate_count, SLOTS_PER_VOXEL),
        device=device,
        dtype=dtype,
    )
    local_offsets = torch.zeros(
        (batch_size, candidate_count, SLOTS_PER_VOXEL, 3),
        device=device,
        dtype=dtype,
    )
    offset_mask = torch.zeros_like(slot_confidence, dtype=torch.bool)
    offset_weight = torch.zeros_like(slot_confidence)
    origin = torch.tensor(LATTICE_ORIGIN_XYZ_M, device=device, dtype=dtype)
    anchors = torch.tensor(SLOT_ANCHORS_XYZ_M, device=device, dtype=dtype)
    limits = torch.tensor(SLOT_RESIDUAL_LIMIT_XYZ_M, device=device, dtype=dtype)

    point_counts: list[int] = []
    occupied_counts: list[int] = []
    represented_counts: list[int] = []
    confidence_coverages: list[float] = []
    for batch, target in enumerate(targets):
        target = target.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(target[:, :4]).all()):
            raise ValueError("R-B2 target contains non-finite XYZ/confidence")
        xyz = target[:, :3]
        confidence = target[:, 3].clamp_min(0.0)
        target_indices = torch.floor((xyz - origin[None, :]) / VOXEL_SIZE_M).long()
        maximum = torch.tensor(
            LATTICE_SHAPE_XYZ,
            device=device,
            dtype=torch.long,
        )
        valid = ((target_indices >= 0) & (target_indices < maximum[None, :])).all(
            dim=1
        )
        if not bool(valid.all()) or not bool(points_in_fov(xyz).all()):
            raise ValueError("R-B2 target contains a point outside the frozen FOV")
        candidate_ids = _linear_ids(candidate_indices_xyz[batch])
        if torch.unique(candidate_ids).numel() != candidate_ids.numel():
            raise ValueError("R-B2 candidate bank contains duplicate voxel IDs")
        candidate_lookup = {
            int(value): index
            for index, value in enumerate(candidate_ids.detach().cpu().tolist())
        }
        target_ids = _linear_ids(target_indices)
        occupied_ids = {int(value) for value in target_ids.detach().cpu().tolist()}
        represented_ids = occupied_ids.intersection(candidate_lookup)
        represented_point_mask = torch.tensor(
            [
                int(value) in candidate_lookup
                for value in target_ids.detach().cpu().tolist()
            ],
            device=device,
            dtype=torch.bool,
        )
        confidence_total = float(confidence.sum().item())
        confidence_covered = float(confidence[represented_point_mask].sum().item())

        # Highest-confidence point wins when multiple targets occupy one slot.
        order = torch.argsort(confidence, descending=True, stable=True)
        assigned: set[tuple[int, int]] = set()
        for target_position in order.detach().cpu().tolist():
            target_id = int(target_ids[target_position].item())
            candidate_position = candidate_lookup.get(target_id)
            if candidate_position is None:
                continue
            local = (
                xyz[target_position]
                - origin
                - (
                    candidate_indices_xyz[batch, candidate_position].to(dtype)
                    + 0.5
                )
                * VOXEL_SIZE_M
            )
            slot = int(local[0] >= 0.0) * 2 + int(local[1] >= 0.0)
            occupancy[batch, candidate_position] = 1.0
            key = (candidate_position, slot)
            if key in assigned:
                continue
            assigned.add(key)
            slot_confidence[batch, candidate_position, slot] = 1.0
            offset_mask[batch, candidate_position, slot] = True
            offset_weight[batch, candidate_position, slot] = confidence[
                target_position
            ].clamp_min(1e-3)
            representable = anchors[slot] + (
                local - anchors[slot]
            ).clamp(min=-limits, max=limits)
            local_offsets[batch, candidate_position, slot] = representable

        point_counts.append(int(xyz.shape[0]))
        occupied_counts.append(len(occupied_ids))
        represented_counts.append(len(represented_ids))
        confidence_coverages.append(
            0.0 if confidence_total <= 0.0 else confidence_covered / confidence_total
        )

    return VoxelSlotTargets(
        occupancy=occupancy,
        slot_confidence=slot_confidence,
        local_offsets_xyz_m=local_offsets,
        offset_mask=offset_mask,
        offset_weight=offset_weight,
        target_point_count=tuple(point_counts),
        target_occupied_voxel_count=tuple(occupied_counts),
        represented_target_occupied_voxel_count=tuple(represented_counts),
        target_confidence_coverage=tuple(confidence_coverages),
    )


def _balanced_focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    positive = target.sum()
    negative = target.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=positive_weight,
    )
    probability = torch.sigmoid(logits)
    correct_probability = probability * target + (1.0 - probability) * (1.0 - target)
    return ((1.0 - correct_probability).pow(gamma) * bce).mean()


def rb2_voxel_slot_loss(
    prediction: VoxelSlotPrediction,
    targets: VoxelSlotTargets,
    *,
    occupancy_weight: float = 1.0,
    slot_confidence_weight: float = 0.5,
    offset_weight: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize candidate ranking and bounded local geometry."""

    if prediction.occupancy_logits.shape != targets.occupancy.shape:
        raise ValueError("R-B2 occupancy prediction/target shapes differ")
    if prediction.slot_confidence_logits.shape != targets.slot_confidence.shape:
        raise ValueError("R-B2 slot-confidence prediction/target shapes differ")
    if prediction.local_offsets_xyz_m.shape != targets.local_offsets_xyz_m.shape:
        raise ValueError("R-B2 offset prediction/target shapes differ")
    occupancy = _balanced_focal_bce(
        prediction.occupancy_logits,
        targets.occupancy,
    )
    slot_confidence = _balanced_focal_bce(
        prediction.slot_confidence_logits,
        targets.slot_confidence,
    )
    if bool(targets.offset_mask.any()):
        pointwise_offset = F.smooth_l1_loss(
            prediction.local_offsets_xyz_m,
            targets.local_offsets_xyz_m,
            beta=0.02,
            reduction="none",
        ).mean(dim=-1)
        weights = targets.offset_weight * targets.offset_mask
        offset = (pointwise_offset * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        offset = prediction.local_offsets_xyz_m.sum() * 0.0
    total = (
        occupancy_weight * occupancy
        + slot_confidence_weight * slot_confidence
        + offset_weight * offset
    )
    return total, {
        "total": total.detach(),
        "occupancy": occupancy.detach(),
        "slot_confidence": slot_confidence.detach(),
        "local_offset": offset.detach(),
        "positive_voxel_fraction": targets.occupancy.mean().detach(),
        "positive_slot_fraction": targets.slot_confidence.mean().detach(),
    }


def candidate_support_report(targets: VoxelSlotTargets) -> dict[str, object]:
    rows = []
    for batch in range(len(targets.target_point_count)):
        denominator = targets.target_occupied_voxel_count[batch]
        rows.append(
            {
                "target_point_count": targets.target_point_count[batch],
                "target_occupied_voxel_count": denominator,
                "represented_target_occupied_voxel_count": (
                    targets.represented_target_occupied_voxel_count[batch]
                ),
                "target_occupied_voxel_recall": (
                    0.0
                    if denominator == 0
                    else targets.represented_target_occupied_voxel_count[batch]
                    / denominator
                ),
                "target_confidence_coverage": targets.target_confidence_coverage[
                    batch
                ],
            }
        )
    return {
        "frames": rows,
        "minimum_target_occupied_voxel_recall": min(
            float(row["target_occupied_voxel_recall"]) for row in rows
        ),
        "minimum_target_confidence_coverage": min(
            float(row["target_confidence_coverage"]) for row in rows
        ),
    }
