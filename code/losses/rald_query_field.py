"""G1D occupancy-query sampling and query-field supervision.

The default objective is

    0.10 * positive-query BCE
  + 1.00 * empty-query BCE
  + 1.00 * geometry Chamfer
  + 0.25 * 2 m outlier hinge
  + 0.10 * confidence existence
  + 0.02 * normalized offset square
  + 0.02 * global nearest-neighbor repulsion.

This module intentionally depends only on tensors and dictionaries, so the
query-field protocol is independent of a particular model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


DEFAULT_QUERY_COUNT = 10_000
DEFAULT_POSITIVE_RATIO = 0.0625


@dataclass(frozen=True)
class OccupancyQueryBatch:
    coordinates_rae: torch.Tensor
    labels: torch.Tensor
    positive_count: int
    negative_count: int
    range_stratum: torch.Tensor


@dataclass(frozen=True)
class RaLDQueryFieldLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    distances: dict[str, torch.Tensor]


def _validate_generator_device(
    generator: torch.Generator | None,
    device: torch.device,
) -> None:
    if generator is None:
        return
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError(
            "Sampling generator and occupancy tensor must use the same device type"
        )


def _range_stratum(range_index: torch.Tensor, range_size: int) -> torch.Tensor:
    return torch.div(
        range_index * 3,
        range_size,
        rounding_mode="floor",
    ).clamp_max(2)


def _balanced_stratum_counts(
    total: int,
    capacities: list[int],
) -> list[int]:
    if len(capacities) != 3:
        raise ValueError("G1D range sampling requires exactly three strata")
    if any(capacity < 0 for capacity in capacities):
        raise ValueError("Range-stratum capacities cannot be negative")
    if sum(capacities) < total:
        raise ValueError(
            "Insufficient unambiguous negative cells: "
            f"need {total}, found {sum(capacities)} across range strata "
            f"{capacities}"
        )

    base, remainder = divmod(total, 3)
    desired = [base + int(index < remainder) for index in range(3)]
    counts = [min(desired[index], capacities[index]) for index in range(3)]
    remaining = total - sum(counts)
    while remaining:
        candidates = [
            index
            for index in range(3)
            if counts[index] < capacities[index]
        ]
        if not candidates:
            raise ValueError(
                "Unable to allocate the requested negatives across range strata"
            )
        index = min(candidates, key=lambda item: (counts[item], item))
        counts[index] += 1
        remaining -= 1
    return counts


def sample_occupancy_queries(
    target_rae_index: torch.Tensor,
    occupancy: torch.Tensor,
    count: int = DEFAULT_QUERY_COUNT,
    positive_ratio: float = DEFAULT_POSITIVE_RATIO,
    generator: torch.Generator | None = None,
) -> OccupancyQueryBatch:
    """Sample occupied and unambiguous free-space RAE queries.

    ``occupancy`` is a dense ``(range, azimuth, elevation)`` confidence grid.
    Positive cells are sampled with replacement according to their confidence
    and jittered uniformly inside the cell. Negative cells are sampled without
    replacement outside the 3x3x3 dilation of all occupied cells.
    """

    if not isinstance(target_rae_index, torch.Tensor):
        raise TypeError("target_rae_index must be a torch.Tensor")
    if not isinstance(occupancy, torch.Tensor):
        raise TypeError("occupancy must be a torch.Tensor")
    if occupancy.ndim != 3:
        raise ValueError(
            "occupancy must have shape (range, azimuth, elevation)"
        )
    if any(size <= 0 for size in occupancy.shape):
        raise ValueError("occupancy dimensions must all be non-empty")
    if occupancy.shape[0] < 3:
        raise ValueError("occupancy range axis must support three non-empty strata")
    if target_rae_index.ndim != 2 or target_rae_index.shape[1] != 3:
        raise ValueError("target_rae_index must have shape (N,3)")
    if target_rae_index.shape[0] == 0:
        raise ValueError("At least one occupied target RAE cell is required")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if not isinstance(positive_ratio, (int, float)) or not math.isfinite(
        float(positive_ratio)
    ):
        raise ValueError("positive_ratio must be finite")
    if not 0.0 < float(positive_ratio) < 1.0:
        raise ValueError("positive_ratio must lie strictly between zero and one")

    positive_exact = count * float(positive_ratio)
    positive_count = int(round(positive_exact))
    if not math.isclose(positive_exact, positive_count, abs_tol=1e-9):
        raise ValueError(
            "count * positive_ratio must be an integer; "
            f"got {positive_exact}"
        )
    negative_count = count - positive_count
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Sampling must include both positive and negative queries")

    if occupancy.is_floating_point():
        if not torch.isfinite(occupancy).all():
            raise ValueError("occupancy must contain only finite values")
        if (occupancy < 0).any():
            raise ValueError("occupancy/confidence values cannot be negative")
        coordinate_dtype = occupancy.dtype
    elif occupancy.dtype == torch.bool:
        coordinate_dtype = torch.float32
    else:
        raise TypeError("occupancy must be boolean or floating point")

    _validate_generator_device(generator, occupancy.device)
    target = target_rae_index.to(device=occupancy.device)
    if target.is_floating_point():
        if not torch.isfinite(target).all():
            raise ValueError("target_rae_index must contain only finite values")
        if not torch.equal(target, target.round()):
            raise ValueError("target_rae_index must contain integer-valued cells")
    elif target.dtype == torch.bool:
        raise TypeError("target_rae_index cannot be boolean")
    target = target.to(torch.long)

    shape = tuple(int(size) for size in occupancy.shape)
    lower_ok = target >= 0
    upper = target.new_tensor(shape)
    if not bool((lower_ok & (target < upper)).all()):
        raise ValueError("target_rae_index contains an out-of-bounds cell")

    target_flat = (
        (target[:, 0] * shape[1] + target[:, 1]) * shape[2]
        + target[:, 2]
    )
    target_flat = torch.unique(target_flat, sorted=True)
    target = torch.stack(
        (
            torch.div(
                target_flat,
                shape[1] * shape[2],
                rounding_mode="floor",
            ),
            torch.div(target_flat, shape[2], rounding_mode="floor")
            % shape[1],
            target_flat % shape[2],
        ),
        dim=1,
    )
    positive_weights = occupancy[
        target[:, 0],
        target[:, 1],
        target[:, 2],
    ].to(torch.float32)
    if not bool((positive_weights > 0).all()):
        raise ValueError(
            "Every target_rae_index cell must have positive occupancy/confidence"
        )

    sampled_positive = torch.multinomial(
        positive_weights,
        positive_count,
        replacement=True,
        generator=generator,
    )
    positive_cells = target[sampled_positive]
    jitter = torch.rand(
        (positive_count, 3),
        dtype=coordinate_dtype,
        device=occupancy.device,
        generator=generator,
    ) - 0.5
    positive_coordinates = positive_cells.to(coordinate_dtype) + jitter

    occupied = occupancy > 0
    dilated = F.max_pool3d(
        occupied[None, None].to(torch.float32),
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0].to(torch.bool)
    negative_cells = (~dilated).nonzero(as_tuple=False)
    negative_strata = _range_stratum(negative_cells[:, 0], shape[0])
    stratum_candidates = [
        negative_cells[negative_strata == index] for index in range(3)
    ]
    stratum_counts = _balanced_stratum_counts(
        negative_count,
        [int(candidates.shape[0]) for candidates in stratum_candidates],
    )

    sampled_negative_parts = []
    sampled_negative_strata = []
    for stratum, (candidates, stratum_count) in enumerate(
        zip(stratum_candidates, stratum_counts)
    ):
        if stratum_count == 0:
            continue
        selection = torch.randperm(
            candidates.shape[0],
            device=occupancy.device,
            generator=generator,
        )[:stratum_count]
        sampled_negative_parts.append(candidates[selection])
        sampled_negative_strata.append(
            torch.full(
                (stratum_count,),
                stratum,
                dtype=torch.long,
                device=occupancy.device,
            )
        )
    sampled_negative = torch.cat(sampled_negative_parts, dim=0)
    sampled_negative_stratum = torch.cat(sampled_negative_strata, dim=0)
    negative_jitter = torch.rand(
        (negative_count, 3),
        dtype=coordinate_dtype,
        device=occupancy.device,
        generator=generator,
    ) - 0.5
    negative_coordinates = (
        sampled_negative.to(coordinate_dtype) + negative_jitter
    )

    positive_stratum = _range_stratum(positive_cells[:, 0], shape[0])
    coordinates = torch.cat(
        (positive_coordinates, negative_coordinates),
        dim=0,
    )
    labels = torch.cat(
        (
            torch.ones(
                positive_count,
                dtype=coordinate_dtype,
                device=occupancy.device,
            ),
            torch.zeros(
                negative_count,
                dtype=coordinate_dtype,
                device=occupancy.device,
            ),
        ),
        dim=0,
    )
    range_stratum = torch.cat(
        (positive_stratum, sampled_negative_stratum),
        dim=0,
    )
    permutation = torch.randperm(
        count,
        device=occupancy.device,
        generator=generator,
    )
    return OccupancyQueryBatch(
        coordinates_rae=coordinates[permutation],
        labels=labels[permutation],
        positive_count=positive_count,
        negative_count=negative_count,
        range_stratum=range_stratum[permutation],
    )


def _single_frame_tensor(
    value: torch.Tensor,
    *,
    name: str,
    trailing_dimensions: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == trailing_dimensions + 1:
        if value.shape[0] != 1:
            raise ValueError(f"{name} only supports a singleton batch dimension")
        value = value[0]
    if value.ndim != trailing_dimensions:
        raise ValueError(f"{name} has an invalid shape")
    return value


def _output_tensor(
    output: dict[str, torch.Tensor],
    names: tuple[str, ...],
) -> torch.Tensor:
    present = [name for name in names if name in output]
    if not present:
        raise KeyError(f"Missing output tensor; expected one of {names}")
    if len(present) > 1:
        raise ValueError(f"Output contains ambiguous aliases {present}")
    return output[present[0]]


def _nearest_assignment(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, source_xyz.shape[0], chunk_size):
        distance = torch.cdist(
            source_xyz[start : start + chunk_size],
            target_xyz,
        ).min(dim=1).values
        parts.append(distance)
    return torch.cat(parts)


def chunked_nearest_other_distance(
    xyz_m: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return each point's nearest distinct point over the complete set."""

    if not isinstance(xyz_m, torch.Tensor):
        raise TypeError("xyz_m must be a torch.Tensor")
    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
        raise ValueError("xyz_m must have shape (N,3)")
    if xyz_m.shape[0] < 2:
        raise ValueError("Global kNN repulsion requires at least two points")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be a positive integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not torch.isfinite(xyz_m).all():
        raise ValueError("xyz_m must contain only finite values")

    all_indices = torch.arange(xyz_m.shape[0], device=xyz_m.device)
    nearest_parts = []
    for start in range(0, xyz_m.shape[0], chunk_size):
        stop = min(start + chunk_size, xyz_m.shape[0])
        distance = torch.cdist(xyz_m[start:stop], xyz_m)
        self_mask = all_indices[None, :] == all_indices[start:stop, None]
        distance = distance.masked_fill(self_mask, torch.inf)
        nearest_parts.append(distance.min(dim=1).values)
    return torch.cat(nearest_parts)


def global_knn_repulsion(
    xyz_m: torch.Tensor,
    *,
    minimum_distance_m: float = 0.10,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Penalize collapse between any pair, including points from other seeds."""

    if not math.isfinite(float(minimum_distance_m)) or minimum_distance_m <= 0:
        raise ValueError("minimum_distance_m must be finite and positive")
    nearest = chunked_nearest_other_distance(xyz_m, chunk_size=chunk_size)
    return torch.relu(
        nearest.new_tensor(minimum_distance_m) - nearest
    ).square().mean()


def rald_query_field_loss(
    output: dict[str, torch.Tensor],
    query_labels: torch.Tensor,
    target_xyz_confidence: torch.Tensor,
    *,
    generated_point_count: int = DEFAULT_QUERY_COUNT,
    positive_weight: float = 0.1,
    negative_weight: float = 1.0,
    geometry_weight: float = 1.0,
    outlier_weight: float = 0.25,
    existence_weight: float = 0.10,
    offset_weight: float = 0.02,
    repulsion_weight: float = 0.02,
    outlier_threshold_m: float = 2.0,
    existence_radius_m: float = 1.0,
    repulsion_distance_m: float = 0.10,
    distance_chunk_size: int = 512,
) -> RaLDQueryFieldLoss:
    """Compute RaLD-style occupancy BCE and fixed-size generated-point losses."""

    if not isinstance(output, dict):
        raise TypeError("output must be a dictionary of tensors")
    if (
        isinstance(generated_point_count, bool)
        or not isinstance(generated_point_count, int)
        or generated_point_count < 2
    ):
        raise ValueError("generated_point_count must be an integer of at least two")
    scalar_values = {
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
        "geometry_weight": geometry_weight,
        "outlier_weight": outlier_weight,
        "existence_weight": existence_weight,
        "offset_weight": offset_weight,
        "repulsion_weight": repulsion_weight,
    }
    for name, value in scalar_values.items():
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name, value in {
        "outlier_threshold_m": outlier_threshold_m,
        "existence_radius_m": existence_radius_m,
        "repulsion_distance_m": repulsion_distance_m,
    }.items():
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        isinstance(distance_chunk_size, bool)
        or not isinstance(distance_chunk_size, int)
        or distance_chunk_size <= 0
    ):
        raise ValueError("distance_chunk_size must be a positive integer")

    query_logits = _single_frame_tensor(
        _output_tensor(output, ("query_logits",)),
        name="query_logits",
        trailing_dimensions=1,
    ).float()
    labels = _single_frame_tensor(
        query_labels,
        name="query_labels",
        trailing_dimensions=1,
    ).to(device=query_logits.device, dtype=query_logits.dtype)
    if query_logits.shape != labels.shape or query_logits.numel() == 0:
        raise ValueError("query_logits and query_labels must be aligned and non-empty")
    if not torch.isfinite(query_logits).all() or not torch.isfinite(labels).all():
        raise ValueError("query logits and labels must be finite")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("query_labels must be binary")
    positive_mask = labels == 1
    negative_mask = labels == 0
    if not bool(positive_mask.any()) or not bool(negative_mask.any()):
        raise ValueError("query_labels must contain both classes")

    generated_xyz = _single_frame_tensor(
        _output_tensor(output, ("generated_xyz_m", "xyz_m")),
        name="generated_xyz_m",
        trailing_dimensions=2,
    ).float()
    if generated_xyz.shape != (generated_point_count, 3):
        raise ValueError(
            "generated_xyz_m must have shape "
            f"({generated_point_count},3), got {tuple(generated_xyz.shape)}"
        )
    confidence_logit = _single_frame_tensor(
        _output_tensor(
            output,
            ("generated_confidence_logit", "confidence_logit"),
        ),
        name="generated_confidence_logit",
        trailing_dimensions=1,
    ).to(device=generated_xyz.device, dtype=generated_xyz.dtype)
    normalized_offset = _single_frame_tensor(
        _output_tensor(
            output,
            ("normalized_offset", "raw_offset_bins", "offset_bins"),
        ),
        name="normalized_offset",
        trailing_dimensions=2,
    ).to(device=generated_xyz.device, dtype=generated_xyz.dtype)
    if confidence_logit.shape != (generated_point_count,):
        raise ValueError(
            "generated_confidence_logit must align with generated points"
        )
    if normalized_offset.shape != (generated_point_count, 3):
        raise ValueError("normalized_offset must have shape (N,3)")
    for name, value in {
        "generated_xyz_m": generated_xyz,
        "generated_confidence_logit": confidence_logit,
        "normalized_offset": normalized_offset,
    }.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    target = _single_frame_tensor(
        target_xyz_confidence,
        name="target_xyz_confidence",
        trailing_dimensions=2,
    ).to(device=generated_xyz.device, dtype=generated_xyz.dtype)
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError("target_xyz_confidence must have non-empty shape (M,4)")
    if not torch.isfinite(target).all():
        raise ValueError("target_xyz_confidence must contain only finite values")
    target_weight = target[:, 3]
    if bool((target_weight < 0).any()) or not bool((target_weight > 0).any()):
        raise ValueError(
            "target confidence must be non-negative with positive total support"
        )

    positive_bce = F.binary_cross_entropy_with_logits(
        query_logits[positive_mask],
        labels[positive_mask],
    )
    negative_bce = F.binary_cross_entropy_with_logits(
        query_logits[negative_mask],
        labels[negative_mask],
    )
    occupancy_bce = (
        positive_weight * positive_bce
        + negative_weight * negative_bce
    )

    prediction_to_target = _nearest_assignment(
        generated_xyz,
        target[:, :3],
        chunk_size=distance_chunk_size,
    )
    target_to_prediction = _nearest_assignment(
        target[:, :3],
        generated_xyz,
        chunk_size=distance_chunk_size,
    )
    completeness = (
        target_to_prediction * target_weight
    ).sum() / target_weight.sum().clamp_min(1e-8)
    chamfer = prediction_to_target.mean() + completeness
    outlier_hinge = torch.relu(
        prediction_to_target
        - prediction_to_target.new_tensor(outlier_threshold_m)
    ).square().mean()

    existence_target = (
        prediction_to_target.detach() <= existence_radius_m
    ).to(confidence_logit)
    existence = F.binary_cross_entropy_with_logits(
        confidence_logit,
        existence_target,
    )
    offset_square = normalized_offset.square().mean()
    nearest_other = chunked_nearest_other_distance(
        generated_xyz,
        chunk_size=distance_chunk_size,
    )
    repulsion = torch.relu(
        nearest_other.new_tensor(repulsion_distance_m) - nearest_other
    ).square().mean()

    total = (
        occupancy_bce
        + geometry_weight * chamfer
        + outlier_weight * outlier_hinge
        + existence_weight * existence
        + offset_weight * offset_square
        + repulsion_weight * repulsion
    )
    components = {
        "occupancy_bce": occupancy_bce.detach(),
        "occupancy_positive_bce": positive_bce.detach(),
        "occupancy_negative_bce": negative_bce.detach(),
        "geometry_chamfer": chamfer.detach(),
        "outlier_hinge_2m": outlier_hinge.detach(),
        "confidence_existence": existence.detach(),
        "normalized_offset_square": offset_square.detach(),
        "global_knn_repulsion": repulsion.detach(),
        "confidence_mean": torch.sigmoid(confidence_logit).mean().detach(),
        "total": total.detach(),
    }
    distances = {
        "prediction_to_target_m": prediction_to_target,
        "target_to_prediction_m": target_to_prediction,
        "nearest_other_m": nearest_other,
        "existence_target": existence_target,
    }
    return RaLDQueryFieldLoss(
        total=total,
        components=components,
        distances=distances,
    )
