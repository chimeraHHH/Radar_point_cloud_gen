"""Frozen loss and query samplers for the R-A2 RaLD-WCE pilots."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


PILOT_MODES = (
    "tiny_memorization",
    "source_classwise",
    "range_class_sampler",
)
RANGE_BOUNDS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
SOURCE_POSITIVE_COUNT = 625
SOURCE_NEGATIVE_COUNT = 9_375
RANGE_POSITIVE_QUOTAS = (700, 200, 100)
RANGE_NEGATIVE_PER_CLASS = 5_000
RANGE_NEGATIVE_GLOBAL_PER_CLASS = 2_500
RANGE_NEGATIVE_SHELL_PER_CLASS = 2_500
AMBIGUITY_RADIUS_CELLS = 1
SURFACE_SHELL_RADIUS_CELLS = (2, 4)
FULL_CELL_JITTER = (-0.5, 0.5)


@dataclass(frozen=True)
class RaLDWCEPilotLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def _validate_spatial_shape(
    spatial_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    if len(spatial_shape) != 3 or any(int(value) <= 1 for value in spatial_shape):
        raise ValueError("R-A2 spatial shape must contain three sizes > 1")
    return tuple(int(value) for value in spatial_shape)


def _validate_target(
    target_rae_index: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    if target_rae_index.ndim != 2 or target_rae_index.shape[1] != 3:
        raise ValueError("R-A2 target RAE indices must have shape (N,3)")
    if target_rae_index.shape[0] <= 0:
        raise ValueError("R-A2 target RAE indices cannot be empty")
    target = target_rae_index.detach().to(device="cpu", dtype=torch.long)
    maximum = torch.tensor(spatial_shape, dtype=torch.long) - 1
    if bool(((target < 0) | (target > maximum)).any()):
        raise ValueError("R-A2 target RAE index is outside the Cube")
    return torch.unique(target, dim=0, sorted=True)


def _flat_indices(
    indices: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    return (
        indices[:, 0] * spatial_shape[1] * spatial_shape[2]
        + indices[:, 1] * spatial_shape[2]
        + indices[:, 2]
    )


def _unflat_indices(
    flat: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    range_index = flat // (spatial_shape[1] * spatial_shape[2])
    remainder = flat % (spatial_shape[1] * spatial_shape[2])
    azimuth_index = remainder // spatial_shape[2]
    elevation_index = remainder % spatial_shape[2]
    return torch.stack((range_index, azimuth_index, elevation_index), dim=-1)


def _range_codes_for_indices(
    indices: torch.Tensor,
    range_m: torch.Tensor,
) -> torch.Tensor:
    values = range_m.detach().to(device="cpu", dtype=torch.float32)[indices[:, 0]]
    codes = torch.full_like(values, -1, dtype=torch.long)
    for code, (lower, upper) in enumerate(RANGE_BOUNDS_M):
        codes[(values >= lower) & (values < upper)] = code
    if bool((codes < 0).any()):
        raise ValueError("R-A2 sampled a cell outside the frozen 0-120 m domain")
    return codes


def _sample_rows_with_replacement(
    rows: torch.Tensor,
    count: int,
    generator: torch.Generator,
    *,
    label: str,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("R-A2 sample count must be positive")
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.shape[0] <= 0:
        raise ValueError(f"R-A2 has no eligible {label} cells")
    selected = torch.randint(
        rows.shape[0],
        (count,),
        generator=generator,
    )
    return rows[selected]


def _ambiguity_flat_set(
    occupied: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> set[int]:
    maximum = torch.tensor(spatial_shape, dtype=torch.long) - 1
    excluded: set[int] = set()
    for delta_r in range(-AMBIGUITY_RADIUS_CELLS, AMBIGUITY_RADIUS_CELLS + 1):
        for delta_a in range(
            -AMBIGUITY_RADIUS_CELLS,
            AMBIGUITY_RADIUS_CELLS + 1,
        ):
            for delta_e in range(
                -AMBIGUITY_RADIUS_CELLS,
                AMBIGUITY_RADIUS_CELLS + 1,
            ):
                shifted = occupied + torch.tensor(
                    (delta_r, delta_a, delta_e),
                    dtype=torch.long,
                )
                valid = ((shifted >= 0) & (shifted <= maximum)).all(dim=1)
                excluded.update(
                    int(value)
                    for value in _flat_indices(shifted[valid], spatial_shape).tolist()
                )
    return excluded


def _sample_global_negative_cells(
    *,
    count: int,
    spatial_shape: tuple[int, int, int],
    excluded_flat: set[int],
    generator: torch.Generator,
    range_m: torch.Tensor | None = None,
    range_code: int | None = None,
) -> torch.Tensor:
    spatial_count = math.prod(spatial_shape)
    selected: list[int] = []
    attempts = 0
    maximum_attempts = max(50_000, count * 200)
    while len(selected) < count and attempts < maximum_attempts:
        draw_count = max(512, 2 * (count - len(selected)))
        candidates = torch.randint(
            spatial_count,
            (draw_count,),
            generator=generator,
        )
        candidate_indices = _unflat_indices(candidates, spatial_shape)
        if range_code is not None:
            if range_m is None:
                raise ValueError("R-A2 range-stratified negatives require range axis")
            keep = _range_codes_for_indices(candidate_indices, range_m) == range_code
            candidates = candidates[keep]
        for value in candidates.tolist():
            attempts += 1
            if int(value) not in excluded_flat:
                selected.append(int(value))
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError("R-A2 cannot fill the requested global negative quota")
    return _unflat_indices(torch.tensor(selected, dtype=torch.long), spatial_shape)


def _surface_shell_offsets() -> torch.Tensor:
    offsets = []
    lower, upper = SURFACE_SHELL_RADIUS_CELLS
    for delta_r in range(-upper, upper + 1):
        for delta_a in range(-upper, upper + 1):
            for delta_e in range(-upper, upper + 1):
                distance = max(abs(delta_r), abs(delta_a), abs(delta_e))
                if lower <= distance <= upper:
                    offsets.append((delta_r, delta_a, delta_e))
    return torch.tensor(offsets, dtype=torch.long)


def _sample_surface_shell_negative_cells(
    *,
    occupied: torch.Tensor,
    count: int,
    range_code: int,
    range_m: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    excluded_flat: set[int],
    generator: torch.Generator,
) -> torch.Tensor:
    offsets = _surface_shell_offsets()
    maximum = torch.tensor(spatial_shape, dtype=torch.long) - 1
    selected: list[torch.Tensor] = []
    attempts = 0
    maximum_attempts = max(100_000, count * 400)
    while len(selected) < count and attempts < maximum_attempts:
        draw_count = max(512, 2 * (count - len(selected)))
        source_rows = torch.randint(
            occupied.shape[0],
            (draw_count,),
            generator=generator,
        )
        offset_rows = torch.randint(
            offsets.shape[0],
            (draw_count,),
            generator=generator,
        )
        candidates = occupied[source_rows] + offsets[offset_rows]
        for candidate in candidates:
            attempts += 1
            if bool(((candidate < 0) | (candidate > maximum)).any()):
                continue
            if int(_range_codes_for_indices(candidate[None], range_m)[0]) != range_code:
                continue
            flat = int(_flat_indices(candidate[None], spatial_shape)[0])
            if flat in excluded_flat:
                continue
            selected.append(candidate.clone())
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError("R-A2 cannot fill the requested surface-shell quota")
    return torch.stack(selected)


def _finalize_queries(
    cell_indices: torch.Tensor,
    occupancy_target: torch.Tensor,
    range_class: torch.Tensor,
    sample_source: torch.Tensor,
    *,
    spatial_shape: tuple[int, int, int],
    generator: torch.Generator,
    mode: str,
    source_counts: dict[str, int],
) -> dict[str, torch.Tensor | int | str | dict[str, int | float | str]]:
    query_count = int(cell_indices.shape[0])
    if occupancy_target.shape != (query_count,):
        raise ValueError("R-A2 occupancy labels do not align with sampled cells")
    if range_class.shape != (query_count,) or sample_source.shape != (query_count,):
        raise ValueError("R-A2 sample metadata does not align with sampled cells")

    jitter = torch.rand((query_count, 3), generator=generator) - 0.5
    maximum = torch.tensor(spatial_shape, dtype=torch.float32) - 1.0
    query_bins = (cell_indices.float() + jitter).clamp(
        min=torch.zeros(3),
        max=maximum,
    )
    fractional_offset = query_bins - cell_indices.float()
    residual_target = torch.where(
        occupancy_target[:, None] > 0.5,
        -fractional_offset,
        torch.zeros_like(fractional_offset),
    )
    normalized = 2.0 * query_bins / maximum - 1.0
    order = torch.randperm(query_count, generator=generator)
    return {
        "normalized_rae": normalized[order].unsqueeze(0),
        "occupancy_target": occupancy_target[order].unsqueeze(0),
        "residual_target_bins": residual_target[order].unsqueeze(0),
        "range_class": range_class[order].unsqueeze(0),
        "sample_source": sample_source[order].unsqueeze(0),
        "cell_indices": cell_indices[order].unsqueeze(0),
        "fractional_offset": fractional_offset[order].unsqueeze(0),
        "positive_count": int((occupancy_target > 0.5).sum()),
        "negative_count": int((occupancy_target <= 0.5).sum()),
        "query_count": query_count,
        "mode": mode,
        "metadata": {
            "jitter_distribution_positive": "uniform_full_cell[-0.5,0.5)",
            "jitter_distribution_negative": "uniform_full_cell[-0.5,0.5)",
            "coordinate_phase_shortcut_allowed": False,
            "sampling_with_replacement": True,
            "ambiguity_radius_cells": (
                AMBIGUITY_RADIUS_CELLS
                if mode == "range_class_sampler"
                else 0
            ),
            **source_counts,
        },
    }


def sample_source_classwise_queries(
    target_rae_index: torch.Tensor,
    *,
    spatial_shape: tuple[int, int, int],
    range_m: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    """Sample RaLD-style 625/9375 classwise queries with shared full-cell jitter."""

    spatial_shape = _validate_spatial_shape(spatial_shape)
    if range_m.ndim != 1 or range_m.numel() != spatial_shape[0]:
        raise ValueError("R-A2 range axis does not match Cube range dimension")
    occupied = _validate_target(target_rae_index, spatial_shape)
    occupied_flat = set(
        int(value) for value in _flat_indices(occupied, spatial_shape).tolist()
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positive = _sample_rows_with_replacement(
        occupied,
        SOURCE_POSITIVE_COUNT,
        generator,
        label="positive",
    )
    negative = _sample_global_negative_cells(
        count=SOURCE_NEGATIVE_COUNT,
        spatial_shape=spatial_shape,
        excluded_flat=occupied_flat,
        generator=generator,
    )
    cells = torch.cat((positive, negative), dim=0)
    target = torch.cat(
        (
            torch.ones(SOURCE_POSITIVE_COUNT),
            torch.zeros(SOURCE_NEGATIVE_COUNT),
        )
    )
    range_class = _range_codes_for_indices(cells, range_m)
    sample_source = torch.cat(
        (
            torch.full((SOURCE_POSITIVE_COUNT,), 0, dtype=torch.long),
            torch.full((SOURCE_NEGATIVE_COUNT,), 1, dtype=torch.long),
        )
    )
    return _finalize_queries(
        cells,
        target,
        range_class,
        sample_source,
        spatial_shape=spatial_shape,
        generator=generator,
        mode="source_classwise",
        source_counts={
            "positive_query_count": SOURCE_POSITIVE_COUNT,
            "negative_global_query_count": SOURCE_NEGATIVE_COUNT,
            "negative_surface_shell_query_count": 0,
        },
    )


def sample_range_class_queries(
    target_rae_index: torch.Tensor,
    *,
    spatial_shape: tuple[int, int, int],
    range_m: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    """Sample frozen range/class quotas and exclude the 3x3x3 ambiguity band."""

    spatial_shape = _validate_spatial_shape(spatial_shape)
    if range_m.ndim != 1 or range_m.numel() != spatial_shape[0]:
        raise ValueError("R-A2 range axis does not match Cube range dimension")
    occupied = _validate_target(target_rae_index, spatial_shape)
    occupied_codes = _range_codes_for_indices(occupied, range_m)
    excluded = _ambiguity_flat_set(occupied, spatial_shape)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    all_cells: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_classes: list[torch.Tensor] = []
    all_sources: list[torch.Tensor] = []
    for range_code, positive_quota in enumerate(RANGE_POSITIVE_QUOTAS):
        positive_pool = occupied[occupied_codes == range_code]
        positive = _sample_rows_with_replacement(
            positive_pool,
            positive_quota,
            generator,
            label=f"range class {range_code} positive",
        )
        global_negative = _sample_global_negative_cells(
            count=RANGE_NEGATIVE_GLOBAL_PER_CLASS,
            spatial_shape=spatial_shape,
            excluded_flat=excluded,
            generator=generator,
            range_m=range_m,
            range_code=range_code,
        )
        shell_negative = _sample_surface_shell_negative_cells(
            occupied=positive_pool,
            count=RANGE_NEGATIVE_SHELL_PER_CLASS,
            range_code=range_code,
            range_m=range_m,
            spatial_shape=spatial_shape,
            excluded_flat=excluded,
            generator=generator,
        )
        cells = torch.cat((positive, global_negative, shell_negative), dim=0)
        all_cells.append(cells)
        all_targets.append(
            torch.cat(
                (
                    torch.ones(positive_quota),
                    torch.zeros(RANGE_NEGATIVE_PER_CLASS),
                )
            )
        )
        all_classes.append(
            torch.full((cells.shape[0],), range_code, dtype=torch.long)
        )
        all_sources.append(
            torch.cat(
                (
                    torch.full((positive_quota,), 0, dtype=torch.long),
                    torch.full(
                        (RANGE_NEGATIVE_GLOBAL_PER_CLASS,),
                        1,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (RANGE_NEGATIVE_SHELL_PER_CLASS,),
                        2,
                        dtype=torch.long,
                    ),
                )
            )
        )
    return _finalize_queries(
        torch.cat(all_cells),
        torch.cat(all_targets),
        torch.cat(all_classes),
        torch.cat(all_sources),
        spatial_shape=spatial_shape,
        generator=generator,
        mode="range_class_sampler",
        source_counts={
            "positive_query_count": sum(RANGE_POSITIVE_QUOTAS),
            "negative_global_query_count": (
                len(RANGE_BOUNDS_M) * RANGE_NEGATIVE_GLOBAL_PER_CLASS
            ),
            "negative_surface_shell_query_count": (
                len(RANGE_BOUNDS_M) * RANGE_NEGATIVE_SHELL_PER_CLASS
            ),
        },
    )


def sample_pilot_queries(
    mode: str,
    target_rae_index: torch.Tensor,
    *,
    spatial_shape: tuple[int, int, int],
    range_m: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    if mode not in PILOT_MODES:
        raise ValueError(f"Unknown R-A2 pilot mode: {mode}")
    if mode == "range_class_sampler":
        return sample_range_class_queries(
            target_rae_index,
            spatial_shape=spatial_shape,
            range_m=range_m,
            seed=seed,
        )
    sampled = sample_source_classwise_queries(
        target_rae_index,
        spatial_shape=spatial_shape,
        range_m=range_m,
        seed=seed,
    )
    sampled["mode"] = mode
    return sampled


def _validate_field_output(
    output: dict[str, torch.Tensor],
    expected_shape: tuple[int, int],
    normalized_rae: torch.Tensor,
    *,
    branch: str,
) -> None:
    if tuple(output.get("occupancy_logit", torch.empty(0)).shape) != expected_shape:
        raise ValueError(f"R-A2 {branch} occupancy logits do not align with queries")
    if tuple(output.get("confidence", torch.empty(0)).shape) != expected_shape:
        raise ValueError(f"R-A2 {branch} confidence does not align with queries")
    if tuple(output.get("residual_bins", torch.empty(0)).shape) != (
        *expected_shape,
        3,
    ):
        raise ValueError(f"R-A2 {branch} residual does not align with queries")
    if "query_normalized_rae" in output and not torch.equal(
        output["query_normalized_rae"],
        normalized_rae,
    ):
        raise ValueError(f"R-A2 {branch} branch changed the frozen query set")


def _source_classwise_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positive = target > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("R-A2 source-classwise loss requires both classes")
    positive_bce = F.binary_cross_entropy_with_logits(
        logits[positive].float(),
        target[positive].float(),
    )
    negative_bce = F.binary_cross_entropy_with_logits(
        logits[negative].float(),
        target[negative].float(),
    )
    return 0.1 * positive_bce + negative_bce, {
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
    }


def _range_class_normalized_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    range_class: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms: list[torch.Tensor] = []
    components: dict[str, torch.Tensor] = {}
    for code in range(len(RANGE_BOUNDS_M)):
        class_mask = range_class == code
        positive = class_mask & (target > 0.5)
        negative = class_mask & (target <= 0.5)
        if not bool(positive.any()) or not bool(negative.any()):
            raise ValueError(f"R-A2 range class {code} lacks a label class")
        positive_bce = F.binary_cross_entropy_with_logits(
            logits[positive].float(),
            target[positive].float(),
        )
        negative_bce = F.binary_cross_entropy_with_logits(
            logits[negative].float(),
            target[negative].float(),
        )
        components[f"range_{code}_positive_bce"] = positive_bce
        components[f"range_{code}_negative_bce"] = negative_bce
        terms.append(0.1 * positive_bce + negative_bce)
    return torch.stack(terms).mean(), components


def rald_wce_pilot_loss(
    matched: dict[str, torch.Tensor],
    wrong: dict[str, torch.Tensor],
    queries: dict[str, Any],
    *,
    mode: str,
    wrong_condition_margin: float = 0.05,
    wrong_condition_weight: float = 0.5,
    residual_weight: float = 0.05,
    brier_weight: float = 0.05,
) -> RaLDWCEPilotLoss:
    """Apply source-classwise or range-class-normalized occupancy supervision."""

    if mode not in PILOT_MODES:
        raise ValueError(f"Unknown R-A2 pilot mode: {mode}")
    normalized = queries["normalized_rae"]
    occupancy = queries["occupancy_target"]
    residual_target = queries["residual_target_bins"]
    range_class = queries["range_class"]
    if normalized.ndim != 3 or normalized.shape[-1] != 3:
        raise ValueError("R-A2 normalized queries must have shape (B,N,3)")
    expected_shape = tuple(normalized.shape[:2])
    if tuple(occupancy.shape) != expected_shape:
        raise ValueError("R-A2 occupancy target does not align with queries")
    if tuple(range_class.shape) != expected_shape:
        raise ValueError("R-A2 range classes do not align with queries")
    if tuple(residual_target.shape) != (*expected_shape, 3):
        raise ValueError("R-A2 residual target does not align with queries")
    if wrong_condition_margin <= 0.0:
        raise ValueError("R-A2 wrong-condition margin must be positive")
    if min(wrong_condition_weight, residual_weight, brier_weight) < 0.0:
        raise ValueError("R-A2 loss weights must be nonnegative")
    _validate_field_output(matched, expected_shape, normalized, branch="matched")
    _validate_field_output(wrong, expected_shape, normalized, branch="wrong")

    if mode == "range_class_sampler":
        matched_occupancy, components = _range_class_normalized_bce(
            matched["occupancy_logit"],
            occupancy,
            range_class,
        )
        wrong_occupancy, wrong_components = _range_class_normalized_bce(
            wrong["occupancy_logit"],
            occupancy,
            range_class,
        )
    else:
        matched_occupancy, components = _source_classwise_bce(
            matched["occupancy_logit"],
            occupancy,
        )
        wrong_occupancy, wrong_components = _source_classwise_bce(
            wrong["occupancy_logit"],
            occupancy,
        )
    components = {f"matched_{key}": value for key, value in components.items()}
    components.update(
        {f"wrong_{key}": value for key, value in wrong_components.items()}
    )

    wrong_margin = torch.relu(
        matched_occupancy.new_tensor(wrong_condition_margin)
        + matched_occupancy
        - wrong_occupancy
    )
    positive = occupancy > 0.5
    residual = F.smooth_l1_loss(
        matched["residual_bins"][positive].float(),
        residual_target[positive].float(),
    )
    brier = (
        matched["confidence"].float() - occupancy.float()
    ).square().mean()
    total = (
        matched_occupancy
        + wrong_condition_weight * wrong_margin
        + residual_weight * residual
        + brier_weight * brier
    )
    components.update(
        {
            "matched_occupancy": matched_occupancy,
            "wrong_occupancy": wrong_occupancy,
            "wrong_condition_margin": wrong_margin,
            "positive_residual": residual,
            "confidence_brier": brier,
        }
    )
    return RaLDWCEPilotLoss(total=total, components=components)
