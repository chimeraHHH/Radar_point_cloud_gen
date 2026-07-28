"""Deterministic wide-query inference and exact export for R-A1 Stage 0.

The candidate domain is GT-free. Q0 is a fixed stratified Sobol domain and Q1
is generated only from matched-condition occupancy scores. Matched and wrong
conditions are then decoded on the identical Q0+Q1 coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch

from models.cube_cycle import continuous_rae_to_xyz
from models.rald_wce_field import RaLDWCEField


RANGE_STRATA_M = (
    ("range_0_30m", 0.0, 30.0),
    ("range_30_60m", 30.0, 60.0),
    ("range_60_120m", 60.0, 120.0),
)
FORMAL_Q0_COUNT = 500_000
FORMAL_Q0_RANGE_QUOTAS = (166_667, 166_667, 166_666)
FORMAL_Q1_ANCHOR_QUOTAS = (20_000, 4_250, 750)
FORMAL_Q1_SAMPLES_PER_ANCHOR = 8
FORMAL_OUTPUT_RANGE_QUOTAS = (8_000, 1_700, 300)
EXPORT_COUNT = 10_000
CAPACITY_DISTANCE_M = 0.05


class ExactExportCapacityError(RuntimeError):
    """Raised when a no-copy exact export cannot satisfy frozen quotas."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class WideInferenceConfig:
    q0_range_quotas: tuple[int, int, int] = FORMAL_Q0_RANGE_QUOTAS
    q1_anchor_quotas: tuple[int, int, int] = FORMAL_Q1_ANCHOR_QUOTAS
    q1_samples_per_anchor: int = FORMAL_Q1_SAMPLES_PER_ANCHOR
    output_range_quotas: tuple[int, int, int] = FORMAL_OUTPUT_RANGE_QUOTAS
    minimum_distance_m: float = CAPACITY_DISTANCE_M
    seed: int = 20260716
    decode_chunk_size: int = 8_192

    def validate(self) -> None:
        for quotas, name in (
            (self.q0_range_quotas, "Q0"),
            (self.q1_anchor_quotas, "Q1 anchors"),
            (self.output_range_quotas, "output"),
        ):
            if len(quotas) != len(RANGE_STRATA_M) or any(
                int(value) <= 0 for value in quotas
            ):
                raise ValueError(f"RaLD-WCE {name} quotas must be three positives")
        if sum(self.output_range_quotas) != EXPORT_COUNT:
            raise ValueError("RaLD-WCE output quotas must sum to exact 10k")
        if sum(self.q0_range_quotas) < EXPORT_COUNT:
            raise ValueError("RaLD-WCE Q0 capacity is below exact 10k")
        if self.q1_samples_per_anchor <= 0:
            raise ValueError("RaLD-WCE Q1 samples per anchor must be positive")
        if self.minimum_distance_m <= 0.0:
            raise ValueError("RaLD-WCE capacity distance must be positive")
        if self.decode_chunk_size <= 0:
            raise ValueError("RaLD-WCE decode chunk size must be positive")

    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "q0_query_count": sum(self.q0_range_quotas),
            "q0_range_quotas": _quota_dict(self.q0_range_quotas),
            "q0_source": "fixed_scrambled_sobol_physical_range_stratified",
            "q1_anchor_count": sum(self.q1_anchor_quotas),
            "q1_anchor_quotas": _quota_dict(self.q1_anchor_quotas),
            "q1_samples_per_anchor": self.q1_samples_per_anchor,
            "q1_query_count": (
                sum(self.q1_anchor_quotas) * self.q1_samples_per_anchor
            ),
            "q1_source": "matched_occupancy_dependent_local_refinement",
            "same_query_wrong_condition": True,
            "output_range_quotas": _quota_dict(self.output_range_quotas),
            "export_count": EXPORT_COUNT,
            "minimum_euclidean_distance_m": self.minimum_distance_m,
            "copy_padding_jitter_duplicate": False,
            "doppler_head": False,
        }


@dataclass(frozen=True)
class ExactWCEExport:
    xyz_m: torch.Tensor
    confidence: torch.Tensor
    selected_candidate_rows: torch.Tensor
    report: dict[str, Any]
    hashes: dict[str, str]


@dataclass(frozen=True)
class WCEInferenceResult:
    matched: ExactWCEExport
    wrong_condition: ExactWCEExport
    report: dict[str, Any]


def _quota_dict(quotas: tuple[int, int, int] | list[int]) -> dict[str, int]:
    return {
        label: int(value)
        for (label, _, _), value in zip(RANGE_STRATA_M, quotas, strict=True)
    }


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_axis(axis: torch.Tensor, name: str) -> None:
    if axis.ndim != 1 or axis.numel() < 2:
        raise ValueError(f"RaLD-WCE {name} axis must contain at least two values")
    if not torch.is_floating_point(axis) or not bool(torch.isfinite(axis).all()):
        raise ValueError(f"RaLD-WCE {name} axis must be finite floating point")
    if not bool((axis[1:] > axis[:-1]).all()):
        raise ValueError(f"RaLD-WCE {name} axis must be strictly increasing")


def _physical_to_fractional_index(
    axis: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    right = torch.searchsorted(axis, values).clamp(1, axis.numel() - 1)
    left = right - 1
    fraction = (values - axis[left]) / (axis[right] - axis[left]).clamp_min(
        torch.finfo(values.dtype).eps
    )
    return left.to(values) + fraction


def _normalize_bins(
    coordinates_rae: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    maximum = coordinates_rae.new_tensor(
        [size - 1 for size in spatial_shape]
    )
    return 2.0 * coordinates_rae / maximum - 1.0


def _denormalize_bins(
    normalized_rae: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    maximum = normalized_rae.new_tensor(
        [size - 1 for size in spatial_shape]
    )
    return (normalized_rae + 1.0) * 0.5 * maximum


def range_stratum_codes(xyz_m: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(xyz_m, dim=1)
    codes = torch.full(
        (xyz_m.shape[0],),
        -1,
        dtype=torch.long,
        device=xyz_m.device,
    )
    for index, (_, lower, upper) in enumerate(RANGE_STRATA_M):
        codes[(radius >= lower) & (radius < upper)] = index
    return codes


def fixed_wide_q0(
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    quotas: tuple[int, int, int],
    seed: int,
) -> torch.Tensor:
    """Build the frame-invariant Q0 domain in normalized continuous RAE."""

    for axis, name in (
        (range_m, "range"),
        (azimuth_rad, "azimuth"),
        (elevation_rad, "elevation"),
    ):
        _validate_axis(axis, name)
    if len(quotas) != len(RANGE_STRATA_M) or any(value <= 0 for value in quotas):
        raise ValueError("RaLD-WCE Q0 quotas must be three positives")
    coordinates: list[torch.Tensor] = []
    for stratum, ((_, lower, upper), count) in enumerate(
        zip(RANGE_STRATA_M, quotas, strict=True)
    ):
        physical_lower = max(lower, float(range_m[0].item()))
        physical_upper = min(upper, float(range_m[-1].item()))
        if physical_upper <= physical_lower:
            raise ValueError("RaLD-WCE range axis misses a frozen Q0 stratum")
        unit = torch.quasirandom.SobolEngine(
            dimension=3,
            scramble=True,
            seed=seed + 101 * stratum,
        ).draw(count, dtype=torch.float32).to(range_m.device)
        physical_range = physical_lower + unit[:, 0] * (
            physical_upper - physical_lower
        )
        coordinates.append(
            torch.stack(
                (
                    _physical_to_fractional_index(range_m, physical_range),
                    unit[:, 1] * (azimuth_rad.numel() - 1),
                    unit[:, 2] * (elevation_rad.numel() - 1),
                ),
                dim=1,
            )
        )
    bins = torch.cat(coordinates, dim=0)
    return _normalize_bins(
        bins,
        (range_m.numel(), azimuth_rad.numel(), elevation_rad.numel()),
    ).unsqueeze(0)


def occupancy_dependent_q1(
    q0_output: dict[str, torch.Tensor],
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    anchor_quotas: tuple[int, int, int],
    samples_per_anchor: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Refine around matched-condition high-occupancy Q0 coordinates."""

    normalized = q0_output["refined_normalized_rae"]
    confidence = q0_output["confidence"]
    if normalized.ndim != 3 or normalized.shape[0] != 1:
        raise ValueError("RaLD-WCE Q1 currently requires batch size one")
    if confidence.shape != normalized.shape[:2]:
        raise ValueError("RaLD-WCE Q0 confidence and coordinates differ")
    spatial_shape = (
        range_m.numel(),
        azimuth_rad.numel(),
        elevation_rad.numel(),
    )
    bins = _denormalize_bins(normalized[0].float(), spatial_shape)
    xyz = continuous_rae_to_xyz(
        bins,
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    codes = range_stratum_codes(xyz)
    anchors: list[torch.Tensor] = []
    anchor_rows: list[torch.Tensor] = []
    for stratum, quota in enumerate(anchor_quotas):
        eligible = torch.nonzero(codes == stratum, as_tuple=False).flatten()
        if eligible.numel() < quota:
            raise ExactExportCapacityError(
                "RaLD-WCE Q0 cannot supply frozen Q1 anchor quota",
                {
                    "stage": "q1_anchor_selection",
                    "required_anchor_quotas": _quota_dict(anchor_quotas),
                    "available_q0_by_range": _quota_dict(
                        [
                            int((codes == index).sum().item())
                            for index in range(len(RANGE_STRATA_M))
                        ]
                    ),
                },
            )
        ordered = eligible[
            torch.argsort(
                confidence[0, eligible].float(),
                descending=True,
                stable=True,
            )
        ]
        selected = ordered[:quota]
        anchors.append(bins[selected])
        anchor_rows.append(selected)

    anchor_bins = torch.cat(anchors, dim=0)
    anchor_indices = torch.cat(anchor_rows, dim=0)
    offset_unit = torch.quasirandom.SobolEngine(
        dimension=3,
        scramble=True,
        seed=seed + 10_007,
    ).draw(
        anchor_bins.shape[0] * samples_per_anchor,
        dtype=torch.float32,
    ).to(anchor_bins.device)
    offsets = (offset_unit * 2.0 - 1.0) * anchor_bins.new_tensor(
        [2.0, 1.0, 1.0]
    )
    refined_bins = anchor_bins.repeat_interleave(
        samples_per_anchor,
        dim=0,
    ) + offsets
    maximum = refined_bins.new_tensor(
        [size - 1 for size in spatial_shape]
    )
    refined_bins = refined_bins.clamp_min(0.0).minimum(maximum)
    q1 = _normalize_bins(refined_bins, spatial_shape).unsqueeze(0)
    return q1, {
        "anchor_count": int(anchor_bins.shape[0]),
        "anchor_quotas": _quota_dict(anchor_quotas),
        "samples_per_anchor": samples_per_anchor,
        "query_count": int(q1.shape[1]),
        "anchor_q0_row_sha256": tensor_sha256(anchor_indices),
        "query_sha256": tensor_sha256(q1),
        "source": "matched_condition_q0_occupancy_only",
    }


def _candidate_tensors(
    outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = torch.cat(
        [output["refined_normalized_rae"] for output in outputs],
        dim=1,
    )[0].float()
    confidence = torch.cat(
        [output["confidence"] for output in outputs],
        dim=1,
    )[0].float()
    bins = _denormalize_bins(
        normalized,
        (range_m.numel(), azimuth_rad.numel(), elevation_rad.numel()),
    )
    xyz = continuous_rae_to_xyz(
        bins,
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    if not bool(torch.isfinite(xyz).all()) or not bool(
        torch.isfinite(confidence).all()
    ):
        raise ValueError("RaLD-WCE candidate field contains non-finite values")
    return xyz, confidence


def exact_capacity_export(
    candidate_xyz_m: torch.Tensor,
    candidate_confidence: torch.Tensor,
    *,
    output_quotas: tuple[int, int, int],
    minimum_distance_m: float,
) -> ExactWCEExport:
    """Select exact 10k by score with true 5 cm capacity and no fallback."""

    candidate_count = candidate_xyz_m.shape[0]
    if candidate_xyz_m.shape != (candidate_count, 3):
        raise ValueError("RaLD-WCE candidate XYZ must have shape (N,3)")
    if candidate_confidence.shape != (candidate_count,):
        raise ValueError("RaLD-WCE candidate confidence must match XYZ")
    if sum(output_quotas) != EXPORT_COUNT:
        raise ValueError("RaLD-WCE output quotas must sum to exact 10k")
    if minimum_distance_m <= 0.0:
        raise ValueError("RaLD-WCE minimum distance must be positive")

    xyz_cpu = candidate_xyz_m.detach().float().cpu()
    confidence_cpu = candidate_confidence.detach().float().cpu()
    codes_cpu = range_stratum_codes(xyz_cpu)
    candidate_ids = np.arange(candidate_count, dtype=np.int64)
    confidence_np = confidence_cpu.numpy()
    selected_rows: list[int] = []
    selected_cells: dict[
        tuple[int, int, int],
        list[tuple[float, float, float]],
    ] = {}
    selected_by_range = [0, 0, 0]
    rejected_distance = [0, 0, 0]
    observed_minimum_squared = float("inf")
    candidate_capacity = [
        int((codes_cpu == index).sum().item())
        for index in range(len(RANGE_STRATA_M))
    ]
    minimum_squared = minimum_distance_m**2

    def accept(row: int, stratum: int) -> bool:
        nonlocal observed_minimum_squared
        point = tuple(float(value) for value in xyz_cpu[row].tolist())
        cell = tuple(math.floor(value / minimum_distance_m) for value in point)
        candidate_minimum_squared = float("inf")
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    neighbours = selected_cells.get(
                        (
                            cell[0] + delta_x,
                            cell[1] + delta_y,
                            cell[2] + delta_z,
                        )
                    )
                    if neighbours is None:
                        continue
                    for neighbour in neighbours:
                        distance_squared = sum(
                            (left - right) ** 2
                            for left, right in zip(point, neighbour, strict=True)
                        )
                        candidate_minimum_squared = min(
                            candidate_minimum_squared,
                            distance_squared,
                        )
                        if distance_squared < minimum_squared:
                            rejected_distance[stratum] += 1
                            return False
        observed_minimum_squared = min(
            observed_minimum_squared,
            candidate_minimum_squared,
        )
        selected_cells.setdefault(cell, []).append(point)
        selected_rows.append(row)
        selected_by_range[stratum] += 1
        return True

    for stratum, quota in enumerate(output_quotas):
        eligible = torch.nonzero(
            codes_cpu == stratum,
            as_tuple=False,
        ).flatten().numpy()
        order = np.lexsort((candidate_ids[eligible], -confidence_np[eligible]))
        for row in eligible[order]:
            accept(int(row), stratum)
            if selected_by_range[stratum] == quota:
                break
        if selected_by_range[stratum] != quota:
            report = {
                "stage": "exact_10000_selection",
                "candidate_count": candidate_count,
                "candidate_capacity_by_range": _quota_dict(candidate_capacity),
                "required_output_quotas": _quota_dict(output_quotas),
                "selected_by_range": _quota_dict(selected_by_range),
                "rejected_minimum_distance_by_range": _quota_dict(
                    rejected_distance
                ),
                "minimum_euclidean_distance_m": minimum_distance_m,
                "copy_padding_jitter_duplicate": False,
            }
            raise ExactExportCapacityError(
                "RaLD-WCE true 5 cm support cannot fill frozen exact-10k quotas",
                report,
            )

    selected = torch.tensor(selected_rows, dtype=torch.long)
    xyz = xyz_cpu[selected]
    confidence = confidence_cpu[selected]
    minimum_observed = (
        math.sqrt(observed_minimum_squared)
        if math.isfinite(observed_minimum_squared)
        else minimum_distance_m
    )
    if xyz.shape != (EXPORT_COUNT, 3):
        raise AssertionError("RaLD-WCE export changed exact cardinality")
    if minimum_observed < minimum_distance_m - 1e-6:
        raise AssertionError("RaLD-WCE export violated minimum point distance")
    report = {
        "method": "confidence_stable_true_5cm_capacity_one_range_quota",
        "candidate_count": candidate_count,
        "candidate_capacity_by_range": _quota_dict(candidate_capacity),
        "output_quotas": _quota_dict(output_quotas),
        "selected_by_range": _quota_dict(selected_by_range),
        "rejected_minimum_distance_by_range": _quota_dict(rejected_distance),
        "minimum_euclidean_distance_m": minimum_distance_m,
        "observed_minimum_pair_distance_m": minimum_observed,
        "exact_point_count": int(xyz.shape[0]),
        "finite_xyz": bool(torch.isfinite(xyz).all()),
        "finite_confidence": bool(torch.isfinite(confidence).all()),
        "unique_selected_candidate_count": int(torch.unique(selected).numel()),
        "copy_padding_jitter_duplicate": False,
        "ground_truth_accessed": False,
    }
    return ExactWCEExport(
        xyz_m=xyz,
        confidence=confidence,
        selected_candidate_rows=selected,
        report=report,
        hashes={
            "xyz_sha256": tensor_sha256(xyz),
            "confidence_sha256": tensor_sha256(confidence),
            "selected_candidate_rows_sha256": tensor_sha256(selected),
        },
    )


@torch.no_grad()
def infer_exact_wce(
    model: RaLDWCEField,
    matched_cube_drae: torch.Tensor,
    wrong_cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    config: WideInferenceConfig,
) -> WCEInferenceResult:
    """Run matched and wrong conditions on an identical two-stage query set."""

    config.validate()
    if matched_cube_drae.shape != wrong_cube_drae.shape:
        raise ValueError("RaLD-WCE matched and wrong Cubes must share shape")
    q0 = fixed_wide_q0(
        range_m,
        azimuth_rad,
        elevation_rad,
        quotas=config.q0_range_quotas,
        seed=config.seed,
    ).to(matched_cube_drae.device)
    matched_condition = model.encode_condition(matched_cube_drae)[
        "condition_latents"
    ]
    wrong_condition = model.encode_condition(wrong_cube_drae)[
        "condition_latents"
    ]
    matched_q0 = model.decode_queries(
        q0,
        matched_condition,
        chunk_size=config.decode_chunk_size,
    )
    wrong_q0 = model.decode_queries(
        q0,
        wrong_condition,
        chunk_size=config.decode_chunk_size,
    )
    q1, q1_report = occupancy_dependent_q1(
        matched_q0,
        range_m,
        azimuth_rad,
        elevation_rad,
        anchor_quotas=config.q1_anchor_quotas,
        samples_per_anchor=config.q1_samples_per_anchor,
        seed=config.seed,
    )
    matched_q1 = model.decode_queries(
        q1,
        matched_condition,
        chunk_size=config.decode_chunk_size,
    )
    wrong_q1 = model.decode_queries(
        q1,
        wrong_condition,
        chunk_size=config.decode_chunk_size,
    )
    matched_xyz, matched_confidence = _candidate_tensors(
        (matched_q0, matched_q1),
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    wrong_xyz, wrong_confidence = _candidate_tensors(
        (wrong_q0, wrong_q1),
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    matched_export = exact_capacity_export(
        matched_xyz,
        matched_confidence,
        output_quotas=config.output_range_quotas,
        minimum_distance_m=config.minimum_distance_m,
    )
    wrong_export = exact_capacity_export(
        wrong_xyz,
        wrong_confidence,
        output_quotas=config.output_range_quotas,
        minimum_distance_m=config.minimum_distance_m,
    )
    query_hashes = {
        "q0_normalized_rae_sha256": tensor_sha256(q0),
        "q1_normalized_rae_sha256": tensor_sha256(q1),
    }
    return WCEInferenceResult(
        matched=matched_export,
        wrong_condition=wrong_export,
        report={
            "config": config.metadata(),
            "q1": q1_report,
            "query_hashes": query_hashes,
            "matched_wrong_query_hashes_identical": True,
            "q1_derived_from": "matched_q0_occupancy_only",
            "wrong_condition_used_for_query_construction": False,
            "candidate_count": int(matched_xyz.shape[0]),
            "condition_exclusive_coordinate_decoder": True,
            "ground_truth_accessed_for_queries_or_selection": False,
            "stage0_output_fields": ["x_m", "y_m", "z_m", "confidence"],
            "doppler_head_locked_post_stage0": True,
        },
    )
