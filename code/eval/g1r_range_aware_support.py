"""Range-aware proposal support for the G1R-R0 diagnostic oracle.

The module changes only proposal support. Geometry selection remains the
unattainable G1F ground-truth oracle, and all geometry metrics remain in
``eval.dense_geometry``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from eval.g1f_candidate_support import (
    EXPORT_COUNT,
    PROPOSAL_COUNT,
    RANGE_BINS_M,
    CandidateSupportSelection,
    select_candidate_support_oracle,
)
from models.cube_cycle import continuous_rae_to_xyz
from models.rald_query_field import (
    RadarProposalSet,
    coarse_query_templates,
    integrated_log_energy,
    proposals_from_flat_index,
)


ARM_NAMES = ("vanilla", "z_only", "range_aware")
BASE_SEED_COUNT = 1_000
COARSE_TEMPLATE_COUNT = 32
VANILLA_NMS_KERNEL = (5, 5, 3)
SEED_QUOTAS = (750, 200, 50)
CANDIDATE_PARENT_QUOTAS = (24_000, 6_400, 1_600)
EXPORT_QUOTAS = (7_500, 2_000, 500)
LATERAL_NMS_RADIUS_M = 1.0
RADIAL_NMS_RADIUS_M = 1.0
COARSE_TEMPLATE_RADIAL_RADIUS_BINS = 2
CALIBRATION_EPSILON = 1e-6
ORACLE_ARTIFACT_LABEL = "g1r_r0_unattainable_gt_support_oracle"


@dataclass(frozen=True)
class RangeEnergyCalibration:
    """Exact train-only robust profile for every radial Cube cell."""

    median: torch.Tensor
    mad: torch.Tensor
    sample_count: torch.Tensor
    training_frame_count: int


@dataclass(frozen=True)
class ExpandedCandidatePool:
    """Frozen G1D 32-way expansion with explicit parent-range ownership."""

    xyz_m: torch.Tensor
    candidate_indices: torch.Tensor
    parent_range_bin: torch.Tensor
    proposal_flat_index: torch.Tensor


def _validate_energy_frame(energy_rae: torch.Tensor) -> None:
    if energy_rae.ndim != 3:
        raise ValueError("G1R range calibration expects one (R,A,E) energy map")
    if not torch.is_floating_point(energy_rae):
        raise TypeError("G1R range calibration requires floating-point energy")
    if not torch.isfinite(energy_rae).all():
        raise ValueError("G1R range calibration requires finite energy")


def fit_range_energy_calibration(
    energy_frames_rae: list[torch.Tensor],
    *,
    epsilon: float = CALIBRATION_EPSILON,
) -> RangeEnergyCalibration:
    """Fit exact per-range-cell median and MAD from training frames only.

    The calculation is deliberately exact rather than histogram-approximated.
    It materializes one radial row across frames at a time, bounding temporary
    memory by ``frames * azimuth * elevation`` values.
    """

    if not energy_frames_rae:
        raise ValueError("G1R calibration requires at least one training frame")
    if epsilon <= 0.0:
        raise ValueError("G1R calibration epsilon must be positive")
    reference_shape = energy_frames_rae[0].shape
    for energy in energy_frames_rae:
        _validate_energy_frame(energy)
        if energy.shape != reference_shape:
            raise ValueError("G1R calibration energy maps must share one shape")
        if energy.device.type != "cpu":
            raise ValueError("G1R calibration profiles must be fitted on CPU maps")

    medians = []
    mads = []
    sample_counts = []
    for range_index in range(reference_shape[0]):
        values = torch.cat(
            [
                energy[range_index].detach().float().reshape(-1)
                for energy in energy_frames_rae
            ]
        )
        median = values.median()
        mad = (values - median).abs().median()
        medians.append(median)
        mads.append(mad)
        sample_counts.append(values.numel())
    return RangeEnergyCalibration(
        median=torch.stack(medians),
        mad=torch.stack(mads).clamp_min(epsilon),
        sample_count=torch.tensor(sample_counts, dtype=torch.long),
        training_frame_count=len(energy_frames_rae),
    )


def calibrated_integrated_energy(
    energy_brae: torch.Tensor,
    calibration: RangeEnergyCalibration,
) -> torch.Tensor:
    """Apply the frozen per-range median/MAD profile to integrated energy."""

    if energy_brae.ndim != 4:
        raise ValueError("G1R calibrated energy expects shape (B,R,A,E)")
    if not torch.is_floating_point(energy_brae):
        raise TypeError("G1R calibrated energy must be floating point")
    if not torch.isfinite(energy_brae).all():
        raise ValueError("G1R calibrated energy must be finite")
    range_count = energy_brae.shape[1]
    if calibration.median.shape != (range_count,):
        raise ValueError("G1R calibration median does not match Cube range axis")
    if calibration.mad.shape != (range_count,):
        raise ValueError("G1R calibration MAD does not match Cube range axis")
    median = calibration.median.to(energy_brae)[None, :, None, None]
    mad = calibration.mad.to(energy_brae)[None, :, None, None]
    return (energy_brae - median) / mad


def calibrated_cube_energy(
    cube_drae: torch.Tensor,
    calibration: RangeEnergyCalibration,
) -> torch.Tensor:
    """Reuse G1D integrated log energy before applying robust calibration."""

    return calibrated_integrated_energy(
        integrated_log_energy(cube_drae),
        calibration,
    )


def _validate_nms_kernel(kernel: tuple[int, int, int]) -> None:
    if len(kernel) != 3 or any(
        size <= 0 or size % 2 == 0 for size in kernel
    ):
        raise ValueError("G1R NMS kernels must contain three positive odd sizes")


def stable_score_proposal_indices(
    score_brae: torch.Tensor,
    *,
    seed_count: int = BASE_SEED_COUNT,
    nms_kernel: tuple[int, int, int] = VANILLA_NMS_KERNEL,
) -> torch.Tensor:
    """Run G1D-compatible stable greedy NMS on an externally supplied score.

    This is required for the z-only attribution arm. Tests lock its flat-index
    output to ``stable_radar_proposals`` when supplied with G1D's own integrated
    energy, preventing a silent control-arm drift.
    """

    if score_brae.ndim != 4:
        raise ValueError("G1R proposal scores must have shape (B,R,A,E)")
    if not torch.is_floating_point(score_brae):
        raise TypeError("G1R proposal scores must be floating point")
    if not torch.isfinite(score_brae).all():
        raise ValueError("G1R proposal scores must be finite")
    if seed_count <= 0:
        raise ValueError("G1R proposal seed count must be positive")
    _validate_nms_kernel(nms_kernel)

    spatial_shape = tuple(int(size) for size in score_brae.shape[1:])
    spatial_count = int(np.prod(spatial_shape))
    if seed_count > spatial_count:
        raise ValueError("G1R proposal count exceeds Cube spatial cells")
    flat_score = score_brae.detach().flatten(start_dim=1)
    order = torch.argsort(flat_score, dim=1, descending=True, stable=True)
    range_count, azimuth_count, elevation_count = spatial_shape
    half_width = tuple(size // 2 for size in nms_kernel)
    selected_batches: list[list[int]] = []
    transfer_chunk = min(spatial_count, 65_536)
    for batch_index in range(flat_score.shape[0]):
        suppressed = np.zeros(spatial_count, dtype=np.bool_)
        selected: list[int] = []
        cursor = 0
        while cursor < spatial_count and len(selected) < seed_count:
            stop = min(cursor + transfer_chunk, spatial_count)
            ranked = order[batch_index, cursor:stop].detach().cpu().tolist()
            for raw_flat_index in ranked:
                flat_index = int(raw_flat_index)
                if suppressed[flat_index]:
                    continue
                selected.append(flat_index)
                radius = flat_index // (azimuth_count * elevation_count)
                remainder = flat_index % (azimuth_count * elevation_count)
                azimuth = remainder // elevation_count
                elevation = remainder % elevation_count
                radius_slice = slice(
                    max(0, radius - half_width[0]),
                    min(range_count, radius + half_width[0] + 1),
                )
                azimuth_slice = slice(
                    max(0, azimuth - half_width[1]),
                    min(azimuth_count, azimuth + half_width[1] + 1),
                )
                elevation_slice = slice(
                    max(0, elevation - half_width[2]),
                    min(elevation_count, elevation + half_width[2] + 1),
                )
                suppressed.reshape(spatial_shape)[
                    radius_slice,
                    azimuth_slice,
                    elevation_slice,
                ] = True
                if len(selected) == seed_count:
                    break
            cursor = stop
        if len(selected) != seed_count:
            raise RuntimeError("G1R fixed NMS cannot pack the requested seeds")
        selected_batches.append(selected)
    return torch.tensor(
        selected_batches,
        dtype=torch.long,
        device=score_brae.device,
    )


def _validate_axes(
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> None:
    axes = (range_m, azimuth_rad, elevation_rad)
    if any(axis.ndim != 1 for axis in axes):
        raise ValueError("G1R axes must be one-dimensional")
    if tuple(axis.numel() for axis in axes) != spatial_shape:
        raise ValueError("G1R axes do not match the proposal score shape")
    if any(not torch.isfinite(axis).all() for axis in axes):
        raise ValueError("G1R axes must be finite")
    if any(axis.numel() < 2 for axis in axes):
        raise ValueError("G1R axes must contain at least two values")


def physical_angular_suppression_mask(
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    range_index: int,
    azimuth_index: int,
    elevation_index: int,
    lateral_radius_m: float = LATERAL_NMS_RADIUS_M,
) -> torch.Tensor:
    """Approximate a fixed lateral-radius neighborhood on the angular grid.

    For a selected cell at range ``r`` the small-angle lateral displacement is
    ``r * sqrt((cos(e) * delta_az)^2 + delta_el^2)``. Consequently, the angular
    footprint shrinks with range instead of suppressing a larger physical area
    at long range as the old fixed index kernel does.
    """

    if lateral_radius_m <= 0.0:
        raise ValueError("G1R physical lateral NMS radius must be positive")
    _validate_axes(
        range_m,
        azimuth_rad,
        elevation_rad,
        (range_m.numel(), azimuth_rad.numel(), elevation_rad.numel()),
    )
    if not 0 <= range_index < range_m.numel():
        raise IndexError("G1R range index is outside its axis")
    if not 0 <= azimuth_index < azimuth_rad.numel():
        raise IndexError("G1R azimuth index is outside its axis")
    if not 0 <= elevation_index < elevation_rad.numel():
        raise IndexError("G1R elevation index is outside its axis")

    axes = [
        axis.detach().cpu().double()
        for axis in (range_m, azimuth_rad, elevation_rad)
    ]
    radial_axis, azimuth_axis, elevation_axis = axes
    positive_steps = torch.diff(radial_axis).abs()
    positive_steps = positive_steps[positive_steps > 0.0]
    minimum_effective_range = (
        float(positive_steps.median().item())
        if positive_steps.numel()
        else lateral_radius_m
    )
    radius = max(abs(float(radial_axis[range_index].item())), minimum_effective_range)
    delta_azimuth = azimuth_axis - azimuth_axis[azimuth_index]
    delta_elevation = elevation_axis - elevation_axis[elevation_index]
    horizontal = (
        radius
        * np.cos(float(elevation_axis[elevation_index].item()))
        * delta_azimuth[:, None]
    )
    vertical = radius * delta_elevation[None, :]
    return (horizontal.square() + vertical.square()) <= lateral_radius_m**2


def _range_bin_ids(range_m: torch.Tensor) -> torch.Tensor:
    ids = torch.full((range_m.numel(),), -1, dtype=torch.long)
    radial = range_m.detach().cpu()
    for index, (_, lower, upper) in enumerate(RANGE_BINS_M):
        ids[(radial >= lower) & (radial < upper)] = index
    if bool((ids < 0).any()):
        raise ValueError("G1R range axis must lie inside the frozen [0,120) m bins")
    return ids


def template_safe_range_mask(
    range_m: torch.Tensor,
    *,
    template_radius_bins: int = COARSE_TEMPLATE_RADIAL_RADIUS_BINS,
) -> torch.Tensor:
    """Mark parent cells whose complete G1D range template stays in one bin."""

    if range_m.ndim != 1 or range_m.numel() < 2:
        raise ValueError("G1R template-safe mask requires a nontrivial range axis")
    if template_radius_bins < 0:
        raise ValueError("G1R template radial radius cannot be negative")
    radial = range_m.detach().cpu().double()
    if not bool((torch.diff(radial) > 0.0).all()):
        raise ValueError("G1R requires a strictly increasing range axis")
    indices = torch.arange(radial.numel())
    lower_index = (indices - template_radius_bins).clamp_min(0)
    upper_index = (indices + template_radius_bins).clamp_max(
        radial.numel() - 1
    )
    lower_value = radial[lower_index]
    upper_value = radial[upper_index]
    safe = torch.zeros(radial.numel(), dtype=torch.bool)
    for _, lower, upper in RANGE_BINS_M:
        safe |= (lower_value >= lower) & (upper_value < upper)
    return safe


def range_aware_proposal_indices(
    score_brae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    seed_quotas: tuple[int, int, int] = SEED_QUOTAS,
    lateral_radius_m: float = LATERAL_NMS_RADIUS_M,
    radial_radius_m: float = RADIAL_NMS_RADIUS_M,
) -> torch.Tensor:
    """Select stable proposals with fixed range quotas and physical angular NMS."""

    if score_brae.ndim != 4:
        raise ValueError("G1R range-aware scores must have shape (B,R,A,E)")
    if not torch.is_floating_point(score_brae):
        raise TypeError("G1R range-aware scores must be floating point")
    if not torch.isfinite(score_brae).all():
        raise ValueError("G1R range-aware scores must be finite")
    if len(seed_quotas) != len(RANGE_BINS_M) or any(
        quota <= 0 for quota in seed_quotas
    ):
        raise ValueError("G1R requires one positive seed quota per range bin")
    if lateral_radius_m <= 0.0 or radial_radius_m <= 0.0:
        raise ValueError("G1R physical NMS radii must be positive")

    spatial_shape = tuple(int(size) for size in score_brae.shape[1:])
    _validate_axes(range_m, azimuth_rad, elevation_rad, spatial_shape)
    expected_count = sum(seed_quotas)
    spatial_count = int(np.prod(spatial_shape))
    if expected_count > spatial_count:
        raise ValueError("G1R range-aware quota exceeds Cube spatial cells")

    flat_score = score_brae.detach().flatten(start_dim=1)
    order = torch.argsort(flat_score, dim=1, descending=True, stable=True)
    radial_axis = range_m.detach().cpu().double()
    radial_bin = _range_bin_ids(range_m)
    template_safe = template_safe_range_mask(range_m)
    range_count, azimuth_count, elevation_count = spatial_shape
    plane_count = azimuth_count * elevation_count
    transfer_chunk = min(spatial_count, 65_536)
    selected_batches: list[list[int]] = []
    for batch_index in range(flat_score.shape[0]):
        suppressed = [
            np.zeros(spatial_count, dtype=np.bool_)
            for _ in RANGE_BINS_M
        ]
        selected: list[int] = []
        bin_counts = [0 for _ in RANGE_BINS_M]
        cursor = 0
        while cursor < spatial_count and any(
            count < quota for count, quota in zip(bin_counts, seed_quotas)
        ):
            stop = min(cursor + transfer_chunk, spatial_count)
            ranked = order[batch_index, cursor:stop].detach().cpu().tolist()
            for raw_flat_index in ranked:
                flat_index = int(raw_flat_index)
                radius_index = flat_index // plane_count
                if not bool(template_safe[radius_index]):
                    continue
                bin_index = int(radial_bin[radius_index].item())
                if bin_counts[bin_index] >= seed_quotas[bin_index]:
                    continue
                if suppressed[bin_index][flat_index]:
                    continue
                selected.append(flat_index)
                bin_counts[bin_index] += 1
                remainder = flat_index % plane_count
                azimuth_index = remainder // elevation_count
                elevation_index = remainder % elevation_count
                angular_mask = physical_angular_suppression_mask(
                    range_m,
                    azimuth_rad,
                    elevation_rad,
                    range_index=radius_index,
                    azimuth_index=azimuth_index,
                    elevation_index=elevation_index,
                    lateral_radius_m=lateral_radius_m,
                ).numpy()
                angular_flat = np.flatnonzero(angular_mask.reshape(-1))
                radial_mask = (
                    (radial_axis - radial_axis[radius_index]).abs()
                    <= radial_radius_m
                )
                radial_indices = torch.nonzero(
                    radial_mask & (radial_bin == bin_index),
                    as_tuple=False,
                ).flatten().tolist()
                for suppressed_radius in radial_indices:
                    offsets = suppressed_radius * plane_count + angular_flat
                    suppressed[bin_index][offsets] = True
                if all(
                    count == quota
                    for count, quota in zip(bin_counts, seed_quotas)
                ):
                    break
            cursor = stop
        if tuple(bin_counts) != tuple(seed_quotas):
            raise RuntimeError(
                "G1R physical NMS cannot pack the frozen range quotas: "
                f"{tuple(bin_counts)}"
            )
        selected_batches.append(selected)
    return torch.tensor(
        selected_batches,
        dtype=torch.long,
        device=score_brae.device,
    )


def _proposal_parent_bins(
    proposal: RadarProposalSet,
    range_m: torch.Tensor,
) -> torch.Tensor:
    radial_index = proposal.coordinates_rae[..., 0].long().detach().cpu()
    radial_value = range_m.detach().cpu()[radial_index]
    parent_bin = torch.full_like(radial_index, -1)
    for index, (_, lower, upper) in enumerate(RANGE_BINS_M):
        parent_bin[(radial_value >= lower) & (radial_value < upper)] = index
    if bool((parent_bin < 0).any()):
        raise ValueError("G1R proposal parents lie outside frozen range bins")
    return parent_bin


def expand_g1d_candidate_pool(
    cube_drae: torch.Tensor,
    proposal_flat_index: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> ExpandedCandidatePool:
    """Reuse G1D's proposal reconstruction and 32-way coarse templates."""

    if cube_drae.shape[0] != 1 or proposal_flat_index.shape[0] != 1:
        raise ValueError("G1R frame oracle requires batch size one")
    proposals = proposals_from_flat_index(cube_drae, proposal_flat_index)
    templates = coarse_query_templates(
        device=cube_drae.device,
        dtype=cube_drae.dtype,
    )
    candidate_rae = (
        proposals.coordinates_rae[:, :, None, :]
        + templates[None, None, :, :]
    ).reshape(-1, 3)
    maximum = candidate_rae.new_tensor(
        [
            cube_drae.shape[2] - 1,
            cube_drae.shape[3] - 1,
            cube_drae.shape[4] - 1,
        ]
    )
    candidate_rae = torch.minimum(
        torch.maximum(candidate_rae, torch.zeros_like(candidate_rae)),
        maximum,
    )
    candidate_xyz = continuous_rae_to_xyz(
        candidate_rae,
        range_m.to(candidate_rae),
        azimuth_rad.to(candidate_rae),
        elevation_rad.to(candidate_rae),
    )
    candidate_count = proposal_flat_index.shape[1] * COARSE_TEMPLATE_COUNT
    if candidate_xyz.shape != (candidate_count, 3):
        raise AssertionError("G1R G1D template expansion changed cardinality")
    parent_bin = _proposal_parent_bins(proposals, range_m)
    parent_bin = parent_bin.reshape(-1).repeat_interleave(COARSE_TEMPLATE_COUNT)
    return ExpandedCandidatePool(
        xyz_m=candidate_xyz,
        candidate_indices=torch.arange(
            candidate_count,
            dtype=torch.long,
            device=cube_drae.device,
        ),
        parent_range_bin=parent_bin.to(cube_drae.device),
        proposal_flat_index=proposal_flat_index[0],
    )


def fixed_quota_oracle_weights(
    target_xyz_m: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    export_quotas: tuple[int, int, int] = EXPORT_QUOTAS,
) -> torch.Tensor:
    """Rescale GT weights only for G1F selection to force export quotas.

    Original target confidence is preserved within each range bin. The caller
    must use the unmodified target weights for all reported geometry metrics.
    """

    if target_xyz_m.ndim != 2 or target_xyz_m.shape[1] != 3:
        raise ValueError("G1R target XYZ must have shape (N,3)")
    if target_weight.shape != (target_xyz_m.shape[0],):
        raise ValueError("G1R target weights must match target rows")
    if len(export_quotas) != len(RANGE_BINS_M):
        raise ValueError("G1R requires one export quota per range bin")
    if sum(export_quotas) != EXPORT_COUNT:
        raise ValueError("G1R export quotas must sum to exactly 10,000")
    if not torch.isfinite(target_weight).all() or bool((target_weight < 0).any()):
        raise ValueError("G1R target weights must be finite and nonnegative")

    radius = torch.linalg.vector_norm(target_xyz_m, dim=1)
    selection_weight = torch.zeros_like(target_weight)
    for quota, (_, lower, upper) in zip(export_quotas, RANGE_BINS_M):
        mask = (radius >= lower) & (radius < upper)
        mass = target_weight[mask].sum()
        if not bool(mask.any()) or float(mass.item()) <= 0.0:
            raise ValueError("G1R fixed oracle quota has no positive target mass")
        scaled = target_weight[mask] * (float(quota) / mass)
        positive = torch.nonzero(scaled > 0.0, as_tuple=False).flatten()
        if positive.numel() == 0:
            raise ValueError("G1R fixed oracle quota lost positive target mass")
        correction = scaled.new_tensor(float(quota)) - scaled.sum()
        scaled[positive[0]] += correction
        selection_weight[mask] = scaled
    return selection_weight


def select_fixed_quota_support_oracle(
    candidate_pool: ExpandedCandidatePool,
    target_xyz_m: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    distance_chunk_size: int = 1024,
) -> CandidateSupportSelection:
    """Reuse the G1F oracle under the frozen 75/20/5 output allocation."""

    if candidate_pool.xyz_m.shape != (PROPOSAL_COUNT, 3):
        raise ValueError("G1R oracle requires exactly 32,000 candidates")
    selection_weight = fixed_quota_oracle_weights(
        target_xyz_m,
        target_weight,
    )
    result = select_candidate_support_oracle(
        candidate_pool.xyz_m.float(),
        candidate_pool.candidate_indices,
        target_xyz_m.float(),
        target_weight=selection_weight.float(),
        distance_chunk_size=distance_chunk_size,
    )
    actual_quotas = tuple(
        int(result.per_range_support[label]["selected_count"])
        for label, _, _ in RANGE_BINS_M
    )
    if actual_quotas != EXPORT_QUOTAS:
        raise AssertionError(
            "G1R G1F oracle did not preserve frozen export quotas: "
            f"{actual_quotas}"
        )
    return result


def range_count_report(
    xyz_m: torch.Tensor,
) -> dict[str, int]:
    """Count actual coordinates in the frozen physical range bins."""

    radius = torch.linalg.vector_norm(xyz_m, dim=1)
    return {
        label: int(((radius >= lower) & (radius < upper)).sum().item())
        for label, lower, upper in RANGE_BINS_M
    }
