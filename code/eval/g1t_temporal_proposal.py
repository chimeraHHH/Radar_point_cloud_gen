"""No-train temporal proposal support for G1T."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cube_dense.parent_prediction import PointPrediction
from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import circular_mean, query_cube_spectrum, wrapped_delta
from models.point_to_cube import trilinear_query_features
from models.rald_query_field import (
    integrated_log_energy,
    proposals_from_flat_index,
    stable_radar_proposals,
)
from models.temporal_baselines import analytic_static_center, warp_prediction


ARM_NAMES = ("t0_current", "t1_ego", "t2_doppler")


@dataclass(frozen=True)
class CubeProposalFrame:
    """Measured Cube proposals and their deterministic G1D flat indices."""

    prediction: PointPrediction
    flat_index: torch.Tensor
    cube_support: torch.Tensor


@dataclass(frozen=True)
class HistoryProposalSource:
    """One measured historical proposal frame expressed relative to current."""

    prediction: PointPrediction
    current_from_source: torch.Tensor
    age_seconds: float
    age_frames: int


@dataclass(frozen=True)
class ProposalSelection:
    """A fixed-count current-frame proposal selection with source provenance."""

    prediction: PointPrediction
    current_cube_support: torch.Tensor
    source_age_seconds: torch.Tensor
    source_age_frames: torch.Tensor
    from_history: torch.Tensor
    dynamic: torch.Tensor
    selected_candidate_index: torch.Tensor
    candidate_count: int
    unique_candidate_voxel_count: int
    deduplicated_count: int
    suppressed_count: int
    fill_count: int
    supplied_history_count: int


def _validate_single_cube(cube_drae: torch.Tensor) -> tuple[int, int, int]:
    if cube_drae.ndim != 5 or cube_drae.shape[0] != 1:
        raise ValueError("G1T expects one Full-RAED Cube")
    if cube_drae.shape[1] != 64:
        raise ValueError("G1T expects all 64 Doppler bins")
    if not torch.is_floating_point(cube_drae):
        raise TypeError("G1T Cube must be floating point")
    if not torch.isfinite(cube_drae).all():
        raise ValueError("G1T Cube must contain only finite values")
    return tuple(int(value) for value in cube_drae.shape[2:])


def _validate_axes(
    spatial_shape: tuple[int, int, int],
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> None:
    if (
        range_m.shape != (spatial_shape[0],)
        or azimuth_rad.shape != (spatial_shape[1],)
        or elevation_rad.shape != (spatial_shape[2],)
    ):
        raise ValueError("G1T axes do not match the Cube spatial shape")


def _normalized_support(
    support: torch.Tensor, maximum_support: torch.Tensor
) -> torch.Tensor:
    return (
        support.clamp_min(0.0)
        / maximum_support.to(support).clamp_min(1e-8)
    ).clamp(0.0, 1.0)


@torch.inference_mode()
def cube_proposal_frame(
    cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    ego_speed_mps: torch.Tensor | float,
    static_hypothesis: str,
    *,
    point_count: int = 10_000,
    nms_kernel: tuple[int, int, int] = (5, 5, 3),
    proposal_flat_index: torch.Tensor | None = None,
) -> CubeProposalFrame:
    """Build deterministic measured proposals without a learned G1D score."""

    spatial_shape = _validate_single_cube(cube_drae)
    _validate_axes(spatial_shape, range_m, azimuth_rad, elevation_rad)
    if point_count <= 0:
        raise ValueError("G1T point count must be positive")
    if proposal_flat_index is None:
        proposals = stable_radar_proposals(
            cube_drae,
            seed_count=point_count,
            nms_kernel=nms_kernel,
        )
    else:
        if proposal_flat_index.shape != (1, point_count):
            raise ValueError("Cached G1D proposal count differs from G1T")
        proposals = proposals_from_flat_index(
            cube_drae, proposal_flat_index.to(cube_drae.device)
        )
    coordinates = proposals.coordinates_rae[0].float()
    xyz_m = continuous_rae_to_xyz(
        coordinates, range_m, azimuth_rad, elevation_rad
    )
    probability = query_cube_spectrum(cube_drae, coordinates)
    support = proposals.integrated_log_energy[0].float()
    maximum_support = integrated_log_energy(cube_drae).amax()
    prediction = PointPrediction(
        xyz_m=xyz_m,
        coordinates_rae=coordinates,
        probability=probability,
        confidence=_normalized_support(support, maximum_support),
        static_center_mps=analytic_static_center(
            xyz_m,
            ego_speed_mps,
            static_hypothesis,
            doppler_lower_mps,
            doppler_period_mps,
        ),
    )
    return CubeProposalFrame(
        prediction=prediction,
        flat_index=proposals.flat_index.detach(),
        cube_support=support,
    )


def _filtered_prediction(
    prediction: PointPrediction, valid: torch.Tensor
) -> PointPrediction:
    return PointPrediction(
        xyz_m=prediction.xyz_m[valid],
        coordinates_rae=prediction.coordinates_rae[valid],
        probability=prediction.probability[valid],
        confidence=prediction.confidence[valid],
        static_center_mps=prediction.static_center_mps[valid],
    )


@torch.inference_mode()
def warp_history_source(
    source: HistoryProposalSource,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    current_ego_speed_mps: torch.Tensor | float,
    static_hypothesis: str,
    *,
    apply_doppler_displacement: bool,
    dynamic_threshold_mps: float = 1.0,
) -> PointPrediction:
    """Advance source by +residual-Doppler*age, then map source to current."""

    if source.age_seconds <= 0.0 or source.age_frames <= 0:
        raise ValueError("Historical proposal age must be positive")
    if source.current_from_source.shape != (4, 4):
        raise ValueError("Historical transform must be 4x4")
    return warp_prediction(
        source.prediction,
        source.current_from_source,
        source.age_seconds,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
        range_m,
        azimuth_rad,
        elevation_rad,
        current_ego_speed_mps,
        static_hypothesis,
        apply_doppler_displacement=apply_doppler_displacement,
        dynamic_threshold_mps=dynamic_threshold_mps,
    )


def _candidate_metadata(
    states: list[PointPrediction],
    source_age_seconds: list[float],
    source_age_frames: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    seconds = torch.cat(
        [
            torch.full(
                (state.xyz_m.shape[0],),
                float(age),
                dtype=state.xyz_m.dtype,
                device=state.xyz_m.device,
            )
            for state, age in zip(states, source_age_seconds, strict=True)
        ]
    )
    frames = torch.cat(
        [
            torch.full(
                (state.xyz_m.shape[0],),
                int(age),
                dtype=torch.long,
                device=state.xyz_m.device,
            )
            for state, age in zip(states, source_age_frames, strict=True)
        ]
    )
    return seconds, frames


def _greedy_fixed_nms(
    coordinates_rae: torch.Tensor,
    ranked_index: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    nms_kernel: tuple[int, int, int],
    output_count: int,
) -> tuple[torch.Tensor, int, int]:
    if len(nms_kernel) != 3 or any(
        size <= 0 or size % 2 == 0 for size in nms_kernel
    ):
        raise ValueError("G1T NMS kernel sizes must be positive odd integers")
    quantized = torch.floor(coordinates_rae).long()
    quantized = torch.stack(
        [
            quantized[:, axis].clamp(0, spatial_shape[axis] - 1)
            for axis in range(3)
        ],
        dim=1,
    )
    coordinates_cpu = quantized.detach().cpu().tolist()
    ranking = ranked_index.detach().cpu().tolist()
    occupied = bytearray(
        spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    )
    half_width = tuple(size // 2 for size in nms_kernel)
    selected: list[int] = []
    selected_set: set[int] = set()
    for raw_index in ranking:
        candidate_index = int(raw_index)
        radius, azimuth, elevation = coordinates_cpu[candidate_index]
        flat = (
            radius * spatial_shape[1] * spatial_shape[2]
            + azimuth * spatial_shape[2]
            + elevation
        )
        if occupied[flat]:
            continue
        selected.append(candidate_index)
        selected_set.add(candidate_index)
        for radius_index in range(
            max(0, radius - half_width[0]),
            min(spatial_shape[0], radius + half_width[0] + 1),
        ):
            radius_offset = (
                radius_index * spatial_shape[1] * spatial_shape[2]
            )
            for azimuth_index in range(
                max(0, azimuth - half_width[1]),
                min(spatial_shape[1], azimuth + half_width[1] + 1),
            ):
                start = (
                    radius_offset
                    + azimuth_index * spatial_shape[2]
                    + max(0, elevation - half_width[2])
                )
                stop = (
                    radius_offset
                    + azimuth_index * spatial_shape[2]
                    + min(
                        spatial_shape[2],
                        elevation + half_width[2] + 1,
                    )
                )
                occupied[start:stop] = b"\x01" * (stop - start)
        if len(selected) == output_count:
            break
    deduplicated_count = len(selected)
    if deduplicated_count < output_count:
        for raw_index in ranking:
            candidate_index = int(raw_index)
            if candidate_index in selected_set:
                continue
            selected.append(candidate_index)
            if len(selected) == output_count:
                break
    if len(selected) != output_count:
        raise RuntimeError("G1T cannot fill the fixed output point budget")
    return (
        torch.tensor(
            selected, dtype=torch.long, device=coordinates_rae.device
        ),
        deduplicated_count,
        output_count - deduplicated_count,
    )


@torch.inference_mode()
def select_current_rescored(
    current_cube_drae: torch.Tensor,
    candidate_states: list[PointPrediction],
    source_age_seconds: list[float],
    source_age_frames: list[int],
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    current_ego_speed_mps: torch.Tensor | float,
    static_hypothesis: str,
    *,
    output_count: int = 10_000,
    nms_kernel: tuple[int, int, int] = (5, 5, 3),
    dynamic_threshold_mps: float = 1.0,
) -> ProposalSelection:
    """Select by current measured Cube support; no target enters this API."""

    spatial_shape = _validate_single_cube(current_cube_drae)
    _validate_axes(spatial_shape, range_m, azimuth_rad, elevation_rad)
    if not candidate_states:
        raise ValueError("G1T requires at least the current proposal state")
    if (
        len(candidate_states) != len(source_age_seconds)
        or len(candidate_states) != len(source_age_frames)
    ):
        raise ValueError("G1T source metadata does not match candidate states")
    if source_age_seconds[0] != 0.0 or source_age_frames[0] != 0:
        raise ValueError("The first G1T candidate state must be current")
    if any(
        seconds < 0.0
        or frames < 0
        or ((seconds == 0.0) != (frames == 0))
        for seconds, frames in zip(
            source_age_seconds, source_age_frames, strict=True
        )
    ):
        raise ValueError("G1T source ages are inconsistent")
    if candidate_states[0].xyz_m.shape[0] < output_count:
        raise ValueError("Current proposals cannot fill the output budget")

    valid_states: list[PointPrediction] = []
    valid_seconds: list[float] = []
    valid_frames: list[int] = []
    for state, seconds, frames in zip(
        candidate_states,
        source_age_seconds,
        source_age_frames,
        strict=True,
    ):
        valid = torch.isfinite(state.coordinates_rae).all(dim=1)
        for axis, size in enumerate(spatial_shape):
            valid &= (state.coordinates_rae[:, axis] >= 0.0) & (
                state.coordinates_rae[:, axis] <= size - 1
            )
        if valid.any():
            valid_states.append(_filtered_prediction(state, valid))
            valid_seconds.append(seconds)
            valid_frames.append(frames)
    if not valid_states or valid_frames[0] != 0:
        raise RuntimeError("G1T current proposals left the current Cube")

    xyz_m = torch.cat([state.xyz_m for state in valid_states])
    coordinates_rae = torch.cat(
        [state.coordinates_rae for state in valid_states]
    )
    ages_seconds, ages_frames = _candidate_metadata(
        valid_states, valid_seconds, valid_frames
    )
    energy_grid = integrated_log_energy(current_cube_drae).unsqueeze(1)
    support = trilinear_query_features(
        energy_grid, coordinates_rae
    ).squeeze(1).float()
    ranked = torch.argsort(
        support, descending=True, stable=True
    )
    selected, deduplicated_count, fill_count = _greedy_fixed_nms(
        coordinates_rae,
        ranked,
        spatial_shape,
        nms_kernel,
        output_count,
    )
    selected_coordinates = coordinates_rae[selected]
    selected_xyz = xyz_m[selected]
    selected_support = support[selected]
    probability = query_cube_spectrum(
        current_cube_drae, selected_coordinates.float()
    )
    confidence = _normalized_support(
        selected_support, energy_grid.amax()
    )
    static_center = analytic_static_center(
        selected_xyz,
        current_ego_speed_mps,
        static_hypothesis,
        doppler_lower_mps,
        doppler_period_mps,
    )
    scalar = circular_mean(
        probability,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
    )
    dynamic = (
        wrapped_delta(
            scalar, static_center, doppler_period_mps
        ).abs()
        >= dynamic_threshold_mps
    )
    flat_voxel = torch.floor(coordinates_rae).long()
    flat_voxel = (
        flat_voxel[:, 0] * spatial_shape[1] * spatial_shape[2]
        + flat_voxel[:, 1] * spatial_shape[2]
        + flat_voxel[:, 2]
    )
    selected_ages_frames = ages_frames[selected]
    prediction = PointPrediction(
        xyz_m=selected_xyz,
        coordinates_rae=selected_coordinates,
        probability=probability,
        confidence=confidence,
        static_center_mps=static_center,
    )
    return ProposalSelection(
        prediction=prediction,
        current_cube_support=selected_support,
        source_age_seconds=ages_seconds[selected],
        source_age_frames=selected_ages_frames,
        from_history=selected_ages_frames > 0,
        dynamic=dynamic,
        selected_candidate_index=selected,
        candidate_count=int(coordinates_rae.shape[0]),
        unique_candidate_voxel_count=int(torch.unique(flat_voxel).numel()),
        deduplicated_count=deduplicated_count,
        suppressed_count=int(coordinates_rae.shape[0] - deduplicated_count),
        fill_count=fill_count,
        supplied_history_count=len(candidate_states) - 1,
    )


@torch.inference_mode()
def build_temporal_proposal_arms(
    current_cube_drae: torch.Tensor,
    current: PointPrediction,
    history_newest_first: list[HistoryProposalSource],
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    current_ego_speed_mps: torch.Tensor | float,
    static_hypothesis: str,
    *,
    output_count: int = 10_000,
    nms_kernel: tuple[int, int, int] = (5, 5, 3),
    dynamic_threshold_mps: float = 1.0,
) -> dict[str, ProposalSelection]:
    """Construct T0/T1/T2 from the same ordered historical source frames."""

    if not history_newest_first:
        raise ValueError("G1T requires a non-empty fixed history")
    ages = [source.age_frames for source in history_newest_first]
    if ages != sorted(ages) or len(set(ages)) != len(ages):
        raise ValueError("G1T history must be unique and newest first")
    ego_history = [
        warp_history_source(
            source,
            doppler_mps,
            doppler_lower_mps,
            doppler_period_mps,
            range_m,
            azimuth_rad,
            elevation_rad,
            current_ego_speed_mps,
            static_hypothesis,
            apply_doppler_displacement=False,
            dynamic_threshold_mps=dynamic_threshold_mps,
        )
        for source in history_newest_first
    ]
    doppler_history = [
        warp_history_source(
            source,
            doppler_mps,
            doppler_lower_mps,
            doppler_period_mps,
            range_m,
            azimuth_rad,
            elevation_rad,
            current_ego_speed_mps,
            static_hypothesis,
            apply_doppler_displacement=True,
            dynamic_threshold_mps=dynamic_threshold_mps,
        )
        for source in history_newest_first
    ]
    age_seconds = [0.0] + [
        source.age_seconds for source in history_newest_first
    ]
    age_frames = [0] + ages
    common = {
        "current_cube_drae": current_cube_drae,
        "range_m": range_m,
        "azimuth_rad": azimuth_rad,
        "elevation_rad": elevation_rad,
        "doppler_mps": doppler_mps,
        "doppler_lower_mps": doppler_lower_mps,
        "doppler_period_mps": doppler_period_mps,
        "current_ego_speed_mps": current_ego_speed_mps,
        "static_hypothesis": static_hypothesis,
        "output_count": output_count,
        "nms_kernel": nms_kernel,
        "dynamic_threshold_mps": dynamic_threshold_mps,
    }
    t0 = select_current_rescored(
        candidate_states=[current],
        source_age_seconds=[0.0],
        source_age_frames=[0],
        **common,
    )
    t1 = select_current_rescored(
        candidate_states=[current] + ego_history,
        source_age_seconds=age_seconds,
        source_age_frames=age_frames,
        **common,
    )
    t2 = select_current_rescored(
        candidate_states=[current] + doppler_history,
        source_age_seconds=age_seconds,
        source_age_frames=age_frames,
        **common,
    )
    if any(
        selection.prediction.xyz_m.shape[0] != output_count
        for selection in (t0, t1, t2)
    ):
        raise AssertionError("G1T arm changed the fixed output count")
    if (
        t1.supplied_history_count != len(history_newest_first)
        or t2.supplied_history_count != len(history_newest_first)
    ):
        raise AssertionError("G1T temporal arms used different history counts")
    return {
        "t0_current": t0,
        "t1_ego": t1,
        "t2_doppler": t2,
    }


def _masked_summary(
    selection: ProposalSelection, mask: torch.Tensor, prefix: str
) -> dict[str, float]:
    if mask.shape != (selection.prediction.xyz_m.shape[0],):
        raise ValueError("G1T report mask does not match selected points")
    count = int(mask.sum().item())
    report: dict[str, float] = {
        f"{prefix}point_count": count,
        f"{prefix}point_fraction": float(mask.float().mean().item()),
    }
    if count == 0:
        return report
    history = selection.from_history[mask]
    age = selection.source_age_seconds[mask]
    support = selection.current_cube_support[mask]
    report.update(
        {
            f"{prefix}history_fraction": float(
                history.float().mean().item()
            ),
            f"{prefix}source_age_seconds_mean": float(age.mean().item()),
            f"{prefix}source_age_seconds_median": float(
                age.median().item()
            ),
            f"{prefix}current_cube_support_mean": float(
                support.mean().item()
            ),
            f"{prefix}current_cube_support_median": float(
                support.median().item()
            ),
        }
    )
    if history.any():
        history_age = age[history]
        history_age_frames = selection.source_age_frames[mask][history]
        report.update(
            {
                f"{prefix}history_source_age_seconds_mean": float(
                    history_age.mean().item()
                ),
                f"{prefix}history_source_age_seconds_median": float(
                    history_age.median().item()
                ),
                f"{prefix}history_source_age_seconds_max": float(
                    history_age.max().item()
                ),
                f"{prefix}history_source_age_frames_mean": float(
                    history_age_frames.float().mean().item()
                ),
            }
        )
    return report


@torch.inference_mode()
def selection_report(
    selection: ProposalSelection,
    *,
    near_boundary_m: float = 60.0,
) -> dict[str, float]:
    """Report source, support, motion, and range slices without GT."""

    if near_boundary_m <= 0.0:
        raise ValueError("G1T near/far boundary must be positive")
    point_count = selection.prediction.xyz_m.shape[0]
    all_points = torch.ones(
        point_count,
        dtype=torch.bool,
        device=selection.prediction.xyz_m.device,
    )
    point_range = torch.linalg.vector_norm(
        selection.prediction.xyz_m, dim=1
    )
    report = {
        "candidate_count": selection.candidate_count,
        "unique_candidate_voxel_count": (
            selection.unique_candidate_voxel_count
        ),
        "deduplicated_count": selection.deduplicated_count,
        "suppressed_count": selection.suppressed_count,
        "fill_count": selection.fill_count,
        "supplied_history_count": selection.supplied_history_count,
        "output_point_count": point_count,
        "source_age_seconds_max": float(
            selection.source_age_seconds.max().item()
        ),
    }
    for mask, prefix in (
        (all_points, ""),
        (~selection.dynamic, "static_"),
        (selection.dynamic, "dynamic_"),
        (point_range < near_boundary_m, "near_"),
        (point_range >= near_boundary_m, "far_"),
    ):
        report.update(_masked_summary(selection, mask, prefix))
    return report
