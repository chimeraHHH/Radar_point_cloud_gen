"""Convention-free current-frame and temporal reports for RaLD G4R."""

from __future__ import annotations

import numpy as np
import torch

from cube_dense.rald_prediction import RaLDPointPrediction
from eval.cube_cycle import aggregate_cycle_reports, cube_cycle_report
from eval.dense_geometry import (
    aggregate_geometry_reports,
    geometry_report,
    nearest_distance,
)
from eval.doppler_distribution import (
    aggregate_doppler_reports,
    cd_doppler_report,
    doppler_distribution_report,
)
from eval.temporal_cube import (
    aggregate_temporal_reports,
    ego_aligned_consistency_report,
)
from losses.doppler_distribution import circular_scalar_target
from losses.temporal_consistency import ego_aligned_match
from models.cube_doppler import circular_mean, query_cube_spectrum
from models.point_to_cube import soft_splat_raed
from models.temporal_baselines import in_spatial_grid


@torch.inference_mode()
def rald_current_frame_report(
    prediction: RaLDPointPrediction,
    cube_drae: torch.Tensor,
    target_xyz_confidence: torch.Tensor,
    target_rae_index: torch.Tensor,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    doppler_step_mps: torch.Tensor,
) -> dict:
    if cube_drae.ndim == 4:
        cube_drae = cube_drae.unsqueeze(0)
    valid = in_spatial_grid(prediction.coordinates_rae, (256, 107, 37))
    if not valid.any():
        raise ValueError("G4R prediction has no point inside the current Cube")
    current_spectrum = query_cube_spectrum(
        cube_drae, prediction.coordinates_rae[valid].float()
    )
    doppler = doppler_distribution_report(
        prediction.probability[valid],
        current_spectrum,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
        doppler_step_mps,
        confidence=prediction.confidence[valid],
    )
    target = target_xyz_confidence
    geometry = geometry_report(
        prediction.xyz_m,
        target[:, :3],
        target_weight=target[:, 3],
    )
    target_distribution = query_cube_spectrum(cube_drae, target_rae_index)
    target_scalar = circular_scalar_target(
        target_distribution,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
    )
    prediction_scalar = circular_mean(
        prediction.probability,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
    )
    cd_doppler = cd_doppler_report(
        prediction.xyz_m,
        prediction_scalar,
        target[:, :3],
        target_scalar,
        target_weight=target[:, 3],
    )
    rendered = soft_splat_raed(
        prediction.coordinates_rae[valid],
        prediction.probability[valid],
        prediction.confidence[valid],
    )
    existence_target = (
        nearest_distance(prediction.xyz_m[valid], target[:, :3]) <= 1.0
    ).float()
    cycle = cube_cycle_report(
        rendered,
        cube_drae[0],
        prediction.confidence[valid],
        existence_target=existence_target,
    )
    cycle["valid_point_fraction"] = float(valid.float().mean().item())
    return {
        "generated_geometry": geometry,
        "doppler": doppler,
        "cd_doppler": cd_doppler,
        "cycle": cycle,
    }


@torch.inference_mode()
def rald_pair_temporal_report(
    previous: RaLDPointPrediction,
    current: RaLDPointPrediction,
    current_from_previous: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> dict[str, float]:
    match = ego_aligned_match(
        previous.xyz_m,
        previous.confidence,
        current.xyz_m,
        current.confidence,
        current_from_previous,
    )
    return ego_aligned_consistency_report(
        match,
        previous.confidence,
        current.xyz_m,
        current.confidence,
        range_m,
        azimuth_rad,
        elevation_rad,
    )


def aggregate_rald_method_frames(frames: list[dict]) -> dict:
    if not frames:
        raise ValueError("Cannot aggregate empty G4R frames")
    temporal = [frame["temporal"] for frame in frames if frame["temporal"]]
    return {
        "frame_count": len(frames),
        "temporal_pair_count": len(temporal),
        "generated_geometry": aggregate_geometry_reports(
            [frame["current"]["generated_geometry"] for frame in frames]
        ),
        "doppler": aggregate_doppler_reports(
            [frame["current"]["doppler"] for frame in frames]
        ),
        "cd_doppler": aggregate_doppler_reports(
            [frame["current"]["cd_doppler"] for frame in frames]
        ),
        "cycle": aggregate_cycle_reports(
            [frame["current"]["cycle"] for frame in frames]
        ),
        "temporal": aggregate_temporal_reports(temporal),
    }


def finite_frame_metrics(frame: dict) -> bool:
    stack = [frame]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (int, float)) and not np.isfinite(value):
            return False
    return True
