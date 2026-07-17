"""Shared current-frame and temporal metrics for all G4 method arms."""

from __future__ import annotations

import numpy as np
import torch

from cube_dense.parent_prediction import PointPrediction
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
    temporal_consistency_report,
)
from losses.doppler_distribution import circular_scalar_target, soft_static_target
from losses.temporal_consistency import temporal_match
from models.cube_doppler import circular_mean, query_cube_spectrum, wrapped_delta
from models.point_to_cube import soft_splat_raed
from models.temporal_baselines import analytic_static_center, in_spatial_grid


def discrete_query_indices(
    coordinates_rae: torch.Tensor,
    spatial_shape: tuple[int, int, int] = (256, 107, 37),
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = in_spatial_grid(coordinates_rae, spatial_shape)
    if not valid.any():
        raise ValueError("Prediction has no points inside the current Cube")
    indices = torch.round(coordinates_rae[valid]).long()
    indices = torch.stack(
        [
            indices[:, axis].clamp(0, spatial_shape[axis] - 1)
            for axis in range(3)
        ],
        dim=1,
    )
    return indices, valid


def weighted_subset_mean(
    values: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    if not mask.any():
        return None
    selected_weight = weight[mask].clamp_min(0.0)
    return float(
        (
            (values[mask] * selected_weight).sum()
            / selected_weight.sum().clamp_min(1e-8)
        ).item()
    )


def stratified_geometry_report(
    prediction: PointPrediction,
    prediction_static_center_mps: torch.Tensor,
    target_xyz_m: torch.Tensor,
    target_weight: torch.Tensor,
    target_scalar_mps: torch.Tensor,
    target_static_center_mps: torch.Tensor,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    dynamic_threshold_mps: float = 1.5,
) -> dict[str, float]:
    prediction_scalar = circular_mean(
        prediction.probability,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
    )
    prediction_dynamic = (
        wrapped_delta(
            prediction_scalar,
            prediction_static_center_mps,
            doppler_period_mps,
        ).abs()
        >= dynamic_threshold_mps
    )
    target_static_probability = soft_static_target(
        target_scalar_mps, target_static_center_mps, doppler_period_mps
    )
    target_dynamic = target_static_probability < 0.5
    target_to_prediction = nearest_distance(target_xyz_m, prediction.xyz_m)
    report: dict[str, float] = {
        "predicted_dynamic_fraction": float(prediction_dynamic.float().mean().item()),
        "target_dynamic_fraction": float(target_dynamic.float().mean().item()),
    }
    for label, target_mask, prediction_mask in (
        ("static", ~target_dynamic, ~prediction_dynamic),
        ("dynamic", target_dynamic, prediction_dynamic),
    ):
        completeness = weighted_subset_mean(
            target_to_prediction, target_weight, target_mask
        )
        if completeness is not None:
            report[f"{label}_target_completeness_mean_distance_m"] = completeness
        if target_mask.any() and prediction_mask.any():
            matched = geometry_report(
                prediction.xyz_m[prediction_mask],
                target_xyz_m[target_mask],
                target_weight=target_weight[target_mask],
            )
            for key in (
                "chamfer_m",
                "precision_mean_distance_m",
                "completeness_mean_distance_m",
                "fscore_0p5m",
                "fscore_1p0m",
                "fscore_2p0m",
            ):
                report[f"{label}_{key}"] = float(matched[key])
    return report


@torch.inference_mode()
def current_frame_report(
    prediction: PointPrediction,
    cube_drae: torch.Tensor,
    target_xyz_confidence: torch.Tensor,
    target_rae_index: torch.Tensor,
    ego_speed_mps: torch.Tensor | float,
    static_hypothesis: str,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    doppler_step_mps: torch.Tensor,
    dynamic_geometry_threshold_mps: float = 1.5,
) -> dict:
    if cube_drae.ndim == 4:
        cube_drae = cube_drae.unsqueeze(0)
    if cube_drae.ndim != 5 or cube_drae.shape[0] != 1:
        raise ValueError("G4 evaluation expects one current RAED Cube")
    indices, valid = discrete_query_indices(prediction.coordinates_rae)
    current_spectrum = query_cube_spectrum(cube_drae, indices)
    current_static_center = analytic_static_center(
        prediction.xyz_m,
        ego_speed_mps,
        static_hypothesis,
        doppler_lower_mps,
        doppler_period_mps,
    )
    doppler = doppler_distribution_report(
        prediction.probability[valid],
        current_spectrum,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
        doppler_step_mps,
        confidence=prediction.confidence[valid],
        static_center_mps=current_static_center[valid],
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
    target_static_center = analytic_static_center(
        target[:, :3],
        ego_speed_mps,
        static_hypothesis,
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
        prediction.coordinates_rae,
        prediction.probability,
        prediction.confidence,
    )
    cycle = cube_cycle_report(rendered, cube_drae[0], prediction.confidence)
    cycle["valid_point_fraction"] = float(valid.float().mean().item())
    stratified = stratified_geometry_report(
        prediction,
        current_static_center,
        target[:, :3],
        target[:, 3],
        target_scalar,
        target_static_center,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
        dynamic_threshold_mps=dynamic_geometry_threshold_mps,
    )
    return {
        "generated_geometry": geometry,
        "doppler": doppler,
        "cd_doppler": cd_doppler,
        "cycle": cycle,
        "stratified_geometry": stratified,
    }


@torch.inference_mode()
def pair_temporal_report(
    previous: PointPrediction,
    current: PointPrediction,
    current_from_previous: torch.Tensor,
    delta_seconds: torch.Tensor | float,
    doppler_mps: torch.Tensor,
    doppler_lower_mps: torch.Tensor,
    doppler_period_mps: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    dynamic_threshold_mps: float = 1.0,
) -> dict[str, float]:
    match = temporal_match(
        previous.xyz_m,
        previous.probability,
        previous.confidence,
        current.xyz_m,
        current.probability,
        current.confidence,
        current_from_previous,
        delta_seconds,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
        previous_static_center_mps=previous.static_center_mps,
        current_static_center_mps=current.static_center_mps,
        dynamic_threshold_mps=dynamic_threshold_mps,
    )
    return temporal_consistency_report(
        match,
        previous.confidence,
        current.xyz_m,
        current.confidence,
        range_m,
        azimuth_rad,
        elevation_rad,
    )


def aggregate_flat_reports(reports: list[dict[str, float]]) -> dict[str, dict]:
    if not reports:
        raise ValueError("Cannot aggregate an empty flat report list")
    keys = sorted({key for report in reports for key in report})
    aggregate = {}
    for key in keys:
        values = np.asarray(
            [
                float(report[key])
                for report in reports
                if key in report and np.isfinite(report[key])
            ],
            dtype=np.float64,
        )
        if values.size:
            aggregate[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "sample_count": int(values.size),
            }
    return aggregate


def aggregate_method_frames(frames: list[dict]) -> dict:
    if not frames:
        raise ValueError("Cannot aggregate empty G4 method frames")
    temporal = [frame["temporal"] for frame in frames if frame["temporal"] is not None]
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
        "stratified_geometry": aggregate_flat_reports(
            [frame["current"]["stratified_geometry"] for frame in frames]
        ),
        "temporal": aggregate_temporal_reports(temporal),
    }
