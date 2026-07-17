"""Shared current-Cube temporal inference for training and strict rollout."""

from __future__ import annotations

import torch

from cube_dense.parent_prediction import PointPrediction
from eval.dense_geometry import occupancy_to_points
from models.cube_doppler import split_query_indices
from models.cube_temporal import CubeTemporalNet
from models.temporal_prior import gated_doppler_warp, rasterize_temporal_prior


def prediction_static_center(
    model: CubeTemporalNet,
    prediction: dict[str, torch.Tensor],
    indices: torch.Tensor,
    ego_speed_mps: torch.Tensor,
) -> torch.Tensor:
    if "static_center_mps" in prediction:
        return prediction["static_center_mps"]
    batch, _, azimuth, elevation = split_query_indices(indices, 1)
    return model.static_center(batch, azimuth, elevation, ego_speed_mps)


def make_temporal_prior(
    model: CubeTemporalNet,
    previous: PointPrediction,
    pair: dict,
    device: torch.device,
    dynamic_threshold_mps: float,
):
    transform = torch.tensor(
        pair["current_from_previous"], dtype=torch.float32, device=device
    ).reshape(4, 4)
    delta_seconds = torch.tensor(
        pair["delta_seconds"], dtype=torch.float32, device=device
    )
    prior = gated_doppler_warp(
        previous.xyz_m,
        previous.probability,
        previous.confidence,
        transform,
        delta_seconds,
        model.doppler_mps,
        model.doppler_lower_mps,
        model.doppler_period_mps,
        model.range_m,
        model.azimuth_rad,
        model.elevation_rad,
        previous_static_center_mps=previous.static_center_mps,
        dynamic_threshold_mps=dynamic_threshold_mps,
    )
    return prior, transform, delta_seconds


def predict_temporal_pair(
    model: CubeTemporalNet,
    current_item: dict,
    previous: PointPrediction,
    pair: dict,
    axes,
    point_count: int,
    dynamic_threshold_mps: float,
    device: torch.device,
    autocast_enabled: bool = True,
) -> dict:
    prior, transform, delta_seconds = make_temporal_prior(
        model, previous, pair, device, dynamic_threshold_mps
    )
    prior_raster = None
    if model.fusion_mode == "concat":
        prior_raster = rasterize_temporal_prior(
            prior,
            model.doppler_mps,
            model.doppler_lower_mps,
            model.doppler_period_mps,
        )
    cube = current_item["cube_drae"].unsqueeze(0).to(device)
    occupancy = current_item["occupancy"].unsqueeze(0).to(device)
    ego_speed = current_item["ego_speed_mps"].reshape(1).to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled and device.type == "cuda",
    ):
        occupancy_logits, features = model.forward_temporal(cube, prior_raster)
    query_xyz, confidence, indices = occupancy_to_points(
        occupancy_logits[0].float(), axes, point_count=point_count
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled and device.type == "cuda",
    ):
        prediction = model.query_temporal(
            features, indices, query_xyz, ego_speed, prior
        )
    current_static = prediction_static_center(
        model, prediction, indices, ego_speed
    )
    return {
        "prior": prior,
        "transform": transform,
        "delta_seconds": delta_seconds,
        "cube": cube,
        "occupancy": occupancy,
        "occupancy_logits": occupancy_logits,
        "features": features,
        "query_xyz": query_xyz,
        "confidence": confidence,
        "indices": indices,
        "prediction": prediction,
        "current_static_center_mps": current_static,
        "ego_speed": ego_speed,
    }
