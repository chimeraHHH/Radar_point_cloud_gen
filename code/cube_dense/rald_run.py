"""Reconstruct frozen single-frame and temporal RaLD-anchor checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.cube_occupancy import CubeOccupancyNet
from models.rald_anchor import FrozenParentRaLDRefiner
from models.rald_anchor_temporal import FrozenParentRaLDTemporalRefiner
from models.rald_matched import FullRAEDRadarTokenEncoder


def load_rald_run(run: Path, *, expected_variant: str = "full") -> dict:
    run = run.resolve()
    config_path = run / "config.json"
    checkpoint_path = run / "best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Incomplete RaLD run: {run}")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if config.get("cycle_variant") != expected_variant:
        raise ValueError(f"RaLD run must use {expected_variant} cycle")
    if config.get("doppler_head_mode") != "distribution":
        raise ValueError("RaLD temporal parent must use the distribution head")
    parent_checkpoint = Path(provenance["parent_g1_checkpoint"]).resolve()
    parent_config_path = parent_checkpoint.parent / "config.json"
    if not parent_checkpoint.is_file() or not parent_config_path.is_file():
        raise FileNotFoundError("RaLD geometry parent artifacts are incomplete")
    return {
        "run": run,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "provenance": provenance,
        "parent_checkpoint": parent_checkpoint,
        "parent_config_path": parent_config_path,
    }


def build_rald_components(run: dict, axes, device: torch.device):
    parent_document = json.loads(
        run["parent_config_path"].read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    config = run["config"]
    parent = CubeOccupancyNet(
        parent_config["mode"],
        torch.from_numpy(axes.doppler_mps),
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
    ).to(device)
    parent_checkpoint = torch.load(
        run["parent_checkpoint"], map_location=device, weights_only=False
    )
    parent.load_state_dict(parent_checkpoint["model"], strict=True)
    radar_encoder = FullRAEDRadarTokenEncoder(
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
        spectral_channels=int(config["radar_spectral_channels"]),
        token_dim=int(config["model_dim"]),
        base_channels=int(config["radar_base_channels"]),
    )
    return parent, radar_encoder


def build_single_frame_rald(run: dict, axes, device: torch.device):
    config = run["config"]
    parent, radar_encoder = build_rald_components(run, axes, device)
    model = FrozenParentRaLDRefiner(
        parent,
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        point_count=int(config["point_count"]),
        latent_count=int(config["latent_count"]),
        model_dim=int(config["model_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
        radar_encoder=radar_encoder,
        radar_token_dim=int(config["model_dim"]),
        doppler_head_mode="distribution",
    ).to(device)
    checkpoint = torch.load(
        run["checkpoint_path"], map_location=device, weights_only=False
    )
    model.refiner.load_state_dict(checkpoint["refiner"], strict=True)
    model.radar_encoder.load_state_dict(checkpoint["radar_encoder"], strict=True)
    return model


def build_temporal_rald(
    run: dict,
    axes,
    device: torch.device,
    fusion_mode: str,
    prior_base_channels: int = 32,
):
    config = run["config"]
    parent, radar_encoder = build_rald_components(run, axes, device)
    model = FrozenParentRaLDTemporalRefiner(
        parent,
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        torch.from_numpy(axes.doppler_mps),
        radar_encoder,
        temporal_fusion_mode=fusion_mode,
        point_count=int(config["point_count"]),
        latent_count=int(config["latent_count"]),
        model_dim=int(config["model_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
        doppler_head_mode="distribution",
        prior_base_channels=prior_base_channels,
    ).to(device)
    checkpoint = torch.load(
        run["checkpoint_path"], map_location=device, weights_only=False
    )
    model.load_single_frame_refiner(
        checkpoint["refiner"], checkpoint["radar_encoder"]
    )
    return model
