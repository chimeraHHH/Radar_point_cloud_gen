#!/usr/bin/env python3
"""Verify native-Cube frozen-parent RaLD trainability without running RH1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_tesseract  # noqa: E402
from models.cube_occupancy import CubeOccupancyNet, parameter_count  # noqa: E402
from models.rald_anchor import FrozenParentRaLDRefiner  # noqa: E402
from models.rald_matched import FullRAEDRadarTokenEncoder  # noqa: E402
from scripts.train_rald_anchor_refiner import gradient_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent-g1-run", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--radar-index", type=int, default=232)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpu-name", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("RaLD anchor integration verification requires CUDA")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    if device_name != args.required_gpu_name:
        raise RuntimeError(f"Required GPU {args.required_gpu_name}, got {device_name}")

    parent_document = json.loads(
        (args.parent_g1_run / "config.json").read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    if parent_config["mode"] not in {"rae_max", "full_raed"}:
        raise ValueError("Integration verification requires a G1 occupancy parent")
    axes = load_axes(args.data_root / "resources")
    parent = CubeOccupancyNet(
        parent_config["mode"],
        torch.from_numpy(axes.doppler_mps),
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
    ).to(device)
    parent_checkpoint_path = args.parent_g1_run / "best.pt"
    parent_checkpoint = torch.load(
        parent_checkpoint_path, map_location=device, weights_only=False
    )
    parent.load_state_dict(parent_checkpoint["model"], strict=True)
    radar_encoder = FullRAEDRadarTokenEncoder(
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
    )
    model = FrozenParentRaLDRefiner(
        parent,
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        radar_encoder=radar_encoder,
        radar_token_dim=512,
    ).to(device)
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-3)
    cube_path = (
        args.data_root
        / str(args.sequence)
        / "radar_tesseract"
        / f"tesseract_{args.radar_index:05d}.mat"
    )
    cube = torch.from_numpy(load_tesseract(cube_path)).float().unsqueeze(0).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    gradients = []
    initial_checks = {}
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(cube)
        if step == 1:
            initial_checks = {
                "anchor_count": int(output["xyz_m"].shape[1]),
                "latent_shape": list(output["latent"].shape),
                "radar_token_count": int(output["radar_token_count"].item()),
                "position_identity": bool(
                    torch.equal(output["xyz_m"], output["anchor_xyz_m"])
                ),
                "spectrum_identity_max_abs_error": float(
                    (
                        output["doppler_probability"].float()
                        - output["anchor_cube_spectrum"].float()
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "confidence_identity_max_abs_error": float(
                    (
                        output["confidence"].float()
                        - output["anchor_parent_confidence"].float()
                    )
                    .abs()
                    .max()
                    .item()
                ),
            }
        desired_offset = torch.full_like(output["offset_bins"].float(), 0.1)
        desired_existence = torch.full_like(
            output["confidence_residual_logit"].float(), 0.6
        )
        desired_spectrum = output["anchor_cube_spectrum"].float().roll(1, dims=-1)
        loss = (
            F.mse_loss(output["offset_bins"].float(), desired_offset)
            + F.binary_cross_entropy_with_logits(
                output["confidence_residual_logit"].float(), desired_existence
            )
            - (
                desired_spectrum
                * output["doppler_probability"].float().clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
        )
        loss.backward()
        gradients.append({"step": step, **gradient_audit(model)})
        if any(parameter.grad is not None for parameter in model.parent.parameters()):
            raise RuntimeError("Frozen parent received gradients during integration test")
        torch.nn.utils.clip_grad_norm_(model.refinement_parameters(), 5.0)
        optimizer.step()
        del output, loss
        torch.cuda.empty_cache()
    peak_memory = torch.cuda.max_memory_allocated(device)
    checks = {
        "native_anchor_count": initial_checks["anchor_count"] == 10_000,
        "native_latent_shape": initial_checks["latent_shape"] == [1, 512, 512],
        "native_radar_token_count": initial_checks["radar_token_count"] == 336,
        "initial_position_identity": initial_checks["position_identity"],
        "initial_spectrum_identity": initial_checks[
            "spectrum_identity_max_abs_error"
        ]
        <= 1e-5,
        "initial_confidence_identity": initial_checks[
            "confidence_identity_max_abs_error"
        ]
        <= 1e-5,
        "step1_physical_head_gradient": gradients[0]["physical_head"] > 0.0,
        "step2_set_latent_gradient": gradients[1]["set_latent_backbone"] > 0.0,
        "step2_radar_encoder_gradient": gradients[1]["radar_token_encoder"] > 0.0,
    }
    report = {
        "protocol": "RaLD-anchor RH0.5 native integration; not an RH1 result",
        "source_commit": args.source_commit,
        "device": device_name,
        "cube": {"path": str(cube_path), "shape": list(cube.shape)},
        "parent": {
            "run": str(args.parent_g1_run),
            "checkpoint": str(parent_checkpoint_path),
            "git_commit": parent_document["provenance"]["git_commit"],
            "mode": parent_config["mode"],
            "parameter_count": parameter_count(parent),
        },
        "refiner_parameter_count": parameter_count(model.refiner),
        "radar_encoder_parameter_count": parameter_count(model.radar_encoder),
        "initial": initial_checks,
        "gradient_steps": gradients,
        "peak_cuda_memory_bytes": peak_memory,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
