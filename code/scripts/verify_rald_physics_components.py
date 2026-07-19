#!/usr/bin/env python3
"""Verify full-scale RaLD-inspired Full-RAED and physical-query components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_tesseract  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_matched import (  # noqa: E402
    FullRAEDRadarTokenEncoder,
    RaLDAnchorLatentRefiner,
    RaLDEDMPreconditioner,
    RaLDPhysicalQueryHead,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--radar-index", type=int, default=232)
    parser.add_argument("--log-center", type=float, required=True)
    parser.add_argument("--log-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpu-name", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("RaLD physical component verification requires CUDA")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    if device_name != args.required_gpu_name:
        raise RuntimeError(
            f"Required GPU {args.required_gpu_name}, got {device_name}"
        )
    cube_path = (
        args.data_root
        / str(args.sequence)
        / "radar_tesseract"
        / f"tesseract_{args.radar_index:05d}.mat"
    )
    cube = torch.from_numpy(load_tesseract(cube_path)).float().unsqueeze(0).to(device)
    encoder = FullRAEDRadarTokenEncoder(
        log_center=args.log_center,
        log_scale=args.log_scale,
    ).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = encoder(cube)
        token_loss = tokens.float().square().mean()
    token_loss.backward()
    spectral_gradient = encoder.spectral_projection.weight.grad
    gradient_per_bin = (
        spectral_gradient.detach().float().abs().flatten(2).sum(dim=(0, 2))
    )
    active_doppler_bins = int(gradient_per_bin.gt(0).sum().item())
    token_peak_memory = torch.cuda.max_memory_allocated(device)
    token_shape = list(tokens.shape)
    del encoder, tokens, token_loss, spectral_gradient
    torch.cuda.empty_cache()

    full_encoder = FullRAEDRadarTokenEncoder(
        log_center=args.log_center,
        log_scale=args.log_scale,
    )
    model = RaLDEDMPreconditioner(radar_encoder=full_encoder).to(device)
    noisy_latent = torch.randn(1, 512, 32, device=device)
    sigma = torch.ones(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        denoised = model(noisy_latent, sigma, cube)
    edm_peak_memory = torch.cuda.max_memory_allocated(device)
    edm_shape = list(denoised.shape)

    head = RaLDPhysicalQueryHead().to(device)
    query_features = torch.randn(1, 1_024, 512, device=device)
    local_spectrum = torch.rand(1, 1_024, 64, device=device)
    physical = head(query_features, local_spectrum)
    measured_probability = local_spectrum / local_spectrum.sum(
        dim=-1, keepdim=True
    )
    spectrum_initialization_error = float(
        (
            physical["doppler_probability"].float()
            - measured_probability.float()
        )
        .abs()
        .max()
        .item()
    )

    hybrid = RaLDAnchorLatentRefiner(anchor_feature_dim=8).to(device)
    anchor_coordinates = torch.rand(1, 10_000, 3, device=device) * 2.0 - 1.0
    anchor_features = torch.randn(1, 10_000, 8, device=device)
    anchor_spectrum = torch.rand(1, 10_000, 64, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hybrid_output = hybrid(
            anchor_coordinates,
            anchor_features,
            anchor_spectrum,
        )
    hybrid_peak_memory = torch.cuda.max_memory_allocated(device)
    checks = {
        "native_token_shape": token_shape == [1, 336, 512],
        "all_doppler_bins_receive_gradient": active_doppler_bins == 64,
        "edm_latent_shape": edm_shape == [1, 512, 32],
        "physical_spectrum_identity": spectrum_initialization_error <= 1e-5,
        "zero_initial_offset": bool(
            torch.count_nonzero(physical["offset_bins"]).item() == 0
        ),
        "neutral_initial_confidence": bool(
            torch.count_nonzero(physical["confidence_logit"]).item() == 0
        ),
        "hybrid_latent_shape": list(hybrid_output["latent"].shape)
        == [1, 512, 512],
        "hybrid_point_shape": list(hybrid_output["offset_bins"].shape)
        == [1, 10_000, 3],
    }
    report = {
        "protocol": "RaLD-inspired physical mainline R0",
        "source_commit": args.source_commit,
        "upstream_rald_commit": "ffec4b41241391734b1eda5c093de843c909eb8e",
        "device": device_name,
        "cube": {
            "path": str(cube_path),
            "shape": list(cube.shape),
        },
        "full_raed_tokens": {
            "shape": token_shape,
            "active_doppler_gradient_bins": active_doppler_bins,
            "peak_cuda_memory_bytes": token_peak_memory,
        },
        "edm": {
            "shape": edm_shape,
            "parameter_count": parameter_count(model),
            "peak_cuda_memory_bytes": edm_peak_memory,
        },
        "physical_head": {
            "parameter_count": parameter_count(head),
            "spectrum_initialization_max_abs_error": spectrum_initialization_error,
        },
        "anchor_hybrid": {
            "anchor_count": 10_000,
            "latent_shape": list(hybrid_output["latent"].shape),
            "offset_shape": list(hybrid_output["offset_bins"].shape),
            "parameter_count": parameter_count(hybrid),
            "peak_cuda_memory_bytes": hybrid_peak_memory,
        },
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
