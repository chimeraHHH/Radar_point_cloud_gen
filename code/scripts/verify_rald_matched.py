#!/usr/bin/env python3
"""Verify full-scale matched RaLD modules and native-grid CUDA behavior."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_matched import (  # noqa: E402
    RaLDEDMPreconditioner,
    RaLDPointAutoencoder,
    RadarTokenEncoder,
    edm_loss,
)


def finite_nonzero(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def gradient_norm(parameters) -> float | None:
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return None
    return float(sum(gradient.square().sum() for gradient in gradients).sqrt().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpu-name", default="NVIDIA H200 NVL")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Matched RaLD verification requires CUDA")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    if device_name != args.required_gpu_name:
        raise RuntimeError(
            f"Matched RaLD verification requires {args.required_gpu_name}, got {device_name}"
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    torch.manual_seed(20260716)
    torch.cuda.manual_seed_all(20260716)
    torch.cuda.reset_peak_memory_stats(device)
    if args.small:
        point_count = 32
        latent_count = 16
        model_dim = 64
        latent_dim = 8
        depth = 2
        heads = 4
        head_dim = 16
        radar_shape = (16, 16, 8)
        encoded_shape = (4, 4, 2)
        radar_base_channels = 8
        radar_encoded_channels = 8
        radar_multipliers = (1, 1, 2)
    else:
        point_count = 2_048
        latent_count = 512
        model_dim = 512
        latent_dim = 32
        depth = 24
        heads = 8
        head_dim = 64
        radar_shape = (256, 107, 37)
        encoded_shape = (16, 7, 3)
        radar_base_channels = 64
        radar_encoded_channels = 16
        radar_multipliers = (1, 1, 2, 2, 4)

    autoencoder = RaLDPointAutoencoder(
        point_count=point_count,
        latent_count=latent_count,
        model_dim=model_dim,
        latent_dim=latent_dim,
        depth=depth,
        heads=heads,
        head_dim=head_dim,
    ).to(device)
    points = torch.rand(1, point_count, 3, device=device) * 2.0 - 1.0
    queries = torch.rand(1, min(point_count, 2_048), 3, device=device) * 2.0 - 1.0
    autoencoder.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, posterior = autoencoder(points, queries)
        ae_loss = logits.square().mean() + 1e-3 * posterior.kl().mean()
    ae_loss.backward()
    ae_gradient = gradient_norm(autoencoder.parameters())
    ae_peak = torch.cuda.max_memory_allocated(device)
    ae_report = {
        "parameters": parameter_count(autoencoder),
        "posterior_shape": list(posterior.mean.shape),
        "query_logit_shape": list(logits.shape),
        "loss": float(ae_loss.item()),
        "gradient_norm": ae_gradient,
        "peak_cuda_memory_bytes": ae_peak,
    }
    del logits, posterior, ae_loss, points, queries
    autoencoder.zero_grad(set_to_none=True)
    del autoencoder
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    radar_encoder = RadarTokenEncoder(
        encoded_shape=encoded_shape,
        encoded_channels=radar_encoded_channels,
        token_dim=model_dim,
        base_channels=radar_base_channels,
        channel_multipliers=radar_multipliers,
        blocks_per_level=2 if not args.small else 1,
    )
    edm = RaLDEDMPreconditioner(
        latent_count=latent_count,
        latent_dim=latent_dim,
        model_dim=model_dim,
        depth=depth,
        heads=heads,
        head_dim=head_dim,
        radar_encoder=radar_encoder,
    ).to(device)
    latent = torch.randn(1, latent_count, latent_dim, device=device)
    radar = torch.randn(1, 1, *radar_shape, device=device)
    optimizer = torch.optim.SGD(edm.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first_loss = edm_loss(edm, latent, radar)
    first_loss.backward()
    output_gradient = gradient_norm(edm.denoiser.output.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        second_loss = edm_loss(edm, latent, radar)
    second_loss.backward()
    radar_gradient = gradient_norm(edm.radar_encoder.parameters())
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = edm.encode_condition(radar)
        denoised = edm(latent, torch.ones(1, device=device), radar)
    torch.cuda.synchronize(device)
    edm_report = {
        "parameters": parameter_count(edm),
        "radar_token_shape": list(tokens.shape),
        "denoised_shape": list(denoised.shape),
        "first_loss": float(first_loss.item()),
        "second_loss": float(second_loss.item()),
        "first_step_output_gradient_norm": output_gradient,
        "second_step_radar_gradient_norm": radar_gradient,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    checks = {
        "h200_identity": device_name == args.required_gpu_name,
        "ae_shape": ae_report["posterior_shape"]
        == [1, latent_count, latent_dim],
        "ae_gradient_finite_nonzero": finite_nonzero(ae_gradient),
        "radar_token_shape": edm_report["radar_token_shape"]
        == [1, encoded_shape[0] * encoded_shape[1] * encoded_shape[2], model_dim],
        "edm_shape": edm_report["denoised_shape"]
        == [1, latent_count, latent_dim],
        "zero_output_receives_gradient": finite_nonzero(output_gradient),
        "radar_condition_receives_second_step_gradient": finite_nonzero(
            radar_gradient
        ),
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "device": device_name,
        "small": args.small,
        "autoencoder": ae_report,
        "edm": edm_report,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
