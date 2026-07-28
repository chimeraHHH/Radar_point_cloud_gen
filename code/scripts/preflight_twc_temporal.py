#!/usr/bin/env python3
"""Run an H200 synthetic preflight for the forced-temporal T-WC scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.twc_physics import twc_physics_loss  # noqa: E402
from losses.wrong_condition import (  # noqa: E402
    WRONG_CONDITIONS,
    apply_wrong_condition,
    wrong_condition_margin_loss,
)
from models.cube_doppler import query_cube_spectrum  # noqa: E402
from models.twc_temporal import ForcedTemporalTWC  # noqa: E402


PROTOCOL = "twc_forced_temporal_synthetic_h200_preflight_v1"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("T-WC preflight requires CUDA")
    device = torch.device(device_name)
    name = torch.cuda.get_device_name(device)
    if "H200" not in name.upper():
        raise RuntimeError(f"T-WC preflight requires H200, got {name}")
    return device, name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("T-WC source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("T-WC source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"T-WC preflight source worktree is dirty: {dirty}")


def synthetic_case(
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    range_m = torch.linspace(0.0, 80.0, 24, device=device)
    azimuth_rad = torch.linspace(-1.0, 1.0, 13, device=device)
    elevation_rad = torch.linspace(-0.35, 0.35, 7, device=device)
    doppler_mps = torch.linspace(-8.0, 8.0, 64, device=device)
    batch_size, frame_count, history_count = 2, 4, 7_000
    cube = torch.rand(
        batch_size,
        doppler_mps.numel(),
        range_m.numel(),
        azimuth_rad.numel(),
        elevation_rad.numel(),
        generator=generator,
        device=device,
    ).requires_grad_(True)
    point_index = torch.arange(
        history_count, dtype=torch.float32, device=device
    )
    point_fraction = point_index / max(history_count - 1, 1)
    radius = 10.0 + 60.0 * point_fraction
    azimuth = -0.8 + 1.6 * point_fraction
    elevation = 0.12 * torch.sin(point_fraction * 8.0 * torch.pi)
    base_xyz = torch.stack(
        (
            radius * torch.cos(elevation) * torch.cos(azimuth),
            radius * torch.cos(elevation) * torch.sin(azimuth),
            radius * torch.sin(elevation),
        ),
        dim=1,
    )
    history_xyz = []
    for batch_index in range(batch_size):
        frames = []
        for frame_index in range(frame_count):
            shift = base_xyz.new_tensor(
                [0.4 * frame_index, 0.1 * batch_index, 0.0]
            )
            frames.append(base_xyz + shift)
        history_xyz.append(torch.stack(frames))
    history_xyz_m = torch.stack(history_xyz)
    doppler_index = (
        torch.arange(history_count, device=device) * 3 + 37
    ) % doppler_mps.numel()
    probability = torch.nn.functional.one_hot(
        doppler_index, doppler_mps.numel()
    ).float()
    history_probability = probability.view(
        1, 1, history_count, -1
    ).expand(batch_size, frame_count, -1, -1).clone()
    history_probability[1] = torch.roll(
        history_probability[1], shifts=5, dims=-1
    )
    history_confidence = torch.linspace(
        0.55, 0.99, history_count, device=device
    ).view(1, 1, -1).expand(batch_size, frame_count, -1).clone()
    history_confidence[:, :-1] *= 0.9
    source_id = torch.arange(
        history_count, dtype=torch.long, device=device
    ).view(1, 1, -1).expand(batch_size, frame_count, -1).clone()
    source_id += (
        torch.arange(batch_size, device=device)[:, None, None] * 10_000
        + torch.arange(frame_count, device=device)[None, :, None]
        * history_count
    )
    current_from_history = torch.eye(
        4, device=device
    )[None].repeat(batch_size, 1, 1)
    current_from_history[:, 0, 3] = 0.25
    inputs = {
        "conditioning_cube_drae": cube,
        "history_xyz_m": history_xyz_m,
        "history_doppler_probability": history_probability,
        "history_confidence": history_confidence,
        "history_source_id": source_id,
        "current_from_history": current_from_history,
        "delta_seconds": torch.full(
            (batch_size,), 0.1, device=device
        ),
    }
    return inputs, (
        range_m,
        azimuth_rad,
        elevation_rad,
        doppler_mps,
    )


def per_example_discrepancy(
    output_xyz_m: torch.Tensor,
    target_xyz_m: torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        output_xyz_m - target_xyz_m, dim=-1
    ).mean(dim=1)


def local_output_doppler_target(
    cube_drae: torch.Tensor,
    coordinates_rae: torch.Tensor,
) -> torch.Tensor:
    batch_size, point_count, coordinate_dim = coordinates_rae.shape
    if coordinate_dim != 3:
        raise ValueError("T-WC output coordinates must end in RAE")
    batch_index = torch.arange(
        batch_size,
        device=coordinates_rae.device,
    )[:, None].expand(-1, point_count)
    query = torch.cat(
        (
            batch_index.reshape(-1, 1).to(coordinates_rae),
            coordinates_rae.reshape(-1, 3),
        ),
        dim=1,
    )
    return query_cube_spectrum(cube_drae, query).reshape(
        batch_size,
        point_count,
        -1,
    )


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    inputs, axes = synthetic_case(device, args.seed)
    model = ForcedTemporalTWC(
        *axes,
        point_count=10_000,
        persistent_fraction=0.70,
        hidden_dim=64,
        maximum_tangential_speed_mps=8.0,
        maximum_radial_correction_mps=0.75,
    ).to(device).train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    matched = model(
        **inputs,
        mode="online_enhancement",
    )
    target = matched["xyz_m"].detach() + matched["xyz_m"].new_tensor(
        [0.05, -0.03, 0.02]
    )
    matched_discrepancy = per_example_discrepancy(
        matched["xyz_m"], target
    )
    intervention_delta = {}
    wrong_outputs = {}
    for condition in WRONG_CONDITIONS:
        wrong_history = apply_wrong_condition(
            history_xyz_m=inputs["history_xyz_m"],
            history_doppler_probability=(
                inputs["history_doppler_probability"]
            ),
            history_confidence=inputs["history_confidence"],
            history_source_id=inputs["history_source_id"],
            condition=condition,
            doppler_mps=axes[-1],
        )
        wrong_input = {**inputs, **wrong_history}
        wrong = model(
            **wrong_input,
            mode="online_enhancement",
        )
        wrong_outputs[condition] = wrong
        intervention_delta[condition] = float(
            (
                wrong["persistent_xyz_m"]
                - matched["persistent_xyz_m"]
            )
            .abs()
            .mean()
            .detach()
            .item()
        )
    replacement_discrepancy = per_example_discrepancy(
        wrong_outputs["history_replacement"]["xyz_m"],
        target,
    )
    ranking = wrong_condition_margin_loss(
        matched_discrepancy,
        replacement_discrepancy,
        margin=0.05,
    )
    physics = twc_physics_loss(matched)
    local_doppler_target = local_output_doppler_target(
        inputs["conditioning_cube_drae"],
        matched["coordinates_rae"],
    ).detach()
    doppler_cross_entropy = -(
        local_doppler_target
        * torch.log_softmax(matched["doppler_logit"].float(), dim=-1)
    ).sum(dim=-1).mean()
    confidence_bce = F.binary_cross_entropy_with_logits(
        matched["confidence_logit"].float(),
        torch.ones_like(matched["confidence_logit"]).float(),
    )
    total_loss = (
        matched_discrepancy.mean()
        + ranking.total
        + physics.total
        + 0.25 * doppler_cross_entropy
        + 0.05 * confidence_bce
    )
    total_loss.backward()
    forecast = model(
        **inputs,
        mode="forecast",
        target_offset_steps=1,
    )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.monotonic() - started
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    cube_gradient = inputs["conditioning_cube_drae"].grad
    cube_channel_gradient = (
        cube_gradient.detach().abs().sum(dim=(0, 2, 3, 4))
        if cube_gradient is not None
        else torch.zeros(64, device=device)
    )
    persistent_doppler_gradient = (
        model.persistent_doppler_head.weight.grad
    )
    birth_doppler_gradient = model.birth_doppler_head.weight.grad
    persistent_doppler_row_gradient = (
        persistent_doppler_gradient.detach().abs().sum(dim=1)
        if persistent_doppler_gradient is not None
        else torch.zeros(64, device=device)
    )
    birth_doppler_row_gradient = (
        birth_doppler_gradient.detach().abs().sum(dim=1)
        if birth_doppler_gradient is not None
        else torch.zeros(64, device=device)
    )
    persistent_mask = matched["persistent_mask"]
    radial_direction = (
        matched["selected_history_xyz_m"]
        / torch.linalg.vector_norm(
            matched["selected_history_xyz_m"],
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
    )
    tangential_dot = (
        matched["tangential_velocity_mps"] * radial_direction
    ).sum(dim=-1).abs()
    checks = {
        "h200_device": "H200" in device_name.upper(),
        "exact_point_count": matched["xyz_m"].shape == (2, 10_000, 3),
        "persistent_birth_split": (
            int(persistent_mask.sum(dim=1).min().item()) == 7_000
            and int((~persistent_mask).sum(dim=1).min().item()) == 3_000
        ),
        "persistent_source_ids_valid": bool(
            (
                matched["history_source_id"][persistent_mask] >= 0
            ).all().item()
        ),
        "birth_source_ids_are_sentinel": bool(
            (
                matched["history_source_id"][~persistent_mask] == -1
            ).all().item()
        ),
        "analytic_radial_displacement_present": bool(
            (
                matched["analytic_radial_displacement_m"].abs() > 1e-6
            ).any().item()
        ),
        "tangential_projection": float(tangential_dot.max().item()) < 1e-4,
        "bounded_radial_correction": float(
            matched["radial_correction_mps"].abs().max().item()
        )
        <= model.maximum_radial_correction_mps + 1e-6,
        "all_interventions_change_output": all(
            value > 1e-7 for value in intervention_delta.values()
        ),
        "finite_loss": bool(torch.isfinite(total_loss).item()),
        "finite_nonzero_gradients": (
            bool(gradients)
            and all(
                bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            )
            and any(
                bool(torch.count_nonzero(gradient).item())
                for gradient in gradients
            )
        ),
        "all_64_cube_channels_receive_gradient": (
            cube_gradient is not None
            and bool(torch.isfinite(cube_gradient).all().item())
            and int(torch.count_nonzero(cube_channel_gradient).item()) == 64
        ),
        "all_64_persistent_doppler_rows_receive_gradient": (
            bool(torch.isfinite(persistent_doppler_row_gradient).all().item())
            and int(
                torch.count_nonzero(
                    persistent_doppler_row_gradient
                ).item()
            )
            == 64
        ),
        "all_64_birth_doppler_rows_receive_gradient": (
            bool(torch.isfinite(birth_doppler_row_gradient).all().item())
            and int(
                torch.count_nonzero(birth_doppler_row_gradient).item()
            )
            == 64
        ),
        "online_metadata": (
            matched["metadata"]["mode"] == "online_enhancement"
            and matched["metadata"]["conditioning_cube_role"]
            == "target_current_cube"
            and not matched["metadata"]["uses_future_cube"]
        ),
        "forecast_metadata": (
            forecast["metadata"]["mode"] == "forecast"
            and forecast["metadata"]["conditioning_cube_role"]
            == "last_observed_cube"
            and forecast["metadata"]["target_offset_steps"] == 1
            and not forecast["metadata"]["uses_future_cube"]
        ),
    }
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in (
                repo / "code/models/twc_temporal.py",
                repo / "code/losses/twc_physics.py",
                repo / "code/losses/wrong_condition.py",
                Path(__file__).resolve(),
            )
        },
        "seed": args.seed,
        "runtime": {
            "device": args.device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "elapsed_seconds": elapsed_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                device
            ),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                device
            ),
        },
        "model": {
            "point_count": model.point_count,
            "persistent_count": model.persistent_count,
            "birth_count": model.birth_count,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "loss": {
            "total": float(total_loss.detach().item()),
            "wrong_condition_margin": float(ranking.total.detach().item()),
            "doppler_cross_entropy": float(
                doppler_cross_entropy.detach().item()
            ),
            "confidence_bce": float(confidence_bce.detach().item()),
            "inverse_radial": float(
                physics.components["inverse_radial"].detach().item()
            ),
            "motion_cycle": float(
                physics.components["motion_cycle"].detach().item()
            ),
        },
        "intervention_mean_abs_persistent_xyz_delta_m": (
            intervention_delta
        ),
        "gradients": {
            "cube_nonzero_channel_count": int(
                torch.count_nonzero(cube_channel_gradient).item()
            ),
            "persistent_doppler_nonzero_row_count": int(
                torch.count_nonzero(
                    persistent_doppler_row_gradient
                ).item()
            ),
            "birth_doppler_nonzero_row_count": int(
                torch.count_nonzero(birth_doppler_row_gradient).item()
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(
                f"T-WC preflight output exists: {args.output}"
            )
        atomic_json(args.output, document)
    print(json.dumps(document, indent=2), flush=True)
    if not document["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
