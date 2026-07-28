#!/usr/bin/env python3
"""Run the D-MHW max-cardinality synthetic mechanism preflight on H200."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmhw.contracts import (  # noqa: E402
    INTERVENTION_NAMES,
    apply_dmhw_intervention,
    audit_direct_horizon_supervision,
)
from dmhw.synthetic_contract import (  # noqa: E402
    make_dmhw_synthetic_contract,
)
from losses.dmhw_objective import dmhw_stage0_loss  # noqa: E402
from models.dmhw_direct_world import (  # noqa: E402
    DirectMultiHorizonDopplerWorld,
)


PROTOCOL = "dmhw_direct_multi_horizon_synthetic_h200_preflight_v1"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SOURCE_FILES = (
    "code/dmhw/__init__.py",
    "code/dmhw/contracts.py",
    "code/dmhw/synthetic_contract.py",
    "code/models/dmhw_direct_world.py",
    "code/losses/dmhw_objective.py",
    "code/scripts/preflight_dmhw_direct_world.py",
    "code/tests/test_dmhw_direct_world.py",
    "docs/dmhw_direct_multi_horizon_stage0.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-manifest", type=Path)
    parser.add_argument("--raw-data-root", type=Path)
    parser.add_argument("--train-target-cache-root", type=Path)
    parser.add_argument("--validation-target-cache-root", type=Path)
    return parser.parse_args()


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("D-MHW source commit must be a full lowercase SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("D-MHW source commit differs from checked-out HEAD")
    dirty = git_output(
        repo, "status", "--porcelain", "--untracked-files=no"
    )
    if dirty:
        raise ValueError(f"D-MHW source tree is dirty: {dirty}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("D-MHW preflight requires CUDA")
    device = torch.device(device_name)
    name = torch.cuda.get_device_name(device)
    if "H200" not in name.upper():
        raise RuntimeError(f"D-MHW preflight requires H200, got {name}")
    return device, name


def count_nonzero_rows(gradient: torch.Tensor | None) -> int:
    if gradient is None:
        return 0
    flat = gradient.detach().abs().reshape(gradient.shape[0], -1)
    return int((flat.sum(dim=1) > 0.0).sum().item())


def intervention_probe(
    model: DirectMultiHorizonDopplerWorld,
    base_inputs: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    name: str,
) -> dict:
    intervened = apply_dmhw_intervention(
        base_inputs, name, model.doppler_mps
    )
    output = model(**intervened)
    xyz_delta = (
        output["xyz_m"] - baseline["xyz_m"]
    ).abs().mean(dim=(0, 2, 3))
    doppler_delta = (
        output["doppler_probability"]
        - baseline["doppler_probability"]
    ).abs().mean(dim=(0, 2, 3))
    confidence_delta = (
        output["confidence"] - baseline["confidence"]
    ).abs().mean(dim=(0, 2))
    combined = xyz_delta + doppler_delta + confidence_delta
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        combined.sum(),
        parameters,
        allow_unused=True,
    )
    gradient_norm = sum(
        float(gradient.detach().abs().sum().item())
        for gradient in gradients
        if gradient is not None
    )
    return {
        "xyz_mean_abs_delta_m_by_horizon": [
            float(value.item()) for value in xyz_delta
        ],
        "doppler_mean_abs_delta_by_horizon": [
            float(value.item()) for value in doppler_delta
        ],
        "confidence_mean_abs_delta_by_horizon": [
            float(value.item()) for value in confidence_delta
        ],
        "combined_delta_by_horizon": [
            float(value.item()) for value in combined
        ],
        "delta_requires_grad": bool(combined.requires_grad),
        "model_gradient_l1": gradient_norm,
        "all_horizons_measurable": bool(
            torch.all(combined > 1e-8).item()
        ),
        "finite_nonzero_graph_gradient": (
            gradient_norm > 0.0
            and torch.isfinite(
                torch.tensor(gradient_norm)
            ).item()
        ),
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    verify_source(repo, args.source_commit)
    if args.output.exists():
        raise FileExistsError(f"D-MHW output exists: {args.output}")
    device, device_name = require_h200(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    inputs, targets, axes = make_dmhw_synthetic_contract(
        device,
        point_count=10_000,
        persistent_fraction=0.70,
        batch_size=2,
        history_frame_count=3,
        full_raed=True,
        seed=args.seed,
    )
    for value in inputs.values():
        if torch.is_floating_point(value):
            value.requires_grad_(True)
    model = DirectMultiHorizonDopplerWorld(
        *axes,
        point_count=10_000,
        persistent_fraction=0.70,
        hidden_dim=96,
        maximum_tangential_speed_mps=8.0,
        maximum_radial_correction_mps=0.75,
    ).to(device)
    model.train()
    started = time.monotonic()
    output = model(**inputs)
    objective = dmhw_stage0_loss(
        output,
        **targets,
        aligned_synthetic_attributes=True,
    )
    objective.total.backward()
    torch.cuda.synchronize(device)

    current_gradient = inputs["current_cube_drae"].grad
    history_gradient = inputs["history_cube_drae"].grad
    current_channel_gradient = current_gradient.abs().sum(
        dim=(0, 2, 3, 4)
    )
    history_channel_gradient = history_gradient.abs().sum(
        dim=(0, 1, 3, 4, 5)
    )
    persistent_doppler_rows = count_nonzero_rows(
        model.persistent_doppler_head.weight.grad
    )
    birth_doppler_rows = count_nonzero_rows(
        model.birth_doppler_head.weight.grad
    )
    horizon_gradient = sum(
        float(parameter.grad.detach().abs().sum().item())
        for parameter in model.horizon_encoder.parameters()
        if parameter.grad is not None
    )
    baseline = {
        "xyz_m": output["xyz_m"].detach(),
        "doppler_probability": output[
            "doppler_probability"
        ].detach(),
        "confidence": output["confidence"].detach(),
    }
    loss_values = {
        name: float(value.detach().item())
        for name, value in objective.components.items()
    }
    loss_values["total"] = float(objective.total.detach().item())
    selected_ids = output["persistent_source_id"].detach()
    persistent_ids_equal = bool(
        torch.equal(selected_ids[:, 0], selected_ids[:, 1])
        and torch.equal(selected_ids[:, 1], selected_ids[:, 2])
    )
    tangential_radial_dot = (
        output["tangential_velocity_mps"]
        * output["radial_direction"]
    ).sum(dim=-1)
    maximum_tangential = float(
        torch.linalg.vector_norm(
            output["tangential_velocity_mps"], dim=-1
        ).max().detach().item()
    )
    maximum_radial_correction = float(
        output["radial_correction_mps"].abs().max().detach().item()
    )
    analytic_displacement_present = bool(
        torch.any(
            output["analytic_radial_displacement_m"].abs() > 1e-8
        ).item()
    )
    del output
    del objective
    for parameter in model.parameters():
        parameter.grad = None
    for value in inputs.values():
        if value.grad is not None:
            value.grad = None

    base_inputs = {
        name: value.detach() for name, value in inputs.items()
    }
    interventions = {
        name: intervention_probe(model, base_inputs, baseline, name)
        for name in INTERVENTION_NAMES
    }
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    all_interventions = all(
        result["all_horizons_measurable"]
        and result["delta_requires_grad"]
        and result["finite_nonzero_graph_gradient"]
        for result in interventions.values()
    )

    data_audit = None
    if args.temporal_manifest is not None:
        cache_roots = {}
        if args.train_target_cache_root is not None:
            cache_roots["train"] = args.train_target_cache_root
        if args.validation_target_cache_root is not None:
            cache_roots["validation"] = (
                args.validation_target_cache_root
            )
        data_audit = audit_direct_horizon_supervision(
            args.temporal_manifest,
            raw_data_root=args.raw_data_root,
            target_cache_roots=cache_roots,
        )

    checks = {
        "h200_device": True,
        "full_raed_shape": tuple(
            inputs["current_cube_drae"].shape[1:]
        )
        == (64, 256, 107, 37),
        "exact_three_horizons": baseline["xyz_m"].shape[1] == 3,
        "exact_10000_points_each_horizon": tuple(
            baseline["xyz_m"].shape[2:]
        )
        == (10_000, 3),
        "persistent_birth_split_7000_3000": (
            model.persistent_count == 7_000
            and model.birth_count == 3_000
        ),
        "persistent_ids_equal_across_horizons": persistent_ids_equal,
        "analytic_radial_displacement_present": (
            analytic_displacement_present
        ),
        "tangential_projection": float(
            tangential_radial_dot.abs().max().detach().item()
        )
        < 1e-5,
        "bounded_tangential_speed": maximum_tangential <= 8.0 + 1e-5,
        "bounded_radial_correction": (
            maximum_radial_correction <= 0.75 + 1e-5
        ),
        "finite_loss": all(
            torch.isfinite(torch.tensor(value)).item()
            for value in loss_values.values()
        ),
        "all_64_current_cube_channels_receive_gradient": int(
            (current_channel_gradient > 0.0).sum().item()
        )
        == 64,
        "all_64_history_cube_channels_receive_gradient": int(
            (history_channel_gradient > 0.0).sum().item()
        )
        == 64,
        "all_64_persistent_doppler_rows_receive_gradient": (
            persistent_doppler_rows == 64
        ),
        "all_64_birth_doppler_rows_receive_gradient": (
            birth_doppler_rows == 64
        ),
        "all_horizon_encoder_parameters_receive_gradient": (
            horizon_gradient > 0.0
        ),
        "five_interventions_measurable_and_differentiable": (
            all_interventions and len(interventions) == 5
        ),
        "direct_not_autoregressive": (
            baseline["xyz_m"].shape[1] == 3
        ),
        "peak_allocated_below_55_gib": (
            peak_allocated_bytes < 55 * 1024**3
        ),
        "peak_reserved_below_65_gib": (
            peak_reserved_bytes < 65 * 1024**3
        ),
    }
    mechanism_passed = all(checks.values())
    formal_blockers = [
        "no frozen geometry parent has passed the full project gate",
        (
            "future birth-point Doppler supervision is unresolved without "
            "reading a future Cube"
        ),
    ]
    if data_audit is None:
        formal_blockers.append("real three-horizon data audit not supplied")
    elif not data_audit["prepared_training_cache_complete"]:
        formal_blockers.append(
            "prepared train/validation dense-target cache is incomplete"
        )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "source_hashes": {
            path: sha256(repo / path) for path in SOURCE_FILES
        },
        "seed": args.seed,
        "runtime": {
            "device": args.device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "elapsed_seconds": elapsed,
            "cuda_peak_allocated_bytes": peak_allocated_bytes,
            "cuda_peak_reserved_bytes": peak_reserved_bytes,
        },
        "model": {
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "point_count": model.point_count,
            "persistent_count": model.persistent_count,
            "birth_count": model.birth_count,
            "horizons_seconds": [
                float(value)
                for value in model.horizons_seconds.detach().cpu()
            ],
            "input_cube_shape": list(
                inputs["current_cube_drae"].shape
            ),
            "history_cube_shape": list(
                inputs["history_cube_drae"].shape
            ),
        },
        "loss": loss_values,
        "gradients": {
            "current_cube_nonzero_channel_count": int(
                (current_channel_gradient > 0.0).sum().item()
            ),
            "history_cube_nonzero_channel_count": int(
                (history_channel_gradient > 0.0).sum().item()
            ),
            "persistent_doppler_nonzero_row_count": (
                persistent_doppler_rows
            ),
            "birth_doppler_nonzero_row_count": birth_doppler_rows,
            "horizon_encoder_gradient_l1": horizon_gradient,
        },
        "motion": {
            "maximum_tangential_speed_mps": maximum_tangential,
            "maximum_radial_correction_mps": (
                maximum_radial_correction
            ),
            "maximum_tangential_radial_dot": float(
                tangential_radial_dot.abs().max().detach().item()
            ),
        },
        "interventions": interventions,
        "data_supervision_audit": data_audit,
        "checks": checks,
        "mechanism_preflight_passed": mechanism_passed,
        "formal_training_authorized": False,
        "formal_training_blockers": formal_blockers,
        "route_status": (
            "mechanism_preflight_passed_real_training_locked"
            if mechanism_passed
            else "mechanism_preflight_failed"
        ),
        "evidence_boundary": {
            "synthetic_only": True,
            "real_training_run": False,
            "real_validation_metric": False,
            "future_cube_read_by_model": False,
            "test_accessed": False,
            "attribute_targets": (
                "synthetic point-aligned labels only; not a real-data claim"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)
    if not mechanism_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
