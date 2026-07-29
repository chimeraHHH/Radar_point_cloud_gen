#!/usr/bin/env python3
"""Run the frozen R-B2 Cube-only one-frame voxel-slot overfit pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_tesseract  # noqa: E402
from eval.dense_geometry import geometry_report  # noqa: E402
from losses.rb2_voxel_slot import (  # noqa: E402
    build_voxel_slot_targets,
    candidate_support_report,
    rb2_voxel_slot_loss,
)
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rb2_voxel_slot_model import (  # noqa: E402
    DEFAULT_CANDIDATE_VOXEL_QUOTAS,
    EXPORT_POINT_COUNT,
    MINIMUM_EXPORT_DISTANCE_M,
    OUTPUT_POINT_QUOTAS,
    PROTOCOL,
    RB2VoxelSlotNet,
    SLOTS_PER_VOXEL,
    VOXEL_SIZE_M,
    candidate_voxels_from_cube,
    export_exact_voxel_slots,
    lattice_indices_to_centers,
    range_stratum_codes,
)


FORMAL_SEED = 20260716
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
FORMAL_TRAIN_COUNT = 76
FORMAL_VALIDATION_COUNT = 24
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_CACHE_ARRAYS = ("target_xyz_confidence",)
FORBIDDEN_RECORD_KEY_FRAGMENTS = ("future", "next_cube", "test")


@dataclass(frozen=True)
class TinyConfig:
    protocol: str = PROTOCOL
    seed: int = FORMAL_SEED
    train_frame_count: int = 1
    evaluation_frame_count: int = 1
    maximum_updates: int = 500
    evaluation_interval_updates: int = 50
    required_consecutive_gate_passes: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 10.0
    hidden_dim: int = 192
    depth: int = 4
    voxel_size_m: float = VOXEL_SIZE_M
    slots_per_voxel: int = SLOTS_PER_VOXEL
    candidate_voxel_quotas: tuple[int, int, int] = (
        DEFAULT_CANDIDATE_VOXEL_QUOTAS
    )
    output_point_quotas: tuple[int, int, int] = OUTPUT_POINT_QUOTAS
    seed_multiplier: int = 8
    maximum_neighbor_radius: int = 4
    minimum_candidate_occupied_voxel_recall: float = 0.20
    minimum_candidate_confidence_coverage: float = 0.30
    maximum_chamfer_m: float = 1.00
    maximum_completeness_mean_m: float = 0.75
    maximum_outlier_fraction_2m: float = 0.10
    minimum_export_distance_m: float = MINIMUM_EXPORT_DISTANCE_M
    numeric_mode: str = "fp32_parameters_bf16_cuda_autocast"
    cube_only_inference: bool = True
    target_used_for_candidate_or_export: bool = False
    cfar_accessed: bool = False
    future_cube_accessed: bool = False
    test_accessed: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    temporary.replace(path)


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("R-B2 source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("R-B2 source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"R-B2 tracked source tree is dirty: {dirty}")


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("R-B2 tiny training requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("R-B2 tiny training is H200 CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"R-B2 tiny training requires H200, got {resolved}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("R-B2 tiny training requires BF16 support")
    return device, resolved


def validate_inputs(
    manifest_path: Path,
    scene_split_path: Path,
    normalization_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    hashes = {
        "manifest_sha256": sha256(manifest_path),
        "scene_split_sha256": sha256(scene_split_path),
        "normalization_sha256": sha256(normalization_path),
    }
    if hashes["manifest_sha256"] != FROZEN_MANIFEST_SHA256:
        raise ValueError("R-B2 manifest differs from the frozen 100-frame cohort")
    if hashes["scene_split_sha256"] != FROZEN_SCENE_SPLIT_SHA256:
        raise ValueError("R-B2 scene split differs from the frozen leakage audit")
    scene_split = json.loads(scene_split_path.read_text(encoding="utf-8"))
    if scene_split.get("gate_pass") is not True:
        raise ValueError("R-B2 scene split failed its leakage gate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("frames")
    if not isinstance(records, list):
        raise ValueError("R-B2 manifest does not contain a frame list")
    train = [record for record in records if record.get("partition") == "train"]
    validation = [
        record for record in records if record.get("partition") == "validation"
    ]
    if len(train) != FORMAL_TRAIN_COUNT or len(validation) != FORMAL_VALIDATION_COUNT:
        raise ValueError("R-B2 requires the frozen 76/24 development split")
    if len(train) + len(validation) != len(records):
        raise ValueError("R-B2 manifest contains a forbidden partition")
    identities: set[tuple[int, int]] = set()
    for record in records:
        lowered = [str(key).lower() for key in record]
        if any(
            fragment in key
            for key in lowered
            for fragment in FORBIDDEN_RECORD_KEY_FRAGMENTS
        ):
            raise ValueError("R-B2 manifest exposes a future/test field")
        identity = (int(record["sequence"]), int(record["radar_index"]))
        if identity in identities:
            raise ValueError("R-B2 manifest contains duplicate frame identities")
        identities.add(identity)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    values = normalization.get("normalization", normalization)
    center = float(values["center"])
    scale = float(values["scale"])
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("R-B2 Cube normalization is invalid")
    selected = min(
        train,
        key=lambda record: (
            int(record["sequence"]),
            int(record["radar_index"]),
        ),
    )
    return {
        "record": selected,
        "log_center": center,
        "log_scale": scale,
    }, hashes


def load_one_frame(
    data_root: Path,
    cache_root: Path,
    record: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    sequence = int(record["sequence"])
    radar_index = int(record["radar_index"])
    cube_path = (
        data_root
        / str(sequence)
        / "radar_tesseract"
        / f"tesseract_{radar_index:05d}.mat"
    )
    cache_path = cache_root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"
    cube = load_tesseract(cube_path).astype(np.float32, copy=False)
    with np.load(cache_path) as cache:
        target = cache[ALLOWED_CACHE_ARRAYS[0]].astype(np.float32)
    return (
        torch.from_numpy(cube).unsqueeze(0),
        torch.from_numpy(target),
        {
            "sequence": sequence,
            "radar_index": radar_index,
            "partition": str(record["partition"]),
            "cube_path": str(cube_path),
            "cube_sha256": sha256(cube_path),
            "cache_path": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "cache_arrays_read": list(ALLOWED_CACHE_ARRAYS),
        },
    )


def candidate_preflight_gate(
    report: dict[str, Any],
    structure: dict[str, Any],
    config: TinyConfig,
) -> dict[str, Any]:
    checks = {
        "candidate_count_exact": (
            int(structure["candidate_count"])
            == sum(config.candidate_voxel_quotas)
        ),
        "candidate_range_quotas_exact": (
            tuple(structure["candidate_voxel_count_by_range"])
            == config.candidate_voxel_quotas
        ),
        "candidate_voxel_ids_unique": (
            int(structure["unique_candidate_voxel_count"])
            == int(structure["candidate_count"])
        ),
        "target_occupied_voxel_recall_at_least_20pct": (
            float(report["minimum_target_occupied_voxel_recall"])
            >= config.minimum_candidate_occupied_voxel_recall
        ),
        "target_confidence_coverage_at_least_30pct": (
            float(report["minimum_target_confidence_coverage"])
            >= config.minimum_candidate_confidence_coverage
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def one_frame_overfit_gate(
    geometry: dict[str, float],
    export_report: dict[str, object],
    config: TinyConfig,
) -> dict[str, Any]:
    checks = {
        "exact_10000_points": (
            int(export_report["exact_point_count"]) == EXPORT_POINT_COUNT
        ),
        "exact_range_quotas": (
            tuple(export_report["range_point_quotas"]) == OUTPUT_POINT_QUOTAS
        ),
        "guaranteed_minimum_distance_at_least_5cm": (
            float(
                export_report[
                    "guaranteed_minimum_pair_distance_lower_bound_m"
                ]
            )
            >= config.minimum_export_distance_m
        ),
        "no_copy_padding_jitter_or_duplicate_candidate": not any(
            bool(export_report[key])
            for key in (
                "copy_used",
                "padding_used",
                "jitter_used",
                "duplicate_candidate_id_used",
            )
        ),
        "cube_only_candidate_and_export": (
            bool(export_report["cube_only_inference"])
            and not bool(export_report["ground_truth_used_for_candidate_or_export"])
        ),
        "chamfer_at_most_1m": (
            float(geometry["chamfer_m"]) <= config.maximum_chamfer_m
        ),
        "completeness_mean_at_most_0p75m": (
            float(geometry["completeness_mean_distance_m"])
            <= config.maximum_completeness_mean_m
        ),
        "outlier_fraction_at_most_10pct": (
            float(geometry["outlier_fraction_2m"])
            <= config.maximum_outlier_fraction_2m
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


@torch.no_grad()
def evaluate(
    model: RB2VoxelSlotNet,
    cube: torch.Tensor,
    candidate_indices: torch.Tensor,
    target: torch.Tensor,
    axes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: TinyConfig,
) -> dict[str, Any]:
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(cube, candidate_indices, *axes)
    export = export_exact_voxel_slots(prediction)[0]
    geometry = geometry_report(
        export.xyz_m.float(),
        target[:, :3].float(),
        target_weight=target[:, 3].float(),
    )
    return {
        "geometry": geometry,
        "export": export.report,
        "gate": one_frame_overfit_gate(geometry, export.report, config),
        "selected_candidate_voxel_count": int(
            export.selected_candidate_positions.numel()
        ),
    }


def save_checkpoint(
    path: Path,
    *,
    model: RB2VoxelSlotNet,
    optimizer: torch.optim.Optimizer,
    update: int,
    config: TinyConfig,
    provenance: dict[str, Any],
    evaluation: dict[str, Any] | None,
) -> None:
    atomic_torch_save(
        path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "config": asdict(config),
            "provenance": provenance,
            "evaluation": evaluation,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"R-B2 output is immutable and non-empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    selected, hashes = validate_inputs(
        args.manifest,
        args.scene_split,
        args.normalization_stats,
    )
    config = TinyConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device, device_name = require_h200(args.device)
    axes_np = load_axes(args.data_root / "resources")
    axes = (
        torch.as_tensor(axes_np.range_m, dtype=torch.float32, device=device),
        torch.as_tensor(axes_np.azimuth_rad, dtype=torch.float32, device=device),
        torch.as_tensor(axes_np.elevation_rad, dtype=torch.float32, device=device),
    )
    cube_cpu, target_cpu, frame = load_one_frame(
        args.data_root,
        args.cache_root,
        selected["record"],
    )
    cube = cube_cpu.to(device)
    target = target_cpu.to(device)
    candidate_indices = candidate_voxels_from_cube(
        cube,
        *axes,
        candidate_voxel_quotas=config.candidate_voxel_quotas,
        seed_multiplier=config.seed_multiplier,
        maximum_neighbor_radius=config.maximum_neighbor_radius,
    )
    candidate_centers = lattice_indices_to_centers(candidate_indices)
    candidate_strata = range_stratum_codes(candidate_centers)
    candidate_structure = {
        "candidate_count": int(candidate_indices.shape[1]),
        "unique_candidate_voxel_count": int(
            torch.unique(candidate_indices[0], dim=0).shape[0]
        ),
        "candidate_voxel_count_by_range": [
            int((candidate_strata[0] == stratum).sum().item())
            for stratum in range(3)
        ],
    }
    targets = build_voxel_slot_targets(candidate_indices, target)
    support = candidate_support_report(targets)
    preflight = candidate_preflight_gate(support, candidate_structure, config)
    provenance = {
        "source_commit": args.source_commit,
        **hashes,
        **frame,
        "device": device_name,
        "torch_version": torch.__version__,
        "candidate_indices_sha256": hashlib.sha256(
            candidate_indices.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "candidate_generation_inputs": [
            "current_cube_drae",
            "frozen_range_axis",
            "frozen_azimuth_axis",
            "frozen_elevation_axis",
        ],
        "candidate_generation_forbidden_inputs": [
            "target_xyz_confidence",
            "target_rae_index",
            "cfar",
            "future_cube",
            "test_partition",
        ],
    }
    run = {
        "config": asdict(config),
        "provenance": provenance,
        "candidate_support": support,
        "candidate_structure": candidate_structure,
        "candidate_preflight_gate": preflight,
    }
    atomic_json(args.output / "config.json", run)
    if not preflight["passed"]:
        atomic_json(
            args.output / "decision.json",
            {
                **run,
                "decision": "no_go_cube_only_candidate_support",
                "training_started": False,
            },
        )
        raise RuntimeError("R-B2 Cube-only candidate support failed its frozen gate")

    model = RB2VoxelSlotNet(
        doppler_bins=cube.shape[1],
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        log_center=float(selected["log_center"]),
        log_scale=float(selected["log_scale"]),
    ).to(device)
    provenance["model_parameter_count"] = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    log_path = args.output / "train_log.jsonl"
    started = time.monotonic()
    consecutive_passes = 0
    final_evaluation: dict[str, Any] | None = None
    best_chamfer = float("inf")
    for update in range(1, config.maximum_updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(cube, candidate_indices, *axes)
            loss, components = rb2_voxel_slot_loss(prediction, targets)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"R-B2 non-finite loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
        )
        optimizer.step()
        record: dict[str, Any] = {
            "update": update,
            "loss": {key: float(value.item()) for key, value in components.items()},
            "gradient_norm": float(gradient_norm.item()),
            "elapsed_seconds": time.monotonic() - started,
        }
        if update % config.evaluation_interval_updates == 0:
            final_evaluation = evaluate(
                model,
                cube,
                candidate_indices,
                target,
                axes,
                config,
            )
            gate_passed = bool(final_evaluation["gate"]["passed"])
            consecutive_passes = consecutive_passes + 1 if gate_passed else 0
            record["evaluation"] = final_evaluation
            record["consecutive_gate_passes"] = consecutive_passes
            chamfer = float(final_evaluation["geometry"]["chamfer_m"])
            if chamfer < best_chamfer:
                best_chamfer = chamfer
                save_checkpoint(
                    args.output / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    config=config,
                    provenance=provenance,
                    evaluation=final_evaluation,
                )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if consecutive_passes >= config.required_consecutive_gate_passes:
            break

    if final_evaluation is None:
        raise AssertionError("R-B2 tiny run ended without an evaluation")
    passed = consecutive_passes >= config.required_consecutive_gate_passes
    decision = {
        "config": asdict(config),
        "provenance": provenance,
        "candidate_support": support,
        "candidate_preflight_gate": preflight,
        "final_evaluation": final_evaluation,
        "completed_updates": update,
        "consecutive_gate_passes": consecutive_passes,
        "decision": (
            "go_one_frame_overfit_only"
            if passed
            else "no_go_representation_or_optimization"
        ),
        "formal_multiframe_or_validation_run_authorized": False,
        "evidence_boundary": (
            "A pass proves one-frame memorization only. It does not establish "
            "generalization, formal validation geometry, or temporal behavior."
        ),
    }
    save_checkpoint(
        args.output / "last.pt",
        model=model,
        optimizer=optimizer,
        update=update,
        config=config,
        provenance=provenance,
        evaluation=final_evaluation,
    )
    atomic_json(args.output / "decision.json", decision)
    if not passed:
        raise RuntimeError("R-B2 failed the frozen one-frame overfit gate")


if __name__ == "__main__":
    main()
