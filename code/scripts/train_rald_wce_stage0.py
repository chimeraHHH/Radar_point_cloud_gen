#!/usr/bin/env python3
"""Train and evaluate the frozen R-A1 RaLD-WCE Stage-0 geometry chain."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
)
from eval.rald_wce_stage0 import (  # noqa: E402
    ExactExportCapacityError,
    FORMAL_OUTPUT_RANGE_QUOTAS,
    FORMAL_Q0_RANGE_QUOTAS,
    FORMAL_Q1_ANCHOR_QUOTAS,
    FORMAL_Q1_SAMPLES_PER_ANCHOR,
    WideInferenceConfig,
    infer_exact_wce,
)
from losses.rald_wce import (  # noqa: E402
    rald_wce_stage0_loss,
    sample_bounded_occupancy_queries,
)
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_wce_field import RaLDWCEField  # noqa: E402
from scripts.g1b_contract import sha256  # noqa: E402


PROTOCOL = "g1_ra1_rald_wce_stage0_v1"
OFFICIAL_RALD_COMMIT = "ffec4b41241391734b1eda5c093de843c909eb8e"
FORMAL_SEED = 20260716
FORMAL_TRAIN_COUNT = 76
FORMAL_VALIDATION_COUNT = 24
FORMAL_FAR_VALIDATION_COUNT = 23
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
CORRECTED_DENSE_GEOMETRY_SHA256 = (
    "e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68"
)
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TrainConfig:
    protocol: str
    seed: int
    smoke: bool
    epochs: int
    eval_every: int
    train_limit: int | None
    validation_limit: int | None
    learning_rate: float
    minimum_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    occupancy_query_count: int
    positive_query_ratio: float
    latent_count: int
    model_dim: int
    depth: int
    heads: int
    head_dim: int
    decode_chunk_size: int
    q0_range_quotas: tuple[int, int, int]
    q1_anchor_quotas: tuple[int, int, int]
    q1_samples_per_anchor: int
    output_range_quotas: tuple[int, int, int]
    minimum_export_distance_m: float
    selection_metric: str
    test_accessed: bool
    doppler_head: bool


def frozen_config(*, smoke: bool) -> TrainConfig:
    """Return the non-overridable smoke or formal Stage-0 contract."""

    return TrainConfig(
        protocol=PROTOCOL,
        seed=FORMAL_SEED,
        smoke=smoke,
        epochs=1 if smoke else 20,
        eval_every=1 if smoke else 5,
        train_limit=2 if smoke else None,
        validation_limit=2 if smoke else None,
        learning_rate=1e-4,
        minimum_learning_rate=1e-6,
        weight_decay=0.05,
        gradient_clip_norm=10.0,
        occupancy_query_count=2_048 if smoke else 16_000,
        positive_query_ratio=0.0625,
        latent_count=512,
        model_dim=512,
        depth=6,
        heads=8,
        head_dim=64,
        decode_chunk_size=8_192,
        q0_range_quotas=(
            (16_000, 9_000, 5_000)
            if smoke
            else FORMAL_Q0_RANGE_QUOTAS
        ),
        q1_anchor_quotas=(
            (1_000, 500, 250)
            if smoke
            else FORMAL_Q1_ANCHOR_QUOTAS
        ),
        q1_samples_per_anchor=(
            4 if smoke else FORMAL_Q1_SAMPLES_PER_ANCHOR
        ),
        output_range_quotas=FORMAL_OUTPUT_RANGE_QUOTAS,
        minimum_export_distance_m=0.05,
        selection_metric=(
            "mean_chamfer + completeness_gate_excess + "
            "2*outlier_gate_excess + far_gate_excess + "
            "10*wrong_condition_gate_deficit"
        ),
        test_accessed=False,
        doppler_head=False,
    )


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
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
        raise ValueError("R-A1 source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("R-A1 source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"R-A1 source worktree is dirty: {dirty}")


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("R-A1 requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("R-A1 is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"R-A1 requires H200, got {resolved}")
    return device, resolved


def load_normalization(path: Path) -> tuple[float, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    values = document.get("normalization", document)
    center = float(values["center"])
    scale = float(values["scale"])
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("R-A1 normalization is invalid")
    return center, scale


def validate_development_manifest(path: Path) -> dict[str, int]:
    document = json.loads(path.read_text(encoding="utf-8"))
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise ValueError("R-A1 manifest does not contain frames")
    counts = Counter(frame.get("partition") for frame in frames)
    expected = {"train": FORMAL_TRAIN_COUNT, "validation": FORMAL_VALIDATION_COUNT}
    if dict(counts) != expected:
        raise ValueError(f"R-A1 requires frozen 76/24 data, got {dict(counts)}")
    if any(frame.get("partition") == "test" for frame in frames):
        raise ValueError("R-A1 development manifest cannot contain test")
    identities = {
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in frames
    }
    if len(identities) != len(frames):
        raise ValueError("R-A1 development manifest has duplicate identities")
    return {
        "train_frame_count": counts["train"],
        "validation_frame_count": counts["validation"],
        "test_frame_count": 0,
    }


def validate_frozen_inputs(
    manifest: Path,
    scene_split: Path,
    normalization: Path,
) -> dict[str, str]:
    actual = {
        "manifest": sha256(manifest),
        "scene_split": sha256(scene_split),
        "normalization": sha256(normalization),
        "corrected_dense_geometry": sha256(
            Path(__file__).resolve().parents[1] / "eval/dense_geometry.py"
        ),
    }
    if actual["manifest"] != FROZEN_MANIFEST_SHA256:
        raise ValueError("R-A1 manifest hash differs from the frozen protocol")
    if actual["scene_split"] != FROZEN_SCENE_SPLIT_SHA256:
        raise ValueError("R-A1 scene-split hash differs from the frozen protocol")
    if actual["corrected_dense_geometry"] != CORRECTED_DENSE_GEOMETRY_SHA256:
        raise ValueError("R-A1 corrected dense-geometry evaluator hash changed")
    split = json.loads(scene_split.read_text(encoding="utf-8"))
    if split.get("gate_pass") is not True:
        raise ValueError("R-A1 scene split failed its leakage gate")
    if not normalization.is_file():
        raise FileNotFoundError(normalization)
    return actual


def cross_scene_condition_indices(records: list[dict]) -> list[int]:
    sequences = [int(record["sequence"]) for record in records]
    for shift in range(1, len(records)):
        candidate = [
            (index + shift) % len(records)
            for index in range(len(records))
        ]
        if all(
            sequences[index] != sequences[other]
            for index, other in enumerate(candidate)
        ):
            return candidate
    raise ValueError("R-A1 cannot construct a cross-scene condition shuffle")


def smoke_indices(records: list[dict]) -> list[int]:
    for first in range(len(records)):
        for second in range(first + 1, len(records)):
            if int(records[first]["sequence"]) != int(
                records[second]["sequence"]
            ):
                return [first, second]
    raise ValueError("R-A1 smoke requires two validation scenes")


def frame_seed(base: int, item: dict, epoch: int) -> int:
    return int(
        (
            base
            + 1_000_003 * epoch
            + 10_007 * int(item["sequence"])
            + 101 * int(item["radar_index"])
        )
        % (2**31 - 1)
    )


def cosine_learning_rate(config: TrainConfig, epoch: int) -> float:
    if config.epochs <= 1:
        return config.learning_rate
    fraction = epoch / (config.epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def aggregate_scalars(reports: list[dict[str, float]]) -> dict[str, dict]:
    if not reports:
        raise ValueError("R-A1 cannot aggregate an empty scalar list")
    keys = sorted(reports[0])
    if any(sorted(report) != keys for report in reports):
        raise ValueError("R-A1 scalar report keys differ")
    return {
        key: {
            "mean": float(np.mean([report[key] for report in reports])),
            "median": float(np.median([report[key] for report in reports])),
            "std": float(np.std([report[key] for report in reports])),
            "sample_count": len(reports),
        }
        for key in keys
    }


def _frame_has_far_target(target_xyz: torch.Tensor) -> bool:
    radius = torch.linalg.vector_norm(target_xyz[:, :3].float(), dim=1)
    return bool(((radius >= 60.0) & (radius < 120.0)).any())


def evaluate(
    model: RaLDWCEField,
    dataset: KRadarCubeDataset,
    validation_records: list[dict],
    validation_indices: list[int],
    wrong_indices: list[int],
    inference_config: WideInferenceConfig,
    axes,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate corrected geometry and same-query wrong-Cube intervention."""

    model.eval()
    range_m = torch.as_tensor(
        axes.range_m,
        dtype=torch.float32,
        device=device,
    )
    azimuth_rad = torch.as_tensor(
        axes.azimuth_rad,
        dtype=torch.float32,
        device=device,
    )
    elevation_rad = torch.as_tensor(
        axes.elevation_rad,
        dtype=torch.float32,
        device=device,
    )
    frames: list[dict[str, Any]] = []
    far_frame_count = 0
    for index in validation_indices:
        item = dataset[index]
        wrong_index = wrong_indices[index]
        wrong_item = dataset[wrong_index]
        cube = item["cube_drae"].unsqueeze(0).to(device)
        wrong_cube = wrong_item["cube_drae"].unsqueeze(0).to(device)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference = infer_exact_wce(
                    model,
                    cube,
                    wrong_cube,
                    range_m,
                    azimuth_rad,
                    elevation_rad,
                    inference_config,
                )
        except ExactExportCapacityError as error:
            error.report.update(
                {
                    "sequence": int(item["sequence"]),
                    "radar_index": int(item["radar_index"]),
                    "partition": str(item["partition"]),
                }
            )
            raise
        target = item["target_xyz_confidence"].to(device)
        target_xyz = target[:, :3].float()
        target_weight = target[:, 3].float()
        matched_xyz = inference.matched.xyz_m.to(device)
        wrong_xyz = inference.wrong_condition.xyz_m.to(device)
        matched_geometry = geometry_report(
            matched_xyz,
            target_xyz,
            target_weight=target_weight,
        )
        wrong_geometry = geometry_report(
            wrong_xyz,
            target_xyz,
            target_weight=target_weight,
        )
        has_far_target = _frame_has_far_target(target)
        far_frame_count += int(has_far_target)
        matched_chamfer = matched_geometry["chamfer_m"]
        wrong_chamfer = wrong_geometry["chamfer_m"]
        frames.append(
            {
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
                "partition": str(item["partition"]),
                "wrong_condition_sequence": int(wrong_item["sequence"]),
                "wrong_condition_radar_index": int(wrong_item["radar_index"]),
                "has_far_target": has_far_target,
                "matched": matched_geometry,
                "wrong_condition": wrong_geometry,
                "condition_intervention": {
                    "wrong_minus_matched_chamfer_fraction": float(
                        wrong_chamfer / max(matched_chamfer, 1e-12) - 1.0
                    ),
                    "matched_chamfer_better": float(
                        matched_chamfer < wrong_chamfer
                    ),
                },
                "matched_export": inference.matched.report,
                "wrong_export": inference.wrong_condition.report,
                "matched_hashes": inference.matched.hashes,
                "wrong_hashes": inference.wrong_condition.hashes,
                "inference": inference.report,
            }
        )
        del item, wrong_item, cube, wrong_cube, target
        torch.cuda.empty_cache()

    matched_aggregate = aggregate_geometry_reports(
        [frame["matched"] for frame in frames]
    )
    wrong_aggregate = aggregate_geometry_reports(
        [frame["wrong_condition"] for frame in frames]
    )
    intervention = aggregate_scalars(
        [frame["condition_intervention"] for frame in frames]
    )
    minimum_pair_distance = min(
        float(frame["matched_export"]["observed_minimum_pair_distance_m"])
        for frame in frames
    )
    return {
        "frame_count": len(frames),
        "scene_count": len({frame["sequence"] for frame in frames}),
        "far_target_frame_count": far_frame_count,
        "frames": frames,
        "matched": matched_aggregate,
        "wrong_condition": wrong_aggregate,
        "condition_intervention": intervention,
        "exact_export": {
            "point_count_per_frame": 10_000,
            "minimum_observed_pair_distance_m": minimum_pair_distance,
            "all_matched_exports_exact_10000": all(
                frame["matched_export"]["exact_point_count"] == 10_000
                for frame in frames
            ),
            "all_wrong_exports_exact_10000": all(
                frame["wrong_export"]["exact_point_count"] == 10_000
                for frame in frames
            ),
            "copy_padding_jitter_duplicate": False,
        },
    }


def stage0_decision(
    metrics: dict[str, Any],
    *,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    formal: bool,
) -> dict[str, Any]:
    """Apply frozen R-A1 gates without threshold relaxation."""

    far = metrics["matched"].get(
        "range_60_120m_completeness_mean_distance_m"
    )
    values = {
        "chamfer_mean_m": float(metrics["matched"]["chamfer_m"]["mean"]),
        "completeness_median_m": float(
            metrics["matched"]["completeness_mean_distance_m"]["median"]
        ),
        "outlier_fraction_mean": float(
            metrics["matched"]["outlier_fraction_2m"]["mean"]
        ),
        "far_completeness_mean_m": (
            float(far["mean"]) if far is not None else 1_000_000.0
        ),
        "condition_wrong_minus_matched_chamfer_fraction_mean": float(
            metrics["condition_intervention"][
                "wrong_minus_matched_chamfer_fraction"
            ]["mean"]
        ),
        "matched_condition_win_fraction": float(
            metrics["condition_intervention"]["matched_chamfer_better"]["mean"]
        ),
        "minimum_pair_distance_m": float(
            metrics["exact_export"]["minimum_observed_pair_distance_m"]
        ),
        "peak_allocated_gib": peak_allocated_bytes / 2**30,
        "peak_reserved_gib": peak_reserved_bytes / 2**30,
    }
    structural_checks = {
        "all_matched_exports_exact_10000": bool(
            metrics["exact_export"]["all_matched_exports_exact_10000"]
        ),
        "all_wrong_exports_exact_10000": bool(
            metrics["exact_export"]["all_wrong_exports_exact_10000"]
        ),
        "minimum_pair_distance_at_least_5cm": (
            values["minimum_pair_distance_m"] >= 0.05 - 1e-6
        ),
        "no_copy_padding_or_jitter_duplicate": (
            metrics["exact_export"]["copy_padding_jitter_duplicate"] is False
        ),
        "allocated_memory_below_55_gib": values["peak_allocated_gib"] < 55.0,
        "reserved_memory_below_65_gib": values["peak_reserved_gib"] < 65.0,
    }
    scientific_checks = {
        "chamfer_mean_at_most_2p50m": values["chamfer_mean_m"] <= 2.50,
        "completeness_median_at_most_2p4946m": (
            values["completeness_median_m"] <= 2.4946
        ),
        "outlier_fraction_mean_at_most_25pct": (
            values["outlier_fraction_mean"] <= 0.25
        ),
        "far_completeness_mean_at_most_46p9407m": (
            values["far_completeness_mean_m"] <= 46.9407
        ),
        "wrong_condition_degrades_chamfer_at_least_1pct": (
            values[
                "condition_wrong_minus_matched_chamfer_fraction_mean"
            ] >= 0.01
        ),
        "matched_condition_wins_at_least_75pct_frames": (
            values["matched_condition_win_fraction"] >= 0.75
        ),
    }
    count_checks = {
        "exact_24_validation_frames": (
            metrics["frame_count"] == FORMAL_VALIDATION_COUNT
        ),
        "exact_23_far_target_frames": (
            metrics["far_target_frame_count"] == FORMAL_FAR_VALIDATION_COUNT
        ),
    }
    eligible = formal and all(count_checks.values())
    return {
        "protocol": PROTOCOL,
        "eligible_for_stage0_scientific_decision": eligible,
        "values": values,
        "structural_checks": structural_checks,
        "scientific_checks": scientific_checks,
        "count_checks": count_checks,
        "promotion_passed": (
            eligible
            and all(structural_checks.values())
            and all(scientific_checks.values())
        ),
        "smoke_only": not formal,
        "doppler_head_evaluated": False,
    }


def save_checkpoint(
    path: Path,
    model: RaLDWCEField,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    source_commit: str,
) -> None:
    torch.save(
        {
            "protocol": PROTOCOL,
            "source_commit": source_commit,
            "epoch": epoch,
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"R-A1 output already exists: {args.output_dir}")
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)
    config = frozen_config(smoke=args.smoke)
    manifest_counts = validate_development_manifest(args.manifest)
    input_hashes = validate_frozen_inputs(
        args.manifest,
        args.scene_split,
        args.normalization,
    )
    log_center, log_scale = load_normalization(args.normalization)
    axes = load_axes(args.data_root / "resources")
    train_dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train",),
    )
    validation_dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    if any(
        record["partition"] == "test"
        for record in train_dataset.records + validation_dataset.records
    ):
        raise ValueError("R-A1 loaded the test partition")
    train_wrong_indices = cross_scene_condition_indices(train_dataset.records)
    validation_wrong_indices = cross_scene_condition_indices(
        validation_dataset.records
    )
    train_indices = (
        list(range(len(train_dataset)))
        if config.train_limit is None
        else list(range(config.train_limit))
    )
    validation_indices = (
        list(range(len(validation_dataset)))
        if config.validation_limit is None
        else smoke_indices(validation_dataset.records)
    )

    args.output_dir.mkdir(parents=True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    model = RaLDWCEField(
        log_center=log_center,
        log_scale=log_scale,
        latent_count=config.latent_count,
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
        decode_chunk_size=config.decode_chunk_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    inference_config = WideInferenceConfig(
        q0_range_quotas=config.q0_range_quotas,
        q1_anchor_quotas=config.q1_anchor_quotas,
        q1_samples_per_anchor=config.q1_samples_per_anchor,
        output_range_quotas=config.output_range_quotas,
        minimum_distance_m=config.minimum_export_distance_m,
        seed=config.seed,
        decode_chunk_size=config.decode_chunk_size,
    )
    source_files = (
        repo / "code/models/rald_wce_field.py",
        repo / "code/losses/rald_wce.py",
        repo / "code/eval/rald_wce_stage0.py",
        repo / "code/eval/dense_geometry.py",
        repo / "code/scripts/train_rald_wce_stage0.py",
        repo / "code/cube_dense/dataset.py",
    )
    run_manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "R-A1 Stage 0 XYZ plus confidence only",
        "source_commit": args.source_commit,
        "official_rald_commit_audited": OFFICIAL_RALD_COMMIT,
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in source_files
        },
        "input_hashes": input_hashes,
        "manifest_counts": manifest_counts,
        "config": asdict(config),
        "inference": inference_config.metadata(),
        "architecture": model.architecture_metadata(),
        "runtime": {
            "device_argument": args.device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "parameter_count": parameter_count(model),
        },
        "evidence_boundary": {
            "test_partition_accessed": False,
            "doppler_head_implemented": False,
            "doppler_head_evaluated": False,
            "edm_implemented": False,
            "long_training_started_by_preflight": False,
        },
    }
    atomic_json(args.output_dir / "run_manifest.json", run_manifest)
    train_log = args.output_dir / "train_log.jsonl"
    best_score = float("inf")
    best_epoch = 0
    started = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch_index in range(config.epochs):
            epoch = epoch_index + 1
            learning_rate = cosine_learning_rate(config, epoch_index)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            order_generator = torch.Generator().manual_seed(
                config.seed + epoch
            )
            order = torch.randperm(
                len(train_indices),
                generator=order_generator,
            ).tolist()
            losses: list[float] = []
            for local_row in order:
                index = train_indices[local_row]
                item = train_dataset[index]
                wrong_item = train_dataset[train_wrong_indices[index]]
                queries = sample_bounded_occupancy_queries(
                    item["target_rae_index"],
                    spatial_shape=tuple(item["cube_drae"].shape[1:]),
                    query_count=config.occupancy_query_count,
                    positive_query_ratio=config.positive_query_ratio,
                    seed=frame_seed(config.seed, item, epoch),
                )
                cube = item["cube_drae"].unsqueeze(0).to(device)
                wrong_cube = wrong_item["cube_drae"].unsqueeze(0).to(device)
                normalized = queries["normalized_rae"].to(device)
                occupancy = queries["occupancy_target"].to(device)
                residual = queries["residual_target_bins"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        cube,
                        normalized,
                        wrong_condition_cube_drae=wrong_cube,
                        chunk_size=config.decode_chunk_size,
                    )
                    loss = rald_wce_stage0_loss(
                        output["matched"],
                        output["wrong"],
                        normalized,
                        occupancy,
                        residual,
                        range_stratified_occupancy=False,
                    )
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
                optimizer.step()
                losses.append(float(loss.total.detach().item()))
                del item, wrong_item, cube, wrong_cube, output, loss
            checkpoint_path = args.output_dir / "last.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                epoch,
                args.source_commit,
            )
            record: dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "mean_train_loss": float(np.mean(losses)),
                "update_count": len(losses),
                "elapsed_seconds": time.monotonic() - started,
            }
            if epoch % config.eval_every == 0 or epoch == config.epochs:
                evaluation_checkpoint = (
                    args.output_dir / f"checkpoint_epoch{epoch:03d}.pt"
                )
                shutil.copy2(checkpoint_path, evaluation_checkpoint)
                metrics = evaluate(
                    model,
                    validation_dataset,
                    validation_dataset.records,
                    validation_indices,
                    validation_wrong_indices,
                    inference_config,
                    axes,
                    device,
                )
                decision = stage0_decision(
                    metrics,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    formal=not config.smoke,
                )
                evaluation = {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "source_commit": args.source_commit,
                    "checkpoint": str(evaluation_checkpoint.resolve()),
                    "checkpoint_sha256": sha256(evaluation_checkpoint),
                    "epoch": epoch,
                    "metrics": metrics,
                    "decision": decision,
                }
                metrics_path = args.output_dir / f"metrics_epoch{epoch:03d}.json"
                atomic_json(metrics_path, evaluation)
                score = (
                    decision["values"]["chamfer_mean_m"]
                    + max(
                        0.0,
                        decision["values"]["completeness_median_m"] - 2.4946,
                    )
                    + 2.0
                    * max(
                        0.0,
                        decision["values"]["outlier_fraction_mean"] - 0.25,
                    )
                    + 0.1
                    * max(
                        0.0,
                        decision["values"]["far_completeness_mean_m"] - 46.9407,
                    )
                    + 10.0
                    * max(
                        0.0,
                        0.01
                        - decision["values"][
                            "condition_wrong_minus_matched_chamfer_fraction_mean"
                        ],
                    )
                )
                record.update(
                    {
                        "metrics_path": str(metrics_path.resolve()),
                        "selection_score": score,
                        "promotion_passed": decision["promotion_passed"],
                    }
                )
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    shutil.copy2(
                        evaluation_checkpoint,
                        args.output_dir / "best.pt",
                    )
                    shutil.copy2(metrics_path, args.output_dir / "best_metrics.json")
            with train_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    except ExactExportCapacityError as error:
        failure = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "source_commit": args.source_commit,
            "status": "failed_exact_10000_capacity",
            "message": str(error),
            "capacity_report": error.report,
            "fallback_attempted": False,
            "copy_padding_jitter_duplicate": False,
            "scientific_decision_eligible": False,
        }
        atomic_json(args.output_dir / "terminal_capacity_failure.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise SystemExit(2) from error

    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "smoke": config.smoke,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "formal_long_training_was_started": not config.smoke,
        "doppler_head_implemented": False,
        "test_partition_accessed": False,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
