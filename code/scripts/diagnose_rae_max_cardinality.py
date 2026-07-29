#!/usr/bin/env python3
"""Diagnose the density-quality effect of frozen RAE-Max output cardinality."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.geometry_cardinality import (  # noqa: E402
    FAR_COMPLETENESS_METRIC,
    FIXED_K_VALUES,
    LEGACY_DENSE_K,
    LEGACY_OVERALL_METRICS,
    AdaptiveCountRule,
    aggregate_cardinality_frames,
    build_density_quality_pareto,
    build_exact10k_factor_diagnosis,
    compare_legacy_overall_metrics,
    evaluate_cardinality_arm,
    freeze_train_adaptive_rule,
    select_cardinality_arms,
    tensor_sha256,
    train_frame_confidence_cutoff,
    validate_k_values,
    validate_legacy_10k_reproduction,
)
from models.cube_occupancy import CubeOccupancyNet  # noqa: E402


PROTOCOL = "frozen_rae_max_geometry_cardinality_v1"
FROZEN_CHECKPOINT_SOURCE_COMMIT = (
    "0e5fe8430892d57996ed26fa18f233cfa5e0c79b"
)
FROZEN_SEEDS = (20260716, 20260717, 20260718)
FORMAL_TRAIN_COUNT = 76
FORMAL_VALIDATION_COUNT = 24
FORMAL_FAR_TARGET_COUNT = 23
PREFLIGHT_VALIDATION_COUNT = 2
LEGACY_METRIC_TOLERANCE = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(
    path: Path,
    document: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Cardinality diagnosis output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _identity(record: dict[str, Any]) -> dict[str, int]:
    return {
        "sequence": int(record["sequence"]),
        "radar_index": int(record["radar_index"]),
    }


def _identity_tuple(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["sequence"]), int(record["radar_index"])


def verify_source_tree(repo: Path, source_commit: str) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if source_commit != head:
        raise ValueError(
            f"Diagnostic source commit {source_commit} differs from HEAD {head}"
        )
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).strip()
    if changed:
        raise ValueError("Diagnostic source tree has tracked modifications")


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available() or not device_name.startswith("cuda"):
        raise RuntimeError("RAE-Max cardinality diagnosis requires an H200 CUDA device")
    device = torch.device(device_name)
    hardware = torch.cuda.get_device_name(device)
    if "H200" not in hardware.upper():
        raise RuntimeError(f"Expected H200, found {hardware}")
    return device, hardware


def _cache_path(cache_root: Path, record: dict[str, Any]) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_"
        f"radar_{int(record['radar_index']):05d}.npz"
    )


def _cube_path(data_root: Path, record: dict[str, Any]) -> Path:
    return (
        data_root
        / str(int(record["sequence"]))
        / "radar_tesseract"
        / f"tesseract_{int(record['radar_index']):05d}.mat"
    )


def _file_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def frame_data_hashes(
    records: list[dict[str, Any]],
    data_root: Path,
    cache_root: Path,
) -> list[dict[str, Any]]:
    documents = []
    for record in records:
        documents.append(
            {
                "partition": record["partition"],
                **_identity(record),
                "cube": _file_document(_cube_path(data_root, record)),
                "dense_cache": _file_document(_cache_path(cache_root, record)),
            }
        )
    return documents


def validate_data_contract(
    manifest: dict[str, Any],
    scene_split: dict[str, Any],
    normalization: dict[str, Any],
    *,
    manifest_hash: str,
    scene_split_hash: str,
) -> dict[str, int]:
    records = manifest.get("frames")
    if not isinstance(records, list):
        raise ValueError("Audit manifest has no frame list")
    partitions = {str(record.get("partition")) for record in records}
    if not partitions <= {"train", "validation"}:
        raise ValueError(
            f"Cardinality diagnosis rejects non-development partitions: {partitions}"
        )
    counts = {
        partition: sum(record["partition"] == partition for record in records)
        for partition in ("train", "validation")
    }
    if counts != {
        "train": FORMAL_TRAIN_COUNT,
        "validation": FORMAL_VALIDATION_COUNT,
    }:
        raise ValueError(f"Frozen G1 frame counts differ: {counts}")
    identities = [_identity_tuple(record) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Audit manifest contains duplicate frame identities")

    splits = scene_split.get("splits", {})
    for record in records:
        partition = record["partition"]
        split_sequences = {
            int(value) for value in splits.get(partition, {}).get("sequences", [])
        }
        if int(record["sequence"]) not in split_sequences:
            raise ValueError(
                f"Frame {_identity(record)} violates the scene split"
            )
    if scene_split.get("gate_pass") is not True:
        raise ValueError("Scene split gate is not passed")

    expected_train_frames = [
        _identity(record) for record in records if record["partition"] == "train"
    ]
    checks = {
        "train_partition_only": normalization.get("partitions") == ["train"],
        "full_train_coverage": normalization.get("frame_limit") is None,
        "train_frame_count": int(normalization.get("frame_count", -1))
        == FORMAL_TRAIN_COUNT,
        "train_frame_identities": normalization.get("frames")
        == expected_train_frames,
        "manifest_hash": normalization.get("manifest_sha256") == manifest_hash,
        "scene_split_hash": normalization.get("scene_split_sha256")
        == scene_split_hash,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Frozen train normalization contract failed: {failed}")
    return counts


def far_target_identities(
    validation_records: list[dict[str, Any]],
    cache_root: Path,
) -> list[dict[str, int]]:
    identities = []
    for record in validation_records:
        with np.load(_cache_path(cache_root, record)) as cache:
            target = cache["target_xyz_confidence"].astype(np.float32)
        radius = np.linalg.norm(target[:, :3], axis=1)
        if np.any((radius >= 60.0) & (radius < 120.0)):
            identities.append(_identity(record))
    return identities


def preflight_indices(
    records: list[dict[str, Any]],
    far_identities: list[dict[str, int]],
) -> list[int]:
    if len(records) < PREFLIGHT_VALIDATION_COUNT:
        raise ValueError("Preflight requires at least two validation records")
    far = {_identity_tuple(record) for record in far_identities}
    nonfar_positions = [
        index
        for index, record in enumerate(records)
        if _identity_tuple(record) not in far
    ]
    first = nonfar_positions[0] if nonfar_positions else 0
    second = next(
        (
            index
            for index, record in enumerate(records)
            if index != first
            and int(record["sequence"]) != int(records[first]["sequence"])
            and _identity_tuple(record) in far
        ),
        None,
    )
    if second is None:
        raise ValueError("Cannot select two cross-scene preflight frames")
    return sorted((first, second))


def _archived_frame_map(metrics: dict[str, Any]) -> dict[tuple[int, int], dict]:
    frames = metrics.get("validation", {}).get("frames", [])
    mapped = {_identity_tuple(frame): frame for frame in frames}
    if len(frames) != FORMAL_VALIDATION_COUNT or len(mapped) != len(frames):
        raise ValueError("Archived RAE-Max metrics do not contain 24 unique frames")
    return mapped


def load_run_contract(run_path: Path) -> dict[str, Any]:
    config_path = run_path / "config.json"
    checkpoint_path = run_path / "best.pt"
    metrics_path = run_path / "best_validation_metrics.json"
    for path in (config_path, checkpoint_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    archived_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = config_document.get("config")
    provenance = config_document.get("provenance")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError(f"Malformed frozen run configuration: {run_path}")
    seed = int(config.get("seed", -1))
    expected_name = (
        f"g1_rae_max_seed{seed}_{FROZEN_CHECKPOINT_SOURCE_COMMIT[:8]}"
    )
    checks = {
        "run_name": run_path.name == expected_name,
        "mode_rae_max": config.get("mode") == "rae_max",
        "formal_seed": seed in FROZEN_SEEDS,
        "exact_10k_training_contract": int(config.get("point_count", -1))
        == LEGACY_DENSE_K,
        "full_validation": config.get("validation_limit") is None,
        "evaluation_frame_count": int(config.get("max_eval_frames", -1))
        == FORMAL_VALIDATION_COUNT,
        "not_overfit": config.get("overfit_one_frame") is False,
        "checkpoint_config": checkpoint.get("config") == config,
        "checkpoint_provenance": checkpoint.get("provenance") == provenance,
        "checkpoint_source_commit": provenance.get("git_commit")
        == FROZEN_CHECKPOINT_SOURCE_COMMIT,
        "model_state_present": isinstance(checkpoint.get("model"), dict),
        "best_epoch_matches": int(checkpoint.get("epoch", -1))
        == int(archived_metrics.get("best_epoch", -2)),
        "archived_frame_count": archived_metrics.get("validation", {}).get(
            "frame_count"
        )
        == FORMAL_VALIDATION_COUNT,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Frozen RAE-Max run contract failed: {failed}")
    archived_frames = _archived_frame_map(archived_metrics)
    return {
        "run_path": run_path,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "config": config,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "archived_metrics": archived_metrics,
        "archived_frames": archived_frames,
        "seed": seed,
        "checks": checks,
    }


def validate_runs(
    runs: list[dict[str, Any]],
    *,
    manifest_hash: str,
    scene_split_hash: str,
    normalization_hash: str,
    validation_identities: list[dict[str, int]],
) -> None:
    if len(runs) != len(FROZEN_SEEDS):
        raise ValueError("Cardinality diagnosis requires three RAE-Max runs")
    if tuple(sorted(run["seed"] for run in runs)) != FROZEN_SEEDS:
        raise ValueError("RAE-Max checkpoint seed set differs from frozen seeds")
    expected_identity_set = {
        _identity_tuple(record) for record in validation_identities
    }
    for run in runs:
        provenance = run["provenance"]
        hashes = {
            "manifest_sha256": manifest_hash,
            "scene_split_sha256": scene_split_hash,
            "normalization_sha256": normalization_hash,
        }
        failed_hashes = [
            name for name, expected in hashes.items()
            if provenance.get(name) != expected
        ]
        if failed_hashes:
            raise ValueError(
                f"Run {run['seed']} data hashes differ: {failed_hashes}"
            )
        if set(run["archived_frames"]) != expected_identity_set:
            raise ValueError(
                f"Run {run['seed']} validation identities differ from manifest"
            )


def source_hashes(repo: Path, script_path: Path) -> dict[str, dict[str, str]]:
    paths = (
        script_path,
        repo / "code/eval/geometry_cardinality.py",
        repo / "code/eval/dense_geometry.py",
        repo / "code/models/cube_occupancy.py",
        repo / "code/cube_dense/dataset.py",
        repo / "code/cube_dense/kradar.py",
    )
    return {
        str(path.resolve()): {"sha256": sha256(path)}
        for path in paths
    }


def _model_from_run(
    run: dict[str, Any],
    axes,
    device: torch.device,
) -> CubeOccupancyNet:
    config = run["config"]
    model = CubeOccupancyNet(
        "rae_max",
        torch.from_numpy(axes.doppler_mps),
        base_channels=int(config["base_channels"]),
        log_center=float(config["log_center"]),
        log_scale=float(config["log_scale"]),
    ).to(device)
    model.load_state_dict(run["checkpoint"]["model"], strict=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def infer_logits(
    model: CubeOccupancyNet,
    item: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(cube)[0].float()
    del cube
    return logits


def fit_adaptive_rule(
    model: CubeOccupancyNet,
    train_set: KRadarCubeDataset,
    indices: list[int],
    device: torch.device,
    *,
    preflight_only: bool,
) -> tuple[AdaptiveCountRule, dict[str, Any]]:
    cutoffs = []
    frames = []
    for index in indices:
        item = train_set[index]
        logits = infer_logits(model, item, device)
        target_effective_count = float(
            item["target_xyz_confidence"][:, 3].sum().item()
        )
        cutoff = train_frame_confidence_cutoff(
            logits,
            target_effective_count,
        )
        cutoffs.append(cutoff)
        frames.append(
            {
                **_identity(item),
                "partition": "train",
                "target_effective_count_fit_only": target_effective_count,
                "confidence_cutoff": cutoff,
                "logits_sha256": tensor_sha256(logits),
            }
        )
        del logits, item
        torch.cuda.empty_cache()
    rule = freeze_train_adaptive_rule(
        cutoffs,
        preflight_only=preflight_only,
    )
    return rule, {
        "rule": rule.to_document(),
        "selection_fit_partition": "train",
        "validation_gt_accessed_for_fit": False,
        "frames": frames,
    }


def _aggregate_metric_reproduction(
    current: dict[str, Any],
    archived: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    differences = {}
    for metric in LEGACY_OVERALL_METRICS:
        if metric not in current or metric not in archived:
            raise ValueError(f"Legacy aggregate metric is missing: {metric}")
        metric_differences = {}
        for statistic in ("mean", "std", "median", "sample_count"):
            current_value = float(current[metric][statistic])
            archived_value = float(archived[metric][statistic])
            difference = abs(current_value - archived_value)
            metric_differences[statistic] = {
                "current": current_value,
                "archived": archived_value,
                "absolute_difference": difference,
                "within_tolerance": difference <= tolerance,
            }
        differences[metric] = metric_differences
    passed = all(
        statistic["within_tolerance"]
        for metric in differences.values()
        for statistic in metric.values()
    )
    return {
        "absolute_tolerance": tolerance,
        "far_metrics_compared": False,
        "metrics": differences,
        "passed": passed,
    }


def _target_diagnostics(
    dataset: KRadarCubeDataset,
    indices: list[int],
) -> dict[str, Any]:
    frames = []
    for index in indices:
        item = dataset[index]
        target = item["target_xyz_confidence"]
        frames.append(
            {
                **_identity(item),
                "target_count": int(target.shape[0]),
                "target_effective_count": float(target[:, 3].sum().item()),
            }
        )
    count = np.asarray([frame["target_count"] for frame in frames], dtype=np.float64)
    effective = np.asarray(
        [frame["target_effective_count"] for frame in frames],
        dtype=np.float64,
    )
    return {
        "selection_dependency": False,
        "purpose": "diagnostic_only",
        "target_count": {
            "mean": float(count.mean()),
            "median": float(np.median(count)),
            "minimum": float(count.min()),
            "maximum": float(count.max()),
        },
        "target_effective_count": {
            "mean": float(effective.mean()),
            "median": float(np.median(effective)),
            "minimum": float(effective.min()),
            "maximum": float(effective.max()),
        },
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument(
        "--rae-max-runs",
        type=Path,
        nargs=3,
        required=True,
        metavar=("SEED16_RUN", "SEED17_RUN", "SEED18_RUN"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--scope",
        choices=("preflight", "full"),
        default="preflight",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(FIXED_K_VALUES),
    )
    parser.add_argument("--fit-train-adaptive-threshold", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Cardinality diagnosis output exists: {args.output}")
    repo = Path(__file__).resolve().parents[2]
    script_path = Path(__file__).resolve()
    verify_source_tree(repo, args.source_commit)
    device, hardware = require_h200(args.device)

    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    normalization = json.loads(
        args.normalization.read_text(encoding="utf-8")
    )
    data_counts = validate_data_contract(
        manifest,
        scene_split,
        normalization,
        manifest_hash=manifest_hash,
        scene_split_hash=scene_split_hash,
    )
    validation_set = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    train_set = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train",),
    )
    validation_records = validation_set.records
    validation_identities = [_identity(record) for record in validation_records]
    full_far_identities = far_target_identities(
        validation_records,
        args.cache_root,
    )
    if len(full_far_identities) != FORMAL_FAR_TARGET_COUNT:
        raise ValueError(
            f"Frozen validation must contain 23 far-target frames, got "
            f"{len(full_far_identities)}"
        )
    if args.scope == "full":
        validation_indices = list(range(FORMAL_VALIDATION_COUNT))
        train_fit_indices = list(range(FORMAL_TRAIN_COUNT))
    else:
        validation_indices = preflight_indices(
            validation_records,
            full_far_identities,
        )
        train_fit_indices = [0, len(train_set) - 1]
    selected_validation_records = [
        validation_records[index] for index in validation_indices
    ]
    selected_far = {
        _identity_tuple(record) for record in full_far_identities
    }
    selected_far_count = sum(
        _identity_tuple(record) in selected_far
        for record in selected_validation_records
    )
    expected_validation_count = (
        FORMAL_VALIDATION_COUNT
        if args.scope == "full"
        else PREFLIGHT_VALIDATION_COUNT
    )
    if len(validation_indices) != expected_validation_count:
        raise ValueError("Selected validation frame count differs from scope")

    axes = load_axes(args.data_root / "resources")
    spatial_cell_count = (
        len(axes.range_m) * len(axes.azimuth_rad) * len(axes.elevation_rad)
    )
    k_values = validate_k_values(args.k_values, spatial_cell_count)
    runs = [load_run_contract(path.resolve()) for path in args.rae_max_runs]
    runs.sort(key=lambda run: run["seed"])
    validate_runs(
        runs,
        manifest_hash=manifest_hash,
        scene_split_hash=scene_split_hash,
        normalization_hash=normalization_hash,
        validation_identities=validation_identities,
    )

    data_records_to_hash = list(selected_validation_records)
    if args.fit_train_adaptive_threshold:
        data_records_to_hash.extend(
            train_set.records[index] for index in train_fit_indices
        )
    data_contract = {
        "manifest": _file_document(args.manifest),
        "scene_split": _file_document(args.scene_split),
        "normalization": _file_document(args.normalization),
        "range_azimuth_elevation_axes": _file_document(
            args.data_root / "resources/info_arr.mat"
        ),
        "doppler_axis": _file_document(
            args.data_root / "resources/arr_doppler.mat"
        ),
        "selected_frames": frame_data_hashes(
            data_records_to_hash,
            args.data_root,
            args.cache_root,
        ),
    }

    seed_results = {}
    combined_frames: dict[str, list[dict[str, Any]]] = {}
    all_legacy_checks = []
    adaptive_fit_documents = {}
    for run in runs:
        seed = run["seed"]
        model = _model_from_run(run, axes, device)
        adaptive_rule = None
        if args.fit_train_adaptive_threshold:
            adaptive_rule, adaptive_fit = fit_adaptive_rule(
                model,
                train_set,
                train_fit_indices,
                device,
                preflight_only=args.scope == "preflight",
            )
            adaptive_fit_documents[str(seed)] = adaptive_fit

        arm_frames: dict[str, list[dict[str, Any]]] = {}
        legacy_frames = []
        for index in validation_indices:
            item = validation_set[index]
            identity = _identity(item)
            logits = infer_logits(model, item, device)
            logits_hash = tensor_sha256(logits)
            arms = select_cardinality_arms(
                logits,
                axes,
                k_values=k_values,
                adaptive_rule=adaptive_rule,
            )
            target = item["target_xyz_confidence"].to(
                device,
                non_blocking=True,
            )
            for arm_name, arm in arms.items():
                evaluated = evaluate_cardinality_arm(arm, target)
                frame = {
                    "seed": seed,
                    **identity,
                    "partition": "validation",
                    "shared_logits_sha256": logits_hash,
                    **evaluated,
                }
                arm_frames.setdefault(arm_name, []).append(frame)
                combined_frames.setdefault(arm_name, []).append(frame)

            exact_10k = arms[f"fixed_k_{LEGACY_DENSE_K}"]
            hash_reproduction = validate_legacy_10k_reproduction(
                logits,
                axes,
                exact_10k,
            )
            current_geometry = arm_frames[
                f"fixed_k_{LEGACY_DENSE_K}"
            ][-1]["geometry"]
            archived_frame = run["archived_frames"][_identity_tuple(identity)]
            metric_reproduction = compare_legacy_overall_metrics(
                current_geometry,
                archived_frame["generated"],
                absolute_tolerance=LEGACY_METRIC_TOLERANCE,
            )
            legacy_frame = {
                **identity,
                "hash_reproduction": hash_reproduction,
                "overall_metric_reproduction": metric_reproduction,
                "passed": hash_reproduction["passed"]
                and metric_reproduction["passed"],
            }
            if not legacy_frame["passed"]:
                raise RuntimeError(
                    f"Seed {seed} frame {identity} failed legacy 10k reproduction"
                )
            legacy_frames.append(legacy_frame)
            all_legacy_checks.append(legacy_frame)
            del logits, target, arms, item
            torch.cuda.empty_cache()

        aggregated_arms = {
            name: aggregate_cardinality_frames(
                frames,
                expected_frame_count=expected_validation_count,
                expected_far_target_count=selected_far_count,
            )
            for name, frames in arm_frames.items()
        }
        aggregate_reproduction = None
        if args.scope == "full":
            aggregate_reproduction = _aggregate_metric_reproduction(
                aggregated_arms[f"fixed_k_{LEGACY_DENSE_K}"]["geometry"],
                run["archived_metrics"]["validation"]["generated"],
                tolerance=LEGACY_METRIC_TOLERANCE,
            )
            if not aggregate_reproduction["passed"]:
                raise RuntimeError(
                    f"Seed {seed} failed archived aggregate 10k reproduction"
                )
        seed_results[str(seed)] = {
            "arms": aggregated_arms,
            "density_quality_pareto": build_density_quality_pareto(
                aggregated_arms
            ),
            "exact10k_factor_diagnosis": build_exact10k_factor_diagnosis(
                aggregated_arms
            ),
            "legacy_10k_reproduction": {
                "frames": legacy_frames,
                "aggregate": aggregate_reproduction,
                "aggregate_status": (
                    "passed"
                    if aggregate_reproduction is not None
                    else "not_applicable_to_2frame_preflight"
                ),
                "passed": all(frame["passed"] for frame in legacy_frames)
                and (
                    aggregate_reproduction is None
                    or aggregate_reproduction["passed"]
                ),
            },
        }
        del model, run["checkpoint"]
        torch.cuda.empty_cache()

    combined_arms = {
        name: aggregate_cardinality_frames(
            frames,
            expected_frame_count=expected_validation_count * len(FROZEN_SEEDS),
            expected_far_target_count=selected_far_count * len(FROZEN_SEEDS),
        )
        for name, frames in combined_frames.items()
    }
    checkpoint_documents = [
        {
            "seed": run["seed"],
            "run_path": str(run["run_path"]),
            "config": _file_document(run["config_path"]),
            "checkpoint": _file_document(run["checkpoint_path"]),
            "archived_metrics": _file_document(run["metrics_path"]),
            "checkpoint_source_commit": run["provenance"]["git_commit"],
            "best_epoch": int(run["archived_metrics"]["best_epoch"]),
            "evaluation_state": "model",
            "contract_checks": run["checks"],
        }
        for run in runs
    ]
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_type": "frozen_rae_max_geometry_cardinality_diagnosis",
        "scope": args.scope,
        "source": {
            "git_commit": args.source_commit,
            "files": source_hashes(repo, script_path),
        },
        "checkpoints": checkpoint_documents,
        "data_contract": data_contract,
        "data_counts": data_counts,
        "validation_frame_identities": [
            _identity(record) for record in selected_validation_records
        ],
        "far_target_frame_identities": [
            _identity(record)
            for record in selected_validation_records
            if _identity_tuple(record) in selected_far
        ],
        "full_validation_far_target_frame_count": len(full_far_identities),
        "evaluator": {
            "corrected_far_semantics": (
                "target-bin completeness uses each target's global nearest "
                "prediction and never requires a same-bin prediction"
            ),
            "dense_geometry_path": str(
                (repo / "code/eval/dense_geometry.py").resolve()
            ),
            "dense_geometry_sha256": sha256(
                repo / "code/eval/dense_geometry.py"
            ),
            "cardinality_evaluator_path": str(
                (repo / "code/eval/geometry_cardinality.py").resolve()
            ),
            "cardinality_evaluator_sha256": sha256(
                repo / "code/eval/geometry_cardinality.py"
            ),
            "full_far_target_requirement": {
                "metric": FAR_COMPLETENESS_METRIC,
                "expected_frame_count": FORMAL_FAR_TARGET_COUNT,
                "observed_frame_count": len(full_far_identities),
                "passed": len(full_far_identities)
                == FORMAL_FAR_TARGET_COUNT,
            },
        },
        "selection_contract": {
            "fixed_k_values": list(k_values),
            "same_logits_for_all_arms": True,
            "unique_rae_cell_centers": True,
            "stable_topk": (
                "sorted torch.topk on sigmoid logits, identical to the "
                "frozen legacy occupancy decoder"
            ),
            "validation_gt_used_for_selection": False,
            "exact_10k_arm": f"fixed_k_{LEGACY_DENSE_K}",
            "smaller_k_interpretation": (
                "reduced-cardinality diagnostics, not 10k dense outputs"
            ),
            "adaptive_enabled": args.fit_train_adaptive_threshold,
            "adaptive_fit": adaptive_fit_documents,
        },
        "validation_gt_diagnostic_only": _target_diagnostics(
            validation_set,
            validation_indices,
        ),
        "results": {
            "seeds": seed_results,
            "combined": {
                "arms": combined_arms,
                "density_quality_pareto": build_density_quality_pareto(
                    combined_arms
                ),
                "exact10k_factor_diagnosis": (
                    build_exact10k_factor_diagnosis(combined_arms)
                ),
            },
        },
        "legacy_10k_reproduction": {
            "frame_check_count": len(all_legacy_checks),
            "all_frame_hash_and_metric_checks_passed": all(
                check["passed"] for check in all_legacy_checks
            ),
            "aggregate_checked_only_for_full_scope": True,
        },
        "device": hardware,
        "torch_version": torch.__version__,
        "test_accessed": False,
    }
    atomic_json(args.output, document, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scope": args.scope,
                "seeds": list(FROZEN_SEEDS),
                "arms": list(combined_arms),
                "validation_frames": expected_validation_count,
                "far_target_frames": selected_far_count,
                "legacy_10k_reproduced": document[
                    "legacy_10k_reproduction"
                ]["all_frame_hash_and_metric_checks_passed"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
