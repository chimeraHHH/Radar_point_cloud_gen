#!/usr/bin/env python3
"""Diagnose formal R-A1/WCE failure factors on its frozen candidate pool."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_tesseract  # noqa: E402
from eval.rald_wce_failure_factors import (  # noqa: E402
    aggregate_failure_factor_frames,
    diagnose_frozen_candidate_pool,
    diagnostic_artifact_label,
    reconstruct_matched_candidate_pool,
    tensor_sha256,
    validate_current_confidence_control,
)
from eval.rald_wce_stage0 import (  # noqa: E402
    FORMAL_OUTPUT_RANGE_QUOTAS,
    FORMAL_Q0_RANGE_QUOTAS,
    FORMAL_Q1_ANCHOR_QUOTAS,
    FORMAL_Q1_SAMPLES_PER_ANCHOR,
    WideInferenceConfig,
)
from models.rald_wce_field import RaLDWCEField  # noqa: E402
from scripts.g1b_contract import sha256  # noqa: E402
from scripts.train_rald_wce_stage0 import (  # noqa: E402
    FORMAL_FAR_VALIDATION_COUNT,
    FORMAL_SEED,
    FORMAL_TRAIN_COUNT,
    FORMAL_VALIDATION_COUNT,
    PROTOCOL as WCE_PROTOCOL,
    git_output,
    load_normalization,
    require_h200,
    validate_development_manifest,
    validate_frozen_inputs,
    verify_source_tree,
)


PROTOCOL = "g1_ra1_wce_frozen_candidate_failure_factors_v1"
CHECKPOINT_SOURCE_FILES = (
    "code/models/rald_wce_field.py",
    "code/losses/rald_wce.py",
    "code/eval/rald_wce_stage0.py",
    "code/eval/dense_geometry.py",
    "code/scripts/train_rald_wce_stage0.py",
    "code/cube_dense/dataset.py",
)


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_formal_checkpoint(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    formal_metrics: dict[str, Any],
) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("WCE diagnosis checkpoint has no configuration")
    checks = {
        "checkpoint_protocol": checkpoint.get("protocol") == WCE_PROTOCOL,
        "checkpoint_source_commit_full": (
            isinstance(checkpoint.get("source_commit"), str)
            and len(checkpoint["source_commit"]) == 40
        ),
        "formal_not_smoke": config.get("smoke") is False,
        "formal_seed": int(config.get("seed", -1)) == FORMAL_SEED,
        "formal_epochs": int(config.get("epochs", -1)) == 20,
        "full_training_set": config.get("train_limit") is None,
        "full_validation_set": config.get("validation_limit") is None,
        "q0_500k": tuple(config.get("q0_range_quotas", ()))
        == FORMAL_Q0_RANGE_QUOTAS,
        "q1_200k": (
            tuple(config.get("q1_anchor_quotas", ()))
            == FORMAL_Q1_ANCHOR_QUOTAS
            and int(config.get("q1_samples_per_anchor", -1))
            == FORMAL_Q1_SAMPLES_PER_ANCHOR
        ),
        "fixed_export_quotas": tuple(config.get("output_range_quotas", ()))
        == FORMAL_OUTPUT_RANGE_QUOTAS,
        "minimum_distance_5cm": float(
            config.get("minimum_export_distance_m", -1.0)
        )
        == 0.05,
        "test_locked": config.get("test_accessed") is False,
        "doppler_head_locked": config.get("doppler_head") is False,
        "model_state_present": isinstance(checkpoint.get("model"), dict),
        "metrics_protocol": formal_metrics.get("protocol") == WCE_PROTOCOL,
        "metrics_source_matches_checkpoint": formal_metrics.get("source_commit")
        == checkpoint.get("source_commit"),
        "metrics_epoch_matches_checkpoint": int(
            formal_metrics.get("epoch", -1)
        )
        == int(checkpoint.get("epoch", -2)),
        "metrics_checkpoint_hash_matches": formal_metrics.get(
            "checkpoint_sha256"
        )
        == sha256(checkpoint_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Formal WCE checkpoint contract failed: {failed}")
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256(checkpoint_path),
        "protocol": checkpoint["protocol"],
        "source_commit": checkpoint["source_commit"],
        "epoch": int(checkpoint["epoch"]),
        "checks": checks,
    }


def _source_hash_by_suffix(
    source_hashes: dict[str, Any],
    suffix: str,
) -> str:
    normalized_suffix = "/" + suffix.replace("\\", "/")
    matches = [
        value
        for path, value in source_hashes.items()
        if str(path).replace("\\", "/").endswith(normalized_suffix)
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError(f"WCE run manifest lacks one source hash for {suffix}")
    return matches[0]


def _json_normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_normalize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def validate_run_manifest(
    repo: Path,
    run_manifest_path: Path,
    checkpoint: dict[str, Any],
    actual_input_hashes: dict[str, str],
) -> dict[str, Any]:
    document = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    checks = {
        "protocol": document.get("protocol") == WCE_PROTOCOL,
        "source_commit": document.get("source_commit")
        == checkpoint.get("source_commit"),
        "config": _json_normalize(document.get("config"))
        == _json_normalize(checkpoint.get("config")),
        "input_hashes": document.get("input_hashes") == actual_input_hashes,
        "test_locked": document.get("evidence_boundary", {}).get(
            "test_partition_accessed"
        )
        is False,
        "doppler_locked": document.get("evidence_boundary", {}).get(
            "doppler_head_implemented"
        )
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"WCE run-manifest contract failed: {failed}")
    recorded_sources = document.get("source_hashes")
    if not isinstance(recorded_sources, dict):
        raise ValueError("WCE run manifest has no source-hash map")
    source_evidence: dict[str, Any] = {}
    for relative in CHECKPOINT_SOURCE_FILES:
        expected = _source_hash_by_suffix(recorded_sources, relative)
        actual = sha256(repo / relative)
        if actual != expected:
            raise ValueError(
                f"Checkpoint-incompatible WCE source {relative}: "
                f"{actual} != {expected}"
            )
        source_evidence[relative] = {
            "sha256": actual,
            "matches_formal_run": True,
        }
    return {
        "path": str(run_manifest_path.resolve()),
        "sha256": sha256(run_manifest_path),
        "checks": checks,
        "checkpoint_source_compatibility": source_evidence,
    }


def validate_formal_metrics(
    formal_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = formal_metrics.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("WCE formal metrics has no metrics document")
    frames = metrics.get("frames")
    if not isinstance(frames, list):
        raise ValueError("WCE formal metrics has no frame reports")
    identities = [
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in frames
    ]
    checks = {
        "frame_count_24": metrics.get("frame_count")
        == FORMAL_VALIDATION_COUNT,
        "frame_report_count_24": len(frames) == FORMAL_VALIDATION_COUNT,
        "far_frame_count_23": metrics.get("far_target_frame_count")
        == FORMAL_FAR_VALIDATION_COUNT,
        "unique_frame_identities": len(set(identities)) == len(identities),
        "validation_only": all(
            frame.get("partition") == "validation" for frame in frames
        ),
        "all_control_hashes_present": all(
            isinstance(frame.get("matched_hashes"), dict)
            and isinstance(
                frame.get("inference", {}).get("query_hashes"),
                dict,
            )
            for frame in frames
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"WCE formal metrics contract failed: {failed}")
    return frames


def validation_records(
    manifest_path: Path,
    formal_frames: list[dict[str, Any]],
    *,
    preflight: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("WCE diagnosis manifest has no frames")
    if any(frame.get("partition") == "test" for frame in frames):
        raise ValueError("WCE diagnosis refuses manifests containing test")
    counts = Counter(frame.get("partition") for frame in frames)
    if counts != Counter(
        {
            "train": FORMAL_TRAIN_COUNT,
            "validation": FORMAL_VALIDATION_COUNT,
        }
    ):
        raise ValueError(f"WCE diagnosis requires frozen 76/24, got {counts}")
    by_identity = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in frames
        if frame.get("partition") == "validation"
    }
    if len(by_identity) != FORMAL_VALIDATION_COUNT:
        raise ValueError("WCE diagnosis validation identities are not unique")
    selected_reference = formal_frames[:2] if preflight else formal_frames
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for reference in selected_reference:
        identity = (
            int(reference["sequence"]),
            int(reference["radar_index"]),
        )
        record = by_identity.get(identity)
        if record is None:
            raise ValueError(
                f"WCE formal reference identity is absent from manifest: {identity}"
            )
        if record.get("partition") != "validation":
            raise ValueError("WCE diagnosis selected a non-validation record")
        selected.append((record, reference))
    expected = 2 if preflight else FORMAL_VALIDATION_COUNT
    if len(selected) != expected:
        raise AssertionError("WCE diagnosis selected the wrong frame count")
    return selected


def load_current_frame(
    data_root: Path,
    cache_root: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
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
        target = cache["target_xyz_confidence"].astype(np.float32)
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError(f"WCE target cache has invalid shape: {cache_path}")
    if not np.isfinite(target).all():
        raise ValueError(f"WCE target cache is non-finite: {cache_path}")
    return {
        "cube_drae": torch.from_numpy(cube),
        "target_xyz_confidence": torch.from_numpy(target),
        "sequence": sequence,
        "radar_index": radar_index,
        "partition": str(record["partition"]),
        "current_cube_path": str(cube_path.resolve()),
        "current_target_cache_path": str(cache_path.resolve()),
        "current_target_cache_sha256": sha256(cache_path),
    }


def build_model(
    checkpoint: dict[str, Any],
    *,
    log_center: float,
    log_scale: float,
    device: torch.device,
) -> RaLDWCEField:
    config = checkpoint["config"]
    model = RaLDWCEField(
        log_center=log_center,
        log_scale=log_scale,
        latent_count=int(config["latent_count"]),
        model_dim=int(config["model_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
        decode_chunk_size=int(config["decode_chunk_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.assert_condition_exclusive_contract()
    return model.eval()


def formal_inference_config(checkpoint: dict[str, Any]) -> WideInferenceConfig:
    config = checkpoint["config"]
    result = WideInferenceConfig(
        q0_range_quotas=tuple(config["q0_range_quotas"]),
        q1_anchor_quotas=tuple(config["q1_anchor_quotas"]),
        q1_samples_per_anchor=int(config["q1_samples_per_anchor"]),
        output_range_quotas=tuple(config["output_range_quotas"]),
        minimum_distance_m=float(config["minimum_export_distance_m"]),
        seed=int(config["seed"]),
        decode_chunk_size=int(config["decode_chunk_size"]),
    )
    result.validate()
    return result


def source_hashes(repo: Path, script_path: Path) -> dict[str, str]:
    paths = (
        script_path,
        repo / "code/eval/rald_wce_failure_factors.py",
        repo / "code/eval/rald_wce_stage0.py",
        repo / "code/eval/dense_geometry.py",
        repo / "code/eval/g1a_wide_support.py",
        repo / "code/models/rald_wce_field.py",
        repo / "code/cube_dense/kradar.py",
    )
    return {
        str(path.relative_to(repo)): sha256(path)
        for path in paths
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--formal-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-chunk-size", type=int, default=8_192)
    parser.add_argument("--target-chunk-size", type=int, default=4_096)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"WCE diagnosis output exists: {args.output}")
    if args.candidate_chunk_size <= 0 or args.target_chunk_size <= 0:
        raise ValueError("WCE diagnostic distance chunks must be positive")

    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)
    validate_development_manifest(args.manifest)
    input_hashes = validate_frozen_inputs(
        args.manifest,
        args.scene_split,
        args.normalization,
    )
    formal_metrics = json.loads(
        args.formal_metrics.read_text(encoding="utf-8")
    )
    reference_frames = validate_formal_metrics(formal_metrics)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_evidence = validate_formal_checkpoint(
        checkpoint,
        checkpoint_path=args.checkpoint,
        formal_metrics=formal_metrics,
    )
    run_manifest_evidence = validate_run_manifest(
        repo,
        args.run_manifest,
        checkpoint,
        input_hashes,
    )
    records = validation_records(
        args.manifest,
        reference_frames,
        preflight=args.preflight,
    )
    log_center, log_scale = load_normalization(args.normalization)
    axes = load_axes(args.data_root / "resources")
    range_m = torch.as_tensor(axes.range_m, dtype=torch.float32, device=device)
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
    model = build_model(
        checkpoint,
        log_center=log_center,
        log_scale=log_scale,
        device=device,
    )
    inference_config = formal_inference_config(checkpoint)

    torch.manual_seed(FORMAL_SEED)
    torch.cuda.manual_seed_all(FORMAL_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    frame_reports: list[dict[str, Any]] = []
    for record, reference in records:
        item = load_current_frame(args.data_root, args.cache_root, record)
        if item["partition"] != "validation":
            raise ValueError("WCE diagnosis refuses non-validation frames")
        cube = item["cube_drae"].unsqueeze(0).to(device)
        target = item["target_xyz_confidence"].to(device)
        target_xyz = target[:, :3].float()
        target_weight = target[:, 3].float()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pool = reconstruct_matched_candidate_pool(
                model,
                cube,
                range_m,
                azimuth_rad,
                elevation_rad,
                inference_config,
            )
        control = validate_current_confidence_control(pool, reference)
        diagnosis = diagnose_frozen_candidate_pool(
            pool,
            target_xyz,
            target_weight,
            candidate_chunk_size=args.candidate_chunk_size,
            target_chunk_size=args.target_chunk_size,
        )
        target_radius = torch.linalg.vector_norm(target_xyz, dim=1)
        has_far_target = bool(
            ((target_radius >= 60.0) & (target_radius < 120.0)).any()
        )
        frame_reports.append(
            {
                "sequence": item["sequence"],
                "radar_index": item["radar_index"],
                "partition": item["partition"],
                "has_far_target": has_far_target,
                "current_inputs": {
                    "cube_path": item["current_cube_path"],
                    "target_cache_path": item["current_target_cache_path"],
                    "target_cache_sha256": item[
                        "current_target_cache_sha256"
                    ],
                    "future_cube_accessed": False,
                    "cfar_accessed": False,
                },
                "candidate_pool": {
                    **pool.report,
                    "xyz_sha256": tensor_sha256(pool.xyz_m),
                    "confidence_sha256": tensor_sha256(pool.confidence),
                },
                "current_confidence": {
                    "formal_hash_control": control,
                    "geometry": diagnosis.current_geometry,
                    "export_report": pool.current_confidence_export.report,
                },
                "gt_only_candidate_diagnostics": {
                    "candidate_to_target": diagnosis.support[
                        "candidate_to_target"
                    ],
                    "target_to_candidate": diagnosis.support[
                        "target_to_candidate"
                    ],
                    "per_range": diagnosis.support["per_range"],
                    "ranking_alignment": diagnosis.ranking,
                },
                "validation_gt_nearest_score": {
                    "geometry": (
                        diagnosis.validation_gt_nearest_score_geometry
                    ),
                    "export_report": (
                        diagnosis.validation_gt_nearest_score_export.report
                    ),
                    "hashes": (
                        diagnosis.validation_gt_nearest_score_export.hashes
                    ),
                },
            }
        )
        del item, cube, target, target_xyz, target_weight, pool, diagnosis
        torch.cuda.empty_cache()

    aggregate = aggregate_failure_factor_frames(
        frame_reports,
        preflight=args.preflight,
    )
    output = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "source_head": git_output(repo, "rev-parse", "HEAD"),
        "preflight": args.preflight,
        "artifact_label": diagnostic_artifact_label(),
        "checkpoint": checkpoint_evidence,
        "formal_run_manifest": run_manifest_evidence,
        "formal_metrics": {
            "path": str(args.formal_metrics.resolve()),
            "sha256": sha256(args.formal_metrics),
            "protocol": formal_metrics["protocol"],
            "epoch": int(formal_metrics["epoch"]),
        },
        "frozen_inputs": {
            "paths": {
                "manifest": str(args.manifest.resolve()),
                "scene_split": str(args.scene_split.resolve()),
                "normalization": str(args.normalization.resolve()),
            },
            "hashes": input_hashes,
            "test_partition_accessed": False,
        },
        "diagnostic_source_hashes": source_hashes(repo, Path(__file__)),
        "runtime": {
            "device_argument": args.device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "candidate_chunk_size": args.candidate_chunk_size,
            "target_chunk_size": args.target_chunk_size,
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        },
        "candidate_contract": inference_config.metadata(),
        "frame_count": len(frame_reports),
        "frames": frame_reports,
        "aggregate": aggregate,
        "evidence_boundary": {
            "training_started": False,
            "checkpoint_modified": False,
            "future_cube_accessed": False,
            "cfar_accessed": False,
            "test_partition_accessed": False,
            "doppler_head_evaluated": False,
            "coverage_aware_oracle_implemented": False,
            "strict_upper_bound_claimed": False,
        },
    }
    atomic_json(args.output, output)
    print(json.dumps(output["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
