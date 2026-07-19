#!/usr/bin/env python3
"""Strict recurrent rollout for the selected RaLD-native G4R adapter."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.rald_prediction import (  # noqa: E402
    FrozenRaLDPredictionCache,
    RaLDPointPrediction,
    rald_prediction_from_output,
)
from cube_dense.rald_run import build_temporal_rald, load_rald_run  # noqa: E402
from cube_dense.temporal_dataset import KRadarTemporalDataset  # noqa: E402
from eval.rald_temporal import (  # noqa: E402
    aggregate_rald_method_frames,
    finite_frame_metrics,
    rald_current_frame_report,
    rald_pair_temporal_report,
)
from models.temporal_prior import ego_pose_warp  # noqa: E402
from scripts.g1b_contract import sha256  # noqa: E402


PROTOCOL = "rald_anchor_g4r_strict_rollout_v1"
ROLLOUT_HORIZONS = (1, 5, 10, 25)


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prediction_path(output: Path, sequence: int, radar_index: int) -> Path:
    return output / "predictions" / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


def write_prediction(
    path: Path,
    state: RaLDPointPrediction,
    sequence: int,
    radar_index: int,
    source_commit: str,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            prediction_schema_version=np.asarray(1, dtype=np.int16),
            sequence=np.asarray(sequence, dtype=np.int16),
            radar_index=np.asarray(radar_index, dtype=np.int32),
            source_commit=np.asarray(source_commit),
            xyz_m=state.xyz_m.detach().cpu().numpy().astype(np.float16),
            coordinates_rae=(
                state.coordinates_rae.detach().cpu().numpy().astype(np.float16)
            ),
            doppler_probability=(
                state.probability.detach().cpu().numpy().astype(np.float16)
            ),
            confidence=state.confidence.detach().cpu().numpy().astype(np.float16),
        )
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "point_count": int(state.xyz_m.shape[0]),
    }


def cached_record(
    cache: FrozenRaLDPredictionCache,
    sequence: int,
    radar_index: int,
) -> dict:
    record = cache.records[(sequence, radar_index)]
    return {
        "path": record["prediction"],
        "sha256": record["prediction_sha256"],
        "point_count": int(cache.configuration["point_count"]),
    }


def prior_from_state(state: RaLDPointPrediction, pair: dict, model):
    transform = torch.as_tensor(
        pair["current_from_previous"],
        dtype=state.xyz_m.dtype,
        device=state.xyz_m.device,
    ).reshape(4, 4)
    return ego_pose_warp(
        state.xyz_m,
        state.probability,
        state.confidence,
        transform,
        model.doppler_mps,
        model.doppler_lower_mps,
        model.doppler_period_mps,
        model.range_m,
        model.azimuth_rad,
        model.elevation_rad,
    )


def rollout_aggregates(frames: list[dict]) -> dict[str, dict]:
    result = {}
    for horizon in ROLLOUT_HORIZONS:
        selected = [frame for frame in frames if frame["rollout_step"] == horizon]
        if not selected:
            raise ValueError(f"Missing strict G4R rollout horizon {horizon}")
        result[str(horizon)] = aggregate_rald_method_frames(selected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--dense-cache-report", type=Path, required=True)
    parser.add_argument("--parent-prediction-cache", type=Path, required=True)
    parser.add_argument("--preflight-selection", type=Path, required=True)
    parser.add_argument("--temporal-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Strict G4R rollout requires CUDA")
    if args.warmup_iterations < 0:
        raise ValueError("G4R rollout warmup iterations must be non-negative")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.output.exists() and any(args.output.iterdir()):
        if args.overwrite:
            shutil.rmtree(args.output)
        elif not args.resume:
            raise FileExistsError(f"G4R rollout output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    dense_hash = sha256(args.dense_cache_report)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dense = json.loads(args.dense_cache_report.read_text(encoding="utf-8"))
    if manifest.get("gate_pass") is not True or dense.get("completed") is not True:
        raise ValueError("G4R rollout data gates are incomplete")
    if dense["configuration"]["source_manifest_sha256"] != manifest_hash:
        raise ValueError("G4R rollout dense cache and manifest differ")
    selection = json.loads(args.preflight_selection.read_text(encoding="utf-8"))
    if (
        selection.get("completed") is not True
        or selection.get("source_commit") != args.source_commit
    ):
        raise ValueError("G4R preflight selection is incomplete or stale")
    run_document = json.loads(
        (args.temporal_run / "config.json").read_text(encoding="utf-8")
    )
    config = run_document["config"]
    provenance = run_document["provenance"]
    if (
        config["fusion_mode"] != selection["selected_fusion_mode"]
        or int(config["epochs"]) != 20
        or provenance["git_commit"] != args.source_commit
        or provenance["zero_gate_identity"]["exact_identity"] is not True
    ):
        raise ValueError("Formal G4R run differs from the frozen selection")
    checkpoint_path = (args.temporal_run / "best.pt").resolve()
    parent_run_path = Path(provenance["parent_checkpoint"]).resolve().parent
    parent_run = load_rald_run(parent_run_path)
    device = torch.device(args.device)
    if "H200" not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError("Formal G4R rollout requires an H200")
    axes = load_axes(args.data_root / "resources")
    model = build_temporal_rald(
        parent_run,
        axes,
        device,
        config["fusion_mode"],
        prior_base_channels=int(config["prior_base_channels"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config"] != config or checkpoint["provenance"] != provenance:
        raise ValueError("G4R rollout checkpoint metadata differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    parent_cache = FrozenRaLDPredictionCache(
        args.parent_prediction_cache, expected_frames=len(manifest["frames"])
    )
    if (
        sha256(parent_cache.manifest_path)
        != provenance["parent_prediction_manifest_sha256"]
        or parent_cache.configuration["g3r_checkpoint_sha256"]
        != provenance["parent_checkpoint_sha256"]
    ):
        raise ValueError("G4R rollout teacher cache differs from training")
    point_count = int(config["point_count"])
    configuration = {
        "source_commit": args.source_commit,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": normalization_hash,
        "dense_cache_report_sha256": dense_hash,
        "parent_prediction_manifest_path": str(parent_cache.manifest_path),
        "parent_prediction_manifest_sha256": sha256(
            parent_cache.manifest_path
        ),
        "g3r_comparison_path": parent_cache.configuration["g3r_comparison"],
        "g3r_comparison_sha256": parent_cache.configuration[
            "g3r_comparison_sha256"
        ],
        "g3r_config_path": parent_cache.configuration["g3r_config"],
        "g3r_config_sha256": parent_cache.configuration["g3r_config_sha256"],
        "g3r_checkpoint_path": parent_cache.configuration["g3r_checkpoint"],
        "g3r_checkpoint_sha256": parent_cache.configuration[
            "g3r_checkpoint_sha256"
        ],
        "temporal_config_path": str((args.temporal_run / "config.json").resolve()),
        "temporal_config_sha256": sha256(args.temporal_run / "config.json"),
        "temporal_checkpoint_path": str(checkpoint_path),
        "temporal_checkpoint_sha256": sha256(checkpoint_path),
        "preflight_selection_path": str(args.preflight_selection.resolve()),
        "preflight_selection_sha256": sha256(args.preflight_selection),
        "model_source_commit": provenance["git_commit"],
        "seed": int(config["seed"]),
        "point_count": point_count,
        "fusion_mode": config["fusion_mode"],
        "partition": "validation",
        "strict_recurrent_rollout": True,
        "warmup_iterations": args.warmup_iterations,
    }

    dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    pairs_by_current = {
        pair["current_dataset_index"]: pair for pair in dataset.pairs
    }
    tensors = tuple(
        torch.as_tensor(value, device=device, dtype=torch.float32)
        for value in (
            axes.doppler_mps,
            axes.range_m,
            axes.azimuth_rad,
            axes.elevation_rad,
        )
    )
    doppler, ranges, azimuth, elevation = tensors
    doppler_step = torch.median(torch.diff(doppler))
    frames = []
    inference_times = []
    warmup_runs = 0
    warmup_complete = args.warmup_iterations == 0
    torch.cuda.reset_peak_memory_stats(device)
    for window in dataset.windows:
        previous = None
        for rollout_step, dataset_index in enumerate(window["dataset_indices"]):
            item = dataset.frame_dataset[dataset_index]
            sequence = int(item["sequence"])
            radar_index = int(item["radar_index"])
            pair = None if rollout_step == 0 else pairs_by_current[dataset_index]
            cube = item["cube_drae"].unsqueeze(0).to(device)
            if pair is None:
                state = parent_cache.load(sequence, radar_index, device)
                record = cached_record(parent_cache, sequence, radar_index)
            else:
                prior = prior_from_state(previous, pair, model)
                if not warmup_complete:
                    for _ in range(args.warmup_iterations):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            warmup_output = model(cube, prior)
                        del warmup_output
                        warmup_runs += 1
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    warmup_complete = True
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(cube, prior)
                torch.cuda.synchronize(device)
                inference_times.append(time.perf_counter() - started)
                state = rald_prediction_from_output(output).detached()
                record = write_prediction(
                    prediction_path(args.output, sequence, radar_index),
                    state,
                    sequence,
                    radar_index,
                    args.source_commit,
                )
            target = item["target_xyz_confidence"].to(device)
            target_index = item["target_rae_index"].to(device)
            current = rald_current_frame_report(
                state,
                cube,
                target,
                target_index,
                doppler,
                doppler[0],
                doppler_step * doppler.numel(),
                doppler_step,
            )
            temporal = None
            if previous is not None and pair is not None:
                transform = torch.as_tensor(
                    pair["current_from_previous"],
                    dtype=torch.float32,
                    device=device,
                ).reshape(4, 4)
                temporal = rald_pair_temporal_report(
                    previous,
                    state,
                    transform,
                    ranges,
                    azimuth,
                    elevation,
                )
            frames.append(
                {
                    "window_id": window["window_id"],
                    "sequence": sequence,
                    "radar_index": radar_index,
                    "rollout_step": rollout_step,
                    "current": current,
                    "temporal": temporal,
                    "prediction": record,
                }
            )
            previous = state
            del item, cube, target, target_index
            torch.cuda.empty_cache()

    aggregate = aggregate_rald_method_frames(frames)
    rollout = rollout_aggregates(frames)
    parent_prediction_hashes = {
        record["prediction_sha256"] for record in parent_cache.records.values()
    }
    checks = {
        "all_validation_frames": len(frames)
        == sum(int(window["frame_count"]) for window in dataset.windows),
        "fixed_point_count": all(
            int(frame["prediction"]["point_count"]) == point_count
            for frame in frames
        ),
        "finite_metrics": all(finite_frame_metrics(frame) for frame in frames),
        "strict_rollout_after_t0_anchor": all(
            (frame["rollout_step"] == 0)
            == (frame["prediction"]["sha256"] in parent_prediction_hashes)
            for frame in frames
        ),
        "rollout_horizons_present": set(rollout)
        == {str(value) for value in ROLLOUT_HORIZONS},
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "training_checks": {
            "formal_20_epochs": int(config["epochs"]) == 20,
            "five_epoch_temporal_warmup": int(config["temporal_warmup_epochs"]) == 5,
            "scheduled_sampling_0p4": float(config["scheduled_sampling_maximum"])
            == 0.4,
            "zero_gate_identity": provenance["zero_gate_identity"]["exact_identity"]
            is True,
        },
        "aggregate": aggregate,
        "rollout": rollout,
        "frames": frames,
        "efficiency": {
            "warmup_iterations_requested": args.warmup_iterations,
            "warmup_iterations_completed": warmup_runs,
            "warmup_excluded_from_timing": warmup_complete,
            "mean_inference_seconds": float(np.mean(inference_times)),
            "median_inference_seconds": float(np.median(inference_times)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "checks": checks,
        "completed": all(checks.values()),
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps({"checks": checks, "completed": report["completed"]}, indent=2))
    if not report["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
