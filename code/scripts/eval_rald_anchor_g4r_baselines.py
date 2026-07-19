#!/usr/bin/env python3
"""Evaluate single-frame and ego-aligned history aggregation for G4R."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.rald_prediction import (  # noqa: E402
    FrozenRaLDPredictionCache,
    RaLDPointPrediction,
)
from cube_dense.temporal_dataset import KRadarTemporalDataset  # noqa: E402
from eval.rald_temporal import (  # noqa: E402
    aggregate_rald_method_frames,
    finite_frame_metrics,
    rald_current_frame_report,
    rald_pair_temporal_report,
)
from models.temporal_baselines import (  # noqa: E402
    ego_warp_rald_prediction,
    raw_doppler_warp_rald_prediction,
    rald_history_aggregate,
)
from scripts.g1b_contract import sha256  # noqa: E402


PROTOCOL = "rald_anchor_g4r_baselines_v1"
ARMS = (
    "t0_single_frame",
    "history_aggregation",
    "raw_doppler_displacement_sensitivity",
)
ROLLOUT_HORIZONS = (1, 5, 10, 25)


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prediction_path(
    output: Path, arm: str, sequence: int, radar_index: int
) -> Path:
    return (
        output
        / "predictions"
        / arm
        / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"
    )


def write_prediction(
    path: Path,
    state: RaLDPointPrediction,
    sequence: int,
    radar_index: int,
    source_commit: str,
    arm: str,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            prediction_schema_version=np.asarray(1, dtype=np.int16),
            arm=np.asarray(arm),
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


def method_frame(
    arm: str,
    state: RaLDPointPrediction,
    previous: RaLDPointPrediction | None,
    pair: dict | None,
    item: dict,
    cube: torch.Tensor,
    target: torch.Tensor,
    target_index: torch.Tensor,
    axes_tensors: tuple[torch.Tensor, ...],
    record: dict,
    window_id: str,
    rollout_step: int,
    aggregation: dict | None,
) -> dict:
    doppler, ranges, azimuth, elevation = axes_tensors
    step = torch.median(torch.diff(doppler))
    current = rald_current_frame_report(
        state,
        cube,
        target,
        target_index,
        doppler,
        doppler[0],
        step * doppler.numel(),
        step,
    )
    temporal = None
    if previous is not None and pair is not None:
        transform = torch.as_tensor(
            pair["current_from_previous"],
            dtype=torch.float32,
            device=cube.device,
        ).reshape(4, 4)
        temporal = rald_pair_temporal_report(
            previous,
            state,
            transform,
            ranges,
            azimuth,
            elevation,
        )
    return {
        "arm": arm,
        "window_id": window_id,
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
        "rollout_step": rollout_step,
        "current": current,
        "temporal": temporal,
        "aggregation": aggregation,
        "prediction": record,
    }


def rollout_aggregates(frames: list[dict]) -> dict[str, dict]:
    result = {}
    for horizon in ROLLOUT_HORIZONS:
        selected = [frame for frame in frames if frame["rollout_step"] == horizon]
        if not selected:
            raise ValueError(f"Missing G4R baseline horizon {horizon}")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("G4R baseline evaluation requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.history_frames != 4:
        raise ValueError("Formal G4R history aggregation uses four frames")
    if args.output.exists() and any(args.output.iterdir()):
        if args.overwrite:
            shutil.rmtree(args.output)
        elif not args.resume:
            raise FileExistsError(f"G4R baseline output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    dense_hash = sha256(args.dense_cache_report)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("gate_pass") is not True:
        raise ValueError("G4R temporal manifest failed its data gate")
    dense = json.loads(args.dense_cache_report.read_text(encoding="utf-8"))
    if (
        dense.get("completed") is not True
        or dense["configuration"]["source_manifest_sha256"] != manifest_hash
    ):
        raise ValueError("G4R dense cache is incomplete or mismatched")
    parent_cache = FrozenRaLDPredictionCache(
        args.parent_prediction_cache, expected_frames=len(manifest["frames"])
    )
    if parent_cache.configuration["temporal_manifest_sha256"] != manifest_hash:
        raise ValueError("G4R parent prediction cache and manifest differ")
    point_count = int(parent_cache.configuration["point_count"])
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
        "seed": int(parent_cache.configuration["g3r_seed"]),
        "point_count": point_count,
        "partition": "validation",
        "history_frames": args.history_frames,
        "raw_doppler_displacement_is_sensitivity_only": True,
    }

    dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    pairs_by_current = {
        pair["current_dataset_index"]: pair for pair in dataset.pairs
    }
    axes = load_axes(args.data_root / "resources")
    device = torch.device(args.device)
    if "H200" not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError("Formal G4R baseline evaluation requires an H200")
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
    period = torch.median(torch.diff(doppler)) * doppler.numel()
    frames_by_arm = {arm: [] for arm in ARMS}
    for window in dataset.windows:
        previous_outputs = {}
        history_sources = []
        raw_doppler_sources = []
        for rollout_step, dataset_index in enumerate(window["dataset_indices"]):
            item = dataset.frame_dataset[dataset_index]
            sequence = int(item["sequence"])
            radar_index = int(item["radar_index"])
            t0 = parent_cache.load(sequence, radar_index, device)
            pair = None if rollout_step == 0 else pairs_by_current[dataset_index]
            if pair is None:
                history = t0
                raw_doppler = t0
                aggregation = None
                raw_aggregation = None
                history_sources = [t0]
                raw_doppler_sources = [t0]
            else:
                transform = torch.as_tensor(
                    pair["current_from_previous"],
                    dtype=torch.float32,
                    device=device,
                ).reshape(4, 4)
                warped = [
                    ego_warp_rald_prediction(
                        state,
                        transform,
                        doppler,
                        doppler[0],
                        period,
                        ranges,
                        azimuth,
                        elevation,
                    )
                    for state in history_sources
                ]
                history, diagnostics = rald_history_aggregate(
                    t0, warped, point_count
                )
                aggregation = asdict(diagnostics)
                history_sources = ([t0] + warped)[: args.history_frames]
                raw_warped = [
                    raw_doppler_warp_rald_prediction(
                        state,
                        transform,
                        pair["delta_seconds"],
                        doppler,
                        doppler[0],
                        period,
                        ranges,
                        azimuth,
                        elevation,
                    )
                    for state in raw_doppler_sources
                ]
                raw_doppler, raw_diagnostics = rald_history_aggregate(
                    t0, raw_warped, point_count
                )
                raw_aggregation = asdict(raw_diagnostics)
                raw_doppler_sources = ([t0] + raw_warped)[: args.history_frames]
            cube = item["cube_drae"].unsqueeze(0).to(device)
            target = item["target_xyz_confidence"].to(device)
            target_index = item["target_rae_index"].to(device)
            states = {
                "t0_single_frame": t0,
                "history_aggregation": history,
                "raw_doppler_displacement_sensitivity": raw_doppler,
            }
            for arm, state in states.items():
                if arm == "t0_single_frame" or rollout_step == 0:
                    record = cached_record(parent_cache, sequence, radar_index)
                else:
                    record = write_prediction(
                        prediction_path(args.output, arm, sequence, radar_index),
                        state,
                        sequence,
                        radar_index,
                        args.source_commit,
                        arm,
                    )
                frame = method_frame(
                    arm,
                    state,
                    previous_outputs.get(arm),
                    pair,
                    item,
                    cube,
                    target,
                    target_index,
                    tensors,
                    record,
                    window["window_id"],
                    rollout_step,
                    (
                        aggregation
                        if arm == "history_aggregation"
                        else raw_aggregation
                        if arm == "raw_doppler_displacement_sensitivity"
                        else None
                    ),
                )
                frames_by_arm[arm].append(frame)
                previous_outputs[arm] = state
            del item, cube, target, target_index, states
            torch.cuda.empty_cache()

    arms = {
        arm: {
            "aggregate": aggregate_rald_method_frames(frames),
            "rollout": rollout_aggregates(frames),
            "frames": frames,
        }
        for arm, frames in frames_by_arm.items()
    }
    expected_frames = sum(int(window["frame_count"]) for window in dataset.windows)
    checks = {
        "complete_arm_matrix": set(arms) == set(ARMS),
        "all_validation_frames": all(
            len(arms[arm]["frames"]) == expected_frames for arm in ARMS
        ),
        "fixed_point_count": all(
            int(frame["prediction"]["point_count"]) == point_count
            for arm in ARMS
            for frame in arms[arm]["frames"]
        ),
        "finite_metrics": all(
            finite_frame_metrics(frame)
            for arm in ARMS
            for frame in arms[arm]["frames"]
        ),
        "rollout_horizons_present": all(
            set(arms[arm]["rollout"]) == {str(value) for value in ROLLOUT_HORIZONS}
            for arm in ARMS
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "arms": arms,
        "checks": checks,
        "completed": all(checks.values()),
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps({"checks": checks, "completed": report["completed"]}, indent=2))
    if not report["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
