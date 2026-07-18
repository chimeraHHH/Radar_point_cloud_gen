#!/usr/bin/env python3
"""Strict recurrent rollout evaluation for one selected formal G4 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.parent_prediction import (  # noqa: E402
    FrozenPredictionCache,
    PointPrediction,
    prediction_from_output,
)
from cube_dense.temporal_dataset import KRadarTemporalDataset  # noqa: E402
from eval.temporal_methods import (  # noqa: E402
    aggregate_flat_reports,
    aggregate_method_frames,
    current_frame_report,
    pair_temporal_report,
)
from models.cube_temporal import CubeTemporalNet  # noqa: E402
from models.temporal_inference import predict_temporal_pair  # noqa: E402


PROTOCOL = "g4_temporal_strict_rollout_v1"
TEST_PROTOCOL = "p5_temporal_strict_rollout_test_v1"
FUSION_ARMS = {
    "concat": "T4",
    "cross_attention": "T5",
    "draft_refinement": "T6",
}
FORMAL_SEEDS = {20260716, 20260717, 20260718}
ROLLOUT_HORIZONS = (1, 5, 10, 25)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prediction_path(
    output: Path,
    arm_id: str,
    sequence: int,
    radar_index: int,
) -> Path:
    return (
        output
        / "predictions"
        / arm_id.lower()
        / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"
    )


def write_prediction(
    path: Path,
    state: PointPrediction,
    arm_id: str,
    sequence: int,
    radar_index: int,
    model_checkpoint_sha256: str,
    source_commit: str,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            prediction_schema_version=np.asarray(2, dtype=np.int16),
            arm=np.asarray(arm_id),
            sequence=np.asarray(sequence, dtype=np.int16),
            radar_index=np.asarray(radar_index, dtype=np.int32),
            model_checkpoint_sha256=np.asarray(model_checkpoint_sha256),
            source_commit=np.asarray(source_commit),
            xyz_m=state.xyz_m.detach().cpu().numpy().astype(np.float16),
            coordinates_rae=(
                state.coordinates_rae.detach().cpu().numpy().astype(np.float16)
            ),
            doppler_probability=(
                state.probability.detach().cpu().numpy().astype(np.float16)
            ),
            confidence=state.confidence.detach().cpu().numpy().astype(np.float16),
            static_center_mps=(
                state.static_center_mps.detach().cpu().numpy().astype(np.float16)
            ),
        )
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "point_count": int(state.xyz_m.shape[0]),
    }


def parent_prediction_record(
    cache: FrozenPredictionCache,
    sequence: int,
    radar_index: int,
    point_count: int,
) -> dict:
    record = cache.records[(sequence, radar_index)]
    return {
        "path": record["prediction"],
        "sha256": record["prediction_sha256"],
        "point_count": point_count,
    }


def valid_window_result(window: dict, point_count: int) -> bool:
    if int(window.get("frame_count", -1)) != 48:
        return False
    for frame in window.get("frames", []):
        prediction = frame.get("prediction", {})
        path = Path(prediction.get("path", ""))
        if (
            not path.is_file()
            or int(prediction.get("point_count", -1)) != point_count
            or sha256(path) != prediction.get("sha256")
        ):
            return False
    return len(window.get("frames", [])) == 48


def validate_training_schedule(run: Path, config: dict) -> dict[str, bool]:
    records = [
        json.loads(line)
        for line in (run / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_epoch = {int(record["epoch"]): record for record in records}
    checks = {
        "all_20_epochs_logged": set(by_epoch) == set(range(1, 21)),
        "first_five_epochs_parent_frozen": all(
            by_epoch[epoch]["parent_trainable"] is False
            for epoch in range(1, 6)
            if epoch in by_epoch
        ),
        "last_15_epochs_joint": all(
            by_epoch[epoch]["parent_trainable"] is True
            for epoch in range(6, 21)
            if epoch in by_epoch
        ),
        "scheduled_sampling_starts_after_epoch_six": all(
            float(by_epoch[epoch]["scheduled_sampling_probability"]) == 0.0
            for epoch in range(1, 7)
            if epoch in by_epoch
        ),
        "scheduled_sampling_reaches_0p4": 20 in by_epoch
        and abs(float(by_epoch[20]["scheduled_sampling_probability"]) - 0.4) < 1e-12,
        "recurrent_exposures_observed": sum(
            int(record["scheduled_exposure_count"]) for record in records
        )
        > 0,
        "formal_config_locked": (
            int(config["epochs"]) == 20
            and int(config["joint_start_epoch"]) == 6
            and float(config["scheduled_sampling_maximum"]) == 0.4
            and config["train_window_limit"] is None
            and config["validation_window_limit"] is None
        ),
    }
    return checks


def rollout_aggregates(frames: list[dict]) -> dict[str, dict]:
    result = {}
    for horizon in ROLLOUT_HORIZONS:
        selected = [frame for frame in frames if frame["rollout_step"] == horizon]
        if not selected:
            raise ValueError(f"Missing selected-model rollout horizon {horizon}")
        result[str(horizon)] = aggregate_method_frames(selected)
    return result


def warm_up(
    model: CubeTemporalNet,
    dataset: KRadarTemporalDataset,
    parent_cache: FrozenPredictionCache,
    axes,
    point_count: int,
    dynamic_threshold_mps: float,
    iterations: int,
    device: torch.device,
) -> None:
    window = dataset.windows[0]
    previous_index, current_index = window["dataset_indices"][:2]
    previous_record = dataset.frame_dataset.records[previous_index]
    previous = parent_cache.load(
        int(previous_record["sequence"]),
        int(previous_record["radar_index"]),
        device,
    )
    pair = next(
        pair
        for pair in dataset.pairs
        if int(pair["current_dataset_index"]) == current_index
    )
    item = dataset.frame_dataset[current_index]
    with torch.inference_mode():
        for _ in range(iterations):
            output = predict_temporal_pair(
                model,
                item,
                previous,
                pair,
                axes,
                point_count,
                dynamic_threshold_mps,
                device,
            )
            del output
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def evaluate_window(
    model: CubeTemporalNet,
    window: dict,
    dataset: KRadarTemporalDataset,
    pairs_by_current: dict[int, dict],
    parent_cache: FrozenPredictionCache,
    axes,
    point_count: int,
    dynamic_threshold_mps: float,
    static_hypothesis: str,
    output_root: Path,
    arm_id: str,
    checkpoint_hash: str,
    source_commit: str,
    device: torch.device,
) -> dict:
    doppler, ranges, azimuth, elevation = (
        model.doppler_mps,
        model.range_m,
        model.azimuth_rad,
        model.elevation_rad,
    )
    frames = []
    previous = None
    for rollout_step, dataset_index in enumerate(window["dataset_indices"]):
        item = dataset.frame_dataset[dataset_index]
        sequence = int(item["sequence"])
        radar_index = int(item["radar_index"])
        pair = None if rollout_step == 0 else pairs_by_current[dataset_index]
        efficiency = None
        if pair is None:
            state = parent_cache.load(sequence, radar_index, device)
            cube = item["cube_drae"].unsqueeze(0).to(device)
            prediction_record = parent_prediction_record(
                parent_cache, sequence, radar_index, point_count
            )
        else:
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode():
                pair_output = predict_temporal_pair(
                    model,
                    item,
                    previous,
                    pair,
                    axes,
                    point_count,
                    dynamic_threshold_mps,
                    device,
                )
            end.record()
            torch.cuda.synchronize(device)
            latency_ms = float(start.elapsed_time(end))
            peak_memory_mib = float(
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            )
            state = prediction_from_output(
                pair_output["prediction"],
                pair_output["confidence"],
                pair_output["current_static_center_mps"],
            ).detached()
            cube = pair_output["cube"]
            efficiency = {
                "latency_ms": latency_ms,
                "peak_memory_mib": peak_memory_mib,
                "point_throughput_per_second": point_count
                / max(latency_ms / 1000.0, 1e-9),
            }
            prediction_record = write_prediction(
                prediction_path(output_root, arm_id, sequence, radar_index),
                state,
                arm_id,
                sequence,
                radar_index,
                checkpoint_hash,
                source_commit,
            )
        target = item["target_xyz_confidence"].to(device)
        target_indices = item["target_rae_index"].to(device)
        current = current_frame_report(
            state,
            cube,
            target,
            target_indices,
            item["ego_speed_mps"].to(device),
            static_hypothesis,
            doppler,
            model.doppler_lower_mps,
            model.doppler_period_mps,
            model.doppler_step_mps,
        )
        temporal = None
        if previous is not None:
            transform = torch.tensor(
                pair["current_from_previous"], dtype=torch.float32, device=device
            ).reshape(4, 4)
            temporal = pair_temporal_report(
                previous,
                state,
                transform,
                pair["delta_seconds"],
                doppler,
                model.doppler_lower_mps,
                model.doppler_period_mps,
                ranges,
                azimuth,
                elevation,
                dynamic_threshold_mps=dynamic_threshold_mps,
            )
        frames.append(
            {
                "arm": arm_id,
                "window_id": window["window_id"],
                "sequence": sequence,
                "radar_index": radar_index,
                "rollout_step": rollout_step,
                "delta_seconds": None
                if pair is None
                else float(pair["delta_seconds"]),
                "current": current,
                "temporal": temporal,
                "efficiency": efficiency,
                "prediction": prediction_record,
            }
        )
        previous = state
        if pair is not None:
            del pair_output
        del item, cube, target, target_indices, state
        torch.cuda.empty_cache()
    return {
        "window_id": window["window_id"],
        "sequence": int(window["sequence"]),
        "frame_count": len(frames),
        "frames": frames,
    }


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
    parser.add_argument(
        "--partition", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Formal G4 rollout evaluation requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.warmup_iterations != 3:
        raise ValueError("Formal G4 latency protocol requires three warm-up iterations")

    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    dense_report_hash = sha256(args.dense_cache_report)
    temporal_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dense_report = json.loads(args.dense_cache_report.read_text(encoding="utf-8"))
    if temporal_manifest.get("gate_pass") is not True:
        raise ValueError("Temporal rollout manifest did not pass its gate")
    if (
        dense_report.get("completed") is not True
        or dense_report["configuration"]["source_manifest_sha256"] != manifest_hash
    ):
        raise ValueError("Temporal rollout dense cache is incomplete or mismatched")
    selection = json.loads(args.preflight_selection.read_text(encoding="utf-8"))
    if selection.get("completed") is not True:
        raise ValueError("G4 preflight selection is incomplete")

    run_document = json.loads(
        (args.temporal_run / "config.json").read_text(encoding="utf-8")
    )
    config = run_document["config"]
    provenance = run_document["provenance"]
    fusion_mode = config["fusion_mode"]
    if fusion_mode != selection["selected_fusion_mode"]:
        raise ValueError("Formal G4 model differs from the frozen preflight selection")
    if provenance["git_commit"] != selection["source_commit"]:
        raise ValueError("Formal G4 model and preflight selection source differ")
    seed = int(config["seed"])
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Unexpected formal G4 seed {seed}")
    if (
        provenance["scene_split_sha256"] != scene_split_hash
        or provenance["normalization_sha256"] != normalization_hash
    ):
        raise ValueError("Temporal model split or normalization provenance differs")
    if args.partition == "validation" and (
        provenance["manifest_sha256"] != manifest_hash
        or provenance["dense_cache_report_sha256"] != dense_report_hash
    ):
        raise ValueError("Formal G4 validation data provenance differs")
    checkpoint_path = (args.temporal_run / "best.pt").resolve()
    checkpoint_hash = sha256(checkpoint_path)
    parent_cache = FrozenPredictionCache(
        args.parent_prediction_cache, expected_frames=len(temporal_manifest["frames"])
    )
    if (
        parent_cache.configuration["parent_checkpoint_sha256"]
        != provenance["parent_checkpoint_sha256"]
        or parent_cache.configuration["dense_cache_report_sha256"]
        != dense_report_hash
    ):
        raise ValueError("Temporal rollout parent cache differs")
    if args.partition == "validation" and (
        sha256(parent_cache.manifest_path)
        != provenance["parent_prediction_manifest_sha256"]
    ):
        raise ValueError("Formal G4 rollout parent cache differs from training")
    point_count = int(config["point_count"])
    if int(parent_cache.configuration["point_count"]) != point_count:
        raise ValueError("Formal G4 rollout point count differs from its parent")
    training_checks = validate_training_schedule(args.temporal_run, config)
    if not all(training_checks.values()):
        raise ValueError(f"Formal G4 training schedule failed checks: {training_checks}")

    nonempty = args.output.exists() and any(args.output.iterdir())
    if nonempty and args.overwrite:
        shutil.rmtree(args.output)
        nonempty = False
    if nonempty and not args.resume:
        raise FileExistsError(f"G4 rollout output is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No G4 rollout to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    axes = load_axes(args.data_root / "resources")
    model = CubeTemporalNet(
        fusion_mode,
        config["parent_head_mode"],
        torch.from_numpy(axes.doppler_mps),
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        base_channels=int(config["base_channels"]),
        log_center=float(config["log_center"]),
        log_scale=float(config["log_scale"]),
        static_hypothesis=config["static_hypothesis"],
        maximum_offset_bins=float(config["maximum_offset_bins"]),
        attention_neighbors=int(config["attention_neighbors"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, (args.partition,)
    )
    if (
        len(dataset.windows) != 8
        or len(dataset.frame_dataset) != 384
        or len(dataset.pairs) != 376
    ):
        raise ValueError("Formal G4 rollout requires 8 windows / 384 frames / 376 pairs")
    pairs_by_current = {
        int(pair["current_dataset_index"]): pair for pair in dataset.pairs
    }
    arm_id = FUSION_ARMS[fusion_mode]
    configuration = {
        "protocol": PROTOCOL if args.partition == "validation" else TEST_PROTOCOL,
        "evaluator_source_commit": args.source_commit,
        "partition": args.partition,
        "model_source_commit": provenance["git_commit"],
        "fusion_mode": fusion_mode,
        "arm_id": arm_id,
        "seed": seed,
        "parent_variant": config["parent_variant"],
        "manifest_sha256": manifest_hash,
        "training_manifest_sha256": provenance["manifest_sha256"],
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": normalization_hash,
        "dense_cache_report_sha256": dense_report_hash,
        "preflight_selection_sha256": sha256(args.preflight_selection),
        "temporal_run": str(args.temporal_run.resolve()),
        "temporal_checkpoint_sha256": checkpoint_hash,
        "parent_prediction_manifest_sha256": sha256(parent_cache.manifest_path),
        "point_count": point_count,
        "dynamic_threshold_mps": float(config["dynamic_threshold_mps"]),
        "warmup_iterations": args.warmup_iterations,
        "strict_recurrent_rollout": True,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    progress_path = args.output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["configuration"] != configuration:
            raise ValueError("G4 rollout resume configuration differs")
    else:
        progress = {
            "schema_version": 1,
            "configuration": configuration,
            "windows": {},
            "completed": False,
        }
        atomic_json(progress_path, progress)

    pending_windows = [
        window
        for window in dataset.windows
        if not valid_window_result(
            progress["windows"].get(window["window_id"], {}), point_count
        )
    ]
    if pending_windows:
        warm_up(
            model,
            dataset,
            parent_cache,
            axes,
            point_count,
            float(config["dynamic_threshold_mps"]),
            args.warmup_iterations,
            device,
        )
    for window in dataset.windows:
        window_id = window["window_id"]
        prior = progress["windows"].get(window_id)
        if prior is not None and valid_window_result(prior, point_count):
            print(json.dumps({"window": window_id, "status": "cached"}), flush=True)
            continue
        if prior is not None:
            del progress["windows"][window_id]
            atomic_json(progress_path, progress)
        result = evaluate_window(
            model,
            window,
            dataset,
            pairs_by_current,
            parent_cache,
            axes,
            point_count,
            float(config["dynamic_threshold_mps"]),
            config["static_hypothesis"],
            args.output,
            arm_id,
            checkpoint_hash,
            args.source_commit,
            device,
        )
        progress["windows"][window_id] = result
        atomic_json(progress_path, progress)
        print(json.dumps({"window": window_id, "status": "complete"}), flush=True)

    frames = [
        frame
        for window_id in sorted(progress["windows"])
        for frame in progress["windows"][window_id]["frames"]
    ]
    efficiency = [
        frame["efficiency"] for frame in frames if frame["efficiency"] is not None
    ]
    checks = {
        "training_schedule_verified": all(training_checks.values()),
        "all_evaluation_windows_complete": len(progress["windows"]) == 8,
        "all_evaluation_frames_complete": len(frames) == 384,
        "all_temporal_pairs_strictly_recurrent": sum(
            frame["temporal"] is not None for frame in frames
        )
        == 376,
        "all_inference_efficiency_records_present": len(efficiency) == 376,
        "all_outputs_have_exact_point_count": all(
            frame["prediction"]["point_count"] == point_count for frame in frames
        ),
        "evaluation_partition_only": all(
            record["partition"] == args.partition
            for record in dataset.frame_dataset.records
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL if args.partition == "validation" else TEST_PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "training_checks": training_checks,
        "aggregate": aggregate_method_frames(frames),
        "rollout": rollout_aggregates(frames),
        "efficiency": aggregate_flat_reports(efficiency),
        "frames": frames,
        "checks": checks,
        "completed": all(checks.values()),
    }
    report_path = args.output / "report.json"
    atomic_json(report_path, report)
    progress["checks"] = checks
    progress["report"] = str(report_path)
    progress["report_sha256"] = sha256(report_path)
    progress["completed"] = report["completed"]
    atomic_json(progress_path, progress)
    print(json.dumps({"checks": checks, "completed": report["completed"]}, indent=2))
    if not report["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
