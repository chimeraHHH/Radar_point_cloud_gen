#!/usr/bin/env python3
"""Evaluate and cache the frozen T0-T3 temporal baseline matrix for G4."""

from __future__ import annotations

import argparse
import hashlib
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
from cube_dense.parent_prediction import (  # noqa: E402
    FrozenPredictionCache,
    PointPrediction,
)
from cube_dense.temporal_dataset import KRadarTemporalDataset  # noqa: E402
from eval.temporal_methods import (  # noqa: E402
    aggregate_method_frames,
    current_frame_report,
    pair_temporal_report,
)
from models.temporal_baselines import (  # noqa: E402
    AggregationDiagnostics,
    doppdrive_aggregate,
    warp_prediction,
)


PROTOCOL = "g4_temporal_baselines_v1"
ARMS = {
    "t0_single_frame": "T0",
    "t1_ego_copy": "T1",
    "t2_doppler_copy": "T2",
    "t3_doppdrive": "T3",
}
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
    arm: str,
    sequence: int,
    radar_index: int,
) -> Path:
    return (
        output
        / "predictions"
        / arm
        / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"
    )


def write_prediction(
    path: Path,
    state: PointPrediction,
    arm: str,
    sequence: int,
    radar_index: int,
    source_commit: str,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            prediction_schema_version=np.asarray(2, dtype=np.int16),
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


def cached_parent_record(
    cache: FrozenPredictionCache,
    sequence: int,
    radar_index: int,
    point_count: int,
) -> dict:
    record = cache.records[(sequence, radar_index)]
    path = Path(record["prediction"])
    return {
        "path": str(path),
        "sha256": record["prediction_sha256"],
        "point_count": point_count,
    }


def valid_window_result(window: dict, point_count: int) -> bool:
    for frames in window.get("arms", {}).values():
        for frame in frames:
            prediction = frame.get("prediction", {})
            path = Path(prediction.get("path", ""))
            if (
                not path.is_file()
                or int(prediction.get("point_count", -1)) != point_count
                or sha256(path) != prediction.get("sha256")
            ):
                return False
    return set(window.get("arms", {})) == set(ARMS)


def method_frame(
    arm: str,
    state: PointPrediction,
    previous: PointPrediction | None,
    pair: dict | None,
    item: dict,
    cube: torch.Tensor,
    target: torch.Tensor,
    target_indices: torch.Tensor,
    axes_tensors: tuple[torch.Tensor, ...],
    static_hypothesis: str,
    dynamic_threshold_mps: float,
    prediction_record: dict,
    window_id: str,
    rollout_step: int,
    aggregation: AggregationDiagnostics | None,
) -> dict:
    doppler, ranges, azimuth, elevation = axes_tensors
    step = torch.median(torch.diff(doppler))
    lower = doppler[0]
    period = step * doppler.numel()
    current = current_frame_report(
        state,
        cube,
        target,
        target_indices,
        item["ego_speed_mps"].to(cube.device),
        static_hypothesis,
        doppler,
        lower,
        period,
        step,
    )
    temporal = None
    if previous is not None and pair is not None:
        transform = torch.tensor(
            pair["current_from_previous"], dtype=torch.float32, device=cube.device
        ).reshape(4, 4)
        temporal = pair_temporal_report(
            previous,
            state,
            transform,
            pair["delta_seconds"],
            doppler,
            lower,
            period,
            ranges,
            azimuth,
            elevation,
            dynamic_threshold_mps=dynamic_threshold_mps,
        )
    return {
        "arm": arm,
        "window_id": window_id,
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
        "rollout_step": rollout_step,
        "delta_seconds": None if pair is None else float(pair["delta_seconds"]),
        "current": current,
        "temporal": temporal,
        "aggregation": None if aggregation is None else asdict(aggregation),
        "prediction": prediction_record,
    }


def evaluate_window(
    window: dict,
    dataset: KRadarTemporalDataset,
    pairs_by_current: dict[int, dict],
    parent_cache: FrozenPredictionCache,
    output: Path,
    point_count: int,
    axes_tensors: tuple[torch.Tensor, ...],
    static_hypothesis: str,
    dynamic_threshold_mps: float,
    history_frames: int,
    source_commit: str,
    device: torch.device,
) -> dict:
    doppler, ranges, azimuth, elevation = axes_tensors
    step = torch.median(torch.diff(doppler))
    lower = doppler[0]
    period = step * doppler.numel()
    arm_frames = {arm: [] for arm in ARMS}
    previous_outputs: dict[str, PointPrediction] = {}
    recurrent_ego = None
    recurrent_doppler = None
    history_sources: list[PointPrediction] = []

    for rollout_step, dataset_index in enumerate(window["dataset_indices"]):
        record = dataset.frame_dataset.records[dataset_index]
        sequence = int(record["sequence"])
        radar_index = int(record["radar_index"])
        t0 = parent_cache.load(sequence, radar_index, device)
        pair = None if rollout_step == 0 else pairs_by_current[dataset_index]
        item = dataset.frame_dataset[dataset_index]
        aggregation = None
        if pair is None:
            recurrent_ego = t0
            recurrent_doppler = t0
            t3 = t0
            history_sources = [t0]
            states = {
                "t0_single_frame": t0,
                "t1_ego_copy": t0,
                "t2_doppler_copy": t0,
                "t3_doppdrive": t0,
            }
        else:
            transform = torch.tensor(
                pair["current_from_previous"], dtype=torch.float32, device=device
            ).reshape(4, 4)
            ego_speed = item["ego_speed_mps"].to(device)
            recurrent_ego = warp_prediction(
                recurrent_ego,
                transform,
                pair["delta_seconds"],
                doppler,
                lower,
                period,
                ranges,
                azimuth,
                elevation,
                ego_speed,
                static_hypothesis,
                apply_doppler_displacement=False,
                dynamic_threshold_mps=dynamic_threshold_mps,
            )
            recurrent_doppler = warp_prediction(
                recurrent_doppler,
                transform,
                pair["delta_seconds"],
                doppler,
                lower,
                period,
                ranges,
                azimuth,
                elevation,
                ego_speed,
                static_hypothesis,
                apply_doppler_displacement=True,
                dynamic_threshold_mps=dynamic_threshold_mps,
            )
            warped_history = [
                warp_prediction(
                    history,
                    transform,
                    pair["delta_seconds"],
                    doppler,
                    lower,
                    period,
                    ranges,
                    azimuth,
                    elevation,
                    ego_speed,
                    static_hypothesis,
                    apply_doppler_displacement=True,
                    dynamic_threshold_mps=dynamic_threshold_mps,
                )
                for history in history_sources
            ]
            t3, aggregation = doppdrive_aggregate(
                t0,
                warped_history,
                point_count,
                ego_speed,
                static_hypothesis,
                lower,
                period,
            )
            history_sources = ([t0] + warped_history)[:history_frames]
            states = {
                "t0_single_frame": t0,
                "t1_ego_copy": recurrent_ego,
                "t2_doppler_copy": recurrent_doppler,
                "t3_doppdrive": t3,
            }
        cube = item["cube_drae"].unsqueeze(0).to(device)
        target = item["target_xyz_confidence"].to(device)
        target_indices = item["target_rae_index"].to(device)
        for arm, state in states.items():
            if rollout_step == 0 or arm == "t0_single_frame":
                cache_record = cached_parent_record(
                    parent_cache, sequence, radar_index, point_count
                )
            else:
                cache_record = write_prediction(
                    prediction_path(output, arm, sequence, radar_index),
                    state,
                    arm,
                    sequence,
                    radar_index,
                    source_commit,
                )
            frame = method_frame(
                arm,
                state,
                previous_outputs.get(arm),
                pair,
                item,
                cube,
                target,
                target_indices,
                axes_tensors,
                static_hypothesis,
                dynamic_threshold_mps,
                cache_record,
                window["window_id"],
                rollout_step,
                aggregation if arm == "t3_doppdrive" else None,
            )
            arm_frames[arm].append(frame)
            previous_outputs[arm] = state
            torch.cuda.empty_cache()
        del item, cube, target, target_indices, states, t0, t3
        torch.cuda.empty_cache()
    return {
        "window_id": window["window_id"],
        "sequence": int(window["sequence"]),
        "frame_count": len(window["dataset_indices"]),
        "arms": arm_frames,
    }


def rollout_aggregates(frames: list[dict]) -> dict[str, dict]:
    result = {}
    for horizon in ROLLOUT_HORIZONS:
        selected = [frame for frame in frames if frame["rollout_step"] == horizon]
        if not selected:
            raise ValueError(f"Missing G4 rollout horizon {horizon}")
        result[str(horizon)] = aggregate_method_frames(selected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--dense-cache-report", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-prediction-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--dynamic-threshold-mps", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Formal G4 baseline evaluation requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.history_frames != 4:
        raise ValueError("Formal T3 requires exactly four historical frames")

    temporal_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if temporal_manifest.get("gate_pass") is not True:
        raise ValueError("G4 baseline manifest did not pass its data gate")
    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    dense_report_hash = sha256(args.dense_cache_report)
    dense_report = json.loads(args.dense_cache_report.read_text(encoding="utf-8"))
    if (
        dense_report.get("completed") is not True
        or dense_report["configuration"]["source_manifest_sha256"] != manifest_hash
    ):
        raise ValueError("G4 dense target cache is incomplete or mismatched")
    parent_document = json.loads(
        (args.parent_run / "config.json").read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    parent_provenance = parent_document["provenance"]
    if parent_config.get("variant") not in ("none", "full"):
        raise ValueError("G4 parent must be matched C0 or C3")
    if (
        parent_provenance["scene_split_sha256"] != scene_split_hash
        or parent_provenance["normalization_sha256"] != normalization_hash
    ):
        raise ValueError("G4 baseline data differs from the single-frame parent")
    parent_checkpoint = (args.parent_run / "best.pt").resolve()
    parent_checkpoint_hash = sha256(parent_checkpoint)
    parent_cache = FrozenPredictionCache(
        args.parent_prediction_cache, expected_frames=len(temporal_manifest["frames"])
    )
    expected_cache = {
        "source_manifest_sha256": manifest_hash,
        "dense_cache_report_sha256": dense_report_hash,
        "parent_checkpoint_sha256": parent_checkpoint_hash,
    }
    for key, expected in expected_cache.items():
        if parent_cache.configuration.get(key) != expected:
            raise ValueError(f"G4 parent prediction cache differs at {key}")
    point_count = int(parent_config["point_count"])
    if int(parent_cache.configuration["point_count"]) != point_count:
        raise ValueError("G4 parent prediction point counts differ")

    nonempty = args.output.exists() and any(args.output.iterdir())
    if nonempty and args.overwrite:
        shutil.rmtree(args.output)
        nonempty = False
    if nonempty and not args.resume:
        raise FileExistsError(f"G4 baseline output is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No G4 baseline evaluation to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    axes = load_axes(args.data_root / "resources")
    axes_tensors = tuple(
        torch.from_numpy(value).float().to(device)
        for value in (
            axes.doppler_mps,
            axes.range_m,
            axes.azimuth_rad,
            axes.elevation_rad,
        )
    )
    dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    if len(dataset.windows) != 8 or len(dataset.frame_dataset) != 384:
        raise ValueError("Formal G4 baseline requires 8 validation windows / 384 frames")
    if len(dataset.pairs) != 376:
        raise ValueError("Formal G4 baseline requires 376 validation pairs")
    pairs_by_current = {
        int(pair["current_dataset_index"]): pair for pair in dataset.pairs
    }
    if len(pairs_by_current) != len(dataset.pairs):
        raise ValueError("Duplicate current frames in G4 validation pairs")

    configuration = {
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": normalization_hash,
        "dense_cache_report_sha256": dense_report_hash,
        "parent_run": str(args.parent_run.resolve()),
        "parent_checkpoint_sha256": parent_checkpoint_hash,
        "parent_prediction_manifest_sha256": sha256(parent_cache.manifest_path),
        "parent_variant": parent_config["variant"],
        "parent_seed": int(parent_config["seed"]),
        "point_count": point_count,
        "history_frames": args.history_frames,
        "dynamic_threshold_mps": args.dynamic_threshold_mps,
        "static_hypothesis": parent_config["static_hypothesis"],
        "validation_window_count": len(dataset.windows),
        "validation_frame_count": len(dataset.frame_dataset),
        "validation_pair_count": len(dataset.pairs),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    progress_path = args.output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["configuration"] != configuration:
            raise ValueError("G4 baseline resume configuration differs")
    else:
        progress = {
            "schema_version": 1,
            "configuration": configuration,
            "windows": {},
            "completed": False,
        }
        atomic_json(progress_path, progress)

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
            window,
            dataset,
            pairs_by_current,
            parent_cache,
            args.output,
            point_count,
            axes_tensors,
            parent_config["static_hypothesis"],
            args.dynamic_threshold_mps,
            args.history_frames,
            args.source_commit,
            device,
        )
        progress["windows"][window_id] = result
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {"window": window_id, "sequence": result["sequence"], "status": "complete"}
            ),
            flush=True,
        )

    arm_reports = {}
    for arm, arm_id in ARMS.items():
        frames = [
            frame
            for window_id in sorted(progress["windows"])
            for frame in progress["windows"][window_id]["arms"][arm]
        ]
        arm_reports[arm] = {
            "arm_id": arm_id,
            "aggregate": aggregate_method_frames(frames),
            "rollout": rollout_aggregates(frames),
            "frames": frames,
        }
    checks = {
        "all_validation_windows_complete": len(progress["windows"])
        == len(dataset.windows),
        "all_arms_have_every_validation_frame": all(
            len(report["frames"]) == 384 for report in arm_reports.values()
        ),
        "all_arms_have_every_temporal_pair": all(
            report["aggregate"]["temporal_pair_count"] == 376
            for report in arm_reports.values()
        ),
        "all_outputs_have_exact_point_count": all(
            frame["prediction"]["point_count"] == point_count
            for report in arm_reports.values()
            for frame in report["frames"]
        ),
        "all_rollout_horizons_present": all(
            set(report["rollout"]) == {str(value) for value in ROLLOUT_HORIZONS}
            for report in arm_reports.values()
        ),
        "validation_partition_only": all(
            record["partition"] == "validation"
            for record in dataset.frame_dataset.records
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "arms": arm_reports,
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
