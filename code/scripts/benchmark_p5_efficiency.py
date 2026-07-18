#!/usr/bin/env python3
"""Matched CUDA efficiency benchmark for the frozen P5 test methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.parent_prediction import FrozenPredictionCache, PointPrediction  # noqa: E402
from cube_dense.temporal_dataset import KRadarTemporalDataset  # noqa: E402
from eval.dense_geometry import occupancy_to_points  # noqa: E402
from models.cube_cycle import CubeCycleNet  # noqa: E402
from models.cube_doppler import split_query_indices  # noqa: E402
from models.cube_temporal import CubeTemporalNet  # noqa: E402
from models.temporal_baselines import doppdrive_aggregate, warp_prediction  # noqa: E402
from models.temporal_inference import predict_temporal_pair  # noqa: E402


PROTOCOL = "p5_efficiency_benchmark_v1"
BASELINE_PROTOCOL = "p5_temporal_baselines_test_v1"
TEMPORAL_PROTOCOL = "p5_temporal_strict_rollout_test_v1"
FORMAL_SEEDS = (20260716, 20260717, 20260718)
BENCHMARK_FRAME_IN_WINDOW = 4
WARMUP_ITERATIONS = 3


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


def load_report(path: Path, protocol: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != protocol or report.get("completed") is not True:
        raise ValueError(f"Incomplete or incompatible P5 report: {path}")
    if report["configuration"].get("partition") != "test":
        raise ValueError(f"Efficiency input is not test-only: {path}")
    return report


def frame_map(frames: list[dict]) -> dict[tuple[int, int], dict]:
    result = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame for frame in frames
    }
    if len(result) != len(frames):
        raise ValueError("Duplicate P5 efficiency frame identities")
    return result


def parent_cache_root(report: dict) -> Path:
    frame = report["arms"]["t0_single_frame"]["frames"][0]
    path = Path(frame["prediction"]["path"])
    if path.parent.name != "predictions":
        raise ValueError(f"Unexpected parent prediction cache path: {path}")
    return path.parent.parent


def point_prediction_from_npz(path: Path, device: torch.device) -> PointPrediction:
    with np.load(path) as cache:
        probability = torch.from_numpy(
            cache["doppler_probability"].astype(np.float32)
        ).to(device)
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return PointPrediction(
            xyz_m=torch.from_numpy(cache["xyz_m"].astype(np.float32)).to(device),
            coordinates_rae=torch.from_numpy(
                cache["coordinates_rae"].astype(np.float32)
            ).to(device),
            probability=probability,
            confidence=torch.from_numpy(cache["confidence"].astype(np.float32)).to(device),
            static_center_mps=torch.from_numpy(
                cache["static_center_mps"].astype(np.float32)
            ).to(device),
        )


def build_parent_model(
    run: Path, axes, device: torch.device
) -> tuple[CubeCycleNet, dict, str]:
    document = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config = document["config"]
    checkpoint_path = (run / "best.pt").resolve()
    model = CubeCycleNet(
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
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, sha256(checkpoint_path)


def build_temporal_model(
    run: Path, axes, device: torch.device
) -> tuple[CubeTemporalNet, dict, str]:
    document = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config = document["config"]
    checkpoint_path = (run / "best.pt").resolve()
    model = CubeTemporalNet(
        config["fusion_mode"],
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
    return model, config, sha256(checkpoint_path)


@torch.inference_mode()
def parent_inference(
    model: CubeCycleNet,
    item: dict,
    axes,
    point_count: int,
    device: torch.device,
) -> PointPrediction:
    cube = item["cube_drae"].unsqueeze(0).to(device)
    ego_speed = item["ego_speed_mps"].reshape(1).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occupancy_logits, features = model(cube)
    _, confidence, indices = occupancy_to_points(
        occupancy_logits[0].float(), axes, point_count=point_count
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.query_cycle(features, indices, ego_speed)
    if "static_center_mps" in prediction:
        static_center = prediction["static_center_mps"]
    else:
        batch, _, azimuth, elevation = split_query_indices(indices, 1)
        static_center = model.static_center(batch, azimuth, elevation, ego_speed)
    return PointPrediction(
        xyz_m=prediction["xyz_m"],
        coordinates_rae=prediction["coordinates_rae"],
        probability=prediction["probability"],
        confidence=confidence,
        static_center_mps=static_center,
    )


def prepare_history(
    window: dict,
    dataset: KRadarTemporalDataset,
    pairs_by_current: dict[int, dict],
    parent_cache: FrozenPredictionCache,
    axes_tensors: tuple[torch.Tensor, ...],
    config: dict,
    device: torch.device,
) -> list[PointPrediction]:
    doppler, ranges, azimuth, elevation = axes_tensors
    lower = doppler[0]
    period = torch.median(torch.diff(doppler)) * doppler.numel()
    history = []
    for rollout_step, dataset_index in enumerate(
        window["dataset_indices"][:BENCHMARK_FRAME_IN_WINDOW]
    ):
        record = dataset.frame_dataset.records[dataset_index]
        state = parent_cache.load(
            int(record["sequence"]), int(record["radar_index"]), device
        )
        if rollout_step == 0:
            history = [state]
            continue
        pair = pairs_by_current[dataset_index]
        item = dataset.frame_dataset[dataset_index]
        transform = torch.tensor(
            pair["current_from_previous"], dtype=torch.float32, device=device
        ).reshape(4, 4)
        warped = [
            warp_prediction(
                prior,
                transform,
                pair["delta_seconds"],
                doppler,
                lower,
                period,
                ranges,
                azimuth,
                elevation,
                item["ego_speed_mps"].to(device),
                config["static_hypothesis"],
                apply_doppler_displacement=True,
                dynamic_threshold_mps=1.0,
            )
            for prior in history
        ]
        history = ([state] + warped)[:4]
    return history


def synchronize_result(result) -> None:
    if isinstance(result, PointPrediction):
        result.xyz_m.sum().item()
    elif isinstance(result, dict):
        result["prediction"]["xyz_m"].sum().item()
    else:
        raise TypeError(type(result))


def warmup(function) -> None:
    for _ in range(WARMUP_ITERATIONS):
        result = function()
        synchronize_result(result)
        del result
    torch.cuda.synchronize()


def measure(function, device: torch.device, point_count: int) -> tuple[dict, object]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = function()
    end.record()
    torch.cuda.synchronize(device)
    latency_ms = float(start.elapsed_time(end))
    if isinstance(result, PointPrediction):
        output_point_count = int(result.xyz_m.shape[0])
    elif isinstance(result, dict):
        output_point_count = int(result["prediction"]["xyz_m"].shape[0])
    else:
        raise TypeError(type(result))
    record = {
        "latency_ms": latency_ms,
        "peak_memory_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
        "point_throughput_per_second": point_count / max(latency_ms / 1000.0, 1e-9),
        "output_point_count": output_point_count,
    }
    return record, result


def aggregate(records: list[dict]) -> dict:
    result = {}
    for method in sorted({record["method"] for record in records}):
        selected = [record for record in records if record["method"] == method]
        latency = np.asarray([record["latency_ms"] for record in selected])
        memory = np.asarray([record["peak_memory_mib"] for record in selected])
        throughput = np.asarray(
            [record["point_throughput_per_second"] for record in selected]
        )
        result[method] = {
            "measurement_count": len(selected),
            "latency_ms_mean": float(latency.mean()),
            "latency_ms_std": float(latency.std()),
            "latency_ms_median": float(np.median(latency)),
            "latency_ms_q90": float(np.quantile(latency, 0.9)),
            "peak_memory_mib_mean": float(memory.mean()),
            "peak_memory_mib_max": float(memory.max()),
            "point_throughput_per_second_mean": float(throughput.mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-reports", type=Path, nargs=3, required=True)
    parser.add_argument("--temporal-reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Formal P5 efficiency benchmark requires CUDA")
    if args.output.exists():
        raise FileExistsError(args.output)

    baseline_reports = [
        load_report(path, BASELINE_PROTOCOL) for path in args.baseline_reports
    ]
    temporal_reports = [
        load_report(path, TEMPORAL_PROTOCOL) for path in args.temporal_reports
    ]
    baseline_by_seed = {
        int(report["configuration"]["parent_seed"]): report
        for report in baseline_reports
    }
    temporal_by_seed = {
        int(report["configuration"]["seed"]): report for report in temporal_reports
    }
    if tuple(sorted(baseline_by_seed)) != FORMAL_SEEDS:
        raise ValueError("P5 efficiency baseline seeds differ")
    if tuple(sorted(temporal_by_seed)) != FORMAL_SEEDS:
        raise ValueError("P5 efficiency temporal seeds differ")
    selected_methods = {
        report["configuration"]["arm_id"] for report in temporal_reports
    }
    if len(selected_methods) != 1:
        raise ValueError("P5 efficiency temporal families differ")
    selected_method = selected_methods.pop()
    manifest_hash = sha256(args.manifest)
    if any(
        report["configuration"]["manifest_sha256"] != manifest_hash
        for report in baseline_reports + temporal_reports
    ):
        raise ValueError("P5 efficiency reports and manifest differ")

    device = torch.device(args.device)
    axes = load_axes(args.data_root / "resources")
    axes_tensors = tuple(
        torch.from_numpy(values).float().to(device)
        for values in (
            axes.doppler_mps,
            axes.range_m,
            axes.azimuth_rad,
            axes.elevation_rad,
        )
    )
    dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, ("test",)
    )
    if len(dataset.windows) != 8 or len(dataset.frame_dataset) != 384:
        raise ValueError("P5 efficiency benchmark requires 8 test windows / 384 frames")
    pairs_by_current = {
        int(pair["current_dataset_index"]): pair for pair in dataset.pairs
    }
    benchmark_windows = sorted(dataset.windows, key=lambda value: value["window_id"])
    records = []
    checkpoint_hashes = {}

    for seed in FORMAL_SEEDS:
        baseline = baseline_by_seed[seed]
        temporal = temporal_by_seed[seed]
        point_count = int(baseline["configuration"]["point_count"])
        if point_count != 10_000 or int(temporal["configuration"]["point_count"]) != 10_000:
            raise ValueError("Formal P5 efficiency requires exactly 10,000 points")
        parent_run = Path(baseline["configuration"]["parent_run"])
        parent_cache = FrozenPredictionCache(
            parent_cache_root(baseline), expected_frames=384
        )
        parent_model, parent_config, parent_hash = build_parent_model(
            parent_run, axes, device
        )
        if parent_hash != baseline["configuration"]["parent_checkpoint_sha256"]:
            raise ValueError("P5 parent checkpoint hash differs")
        checkpoint_hashes[f"seed{seed}_parent"] = parent_hash

        first_window = benchmark_windows[0]
        first_index = first_window["dataset_indices"][BENCHMARK_FRAME_IN_WINDOW]
        first_item = dataset.frame_dataset[first_index]
        warmup(
            lambda: parent_inference(
                parent_model, first_item, axes, point_count, device
            )
        )
        for window in benchmark_windows:
            dataset_index = window["dataset_indices"][BENCHMARK_FRAME_IN_WINDOW]
            item = dataset.frame_dataset[dataset_index]
            measurement, output = measure(
                lambda item=item: parent_inference(
                    parent_model, item, axes, point_count, device
                ),
                device,
                point_count,
            )
            records.append(
                measurement
                | {
                    "method": "T0",
                    "seed": seed,
                    "window_id": window["window_id"],
                    "sequence": int(window["sequence"]),
                    "frame_in_window": BENCHMARK_FRAME_IN_WINDOW,
                }
            )
            del output, item

        def t3_call(item, pair, history):
            doppler, ranges, azimuth, elevation = axes_tensors
            lower = doppler[0]
            period = torch.median(torch.diff(doppler)) * doppler.numel()
            transform = torch.tensor(
                pair["current_from_previous"], dtype=torch.float32, device=device
            ).reshape(4, 4)
            current = parent_inference(parent_model, item, axes, point_count, device)
            warped = [
                warp_prediction(
                    prior,
                    transform,
                    pair["delta_seconds"],
                    doppler,
                    lower,
                    period,
                    ranges,
                    azimuth,
                    elevation,
                    item["ego_speed_mps"].to(device),
                    parent_config["static_hypothesis"],
                    apply_doppler_displacement=True,
                    dynamic_threshold_mps=1.0,
                )
                for prior in history
            ]
            return doppdrive_aggregate(
                current,
                warped,
                point_count,
                item["ego_speed_mps"].to(device),
                parent_config["static_hypothesis"],
                lower,
                period,
            )[0]

        for window_index, window in enumerate(benchmark_windows):
            history = prepare_history(
                window,
                dataset,
                pairs_by_current,
                parent_cache,
                axes_tensors,
                parent_config,
                device,
            )
            dataset_index = window["dataset_indices"][BENCHMARK_FRAME_IN_WINDOW]
            item = dataset.frame_dataset[dataset_index]
            pair = pairs_by_current[dataset_index]
            if window_index == 0:
                warmup(lambda: t3_call(item, pair, history))
            measurement, output = measure(
                lambda item=item, pair=pair, history=history: t3_call(
                    item, pair, history
                ),
                device,
                point_count,
            )
            records.append(
                measurement
                | {
                    "method": "T3",
                    "seed": seed,
                    "window_id": window["window_id"],
                    "sequence": int(window["sequence"]),
                    "frame_in_window": BENCHMARK_FRAME_IN_WINDOW,
                }
            )
            del output, item, history
            torch.cuda.empty_cache()
        del parent_model, parent_cache
        torch.cuda.empty_cache()

        temporal_run = Path(temporal["configuration"]["temporal_run"])
        temporal_model, temporal_config, temporal_hash = build_temporal_model(
            temporal_run, axes, device
        )
        if temporal_hash != temporal["configuration"]["temporal_checkpoint_sha256"]:
            raise ValueError("P5 temporal checkpoint hash differs")
        checkpoint_hashes[f"seed{seed}_temporal"] = temporal_hash
        temporal_frames = frame_map(temporal["frames"])
        def temporal_call(item, pair, previous):
            return predict_temporal_pair(
                temporal_model,
                item,
                previous,
                pair,
                axes,
                point_count,
                float(temporal_config["dynamic_threshold_mps"]),
                device,
            )

        for window_index, window in enumerate(benchmark_windows):
            previous_index = window["dataset_indices"][BENCHMARK_FRAME_IN_WINDOW - 1]
            current_index = window["dataset_indices"][BENCHMARK_FRAME_IN_WINDOW]
            previous_record = dataset.frame_dataset.records[previous_index]
            previous_key = (
                int(previous_record["sequence"]),
                int(previous_record["radar_index"]),
            )
            previous = point_prediction_from_npz(
                Path(temporal_frames[previous_key]["prediction"]["path"]), device
            )
            item = dataset.frame_dataset[current_index]
            pair = pairs_by_current[current_index]
            if window_index == 0:
                warmup(lambda: temporal_call(item, pair, previous))
            measurement, output = measure(
                lambda item=item, pair=pair, previous=previous: temporal_call(
                    item, pair, previous
                ),
                device,
                point_count,
            )
            records.append(
                measurement
                | {
                    "method": selected_method,
                    "seed": seed,
                    "window_id": window["window_id"],
                    "sequence": int(window["sequence"]),
                    "frame_in_window": BENCHMARK_FRAME_IN_WINDOW,
                }
            )
            del output, previous, item
            torch.cuda.empty_cache()
        del temporal_model
        torch.cuda.empty_cache()

    expected_methods = {"T0", "T3", selected_method}
    checks = {
        "three_formal_seeds": {record["seed"] for record in records}
        == set(FORMAL_SEEDS),
        "all_methods_measured": {record["method"] for record in records}
        == expected_methods,
        "eight_scenes_per_seed_method": all(
            len(
                [
                    record
                    for record in records
                    if record["method"] == method and record["seed"] == seed
                ]
            )
            == 8
            for method in expected_methods
            for seed in FORMAL_SEEDS
        ),
        "exact_10000_point_outputs": all(
            record["output_point_count"] == 10_000 for record in records
        ),
        "three_warmup_iterations": WARMUP_ITERATIONS == 3,
        "finite_positive_measurements": all(
            np.isfinite(record["latency_ms"])
            and record["latency_ms"] > 0.0
            and np.isfinite(record["peak_memory_mib"])
            and record["peak_memory_mib"] > 0.0
            for record in records
        ),
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "source_commit": args.source_commit,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_hash,
            "partition": "test",
            "formal_seeds": list(FORMAL_SEEDS),
            "methods": sorted(expected_methods),
            "benchmark_frame_in_window": BENCHMARK_FRAME_IN_WINDOW,
            "scene_count": 8,
            "point_count": 10_000,
            "warmup_iterations": WARMUP_ITERATIONS,
            "timing": "CUDA events around end-to-end GPU method inference; dataset I/O excluded",
            "device": args.device,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "checkpoint_sha256": checkpoint_hashes,
        },
        "aggregate": aggregate(records),
        "records": records,
        "checks": checks,
        "completed": all(checks.values()),
    }
    atomic_json(args.output, report)
    print(json.dumps({"checks": checks, "aggregate": report["aggregate"]}, indent=2))
    if not report["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
