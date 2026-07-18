#!/usr/bin/env python3
"""Evaluate object-centric radial velocity on the frozen P5 test cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_calibration  # noqa: E402


PROTOCOL = "p5_object_radial_velocity_test_v1"
BASELINE_PROTOCOL = "p5_temporal_baselines_test_v1"
TEMPORAL_PROTOCOL = "p5_temporal_strict_rollout_test_v1"
FORMAL_SEEDS = (20260716, 20260717, 20260718)
DISTANCE_BINS = ("0-30", "30-60", "60-120", ">=120")
SPEED_BINS = ("0-0.5", "0.5-2", ">=2")
MOTION_STATES = ("static_like", "dynamic")


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


def wrap_scalar(value: float, lower: float, period: float) -> float:
    return float(np.remainder(value - lower, period) + lower)


def circular_error(observed: float, target: float, period: float) -> float:
    return float(np.remainder(observed - target + period / 2.0, period) - period / 2.0)


def parse_boxes(path: Path, calibration_xyz_m: np.ndarray) -> list[dict]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        values = [value.strip() for value in line.split(",")]
        if len(values) < 11 or values[0] != "*":
            continue
        center = np.asarray(values[4:7], dtype=np.float64) + calibration_xyz_m
        boxes.append(
            {
                "object_index": values[1],
                "track_id": values[2],
                "class": values[3],
                "center_xyz_m": center,
                "yaw_rad": math.radians(float(values[7])),
                "half_size_xyz_m": np.asarray(values[8:11], dtype=np.float64),
            }
        )
    return boxes


def unique_tracks(boxes: list[dict]) -> tuple[dict[str, dict], set[str]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for box in boxes:
        grouped[box["track_id"]].append(box)
    duplicates = {track_id for track_id, values in grouped.items() if len(values) != 1}
    return (
        {
            track_id: values[0]
            for track_id, values in grouped.items()
            if track_id not in duplicates
        },
        duplicates,
    )


def points_in_box(points_xyz_m: np.ndarray, box: dict) -> np.ndarray:
    centered = points_xyz_m - box["center_xyz_m"]
    cosine = math.cos(box["yaw_rad"])
    sine = math.sin(box["yaw_rad"])
    local = np.empty_like(centered)
    local[:, 0] = cosine * centered[:, 0] + sine * centered[:, 1]
    local[:, 1] = -sine * centered[:, 0] + cosine * centered[:, 1]
    local[:, 2] = centered[:, 2]
    return np.all(np.abs(local) <= box["half_size_xyz_m"], axis=1)


def circular_object_estimate(
    probability: np.ndarray,
    confidence: np.ndarray,
    doppler_mps: np.ndarray,
    lower: float,
    period: float,
) -> tuple[float, float]:
    if probability.ndim != 2 or probability.shape[1] != doppler_mps.size:
        raise ValueError("Prediction Doppler distribution has the wrong shape")
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if probability.shape[0] != confidence.size or confidence.size == 0:
        raise ValueError("Prediction confidence and Doppler rows differ")
    probability = np.asarray(probability, dtype=np.float64)
    probability = probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    weights = np.clip(confidence, 0.0, None)
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights)
    angle = 2.0 * np.pi * (doppler_mps - lower) / period
    unit = np.exp(1j * angle)
    vector = np.sum(weights * (probability @ unit)) / np.sum(weights)
    phase = float(np.mod(np.angle(vector), 2.0 * np.pi))
    estimate = wrap_scalar(lower + period * phase / (2.0 * np.pi), lower, period)
    return estimate, float(np.abs(vector))


def object_geometry(
    prediction_xyz_m: np.ndarray,
    target_xyz_confidence: np.ndarray,
) -> dict[str, float] | None:
    if prediction_xyz_m.size == 0 or target_xyz_confidence.size == 0:
        return None
    target_xyz_m = target_xyz_confidence[:, :3]
    target_weight = np.clip(target_xyz_confidence[:, 3], 0.0, None)
    if float(target_weight.sum()) <= 0.0:
        target_weight = np.ones_like(target_weight)
    prediction_to_target = cKDTree(target_xyz_m).query(prediction_xyz_m, workers=1)[0]
    target_to_prediction = cKDTree(prediction_xyz_m).query(target_xyz_m, workers=1)[0]
    completeness = float(np.average(target_to_prediction, weights=target_weight))
    precision = float(np.mean(prediction_to_target <= 1.0))
    recall = float(np.average(target_to_prediction <= 1.0, weights=target_weight))
    fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "chamfer_m": float(prediction_to_target.mean() + completeness),
        "precision_mean_distance_m": float(prediction_to_target.mean()),
        "completeness_mean_distance_m": completeness,
        "fscore_1m": fscore,
    }


def distance_bin(value: float) -> str:
    if value < 30.0:
        return "0-30"
    if value < 60.0:
        return "30-60"
    if value < 120.0:
        return "60-120"
    return ">=120"


def speed_bin(value: float) -> str:
    value = abs(value)
    if value < 0.5:
        return "0-0.5"
    if value < 2.0:
        return "0.5-2"
    return ">=2"


def summarize(observations: list[dict]) -> dict:
    supported = [item for item in observations if item["prediction_mps"] is not None]
    geometry_supported = [
        item for item in observations if item["object_geometry"] is not None
    ]
    absolute = np.asarray([item["absolute_error_mps"] for item in supported])
    resultant = np.asarray([item["resultant_strength"] for item in supported])
    report = {
        "box_count": len(observations),
        "scene_count": len({item["sequence"] for item in observations}),
        "frame_count": len(
            {(item["sequence"], item["radar_index"]) for item in observations}
        ),
        "supported_box_count": len(supported),
        "unsupported_box_count": len(observations) - len(supported),
        "support_rate_at_least_1": float(
            np.mean([item["point_count"] >= 1 for item in observations])
        )
        if observations
        else None,
        "support_rate_at_least_5": float(
            np.mean([item["point_count"] >= 5 for item in observations])
        )
        if observations
        else None,
        "support_rate_at_least_10": float(
            np.mean([item["point_count"] >= 10 for item in observations])
        )
        if observations
        else None,
        "mae_mps": None,
        "rmse_mps": None,
        "median_absolute_error_mps": None,
        "p90_absolute_error_mps": None,
        "within_0p5_mps_fraction_supported": None,
        "within_1p0_mps_fraction_supported": None,
        "mean_resultant_strength": None,
        "geometry_supported_box_count": len(geometry_supported),
        "object_chamfer_mean_m": None,
        "object_completeness_mean_m": None,
        "object_fscore_1m_mean": None,
    }
    if supported:
        report.update(
            {
                "mae_mps": float(absolute.mean()),
                "rmse_mps": float(np.sqrt(np.mean(np.square(absolute)))),
                "median_absolute_error_mps": float(np.median(absolute)),
                "p90_absolute_error_mps": float(np.quantile(absolute, 0.9)),
                "within_0p5_mps_fraction_supported": float(np.mean(absolute <= 0.5)),
                "within_1p0_mps_fraction_supported": float(np.mean(absolute <= 1.0)),
                "mean_resultant_strength": float(resultant.mean()),
            }
        )
    if geometry_supported:
        report.update(
            {
                "object_chamfer_mean_m": float(
                    np.mean(
                        [item["object_geometry"]["chamfer_m"] for item in geometry_supported]
                    )
                ),
                "object_completeness_mean_m": float(
                    np.mean(
                        [
                            item["object_geometry"]["completeness_mean_distance_m"]
                            for item in geometry_supported
                        ]
                    )
                ),
                "object_fscore_1m_mean": float(
                    np.mean(
                        [item["object_geometry"]["fscore_1m"] for item in geometry_supported]
                    )
                ),
            }
        )
    return report


def slice_reports(observations: list[dict], category_universe: dict[str, list]) -> dict:
    result = {}
    for dimension, categories in category_universe.items():
        result[dimension] = {}
        for category in categories:
            category_key = str(category)
            by_method = {}
            for method in sorted({item["method"] for item in observations}):
                selected = [
                    item
                    for item in observations
                    if item["method"] == method and item[dimension] == category
                ]
                by_method[method] = summarize(selected)
            result[dimension][category_key] = by_method
    return result


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_scene_bootstrap(
    observations: list[dict],
    reference: str,
    candidate: str,
    samples: int,
    random_seed: int,
) -> dict:
    methods = defaultdict(dict)
    for item in observations:
        key = (
            item["seed"],
            item["sequence"],
            item["radar_index"],
            item["track_id"],
        )
        methods[item["method"]][key] = item
    shared_keys = sorted(set(methods[reference]) & set(methods[candidate]))
    by_scene_seed: dict[tuple[int, int], dict] = {}
    for sequence in sorted({key[1] for key in shared_keys}):
        for seed in FORMAL_SEEDS:
            keys = [key for key in shared_keys if key[0] == seed and key[1] == sequence]
            common_supported = [
                key
                for key in keys
                if methods[reference][key]["prediction_mps"] is not None
                and methods[candidate][key]["prediction_mps"] is not None
            ]
            if not keys or not common_supported:
                continue
            by_scene_seed[(sequence, seed)] = {
                "mae_difference_mps": float(
                    np.mean(
                        [
                            methods[candidate][key]["absolute_error_mps"]
                            - methods[reference][key]["absolute_error_mps"]
                            for key in common_supported
                        ]
                    )
                ),
                "support10_difference": float(
                    np.mean(
                        [
                            (methods[candidate][key]["point_count"] >= 10)
                            - (methods[reference][key]["point_count"] >= 10)
                            for key in keys
                        ]
                    )
                ),
                "box_count": len(keys),
                "common_supported_box_count": len(common_supported),
            }
    scenes = sorted({key[0] for key in by_scene_seed})
    if not scenes:
        raise ValueError(f"No paired scene/seed observations for {candidate} vs {reference}")
    rng = np.random.default_rng(random_seed)
    bootstrap_mae = np.empty(samples, dtype=np.float64)
    bootstrap_support = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        mae_values = []
        support_values = []
        for sequence in sampled_scenes:
            available = [
                seed for seed in FORMAL_SEEDS if (int(sequence), seed) in by_scene_seed
            ]
            sampled_seeds = rng.choice(available, size=len(available), replace=True)
            mae_values.append(
                np.mean(
                    [
                        by_scene_seed[(int(sequence), int(seed))]["mae_difference_mps"]
                        for seed in sampled_seeds
                    ]
                )
            )
            support_values.append(
                np.mean(
                    [
                        by_scene_seed[(int(sequence), int(seed))]["support10_difference"]
                        for seed in sampled_seeds
                    ]
                )
            )
        bootstrap_mae[index] = float(np.mean(mae_values))
        bootstrap_support[index] = float(np.mean(support_values))
    paired_values = list(by_scene_seed.values())
    return {
        "reference": reference,
        "candidate": candidate,
        "sampling_unit": "scene first, then seed within scene",
        "bootstrap_samples": samples,
        "random_seed": random_seed,
        "scene_count": len(scenes),
        "scene_seed_count": len(by_scene_seed),
        "common_supported_box_count": sum(
            value["common_supported_box_count"] for value in paired_values
        ),
        "mae_difference_mps_candidate_minus_reference": {
            "estimate": float(np.mean([value["mae_difference_mps"] for value in paired_values])),
            "confidence_interval_95": percentile_interval(bootstrap_mae),
            "probability_candidate_better": float(np.mean(bootstrap_mae < 0.0)),
        },
        "support10_difference_candidate_minus_reference": {
            "estimate": float(np.mean([value["support10_difference"] for value in paired_values])),
            "confidence_interval_95": percentile_interval(bootstrap_support),
            "probability_candidate_better": float(np.mean(bootstrap_support > 0.0)),
        },
        "scene_seed_values": {
            f"seq{sequence:02d}_seed{seed}": value
            for (sequence, seed), value in sorted(by_scene_seed.items())
        },
    }


def load_report(path: Path, protocol: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != protocol or report.get("completed") is not True:
        raise ValueError(f"Incomplete or wrong protocol report: {path}")
    if report.get("configuration", {}).get("partition") != "test":
        raise ValueError(f"Report is not test-only: {path}")
    return report


def frame_map(frames: list[dict]) -> dict[tuple[int, int], dict]:
    mapped = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame for frame in frames
    }
    if len(mapped) != len(frames):
        raise ValueError("Duplicate frame keys in method report")
    return mapped


def load_prediction(
    record: dict,
    doppler_mps: np.ndarray,
    verified: dict[Path, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(record["path"])
    expected = record["sha256"]
    if path not in verified:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Prediction hash mismatch: {path}")
        verified[path] = actual
    elif verified[path] != expected:
        raise ValueError(f"Prediction report hashes disagree: {path}")
    with np.load(path) as cache:
        xyz = cache["xyz_m"].astype(np.float64)
        probability = cache["doppler_probability"].astype(np.float64)
        confidence = cache["confidence"].astype(np.float64).reshape(-1)
    if (
        xyz.shape != (int(record["point_count"]), 3)
        or probability.shape != (xyz.shape[0], doppler_mps.size)
        or confidence.shape != (xyz.shape[0],)
    ):
        raise ValueError(f"Malformed prediction arrays: {path}")
    return xyz, probability, confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--static-doppler-audit", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, nargs=3, required=True)
    parser.add_argument("--temporal-report", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.bootstrap_samples != 10_000 or args.bootstrap_seed != 20260718:
        raise ValueError("Formal P5 bootstrap is frozen at 10,000 samples / seed 20260718")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"P5 object velocity output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    static_audit = json.loads(args.static_doppler_audit.read_text(encoding="utf-8"))
    records = [record for record in manifest["frames"] if record["partition"] == "test"]
    if len(records) != 384 or len(manifest.get("windows", [])) != 8:
        raise ValueError("Formal P5 cohort requires 8 test windows / 384 frames")
    if len(records) != len(manifest["frames"]):
        raise ValueError("P5 manifest must contain only the test partition")
    if static_audit.get("passed") is not True:
        raise ValueError("Frozen static Doppler convention audit did not pass")
    hypothesis = static_audit["frozen_hypothesis"]
    if hypothesis not in ("positive_ego", "negative_ego"):
        raise ValueError(f"Unsupported frozen Doppler convention: {hypothesis}")

    axes = load_axes(args.data_root / "resources")
    doppler_mps = axes.doppler_mps.astype(np.float64)
    step = float(np.median(np.diff(doppler_mps)))
    lower = float(doppler_mps[0])
    period = step * doppler_mps.size
    if not np.isclose(period, static_audit["protocol"]["doppler_period_mps"], atol=1e-6):
        raise ValueError("Test Doppler axis differs from the frozen convention audit")

    baseline_reports = [
        load_report(path, BASELINE_PROTOCOL) for path in args.baseline_report
    ]
    temporal_reports = [
        load_report(path, TEMPORAL_PROTOCOL) for path in args.temporal_report
    ]
    baseline_by_seed = {
        int(report["configuration"]["parent_seed"]): report
        for report in baseline_reports
    }
    temporal_by_seed = {
        int(report["configuration"]["seed"]): report for report in temporal_reports
    }
    if tuple(sorted(baseline_by_seed)) != FORMAL_SEEDS:
        raise ValueError("P5 baseline reports do not cover the three formal seeds")
    if tuple(sorted(temporal_by_seed)) != FORMAL_SEEDS:
        raise ValueError("P5 temporal reports do not cover the three formal seeds")
    temporal_methods = {
        report["configuration"]["arm_id"] for report in temporal_reports
    }
    temporal_family_count = len(temporal_methods)
    if temporal_family_count != 1:
        raise ValueError("P5 temporal reports use different selected families")
    temporal_method = temporal_methods.pop()

    record_by_key = {
        (int(record["sequence"]), int(record["radar_index"])): record
        for record in records
    }
    by_window_frame = {
        (record["window_id"], int(record["frame_in_window"])): record
        for record in records
    }
    if len(record_by_key) != 384 or len(by_window_frame) != 384:
        raise ValueError("Duplicate P5 manifest frame identities")

    calibration_by_sequence = {
        sequence: load_calibration(
            args.data_root / str(sequence) / "info_calib" / "calib_radar_lidar.txt"
        ).translation_xyz_m.astype(np.float64)
        for sequence in sorted({int(record["sequence"]) for record in records})
    }
    boxes_by_key = {}
    duplicate_tracks = 0
    for key, record in record_by_key.items():
        boxes = parse_boxes(
            args.data_root / str(key[0]) / "info_label" / record["label"],
            calibration_by_sequence[key[0]],
        )
        unique, duplicates = unique_tracks(boxes)
        boxes_by_key[key] = unique
        duplicate_tracks += len(duplicates)

    targets_by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    unmatched_tracks = 0
    class_mismatches = 0
    for key, record in record_by_key.items():
        frame_in_window = int(record["frame_in_window"])
        if frame_in_window == 0:
            continue
        previous_record = by_window_frame[(record["window_id"], frame_in_window - 1)]
        previous_key = (
            int(previous_record["sequence"]),
            int(previous_record["radar_index"]),
        )
        delta_seconds = float(record["delta_seconds_from_previous"])
        if delta_seconds <= 0.0:
            raise ValueError(f"Non-positive P5 frame interval at {key}")
        for track_id, current_box in boxes_by_key[key].items():
            previous_box = boxes_by_key[previous_key].get(track_id)
            if previous_box is None:
                unmatched_tracks += 1
                continue
            if previous_box["class"] != current_box["class"]:
                class_mismatches += 1
                continue
            current_range = float(np.linalg.norm(current_box["center_xyz_m"]))
            previous_range = float(np.linalg.norm(previous_box["center_xyz_m"]))
            range_rate = (current_range - previous_range) / delta_seconds
            target_unwrapped = -range_rate if hypothesis == "positive_ego" else range_rate
            description = list(record["description"])
            if len(description) != 3:
                raise ValueError(f"Expected road/time/weather tags at {key}")
            targets_by_key[key].append(
                {
                    "sequence": key[0],
                    "radar_index": key[1],
                    "window_id": record["window_id"],
                    "frame_in_window": frame_in_window,
                    "track_id": track_id,
                    "class": current_box["class"],
                    "box": current_box,
                    "delta_seconds": delta_seconds,
                    "range_rate_mps": range_rate,
                    "target_unwrapped_mps": target_unwrapped,
                    "target_wrapped_mps": wrap_scalar(target_unwrapped, lower, period),
                    "distance_m": current_range,
                    "distance_bin": distance_bin(current_range),
                    "absolute_speed_bin": speed_bin(target_unwrapped),
                    "motion_state": "static_like"
                    if abs(target_unwrapped) < 0.5
                    else "dynamic",
                    "road_type": description[0],
                    "time_of_day": description[1],
                    "weather": description[2],
                }
            )

    observations = []
    verified_predictions: dict[Path, str] = {}
    expected_keys = set(record_by_key)
    baseline_report_hashes = {
        str(path): sha256(path) for path in args.baseline_report
    }
    temporal_report_hashes = {
        str(path): sha256(path) for path in args.temporal_report
    }
    target_cache_files = set()
    for seed in FORMAL_SEEDS:
        baseline = baseline_by_seed[seed]
        temporal = temporal_by_seed[seed]
        method_frames = {
            "T0": frame_map(baseline["arms"]["t0_single_frame"]["frames"]),
            "T3": frame_map(baseline["arms"]["t3_doppdrive"]["frames"]),
            temporal_method: frame_map(temporal["frames"]),
        }
        for method, frames in method_frames.items():
            if set(frames) != expected_keys:
                raise ValueError(f"{method} seed {seed} frame cohort differs from P5 manifest")
        for key in sorted(expected_keys):
            targets = targets_by_key.get(key, [])
            if not targets:
                continue
            target_cache_path = (
                args.cache_root / f"seq{key[0]:02d}_radar_{key[1]:05d}.npz"
            )
            with np.load(target_cache_path) as cache:
                target_cloud = cache["target_xyz_confidence"].astype(np.float64)
            target_cache_files.add(target_cache_path)
            target_subsets = {
                target["track_id"]: target_cloud[
                    points_in_box(target_cloud[:, :3], target["box"])
                ]
                for target in targets
            }
            for method, frames in method_frames.items():
                xyz, probability, confidence = load_prediction(
                    frames[key]["prediction"], doppler_mps, verified_predictions
                )
                for target in targets:
                    selected = points_in_box(xyz, target["box"])
                    count = int(selected.sum())
                    prediction = None
                    strength = None
                    error = None
                    if count:
                        prediction, strength = circular_object_estimate(
                            probability[selected],
                            confidence[selected],
                            doppler_mps,
                            lower,
                            period,
                        )
                        error = circular_error(
                            prediction, target["target_wrapped_mps"], period
                        )
                    geometry = object_geometry(
                        xyz[selected], target_subsets[target["track_id"]]
                    )
                    observations.append(
                        {
                            key_name: value
                            for key_name, value in target.items()
                            if key_name != "box"
                        }
                        | {
                            "seed": seed,
                            "method": method,
                            "point_count": count,
                            "prediction_mps": prediction,
                            "resultant_strength": strength,
                            "signed_error_mps": error,
                            "absolute_error_mps": None if error is None else abs(error),
                            "target_point_count": int(
                                target_subsets[target["track_id"]].shape[0]
                            ),
                            "object_geometry": geometry,
                        }
                    )

    methods = ("T0", "T3", temporal_method)
    by_method = {
        method: summarize([item for item in observations if item["method"] == method])
        for method in methods
    }
    classes = sorted(
        {
            box["class"]
            for boxes in boxes_by_key.values()
            for box in boxes.values()
        }
    )
    category_universe = {
        "road_type": sorted({record["description"][0] for record in records}),
        "time_of_day": sorted({record["description"][1] for record in records}),
        "weather": sorted({record["description"][2] for record in records}),
        "distance_bin": list(DISTANCE_BINS),
        "absolute_speed_bin": list(SPEED_BINS),
        "class": classes,
        "motion_state": list(MOTION_STATES),
        "sequence": sorted({int(record["sequence"]) for record in records}),
    }
    bootstrap = {
        "T3_vs_T0": paired_scene_bootstrap(
            observations, "T0", "T3", args.bootstrap_samples, args.bootstrap_seed
        ),
        f"{temporal_method}_vs_T0": paired_scene_bootstrap(
            observations,
            "T0",
            temporal_method,
            args.bootstrap_samples,
            args.bootstrap_seed,
        ),
        f"{temporal_method}_vs_T3": paired_scene_bootstrap(
            observations,
            "T3",
            temporal_method,
            args.bootstrap_samples,
            args.bootstrap_seed,
        ),
    }

    observation_path = args.output / "observations.jsonl"
    observation_temporary = observation_path.with_suffix(".jsonl.tmp")
    with observation_temporary.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation, sort_keys=True) + "\n")
    observation_temporary.replace(observation_path)
    expected_box_count_per_seed = sum(len(values) for values in targets_by_key.values())
    checks = {
        "test_manifest_gate_pass": manifest.get("gate_pass") is True,
        "test_partition_only": all(record["partition"] == "test" for record in records),
        "eight_windows_384_frames_376_pairs": len(records) == 384
        and len(manifest["windows"]) == 8
        and sum(max(int(window["frame_count"]) - 1, 0) for window in manifest["windows"])
        == 376,
        "three_formal_seeds": tuple(sorted(baseline_by_seed)) == FORMAL_SEEDS
        and tuple(sorted(temporal_by_seed)) == FORMAL_SEEDS,
        "one_frozen_temporal_family": temporal_family_count == 1,
        "matched_boxes_nonempty": expected_box_count_per_seed > 0,
        "all_methods_cover_every_matched_box": len(observations)
        == expected_box_count_per_seed * len(FORMAL_SEEDS) * len(methods),
        "prediction_hashes_verified": bool(verified_predictions),
        "all_matched_target_cache_files_present": len(target_cache_files)
        == sum(bool(values) for values in targets_by_key.values()),
        "scene_split_hash_matches_manifest": manifest.get("source_split_sha256")
        == sha256(args.scene_split),
        "manifest_sequences_equal_frozen_test_split": sorted(
            {int(record["sequence"]) for record in records}
        )
        == sorted(int(value) for value in scene_split["splits"]["test"]["sequences"]),
        "manifest_descriptions_match_frozen_split": all(
            list(record["description"])
            == scene_split["sequence_descriptions"][str(record["sequence"])]
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
            "manifest_sha256": sha256(args.manifest),
            "cache_root": str(args.cache_root.resolve()),
            "scene_split": str(args.scene_split.resolve()),
            "scene_split_sha256": sha256(args.scene_split),
            "static_doppler_audit": str(args.static_doppler_audit.resolve()),
            "static_doppler_audit_sha256": sha256(args.static_doppler_audit),
            "baseline_report_sha256": baseline_report_hashes,
            "temporal_report_sha256": temporal_report_hashes,
            "formal_seeds": list(FORMAL_SEEDS),
            "methods": list(methods),
            "box_margin_m": 0.0,
            "box_coordinate_transform": "official LiDAR box center plus frozen lidar-to-radar translation",
            "target_definition": "signed adjacent-frame box-center range rate mapped by frozen Doppler convention",
            "frozen_doppler_hypothesis": hypothesis,
            "doppler_lower_mps": lower,
            "doppler_period_mps": period,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "matching": {
            "matched_box_count_per_seed": expected_box_count_per_seed,
            "unmatched_current_track_count": unmatched_tracks,
            "class_mismatch_count": class_mismatches,
            "duplicate_track_ids_excluded": duplicate_tracks,
        },
        "aggregate": by_method,
        "slices": slice_reports(observations, category_universe),
        "paired_scene_bootstrap": bootstrap,
        "observations": {
            "path": str(observation_path),
            "sha256": sha256(observation_path),
            "count": len(observations),
        },
        "checks": checks,
        "completed": all(checks.values()),
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps({"checks": checks, "aggregate": by_method}, indent=2))
    if not report["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
