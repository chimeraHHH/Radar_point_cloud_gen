#!/usr/bin/env python3
"""Consolidate the frozen P5 test reports without test-time selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROTOCOL = "p5_final_test_reporting_v1"
BASELINE_PROTOCOL = "p5_temporal_baselines_test_v1"
TEMPORAL_PROTOCOL = "p5_temporal_strict_rollout_test_v1"
OBJECT_PROTOCOL = "p5_object_radial_velocity_test_v1"
EFFICIENCY_PROTOCOL = "p5_efficiency_benchmark_v1"
FORMAL_SEEDS = (20260716, 20260717, 20260718)
ROLLOUT_HORIZONS = (1, 5, 10, 25)
ENDPOINTS = {
    "geometry_chamfer_m": (("current", "generated_geometry", "chamfer_m"), "lower"),
    "geometry_completeness_m": (
        ("current", "generated_geometry", "completeness_mean_distance_m"),
        "lower",
    ),
    "geometry_fscore_1m": (
        ("current", "generated_geometry", "fscore_1p0m"),
        "higher",
    ),
    "doppler_spectrum_kl": (("current", "doppler", "spectrum_kl"), "lower"),
    "doppler_circular_w1_mps": (
        ("current", "doppler", "circular_w1_mps"),
        "lower",
    ),
    "doppler_scalar_mae_mps": (
        ("current", "doppler", "circular_scalar_mae_mps"),
        "lower",
    ),
    "doppler_soft_ece": (("current", "doppler", "soft_ece_10bin"), "lower"),
    "cd_doppler": (("current", "cd_doppler", "cd_doppler"), "lower"),
    "static_pce_median_mps": (
        ("current", "doppler", "static_pce_median_mps"),
        "lower",
    ),
    "dynamic_scalar_mae_mps": (
        ("current", "doppler", "dynamic_scalar_mae_mps"),
        "lower",
    ),
    "local_spectrum_kl": (("current", "cycle", "local_spectrum_kl"), "lower"),
    "confidence_mean": (("current", "cycle", "confidence_mean"), "higher"),
    "covered_cell_count": (("current", "cycle", "covered_cell_count"), "higher"),
    "temporal_radial_error_m": (
        ("temporal", "temporal_radial_error_mean_m"),
        "lower",
    ),
    "occupancy_flicker": (("temporal", "occupancy_flicker"), "lower"),
}
FAILURE_CATEGORIES = {
    "geometry_outlier": (("current", "generated_geometry", "chamfer_m"), "higher"),
    "doppler_mismatch": (("current", "cycle", "local_spectrum_kl"), "higher"),
    "static_pce_failure": (
        ("current", "doppler", "static_pce_median_mps"),
        "higher",
    ),
    "confidence_collapse": (("current", "cycle", "confidence_mean"), "lower"),
    "coverage_collapse": (("current", "cycle", "covered_cell_count"), "lower"),
    "long_rollout_drift": (
        ("temporal", "temporal_radial_error_mean_m"),
        "higher",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def nested_value(document: dict, path: tuple[str, ...]) -> float | None:
    current = document
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if not isinstance(current, (int, float)) or not np.isfinite(current):
        return None
    return float(current)


def frame_index(frames: list[dict]) -> dict[tuple[int, int], dict]:
    indexed = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame for frame in frames
    }
    if len(indexed) != len(frames):
        raise ValueError("Duplicate P5 frame identities")
    return indexed


def load_json_report(path: Path, protocol: str) -> tuple[Path, dict]:
    path = path.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != protocol or report.get("completed") is not True:
        raise ValueError(f"Incomplete or incompatible report: {path}")
    if report.get("checks") and not all(report["checks"].values()):
        raise ValueError(f"Report contains failed checks: {path}")
    return path, report


def load_baseline(path: Path) -> dict:
    path, report = load_json_report(path, BASELINE_PROTOCOL)
    config = report["configuration"]
    required = {"t0_single_frame", "t1_ego_copy", "t2_doppler_copy", "t3_doppdrive"}
    if set(report["arms"]) != required:
        raise ValueError(f"P5 baseline arm matrix is incomplete: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "seed": int(config["parent_seed"]),
        "config": config,
        "T0": frame_index(report["arms"]["t0_single_frame"]["frames"]),
        "T3": frame_index(report["arms"]["t3_doppdrive"]["frames"]),
    }


def load_temporal(path: Path) -> dict:
    path, report = load_json_report(path, TEMPORAL_PROTOCOL)
    if not report.get("training_checks") or not all(report["training_checks"].values()):
        raise ValueError(f"Temporal training checks failed: {path}")
    config = report["configuration"]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "seed": int(config["seed"]),
        "method": config["arm_id"],
        "config": config,
        "frames": frame_index(report["frames"]),
        "efficiency": report["efficiency"],
    }


def by_seed(paths: list[Path], loader, label: str) -> dict[int, dict]:
    runs = [loader(path) for path in paths]
    indexed = {run["seed"]: run for run in runs}
    if tuple(sorted(indexed)) != FORMAL_SEEDS or len(indexed) != len(runs):
        raise ValueError(f"P5 {label} reports do not cover the three formal seeds")
    return indexed


def methods_by_seed(
    baselines: dict[int, dict], temporal: dict[int, dict]
) -> tuple[dict[int, dict[str, dict]], str]:
    temporal_methods = {run["method"] for run in temporal.values()}
    if len(temporal_methods) != 1:
        raise ValueError("P5 temporal reports use different selected families")
    selected = temporal_methods.pop()
    return (
        {
            seed: {
                "T0": baselines[seed]["T0"],
                "T3": baselines[seed]["T3"],
                selected: temporal[seed]["frames"],
            }
            for seed in FORMAL_SEEDS
        },
        selected,
    )


def validate_matched(
    methods: dict[int, dict[str, dict]],
    baselines: dict[int, dict],
    temporal: dict[int, dict],
    selected: str,
    manifest: dict,
    manifest_hash: str,
) -> dict[str, bool]:
    expected = {
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in manifest["frames"]
    }
    checks = {
        "three_formal_seeds": tuple(sorted(methods)) == FORMAL_SEEDS,
        "all_methods_cover_identical_384_frames": len(expected) == 384,
        "test_partition_only": all(
            frame["partition"] == "test" for frame in manifest["frames"]
        ),
        "same_frozen_data_point_count_and_parent": True,
        "same_selected_temporal_family": True,
        "first_frame_is_matched_t0_anchor": True,
        "required_metrics_cover_every_test_sequence": True,
        "strict_recurrent_rollout": True,
    }
    sequences = sorted({key[0] for key in expected})
    for seed in FORMAL_SEEDS:
        baseline = baselines[seed]
        learned = temporal[seed]
        if any(set(frames) != expected for frames in methods[seed].values()):
            checks["all_methods_cover_identical_384_frames"] = False
        baseline_config = baseline["config"]
        learned_config = learned["config"]
        if (
            baseline_config.get("partition") != "test"
            or learned_config.get("partition") != "test"
            or baseline_config["manifest_sha256"] != manifest_hash
            or learned_config["manifest_sha256"] != manifest_hash
            or baseline_config["scene_split_sha256"]
            != learned_config["scene_split_sha256"]
            or baseline_config["normalization_sha256"]
            != learned_config["normalization_sha256"]
            or baseline_config["dense_cache_report_sha256"]
            != learned_config["dense_cache_report_sha256"]
            or baseline_config["parent_prediction_manifest_sha256"]
            != learned_config["parent_prediction_manifest_sha256"]
            or baseline_config["parent_variant"] != learned_config["parent_variant"]
            or int(baseline_config["point_count"]) != int(learned_config["point_count"])
        ):
            checks["same_frozen_data_point_count_and_parent"] = False
        if learned["method"] != selected:
            checks["same_selected_temporal_family"] = False
        if learned_config.get("strict_recurrent_rollout") is not True:
            checks["strict_recurrent_rollout"] = False
        for identity in expected:
            if (
                int(learned["frames"][identity]["rollout_step"]) == 0
                and learned["frames"][identity]["prediction"]["sha256"]
                != baseline["T0"][identity]["prediction"]["sha256"]
            ):
                checks["first_frame_is_matched_t0_anchor"] = False
        for method_frames in methods[seed].values():
            for path, _ in ENDPOINTS.values():
                for sequence in sequences:
                    if not any(
                        nested_value(frame, path) is not None
                        for (current_sequence, _), frame in method_frames.items()
                        if current_sequence == sequence
                    ):
                        checks["required_metrics_cover_every_test_sequence"] = False
    return checks


def paired_groups(
    methods: dict[int, dict[str, dict]],
    reference: str,
    candidate: str,
    path: tuple[str, ...],
    horizon: int | None = None,
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    grouped: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed in FORMAL_SEEDS:
        first = methods[seed][reference]
        second = methods[seed][candidate]
        for identity in sorted(set(first) & set(second)):
            if horizon is not None and int(second[identity]["rollout_step"]) != horizon:
                continue
            first_value = nested_value(first[identity], path)
            second_value = nested_value(second[identity], path)
            if first_value is not None and second_value is not None:
                grouped[seed][identity[0]].append((first_value, second_value))
    return grouped


def summarize_groups(
    grouped: dict[int, dict[int, list[tuple[float, float]]]],
    direction: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    seeds = sorted(grouped)
    scenes = sorted({scene for values in grouped.values() for scene in values})
    if tuple(seeds) != FORMAL_SEEDS or len(scenes) != 8:
        raise ValueError("P5 comparison lacks the complete seed x scene matrix")
    if any(not grouped[seed].get(scene) for seed in seeds for scene in scenes):
        raise ValueError("P5 comparison has a missing seed-scene endpoint")

    def statistic(sampled_scenes, sampled_seeds) -> tuple[float, float, float]:
        first_values = []
        second_values = []
        improvements = []
        for scene in sampled_scenes:
            for seed in sampled_seeds:
                values = np.asarray(grouped[int(seed)][int(scene)], dtype=np.float64)
                first = float(values[:, 0].mean())
                second = float(values[:, 1].mean())
                first_values.append(first)
                second_values.append(second)
                improvements.append(first - second if direction == "lower" else second - first)
        return (
            float(np.mean(first_values)),
            float(np.mean(second_values)),
            float(np.mean(improvements)),
        )

    first, second, improvement = statistic(scenes, seeds)
    bootstrap = np.empty((bootstrap_samples, 3), dtype=np.float64)
    for index in range(bootstrap_samples):
        bootstrap[index] = statistic(
            rng.choice(scenes, size=len(scenes), replace=True),
            rng.choice(seeds, size=len(seeds), replace=True),
        )
    return {
        "direction": direction,
        "reference_mean": first,
        "candidate_mean": second,
        "improvement": improvement,
        "improvement_ci95": np.quantile(bootstrap[:, 2], (0.025, 0.975)).tolist(),
        "probability_candidate_better": float(np.mean(bootstrap[:, 2] > 0.0)),
        "seed_count": len(seeds),
        "scene_count": len(scenes),
        "paired_frame_seed_count": int(
            sum(len(items) for by_scene in grouped.values() for items in by_scene.values())
        ),
    }


def compare_methods(
    methods: dict[int, dict[str, dict]],
    reference: str,
    candidate: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
    horizon: int | None = None,
) -> dict:
    return {
        endpoint: summarize_groups(
            paired_groups(methods, reference, candidate, path, horizon),
            direction,
            bootstrap_samples,
            rng,
        )
        for endpoint, (path, direction) in ENDPOINTS.items()
    }


def aggregate_values(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"mean": None, "std": None, "median": None, "sample_count": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "sample_count": int(array.size),
    }


def aggregate_frames(frames: list[dict]) -> dict:
    return {
        endpoint: aggregate_values(
            [value for frame in frames if (value := nested_value(frame, path)) is not None]
        )
        for endpoint, (path, _) in ENDPOINTS.items()
    }


def primary_slices(methods: dict[int, dict[str, dict]], manifest: dict) -> dict:
    record_by_key = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in manifest["frames"]
    }
    result = {}
    for dimension, description_index in (("road_type", 0), ("time_of_day", 1), ("weather", 2)):
        categories = sorted({frame["description"][description_index] for frame in manifest["frames"]})
        result[dimension] = {}
        for category in categories:
            result[dimension][category] = {}
            for method in next(iter(methods.values())):
                selected = [
                    frame
                    for seed in FORMAL_SEEDS
                    for identity, frame in methods[seed][method].items()
                    if record_by_key[identity]["description"][description_index] == category
                ]
                result[dimension][category][method] = {
                    "frame_seed_count": len(selected),
                    "scene_count": len(
                        {
                            identity[0]
                            for identity, record in record_by_key.items()
                            if record["description"][description_index] == category
                        }
                    ),
                    "metrics": aggregate_frames(selected),
                }
    range_paths = {
        "0-30": "range_0_30m",
        "30-60": "range_30_60m",
        "60-120": "range_60_120m",
    }
    result["distance"] = {}
    for label, prefix in range_paths.items():
        result["distance"][label] = {}
        for method in next(iter(methods.values())):
            frames = [
                frame for seed in FORMAL_SEEDS for frame in methods[seed][method].values()
            ]
            result["distance"][label][method] = {
                "precision_mean_distance_m": aggregate_values(
                    [
                        value
                        for frame in frames
                        if (
                            value := nested_value(
                                frame,
                                (
                                    "current",
                                    "generated_geometry",
                                    f"{prefix}_precision_mean_distance_m",
                                ),
                            )
                        )
                        is not None
                    ]
                ),
                "completeness_mean_distance_m": aggregate_values(
                    [
                        value
                        for frame in frames
                        if (
                            value := nested_value(
                                frame,
                                (
                                    "current",
                                    "generated_geometry",
                                    f"{prefix}_completeness_mean_distance_m",
                                ),
                            )
                        )
                        is not None
                    ]
                ),
                "fscore_1m": aggregate_values(
                    [
                        value
                        for frame in frames
                        if (
                            value := nested_value(
                                frame,
                                (
                                    "current",
                                    "generated_geometry",
                                    f"{prefix}_fscore_1m",
                                ),
                            )
                        )
                        is not None
                    ]
                ),
            }
    result["motion_state"] = {}
    for state in ("static", "dynamic"):
        result["motion_state"][state] = {}
        for method in next(iter(methods.values())):
            frames = [
                frame for seed in FORMAL_SEEDS for frame in methods[seed][method].values()
            ]
            result["motion_state"][state][method] = {
                key: aggregate_values(
                    [
                        value
                        for frame in frames
                        if (
                            value := nested_value(
                                frame,
                                ("current", "stratified_geometry", f"{state}_{key}"),
                            )
                        )
                        is not None
                    ]
                )
                for key in (
                    "target_completeness_mean_distance_m",
                    "chamfer_m",
                    "fscore_1p0m",
                )
            }
    return result


def failure_taxonomy(methods: dict[int, dict[str, dict]]) -> dict:
    result = {}
    for seed in FORMAL_SEEDS:
        result[str(seed)] = {}
        for method, frames in methods[seed].items():
            result[str(seed)][method] = {}
            for category, (path, direction) in FAILURE_CATEGORIES.items():
                candidates = []
                for frame in frames.values():
                    if category == "long_rollout_drift" and int(frame["rollout_step"]) < 10:
                        continue
                    value = nested_value(frame, path)
                    if value is None:
                        continue
                    candidates.append(
                        {
                            "sequence": int(frame["sequence"]),
                            "radar_index": int(frame["radar_index"]),
                            "window_id": frame["window_id"],
                            "rollout_step": int(frame["rollout_step"]),
                            "value": value,
                            "prediction": frame["prediction"],
                        }
                    )
                candidates.sort(key=lambda item: item["value"], reverse=direction == "higher")
                result[str(seed)][method][category] = candidates[:5]
    return result


def markdown(report: dict) -> str:
    return "\n".join(
        [
            "# P5 Frozen Test Summary",
            "",
            f"Temporal arm: **{report['selected_temporal_arm']}**",
            "",
            "This report is descriptive. Test output did not select a model, checkpoint, threshold, or gate.",
            "",
            f"Completed: **{report['completed']}**",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--baseline-reports", type=Path, nargs=3, required=True)
    parser.add_argument("--temporal-reports", type=Path, nargs=3, required=True)
    parser.add_argument("--object-velocity-report", type=Path, required=True)
    parser.add_argument("--efficiency-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.bootstrap_samples != 10_000 or args.bootstrap_seed != 20260718:
        raise ValueError("Formal P5 comparison is frozen at 10,000 bootstraps / seed 20260718")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"P5 final output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    manifest_hash = sha256(args.manifest)
    baselines = by_seed(args.baseline_reports, load_baseline, "baseline")
    temporal = by_seed(args.temporal_reports, load_temporal, "temporal")
    methods, selected = methods_by_seed(baselines, temporal)
    matched_checks = validate_matched(
        methods, baselines, temporal, selected, manifest, manifest_hash
    )
    _, object_report = load_json_report(args.object_velocity_report, OBJECT_PROTOCOL)
    _, efficiency_report = load_json_report(args.efficiency_report, EFFICIENCY_PROTOCOL)
    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = {
        "T3_vs_T0": compare_methods(
            methods, "T0", "T3", args.bootstrap_samples, rng
        ),
        f"{selected}_vs_T0": compare_methods(
            methods, "T0", selected, args.bootstrap_samples, rng
        ),
        f"{selected}_vs_T3": compare_methods(
            methods, "T3", selected, args.bootstrap_samples, rng
        ),
        "rollout": {
            str(horizon): compare_methods(
                methods, "T0", selected, args.bootstrap_samples, rng, horizon=horizon
            )
            for horizon in ROLLOUT_HORIZONS
        },
    }
    slices = primary_slices(methods, manifest)
    failures = failure_taxonomy(methods)
    atomic_text(
        args.output / "generalization_slices.json", json.dumps(slices, indent=2) + "\n"
    )
    atomic_text(
        args.output / "failure_taxonomy.json", json.dumps(failures, indent=2) + "\n"
    )
    test_sequences = sorted({int(frame["sequence"]) for frame in manifest["frames"]})
    checks = matched_checks | {
        "manifest_gate_pass": manifest.get("gate_pass") is True,
        "manifest_is_exact_frozen_test_split": test_sequences
        == sorted(int(value) for value in scene_split["splits"]["test"]["sequences"]),
        "object_report_matches_manifest": object_report["configuration"]["manifest_sha256"]
        == manifest_hash,
        "object_report_matches_methods": set(object_report["configuration"]["methods"])
        == {"T0", "T3", selected},
        "efficiency_report_matches_manifest": efficiency_report["configuration"][
            "manifest_sha256"
        ]
        == manifest_hash,
        "test_results_not_used_for_selection": True,
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "source_commit": args.source_commit,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_hash,
            "scene_split_sha256": sha256(args.scene_split),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "selection_partition": "validation",
            "reporting_partition": "test",
            "test_derived_decision": False,
        },
        "selected_temporal_arm": selected,
        "baseline_reports": [
            {"seed": seed, "path": run["path"], "sha256": run["sha256"]}
            for seed, run in sorted(baselines.items())
        ],
        "temporal_reports": [
            {"seed": seed, "path": run["path"], "sha256": run["sha256"]}
            for seed, run in sorted(temporal.items())
        ],
        "aggregate": {
            method: aggregate_frames(
                [frame for seed in FORMAL_SEEDS for frame in methods[seed][method].values()]
            )
            for method in ("T0", "T3", selected)
        },
        "comparisons": comparisons,
        "object_velocity_report": {
            "path": str(args.object_velocity_report.resolve()),
            "sha256": sha256(args.object_velocity_report),
            "aggregate": object_report["aggregate"],
        },
        "efficiency_report": {
            "path": str(args.efficiency_report.resolve()),
            "sha256": sha256(args.efficiency_report),
            "aggregate": efficiency_report["aggregate"],
        },
        "generalization_slices": {
            "path": str((args.output / "generalization_slices.json").resolve()),
            "sha256": sha256(args.output / "generalization_slices.json"),
        },
        "failure_taxonomy": {
            "path": str((args.output / "failure_taxonomy.json").resolve()),
            "sha256": sha256(args.output / "failure_taxonomy.json"),
        },
        "checks": checks,
        "completed": all(checks.values()),
    }
    atomic_text(args.output / "report.json", json.dumps(report, indent=2) + "\n")
    atomic_text(args.output / "summary.md", markdown(report))
    print(json.dumps({"checks": checks, "completed": report["completed"]}, indent=2))
    if not report["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
