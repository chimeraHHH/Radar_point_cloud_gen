#!/usr/bin/env python3
"""Compare three-seed G1 Cube occupancy runs with paired scene bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ENDPOINTS = {
    "chamfer_m": "lower",
    "fscore_1p0m": "higher",
    "outlier_fraction_2m": "lower",
    "range_60_120m_completeness_mean_distance_m": "lower",
    "range_60_120m_fscore_1m": "higher",
}
DOPPLER_SENSITIVE_ENDPOINTS = (
    "chamfer_m",
    "range_60_120m_completeness_mean_distance_m",
    "range_60_120m_fscore_1m",
)
CONFIG_PAIR_EXCLUSIONS = {"mode", "seed"}


def load_run(path: Path, expected_mode: str) -> dict:
    config_document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config = config_document["config"]
    provenance = config_document["provenance"]
    if config["mode"] != expected_mode:
        raise ValueError(f"Expected {expected_mode}, found {config['mode']} in {path}")
    metrics = json.loads(
        (path / "best_validation_metrics.json").read_text(encoding="utf-8")
    )
    frames = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in metrics["validation"]["frames"]
    }
    return {
        "path": str(path),
        "seed": int(config["seed"]),
        "config": config,
        "provenance": provenance,
        "best_epoch": int(metrics["best_epoch"]),
        "frames": frames,
    }


def runs_by_seed(paths: list[Path], expected_mode: str) -> dict[int, dict]:
    runs = [load_run(path, expected_mode) for path in paths]
    indexed = {run["seed"]: run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError(f"Duplicate {expected_mode} seeds")
    return indexed


def validate_pairs(rae_max: dict[int, dict], full_raed: dict[int, dict]) -> None:
    if set(rae_max) != set(full_raed):
        raise ValueError("RAE-Max and Full-RAED seed sets differ")
    reference_frames = None
    reference_hashes = None
    reference_runtime = None
    reference_config = None
    parameter_counts: dict[str, set[int]] = defaultdict(set)
    for mode_runs in (rae_max, full_raed):
        for run in mode_runs.values():
            frame_keys = set(run["frames"])
            hashes = (
                run["provenance"]["manifest_sha256"],
                run["provenance"]["scene_split_sha256"],
                run["provenance"]["normalization_sha256"],
            )
            runtime = (
                run["provenance"]["git_commit"],
                run["provenance"]["torch_version"],
                run["provenance"]["device"],
            )
            if reference_frames is None:
                reference_frames = frame_keys
                reference_hashes = hashes
                reference_runtime = runtime
            if frame_keys != reference_frames:
                raise ValueError("Validation frame sets differ across G1 runs")
            if hashes != reference_hashes:
                raise ValueError("Manifest, split, or normalization hashes differ")
            if runtime != reference_runtime:
                raise ValueError("Source commit or runtime differs across G1 runs")
            paired_config = {
                key: value
                for key, value in run["config"].items()
                if key not in CONFIG_PAIR_EXCLUSIONS
            }
            if reference_config is None:
                reference_config = paired_config
            if paired_config != reference_config:
                raise ValueError("G1 training configurations differ beyond mode and seed")
            if "model_parameter_count" not in run["provenance"]:
                raise ValueError("G1 run provenance lacks model parameter count")
            parameter_counts[run["config"]["mode"]].add(
                int(run["provenance"]["model_parameter_count"])
            )
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise ValueError("Model parameter counts differ across seeds")
    rae_parameters = next(iter(parameter_counts["rae_max"]))
    full_parameters = next(iter(parameter_counts["full_raed"]))
    relative_increase = (full_parameters - rae_parameters) / rae_parameters
    if relative_increase > 0.01:
        raise ValueError(
            f"Full-RAED parameter increase {relative_increase:.4%} exceeds 1%"
        )


def parameter_parity(rae_max: dict[int, dict], full_raed: dict[int, dict]) -> dict:
    rae_parameters = int(
        next(iter(rae_max.values()))["provenance"]["model_parameter_count"]
    )
    full_parameters = int(
        next(iter(full_raed.values()))["provenance"]["model_parameter_count"]
    )
    return {
        "rae_max": rae_parameters,
        "full_raed": full_parameters,
        "relative_increase": (full_parameters - rae_parameters) / rae_parameters,
        "maximum_relative_increase": 0.01,
        "passed": (full_parameters - rae_parameters) / rae_parameters <= 0.01,
    }


def paired_groups(
    first_runs: dict[int, dict],
    second_runs: dict[int, dict] | None,
    endpoint: str,
    first_source: str,
    second_source: str,
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    grouped: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed, first_run in first_runs.items():
        second_run = first_run if second_runs is None else second_runs[seed]
        common_frames = sorted(set(first_run["frames"]) & set(second_run["frames"]))
        for frame_key in common_frames:
            first_frame = first_run["frames"][frame_key]
            second_frame = second_run["frames"][frame_key]
            first_metrics = first_frame[first_source]
            second_metrics = second_frame[second_source]
            if endpoint not in first_metrics or endpoint not in second_metrics:
                continue
            grouped[seed][frame_key[0]].append(
                (float(first_metrics[endpoint]), float(second_metrics[endpoint]))
            )
    return grouped


def summarize_groups(
    grouped: dict[int, dict[int, list[tuple[float, float]]]],
    direction: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    seeds = sorted(grouped)
    scenes = sorted({scene for seed in seeds for scene in grouped[seed]})
    if not seeds or not scenes:
        raise ValueError("No paired observations are available for an endpoint")

    def sample_summary(sampled_seeds, sampled_scenes) -> tuple[float, float, float]:
        first_values = []
        second_values = []
        improvements = []
        for seed in sampled_seeds:
            for scene in sampled_scenes:
                pairs = grouped[int(seed)].get(int(scene))
                if not pairs:
                    continue
                values = np.asarray(pairs, dtype=np.float64)
                first = float(values[:, 0].mean())
                second = float(values[:, 1].mean())
                first_values.append(first)
                second_values.append(second)
                improvements.append(
                    first - second if direction == "lower" else second - first
                )
        if not improvements:
            return float("nan"), float("nan"), float("nan")
        first_mean = float(np.mean(first_values))
        second_mean = float(np.mean(second_values))
        relative_change = (
            (second_mean - first_mean) / max(abs(first_mean), 1e-12)
        )
        return float(np.mean(improvements)), relative_change, first_mean

    point, relative_change, first_mean = sample_summary(seeds, scenes)
    second_mean = first_mean * (1.0 + relative_change)
    bootstrap = []
    relative_bootstrap = []
    while len(bootstrap) < bootstrap_samples:
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        improvement, relative, _ = sample_summary(sampled_seeds, sampled_scenes)
        if np.isfinite(improvement):
            bootstrap.append(improvement)
            relative_bootstrap.append(relative)
    confidence = np.quantile(bootstrap, (0.025, 0.975))
    relative_confidence = np.quantile(relative_bootstrap, (0.025, 0.975))
    observation_count = sum(
        len(values)
        for scenes_by_seed in grouped.values()
        for values in scenes_by_seed.values()
    )
    return {
        "direction": direction,
        "first_mean": first_mean,
        "second_mean": second_mean,
        "improvement": point,
        "improvement_ci95": confidence.tolist(),
        "relative_change": relative_change,
        "relative_change_ci95": relative_confidence.tolist(),
        "seed_count": len(seeds),
        "scene_count": len(scenes),
        "paired_frame_seed_count": observation_count,
    }


def compare(
    first_runs: dict[int, dict],
    second_runs: dict[int, dict] | None,
    first_source: str,
    second_source: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    reports = {}
    for endpoint, direction in ENDPOINTS.items():
        groups = paired_groups(
            first_runs,
            second_runs,
            endpoint,
            first_source,
            second_source,
        )
        reports[endpoint] = summarize_groups(
            groups, direction, bootstrap_samples, rng
        )
    return reports


def confidently_better(report: dict) -> bool:
    return report["improvement_ci95"][0] > 0.0


def dense_beats_cfar(reports: dict) -> bool:
    return (
        confidently_better(reports["chamfer_m"])
        and confidently_better(reports["fscore_1p0m"])
        and reports["outlier_fraction_2m"]["second_mean"] <= 0.25
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rae-max-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--full-raed-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    rae_max = runs_by_seed(args.rae_max_runs, "rae_max")
    full_raed = runs_by_seed(args.full_raed_runs, "full_raed")
    if len(rae_max) != args.required_seeds or len(full_raed) != args.required_seeds:
        raise ValueError(
            f"G1 requires {args.required_seeds} seeds per mode, received "
            f"{len(rae_max)} and {len(full_raed)}"
        )
    validate_pairs(rae_max, full_raed)
    rng = np.random.default_rng(args.seed)
    e2_vs_e1 = compare(
        rae_max, full_raed, "generated", "generated", args.bootstrap_samples, rng
    )
    e1_vs_e0 = compare(
        rae_max, None, "cfar", "generated", args.bootstrap_samples, rng
    )
    e2_vs_e0 = compare(
        full_raed, None, "cfar", "generated", args.bootstrap_samples, rng
    )
    doppler_sensitive_pass = any(
        confidently_better(e2_vs_e1[endpoint])
        for endpoint in DOPPLER_SENSITIVE_ENDPOINTS
    )
    chamfer_nondegradation = (
        e2_vs_e1["chamfer_m"]["relative_change_ci95"][1] <= 0.02
    )
    e1_dense_pass = dense_beats_cfar(e1_vs_e0)
    e2_dense_pass = dense_beats_cfar(e2_vs_e0)
    decision = {
        "dense_beats_cfar": e1_dense_pass or e2_dense_pass,
        "rae_max_beats_cfar": e1_dense_pass,
        "full_raed_beats_cfar": e2_dense_pass,
        "full_raed_doppler_sensitive_gain": doppler_sensitive_pass,
        "full_raed_chamfer_nondegradation": chamfer_nondegradation,
    }
    decision["g1_passed"] = all(
        (
            decision["dense_beats_cfar"],
            decision["full_raed_doppler_sensitive_gain"],
            decision["full_raed_chamfer_nondegradation"],
        )
    )
    report = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "seeds": sorted(rae_max),
        "model_parameter_parity": parameter_parity(rae_max, full_raed),
        "gate_thresholds": {
            "maximum_dense_outlier_fraction_2m": 0.25,
            "maximum_full_raed_chamfer_relative_degradation": 0.02,
        },
        "e2_full_raed_vs_e1_rae_max": e2_vs_e1,
        "e1_rae_max_vs_e0_cfar": e1_vs_e0,
        "e2_full_raed_vs_e0_cfar": e2_vs_e0,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
