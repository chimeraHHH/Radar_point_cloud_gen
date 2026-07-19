#!/usr/bin/env python3
"""Compare C0-C3 with paired scene-first bootstrap and decide G3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANTS = ("none", "local_peak", "marginal", "full")
ENDPOINTS = {
    "local_spectrum_kl": ("cycle", "local_spectrum_kl", "lower"),
    "static_pce_median_mps": ("doppler", "static_pce_median_mps", "lower"),
    "geometry_chamfer_m": ("generated_geometry", "chamfer_m", "lower"),
    "geometry_fscore_1m": ("generated_geometry", "fscore_1p0m", "higher"),
    "cd_doppler": ("cd_doppler", "cd_doppler", "lower"),
    "confidence_mean": ("cycle", "confidence_mean", "higher"),
    "covered_cell_count": ("cycle", "covered_cell_count", "higher"),
    "confidence_ece": ("cycle", "existence_ece_10bin", "lower"),
}
CONFIG_PAIR_EXCLUSIONS = {"variant", "seed"}
ROBUSTNESS_PROTOCOL = "g3_cube_cycle_robustness_v1"
REQUIRED_ROBUSTNESS_CONDITIONS = {
    "clean",
    "log_power_noise_snr20db",
    "log_power_noise_snr10db",
    "log_power_noise_snr5db",
    "doppler_shift_m2",
    "doppler_shift_m1",
    "doppler_shift_p1",
    "doppler_shift_p2",
    "azimuth_offset_p0p25_bin",
    "azimuth_offset_p0p5_bin",
    "elevation_offset_p0p25_bin",
    "elevation_offset_p0p5_bin",
    "confidence_temperature_0p5",
    "confidence_temperature_1p0",
    "confidence_temperature_2p0",
}
REQUIRED_ROBUSTNESS_AGGREGATES = (
    ("cycle", "local_spectrum_kl"),
    ("doppler", "static_pce_median_mps"),
    ("generated_geometry", "chamfer_m"),
    ("cycle", "covered_cell_count"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path, expected_variant: str) -> dict:
    path = path.resolve()
    config_path = path / "config.json"
    checkpoint_path = path / "best.pt"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if config["variant"] != expected_variant:
        raise ValueError(
            f"Expected {expected_variant}, found {config['variant']} in {path}"
        )
    metrics = json.loads(
        (path / "best_validation_metrics.json").read_text(encoding="utf-8")
    )
    frames = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in metrics["validation"]["frames"]
    }
    return {
        "path": str(path),
        "config_sha256": sha256(config_path),
        "best_checkpoint_sha256": sha256(checkpoint_path),
        "seed": int(config["seed"]),
        "config": config,
        "provenance": provenance,
        "best_epoch": int(metrics["best_epoch"]),
        "frames": frames,
    }


def validate_robustness(
    robustness: dict,
    arms: dict[str, dict[int, dict]],
) -> dict:
    if robustness.get("protocol") != ROBUSTNESS_PROTOCOL:
        raise ValueError("G3 robustness report uses the wrong protocol")
    if robustness.get("schema_version") != 1:
        raise ValueError("Unsupported G3 robustness schema")
    if robustness.get("completed") is not True:
        raise ValueError("G3 robustness report is not complete")
    report_checks = robustness.get("checks")
    if not isinstance(report_checks, dict) or not report_checks or not all(
        value is True for value in report_checks.values()
    ):
        raise ValueError("G3 robustness report contains failed completion checks")
    definitions = robustness.get("condition_definitions", [])
    condition_ids = {definition.get("condition_id") for definition in definitions}
    if condition_ids != REQUIRED_ROBUSTNESS_CONDITIONS:
        missing = sorted(REQUIRED_ROBUSTNESS_CONDITIONS - condition_ids)
        extra = sorted(condition_ids - REQUIRED_ROBUSTNESS_CONDITIONS)
        raise ValueError(
            f"G3 robustness conditions differ: missing={missing}, extra={extra}"
        )

    expected_runs = {}
    for report_variant, arm_name in (("none", "c0_none"), ("full", "c3_full")):
        for seed, run in arms[arm_name].items():
            expected_runs[(report_variant, seed)] = run
    report_runs = {
        (run.get("variant"), int(run.get("seed"))): run
        for run in robustness.get("runs", [])
    }
    if set(report_runs) != set(expected_runs):
        raise ValueError("G3 robustness C0/C3 seed matrix differs from clean runs")

    reference_frames = set(next(iter(arms["c0_none"].values()))["frames"])
    if int(robustness.get("full_validation_frame_count", -1)) != len(reference_frames):
        raise ValueError("G3 robustness report does not cover the frozen validation set")
    for run_key, expected in expected_runs.items():
        reported = report_runs[run_key]
        if str(Path(reported.get("run_path", "")).resolve()) != expected["path"]:
            raise ValueError(f"Robustness run path differs for {run_key}")
        if reported.get("config_sha256") != expected["config_sha256"]:
            raise ValueError(f"Robustness config hash differs for {run_key}")
        if reported.get("best_checkpoint_sha256") != expected["best_checkpoint_sha256"]:
            raise ValueError(f"Robustness checkpoint hash differs for {run_key}")
        if reported.get("model_source_commit") != expected["provenance"]["git_commit"]:
            raise ValueError(f"Robustness model source differs for {run_key}")
        conditions = reported.get("conditions", {})
        if set(conditions) != REQUIRED_ROBUSTNESS_CONDITIONS:
            raise ValueError(f"Robustness condition matrix incomplete for {run_key}")
        for condition_id, result in conditions.items():
            if int(result.get("frame_count", -1)) != len(reference_frames):
                raise ValueError(
                    f"Robustness frame count differs for {run_key} {condition_id}"
                )
            frame_keys = {
                (int(frame["sequence"]), int(frame["radar_index"]))
                for frame in result.get("frames", [])
            }
            if frame_keys != reference_frames:
                raise ValueError(
                    f"Robustness frame identities differ for {run_key} {condition_id}"
                )
            for source, metric in REQUIRED_ROBUSTNESS_AGGREGATES:
                if metric not in result.get(source, {}):
                    raise ValueError(
                        f"Missing robustness metric {source}.{metric} for "
                        f"{run_key} {condition_id}"
                    )
    return {
        "protocol": ROBUSTNESS_PROTOCOL,
        "condition_count": len(REQUIRED_ROBUSTNESS_CONDITIONS),
        "run_count": len(report_runs),
        "frame_count_per_condition": len(reference_frames),
    }


def runs_by_seed(paths: list[Path], variant: str) -> dict[int, dict]:
    runs = [load_run(path, variant) for path in paths]
    indexed = {run["seed"]: run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError(f"Duplicate seeds for cycle variant {variant}")
    return indexed


def validate_runs(arms: dict[str, dict[int, dict]], required_seeds: int) -> None:
    seed_sets = [set(runs) for runs in arms.values()]
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("C0-C3 seed sets differ")
    if len(seed_sets[0]) != required_seeds:
        raise ValueError(f"G3 requires {required_seeds} seeds per variant")
    reference_frames = None
    reference_config = None
    reference_global = None
    for seed in sorted(seed_sets[0]):
        parent_hash = None
        for runs in arms.values():
            run = runs[seed]
            frame_keys = set(run["frames"])
            paired_config = {
                key: value
                for key, value in run["config"].items()
                if key not in CONFIG_PAIR_EXCLUSIONS
            }
            global_provenance = (
                run["provenance"]["git_commit"],
                run["provenance"]["manifest_sha256"],
                run["provenance"]["scene_split_sha256"],
                run["provenance"]["normalization_sha256"],
                run["provenance"]["torch_version"],
                run["provenance"]["device"],
            )
            if reference_frames is None:
                reference_frames = frame_keys
                reference_config = paired_config
                reference_global = global_provenance
            if frame_keys != reference_frames:
                raise ValueError("C0-C3 validation frame sets differ")
            if paired_config != reference_config:
                raise ValueError("C0-C3 configurations differ beyond variant and seed")
            if global_provenance != reference_global:
                raise ValueError("C0-C3 source, data, or runtime provenance differs")
            current_parent = run["provenance"]["parent_g2_checkpoint_sha256"]
            if parent_hash is None:
                parent_hash = current_parent
            if current_parent != parent_hash:
                raise ValueError(f"C0-C3 for seed {seed} use different G2 parents")


def paired_groups(
    first_runs: dict[int, dict],
    second_runs: dict[int, dict],
    source: str,
    key: str,
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    grouped: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed, first in first_runs.items():
        second = second_runs[seed]
        for frame_key in sorted(set(first["frames"]) & set(second["frames"])):
            first_report = first["frames"][frame_key][source]
            second_report = second["frames"][frame_key][source]
            if key not in first_report or key not in second_report:
                continue
            grouped[seed][frame_key[0]].append(
                (float(first_report[key]), float(second_report[key]))
            )
    return grouped


def summarize_groups(
    grouped: dict[int, dict[int, list[tuple[float, float]]]],
    direction: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    seeds = sorted(grouped)
    scenes = sorted({scene for by_scene in grouped.values() for scene in by_scene})
    if not seeds or not scenes:
        raise ValueError("No paired Cube-cycle observations")

    def statistic(sampled_seeds, sampled_scenes) -> tuple[float, float, float, float]:
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
            return (float("nan"),) * 4
        first_mean = float(np.mean(first_values))
        second_mean = float(np.mean(second_values))
        relative = (second_mean - first_mean) / max(abs(first_mean), 1e-12)
        return float(np.mean(improvements)), relative, first_mean, second_mean

    point, relative, first_mean, second_mean = statistic(seeds, scenes)
    improvements = []
    relatives = []
    while len(improvements) < bootstrap_samples:
        value, relative_value, _, _ = statistic(
            rng.choice(seeds, size=len(seeds), replace=True),
            rng.choice(scenes, size=len(scenes), replace=True),
        )
        if np.isfinite(value):
            improvements.append(value)
            relatives.append(relative_value)
    return {
        "direction": direction,
        "first_mean": first_mean,
        "second_mean": second_mean,
        "improvement": point,
        "improvement_ci95": np.quantile(improvements, (0.025, 0.975)).tolist(),
        "relative_change": relative,
        "relative_change_ci95": np.quantile(relatives, (0.025, 0.975)).tolist(),
        "seed_count": len(seeds),
        "scene_count": len(scenes),
        "paired_frame_seed_count": int(
            sum(
                len(values)
                for by_scene in grouped.values()
                for values in by_scene.values()
            )
        ),
    }


def compare(
    first: dict[int, dict],
    second: dict[int, dict],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    return {
        endpoint: summarize_groups(
            paired_groups(first, second, source, key),
            direction,
            bootstrap_samples,
            rng,
        )
        for endpoint, (source, key, direction) in ENDPOINTS.items()
    }


def confidently_better(report: dict) -> bool:
    return report["improvement_ci95"][0] > 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--none-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--local-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--marginal-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--full-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--renderer-test-report", type=Path, required=True)
    parser.add_argument("--robustness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    arms = {
        "c0_none": runs_by_seed(args.none_runs, "none"),
        "c1_local": runs_by_seed(args.local_runs, "local_peak"),
        "c2_marginal": runs_by_seed(args.marginal_runs, "marginal"),
        "c3_full": runs_by_seed(args.full_runs, "full"),
    }
    validate_runs(arms, args.required_seeds)
    renderer_test = json.loads(
        args.renderer_test_report.read_text(encoding="utf-8")
    )
    robustness = json.loads(args.robustness_report.read_text(encoding="utf-8"))
    robustness_validation = validate_robustness(robustness, arms)
    rng = np.random.default_rng(args.seed)
    comparisons = {
        "c1_local_vs_c0_none": compare(
            arms["c0_none"], arms["c1_local"], args.bootstrap_samples, rng
        ),
        "c2_marginal_vs_c0_none": compare(
            arms["c0_none"], arms["c2_marginal"], args.bootstrap_samples, rng
        ),
        "c3_full_vs_c0_none": compare(
            arms["c0_none"], arms["c3_full"], args.bootstrap_samples, rng
        ),
    }
    primary = comparisons["c3_full_vs_c0_none"]
    geometry_improved = any(
        confidently_better(primary[key])
        for key in ("geometry_chamfer_m", "geometry_fscore_1m")
    )
    geometry_nondegraded = primary["geometry_chamfer_m"][
        "relative_change_ci95"
    ][1] <= 0.02
    confidence_mean_ok = primary["confidence_mean"]["relative_change_ci95"][0] >= -0.1
    coverage_ok = primary["covered_cell_count"]["relative_change_ci95"][0] >= -0.1
    ece_degradation_upper = -primary["confidence_ece"]["improvement_ci95"][0]
    checks = {
        "renderer_unit_test_passed": renderer_test.get("passed") is True,
        "robustness_matrix_completed": True,
        "local_spectrum_kl_gain": confidently_better(
            primary["local_spectrum_kl"]
        ),
        "second_metric_class_gain": any(
            confidently_better(primary[key])
            for key in (
                "static_pce_median_mps",
                "geometry_chamfer_m",
                "geometry_fscore_1m",
            )
        ),
        "geometry_safeguard": geometry_improved or geometry_nondegraded,
        "confidence_mean_retained": confidence_mean_ok,
        "coverage_retained": coverage_ok,
        "confidence_ece_nondegradation": ece_degradation_upper <= 0.02,
    }
    report = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "seeds": sorted(arms["c0_none"]),
        "gate_thresholds": {
            "maximum_geometry_chamfer_relative_degradation": 0.02,
            "minimum_confidence_mean_ratio": 0.9,
            "minimum_covered_cell_ratio": 0.9,
            "maximum_confidence_ece_absolute_degradation": 0.02,
        },
        "comparisons": comparisons,
        "renderer_test_report": str(args.renderer_test_report),
        "robustness_report": str(args.robustness_report),
        "robustness_validation": robustness_validation,
        "confidence_ece_degradation_upper_ci95": ece_degradation_upper,
        "checks": checks,
        "g3_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["g3_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
