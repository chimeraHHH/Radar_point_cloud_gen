#!/usr/bin/env python3
"""Compare C0-C3 with paired scene-first bootstrap and decide G3."""

from __future__ import annotations

import argparse
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
    "confidence_ece": ("doppler", "soft_ece_10bin", "lower"),
}
CONFIG_PAIR_EXCLUSIONS = {"variant", "seed"}


def load_run(path: Path, expected_variant: str) -> dict:
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
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
        "seed": int(config["seed"]),
        "config": config,
        "provenance": provenance,
        "best_epoch": int(metrics["best_epoch"]),
        "frames": frames,
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
        "robustness_matrix_completed": robustness.get("completed") is True,
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
