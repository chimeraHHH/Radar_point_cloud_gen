#!/usr/bin/env python3
"""Compare E3-E5 with paired scene-first bootstrap and decide G2."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ENDPOINTS = {
    "spectrum_nll": ("doppler", "spectrum_nll", "lower"),
    "circular_w1_mps": ("doppler", "circular_w1_mps", "lower"),
    "soft_ece_10bin": ("doppler", "soft_ece_10bin", "lower"),
    "static_pce_median_mps": ("doppler", "static_pce_median_mps", "lower"),
    "cd_doppler": ("cd_doppler", "cd_doppler", "lower"),
    "geometry_chamfer_m": ("generated_geometry", "chamfer_m", "lower"),
}
CONFIG_PAIR_EXCLUSIONS = {"head_mode", "seed"}


def load_run(path: Path, expected_mode: str) -> dict:
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if config["head_mode"] != expected_mode:
        raise ValueError(
            f"Expected {expected_mode}, found {config['head_mode']} in {path}"
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


def runs_by_seed(paths: list[Path], mode: str) -> dict[int, dict]:
    runs = [load_run(path, mode) for path in paths]
    indexed = {run["seed"]: run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError(f"Duplicate seeds for {mode}")
    return indexed


def validate_runs(arms: dict[str, dict[int, dict]], required_seeds: int) -> None:
    seed_sets = [set(runs) for runs in arms.values()]
    if not seed_sets or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("E3-E5 seed sets differ")
    if len(seed_sets[0]) != required_seeds:
        raise ValueError(f"G2 requires {required_seeds} seeds per arm")
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
                run["provenance"]["static_doppler_audit_sha256"],
                run["provenance"]["static_doppler_audit_passed"],
                run["provenance"]["torch_version"],
                run["provenance"]["device"],
            )
            if reference_frames is None:
                reference_frames = frame_keys
                reference_config = paired_config
                reference_global = global_provenance
            if frame_keys != reference_frames:
                raise ValueError("G2 validation frame sets differ")
            if paired_config != reference_config:
                raise ValueError("G2 configurations differ beyond head mode and seed")
            if global_provenance != reference_global:
                raise ValueError("G2 source, data, audit, or runtime provenance differs")
            current_parent = run["provenance"]["parent_e2_checkpoint_sha256"]
            if parent_hash is None:
                parent_hash = current_parent
            if current_parent != parent_hash:
                raise ValueError(f"G2 arms for seed {seed} use different E2 parents")
    if reference_global[5] is not True:
        raise ValueError("G2 decision requires a passed static Doppler audit")


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
        raise ValueError("No paired G2 observations are available")

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
    relative_changes = []
    while len(improvements) < bootstrap_samples:
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        improvement, relative_change, _, _ = statistic(
            sampled_seeds, sampled_scenes
        )
        if np.isfinite(improvement):
            improvements.append(improvement)
            relative_changes.append(relative_change)
    return {
        "direction": direction,
        "first_mean": first_mean,
        "second_mean": second_mean,
        "improvement": point,
        "improvement_ci95": np.quantile(improvements, (0.025, 0.975)).tolist(),
        "relative_change": relative,
        "relative_change_ci95": np.quantile(
            relative_changes, (0.025, 0.975)
        ).tolist(),
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


def compare_arms(
    first: dict[int, dict],
    second: dict[int, dict],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    reports = {}
    for endpoint, (source, key, direction) in ENDPOINTS.items():
        reports[endpoint] = summarize_groups(
            paired_groups(first, second, source, key),
            direction,
            bootstrap_samples,
            rng,
        )
    return reports


def confidently_better(report: dict) -> bool:
    return report["improvement_ci95"][0] > 0.0


def arm_mean(runs: dict[int, dict], source: str, key: str) -> float:
    scene_seed_values = []
    for run in runs.values():
        by_scene: dict[int, list[float]] = defaultdict(list)
        for (scene, _), frame in run["frames"].items():
            if key in frame[source]:
                by_scene[scene].append(float(frame[source][key]))
        scene_seed_values.extend(np.mean(values) for values in by_scene.values())
    if not scene_seed_values:
        raise ValueError(f"No values found for {source}.{key}")
    return float(np.mean(scene_seed_values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--distribution-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--physics-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--counterfactual-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")

    arms = {
        "e3_scalar": runs_by_seed(args.scalar_runs, "scalar"),
        "e4_distribution": runs_by_seed(args.distribution_runs, "distribution"),
        "e5_physics": runs_by_seed(args.physics_runs, "physics_distribution"),
    }
    validate_runs(arms, args.required_seeds)
    rng = np.random.default_rng(args.seed)
    e4_vs_e3 = compare_arms(
        arms["e3_scalar"],
        arms["e4_distribution"],
        args.bootstrap_samples,
        rng,
    )
    e5_vs_e4 = compare_arms(
        arms["e4_distribution"],
        arms["e5_physics"],
        args.bootstrap_samples,
        rng,
    )
    predicted_dynamic = arm_mean(
        arms["e5_physics"], "doppler", "predicted_dynamic_fraction"
    )
    target_dynamic = arm_mean(
        arms["e5_physics"], "doppler", "target_dynamic_fraction"
    )
    dynamic_ratio = predicted_dynamic / max(target_dynamic, 1e-12)
    q0_reference = {
        key: arm_mean(arms["e4_distribution"], "q0_direct_query", key)
        for key in (
            "spectrum_nll",
            "spectrum_kl",
            "circular_w1_mps",
            "soft_ece_10bin",
        )
    }
    counterfactual = json.loads(
        args.counterfactual_report.read_text(encoding="utf-8")
    )
    checks = {
        "distribution_spectrum_nll_gain": confidently_better(
            e4_vs_e3["spectrum_nll"]
        ),
        "distribution_secondary_gain": any(
            confidently_better(e4_vs_e3[key])
            for key in ("circular_w1_mps", "cd_doppler")
        ),
        "physics_static_pce_gain": confidently_better(
            e5_vs_e4["static_pce_median_mps"]
        ),
        "physics_secondary_gain": any(
            confidently_better(e5_vs_e4[key])
            for key in ("spectrum_nll", "soft_ece_10bin", "cd_doppler")
        ),
        "geometry_chamfer_nondegradation": e5_vs_e4["geometry_chamfer_m"][
            "relative_change_ci95"
        ][1]
        <= 0.02,
        "dynamic_fraction_not_collapsed": predicted_dynamic >= 0.05
        and 0.5 <= dynamic_ratio <= 1.5,
        "counterfactual_convention_response": counterfactual.get("passed") is True,
    }
    report = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "seeds": sorted(arms["e3_scalar"]),
        "gate_thresholds": {
            "maximum_geometry_chamfer_relative_degradation": 0.02,
            "minimum_predicted_dynamic_fraction": 0.05,
            "dynamic_fraction_ratio_interval": [0.5, 1.5],
        },
        "e4_distribution_vs_e3_scalar": e4_vs_e3,
        "e5_physics_vs_e4_distribution": e5_vs_e4,
        "q0_direct_query_reference": q0_reference,
        "dynamic_fraction": {
            "predicted": predicted_dynamic,
            "target": target_dynamic,
            "ratio": dynamic_ratio,
        },
        "counterfactual_report": str(args.counterfactual_report),
        "counterfactual": counterfactual,
        "checks": checks,
        "g2_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["g2_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
