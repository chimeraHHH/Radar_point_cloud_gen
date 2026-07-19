#!/usr/bin/env python3
"""Decide RaLD-anchor G3R from matched Cube-cycle ablations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_g1_cube_occupancy import (  # noqa: E402
    confidently_better,
    summarize_groups,
)
from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256  # noqa: E402


VARIANTS = ("none", "local_peak", "marginal", "full")
ENDPOINTS = {
    "local_spectrum_kl": ("cycle", "local_spectrum_kl", "lower"),
    "doppler_marginal_kl": ("cycle", "doppler_marginal_kl", "lower"),
    "spectrum_nll": ("refined_doppler", "spectrum_nll", "lower"),
    "circular_w1_mps": ("refined_doppler", "circular_w1_mps", "lower"),
    "cd_doppler": ("refined_cd_doppler", "cd_doppler", "lower"),
    "geometry_chamfer_m": ("refined_geometry", "chamfer_m", "lower"),
    "geometry_fscore_1m": ("refined_geometry", "fscore_1p0m", "higher"),
    "confidence_mean": ("cycle", "confidence_mean", "higher"),
    "covered_cell_count": ("cycle", "covered_cell_count", "higher"),
    "confidence_ece": ("cycle", "existence_ece_10bin", "lower"),
    "offset_saturation_fraction": (
        "cycle",
        "offset_saturation_fraction",
        "lower",
    ),
}
CONFIG_GLOBAL_EXCLUSIONS = {
    "cycle_variant",
    "seed",
    "initial_refiner_run",
}
ROBUSTNESS_PROTOCOL = "rald_anchor_g3r_robustness_v1"


def load_run(path: Path, expected_variant: str) -> dict:
    path = path.resolve()
    config_path = path / "config.json"
    checkpoint_path = path / "best.pt"
    metrics_path = path / "best_validation_metrics.json"
    if not all(candidate.is_file() for candidate in (config_path, checkpoint_path, metrics_path)):
        raise FileNotFoundError(f"Incomplete G3R run: {path}")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if config["cycle_variant"] != expected_variant:
        raise ValueError(f"Expected {expected_variant} cycle run in {path}")
    if config["doppler_head_mode"] != "distribution":
        raise ValueError(f"G3R requires the selected distribution head: {path}")
    if config.get("initial_refiner_run") is None:
        raise ValueError(f"G3R must start from the selected G2R checkpoint: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frames = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in metrics["final"]["frames"]
    }
    return {
        "path": str(path),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "seed": int(config["seed"]),
        "config": config,
        "provenance": provenance,
        "frames": frames,
    }


def runs_by_seed(paths: list[Path], variant: str) -> dict[int, dict]:
    loaded = [load_run(path, variant) for path in paths]
    runs = {run["seed"]: run for run in loaded}
    if len(runs) != len(loaded):
        raise ValueError(f"Duplicate G3R {variant} seeds")
    return runs


def validate_g2r_parent(g2r: dict, arms: dict[str, dict[int, dict]]) -> None:
    if g2r.get("protocol") != "rald_anchor_g2r_v1":
        raise ValueError("G3R received an incompatible G2R report")
    if g2r.get("decision", {}).get("g2r_passed") is not True:
        raise ValueError("G3R requires a passing G2R decision")
    distribution_paths = g2r.get("runs", {}).get("distribution", {})
    distribution_hashes = g2r.get("run_hashes", {}).get("distribution", {})
    for seed in FROZEN_G1B_SEEDS:
        expected_path = str(Path(distribution_paths[str(seed)]).resolve())
        expected_hash = distribution_hashes[str(seed)]["best_checkpoint_sha256"]
        for runs in arms.values():
            run = runs[seed]
            provenance = run["provenance"]
            if str(Path(run["config"]["initial_refiner_run"]).resolve()) != expected_path:
                raise ValueError(f"G3R seed {seed} uses the wrong G2R run")
            if provenance["initial_refiner_checkpoint_sha256"] != expected_hash:
                raise ValueError(f"G3R seed {seed} uses the wrong G2R checkpoint")


def validate_runs(
    arms: dict[str, dict[int, dict]], required_seeds: int, g2r: dict
) -> None:
    seed_sets = [set(runs) for runs in arms.values()]
    expected = set(FROZEN_G1B_SEEDS)
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("G3R cycle seed sets differ")
    if len(seed_sets[0]) != required_seeds or seed_sets[0] != expected:
        raise ValueError("G3R requires the exact frozen three-seed matrix")
    reference_frames = None
    reference_global_config = None
    reference_global_provenance = None
    for seed in sorted(seed_sets[0]):
        reference_seed_config = None
        initial_hash = None
        parent_hash = None
        for runs in arms.values():
            run = runs[seed]
            frame_keys = set(run["frames"])
            seed_config = {
                key: value
                for key, value in run["config"].items()
                if key != "cycle_variant"
            }
            global_config = {
                key: value
                for key, value in run["config"].items()
                if key not in CONFIG_GLOBAL_EXCLUSIONS
            }
            provenance = run["provenance"]
            global_provenance = tuple(
                provenance.get(key)
                for key in (
                    "git_commit",
                    "manifest_sha256",
                    "scene_split_sha256",
                    "normalization_sha256",
                    "g1_comparison_sha256",
                    "g1b_summary_sha256",
                    "g1b_training_source_commit",
                    "g1b_decision_source_commit",
                    "initial_refiner_source_commit",
                    "torch_version",
                    "device",
                )
            )
            if reference_frames is None:
                reference_frames = frame_keys
                reference_global_config = global_config
                reference_global_provenance = global_provenance
            if frame_keys != reference_frames:
                raise ValueError("G3R validation frame sets differ")
            if global_config != reference_global_config:
                raise ValueError("G3R global configurations differ")
            if global_provenance != reference_global_provenance:
                raise ValueError("G3R source, data, gate, or runtime provenance differs")
            if reference_seed_config is None:
                reference_seed_config = seed_config
                initial_hash = provenance["initial_refiner_checkpoint_sha256"]
                parent_hash = provenance["parent_g1_checkpoint_sha256"]
            if seed_config != reference_seed_config:
                raise ValueError(f"G3R seed {seed} differs beyond cycle variant")
            if provenance["initial_refiner_checkpoint_sha256"] != initial_hash:
                raise ValueError(f"G3R seed {seed} has different initial checkpoints")
            if provenance["parent_g1_checkpoint_sha256"] != parent_hash:
                raise ValueError(f"G3R seed {seed} has different geometry parents")
    validate_g2r_parent(g2r, arms)


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
        for frame_key in sorted(first["frames"]):
            grouped[seed][frame_key[0]].append(
                (
                    float(first["frames"][frame_key][source][key]),
                    float(second["frames"][frame_key][source][key]),
                )
            )
    return grouped


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


def validate_robustness(report: dict, arms: dict[str, dict[int, dict]]) -> dict:
    if report.get("protocol") != ROBUSTNESS_PROTOCOL:
        raise ValueError("G3R robustness report uses the wrong protocol")
    if report.get("completed") is not True:
        raise ValueError("G3R robustness report is incomplete")
    checks = report.get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("G3R robustness completion checks failed")
    expected = {
        (variant, seed): (
            runs[seed]["path"],
            runs[seed]["config_sha256"],
            runs[seed]["checkpoint_sha256"],
        )
        for variant, runs in arms.items()
        if variant in ("none", "full")
        for seed in runs
    }
    reported = {
        (run["variant"], int(run["seed"])): (
            str(Path(run["run_path"]).resolve()),
            run["config_sha256"],
            run["best_checkpoint_sha256"],
        )
        for run in report.get("runs", [])
    }
    if reported != expected:
        raise ValueError("G3R robustness run identities differ from clean runs")
    return {
        "protocol": ROBUSTNESS_PROTOCOL,
        "run_count": len(reported),
        "condition_count": len(report.get("condition_definitions", [])),
        "frame_count": int(report.get("full_validation_frame_count", 0)),
    }


def gate_decision(primary: dict, full_runs: dict[int, dict]) -> dict:
    confidence_ok = primary["confidence_mean"]["relative_change_ci95"][0] >= -0.1
    coverage_ok = primary["covered_cell_count"]["relative_change_ci95"][0] >= -0.1
    ece_degradation_upper = -primary["confidence_ece"]["improvement_ci95"][0]
    maximum_saturation = max(
        float(frame["cycle"]["offset_saturation_fraction"])
        for run in full_runs.values()
        for frame in run["frames"].values()
    )
    checks = {
        "local_spectrum_kl_gain": confidently_better(
            primary["local_spectrum_kl"]
        ),
        "second_metric_class_gain": any(
            confidently_better(primary[key])
            for key in (
                "spectrum_nll",
                "circular_w1_mps",
                "cd_doppler",
                "geometry_chamfer_m",
                "geometry_fscore_1m",
            )
        ),
        "geometry_chamfer_nondegradation": primary["geometry_chamfer_m"][
            "relative_change_ci95"
        ][1]
        <= 0.02,
        "confidence_mean_retained": confidence_ok,
        "coverage_retained": coverage_ok,
        "confidence_ece_nondegradation": ece_degradation_upper <= 0.02,
        "offset_saturation_bounded": maximum_saturation <= 0.1,
    }
    return {
        "g3r_statistical_gate_passed": all(checks.values()),
        **checks,
        "confidence_ece_degradation_upper_ci95": ece_degradation_upper,
        "maximum_frame_offset_saturation_fraction": maximum_saturation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--none-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--local-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--marginal-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--full-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--g2r-comparison", type=Path, required=True)
    parser.add_argument("--renderer-test-report", type=Path, required=True)
    parser.add_argument("--robustness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    arms = {
        "none": runs_by_seed(args.none_runs, "none"),
        "local_peak": runs_by_seed(args.local_runs, "local_peak"),
        "marginal": runs_by_seed(args.marginal_runs, "marginal"),
        "full": runs_by_seed(args.full_runs, "full"),
    }
    g2r = json.loads(args.g2r_comparison.read_text(encoding="utf-8"))
    validate_runs(arms, args.required_seeds, g2r)
    renderer = json.loads(args.renderer_test_report.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness_report.read_text(encoding="utf-8"))
    robustness_validation = validate_robustness(robustness, arms)
    rng = np.random.default_rng(args.seed)
    comparisons = {
        f"{variant}_vs_none": compare(
            arms["none"], arms[variant], args.bootstrap_samples, rng
        )
        for variant in ("local_peak", "marginal", "full")
    }
    statistical = gate_decision(comparisons["full_vs_none"], arms["full"])
    checks = {
        "renderer_unit_test_passed": renderer.get("passed") is True,
        "robustness_matrix_completed": robustness.get("completed") is True,
        "statistical_gate_passed": statistical["g3r_statistical_gate_passed"],
    }
    report = {
        "protocol": "rald_anchor_g3r_v1",
        "seeds": sorted(arms["none"]),
        "runs": {
            variant: {
                str(seed): run["path"] for seed, run in sorted(runs.items())
            }
            for variant, runs in arms.items()
        },
        "run_hashes": {
            variant: {
                str(seed): {
                    "config_sha256": run["config_sha256"],
                    "best_checkpoint_sha256": run["checkpoint_sha256"],
                }
                for seed, run in sorted(runs.items())
            }
            for variant, runs in arms.items()
        },
        "g2r_comparison": str(args.g2r_comparison),
        "g2r_comparison_sha256": sha256(args.g2r_comparison),
        "renderer_test_report": str(args.renderer_test_report),
        "renderer_test_report_sha256": sha256(args.renderer_test_report),
        "robustness_report": str(args.robustness_report),
        "robustness_report_sha256": sha256(args.robustness_report),
        "robustness_validation": robustness_validation,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "gate_thresholds": {
            "maximum_chamfer_relative_degradation": 0.02,
            "minimum_confidence_and_coverage_retention": 0.9,
            "maximum_confidence_ece_absolute_degradation": 0.02,
            "maximum_frame_offset_saturation_fraction": 0.1,
        },
        "comparisons": comparisons,
        "statistical_decision": statistical,
        "checks": checks,
        "decision": {"g3r_passed": all(checks.values())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["decision"]["g3r_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
