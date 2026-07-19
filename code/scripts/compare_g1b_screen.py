#!/usr/bin/env python3
"""Apply the preregistered one-seed G1B no-go screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODES = (
    "rae_max",
    "rae_moments",
    "rae_circular_harmonics",
    "full_raed_rank2",
)
CANDIDATE_MODES = MODES[1:]
CONFIG_PAIR_EXCLUSIONS = {"mode"}


def load_run(path: Path) -> dict:
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config = document["config"]
    mode = config["mode"]
    if mode not in MODES:
        raise ValueError(f"Unsupported G1B mode {mode} in {path}")
    metrics = json.loads(
        (path / "best_validation_metrics.json").read_text(encoding="utf-8")
    )
    log_records = [
        json.loads(line)
        for line in (path / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not log_records:
        raise ValueError(f"Empty training log in {path}")
    return {
        "path": str(path),
        "mode": mode,
        "config": config,
        "provenance": document["provenance"],
        "best_epoch": int(metrics["best_epoch"]),
        "metrics": metrics["validation"]["generated"],
        "spectral_diagnostics": log_records[-1].get("spectral_diagnostics", {}),
    }


def validate_runs(runs: list[dict], required_seed: int) -> dict:
    indexed = {run["mode"]: run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError("G1B screen contains duplicate modes")
    if set(indexed) != set(MODES):
        raise ValueError(
            f"G1B screen requires exactly {MODES}, received {tuple(indexed)}"
        )
    reference = indexed["rae_max"]
    reference_config = {
        key: value
        for key, value in reference["config"].items()
        if key not in CONFIG_PAIR_EXCLUSIONS
    }
    reference_runtime = tuple(
        reference["provenance"][key]
        for key in (
            "git_commit",
            "manifest_sha256",
            "scene_split_sha256",
            "normalization_sha256",
            "device",
            "torch_version",
        )
    )
    baseline_parameters = int(reference["provenance"]["model_parameter_count"])
    parameter_report = {}
    for mode in MODES:
        run = indexed[mode]
        if int(run["config"]["seed"]) != required_seed:
            raise ValueError(f"{mode} does not use required seed {required_seed}")
        paired_config = {
            key: value
            for key, value in run["config"].items()
            if key not in CONFIG_PAIR_EXCLUSIONS
        }
        if paired_config != reference_config:
            raise ValueError(f"{mode} configuration differs beyond mode")
        runtime = tuple(
            run["provenance"][key]
            for key in (
                "git_commit",
                "manifest_sha256",
                "scene_split_sha256",
                "normalization_sha256",
                "device",
                "torch_version",
            )
        )
        if runtime != reference_runtime:
            raise ValueError(f"{mode} provenance differs from RAE-Max")
        parameters = int(run["provenance"]["model_parameter_count"])
        relative_increase = (parameters - baseline_parameters) / baseline_parameters
        if relative_increase > 0.01:
            raise ValueError(f"{mode} parameter increase exceeds 1%")
        parameter_report[mode] = {
            "count": parameters,
            "relative_increase_over_rae_max": relative_increase,
        }
        if mode in CANDIDATE_MODES:
            diagnostics = run["spectral_diagnostics"]
            required = (
                diagnostics.get("first_step_gradient_norm"),
                diagnostics.get("spectral_branch_weight_rms"),
                diagnostics.get("spectral_to_trunk_weight_rms_ratio"),
            )
            if any(
                value is None or not math.isfinite(float(value))
                for value in required
            ):
                raise ValueError(f"{mode} lacks finite spectral diagnostics")
            if float(required[0]) <= 0.0 or float(required[1]) <= 0.0:
                raise ValueError(f"{mode} spectral branch did not learn")
    return {
        "source_commit": reference["provenance"]["git_commit"],
        "required_seed": required_seed,
        "parameters": parameter_report,
    }


def metric(run: dict, name: str, statistic: str) -> float:
    return float(run["metrics"][name][statistic])


def evaluate_screen(runs: list[dict], required_seed: int) -> dict:
    validation = validate_runs(runs, required_seed)
    indexed = {run["mode"]: run for run in runs}
    baseline = indexed["rae_max"]
    baseline_chamfer = metric(baseline, "chamfer_m", "median")
    candidates = {}
    survivors = []
    for mode in CANDIDATE_MODES:
        run = indexed[mode]
        chamfer = metric(run, "chamfer_m", "median")
        far_completeness = metric(
            run, "range_60_120m_completeness_mean_distance_m", "mean"
        )
        far_fscore = metric(run, "range_60_120m_fscore_1m", "mean")
        improvements = {
            "overall_chamfer": baseline_chamfer - chamfer,
            "far_completeness": metric(
                baseline,
                "range_60_120m_completeness_mean_distance_m",
                "mean",
            )
            - far_completeness,
            "far_fscore": far_fscore
            - metric(baseline, "range_60_120m_fscore_1m", "mean"),
        }
        checks = {
            "chamfer_nondegradation": (chamfer - baseline_chamfer)
            / max(abs(baseline_chamfer), 1e-12)
            <= 0.02,
            "outlier_at_most_25_percent": metric(
                run, "outlier_fraction_2m", "mean"
            )
            <= 0.25,
            "improves_preregistered_endpoint": any(
                value > 0.0 for value in improvements.values()
            ),
        }
        survived = all(checks.values())
        if survived:
            survivors.append(mode)
        candidates[mode] = {
            "run": run["path"],
            "best_epoch": run["best_epoch"],
            "metrics": {
                "chamfer_median_m": chamfer,
                "outlier_fraction_2m_mean": metric(
                    run, "outlier_fraction_2m", "mean"
                ),
                "far_completeness_mean_distance_m": far_completeness,
                "far_fscore_1m_mean": far_fscore,
            },
            "improvements_over_rae_max": improvements,
            "checks": checks,
            "survived": survived,
            "spectral_diagnostics": run["spectral_diagnostics"],
        }
    selected = None
    if survivors:
        selected = min(
            survivors,
            key=lambda mode: (
                candidates[mode]["metrics"]["chamfer_median_m"],
                candidates[mode]["metrics"][
                    "far_completeness_mean_distance_m"
                ],
                validation["parameters"][mode]["count"],
                mode,
            ),
        )
    return {
        "protocol": "G1B Stage A one-seed no-go screen",
        "validation": validation,
        "thresholds": {
            "maximum_chamfer_relative_degradation": 0.02,
            "maximum_outlier_fraction_2m_mean": 0.25,
        },
        "rae_max_run": baseline["path"],
        "candidates": candidates,
        "survivors": survivors,
        "selected_candidate": selected,
        "stage_b_authorized": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    report = evaluate_screen(
        [load_run(path) for path in args.runs], args.required_seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
