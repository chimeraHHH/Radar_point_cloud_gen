#!/usr/bin/env python3
"""Independent three-seed Stage B decision for a frozen G1B representation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_g1_cube_occupancy import (  # noqa: E402
    DOPPLER_SENSITIVE_ENDPOINTS,
    compare,
    confidently_better,
    runs_by_seed,
)


def validate_pairs(
    baseline: dict[int, dict], candidate: dict[int, dict], candidate_mode: str
) -> dict:
    if set(baseline) != set(candidate):
        raise ValueError("G1B baseline and candidate seed sets differ")
    reference_frames = None
    reference_hashes = None
    reference_runtime = None
    reference_config = None
    counts = {"rae_max": set(), candidate_mode: set()}
    for mode, runs in (("rae_max", baseline), (candidate_mode, candidate)):
        for run in runs.values():
            frames = set(run["frames"])
            hashes = tuple(
                run["provenance"][key]
                for key in (
                    "manifest_sha256",
                    "scene_split_sha256",
                    "normalization_sha256",
                )
            )
            runtime = tuple(
                run["provenance"][key]
                for key in ("git_commit", "torch_version", "device")
            )
            paired_config = {
                key: value
                for key, value in run["config"].items()
                if key not in {"mode", "seed"}
            }
            if reference_frames is None:
                reference_frames = frames
                reference_hashes = hashes
                reference_runtime = runtime
                reference_config = paired_config
            if frames != reference_frames:
                raise ValueError("G1B validation frame sets differ")
            if hashes != reference_hashes:
                raise ValueError("G1B data artifact hashes differ")
            if runtime != reference_runtime:
                raise ValueError("G1B source or runtime differs")
            if paired_config != reference_config:
                raise ValueError("G1B configurations differ beyond mode and seed")
            counts[mode].add(int(run["provenance"]["model_parameter_count"]))
    if any(len(values) != 1 for values in counts.values()):
        raise ValueError("G1B parameter counts vary across seeds")
    baseline_count = next(iter(counts["rae_max"]))
    candidate_count = next(iter(counts[candidate_mode]))
    relative = (candidate_count - baseline_count) / baseline_count
    if relative > 0.01:
        raise ValueError("G1B candidate exceeds the 1% parameter budget")
    return {
        "baseline": baseline_count,
        "candidate": candidate_count,
        "relative_increase": relative,
        "maximum_relative_increase": 0.01,
        "passed": relative <= 0.01,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-mode", required=True)
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--launch-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    screen = json.loads(args.screen_report.read_text(encoding="utf-8"))
    launch = json.loads(args.launch_decision.read_text(encoding="utf-8"))
    if (
        screen.get("stage_b_authorized") is not True
        or screen.get("selected_candidate") != args.candidate_mode
        or launch.get("authorized") is not True
        or launch.get("candidate_mode") != args.candidate_mode
    ):
        raise ValueError("G1B candidate is not frozen by screen and launch decision")
    baseline = runs_by_seed(args.baseline_runs, "rae_max")
    candidate = runs_by_seed(args.candidate_runs, args.candidate_mode)
    if len(baseline) != args.required_seeds or len(candidate) != args.required_seeds:
        raise ValueError("G1B Stage B requires the preregistered seed count")
    parameter_parity = validate_pairs(baseline, candidate, args.candidate_mode)
    rng = np.random.default_rng(args.seed)
    candidate_vs_baseline = compare(
        baseline,
        candidate,
        "generated",
        "generated",
        args.bootstrap_samples,
        rng,
    )
    doppler_gain = any(
        confidently_better(candidate_vs_baseline[endpoint])
        for endpoint in DOPPLER_SENSITIVE_ENDPOINTS
    )
    checks = {
        "chamfer_nondegradation": candidate_vs_baseline["chamfer_m"][
            "relative_change_ci95"
        ][1]
        <= 0.02,
        "candidate_outlier_within_limit": candidate_vs_baseline[
            "outlier_fraction_2m"
        ]["second_mean"]
        <= 0.25,
        "doppler_sensitive_confident_gain": doppler_gain,
        "parameter_budget": parameter_parity["passed"],
    }
    report = {
        "protocol": "G1B Stage B independent three-seed decision",
        "candidate_mode": args.candidate_mode,
        "seeds": sorted(baseline),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "screen_report": str(args.screen_report),
        "launch_decision": str(args.launch_decision),
        "parameter_parity": parameter_parity,
        "gate_thresholds": {
            "maximum_chamfer_relative_degradation": 0.02,
            "maximum_candidate_outlier_fraction_2m": 0.25,
        },
        "candidate_vs_rae_max": candidate_vs_baseline,
        "checks": checks,
        "g1b_passed": all(checks.values()),
        "boundary": "This decision does not alter or reopen the original G1.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
