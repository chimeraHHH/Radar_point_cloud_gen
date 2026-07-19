#!/usr/bin/env python3
"""Scene-first paired comparison of frozen parents and RaLD anchor refiners."""

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


GEOMETRY_ENDPOINTS = {
    "chamfer_m": "lower",
    "fscore_1p0m": "higher",
    "outlier_fraction_2m": "lower",
    "range_60_120m_completeness_mean_distance_m": "lower",
    "range_60_120m_fscore_1m": "higher",
}
DOPPLER_ENDPOINTS = {
    "spectrum_nll": "lower",
    "spectrum_kl": "lower",
    "circular_w1_mps": "lower",
    "circular_scalar_mae_mps": "lower",
}


def load_run(path: Path) -> dict:
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if document["config"]["rh1_one_frame"]:
        raise ValueError(f"RH2 cannot consume one-frame RH1 run {path}")
    metrics = json.loads(
        (path / "best_validation_metrics.json").read_text(encoding="utf-8")
    )
    frames = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in metrics["final"]["frames"]
    }
    return {
        "path": str(path),
        "seed": int(document["config"]["seed"]),
        "config": document["config"],
        "provenance": document["provenance"],
        "best_epoch": int(metrics["best_epoch"]),
        "frames": frames,
    }


def validate_runs(runs: dict[int, dict], required_seeds: int) -> None:
    if len(runs) != required_seeds:
        raise ValueError(f"RH2 requires {required_seeds} unique seeds")
    reference_frames = None
    reference_hashes = None
    reference_source = None
    reference_config = None
    for run in runs.values():
        frame_keys = set(run["frames"])
        hashes = tuple(
            run["provenance"][key]
            for key in (
                "manifest_sha256",
                "scene_split_sha256",
                "normalization_sha256",
                "g1_comparison_sha256",
            )
        )
        source = (
            run["provenance"]["git_commit"],
            run["provenance"]["torch_version"],
            run["provenance"]["device"],
        )
        paired_config = {
            key: value for key, value in run["config"].items() if key != "seed"
        }
        if reference_frames is None:
            reference_frames = frame_keys
            reference_hashes = hashes
            reference_source = source
            reference_config = paired_config
        if frame_keys != reference_frames:
            raise ValueError("RH2 validation frame sets differ")
        if hashes != reference_hashes:
            raise ValueError("RH2 data or G1 comparison hashes differ")
        if source != reference_source:
            raise ValueError("RH2 source or runtime differs")
        if paired_config != reference_config:
            raise ValueError("RH2 configurations differ beyond seed")


def paired_groups(
    runs: dict[int, dict],
    endpoint: str,
    first_source: str,
    second_source: str,
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    grouped: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed, run in runs.items():
        for (sequence, _), frame in sorted(run["frames"].items()):
            first = frame[first_source]
            second = frame[second_source]
            if endpoint not in first or endpoint not in second:
                continue
            grouped[seed][sequence].append(
                (float(first[endpoint]), float(second[endpoint]))
            )
    return grouped


def compare_endpoints(
    runs: dict[int, dict],
    endpoints: dict[str, str],
    first_source: str,
    second_source: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    return {
        endpoint: summarize_groups(
            paired_groups(runs, endpoint, first_source, second_source),
            direction,
            bootstrap_samples,
            rng,
        )
        for endpoint, direction in endpoints.items()
    }


def gate_decision(geometry: dict, doppler: dict, runs: dict[int, dict]) -> dict:
    confidence_values = []
    saturation_values = []
    for run in runs.values():
        for frame in run["frames"].values():
            confidence_values.append(float(frame["cycle"]["confidence_mean"]))
            saturation_values.append(
                float(frame["cycle"]["offset_saturation_fraction"])
            )
    checks = {
        "chamfer_nondegradation": geometry["chamfer_m"][
            "relative_change_ci95"
        ][1]
        <= 0.02,
        "geometry_confident_gain": any(
            confidently_better(report) for report in geometry.values()
        ),
        "doppler_nll_confident_gain": confidently_better(doppler["spectrum_nll"]),
        "confidence_not_collapsed": min(confidence_values) >= 0.1,
        "offset_saturation_bounded": max(saturation_values) <= 0.1,
    }
    return {
        "rh2_passed": all(checks.values()),
        **checks,
        "minimum_frame_confidence_mean": min(confidence_values),
        "maximum_frame_offset_saturation_fraction": max(saturation_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    loaded = [load_run(path) for path in args.runs]
    runs = {run["seed"]: run for run in loaded}
    if len(runs) != len(loaded):
        raise ValueError("Duplicate RH2 seeds")
    validate_runs(runs, args.required_seeds)
    rng = np.random.default_rng(args.seed)
    geometry = compare_endpoints(
        runs,
        GEOMETRY_ENDPOINTS,
        "parent_geometry",
        "refined_geometry",
        args.bootstrap_samples,
        rng,
    )
    doppler = compare_endpoints(
        runs,
        DOPPLER_ENDPOINTS,
        "direct_cube_doppler",
        "refined_doppler",
        args.bootstrap_samples,
        rng,
    )
    report = {
        "seeds": sorted(runs),
        "runs": {str(seed): run["path"] for seed, run in sorted(runs.items())},
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "gate_thresholds": {
            "maximum_chamfer_relative_degradation": 0.02,
            "minimum_frame_confidence_mean": 0.1,
            "maximum_frame_offset_saturation_fraction": 0.1,
        },
        "geometry_parent_vs_rald_anchor": geometry,
        "doppler_direct_cube_vs_rald_anchor": doppler,
        "decision": gate_decision(geometry, doppler, runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
