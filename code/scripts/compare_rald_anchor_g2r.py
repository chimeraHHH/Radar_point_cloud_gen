#!/usr/bin/env python3
"""Decide RaLD-anchor G2R from matched scalar and distribution runs."""

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


HEAD_MODES = ("scalar", "distribution")
ENDPOINTS = {
    "spectrum_nll": ("refined_doppler", "spectrum_nll", "lower"),
    "spectrum_kl": ("refined_doppler", "spectrum_kl", "lower"),
    "circular_w1_mps": ("refined_doppler", "circular_w1_mps", "lower"),
    "circular_scalar_mae_mps": (
        "refined_doppler",
        "circular_scalar_mae_mps",
        "lower",
    ),
    "soft_ece_10bin": ("refined_doppler", "soft_ece_10bin", "lower"),
    "cd_doppler": ("refined_cd_doppler", "cd_doppler", "lower"),
    "geometry_chamfer_m": ("refined_geometry", "chamfer_m", "lower"),
    "geometry_fscore_1m": ("refined_geometry", "fscore_1p0m", "higher"),
}
DIRECT_ENDPOINTS = {
    "spectrum_nll": (
        "refined_cube_query_doppler",
        "refined_doppler",
        "spectrum_nll",
        "lower",
    ),
    "spectrum_kl": (
        "refined_cube_query_doppler",
        "refined_doppler",
        "spectrum_kl",
        "lower",
    ),
    "circular_w1_mps": (
        "refined_cube_query_doppler",
        "refined_doppler",
        "circular_w1_mps",
        "lower",
    ),
    "soft_ece_10bin": (
        "refined_cube_query_doppler",
        "refined_doppler",
        "soft_ece_10bin",
        "lower",
    ),
    "cd_doppler": (
        "refined_cube_query_cd_doppler",
        "refined_cd_doppler",
        "cd_doppler",
        "lower",
    ),
}
CONFIG_PAIR_EXCLUSIONS = {"doppler_head_mode", "seed"}


def load_run(path: Path, expected_head: str) -> dict:
    path = path.resolve()
    config_path = path / "config.json"
    checkpoint_path = path / "best.pt"
    metrics_path = path / "best_validation_metrics.json"
    if not all(candidate.is_file() for candidate in (config_path, checkpoint_path, metrics_path)):
        raise FileNotFoundError(f"Incomplete G2R run: {path}")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    config = document["config"]
    if config["doppler_head_mode"] != expected_head:
        raise ValueError(f"Expected {expected_head} head in {path}")
    if config["cycle_variant"] != "none":
        raise ValueError(f"G2R must be cycle-free: {path}")
    if config.get("initial_refiner_run") is not None:
        raise ValueError(f"G2R cannot start from a trained RaLD checkpoint: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frames = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in metrics["final"]["frames"]
    }
    return {
        "path": str(path),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "seed": int(config["seed"]),
        "config": config,
        "provenance": document["provenance"],
        "best_epoch": int(metrics["best_epoch"]),
        "frames": frames,
    }


def runs_by_seed(paths: list[Path], head: str) -> dict[int, dict]:
    loaded = [load_run(path, head) for path in paths]
    runs = {run["seed"]: run for run in loaded}
    if len(runs) != len(loaded):
        raise ValueError(f"Duplicate G2R {head} seeds")
    return runs


def validate_runs(arms: dict[str, dict[int, dict]], required_seeds: int) -> None:
    seed_sets = [set(runs) for runs in arms.values()]
    expected = set(FROZEN_G1B_SEEDS)
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("G2R scalar and distribution seed sets differ")
    if len(seed_sets[0]) != required_seeds or seed_sets[0] != expected:
        raise ValueError("G2R requires the exact frozen three-seed matrix")
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
                    "torch_version",
                    "device",
                )
            )
            if reference_frames is None:
                reference_frames = frame_keys
                reference_config = paired_config
                reference_global = global_provenance
            if frame_keys != reference_frames:
                raise ValueError("G2R validation frame sets differ")
            if paired_config != reference_config:
                raise ValueError("G2R configurations differ beyond head and seed")
            if global_provenance != reference_global:
                raise ValueError("G2R source, data, gate, or runtime provenance differs")
            current_parent = provenance["parent_g1_checkpoint_sha256"]
            if parent_hash is None:
                parent_hash = current_parent
            if current_parent != parent_hash:
                raise ValueError(f"G2R seed {seed} uses different geometry parents")


def paired_arm_groups(
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
            first_value = first["frames"][frame_key][source][key]
            second_value = second["frames"][frame_key][source][key]
            grouped[seed][frame_key[0]].append(
                (float(first_value), float(second_value))
            )
    return grouped


def paired_source_groups(
    runs: dict[int, dict], first_source: str, second_source: str, key: str
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    grouped: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed, run in runs.items():
        for frame_key, frame in sorted(run["frames"].items()):
            grouped[seed][frame_key[0]].append(
                (float(frame[first_source][key]), float(frame[second_source][key]))
            )
    return grouped


def compare_heads(
    scalar: dict[int, dict],
    distribution: dict[int, dict],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    return {
        endpoint: summarize_groups(
            paired_arm_groups(scalar, distribution, source, key),
            direction,
            bootstrap_samples,
            rng,
        )
        for endpoint, (source, key, direction) in ENDPOINTS.items()
    }


def compare_direct_query(
    distribution: dict[int, dict],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    return {
        endpoint: summarize_groups(
            paired_source_groups(distribution, first, second, key),
            direction,
            bootstrap_samples,
            rng,
        )
        for endpoint, (first, second, key, direction) in DIRECT_ENDPOINTS.items()
    }


def gate_decision(
    head_comparison: dict,
    direct_comparison: dict,
    distribution_runs: dict[int, dict],
) -> dict:
    confidence_values = []
    saturation_values = []
    for run in distribution_runs.values():
        for frame in run["frames"].values():
            confidence_values.append(float(frame["cycle"]["confidence_mean"]))
            saturation_values.append(
                float(frame["cycle"]["offset_saturation_fraction"])
            )
    checks = {
        "distribution_nll_gain": confidently_better(
            head_comparison["spectrum_nll"]
        ),
        "distribution_secondary_gain": any(
            confidently_better(head_comparison[key])
            for key in ("circular_w1_mps", "cd_doppler")
        ),
        "geometry_chamfer_nondegradation": head_comparison[
            "geometry_chamfer_m"
        ]["relative_change_ci95"][1]
        <= 0.02,
        "learned_spectrum_beats_direct_query": any(
            confidently_better(direct_comparison[key])
            for key in (
                "spectrum_nll",
                "circular_w1_mps",
                "soft_ece_10bin",
                "cd_doppler",
            )
        ),
        "confidence_not_collapsed": min(confidence_values) >= 0.1,
        "offset_saturation_bounded": max(saturation_values) <= 0.1,
    }
    return {
        "g2r_passed": all(checks.values()),
        **checks,
        "minimum_frame_confidence_mean": min(confidence_values),
        "maximum_frame_offset_saturation_fraction": max(saturation_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--distribution-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    arms = {
        "scalar": runs_by_seed(args.scalar_runs, "scalar"),
        "distribution": runs_by_seed(args.distribution_runs, "distribution"),
    }
    validate_runs(arms, args.required_seeds)
    rng = np.random.default_rng(args.seed)
    head_comparison = compare_heads(
        arms["scalar"], arms["distribution"], args.bootstrap_samples, rng
    )
    direct_comparison = compare_direct_query(
        arms["distribution"], args.bootstrap_samples, rng
    )
    report = {
        "protocol": "rald_anchor_g2r_v1",
        "seeds": sorted(arms["scalar"]),
        "runs": {
            head: {str(seed): run["path"] for seed, run in sorted(runs.items())}
            for head, runs in arms.items()
        },
        "run_hashes": {
            head: {
                str(seed): {
                    "config_sha256": run["config_sha256"],
                    "best_checkpoint_sha256": run["checkpoint_sha256"],
                }
                for seed, run in sorted(runs.items())
            }
            for head, runs in arms.items()
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "gate_thresholds": {
            "maximum_chamfer_relative_degradation": 0.02,
            "minimum_frame_confidence_mean": 0.1,
            "maximum_frame_offset_saturation_fraction": 0.1,
        },
        "distribution_vs_scalar": head_comparison,
        "learned_distribution_vs_same_position_direct_cube_query": direct_comparison,
        "decision": gate_decision(
            head_comparison, direct_comparison, arms["distribution"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["decision"]["g2r_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
