#!/usr/bin/env python3
"""Apply the frozen three-seed gates to G3L VAE and EDM evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_g1_cube_occupancy import summarize_groups  # noqa: E402
from scripts.eval_rald_anchor_g3l import (  # noqa: E402
    FORMAL_SEEDS,
    MODES,
    PROTOCOL as EVALUATION_PROTOCOL,
    VAE_PROTOCOL,
)
from scripts.g1b_contract import sha256  # noqa: E402


G3L1_GATE_PROTOCOL = "rald_anchor_g3l1_physical_vae_gate_v1"
G3L2_GATE_PROTOCOL = "rald_anchor_g3l2_full_raed_edm_gate_v1"
ENDPOINTS = {
    "chamfer_m": ("geometry", "chamfer_m", "lower"),
    "local_spectrum_kl": ("cycle", "local_spectrum_kl", "lower"),
    "circular_w1_mps": ("doppler", "circular_w1_mps", "lower"),
    "confidence_mean": ("cycle", "confidence_mean", "higher"),
    "covered_cell_count": ("cycle", "covered_cell_count", "higher"),
}
BASELINE_SOURCES = {
    "geometry": "refined_geometry",
    "doppler": "refined_doppler",
    "cycle": "cycle",
}


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return document


def _frame_map(frames: list[dict]) -> dict[tuple[int, int], dict]:
    mapped = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in frames
    }
    if len(mapped) != len(frames):
        raise ValueError("G3L evaluation contains duplicate frames")
    return mapped


def load_evaluation(path: Path, mode: str) -> dict:
    path = path.expanduser().resolve()
    report = _load_json(path)
    if report.get("protocol") != EVALUATION_PROTOCOL or report.get("mode") != mode:
        raise ValueError(f"Expected a {mode} G3L evaluation: {path}")
    if report.get("completed") is not True:
        raise ValueError(f"Incomplete G3L evaluation: {path}")
    if report.get("single_sample_per_frame") is not True:
        raise ValueError("G3L evaluation must use one sample per frame")
    if report.get("best_of_k") is not False:
        raise ValueError("G3L evaluation used prohibited best-of-k selection")
    if report.get("partitions") != ["validation"] or report.get("test_accessed") is not False:
        raise ValueError("G3L evaluation is not validation-only")
    seed = int(report["seed"])
    if seed not in FORMAL_SEEDS:
        raise ValueError("G3L evaluation seed is not in the frozen formal set")

    vae_run = Path(report["vae_run"]).expanduser().resolve()
    vae_config_path = vae_run / "config.json"
    vae_checkpoint_path = vae_run / "best.pt"
    if sha256(vae_config_path) != report["vae_config_sha256"]:
        raise ValueError("G3L evaluation VAE config hash is stale")
    if sha256(vae_checkpoint_path) != report["vae_checkpoint_sha256"]:
        raise ValueError("G3L evaluation VAE checkpoint hash is stale")
    vae_document = _load_json(vae_config_path)
    if vae_document.get("config", {}).get("protocol") != VAE_PROTOCOL:
        raise ValueError("G3L evaluation references an incompatible VAE")
    if int(vae_document["config"]["seed"]) != seed:
        raise ValueError("G3L evaluation and VAE seeds differ")

    parent_run = Path(
        vae_document["provenance"]["g3r_selected_run"]
    ).expanduser().resolve()
    parent_config_path = parent_run / "config.json"
    parent_checkpoint_path = parent_run / "best.pt"
    parent_metrics_path = parent_run / "best_validation_metrics.json"
    for candidate in (parent_config_path, parent_checkpoint_path, parent_metrics_path):
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
    if sha256(parent_config_path) != vae_document["provenance"]["g3r_selected_config_sha256"]:
        raise ValueError("G3L parent config hash is stale")
    if sha256(parent_checkpoint_path) != vae_document["provenance"]["g3r_selected_checkpoint_sha256"]:
        raise ValueError("G3L parent checkpoint hash is stale")
    parent_metrics = _load_json(parent_metrics_path)
    baseline_frames = _frame_map(parent_metrics["final"]["frames"])

    if mode == "posterior_mean":
        frames = _frame_map(report["evaluation"]["frames"])
        shuffled_frames = None
        posterior = report["evaluation"]["posterior"]
        diversity = None
    else:
        frames = _frame_map(report["evaluation"]["clean"]["frames"])
        shuffled_frames = _frame_map(
            report["evaluation"]["condition_shuffle"]["frames"]
        )
        posterior = None
        diversity = report["evaluation"]["diversity"]
        edm_run = Path(report["edm_run"]).expanduser().resolve()
        if sha256(edm_run / "config.json") != report["edm_config_sha256"]:
            raise ValueError("G3L evaluation EDM config hash is stale")
        if sha256(edm_run / "final.pt") != report["edm_checkpoint_sha256"]:
            raise ValueError("G3L evaluation EDM checkpoint hash is stale")
    if set(frames) != set(baseline_frames):
        raise ValueError("G3L and G3R validation frame sets differ")
    if shuffled_frames is not None and set(shuffled_frames) != set(frames):
        raise ValueError("G3L clean and shuffled frame sets differ")
    return {
        "path": path,
        "sha256": sha256(path),
        "report": report,
        "seed": seed,
        "frames": frames,
        "shuffled_frames": shuffled_frames,
        "baseline_frames": baseline_frames,
        "posterior": posterior,
        "diversity": diversity,
        "vae_run": vae_run,
        "vae_config_sha256": report["vae_config_sha256"],
        "vae_checkpoint_sha256": report["vae_checkpoint_sha256"],
        "vae_provenance": vae_document["provenance"],
    }


def runs_by_seed(paths: list[Path], mode: str) -> dict[int, dict]:
    loaded = [load_evaluation(path, mode) for path in paths]
    runs = {run["seed"]: run for run in loaded}
    if len(runs) != len(loaded):
        raise ValueError("Duplicate G3L formal seeds")
    if set(runs) != set(FORMAL_SEEDS):
        raise ValueError("G3L requires the exact frozen three-seed matrix")
    reference_frames = None
    reference_artifacts = None
    reference_sources = None
    for run in runs.values():
        frames = set(run["frames"])
        artifacts = tuple(sorted(run["report"]["artifact_hashes"].items()))
        provenance = run["vae_provenance"]
        sources = (
            provenance["git_commit"],
            provenance["g3r_source_commit"],
            provenance["torch_version"],
            provenance["device"],
        )
        if reference_frames is None:
            reference_frames = frames
            reference_artifacts = artifacts
            reference_sources = sources
        if frames != reference_frames:
            raise ValueError("G3L validation frames differ across seeds")
        if artifacts != reference_artifacts:
            raise ValueError("G3L data artifacts differ across seeds")
        if sources != reference_sources:
            raise ValueError("G3L source or runtime provenance differs across seeds")
    return runs


def paired_groups(
    runs: dict[int, dict],
    endpoint: str,
    *,
    first: str,
    second: str,
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    source, key, _ = ENDPOINTS[endpoint]
    grouped: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed, run in runs.items():
        first_frames = run[first]
        second_frames = run[second]
        for frame_key in sorted(run["frames"]):
            first_source = BASELINE_SOURCES[source] if first == "baseline_frames" else source
            second_source = (
                BASELINE_SOURCES[source] if second == "baseline_frames" else source
            )
            grouped[seed][frame_key[0]].append(
                (
                    float(first_frames[frame_key][first_source][key]),
                    float(second_frames[frame_key][second_source][key]),
                )
            )
    return grouped


def compare_runs(
    runs: dict[int, dict],
    *,
    first: str,
    second: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    return {
        endpoint: summarize_groups(
            paired_groups(runs, endpoint, first=first, second=second),
            direction,
            bootstrap_samples,
            rng,
        )
        for endpoint, (_, _, direction) in ENDPOINTS.items()
    }


def g3l1_decision(comparison: dict, runs: dict[int, dict]) -> dict:
    diagnostics = [run["posterior"] for run in runs.values()]
    checks = {
        "chamfer_retained_within_2pct": comparison["chamfer_m"]["relative_change_ci95"][1] <= 0.02,
        "local_spectrum_kl_retained_within_5pct": comparison["local_spectrum_kl"]["relative_change_ci95"][1] <= 0.05,
        "circular_w1_retained_within_5pct": comparison["circular_w1_mps"]["relative_change_ci95"][1] <= 0.05,
        "confidence_retained_at_least_90pct": comparison["confidence_mean"]["relative_change_ci95"][0] >= -0.10,
        "coverage_retained_at_least_90pct": comparison["covered_cell_count"]["relative_change_ci95"][0] >= -0.10,
        "posterior_finite": all(item["all_finite"] for item in diagnostics),
        "posterior_variance_nonzero": min(item["variance_mean"] for item in diagnostics) > 1e-6,
        "posterior_variance_bounded": max(item["variance_mean"] for item in diagnostics) < 1e3,
        "posterior_not_constant": min(item["across_frame_mean_std"] for item in diagnostics) > 1e-5,
        "doppler_state_used": min(
            min(
                item["doppler_intervention_latent_rms_mean"],
                item["doppler_intervention_decoder_rms_mean"],
            )
            for item in diagnostics
        ) > 1e-6,
        "confidence_state_used": min(
            min(
                item["confidence_intervention_latent_rms_mean"],
                item["confidence_intervention_decoder_rms_mean"],
            )
            for item in diagnostics
        ) > 1e-6,
    }
    return {"g3l1_passed": all(checks.values()), "checks": checks}


def g3l2_decision(comparison: dict, shuffle: dict) -> dict:
    # For lower-is-better endpoints, a shuffled-condition regression makes the
    # clean-minus-shuffled improvement strictly negative.
    condition_effect = {
        endpoint: shuffle[endpoint]["improvement_ci95"][1] < 0.0
        for endpoint in ("chamfer_m", "local_spectrum_kl")
    }
    checks = {
        "chamfer_retained_within_5pct": comparison["chamfer_m"]["relative_change_ci95"][1] <= 0.05,
        "local_spectrum_kl_retained_within_5pct": comparison["local_spectrum_kl"]["relative_change_ci95"][1] <= 0.05,
        "circular_w1_retained_within_5pct": comparison["circular_w1_mps"]["relative_change_ci95"][1] <= 0.05,
        "confidence_retained_at_least_90pct": comparison["confidence_mean"]["relative_change_ci95"][0] >= -0.10,
        "coverage_retained_at_least_90pct": comparison["covered_cell_count"]["relative_change_ci95"][0] >= -0.10,
        "full_raed_condition_effect": any(condition_effect.values()),
    }
    return {
        "g3l2_passed": all(checks.values()),
        "checks": checks,
        "condition_effect": condition_effect,
    }


def validate_g3l1_gate(path: Path, runs: dict[int, dict]) -> dict:
    report = _load_json(path)
    if report.get("protocol") != G3L1_GATE_PROTOCOL:
        raise ValueError("G3L-2 received the wrong G3L-1 gate protocol")
    if report.get("decision", {}).get("g3l1_passed") is not True:
        raise ValueError("G3L-2 requires a passing G3L-1 gate")
    expected_runs = {str(seed): str(run["vae_run"]) for seed, run in runs.items()}
    expected_hashes = {
        str(seed): {
            "config_sha256": run["vae_config_sha256"],
            "best_checkpoint_sha256": run["vae_checkpoint_sha256"],
        }
        for seed, run in runs.items()
    }
    if report.get("selected_runs") != expected_runs:
        raise ValueError("G3L-1 gate selected runs differ from G3L-2 evaluations")
    if report.get("selected_run_hashes") != expected_hashes:
        raise ValueError("G3L-1 gate selected hashes differ from G3L-2 evaluations")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("g3l1", "g3l2"), required=True)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--g3l1-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    if args.bootstrap_samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    mode = MODES[0] if args.mode == "g3l1" else MODES[1]
    runs = runs_by_seed(args.evaluations, mode)
    rng = np.random.default_rng(args.bootstrap_seed)
    baseline_comparison = compare_runs(
        runs,
        first="baseline_frames",
        second="frames",
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    selected_runs = {
        str(seed): str(run["vae_run"]) for seed, run in sorted(runs.items())
    }
    selected_hashes = {
        str(seed): {
            "config_sha256": run["vae_config_sha256"],
            "best_checkpoint_sha256": run["vae_checkpoint_sha256"],
        }
        for seed, run in sorted(runs.items())
    }
    if args.mode == "g3l1":
        if args.g3l1_gate is not None:
            raise ValueError("G3L-1 comparison cannot consume its own gate")
        decision = g3l1_decision(baseline_comparison, runs)
        report = {
            "protocol": G3L1_GATE_PROTOCOL,
            "completed": True,
            "seeds": sorted(runs),
            "selected_runs": selected_runs,
            "selected_run_hashes": selected_hashes,
            "evaluations": {str(seed): str(run["path"]) for seed, run in runs.items()},
            "evaluation_hashes": {str(seed): run["sha256"] for seed, run in runs.items()},
            "best_of_k": False,
            "test_accessed": False,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "gate_thresholds": {
                "maximum_chamfer_relative_degradation": 0.02,
                "maximum_physics_relative_degradation": 0.05,
                "minimum_confidence_and_coverage_retention": 0.90,
                "minimum_posterior_variance": 1e-6,
                "minimum_state_response_rms": 1e-6,
            },
            "g3r_vs_posterior_mean": baseline_comparison,
            "posterior_diagnostics": {
                str(seed): run["posterior"] for seed, run in runs.items()
            },
            "decision": decision,
        }
    else:
        if args.g3l1_gate is None:
            raise ValueError("G3L-2 comparison requires the passing G3L-1 gate")
        g3l1 = validate_g3l1_gate(args.g3l1_gate, runs)
        shuffle = compare_runs(
            runs,
            first="frames",
            second="shuffled_frames",
            bootstrap_samples=args.bootstrap_samples,
            rng=rng,
        )
        decision = g3l2_decision(baseline_comparison, shuffle)
        report = {
            "protocol": G3L2_GATE_PROTOCOL,
            "completed": True,
            "seeds": sorted(runs),
            "g3l1_gate": str(args.g3l1_gate.resolve()),
            "g3l1_gate_sha256": sha256(args.g3l1_gate),
            "g3l1_gate_protocol": g3l1["protocol"],
            "evaluations": {str(seed): str(run["path"]) for seed, run in runs.items()},
            "evaluation_hashes": {str(seed): run["sha256"] for seed, run in runs.items()},
            "single_sample_per_frame": True,
            "best_of_k": False,
            "test_accessed": False,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "gate_thresholds": {
                "maximum_geometry_and_physics_relative_degradation": 0.05,
                "minimum_confidence_and_coverage_retention": 0.90,
                "condition_shuffle_requires_upper_ci_below_zero": True,
            },
            "g3r_vs_edm_sample": baseline_comparison,
            "clean_vs_condition_shuffle": shuffle,
            "diversity_descriptive_only": {
                str(seed): run["diversity"] for seed, run in runs.items()
            },
            "decision": {
                "g3l_passed": decision["g3l2_passed"],
                **decision,
            },
        }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    passed = (
        report["decision"]["g3l1_passed"]
        if args.mode == "g3l1"
        else report["decision"]["g3l_passed"]
    )
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
