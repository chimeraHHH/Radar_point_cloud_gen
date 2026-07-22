#!/usr/bin/env python3
"""Apply the preregistered G1C Stage A and Stage B geometry gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256  # noqa: E402
from scripts.train_rald_guided_query import PROTOCOL as TRAIN_PROTOCOL  # noqa: E402


PROTOCOL = "g1c_rald_guided_query_gate_v1"
STAGES = ("stage_a", "stage_b")
THRESHOLDS = {
    "chamfer_median_m": 2.50,
    "outlier_fraction_2m_mean": 0.25,
    "completeness_median_m": 0.65,
    "far_completeness_mean_m": 8.0,
    "duplicate_fraction_mean": 0.10,
    "confidence_mean": 0.10,
}
CONFIG_SEED_EXCLUSIONS = {"seed"}


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


def load_run(path: Path) -> dict:
    path = path.expanduser().resolve()
    config_path = path / "config.json"
    checkpoint_path = path / "best.pt"
    metrics_path = path / "best_validation_metrics.json"
    document = _load_json(config_path)
    config = document.get("config")
    provenance = document.get("provenance")
    metrics = _load_json(metrics_path)
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("G1C run config or provenance is invalid")
    if config.get("protocol") != TRAIN_PROTOCOL or metrics.get("protocol") != TRAIN_PROTOCOL:
        raise ValueError("G1C run protocol differs")
    if metrics.get("completed") is not True or metrics.get("test_accessed") is not False:
        raise ValueError("G1C run is incomplete or accessed test")
    if provenance.get("test_accessed") is not False:
        raise ValueError("G1C provenance does not attest no test access")
    if provenance.get("cfar_query_helper") is not False:
        raise ValueError("G1C must not use a CFAR query helper")
    if provenance.get("occupancy_checkpoint") is not None:
        raise ValueError("G1C must not inherit a failed occupancy checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if metrics.get("best_checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("G1C selected checkpoint hash differs")
    if metrics.get("provenance") != provenance:
        raise ValueError("G1C metrics provenance differs from config.json")
    for key, source_key in (
        ("training_script_sha256", "training_script"),
        ("model_source_sha256", "model_source"),
        ("loss_source_sha256", "loss_source"),
    ):
        if sha256(Path(provenance[source_key])) != provenance[key]:
            raise ValueError(f"G1C live source hash differs: {source_key}")
    frames = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in metrics["validation"]["frames"]
    }
    if len(frames) != metrics["validation"]["frame_count"]:
        raise ValueError("G1C validation frame identities are duplicated")
    return {
        "path": path,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "metrics_sha256": sha256(metrics_path),
        "config": config,
        "provenance": provenance,
        "metrics": metrics,
        "seed": int(config["seed"]),
        "frames": frames,
    }


def stage_a_values(run: dict) -> dict[str, float]:
    validation = run["metrics"]["validation"]
    return {
        "chamfer_median_m": float(validation["generated"]["chamfer_m"]["median"]),
        "outlier_fraction_2m_mean": float(
            validation["generated"]["outlier_fraction_2m"]["mean"]
        ),
        "completeness_median_m": float(
            validation["generated"]["completeness_mean_distance_m"]["median"]
        ),
        "far_completeness_mean_m": float(
            validation["generated"][
                "range_60_120m_completeness_mean_distance_m"
            ]["mean"]
        ),
        "duplicate_fraction_mean": float(
            validation["duplicates"]["duplicate_fraction_0p05m"]["mean"]
        ),
        "confidence_mean": float(validation["confidence_mean"]["mean"]),
    }


def gradient_checks(run: dict) -> dict[str, bool]:
    steps = run["metrics"].get("gradient_steps", [])
    if len(steps) < 2:
        return {
            "first_step_physical_gradient": False,
            "second_step_mixed_latent_gradient": False,
            "second_step_full_raed_gradient": False,
            "second_step_local_64bin_gradient": False,
        }
    return {
        "first_step_physical_gradient": steps[0]["gradients"]["physical_head"] > 0.0,
        "second_step_mixed_latent_gradient": steps[1]["gradients"][
            "mixed_latent_and_query_decoder"
        ] > 0.0,
        "second_step_full_raed_gradient": steps[1]["gradients"][
            "full_raed_radar_encoder"
        ] > 0.0,
        "second_step_local_64bin_gradient": steps[1]["gradients"][
            "local_64bin_spectrum_projection"
        ] > 0.0,
    }


def stage_a_decision(run: dict) -> dict:
    values = stage_a_values(run)
    checks = {
        "chamfer": values["chamfer_median_m"] <= THRESHOLDS["chamfer_median_m"],
        "outlier": values["outlier_fraction_2m_mean"] <= THRESHOLDS["outlier_fraction_2m_mean"],
        "completeness": values["completeness_median_m"] <= THRESHOLDS["completeness_median_m"],
        "far_completeness": values["far_completeness_mean_m"] <= THRESHOLDS["far_completeness_mean_m"],
        "duplicates": values["duplicate_fraction_mean"] <= THRESHOLDS["duplicate_fraction_mean"],
        "confidence": values["confidence_mean"] >= THRESHOLDS["confidence_mean"],
        **gradient_checks(run),
    }
    return {"passed": all(checks.values()), "checks": checks, "values": values}


def validate_stage_b_runs(runs: dict[int, dict], stage_a: dict) -> None:
    if set(runs) != set(FROZEN_G1B_SEEDS):
        raise ValueError("G1C Stage B requires the exact frozen three seeds")
    if stage_a.get("protocol") != PROTOCOL or stage_a.get("stage") != "stage_a":
        raise ValueError("G1C Stage B received an incompatible Stage A report")
    if stage_a.get("decision", {}).get("passed") is not True:
        raise ValueError("G1C Stage B requires a passing Stage A decision")
    seed_a = FROZEN_G1B_SEEDS[0]
    selected = stage_a.get("runs", {}).get(str(seed_a))
    selected_hash = stage_a.get("run_hashes", {}).get(str(seed_a), {})
    if str(runs[seed_a]["path"]) != selected:
        raise ValueError("G1C Stage B seed-A run differs from the passing screen")
    if runs[seed_a]["checkpoint_sha256"] != selected_hash.get("best_checkpoint_sha256"):
        raise ValueError("G1C Stage B seed-A checkpoint differs from the screen")

    reference_config = None
    reference_frames = None
    reference_provenance = None
    for run in runs.values():
        config = {
            key: value
            for key, value in run["config"].items()
            if key not in CONFIG_SEED_EXCLUSIONS
        }
        frames = set(run["frames"])
        provenance = tuple(
            run["provenance"][key]
            for key in (
                "git_commit",
                "manifest_sha256",
                "scene_split_sha256",
                "normalization_sha256",
                "torch_version",
                "device",
            )
        )
        if reference_config is None:
            reference_config = config
            reference_frames = frames
            reference_provenance = provenance
        if config != reference_config:
            raise ValueError("G1C Stage B configurations differ beyond seed")
        if frames != reference_frames:
            raise ValueError("G1C Stage B validation frames differ")
        if provenance != reference_provenance:
            raise ValueError("G1C Stage B source, data, or runtime differs")


def frame_endpoint(frame: dict, endpoint: str) -> float:
    if endpoint == "chamfer_m":
        return float(frame["generated"]["chamfer_m"])
    if endpoint == "outlier_fraction_2m":
        return float(frame["generated"]["outlier_fraction_2m"])
    if endpoint == "completeness_mean_distance_m":
        return float(frame["generated"]["completeness_mean_distance_m"])
    if endpoint == "duplicate_fraction_0p05m":
        return float(frame["duplicates"]["duplicate_fraction_0p05m"])
    raise KeyError(endpoint)


def absolute_scene_bootstrap(
    runs: dict[int, dict],
    endpoint: str,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seed, run in runs.items():
        for frame_key, frame in run["frames"].items():
            grouped[seed][frame_key[0]].append(frame_endpoint(frame, endpoint))
    seeds = sorted(grouped)
    scenes = sorted({scene for per_seed in grouped.values() for scene in per_seed})

    def statistic(sampled_seeds, sampled_scenes) -> float:
        values = []
        for seed in sampled_seeds:
            for scene in sampled_scenes:
                observations = grouped[int(seed)].get(int(scene))
                if observations:
                    values.append(float(np.mean(observations)))
        return float(np.mean(values))

    point = statistic(seeds, scenes)
    samples = [
        statistic(
            rng.choice(seeds, size=len(seeds), replace=True),
            rng.choice(scenes, size=len(scenes), replace=True),
        )
        for _ in range(bootstrap_samples)
    ]
    return {
        "mean": point,
        "ci95": np.quantile(samples, (0.025, 0.975)).tolist(),
        "seed_count": len(seeds),
        "scene_count": len(scenes),
        "frame_seed_count": sum(
            len(values) for per_seed in grouped.values() for values in per_seed.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--stage-a-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    loaded = [load_run(path) for path in args.runs]
    runs = {run["seed"]: run for run in loaded}
    if len(runs) != len(loaded):
        raise ValueError("Duplicate G1C seeds")
    if args.stage == "stage_a":
        if set(runs) != {FROZEN_G1B_SEEDS[0]}:
            raise ValueError("G1C Stage A requires only seed 20260716")
        if args.stage_a_report is not None:
            raise ValueError("G1C Stage A cannot consume a prior report")
        decision = stage_a_decision(runs[FROZEN_G1B_SEEDS[0]])
        statistics = None
    else:
        if args.stage_a_report is None:
            raise ValueError("G1C Stage B requires its passing Stage A report")
        stage_a = _load_json(args.stage_a_report)
        validate_stage_b_runs(runs, stage_a)
        rng = np.random.default_rng(args.bootstrap_seed)
        statistics = {
            endpoint: absolute_scene_bootstrap(
                runs,
                endpoint,
                bootstrap_samples=args.bootstrap_samples,
                rng=rng,
            )
            for endpoint in (
                "chamfer_m",
                "outlier_fraction_2m",
                "completeness_mean_distance_m",
                "duplicate_fraction_0p05m",
            )
        }
        gradient = {str(seed): gradient_checks(run) for seed, run in runs.items()}
        checks = {
            "chamfer_upper_ci": statistics["chamfer_m"]["ci95"][1] <= 2.50,
            "outlier_upper_ci": statistics["outlier_fraction_2m"]["ci95"][1] <= 0.25,
            "completeness_upper_ci": statistics[
                "completeness_mean_distance_m"
            ]["ci95"][1] <= 0.65,
            "duplicate_upper_ci": statistics[
                "duplicate_fraction_0p05m"
            ]["ci95"][1] <= 0.10,
            "all_seed_gradient_contracts": all(
                all(seed_checks.values()) for seed_checks in gradient.values()
            ),
        }
        decision = {
            "passed": all(checks.values()),
            "checks": checks,
            "gradient_checks": gradient,
        }
    report = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "completed": True,
        "seeds": sorted(runs),
        "runs": {str(seed): str(run["path"]) for seed, run in runs.items()},
        "run_hashes": {
            str(seed): {
                "config_sha256": run["config_sha256"],
                "best_checkpoint_sha256": run["checkpoint_sha256"],
                "metrics_sha256": run["metrics_sha256"],
            }
            for seed, run in runs.items()
        },
        "thresholds": THRESHOLDS,
        "bootstrap_samples": args.bootstrap_samples if statistics is not None else None,
        "bootstrap_seed": args.bootstrap_seed if statistics is not None else None,
        "statistics": statistics,
        "stage_a_report": (
            None if args.stage_a_report is None else str(args.stage_a_report.resolve())
        ),
        "stage_a_report_sha256": (
            None if args.stage_a_report is None else sha256(args.stage_a_report)
        ),
        "decision": decision,
        "test_accessed": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not decision["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
