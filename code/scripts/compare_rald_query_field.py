#!/usr/bin/env python3
"""Apply the preregistered G1D RaLD query-field gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256  # noqa: E402
from scripts.train_rald_query_field import (  # noqa: E402
    OFFICIAL_RALD_COMMIT,
    PROTOCOL as TRAIN_PROTOCOL,
)


PROTOCOL = "g1d_rald_query_field_gate_v1"
STAGES = ("stage_a", "stage_b")
THRESHOLDS = {
    "chamfer_median_m": 2.50,
    "outlier_fraction_2m_mean": 0.25,
    "completeness_median_m": 0.65,
    "far_completeness_mean_m": 8.0,
    "duplicate_fraction_mean": 0.10,
    "confidence_mean": 0.10,
    "positive_occupancy_recall_mean": 0.80,
    "empty_false_positive_rate_mean": 0.20,
    "condition_shuffle_degradation_fraction": 0.01,
}
FORMAL_CONFIG = {
    "epochs": 150,
    "occupancy_query_count": 10_000,
    "positive_query_ratio": 0.0625,
    "positive_occupancy_weight": 0.1,
    "negative_occupancy_weight": 1.0,
    "base_seed_count": 1_000,
    "coarse_templates_per_seed": 32,
    "coarse_query_count": 32_000,
    "selected_coarse_count": 2_500,
    "local_templates_per_selection": 4,
    "point_count": 10_000,
    "latent_count": 512,
    "model_dim": 512,
    "depth": 24,
    "radar_token_count": 336,
    "train_limit": None,
    "validation_limit": None,
    "test_accessed": False,
}
REQUIRED_SOURCE_BASENAMES = {
    "train_rald_query_field.py",
    "rald_query_field.py",
    "rald_matched.py",
    "cube_doppler.py",
    "cube_occupancy.py",
    "cube_cycle.py",
    "point_to_cube.py",
    "dataset.py",
    "kradar.py",
    "dense_geometry.py",
    "rald_guided_query.py",
    "g1b_contract.py",
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


def _verify_hashed_path(path_value: object, digest_value: object, name: str) -> None:
    if not isinstance(path_value, str) or not isinstance(digest_value, str):
        raise ValueError(f"G1D {name} provenance is incomplete")
    path = Path(path_value)
    if not path.is_file() or sha256(path) != digest_value:
        raise ValueError(f"G1D {name} hash differs")


def verify_source_hashes(provenance: dict) -> None:
    source_hashes = provenance.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("G1D transitive source hashes are missing")
    observed_basenames = set()
    for raw_path, record in source_hashes.items():
        if not isinstance(raw_path, str) or not isinstance(record, dict):
            raise ValueError("G1D transitive source hash entry is invalid")
        digest = record.get("sha256")
        path = Path(raw_path)
        if not isinstance(digest, str) or not path.is_file():
            raise ValueError(f"G1D source path is unavailable: {raw_path}")
        if sha256(path) != digest:
            raise ValueError(f"G1D live source hash differs: {raw_path}")
        observed_basenames.add(path.name)
    missing = REQUIRED_SOURCE_BASENAMES - observed_basenames
    if missing:
        raise ValueError(f"G1D transitive source set is incomplete: {sorted(missing)}")


def _formal_config_checks(config: dict) -> dict[str, bool]:
    return {
        f"config_{key}": config.get(key) == expected
        for key, expected in FORMAL_CONFIG.items()
    }


def _validation_manifest_count(provenance: dict) -> int:
    manifest = _load_json(Path(provenance["manifest"]))
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("G1D manifest does not contain a frame list")
    return sum(frame.get("partition") == "validation" for frame in frames)


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
        raise ValueError("G1D run config or provenance is invalid")
    if config.get("protocol") != TRAIN_PROTOCOL:
        raise ValueError("G1D run configuration protocol differs")
    if metrics.get("protocol") != TRAIN_PROTOCOL:
        raise ValueError("G1D run metrics protocol differs")
    if metrics.get("completed") is not True:
        raise ValueError("G1D run is incomplete")
    if metrics.get("test_accessed") is not False:
        raise ValueError("G1D run accessed test")
    if provenance.get("test_accessed") is not False:
        raise ValueError("G1D provenance does not attest no test access")
    if provenance.get("worktree_clean") is not True:
        raise ValueError("G1D source worktree was not clean")
    if provenance.get("external_pretraining") is not False:
        raise ValueError("G1D must train from scratch")
    if provenance.get("cfar_query_helper") is not False:
        raise ValueError("G1D must not use a CFAR query helper")
    if provenance.get("occupancy_checkpoint") is not None:
        raise ValueError("G1D must not inherit a failed occupancy checkpoint")
    if provenance.get("official_rald_commit") != OFFICIAL_RALD_COMMIT:
        raise ValueError("G1D RaLD source reference differs")
    if provenance.get("doppler_geometry_status") != (
        "measured_cube_spectrum_attached_not_learned"
    ):
        raise ValueError("G1D Doppler geometry claim boundary differs")
    if provenance.get("proposal_index_cache") != (
        "deterministic_flat_indices_only"
    ):
        raise ValueError("G1D proposal-cache contract differs")
    if not all(_formal_config_checks(config).values()):
        raise ValueError("G1D formal configuration differs from the frozen protocol")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if metrics.get("best_checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("G1D selected checkpoint hash differs")
    if metrics.get("provenance") != provenance:
        raise ValueError("G1D metrics provenance differs from config.json")
    for name in ("manifest", "scene_split", "normalization"):
        _verify_hashed_path(
            provenance.get(name),
            provenance.get(f"{name}_sha256"),
            name,
        )
    _verify_hashed_path(
        provenance.get("protocol_document"),
        provenance.get("protocol_document_sha256"),
        "protocol_document",
    )
    _verify_hashed_path(
        provenance.get("rald_source_map"),
        provenance.get("rald_source_map_sha256"),
        "rald_source_map",
    )
    verify_source_hashes(provenance)
    validation = metrics.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("G1D validation report is missing")
    frames_list = validation.get("frames")
    if not isinstance(frames_list, list):
        raise ValueError("G1D validation frame records are missing")
    frames = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in frames_list
    }
    if len(frames) != int(validation.get("frame_count", -1)):
        raise ValueError("G1D validation frame identities are duplicated")
    if len(frames) != _validation_manifest_count(provenance):
        raise ValueError("G1D did not evaluate the complete validation partition")
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


def _finite_positive(values: object, expected_count: int) -> bool:
    if not isinstance(values, list) or len(values) != expected_count:
        return False
    return all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
        for value in values
    )


def gradient_checks(run: dict) -> dict[str, bool]:
    steps = run["metrics"].get("gradient_steps", [])
    if not isinstance(steps, list) or len(steps) != 2:
        return {"exactly_two_gradient_audits": False}
    if [step.get("update") for step in steps] != [1, 2]:
        return {"exactly_two_gradient_audits": False}
    first = steps[0].get("gradients", {})
    second = steps[1].get("gradients", {})
    scalar_names = (
        "output_heads",
        "mixed_latent",
        "query_decoder",
        "full_raed_radar_encoder",
    )
    return {
        "exactly_two_gradient_audits": True,
        "first_step_output_head_gradient": (
            float(first.get("output_heads", 0.0)) > 0.0
        ),
        **{
            f"second_step_{name}_gradient": (
                isinstance(second.get(name), (int, float))
                and math.isfinite(float(second[name]))
                and float(second[name]) > 0.0
            )
            for name in scalar_names
        },
        "second_step_all_cube_channels": _finite_positive(
            second.get("cube_input_channel_norms"), 64
        ),
        "second_step_all_local_spectrum_columns": _finite_positive(
            second.get("local_spectrum_input_column_norms"), 64
        ),
        "second_step_absolute_energy_column": _finite_positive(
            second.get("absolute_energy_input_column_norms"), 1
        ),
        "second_step_normalized_range_column": _finite_positive(
            second.get("normalized_range_input_column_norms"), 1
        ),
        "second_step_all_radar_projection_columns": _finite_positive(
            second.get("radar_projection_input_column_norms"), 64
        ),
        "second_step_all_24_condition_blocks": _finite_positive(
            second.get("condition_block_gradient_norms"), 24
        ),
    }


def frame_count_checks(run: dict) -> dict[str, bool]:
    frames = list(run["frames"].values())
    expected = {
        "radar_token_count": 336,
        "coarse_query_count": 32_000,
        "selected_coarse_count": 2_500,
        "final_point_count": 10_000,
    }
    checks = {
        f"all_frames_{name}": bool(frames)
        and all(int(frame.get(name, -1)) == count for frame in frames)
        for name, count in expected.items()
    }
    checks["all_frames_generated_10000_points"] = bool(frames) and all(
        int(frame.get("generated", {}).get("prediction_count", -1)) == 10_000
        for frame in frames
    )
    checks["all_frames_occupancy_queries_10000"] = bool(frames) and all(
        int(frame.get("occupancy_query_count", -1)) == 10_000
        for frame in frames
    )
    checks["all_frames_positive_queries_625"] = bool(frames) and all(
        int(frame.get("positive_occupancy_query_count", -1)) == 625
        for frame in frames
    )
    checks["all_frames_empty_queries_9375"] = bool(frames) and all(
        int(frame.get("empty_occupancy_query_count", -1)) == 9_375
        for frame in frames
    )
    checks["all_frames_used_deterministic_proposal_cache"] = bool(frames) and all(
        frame.get("proposal_cache_used") is True for frame in frames
    )
    checks["all_frames_matched_query_fractional_coordinate_rates"] = (
        bool(frames)
        and all(
            float(frame.get("positive_fractional_coordinate_rate", -1.0))
            >= 0.90
            and float(frame.get("empty_fractional_coordinate_rate", -1.0))
            >= 0.90
            and abs(
                float(frame["positive_fractional_coordinate_rate"])
                - float(frame["empty_fractional_coordinate_rate"])
            )
            <= 0.05
            for frame in frames
        )
    )
    checks["all_condition_pairs_cross_scene"] = bool(frames) and all(
        int(frame.get("sequence", -1))
        != int(frame.get("shuffled_condition_sequence", -1))
        for frame in frames
    )
    checks["all_frames_normalized_query_energy_bounded"] = bool(frames) and all(
        math.isfinite(float(frame.get("normalized_log_energy_abs_max", math.nan)))
        and 0.0 <= float(frame["normalized_log_energy_abs_max"]) <= 4.0
        for frame in frames
    )
    return checks


def stage_a_values(run: dict) -> dict[str, float]:
    validation = run["metrics"]["validation"]
    generated = validation["generated"]
    control = validation["zero_offset_control"]
    occupancy = validation["occupancy"]
    shuffle = validation["condition_shuffle"]
    return {
        "chamfer_median_m": float(generated["chamfer_m"]["median"]),
        "outlier_fraction_2m_mean": float(
            generated["outlier_fraction_2m"]["mean"]
        ),
        "completeness_median_m": float(
            generated["completeness_mean_distance_m"]["median"]
        ),
        "far_completeness_mean_m": float(
            generated["range_60_120m_completeness_mean_distance_m"]["mean"]
        ),
        "duplicate_fraction_mean": float(
            validation["duplicates"]["duplicate_fraction_0p05m"]["mean"]
        ),
        "confidence_mean": float(validation["confidence_mean"]["mean"]),
        "positive_occupancy_recall_mean": float(
            occupancy["positive_recall"]["mean"]
        ),
        "empty_false_positive_rate_mean": float(
            occupancy["empty_false_positive_rate"]["mean"]
        ),
        "condition_shuffle_occupancy_bce_fraction_mean": float(
            shuffle["occupancy_bce_fraction"]["mean"]
        ),
        "condition_shuffle_chamfer_fraction_mean": float(
            shuffle["chamfer_fraction"]["mean"]
        ),
        "generated_chamfer_mean_m": float(generated["chamfer_m"]["mean"]),
        "control_chamfer_mean_m": float(control["chamfer_m"]["mean"]),
        "generated_outlier_fraction_2m_mean": float(
            generated["outlier_fraction_2m"]["mean"]
        ),
        "control_outlier_fraction_2m_mean": float(
            control["outlier_fraction_2m"]["mean"]
        ),
    }


def stage_a_decision(run: dict) -> dict:
    values = stage_a_values(run)
    shuffle_degradation = max(
        values["condition_shuffle_occupancy_bce_fraction_mean"],
        values["condition_shuffle_chamfer_fraction_mean"],
    )
    regresses_chamfer = (
        values["generated_chamfer_mean_m"] > values["control_chamfer_mean_m"]
    )
    regresses_outlier = (
        values["generated_outlier_fraction_2m_mean"]
        > values["control_outlier_fraction_2m_mean"]
    )
    checks = {
        "chamfer": (
            values["chamfer_median_m"] <= THRESHOLDS["chamfer_median_m"]
        ),
        "outlier": (
            values["outlier_fraction_2m_mean"]
            <= THRESHOLDS["outlier_fraction_2m_mean"]
        ),
        "completeness": (
            values["completeness_median_m"]
            <= THRESHOLDS["completeness_median_m"]
        ),
        "far_completeness": (
            values["far_completeness_mean_m"]
            <= THRESHOLDS["far_completeness_mean_m"]
        ),
        "duplicates": (
            values["duplicate_fraction_mean"]
            <= THRESHOLDS["duplicate_fraction_mean"]
        ),
        "confidence": (
            values["confidence_mean"] >= THRESHOLDS["confidence_mean"]
        ),
        "positive_occupancy_recall": (
            values["positive_occupancy_recall_mean"]
            >= THRESHOLDS["positive_occupancy_recall_mean"]
        ),
        "empty_false_positive_rate": (
            values["empty_false_positive_rate_mean"]
            <= THRESHOLDS["empty_false_positive_rate_mean"]
        ),
        "condition_shuffle_degradation": (
            shuffle_degradation
            >= THRESHOLDS["condition_shuffle_degradation_fraction"]
        ),
        "zero_offset_causal_no_double_regression": not (
            regresses_chamfer and regresses_outlier
        ),
        **_formal_config_checks(run["config"]),
        **gradient_checks(run),
        **frame_count_checks(run),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "values": values,
        "condition_shuffle_degradation_fraction": shuffle_degradation,
    }


def validate_stage_b_runs(runs: dict[int, dict], stage_a: dict) -> None:
    if set(runs) != set(FROZEN_G1B_SEEDS):
        raise ValueError("G1D Stage B requires the exact frozen three seeds")
    if stage_a.get("protocol") != PROTOCOL or stage_a.get("stage") != "stage_a":
        raise ValueError("G1D Stage B received an incompatible Stage A report")
    if stage_a.get("decision", {}).get("passed") is not True:
        raise ValueError("G1D Stage B requires a passing Stage A decision")
    seed_a = FROZEN_G1B_SEEDS[0]
    selected = stage_a.get("runs", {}).get(str(seed_a))
    selected_hash = stage_a.get("run_hashes", {}).get(str(seed_a), {})
    if str(runs[seed_a]["path"]) != selected:
        raise ValueError("G1D Stage B seed-A run differs from the passing screen")
    if runs[seed_a]["checkpoint_sha256"] != selected_hash.get(
        "best_checkpoint_sha256"
    ):
        raise ValueError("G1D Stage B seed-A checkpoint differs from the screen")

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
        provenance = (
            run["provenance"]["git_commit"],
            run["provenance"]["manifest_sha256"],
            run["provenance"]["scene_split_sha256"],
            run["provenance"]["normalization_sha256"],
            run["provenance"]["source_hashes"],
            run["provenance"]["torch_version"],
            run["provenance"]["device"],
        )
        if reference_config is None:
            reference_config = config
            reference_frames = frames
            reference_provenance = provenance
        if config != reference_config:
            raise ValueError("G1D Stage B configurations differ beyond seed")
        if frames != reference_frames:
            raise ValueError("G1D Stage B validation frames differ")
        if provenance != reference_provenance:
            raise ValueError("G1D Stage B source, data, or runtime differs")


def frame_endpoint(frame: dict, endpoint: str, arm: str = "generated") -> float:
    if endpoint in {
        "chamfer_m",
        "outlier_fraction_2m",
        "completeness_mean_distance_m",
    }:
        return float(frame[arm][endpoint])
    if endpoint == "duplicate_fraction_0p05m":
        return float(frame["duplicates"][endpoint])
    if endpoint == "positive_occupancy_false_negative_rate":
        return 1.0 - float(frame["occupancy"]["positive_recall"])
    if endpoint == "empty_false_positive_rate":
        return float(frame["occupancy"]["empty_false_positive_rate"])
    raise KeyError(endpoint)


def _scene_bootstrap(
    grouped: dict[int, dict[int, list[float]]],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    seeds = sorted(grouped)
    scenes = sorted({scene for per_seed in grouped.values() for scene in per_seed})
    if not seeds or not scenes:
        raise ValueError("G1D bootstrap requires non-empty seed and scene groups")

    def statistic(sampled_seeds, sampled_scenes) -> float:
        values = []
        for seed in sampled_seeds:
            for scene in sampled_scenes:
                observations = grouped[int(seed)].get(int(scene))
                if observations:
                    values.append(float(np.mean(observations)))
        if not values:
            raise ValueError("G1D bootstrap sample contains no observations")
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
            len(values)
            for per_seed in grouped.values()
            for values in per_seed.values()
        ),
    }


def absolute_scene_bootstrap(
    runs: dict[int, dict],
    endpoint: str,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    grouped: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed, run in runs.items():
        for frame_key, frame in run["frames"].items():
            grouped[seed][frame_key[0]].append(frame_endpoint(frame, endpoint))
    return _scene_bootstrap(
        grouped,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )


def paired_control_bootstrap(
    runs: dict[int, dict],
    endpoint: str,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    grouped: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed, run in runs.items():
        for frame_key, frame in run["frames"].items():
            delta = frame_endpoint(frame, endpoint, "generated") - frame_endpoint(
                frame, endpoint, "zero_offset_control"
            )
            grouped[seed][frame_key[0]].append(delta)
    report = _scene_bootstrap(
        grouped,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    report["contrast"] = "query_field_minus_zero_offset_control"
    return report


def stage_b_decision(
    runs: dict[int, dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(bootstrap_seed)
    absolute_endpoints = (
        "chamfer_m",
        "outlier_fraction_2m",
        "completeness_mean_distance_m",
        "duplicate_fraction_0p05m",
        "positive_occupancy_false_negative_rate",
        "empty_false_positive_rate",
    )
    absolute = {
        endpoint: absolute_scene_bootstrap(
            runs,
            endpoint,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        for endpoint in absolute_endpoints
    }
    paired = {
        endpoint: paired_control_bootstrap(
            runs,
            endpoint,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        for endpoint in ("chamfer_m", "outlier_fraction_2m")
    }
    paired_upper = [paired[name]["ci95"][1] for name in paired]
    per_seed_contracts = {
        str(seed): {
            **gradient_checks(run),
            **frame_count_checks(run),
            **_formal_config_checks(run["config"]),
        }
        for seed, run in runs.items()
    }
    checks = {
        "chamfer_upper_ci": absolute["chamfer_m"]["ci95"][1] <= 2.50,
        "outlier_upper_ci": (
            absolute["outlier_fraction_2m"]["ci95"][1] <= 0.25
        ),
        "completeness_upper_ci": (
            absolute["completeness_mean_distance_m"]["ci95"][1] <= 0.65
        ),
        "duplicate_upper_ci": (
            absolute["duplicate_fraction_0p05m"]["ci95"][1] <= 0.10
        ),
        "occupancy_false_negative_upper_ci": (
            absolute["positive_occupancy_false_negative_rate"]["ci95"][1]
            <= 0.20
        ),
        "empty_false_positive_upper_ci": (
            absolute["empty_false_positive_rate"]["ci95"][1] <= 0.20
        ),
        "paired_control_both_upper_nonpositive": all(
            value <= 0.0 for value in paired_upper
        ),
        "paired_control_one_upper_strictly_negative": any(
            value < 0.0 for value in paired_upper
        ),
        "all_seed_structural_contracts": all(
            all(contract.values()) for contract in per_seed_contracts.values()
        ),
    }
    statistics = {"absolute": absolute, "paired_control": paired}
    decision = {
        "passed": all(checks.values()),
        "checks": checks,
        "per_seed_contracts": per_seed_contracts,
    }
    return statistics, decision


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
    if args.bootstrap_samples <= 0:
        raise ValueError("G1D bootstrap sample count must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    loaded = [load_run(path) for path in args.runs]
    runs = {run["seed"]: run for run in loaded}
    if len(runs) != len(loaded):
        raise ValueError("Duplicate G1D seeds")
    if args.stage == "stage_a":
        if set(runs) != {FROZEN_G1B_SEEDS[0]}:
            raise ValueError("G1D Stage A requires only the frozen first seed")
        if args.stage_a_report is not None:
            raise ValueError("G1D Stage A cannot consume a prior report")
        decision = stage_a_decision(runs[FROZEN_G1B_SEEDS[0]])
        statistics = None
    else:
        if args.stage_a_report is None:
            raise ValueError("G1D Stage B requires its passing Stage A report")
        stage_a = _load_json(args.stage_a_report)
        validate_stage_b_runs(runs, stage_a)
        statistics, decision = stage_b_decision(
            runs,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    report = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "completed": True,
        "seeds": sorted(runs),
        "source_commits": {
            str(seed): run["provenance"]["git_commit"]
            for seed, run in runs.items()
        },
        "official_rald_commit": OFFICIAL_RALD_COMMIT,
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
        "formal_config": FORMAL_CONFIG,
        "bootstrap_samples": (
            args.bootstrap_samples if statistics is not None else None
        ),
        "bootstrap_seed": (
            args.bootstrap_seed if statistics is not None else None
        ),
        "statistics": statistics,
        "stage_a_report": (
            None
            if args.stage_a_report is None
            else str(args.stage_a_report.resolve())
        ),
        "stage_a_report_sha256": (
            None
            if args.stage_a_report is None
            else sha256(args.stage_a_report)
        ),
        "gate_source": str(Path(__file__).resolve()),
        "gate_source_sha256": sha256(Path(__file__).resolve()),
        "decision": decision,
        "test_accessed": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not decision["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
