#!/usr/bin/env python3
"""Compare three-seed RaLD-native G4R rollouts with scene-first bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


COMPARISON_PROTOCOL = "rald_anchor_g4r_comparison_v1"
BASELINE_PROTOCOL = "rald_anchor_g4r_baselines_v1"
TEMPORAL_PROTOCOL = "rald_anchor_g4r_strict_rollout_v1"
FORMAL_SEEDS = {20260716, 20260717, 20260718}
BASELINE_ARMS = {
    "t0_single_frame",
    "history_aggregation",
    "raw_doppler_displacement_sensitivity",
}
ENDPOINTS = {
    "ego_aligned_matched_distance_m": (
        ("temporal", "ego_aligned_matched_distance_mean_m"),
        "lower",
    ),
    "occupancy_flicker": (("temporal", "occupancy_flicker"), "lower"),
    "geometry_chamfer_m": (
        ("current", "generated_geometry", "chamfer_m"),
        "lower",
    ),
    "local_spectrum_kl": (
        ("current", "cycle", "local_spectrum_kl"),
        "lower",
    ),
    "circular_w1_mps": (
        ("current", "doppler", "circular_w1_mps"),
        "lower",
    ),
    "confidence_mean": (("current", "cycle", "confidence_mean"), "higher"),
    "covered_cell_count": (
        ("current", "cycle", "covered_cell_count"),
        "higher",
    ),
}
COMMON_SHA256_FIELDS = (
    "manifest_sha256",
    "scene_split_sha256",
    "normalization_sha256",
    "dense_cache_report_sha256",
    "g3r_comparison_sha256",
    "g3r_config_sha256",
    "g3r_checkpoint_sha256",
    "parent_prediction_manifest_sha256",
)
TEMPORAL_SHA256_FIELDS = (
    "temporal_config_sha256",
    "temporal_checkpoint_sha256",
    "preflight_selection_sha256",
)
COMMON_ARTIFACT_FIELDS = (
    ("parent_prediction_manifest_path", "parent_prediction_manifest_sha256"),
    ("g3r_comparison_path", "g3r_comparison_sha256"),
    ("g3r_config_path", "g3r_config_sha256"),
    ("g3r_checkpoint_path", "g3r_checkpoint_sha256"),
)
TEMPORAL_ARTIFACT_FIELDS = (
    ("temporal_config_path", "temporal_config_sha256"),
    ("temporal_checkpoint_path", "temporal_checkpoint_sha256"),
    ("preflight_selection_path", "preflight_selection_sha256"),
)
GLOBAL_DATA_FIELDS = (
    "manifest_sha256",
    "scene_split_sha256",
    "normalization_sha256",
    "dense_cache_report_sha256",
    "g3r_comparison_sha256",
    "partition",
    "point_count",
)
PER_SEED_PARENT_FIELDS = (
    "g3r_config_sha256",
    "g3r_checkpoint_sha256",
    "parent_prediction_manifest_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def is_git_commit(value: object) -> bool:
    if not isinstance(value, str) or not 7 <= len(value) <= 40:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def nested_value(document: dict, path: tuple[str, ...]) -> float | None:
    current: object = document
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if not isinstance(current, (int, float)) or not np.isfinite(current):
        return None
    return float(current)


def frame_index(frames: list[dict], label: str) -> dict[tuple[int, int], dict]:
    indexed = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in frames
    }
    if len(indexed) != len(frames):
        raise ValueError(f"Duplicate frame identities in {label}")
    if not indexed:
        raise ValueError(f"No frames in {label}")
    return indexed


def completed_report(path: Path, expected_protocol: str, label: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != expected_protocol:
        raise ValueError(f"Incompatible {label} protocol: {path}")
    if report.get("completed") is not True:
        raise ValueError(f"Incomplete {label}: {path}")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError(f"Failed completion checks in {label}: {path}")
    return report


def load_baseline(path: Path) -> dict:
    path = path.resolve()
    report = completed_report(path, BASELINE_PROTOCOL, "G4R baseline report")
    configuration = report.get("configuration", {})
    arms = report.get("arms", {})
    if not BASELINE_ARMS.issubset(arms):
        raise ValueError(f"G4R baseline arms are incomplete: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "seed": int(configuration["seed"]),
        "configuration": configuration,
        "t0": frame_index(
            arms["t0_single_frame"]["frames"], f"{path}:t0_single_frame"
        ),
        "history": frame_index(
            arms["history_aggregation"]["frames"],
            f"{path}:history_aggregation",
        ),
        "raw_doppler": frame_index(
            arms["raw_doppler_displacement_sensitivity"]["frames"],
            f"{path}:raw_doppler_displacement_sensitivity",
        ),
    }


def load_temporal(path: Path) -> dict:
    path = path.resolve()
    report = completed_report(path, TEMPORAL_PROTOCOL, "G4R rollout report")
    configuration = report.get("configuration", {})
    training_checks = report.get("training_checks")
    if not isinstance(training_checks, dict) or not training_checks or not all(
        value is True for value in training_checks.values()
    ):
        raise ValueError(f"Failed G4R training checks: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "seed": int(configuration["seed"]),
        "configuration": configuration,
        "frames": frame_index(report["frames"], f"{path}:temporal"),
    }


def reports_by_seed(paths: list[Path], loader, label: str) -> dict[int, dict]:
    reports = [loader(path) for path in paths]
    indexed = {report["seed"]: report for report in reports}
    if len(indexed) != len(reports):
        raise ValueError(f"Duplicate seed in G4R {label} reports")
    if set(indexed) != FORMAL_SEEDS:
        raise ValueError(
            f"G4R {label} seed set is {set(indexed)}, expected {FORMAL_SEEDS}"
        )
    return indexed


def metric_covers_expected_frames(
    frames: dict[tuple[int, int], dict], path: tuple[str, ...]
) -> bool:
    for frame in frames.values():
        is_temporal_anchor = (
            path[0] == "temporal" and int(frame.get("rollout_step", -1)) == 0
        )
        if not is_temporal_anchor and nested_value(frame, path) is None:
            return False
    return True


def prediction_hashes_complete(frames: dict[tuple[int, int], dict]) -> bool:
    for frame in frames.values():
        prediction = frame.get("prediction", {})
        expected = prediction.get("sha256")
        path = Path(prediction.get("path", ""))
        if not is_sha256(expected) or not path.is_file() or sha256(path) != expected:
            return False
    return True


def configuration_hashes_complete(configuration: dict, temporal: bool) -> bool:
    sha_fields = COMMON_SHA256_FIELDS + (TEMPORAL_SHA256_FIELDS if temporal else ())
    required_scalars = (
        isinstance(configuration.get("point_count"), int)
        and int(configuration["point_count"]) > 0
        and configuration.get("partition") == "validation"
        and is_git_commit(configuration.get("source_commit"))
    )
    if temporal:
        required_scalars = (
            required_scalars
            and is_git_commit(configuration.get("model_source_commit"))
            and configuration.get("model_source_commit")
            == configuration.get("source_commit")
            and isinstance(configuration.get("fusion_mode"), str)
            and bool(configuration["fusion_mode"])
            and configuration.get("strict_recurrent_rollout") is True
        )
    hashes_complete = all(
        is_sha256(configuration.get(field)) for field in sha_fields
    )
    artifact_fields = COMMON_ARTIFACT_FIELDS + (
        TEMPORAL_ARTIFACT_FIELDS if temporal else ()
    )
    artifacts_complete = all(
        isinstance(configuration.get(path_field), str)
        and Path(configuration[path_field]).is_file()
        and sha256(Path(configuration[path_field])) == configuration.get(hash_field)
        for path_field, hash_field in artifact_fields
    )
    return required_scalars and hashes_complete and artifacts_complete


def validate_matched_reports(
    baselines: dict[int, dict],
    temporal: dict[int, dict],
    source_commit: str,
) -> dict[str, bool]:
    checks = {
        "three_frozen_seeds": set(baselines) == set(temporal) == FORMAL_SEEDS,
        "identical_frame_identities": True,
        "same_frozen_evaluation_data": True,
        "same_seed_g3r_parent": True,
        "same_selected_temporal_family": True,
        "same_temporal_training_source": True,
        "same_preflight_selection": True,
        "strict_recurrent_rollout": True,
        "step0_matches_t0": True,
        "all_endpoint_frames_present": True,
        "step25_covers_every_scene": True,
        "validation_only_test_untouched": True,
        "source_commit_bound": True,
        "complete_hash_provenance": True,
    }
    reference_identities = None
    reference_global = None
    reference_fusion = None
    reference_model_source = None
    reference_preflight = None
    for seed in sorted(FORMAL_SEEDS):
        baseline = baselines[seed]
        selected = temporal[seed]
        baseline_configuration = baseline["configuration"]
        selected_configuration = selected["configuration"]
        identities = set(baseline["t0"])
        if (
            set(baseline["history"]) != identities
            or set(baseline["raw_doppler"]) != identities
            or set(selected["frames"]) != identities
        ):
            checks["identical_frame_identities"] = False
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            checks["identical_frame_identities"] = False

        baseline_global = tuple(
            baseline_configuration.get(field) for field in GLOBAL_DATA_FIELDS
        )
        selected_global = tuple(
            selected_configuration.get(field) for field in GLOBAL_DATA_FIELDS
        )
        if baseline_global != selected_global:
            checks["same_frozen_evaluation_data"] = False
        if reference_global is None:
            reference_global = baseline_global
        elif baseline_global != reference_global:
            checks["same_frozen_evaluation_data"] = False

        baseline_parent = tuple(
            baseline_configuration.get(field) for field in PER_SEED_PARENT_FIELDS
        )
        selected_parent = tuple(
            selected_configuration.get(field) for field in PER_SEED_PARENT_FIELDS
        )
        if baseline_parent != selected_parent:
            checks["same_seed_g3r_parent"] = False

        fusion_mode = selected_configuration.get("fusion_mode")
        if reference_fusion is None:
            reference_fusion = fusion_mode
        elif fusion_mode != reference_fusion:
            checks["same_selected_temporal_family"] = False
        model_source = selected_configuration.get("model_source_commit")
        if reference_model_source is None:
            reference_model_source = model_source
        elif model_source != reference_model_source:
            checks["same_temporal_training_source"] = False
        preflight = selected_configuration.get("preflight_selection_sha256")
        if reference_preflight is None:
            reference_preflight = preflight
        elif preflight != reference_preflight:
            checks["same_preflight_selection"] = False
        if selected_configuration.get("strict_recurrent_rollout") is not True:
            checks["strict_recurrent_rollout"] = False

        scenes = {identity[0] for identity in identities}
        step0_identities = {
            identity
            for identity, frame in selected["frames"].items()
            if int(frame.get("rollout_step", -1)) == 0
        }
        if (
            len(step0_identities) != len(scenes)
            or {identity[0] for identity in step0_identities} != scenes
        ):
            checks["step0_matches_t0"] = False
        for identity in step0_identities:
            selected_frame = selected["frames"][identity]
            selected_hash = selected_frame.get("prediction", {}).get("sha256")
            baseline_hash = (
                baseline["t0"].get(identity, {}).get("prediction", {}).get("sha256")
            )
            if selected_hash != baseline_hash:
                checks["step0_matches_t0"] = False

        for endpoint_path, _ in ENDPOINTS.values():
            for frames in (
                baseline["t0"],
                baseline["history"],
                baseline["raw_doppler"],
                selected["frames"],
            ):
                if not metric_covers_expected_frames(frames, endpoint_path):
                    checks["all_endpoint_frames_present"] = False

        step25_scenes = {
            identity[0]
            for identity, frame in selected["frames"].items()
            if int(frame.get("rollout_step", -1)) == 25
        }
        step25_count = sum(
            int(frame.get("rollout_step", -1)) == 25
            for frame in selected["frames"].values()
        )
        if step25_count != len(scenes) or step25_scenes != scenes:
            checks["step25_covers_every_scene"] = False

        if (
            baseline_configuration.get("partition") != "validation"
            or selected_configuration.get("partition") != "validation"
        ):
            checks["validation_only_test_untouched"] = False
        if (
            baseline_configuration.get("source_commit") != source_commit
            or selected_configuration.get("source_commit") != source_commit
        ):
            checks["source_commit_bound"] = False
        if not (
            configuration_hashes_complete(baseline_configuration, temporal=False)
            and configuration_hashes_complete(selected_configuration, temporal=True)
            and prediction_hashes_complete(baseline["t0"])
            and prediction_hashes_complete(baseline["history"])
            and prediction_hashes_complete(baseline["raw_doppler"])
            and prediction_hashes_complete(selected["frames"])
        ):
            checks["complete_hash_provenance"] = False
    return checks


def paired_groups(
    baselines: dict[int, dict],
    temporal: dict[int, dict],
    baseline_arm: str,
    path: tuple[str, ...],
    horizon: int | None = None,
) -> dict[int, dict[int, list[tuple[float, float]]]]:
    grouped: dict[int, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed in sorted(FORMAL_SEEDS):
        baseline_frames = baselines[seed][baseline_arm]
        temporal_frames = temporal[seed]["frames"]
        for identity in sorted(set(baseline_frames) & set(temporal_frames)):
            temporal_frame = temporal_frames[identity]
            if (
                horizon is not None
                and int(temporal_frame.get("rollout_step", -1)) != horizon
            ):
                continue
            baseline_value = nested_value(baseline_frames[identity], path)
            temporal_value = nested_value(temporal_frame, path)
            if baseline_value is None or temporal_value is None:
                continue
            grouped[seed][identity[0]].append((baseline_value, temporal_value))
    return grouped


def summarize_groups(
    grouped: dict[int, dict[int, list[tuple[float, float]]]],
    direction: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    seeds = sorted(grouped)
    scenes = sorted({scene for by_scene in grouped.values() for scene in by_scene})
    if seeds != sorted(FORMAL_SEEDS) or not scenes:
        raise ValueError("G4R comparison lacks the complete seed-scene matrix")
    if any(not grouped[seed].get(scene) for scene in scenes for seed in seeds):
        raise ValueError("G4R comparison has a missing seed-scene endpoint")

    def statistic(sampled_scenes, sampled_seeds):
        baseline_values = []
        temporal_values = []
        improvements = []
        for scene in sampled_scenes:
            for seed in sampled_seeds:
                observations = np.asarray(
                    grouped[int(seed)][int(scene)], dtype=np.float64
                )
                baseline_mean = float(observations[:, 0].mean())
                temporal_mean = float(observations[:, 1].mean())
                baseline_values.append(baseline_mean)
                temporal_values.append(temporal_mean)
                improvements.append(
                    baseline_mean - temporal_mean
                    if direction == "lower"
                    else temporal_mean - baseline_mean
                )
        baseline_mean = float(np.mean(baseline_values))
        temporal_mean = float(np.mean(temporal_values))
        relative_change = (temporal_mean - baseline_mean) / max(
            abs(baseline_mean), 1e-12
        )
        retention_ratio = temporal_mean / max(abs(baseline_mean), 1e-12)
        return (
            float(np.mean(improvements)),
            relative_change,
            retention_ratio,
            baseline_mean,
            temporal_mean,
        )

    point = statistic(scenes, seeds)
    bootstrap = np.asarray(
        [
            statistic(
                rng.choice(scenes, size=len(scenes), replace=True),
                rng.choice(seeds, size=len(seeds), replace=True),
            )[:3]
            for _ in range(bootstrap_samples)
        ],
        dtype=np.float64,
    )
    return {
        "direction": direction,
        "baseline_mean": point[3],
        "temporal_mean": point[4],
        "improvement": point[0],
        "improvement_ci95": np.quantile(bootstrap[:, 0], (0.025, 0.975)).tolist(),
        "relative_change": point[1],
        "relative_change_ci95": np.quantile(
            bootstrap[:, 1], (0.025, 0.975)
        ).tolist(),
        "retention_ratio": point[2],
        "retention_ratio_ci95": np.quantile(
            bootstrap[:, 2], (0.025, 0.975)
        ).tolist(),
        "seed_count": len(seeds),
        "scene_count": len(scenes),
        "paired_frame_seed_count": sum(
            len(observations)
            for by_scene in grouped.values()
            for observations in by_scene.values()
        ),
    }


def compare_arm(
    baselines: dict[int, dict],
    temporal: dict[int, dict],
    baseline_arm: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
    horizon: int | None = None,
) -> dict:
    return {
        endpoint: summarize_groups(
            paired_groups(baselines, temporal, baseline_arm, path, horizon),
            direction,
            bootstrap_samples,
            rng,
        )
        for endpoint, (path, direction) in ENDPOINTS.items()
    }


def confidently_better(endpoint: dict) -> bool:
    return float(endpoint["improvement_ci95"][0]) > 0.0


def gate_decision(
    versus_t0: dict,
    versus_history_aggregation: dict,
    step25_versus_t0: dict,
    complete_provenance: bool,
) -> dict:
    checks = {
        "matching_improves_vs_t0": confidently_better(
            versus_t0["ego_aligned_matched_distance_m"]
        ),
        "flicker_improves_vs_t0": confidently_better(
            versus_t0["occupancy_flicker"]
        ),
        "chamfer_improves_vs_history_aggregation": confidently_better(
            versus_history_aggregation["geometry_chamfer_m"]
        ),
        "spectrum_improves_vs_history_aggregation": any(
            confidently_better(versus_history_aggregation[endpoint])
            for endpoint in ("local_spectrum_kl", "circular_w1_mps")
        ),
        "chamfer_nondegradation_vs_t0": versus_t0["geometry_chamfer_m"][
            "relative_change_ci95"
        ][1]
        <= 0.02,
        "local_kl_nondegradation_vs_t0": versus_t0["local_spectrum_kl"][
            "relative_change_ci95"
        ][1]
        <= 0.05,
        "circular_w1_nondegradation_vs_t0": versus_t0["circular_w1_mps"][
            "relative_change_ci95"
        ][1]
        <= 0.05,
        "step25_confidence_retention": step25_versus_t0["confidence_mean"][
            "retention_ratio_ci95"
        ][0]
        >= 0.90,
        "step25_covered_cells_retention": step25_versus_t0[
            "covered_cell_count"
        ]["retention_ratio_ci95"][0]
        >= 0.90,
        "complete_provenance": bool(complete_provenance),
    }
    return {"g4r_passed": all(checks.values()), **checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--temporal-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        raise ValueError("Formal G4R comparison requires at least 1,000 bootstraps")
    if not is_git_commit(args.source_commit):
        raise ValueError("G4R source commit must be a hexadecimal Git commit")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")

    baselines = reports_by_seed(args.baseline_reports, load_baseline, "baseline")
    temporal = reports_by_seed(args.temporal_reports, load_temporal, "temporal")
    validation_checks = validate_matched_reports(
        baselines, temporal, args.source_commit
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    versus_t0 = compare_arm(
        baselines, temporal, "t0", args.bootstrap_samples, rng
    )
    versus_history = compare_arm(
        baselines, temporal, "history", args.bootstrap_samples, rng
    )
    versus_raw_doppler = compare_arm(
        baselines, temporal, "raw_doppler", args.bootstrap_samples, rng
    )
    step25_versus_t0 = compare_arm(
        baselines, temporal, "t0", args.bootstrap_samples, rng, horizon=25
    )
    decision = gate_decision(
        versus_t0,
        versus_history,
        step25_versus_t0,
        complete_provenance=all(validation_checks.values()),
    )
    first_temporal = temporal[min(temporal)]
    report = {
        "schema_version": 1,
        "protocol": COMPARISON_PROTOCOL,
        "source_commit": args.source_commit,
        "seeds": sorted(FORMAL_SEEDS),
        "selected_fusion_mode": first_temporal["configuration"]["fusion_mode"],
        "baseline_reports": [
            {"seed": seed, "path": run["path"], "sha256": run["sha256"]}
            for seed, run in sorted(baselines.items())
        ],
        "temporal_reports": [
            {"seed": seed, "path": run["path"], "sha256": run["sha256"]}
            for seed, run in sorted(temporal.items())
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "gate_thresholds": {
            "maximum_chamfer_relative_degradation_vs_t0": 0.02,
            "maximum_local_kl_or_w1_relative_degradation_vs_t0": 0.05,
            "minimum_step25_confidence_and_coverage_retention": 0.90,
        },
        "validation_checks": validation_checks,
        "comparisons": {
            "selected_vs_t0": versus_t0,
            "selected_vs_history_aggregation": versus_history,
            "selected_vs_raw_doppler_displacement_sensitivity": versus_raw_doppler,
            "selected_vs_t0_step25": step25_versus_t0,
        },
        "decision": decision,
        "completed": True,
    }
    atomic_json(args.output, report)
    print(json.dumps({"decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
