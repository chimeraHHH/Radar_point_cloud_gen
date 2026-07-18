#!/usr/bin/env python3
"""Scene-first paired bootstrap and conservative G4 decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASELINE_PROTOCOL = "g4_temporal_baselines_v1"
TEMPORAL_PROTOCOL = "g4_temporal_strict_rollout_v1"
FORMAL_SEEDS = {20260716, 20260717, 20260718}
ENDPOINTS = {
    "temporal_radial_error_m": (
        ("temporal", "temporal_radial_error_mean_m"),
        "lower",
    ),
    "occupancy_flicker": (("temporal", "occupancy_flicker"), "lower"),
    "geometry_chamfer_m": (("current", "generated_geometry", "chamfer_m"), "lower"),
    "local_spectrum_kl": (("current", "cycle", "local_spectrum_kl"), "lower"),
    "static_pce_median_mps": (
        ("current", "doppler", "static_pce_median_mps"),
        "lower",
    ),
    "confidence_mean": (("current", "cycle", "confidence_mean"), "higher"),
    "covered_cell_count": (("current", "cycle", "covered_cell_count"), "higher"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def nested_value(document: dict, path: tuple[str, ...]) -> float | None:
    current = document
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if not isinstance(current, (int, float)) or not np.isfinite(current):
        return None
    return float(current)


def frame_index(frames: list[dict]) -> dict[tuple[int, int], dict]:
    indexed = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in frames
    }
    if len(indexed) != len(frames):
        raise ValueError("Duplicate G4 frame identities")
    return indexed


def load_baseline(path: Path) -> dict:
    path = path.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != BASELINE_PROTOCOL or report.get("completed") is not True:
        raise ValueError(f"Incomplete or incompatible G4 baseline report: {path}")
    if not report.get("checks") or not all(report["checks"].values()):
        raise ValueError(f"G4 baseline report contains failed checks: {path}")
    required_arms = {"t0_single_frame", "t1_ego_copy", "t2_doppler_copy", "t3_doppdrive"}
    if set(report.get("arms", {})) != required_arms:
        raise ValueError(f"G4 baseline arm matrix is incomplete: {path}")
    config = report["configuration"]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "seed": int(config["parent_seed"]),
        "config": config,
        "t0": frame_index(report["arms"]["t0_single_frame"]["frames"]),
        "t3": frame_index(report["arms"]["t3_doppdrive"]["frames"]),
    }


def load_temporal(path: Path) -> dict:
    path = path.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != TEMPORAL_PROTOCOL or report.get("completed") is not True:
        raise ValueError(f"Incomplete or incompatible G4 rollout report: {path}")
    if not report.get("checks") or not all(report["checks"].values()):
        raise ValueError(f"G4 rollout report contains failed checks: {path}")
    if not report.get("training_checks") or not all(report["training_checks"].values()):
        raise ValueError(f"G4 rollout training checks failed: {path}")
    config = report["configuration"]
    if not report.get("efficiency"):
        raise ValueError(f"G4 rollout efficiency report is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "seed": int(config["seed"]),
        "config": config,
        "frames": frame_index(report["frames"]),
    }


def by_seed(paths: list[Path], loader, label: str) -> dict[int, dict]:
    runs = [loader(path) for path in paths]
    indexed = {run["seed"]: run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError(f"Duplicate seed in G4 {label} reports")
    if set(indexed) != FORMAL_SEEDS:
        raise ValueError(f"G4 {label} seed set is {set(indexed)}, expected {FORMAL_SEEDS}")
    return indexed


def sequence_metric_coverage(frames: dict, path: tuple[str, ...]) -> bool:
    sequences = {key[0] for key in frames}
    return all(
        any(
            nested_value(frame, path) is not None
            for (sequence, _), frame in frames.items()
            if sequence == expected
        )
        for expected in sequences
    )


def stratified_coverage(frames: dict) -> bool:
    sequences = {key[0] for key in frames}
    for sequence in sequences:
        sequence_frames = [
            frame for (current, _), frame in frames.items() if current == sequence
        ]
        if not all(
            nested_value(
                frame, ("current", "stratified_geometry", "target_dynamic_fraction")
            )
            is not None
            for frame in sequence_frames
        ):
            return False
        stratified_keys = {
            key
            for frame in sequence_frames
            for key in frame["current"]["stratified_geometry"]
        }
        if not {
            "static_target_completeness_mean_distance_m",
            "dynamic_target_completeness_mean_distance_m",
        }.issubset(stratified_keys):
            return False
        geometry_keys = {
            key
            for frame in sequence_frames
            for key in frame["current"]["generated_geometry"]
        }
        if not any(key.startswith("range_") for key in geometry_keys):
            return False
    return True


def validate_matched(
    baselines: dict[int, dict], temporal: dict[int, dict]
) -> dict[str, bool]:
    reference_identities = None
    reference_global = None
    reference_fusion = None
    checks = {
        "three_matched_seeds": set(baselines) == set(temporal) == FORMAL_SEEDS,
        "all_reports_cover_384_identical_frames": True,
        "same_frozen_data_and_point_count": True,
        "same_seed_parent_prediction_cache": True,
        "same_selected_temporal_family": True,
        "first_frame_is_the_matched_t0_anchor": True,
        "all_required_metrics_cover_every_sequence": True,
        "dynamic_static_and_distance_strata_present": True,
        "validation_only_and_test_untouched": True,
    }
    for seed in sorted(FORMAL_SEEDS):
        baseline = baselines[seed]
        selected = temporal[seed]
        identities = set(baseline["t0"])
        if (
            len(identities) != 384
            or set(baseline["t3"]) != identities
            or set(selected["frames"]) != identities
        ):
            checks["all_reports_cover_384_identical_frames"] = False
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            checks["all_reports_cover_384_identical_frames"] = False
        global_provenance = (
            baseline["config"]["manifest_sha256"],
            baseline["config"]["scene_split_sha256"],
            baseline["config"]["normalization_sha256"],
            baseline["config"]["dense_cache_report_sha256"],
            int(baseline["config"]["point_count"]),
        )
        selected_global = (
            selected["config"]["manifest_sha256"],
            selected["config"]["scene_split_sha256"],
            selected["config"]["normalization_sha256"],
            selected["config"]["dense_cache_report_sha256"],
            int(selected["config"]["point_count"]),
        )
        if selected_global != global_provenance:
            checks["same_frozen_data_and_point_count"] = False
        if reference_global is None:
            reference_global = global_provenance
        elif global_provenance != reference_global:
            checks["same_frozen_data_and_point_count"] = False
        if (
            baseline["config"]["parent_prediction_manifest_sha256"]
            != selected["config"]["parent_prediction_manifest_sha256"]
            or baseline["config"]["parent_variant"]
            != selected["config"]["parent_variant"]
        ):
            checks["same_seed_parent_prediction_cache"] = False
        fusion = selected["config"]["fusion_mode"]
        if reference_fusion is None:
            reference_fusion = fusion
        elif fusion != reference_fusion:
            checks["same_selected_temporal_family"] = False
        for identity in identities:
            baseline_frame = baseline["t0"][identity]
            selected_frame = selected["frames"][identity]
            if int(selected_frame["rollout_step"]) == 0 and (
                selected_frame["prediction"]["sha256"]
                != baseline_frame["prediction"]["sha256"]
            ):
                checks["first_frame_is_the_matched_t0_anchor"] = False
        for endpoint_path, _ in ENDPOINTS.values():
            if not sequence_metric_coverage(baseline["t0"], endpoint_path):
                checks["all_required_metrics_cover_every_sequence"] = False
            if not sequence_metric_coverage(selected["frames"], endpoint_path):
                checks["all_required_metrics_cover_every_sequence"] = False
        if not stratified_coverage(baseline["t0"]) or not stratified_coverage(
            selected["frames"]
        ):
            checks["dynamic_static_and_distance_strata_present"] = False
        if (
            baseline["config"].get("evaluation_frame_count") != 384
            or baseline["config"].get("partition") != "validation"
            or selected["config"].get("partition") != "validation"
            or selected["config"]["strict_recurrent_rollout"] is not True
        ):
            checks["validation_only_and_test_untouched"] = False
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
        first = baselines[seed][baseline_arm]
        second = temporal[seed]["frames"]
        for identity in sorted(set(first) & set(second)):
            if horizon is not None and int(second[identity]["rollout_step"]) != horizon:
                continue
            first_value = nested_value(first[identity], path)
            second_value = nested_value(second[identity], path)
            if first_value is None or second_value is None:
                continue
            grouped[seed][identity[0]].append((first_value, second_value))
    return grouped


def summarize_groups(
    grouped: dict[int, dict[int, list[tuple[float, float]]]],
    direction: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    seeds = sorted(grouped)
    scenes = sorted({scene for by_scene in grouped.values() for scene in by_scene})
    if seeds != sorted(FORMAL_SEEDS) or len(scenes) != 8:
        raise ValueError("G4 comparison lacks the complete seed x scene matrix")
    if any(not grouped[seed].get(scene) for seed in seeds for scene in scenes):
        raise ValueError("G4 comparison has a missing seed-scene endpoint")

    def statistic(sampled_seeds, sampled_scenes):
        first_values = []
        second_values = []
        improvements = []
        for seed in sampled_seeds:
            for scene in sampled_scenes:
                values = np.asarray(
                    grouped[int(seed)][int(scene)], dtype=np.float64
                )
                first = float(values[:, 0].mean())
                second = float(values[:, 1].mean())
                first_values.append(first)
                second_values.append(second)
                improvements.append(
                    first - second if direction == "lower" else second - first
                )
        first_mean = float(np.mean(first_values))
        second_mean = float(np.mean(second_values))
        relative = (second_mean - first_mean) / max(abs(first_mean), 1e-12)
        ratio = second_mean / max(first_mean, 1e-12)
        return float(np.mean(improvements)), relative, ratio, first_mean, second_mean

    point, relative, ratio, first_mean, second_mean = statistic(seeds, scenes)
    bootstraps = []
    for _ in range(bootstrap_samples):
        bootstraps.append(
            statistic(
                rng.choice(seeds, size=len(seeds), replace=True),
                rng.choice(scenes, size=len(scenes), replace=True),
            )[:3]
        )
    values = np.asarray(bootstraps, dtype=np.float64)
    return {
        "direction": direction,
        "baseline_mean": first_mean,
        "temporal_mean": second_mean,
        "improvement": point,
        "improvement_ci95": np.quantile(values[:, 0], (0.025, 0.975)).tolist(),
        "relative_change": relative,
        "relative_change_ci95": np.quantile(
            values[:, 1], (0.025, 0.975)
        ).tolist(),
        "retention_ratio": ratio,
        "retention_ratio_ci95": np.quantile(
            values[:, 2], (0.025, 0.975)
        ).tolist(),
        "seed_count": len(seeds),
        "scene_count": len(scenes),
        "paired_frame_seed_count": int(
            sum(len(values) for by_scene in grouped.values() for values in by_scene.values())
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


def markdown(report: dict) -> str:
    decision = report["decision"]
    rows = [
        "# G4 Current-Cube Temporal Decision",
        "",
        f"Selected arm: **{report['selected_arm_id']} "
        f"(`{report['selected_fusion_mode']}`)**",
        "",
        f"G4 passed: **{decision['g4_passed']}**",
        "",
        "| Gate condition | Passed |",
        "|---|---:|",
    ]
    for key, value in decision.items():
        if key != "g4_passed":
            rows.append(f"| `{key}` | {value} |")
    rows.extend(
        [
            "",
            "The decision uses paired seed-by-scene bootstrap intervals. A failed "
            "G4 moves temporal fusion to the appendix and does not alter G2/G3.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--temporal-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        raise ValueError("Formal G4 comparison requires at least 1,000 bootstraps")
    for path in (args.output, args.decision_markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}")
    baselines = by_seed(args.baseline_reports, load_baseline, "baseline")
    temporal = by_seed(args.temporal_reports, load_temporal, "temporal")
    validation_checks = validate_matched(baselines, temporal)
    if not all(validation_checks.values()):
        raise ValueError(f"G4 matched-report checks failed: {validation_checks}")
    rng = np.random.default_rng(args.bootstrap_seed)
    vs_t0 = compare_arm(
        baselines, temporal, "t0", args.bootstrap_samples, rng
    )
    vs_t3 = compare_arm(
        baselines, temporal, "t3", args.bootstrap_samples, rng
    )
    rollout25_vs_t0 = compare_arm(
        baselines, temporal, "t0", args.bootstrap_samples, rng, horizon=25
    )
    decision = {
        "temporal_radial_improves_vs_t0": vs_t0[
            "temporal_radial_error_m"
        ]["improvement_ci95"][0]
        > 0.0,
        "occupancy_flicker_improves_vs_t0": vs_t0[
            "occupancy_flicker"
        ]["improvement_ci95"][0]
        > 0.0,
        "geometry_chamfer_improves_vs_t3": vs_t3[
            "geometry_chamfer_m"
        ]["improvement_ci95"][0]
        > 0.0,
        "chamfer_nondegradation_vs_t0": vs_t0[
            "geometry_chamfer_m"
        ]["relative_change_ci95"][1]
        <= 0.02,
        "local_kl_nondegradation_vs_t0": vs_t0[
            "local_spectrum_kl"
        ]["relative_change_ci95"][1]
        <= 0.05,
        "static_pce_nondegradation_vs_t0": vs_t0[
            "static_pce_median_mps"
        ]["relative_change_ci95"][1]
        <= 0.05,
        "step25_confidence_retention": rollout25_vs_t0[
            "confidence_mean"
        ]["retention_ratio_ci95"][0]
        >= 0.90,
        "step25_covered_cell_retention": rollout25_vs_t0[
            "covered_cell_count"
        ]["retention_ratio_ci95"][0]
        >= 0.90,
        "complete_provenance_and_metrics": all(validation_checks.values()),
    }
    decision["g4_passed"] = all(decision.values())
    first_temporal = temporal[min(temporal)]
    report = {
        "schema_version": 1,
        "protocol": "g4_scene_first_paired_bootstrap_v1",
        "source_commit": args.source_commit,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "selected_arm_id": first_temporal["config"]["arm_id"],
        "selected_fusion_mode": first_temporal["config"]["fusion_mode"],
        "baseline_reports": [
            {"seed": seed, "path": run["path"], "sha256": run["sha256"]}
            for seed, run in sorted(baselines.items())
        ],
        "temporal_reports": [
            {"seed": seed, "path": run["path"], "sha256": run["sha256"]}
            for seed, run in sorted(temporal.items())
        ],
        "validation_checks": validation_checks,
        "comparisons": {
            "selected_vs_t0": vs_t0,
            "selected_vs_t3": vs_t3,
            "selected_vs_t0_step25": rollout25_vs_t0,
        },
        "decision": decision,
        "completed": True,
    }
    atomic_text(args.output, json.dumps(report, indent=2) + "\n")
    atomic_text(args.decision_markdown, markdown(report))
    print(json.dumps({"decision": decision}, indent=2))


if __name__ == "__main__":
    main()
