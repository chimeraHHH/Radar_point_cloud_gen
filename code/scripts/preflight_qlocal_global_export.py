#!/usr/bin/env python3
"""Run the source-bound Q-Local-F0R global-export capacity gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import geometry_report, nearest_distance  # noqa: E402
from eval.rald_wce_stage0 import (  # noqa: E402
    CAPACITY_DISTANCE_M,
    FORMAL_OUTPUT_RANGE_QUOTAS,
    ExactExportCapacityError,
    exact_capacity_export,
    global_exact_capacity_export,
    tensor_sha256,
)
from losses.rald_wce_quality import continuous_geometry_quality_target  # noqa: E402
from scripts.preflight_qlocal_distributional_risk import (  # noqa: E402
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_NORMALIZATION_SHA256,
    EXPECTED_SCENE_SPLIT_SHA256,
    FIT_IDENTITIES,
    FRESH_PARENT_CHECKPOINT_SHA256,
    FRESH_PARENT_MANIFEST_SHA256,
    FRESH_PARENT_METRICS_SHA256,
    FRESH_PARENT_PROTOCOL,
    FRESH_PARENT_SOURCE_COMMIT,
    UNSEEN_IDENTITIES,
    WRONG_IDENTITY_MAP,
    build_inference_config,
    cache_path,
    canonical_digest,
    cube_path,
    frame_identity,
    git_output,
    require_h200_with_identity,
    resolve_frozen_cohort,
    sha256_file,
    verify_source_tree,
)
from scripts.train_rald_wce_pilot import (  # noqa: E402
    PilotCubeDataset,
    load_normalization,
    validate_frozen_inputs,
)
from scripts.train_rald_wce_quality_tiny import (  # noqa: E402
    build_formal_base,
    frozen_candidate_field,
    state_dict_sha256,
)


PROTOCOL = "g1_qlocal_f0r_global_export_capacity_v1"
ARCHIVED_QLOCAL_SOURCE_COMMIT = "bf1a166c48be60f82c5283b5937edf0cf6ad64c2"
ARCHIVED_QLOCAL_SHA256 = (
    "b1b740dff2515df274dfbf6fdb15ece53c1c3129d7847c6e803abc568c97fef4"
)
ORACLE_CHAMFER_GATE_M = 0.8
ORACLE_OUTLIER_GATE = 0.05
RETENTION_TOLERANCE = 1e-6
RANGE_INTERVALS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
FROZEN_QLOCAL_SCORER_PATHS = (
    "code/losses/qlocal_distributional_risk.py",
    "code/losses/rald_wce_quality.py",
    "code/models/qlocal_distributional_risk.py",
)


def source_hashes(repo: Path) -> dict[str, str]:
    paths = (
        "code/eval/rald_wce_stage0.py",
        "code/eval/dense_geometry.py",
        "code/losses/qlocal_distributional_risk.py",
        "code/losses/rald_wce_quality.py",
        "code/models/qlocal_distributional_risk.py",
        "code/scripts/preflight_qlocal_distributional_risk.py",
        "code/scripts/preflight_qlocal_global_export.py",
        "code/scripts/train_rald_wce_pilot.py",
        "code/scripts/train_rald_wce_quality_tiny.py",
        "docs/qlocal_f0r_global_export_capacity_protocol.md",
    )
    return {path: sha256_file(repo / path) for path in paths}


def frozen_qlocal_scorer_blob_ids(repo: Path) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for path in FROZEN_QLOCAL_SCORER_PATHS:
        archived = git_output(
            repo,
            "rev-parse",
            f"{ARCHIVED_QLOCAL_SOURCE_COMMIT}:{path}",
        )
        current = git_output(repo, "hash-object", path)
        unchanged = current == archived
        report[path] = {
            "archived_blob_id": archived,
            "current_blob_id": current,
            "unchanged": unchanged,
        }
        if not unchanged:
            raise ValueError(f"Q-Local-F0R changed frozen scorer source: {path}")
    return report


def atomic_commit_report(output_dir: Path, report: dict[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Q-Local-F0R output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        path = staging / "preflight.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "preflight.json"


def global_structural_checks(export: Any) -> dict[str, bool]:
    report = export.report
    return {
        "exact_10000": int(report["exact_point_count"]) == 10_000,
        "minimum_spacing_5cm": float(
            report["observed_minimum_pair_distance_m"]
        )
        >= CAPACITY_DISTANCE_M - 1e-6,
        "unique_rows": int(report["unique_selected_candidate_count"]) == 10_000,
        "no_range_quotas": report["range_quotas_enforced"] is False,
        "no_copy_padding_jitter_duplicate": report[
            "copy_padding_jitter_duplicate"
        ]
        is False,
        "exporter_api_target_free": report["ground_truth_accessed"] is False,
    }


def target_stratum_retention(
    prediction_xyz_m: torch.Tensor,
    target_xyz_confidence: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    target_xyz = target_xyz_confidence[:, :3]
    target_weight = target_xyz_confidence[:, 3].clamp_min(0.0)
    distance = nearest_distance(target_xyz, prediction_xyz_m, chunk_size=1_024)
    radius = torch.linalg.vector_norm(target_xyz, dim=1)
    report: dict[str, dict[str, float | int]] = {}
    for lower, upper in RANGE_INTERVALS_M:
        mask = (radius >= lower) & (radius < upper)
        if not bool(mask.any()):
            continue
        weight = target_weight[mask]
        effective_count = weight.sum()
        if float(effective_count.item()) <= 0.0:
            raise ValueError("Q-Local-F0R target stratum has zero effective weight")
        weight_sum = effective_count.clamp_min(1e-8)
        values = distance[mask]
        key = f"range_{int(lower)}_{int(upper)}m"
        report[key] = {
            "target_count": int(mask.sum().item()),
            "target_effective_count": float(weight.sum().item()),
            "completeness_mean_distance_m": float(
                ((values * weight).sum() / weight_sum).item()
            ),
            "recall_1m": float(
                (((values <= 1.0).to(weight) * weight).sum() / weight_sum).item()
            ),
        }
    return report


def retention_gate(
    fixed: dict[str, dict[str, float | int]],
    global_report: dict[str, dict[str, float | int]],
) -> tuple[dict[str, dict[str, Any]], bool]:
    if set(fixed) != set(global_report):
        raise ValueError("Q-Local-F0R target-bearing strata changed")
    checks: dict[str, dict[str, Any]] = {}
    for key in sorted(fixed):
        fixed_row = fixed[key]
        global_row = global_report[key]
        if fixed_row["target_count"] != global_row["target_count"]:
            raise ValueError("Q-Local-F0R target stratum count changed")
        if not np.isclose(
            float(fixed_row["target_effective_count"]),
            float(global_row["target_effective_count"]),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("Q-Local-F0R target stratum weight changed")
        completeness_ok = float(
            global_row["completeness_mean_distance_m"]
        ) <= float(fixed_row["completeness_mean_distance_m"]) + RETENTION_TOLERANCE
        recall_ok = float(global_row["recall_1m"]) >= float(
            fixed_row["recall_1m"]
        ) - RETENTION_TOLERANCE
        checks[key] = {
            "fixed": fixed_row,
            "global": global_row,
            "global_completeness_not_worse": completeness_ok,
            "global_recall_1m_not_worse": recall_ok,
            "passed": completeness_ok and recall_ok,
        }
    return checks, all(row["passed"] for row in checks.values())


def archived_frame_map(document: dict[str, Any]) -> dict[tuple[int, int], dict]:
    if document.get("protocol") != "g1_qlocal_distributional_risk_f0_preflight_v1":
        raise ValueError("Q-Local-F0R archived report protocol changed")
    if document.get("source_commit") != ARCHIVED_QLOCAL_SOURCE_COMMIT:
        raise ValueError("Q-Local-F0R archived source changed")
    if document.get("status") != "qlocal_f0_capacity_no_go":
        raise ValueError("Q-Local-F0R archived terminal status changed")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != 12:
        raise ValueError("Q-Local-F0R archived report needs 12 frames")
    mapping = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in frames
    }
    if len(mapping) != 12:
        raise ValueError("Q-Local-F0R archived identities are not unique")
    return mapping


def verify_candidate_reconstruction(
    report: dict[str, Any],
    field: dict[str, Any],
    target: torch.Tensor,
    distance: torch.Tensor,
) -> dict[str, bool]:
    archived_candidate = report["candidate_field"]
    observed_candidate = field["report"]
    checks = {
        "candidate_xyz_sha256": observed_candidate["candidate_xyz_sha256"]
        == archived_candidate["candidate_xyz_sha256"],
        "base_confidence_sha256": observed_candidate[
            "base_confidence_sha256"
        ]
        == archived_candidate["base_confidence_sha256"],
        "q0_sha256": observed_candidate["q0_sha256"]
        == archived_candidate["q0_sha256"],
        "q1_sha256": observed_candidate["q1_sha256"]
        == archived_candidate["q1_sha256"],
        "target_tensor_sha256": tensor_sha256(target)
        == report["raw_inputs"]["target_tensor_sha256"],
        "nearest_distance_sha256": tensor_sha256(distance)
        == report["nearest_distance_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"Q-Local-F0R candidate reconstruction changed: {checks}")
    return checks


def frame_report(
    *,
    dataset: PilotCubeDataset,
    index: int,
    archived: dict[str, Any],
    base_model: torch.nn.Module,
    axes_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inference_config: Any,
    device: torch.device,
    data_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    item = dataset[index]
    record = dataset.records[index]
    cube = item["cube_drae"].unsqueeze(0).to(device)
    range_m, azimuth_rad, elevation_rad = axes_tensors
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        field = frozen_candidate_field(
            base_model,
            cube,
            range_m,
            azimuth_rad,
            elevation_rad,
            inference_config,
        )
    target = item["target_xyz_confidence"].to(device).float()
    distance = continuous_geometry_quality_target(
        field["xyz_m"].float(),
        target,
        distance_temperature_m=1.0,
        candidate_chunk_size=4_096,
    )["nearest_distance_m"]
    reconstruction = verify_candidate_reconstruction(
        archived,
        field,
        target,
        distance,
    )

    try:
        fixed_oracle = exact_capacity_export(
            field["xyz_m"],
            -distance,
            output_quotas=FORMAL_OUTPUT_RANGE_QUOTAS,
            minimum_distance_m=CAPACITY_DISTANCE_M,
        )
    except ExactExportCapacityError as error:
        raise ValueError(
            "Q-Local-F0R archived fixed oracle no longer has capacity"
        ) from error
    fixed_hash_checks = {
        key: fixed_oracle.hashes[key]
        == archived["gt_nearest_capacity_oracle"]["hashes"][key]
        for key in fixed_oracle.hashes
    }
    if not all(fixed_hash_checks.values()):
        raise ValueError("Q-Local-F0R fixed oracle did not reproduce")

    try:
        global_base = global_exact_capacity_export(
            field["xyz_m"],
            field["base_confidence"][0].float(),
            minimum_distance_m=CAPACITY_DISTANCE_M,
        )
    except ExactExportCapacityError as error:
        global_base_report: dict[str, Any] = {
            "descriptive_only": True,
            "capacity_error": error.report,
        }
    else:
        base_geometry = geometry_report(
            global_base.xyz_m.to(device),
            target[:, :3],
            target_weight=target[:, 3],
        )
        global_base_report = {
            "descriptive_only": True,
            "geometry": base_geometry,
            "export": global_base.report,
            "hashes": global_base.hashes,
            "structural_checks": global_structural_checks(global_base),
        }
    try:
        global_oracle = global_exact_capacity_export(
            field["xyz_m"],
            -distance,
            minimum_distance_m=CAPACITY_DISTANCE_M,
        )
    except ExactExportCapacityError as error:
        raise ExactExportCapacityError(
            str(error),
            {
                **error.report,
                "oracle_role": "global_gt_nearest_capacity",
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
            },
        ) from error
    oracle_geometry = geometry_report(
        global_oracle.xyz_m.to(device),
        target[:, :3],
        target_weight=target[:, 3],
    )
    fixed_retention = target_stratum_retention(
        fixed_oracle.xyz_m.to(device),
        target,
    )
    global_retention = target_stratum_retention(
        global_oracle.xyz_m.to(device),
        target,
    )
    retention, retention_passed = retention_gate(
        fixed_retention,
        global_retention,
    )
    oracle_checks = global_structural_checks(global_oracle)
    gate = {
        "chamfer_at_most_0p8m": float(oracle_geometry["chamfer_m"])
        <= ORACLE_CHAMFER_GATE_M,
        "outlier_at_most_5pct": float(oracle_geometry["outlier_fraction_2m"])
        <= ORACLE_OUTLIER_GATE,
        "all_structural_checks": all(oracle_checks.values()),
        "all_target_strata_retained": retention_passed,
    }
    return {
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
        "partition": str(item["partition"]),
        "raw_inputs": {
            "cube_path": str(cube_path(data_root, record).resolve()),
            "cube_sha256": sha256_file(cube_path(data_root, record)),
            "target_cache_path": str(cache_path(cache_root, record).resolve()),
            "target_cache_sha256": sha256_file(cache_path(cache_root, record)),
            "target_tensor_sha256": tensor_sha256(target),
        },
        "candidate_field": field["report"],
        "candidate_reconstruction_checks": reconstruction,
        "fixed_oracle_reproduction_checks": fixed_hash_checks,
        "global_base_confidence": global_base_report,
        "global_gt_nearest_capacity_oracle": {
            "non_deployable": True,
            "target_used_for_selection": True,
            "geometry": oracle_geometry,
            "export": global_oracle.report,
            "hashes": global_oracle.hashes,
            "structural_checks": oracle_checks,
            "target_stratum_retention": retention,
            "gate": gate,
            "passed": all(gate.values()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--fresh-parent-checkpoint", type=Path, required=True)
    parser.add_argument("--fresh-parent-metrics", type=Path, required=True)
    parser.add_argument("--fresh-parent-run-manifest", type=Path, required=True)
    parser.add_argument("--archived-qlocal-preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def terminal_status(
    *,
    all_frames_passed: bool,
    parent_unchanged: bool,
) -> str:
    if not parent_unchanged:
        return "qlocal_f0r_implementation_invalid"
    if all_frames_passed:
        return "qlocal_f0r_capacity_passed"
    return "qlocal_f0r_capacity_no_go"


def main() -> None:
    args = build_parser().parse_args()
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    frozen_scorer_blobs = frozen_qlocal_scorer_blob_ids(repo)
    device, gpu = require_h200_with_identity(args.device)
    torch.manual_seed(20260716)
    np.random.seed(20260716)

    archived_hash = sha256_file(args.archived_qlocal_preflight)
    if archived_hash != ARCHIVED_QLOCAL_SHA256:
        raise ValueError("Q-Local-F0R archived report hash changed")
    archived_document = json.loads(
        args.archived_qlocal_preflight.read_text(encoding="utf-8")
    )
    archived_frames = archived_frame_map(archived_document)

    input_hashes, manifest_counts = validate_frozen_inputs(
        args.manifest,
        args.scene_split,
        args.normalization,
        repo,
    )
    expected_inputs = {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "scene_split_sha256": EXPECTED_SCENE_SPLIT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
    }
    for key, expected in expected_inputs.items():
        if input_hashes[key] != expected:
            raise ValueError(f"Q-Local-F0R frozen {key} hash changed")

    checkpoint_hash = sha256_file(args.fresh_parent_checkpoint)
    metrics_hash = sha256_file(args.fresh_parent_metrics)
    parent_manifest_hash = sha256_file(args.fresh_parent_run_manifest)
    if checkpoint_hash != FRESH_PARENT_CHECKPOINT_SHA256:
        raise ValueError("Q-Local-F0R checkpoint hash changed")
    if metrics_hash != FRESH_PARENT_METRICS_SHA256:
        raise ValueError("Q-Local-F0R metrics hash changed")
    if parent_manifest_hash != FRESH_PARENT_MANIFEST_SHA256:
        raise ValueError("Q-Local-F0R parent manifest hash changed")
    checkpoint = torch.load(
        args.fresh_parent_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("protocol") != FRESH_PARENT_PROTOCOL:
        raise ValueError("Q-Local-F0R parent protocol changed")
    if checkpoint.get("source_commit") != FRESH_PARENT_SOURCE_COMMIT:
        raise ValueError("Q-Local-F0R parent source changed")
    if int(checkpoint.get("epoch", -1)) != 20:
        raise ValueError("Q-Local-F0R parent must be epoch 20")
    parent_manifest = json.loads(
        args.fresh_parent_run_manifest.read_text(encoding="utf-8")
    )
    if parent_manifest.get("source_commit") != FRESH_PARENT_SOURCE_COMMIT:
        raise ValueError("Q-Local-F0R parent manifest source changed")
    if parent_manifest.get("input_hashes") != {
        "manifest": EXPECTED_MANIFEST_SHA256,
        "scene_split": EXPECTED_SCENE_SPLIT_SHA256,
        "normalization": EXPECTED_NORMALIZATION_SHA256,
        "corrected_dense_geometry": input_hashes[
            "dense_geometry_evaluator_sha256"
        ],
    }:
        raise ValueError("Q-Local-F0R parent data binding changed")

    dataset = PilotCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        "train",
    )
    fit_indices, unseen_indices, wrong_indices = resolve_frozen_cohort(
        dataset.records
    )
    cohort_indices = fit_indices + unseen_indices
    cohort_binding = []
    for index in cohort_indices:
        record = dataset.records[index]
        cube = cube_path(args.data_root, record)
        cache = cache_path(args.cache_root, record)
        cohort_binding.append(
            {
                "identity": list(frame_identity(record)),
                "group": "fit" if index in fit_indices else "unseen_train",
                "cube_path": str(cube.resolve()),
                "cube_sha256": sha256_file(cube),
                "target_cache_path": str(cache.resolve()),
                "target_cache_sha256": sha256_file(cache),
                "wrong_identity": list(
                    frame_identity(dataset.records[wrong_indices[index]])
                ),
            }
        )
    cohort_digest = canonical_digest(cohort_binding)
    if cohort_digest != archived_document["input_binding"][
        "cohort_ordered_digest_sha256"
    ]:
        raise ValueError("Q-Local-F0R cohort byte binding changed")

    log_center, log_scale = load_normalization(args.normalization)
    axes = load_axes(args.data_root / "resources")
    axes_tensors = (
        torch.as_tensor(axes.range_m, dtype=torch.float32, device=device),
        torch.as_tensor(axes.azimuth_rad, dtype=torch.float32, device=device),
        torch.as_tensor(axes.elevation_rad, dtype=torch.float32, device=device),
    )
    base_model = build_formal_base(
        checkpoint,
        log_center=log_center,
        log_scale=log_scale,
        device=device,
    )
    parent_state_before = state_dict_sha256(base_model)
    inference_config = build_inference_config()
    reports: list[dict[str, Any]] = []
    capacity_error: dict[str, Any] | None = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for position, index in enumerate(cohort_indices):
            identity = frame_identity(dataset.records[index])
            report = frame_report(
                dataset=dataset,
                index=index,
                archived=archived_frames[identity],
                base_model=base_model,
                axes_tensors=axes_tensors,
                inference_config=inference_config,
                device=device,
                data_root=args.data_root,
                cache_root=args.cache_root,
            )
            report["group"] = "fit" if index in fit_indices else "unseen_train"
            report["cohort_position"] = position
            reports.append(report)
            torch.cuda.empty_cache()
    except ExactExportCapacityError as error:
        capacity_error = error.report

    parent_state_after = state_dict_sha256(base_model)
    parent_unchanged = parent_state_after == parent_state_before
    all_frames_passed = (
        capacity_error is None
        and len(reports) == len(cohort_indices)
        and all(
            frame["global_gt_nearest_capacity_oracle"]["passed"]
            for frame in reports
        )
    )
    status = terminal_status(
        all_frames_passed=all_frames_passed,
        parent_unchanged=parent_unchanged,
    )
    passed = status == "qlocal_f0r_capacity_passed"
    scientific_no_go = status == "qlocal_f0r_capacity_no_go"
    elapsed = time.monotonic() - started
    terminal = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "status": status,
        "passed": passed,
        "decision": {
            "global_export_training_authorized": passed,
            "all_12_global_oracle_frames_passed": all_frames_passed,
            "fresh_parent_unchanged": parent_unchanged,
            "implementation_valid": parent_unchanged,
            "capacity_error": capacity_error,
            "ray_hazard_authorized_if_no_go": scientific_no_go,
        },
        "source_hashes": source_hashes(repo),
        "input_binding": {
            "frozen_inputs": input_hashes,
            "manifest_counts": manifest_counts,
            "fresh_parent_checkpoint_sha256": checkpoint_hash,
            "fresh_parent_metrics_sha256": metrics_hash,
            "fresh_parent_run_manifest_sha256": parent_manifest_hash,
            "archived_qlocal_preflight_sha256": archived_hash,
            "cohort_ordered_digest_sha256": cohort_digest,
            "cohort": cohort_binding,
            "frozen_qlocal_scorer_blob_ids": frozen_scorer_blobs,
        },
        "fresh_parent": {
            "name": "Fresh-WCE-20",
            "source_commit": FRESH_PARENT_SOURCE_COMMIT,
            "epoch": 20,
            "state_sha256_before": parent_state_before,
            "state_sha256_after": parent_state_after,
            "unchanged": parent_unchanged,
        },
        "cohort_contract": {
            "fit_identities": [list(identity) for identity in FIT_IDENTITIES],
            "unseen_train_identities": [
                list(identity) for identity in UNSEEN_IDENTITIES
            ],
            "wrong_identity_map": {
                f"{source[0]}:{source[1]}": list(wrong)
                for source, wrong in WRONG_IDENTITY_MAP.items()
            },
            "validation_accessed": False,
            "test_accessed": False,
            "future_accessed": False,
        },
        "oracle_gate": {
            "non_deployable": True,
            "per_frame_chamfer_m_at_most": ORACLE_CHAMFER_GATE_M,
            "per_frame_outlier_fraction_at_most": ORACLE_OUTLIER_GATE,
            "exact_output_count": 10_000,
            "range_quotas_enforced": False,
            "minimum_spacing_m": CAPACITY_DISTANCE_M,
            "target_stratum_retention_tolerance": RETENTION_TOLERANCE,
        },
        "frames": reports,
        "runtime": {
            "gpu": gpu,
            "device_argument": args.device,
            "torch_version": torch.__version__,
            "elapsed_seconds": elapsed,
            "measured_gpu_seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "evidence_boundary": {
            "target_used_only_for_non_deployable_capacity_selection_and_metrics": True,
            "deployable_exporter_target_input": False,
            "model_trained": False,
            "range_quota_used": False,
            "validation_accessed": False,
            "test_accessed": False,
            "future_cube_accessed": False,
            "best_of_k": False,
        },
    }
    output = atomic_commit_report(args.output_dir, terminal)
    print(
        json.dumps(
            {
                "status": status,
                "passed": passed,
                "output": str(output),
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
