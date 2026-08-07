#!/usr/bin/env python3
"""Certify a source-equivalent R-A1 replay as a Q1-R tiny parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.dense_geometry import aggregate_geometry_reports  # noqa: E402
from eval.rald_wce_failure_factors import (  # noqa: E402
    aggregate_failure_factor_frames,
    diagnostic_artifact_label,
)
from scripts.train_rald_wce_stage0 import stage0_decision  # noqa: E402


PROTOCOL = "g1_q1r_replay_parent_certificate_v1"
FORMAL_PROTOCOL = "g1_ra1_rald_wce_stage0_v1"
DIAGNOSIS_PROTOCOL = "g1_ra1_wce_frozen_candidate_failure_factors_v1"
FORMAL_SOURCE_COMMIT = "f2a9489d40323d1ef45d85de958f4aea8126e1c8"
ORIGINAL_CHECKPOINT_SHA256 = (
    "5be30e0f1ca23ea3b603abb0f5e330efd3599167362a8e23ab3a5967c411a2a0"
)
ARCHIVED_METRICS_SHA256 = (
    "ae5bca6273fd3e8c18672b5e3704536a43bb897b81b389f07862ead7922df5a6"
)
ARCHIVED_RUN_MANIFEST_SHA256 = (
    "8651214e62eaa03d41cf5116a5a6a8605b9ad506dd3b30a24c7e961c232ee68a"
)
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FORMAL_SOURCE_SUFFIXES = (
    "code/models/rald_wce_field.py",
    "code/losses/rald_wce.py",
    "code/eval/rald_wce_stage0.py",
    "code/eval/dense_geometry.py",
    "code/scripts/train_rald_wce_stage0.py",
    "code/cube_dense/dataset.py",
)
DIAGNOSTIC_SOURCE_RELATIVE_PATHS = (
    "code/scripts/diagnose_rald_wce_failure_factors.py",
    "code/eval/rald_wce_failure_factors.py",
    "code/eval/rald_wce_stage0.py",
    "code/eval/dense_geometry.py",
    "code/eval/g1a_wide_support.py",
    "code/models/rald_wce_field.py",
    "code/cube_dense/kradar.py",
)
EXPECTED_H200_DEVICE_NAME = "NVIDIA H200 NVL"
VALUE_TOLERANCES = {
    "chamfer_mean_m": 0.02,
    "completeness_median_m": 0.02,
    "outlier_fraction_mean": 0.002,
    "far_completeness_mean_m": 0.10,
    "condition_wrong_minus_matched_chamfer_fraction_mean": 0.01,
    "matched_condition_win_fraction": 1.0 / 24.0 + 1e-12,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Q1-R parent certificate exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("Q1-R certifier source must be a full lowercase SHA")
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repo,
        text=True,
    ).strip()
    if head != source_commit or dirty:
        raise ValueError("Q1-R certifier requires its clean source-bound snapshot")


def _json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Q1-R expected a JSON object: {path}")
    return document


def _source_hash_by_suffix(document: dict[str, Any], suffix: str) -> str:
    normalized = "/" + suffix.replace("\\", "/")
    matches = [
        value
        for path, value in document.items()
        if str(path).replace("\\", "/").endswith(normalized)
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError(f"Q1-R manifest lacks one source hash for {suffix}")
    return matches[0]


def _json_normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_normalize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def _aggregate_scalar_reports(
    reports: list[dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    if not reports:
        raise ValueError("Q1-R cannot aggregate an empty scalar report list")
    keys = sorted(reports[0])
    if any(sorted(report) != keys for report in reports):
        raise ValueError("Q1-R scalar report keys differ")
    return {
        key: {
            "mean": float(np.mean([report[key] for report in reports])),
            "median": float(np.median([report[key] for report in reports])),
            "std": float(np.std([report[key] for report in reports])),
            "sample_count": len(reports),
        }
        for key in keys
    }


def _derived_condition_intervention(frame: dict[str, Any]) -> dict[str, float]:
    matched = float(frame["matched"]["chamfer_m"])
    wrong = float(frame["wrong_condition"]["chamfer_m"])
    if not math.isfinite(matched) or not math.isfinite(wrong) or matched <= 0.0:
        raise ValueError("Q1-R formal frame has invalid Chamfer values")
    return {
        "wrong_minus_matched_chamfer_fraction": wrong / matched - 1.0,
        "matched_chamfer_better": float(matched < wrong),
    }


def _recompute_formal_metrics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        raise ValueError("Q1-R formal metrics cannot have an empty frame list")
    minimum_pair_distance = min(
        float(frame["matched_export"]["observed_minimum_pair_distance_m"])
        for frame in frames
    )
    return {
        "frame_count": len(frames),
        "scene_count": len({int(frame["sequence"]) for frame in frames}),
        "far_target_frame_count": sum(
            bool(frame["has_far_target"]) for frame in frames
        ),
        "frames": frames,
        "matched": aggregate_geometry_reports(
            [frame["matched"] for frame in frames]
        ),
        "wrong_condition": aggregate_geometry_reports(
            [frame["wrong_condition"] for frame in frames]
        ),
        "condition_intervention": _aggregate_scalar_reports(
            [_derived_condition_intervention(frame) for frame in frames]
        ),
        "exact_export": {
            "point_count_per_frame": 10_000,
            "minimum_observed_pair_distance_m": minimum_pair_distance,
            "all_matched_exports_exact_10000": all(
                frame["matched_export"]["exact_point_count"] == 10_000
                for frame in frames
            ),
            "all_wrong_exports_exact_10000": all(
                frame["wrong_export"]["exact_point_count"] == 10_000
                for frame in frames
            ),
            "copy_padding_jitter_duplicate": any(
                frame[arm]["copy_padding_jitter_duplicate"] is not False
                for frame in frames
                for arm in ("matched_export", "wrong_export")
            ),
        },
    }


def _recompute_formal_document(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    stored_metrics = document.get("metrics")
    stored_decision = document.get("decision")
    if not isinstance(stored_metrics, dict) or not isinstance(
        stored_decision, dict
    ):
        raise ValueError("Q1-R formal document lacks metrics or decision")
    frames = stored_metrics.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Q1-R formal document lacks frame reports")
    recomputed_metrics = _recompute_formal_metrics(frames)
    frame_interventions_consistent = all(
        _json_normalize(frame.get("condition_intervention"))
        == _json_normalize(_derived_condition_intervention(frame))
        for frame in frames
    )
    stored_values = stored_decision.get("values")
    if not isinstance(stored_values, dict):
        raise ValueError("Q1-R formal decision lacks values")
    peak_allocated_gib = float(stored_values["peak_allocated_gib"])
    peak_reserved_gib = float(stored_values["peak_reserved_gib"])
    if not math.isfinite(peak_allocated_gib) or not math.isfinite(
        peak_reserved_gib
    ):
        raise FloatingPointError("Q1-R formal memory values are non-finite")
    recomputed_decision = stage0_decision(
        recomputed_metrics,
        peak_allocated_bytes=round(peak_allocated_gib * 2**30),
        peak_reserved_bytes=round(peak_reserved_gib * 2**30),
        formal=True,
    )
    aggregate_keys = (
        "frame_count",
        "scene_count",
        "far_target_frame_count",
        "matched",
        "wrong_condition",
        "condition_intervention",
        "exact_export",
    )
    metrics_consistent = all(
        _json_normalize(stored_metrics.get(key))
        == _json_normalize(recomputed_metrics.get(key))
        for key in aggregate_keys
    )
    decision_consistent = _json_normalize(stored_decision) == _json_normalize(
        recomputed_decision
    )
    return recomputed_metrics, recomputed_decision, (
        metrics_consistent
        and frame_interventions_consistent
        and decision_consistent
    )


def _query_hash(frame: dict[str, Any], key: str) -> str:
    value = frame.get("inference", {}).get("query_hashes", {}).get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Q1-R formal frame lacks query hash {key}")
    return value


def _diagnosis_query_hash(frame: dict[str, Any], key: str) -> str:
    value = frame.get("candidate_pool", {}).get("query_hashes", {}).get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Q1-R diagnosis frame lacks query hash {key}")
    return value


def _diagnosis_control_passed(
    frame: dict[str, Any],
    replay_frame: dict[str, Any],
) -> bool:
    control = frame.get("current_confidence", {}).get("formal_hash_control", {})
    export_checks = control.get("export_hash_checks", {})
    query_checks = control.get("query_hash_checks", {})
    replay_export_hashes = replay_frame.get("matched_hashes")
    replay_query_hashes = replay_frame.get("inference", {}).get("query_hashes")
    actual_export_hashes = control.get("actual_export_hashes")
    expected_export_hashes = control.get("expected_export_hashes")
    actual_query_hashes = control.get("actual_query_hashes")
    expected_query_hashes = control.get("expected_query_hashes")
    export_hashes_match = (
        isinstance(replay_export_hashes, dict)
        and actual_export_hashes == replay_export_hashes
        and expected_export_hashes == replay_export_hashes
    )
    query_hashes_match = (
        isinstance(replay_query_hashes, dict)
        and actual_query_hashes == replay_query_hashes
        and expected_query_hashes == replay_query_hashes
    )
    expected_export_checks = {
        key: export_hashes_match
        for key in (
            "xyz_sha256",
            "confidence_sha256",
            "selected_candidate_rows_sha256",
        )
    }
    expected_query_checks = {
        key: query_hashes_match
        for key in (
            "q0_normalized_rae_sha256",
            "q1_normalized_rae_sha256",
        )
    }
    current = frame.get("current_confidence", {})
    return (
        control.get("passed") is True
        and control.get("bit_exact") is True
        and export_checks == expected_export_checks
        and query_checks == expected_query_checks
        and export_hashes_match
        and query_hashes_match
        and _json_normalize(current.get("geometry"))
        == _json_normalize(replay_frame.get("matched"))
        and _json_normalize(current.get("export_report"))
        == _json_normalize(replay_frame.get("matched_export"))
    )


def _frame_identity(frame: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(frame["sequence"]),
        int(frame["radar_index"]),
        int(frame["wrong_condition_sequence"]),
        int(frame["wrong_condition_radar_index"]),
    )


def certify_replay_parent(
    *,
    archived_metrics_path: Path,
    archived_manifest_path: Path,
    replay_checkpoint_path: Path,
    replay_metrics_path: Path,
    replay_manifest_path: Path,
    replay_diagnosis_path: Path,
    certifier_source_commit: str,
) -> dict[str, Any]:
    archived_metrics = _json(archived_metrics_path)
    archived_manifest = _json(archived_manifest_path)
    replay_metrics = _json(replay_metrics_path)
    replay_manifest = _json(replay_manifest_path)
    diagnosis = _json(replay_diagnosis_path)
    checkpoint_sha = sha256_file(replay_checkpoint_path)
    checkpoint = torch.load(
        replay_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not SOURCE_PATTERN.fullmatch(certifier_source_commit):
        raise ValueError("Q1-R certifier source commit must be a full SHA")
    archived_sources = archived_manifest.get("source_hashes")
    replay_sources = replay_manifest.get("source_hashes")
    if not isinstance(archived_sources, dict) or not isinstance(
        replay_sources, dict
    ):
        raise ValueError("Q1-R formal manifests lack source-hash maps")
    source_comparison = {
        suffix: {
            "archived_sha256": _source_hash_by_suffix(
                archived_sources, suffix
            ),
            "replay_sha256": _source_hash_by_suffix(replay_sources, suffix),
        }
        for suffix in FORMAL_SOURCE_SUFFIXES
    }
    source_hashes_match = all(
        row["archived_sha256"] == row["replay_sha256"]
        for row in source_comparison.values()
    )

    (
        archived_recomputed_metrics,
        archived_recomputed_decision,
        archived_formal_consistent,
    ) = _recompute_formal_document(archived_metrics)
    (
        replay_recomputed_metrics,
        replay_recomputed_decision,
        replay_formal_consistent,
    ) = _recompute_formal_document(replay_metrics)
    archived_frames = archived_recomputed_metrics["frames"]
    replay_frames = replay_recomputed_metrics["frames"]
    if not isinstance(archived_frames, list) or not isinstance(
        replay_frames, list
    ):
        raise ValueError("Q1-R formal metrics lack frame lists")
    archived_identities = [_frame_identity(frame) for frame in archived_frames]
    replay_identities = [_frame_identity(frame) for frame in replay_frames]
    q0_hash_matches = sum(
        _query_hash(archived, "q0_normalized_rae_sha256")
        == _query_hash(replay, "q0_normalized_rae_sha256")
        for archived, replay in zip(
            archived_frames,
            replay_frames,
            strict=False,
        )
    )
    q1_hash_matches = sum(
        _query_hash(archived, "q1_normalized_rae_sha256")
        == _query_hash(replay, "q1_normalized_rae_sha256")
        for archived, replay in zip(
            archived_frames,
            replay_frames,
            strict=False,
        )
    )
    inference_contracts_match = (
        len(archived_frames) == len(replay_frames) == 24
        and all(
            archived["inference"]["config"] == replay["inference"]["config"]
            for archived, replay in zip(
                archived_frames,
                replay_frames,
                strict=True,
            )
        )
    )

    archived_values = {
        key: float(archived_recomputed_decision["values"][key])
        for key in VALUE_TOLERANCES
    }
    replay_values = {
        key: float(replay_recomputed_decision["values"][key])
        for key in VALUE_TOLERANCES
    }
    if not all(
        math.isfinite(value)
        for value in (*archived_values.values(), *replay_values.values())
    ):
        raise FloatingPointError("Q1-R formal metrics are non-finite")
    value_comparison = {
        key: {
            "archived": archived_values[key],
            "replay": replay_values[key],
            "absolute_difference": abs(
                replay_values[key] - archived_values[key]
            ),
            "absolute_tolerance": tolerance,
            "passed": abs(replay_values[key] - archived_values[key])
            <= tolerance,
        }
        for key, tolerance in VALUE_TOLERANCES.items()
    }
    values_within_tolerance = all(
        row["passed"] for row in value_comparison.values()
    )

    archived_decision = archived_recomputed_decision
    replay_decision = replay_recomputed_decision
    decision_preserved = (
        archived_decision.get("protocol") == FORMAL_PROTOCOL
        and replay_decision.get("protocol") == FORMAL_PROTOCOL
        and archived_decision.get("eligible_for_stage0_scientific_decision")
        is True
        and replay_decision.get("eligible_for_stage0_scientific_decision")
        is True
        and archived_decision.get("scientific_checks")
        == replay_decision.get("scientific_checks")
        and archived_decision.get("count_checks")
        == replay_decision.get("count_checks")
        and archived_decision.get("structural_checks")
        == replay_decision.get("structural_checks")
        and replay_decision.get("promotion_passed") is False
        and replay_decision.get("smoke_only") is False
        and replay_decision.get("doppler_head_evaluated") is False
        and all(replay_decision.get("structural_checks", {}).values())
    )
    diagnosis_frames = diagnosis.get("frames")
    if not isinstance(diagnosis_frames, list):
        raise ValueError("Q1-R replay diagnosis lacks frame reports")
    diagnosis_identities = [
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in diagnosis_frames
    ]
    replay_frame_pairs = [identity[:2] for identity in replay_identities]
    diagnosis_q0_hash_matches = sum(
        _diagnosis_query_hash(diagnosis_frame, "q0_normalized_rae_sha256")
        == _query_hash(replay_frame, "q0_normalized_rae_sha256")
        for diagnosis_frame, replay_frame in zip(
            diagnosis_frames,
            replay_frames,
            strict=False,
        )
    )
    diagnosis_q1_hash_matches = sum(
        _diagnosis_query_hash(diagnosis_frame, "q1_normalized_rae_sha256")
        == _query_hash(replay_frame, "q1_normalized_rae_sha256")
        for diagnosis_frame, replay_frame in zip(
            diagnosis_frames,
            replay_frames,
            strict=False,
        )
    )
    diagnosis_control_count = sum(
        _diagnosis_control_passed(diagnosis_frame, replay_frame)
        for diagnosis_frame, replay_frame in zip(
            diagnosis_frames,
            replay_frames,
            strict=False,
        )
    )
    diagnosis_candidate_contracts = all(
        frame.get("partition") == "validation"
        and frame.get("candidate_pool", {}).get("candidate_count") == 700_000
        and frame.get("candidate_pool", {}).get("q0_query_count") == 500_000
        and frame.get("candidate_pool", {}).get("q1_query_count") == 200_000
        and frame.get("candidate_pool", {}).get("ground_truth_accessed") is False
        and frame.get("candidate_pool", {}).get("future_cube_accessed") is False
        and frame.get("candidate_pool", {}).get("cfar_accessed") is False
        for frame in diagnosis_frames
    )
    diagnosis_aggregate = diagnosis.get("aggregate", {})
    diagnosis_recomputed_aggregate = aggregate_failure_factor_frames(
        diagnosis_frames,
        preflight=False,
    )
    diagnosis_aggregate_consistent = _json_normalize(
        diagnosis_aggregate
    ) == _json_normalize(diagnosis_recomputed_aggregate)
    diagnosis_decision = diagnosis_recomputed_aggregate["decision"]
    diagnosis_boundary = diagnosis.get("evidence_boundary", {})
    diagnosis_checkpoint = diagnosis.get("checkpoint", {})
    diagnosis_metrics = diagnosis.get("formal_metrics", {})
    diagnosis_manifest = diagnosis.get("formal_run_manifest", {})
    diagnosis_manifest_checks = diagnosis_manifest.get("checks", {})
    required_manifest_checks = (
        "protocol",
        "source_commit",
        "config",
        "input_hashes",
        "test_locked",
        "doppler_locked",
    )
    diagnosis_count_checks = diagnosis_decision.get("count_checks", {})
    diagnosis_checkpoint_checks = diagnosis_checkpoint.get("checks", {})
    diagnosis_source_hashes = diagnosis.get("diagnostic_source_hashes", {})
    repo = Path(__file__).resolve().parents[2]
    expected_diagnosis_source_hashes = {
        relative: sha256_file(repo / relative)
        for relative in DIAGNOSTIC_SOURCE_RELATIVE_PATHS
    }
    diagnosis_frozen_inputs = diagnosis.get("frozen_inputs", {})
    diagnosis_candidate_contract = diagnosis.get("candidate_contract")
    replay_candidate_contracts = [
        frame.get("inference", {}).get("config") for frame in replay_frames
    ]
    source_and_input_binding = (
        diagnosis_source_hashes == expected_diagnosis_source_hashes
        and diagnosis_frozen_inputs.get("hashes")
        == replay_manifest.get("input_hashes")
        and diagnosis_frozen_inputs.get("test_partition_accessed") is False
        and diagnosis_candidate_contract is not None
        and all(
            contract == diagnosis_candidate_contract
            for contract in replay_candidate_contracts
        )
        and diagnosis.get("artifact_label") == diagnostic_artifact_label()
    )
    checkpoint_and_manifest_checks = (
        isinstance(diagnosis_checkpoint_checks, dict)
        and bool(diagnosis_checkpoint_checks)
        and all(value is True for value in diagnosis_checkpoint_checks.values())
        and isinstance(diagnosis_manifest_checks, dict)
        and all(
            diagnosis_manifest_checks.get(key) is True
            for key in required_manifest_checks
        )
        and all(
            row.get("matches_formal_run") is True
            and row.get("sha256")
            == _source_hash_by_suffix(replay_sources, suffix)
            for suffix, row in diagnosis_manifest.get(
                "checkpoint_source_compatibility", {}
            ).items()
        )
        and set(
            diagnosis_manifest.get("checkpoint_source_compatibility", {})
        )
        == set(FORMAL_SOURCE_SUFFIXES)
    )
    archived_runtime = archived_manifest.get("runtime", {})
    replay_runtime = replay_manifest.get("runtime", {})
    diagnosis_runtime = diagnosis.get("runtime", {})
    runtime_device_names = (
        archived_runtime.get("device_name"),
        replay_runtime.get("device_name"),
        diagnosis_runtime.get("device_name"),
    )
    runtime_torch_versions = (
        archived_runtime.get("torch_version"),
        replay_runtime.get("torch_version"),
        diagnosis_runtime.get("torch_version"),
    )
    h200_runtime_match = all(
        name == EXPECTED_H200_DEVICE_NAME for name in runtime_device_names
    ) and (
        isinstance(runtime_torch_versions[0], str)
        and len(set(runtime_torch_versions)) == 1
    )
    diagnosis_physical_gpu_policy = (
        diagnosis_runtime.get("cuda_device_order") == "PCI_BUS_ID"
        and diagnosis_runtime.get("cuda_visible_devices") in ("0", "2")
        and diagnosis_runtime.get("visible_cuda_device_count") == 1
        and diagnosis_runtime.get("device_argument") == "cuda:0"
    )
    diagnosis_replay_binding = (
        diagnosis.get("protocol") == DIAGNOSIS_PROTOCOL
        and diagnosis.get("source_commit") == certifier_source_commit
        and diagnosis.get("source_head") == certifier_source_commit
        and diagnosis.get("preflight") is False
        and diagnosis.get("frame_count") == 24
        and diagnosis_identities == replay_frame_pairs
        and diagnosis_q0_hash_matches == 24
        and diagnosis_q1_hash_matches == 24
        and diagnosis_control_count == 24
        and diagnosis_candidate_contracts
        and diagnosis_aggregate_consistent
        and source_and_input_binding
        and checkpoint_and_manifest_checks
        and diagnosis_physical_gpu_policy
        and diagnosis_aggregate.get("frame_count") == 24
        and diagnosis_aggregate.get("far_target_frame_count") == 23
        and diagnosis_checkpoint.get("sha256") == checkpoint_sha
        and diagnosis_checkpoint.get("protocol") == FORMAL_PROTOCOL
        and diagnosis_checkpoint.get("source_commit") == FORMAL_SOURCE_COMMIT
        and diagnosis_checkpoint.get("epoch") == 20
        and diagnosis_metrics.get("sha256") == sha256_file(replay_metrics_path)
        and diagnosis_metrics.get("protocol") == FORMAL_PROTOCOL
        and diagnosis_metrics.get("epoch") == 20
        and diagnosis_manifest.get("sha256")
        == sha256_file(replay_manifest_path)
    )
    oracle_passed = (
        diagnosis_replay_binding
        and diagnosis_decision.get("formal_failure_factor_decision_eligible")
        is True
        and diagnosis_decision.get("branch")
        == "confidence_ranking_bottleneck_indicated"
        and diagnosis_decision.get(
            "validation_gt_nearest_score_geometry_passed"
        )
        is True
        and diagnosis_count_checks.get("exact_24_validation_frames") is True
        and diagnosis_count_checks.get("exact_23_far_target_frames") is True
        and diagnosis_decision.get("current_confidence_geometry_passed")
        is False
        and diagnosis_decision.get(
            "validation_gt_nearest_score_unattainable"
        )
        is True
        and diagnosis_decision.get(
            "validation_gt_nearest_score_strict_upper_bound"
        )
        is False
        and diagnosis_decision.get("method_promotion_eligible") is False
    )
    checks = {
        "source_config_data_seed_epoch_match": (
            checkpoint.get("protocol") == FORMAL_PROTOCOL
            and checkpoint.get("source_commit") == FORMAL_SOURCE_COMMIT
            and int(checkpoint.get("epoch", -1)) == 20
            and isinstance(checkpoint.get("model"), dict)
            and _json_normalize(checkpoint.get("config"))
            == _json_normalize(replay_manifest.get("config"))
            and replay_manifest.get("config") == archived_manifest.get("config")
            and replay_manifest.get("input_hashes")
            == archived_manifest.get("input_hashes")
            and archived_manifest.get("protocol") == FORMAL_PROTOCOL
            and archived_manifest.get("source_commit") == FORMAL_SOURCE_COMMIT
            and replay_manifest.get("protocol") == FORMAL_PROTOCOL
            and replay_manifest.get("source_commit") == FORMAL_SOURCE_COMMIT
            and archived_metrics.get("protocol") == FORMAL_PROTOCOL
            and archived_metrics.get("source_commit") == FORMAL_SOURCE_COMMIT
            and archived_metrics.get("epoch") == 20
            and archived_metrics.get("checkpoint_sha256")
            == ORIGINAL_CHECKPOINT_SHA256
            and sha256_file(archived_metrics_path) == ARCHIVED_METRICS_SHA256
            and sha256_file(archived_manifest_path)
            == ARCHIVED_RUN_MANIFEST_SHA256
            and replay_metrics.get("protocol") == FORMAL_PROTOCOL
            and replay_metrics.get("source_commit") == FORMAL_SOURCE_COMMIT
            and source_hashes_match
            and replay_metrics.get("checkpoint_sha256") == checkpoint_sha
            and replay_metrics.get("epoch") == 20
            and h200_runtime_match
            and diagnosis_physical_gpu_policy
        ),
        "formal_metrics_recomputed": (
            archived_formal_consistent and replay_formal_consistent
        ),
        "formal_stage0_decision_preserved": (
            decision_preserved and values_within_tolerance
        ),
        "validation_frame_contract_preserved": (
            archived_identities == replay_identities
            and len(replay_identities) == 24
            and archived_metrics.get("metrics", {}).get("frame_count") == 24
            and replay_metrics.get("metrics", {}).get("frame_count") == 24
            and replay_metrics.get("metrics", {}).get("far_target_frame_count")
            == 23
            and all(frame.get("partition") == "validation" for frame in replay_frames)
        ),
        "candidate_query_contract_preserved": (
            inference_contracts_match
            and q0_hash_matches == 24
        ),
        "replay_diagnosis_bound": diagnosis_replay_binding,
        "diagnosis_aggregate_recomputed": diagnosis_aggregate_consistent,
        "diagnosis_hash_geometry_chain_recomputed": (
            diagnosis_control_count == 24
        ),
        "diagnostic_source_and_input_binding": source_and_input_binding,
        "physical_gpu_policy": diagnosis_physical_gpu_policy,
        "validation_gt_ranking_oracle_passed": oracle_passed,
        "training_started": diagnosis_boundary.get("training_started")
        is not False,
        "checkpoint_modified": diagnosis_boundary.get("checkpoint_modified")
        is not False,
        "test_partition_accessed": diagnosis_boundary.get(
            "test_partition_accessed"
        )
        is not False,
        "future_cube_accessed": diagnosis_boundary.get("future_cube_accessed")
        is not False,
        "cfar_accessed": diagnosis_boundary.get("cfar_accessed") is not False,
        "doppler_head_evaluated": diagnosis_boundary.get(
            "doppler_head_evaluated"
        )
        is not False,
    }
    authorized = (
        all(
            checks[key]
            for key in (
                "source_config_data_seed_epoch_match",
                "formal_metrics_recomputed",
                "formal_stage0_decision_preserved",
                "validation_frame_contract_preserved",
                "candidate_query_contract_preserved",
                "replay_diagnosis_bound",
                "diagnosis_aggregate_recomputed",
                "diagnosis_hash_geometry_chain_recomputed",
                "diagnostic_source_and_input_binding",
                "physical_gpu_policy",
                "validation_gt_ranking_oracle_passed",
            )
        )
        and checks["training_started"] is False
        and checks["checkpoint_modified"] is False
        and checks["test_partition_accessed"] is False
        and checks["future_cube_accessed"] is False
        and checks["cfar_accessed"] is False
        and checks["doppler_head_evaluated"] is False
    )

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": (
            "replay_parent_authorized_for_q1r_tiny"
            if authorized
            else "replay_parent_not_authorized_for_q1r_tiny"
        ),
        "q1r_tiny_authorized": authorized,
        "identity": {
            "original_checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "replay_checkpoint_sha256": checkpoint_sha,
            "replay_metrics_sha256": sha256_file(replay_metrics_path),
            "replay_run_manifest_sha256": sha256_file(replay_manifest_path),
            "replay_failure_diagnosis_sha256": sha256_file(
                replay_diagnosis_path
            ),
            "formal_source_commit": FORMAL_SOURCE_COMMIT,
            "formal_epoch": 20,
            "exact_original_checkpoint": (
                checkpoint_sha == ORIGINAL_CHECKPOINT_SHA256
            ),
            "certifier_source_commit": certifier_source_commit,
        },
        "checks": checks,
        "comparison": {
            "formal_value_tolerances": value_comparison,
            "formal_source_hashes": source_comparison,
            "validation_q0_query_hash_match_count": q0_hash_matches,
            "validation_q1_query_hash_match_count": q1_hash_matches,
            "archived_q1_query_hash_exact_match_required": False,
            "diagnosis_replay_q0_query_hash_match_count": (
                diagnosis_q0_hash_matches
            ),
            "diagnosis_replay_q1_query_hash_match_count": (
                diagnosis_q1_hash_matches
            ),
            "validation_frame_count": len(replay_frames),
            "diagnosis_frame_count": len(diagnosis_frames),
            "diagnosis_formal_hash_control_pass_count": (
                diagnosis_control_count
            ),
            "replay_is_original_checkpoint": (
                checkpoint_sha == ORIGINAL_CHECKPOINT_SHA256
            ),
            "runtime": {
                "archived_device_name": runtime_device_names[0],
                "replay_device_name": runtime_device_names[1],
                "diagnosis_device_name": runtime_device_names[2],
                "archived_torch_version": runtime_torch_versions[0],
                "replay_torch_version": runtime_torch_versions[1],
                "diagnosis_torch_version": runtime_torch_versions[2],
                "h200_and_torch_version_match": h200_runtime_match,
                "diagnosis_physical_gpu_policy": (
                    diagnosis_physical_gpu_policy
                ),
            },
            "diagnostic_source_hashes": {
                "actual": diagnosis_source_hashes,
                "expected": expected_diagnosis_source_hashes,
                "passed": diagnosis_source_hashes
                == expected_diagnosis_source_hashes,
            },
        },
        "inputs": {
            "archived_metrics_sha256": sha256_file(archived_metrics_path),
            "expected_archived_metrics_sha256": ARCHIVED_METRICS_SHA256,
            "archived_run_manifest_sha256": sha256_file(
                archived_manifest_path
            ),
            "expected_archived_run_manifest_sha256": (
                ARCHIVED_RUN_MANIFEST_SHA256
            ),
            "replay_checkpoint_path": str(replay_checkpoint_path.resolve()),
            "replay_metrics_path": str(replay_metrics_path.resolve()),
            "replay_run_manifest_path": str(replay_manifest_path.resolve()),
            "replay_failure_diagnosis_path": str(
                replay_diagnosis_path.resolve()
            ),
        },
        "claim_boundary": (
            "This certificate authorizes only an eight-train-frame Q1-R "
            "memorization gate on a source/config/data-equivalent replay. It "
            "does not identify the replay as the deleted original checkpoint "
            "unless exact_original_checkpoint is true. Fixed Q0 hashes must "
            "match the archive; learned occupancy-dependent Q1 hashes must "
            "instead be reproduced exactly by the replay-bound diagnosis. "
            "The certificate does not unlock validation, test, Doppler, "
            "cycle, or temporal claims. Legacy formal peak-memory values are "
            "pinned from the source-bound endpoint decision because the old "
            "artifact did not store independent raw peak-byte fields."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived-metrics", type=Path, required=True)
    parser.add_argument("--archived-run-manifest", type=Path, required=True)
    parser.add_argument("--replay-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-metrics", type=Path, required=True)
    parser.add_argument("--replay-run-manifest", type=Path, required=True)
    parser.add_argument("--replay-failure-diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    document = certify_replay_parent(
        archived_metrics_path=args.archived_metrics,
        archived_manifest_path=args.archived_run_manifest,
        replay_checkpoint_path=args.replay_checkpoint,
        replay_metrics_path=args.replay_metrics,
        replay_manifest_path=args.replay_run_manifest,
        replay_diagnosis_path=args.replay_failure_diagnosis,
        certifier_source_commit=args.source_commit,
    )
    atomic_json(args.output, document)
    print(json.dumps(document["status"]), flush=True)
    if document["q1r_tiny_authorized"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
