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
from typing import Any

import torch


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


def _diagnosis_control_passed(frame: dict[str, Any]) -> bool:
    control = frame.get("current_confidence", {}).get("formal_hash_control", {})
    export_checks = control.get("export_hash_checks", {})
    query_checks = control.get("query_hash_checks", {})
    return (
        control.get("passed") is True
        and control.get("bit_exact") is True
        and all(
            export_checks.get(key) is True
            for key in (
                "xyz_sha256",
                "confidence_sha256",
                "selected_candidate_rows_sha256",
            )
        )
        and all(
            query_checks.get(key) is True
            for key in (
                "q0_normalized_rae_sha256",
                "q1_normalized_rae_sha256",
            )
        )
    )


def _frame_identity(frame: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(frame["sequence"]),
        int(frame["radar_index"]),
        int(frame["wrong_condition_sequence"]),
        int(frame["wrong_condition_radar_index"]),
    )


def _formal_values(document: dict[str, Any]) -> dict[str, float]:
    values = document.get("decision", {}).get("values")
    if not isinstance(values, dict):
        raise ValueError("Q1-R formal metrics lack decision values")
    return {key: float(values[key]) for key in VALUE_TOLERANCES}


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

    archived_frames = archived_metrics.get("metrics", {}).get("frames", [])
    replay_frames = replay_metrics.get("metrics", {}).get("frames", [])
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

    archived_values = _formal_values(archived_metrics)
    replay_values = _formal_values(replay_metrics)
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

    archived_decision = archived_metrics.get("decision", {})
    replay_decision = replay_metrics.get("decision", {})
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
        _diagnosis_control_passed(frame) for frame in diagnosis_frames
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
    diagnosis_decision = diagnosis.get("aggregate", {}).get("decision", {})
    diagnosis_aggregate = diagnosis.get("aggregate", {})
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
        and isinstance(diagnosis_manifest_checks, dict)
        and all(
            diagnosis_manifest_checks.get(key) is True
            for key in required_manifest_checks
        )
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
    forbidden_access_locked = (
        diagnosis_boundary.get("training_started") is False
        and diagnosis_boundary.get("checkpoint_modified") is False
        and diagnosis_boundary.get("test_partition_accessed") is False
        and diagnosis_boundary.get("future_cube_accessed") is False
        and diagnosis_boundary.get("cfar_accessed") is False
        and diagnosis_boundary.get("doppler_head_evaluated") is False
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
        "validation_gt_ranking_oracle_passed": oracle_passed,
        "test_partition_accessed": False if forbidden_access_locked else True,
        "future_cube_accessed": False if forbidden_access_locked else True,
        "cfar_accessed": False if forbidden_access_locked else True,
        "doppler_head_evaluated": False if forbidden_access_locked else True,
    }
    authorized = (
        all(
            checks[key]
            for key in (
                "source_config_data_seed_epoch_match",
                "formal_stage0_decision_preserved",
                "validation_frame_contract_preserved",
                "candidate_query_contract_preserved",
                "replay_diagnosis_bound",
                "validation_gt_ranking_oracle_passed",
            )
        )
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
            "cycle, or temporal claims."
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
