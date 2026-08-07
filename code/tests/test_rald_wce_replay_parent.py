import hashlib
import json
from pathlib import Path

import torch

from scripts import certify_rald_wce_replay_parent as cert


CERTIFIER_SOURCE_COMMIT = "a" * 40


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_frozen_archive_hashes_match_repository_evidence() -> None:
    repo = Path(__file__).resolve().parents[2]
    assert cert.sha256_file(
        repo / "artifacts/g1/wce_formal_f2a9489/metrics_epoch020.json"
    ) == cert.ARCHIVED_METRICS_SHA256
    assert cert.sha256_file(
        repo / "artifacts/g1/wce_formal_f2a9489/run_manifest.json"
    ) == cert.ARCHIVED_RUN_MANIFEST_SHA256


def _formal_decision(*, chamfer_mean_m: float = 4.0) -> dict:
    return {
        "protocol": cert.FORMAL_PROTOCOL,
        "eligible_for_stage0_scientific_decision": True,
        "values": {
            "chamfer_mean_m": chamfer_mean_m,
            "completeness_median_m": 1.5,
            "outlier_fraction_mean": 0.31,
            "far_completeness_mean_m": 11.1,
            "condition_wrong_minus_matched_chamfer_fraction_mean": 0.13,
            "matched_condition_win_fraction": 16.0 / 24.0,
        },
        "structural_checks": {
            "exact_matched": True,
            "exact_wrong": True,
            "minimum_distance": True,
        },
        "scientific_checks": {
            "chamfer": False,
            "completeness": True,
            "outlier": False,
            "condition": False,
        },
        "count_checks": {
            "exact_24_validation_frames": True,
            "exact_23_far_target_frames": True,
        },
        "promotion_passed": False,
        "smoke_only": False,
        "doppler_head_evaluated": False,
    }


def _formal_frames(
    *,
    q0_drift: bool = False,
    q1_drift: bool = False,
) -> tuple[list[dict], list[dict]]:
    archived: list[dict] = []
    replay: list[dict] = []
    for index in range(24):
        common = {
            "sequence": index + 1,
            "radar_index": 100 + index,
            "partition": "validation",
            "wrong_condition_sequence": 40 + index,
            "wrong_condition_radar_index": 500 + index,
            "inference": {
                "config": {
                    "q0_query_count": 500_000,
                    "q1_query_count": 200_000,
                    "candidate_count": 700_000,
                },
                "query_hashes": {
                    "q0_normalized_rae_sha256": _digest(f"q0-{index}"),
                    "q1_normalized_rae_sha256": _digest(f"q1-{index}"),
                },
            },
        }
        archived.append(json.loads(json.dumps(common)))
        replay.append(json.loads(json.dumps(common)))
    if q0_drift:
        replay[7]["inference"]["query_hashes"][
            "q0_normalized_rae_sha256"
        ] = _digest("drifted-q0")
    if q1_drift:
        replay[7]["inference"]["query_hashes"][
            "q1_normalized_rae_sha256"
        ] = _digest("drifted-q1")
    return archived, replay


def _write_bundle(
    tmp_path: Path,
    *,
    q0_drift: bool = False,
    q1_drift: bool = False,
    diagnosis_q1_drift: bool = False,
    diagnosis_control_passed: bool = True,
    replay_chamfer_mean_m: float = 4.0,
) -> dict[str, Path]:
    archived_metrics_path = tmp_path / "archived_metrics.json"
    archived_manifest_path = tmp_path / "archived_manifest.json"
    replay_checkpoint_path = tmp_path / "replay_checkpoint.pt"
    replay_metrics_path = tmp_path / "replay_metrics.json"
    replay_manifest_path = tmp_path / "replay_manifest.json"
    diagnosis_path = tmp_path / "replay_diagnosis.json"

    checkpoint_config = {
        "seed": 20260716,
        "epochs": 20,
        "q0_range_quotas": (166_667, 166_667, 166_666),
        "q1_anchor_quotas": (20_000, 4_250, 750),
        "output_range_quotas": (8_000, 1_700, 300),
        "test_accessed": False,
        "doppler_head": False,
    }
    manifest_config = json.loads(json.dumps(checkpoint_config))
    source_hashes = {
        f"/formal/source/{suffix}": _digest(suffix)
        for suffix in cert.FORMAL_SOURCE_SUFFIXES
    }
    input_hashes = {
        "manifest": _digest("manifest"),
        "scene_split": _digest("scene-split"),
        "normalization": _digest("normalization"),
        "corrected_dense_geometry": _digest("geometry"),
    }
    manifest = {
        "protocol": cert.FORMAL_PROTOCOL,
        "source_commit": cert.FORMAL_SOURCE_COMMIT,
        "config": manifest_config,
        "input_hashes": input_hashes,
        "source_hashes": source_hashes,
    }
    _write_json(archived_manifest_path, manifest)
    _write_json(replay_manifest_path, manifest)

    torch.save(
        {
            "protocol": cert.FORMAL_PROTOCOL,
            "source_commit": cert.FORMAL_SOURCE_COMMIT,
            "epoch": 20,
            "config": checkpoint_config,
            "model": {"weight": torch.ones(1)},
        },
        replay_checkpoint_path,
    )
    checkpoint_sha = cert.sha256_file(replay_checkpoint_path)
    archived_frames, replay_frames = _formal_frames(
        q0_drift=q0_drift,
        q1_drift=q1_drift,
    )
    archived_metrics = {
        "protocol": cert.FORMAL_PROTOCOL,
        "source_commit": cert.FORMAL_SOURCE_COMMIT,
        "epoch": 20,
        "checkpoint_sha256": cert.ORIGINAL_CHECKPOINT_SHA256,
        "metrics": {
            "frames": archived_frames,
            "frame_count": 24,
            "far_target_frame_count": 23,
        },
        "decision": _formal_decision(),
    }
    replay_metrics = {
        "protocol": cert.FORMAL_PROTOCOL,
        "source_commit": cert.FORMAL_SOURCE_COMMIT,
        "epoch": 20,
        "checkpoint_sha256": checkpoint_sha,
        "metrics": {
            "frames": replay_frames,
            "frame_count": 24,
            "far_target_frame_count": 23,
        },
        "decision": _formal_decision(
            chamfer_mean_m=replay_chamfer_mean_m
        ),
    }
    _write_json(archived_metrics_path, archived_metrics)
    _write_json(replay_metrics_path, replay_metrics)

    diagnosis_frames = []
    for index, frame in enumerate(replay_frames):
        replay_query_hashes = frame["inference"]["query_hashes"]
        diagnosis_q1_hash = replay_query_hashes[
            "q1_normalized_rae_sha256"
        ]
        if diagnosis_q1_drift and index == 7:
            diagnosis_q1_hash = _digest("diagnosis-drifted-q1")
        diagnosis_frames.append(
            {
                "sequence": frame["sequence"],
                "radar_index": frame["radar_index"],
                "partition": "validation",
                "candidate_pool": {
                    "candidate_count": 700_000,
                    "q0_query_count": 500_000,
                    "q1_query_count": 200_000,
                    "ground_truth_accessed": False,
                    "future_cube_accessed": False,
                    "cfar_accessed": False,
                    "query_hashes": {
                        "q0_normalized_rae_sha256": replay_query_hashes[
                            "q0_normalized_rae_sha256"
                        ],
                        "q1_normalized_rae_sha256": diagnosis_q1_hash,
                    },
                },
                "current_confidence": {
                    "formal_hash_control": {
                        "passed": diagnosis_control_passed,
                        "bit_exact": diagnosis_control_passed,
                        "export_hash_checks": {
                            "xyz_sha256": diagnosis_control_passed,
                            "confidence_sha256": diagnosis_control_passed,
                            "selected_candidate_rows_sha256": (
                                diagnosis_control_passed
                            ),
                        },
                        "query_hash_checks": {
                            "q0_normalized_rae_sha256": (
                                diagnosis_control_passed
                            ),
                            "q1_normalized_rae_sha256": (
                                diagnosis_control_passed
                            ),
                        },
                    }
                },
            }
        )
    diagnosis = {
        "protocol": cert.DIAGNOSIS_PROTOCOL,
        "source_commit": CERTIFIER_SOURCE_COMMIT,
        "source_head": CERTIFIER_SOURCE_COMMIT,
        "preflight": False,
        "frame_count": 24,
        "checkpoint": {
            "sha256": checkpoint_sha,
            "protocol": cert.FORMAL_PROTOCOL,
            "source_commit": cert.FORMAL_SOURCE_COMMIT,
            "epoch": 20,
        },
        "formal_metrics": {
            "sha256": cert.sha256_file(replay_metrics_path),
            "protocol": cert.FORMAL_PROTOCOL,
            "epoch": 20,
        },
        "formal_run_manifest": {
            "sha256": cert.sha256_file(replay_manifest_path),
            "checks": {
                "protocol": True,
                "source_commit": True,
                "config": True,
                "input_hashes": True,
                "test_locked": True,
                "doppler_locked": True,
            },
        },
        "frames": diagnosis_frames,
        "aggregate": {
            "frame_count": 24,
            "far_target_frame_count": 23,
            "decision": {
                "formal_failure_factor_decision_eligible": True,
                "branch": "confidence_ranking_bottleneck_indicated",
                "validation_gt_nearest_score_geometry_passed": True,
                "count_checks": {
                    "exact_24_validation_frames": True,
                    "exact_23_far_target_frames": True,
                },
                "current_confidence_geometry_passed": False,
                "validation_gt_nearest_score_unattainable": True,
                "validation_gt_nearest_score_strict_upper_bound": False,
                "method_promotion_eligible": False,
            },
        },
        "evidence_boundary": {
            "training_started": False,
            "checkpoint_modified": False,
            "test_partition_accessed": False,
            "future_cube_accessed": False,
            "cfar_accessed": False,
            "doppler_head_evaluated": False,
        },
    }
    _write_json(diagnosis_path, diagnosis)
    return {
        "archived_metrics_path": archived_metrics_path,
        "archived_manifest_path": archived_manifest_path,
        "replay_checkpoint_path": replay_checkpoint_path,
        "replay_metrics_path": replay_metrics_path,
        "replay_manifest_path": replay_manifest_path,
        "replay_diagnosis_path": diagnosis_path,
    }


def _certify(paths: dict[str, Path]) -> dict:
    original_metrics_sha = cert.ARCHIVED_METRICS_SHA256
    original_manifest_sha = cert.ARCHIVED_RUN_MANIFEST_SHA256
    try:
        cert.ARCHIVED_METRICS_SHA256 = cert.sha256_file(
            paths["archived_metrics_path"]
        )
        cert.ARCHIVED_RUN_MANIFEST_SHA256 = cert.sha256_file(
            paths["archived_manifest_path"]
        )
        return cert.certify_replay_parent(
            **paths,
            certifier_source_commit=CERTIFIER_SOURCE_COMMIT,
        )
    finally:
        cert.ARCHIVED_METRICS_SHA256 = original_metrics_sha
        cert.ARCHIVED_RUN_MANIFEST_SHA256 = original_manifest_sha


def test_certifier_authorizes_source_equivalent_replay_and_normalizes_config(
    tmp_path: Path,
) -> None:
    document = _certify(_write_bundle(tmp_path))

    assert document["q1r_tiny_authorized"] is True
    assert document["identity"]["exact_original_checkpoint"] is False
    assert document["checks"]["replay_diagnosis_bound"] is True
    assert document["comparison"]["validation_q0_query_hash_match_count"] == 24
    assert document["comparison"]["validation_q1_query_hash_match_count"] == 24
    assert (
        document["comparison"][
            "diagnosis_replay_q1_query_hash_match_count"
        ]
        == 24
    )
    assert (
        document["comparison"]["diagnosis_formal_hash_control_pass_count"]
        == 24
    )


def test_certifier_rejects_fixed_q0_query_drift(tmp_path: Path) -> None:
    document = _certify(_write_bundle(tmp_path, q0_drift=True))

    assert document["q1r_tiny_authorized"] is False
    assert document["checks"]["candidate_query_contract_preserved"] is False
    assert document["comparison"]["validation_q0_query_hash_match_count"] == 23


def test_certifier_allows_replay_specific_q1_when_diagnosis_reproduces_it(
    tmp_path: Path,
) -> None:
    document = _certify(_write_bundle(tmp_path, q1_drift=True))

    assert document["q1r_tiny_authorized"] is True
    assert document["comparison"]["validation_q1_query_hash_match_count"] == 23
    assert (
        document["comparison"][
            "diagnosis_replay_q1_query_hash_match_count"
        ]
        == 24
    )


def test_certifier_rejects_q1_not_reproduced_by_diagnosis(
    tmp_path: Path,
) -> None:
    document = _certify(
        _write_bundle(tmp_path, diagnosis_q1_drift=True)
    )

    assert document["q1r_tiny_authorized"] is False
    assert document["checks"]["replay_diagnosis_bound"] is False
    assert (
        document["comparison"][
            "diagnosis_replay_q1_query_hash_match_count"
        ]
        == 23
    )


def test_certifier_rejects_failed_diagnosis_control(tmp_path: Path) -> None:
    document = _certify(
        _write_bundle(tmp_path, diagnosis_control_passed=False)
    )

    assert document["q1r_tiny_authorized"] is False
    assert document["checks"]["replay_diagnosis_bound"] is False
    assert document["checks"]["validation_gt_ranking_oracle_passed"] is False


def test_certifier_rejects_formal_metric_drift(tmp_path: Path) -> None:
    document = _certify(
        _write_bundle(tmp_path, replay_chamfer_mean_m=4.2)
    )

    assert document["q1r_tiny_authorized"] is False
    assert document["checks"]["formal_stage0_decision_preserved"] is False
    assert (
        document["comparison"]["formal_value_tolerances"]["chamfer_mean_m"]
        ["passed"]
        is False
    )
