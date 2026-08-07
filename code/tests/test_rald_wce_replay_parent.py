import hashlib
import json
from pathlib import Path

import pytest
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_archive_hashes_match_repository_evidence() -> None:
    repo = Path(__file__).resolve().parents[2]
    assert cert.sha256_file(
        repo / "artifacts/g1/wce_formal_f2a9489/metrics_epoch020.json"
    ) == cert.ARCHIVED_METRICS_SHA256
    assert cert.sha256_file(
        repo / "artifacts/g1/wce_formal_f2a9489/run_manifest.json"
    ) == cert.ARCHIVED_RUN_MANIFEST_SHA256


def _geometry(
    *,
    chamfer: float,
    completeness: float,
    outlier: float,
    far: float | None,
) -> dict:
    report = {
        "chamfer_m": chamfer,
        "precision_mean_distance_m": max(chamfer - completeness, 0.0),
        "completeness_mean_distance_m": completeness,
        "outlier_fraction_2m": outlier,
        "prediction_count": 10_000,
        "target_count": 1_000,
        "target_effective_count": 900.0,
    }
    if far is not None:
        report["range_60_120m_completeness_mean_distance_m"] = far
    return report


def _formal_frames(
    *,
    chamfer: float = 4.0,
    q0_drift: bool = False,
    q1_drift: bool = False,
) -> tuple[list[dict], list[dict]]:
    archived: list[dict] = []
    replay: list[dict] = []
    for index in range(24):
        has_far_target = index < 23
        common = {
            "sequence": index + 1,
            "radar_index": 100 + index,
            "partition": "validation",
            "wrong_condition_sequence": 40 + index,
            "wrong_condition_radar_index": 500 + index,
            "has_far_target": has_far_target,
            "matched": _geometry(
                chamfer=chamfer,
                completeness=1.5,
                outlier=0.31,
                far=11.1 if has_far_target else None,
            ),
            "wrong_condition": _geometry(
                chamfer=chamfer * 1.13,
                completeness=1.7,
                outlier=0.35,
                far=12.0 if has_far_target else None,
            ),
            "condition_intervention": {
                "wrong_minus_matched_chamfer_fraction": 0.13,
                "matched_chamfer_better": 1.0 if index < 16 else 0.0,
            },
            "matched_export": {
                "exact_point_count": 10_000,
                "observed_minimum_pair_distance_m": 0.06,
                "copy_padding_jitter_duplicate": False,
            },
            "wrong_export": {
                "exact_point_count": 10_000,
                "observed_minimum_pair_distance_m": 0.06,
                "copy_padding_jitter_duplicate": False,
            },
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


def _formal_document(
    frames: list[dict],
    *,
    checkpoint_sha256: str,
) -> dict:
    metrics = cert._recompute_formal_metrics(frames)
    decision = cert.stage0_decision(
        metrics,
        peak_allocated_bytes=2**30,
        peak_reserved_bytes=2 * 2**30,
        formal=True,
    )
    return {
        "protocol": cert.FORMAL_PROTOCOL,
        "source_commit": cert.FORMAL_SOURCE_COMMIT,
        "epoch": 20,
        "checkpoint_sha256": checkpoint_sha256,
        "metrics": metrics,
        "decision": decision,
    }


def _diagnosis_geometry(*, rescued: bool, has_far_target: bool) -> dict:
    if rescued:
        return _geometry(
            chamfer=0.5,
            completeness=0.6,
            outlier=0.02,
            far=1.0 if has_far_target else None,
        )
    return _geometry(
        chamfer=4.0,
        completeness=1.5,
        outlier=0.31,
        far=11.1 if has_far_target else None,
    )


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
        "runtime": {
            "device_name": "NVIDIA H200 NVL",
            "torch_version": "2.12.1+cu130",
        },
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
    archived_frames, _ = _formal_frames()
    _, replay_frames = _formal_frames(
        chamfer=replay_chamfer_mean_m,
        q0_drift=q0_drift,
        q1_drift=q1_drift,
    )
    archived_metrics = _formal_document(
        archived_frames,
        checkpoint_sha256=cert.ORIGINAL_CHECKPOINT_SHA256,
    )
    replay_metrics = _formal_document(
        replay_frames,
        checkpoint_sha256=checkpoint_sha,
    )
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
                "has_far_target": frame["has_far_target"],
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
                    "geometry": _diagnosis_geometry(
                        rescued=False,
                        has_far_target=frame["has_far_target"],
                    ),
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
                    },
                },
                "validation_gt_nearest_score": {
                    "geometry": _diagnosis_geometry(
                        rescued=True,
                        has_far_target=frame["has_far_target"],
                    ),
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
        "runtime": {
            "device_name": "NVIDIA H200 NVL",
            "torch_version": "2.12.1+cu130",
        },
        "frames": diagnosis_frames,
        "aggregate": cert.aggregate_failure_factor_frames(
            diagnosis_frames,
            preflight=False,
        ),
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
    assert document["checks"]["formal_metrics_recomputed"] is True
    assert document["checks"]["diagnosis_aggregate_recomputed"] is True
    assert document["checks"]["replay_diagnosis_bound"] is True
    assert document["comparison"]["validation_q0_query_hash_match_count"] == 24
    assert document["comparison"]["validation_q1_query_hash_match_count"] == 24
    assert (
        document["comparison"]["diagnosis_replay_q1_query_hash_match_count"]
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
        document["comparison"]["diagnosis_replay_q1_query_hash_match_count"]
        == 24
    )


def test_certifier_rejects_q1_not_reproduced_by_diagnosis(
    tmp_path: Path,
) -> None:
    document = _certify(_write_bundle(tmp_path, diagnosis_q1_drift=True))

    assert document["q1r_tiny_authorized"] is False
    assert document["checks"]["replay_diagnosis_bound"] is False
    assert (
        document["comparison"]["diagnosis_replay_q1_query_hash_match_count"]
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


def test_certifier_rejects_stale_formal_aggregate_and_decision(
    tmp_path: Path,
) -> None:
    paths = _write_bundle(tmp_path)
    replay = _read_json(paths["replay_metrics_path"])
    replay["metrics"]["frames"][0]["matched"]["chamfer_m"] = 9.0
    _write_json(paths["replay_metrics_path"], replay)
    diagnosis = _read_json(paths["replay_diagnosis_path"])
    diagnosis["formal_metrics"]["sha256"] = cert.sha256_file(
        paths["replay_metrics_path"]
    )
    _write_json(paths["replay_diagnosis_path"], diagnosis)

    document = _certify(paths)

    assert document["q1r_tiny_authorized"] is False
    assert document["checks"]["formal_metrics_recomputed"] is False


def test_certifier_rejects_stale_diagnosis_aggregate(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path)
    diagnosis = _read_json(paths["replay_diagnosis_path"])
    diagnosis["frames"][0]["current_confidence"]["geometry"][
        "chamfer_m"
    ] = 9.0
    _write_json(paths["replay_diagnosis_path"], diagnosis)

    document = _certify(paths)

    assert document["q1r_tiny_authorized"] is False
    assert document["checks"]["diagnosis_aggregate_recomputed"] is False
    assert document["checks"]["replay_diagnosis_bound"] is False


def test_certifier_rejects_non_h200_runtime(tmp_path: Path) -> None:
    paths = _write_bundle(tmp_path)
    replay_manifest = _read_json(paths["replay_manifest_path"])
    replay_manifest["runtime"]["device_name"] = "NVIDIA RTX 5000 Ada"
    _write_json(paths["replay_manifest_path"], replay_manifest)
    diagnosis = _read_json(paths["replay_diagnosis_path"])
    diagnosis["formal_run_manifest"]["sha256"] = cert.sha256_file(
        paths["replay_manifest_path"]
    )
    _write_json(paths["replay_diagnosis_path"], diagnosis)

    document = _certify(paths)

    assert document["q1r_tiny_authorized"] is False
    assert document["checks"]["source_config_data_seed_epoch_match"] is False
    assert (
        document["comparison"]["runtime"]["h200_and_torch_version_match"]
        is False
    )


@pytest.mark.parametrize(
    "boundary_key",
    (
        "training_started",
        "checkpoint_modified",
        "test_partition_accessed",
        "future_cube_accessed",
        "cfar_accessed",
        "doppler_head_evaluated",
    ),
)
def test_certifier_keeps_forbidden_boundary_flags_independent(
    tmp_path: Path,
    boundary_key: str,
) -> None:
    paths = _write_bundle(tmp_path)
    diagnosis = _read_json(paths["replay_diagnosis_path"])
    diagnosis["evidence_boundary"][boundary_key] = True
    _write_json(paths["replay_diagnosis_path"], diagnosis)

    document = _certify(paths)

    assert document["q1r_tiny_authorized"] is False
    for key in diagnosis["evidence_boundary"]:
        assert document["checks"][key] is (key == boundary_key)
