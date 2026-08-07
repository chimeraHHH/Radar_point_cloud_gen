from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import preflight_stda_f0_capacity as capacity
from scripts import stda_f0_bundle_verify as bundle_verify


SOURCE_COMMIT = "a" * 40
GPU = {
    "index": 2,
    "uuid": "GPU-000b6236-3632-a001-9667-1f02cbb61c8b",
    "pci_bus_id": "00000000:D1:00.0",
    "name": "NVIDIA H200 NVL",
}


def _write_canonical_record(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    payload = capacity.canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": None,
        "size_bytes": len(payload),
        "sha256": capacity.sha256_bytes(payload),
        "fsynced": True,
        "full_rehash_passed": True,
    }


def _synthetic_support_evidence(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    record = {
        "position": 0,
        "sequence": 1,
        "radar_index": 2,
        "partition": "train",
    }
    frame_directory = root / "frames/seq01_radar00002"
    frame_directory.mkdir(parents=True)
    support_payload = b"synthetic packed support"
    support_path = frame_directory / "support.bin"
    support_path.write_bytes(support_payload)
    support_sha256 = capacity.sha256_bytes(support_payload)
    support_file = {
        "relative_path": "frames/seq01_radar00002/support.bin",
        "size_bytes": len(support_payload),
        "sha256": support_sha256,
        "fsynced": True,
        "full_rehash_passed": True,
    }
    files = {"support.bin": support_file}
    cube = {"sha256": "c" * 64, "path_read_calls": 1}
    candidate = {
        "passed": True,
        "hashes": {"xyz_m": "d" * 64},
        "predecessor_expected": False,
        "predecessor_hashes_match": True,
    }
    support = {
        "candidate_count": 700_000,
        "support_count": 10_000,
        "selected_color_id": 3,
        "color_cardinalities": [10_000],
        "candidate_field_sha256": "e" * 64,
        "support_sha256": support_sha256,
        "formal_independent_spacing_verified": True,
        "independent_verifier": {"passed": True},
    }
    source_paths = capacity.SUPPORT_RUNTIME_SOURCE_PATHS
    source_hashes = {
        path: f"{index + 1:064x}"
        for index, path in enumerate(source_paths)
    }
    critical_hashes = {
        path: source_hashes[path] for path in capacity.SUPPORT_CHILD_SOURCE_PATHS
    }
    orchestrator_hashes = {
        path: source_hashes[path]
        for path in capacity.SUPPORT_ORCHESTRATOR_SOURCE_PATHS
    }
    commitment = {
        "schema": "stda_f0_support_frame_commitment_v1",
        "protocol_sha256": capacity.PROTOCOL_SHA256,
        "protocol_freeze_commit": capacity.PROTOCOL_FREEZE_COMMIT,
        "position": 0,
        "sequence": 1,
        "radar_index": 2,
        "cube_sha256": cube["sha256"],
        "cube": cube,
        "candidate_hashes": candidate["hashes"],
        "candidate_predecessor_expected": candidate["predecessor_expected"],
        "candidate_predecessor_hashes_match": candidate[
            "predecessor_hashes_match"
        ],
        "candidate_field_sha256": support["candidate_field_sha256"],
        "support_sha256": support_sha256,
        "support_count": support["support_count"],
        "selected_color_id": support["selected_color_id"],
        "color_cardinalities": support["color_cardinalities"],
        "files": files,
        "independent_support_verification_passed": True,
        "support_target_input": False,
        "ground_truth_accessed": False,
    }
    commitment_record = _write_canonical_record(
        frame_directory / "support_record.json",
        commitment,
    )
    commitment_record["relative_path"] = (
        "frames/seq01_radar00002/support_record.json"
    )
    entry = {
        "schema": "stda_f0_support_manifest_entry_evidence_v1",
        "protocol_sha256": capacity.PROTOCOL_SHA256,
        "protocol_freeze_commit": capacity.PROTOCOL_FREEZE_COMMIT,
        "position": 0,
        "sequence": 1,
        "radar_index": 2,
        "frame_key": "seq01/radar00002",
        "partition": "train",
        "cube": cube,
        "candidate": candidate,
        "support": support,
        "files": files,
        "frame_commitment": commitment_record,
        "runtime_source_sha256": source_hashes,
        "critical_runtime_source_sha256": critical_hashes,
        "orchestrator_source_sha256": orchestrator_hashes,
        "timing_boundary_contract": {
            "support_frame_ns_in_final_manifest": True,
            "manifest_entry_write_fsync_rehash_inside_support_frame_ns": True,
            "cleanup_and_final_cuda_sync_inside_support_frame_ns": True,
            "elapsed_time_excluded_here_to_avoid_self_reference": True,
        },
        "support_target_input": False,
        "ground_truth_accessed": False,
    }
    entry_record = _write_canonical_record(
        frame_directory / "support_manifest_entry.json",
        entry,
    )
    entry_record["relative_path"] = (
        "frames/seq01_radar00002/support_manifest_entry.json"
    )
    frame = {
        **record,
        "frame_key": "seq01/radar00002",
        "cube": cube,
        "candidate": candidate,
        "support": support,
        "files": files,
        "frame_commitment": commitment_record,
        "manifest_entry_evidence": entry_record,
        "runtime_source_sha256": source_hashes,
        "critical_runtime_source_sha256": critical_hashes,
        "orchestrator_source_sha256": orchestrator_hashes,
        "support_target_input": False,
        "ground_truth_accessed": False,
    }
    manifest = {
        "schema": "stda_f0_support_manifest_v1",
        "protocol_sha256": capacity.PROTOCOL_SHA256,
        "protocol_freeze_commit": capacity.PROTOCOL_FREEZE_COMMIT,
        "source_commit": SOURCE_COMMIT,
        "source_hashes": source_hashes,
        "frames": [frame],
        "formal_contract": {
            "partition": "train",
            "frame_count": 1,
            "candidate_count_per_frame": 700_000,
            "predecessor_hash_frames": 12,
            "fresh_hash_frames": 64,
            "single_cuda_visible_os_process": True,
            "descendant_processes_created": False,
            "target_loader_started": False,
        },
        "summary": {
            "predecessor_hashes_verified": 12,
            "fresh_candidate_hashes_recorded": 64,
            "all_independent_support_verifications_passed": True,
        },
    }
    _write_canonical_record(root / "support_manifest.json", manifest)
    _write_canonical_record(
        root / "open_ledger.json",
        {
            "schema": "stda_f0_support_open_ledger_v1",
            "source_commit": SOURCE_COMMIT,
            "support_target_input": False,
            "decision": {"passed": True},
        },
    )
    return manifest, [record], source_hashes


@pytest.fixture(autouse=True)
def _require_cpu_only_h200() -> None:
    assert socket.gethostname() == "WHUServer-H200"
    assert "CUDA_VISIBLE_DEVICES" in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def _gate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    export = {
        "point_count": 10_000,
        "finite_xyz": True,
        "unique_xyz_bytes": True,
        "spacing": {"strict_spacing_5cm": True},
    }
    metric = {
        "geometry": {
            "chamfer_m": 0.8,
            "outlier_fraction_2m": 0.05,
        },
        "range_metrics": {
            prefix: {
                "applicable": True,
                "completeness_mean_distance_m": 1.0,
                "recall_1m": 0.8,
            }
            for prefix in capacity.RANGE_GATE_PREFIXES
        },
    }
    structure = {
        "structural_domain_valid": True,
        "class_metrics": {
            f"{prefix}_{suffix}": {
                "applicable": True,
                "completeness_mean_distance_m": 1.0,
                "recall_1m": 0.8,
            }
            for prefix in capacity.RANGE_GATE_PREFIXES
            for suffix in capacity.RETURN_GATE_SUFFIXES
        },
    }
    return export, metric, structure


def _passing_gate_vector() -> dict[str, dict[str, bool]]:
    return capacity.build_shared_gate_vector(*_gate_inputs())


def _arm(vector: dict[str, dict[str, bool]]) -> dict[str, Any]:
    return {
        "passed": capacity.gate_vector_passed(vector),
        "gate_vector": vector,
    }


def _base_scientific_frame() -> dict[str, Any]:
    decision = _passing_gate_vector()
    pointwise = copy.deepcopy(decision)
    greedy = copy.deepcopy(decision)
    return {
        "support": {"capacity_sufficient": True},
        "graph": {"maximum_cardinality": 10_000},
        "arms": {
            "decision": _arm(decision),
            "pointwise": _arm(pointwise),
            "greedy": _arm(greedy),
        },
    }


def _frames_for_status(status: str) -> list[dict[str, Any]]:
    frames = [
        copy.deepcopy(_base_scientific_frame())
        for _ in range(capacity.EXPECTED_TRAIN_FRAMES)
    ]
    if status == capacity.TERMINAL_STATUSES[2]:
        frames[0]["support"]["capacity_sufficient"] = False
    elif status == capacity.TERMINAL_STATUSES[3]:
        frames[0]["graph"]["maximum_cardinality"] = 9_999
    elif status == capacity.TERMINAL_STATUSES[4]:
        vector = frames[0]["arms"]["decision"]["gate_vector"]
        vector["chamfer_le_0p8"]["passed"] = False
        frames[0]["arms"]["decision"]["passed"] = False
    elif status == capacity.TERMINAL_STATUSES[5]:
        pass
    elif status == capacity.TERMINAL_STATUSES[6]:
        vector = frames[0]["arms"]["pointwise"]["gate_vector"]
        vector["chamfer_le_0p8"]["passed"] = False
        frames[0]["arms"]["pointwise"]["passed"] = False
    elif status == capacity.TERMINAL_STATUSES[7]:
        for arm_name, key in (
            ("pointwise", "chamfer_le_0p8"),
            ("greedy", "outlier_le_0p05"),
        ):
            vector = frames[0]["arms"][arm_name]["gate_vector"]
            vector[key]["passed"] = False
            frames[0]["arms"][arm_name]["passed"] = False
    else:
        raise AssertionError(f"not a scientific terminal status: {status}")
    return frames


def _resource_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    support_frames = [
        {
            "timing": {
                "support_allocation_ns": 3,
                "support_frame_ns": 11,
            }
        }
        for _ in range(capacity.EXPECTED_TRAIN_FRAMES)
    ]
    frames = [
        {
            "resources": {
                "oracle_report_allocation_ns": 2,
                "independent_verifier_wall_ns": 2,
                "independent_verifier_serialization_ns": 1,
                "oracle_allocation_ns": 5,
                "oracle_frame_ns": 13,
                "oracle_child_pre_metric_ns": 1,
                "oracle_parent_pre_target_ns_excluded": 1,
                "oracle_parent_child_wall_ns_reported": 2,
                "metric_wall_ns": 0,
                "oracle_frame_measured_lower_bound_ns": 4,
                "oracle_frame_timing_lower_bound_valid": True,
                "torch_peak_allocated_bytes": 101,
                "torch_peak_reserved_bytes": 202,
            }
        }
        for _ in range(capacity.EXPECTED_TRAIN_FRAMES)
    ]
    support_frames[0]["timing"]["support_allocation_ns"] = 7
    support_frames[0]["timing"]["support_frame_ns"] = 17
    frames[0]["resources"]["oracle_report_allocation_ns"] = (
        capacity.MAX_ALLOCATION_NS - 10
    )
    frames[0]["resources"]["oracle_allocation_ns"] = capacity.MAX_ALLOCATION_NS - 7
    frames[0]["resources"]["oracle_frame_ns"] = capacity.MAX_FRAME_NS - 17
    support_manifest = {
        "frames": support_frames,
        "summary": {
            "maximum_torch_peak_allocated_bytes": 303,
            "maximum_torch_peak_reserved_bytes": 404,
        },
    }
    monitor = {
        "peak_process_tree_rss_bytes": capacity.MAX_HOST_BYTES,
        "peak_process_tree_swap_bytes": 0,
        "peak_summed_process_tree_nvml_bytes": 505,
        "sample_interval_target_ns": capacity.MONITOR_SAMPLE_INTERVAL_NS,
        "maximum_interval_ns": capacity.MAX_MONITOR_SAMPLE_INTERVAL_NS,
        "periodic_sample_count": 100,
        "minimum_periodic_sample_count": 100,
        "phase_peaks": {
            "support": {"child_descendant_seen": False, "samples": 2}
        },
        "cuda_overlap_detected": False,
        "maximum_cuda_pid_count": 1,
        "multiple_cuda_pid_sample_count": 0,
        "unexpected_tree_cuda_pid_sample_count": 0,
        "foreign_cuda_pid_sample_count": 0,
        "foreign_cuda_pids": {},
        "unexpected_tree_cuda_pids": {},
        "monitor_errors": [],
    }
    return monitor, support_manifest, frames


def _required_argv(tmp_path: Path) -> list[str]:
    value = str(tmp_path / "synthetic")
    return [
        "--repo",
        value,
        "--source-commit",
        SOURCE_COMMIT,
        "--manifest",
        value,
        "--scene-split",
        value,
        "--normalization",
        value,
        "--checkpoint",
        value,
        "--formal-metrics",
        value,
        "--formal-run-manifest",
        value,
        "--candidate-hash-manifest",
        value,
        "--data-root",
        value,
        "--cache-root",
        value,
        "--shared-mount-root",
        value,
        "--l40s-mount-root",
        value,
        "--shared-evidence-parent",
        str(tmp_path / "shared-evidence"),
        "--local-evidence-parent",
        str(tmp_path / "local-evidence"),
        "--transaction-parent",
        str(tmp_path / "transactions"),
    ]


def _preflight_args(tmp_path: Path) -> argparse.Namespace:
    return capacity.build_parser().parse_args(
        [*_required_argv(tmp_path), "--preflight-only"]
    )


def _phase_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo=tmp_path / "repo",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "target-cache",
        manifest=tmp_path / "manifest.json",
        scene_split=tmp_path / "scene-split.json",
        normalization=tmp_path / "normalization.json",
        checkpoint=tmp_path / "checkpoint.pt",
        candidate_hash_manifest=tmp_path / "candidate-hashes.json",
        source_commit=SOURCE_COMMIT,
        physical_gpu=2,
        gpu_uuid=GPU["uuid"],
        support_timeout_seconds=1.0,
        frame_child_timeout_seconds=1.0,
    )


def test_canonical_json_and_exclusive_publication_are_byte_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = {"z": True, "ascii": "ok", "escaped": "\u96f7\u8fbe", "n": 2}
    expected = (
        b'{"ascii":"ok","escaped":"\\u96f7\\u8fbe","n":2,"z":true}\n'
    )
    assert capacity.canonical_json_bytes(document) == expected
    with pytest.raises(ValueError):
        capacity.canonical_json_bytes({"not_finite": float("nan")})

    canonical_path = tmp_path / "canonical.json"
    assert capacity.atomic_write_json(canonical_path, document) == expected
    assert canonical_path.read_bytes() == expected

    immutable_path = tmp_path / "immutable.bin"
    rename_observations: list[tuple[Path, Path, str]] = []
    original_rename = capacity._rename_noreplace

    def forbidden_hard_link(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("hard-link publication is forbidden")

    def observed_rename(staging: Path, destination: Path) -> None:
        staging_payload = staging.read_bytes()
        assert staging_payload == b"first"
        assert capacity.sha256_file(staging) == capacity.sha256_bytes(b"first")
        assert not os.path.lexists(destination)
        rename_observations.append(
            (staging, destination, capacity.sha256_bytes(staging_payload))
        )
        original_rename(staging, destination)

    monkeypatch.setattr(os, "link", forbidden_hard_link)
    monkeypatch.setattr(capacity, "_rename_noreplace", observed_rename)
    capacity.exclusive_write_bytes(immutable_path, b"first")
    inode = immutable_path.stat().st_ino
    with pytest.raises(FileExistsError):
        capacity.exclusive_write_bytes(immutable_path, b"second")
    assert immutable_path.read_bytes() == b"first"
    assert immutable_path.stat().st_ino == inode
    assert not list(tmp_path.glob(".immutable.bin.staging-*"))
    assert len(rename_observations) == 1
    assert not rename_observations[0][0].exists()
    assert rename_observations[0][1] == immutable_path


def test_rename_noreplace_preserves_racing_destination_and_bundle_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "bundle-staging"
    staging.mkdir()
    (staging / "payload").write_bytes(b"staging")
    destination = tmp_path / "bundle-final"
    original_rename = capacity._rename_noreplace

    def racing_rename(source: Path, final: Path) -> None:
        assert source == staging
        assert not os.path.lexists(final)
        final.mkdir()
        (final / "payload").write_bytes(b"racer")
        original_rename(source, final)

    monkeypatch.setattr(capacity, "_rename_noreplace", racing_rename)
    with pytest.raises(FileExistsError):
        capacity.rename_bundle(staging, destination)
    assert (staging / "payload").read_bytes() == b"staging"
    assert (destination / "payload").read_bytes() == b"racer"


def test_bundle_seal_verification_and_payload_tamper_detection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    (root / "nested").mkdir(parents=True)
    (root / "alpha.txt").write_bytes(b"alpha")
    (root / "nested/beta.bin").write_bytes(b"beta")

    sealed = capacity.seal_bundle(root, source_commit=SOURCE_COMMIT)
    verified = bundle_verify.verify_bundle(root)
    assert verified["passed"] is True
    assert verified["bundle_root"] == sealed["bundle_root"]
    assert verified["payload_count"] == 2
    assert verified["source_commit"] == SOURCE_COMMIT

    (root / "nested/beta.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload (size|hash) changed"):
        bundle_verify.verify_bundle(root)


def test_bundle_verifier_rejects_unlisted_payload_after_seal(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "payload.txt").write_bytes(b"bound")
    capacity.seal_bundle(root, source_commit=SOURCE_COMMIT)
    (root / "late.txt").write_bytes(b"not in manifest")
    with pytest.raises(ValueError, match="unlisted or missing payloads"):
        bundle_verify.verify_bundle(root)


def test_shared_gate_vector_has_exact_frozen_25_keys_and_neutral_classes() -> None:
    export, metric, structure = _gate_inputs()
    metric["range_metrics"]["range_60_120"] = {"applicable": False}
    structure["class_metrics"]["range_60_120_later"] = {"applicable": False}

    vector = capacity.build_shared_gate_vector(export, metric, structure)
    expected = set(capacity.GLOBAL_GATE_KEYS)
    for prefix in capacity.RANGE_GATE_PREFIXES:
        expected.update(
            {
                f"{prefix}_completeness_le_1m",
                f"{prefix}_recall_1m_ge_0p8",
            }
        )
        for suffix in capacity.RETURN_GATE_SUFFIXES:
            expected.update(
                {
                    f"{prefix}_{suffix}_completeness_le_1m",
                    f"{prefix}_{suffix}_recall_1m_ge_0p8",
                }
            )

    assert len(vector) == 25
    assert set(vector) == expected
    assert all(vector[key]["applicable"] for key in capacity.GLOBAL_GATE_KEYS)
    assert vector["range_60_120_completeness_le_1m"] == {
        "applicable": False,
        "passed": True,
    }
    assert vector["range_60_120_later_recall_1m_ge_0p8"] == {
        "applicable": False,
        "passed": True,
    }
    assert capacity.gate_vector_passed(vector) is True


def test_cross_arm_applicability_must_be_identical_but_pass_bits_may_differ() -> None:
    decision = _passing_gate_vector()
    pointwise = copy.deepcopy(decision)
    greedy = copy.deepcopy(decision)
    pointwise["chamfer_le_0p8"]["passed"] = False
    capacity.validate_cross_arm_applicability(
        {"decision": decision, "pointwise": pointwise, "greedy": greedy}
    )

    greedy["range_60_120_recall_1m_ge_0p8"]["applicable"] = False
    with pytest.raises(capacity.StageFailure) as captured:
        capacity.validate_cross_arm_applicability(
            {"decision": decision, "pointwise": pointwise, "greedy": greedy}
        )
    assert captured.value.stage == "gate"
    assert captured.value.code == "cross_arm_applicability"


def test_all_eight_terminal_statuses_are_unique_and_scientific_partition_is_exact() -> None:
    assert capacity.TERMINAL_STATUSES == (
        "stda_f0_implementation_invalid",
        "stda_f0_resource_invalid",
        "stda_f0_packed_support_capacity_no_go",
        "stda_f0_graph_cardinality_no_go",
        "stda_f0_assignment_recipe_no_go",
        "stda_f0_packed_support_only",
        "stda_f0_demand_allocation_only",
        "stda_f0_assignment_utility_passed",
    )
    assert len(set(capacity.TERMINAL_STATUSES)) == 8

    observed = {
        capacity.determine_scientific_status(_frames_for_status(status))
        for status in capacity.TERMINAL_STATUSES[2:]
    }
    assert observed == set(capacity.TERMINAL_STATUSES[2:])

    formal_source = inspect.getsource(capacity.run_formal)
    assert formal_source.index('if not resources["implementation_valid"]') < (
        formal_source.index('if not resources["resource_valid"]')
    )


def test_scientific_partition_rejects_unexplained_control_failure() -> None:
    frames = _frames_for_status(capacity.TERMINAL_STATUSES[5])
    frames[0]["arms"]["pointwise"]["passed"] = False
    frames[0]["arms"]["greedy"]["passed"] = False
    with pytest.raises(capacity.StageFailure) as captured:
        capacity.determine_scientific_status(frames)
    assert captured.value.stage == "status"
    assert captured.value.code == "terminal_partition"


def test_support_and_oracle_nested_timings_drive_resource_summary() -> None:
    monitor, support_manifest, frames = _resource_inputs()
    report = capacity.summarize_resources(
        monitor_report=monitor,
        support_manifest=support_manifest,
        frames=frames,
        transaction_work_ns=capacity.MAX_TRANSACTION_NS,
    )

    assert report["allocation_core_ns_by_frame"][0] == capacity.MAX_ALLOCATION_NS
    assert report["frame_total_ns_by_frame"][0] == capacity.MAX_FRAME_NS
    assert report["maximum_allocation_core_ns"] == capacity.MAX_ALLOCATION_NS
    assert report["maximum_frame_total_ns"] == capacity.MAX_FRAME_NS
    assert report["torch_peak_allocated_bytes"] == 303
    assert report["torch_peak_reserved_bytes"] == 404
    assert report["deciding_cuda_peak_bytes"] == 505
    assert report["implementation_valid"] is True
    assert report["resource_valid"] is True
    assert frames[0]["resources"]["oracle_allocation_ns"] == (
        frames[0]["resources"]["oracle_report_allocation_ns"]
        + frames[0]["resources"]["independent_verifier_wall_ns"]
        + frames[0]["resources"]["independent_verifier_serialization_ns"]
    )

    support_manifest["frames"][0]["timing"]["support_frame_ns"] += 1
    exceeded = capacity.summarize_resources(
        monitor_report=monitor,
        support_manifest=support_manifest,
        frames=frames,
        transaction_work_ns=capacity.MAX_TRANSACTION_NS,
    )
    assert exceeded["checks"]["frame_total_le_120s_each"] is False
    assert exceeded["resource_valid"] is False

    monitor["phase_peaks"]["support"]["child_descendant_seen"] = True
    invalid = capacity.summarize_resources(
        monitor_report=monitor,
        support_manifest=support_manifest,
        frames=frames,
        transaction_work_ns=capacity.MAX_TRANSACTION_NS,
    )
    assert invalid["checks"]["support_has_no_descendants"] is False
    assert invalid["implementation_valid"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (
            lambda monitor: monitor.__setitem__(
                "maximum_interval_ns",
                capacity.MAX_MONITOR_SAMPLE_INTERVAL_NS + 1,
            ),
            "sampling_interval_le_50ms",
        ),
        (
            lambda monitor: monitor.update(
                {
                    "foreign_cuda_pids": {"999": 1},
                    "foreign_cuda_pid_sample_count": 1,
                }
            ),
            "no_foreign_cuda_pid",
        ),
        (
            lambda monitor: monitor.update(
                {
                    "maximum_cuda_pid_count": 2,
                    "multiple_cuda_pid_sample_count": 1,
                }
            ),
            "single_cuda_pid",
        ),
        (
            lambda monitor: monitor.update(
                {
                    "unexpected_tree_cuda_pids": {"123": 1},
                    "unexpected_tree_cuda_pid_sample_count": 1,
                }
            ),
            "cuda_only_in_registered_cuda_child",
        ),
    ),
)
def test_monitor_sampling_and_cuda_pid_contracts_are_implementation_gates(
    mutation: Any,
    failed_check: str,
) -> None:
    monitor, support_manifest, frames = _resource_inputs()
    mutation(monitor)
    report = capacity.summarize_resources(
        monitor_report=monitor,
        support_manifest=support_manifest,
        frames=frames,
        transaction_work_ns=1,
    )
    assert report["checks"][failed_check] is False
    assert report["implementation_valid"] is False


def test_implementation_invalid_precedes_resource_failure() -> None:
    monitor, _, _ = _resource_inputs()
    timeout = capacity.StageFailure("resource", "child_timeout", "synthetic")
    assert capacity._classify_execution_failure(
        timeout,
        monitor_report=monitor,
        resources={"implementation_valid": True},
    ) == capacity.TERMINAL_STATUSES[1]

    monitor["foreign_cuda_pids"] = {"777": 1}
    monitor["foreign_cuda_pid_sample_count"] = 1
    assert capacity._classify_execution_failure(
        timeout,
        monitor_report=monitor,
        resources={"implementation_valid": False},
    ) == capacity.TERMINAL_STATUSES[0]

    mislabeled = capacity.StageFailure("resource", "source_hash_lock_mismatch", "x")
    assert capacity._classify_execution_failure(
        mislabeled,
        monitor_report=None,
        resources=None,
    ) == capacity.TERMINAL_STATUSES[0]


def test_parser_requires_one_mode_and_main_requires_exact_formal_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = capacity.build_parser()
    base = _required_argv(tmp_path)
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *base,
                "--preflight-only",
                "--formal-launch-token",
                capacity.FORMAL_LAUNCH_TOKEN,
            ]
        )

    preflight_args = parser.parse_args([*base, "--preflight-only"])
    formal_args = parser.parse_args(
        [*base, "--formal-launch-token", capacity.FORMAL_LAUNCH_TOKEN]
    )
    assert preflight_args.preflight_only is True
    assert preflight_args.formal_launch_token is None
    assert formal_args.preflight_only is False
    assert formal_args.formal_launch_token == capacity.FORMAL_LAUNCH_TOKEN

    preflight_called = False

    def forbidden_preflight(_args: argparse.Namespace) -> dict[str, Any]:
        nonlocal preflight_called
        preflight_called = True
        raise AssertionError("wrong token reached preflight")

    monkeypatch.setattr(capacity, "run_preflight", forbidden_preflight)
    with pytest.raises(SystemExit):
        capacity.main([*base, "--formal-launch-token", "WRONG"])
    assert preflight_called is False

    monkeypatch.setattr(capacity, "run_preflight", lambda _args: {"passed": True})
    formal_calls: list[tuple[argparse.Namespace, dict[str, Any]]] = []

    def fake_formal(
        args: argparse.Namespace,
        preflight: dict[str, Any],
    ) -> dict[str, Any]:
        formal_calls.append((args, preflight))
        return {
            "terminal_status": capacity.TERMINAL_STATUSES[2],
            "bundle_root": None,
            "transaction": None,
            "failure": None,
            "formal_log_path": None,
        }

    monkeypatch.setattr(capacity, "run_formal", fake_formal)
    assert (
        capacity.main(
            [*base, "--formal-launch-token", capacity.FORMAL_LAUNCH_TOKEN]
        )
        == 0
    )
    assert len(formal_calls) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["terminal_status"] == capacity.TERMINAL_STATUSES[2]


def test_preflight_never_derives_or_opens_target_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _preflight_args(tmp_path)
    args.cache_root = tmp_path / "target-cache"
    args.cache_root.mkdir()
    tripwire = args.cache_root / "seq01_radar_00001.npz"
    tripwire.write_bytes(b"must not be opened")

    monkeypatch.setattr(
        capacity,
        "_resolved_existing_directory",
        lambda path, *, label: Path(path).resolve(strict=False),
    )
    monkeypatch.setattr(
        capacity,
        "_resolved_existing_file",
        lambda path, *, label: Path(path).resolve(strict=False),
    )
    monkeypatch.setattr(
        capacity,
        "verify_source_tree",
        lambda _repo, _commit: {"verified": True},
    )
    monkeypatch.setattr(
        capacity,
        "require_orchestrator_isolation",
        lambda: {"passed": True},
    )
    monkeypatch.setattr(
        capacity,
        "verify_frozen_inputs",
        lambda _args: ({"verified": True}, [{"position": 0}]),
    )
    monkeypatch.setattr(capacity, "require_allowed_h200", lambda _index: dict(GPU))
    monkeypatch.setattr(capacity, "environment_report", lambda _args, _gpu: {})
    monkeypatch.setattr(capacity, "_ensure_distinct_mounts", lambda _args: ({}, {}))
    monkeypatch.setattr(capacity, "_source_binding", lambda _repo: {})
    monkeypatch.setattr(
        capacity,
        "_assert_source_lock",
        lambda **kwargs: {"label": kwargs["label"], "hashes_match": True},
    )
    monkeypatch.setattr(
        capacity,
        "run_mandatory_synthetic_tests",
        lambda **_kwargs: {
            "passed": True,
            "cache_root_opened": False,
            "report_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(capacity, "_foreign_gpu_processes", lambda _index: [])

    def forbidden_cache_derivation(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("preflight derived a target-cache file")

    monkeypatch.setattr(capacity, "_target_cache_path", forbidden_cache_derivation)
    original_open = Path.open
    opened_cache_paths: list[Path] = []
    cache_root = args.cache_root.resolve()

    def guarded_open(path: Path, *open_args: object, **open_kwargs: object):
        resolved = path.resolve(strict=False)
        if resolved == cache_root or cache_root in resolved.parents:
            opened_cache_paths.append(resolved)
            raise AssertionError(f"preflight opened target cache: {resolved}")
        return original_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = capacity.run_preflight(args)
    assert opened_cache_paths == []
    assert report["target_cache_opened"] is False
    assert report["formal_launch_consumed"] is False
    assert report["passed"] is True


def test_mandatory_pytest_lock_uses_isolated_synthetic_groups_and_exact_gpu_uuid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    tests_root = repo / "code/tests"
    tests_root.mkdir(parents=True)
    source_hashes: dict[str, str] = {}
    for relative in capacity.MANDATORY_TEST_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {relative}\n", encoding="ascii")
        source_hashes[relative] = capacity.sha256_file(path)
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(capacity, "_explicit_site_packages", lambda: site_packages)
    monkeypatch.setattr(capacity, "_foreign_gpu_processes", lambda _index: [])

    calls: list[dict[str, Any]] = []

    def fake_run(command: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append({"command": tuple(command), **kwargs})
        return SimpleNamespace(returncode=0, stdout=b"synthetic pytest passed\n")

    monkeypatch.setattr(capacity.subprocess, "run", fake_run)
    report = capacity.run_mandatory_synthetic_tests(
        repo=repo,
        physical_gpu=2,
        gpu_uuid=GPU["uuid"],
        gpu_pci=GPU["pci_bus_id"],
        gpu_name=GPU["name"],
        source_hashes=source_hashes,
    )

    assert report["passed"] is True
    assert report["cache_root_opened"] is False
    assert len(report["gpu_idle_snapshots"]) == 6
    assert [call["env"]["CUDA_VISIBLE_DEVICES"] for call in calls] == [
        "",
        GPU["uuid"],
        GPU["uuid"],
    ]
    assert all(call["command"][1:4] == ("-I", "-S", "-B") for call in calls)
    assert all("no:cacheprovider" in call["command"] for call in calls)
    assert calls[2]["env"]["STDA_F0_CUDA_SMOKE"] == "1"
    assert calls[2]["env"]["STDA_EXPECTED_GPU_UUID"] == GPU["uuid"]
    assert report["candidate_700k_included"] is True
    assert report["metric_isolated_import_test_included"] is True
    assert report["explicit_metric_smoke_enabled"] is True
    assert all(
        "target-cache" not in argument and "cache-root" not in argument
        for call in calls
        for argument in call["command"]
    )


def test_source_lock_and_child_source_evidence_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"code/scripts/stda_f0_x.py": "a" * 64}
    monkeypatch.setattr(
        capacity,
        "verify_source_tree",
        lambda _repo, _commit: {"worktree_clean": True},
    )
    monkeypatch.setattr(capacity, "_source_binding", lambda _repo: dict(expected))
    lock = capacity._assert_source_lock(
        repo=tmp_path,
        source_commit=SOURCE_COMMIT,
        expected_hashes=expected,
        label="synthetic",
    )
    assert lock["hashes_match"] is True

    monkeypatch.setattr(
        capacity,
        "_source_binding",
        lambda _repo: {"code/scripts/stda_f0_x.py": "b" * 64},
    )
    with pytest.raises(capacity.StageFailure) as captured:
        capacity._assert_source_lock(
            repo=tmp_path,
            source_commit=SOURCE_COMMIT,
            expected_hashes=expected,
            label="changed",
        )
    assert captured.value.stage == "implementation"
    assert captured.value.code == "source_hash_lock_mismatch"

    evidence = capacity._validate_child_source_evidence(
        child_name="synthetic",
        observed_hashes=expected,
        expected_hashes=expected,
        required_paths=tuple(expected),
    )
    assert evidence["passed"] is True
    with pytest.raises(capacity.StageFailure, match="changed"):
        capacity._validate_child_source_evidence(
            child_name="synthetic",
            observed_hashes={"code/scripts/stda_f0_x.py": "b" * 64},
            expected_hashes=expected,
            required_paths=tuple(expected),
        )


def test_locked_source_set_covers_all_stda_scripts_critical_models_and_tests() -> None:
    repo = Path(capacity.__file__).resolve().parents[2]
    locked = set(capacity._locked_source_paths(repo))
    required = {
        *capacity.SUPPORT_CHILD_SOURCE_PATHS,
        *capacity.SUPPORT_ORCHESTRATOR_SOURCE_PATHS,
        *capacity.ORACLE_CHILD_SOURCE_PATHS,
        *capacity.VERIFY_CHILD_SOURCE_PATHS,
        *capacity.MANDATORY_TEST_PATHS,
        "code/eval/dense_geometry.py",
        "code/eval/vrh_f0_metrics.py",
    }
    assert required <= locked
    for pattern in (
        "code/eval/stda_f0_*.py",
        "code/scripts/stda_f0_*.py",
        "code/tests/test_stda_f0_*.py",
    ):
        assert {
            str(path.relative_to(repo)) for path in repo.glob(pattern) if path.is_file()
        } <= locked


def test_support_manifest_validates_commitment_and_manifest_entry_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_root = tmp_path / "support"
    manifest, records, source_hashes = _synthetic_support_evidence(support_root)
    monkeypatch.setattr(capacity, "EXPECTED_TRAIN_FRAMES", 1)
    monkeypatch.setattr(capacity, "_freeze_tree_readonly", lambda _root: None)

    observed = capacity._validate_support_manifest(
        support_root,
        records,
        source_commit=SOURCE_COMMIT,
        expected_source_hashes=source_hashes,
    )
    assert observed == manifest


def test_support_manifest_rejects_missing_manifest_entry_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_root = tmp_path / "support"
    manifest, records, source_hashes = _synthetic_support_evidence(support_root)
    del manifest["frames"][0]["manifest_entry_evidence"]
    _write_canonical_record(support_root / "support_manifest.json", manifest)
    monkeypatch.setattr(capacity, "EXPECTED_TRAIN_FRAMES", 1)
    monkeypatch.setattr(capacity, "_freeze_tree_readonly", lambda _root: None)

    with pytest.raises(capacity.StageFailure) as captured:
        capacity._validate_support_manifest(
            support_root,
            records,
            source_commit=SOURCE_COMMIT,
            expected_source_hashes=source_hashes,
        )
    assert captured.value.code == "support_sidecar_record_absent"


def test_support_manifest_rejects_semantically_rehashed_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_root = tmp_path / "support"
    manifest, records, source_hashes = _synthetic_support_evidence(support_root)
    frame = manifest["frames"][0]
    frame_root = support_root / "frames/seq01_radar00002"
    commitment_path = frame_root / "support_record.json"
    commitment = capacity.canonical_json_file(commitment_path)
    commitment["support_sha256"] = "f" * 64
    commitment_record = _write_canonical_record(commitment_path, commitment)
    commitment_record["relative_path"] = (
        "frames/seq01_radar00002/support_record.json"
    )
    frame["frame_commitment"] = commitment_record
    entry_path = frame_root / "support_manifest_entry.json"
    entry = capacity.canonical_json_file(entry_path)
    entry["frame_commitment"] = commitment_record
    entry_record = _write_canonical_record(entry_path, entry)
    entry_record["relative_path"] = (
        "frames/seq01_radar00002/support_manifest_entry.json"
    )
    frame["manifest_entry_evidence"] = entry_record
    _write_canonical_record(support_root / "support_manifest.json", manifest)
    monkeypatch.setattr(capacity, "EXPECTED_TRAIN_FRAMES", 1)
    monkeypatch.setattr(capacity, "_freeze_tree_readonly", lambda _root: None)

    with pytest.raises(capacity.StageFailure) as captured:
        capacity._validate_support_manifest(
            support_root,
            records,
            source_commit=SOURCE_COMMIT,
            expected_source_hashes=source_hashes,
        )
    assert captured.value.code == "support_frame_commitment_binding"


def test_support_manifest_entry_must_bind_exact_commitment_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_root = tmp_path / "support"
    manifest, records, source_hashes = _synthetic_support_evidence(support_root)
    frame = manifest["frames"][0]
    entry_path = support_root / "frames/seq01_radar00002/support_manifest_entry.json"
    entry = capacity.canonical_json_file(entry_path)
    entry["frame_commitment"] = {
        **entry["frame_commitment"],
        "sha256": "f" * 64,
    }
    entry_record = _write_canonical_record(entry_path, entry)
    entry_record["relative_path"] = (
        "frames/seq01_radar00002/support_manifest_entry.json"
    )
    frame["manifest_entry_evidence"] = entry_record
    _write_canonical_record(support_root / "support_manifest.json", manifest)
    monkeypatch.setattr(capacity, "EXPECTED_TRAIN_FRAMES", 1)
    monkeypatch.setattr(capacity, "_freeze_tree_readonly", lambda _root: None)

    with pytest.raises(capacity.StageFailure) as captured:
        capacity._validate_support_manifest(
            support_root,
            records,
            source_commit=SOURCE_COMMIT,
            expected_source_hashes=source_hashes,
        )
    assert captured.value.code == "support_manifest_entry_binding"


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("opened", False),
        ("array_loader_called", True),
        ("cache_arrays_read", ["target_xyz_confidence"]),
        ("target_array_materialized", True),
        ("cache_sha256", "bad"),
    ),
)
def test_capacity_no_go_requires_read_once_cache_provenance(
    tmp_path: Path,
    field: str,
    invalid: Any,
) -> None:
    output = tmp_path / field
    output.mkdir()
    expected_source_hashes = {
        path: f"{index + 1:064x}"
            for index, path in enumerate(capacity.ORACLE_RUNTIME_SOURCE_PATHS)
    }
    stat_record = {
        "device": 1,
        "inode": 2,
        "mode": 0o100600,
        "size_bytes": 123,
        "mtime_ns": 3,
        "ctime_ns": 4,
    }
    target = {
        "opened": True,
        "array_loader_called": False,
        "cache_arrays_read": [],
        "target_array_materialized": False,
        "path_read_calls": 1,
        "immutable_bytes_materialized": True,
        "hash_consumed_same_payload": True,
        "clock": "time.perf_counter_ns",
        "read_started_perf_counter_ns": 100,
        "cache_sha256": "a" * 64,
        "cache_size_bytes": 123,
        "stat_before": stat_record,
        "stat_after": dict(stat_record),
    }
    report = {
        "schema": "stda_f0_oracle_frame_v1",
        "protocol_sha256": capacity.PROTOCOL_SHA256,
        "protocol_freeze_commit": capacity.PROTOCOL_FREEZE_COMMIT,
        "status": "stda_f0_packed_support_capacity_no_go",
        "frame": {"frame_key": "seq01/radar00002"},
        "cpu_only": {
            "cuda_visible_devices": "",
            "forbidden_cuda_modules_loaded": [],
            "geometry_evaluator_used": False,
            "candidate_reconstruction_used": False,
        },
        "target": target,
        "source_sha256": expected_source_hashes,
    }
    report_path = output / "oracle_report.json"
    report_record = _write_canonical_record(report_path, report)
    complete = {
        "schema": "stda_f0_oracle_frame_complete_v1",
        "oracle_report_path": "oracle_report.json",
        "oracle_report_bytes": report_record["size_bytes"],
        "oracle_report_sha256": report_record["sha256"],
        "status": report["status"],
    }
    _write_canonical_record(output / "ORACLE_COMPLETE.json", complete)
    loaded = capacity._load_oracle_report(
        output,
        expected_source_hashes=expected_source_hashes,
    )
    assert loaded["target"] == target

    target[field] = invalid
    report_record = _write_canonical_record(report_path, report)
    complete["oracle_report_bytes"] = report_record["size_bytes"]
    complete["oracle_report_sha256"] = report_record["sha256"]
    _write_canonical_record(output / "ORACLE_COMPLETE.json", complete)
    with pytest.raises(capacity.StageFailure) as captured:
        capacity._load_oracle_report(
            output,
            expected_source_hashes=expected_source_hashes,
        )
    assert captured.value.code == "oracle_target_boundary"


def test_oracle_child_timing_receipt_is_authoritative(tmp_path: Path) -> None:
    oracle = {
        "status": "stda_f0_packed_support_capacity_no_go",
        "frame": {"frame_key": "seq01/radar00002"},
        "timings_ns": {
            "oracle_frame_started_perf_counter_ns": 900,
            "oracle_child_pre_metric_ended_perf_counter_ns": 901,
            "oracle_child_pre_metric_ns": 1,
        },
    }
    timing = {
        "schema": "stda_f0_oracle_child_timing_v2",
        "protocol_sha256": capacity.PROTOCOL_SHA256,
        "protocol_freeze_commit": capacity.PROTOCOL_FREEZE_COMMIT,
        "status": oracle["status"],
        "frame_key": oracle["frame"]["frame_key"],
        "oracle_frame_started_perf_counter_ns": 100,
        "oracle_child_pre_metric_ended_perf_counter_ns": 140,
        "oracle_child_pre_metric_ns": 40,
        "authoritative_oracle_frame_ns": False,
        "parent_is_sole_oracle_frame_authority": True,
        "support_reverification_excluded": True,
        "timing_receipt_publication_excluded": True,
    }
    _write_canonical_record(tmp_path / "ORACLE_FRAME_TIMING.json", timing)

    observed = capacity._load_oracle_child_timing(tmp_path, oracle=oracle)
    assert observed["oracle_frame_started_perf_counter_ns"] == 100
    assert observed["oracle_child_pre_metric_ended_perf_counter_ns"] == 140
    assert observed["oracle_child_pre_metric_ns"] == 40


def _verifier_parent_timing_fixture(
    tmp_path: Path,
    *,
    mutation: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    output = tmp_path / "independent_verification.json"
    log = tmp_path / "independent_verification.log"
    module_started_ns = 110
    verification = {
        "schema": "stda_f0_independent_frame_verification_v1",
        "passed": True,
        "timing_authority": {
            "parent_process_wall_authoritative": True,
            "internal_process_wall_emitted_after_output_rehash": True,
            "module_wall_started_perf_counter_ns": module_started_ns,
        },
    }
    if mutation == "report-authority":
        verification["timing_authority"]["parent_process_wall_authoritative"] = False
    elif mutation == "module-before-parent":
        verification["timing_authority"]["module_wall_started_perf_counter_ns"] = 99
    _write_canonical_record(output, verification)
    summary = {
        "schema": "stda_f0_independent_frame_verification_v1",
        "output": str(output.absolute()),
        "output_sha256": capacity.sha256_file(output),
        "output_size_bytes": output.stat().st_size,
        "parent_process_wall_authoritative": True,
        "internal_full_wall_through_output_rehash_ns": 20,
        "internal_full_wall_boundary": (
            "verifier_module_start_through_output_fsync_and_independent_rehash"
        ),
        "passed": True,
    }
    if mutation == "output-hash":
        summary["output_sha256"] = "0" * 64
    elif mutation == "rehash-after-parent":
        summary["internal_full_wall_through_output_rehash_ns"] = 31
    _write_canonical_record(log, summary)
    return output, log, verification


def test_verifier_parent_timing_receipt_is_independently_bound(tmp_path: Path) -> None:
    output, log, verification = _verifier_parent_timing_fixture(tmp_path)
    evidence = capacity._load_verifier_parent_timing_evidence(
        verification_path=output,
        log_path=log,
        verification=verification,
        parent_started_ns=100,
        parent_completed_ns=140,
    )
    assert evidence["parent_process_wall_authoritative"] is True
    assert evidence["parent_wall_ns"] == 40
    assert evidence["child_output_rehash_completed_perf_counter_ns"] == 130
    assert evidence["passed"] is True


@pytest.mark.parametrize(
    "mutation",
    ("report-authority", "module-before-parent", "output-hash", "rehash-after-parent"),
)
def test_verifier_parent_timing_receipt_rejects_hostile_claims(
    tmp_path: Path,
    mutation: str,
) -> None:
    output, log, verification = _verifier_parent_timing_fixture(
        tmp_path,
        mutation=mutation,
    )
    with pytest.raises(capacity.StageFailure) as captured:
        capacity._load_verifier_parent_timing_evidence(
            verification_path=output,
            log_path=log,
            verification=verification,
            parent_started_ns=100,
            parent_completed_ns=140,
        )
    assert captured.value.code == "verifier_timing_authority_invalid"


def test_transaction_boundary_stops_monitor_before_clock_and_record_publication() -> None:
    events: list[str] = []

    class SyntheticMonitor:
        def stop(self) -> dict[str, Any]:
            events.append("monitor.stop")
            return {"monitor": "stopped"}

    original = capacity.time.perf_counter_ns
    try:
        capacity.time.perf_counter_ns = lambda: (
            events.append("clock") or 1_000
        )
        report, elapsed, boundary = capacity._stop_transaction_monitor(
            SyntheticMonitor(), 100
        )
    finally:
        capacity.time.perf_counter_ns = original
    assert events == ["monitor.stop", "clock"]
    assert report == {"monitor": "stopped"}
    assert elapsed == 900
    assert boundary == 1_000


def test_isolation_source_and_atomic_publish_contracts_are_structurally_bound() -> None:
    preflight_source = inspect.getsource(capacity.run_preflight)
    environment_source = inspect.getsource(capacity.environment_report)
    formal_source = inspect.getsource(capacity.run_formal)
    publisher_source = inspect.getsource(capacity.exclusive_write_bytes)
    bundle_source = inspect.getsource(capacity.rename_bundle)
    transaction_source = inspect.getsource(capacity.publish_transaction_record)
    failure_source = inspect.getsource(capacity.publish_failure_record)
    local_verifier_source = inspect.getsource(capacity.run_local_bundle_verifier)
    remote_verifier_source = inspect.getsource(capacity.run_l40s_bundle_verifier)

    assert preflight_source.index("require_orchestrator_isolation") < preflight_source.index(
        "CUDA_VISIBLE_DEVICES"
    )
    for source in (local_verifier_source, remote_verifier_source, environment_source):
        assert source.index('"-I"') < source.index('"-S"') < source.index('"-B"')
    assert '"python3"' in remote_verifier_source
    assert "_stop_transaction_monitor" in formal_source
    assert formal_source.index("after_final_bundle_reverification") < formal_source.index(
        "_stop_transaction_monitor"
    )
    assert "_rename_noreplace(temporary, path)" in publisher_source
    assert "os.link" not in publisher_source
    assert "_rename_noreplace(staging, destination)" in bundle_source
    assert "exclusive_write_bytes(path, payload)" in transaction_source
    assert "exclusive_write_bytes(path, payload)" in failure_source
    assert "os.link" not in inspect.getsource(capacity)


def test_formal_launch_marker_is_exclusive_and_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local"
    transactions = tmp_path / "transactions"
    for path in (shared, local, transactions):
        path.mkdir()
    args = argparse.Namespace(
        shared_evidence_parent=shared,
        local_evidence_parent=local,
        transaction_parent=transactions,
        source_commit=SOURCE_COMMIT,
        physical_gpu=2,
    )

    marker, first_document = capacity._create_formal_launch_marker(
        args,
        {"gpu": GPU},
    )
    first_bytes = marker.read_bytes()
    first_log = Path(first_document["log_path"])
    assert first_bytes == capacity.canonical_json_bytes(first_document)
    assert first_log.read_bytes() == b""

    with pytest.raises(capacity.StageFailure) as captured:
        capacity._create_formal_launch_marker(args, {"gpu": GPU})
    assert captured.value.stage == "launch"
    assert captured.value.code == "prior_formal_evidence"
    assert marker.read_bytes() == first_bytes
    assert first_log.read_bytes() == b""


def test_orchestrator_source_has_no_numpy_or_torch_import_or_symbol_use() -> None:
    source = inspect.getsource(capacity)
    tree = ast.parse(source)
    assert source.count("records[relative] = sha256_file(path)") == 1
    assert source.count('if os.environ.get("CUDA_VISIBLE_DEVICES") != "":') == 1
    assert source.count('"logical_core_count": os.cpu_count(),') == 1
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert imported_roots.isdisjoint({"numpy", "torch"})
    assert loaded_names.isdisjoint({"numpy", "torch"})
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "__import__"
        for node in ast.walk(tree)
    )


def test_phase_commands_enforce_serial_cuda_visibility_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _phase_args(tmp_path)
    args.repo.mkdir()
    args.data_root.mkdir()
    args.cache_root.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(capacity, "event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        capacity,
        "_assert_source_lock",
        lambda **kwargs: {"label": kwargs["label"], "hashes_match": True},
    )
    monkeypatch.setattr(
        capacity,
        "_remaining_timeout_seconds",
        lambda _started, requested: float(requested),
    )
    monkeypatch.setattr(
        capacity,
        "_validate_support_manifest",
        lambda *_args, **_kwargs: {"frames": []},
    )

    def fake_support_child(command: Any, **kwargs: Any) -> int:
        calls.append({"command": tuple(command), **kwargs})
        capacity.atomic_write_json(
            kwargs["log_path"],
            {
                "schema": "stda_f0_support_child_completion_v1",
                "completed": True,
                "frame_count": capacity.EXPECTED_TRAIN_FRAMES,
                "ground_truth_accessed": False,
                "support_target_input": False,
            },
        )
        (staging / "support/support_manifest.json").write_bytes(b"synthetic")
        return 0

    monkeypatch.setattr(capacity, "run_child", fake_support_child)
    capacity._run_support_child(
        args=args,
        staging=staging,
        monitor=object(),
        transaction_started_ns=0,
        transaction_clock=None,
        records=[],
        expected_source_hashes={},
    )
    support_call = calls.pop()
    assert support_call["phase"] == "support"
    assert support_call["cuda_visible"] is True
    assert support_call["environment"]["CUDA_VISIBLE_DEVICES"] == GPU["uuid"]
    assert support_call["environment"]["STDA_F0_PHASE"] == "support"
    assert support_call["command"][1:4] == ("-I", "-S", "-B")
    assert support_call["command"][-2:] == ("--device", "cuda:0")
    assert "--target-cache" not in support_call["command"]

    support_root = staging / "support"
    support_frame = {"support": {"support_sha256": "b" * 64}}
    record = {"position": 0, "sequence": 1, "radar_index": 2}
    frame_key = "seq01/radar00002"
    oracle = {
        "frame": {"frame_key": frame_key},
        "status": "stda_f0_oracle_ready_for_cuda_metrics",
        "fit_binding": {
            "fit_evidence_files_sha256": {
                "target_xyz_confidence.npy": "c" * 64,
            }
        },
        "exports": {
            name: {"npy": {"sha256": character * 64}}
            for name, character in (
                ("decision", "d"),
                ("packed_pointwise", "e"),
                ("round_robin_greedy", "f"),
            )
        },
    }
    verification = {
        "passed": True,
        "protocol_sha256": capacity.PROTOCOL_SHA256,
        "protocol_freeze_commit": capacity.PROTOCOL_FREEZE_COMMIT,
        "frame": {"frame_key": frame_key},
        "oracle_status": oracle["status"],
        "source_sha256": {
            path: character * 64
            for path, character in zip(
                capacity.VERIFY_CHILD_SOURCE_PATHS,
                ("1", "2"),
                strict=True,
            )
        },
    }
    metric = {
        "frame_key": frame_key,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": capacity.PROTOCOL_SHA256,
    }
    monkeypatch.setattr(
        capacity,
        "_support_frame_directory",
        lambda _root, _frame: support_root,
    )
    monkeypatch.setattr(
        capacity,
        "_load_oracle_report",
        lambda _path, **_kwargs: oracle,
    )
    oracle_child_window: dict[str, int] = {}
    monkeypatch.setattr(
        capacity,
        "_load_oracle_child_timing",
        lambda _path, **_kwargs: {
            "oracle_frame_started_perf_counter_ns": oracle_child_window["start"],
            "oracle_child_pre_metric_ended_perf_counter_ns": (
                oracle_child_window["end"]
            ),
            "oracle_child_pre_metric_ns": (
                oracle_child_window["end"] - oracle_child_window["start"]
            ),
            "timing_receipt_sha256": "9" * 64,
        },
    )

    def fake_canonical_record(path: Path, *, schema: str | None = None) -> dict[str, Any]:
        if path.name == "independent_verification.json":
            assert schema == "stda_f0_independent_frame_verification_v1"
            return verification
        if path.name == "independent_verification.log":
            assert schema == "stda_f0_independent_frame_verification_v1"
            return capacity.canonical_json_file(path)
        if path.name == "cuda_metrics.json":
            assert schema == "stda_f0_metric_phase_v1"
            return metric
        if path.name == "oracle_frame_timing.json":
            assert schema == "stda_f0_oracle_frame_timing_v2"
            return capacity.canonical_json_file(path)
        raise AssertionError(f"unexpected canonical record: {path}")

    monkeypatch.setattr(capacity, "_canonical_record", fake_canonical_record)

    normalization_calls: list[dict[str, Any]] = []

    def normalized_frame(**kwargs: Any) -> dict[str, Any]:
        normalization_calls.append(kwargs)
        return {
            "frame_key": frame_key,
            "support": {"capacity_sufficient": True},
            "graph": {"maximum_cardinality": 10_000},
            "arms": {},
            "resources": {
                "oracle_allocation_ns": 0,
                "oracle_frame_ns": kwargs["oracle_frame_ns"],
                "torch_peak_allocated_bytes": 0,
                "torch_peak_reserved_bytes": 0,
            },
        }

    monkeypatch.setattr(capacity, "_normalize_scientific_frame", normalized_frame)

    def fake_oracle_children(command: Any, **kwargs: Any) -> int:
        command_tuple = tuple(command)
        calls.append({"command": command_tuple, **kwargs})
        phase = kwargs["phase"]
        if phase.startswith("oracle_"):
            oracle_child_window["start"] = capacity.time.perf_counter_ns()
            oracle_dir = Path(command_tuple[command_tuple.index("--output-dir") + 1])
            oracle_dir.mkdir(parents=True)
            (oracle_dir / "oracle_report.json").write_bytes(b"oracle")
            oracle_child_window["end"] = capacity.time.perf_counter_ns()
        else:
            output = Path(command_tuple[command_tuple.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(phase.encode("ascii"))
            if phase.startswith("verify_"):
                module_started_ns = capacity.time.perf_counter_ns()
                verification["timing_authority"] = {
                    "parent_process_wall_authoritative": True,
                    "internal_process_wall_emitted_after_output_rehash": True,
                    "module_wall_started_perf_counter_ns": module_started_ns,
                }
                capacity.atomic_write_json(
                    kwargs["log_path"],
                    {
                        "schema": "stda_f0_independent_frame_verification_v1",
                        "output": str(output.absolute()),
                        "output_sha256": capacity.sha256_file(output),
                        "output_size_bytes": output.stat().st_size,
                        "parent_process_wall_authoritative": True,
                        "internal_full_wall_through_output_rehash_ns": 1,
                        "internal_full_wall_boundary": (
                            "verifier_module_start_through_output_fsync_and_"
                            "independent_rehash"
                        ),
                        "passed": True,
                    },
                )
        return 0

    monkeypatch.setattr(capacity, "run_child", fake_oracle_children)
    normalized = capacity._run_one_oracle_frame(
        args=args,
        staging=staging,
        support_root=support_root,
        support_frame=support_frame,
        record=record,
        monitor=object(),
        transaction_started_ns=0,
        gpu=GPU,
        expected_source_hashes={
            path: character * 64
            for path, character in zip(
                capacity.VERIFY_CHILD_SOURCE_PATHS,
                ("1", "2"),
                strict=True,
            )
        },
    )
    assert normalized["frame_key"] == frame_key
    assert [call["phase"] for call in calls] == [
        "oracle_00",
        "verify_00",
        "metric_00",
    ]
    for call in calls[:2]:
        assert call["cuda_visible"] is False
        assert call["environment"]["CUDA_VISIBLE_DEVICES"] == ""
        assert call["environment"]["STDA_F0_PHASE"] == call["phase"]
        assert call["command"][1:4] == ("-I", "-S", "-B")
    metric_call = calls[2]
    assert metric_call["cuda_visible"] is True
    assert metric_call["environment"]["CUDA_VISIBLE_DEVICES"] == GPU["uuid"]
    assert metric_call["environment"]["STDA_F0_PHASE"] == "metric_00"
    assert metric_call["command"][1:4] == ("-I", "-S", "-B")
    assert "--target-cache" in calls[0]["command"]
    assert "--target-cache" in calls[1]["command"]
    assert calls[0]["command"][calls[0]["command"].index("--target-cache") + 1] == (
        calls[1]["command"][calls[1]["command"].index("--target-cache") + 1]
    )
    assert "--target" in metric_call["command"]
    assert len(normalization_calls) == 2
    final_timing = normalization_calls[-1]
    measured_components = (
        final_timing["oracle_child_pre_metric_ns"]
        + final_timing["independent_verifier_wall_ns"]
        + final_timing["independent_verifier_serialization_ns"]
        + final_timing["metric_wall_ns"]
    )
    assert final_timing["oracle_frame_ns"] >= measured_components
    assert normalized["resources"]["oracle_frame_ns"] == final_timing[
        "oracle_frame_ns"
    ]
    assert normalized["resources"]["timing_receipt_publication_included"] is True
    timing_document = capacity.canonical_json_file(
        staging / "frames/seq01_radar00002/oracle_frame_timing.json"
    )
    assert timing_document["receipt_publication_included_in_oracle_frame_ns"] is True
    assert timing_document["independent_verifier_parent_timing"]["passed"] is True
    assert timing_document["independent_verifier_parent_timing"][
        "parent_process_wall_authoritative"
    ] is True
    assert timing_document["self_referential_end_fields_omitted"] is True
    assert "oracle_frame_ns" not in timing_document
    assert "evidence_boundary_perf_counter_ns" not in timing_document


def test_monitor_treats_historical_owned_pid_outside_current_tree_as_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_pid = 10
    reused_pid = 99
    monitor = object.__new__(capacity.ProcessTreeMonitor)
    monitor.root_pid = root_pid
    monitor._nvml = SimpleNamespace(
        process_memory=lambda: {reused_pid: 4_096},
    )
    monitor._lock = capacity.threading.Lock()
    monitor._active = None
    monitor._owned_child_pids = {reused_pid}
    monitor._phase_peaks = {}
    monitor.sample_count = 0
    monitor.periodic_sample_count = 0
    monitor.peak_rss_bytes = 0
    monitor.peak_swap_bytes = 0
    monitor.peak_nvml_bytes = 0
    monitor.maximum_process_count = 0
    monitor.maximum_sample_duration_ns = 0
    monitor.maximum_process_tree_sample_ns = 0
    monitor.maximum_nvml_query_ns = 0
    monitor.maximum_cuda_pid_count = 0
    monitor.multiple_cuda_pid_sample_count = 0
    monitor.unexpected_tree_cuda_pid_sample_count = 0
    monitor.foreign_cuda_pid_sample_count = 0
    monitor.observed_cuda_pids = {}
    monitor.foreign_cuda_pids = {}
    monitor.unexpected_tree_cuda_pids = {}
    monkeypatch.setattr(capacity, "process_tree", lambda _pid: {root_pid})
    monkeypatch.setattr(capacity, "process_memory", lambda _pid: (1_024, 0))

    monitor._sample(periodic=True)

    assert monitor.foreign_cuda_pid_sample_count == 1
    assert monitor.foreign_cuda_pids == {reused_pid: 4_096}
    assert monitor.maximum_cuda_pid_count == 1


def test_h200_gpu2_uuid_nvml_process_tree_monitor_captures_child(
    tmp_path: Path,
) -> None:
    repo = Path(capacity.__file__).resolve().parents[2]
    monitor = capacity.ProcessTreeMonitor(os.getpid(), GPU["uuid"])
    monitor.start()
    try:
        capacity.run_child(
            (
                sys.executable,
                "-B",
                "-c",
                (
                    "import time,torch;"
                    "assert torch.cuda.device_count()==1;"
                    "assert torch.cuda.get_device_name(0)=='NVIDIA H200 NVL';"
                    "x=torch.ones((1024,1024),device='cuda:0');"
                    "torch.cuda.synchronize();time.sleep(0.40);"
                    "print(int(x.numel()))"
                ),
            ),
            environment=capacity.child_environment(
                repo=repo,
                physical_gpu=GPU["uuid"],
                phase="synthetic_gpu",
            ),
            log_path=tmp_path / "synthetic_gpu.log",
            phase="synthetic_gpu",
            cuda_visible=True,
            monitor=monitor,
            timeout_seconds=20.0,
        )
    finally:
        report = monitor.stop()
    peak = report["phase_peaks"]["synthetic_gpu"]
    assert peak["peak_nvml_bytes"] > 0
    assert report["peak_summed_process_tree_nvml_bytes"] >= peak["peak_nvml_bytes"]
    assert report["peak_process_tree_swap_bytes"] == 0
    assert report["cuda_overlap_detected"] is False
    assert report["sample_interval_target_ns"] == capacity.MONITOR_SAMPLE_INTERVAL_NS
    assert report["maximum_interval_ns"] <= capacity.MAX_MONITOR_SAMPLE_INTERVAL_NS
    assert report["maximum_cuda_pid_count"] <= 1
    assert report["foreign_cuda_pids"] == {}
    assert report["multiple_cuda_pid_sample_count"] == 0
    assert report["unexpected_tree_cuda_pids"] == {}
    assert report["monitor_errors"] == []
