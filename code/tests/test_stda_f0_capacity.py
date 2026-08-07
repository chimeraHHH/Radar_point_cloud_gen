from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import socket
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
                "oracle_allocation_ns": 5,
                "oracle_frame_ns": 13,
                "torch_peak_allocated_bytes": 101,
                "torch_peak_reserved_bytes": 202,
            }
        }
        for _ in range(capacity.EXPECTED_TRAIN_FRAMES)
    ]
    support_frames[0]["timing"]["support_allocation_ns"] = 7
    support_frames[0]["timing"]["support_frame_ns"] = 17
    frames[0]["resources"]["oracle_allocation_ns"] = (
        capacity.MAX_ALLOCATION_NS - 7
    )
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
        "phase_peaks": {"support": {"child_descendant_seen": False}},
        "cuda_overlap_detected": False,
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
        support_timeout_seconds=1.0,
        frame_child_timeout_seconds=1.0,
    )


def test_canonical_json_and_exclusive_publication_are_byte_strict(
    tmp_path: Path,
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
    capacity.exclusive_write_bytes(immutable_path, b"first")
    inode = immutable_path.stat().st_ino
    with pytest.raises(FileExistsError):
        capacity.exclusive_write_bytes(immutable_path, b"second")
    assert immutable_path.read_bytes() == b"first"
    assert immutable_path.stat().st_ino == inode
    assert not list(tmp_path.glob(".immutable.bin.staging-*"))


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
        "verify_frozen_inputs",
        lambda _args: ({"verified": True}, [{"position": 0}]),
    )
    monkeypatch.setattr(capacity, "require_allowed_h200", lambda _index: dict(GPU))
    monkeypatch.setattr(capacity, "environment_report", lambda _args, _gpu: {})
    monkeypatch.setattr(capacity, "_ensure_distinct_mounts", lambda _args: ({}, {}))
    monkeypatch.setattr(capacity, "_source_binding", lambda _repo: {})
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
        records=[],
    )
    support_call = calls.pop()
    assert support_call["phase"] == "support"
    assert support_call["cuda_visible"] is True
    assert support_call["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert support_call["environment"]["STDA_F0_PHASE"] == "support"
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
    verification = {"passed": True}
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
    monkeypatch.setattr(capacity, "_load_oracle_report", lambda _path: oracle)

    def fake_canonical_record(path: Path, *, schema: str | None = None) -> dict[str, Any]:
        if path.name == "independent_verification.json":
            assert schema == "stda_f0_independent_frame_verification_v1"
            return verification
        if path.name == "cuda_metrics.json":
            assert schema == "stda_f0_metric_phase_v1"
            return metric
        raise AssertionError(f"unexpected canonical record: {path}")

    monkeypatch.setattr(capacity, "_canonical_record", fake_canonical_record)

    def normalized_frame(**_kwargs: Any) -> dict[str, Any]:
        return {
            "frame_key": frame_key,
            "support": {"capacity_sufficient": True},
            "graph": {"maximum_cardinality": 10_000},
            "arms": {},
            "resources": {
                "oracle_allocation_ns": 0,
                "oracle_frame_ns": 0,
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
            oracle_dir = Path(command_tuple[command_tuple.index("--output-dir") + 1])
            oracle_dir.mkdir(parents=True)
            (oracle_dir / "oracle_report.json").write_bytes(b"oracle")
        else:
            output = Path(command_tuple[command_tuple.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(phase.encode("ascii"))
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
    metric_call = calls[2]
    assert metric_call["cuda_visible"] is True
    assert metric_call["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert metric_call["environment"]["STDA_F0_PHASE"] == "metric_00"
    assert "--target-cache" in calls[0]["command"]
    assert "--target-cache" not in calls[1]["command"]
    assert "--target" in metric_call["command"]
