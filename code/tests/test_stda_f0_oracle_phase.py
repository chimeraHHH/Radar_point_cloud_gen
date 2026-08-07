from __future__ import annotations

import ast
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.stda_f0_assignment_replay as replay_module
import scripts.stda_f0_oracle_phase as oracle_module
from eval.stda_f0_structure import (
    StructuralDomain,
    evaluate_structure,
    freeze_structural_inputs,
)
from eval.stda_f0_support import (
    CAPACITY_NO_GO_STATUS,
    build_packed_support,
    serialize_packed_support,
)
from eval.stda_f0_verify import verify_structural_replay


EXPECTED_PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
EXPECTED_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"


def _imports(source: str) -> set[str]:
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _calls(source: str) -> list[str]:
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def test_oracle_and_replay_bind_frozen_protocol() -> None:
    assert oracle_module.PROTOCOL_SHA256 == EXPECTED_PROTOCOL_SHA256
    assert oracle_module.PROTOCOL_FREEZE_COMMIT == EXPECTED_FREEZE_COMMIT
    assert replay_module.PROTOCOL_SHA256 == EXPECTED_PROTOCOL_SHA256
    assert replay_module.PROTOCOL_FREEZE_COMMIT == EXPECTED_FREEZE_COMMIT


def test_oracle_has_cpu_only_target_and_metric_boundaries() -> None:
    source = inspect.getsource(oracle_module)
    imported = _imports(source)
    forbidden_imports = {
        name
        for name in imported
        if name == "torch"
        or name.startswith("torch.")
        or "dense_geometry" in name
        or "stda_f0_candidate" in name
    }
    assert not forbidden_imports
    assert "eval.stda_f0_fit" in imported
    assert "eval.stda_f0_round" in imported
    assert "eval.stda_f0_structure" in imported
    assert "eval.vrh_f0_support" in imported
    assert "load_target_xyz_confidence_bytes" in _calls(source)
    assert _calls(source).count("load_target_xyz_confidence_bytes") == 1
    assert "np.load" not in source
    assert "geometry_report" not in source
    assert "load_tesseract" not in source


def test_clean_replay_imports_only_round_solver_boundary() -> None:
    source = inspect.getsource(replay_module)
    imported = _imports(source)
    assert "eval.stda_f0_round" in imported
    assert not any("stda_f0_fit" in name for name in imported)
    assert not any("stda_f0_structure" in name for name in imported)
    assert not any("dense_geometry" in name for name in imported)
    assert not any("candidate" in name for name in imported)
    assert "target" not in " ".join(replay_module.ALLOWED_INPUT_FILENAMES).lower()
    assert set(replay_module.ALLOWED_INPUT_FILENAMES) == {
        filename
        for _, filename, _, _ in (
            replay_module.SUPPORT_ARRAY_FILES + replay_module.GRAPH_ARRAY_FILES
        )
    }


def test_cpu_only_environment_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle_module, "FORBIDDEN_MODULE_PREFIXES", ())
    monkeypatch.setattr(replay_module, "FORBIDDEN_MODULE_PREFIXES", ())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    oracle_module.require_cpu_only_environment()
    replay_module.require_cpu_only_environment()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES empty"):
        oracle_module.require_cpu_only_environment()
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES empty"):
        replay_module.require_cpu_only_environment()

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES empty"):
        oracle_module.require_cpu_only_environment()


def test_clean_replay_environment_drops_target_and_cuda_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TARGET_CACHE_PATH", "/forbidden/target.npz")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("CUDA_HOME", "/forbidden/cuda")
    environment = oracle_module._clean_replay_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "TARGET_CACHE_PATH" not in environment
    assert "NVIDIA_VISIBLE_DEVICES" not in environment
    assert "CUDA_HOME" not in environment


def test_replay_request_rejects_any_extra_target_file(tmp_path: Path) -> None:
    input_root = tmp_path / "solver_inputs"
    input_root.mkdir()
    empty_sha = hashlib.sha256(b"").hexdigest()
    file_hashes = {}
    for filename in replay_module.ALLOWED_INPUT_FILENAMES:
        path = input_root / filename
        path.write_bytes(b"")
        file_hashes[filename] = empty_sha

    script_sha = replay_module.sha256_file(Path(replay_module.__file__).resolve())
    request_without_self = {
        "schema": replay_module.REQUEST_SCHEMA,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_freeze_commit": EXPECTED_FREEZE_COMMIT,
        "support_cardinality": 10_000,
        "input_files_sha256": file_hashes,
        "replay_script_sha256": script_sha,
    }
    request = dict(request_without_self)
    request["request_payload_sha256"] = replay_module.sha256_bytes(
        replay_module.canonical_json_bytes(request_without_self)
    )
    request_payload = replay_module.canonical_json_bytes(request)

    cardinality, observed = replay_module._validate_request(
        request,
        request_payload=request_payload,
        input_root=input_root,
        observed_input_hashes=file_hashes,
    )
    assert cardinality == 10_000
    assert observed == file_hashes

    (input_root / "target_xyz_confidence.npy").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="unexpected file"):
        replay_module._validate_request(
            request,
            request_payload=request_payload,
            input_root=input_root,
            observed_input_hashes=file_hashes,
        )


def _assignment(value: int = 0) -> SimpleNamespace:
    slot = np.asarray([0, 1], dtype="<i8")
    support = np.asarray([2, 3], dtype="<i8")
    export = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype="<f4")
    return SimpleNamespace(
        slot_row=slot,
        slot_id=slot,
        support_rank=support,
        support_id=support + 100,
        edge_cost=np.asarray([5, 7], dtype="<i8"),
        selected_support_rank=support,
        selected_support_id=support + 100,
        export_xyz=export + np.float32(value),
        objective=12,
        assignment_sha256="01" * 32,
        selected_id_sha256="02" * 32,
        export_sha256="03" * 32,
        objective_sha256="04" * 32,
        digest_sha256="05" * 32,
    )


def test_inprocess_assignment_identity_requires_arrays_hashes_and_objective() -> None:
    first = _assignment()
    second = _assignment()
    oracle_module._require_identical_assignments(first, second)

    with pytest.raises(AssertionError, match="arrays differ"):
        oracle_module._require_identical_assignments(first, _assignment(1))
    changed_hash = _assignment()
    changed_hash.export_sha256 = "ff" * 32
    with pytest.raises(AssertionError, match="hashes differ"):
        oracle_module._require_identical_assignments(first, changed_hash)


def test_all_emitted_arm_reports_pass_independent_structural_replay() -> None:
    domain = StructuralDomain.from_edges((-0.2, 0.2), (-0.2, 0.2))
    target = np.asarray(
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        dtype="<f4",
    )
    emitted_arms = {
        "decision": np.asarray(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype="<f4",
        ),
        "packed_pointwise": np.asarray(
            [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype="<f4",
        ),
        "round_robin_greedy": np.asarray(
            [[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
            dtype="<f4",
        ),
    }
    expected_report_keys = {
        "schema",
        "protocol_sha256",
        "protocol_freeze_commit",
        "domain_sha256",
        "raw_inputs_sha256",
        "canonical_inputs_sha256",
        "output",
        "target",
        "structural_domain_valid",
        "evaluation_complete",
        "failure_reasons",
        "groups",
        "assignments",
        "class_metrics",
        "integer_cost",
        "matched_count",
        "complete_assignment_tuple",
        "objective_key",
        "groups_sha256",
        "assignments_sha256",
        "complete_mapping_sha256",
        "class_metrics_sha256",
        "result_sha256",
    }

    replayed_arms: set[str] = set()
    for arm, output in emitted_arms.items():
        inputs = freeze_structural_inputs(output, target, domain=domain)
        result = evaluate_structure(inputs)
        report = oracle_module._plain_structure_report(result, domain)

        assert report["schema"] == "stda_f0_structural_report_v1"
        assert set(report) == expected_report_keys
        assert set(report["output"]) == {
            "source_row_count",
            "duplicate_row_count",
            "invalid_event_ids",
            "digest_sha256",
        }
        assert set(report["target"]) == {
            "source_row_count",
            "zero_confidence_row_count",
            "invalid_target_ids",
            "digest_sha256",
        }
        replay = verify_structural_replay(
            inputs.output_xyz_le_f4,
            inputs.target_xyz_confidence_le_f4,
            domain.azimuth_edges_rad,
            domain.elevation_edges_rad,
            report,
        )

        assert replay["passed"] is True, (arm, replay["checks"])
        assert replay["computed_report"] == report
        assert oracle_module.canonical_json_bytes(report).endswith(b"\n")
        replayed_arms.add(arm)

    assert replayed_arms == {
        "decision",
        "packed_pointwise",
        "round_robin_greedy",
    }


def test_capacity_no_go_hashes_cache_once_without_array_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(oracle_module, "FORBIDDEN_MODULE_PREFIXES", ())
    monkeypatch.setattr(oracle_module, "_require_isolated_interpreter", lambda: None)
    target_calls: list[Path] = []

    def forbidden_target_loader(_: bytes, **kwargs: object) -> object:
        target_calls.append(Path(str(kwargs)))
        raise AssertionError("capacity no-go must not open target")

    monkeypatch.setattr(
        oracle_module,
        "load_target_xyz_confidence_bytes",
        forbidden_target_loader,
    )
    support = build_packed_support(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype="<f4",
        ),
        np.ones(4, dtype="<f4"),
    )
    support_bytes = serialize_packed_support(support)
    support_dir = tmp_path / "support_frame"
    support_dir.mkdir()
    (support_dir / "support.bin").write_bytes(support_bytes)
    target_path = tmp_path / "must_not_open.npz"
    target_payload = b"not parsed as a target cache"
    target_path.write_bytes(target_payload)
    resources = tmp_path / "resources"
    resources.mkdir()
    output = tmp_path / "oracle_output"

    report = oracle_module.run_oracle_phase(
        support_frame_dir=support_dir,
        target_cache_path=target_path,
        resources_dir=resources,
        output_dir=output,
        sequence=1,
        radar_index=2,
        expected_support_sha256=hashlib.sha256(support_bytes).hexdigest(),
    )

    assert report["status"] == CAPACITY_NO_GO_STATUS
    assert report["target"]["opened"] is True
    assert report["target"]["array_loader_called"] is False
    assert report["target"]["cache_arrays_read"] == []
    assert report["target"]["target_array_materialized"] is False
    assert report["target"]["cache_sha256"] == hashlib.sha256(
        target_payload
    ).hexdigest()
    assert report["target"]["cache_size_bytes"] == len(target_payload)
    assert report["target"]["stat_before"] == report["target"]["stat_after"]
    assert report["target"]["path_read_calls"] == 1
    assert report["target"]["descriptor_read_calls"] >= 1
    assert target_calls == []
    assert (output / "oracle_report.json").is_file()
    assert (output / "ORACLE_COMPLETE.json").is_file()
    timing, _ = oracle_module._load_canonical_json(
        output / "ORACLE_FRAME_TIMING.json"
    )
    assert timing["oracle_child_pre_metric_ns"] > 0
    assert timing["authoritative_oracle_frame_ns"] is False
    assert timing["support_reverification_excluded"] is True
    assert timing["timing_receipt_publication_excluded"] is True
    assert "sole_target_cache_byte_read" in timing["boundary"]
    assert not (output / "solver_inputs").exists()


def test_capacity_no_go_cache_provenance_rejects_concurrent_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "target.npz"
    cache.write_bytes(b"before")
    original_read = oracle_module.os.read
    mutated = False

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        payload = original_read(descriptor, count)
        if payload and not mutated:
            mutated = True
            with cache.open("ab") as handle:
                handle.write(b"changed")
                handle.flush()
                os.fsync(handle.fileno())
        return payload

    monkeypatch.setattr(oracle_module.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="sole byte read"):
        oracle_module._read_immutable_path_bytes(
            cache,
            label="STDA capacity-no-go target cache",
        )


def test_clean_solver_uses_isolated_no_site_flags() -> None:
    source = inspect.getsource(oracle_module.run_oracle_phase)
    assert '"-I"' in source
    assert '"-S"' in source
    assert '"-B"' in source
    assert source.index('"-I"') < source.index("str(replay_script)")
    assert "expected_replay_source_hashes" in source


def test_oracle_and_assignment_scripts_import_under_isolated_no_site(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "sitecustomize-ran"
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="ascii",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(hostile),
            "PYTHONNOUSERSITE": "1",
        }
    )
    for script in (Path(oracle_module.__file__), Path(replay_module.__file__)):
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(script), "--help"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            check=False,
            timeout=30.0,
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert not marker.exists()


def test_assignment_runtime_source_hashes_are_explicit_and_stable() -> None:
    hashes = replay_module._runtime_source_hashes()
    assert set(hashes) == {
        "code/scripts/stda_f0_assignment_replay.py",
        "code/eval/stda_f0_round.py",
    }
    assert replay_module._require_runtime_source_hashes(hashes) == hashes


def test_oracle_source_hashes_expose_exact_orchestrator_binding() -> None:
    hashes = oracle_module._source_hashes()
    orchestrator_hashes = {
        relative: hashes[relative]
        for relative in oracle_module.ORCHESTRATOR_SOURCE_PATHS
    }
    assert tuple(orchestrator_hashes) == oracle_module.ORCHESTRATOR_SOURCE_PATHS
    assert all(len(value) == 64 for value in orchestrator_hashes.values())


def test_fsynced_array_writer_preserves_explicit_little_endian_dtype(
    tmp_path: Path,
) -> None:
    records: dict[str, dict[str, object]] = {}
    hashes = oracle_module._write_array_set(
        root=tmp_path,
        directory=tmp_path / "arrays",
        arrays={
            "integer.npy": (np.asarray([1, 2], dtype=np.int64), "<i8"),
            "float.npy": (np.asarray([[1.5, 2.5]], dtype=np.float64), "<f8"),
        },
        records=records,
    )

    integer = np.load(tmp_path / "arrays/integer.npy", allow_pickle=False)
    floating = np.load(tmp_path / "arrays/float.npy", allow_pickle=False)
    assert integer.dtype.str == "<i8"
    assert floating.dtype.str == "<f8"
    assert hashes["integer.npy"] == hashlib.sha256(
        (tmp_path / "arrays/integer.npy").read_bytes()
    ).hexdigest()
    assert set(records) == {"arrays/integer.npy", "arrays/float.npy"}
