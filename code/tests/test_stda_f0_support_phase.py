from __future__ import annotations

import ast
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import savemat

import cube_dense.kradar as kradar
import scripts.stda_f0_support_phase as phase


EXPECTED_PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
EXPECTED_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"


def _manifest_document() -> dict[str, object]:
    frames = [
        {
            "sequence": index // 2 + 1,
            "radar_index": 1000 + index,
            "partition": "train",
            "label": f"ignored_{index}.txt",
            "lidar64_index": index,
        }
        for index in range(76)
    ]
    frames.extend(
        {
            "sequence": 100 + index,
            "radar_index": 2000 + index,
            "partition": "validation",
        }
        for index in range(24)
    )
    return {"frames": frames}


def _audit_policy(tmp_path: Path) -> tuple[phase.OpenAuditPolicy, dict[str, Path]]:
    roots = {
        "environment": tmp_path / "env",
        "source": tmp_path / "source",
        "input": tmp_path / "manifest.json",
        "staging": tmp_path / "staging",
        "cube": tmp_path / "data/1/radar_tesseract/tesseract_00232.mat",
    }
    for key in ("environment", "source", "staging"):
        roots[key].mkdir(parents=True)
    roots["input"].write_text("{}\n", encoding="ascii")
    roots["cube"].parent.mkdir(parents=True)
    roots["cube"].write_bytes(b"cube")
    policy = phase.OpenAuditPolicy(
        interpreter_roots=(roots["environment"],),
        source_roots=(roots["source"],),
        exact_read_paths=(roots["input"],),
        staging_root=roots["staging"],
    )
    return policy, roots


def test_freeze_constants_and_formal_counts_are_bound() -> None:
    assert phase.PROTOCOL_SHA256 == EXPECTED_PROTOCOL_SHA256
    assert phase.PROTOCOL_FREEZE_COMMIT == EXPECTED_FREEZE_COMMIT
    assert phase.FORMAL_TRAIN_FRAME_COUNT == 76
    assert phase.FORMAL_VALIDATION_FRAME_COUNT == 24
    assert phase.FORMAL_CANDIDATE_COUNT == 700_000
    assert phase.PREDECESSOR_HASH_FRAME_COUNT == 12
    assert phase.EXPECTED_DEVICE_ARGUMENT == "cuda:0"
    assert phase.AUDIT_HOOK_INSTALLED_BEFORE_SCIENTIFIC_IMPORTS is True


def test_audit_hook_precedes_every_scientific_or_project_import() -> None:
    source = inspect.getsource(phase)
    hook_offset = source.index("sys.addaudithook(_stda_f0_early_audit_hook)")
    runtime_offset = source.index('importlib.import_module("numpy")')
    assert hook_offset < runtime_offset

    tree = ast.parse(source)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level_imports.add(node.module)
    assert "numpy" not in top_level_imports
    assert "torch" not in top_level_imports
    assert "scipy" not in top_level_imports
    assert not any(
        name.startswith(("cube_dense", "eval", "models", "scripts"))
        for name in top_level_imports
    )


def test_support_child_has_no_process_launcher_import_or_call() -> None:
    source = inspect.getsource(phase)
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert "subprocess" not in imported
    assert "multiprocessing" not in imported
    assert "os.fork(" not in source
    assert "os.posix_spawn(" not in source
    assert "os.system(" not in source


def test_cli_surface_is_exact_and_has_no_gt_or_future_input() -> None:
    parser = phase.build_parser()
    phase.validate_parser_surface(parser)
    options = set(phase.parser_option_strings(parser))
    assert options == {
        "--data-root",
        "--manifest",
        "--scene-split",
        "--normalization",
        "--checkpoint",
        "--candidate-hash-manifest",
        "--repo",
        "--output-root",
        "--source-commit",
        "--device",
    }
    assert not any(
        fragment in option
        for option in options
        for fragment in phase.FORBIDDEN_OPTION_FRAGMENTS
    )


def test_environment_sanitization_removes_cache_knobs_and_rejects_gt_input() -> None:
    environment = {
        "XDG_CACHE_HOME": "/tmp/ordinary",
        "PIP_CACHE_DIR": "/tmp/pip",
        "PATH": "/usr/bin",
        "UNRELATED": "value",
    }
    report = phase.sanitize_environment(environment, enforce_required=False)
    assert "XDG_CACHE_HOME" not in environment
    assert "PIP_CACHE_DIR" not in environment
    assert "UNRELATED" not in environment
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert {row["key"] for row in report["removed_entries"]} == {
        "PIP_CACHE_DIR",
        "UNRELATED",
        "XDG_CACHE_HOME",
    }

    with pytest.raises(phase.SupportPhaseContractError, match="forbidden"):
        phase.sanitize_environment(
            {"CURRENT_TARGET_PATH": "/secret/frame.npz"},
            enforce_required=False,
        )
    with pytest.raises(phase.SupportPhaseContractError, match="forbidden"):
        phase.sanitize_environment(
            {"PATH": "/usr/bin:/dataset/labels/train"},
            enforce_required=False,
        )


def test_formal_environment_requires_frozen_h200_gpu2_uuid() -> None:
    environment = {
        **phase.REQUIRED_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": phase.EXPECTED_GPU_UUID_TOKEN,
        "CONDA_PREFIX": str(Path(sys.executable).resolve().parent.parent),
        "PATH": str(Path(sys.executable).resolve().parent),
    }
    report = phase.sanitize_environment(environment, enforce_required=True)
    assert report["required_checks"][
        "CUDA_VISIBLE_DEVICES_is_frozen_H200_GPU2_UUID"
    ] is True

    wrong = {
        **phase.REQUIRED_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": "2",
        "CONDA_PREFIX": str(Path(sys.executable).resolve().parent.parent),
    }
    with pytest.raises(phase.SupportPhaseContractError, match="GPU2 UUID"):
        phase.sanitize_environment(wrong, enforce_required=True)


def test_audit_policy_allows_only_exact_cube_source_env_and_staging(
    tmp_path: Path,
) -> None:
    policy, roots = _audit_policy(tmp_path)
    assert policy.classify_path(roots["input"], write=False)["allowed"] is True
    assert (
        policy.classify_path(roots["environment"] / "numpy.py", write=False)[
            "category"
        ]
        == "interpreter_environment"
    )
    assert (
        policy.classify_path(roots["source"] / "module.py", write=False)[
            "category"
        ]
        == "clean_source_snapshot"
    )
    assert policy.classify_path(
        roots["staging"] / "support.bin", write=True
    )["allowed"] is True
    assert policy.classify_path(
        tmp_path / "outside.bin", write=True
    )["allowed"] is False

    assert policy.classify_path(roots["cube"], write=False)["allowed"] is False
    policy.set_current_cube(roots["cube"])
    decision = policy.classify_path(roots["cube"], write=False)
    assert decision["allowed"] is True
    assert decision["category"] == "current_cube"
    policy.clear_current_cube(roots["cube"])
    assert policy.classify_path(roots["cube"], write=False)["allowed"] is False


def test_audit_policy_rejects_adversarial_target_sidecar_even_under_source(
    tmp_path: Path,
) -> None:
    policy, roots = _audit_policy(tmp_path)
    fake = roots["source"] / "target_cache" / "seq01_radar_00232.npz"
    decision = policy.classify_path(fake, write=False)
    assert decision["allowed"] is False
    assert decision["category"] == "forbidden_data_path"

    with pytest.raises(phase.AuditAccessError, match="cannot create descendants"):
        policy.handle("subprocess.Popen", ("python",))
    assert policy.process_attempt_count == 1
    assert policy.denied_count == 1


def test_audit_seal_rejects_later_file_access(tmp_path: Path) -> None:
    policy, roots = _audit_policy(tmp_path)
    policy.handle("open", (str(roots["input"]), "r", 0))
    sealed = policy.seal()
    assert len(sealed) == 1
    with pytest.raises(phase.AuditAccessError, match="after audit ledger seal"):
        policy.handle("open", (str(roots["input"]), "r", 0))


def test_canonical_train_records_preserve_archived_order_and_drop_metadata() -> None:
    document = _manifest_document()
    records = phase.canonical_train_records(document)
    assert len(records) == 76
    assert records[0] == {"sequence": 1, "radar_index": 1000}
    assert records[-1] == {"sequence": 38, "radar_index": 1075}
    assert all(set(record) == {"sequence", "radar_index"} for record in records)

    duplicate = _manifest_document()
    duplicate["frames"][1]["sequence"] = duplicate["frames"][0]["sequence"]
    duplicate["frames"][1]["radar_index"] = duplicate["frames"][0][
        "radar_index"
    ]
    with pytest.raises(phase.SupportPhaseContractError, match="repeats"):
        phase.canonical_train_records(duplicate)

    missing = _manifest_document()
    missing["frames"].pop()
    with pytest.raises(phase.SupportPhaseContractError, match="76/24"):
        phase.canonical_train_records(missing)


def test_candidate_hash_map_requires_twelve_unique_hash_bound_frames() -> None:
    keys = (
        "q0_sha256",
        "q1_sha256",
        "candidate_xyz_sha256",
        "base_confidence_sha256",
    )
    document = {
        "kind": "stda_f0_target_free_candidate_hash_manifest_v1",
        "schema_version": 1,
        "frames": [
            {
                "sequence": index + 1,
                "radar_index": 200 + index,
                **{key: hashlib.sha256(f"{key}:{index}".encode()).hexdigest() for key in keys},
            }
            for index in range(12)
        ],
    }
    mapping = phase.candidate_hash_map(document)
    assert len(mapping) == 12
    assert set(mapping[(1, 200)]) == set(keys)

    document["frames"][1]["sequence"] = 1
    document["frames"][1]["radar_index"] = 200
    with pytest.raises(phase.SupportPhaseContractError, match="malformed"):
        phase.candidate_hash_map(document)


def test_canonical_json_and_support_file_schema_are_stable() -> None:
    payload = phase.canonical_json_bytes({"z": 1, "a": [True, False]})
    assert payload == b'{"a":[true,false],"z":1}\n'
    assert json.loads(payload) == {"a": [True, False], "z": 1}
    assert phase.sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()
    assert tuple(row[0] for row in phase.SUPPORT_FILE_SCHEMA) == (
        "support.bin",
        "support_stable_candidate_id.npy",
        "support_grid_cell.npy",
        "support_xyz.npy",
        "support_base_confidence.npy",
        "support_color.npy",
    )


def test_immutable_cube_reader_reads_once_and_binds_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cube = tmp_path / "cube.mat"
    payload = b"immutable synthetic Cube bytes"
    cube.write_bytes(payload)
    observed, provenance = phase._read_immutable_path_bytes(cube, label="raw Cube")

    assert observed == payload
    assert provenance["path_read_calls"] == 1
    assert provenance["descriptor_read_calls"] >= 1
    assert provenance["sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["stat_before"] == provenance["stat_after"]
    assert provenance["hash_consumed_same_payload"] is True


def test_immutable_cube_reader_rejects_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cube = tmp_path / "cube.mat"
    cube.write_bytes(b"before")
    original_read = phase.os.read
    mutated = False

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        payload = original_read(descriptor, count)
        if payload and not mutated:
            mutated = True
            with cube.open("ab") as handle:
                handle.write(b"changed")
                handle.flush()
                os.fsync(handle.fileno())
        return payload

    monkeypatch.setattr(phase.os, "read", mutate_after_read)
    with pytest.raises(phase.SupportPhaseContractError, match="sole byte read"):
        phase._read_immutable_path_bytes(cube, label="raw Cube")


def test_tesseract_bytesio_errors_preserve_source_label() -> None:
    buffer = io.BytesIO()
    savemat(buffer, {"arrDREA": np.zeros((1, 2), dtype=np.float32)})
    buffer.seek(0)
    with pytest.raises(ValueError, match="synthetic_cube.mat"):
        kradar.load_tesseract(buffer, source_label="synthetic_cube.mat")


def test_frame_runtime_source_contains_required_boundaries_and_payload_loader() -> None:
    source = inspect.getsource(phase._process_frame)
    assert "_read_immutable_path_bytes(" in source
    assert "io.BytesIO(cube_payload)" in source
    assert "source_label=cube_path" in source
    assert "sha256_file(cube_path)" not in source
    assert "torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16)" in source
    assert "runtime.candidate.reconstruct_candidate_field" in source
    assert "runtime.candidate.verify_candidate_only" in source
    assert "runtime.support.build_packed_support" in source
    assert "runtime.verify.verify_packed_support_bytes" in source
    assert 'frame_root / "support_record.json"' in source
    assert 'frame_root / "support_manifest_entry.json"' in source
    assert "manifest_entry_record = _write_fsync_rehash(" in source
    assert source.index("manifest_entry_record = _write_fsync_rehash(") < source.index(
        "support_frame_ended = time.perf_counter_ns()"
    )
    assert source.index("torch.cuda.empty_cache()") < source.index(
        "support_frame_ended = time.perf_counter_ns()"
    )
    assert source.count("torch.cuda.synchronize(device)") >= 5
    assert "time.perf_counter_ns()" in source
    assert "peak_allocated_bytes" in source
    assert "peak_reserved_bytes" in source
    assert "PilotCubeDataset" not in source


def test_trusted_site_packages_are_explicit_without_site_processing() -> None:
    roots = phase._interpreter_roots(os.environ)
    paths = phase._trusted_site_package_paths(roots)
    assert paths
    assert any((path / "numpy").is_dir() for path in paths)
    source = inspect.getsource(phase._trusted_site_package_paths)
    assert "addsitedir" not in source
    assert "sitecustomize" not in source


def test_runtime_source_hashes_expose_exact_orchestrator_binding() -> None:
    repo = Path(phase.__file__).resolve().parents[2]
    runtime_hashes = phase._runtime_source_hashes(repo)
    orchestrator_hashes = phase._source_hash_subset(
        runtime_hashes,
        phase.ORCHESTRATOR_SOURCE_PATHS,
    )
    assert tuple(orchestrator_hashes) == phase.ORCHESTRATOR_SOURCE_PATHS
    assert all(len(value) == 64 for value in orchestrator_hashes.values())


def test_support_script_starts_under_isolated_no_site_interpreter(
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
    environment["PYTHONPATH"] = str(hostile)
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(Path(phase.__file__)), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert not marker.exists()


def test_completion_writer_seals_after_opening_ledger_file() -> None:
    source = inspect.getsource(phase._write_final_ledger)
    assert source.index('ledger_path.open("x+b")') < source.index("auditor.seal()")
    assert source.index("os.open(output_root") < source.index("auditor.seal()")
    assert "os.fsync(handle.fileno())" in source
    assert "handle.seek(0)" in source
    assert "handle.read()" in source
