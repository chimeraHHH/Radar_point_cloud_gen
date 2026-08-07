#!/usr/bin/env python3
"""Run one CPU-only, target-conditioned frame of the frozen STDA-F0 oracle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence, TypeVar

CODE_ROOT = Path(__file__).resolve().parents[1]


def _require_isolated_interpreter() -> None:
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.no_user_site
        and "sitecustomize" not in sys.modules
    ):
        raise RuntimeError(
            "STDA oracle requires python -I -S with no user site or sitecustomize"
        )


if __name__ == "__main__":
    _require_isolated_interpreter()


def _path_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _trusted_site_package_paths() -> tuple[Path, ...]:
    """Expose only this interpreter's package dirs; never process .pth hooks."""

    prefix = Path(sys.executable).resolve().parent.parent
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    paths: set[Path] = set()
    for relative in (
        Path("lib") / version / "site-packages",
        Path("lib64") / version / "site-packages",
    ):
        candidate = (prefix / relative).resolve()
        if candidate.is_dir() and _path_within(candidate, prefix):
            paths.add(candidate)
    if not paths:
        raise RuntimeError("STDA oracle cannot locate trusted site-packages under -I -S")
    return tuple(sorted(paths, key=str))


def _bootstrap_sys_path(trusted_site_packages: Sequence[Path]) -> None:
    retained: list[str] = []
    trusted = {path.resolve() for path in trusted_site_packages}
    for entry in sys.path:
        if not entry:
            continue
        resolved = Path(entry).resolve()
        if resolved == CODE_ROOT or resolved in trusted or "site-packages" in resolved.parts:
            continue
        value = str(resolved)
        if value not in retained:
            retained.append(value)
    sys.path[:] = [
        str(CODE_ROOT),
        *retained,
        *(str(path) for path in trusted_site_packages),
    ]


TRUSTED_SITE_PACKAGE_PATHS = _trusted_site_package_paths()
_bootstrap_sys_path(TRUSTED_SITE_PACKAGE_PATHS)


import numpy as np  # noqa: E402

from cube_dense.kradar import load_axes  # noqa: E402
from eval.stda_f0_fit import (  # noqa: E402
    build_midpoint_demand_slots,
    build_packed_pointwise_sidecar,
    build_sparse_assignment_graph,
    canonicalize_support,
    canonicalize_target_atoms,
    load_target_xyz_confidence_bytes,
)
from eval.stda_f0_round import (  # noqa: E402
    GRAPH_ARRAY_FILES,
    GREEDY_ARRAY_FILES,
    POINTWISE_ARRAY_FILES,
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_SHA256,
    SUPPORT_ARRAY_FILES,
    load_assignment_graph,
    load_greedy_sidecar,
    load_packed_support,
    load_pointwise_sidecar,
    maximum_cardinality_certificate,
    packed_pointwise_control,
    solve_full_assignment,
    target_atom_round_robin_greedy,
    verify_full_assignment,
    verify_greedy_control,
    verify_hall_witness,
    verify_matching,
    verify_packed_pointwise,
)
from eval.stda_f0_structure import (  # noqa: E402
    StructuralDomain,
    evaluate_structure,
    freeze_structural_inputs,
)
from eval.stda_f0_support import (  # noqa: E402
    CAPACITY_NO_GO_STATUS,
    REQUIRED_EXPORT_COUNT,
    commit_packed_support,
    deserialize_packed_support,
)
from eval.vrh_f0_support import build_vrh_support  # noqa: E402


ORACLE_SCHEMA = "stda_f0_oracle_frame_v1"
ORACLE_COMPLETE_SCHEMA = "stda_f0_oracle_frame_complete_v1"
ORACLE_FAILURE_SCHEMA = "stda_f0_oracle_frame_failure_v1"
REPLAY_REQUEST_SCHEMA = "stda_f0_assignment_replay_request_v1"
REPLAY_RESULT_SCHEMA = "stda_f0_assignment_replay_result_v1"
STRUCTURAL_REPORT_SCHEMA = "stda_f0_structural_report_v1"
ORACLE_READY_STATUS = "stda_f0_oracle_ready_for_cuda_metrics"
GRAPH_NO_GO_STATUS = "stda_f0_graph_cardinality_no_go"
IMPLEMENTATION_INVALID_STATUS = "stda_f0_implementation_invalid"

INFO_ARR_SHA256 = "53f72b22544aa11bc0057f9b8c2177a7a844fddd0e8ce3f753a989d07159767a"
ARR_DOPPLER_SHA256 = "f81e56889c2cedc98eb3eb8a4828e382845e3d4758a36f3b8fc0fce4839e0493"
FORBIDDEN_MODULE_PREFIXES = ("torch", "cupy", "jax", "tensorflow", "pynvml")
ASSIGNMENT_REPLAY_SOURCE_PATHS = (
    "code/scripts/stda_f0_assignment_replay.py",
    "code/eval/stda_f0_round.py",
)
ORACLE_CRITICAL_SOURCE_PATHS = (
    "code/scripts/stda_f0_oracle_phase.py",
    "code/scripts/stda_f0_assignment_replay.py",
    "code/cube_dense/__init__.py",
    "code/cube_dense/kradar.py",
    "code/eval/stda_f0_support.py",
    "code/eval/stda_f0_fit.py",
    "code/eval/stda_f0_round.py",
    "code/eval/stda_f0_structure.py",
    "code/eval/vrh_f0_support.py",
    "docs/stda_f0_sparse_target_demand_assignment_protocol.md",
)
ORCHESTRATOR_SOURCE_PATHS = (
    "docs/stda_f0_sparse_target_demand_assignment_protocol.md",
    "artifacts/idea/stda_f0_freeze_record.json",
    "artifacts/idea/stda_f0_prefreeze_audit_round6.md",
    "code/eval/stda_f0_candidate.py",
    "code/eval/stda_f0_support.py",
    "code/eval/stda_f0_fit.py",
    "code/eval/stda_f0_round.py",
    "code/eval/stda_f0_structure.py",
    "code/eval/stda_f0_verify.py",
    "code/scripts/stda_f0_support_phase.py",
    "code/scripts/stda_f0_oracle_phase.py",
    "code/scripts/stda_f0_assignment_replay.py",
    "code/scripts/stda_f0_verify_phase.py",
    "code/scripts/stda_f0_metric_phase.py",
    "code/scripts/stda_f0_bundle_verify.py",
    "code/scripts/preflight_stda_f0_capacity.py",
)
RUNTIME_SOURCE_PATHS = tuple(
    dict.fromkeys((*ORACLE_CRITICAL_SOURCE_PATHS, *ORCHESTRATOR_SOURCE_PATHS))
)
REQUIRED_PROJECT_MODULE_PATHS = {
    "cube_dense": "code/cube_dense/__init__.py",
    "cube_dense.kradar": "code/cube_dense/kradar.py",
    "eval.stda_f0_fit": "code/eval/stda_f0_fit.py",
    "eval.stda_f0_round": "code/eval/stda_f0_round.py",
    "eval.stda_f0_structure": "code/eval/stda_f0_structure.py",
    "eval.stda_f0_support": "code/eval/stda_f0_support.py",
    "eval.vrh_f0_support": "code/eval/vrh_f0_support.py",
}
PROJECT_IMPORT_ROOTS = frozenset({"cube_dense", "eval", "models"})
ASSIGNMENT_IDENTITY_FIELDS = (
    "assignment_sha256",
    "selected_id_sha256",
    "export_sha256",
    "objective_sha256",
    "digest_sha256",
)

ALLOCATION_COMPONENT_KEYS = (
    "canonicalize_target_atoms_ns",
    "build_midpoint_demand_slots_ns",
    "canonicalize_support_ns",
    "build_sparse_assignment_graph_ns",
    "build_packed_pointwise_sidecar_ns",
    "fit_input_serialization_ns",
    "fit_binding_serialization_ns",
    "round_input_reload_ns",
    "maximum_cardinality_certificate_ns",
    "matching_replay_custom_ns",
    "matching_replay_scipy_ns",
    "hall_replay_ns",
    "decision_solver_inprocess_1_ns",
    "decision_solver_inprocess_2_ns",
    "decision_solver_inprocess_replay_ns",
    "assignment_replay_request_serialization_ns",
    "decision_solver_subprocess_3_ns",
    "decision_solver_subprocess_replay_ns",
    "packed_pointwise_control_ns",
    "packed_pointwise_replay_ns",
    "round_robin_greedy_control_ns",
    "round_robin_greedy_replay_ns",
    "result_array_serialization_ns",
    "export_serialization_ns",
    "structural_domain_load_ns",
    "structural_target_snapshot_ns",
    "structure_evidence_directory_ns",
    "structure_decision_ns",
    "structure_decision_serialization_ns",
    "structure_packed_pointwise_ns",
    "structure_packed_pointwise_serialization_ns",
    "structure_round_robin_greedy_ns",
    "structure_round_robin_greedy_serialization_ns",
    "structure_directory_fsync_ns",
    "final_input_reverification_ns",
)

T = TypeVar("T")


def canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_record(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _read_immutable_path_bytes(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, dict[str, object]]:
    """Read one regular file through one no-follow descriptor."""

    path = Path(path)
    read_started_ns = time.perf_counter_ns()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a regular non-symlink file") from error
    read_calls = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            read_calls += 1
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        path_after = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    before_record = _stat_record(before)
    after_record = _stat_record(after)
    path_after_record = _stat_record(path_after)
    if (
        before_record != after_record
        or path_after_record != after_record
        or len(payload) != before.st_size
    ):
        raise ValueError(f"{label} changed during its sole byte read")
    return payload, {
        "path": str(path),
        "cache_sha256": sha256_bytes(payload),
        "cache_size_bytes": len(payload),
        "stat_before": before_record,
        "stat_after": after_record,
        "path_stat_after": path_after_record,
        "path_read_calls": 1,
        "descriptor_read_calls": read_calls,
        "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
        "read_started_perf_counter_ns": read_started_ns,
        "clock": "time.perf_counter_ns",
        "immutable_bytes_materialized": True,
        "hash_consumed_same_payload": True,
    }


def _validate_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def require_cpu_only_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("STDA oracle requires CUDA_VISIBLE_DEVICES empty")
    loaded = sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in FORBIDDEN_MODULE_PREFIXES
    )
    if loaded:
        raise RuntimeError(f"STDA oracle loaded CUDA-capable modules: {loaded}")


def _timed(timings: dict[str, int], name: str, operation: Callable[[], T]) -> T:
    started = time.perf_counter_ns()
    try:
        return operation()
    finally:
        elapsed = time.perf_counter_ns() - started
        if elapsed < 0:
            raise AssertionError("STDA perf_counter_ns moved backwards")
        if name in timings:
            raise AssertionError(f"STDA timing component recorded twice: {name}")
        timings[name] = elapsed


def _allocation_core_ns(timings: Mapping[str, int]) -> int:
    return sum(int(timings.get(name, 0)) for name in ALLOCATION_COMPONENT_KEYS)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_fsynced(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npy_new_fsynced(path: Path, values: np.ndarray, dtype: str) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _file_record(root: Path, path: Path) -> dict[str, object]:
    path = Path(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _remember_file(
    records: dict[str, dict[str, object]],
    *,
    root: Path,
    path: Path,
) -> dict[str, object]:
    record = _file_record(root, path)
    relative = str(record["path"])
    if relative in records:
        raise AssertionError(f"STDA output file recorded twice: {relative}")
    records[relative] = record
    return record


def _write_array_set(
    *,
    root: Path,
    directory: Path,
    arrays: Mapping[str, tuple[np.ndarray, str]],
    records: dict[str, dict[str, object]],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    directory.mkdir(parents=True, exist_ok=False)
    for filename, (values, dtype) in arrays.items():
        path = directory / filename
        _write_npy_new_fsynced(path, values, dtype)
        record = _remember_file(records, root=root, path=path)
        hashes[filename] = str(record["sha256"])
    _fsync_directory(directory)
    for filename, expected in hashes.items():
        if sha256_file(directory / filename) != expected:
            raise ValueError(f"STDA immutable array changed after fsync: {filename}")
        os.chmod(directory / filename, 0o444)
    _fsync_directory(directory)
    return hashes


def _checks_dict(checks: object) -> dict[str, bool]:
    values = getattr(checks, "__dict__", None)
    if not isinstance(values, dict):
        raise TypeError("STDA replay checks are not a dataclass-like object")
    return {name: bool(value) for name, value in values.items()}


def _load_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = Path(path).read_bytes()
    try:
        decoded = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("STDA JSON artifact is not ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("STDA JSON artifact must be an object")
    document = dict(decoded)
    if canonical_json_bytes(document) != payload:
        raise ValueError("STDA JSON artifact is not canonical")
    return document, payload


def _source_hashes() -> dict[str, str]:
    return {
        relative: sha256_file(CODE_ROOT.parent / relative)
        for relative in RUNTIME_SOURCE_PATHS
    }


def _require_project_import_closure(
    modules: Mapping[str, object] | None = None,
) -> dict[str, str]:
    repo = CODE_ROOT.parent.resolve(strict=True)
    code_root = CODE_ROOT.resolve(strict=True)
    loaded = sys.modules if modules is None else modules
    allowed = set(RUNTIME_SOURCE_PATHS)
    origins: dict[str, str] = {}
    for name, module in sorted(loaded.items()):
        if name.split(".", 1)[0] not in PROJECT_IMPORT_ROOTS or module is None:
            continue
        raw_file = getattr(module, "__file__", None)
        if raw_file is None:
            locations = getattr(module, "__path__", None)
            if locations is not None:
                for location in locations:
                    if not _path_within(Path(location).resolve(strict=True), code_root):
                        raise ValueError(f"project namespace escaped CODE_ROOT: {name}")
            continue
        resolved = Path(raw_file).resolve(strict=True)
        if not _path_within(resolved, code_root):
            raise ValueError(f"project module escaped CODE_ROOT: {name} -> {resolved}")
        relative = resolved.relative_to(repo).as_posix()
        if relative not in allowed:
            raise ValueError(
                f"loaded project module is absent from source lock: {relative}"
            )
        origins[name] = relative
    for name, expected in REQUIRED_PROJECT_MODULE_PATHS.items():
        if origins.get(name) != expected:
            raise ValueError(f"required project module origin changed: {name}")
    return origins


def _require_source_hashes(expected: Mapping[str, str]) -> dict[str, str]:
    observed = _source_hashes()
    if observed != dict(expected):
        raise ValueError("STDA oracle runtime source bytes changed during execution")
    return observed


def _verify_support(
    support_frame_dir: Path,
    *,
    expected_support_sha256: str,
) -> tuple[object, object, bytes, dict[str, object]]:
    support_frame_dir = Path(support_frame_dir).resolve(strict=True)
    support_path = support_frame_dir / "support.bin"
    if not support_path.is_file() or support_path.is_symlink():
        raise ValueError("STDA support frame must contain one regular support.bin")
    payload = support_path.read_bytes()
    observed = sha256_bytes(payload)
    if observed != expected_support_sha256:
        raise ValueError("STDA support.bin SHA-256 differs from orchestrator commitment")
    support = deserialize_packed_support(payload, expected_sha256=observed)
    commitment = commit_packed_support(payload, expected_sha256=observed)
    if commitment.support_sha256 != observed:
        raise AssertionError("STDA support commitment differs from support bytes")
    stat = support_path.stat()
    report = {
        "path": str(support_path),
        "sha256": observed,
        "bytes": len(payload),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "candidate_field_sha256": commitment.candidate_field_sha256,
        "candidate_count": commitment.candidate_count,
        "support_count": commitment.support_count,
        "selected_color_id": commitment.selected_color_id,
        "color_cardinalities": list(commitment.color_cardinalities),
        "capacity_status": commitment.capacity_status.as_dict(),
        "builder_spacing_self_check_passed": (
            commitment.builder_spacing_self_check_passed
        ),
        "builder_spacing_self_check_authoritative": False,
        "formal_independent_spacing_verified": False,
    }
    return support, commitment, payload, report


def _support_unchanged(support_report: Mapping[str, object]) -> None:
    path = Path(str(support_report["path"]))
    stat = path.stat()
    checks = (
        sha256_file(path) == support_report["sha256"],
        stat.st_size == support_report["bytes"],
        stat.st_dev == support_report["device"],
        stat.st_ino == support_report["inode"],
        stat.st_mtime_ns == support_report["mtime_ns"],
    )
    if not all(checks):
        raise ValueError("STDA support.bin changed during oracle execution")


def _reverify_round_inputs(
    *,
    solver_dir: Path,
    solver_hashes: Mapping[str, str],
    control_dir: Path,
    control_hashes: Mapping[str, str],
    support_report: Mapping[str, object],
) -> None:
    for filename, expected in solver_hashes.items():
        if sha256_file(solver_dir / filename) != expected:
            raise ValueError(f"STDA solver input changed after all replays: {filename}")
    for filename, expected in control_hashes.items():
        if sha256_file(control_dir / filename) != expected:
            raise ValueError(f"STDA control input changed after all replays: {filename}")
    _support_unchanged(support_report)
    require_cpu_only_environment()


def _load_structural_domain(resources_dir: Path) -> tuple[StructuralDomain, dict[str, object]]:
    resources_dir = Path(resources_dir).resolve(strict=True)
    resource_hashes = {
        "info_arr.mat": sha256_file(resources_dir / "info_arr.mat"),
        "arr_doppler.mat": sha256_file(resources_dir / "arr_doppler.mat"),
    }
    expected = {
        "info_arr.mat": INFO_ARR_SHA256,
        "arr_doppler.mat": ARR_DOPPLER_SHA256,
    }
    if resource_hashes != expected:
        raise ValueError("STDA pinned axis resource hashes changed")
    axes = load_axes(resources_dir)
    vrh = build_vrh_support(axes)
    domain = StructuralDomain.from_edges(
        vrh.azimuth_axis.edges,
        vrh.elevation_axis.edges,
    )
    report = {
        "resources_dir": str(resources_dir),
        "resource_sha256": resource_hashes,
        "vrh_support_sha256": vrh.digest_sha256,
        "vrh_schema_header_sha256": vrh.schema_header_sha256,
        "azimuth_edge_count": len(domain.azimuth_edges_rad),
        "elevation_edge_count": len(domain.elevation_edges_rad),
        "azimuth_edges_rad": list(domain.azimuth_edges_rad),
        "elevation_edges_rad": list(domain.elevation_edges_rad),
        "azimuth_edges_sha256": sha256_bytes(
            np.asarray(domain.azimuth_edges_rad, dtype="<f8").tobytes(order="C")
        ),
        "elevation_edges_sha256": sha256_bytes(
            np.asarray(domain.elevation_edges_rad, dtype="<f8").tobytes(order="C")
        ),
        "structural_domain_sha256": domain.digest_sha256,
    }
    return domain, report


def _plain_structure_report(result: object, domain: StructuralDomain) -> dict[str, object]:
    output = result.output
    target = result.target
    return {
        "schema": STRUCTURAL_REPORT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "domain_sha256": domain.digest_sha256,
        "raw_inputs_sha256": result.raw_inputs_sha256,
        "canonical_inputs_sha256": result.canonical_inputs_sha256,
        "output": {
            "source_row_count": output.source_row_count,
            "duplicate_row_count": output.duplicate_row_count,
            "invalid_event_ids": list(output.invalid_event_ids),
            "digest_sha256": output.digest_sha256,
        },
        "target": {
            "source_row_count": target.source_row_count,
            "zero_confidence_row_count": target.zero_confidence_row_count,
            "invalid_target_ids": list(target.invalid_target_ids),
            "digest_sha256": target.digest_sha256,
        },
        "structural_domain_valid": result.structural_domain_valid,
        "evaluation_complete": result.evaluation_complete,
        "failure_reasons": list(result.failure_reasons),
        "groups": [
            {
                "ray_id": group.ray_id,
                "group_index": group.group_index,
                "return_class": group.return_class,
                "range_stratum": group.range_stratum,
                "representative_range_m": group.representative_range_m,
                "group_weight": group.group_weight,
                "canonical_target_ids": list(group.canonical_target_ids),
            }
            for group in result.groups
        ],
        "assignments": [
            {
                "ray_id": assignment.ray_id,
                "group_index": assignment.group_index,
                "return_class": assignment.return_class,
                "range_stratum": assignment.range_stratum,
                "representative_range_m": assignment.representative_range_m,
                "group_weight": assignment.group_weight,
                "assigned_distance_m": assignment.assigned_distance_m,
                "matched_event_id": assignment.matched_event_id,
                "matched_depth": assignment.matched_depth,
                "matched_event_range_m": assignment.matched_event_range_m,
                "integer_cost": assignment.integer_cost,
            }
            for assignment in result.assignments
        ],
        "class_metrics": [
            {
                "key": metric.key,
                "range_stratum": metric.range_stratum,
                "return_class": metric.return_class,
                "applicable": metric.applicable,
                "group_count": metric.group_count,
                "effective_weight": metric.effective_weight,
                "completeness_mean_distance_m": (
                    metric.completeness_mean_distance_m
                ),
                "recall_1m": metric.recall_1m,
                "passed": metric.passed,
            }
            for metric in result.class_metrics
        ],
        "integer_cost": result.integer_cost,
        "matched_count": result.matched_count,
        "complete_assignment_tuple": list(result.complete_assignment_tuple),
        "objective_key": (
            None
            if result.objective_key is None
            else [
                result.objective_key[0],
                result.objective_key[1],
                list(result.objective_key[2]),
            ]
        ),
        "groups_sha256": result.groups_sha256,
        "assignments_sha256": result.assignments_sha256,
        "complete_mapping_sha256": result.complete_mapping_sha256,
        "class_metrics_sha256": result.class_metrics_sha256,
        "result_sha256": result.result_sha256,
    }


def _assignment_arrays_equal(first: object, second: object) -> bool:
    arrays = (
        "slot_row",
        "slot_id",
        "support_rank",
        "support_id",
        "edge_cost",
        "selected_support_rank",
        "selected_support_id",
        "export_xyz",
    )
    return all(np.array_equal(getattr(first, name), getattr(second, name)) for name in arrays)


def _require_identical_assignments(first: object, second: object) -> None:
    if first.objective != second.objective:
        raise AssertionError("STDA in-process assignment objectives differ")
    if not _assignment_arrays_equal(first, second):
        raise AssertionError("STDA in-process assignment/export arrays differ")
    if any(getattr(first, name) != getattr(second, name) for name in ASSIGNMENT_IDENTITY_FIELDS):
        raise AssertionError("STDA in-process assignment hashes differ")


def _clean_replay_environment() -> dict[str, str]:
    environment: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in (
        "PATH",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
    ):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _build_replay_request(
    *,
    support_cardinality: int,
    input_hashes: Mapping[str, str],
    replay_script_sha256: str,
) -> dict[str, object]:
    request: dict[str, object] = {
        "schema": REPLAY_REQUEST_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "support_cardinality": support_cardinality,
        "input_files_sha256": dict(sorted(input_hashes.items())),
        "replay_script_sha256": replay_script_sha256,
    }
    request["request_payload_sha256"] = sha256_bytes(canonical_json_bytes(request))
    return request


def _replay_snapshot_valid(
    snapshot: object,
    *,
    expected_path: str,
    expected_sha256: str,
) -> bool:
    expected_keys = {
        "path",
        "size_bytes",
        "sha256",
        "stat_before",
        "stat_after",
        "path_stat_after",
        "path_read_calls",
        "descriptor_read_calls",
        "open_flags",
        "read_started_perf_counter_ns",
        "clock",
        "immutable_bytes_materialized",
        "hash_consumed_same_payload",
        "parse_consumed_same_payload",
    }
    if not isinstance(snapshot, Mapping) or set(snapshot) != expected_keys:
        return False
    stat_before = snapshot.get("stat_before")
    stat_after = snapshot.get("stat_after")
    path_stat_after = snapshot.get("path_stat_after")
    if not (
        isinstance(stat_before, Mapping)
        and stat_before == stat_after == path_stat_after
        and type(stat_before.get("mode")) is int
        and stat.S_ISREG(stat_before["mode"])
    ):
        return False
    return all(
        (
            snapshot.get("path") == expected_path,
            snapshot.get("sha256") == expected_sha256,
            type(snapshot.get("size_bytes")) is int,
            snapshot.get("size_bytes") == stat_before.get("size_bytes"),
            snapshot.get("path_read_calls") == 1,
            type(snapshot.get("descriptor_read_calls")) is int,
            snapshot.get("descriptor_read_calls") >= 1,
            snapshot.get("open_flags")
            == ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
            type(snapshot.get("read_started_perf_counter_ns")) is int,
            snapshot.get("read_started_perf_counter_ns") > 0,
            snapshot.get("clock") == "time.perf_counter_ns",
            snapshot.get("immutable_bytes_materialized") is True,
            snapshot.get("hash_consumed_same_payload") is True,
            snapshot.get("parse_consumed_same_payload") is True,
        )
    )


def _verify_subprocess_assignment(
    *,
    report: Mapping[str, Any],
    report_payload: bytes,
    request: Mapping[str, object],
    reference: object,
    replay_export_bytes: bytes,
    replay_assignment_bytes: bytes,
    expected_input_hashes: Mapping[str, str],
    expected_source_hashes: Mapping[str, str],
) -> dict[str, bool]:
    reference_export = np.ascontiguousarray(reference.export_xyz, dtype="<f4").tobytes(
        order="C"
    )
    reference_assignment = np.ascontiguousarray(
        reference.support_rank,
        dtype="<i8",
    ).tobytes(order="C")
    input_snapshots = report.get("input_file_snapshots")
    input_snapshot_checks = isinstance(input_snapshots, Mapping) and set(
        input_snapshots
    ) == set(expected_input_hashes)
    if input_snapshot_checks:
        input_snapshot_checks = all(
            _replay_snapshot_valid(
                input_snapshots[name],
                expected_path=f"solver_inputs/{name}",
                expected_sha256=expected_input_hashes[name],
            )
            for name in expected_input_hashes
        )
    request_snapshot_check = _replay_snapshot_valid(
        report.get("request_snapshot"),
        expected_path="assignment_replay/request.json",
        expected_sha256=sha256_bytes(canonical_json_bytes(request)),
    )
    expected_project_module_paths = {
        "eval.stda_f0_round": str(
            (CODE_ROOT / "eval/stda_f0_round.py").resolve(strict=True)
        )
    }
    checks = {
        "schema": report.get("schema") == REPLAY_RESULT_SCHEMA,
        "protocol_sha256": report.get("protocol_sha256") == PROTOCOL_SHA256,
        "protocol_freeze_commit": (
            report.get("protocol_freeze_commit") == PROTOCOL_FREEZE_COMMIT
        ),
        "request_file_sha256": report.get("request_file_sha256")
        == sha256_bytes(canonical_json_bytes(request)),
        "request_payload_sha256": report.get("request_payload_sha256")
        == request.get("request_payload_sha256"),
        "cuda_visible_devices_empty": report.get("cuda_visible_devices") == "",
        "no_cuda_modules": report.get("forbidden_cuda_modules_loaded") == [],
        "runtime_source_hashes": report.get("runtime_source_sha256")
        == dict(sorted(expected_source_hashes.items())),
        "runtime_source_hashes_stable": (
            report.get("runtime_source_hashes_stable") is True
        ),
        "project_module_paths": report.get("project_module_paths")
        == expected_project_module_paths,
        "project_module_paths_stable": (
            report.get("project_module_paths_stable") is True
        ),
        "isolated_interpreter": isinstance(
            report.get("interpreter_isolation"), Mapping
        )
        and report["interpreter_isolation"].get("isolated") is True
        and report["interpreter_isolation"].get("no_site") is True
        and report["interpreter_isolation"].get("no_user_site") is True
        and report["interpreter_isolation"].get("sitecustomize_loaded") is False,
        "input_hashes": report.get("input_files_sha256")
        == dict(sorted(expected_input_hashes.items())),
        "input_snapshots": input_snapshot_checks,
        "request_snapshot": request_snapshot_check,
        "objective": report.get("objective") == reference.objective,
        "assignment_sha256": report.get("assignment_sha256")
        == reference.assignment_sha256,
        "selected_id_sha256": report.get("selected_id_sha256")
        == reference.selected_id_sha256,
        "export_sha256": report.get("export_sha256") == reference.export_sha256,
        "objective_sha256": report.get("objective_sha256")
        == reference.objective_sha256,
        "result_digest_sha256": report.get("result_digest_sha256")
        == reference.digest_sha256,
        "assignment_bytes_identical": replay_assignment_bytes
        == reference_assignment,
        "assignment_raw_sha256": report.get("assignment_raw_sha256")
        == sha256_bytes(reference_assignment),
        "assignment_count": report.get("assignment_count")
        == reference.support_rank.size,
        "export_bytes_identical": replay_export_bytes == reference_export,
        "export_raw_sha256": report.get("export_raw_sha256")
        == sha256_bytes(reference_export),
        "export_count": report.get("export_count") == reference.export_xyz.shape[0],
        "canonical_report": canonical_json_bytes(report) == report_payload,
        "internal_verification": isinstance(report.get("verification"), dict)
        and all(report["verification"].values()),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"STDA clean assignment replay mismatch: {failed}")
    return checks


def _hall_target_summary(
    *,
    witness: object,
    greedy_sidecar: object,
    domain: StructuralDomain,
) -> dict[str, object]:
    slot_atom = greedy_sidecar.slot_atom_id[witness.slot_rows]
    unique_atom, multiplicity = np.unique(slot_atom, return_counts=True)
    atom_rows = np.searchsorted(greedy_sidecar.atom_id, unique_atom)
    xyz = greedy_sidecar.atom_xyz[atom_rows]
    radius = np.linalg.norm(xyz, axis=1)
    rays: dict[int, int] = {}
    range_atoms = [0, 0, 0, 0]
    range_slots = [0, 0, 0, 0]
    multiplicity_rows: list[dict[str, int]] = []
    for atom_id, count, point, atom_radius in zip(
        unique_atom,
        multiplicity,
        xyz,
        radius,
    ):
        x, y, z = (float(value) for value in point)
        azimuth = math.atan2(y, x)
        if azimuth == math.pi:
            azimuth = -math.pi
        ratio = 0.0 if atom_radius == 0.0 else z / float(atom_radius)
        elevation = math.asin(max(-1.0, min(1.0, ratio)))
        ray_id = domain.ray_id(azimuth, elevation)
        rays[ray_id] = rays.get(ray_id, 0) + int(count)
        if atom_radius < 30.0:
            stratum = 0
        elif atom_radius < 60.0:
            stratum = 1
        elif atom_radius < 120.0:
            stratum = 2
        else:
            stratum = 3
        range_atoms[stratum] += 1
        range_slots[stratum] += int(count)
        multiplicity_rows.append(
            {"canonical_atom_id": int(atom_id), "slot_multiplicity": int(count)}
        )
    return {
        "reachable_slot_count": int(witness.slot_rows.size),
        "unique_canonical_atom_count": int(unique_atom.size),
        "atom_multiplicity": multiplicity_rows,
        "multiplicity_sha256": sha256_bytes(
            np.column_stack((unique_atom, multiplicity)).astype("<i8").tobytes(order="C")
        ),
        "range_labels": ["0_30", "30_60", "60_120", "outside"],
        "range_unique_atom_count": range_atoms,
        "range_slot_count": range_slots,
        "ray_slot_count": [
            {"ray_id": ray_id, "slot_count": rays[ray_id]} for ray_id in sorted(rays)
        ],
    }


def _write_result_arrays(
    *,
    root: Path,
    directory: Path,
    certificate: object,
    decision: object | None,
    pointwise: object | None,
    greedy: object | None,
    records: dict[str, dict[str, object]],
) -> dict[str, str]:
    arrays: dict[str, tuple[np.ndarray, str]] = {
        "matching_custom_slot_to_support_rank.npy": (
            certificate.custom_matching.slot_to_support_rank,
            "<i8",
        ),
        "matching_custom_support_to_slot_row.npy": (
            certificate.custom_matching.support_to_slot_row,
            "<i8",
        ),
        "matching_scipy_slot_to_support_rank.npy": (
            certificate.scipy_matching.slot_to_support_rank,
            "<i8",
        ),
        "matching_scipy_support_to_slot_row.npy": (
            certificate.scipy_matching.support_to_slot_row,
            "<i8",
        ),
    }
    if certificate.hall_witness is not None:
        witness = certificate.hall_witness
        arrays.update(
            {
                "hall_reachable_slot_rows.npy": (witness.slot_rows, "<i8"),
                "hall_reachable_support_ranks.npy": (witness.support_ranks, "<i8"),
                "hall_reachable_slot_ids.npy": (witness.slot_ids, "<i8"),
                "hall_reachable_support_ids.npy": (witness.support_ids, "<i8"),
            }
        )
    if decision is not None:
        arrays.update(
            {
                "decision_slot_row.npy": (decision.slot_row, "<i8"),
                "decision_slot_id.npy": (decision.slot_id, "<i8"),
                "decision_support_rank.npy": (decision.support_rank, "<i8"),
                "decision_support_id.npy": (decision.support_id, "<i8"),
                "decision_edge_cost.npy": (decision.edge_cost, "<i8"),
                "decision_selected_support_rank.npy": (
                    decision.selected_support_rank,
                    "<i8",
                ),
                "decision_selected_support_id.npy": (
                    decision.selected_support_id,
                    "<i8",
                ),
            }
        )
    if pointwise is not None:
        arrays.update(
            {
                "packed_pointwise_selected_support_rank.npy": (
                    pointwise.selected_support_rank,
                    "<i8",
                ),
                "packed_pointwise_selected_support_id.npy": (
                    pointwise.selected_support_id,
                    "<i8",
                ),
                "packed_pointwise_nearest_atom_id.npy": (
                    pointwise.nearest_atom_id,
                    "<i8",
                ),
                "packed_pointwise_squared_distance.npy": (
                    pointwise.squared_distance,
                    "<f8",
                ),
                "packed_pointwise_distance_m.npy": (pointwise.distance_m, "<f8"),
            }
        )
    if greedy is not None:
        arrays.update(
            {
                "round_robin_greedy_round_index.npy": (greedy.round_index, "<i8"),
                "round_robin_greedy_atom_id.npy": (greedy.atom_id, "<i8"),
                "round_robin_greedy_slot_row.npy": (greedy.slot_row, "<i8"),
                "round_robin_greedy_slot_id.npy": (greedy.slot_id, "<i8"),
                "round_robin_greedy_support_rank.npy": (greedy.support_rank, "<i8"),
                "round_robin_greedy_support_id.npy": (greedy.support_id, "<i8"),
                "round_robin_greedy_edge_cost.npy": (greedy.edge_cost, "<i8"),
                "round_robin_greedy_selected_support_rank.npy": (
                    greedy.selected_support_rank,
                    "<i8",
                ),
                "round_robin_greedy_selected_support_id.npy": (
                    greedy.selected_support_id,
                    "<i8",
                ),
                "round_robin_greedy_ordered_atom_id.npy": (
                    greedy.atom_order.ordered_atom_id,
                    "<i8",
                ),
                "round_robin_greedy_order_digest_bytes.npy": (
                    greedy.atom_order.ordered_digest_bytes,
                    "<u1",
                ),
            }
        )
    return _write_array_set(
        root=root,
        directory=directory,
        arrays=arrays,
        records=records,
    )


def _write_export_pair(
    *,
    root: Path,
    exports_dir: Path,
    arm: str,
    xyz: np.ndarray,
    records: dict[str, dict[str, object]],
) -> dict[str, object]:
    export = np.ascontiguousarray(np.asarray(xyz, dtype="<f4"))
    if export.ndim != 2 or export.shape[1] != 3 or not np.isfinite(export).all():
        raise ValueError(f"STDA {arm} export must be finite float32 XYZ")
    binary_path = exports_dir / f"{arm}.bin"
    npy_path = exports_dir / f"{arm}.npy"
    _write_new_fsynced(binary_path, export.tobytes(order="C"))
    _write_npy_new_fsynced(npy_path, export, "<f4")
    binary_record = _remember_file(records, root=root, path=binary_path)
    npy_record = _remember_file(records, root=root, path=npy_path)
    return {
        "arm": arm,
        "count": int(export.shape[0]),
        "dtype": "<f4",
        "bin": binary_record,
        "npy": npy_record,
        "raw_xyz_sha256": str(binary_record["sha256"]),
        "formal_spacing_verification_pending": True,
    }


def _cardinality_report(certificate: object) -> dict[str, object]:
    witness = certificate.hall_witness
    return {
        "source_capacity": certificate.source_capacity,
        "requested_target_mass": certificate.requested_target_mass,
        "transported_mass": certificate.transported_mass,
        "unmatched_target_mass": certificate.unmatched_target_mass,
        "unused_source_mass": certificate.unused_source_mass,
        "custom_matching": {
            "method": certificate.custom_matching.method,
            "cardinality": certificate.custom_matching.cardinality,
            "digest_sha256": certificate.custom_matching.digest_sha256,
        },
        "scipy_matching": {
            "method": certificate.scipy_matching.method,
            "cardinality": certificate.scipy_matching.cardinality,
            "digest_sha256": certificate.scipy_matching.digest_sha256,
        },
        "hall_witness": (
            None
            if witness is None
            else {
                "reachable_slot_count": int(witness.slot_rows.size),
                "reachable_support_count": int(witness.support_ranks.size),
                "hall_deficit": witness.hall_deficit,
                "global_deficit": witness.global_deficit,
                "slot_set_sha256": witness.slot_set_sha256,
                "support_set_sha256": witness.support_set_sha256,
                "matching_sha256": witness.matching_sha256,
                "digest_sha256": witness.digest_sha256,
            }
        ),
        "digest_sha256": certificate.digest_sha256,
    }


def _finalize_report(
    *,
    output_dir: Path,
    report: dict[str, object],
    expected_source_hashes: Mapping[str, str],
    oracle_frame_started_ns: int,
) -> dict[str, object]:
    _require_source_hashes(expected_source_hashes)
    report_path = output_dir / "oracle_report.json"
    report_payload = canonical_json_bytes(report)
    _write_new_fsynced(report_path, report_payload)
    report_sha = sha256_file(report_path)
    if report_sha != sha256_bytes(report_payload):
        raise AssertionError("STDA oracle report changed after fsync")
    complete = {
        "schema": ORACLE_COMPLETE_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "status": report["status"],
        "oracle_report_path": "oracle_report.json",
        "oracle_report_bytes": len(report_payload),
        "oracle_report_sha256": report_sha,
    }
    complete_path = output_dir / "ORACLE_COMPLETE.json"
    complete_payload = canonical_json_bytes(complete)
    _write_new_fsynced(complete_path, complete_payload)
    _fsync_directory(output_dir)
    if sha256_file(report_path) != report_sha:
        raise AssertionError("STDA oracle report changed after completion marker")
    if sha256_file(complete_path) != sha256_bytes(complete_payload):
        raise AssertionError("STDA oracle completion marker changed after fsync")
    _require_source_hashes(expected_source_hashes)
    oracle_child_pre_metric_ended_ns = time.perf_counter_ns()
    timing = {
        "schema": "stda_f0_oracle_child_timing_v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "status": report["status"],
        "frame_key": report["frame"]["frame_key"],
        "oracle_frame_started_perf_counter_ns": oracle_frame_started_ns,
        "oracle_child_pre_metric_ended_perf_counter_ns": (
            oracle_child_pre_metric_ended_ns
        ),
        "oracle_child_pre_metric_ns": (
            oracle_child_pre_metric_ended_ns - oracle_frame_started_ns
        ),
        "authoritative_oracle_frame_ns": False,
        "parent_is_sole_oracle_frame_authority": True,
        "support_reverification_excluded": True,
        "boundary": (
            "immediately_before_the_sole_target_cache_byte_read_to_after_oracle_"
            "report_and_completion_fsync_rehash_and_runtime_source_rehash_before_"
            "cuda_metrics_and_independent_verifier"
        ),
        "timing_receipt_publication_excluded": True,
    }
    timing_path = output_dir / "ORACLE_FRAME_TIMING.json"
    timing_payload = canonical_json_bytes(timing)
    _write_new_fsynced(timing_path, timing_payload)
    _fsync_directory(output_dir)
    if sha256_file(timing_path) != sha256_bytes(timing_payload):
        raise AssertionError("STDA oracle timing receipt changed after fsync")
    return complete


def _base_report(
    *,
    status: str,
    sequence: int,
    radar_index: int,
    support_report: Mapping[str, object],
    target_report: Mapping[str, object],
    timings: Mapping[str, int],
    records: Mapping[str, Mapping[str, object]],
    started_ns: int,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    runtime_source_hashes = _require_source_hashes(source_hashes)
    project_module_origins = _require_project_import_closure()
    critical_source_hashes = {
        relative: runtime_source_hashes[relative]
        for relative in ORACLE_CRITICAL_SOURCE_PATHS
    }
    orchestrator_source_hashes = {
        relative: runtime_source_hashes[relative]
        for relative in ORCHESTRATOR_SOURCE_PATHS
    }
    allocation_core_ns = _allocation_core_ns(timings)
    return {
        "schema": ORACLE_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "status": status,
        "frame": {
            "sequence": sequence,
            "radar_index": radar_index,
            "frame_key": f"seq{sequence:02d}/radar{radar_index:05d}",
        },
        "cpu_only": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "forbidden_cuda_modules_loaded": [],
            "geometry_evaluator_used": False,
            "candidate_reconstruction_used": False,
            "trusted_site_package_paths": [
                str(path) for path in TRUSTED_SITE_PACKAGE_PATHS
            ],
            "interpreter_flags": {
                "isolated": bool(sys.flags.isolated),
                "no_site": bool(sys.flags.no_site),
                "no_user_site": bool(sys.flags.no_user_site),
            },
            "sitecustomize_loaded": "sitecustomize" in sys.modules,
        },
        "support": dict(support_report),
        "target": dict(target_report),
        "timings_ns": {
            **dict(sorted(timings.items())),
            "allocation_core_ns": allocation_core_ns,
            "oracle_pre_report_ns": time.perf_counter_ns() - started_ns,
        },
        "resource_snapshot": {
            "pid": os.getpid(),
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "process_tree_monitoring_owned_by_orchestrator": True,
        },
        "source_sha256": runtime_source_hashes,
        "critical_runtime_source_sha256": critical_source_hashes,
        "orchestrator_source_sha256": orchestrator_source_hashes,
        "runtime_source_hashes_stable": True,
        "project_module_origins": project_module_origins,
        "payload_files": [records[name] for name in sorted(records)],
    }


def run_oracle_phase(
    *,
    support_frame_dir: Path,
    target_cache_path: Path,
    resources_dir: Path,
    output_dir: Path,
    sequence: int,
    radar_index: int,
    expected_support_sha256: str,
    replay_script: Path | None = None,
    python_executable: Path | None = None,
) -> dict[str, object]:
    """Execute one frozen frame and return its canonical report document."""

    started_ns = time.perf_counter_ns()
    _require_isolated_interpreter()
    require_cpu_only_environment()
    _require_project_import_closure()
    runtime_source_hashes = _source_hashes()
    expected_support_sha256 = _validate_sha256(
        expected_support_sha256,
        label="expected_support_sha256",
    )
    if sequence < 0 or radar_index < 0:
        raise ValueError("STDA frame indices must be nonnegative")
    support_frame_dir = Path(support_frame_dir).resolve(strict=True)
    target_cache_path = Path(os.path.abspath(os.fspath(target_cache_path)))
    if not target_cache_path.parent.resolve(strict=True).is_dir():
        raise ValueError("STDA target cache parent is unavailable")
    resources_dir = Path(resources_dir).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if output_dir == support_frame_dir or support_frame_dir in output_dir.parents:
        raise ValueError("STDA oracle output cannot modify the immutable support directory")
    if output_dir.exists():
        raise FileExistsError(f"STDA oracle output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output_dir.parent)

    timings: dict[str, int] = {}
    records: dict[str, dict[str, object]] = {}
    support, commitment, support_payload, support_report = _timed(
        timings,
        "support_reverification_ns",
        lambda: _verify_support(
            support_frame_dir,
            expected_support_sha256=expected_support_sha256,
        ),
    )
    if sha256_bytes(support_payload) != commitment.support_sha256:
        raise AssertionError("STDA support bytes changed between verifier layers")

    if not commitment.capacity_status.capacity_sufficient:
        if commitment.capacity_status.terminal_status != CAPACITY_NO_GO_STATUS:
            raise AssertionError("STDA insufficient support lacks the frozen no-go status")
        _support_unchanged(support_report)
        target_cache_payload, target_cache_provenance = _timed(
            timings,
            "target_cache_provenance_ns",
            lambda: _read_immutable_path_bytes(
                target_cache_path,
                label="STDA capacity-no-go target cache",
            ),
        )
        if sha256_bytes(target_cache_payload) != target_cache_provenance[
            "cache_sha256"
        ]:
            raise AssertionError("STDA target cache immutable payload hash changed")
        oracle_frame_started_ns = int(
            target_cache_provenance["read_started_perf_counter_ns"]
        )
        del target_cache_payload
        require_cpu_only_environment()
        report = _base_report(
            status=CAPACITY_NO_GO_STATUS,
            sequence=sequence,
            radar_index=radar_index,
            support_report=support_report,
            target_report={
                **target_cache_provenance,
                "opened": True,
                "reason": "capacity_no_go_records_cache_provenance_without_array_parse",
                "path_disclosed_to_support": False,
                "array_loader_called": False,
                "cache_arrays_read": [],
                "target_array_materialized": False,
            },
            timings=timings,
            records=records,
            started_ns=started_ns,
            source_hashes=runtime_source_hashes,
        )
        report["scientific_scope"] = commitment.capacity_status.conclusion_scope
        _finalize_report(
            output_dir=output_dir,
            report=report,
            expected_source_hashes=runtime_source_hashes,
            oracle_frame_started_ns=oracle_frame_started_ns,
        )
        return report

    target_cache_payload, target_cache_provenance = _timed(
        timings,
        "target_cache_snapshot_ns",
        lambda: _read_immutable_path_bytes(
            target_cache_path,
            label="STDA target cache",
        ),
    )
    oracle_frame_started_ns = int(
        target_cache_provenance["read_started_perf_counter_ns"]
    )
    target_started = oracle_frame_started_ns
    target_load = _timed(
        timings,
        "target_load_ns",
        lambda: load_target_xyz_confidence_bytes(
            target_cache_payload,
            support_commit_sha256=commitment.support_sha256,
        ),
    )
    if target_load.support_commit_sha256 != commitment.support_sha256:
        raise AssertionError("STDA target loader lost the support commitment gate")
    target_shape = tuple(int(value) for value in target_load.target_xyz_confidence.shape)
    target_bytes = np.ascontiguousarray(
        target_load.target_xyz_confidence,
        dtype="<f4",
    ).tobytes(order="C")
    if sha256_bytes(target_bytes) != target_load.target_tensor_sha256:
        raise AssertionError("STDA target tensor bytes changed after sole approved load")
    target_report: dict[str, object] = {
        **target_cache_provenance,
        "opened": True,
        "array_loader_called": True,
        "loader": "eval.stda_f0_fit.load_target_xyz_confidence_bytes",
        "target_tensor_sha256": target_load.target_tensor_sha256,
        "target_shape": list(target_shape),
        "target_dtype": "<f4",
        "cache_arrays_read": list(target_load.cache_arrays_read),
        "support_commit_sha256": target_load.support_commit_sha256,
        "target_open_to_load_complete_ns": time.perf_counter_ns() - target_started,
        "loader_consumed_same_payload_as_cache_sha256": (
            target_load.cache_sha256 == target_cache_provenance["cache_sha256"]
        ),
    }
    if not target_report["loader_consumed_same_payload_as_cache_sha256"]:
        raise AssertionError("STDA target parser did not consume the bound cache payload")
    del target_cache_payload

    atoms = _timed(
        timings,
        "canonicalize_target_atoms_ns",
        lambda: canonicalize_target_atoms(target_load.target_xyz_confidence),
    )
    demand = _timed(
        timings,
        "build_midpoint_demand_slots_ns",
        lambda: build_midpoint_demand_slots(atoms),
    )
    canonical_support = _timed(
        timings,
        "canonicalize_support_ns",
        lambda: canonicalize_support(support.xyz_m, support.stable_candidate_id),
    )
    if not np.array_equal(canonical_support.stable_candidate_id, support.stable_candidate_id):
        raise ValueError("STDA packed support order differs from canonical support order")
    if not np.array_equal(canonical_support.xyz_float32, support.xyz_m):
        raise ValueError("STDA packed support XYZ differs from canonical support view")
    fitted_graph = _timed(
        timings,
        "build_sparse_assignment_graph_ns",
        lambda: build_sparse_assignment_graph(
            demand,
            canonical_support.xyz_float32,
            canonical_support.stable_candidate_id,
        ),
    )
    fitted_pointwise = _timed(
        timings,
        "build_packed_pointwise_sidecar_ns",
        lambda: build_packed_pointwise_sidecar(
            atoms,
            canonical_support.xyz_float32,
            canonical_support.stable_candidate_id,
        ),
    )

    solver_arrays = {
        "support_stable_candidate_id.npy": (support.stable_candidate_id, "<i8"),
        "support_grid_cell.npy": (support.cell_xyz, "<i8"),
        "support_xyz.npy": (support.xyz_m, "<f4"),
        "support_base_confidence.npy": (support.base_confidence, "<f4"),
        "support_color.npy": (support.color_id, "<u1"),
        "graph_indptr.npy": (fitted_graph.indptr, "<i8"),
        "graph_indices.npy": (fitted_graph.indices, "<i4"),
        "graph_data.npy": (fitted_graph.data, "<i8"),
        "graph_edge_squared_distance_m2.npy": (
            fitted_graph.squared_distance_m2,
            "<f8",
        ),
        "graph_edge_distance_m.npy": (fitted_graph.distance_m, "<f8"),
        "demand_slot_id.npy": (demand.slot_id, "<i8"),
    }
    expected_solver_names = {
        filename for _, filename, _, _ in SUPPORT_ARRAY_FILES + GRAPH_ARRAY_FILES
    }
    if set(solver_arrays) != expected_solver_names:
        raise AssertionError("STDA solver serialization names differ from round loader")
    control_arrays = {
        "demand_slot_atom_id.npy": (demand.canonical_atom_id, "<i8"),
        "demand_atom_id.npy": (atoms.canonical_atom_id, "<i8"),
        "demand_atom_xyz.npy": (atoms.xyz_float64, "<f8"),
        "demand_atom_weight.npy": (atoms.aggregate_weight, "<f8"),
        "pointwise_nearest_atom_id.npy": (fitted_pointwise.nearest_atom_id, "<i8"),
        "pointwise_squared_distance.npy": (
            fitted_pointwise.squared_distance_m2,
            "<f8",
        ),
        "pointwise_distance_m.npy": (fitted_pointwise.distance_m, "<f8"),
    }
    expected_control_names = {
        filename for _, filename, _, _ in GREEDY_ARRAY_FILES + POINTWISE_ARRAY_FILES
    }
    if set(control_arrays) != expected_control_names:
        raise AssertionError("STDA control serialization names differ from round loader")
    fit_arrays = {
        "target_xyz_confidence.npy": (
            target_load.target_xyz_confidence,
            "<f4",
        ),
        "target_atom_xyz_float32.npy": (atoms.xyz_float32, "<f4"),
        "target_atom_polar_rae.npy": (atoms.polar_rae, "<f8"),
        "target_atom_cdf.npy": (atoms.cdf, "<f8"),
        "demand_slot_xyz.npy": (demand.slot_xyz, "<f8"),
        "canonical_support_xyz_float64.npy": (
            canonical_support.xyz_float64,
            "<f8",
        ),
    }

    solver_dir = output_dir / "solver_inputs"
    control_dir = output_dir / "control_inputs"
    fit_dir = output_dir / "fit_evidence"

    def serialize_fit_inputs() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        solver_hashes = _write_array_set(
            root=output_dir,
            directory=solver_dir,
            arrays=solver_arrays,
            records=records,
        )
        control_hashes = _write_array_set(
            root=output_dir,
            directory=control_dir,
            arrays=control_arrays,
            records=records,
        )
        fit_hashes = _write_array_set(
            root=output_dir,
            directory=fit_dir,
            arrays=fit_arrays,
            records=records,
        )
        target_bin = fit_dir / "target_xyz_confidence.bin"
        _write_new_fsynced(target_bin, target_bytes)
        target_record = _remember_file(records, root=output_dir, path=target_bin)
        fit_hashes[target_bin.name] = str(target_record["sha256"])
        os.chmod(target_bin, 0o444)
        _fsync_directory(fit_dir)
        return solver_hashes, control_hashes, fit_hashes

    solver_hashes, control_hashes, fit_hashes = _timed(
        timings,
        "fit_input_serialization_ns",
        serialize_fit_inputs,
    )
    fit_binding = {
        "schema": "stda_f0_fit_binding_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "support_bin_sha256": commitment.support_sha256,
        "target_cache_sha256": target_load.cache_sha256,
        "target_tensor_sha256": target_load.target_tensor_sha256,
        "canonical_target_atoms_sha256": atoms.digest_sha256,
        "demand_slots_sha256": demand.digest_sha256,
        "canonical_support_sha256": canonical_support.digest_sha256,
        "sparse_graph_sha256": fitted_graph.digest_sha256,
        "packed_pointwise_sidecar_sha256": fitted_pointwise.digest_sha256,
        "solver_input_files_sha256": dict(sorted(solver_hashes.items())),
        "control_input_files_sha256": dict(sorted(control_hashes.items())),
        "fit_evidence_files_sha256": dict(sorted(fit_hashes.items())),
    }
    fit_binding_path = output_dir / "fit_binding.json"

    def serialize_fit_binding() -> None:
        _write_new_fsynced(fit_binding_path, canonical_json_bytes(fit_binding))
        _remember_file(records, root=output_dir, path=fit_binding_path)
        _fsync_directory(output_dir)

    _timed(
        timings,
        "fit_binding_serialization_ns",
        serialize_fit_binding,
    )

    fit_digests = {
        "canonical_target_atoms_sha256": atoms.digest_sha256,
        "demand_slots_sha256": demand.digest_sha256,
        "canonical_support_sha256": canonical_support.digest_sha256,
        "sparse_graph_sha256": fitted_graph.digest_sha256,
        "packed_pointwise_sidecar_sha256": fitted_pointwise.digest_sha256,
    }
    del serialize_fit_inputs
    del solver_arrays, control_arrays, fit_arrays, target_bytes
    del target_load, atoms, demand, canonical_support
    del fitted_graph, fitted_pointwise
    gc.collect()

    def reload_round_inputs() -> tuple[object, object, object, object]:
        round_support = load_packed_support(
            solver_dir,
            expected_sha256=solver_hashes,
        )
        round_graph = load_assignment_graph(
            solver_dir,
            support_cardinality=round_support.count,
            expected_sha256=solver_hashes,
            require_formal_shape=True,
        )
        greedy_sidecar = load_greedy_sidecar(
            control_dir,
            expected_sha256=control_hashes,
        )
        pointwise_sidecar = load_pointwise_sidecar(
            control_dir,
            expected_sha256=control_hashes,
        )
        return round_support, round_graph, greedy_sidecar, pointwise_sidecar

    round_support, round_graph, greedy_sidecar, pointwise_sidecar = _timed(
        timings,
        "round_input_reload_ns",
        reload_round_inputs,
    )
    os.chmod(solver_dir, 0o555)
    os.chmod(control_dir, 0o555)
    os.chmod(fit_dir, 0o555)
    _fsync_directory(output_dir)

    certificate = _timed(
        timings,
        "maximum_cardinality_certificate_ns",
        lambda: maximum_cardinality_certificate(round_support, round_graph),
    )
    custom_checks = _timed(
        timings,
        "matching_replay_custom_ns",
        lambda: verify_matching(round_graph, certificate.custom_matching),
    )
    scipy_checks = _timed(
        timings,
        "matching_replay_scipy_ns",
        lambda: verify_matching(round_graph, certificate.scipy_matching),
    )
    if not custom_checks.passed or not scipy_checks.passed:
        raise AssertionError("STDA maximum-matching verification failed")
    if certificate.custom_matching.cardinality != certificate.scipy_matching.cardinality:
        raise AssertionError("STDA dual maximum-matching cardinalities differ")

    hall_checks: dict[str, bool] | None = None
    hall_summary: dict[str, object] | None = None
    structural_domain: StructuralDomain | None = None
    structural_domain_report: dict[str, object] | None = None
    if certificate.hall_witness is not None:
        def replay_hall() -> tuple[
            dict[str, bool],
            dict[str, object],
            StructuralDomain,
            dict[str, object],
        ]:
            checks = verify_hall_witness(
                round_support,
                round_graph,
                certificate.custom_matching,
                certificate.hall_witness,
            )
            if not checks.passed:
                raise AssertionError("STDA Hall witness replay failed")
            domain, domain_report = _load_structural_domain(resources_dir)
            summary = _hall_target_summary(
                witness=certificate.hall_witness,
                greedy_sidecar=greedy_sidecar,
                domain=domain,
            )
            return _checks_dict(checks), summary, domain, domain_report

        hall_checks, hall_summary, structural_domain, structural_domain_report = _timed(
            timings,
            "hall_replay_ns",
            replay_hall,
        )

    if certificate.transported_mass < REQUIRED_EXPORT_COUNT:
        result_hashes = _timed(
            timings,
            "result_array_serialization_ns",
            lambda: _write_result_arrays(
                root=output_dir,
                directory=output_dir / "round_results",
                certificate=certificate,
                decision=None,
                pointwise=None,
                greedy=None,
                records=records,
            ),
        )
        _timed(
            timings,
            "final_input_reverification_ns",
            lambda: _reverify_round_inputs(
                solver_dir=solver_dir,
                solver_hashes=solver_hashes,
                control_dir=control_dir,
                control_hashes=control_hashes,
                support_report=support_report,
            ),
        )
        report = _base_report(
            status=GRAPH_NO_GO_STATUS,
            sequence=sequence,
            radar_index=radar_index,
            support_report=support_report,
            target_report=target_report,
            timings=timings,
            records=records,
            started_ns=started_ns,
            source_hashes=runtime_source_hashes,
        )
        report.update(
            {
                "fit_binding": fit_binding,
                "fit_digests": fit_digests,
                "round_input_binding": {
                    "solver_input_files_sha256": dict(sorted(solver_hashes.items())),
                    "control_input_files_sha256": dict(sorted(control_hashes.items())),
                    "fit_evidence_files_sha256": dict(sorted(fit_hashes.items())),
                    "round_support_sha256": round_support.digest_sha256,
                    "round_graph_sha256": round_graph.digest_sha256,
                    "round_greedy_sidecar_sha256": greedy_sidecar.digest_sha256,
                    "round_pointwise_sidecar_sha256": pointwise_sidecar.digest_sha256,
                },
                "cardinality": _cardinality_report(certificate),
                "matching_replay": {
                    "custom": _checks_dict(custom_checks),
                    "scipy": _checks_dict(scipy_checks),
                    "hall": hall_checks,
                },
                "hall_target_summary": hall_summary,
                "result_array_files_sha256": dict(sorted(result_hashes.items())),
                "structural_domain": structural_domain_report,
                "available_arms": [],
                "scientific_scope": (
                    "full-cardinality feasibility of this committed parity support "
                    "and frozen K=256 graph only"
                ),
            }
        )
        _finalize_report(
            output_dir=output_dir,
            report=report,
            expected_source_hashes=runtime_source_hashes,
            oracle_frame_started_ns=oracle_frame_started_ns,
        )
        return report

    decision_first = _timed(
        timings,
        "decision_solver_inprocess_1_ns",
        lambda: solve_full_assignment(round_support, round_graph),
    )
    decision_second = _timed(
        timings,
        "decision_solver_inprocess_2_ns",
        lambda: solve_full_assignment(round_support, round_graph),
    )

    def replay_inprocess_decisions() -> dict[str, bool]:
        _require_identical_assignments(decision_first, decision_second)
        checks = verify_full_assignment(
            round_support,
            round_graph,
            decision_second,
            reference=decision_first,
        )
        if not checks.passed:
            raise AssertionError("STDA second in-process assignment replay failed")
        return _checks_dict(checks)

    inprocess_replay_checks = _timed(
        timings,
        "decision_solver_inprocess_replay_ns",
        replay_inprocess_decisions,
    )

    frozen_replay_script = (
        CODE_ROOT / "scripts/stda_f0_assignment_replay.py"
    ).resolve(strict=True)
    replay_script = (
        frozen_replay_script if replay_script is None else Path(replay_script)
    ).resolve(strict=True)
    if replay_script != frozen_replay_script:
        raise ValueError("STDA clean replay script path differs from the frozen source")
    python_executable = (
        Path(sys.executable) if python_executable is None else Path(python_executable)
    ).resolve(strict=True)
    replay_script_sha = sha256_file(replay_script)
    expected_replay_source_hashes = {
        relative: runtime_source_hashes[relative]
        for relative in ASSIGNMENT_REPLAY_SOURCE_PATHS
    }
    if replay_script_sha != expected_replay_source_hashes[
        "code/scripts/stda_f0_assignment_replay.py"
    ]:
        raise ValueError("STDA replay script differs from oracle runtime binding")
    replay_request = _build_replay_request(
        support_cardinality=round_support.count,
        input_hashes=solver_hashes,
        replay_script_sha256=replay_script_sha,
    )
    replay_dir = output_dir / "assignment_replay"
    replay_dir.mkdir(parents=False, exist_ok=False)
    replay_request_path = replay_dir / "request.json"
    replay_result_path = replay_dir / "result.json"
    replay_export_path = replay_dir / "export.bin"
    replay_assignment_path = replay_dir / "assignment.bin"

    def serialize_replay_request() -> None:
        _write_new_fsynced(replay_request_path, canonical_json_bytes(replay_request))
        _remember_file(records, root=output_dir, path=replay_request_path)
        _fsync_directory(replay_dir)

    _timed(
        timings,
        "assignment_replay_request_serialization_ns",
        serialize_replay_request,
    )

    def run_clean_solver() -> subprocess.CompletedProcess[bytes]:
        command = (
            str(python_executable),
            "-I",
            "-S",
            "-B",
            str(replay_script),
            "--input-root",
            str(solver_dir),
            "--request",
            str(replay_request_path),
            "--result",
            str(replay_result_path),
            "--export",
            str(replay_export_path),
            "--assignment",
            str(replay_assignment_path),
        )
        completed = subprocess.run(
            command,
            cwd=solver_dir,
            env=_clean_replay_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120.0,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "STDA clean assignment replay failed with return code "
                f"{completed.returncode}; stderr_sha256={sha256_bytes(completed.stderr)}"
            )
        if completed.stdout or completed.stderr:
            raise RuntimeError("STDA clean assignment replay emitted unexpected output")
        return completed

    _timed(
        timings,
        "decision_solver_subprocess_3_ns",
        run_clean_solver,
    )
    def replay_subprocess_evidence() -> dict[str, bool]:
        replay_result, replay_result_payload = _load_canonical_json(
            replay_result_path
        )
        replay_export_bytes = replay_export_path.read_bytes()
        replay_assignment_bytes = replay_assignment_path.read_bytes()
        checks = _verify_subprocess_assignment(
            report=replay_result,
            report_payload=replay_result_payload,
            request=replay_request,
            reference=decision_first,
            replay_export_bytes=replay_export_bytes,
            replay_assignment_bytes=replay_assignment_bytes,
            expected_input_hashes=solver_hashes,
            expected_source_hashes=expected_replay_source_hashes,
        )
        _remember_file(records, root=output_dir, path=replay_result_path)
        _remember_file(records, root=output_dir, path=replay_export_path)
        _remember_file(records, root=output_dir, path=replay_assignment_path)
        _fsync_directory(replay_dir)
        return checks

    subprocess_replay_checks = _timed(
        timings,
        "decision_solver_subprocess_replay_ns",
        replay_subprocess_evidence,
    )

    pointwise_result = _timed(
        timings,
        "packed_pointwise_control_ns",
        lambda: packed_pointwise_control(
            round_support,
            pointwise_sidecar,
            output_count=REQUIRED_EXPORT_COUNT,
        ),
    )
    pointwise_checks = _timed(
        timings,
        "packed_pointwise_replay_ns",
        lambda: verify_packed_pointwise(
            round_support,
            pointwise_sidecar,
            pointwise_result,
        ),
    )
    if not pointwise_checks.passed:
        raise AssertionError("STDA packed-pointwise replay failed")

    greedy_result = _timed(
        timings,
        "round_robin_greedy_control_ns",
        lambda: target_atom_round_robin_greedy(
            round_support,
            round_graph,
            greedy_sidecar,
            sequence=sequence,
            radar_index=radar_index,
        ),
    )
    greedy_checks = _timed(
        timings,
        "round_robin_greedy_replay_ns",
        lambda: verify_greedy_control(
            round_support,
            round_graph,
            greedy_sidecar,
            greedy_result,
            sequence=sequence,
            radar_index=radar_index,
        ),
    )
    if not greedy_checks.passed:
        raise AssertionError("STDA round-robin greedy replay failed")

    result_hashes = _timed(
        timings,
        "result_array_serialization_ns",
        lambda: _write_result_arrays(
            root=output_dir,
            directory=output_dir / "round_results",
            certificate=certificate,
            decision=decision_first,
            pointwise=pointwise_result,
            greedy=greedy_result,
            records=records,
        ),
    )

    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=False, exist_ok=False)

    def serialize_exports() -> dict[str, dict[str, object]]:
        exports = {
            "decision": _write_export_pair(
                root=output_dir,
                exports_dir=exports_dir,
                arm="decision",
                xyz=decision_first.export_xyz,
                records=records,
            ),
            "packed_pointwise": _write_export_pair(
                root=output_dir,
                exports_dir=exports_dir,
                arm="packed_pointwise",
                xyz=pointwise_result.export_xyz,
                records=records,
            ),
            "round_robin_greedy": _write_export_pair(
                root=output_dir,
                exports_dir=exports_dir,
                arm="round_robin_greedy",
                xyz=greedy_result.export_xyz,
                records=records,
            ),
        }
        _fsync_directory(exports_dir)
        return exports

    export_reports = _timed(
        timings,
        "export_serialization_ns",
        serialize_exports,
    )

    if structural_domain is None:
        structural_domain, structural_domain_report = _timed(
            timings,
            "structural_domain_load_ns",
            lambda: _load_structural_domain(resources_dir),
        )
    if structural_domain_report is None:
        raise AssertionError("STDA structural domain report is missing")

    target_snapshot_path = fit_dir / "target_xyz_confidence.bin"

    def load_structural_target_snapshot() -> tuple[bytes, np.ndarray]:
        target_structure_bytes, _ = _read_immutable_path_bytes(
            target_snapshot_path,
            label="STDA structural target snapshot",
        )
        if sha256_bytes(target_structure_bytes) != target_report[
            "target_tensor_sha256"
        ]:
            raise ValueError("STDA immutable target snapshot changed before structure")
        target_for_structure = np.frombuffer(
            target_structure_bytes,
            dtype="<f4",
        ).reshape(target_shape)
        return target_structure_bytes, target_for_structure

    target_structure_bytes, target_for_structure = _timed(
        timings,
        "structural_target_snapshot_ns",
        load_structural_target_snapshot,
    )
    structure_dir = output_dir / "structure"
    _timed(
        timings,
        "structure_evidence_directory_ns",
        lambda: structure_dir.mkdir(parents=False, exist_ok=False),
    )
    structural_reports: dict[str, dict[str, object]] = {}
    arm_exports = {
        "decision": decision_first.export_xyz,
        "packed_pointwise": pointwise_result.export_xyz,
        "round_robin_greedy": greedy_result.export_xyz,
    }
    for arm, export_xyz in arm_exports.items():
        def evaluate_arm_structure(
            export: np.ndarray = export_xyz,
        ) -> tuple[object, dict[str, object]]:
            inputs = freeze_structural_inputs(
                export,
                target_for_structure,
                domain=structural_domain,
            )
            result = evaluate_structure(inputs)
            return result, _plain_structure_report(result, structural_domain)

        structure_result, structure_report = _timed(
            timings,
            f"structure_{arm}_ns",
            evaluate_arm_structure,
        )
        structure_path = structure_dir / f"{arm}.json"

        def serialize_structure_report() -> dict[str, object]:
            _write_new_fsynced(
                structure_path,
                canonical_json_bytes(structure_report),
            )
            return _remember_file(records, root=output_dir, path=structure_path)

        record = _timed(
            timings,
            f"structure_{arm}_serialization_ns",
            serialize_structure_report,
        )
        structural_reports[arm] = {
            "structural_domain_valid": structure_result.structural_domain_valid,
            "evaluation_complete": structure_result.evaluation_complete,
            "failure_reasons": list(structure_result.failure_reasons),
            "integer_cost": structure_result.integer_cost,
            "matched_count": structure_result.matched_count,
            "complete_mapping_sha256": structure_result.complete_mapping_sha256,
            "result_sha256": structure_result.result_sha256,
            "report": record,
        }
    _timed(
        timings,
        "structure_directory_fsync_ns",
        lambda: _fsync_directory(structure_dir),
    )
    del target_for_structure, target_structure_bytes

    _timed(
        timings,
        "final_input_reverification_ns",
        lambda: _reverify_round_inputs(
            solver_dir=solver_dir,
            solver_hashes=solver_hashes,
            control_dir=control_dir,
            control_hashes=control_hashes,
            support_report=support_report,
        ),
    )

    report = _base_report(
        status=ORACLE_READY_STATUS,
        sequence=sequence,
        radar_index=radar_index,
        support_report=support_report,
        target_report=target_report,
        timings=timings,
        records=records,
        started_ns=started_ns,
        source_hashes=runtime_source_hashes,
    )
    report.update(
        {
            "fit_binding": fit_binding,
            "fit_digests": fit_digests,
            "round_input_binding": {
                "solver_input_files_sha256": dict(sorted(solver_hashes.items())),
                "control_input_files_sha256": dict(sorted(control_hashes.items())),
                "fit_evidence_files_sha256": dict(sorted(fit_hashes.items())),
                "round_support_sha256": round_support.digest_sha256,
                "round_graph_sha256": round_graph.digest_sha256,
                "round_greedy_sidecar_sha256": greedy_sidecar.digest_sha256,
                "round_pointwise_sidecar_sha256": pointwise_sidecar.digest_sha256,
            },
            "cardinality": _cardinality_report(certificate),
            "matching_replay": {
                "custom": _checks_dict(custom_checks),
                "scipy": _checks_dict(scipy_checks),
                "hall": hall_checks,
            },
            "decision": {
                "objective": decision_first.objective,
                "assignment_sha256": decision_first.assignment_sha256,
                "selected_id_sha256": decision_first.selected_id_sha256,
                "export_sha256": decision_first.export_sha256,
                "objective_sha256": decision_first.objective_sha256,
                "digest_sha256": decision_first.digest_sha256,
                "second_inprocess_replay": inprocess_replay_checks,
                "clean_subprocess_replay": subprocess_replay_checks,
                "three_solver_runs_identical": True,
            },
            "packed_pointwise": {
                "selection_sha256": pointwise_result.selection_sha256,
                "selected_id_sha256": pointwise_result.selected_id_sha256,
                "export_sha256": pointwise_result.export_sha256,
                "digest_sha256": pointwise_result.digest_sha256,
                "replay": _checks_dict(pointwise_checks),
            },
            "round_robin_greedy": {
                "failed_slot_count": greedy_result.failed_slot_count,
                "full_capacity": greedy_result.full_capacity,
                "atom_order_sha256": greedy_result.atom_order.digest_sha256,
                "assignment_sha256": greedy_result.assignment_sha256,
                "selected_id_sha256": greedy_result.selected_id_sha256,
                "export_sha256": greedy_result.export_sha256,
                "digest_sha256": greedy_result.digest_sha256,
                "replay": _checks_dict(greedy_checks),
            },
            "result_array_files_sha256": dict(sorted(result_hashes.items())),
            "exports": export_reports,
            "structural_domain": structural_domain_report,
            "structure": structural_reports,
            "available_arms": [
                "decision",
                "packed_pointwise",
                "round_robin_greedy",
            ],
            "cuda_geometry_metrics_pending": True,
            "formal_independent_byte_replay_pending": True,
        }
    )
    _finalize_report(
        output_dir=output_dir,
        report=report,
        expected_source_hashes=runtime_source_hashes,
        oracle_frame_started_ns=oracle_frame_started_ns,
    )
    return report


def _write_failure(output_dir: Path, error: BaseException) -> None:
    try:
        output_dir = Path(output_dir).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "ORACLE_FAILURE.json"
        if failure_path.exists() or (output_dir / "ORACLE_COMPLETE.json").exists():
            return
        document = {
            "schema": ORACLE_FAILURE_SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "status": IMPLEMENTATION_INVALID_STATUS,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "event_time_ns": time.time_ns(),
        }
        _write_new_fsynced(failure_path, canonical_json_bytes(document))
        _fsync_directory(output_dir)
    except BaseException:
        return


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-frame-dir", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--radar-index", type=int, required=True)
    parser.add_argument("--expected-support-sha256", required=True)
    parser.add_argument("--replay-script", type=Path)
    parser.add_argument("--python-executable", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_oracle_phase(
            support_frame_dir=args.support_frame_dir,
            target_cache_path=args.target_cache,
            resources_dir=args.resources,
            output_dir=args.output_dir,
            sequence=args.sequence,
            radar_index=args.radar_index,
            expected_support_sha256=args.expected_support_sha256,
            replay_script=args.replay_script,
            python_executable=args.python_executable,
        )
    except BaseException as error:
        _write_failure(args.output_dir, error)
        traceback.print_exc(file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
