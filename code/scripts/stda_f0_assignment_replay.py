#!/usr/bin/env python3
"""Clean CPU-only third execution of the frozen STDA-F0 assignment solver."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]


def _require_isolated_interpreter() -> None:
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.no_user_site
        and "sitecustomize" not in sys.modules
    ):
        raise RuntimeError(
            "STDA assignment replay requires python -I -S with no user site or "
            "sitecustomize"
        )


if __name__ == "__main__":
    _require_isolated_interpreter()


def _path_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _trusted_site_package_paths() -> tuple[Path, ...]:
    """Expose only package dirs owned by this interpreter, without site hooks."""

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
        raise RuntimeError(
            "STDA assignment replay cannot locate trusted site-packages under -I -S"
        )
    return tuple(sorted(paths, key=str))


TRUSTED_SITE_PACKAGE_PATHS = _trusted_site_package_paths()
_trusted_paths = {path.resolve() for path in TRUSTED_SITE_PACKAGE_PATHS}
_retained_sys_path: list[str] = []
for _entry in sys.path:
    if not _entry:
        continue
    _resolved_entry = Path(_entry).resolve()
    if (
        _resolved_entry == CODE_ROOT
        or _resolved_entry in _trusted_paths
        or "site-packages" in _resolved_entry.parts
    ):
        continue
    _value = str(_resolved_entry)
    if _value not in _retained_sys_path:
        _retained_sys_path.append(_value)
sys.path[:] = [
    str(CODE_ROOT),
    *_retained_sys_path,
    *(str(path) for path in TRUSTED_SITE_PACKAGE_PATHS),
]


import numpy as np  # noqa: E402

from eval.stda_f0_round import (  # noqa: E402
    AssignmentGraph,
    FORMAL_NEIGHBOR_COUNT,
    FORMAL_SLOT_COUNT,
    GRAPH_ARRAY_FILES,
    PackedSupport,
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_SHA256,
    SUPPORT_ARRAY_FILES,
    solve_full_assignment,
    verify_full_assignment,
)


REQUEST_SCHEMA = "stda_f0_assignment_replay_request_v1"
RESULT_SCHEMA = "stda_f0_assignment_replay_result_v1"
ALLOWED_INPUT_FILENAMES = tuple(
    filename for _, filename, _, _ in SUPPORT_ARRAY_FILES + GRAPH_ARRAY_FILES
)
FORBIDDEN_MODULE_PREFIXES = ("torch", "cupy", "jax", "tensorflow", "pynvml")
RUNTIME_SOURCE_PATHS = (
    CODE_ROOT / "scripts/stda_f0_assignment_replay.py",
    CODE_ROOT / "eval/stda_f0_round.py",
)
REQUIRED_PROJECT_MODULE_PATHS = {
    "eval.stda_f0_round": "code/eval/stda_f0_round.py",
}


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


def _runtime_source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(CODE_ROOT.parent)): sha256_file(path)
        for path in RUNTIME_SOURCE_PATHS
    }


def _require_runtime_source_hashes(
    expected: Mapping[str, str],
) -> dict[str, str]:
    observed = _runtime_source_hashes()
    if observed != dict(expected):
        raise ValueError("STDA assignment replay source bytes changed during execution")
    return observed


def _require_project_module_paths() -> dict[str, str]:
    observed: dict[str, str] = {}
    for module_name, relative_path in REQUIRED_PROJECT_MODULE_PATHS.items():
        module = sys.modules.get(module_name)
        loaded_path = getattr(module, "__file__", None)
        if not isinstance(loaded_path, str):
            raise RuntimeError(f"STDA replay module is not loaded: {module_name}")
        resolved = Path(loaded_path).resolve(strict=True)
        expected = (CODE_ROOT.parent / relative_path).resolve(strict=True)
        if resolved != expected:
            raise RuntimeError(
                f"STDA replay module path changed for {module_name}: {resolved}"
            )
        observed[module_name] = str(resolved)
    return observed


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
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
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
        "parse_consumed_same_payload": True,
    }


def require_cpu_only_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("STDA assignment replay requires CUDA_VISIBLE_DEVICES empty")
    loaded = sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in FORBIDDEN_MODULE_PREFIXES
    )
    if loaded:
        raise RuntimeError(f"STDA assignment replay loaded CUDA-capable modules: {loaded}")


def _load_canonical_json_bytes(payload: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("STDA replay request is not ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("STDA replay request must be a JSON object")
    document = dict(decoded)
    if canonical_json_bytes(document) != payload:
        raise ValueError("STDA replay request is not canonical JSON")
    return document


def _load_canonical_json(path: Path) -> tuple[dict[str, Any], bytes, dict[str, object]]:
    payload, snapshot = _read_immutable_path_bytes(
        path,
        label="STDA replay request",
    )
    return _load_canonical_json_bytes(payload), payload, snapshot


def _validate_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _request_file_hashes(document: Mapping[str, Any]) -> dict[str, str]:
    raw = document.get("input_files_sha256")
    if not isinstance(raw, dict) or set(raw) != set(ALLOWED_INPUT_FILENAMES):
        raise ValueError("STDA replay request does not name the exact solver input set")
    return {
        name: _validate_sha256(raw[name], label=f"input_files_sha256[{name}]")
        for name in ALLOWED_INPUT_FILENAMES
    }


def _validate_request(
    document: Mapping[str, Any],
    *,
    request_payload: bytes,
    input_root: Path,
    observed_input_hashes: Mapping[str, str],
) -> tuple[int, dict[str, str]]:
    if document.get("schema") != REQUEST_SCHEMA:
        raise ValueError("STDA replay request schema changed")
    if document.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("STDA replay protocol SHA-256 changed")
    if document.get("protocol_freeze_commit") != PROTOCOL_FREEZE_COMMIT:
        raise ValueError("STDA replay freeze commit changed")
    expected_request_sha = document.get("request_payload_sha256")
    request_without_self = dict(document)
    request_without_self.pop("request_payload_sha256", None)
    observed_request_sha = sha256_bytes(canonical_json_bytes(request_without_self))
    if expected_request_sha != observed_request_sha:
        raise ValueError("STDA replay request self-binding changed")
    support_cardinality = document.get("support_cardinality")
    if type(support_cardinality) is not int or support_cardinality < 10_000:
        raise ValueError("STDA replay support cardinality is invalid")
    replay_script_sha = _validate_sha256(
        document.get("replay_script_sha256"),
        label="replay_script_sha256",
    )
    if replay_script_sha != sha256_file(Path(__file__).resolve()):
        raise ValueError("STDA replay script bytes changed")

    input_root = input_root.resolve(strict=True)
    actual_names = {path.name for path in input_root.iterdir() if path.is_file()}
    if actual_names != set(ALLOWED_INPUT_FILENAMES):
        raise ValueError("STDA replay input directory contains an unexpected file")
    if any(path.is_dir() or path.is_symlink() for path in input_root.iterdir()):
        raise ValueError("STDA replay input directory contains a non-regular entry")

    hashes = _request_file_hashes(document)
    if hashes != dict(observed_input_hashes):
        raise ValueError("STDA replay immutable solver input hashes changed")
    if not request_payload.endswith(b"\n"):
        raise ValueError("STDA replay request framing changed")
    return support_cardinality, hashes


def _load_npy_payload(
    payload: bytes,
    *,
    filename: str,
    dtype: str,
    ndim: int,
) -> np.ndarray:
    loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise TypeError(f"{filename} is not a NumPy array")
    if loaded.dtype != np.dtype(dtype):
        raise ValueError(f"{filename} dtype {loaded.dtype.str} does not equal {dtype}")
    if loaded.ndim != ndim:
        raise ValueError(f"{filename} rank {loaded.ndim} does not equal {ndim}")
    if not loaded.flags.c_contiguous:
        raise ValueError(f"{filename} is not canonical C-contiguous storage")
    return loaded


def _load_solver_inputs(
    input_root: Path,
    *,
    expected_sha256: Mapping[str, str],
) -> tuple[PackedSupport, AssignmentGraph, dict[str, dict[str, object]]]:
    arrays: dict[str, np.ndarray] = {}
    snapshots: dict[str, dict[str, object]] = {}
    source_hashes: list[tuple[str, str]] = []
    for field, filename, dtype, ndim in SUPPORT_ARRAY_FILES + GRAPH_ARRAY_FILES:
        payload, snapshot = _read_immutable_path_bytes(
            input_root / filename,
            label=f"STDA replay solver input {filename}",
        )
        observed_sha256 = str(snapshot["sha256"])
        if observed_sha256 != expected_sha256.get(filename):
            raise ValueError(f"STDA replay immutable input changed: {filename}")
        arrays[field] = _load_npy_payload(
            payload,
            filename=filename,
            dtype=dtype,
            ndim=ndim,
        )
        snapshot["path"] = f"solver_inputs/{filename}"
        snapshots[filename] = snapshot
        source_hashes.append((filename, observed_sha256))

    support_fields = {field for field, _, _, _ in SUPPORT_ARRAY_FILES}
    graph_fields = {field for field, _, _, _ in GRAPH_ARRAY_FILES}
    support = PackedSupport(
        **{field: arrays[field] for field in support_fields},
        source_hashes=tuple(
            item for item in source_hashes if item[0] in {
                filename for _, filename, _, _ in SUPPORT_ARRAY_FILES
            }
        ),
    )
    graph = AssignmentGraph(
        **{field: arrays[field] for field in graph_fields},
        support_cardinality=support.count,
        source_hashes=tuple(
            item for item in source_hashes if item[0] in {
                filename for _, filename, _, _ in GRAPH_ARRAY_FILES
            }
        ),
    )
    if graph.slot_count != FORMAL_SLOT_COUNT:
        raise ValueError("Formal graph must contain exactly 10,000 slots")
    degrees = np.diff(graph.indptr)
    if not np.all(degrees == FORMAL_NEIGHBOR_COUNT):
        raise ValueError("Formal graph must contain exactly 256 edges per slot")
    if graph.edge_count != FORMAL_SLOT_COUNT * FORMAL_NEIGHBOR_COUNT:
        raise ValueError("Formal graph edge count changed")
    if not np.array_equal(
        graph.slot_id,
        np.arange(FORMAL_SLOT_COUNT, dtype="<i8"),
    ):
        raise ValueError("Formal graph slot IDs must be 0 through 9,999")
    return support, graph, snapshots


def _require_snapshot_paths_stable(
    snapshots: Mapping[str, Mapping[str, object]],
    *,
    actual_paths: Mapping[str, Path],
) -> None:
    for name, snapshot in snapshots.items():
        path = actual_paths[name]
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"STDA replay immutable input disappeared: {name}") from error
        if _stat_record(current) != snapshot.get("stat_after"):
            raise ValueError(f"STDA replay immutable input changed after read: {name}")


def _write_new_fsynced(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_assignment_replay(
    *,
    input_root: Path,
    request_path: Path,
    result_path: Path,
    export_path: Path,
    assignment_path: Path,
) -> dict[str, Any]:
    """Load only frozen solver bytes, solve once, and fsync a canonical result."""

    runtime_source_hashes = _runtime_source_hashes()
    project_module_paths = _require_project_module_paths()
    require_cpu_only_environment()
    input_root = Path(input_root).resolve(strict=True)
    request_path = Path(request_path).resolve(strict=True)
    result_path = Path(result_path).absolute()
    export_path = Path(export_path).absolute()
    assignment_path = Path(assignment_path).absolute()
    for label, path in (
        ("request", request_path),
        ("result", result_path),
        ("export", export_path),
        ("assignment", assignment_path),
    ):
        if path == input_root or input_root in path.parents:
            raise ValueError(f"STDA replay {label} path cannot enter solver input root")
    request, request_payload, request_snapshot = _load_canonical_json(request_path)
    request_snapshot["path"] = "assignment_replay/request.json"
    requested_hashes = _request_file_hashes(request)
    support, graph, input_snapshots = _load_solver_inputs(
        input_root,
        expected_sha256=requested_hashes,
    )
    observed_hashes = {
        name: str(snapshot["sha256"])
        for name, snapshot in input_snapshots.items()
    }
    support_cardinality, expected_hashes = _validate_request(
        request,
        request_payload=request_payload,
        input_root=input_root,
        observed_input_hashes=observed_hashes,
    )
    if support.count != support_cardinality:
        raise ValueError("STDA replay support cardinality differs from request")
    assignment = solve_full_assignment(support, graph)
    checks = verify_full_assignment(support, graph, assignment)
    if not checks.passed:
        raise AssertionError("STDA clean assignment replay verification failed")

    export_bytes = np.ascontiguousarray(
        assignment.export_xyz,
        dtype="<f4",
    ).tobytes(order="C")
    assignment_bytes = np.ascontiguousarray(
        assignment.support_rank,
        dtype="<i8",
    ).tobytes(order="C")
    _write_new_fsynced(export_path, export_bytes)
    _write_new_fsynced(assignment_path, assignment_bytes)
    export_file_sha = sha256_file(export_path)
    assignment_file_sha = sha256_file(assignment_path)
    if export_file_sha != sha256_bytes(export_bytes):
        raise AssertionError("STDA replay export changed after fsync")
    if assignment_file_sha != sha256_bytes(assignment_bytes):
        raise AssertionError("STDA replay assignment changed after fsync")

    input_paths = {name: input_root / name for name in input_snapshots}
    _require_snapshot_paths_stable(input_snapshots, actual_paths=input_paths)
    _require_snapshot_paths_stable(
        {"request.json": request_snapshot},
        actual_paths={"request.json": request_path},
    )
    final_input_hashes = dict(sorted(observed_hashes.items()))
    if final_input_hashes != expected_hashes:
        raise ValueError("STDA replay solver input changed during execution")
    final_input_names = {path.name for path in input_root.iterdir() if path.is_file()}
    if final_input_names != set(ALLOWED_INPUT_FILENAMES):
        raise ValueError("STDA replay input directory changed during execution")
    require_cpu_only_environment()
    _require_runtime_source_hashes(runtime_source_hashes)
    if _require_project_module_paths() != project_module_paths:
        raise ValueError("STDA replay project module paths changed during execution")

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "request_file_sha256": sha256_bytes(request_payload),
        "request_payload_sha256": request["request_payload_sha256"],
        "replay_script_sha256": sha256_file(Path(__file__).resolve()),
        "runtime_source_sha256": dict(sorted(runtime_source_hashes.items())),
        "runtime_source_hashes_stable": True,
        "project_module_paths": dict(sorted(project_module_paths.items())),
        "project_module_paths_stable": True,
        "interpreter_isolation": {
            "isolated": bool(sys.flags.isolated),
            "no_site": bool(sys.flags.no_site),
            "no_user_site": bool(sys.flags.no_user_site),
            "sitecustomize_loaded": "sitecustomize" in sys.modules,
            "trusted_site_package_paths": [
                str(path) for path in TRUSTED_SITE_PACKAGE_PATHS
            ],
        },
        "cuda_visible_devices": "",
        "forbidden_cuda_modules_loaded": [],
        "input_files_sha256": final_input_hashes,
        "input_file_snapshots": {
            name: input_snapshots[name] for name in sorted(input_snapshots)
        },
        "request_snapshot": request_snapshot,
        "support_cardinality": support.count,
        "support_digest_sha256": support.digest_sha256,
        "graph_digest_sha256": graph.digest_sha256,
        "objective": assignment.objective,
        "assignment_sha256": assignment.assignment_sha256,
        "selected_id_sha256": assignment.selected_id_sha256,
        "export_sha256": assignment.export_sha256,
        "objective_sha256": assignment.objective_sha256,
        "result_digest_sha256": assignment.digest_sha256,
        "assignment_raw_sha256": assignment_file_sha,
        "assignment_bytes": len(assignment_bytes),
        "assignment_count": int(assignment.support_rank.size),
        "assignment_dtype": "<i8",
        "export_raw_sha256": export_file_sha,
        "export_bytes": len(export_bytes),
        "export_count": int(assignment.export_xyz.shape[0]),
        "export_dtype": "<f4",
        "verification": {
            name: bool(value)
            for name, value in checks.__dict__.items()
        },
    }
    _write_new_fsynced(result_path, canonical_json_bytes(result))
    _fsync_directory(result_path.parent)
    for parent in {export_path.parent, assignment_path.parent}:
        if parent != result_path.parent:
            _fsync_directory(parent)
    _require_snapshot_paths_stable(input_snapshots, actual_paths=input_paths)
    _require_snapshot_paths_stable(
        {"request.json": request_snapshot},
        actual_paths={"request.json": request_path},
    )
    _require_runtime_source_hashes(runtime_source_hashes)
    if _require_project_module_paths() != project_module_paths:
        raise ValueError("STDA replay project module paths changed after publication")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_assignment_replay(
        input_root=args.input_root,
        request_path=args.request,
        result_path=args.result,
        export_path=args.export,
        assignment_path=args.assignment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
