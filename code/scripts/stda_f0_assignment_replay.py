#!/usr/bin/env python3
"""Clean CPU-only third execution of the frozen STDA-F0 assignment solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eval.stda_f0_round import (  # noqa: E402
    GRAPH_ARRAY_FILES,
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_SHA256,
    SUPPORT_ARRAY_FILES,
    load_assignment_graph,
    load_packed_support,
    solve_full_assignment,
    verify_full_assignment,
)


REQUEST_SCHEMA = "stda_f0_assignment_replay_request_v1"
RESULT_SCHEMA = "stda_f0_assignment_replay_result_v1"
ALLOWED_INPUT_FILENAMES = tuple(
    filename for _, filename, _, _ in SUPPORT_ARRAY_FILES + GRAPH_ARRAY_FILES
)
FORBIDDEN_MODULE_PREFIXES = ("torch", "cupy", "jax", "tensorflow", "pynvml")


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


def _load_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = Path(path).read_bytes()
    try:
        decoded = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("STDA replay request is not ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("STDA replay request must be a JSON object")
    document = dict(decoded)
    if canonical_json_bytes(document) != payload:
        raise ValueError("STDA replay request is not canonical JSON")
    return document, payload


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
    for name, expected in hashes.items():
        if sha256_file(input_root / name) != expected:
            raise ValueError(f"STDA replay immutable input changed: {name}")
    if not request_payload.endswith(b"\n"):
        raise ValueError("STDA replay request framing changed")
    return support_cardinality, hashes


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
    request, request_payload = _load_canonical_json(request_path)
    support_cardinality, expected_hashes = _validate_request(
        request,
        request_payload=request_payload,
        input_root=input_root,
    )

    support = load_packed_support(input_root, expected_sha256=expected_hashes)
    if support.count != support_cardinality:
        raise ValueError("STDA replay support cardinality differs from request")
    graph = load_assignment_graph(
        input_root,
        support_cardinality=support.count,
        expected_sha256=expected_hashes,
        require_formal_shape=True,
    )
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

    final_input_hashes = {
        name: sha256_file(input_root / name) for name in ALLOWED_INPUT_FILENAMES
    }
    if final_input_hashes != expected_hashes:
        raise ValueError("STDA replay solver input changed during execution")
    final_input_names = {path.name for path in input_root.iterdir() if path.is_file()}
    if final_input_names != set(ALLOWED_INPUT_FILENAMES):
        raise ValueError("STDA replay input directory changed during execution")
    require_cpu_only_environment()

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "request_file_sha256": sha256_bytes(request_payload),
        "request_payload_sha256": request["request_payload_sha256"],
        "replay_script_sha256": sha256_file(Path(__file__).resolve()),
        "cuda_visible_devices": "",
        "forbidden_cuda_modules_loaded": [],
        "input_files_sha256": final_input_hashes,
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
