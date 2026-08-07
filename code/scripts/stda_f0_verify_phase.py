#!/usr/bin/env python3
"""Independently replay one immutable STDA-F0 oracle frame on CPU."""

from __future__ import annotations

from time import perf_counter_ns

MODULE_WALL_STARTED_NS = perf_counter_ns()

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pwd
import stat
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


def _bootstrap_isolated_site_packages() -> str | None:
    """Add only the interpreter-owned package directory under ``-I -S``."""

    if not sys.flags.no_site:
        return None
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    roots: list[Path] = []
    for raw_root in (Path(sys.prefix), Path(sys.executable).absolute().parent.parent):
        try:
            root = raw_root.resolve(strict=True)
        except OSError:
            continue
        if root not in roots:
            roots.append(root)
    candidates: list[Path] = []
    for root in roots:
        raw_candidate = root / "lib" / version / "site-packages"
        try:
            candidate = raw_candidate.resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        if (
            candidate.is_dir()
            and (candidate / "numpy/__init__.py").is_file()
            and (candidate / "scipy/__init__.py").is_file()
        ):
            candidates.append(candidate)
    if len(set(candidates)) != 1:
        raise RuntimeError(
            "STDA isolated verifier requires one interpreter-owned NumPy/SciPy site-packages"
        )
    selected = str(candidates[0])
    if selected not in sys.path:
        sys.path.insert(0, selected)
    return selected


ISOLATED_SITE_PACKAGES = _bootstrap_isolated_site_packages()

import numpy as np
import scipy


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eval.stda_f0_verify import (  # noqa: E402
    BOUNDARY_SEMANTICS,
    CONTROL_INPUT_FILENAMES,
    FIT_EVIDENCE_FILENAMES,
    GRAPH_NO_GO_RESULT_FILENAMES,
    MATCHING_RESULT_FILENAMES,
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_SHA256,
    READY_RESULT_FILENAMES,
    SOLVER_INPUT_FILENAMES,
    verify_csr,
    verify_hall_certificate,
    verify_matching,
    verify_packed_support_bytes,
    verify_control_export_replay,
    verify_fitted_semantic_reconstruction,
    verify_structural_replay,
)


SCHEMA = "stda_f0_independent_frame_verification_v1"
READY_STATUS = "stda_f0_oracle_ready_for_cuda_metrics"
GRAPH_NO_GO_STATUS = "stda_f0_graph_cardinality_no_go"
SUPPORT_NO_GO_STATUS = "stda_f0_packed_support_capacity_no_go"
ALLOWED_STATUSES = (SUPPORT_NO_GO_STATUS, GRAPH_NO_GO_STATUS, READY_STATUS)
ORACLE_ARM_NAMES = ("decision", "packed_pointwise", "round_robin_greedy")
ARM_ALIASES = {
    "decision": "decision",
    "packed_pointwise": "pointwise",
    "round_robin_greedy": "greedy",
}
RANGE_LABELS = ("range_0_30", "range_30_60", "range_60_120")
FORBIDDEN_MODULE_PREFIXES = ("torch", "cupy", "jax", "tensorflow", "pynvml")
PINNED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONHASHSEED": "0",
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
}
PINNED_PYTHON_VERSION = "3.10.20"
PINNED_NUMPY_VERSION = "2.2.6"
PINNED_SCIPY_VERSION = "1.15.3"
INPROCESS_REPLAY_CHECK_NAMES = frozenset(
    {
        "shape_valid",
        "all_slots_once",
        "support_capacity",
        "assignment_sidecars_equal",
        "edge_membership",
        "edge_cost_equal",
        "objective_equal",
        "export_finite",
        "export_unique_bytes",
        "canonical_export_order",
        "hashes_valid",
        "reference_equal",
    }
)
CLEAN_REPLAY_CHECK_NAMES = frozenset(
    {
        "schema",
        "protocol_sha256",
        "protocol_freeze_commit",
        "request_file_sha256",
        "request_payload_sha256",
        "cuda_visible_devices_empty",
        "no_cuda_modules",
        "runtime_source_hashes",
        "runtime_source_hashes_stable",
        "project_module_paths",
        "project_module_paths_stable",
        "isolated_interpreter",
        "input_hashes",
        "input_snapshots",
        "request_snapshot",
        "objective",
        "assignment_sha256",
        "selected_id_sha256",
        "export_sha256",
        "objective_sha256",
        "result_digest_sha256",
        "assignment_bytes_identical",
        "assignment_raw_sha256",
        "assignment_count",
        "export_bytes_identical",
        "export_raw_sha256",
        "export_count",
        "canonical_report",
        "internal_verification",
    }
)
ASSIGNMENT_REPLAY_FILENAMES = frozenset(
    {"request.json", "result.json", "export.bin", "assignment.bin"}
)
ASSIGNMENT_REPLAY_VERIFICATION_NAMES = INPROCESS_REPLAY_CHECK_NAMES


def canonical_json_bytes(document: Any) -> bytes:
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _reject_symlink_components(path: Path, *, include_leaf: bool = True) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    current = Path(parts[0])
    stop = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:stop]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ValueError(f"STDA verifier path component is unavailable: {current}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"STDA verifier path contains a symlink: {current}")


def _resolve_input_path(path: Path, *, directory: bool, label: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError(f"STDA verifier {label} path must be absolute")
    if ".." in raw.parts:
        raise ValueError(f"STDA verifier {label} path contains parent traversal")
    _reject_symlink_components(raw)
    canonical = raw.resolve(strict=True)
    mode = canonical.lstat().st_mode
    valid = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not valid:
        kind = "directory" if directory else "regular file"
        raise ValueError(f"STDA verifier {label} is not one {kind}: {canonical}")
    return canonical


def _resolve_output_path(
    path: Path,
    *,
    forbidden_roots: Sequence[Path],
    forbidden_files: Sequence[Path],
) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError("STDA verifier output path must be absolute")
    if ".." in raw.parts:
        raise ValueError("STDA verifier output path contains parent traversal")
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(raw)
    _reject_symlink_components(raw, include_leaf=False)
    parent = raw.parent.resolve(strict=True)
    canonical = parent / raw.name
    if any(_path_within(canonical, root) for root in forbidden_roots):
        raise ValueError("STDA verifier output cannot enter immutable evidence")
    if canonical in forbidden_files:
        raise ValueError("STDA verifier output aliases an immutable input")
    return canonical


def _formal_runtime_report() -> dict[str, Any]:
    environment_checks = {
        name: os.environ.get(name) == expected
        for name, expected in PINNED_ENVIRONMENT.items()
    }
    phase = os.environ.get("STDA_F0_PHASE", "")
    pythonpath = os.environ.get("PYTHONPATH")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_default = os.environ.get("CONDA_DEFAULT_ENV")
    checks = {
        "isolated": bool(sys.flags.isolated),
        "no_site": bool(sys.flags.no_site),
        "no_user_site": bool(sys.flags.no_user_site),
        "sitecustomize_absent": "sitecustomize" not in sys.modules,
        "usercustomize_absent": "usercustomize" not in sys.modules,
        "isolated_site_packages_selected": ISOLATED_SITE_PACKAGES is not None,
        "python_3_10_20": sys.version.split()[0] == PINNED_PYTHON_VERSION,
        "numpy_2_2_6": np.__version__ == PINNED_NUMPY_VERSION,
        "scipy_1_15_3": scipy.__version__ == PINNED_SCIPY_VERSION,
        "user_wangning": pwd.getpwuid(os.getuid()).pw_name == "wangning",
        "hym_environment": Path(sys.prefix).name.startswith("hym"),
        "conda_prefix_bound": conda_prefix is not None
        and Path(conda_prefix).resolve() == Path(sys.prefix).resolve(),
        "conda_default_hym": isinstance(conda_default, str)
        and conda_default.startswith("hym"),
        "pythonpath_bound_to_code_root": isinstance(pythonpath, str)
        and Path(pythonpath).resolve() == CODE_ROOT,
        "formal_phase_name": phase.startswith("verify_")
        and phase.removeprefix("verify_").isdigit(),
        "environment_exact": all(environment_checks.values()),
    }
    report = {
        "schema": "stda_f0_verifier_formal_runtime_v1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "interpreter_flags": {
            "isolated": bool(sys.flags.isolated),
            "no_site": bool(sys.flags.no_site),
            "no_user_site": bool(sys.flags.no_user_site),
        },
        "isolated_site_packages": ISOLATED_SITE_PACKAGES,
        "environment": {
            **{name: os.environ.get(name) for name in PINNED_ENVIRONMENT},
            "PYTHONPATH": pythonpath,
            "CONDA_PREFIX": conda_prefix,
            "CONDA_DEFAULT_ENV": conda_default,
            "STDA_F0_PHASE": phase,
        },
        "environment_checks": environment_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        failed.extend(
            f"environment:{name}"
            for name, passed in environment_checks.items()
            if not passed
        )
        raise RuntimeError(f"STDA formal verifier runtime is not pinned: {failed}")
    return report


def _canonical_document(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = _regular_bytes(path)
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"STDA verifier JSON is invalid: {path}") from error
    if not isinstance(document, dict) or canonical_json_bytes(document) != payload:
        raise ValueError(f"STDA verifier JSON is not canonical: {path}")
    return document, payload


def _regular_bytes(path: Path) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError(f"STDA verifier input is unavailable: {path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"STDA verifier input is not one regular file: {path}")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(payload) != after.st_size:
        raise ValueError(f"STDA verifier input changed during read: {path}")
    return payload


def _target_cache_provenance(
    target_cache: Path,
    *,
    oracle: Mapping[str, Any],
    parse_array: bool,
) -> tuple[dict[str, Any], bytes, bytes | None]:
    canonical_path = target_cache.resolve(strict=True)
    cache_payload = _regular_bytes(canonical_path)
    cache_sha256 = sha256_bytes(cache_payload)
    target_report = oracle.get("target")
    provenance_checks = {
        "oracle_target_mapping": isinstance(target_report, Mapping),
        "canonical_path_matches_oracle_provenance": isinstance(target_report, Mapping)
        and target_report.get("path") == str(canonical_path),
        "cache_sha256_matches_oracle_provenance": isinstance(target_report, Mapping)
        and target_report.get("cache_sha256") == cache_sha256,
    }
    target_bytes: bytes | None = None
    array_report: dict[str, Any] = {
        "parsed": False,
        "reason": "packed_support_capacity_no_go_does_not_parse_target_array",
    }
    if parse_array:
        stream = io.BytesIO(cache_payload)
        loaded: Any = None
        try:
            loaded = np.load(stream, allow_pickle=False)
            files = getattr(loaded, "files", None)
            if not isinstance(files, list) or "target_xyz_confidence" not in files:
                raise ValueError(
                    "STDA verifier target cache lacks target_xyz_confidence"
                )
            raw_target = loaded["target_xyz_confidence"]
            if (
                not isinstance(raw_target, np.ndarray)
                or raw_target.dtype != np.dtype("<f4")
                or raw_target.ndim != 2
                or raw_target.shape[0] == 0
                or raw_target.shape[1] != 4
                or not raw_target.flags.c_contiguous
            ):
                raise ValueError(
                    "STDA verifier target cache member is not canonical C-order <f4 (N,4)"
                )
            target = np.array(raw_target, dtype="<f4", order="C", copy=True)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError("STDA verifier target cache is invalid") from error
        finally:
            close = getattr(loaded, "close", None)
            if callable(close):
                close()
        if (
            target.ndim != 2
            or target.shape[0] == 0
            or target.shape[1] != 4
            or not bool(np.isfinite(target).all())
            or bool(np.any(target[:, 3] < 0.0))
        ):
            raise ValueError("STDA verifier target array contract changed")
        target_bytes = target.tobytes(order="C")
        tensor_sha256 = sha256_bytes(target_bytes)
        provenance_checks.update(
            {
                "oracle_marks_target_opened": isinstance(target_report, Mapping)
                and target_report.get("opened") is True,
                "target_tensor_sha256_matches_oracle": isinstance(
                    target_report, Mapping
                )
                and target_report.get("target_tensor_sha256") == tensor_sha256,
                "target_shape_matches_oracle": isinstance(target_report, Mapping)
                and target_report.get("target_shape")
                == [int(value) for value in target.shape],
                "target_dtype_matches_oracle": isinstance(target_report, Mapping)
                and target_report.get("target_dtype") == "<f4",
                "sole_cache_array_recorded": isinstance(target_report, Mapping)
                and target_report.get("cache_arrays_read")
                == ["target_xyz_confidence"],
            }
        )
        array_report = {
            "parsed": True,
            "member": "target_xyz_confidence",
            "shape": [int(value) for value in target.shape],
            "dtype": "<f4",
            "tensor_bytes": len(target_bytes),
            "tensor_sha256": tensor_sha256,
        }
    else:
        provenance_checks.update(
            {
                "oracle_marks_cache_opened": isinstance(target_report, Mapping)
                and target_report.get("opened") is True,
                "oracle_marks_array_loader_not_called": isinstance(
                    target_report, Mapping
                )
                and target_report.get("array_loader_called") is False,
                "oracle_marks_target_not_materialized": isinstance(
                    target_report, Mapping
                )
                and target_report.get("target_array_materialized") is False,
                "oracle_records_no_cache_array_reads": isinstance(
                    target_report, Mapping
                )
                and target_report.get("cache_arrays_read") == [],
            }
        )
    return (
        {
            "canonical_path": str(canonical_path),
            "cache_bytes": len(cache_payload),
            "cache_sha256": cache_sha256,
            "array": array_report,
            "checks": provenance_checks,
            "passed": all(provenance_checks.values()),
        },
        cache_payload,
        target_bytes,
    )


def _npy(payload: bytes, *, dtype: str, ndim: int, label: str) -> np.ndarray:
    stream = io.BytesIO(payload)
    loaded = np.load(stream, allow_pickle=False)
    if stream.read(1):
        raise ValueError(f"STDA verifier NPY has trailing bytes: {label}")
    if (
        not isinstance(loaded, np.ndarray)
        or loaded.dtype != np.dtype(dtype)
        or loaded.ndim != ndim
        or not loaded.flags.c_contiguous
    ):
        raise ValueError(f"STDA verifier NPY contract changed: {label}")
    return np.frombuffer(
        loaded.tobytes(order="C"),
        dtype=np.dtype(dtype),
    ).reshape(loaded.shape)


def _directory_payloads(
    path: Path,
    *,
    expected_names: Sequence[str] | frozenset[str],
) -> dict[str, bytes]:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError(f"STDA verifier directory is unavailable: {path}") from error
    if not stat.S_ISDIR(mode):
        raise ValueError(f"STDA verifier directory is invalid: {path}")
    expected = set(expected_names)
    entries = list(path.iterdir())
    observed = {child.name for child in entries}
    if observed != expected:
        raise ValueError(
            "STDA verifier directory file set changed: "
            f"{path}; unexpected={sorted(observed - expected)}, "
            f"missing={sorted(expected - observed)}"
        )
    return {name: _regular_bytes(path / name) for name in sorted(expected)}


def _payload_binding(
    payloads: Mapping[str, bytes],
    expected: Any,
) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise ValueError("STDA verifier expected file-hash mapping is absent")
    observed = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    checks = {
        "exact_file_names": set(observed) == set(expected),
        "all_hashes_match": observed == dict(expected),
    }
    return {
        "observed_sha256": dict(sorted(observed.items())),
        "expected_sha256": dict(sorted(expected.items())),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _exact_true_boolean_mapping(
    value: Any,
    *,
    expected_names: Sequence[str] | frozenset[str],
    label: str,
) -> dict[str, Any]:
    expected = set(expected_names)
    mapping = dict(value) if isinstance(value, Mapping) else {}
    checks = {
        "mapping": isinstance(value, Mapping),
        "exact_names": set(mapping) == expected,
        "all_values_are_true_booleans": set(mapping) == expected
        and all(mapping[name] is True for name in expected),
    }
    return {
        "label": label,
        "observed": mapping,
        "expected_names": sorted(expected),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _exclusive_json(path: Path, document: Mapping[str, Any]) -> tuple[str, int]:
    if not path.is_absolute():
        raise ValueError("STDA verifier output path must be absolute")
    parent = path.parent.resolve(strict=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    payload = canonical_json_bytes(document)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-",
        dir=parent,
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        linked = True
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if _regular_bytes(path) != payload:
            raise ValueError("STDA verifier output replay changed")
    except BaseException:
        temporary.unlink(missing_ok=True)
        if linked:
            path.unlink(missing_ok=True)
        raise
    return sha256_bytes(payload), len(payload)


def _load_oracle(oracle_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    complete, _ = _canonical_document(oracle_dir / "ORACLE_COMPLETE.json")
    if (
        complete.get("schema") != "stda_f0_oracle_frame_complete_v1"
        or complete.get("protocol_sha256") != PROTOCOL_SHA256
        or complete.get("protocol_freeze_commit") != PROTOCOL_FREEZE_COMMIT
        or complete.get("oracle_report_path") != "oracle_report.json"
    ):
        raise ValueError("STDA oracle completion schema changed")
    report_path = oracle_dir / "oracle_report.json"
    report, payload = _canonical_document(report_path)
    if (
        report.get("schema") != "stda_f0_oracle_frame_v1"
        or complete.get("oracle_report_sha256") != sha256_bytes(payload)
        or complete.get("oracle_report_bytes") != len(payload)
        or complete.get("status") != report.get("status")
    ):
        raise ValueError("STDA oracle report completion binding failed")
    return report, complete


def _support_replay(
    support_frame_dir: Path,
    *,
    expected_support_sha256: str,
    oracle: Mapping[str, Any],
    payload: bytes | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = _regular_bytes(support_frame_dir / "support.bin")
    replay = verify_packed_support_bytes(payload)
    oracle_support = oracle.get("support")
    checks = {
        "expected_sha256": sha256_bytes(payload) == expected_support_sha256,
        "oracle_mapping": isinstance(oracle_support, Mapping),
        "oracle_sha256": isinstance(oracle_support, Mapping)
        and oracle_support.get("sha256") == expected_support_sha256,
        "support_count": isinstance(oracle_support, Mapping)
        and oracle_support.get("support_count") == replay.get("support_count"),
        "candidate_field_sha256": isinstance(oracle_support, Mapping)
        and oracle_support.get("candidate_field_sha256")
        == replay.get("candidate_field_sha256"),
        "selected_color": isinstance(oracle_support, Mapping)
        and oracle_support.get("selected_color_id") == replay.get("selected_color_id"),
        "color_cardinalities": isinstance(oracle_support, Mapping)
        and oracle_support.get("color_cardinalities")
        == replay.get("color_cardinalities"),
        "independent_support_replay": replay.get("passed") is True,
    }
    return {"checks": checks, "replay": replay, "passed": all(checks.values())}


def _payload_record_replay(
    oracle: Mapping[str, Any],
    *,
    relative_path: str,
    payload: bytes,
) -> dict[str, Any]:
    records = oracle.get("payload_files")
    matches = (
        [
            record
            for record in records
            if isinstance(record, Mapping) and record.get("path") == relative_path
        ]
        if isinstance(records, list)
        else []
    )
    checks = {
        "exactly_one_record": len(matches) == 1,
        "bytes": len(matches) == 1 and matches[0].get("bytes") == len(payload),
        "sha256": len(matches) == 1
        and matches[0].get("sha256") == sha256_bytes(payload),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _fit_semantic_replay(
    *,
    support_payload: bytes,
    target_array_bytes: bytes,
    target_cache_sha256: str,
    oracle_dir: Path,
    oracle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    solver_payloads = _directory_payloads(
        oracle_dir / "solver_inputs", expected_names=SOLVER_INPUT_FILENAMES
    )
    control_payloads = _directory_payloads(
        oracle_dir / "control_inputs", expected_names=CONTROL_INPUT_FILENAMES
    )
    fit_payloads = _directory_payloads(
        oracle_dir / "fit_evidence", expected_names=FIT_EVIDENCE_FILENAMES
    )
    target_snapshot = fit_payloads["target_xyz_confidence.bin"]
    fit_binding, fit_binding_payload = _canonical_document(
        oracle_dir / "fit_binding.json"
    )
    embedded_fit_binding = oracle.get("fit_binding")
    fit_digests = oracle.get("fit_digests")
    round_input_binding = oracle.get("round_input_binding")
    if not isinstance(fit_digests, Mapping) or not isinstance(
        round_input_binding, Mapping
    ):
        raise ValueError("STDA semantic replay bindings are absent")
    arithmetic = verify_fitted_semantic_reconstruction(
        support_payload=support_payload,
        target_xyz_confidence_payload=target_array_bytes,
        authoritative_target_cache_sha256=target_cache_sha256,
        solver_inputs=solver_payloads,
        control_inputs=control_payloads,
        fit_evidence=fit_payloads,
        fit_binding=fit_binding,
        oracle_fit_digests=fit_digests,
        round_input_binding=round_input_binding,
    )
    checks = {
        "embedded_fit_binding_mapping": isinstance(embedded_fit_binding, Mapping),
        "fit_binding_document_equals_oracle": isinstance(
            embedded_fit_binding, Mapping
        )
        and dict(embedded_fit_binding) == fit_binding,
        "fit_binding_payload_record": _payload_record_replay(
            oracle,
            relative_path="fit_binding.json",
            payload=fit_binding_payload,
        )["passed"],
        "authoritative_cache_array_matches_fit_tensor_hash": fit_binding.get(
            "target_tensor_sha256"
        )
        == sha256_bytes(target_array_bytes),
        "authoritative_cache_bytes_match_fit_cache_hash": fit_binding.get(
            "target_cache_sha256"
        )
        == target_cache_sha256,
        "independent_arithmetic": arithmetic.get("passed") is True,
    }
    return (
        {
            "fit_binding_sha256": sha256_bytes(fit_binding_payload),
            "target_snapshot_sha256": sha256_bytes(target_snapshot),
            "arithmetic": arithmetic,
            "checks": checks,
            "passed": all(checks.values()),
        },
        solver_payloads,
        control_payloads,
    )


def _graph_replay(
    oracle_dir: Path,
    oracle: Mapping[str, Any],
    *,
    solver_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    expected_results = (
        GRAPH_NO_GO_RESULT_FILENAMES
        if oracle.get("status") == GRAPH_NO_GO_STATUS
        else READY_RESULT_FILENAMES
    )
    result_payloads = _directory_payloads(
        oracle_dir / "round_results", expected_names=expected_results
    )
    round_binding = oracle.get("round_input_binding")
    if not isinstance(round_binding, Mapping):
        raise ValueError("STDA round input binding is absent")
    solver_binding = _payload_binding(
        solver_payloads,
        round_binding.get("solver_input_files_sha256"),
    )
    result_binding = _payload_binding(
        result_payloads,
        oracle.get("result_array_files_sha256"),
    )
    support_id = _npy(
        solver_payloads["support_stable_candidate_id.npy"],
        dtype="<i8",
        ndim=1,
        label="support_stable_candidate_id.npy",
    )
    indptr = _npy(
        solver_payloads["graph_indptr.npy"],
        dtype="<i8",
        ndim=1,
        label="graph_indptr.npy",
    )
    indices = _npy(
        solver_payloads["graph_indices.npy"],
        dtype="<i4",
        ndim=1,
        label="graph_indices.npy",
    )
    data = _npy(
        solver_payloads["graph_data.npy"],
        dtype="<i8",
        ndim=1,
        label="graph_data.npy",
    )
    custom = _npy(
        result_payloads["matching_custom_slot_to_support_rank.npy"],
        dtype="<i8",
        ndim=1,
        label="matching_custom_slot_to_support_rank.npy",
    )
    scipy = _npy(
        result_payloads["matching_scipy_slot_to_support_rank.npy"],
        dtype="<i8",
        ndim=1,
        label="matching_scipy_slot_to_support_rank.npy",
    )
    custom_reverse = _npy(
        result_payloads["matching_custom_support_to_slot_row.npy"],
        dtype="<i8",
        ndim=1,
        label="matching_custom_support_to_slot_row.npy",
    )
    scipy_reverse = _npy(
        result_payloads["matching_scipy_support_to_slot_row.npy"],
        dtype="<i8",
        ndim=1,
        label="matching_scipy_support_to_slot_row.npy",
    )
    csr = verify_csr(
        indptr,
        indices,
        data,
        support_cardinality=int(support_id.size),
    )
    custom_replay = verify_matching(
        custom,
        custom_reverse,
        indptr,
        indices,
        support_cardinality=int(support_id.size),
    )
    scipy_replay = verify_matching(
        scipy,
        scipy_reverse,
        indptr,
        indices,
        support_cardinality=int(support_id.size),
    )
    cardinality = oracle.get("cardinality")
    matching_replay_report = oracle.get("matching_replay")
    custom_oracle_checks = _exact_true_boolean_mapping(
        matching_replay_report.get("custom")
        if isinstance(matching_replay_report, Mapping)
        else None,
        expected_names={
            "shape_valid",
            "sentinels_and_bounds_valid",
            "cardinality_valid",
            "reciprocal",
            "support_capacity",
            "edge_membership",
            "digest_valid",
        },
        label="oracle.matching_replay.custom",
    )
    scipy_oracle_checks = _exact_true_boolean_mapping(
        matching_replay_report.get("scipy")
        if isinstance(matching_replay_report, Mapping)
        else None,
        expected_names={
            "shape_valid",
            "sentinels_and_bounds_valid",
            "cardinality_valid",
            "reciprocal",
            "support_capacity",
            "edge_membership",
            "digest_valid",
        },
        label="oracle.matching_replay.scipy",
    )
    cardinality_checks = {
        "mapping": isinstance(cardinality, Mapping),
        "matching_replay_mapping": isinstance(matching_replay_report, Mapping),
        "custom_equals_scipy": custom_replay["cardinality"]
        == scipy_replay["cardinality"],
        "oracle_transported_mass": isinstance(cardinality, Mapping)
        and cardinality.get("transported_mass") == custom_replay["cardinality"],
        "oracle_custom": isinstance(cardinality, Mapping)
        and cardinality.get("custom_matching", {}).get("cardinality")
        == custom_replay["cardinality"],
        "oracle_custom_digest": isinstance(cardinality, Mapping)
        and cardinality.get("custom_matching", {}).get("digest_sha256")
        == custom_replay["digest_sha256"],
        "oracle_scipy": isinstance(cardinality, Mapping)
        and cardinality.get("scipy_matching", {}).get("cardinality")
        == scipy_replay["cardinality"],
        "oracle_scipy_digest": isinstance(cardinality, Mapping)
        and cardinality.get("scipy_matching", {}).get("digest_sha256")
        == scipy_replay["digest_sha256"],
        "oracle_custom_replay": custom_oracle_checks["passed"],
        "oracle_scipy_replay": scipy_oracle_checks["passed"],
    }
    hall: dict[str, Any] | None = None
    hall_identifier_binding = True
    if custom_replay["cardinality"] < 10_000:
        hall_slot_rows = _npy(
            result_payloads["hall_reachable_slot_rows.npy"],
            dtype="<i8",
            ndim=1,
            label="hall_reachable_slot_rows.npy",
        )
        hall_support_ranks = _npy(
            result_payloads["hall_reachable_support_ranks.npy"],
            dtype="<i8",
            ndim=1,
            label="hall_reachable_support_ranks.npy",
        )
        hall = verify_hall_certificate(
            hall_slot_rows,
            hall_support_ranks,
            custom,
            indptr,
            indices,
            support_cardinality=int(support_id.size),
        )
        demand_slot_id = _npy(
            solver_payloads["demand_slot_id.npy"],
            dtype="<i8",
            ndim=1,
            label="demand_slot_id.npy",
        )
        hall_identifier_binding = bool(
            np.array_equal(
                _npy(
                    result_payloads["hall_reachable_slot_ids.npy"],
                    dtype="<i8",
                    ndim=1,
                    label="hall_reachable_slot_ids.npy",
                ),
                demand_slot_id[hall_slot_rows],
            )
            and np.array_equal(
                _npy(
                    result_payloads["hall_reachable_support_ids.npy"],
                    dtype="<i8",
                    ndim=1,
                    label="hall_reachable_support_ids.npy",
                ),
                support_id[hall_support_ranks],
            )
        )
    hall_report = (
        cardinality.get("hall_witness")
        if isinstance(cardinality, Mapping)
        else None
    )
    hall_binding = hall is None and hall_report is None
    if hall is not None and isinstance(hall_report, Mapping):
        hall_binding = all(
            (
                hall_report.get("reachable_slot_count")
                == hall.get("reachable_slot_count"),
                hall_report.get("reachable_support_count")
                == hall.get("reachable_support_count"),
                hall_report.get("hall_deficit") == hall.get("hall_deficit"),
                hall_report.get("slot_set_sha256")
                == hall.get("reachable_slots_sha256"),
                hall_report.get("support_set_sha256")
                == hall.get("reachable_support_sha256"),
            )
        )
    checks = {
        "solver_file_binding": solver_binding["passed"],
        "result_file_binding": result_binding["passed"],
        "csr": csr.get("passed") is True,
        "custom_matching": custom_replay.get("passed") is True,
        "scipy_matching": scipy_replay.get("passed") is True,
        "cardinality": all(cardinality_checks.values()),
        "hall_when_required": hall is None or hall.get("passed") is True,
        "hall_presence_exact": (hall is not None)
        is (custom_replay["cardinality"] < 10_000),
        "hall_report_binding": hall_binding,
        "hall_identifier_binding": hall_identifier_binding,
    }
    return {
        "csr": csr,
        "custom_matching": custom_replay,
        "scipy_matching": scipy_replay,
        "cardinality_checks": cardinality_checks,
        "oracle_matching_replay": {
            "custom": custom_oracle_checks,
            "scipy": scipy_oracle_checks,
        },
        "hall": hall,
        "solver_file_binding": solver_binding,
        "result_file_binding": result_binding,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _compact_structure_replay(replay: Mapping[str, Any]) -> dict[str, Any]:
    output_events = replay.get("canonical_output_events", [])
    target_atoms = replay.get("canonical_positive_targets", [])
    return {
        "schema": replay.get("schema"),
        "protocol_sha256": replay.get("protocol_sha256"),
        "protocol_freeze_commit": replay.get("protocol_freeze_commit"),
        "passed": replay.get("passed"),
        "structural_domain_valid": replay.get("structural_domain_valid"),
        "evaluation_complete": replay.get("evaluation_complete"),
        "checks": replay.get("checks"),
        "computed_report": replay.get("computed_report"),
        "canonical_output_event_count": len(output_events),
        "canonical_output_events_sha256": sha256_bytes(
            canonical_json_bytes(output_events)
        ),
        "canonical_positive_target_count": len(target_atoms),
        "canonical_positive_targets_sha256": sha256_bytes(
            canonical_json_bytes(target_atoms)
        ),
    }


def _assignment_replay_snapshot_valid(
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
    if not (
        isinstance(stat_before, Mapping)
        and stat_before == snapshot.get("stat_after") == snapshot.get("path_stat_after")
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


def _decision_reproducibility_replay(
    *,
    oracle_dir: Path,
    oracle: Mapping[str, Any],
    solver_payloads: Mapping[str, bytes],
    result_payloads: Mapping[str, bytes],
    decision_replay: Mapping[str, Any],
    decision_export_bytes: bytes,
) -> dict[str, Any]:
    decision_report = oracle.get("decision")
    if not isinstance(decision_report, Mapping):
        raise ValueError("STDA decision reproducibility report is absent")
    inprocess_receipt = _exact_true_boolean_mapping(
        decision_report.get("second_inprocess_replay"),
        expected_names=INPROCESS_REPLAY_CHECK_NAMES,
        label="decision.second_inprocess_replay",
    )
    clean_receipt = _exact_true_boolean_mapping(
        decision_report.get("clean_subprocess_replay"),
        expected_names=CLEAN_REPLAY_CHECK_NAMES,
        label="decision.clean_subprocess_replay",
    )
    timing_report = oracle.get("timings_ns")
    timing_names = (
        "decision_solver_inprocess_1_ns",
        "decision_solver_inprocess_2_ns",
        "decision_solver_inprocess_replay_ns",
        "decision_solver_subprocess_3_ns",
        "decision_solver_subprocess_replay_ns",
    )
    timing_checks = {
        name: isinstance(timing_report, Mapping)
        and type(timing_report.get(name)) is int
        and timing_report[name] > 0
        for name in timing_names
    }

    replay_dir = oracle_dir / "assignment_replay"
    replay_payloads = _directory_payloads(
        replay_dir, expected_names=ASSIGNMENT_REPLAY_FILENAMES
    )
    request, request_payload = _canonical_document(replay_dir / "request.json")
    result, result_payload = _canonical_document(replay_dir / "result.json")
    if replay_payloads["request.json"] != request_payload:
        raise ValueError("STDA clean replay request changed between immutable reads")
    if replay_payloads["result.json"] != result_payload:
        raise ValueError("STDA clean replay result changed between immutable reads")

    request_without_self = dict(request)
    claimed_request_payload_sha256 = request_without_self.pop(
        "request_payload_sha256", None
    )
    solver_hashes = {
        name: sha256_bytes(payload) for name, payload in sorted(solver_payloads.items())
    }
    replay_source_hashes = {
        "code/scripts/stda_f0_assignment_replay.py": sha256_file(
            CODE_ROOT / "scripts/stda_f0_assignment_replay.py"
        ),
        "code/eval/stda_f0_round.py": sha256_file(
            CODE_ROOT / "eval/stda_f0_round.py"
        ),
    }
    replay_script_sha256 = replay_source_hashes[
        "code/scripts/stda_f0_assignment_replay.py"
    ]
    expected_project_module_paths = {
        "eval.stda_f0_round": str(
            (CODE_ROOT / "eval/stda_f0_round.py").resolve(strict=True)
        )
    }
    support_count = int(
        _npy(
            solver_payloads["support_stable_candidate_id.npy"],
            dtype="<i8",
            ndim=1,
            label="support_stable_candidate_id.npy",
        ).size
    )
    decision_support_rank = _npy(
        result_payloads["decision_support_rank.npy"],
        dtype="<i8",
        ndim=1,
        label="decision_support_rank.npy",
    )
    expected_assignment_bytes = decision_support_rank.tobytes(order="C")
    assignment_bytes = replay_payloads["assignment.bin"]
    export_bytes = replay_payloads["export.bin"]
    computed_hashes = decision_replay.get("computed_hashes")
    if not isinstance(computed_hashes, Mapping):
        raise ValueError("STDA independent decision hashes are absent")
    round_binding = oracle.get("round_input_binding")
    if not isinstance(round_binding, Mapping):
        raise ValueError("STDA round binding is absent during clean replay")
    result_verification = _exact_true_boolean_mapping(
        result.get("verification"),
        expected_names=ASSIGNMENT_REPLAY_VERIFICATION_NAMES,
        label="assignment_replay.result.verification",
    )
    isolation = result.get("interpreter_isolation")
    isolation_checks = {
        "mapping": isinstance(isolation, Mapping),
        "isolated": isinstance(isolation, Mapping)
        and isolation.get("isolated") is True,
        "no_site": isinstance(isolation, Mapping)
        and isolation.get("no_site") is True,
        "no_user_site": isinstance(isolation, Mapping)
        and isolation.get("no_user_site") is True,
        "sitecustomize_absent": isinstance(isolation, Mapping)
        and isolation.get("sitecustomize_loaded") is False,
        "trusted_site_paths_recorded": isinstance(isolation, Mapping)
        and isinstance(isolation.get("trusted_site_package_paths"), list)
        and bool(isolation.get("trusted_site_package_paths")),
    }
    input_snapshots = result.get("input_file_snapshots")
    input_snapshot_checks = isinstance(input_snapshots, Mapping) and set(
        input_snapshots
    ) == set(solver_hashes)
    if input_snapshot_checks:
        input_snapshot_checks = all(
            _assignment_replay_snapshot_valid(
                input_snapshots[name],
                expected_path=f"solver_inputs/{name}",
                expected_sha256=solver_hashes[name],
            )
            for name in solver_hashes
        )
    request_snapshot_check = _assignment_replay_snapshot_valid(
        result.get("request_snapshot"),
        expected_path="assignment_replay/request.json",
        expected_sha256=sha256_bytes(request_payload),
    )
    request_checks = {
        "schema": request.get("schema") == "stda_f0_assignment_replay_request_v1",
        "protocol_sha256": request.get("protocol_sha256") == PROTOCOL_SHA256,
        "protocol_freeze_commit": request.get("protocol_freeze_commit")
        == PROTOCOL_FREEZE_COMMIT,
        "support_cardinality": request.get("support_cardinality") == support_count,
        "input_files_sha256": request.get("input_files_sha256") == solver_hashes,
        "replay_script_sha256": request.get("replay_script_sha256")
        == replay_script_sha256,
        "request_payload_sha256": claimed_request_payload_sha256
        == sha256_bytes(canonical_json_bytes(request_without_self)),
    }
    result_checks = {
        "schema": result.get("schema") == "stda_f0_assignment_replay_result_v1",
        "protocol_sha256": result.get("protocol_sha256") == PROTOCOL_SHA256,
        "protocol_freeze_commit": result.get("protocol_freeze_commit")
        == PROTOCOL_FREEZE_COMMIT,
        "request_file_sha256": result.get("request_file_sha256")
        == sha256_bytes(request_payload),
        "request_payload_sha256": result.get("request_payload_sha256")
        == claimed_request_payload_sha256,
        "replay_script_sha256": result.get("replay_script_sha256")
        == replay_script_sha256,
        "runtime_source_sha256": result.get("runtime_source_sha256")
        == replay_source_hashes,
        "runtime_source_hashes_stable": result.get("runtime_source_hashes_stable")
        is True,
        "project_module_paths": result.get("project_module_paths")
        == expected_project_module_paths,
        "project_module_paths_stable": result.get("project_module_paths_stable")
        is True,
        "cuda_visible_devices": result.get("cuda_visible_devices") == "",
        "forbidden_cuda_modules": result.get("forbidden_cuda_modules_loaded") == [],
        "input_files_sha256": result.get("input_files_sha256") == solver_hashes,
        "input_file_snapshots": input_snapshot_checks,
        "request_snapshot": request_snapshot_check,
        "support_cardinality": result.get("support_cardinality") == support_count,
        "support_digest_sha256": result.get("support_digest_sha256")
        == round_binding.get("round_support_sha256"),
        "graph_digest_sha256": result.get("graph_digest_sha256")
        == round_binding.get("round_graph_sha256"),
        "objective": result.get("objective") == decision_replay.get("objective"),
        "assignment_sha256": result.get("assignment_sha256")
        == computed_hashes.get("assignment_sha256"),
        "selected_id_sha256": result.get("selected_id_sha256")
        == computed_hashes.get("selected_id_sha256"),
        "export_sha256": result.get("export_sha256")
        == computed_hashes.get("export_sha256"),
        "objective_sha256": result.get("objective_sha256")
        == computed_hashes.get("objective_sha256"),
        "result_digest_sha256": result.get("result_digest_sha256")
        == computed_hashes.get("digest_sha256"),
        "assignment_bytes": assignment_bytes == expected_assignment_bytes,
        "assignment_raw_sha256": result.get("assignment_raw_sha256")
        == sha256_bytes(expected_assignment_bytes),
        "assignment_size": result.get("assignment_bytes")
        == len(expected_assignment_bytes),
        "assignment_count": result.get("assignment_count")
        == int(decision_support_rank.size),
        "assignment_dtype": result.get("assignment_dtype") == "<i8",
        "export_bytes": export_bytes == decision_export_bytes,
        "export_raw_sha256": result.get("export_raw_sha256")
        == sha256_bytes(decision_export_bytes),
        "export_size": result.get("export_bytes") == len(decision_export_bytes),
        "export_count": result.get("export_count") == len(decision_export_bytes) // 12,
        "export_dtype": result.get("export_dtype") == "<f4",
        "internal_verification": result_verification["passed"],
        "interpreter_isolation": all(isolation_checks.values()),
    }
    payload_record_checks = {
        name: _payload_record_replay(
            oracle,
            relative_path=f"assignment_replay/{name}",
            payload=replay_payloads[name],
        )["passed"]
        for name in sorted(ASSIGNMENT_REPLAY_FILENAMES)
    }
    checks = {
        "three_solver_runs_identical": decision_report.get(
            "three_solver_runs_identical"
        )
        is True,
        "second_inprocess_receipt": inprocess_receipt["passed"],
        "clean_subprocess_receipt": clean_receipt["passed"],
        "all_solver_timings_positive": all(timing_checks.values()),
        "request": all(request_checks.values()),
        "result": all(result_checks.values()),
        "payload_records": all(payload_record_checks.values()),
    }
    return {
        "second_inprocess_receipt": inprocess_receipt,
        "clean_subprocess_receipt": clean_receipt,
        "solver_timing_checks": timing_checks,
        "request_checks": request_checks,
        "result_checks": result_checks,
        "result_verification": result_verification,
        "interpreter_isolation_checks": isolation_checks,
        "payload_record_checks": payload_record_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _target_class_applicability(
    target_atoms: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    by_ray: dict[int, list[Mapping[str, Any]]] = {}
    for atom in target_atoms:
        ray_id = int(atom["ray_id"])
        if ray_id < 0:
            continue
        by_ray.setdefault(ray_id, []).append(atom)
    applicable = {
        f"{range_label}_{return_label}": False
        for range_label in RANGE_LABELS
        for return_label in ("first", "later")
    }
    for atoms in by_ray.values():
        ordered = sorted(
            atoms,
            key=lambda atom: (
                float(atom["range_m"]),
                int(atom["canonical_target_id"]),
            ),
        )
        start = 0
        group_index = 0
        while start < len(ordered):
            stop = start + 1
            first_range = float(ordered[start]["range_m"])
            while (
                stop < len(ordered)
                and float(ordered[stop]["range_m"]) - first_range < 0.05
            ):
                stop += 1
            members = ordered[start:stop]
            weight = math.fsum(float(atom["weight"]) for atom in members)
            representative = math.fsum(
                float(atom["range_m"]) * float(atom["weight"])
                for atom in members
            ) / weight
            if 0.0 <= representative < 30.0:
                range_index = 0
            elif representative < 60.0:
                range_index = 1
            elif representative < 120.0:
                range_index = 2
            else:
                raise ValueError("STDA target group escaped [0,120) during replay")
            return_label = "first" if group_index == 0 else "later"
            applicable[f"{RANGE_LABELS[range_index]}_{return_label}"] = True
            start = stop
            group_index += 1
    return applicable


def _full_arm_replay(
    oracle_dir: Path,
    oracle: Mapping[str, Any],
    *,
    solver: Mapping[str, bytes],
    controls: Mapping[str, bytes],
    target_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = oracle.get("frame")
    if not isinstance(frame, Mapping):
        raise ValueError("STDA oracle frame identity is absent")
    results = _directory_payloads(
        oracle_dir / "round_results", expected_names=READY_RESULT_FILENAMES
    )
    exports = _directory_payloads(
        oracle_dir / "exports",
        expected_names=frozenset(
            f"{arm}.{suffix}"
            for arm in ORACLE_ARM_NAMES
            for suffix in ("bin", "npy")
        ),
    )
    control_replay = verify_control_export_replay(
        sequence=int(frame["sequence"]),
        radar_index=int(frame["radar_index"]),
        solver_inputs=solver,
        control_inputs=controls,
        round_results=results,
        exports=exports,
        oracle_report=oracle,
    )
    decision_reproducibility = _decision_reproducibility_replay(
        oracle_dir=oracle_dir,
        oracle=oracle,
        solver_payloads=solver,
        result_payloads=results,
        decision_replay=control_replay["decision"],
        decision_export_bytes=exports["decision.bin"],
    )
    control_replay = dict(control_replay)
    control_replay["decision_reproducibility"] = decision_reproducibility
    control_replay["passed"] = bool(
        control_replay.get("passed") is True
        and decision_reproducibility.get("passed") is True
    )
    target = oracle.get("target")
    if not isinstance(target, Mapping) or target.get("target_tensor_sha256") != sha256_bytes(
        target_bytes
    ):
        raise ValueError("STDA target snapshot binding changed before structural replay")
    domain = oracle.get("structural_domain")
    if not isinstance(domain, Mapping):
        raise ValueError("STDA structural domain evidence is absent")
    azimuth = domain.get("azimuth_edges_rad")
    elevation = domain.get("elevation_edges_rad")
    if not isinstance(azimuth, list) or not isinstance(elevation, list):
        raise ValueError("STDA structural edge arrays are absent")
    arms: dict[str, Any] = {}
    target_class_applicability: dict[str, bool] | None = None
    canonical_target_sha256: str | None = None
    for oracle_name in ORACLE_ARM_NAMES:
        structure, structure_payload = _canonical_document(
            oracle_dir / "structure" / f"{oracle_name}.json"
        )
        expected_structure = oracle.get("structure", {}).get(oracle_name, {}).get(
            "report", {}
        )
        if (
            expected_structure.get("sha256") != sha256_bytes(structure_payload)
            or expected_structure.get("bytes") != len(structure_payload)
        ):
            raise ValueError(f"STDA structural report binding failed: {oracle_name}")
        replay = verify_structural_replay(
            exports[f"{oracle_name}.bin"],
            target_bytes,
            azimuth,
            elevation,
            structure,
        )
        replay_targets = replay.get("canonical_positive_targets")
        if not isinstance(replay_targets, list):
            raise ValueError("STDA structural replay omitted canonical targets")
        replay_target_sha = sha256_bytes(canonical_json_bytes(replay_targets))
        replay_applicability = _target_class_applicability(replay_targets)
        if target_class_applicability is None:
            target_class_applicability = replay_applicability
            canonical_target_sha256 = replay_target_sha
        elif (
            replay_applicability != target_class_applicability
            or replay_target_sha != canonical_target_sha256
        ):
            raise ValueError("STDA target structural applicability changed across arms")
        compact = _compact_structure_replay(replay)
        export_replay = control_replay["exports"][oracle_name]
        alias = ARM_ALIASES[oracle_name]
        arms[alias] = {
            "oracle_arm": oracle_name,
            "export_replay": export_replay,
            "spacing": export_replay["spacing"],
            "structural_replay": compact,
            "target_class_applicability": replay_applicability,
            "passed": export_replay.get("passed") is True
            and compact.get("passed") is True,
        }
    return control_replay, arms


def run(
    arguments: argparse.Namespace,
    *,
    formal_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_started_ns = time.perf_counter_ns()
    timings: dict[str, int] = {}
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("STDA independent verifier requires CUDA_VISIBLE_DEVICES empty")
    forbidden = sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in FORBIDDEN_MODULE_PREFIXES
    )
    if forbidden:
        raise RuntimeError(f"STDA independent verifier loaded forbidden modules: {forbidden}")
    support_frame_dir = _resolve_input_path(
        arguments.support_frame_dir, directory=True, label="support-frame"
    )
    oracle_dir = _resolve_input_path(
        arguments.oracle_dir, directory=True, label="oracle"
    )
    target_cache = _resolve_input_path(
        arguments.target_cache, directory=False, label="target-cache"
    )
    if (
        _path_within(support_frame_dir, oracle_dir)
        or _path_within(oracle_dir, support_frame_dir)
        or _path_within(target_cache, support_frame_dir)
        or _path_within(target_cache, oracle_dir)
    ):
        raise ValueError("STDA verifier immutable inputs overlap")
    output = _resolve_output_path(
        arguments.output,
        forbidden_roots=(support_frame_dir, oracle_dir),
        forbidden_files=(target_cache,),
    )
    arguments.output = output
    source_before = {
        "code/eval/stda_f0_verify.py": sha256_file(
            CODE_ROOT / "eval/stda_f0_verify.py"
        ),
        "code/scripts/stda_f0_verify_phase.py": sha256_file(Path(__file__)),
    }
    interval = time.perf_counter_ns()
    oracle, complete = _load_oracle(oracle_dir)
    timings["oracle_completion_and_report_load_ns"] = time.perf_counter_ns() - interval
    status = oracle.get("status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"STDA oracle status is not independently replayable: {status}")
    target_conditioned_directories = (
        "solver_inputs",
        "control_inputs",
        "fit_evidence",
        "round_results",
        "assignment_replay",
        "exports",
        "structure",
    )
    support_no_go_stage_absence = status != SUPPORT_NO_GO_STATUS or all(
        not (oracle_dir / name).exists() and not (oracle_dir / name).is_symlink()
        for name in target_conditioned_directories
    )
    interval = time.perf_counter_ns()
    support_payload = _regular_bytes(support_frame_dir / "support.bin")
    support = _support_replay(
        support_frame_dir,
        expected_support_sha256=arguments.expected_support_sha256,
        oracle=oracle,
        payload=support_payload,
    )
    timings["independent_support_replay_ns"] = time.perf_counter_ns() - interval

    interval = time.perf_counter_ns()
    target_provenance, _, target_array_bytes = _target_cache_provenance(
        target_cache,
        oracle=oracle,
        parse_array=status in (GRAPH_NO_GO_STATUS, READY_STATUS),
    )
    timings["authoritative_target_cache_provenance_ns"] = (
        time.perf_counter_ns() - interval
    )

    semantic: dict[str, Any] | None = None
    solver_payloads: dict[str, bytes] = {}
    control_payloads: dict[str, bytes] = {}
    graph: dict[str, Any] | None = None
    controls: dict[str, Any] | None = None
    arms: dict[str, Any] = {}
    if status in (GRAPH_NO_GO_STATUS, READY_STATUS):
        if target_array_bytes is None:
            raise AssertionError("STDA target cache parser omitted required array bytes")
        interval = time.perf_counter_ns()
        semantic, solver_payloads, control_payloads = _fit_semantic_replay(
            support_payload=support_payload,
            target_array_bytes=target_array_bytes,
            target_cache_sha256=target_provenance["cache_sha256"],
            oracle_dir=oracle_dir,
            oracle=oracle,
        )
        timings["independent_fit_semantic_reconstruction_ns"] = (
            time.perf_counter_ns() - interval
        )
        interval = time.perf_counter_ns()
        graph = _graph_replay(
            oracle_dir,
            oracle,
            solver_payloads=solver_payloads,
        )
        timings["independent_graph_certificate_replay_ns"] = (
            time.perf_counter_ns() - interval
        )
    if status == READY_STATUS:
        if target_array_bytes is None:
            raise AssertionError("STDA ready replay lacks authoritative target bytes")
        interval = time.perf_counter_ns()
        controls, arms = _full_arm_replay(
            oracle_dir,
            oracle,
            solver=solver_payloads,
            controls=control_payloads,
            target_bytes=target_array_bytes,
        )
        timings["independent_controls_exports_and_structure_replay_ns"] = (
            time.perf_counter_ns() - interval
        )
    status_checks = {
        "support_no_go_has_no_graph": status != SUPPORT_NO_GO_STATUS
        or graph is None,
        "support_no_go_has_no_target_conditioned_artifacts": support_no_go_stage_absence,
        "support_no_go_has_insufficient_capacity": status != SUPPORT_NO_GO_STATUS
        or int(support["replay"]["support_count"]) < 10_000,
        "graph_or_ready_has_support_capacity": status == SUPPORT_NO_GO_STATUS
        or int(support["replay"]["support_count"]) >= 10_000,
        "graph_no_go_has_verified_deficit": status != GRAPH_NO_GO_STATUS
        or (
            graph is not None
            and graph["custom_matching"]["cardinality"] < 10_000
            and graph["hall"] is not None
        ),
        "ready_has_full_graph": status != READY_STATUS
        or (
            graph is not None
            and graph["custom_matching"]["cardinality"] == 10_000
        ),
        "ready_has_three_arms": status != READY_STATUS or set(arms) == {
            "decision",
            "pointwise",
            "greedy",
        },
        "support_no_go_did_not_parse_target_array": status != SUPPORT_NO_GO_STATUS
        or target_provenance["array"].get("parsed") is False,
        "graph_and_ready_rebuilt_fit_semantics": status == SUPPORT_NO_GO_STATUS
        or (semantic is not None and semantic.get("passed") is True),
    }
    timings["independent_replay_core_ns"] = time.perf_counter_ns() - run_started_ns
    source_after = {
        "code/eval/stda_f0_verify.py": sha256_file(
            CODE_ROOT / "eval/stda_f0_verify.py"
        ),
        "code/scripts/stda_f0_verify_phase.py": sha256_file(Path(__file__)),
    }
    checks = {
        "protocol_sha256": oracle.get("protocol_sha256") == PROTOCOL_SHA256,
        "protocol_freeze_commit": oracle.get("protocol_freeze_commit")
        == PROTOCOL_FREEZE_COMMIT,
        "oracle_completion_status": complete.get("status") == status,
        "support": support["passed"],
        "authoritative_target_cache_provenance": target_provenance["passed"],
        "fit_semantics_when_present": semantic is None or semantic["passed"],
        "graph_when_present": graph is None or graph["passed"],
        "controls_when_present": controls is None or controls["passed"],
        "arms_when_present": not arms or all(arm["passed"] for arm in arms.values()),
        "status_semantics": all(status_checks.values()),
        "boundary_semantics_frozen": BOUNDARY_SEMANTICS
        == {
            "support_target_input": False,
            "demand_target_conditioned": True,
            "graph_target_conditioned": True,
            "cost_target_conditioned": True,
            "control_score_target_conditioned": True,
            "assignment_target_conditioned": True,
            "solver_raw_target": False,
            "deployment_claim": False,
        },
        "source_hashes_stable": source_before == source_after,
        "formal_runtime_when_provided": formal_runtime is None
        or formal_runtime.get("passed") is True,
        "cpu_only": True,
        "forbidden_modules_absent": not forbidden,
    }
    return {
        "schema": SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "boundary_semantics": dict(BOUNDARY_SEMANTICS),
        "formal_runtime": dict(formal_runtime) if formal_runtime is not None else None,
        "timing_authority": {
            "parent_process_wall_authoritative": True,
            "internal_process_wall_emitted_after_output_rehash": True,
            "module_wall_started_perf_counter_ns": MODULE_WALL_STARTED_NS,
        },
        "oracle_status": status,
        "frame": oracle.get("frame"),
        "support": support,
        "target_cache_provenance": target_provenance,
        "fit_semantic_replay": semantic,
        "graph": graph,
        "controls": controls,
        "arms": arms,
        "status_checks": status_checks,
        "timings_ns": timings,
        "checks": checks,
        "source_sha256": source_after,
        "passed": all(checks.values()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-frame-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--expected-support-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    formal_runtime = _formal_runtime_report()
    arguments = parse_args(argv)
    report = run(arguments, formal_runtime=formal_runtime)
    output_sha256, output_size = _exclusive_json(arguments.output, report)
    output_rehash_completed_ns = time.perf_counter_ns()
    summary = {
        "schema": SCHEMA,
        "output": str(arguments.output.absolute()),
        "output_sha256": output_sha256,
        "output_size_bytes": output_size,
        "parent_process_wall_authoritative": True,
        "internal_full_wall_through_output_rehash_ns": (
            output_rehash_completed_ns - MODULE_WALL_STARTED_NS
        ),
        "internal_full_wall_boundary": (
            "verifier_module_start_through_output_fsync_and_independent_rehash"
        ),
        "passed": report["passed"],
    }
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    sys.stdout.buffer.flush()
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
