#!/usr/bin/env python3
"""Orchestrate the frozen all-76 STDA-F0 transaction on one allowed H200."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence


PROTOCOL = "stda_f0_sparse_target_demand_assignment"
PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
PROTOCOL_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
FREEZE_RECORD_SHA256 = (
    "af0134038cb501b03fe1dd25937c46f9bdfacbca08fa5c6068afbc3dcfe42c7d"
)
FREEZE_AUDIT_SHA256 = (
    "5b27abd63a114c1f107dd22b65590b39a5425c60dc72809d3bc6ce8027ade7d5"
)
EXPECTED_INPUT_HASHES = {
    "manifest": "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4",
    "scene_split": "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc",
    "normalization": "4d0bca7d027a1a9f457c526f21a034a406ce4973e2dccee55ddd2d7019b41b77",
    "checkpoint": "c0ebb4c1509c3d36b6834bd39977cb1cda2fcaff1747eabc97ebb41af0f06ce4",
    "formal_metrics": "d9c0d6c41b74392b123b11e770bc122aba2ec30a835b45c157a21ad96fd14d7a",
    "formal_run_manifest": "26ebab679d112e07c5bb364b8799cffc24b573c4abc6a71623705a4cfe962e9f",
    "candidate_hash_manifest": "1d650272a021c36ee2d92219ebec3b1d91508aefdb5242c0e6fcd323d6bed2bf",
    "dense_geometry": "e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68",
    "vrh_structure": "3450f83aa7c4add4198f008995b4928e759391a5d74e8a72b0eedf6da2185a42",
    "qlocal_f0r_report": "505644667ba3f73b58ee138608307e82632fe92bc3336f538c74767ab9c148ae",
    "qlocal_f0r_protocol": "86683108d8b012a04e74b57cb67f5d23f197fce1a01448a480bea070701f45bd",
    "vrh_report": "944b3e338823498ae61c545b067bf92dbd0d3600f1aef6aa099eb7dc498c1eed",
    "vrh_protocol": "78c62f29e8f325287d79eefea141bf2b014f1388448d4ed1a3b5365071cd7946",
    "info_arr": "53f72b22544aa11bc0057f9b8c2177a7a844fddd0e8ce3f753a989d07159767a",
    "arr_doppler": "f81e56889c2cedc98eb3eb8a4828e382845e3d4758a36f3b8fc0fce4839e0493",
}
EXPECTED_TRAIN_FRAMES = 76
EXPECTED_VALIDATION_FRAMES = 24
MAX_HOST_BYTES = 60 * 1024**3
MAX_CUDA_BYTES = 60 * 1024**3
MAX_ALLOCATION_NS = 30_000_000_000
MAX_FRAME_NS = 120_000_000_000
MAX_TRANSACTION_NS = 7_200_000_000_000
MONITOR_SAMPLE_INTERVAL_NS = 2_000_000
MAX_MONITOR_SAMPLE_INTERVAL_NS = 50_000_000
TRANSACTION_DOMAIN = b"stda_f0_transaction_v1\0"
FAILURE_DOMAIN = b"stda_f0_failure_v1\0"
BUNDLE_DOMAIN = b"stda_f0_bundle_v2\0"
FORMAL_LAUNCH_TOKEN = "STDA-F0-ALL-76-A1BACD61"
AT_FDCWD = -100
RENAME_NOREPLACE = 1
_EVENT_LOG_PATH: Path | None = None
TERMINAL_STATUSES = (
    "stda_f0_implementation_invalid",
    "stda_f0_resource_invalid",
    "stda_f0_packed_support_capacity_no_go",
    "stda_f0_graph_cardinality_no_go",
    "stda_f0_assignment_recipe_no_go",
    "stda_f0_packed_support_only",
    "stda_f0_demand_allocation_only",
    "stda_f0_assignment_utility_passed",
)
RESOURCE_FAILURE_CODES = frozenset(("child_timeout", "transaction_timeout"))

SUPPORT_CHILD_SOURCE_PATHS = (
    "code/scripts/stda_f0_support_phase.py",
    "code/cube_dense/__init__.py",
    "code/cube_dense/kradar.py",
    "code/eval/stda_f0_candidate.py",
    "code/eval/stda_f0_support.py",
    "code/eval/stda_f0_verify.py",
    "code/eval/rald_wce_stage0.py",
    "code/models/rald_wce_field.py",
    "code/models/rald_matched.py",
    "code/models/cube_cycle.py",
    "code/models/cube_doppler.py",
    "code/models/cube_occupancy.py",
    "code/models/point_to_cube.py",
)
SUPPORT_ORCHESTRATOR_SOURCE_PATHS = (
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
SUPPORT_RUNTIME_SOURCE_PATHS = tuple(
    dict.fromkeys((*SUPPORT_CHILD_SOURCE_PATHS, *SUPPORT_ORCHESTRATOR_SOURCE_PATHS))
)
ORACLE_CHILD_SOURCE_PATHS = (
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
ORACLE_RUNTIME_SOURCE_PATHS = tuple(
    dict.fromkeys((*ORACLE_CHILD_SOURCE_PATHS, *SUPPORT_ORCHESTRATOR_SOURCE_PATHS))
)
VERIFY_CHILD_SOURCE_PATHS = (
    "code/eval/stda_f0_verify.py",
    "code/scripts/stda_f0_verify_phase.py",
)
MANDATORY_CPU_TEST_PATHS = (
    "code/tests/test_stda_f0_support.py",
    "code/tests/test_stda_f0_fit.py",
    "code/tests/test_stda_f0_round.py",
    "code/tests/test_stda_f0_structure.py",
    "code/tests/test_stda_f0_verify.py",
    "code/tests/test_stda_f0_oracle_phase.py",
    "code/tests/test_stda_f0_verify_phase.py",
    "code/tests/test_stda_f0_support_phase.py",
    "code/tests/test_stda_f0_capacity.py",
)
MANDATORY_CANDIDATE_TEST_PATHS = ("code/tests/test_stda_f0_candidate.py",)
MANDATORY_METRIC_TEST_PATHS = ("code/tests/test_stda_f0_metric_phase.py",)
MANDATORY_TEST_PATHS = tuple(
    dict.fromkeys(
        MANDATORY_CPU_TEST_PATHS
        + MANDATORY_CANDIDATE_TEST_PATHS
        + MANDATORY_METRIC_TEST_PATHS
    )
)


class StageFailure(RuntimeError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(message)


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


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != payload:
            raise IOError(f"atomic write replay changed bytes: {path}")
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, document: Any) -> bytes:
    payload = canonical_json_bytes(document)
    atomic_write_bytes(path, payload)
    return payload


def _rename_noreplace(staging: Path, destination: Path) -> None:
    """Atomically rename within one directory and fail if destination exists."""

    staging = Path(staging)
    destination = Path(destination)
    if staging.parent.resolve() != destination.parent.resolve():
        raise OSError(errno.EXDEV, "renameat2 publication must stay in one directory")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "libc renameat2 is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(staging),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def exclusive_write_bytes(path: Path, payload: bytes) -> None:
    """Publish immutable bytes through one fsynced renameat2 no-replace commit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    committed = False
    expected_sha256 = sha256_bytes(payload)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
        replay = temporary.read_bytes()
        if len(replay) != len(payload) or sha256_bytes(replay) != expected_sha256:
            raise IOError(f"exclusive staging replay changed bytes: {path}")
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
        _rename_noreplace(temporary, path)
        committed = True
        fsync_directory(path.parent)
        published = path.read_bytes()
        if (
            len(published) != len(payload)
            or sha256_bytes(published) != expected_sha256
        ):
            raise IOError(f"exclusive write replay changed bytes: {path}")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if not committed:
            temporary.unlink(missing_ok=True)


def canonical_json_file(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageFailure("serialization", "json_decode", str(path)) from error
    if not isinstance(document, dict) or canonical_json_bytes(document) != payload:
        raise StageFailure("serialization", "json_noncanonical", str(path))
    return document


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _valid_git_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def verify_source_tree(repo: Path, source_commit: str) -> dict[str, Any]:
    if not _valid_git_sha(source_commit):
        raise StageFailure("preflight", "source_sha_invalid", source_commit)
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise StageFailure("preflight", "source_head_mismatch", source_commit)
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise StageFailure("preflight", "source_dirty", dirty)
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", PROTOCOL_FREEZE_COMMIT, source_commit),
        cwd=repo,
        check=False,
    ).returncode == 0
    if not ancestor or source_commit == PROTOCOL_FREEZE_COMMIT:
        raise StageFailure("preflight", "freeze_ancestry_invalid", source_commit)
    return {
        "source_commit": source_commit,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "freeze_is_strict_ancestor": True,
        "worktree_clean": True,
    }


def _require_hash(name: str, path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageFailure("preflight", f"missing_{name}", str(path))
    observed = sha256_file(path)
    if observed != expected:
        raise StageFailure(
            "preflight",
            f"hash_{name}",
            f"{path}: expected {expected}, observed {observed}",
        )
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": observed}


def _load_canonical_train_records(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise StageFailure("preflight", "manifest_frames", str(manifest))
    train = [row for row in frames if row.get("partition") == "train"]
    validation = [row for row in frames if row.get("partition") == "validation"]
    if len(train) != EXPECTED_TRAIN_FRAMES or len(validation) != EXPECTED_VALIDATION_FRAMES:
        raise StageFailure("preflight", "manifest_cohort", f"{len(train)}/{len(validation)}")
    identities: set[tuple[int, int]] = set()
    canonical: list[dict[str, Any]] = []
    for row in train:
        identity = (int(row["sequence"]), int(row["radar_index"]))
        if identity in identities:
            raise StageFailure("preflight", "duplicate_train_frame", str(identity))
        identities.add(identity)
        lowered = [str(key).lower() for key in row]
        if any(fragment in key for key in lowered for fragment in ("test", "future")):
            raise StageFailure("preflight", "forbidden_manifest_field", str(identity))
        canonical.append(
            {
                "position": len(canonical),
                "sequence": identity[0],
                "radar_index": identity[1],
                "partition": "train",
            }
        )
    return canonical, {
        "train": len(train),
        "validation": len(validation),
        "test": 0,
        "future": 0,
    }


def verify_frozen_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repo = args.repo.resolve()
    paths = {
        "manifest": args.manifest,
        "scene_split": args.scene_split,
        "normalization": args.normalization,
        "checkpoint": args.checkpoint,
        "formal_metrics": args.formal_metrics,
        "formal_run_manifest": args.formal_run_manifest,
        "candidate_hash_manifest": args.candidate_hash_manifest,
        "dense_geometry": repo / "code/eval/dense_geometry.py",
        "vrh_structure": repo / "code/eval/vrh_f0_metrics.py",
        "qlocal_f0r_report": repo / "artifacts/g1/qlocal_f0r_global_export_34579a2/preflight.json",
        "qlocal_f0r_protocol": repo / "docs/qlocal_f0r_global_export_capacity_protocol.md",
        "vrh_report": repo / "artifacts/g1/vrh_f0_capacity_380f3ea/preflight_compact.json",
        "vrh_protocol": repo / "docs/vrh_f0_variable_return_capacity_protocol.md",
        "info_arr": args.data_root / "resources/info_arr.mat",
        "arr_doppler": args.data_root / "resources/arr_doppler.mat",
    }
    evidence = {
        name: _require_hash(name, Path(path), EXPECTED_INPUT_HASHES[name])
        for name, path in paths.items()
    }
    evidence["protocol"] = _require_hash(
        "protocol",
        repo / "docs/stda_f0_sparse_target_demand_assignment_protocol.md",
        PROTOCOL_SHA256,
    )
    evidence["freeze_record"] = _require_hash(
        "freeze_record",
        repo / "artifacts/idea/stda_f0_freeze_record.json",
        FREEZE_RECORD_SHA256,
    )
    evidence["freeze_audit"] = _require_hash(
        "freeze_audit",
        repo / "artifacts/idea/stda_f0_prefreeze_audit_round6.md",
        FREEZE_AUDIT_SHA256,
    )
    freeze_record = json.loads(
        (repo / "artifacts/idea/stda_f0_freeze_record.json").read_text(encoding="ascii")
    )
    if (
        freeze_record.get("protocol_sha256") != PROTOCOL_SHA256
        or freeze_record.get("audit_artifact_sha256") != FREEZE_AUDIT_SHA256
        or freeze_record.get("verdict") != "FREEZE"
    ):
        raise StageFailure("preflight", "freeze_record_semantics", str(freeze_record))
    records, counts = _load_canonical_train_records(args.manifest)
    candidate_manifest = json.loads(args.candidate_hash_manifest.read_text(encoding="utf-8"))
    expected_candidates = candidate_manifest.get("frames")
    if not isinstance(expected_candidates, list) or len(expected_candidates) != 12:
        raise StageFailure("preflight", "candidate_manifest_frames", "expected 12")
    for record in records:
        cube = (
            args.data_root
            / str(record["sequence"])
            / "radar_tesseract"
            / f"tesseract_{record['radar_index']:05d}.mat"
        )
        if not cube.is_file():
            raise StageFailure("preflight", "raw_cube_missing", str(cube))
    evidence["cohort"] = counts
    evidence["candidate_expected_frame_count"] = len(expected_candidates)
    return evidence, records


def parse_gpu_inventory() -> dict[int, dict[str, Any]]:
    output = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        stderr=subprocess.STDOUT,
    )
    rows: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) != 6:
            raise StageFailure("preflight", "gpu_inventory_parse", line)
        index = int(columns[0])
        rows[index] = {
            "physical_index": index,
            "uuid": columns[1],
            "pci_bus_id": columns[2],
            "name": columns[3],
            "memory_total_mib": int(columns[4]),
            "memory_used_mib_preflight": int(columns[5]),
        }
    return rows


def cuda_driver_version(physical_index: int) -> str:
    output = subprocess.check_output(
        (
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise StageFailure("preflight", "cuda_driver_parse", output)
    return rows[0]


def require_allowed_h200(physical_index: int) -> dict[str, Any]:
    if physical_index not in (0, 2):
        raise StageFailure("preflight", "gpu_index_forbidden", str(physical_index))
    inventory = parse_gpu_inventory()
    if physical_index not in inventory:
        raise StageFailure("preflight", "gpu_index_absent", str(physical_index))
    selected = inventory[physical_index]
    if selected["name"] != "NVIDIA H200 NVL":
        raise StageFailure("preflight", "gpu_not_h200", str(selected))
    if physical_index == 1 or "RTX" in selected["name"].upper():
        raise StageFailure("preflight", "rtx_forbidden", str(selected))
    return {
        **selected,
        "driver_version": cuda_driver_version(physical_index),
        "inventory": inventory,
    }


def child_environment(
    *,
    repo: Path,
    physical_gpu: int | str | None,
    phase: str,
) -> dict[str, str]:
    python_path = Path(sys.executable).resolve()
    python_bin = str(python_path.parent)
    conda_prefix = str(python_path.parent.parent)
    base_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    path_parts = [python_bin] + [part for part in base_path.split(":") if part and part != python_bin]
    environment = {
        "PATH": ":".join(path_parts),
        "HOME": str(Path.home()),
        "USER": pwd.getpwuid(os.getuid()).pw_name,
        "LOGNAME": pwd.getpwuid(os.getuid()).pw_name,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(repo / "code"),
        "PYTHONNOUSERSITE": "1",
        "CONDA_PREFIX": conda_prefix,
        "CONDA_DEFAULT_ENV": Path(conda_prefix).name,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "" if physical_gpu is None else str(physical_gpu),
        "STDA_F0_PHASE": phase,
    }
    for key in ("LD_LIBRARY_PATH", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    return environment


def interpreter_isolation_report() -> dict[str, Any]:
    checks = {
        "isolated_flag": bool(sys.flags.isolated),
        "no_site_flag": bool(sys.flags.no_site),
    }
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "sys_flags_isolated": int(sys.flags.isolated),
        "sys_flags_no_site": int(sys.flags.no_site),
        "checks": checks,
        "passed": all(checks.values()),
    }


def require_orchestrator_isolation() -> dict[str, Any]:
    report = interpreter_isolation_report()
    if report["passed"] is not True:
        raise StageFailure(
            "preflight",
            "orchestrator_not_isolated_no_site",
            str(report["checks"]),
        )
    return report


def _explicit_site_packages() -> Path:
    path = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    ).resolve(strict=True)
    if not path.is_dir():
        raise StageFailure("preflight", "site_packages_absent", str(path))
    return path


def _isolated_python_bootstrap(*, repo: Path, statement: str) -> str:
    search_paths = [str(_explicit_site_packages()), str((repo / "code").resolve())]
    return (
        "import sys;"
        f"sys.path[:0]={search_paths!r};"
        f"{statement}"
    )


def process_tree(root_pid: int) -> set[int]:
    pending = [root_pid]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in observed or not Path(f"/proc/{pid}").exists():
            continue
        observed.add(pid)
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = children_path.read_text(encoding="ascii").split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            children = []
        pending.extend(int(value) for value in children)
    return observed


def process_memory(pid: int) -> tuple[int, int]:
    rss = 0
    swap = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("VmSwap:"):
                swap = int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return rss, swap


@dataclass
class PhasePeak:
    samples: int = 0
    peak_rss_bytes: int = 0
    peak_swap_bytes: int = 0
    peak_nvml_bytes: int = 0
    maximum_process_count: int = 0
    child_descendant_seen: bool = False


class NvmlBinding:
    class ProcessInfo(ctypes.Structure):
        _fields_ = (
            ("pid", ctypes.c_uint),
            ("usedGpuMemory", ctypes.c_ulonglong),
        )

    NVML_SUCCESS = 0
    NVML_ERROR_INSUFFICIENT_SIZE = 7
    NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1

    def __init__(self, gpu_uuid: str) -> None:
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlShutdown.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByUUID.argtypes = (
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self.library.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
        self.handle = ctypes.c_void_p()
        self._require(self.library.nvmlInit_v2(), "nvmlInit_v2")
        self._require(
            self.library.nvmlDeviceGetHandleByUUID(
                gpu_uuid.encode("ascii"),
                ctypes.byref(self.handle),
            ),
            "nvmlDeviceGetHandleByUUID",
        )
        self.function = getattr(
            self.library,
            "nvmlDeviceGetComputeRunningProcesses",
        )
        self.function.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(self.ProcessInfo),
        )
        self.function.restype = ctypes.c_int

    @classmethod
    def _require(cls, result: int, operation: str) -> None:
        if result != cls.NVML_SUCCESS:
            raise RuntimeError(f"{operation} returned NVML status {result}")

    def process_memory(self) -> dict[int, int]:
        observed: dict[int, int] = {}
        capacity = 64
        while True:
            array = (self.ProcessInfo * capacity)()
            count = ctypes.c_uint(capacity)
            result = self.function(self.handle, ctypes.byref(count), array)
            if result == self.NVML_ERROR_INSUFFICIENT_SIZE:
                capacity = max(capacity * 2, int(count.value))
                continue
            self._require(result, self.function.__name__)
            for index in range(int(count.value)):
                pid = int(array[index].pid)
                used = int(array[index].usedGpuMemory)
                if used != self.NVML_VALUE_NOT_AVAILABLE:
                    observed[pid] = max(observed.get(pid, 0), used)
            break
        return observed

    def close(self) -> None:
        self._require(self.library.nvmlShutdown(), "nvmlShutdown")


class ProcessTreeMonitor:
    def __init__(self, root_pid: int, gpu_uuid: str) -> None:
        self.root_pid = root_pid
        self.gpu_uuid = gpu_uuid
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active: tuple[str, int, bool] | None = None
        self._owned_child_pids: set[int] = set()
        self._phase_peaks: dict[str, PhasePeak] = {}
        self._errors: list[str] = []
        self.sample_count = 0
        self.periodic_sample_count = 0
        self.peak_rss_bytes = 0
        self.peak_swap_bytes = 0
        self.peak_nvml_bytes = 0
        self.maximum_process_count = 0
        self.maximum_interval_ns = 0
        self.maximum_sample_duration_ns = 0
        self.maximum_process_tree_sample_ns = 0
        self.maximum_nvml_query_ns = 0
        self.cuda_overlap_detected = False
        self.maximum_cuda_pid_count = 0
        self.multiple_cuda_pid_sample_count = 0
        self.unexpected_tree_cuda_pid_sample_count = 0
        self.foreign_cuda_pid_sample_count = 0
        self.observed_cuda_pids: dict[int, int] = {}
        self.foreign_cuda_pids: dict[int, int] = {}
        self.unexpected_tree_cuda_pids: dict[int, int] = {}
        self._started_ns: int | None = None
        self._last_periodic_sample_ns: int | None = None
        try:
            self._nvml = NvmlBinding(gpu_uuid)
        except Exception as error:
            raise StageFailure("preflight", "nvml_unavailable", repr(error)) from error

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("process monitor already started")
        try:
            self._nvml.process_memory()
        except Exception as error:
            raise StageFailure("preflight", "nvml_warmup_failed", repr(error)) from error
        self._started_ns = time.perf_counter_ns()
        self._thread = threading.Thread(target=self._run, name="stda-resource-monitor", daemon=True)
        self._thread.start()

    def enter_phase(self, name: str, child_pid: int, cuda_visible: bool) -> None:
        with self._lock:
            if self._active is not None:
                if cuda_visible and self._active[2]:
                    self.cuda_overlap_detected = True
                raise RuntimeError(f"overlapping STDA phases: {self._active} / {name}")
            self._active = (name, child_pid, cuda_visible)
            self._owned_child_pids.add(child_pid)
            self._phase_peaks.setdefault(name, PhasePeak())

    def leave_phase(self, name: str, child_pid: int) -> None:
        with self._lock:
            if (
                self._active is None
                or self._active[0] != name
                or self._active[1] != child_pid
            ):
                raise RuntimeError(f"STDA phase registry mismatch: {self._active}")
            self._active = None

    @staticmethod
    def _record_cuda_pids(
        destination: dict[int, int],
        pids: set[int],
        memory: Mapping[int, int],
    ) -> None:
        for pid in pids:
            destination[pid] = max(destination.get(pid, 0), int(memory[pid]))

    def _sample(self, *, periodic: bool) -> None:
        sample_started_ns = time.perf_counter_ns()
        tree = process_tree(self.root_pid)
        rss_swap = [process_memory(pid) for pid in tree]
        rss = sum(value[0] for value in rss_swap)
        swap = sum(value[1] for value in rss_swap)
        tree_completed_ns = time.perf_counter_ns()
        nvml_processes = self._nvml.process_memory()
        nvml_completed_ns = time.perf_counter_ns()
        all_cuda_pids = set(nvml_processes)
        tree_cuda_pids = all_cuda_pids & tree
        foreign_cuda_pids = all_cuda_pids - tree
        nvml = sum(int(nvml_processes[pid]) for pid in tree_cuda_pids)
        with self._lock:
            active = self._active
        expected_cuda_pids: set[int] = set()
        if active is not None and active[2]:
            expected_cuda_pids = process_tree(active[1])
        unexpected_tree_cuda_pids = tree_cuda_pids - expected_cuda_pids
        with self._lock:
            self.sample_count += 1
            if periodic:
                self.periodic_sample_count += 1
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.peak_swap_bytes = max(self.peak_swap_bytes, swap)
            self.peak_nvml_bytes = max(self.peak_nvml_bytes, nvml)
            self.maximum_process_count = max(self.maximum_process_count, len(tree))
            self.maximum_sample_duration_ns = max(
                self.maximum_sample_duration_ns,
                nvml_completed_ns - sample_started_ns,
            )
            self.maximum_process_tree_sample_ns = max(
                self.maximum_process_tree_sample_ns,
                tree_completed_ns - sample_started_ns,
            )
            self.maximum_nvml_query_ns = max(
                self.maximum_nvml_query_ns,
                nvml_completed_ns - tree_completed_ns,
            )
            self.maximum_cuda_pid_count = max(
                self.maximum_cuda_pid_count,
                len(all_cuda_pids),
            )
            if len(all_cuda_pids) > 1:
                self.multiple_cuda_pid_sample_count += 1
            if foreign_cuda_pids:
                self.foreign_cuda_pid_sample_count += 1
            if unexpected_tree_cuda_pids:
                self.unexpected_tree_cuda_pid_sample_count += 1
            self._record_cuda_pids(
                self.observed_cuda_pids,
                all_cuda_pids,
                nvml_processes,
            )
            self._record_cuda_pids(
                self.foreign_cuda_pids,
                foreign_cuda_pids,
                nvml_processes,
            )
            self._record_cuda_pids(
                self.unexpected_tree_cuda_pids,
                unexpected_tree_cuda_pids,
                nvml_processes,
            )
            if active is not None:
                name, child_pid, _ = active
                peak = self._phase_peaks[name]
                peak.samples += 1
                peak.peak_rss_bytes = max(peak.peak_rss_bytes, rss)
                peak.peak_swap_bytes = max(peak.peak_swap_bytes, swap)
                peak.peak_nvml_bytes = max(peak.peak_nvml_bytes, nvml)
                peak.maximum_process_count = max(peak.maximum_process_count, len(tree))
                child_tree = process_tree(child_pid)
                peak.child_descendant_seen |= len(child_tree) > 1

    def _sample_guarded(self, *, periodic: bool) -> None:
        try:
            self._sample(periodic=periodic)
        except Exception:
            with self._lock:
                self._errors.append(traceback.format_exc())

    def _run(self) -> None:
        previous = time.perf_counter_ns()
        self._last_periodic_sample_ns = previous
        deadline = time.monotonic()
        while not self._stop.is_set():
            self._sample_guarded(periodic=True)
            current = time.perf_counter_ns()
            with self._lock:
                self.maximum_interval_ns = max(self.maximum_interval_ns, current - previous)
                self._last_periodic_sample_ns = current
            previous = current
            deadline += MONITOR_SAMPLE_INTERVAL_NS / 1_000_000_000
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                self._errors.append("resource monitor thread did not stop")
        stopped_ns = time.perf_counter_ns()
        if self._last_periodic_sample_ns is not None:
            self.maximum_interval_ns = max(
                self.maximum_interval_ns,
                stopped_ns - self._last_periodic_sample_ns,
            )
        try:
            self._nvml.close()
        except Exception:
            self._errors.append(traceback.format_exc())
        with self._lock:
            duration_ns = (
                0 if self._started_ns is None else stopped_ns - self._started_ns
            )
            minimum_periodic_samples = max(
                1,
                duration_ns // MAX_MONITOR_SAMPLE_INTERVAL_NS,
            )
            return {
                "sample_interval_target_ns": MONITOR_SAMPLE_INTERVAL_NS,
                "maximum_allowed_sample_interval_ns": MAX_MONITOR_SAMPLE_INTERVAL_NS,
                "monitor_duration_ns": duration_ns,
                "sample_count": self.sample_count,
                "periodic_sample_count": self.periodic_sample_count,
                "minimum_periodic_sample_count": minimum_periodic_samples,
                "maximum_interval_ns": self.maximum_interval_ns,
                "maximum_sample_duration_ns": self.maximum_sample_duration_ns,
                "maximum_process_tree_sample_ns": (
                    self.maximum_process_tree_sample_ns
                ),
                "maximum_nvml_query_ns": self.maximum_nvml_query_ns,
                "peak_process_tree_rss_bytes": self.peak_rss_bytes,
                "peak_process_tree_swap_bytes": self.peak_swap_bytes,
                "peak_summed_process_tree_nvml_bytes": self.peak_nvml_bytes,
                "maximum_process_count": self.maximum_process_count,
                "cuda_overlap_detected": self.cuda_overlap_detected,
                "maximum_cuda_pid_count": self.maximum_cuda_pid_count,
                "multiple_cuda_pid_sample_count": self.multiple_cuda_pid_sample_count,
                "unexpected_tree_cuda_pid_sample_count": (
                    self.unexpected_tree_cuda_pid_sample_count
                ),
                "foreign_cuda_pid_sample_count": self.foreign_cuda_pid_sample_count,
                "observed_cuda_pids": {
                    str(pid): used for pid, used in sorted(self.observed_cuda_pids.items())
                },
                "foreign_cuda_pids": {
                    str(pid): used for pid, used in sorted(self.foreign_cuda_pids.items())
                },
                "unexpected_tree_cuda_pids": {
                    str(pid): used
                    for pid, used in sorted(self.unexpected_tree_cuda_pids.items())
                },
                "phase_peaks": {name: asdict(value) for name, value in self._phase_peaks.items()},
                "monitor_errors": list(self._errors),
            }


def run_child(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    phase: str,
    cuda_visible: bool,
    monitor: ProcessTreeMonitor,
    timeout_seconds: float,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            tuple(command),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        monitor.enter_phase(phase, process.pid, cuda_visible)
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise StageFailure(phase, "child_timeout", str(command)) from error
        finally:
            monitor.leave_phase(phase, process.pid)
            log.flush()
            os.fsync(log.fileno())
    fsync_directory(log_path.parent)
    if return_code != 0:
        raise StageFailure(phase, "child_nonzero", f"exit={return_code}, log={log_path}")
    return return_code


def event(kind: str, **values: Any) -> None:
    payload = canonical_json_bytes(
        {"event": kind, "event_time_ns": time.time_ns(), **values}
    )
    if _EVENT_LOG_PATH is not None:
        with _EVENT_LOG_PATH.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(_EVENT_LOG_PATH.parent)
    print(payload.decode("ascii"), end="", flush=True)


def mount_information(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        ("findmnt", "--json", "--target", str(path.resolve())),
        text=True,
        stderr=subprocess.STDOUT,
    )
    document = json.loads(output)
    filesystems = document.get("filesystems")
    if not isinstance(filesystems, list) or len(filesystems) != 1:
        raise StageFailure("preflight", "findmnt_result", output)
    row = filesystems[0]
    return {
        "target": row.get("target"),
        "source": row.get("source"),
        "fstype": row.get("fstype"),
        "options": row.get("options"),
        "device_id": int(path.stat().st_dev),
    }


def _safe_payload_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise StageFailure("bundle", "unsafe_payload_path", relative)


def build_payload_manifest(root: Path) -> tuple[dict[str, Any], bytes]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in ("payload_manifest.json", "BUNDLE_COMPLETE.json"):
            continue
        _safe_payload_path(relative)
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    document = {
        "schema": "stda_f0_payload_manifest_v2",
        "files": rows,
        "payload_count": len(rows),
        "payload_bytes": sum(row["size"] for row in rows),
    }
    return document, canonical_json_bytes(document)


def seal_bundle(root: Path, *, source_commit: str) -> dict[str, Any]:
    manifest, manifest_bytes = build_payload_manifest(root)
    atomic_write_bytes(root / "payload_manifest.json", manifest_bytes)
    complete = {
        "schema": "stda_f0_bundle_complete_v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": source_commit,
        "payload_count": manifest["payload_count"],
        "payload_manifest_sha256": sha256_bytes(manifest_bytes),
    }
    complete_bytes = atomic_write_json(root / "BUNDLE_COMPLETE.json", complete)
    fsync_directory(root)
    return {
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "complete": complete,
        "complete_bytes": complete_bytes,
        "bundle_root": sha256_bytes(
            BUNDLE_DOMAIN + manifest_bytes + b"\0" + complete_bytes
        ),
    }


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            fsync_directory(path)
    fsync_directory(root)
    fsync_directory(root.parent)


def copy_bundle(source: Path, destination: Path) -> None:
    if destination.exists():
        raise StageFailure("bundle_copy", "destination_exists", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)
    _fsync_tree(destination)


def run_local_bundle_verifier(
    verifier: Path,
    root: Path,
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    output = subprocess.check_output(
        (sys.executable, "-I", "-S", "-B", str(verifier), str(root)),
        env=dict(environment),
        text=True,
        stderr=subprocess.STDOUT,
    )
    document = json.loads(output)
    if document.get("passed") is not True:
        raise StageFailure("bundle_verify", "local_verifier_failed", output)
    return document


def _remote_shared_path(
    path: Path,
    *,
    local_mount_root: Path,
    remote_mount_root: Path,
) -> Path:
    try:
        relative = path.resolve().relative_to(local_mount_root.resolve())
    except ValueError as error:
        raise StageFailure("bundle_verify", "shared_path_mapping", str(path)) from error
    return remote_mount_root.joinpath(*relative.parts)


def run_l40s_bundle_verifier(
    root: Path,
    *,
    shared_mount_root: Path,
    l40s_mount_root: Path,
    l40s_host: str,
) -> dict[str, Any]:
    remote_root = _remote_shared_path(
        root,
        local_mount_root=shared_mount_root,
        remote_mount_root=l40s_mount_root,
    )
    remote_verifier = remote_root / "tools/stda_f0_bundle_verify.py"
    output = subprocess.check_output(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            l40s_host,
            "env",
            "CUDA_VISIBLE_DEVICES=",
            "PYTHONHASHSEED=0",
            "LC_ALL=C",
            "LANG=C",
            "TZ=UTC",
            "python3",
            "-I",
            "-S",
            "-B",
            str(remote_verifier),
            str(remote_root),
        ),
        text=True,
        stderr=subprocess.STDOUT,
    )
    document = json.loads(output)
    if document.get("passed") is not True:
        raise StageFailure("bundle_verify", "l40s_verifier_failed", output)
    return document


def rename_bundle(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise StageFailure("bundle_publish", "final_bundle_exists", str(destination))
    _rename_noreplace(staging, destination)
    fsync_directory(destination.parent)


def available_artifacts(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def _probe_evidence_parent(parent: Path) -> bool:
    if not parent.is_dir():
        return False
    probe = parent / f".stda_f0_write_probe_{os.getpid()}_{time.time_ns()}"
    try:
        exclusive_write_bytes(probe, b"stda-f0-probe\n")
        if probe.read_bytes() != b"stda-f0-probe\n":
            return False
        probe.unlink()
        fsync_directory(parent)
        return True
    except OSError:
        return False
    finally:
        probe.unlink(missing_ok=True)


def publish_failure_record(
    *,
    args: argparse.Namespace,
    source_commit: str,
    terminal_status: str,
    failed_stage: str,
    failed_code: str,
    resources: Mapping[str, Any] | None,
    environment: Mapping[str, Any],
    input_hashes: Mapping[str, Any] | None,
    evidence_root: Path | None,
    bundle_root: str | None,
    bundle_paths: Sequence[str],
    transaction_root: str | None,
) -> dict[str, Any]:
    if terminal_status not in TERMINAL_STATUSES[:2]:
        raise ValueError("FAILURE.json is reserved for terminal statuses 1-2")
    destinations = []
    for parent in (args.shared_evidence_parent, args.local_evidence_parent):
        destinations.append(
            {
                "parent": str(parent.resolve()),
                "available": _probe_evidence_parent(parent),
            }
        )
    document = {
        "schema": "stda_f0_failure_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "source_commit": source_commit,
        "terminal_status": terminal_status,
        "failed_stage": failed_stage,
        "failed_code": failed_code,
        "event_time_ns": time.time_ns(),
        "measured_resources": dict(resources) if resources is not None else None,
        "available_artifacts": available_artifacts(evidence_root),
        "bundle_root": bundle_root,
        "bundle_paths": list(bundle_paths),
        "transaction_root": transaction_root,
        "destinations": destinations,
        "environment": dict(environment),
        "input_hashes": dict(input_hashes) if input_hashes is not None else None,
    }
    payload = canonical_json_bytes(document)
    failure_root = sha256_bytes(FAILURE_DOMAIN + payload)
    filename = f"stda_f0_failure_{source_commit[:8]}_{PROTOCOL_SHA256[:8]}_{failure_root[:16]}.json"
    verified: list[str] = []
    for destination in destinations:
        if not destination["available"]:
            continue
        path = Path(destination["parent"]) / filename
        try:
            exclusive_write_bytes(path, payload)
            if path.read_bytes() != payload:
                raise StageFailure("failure_publish", "failure_replay", str(path))
            verified.append(str(path.resolve()))
        except (OSError, StageFailure):
            continue
    if not verified:
        raise StageFailure("failure_publish", "no_server_failure_copy", filename)
    repository_path = args.repo / "artifacts/idea" / filename
    exclusive_write_bytes(repository_path, payload)
    if repository_path.read_bytes() != payload:
        raise StageFailure("failure_publish", "repository_failure_replay", str(repository_path))
    return {
        "failure_root": failure_root,
        "failure_sha256": sha256_bytes(payload),
        "server_paths": verified,
        "repository_path": str(repository_path.relative_to(args.repo)),
        "repository_sha256": sha256_file(repository_path),
        "document": document,
    }


def publish_transaction_record(
    *,
    args: argparse.Namespace,
    source_commit: str,
    terminal_status: str,
    resources: Mapping[str, Any],
    shared_final: Path,
    local_final: Path,
    shared_mount: Mapping[str, Any],
    local_mount: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    if terminal_status not in TERMINAL_STATUSES[2:]:
        raise ValueError("TRANSACTION.json is reserved for statuses 3-8")
    document = {
        "schema": "stda_f0_transaction_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "source_commit": source_commit,
        "terminal_status": terminal_status,
        "measured_resources": dict(resources),
        "shared_final_path": str(shared_final.resolve()),
        "local_final_path": str(local_final.resolve()),
        "shared_mount": dict(shared_mount),
        "local_mount": dict(local_mount),
        "payload_count": verification["payload_count"],
        "payload_bytes": verification["payload_bytes"],
        "complete_bundle_bytes": verification["complete_bundle_bytes"],
        "bundle_root": verification["bundle_root"],
        "verification_timestamps_ns": dict(verification["timestamps_ns"]),
    }
    payload = canonical_json_bytes(document)
    root = sha256_bytes(TRANSACTION_DOMAIN + payload)
    filename = f"stda_f0_transaction_{source_commit[:8]}_{PROTOCOL_SHA256[:8]}_{root[:16]}.json"
    args.transaction_parent.mkdir(parents=True, exist_ok=True)
    path = args.transaction_parent / filename
    exclusive_write_bytes(path, payload)
    if path.read_bytes() != payload:
        raise StageFailure("transaction_publish", "transaction_replay", str(path))
    return {
        "transaction_root": root,
        "transaction_sha256": sha256_bytes(payload),
        "transaction_path": str(path.resolve()),
        "document": document,
    }


GLOBAL_GATE_KEYS = (
    "exact_10000",
    "finite_xyz",
    "unique_xyz_bytes",
    "strict_spacing_5cm",
    "structural_domain_valid",
    "chamfer_le_0p8",
    "outlier_le_0p05",
)
RANGE_GATE_PREFIXES = (
    "range_0_30",
    "range_30_60",
    "range_60_120",
)
RETURN_GATE_SUFFIXES = (
    "first",
    "later",
)


def build_shared_gate_vector(
    export: Mapping[str, Any],
    metric: Mapping[str, Any],
    structure: Mapping[str, Any],
) -> dict[str, dict[str, bool]]:
    geometry = metric.get("geometry")
    if not isinstance(geometry, Mapping):
        raise StageFailure("gate", "geometry_absent", str(metric))
    spacing = export.get("spacing")
    if not isinstance(spacing, Mapping):
        raise StageFailure("gate", "spacing_absent", str(export))
    vector: dict[str, dict[str, bool]] = {
        "exact_10000": {"applicable": True, "passed": int(export.get("point_count", -1)) == 10_000},
        "finite_xyz": {"applicable": True, "passed": export.get("finite_xyz") is True},
        "unique_xyz_bytes": {"applicable": True, "passed": export.get("unique_xyz_bytes") is True},
        "strict_spacing_5cm": {"applicable": True, "passed": spacing.get("strict_spacing_5cm") is True},
        "structural_domain_valid": {"applicable": True, "passed": structure.get("structural_domain_valid") is True},
        "chamfer_le_0p8": {"applicable": True, "passed": float(geometry["chamfer_m"]) <= 0.8},
        "outlier_le_0p05": {"applicable": True, "passed": float(geometry["outlier_fraction_2m"]) <= 0.05},
    }
    ranges = metric.get("range_metrics")
    if not isinstance(ranges, Mapping):
        raise StageFailure("gate", "range_metrics_absent", str(metric))
    for prefix in RANGE_GATE_PREFIXES:
        row = ranges.get(prefix)
        if not isinstance(row, Mapping):
            raise StageFailure("gate", "range_row_absent", prefix)
        applicable = row.get("applicable") is True
        vector[f"{prefix}_completeness_le_1m"] = {
            "applicable": applicable,
            "passed": (not applicable) or float(row["completeness_mean_distance_m"]) <= 1.0,
        }
        vector[f"{prefix}_recall_1m_ge_0p8"] = {
            "applicable": applicable,
            "passed": (not applicable) or float(row["recall_1m"]) >= 0.8,
        }
    class_rows = structure.get("class_metrics")
    if not isinstance(class_rows, Mapping):
        raise StageFailure("gate", "structure_classes_absent", str(structure))
    for prefix in RANGE_GATE_PREFIXES:
        for suffix in RETURN_GATE_SUFFIXES:
            key = f"{prefix}_{suffix}"
            row = class_rows.get(key)
            if not isinstance(row, Mapping):
                raise StageFailure("gate", "structure_class_absent", key)
            applicable = row.get("applicable") is True
            vector[f"{key}_completeness_le_1m"] = {
                "applicable": applicable,
                "passed": (not applicable) or float(row["completeness_mean_distance_m"]) <= 1.0,
            }
            vector[f"{key}_recall_1m_ge_0p8"] = {
                "applicable": applicable,
                "passed": (not applicable) or float(row["recall_1m"]) >= 0.8,
            }
    expected_keys = set(GLOBAL_GATE_KEYS)
    for prefix in RANGE_GATE_PREFIXES:
        expected_keys.update(
            (f"{prefix}_completeness_le_1m", f"{prefix}_recall_1m_ge_0p8")
        )
        for suffix in RETURN_GATE_SUFFIXES:
            expected_keys.update(
                (
                    f"{prefix}_{suffix}_completeness_le_1m",
                    f"{prefix}_{suffix}_recall_1m_ge_0p8",
                )
            )
    if set(vector) != expected_keys:
        raise StageFailure("gate", "gate_key_set", str(sorted(vector)))
    return vector


def gate_vector_passed(vector: Mapping[str, Mapping[str, bool]]) -> bool:
    return all((not row["applicable"]) or row["passed"] for row in vector.values())


def validate_cross_arm_applicability(
    arm_vectors: Mapping[str, Mapping[str, Mapping[str, bool]]],
) -> None:
    arms = list(arm_vectors)
    if not arms:
        return
    reference = {
        key: bool(value["applicable"])
        for key, value in arm_vectors[arms[0]].items()
    }
    for arm in arms[1:]:
        observed = {
            key: bool(value["applicable"])
            for key, value in arm_vectors[arm].items()
        }
        if observed != reference:
            raise StageFailure("gate", "cross_arm_applicability", arm)


def determine_scientific_status(frames: Sequence[Mapping[str, Any]]) -> str:
    if len(frames) != EXPECTED_TRAIN_FRAMES:
        raise StageFailure("status", "frame_count", str(len(frames)))
    if any(frame["support"]["capacity_sufficient"] is not True for frame in frames):
        return TERMINAL_STATUSES[2]
    if any(int(frame["graph"]["maximum_cardinality"]) < 10_000 for frame in frames):
        return TERMINAL_STATUSES[3]
    decision_pass = all(frame["arms"]["decision"]["passed"] is True for frame in frames)
    if not decision_pass:
        return TERMINAL_STATUSES[4]
    pointwise_failures: list[tuple[int, str]] = []
    greedy_failures: list[tuple[int, str]] = []
    for position, frame in enumerate(frames):
        decision = frame["arms"]["decision"]["gate_vector"]
        for arm_name, sink in (("pointwise", pointwise_failures), ("greedy", greedy_failures)):
            control = frame["arms"][arm_name]["gate_vector"]
            for key in decision:
                if (
                    decision[key]["applicable"]
                    and decision[key]["passed"]
                    and control[key]["applicable"]
                    and not control[key]["passed"]
                ):
                    sink.append((position, key))
    pointwise_all_pass = all(frame["arms"]["pointwise"]["passed"] is True for frame in frames)
    greedy_all_pass = all(frame["arms"]["greedy"]["passed"] is True for frame in frames)
    if pointwise_all_pass:
        return TERMINAL_STATUSES[5]
    if greedy_all_pass:
        return TERMINAL_STATUSES[6]
    if pointwise_failures and greedy_failures:
        return TERMINAL_STATUSES[7]
    raise StageFailure("status", "terminal_partition", "valid gates match no status")


def _monitor_implementation_checks(
    monitor_report: Mapping[str, Any],
) -> dict[str, bool]:
    phase_peaks = monitor_report.get("phase_peaks", {})
    if not isinstance(phase_peaks, Mapping):
        phase_peaks = {}
    support_phase = phase_peaks.get("support", {})
    if not isinstance(support_phase, Mapping):
        support_phase = {}
    return {
        "monitor_clean": not monitor_report.get("monitor_errors"),
        "sampling_target_matches_reported_constant": int(
            monitor_report.get("sample_interval_target_ns", -1)
        )
        == MONITOR_SAMPLE_INTERVAL_NS,
        "sampling_interval_le_50ms": int(
            monitor_report.get(
                "maximum_interval_ns",
                MAX_MONITOR_SAMPLE_INTERVAL_NS + 1,
            )
        )
        <= MAX_MONITOR_SAMPLE_INTERVAL_NS,
        "sampling_periodic_coverage": int(
            monitor_report.get("periodic_sample_count", -1)
        )
        >= int(monitor_report.get("minimum_periodic_sample_count", 1)),
        "all_registered_phases_sampled": bool(phase_peaks)
        and all(int(phase.get("samples", 0)) > 0 for phase in phase_peaks.values()),
        "no_cuda_overlap": monitor_report.get("cuda_overlap_detected") is False,
        "no_foreign_cuda_pid": (
            not monitor_report.get("foreign_cuda_pids")
            and int(monitor_report.get("foreign_cuda_pid_sample_count", -1)) == 0
        ),
        "single_cuda_pid": (
            int(monitor_report.get("maximum_cuda_pid_count", -1)) <= 1
            and int(monitor_report.get("multiple_cuda_pid_sample_count", -1)) == 0
        ),
        "cuda_only_in_registered_cuda_child": (
            not monitor_report.get("unexpected_tree_cuda_pids")
            and int(
                monitor_report.get("unexpected_tree_cuda_pid_sample_count", -1)
            )
            == 0
        ),
        "support_has_no_descendants": support_phase.get("child_descendant_seen")
        is False,
    }


def summarize_resources(
    *,
    monitor_report: Mapping[str, Any],
    support_manifest: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    transaction_work_ns: int,
) -> dict[str, Any]:
    support_frames = support_manifest.get("frames")
    if not isinstance(support_frames, list) or len(support_frames) != EXPECTED_TRAIN_FRAMES:
        raise StageFailure("resource", "support_timing_frames", str(type(support_frames)))
    if len(frames) != EXPECTED_TRAIN_FRAMES:
        raise StageFailure("resource", "oracle_timing_frames", str(len(frames)))
    allocation_ns: list[int] = []
    frame_total_ns: list[int] = []
    support_summary = support_manifest.get("summary", {})
    child_allocated = int(
        support_summary.get("maximum_torch_peak_allocated_bytes", 0)
    )
    child_reserved = int(
        support_summary.get("maximum_torch_peak_reserved_bytes", 0)
    )
    verifier_timing_consistent = True
    verifier_timing_within_frame = True
    oracle_frame_boundary_consistent = True
    for support, frame in zip(support_frames, frames, strict=True):
        support_timing = support.get("timing", {})
        frame_resources = frame["resources"]
        verifier_ns = int(frame_resources["independent_verifier_wall_ns"])
        verifier_serialization_ns = int(
            frame_resources["independent_verifier_serialization_ns"]
        )
        oracle_report_allocation_ns = int(
            frame_resources["oracle_report_allocation_ns"]
        )
        joined_oracle_allocation_ns = (
            oracle_report_allocation_ns + verifier_ns + verifier_serialization_ns
        )
        verifier_timing_consistent &= (
            int(frame_resources["oracle_allocation_ns"])
            == joined_oracle_allocation_ns
        )
        verifier_timing_within_frame &= (
            int(frame_resources["oracle_frame_ns"])
            >= verifier_ns + verifier_serialization_ns
        )
        oracle_frame_boundary_consistent &= (
            frame_resources.get("oracle_frame_timing_lower_bound_valid") is True
            and int(frame_resources["oracle_parent_pre_target_ns_excluded"]) >= 0
            and int(frame_resources["oracle_parent_child_wall_ns_reported"])
            >= int(frame_resources["oracle_child_pre_metric_ns"])
            and int(frame_resources["oracle_frame_ns"])
            >= int(frame_resources["oracle_frame_measured_lower_bound_ns"])
        )
        allocation_ns.append(
            int(support_timing["support_allocation_ns"])
            + joined_oracle_allocation_ns
        )
        frame_total_ns.append(
            int(support_timing["support_frame_ns"])
            + int(frame_resources["oracle_frame_ns"])
        )
        child_allocated = max(
            child_allocated,
            int(frame_resources.get("torch_peak_allocated_bytes", 0)),
        )
        child_reserved = max(
            child_reserved,
            int(frame_resources.get("torch_peak_reserved_bytes", 0)),
        )
    nvml = int(monitor_report["peak_summed_process_tree_nvml_bytes"])
    cuda_peak = max(child_allocated, child_reserved, nvml)
    checks = {
        **_monitor_implementation_checks(monitor_report),
        "verifier_timing_consistent": verifier_timing_consistent,
        "verifier_timing_within_frame": verifier_timing_within_frame,
        "oracle_frame_boundary_consistent": oracle_frame_boundary_consistent,
        "host_rss_le_60gib": int(monitor_report["peak_process_tree_rss_bytes"]) <= MAX_HOST_BYTES,
        "swap_zero": int(monitor_report["peak_process_tree_swap_bytes"]) == 0,
        "cuda_le_60gib": cuda_peak <= MAX_CUDA_BYTES,
        "allocation_le_30s_each": all(value <= MAX_ALLOCATION_NS for value in allocation_ns),
        "frame_total_le_120s_each": all(value <= MAX_FRAME_NS for value in frame_total_ns),
        "transaction_le_7200s": transaction_work_ns <= MAX_TRANSACTION_NS,
    }
    return {
        "transaction_work_ns": transaction_work_ns,
        "peak_process_tree_rss_bytes": int(monitor_report["peak_process_tree_rss_bytes"]),
        "peak_process_tree_swap_bytes": int(monitor_report["peak_process_tree_swap_bytes"]),
        "torch_peak_allocated_bytes": child_allocated,
        "torch_peak_reserved_bytes": child_reserved,
        "peak_summed_process_tree_nvml_bytes": nvml,
        "deciding_cuda_peak_bytes": cuda_peak,
        "maximum_allocation_core_ns": max(allocation_ns, default=0),
        "maximum_frame_total_ns": max(frame_total_ns, default=0),
        "allocation_core_ns_by_frame": allocation_ns,
        "frame_total_ns_by_frame": frame_total_ns,
        "monitor": dict(monitor_report),
        "checks": checks,
        "implementation_valid": all(
            checks[key]
            for key in (
                "monitor_clean",
                "sampling_target_matches_reported_constant",
                "sampling_interval_le_50ms",
                "sampling_periodic_coverage",
                "all_registered_phases_sampled",
                "no_cuda_overlap",
                "no_foreign_cuda_pid",
                "single_cuda_pid",
                "cuda_only_in_registered_cuda_child",
                "support_has_no_descendants",
                "verifier_timing_consistent",
                "verifier_timing_within_frame",
                "oracle_frame_boundary_consistent",
            )
        ),
        "resource_valid": all(
            checks[key]
            for key in (
                "host_rss_le_60gib",
                "swap_zero",
                "cuda_le_60gib",
                "allocation_le_30s_each",
                "frame_total_le_120s_each",
                "transaction_le_7200s",
            )
        ),
    }


def environment_report(args: argparse.Namespace, gpu: Mapping[str, Any]) -> dict[str, Any]:
    probe_statement = (
        "import json,numpy,scipy,torch;"
        "print(json.dumps({'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'torch':torch.__version__,'cuda':torch.version.cuda},sort_keys=True))"
    )
    probe = subprocess.check_output(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _isolated_python_bootstrap(repo=args.repo, statement=probe_statement),
        ),
        env=child_environment(repo=args.repo, physical_gpu=None, phase="environment_probe"),
        text=True,
        stderr=subprocess.STDOUT,
    )
    versions = json.loads(probe)
    cpu_model = ""
    for line in Path("/proc/cpuinfo").read_text(encoding="ascii").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    report = {
        "hostname": os.uname().nodename,
        "user": pwd.getpwuid(os.getuid()).pw_name,
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "versions": versions,
        "cpu_model": cpu_model,
        "logical_core_count": os.cpu_count(),
        "gpu": dict(gpu),
        "cuda_driver_version": str(gpu.get("driver_version", "")),
        "orchestrator_interpreter": interpreter_isolation_report(),
        "solver_environment": {
            "PYTHONHASHSEED": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }
    checks = {
        "hostname_h200": report["hostname"] == "WHUServer-H200",
        "user_wangning": report["user"] == "wangning",
        "hym_environment": "/envs/hym_" in report["python_executable"],
        "python_3_10_20": report["python"] == "3.10.20",
        "numpy_2_2_6": versions["numpy"] == "2.2.6",
        "scipy_1_15_3": versions["scipy"] == "1.15.3",
        "torch_2_12_1_cu130": versions["torch"] == "2.12.1+cu130" and versions["cuda"] == "13.0",
        "cuda_driver_recorded": bool(report["cuda_driver_version"]),
        "orchestrator_isolated_no_site": report["orchestrator_interpreter"]["passed"]
        is True,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    if not report["passed"]:
        raise StageFailure("preflight", "environment_contract", str(checks))
    return report


def _resolved_existing_directory(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise StageFailure("preflight", f"missing_{label}", str(path)) from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise StageFailure("preflight", f"invalid_{label}", str(resolved))
    return resolved


def _resolved_existing_file(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise StageFailure("preflight", f"missing_{label}", str(path)) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise StageFailure("preflight", f"invalid_{label}", str(resolved))
    return resolved


def _canonical_record(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    document = canonical_json_file(path)
    if schema is not None and document.get("schema") != schema:
        raise StageFailure(
            "serialization",
            "schema_mismatch",
            f"{path}: expected {schema}, got {document.get('schema')}",
        )
    return document


def _load_verifier_parent_timing_evidence(
    *,
    verification_path: Path,
    log_path: Path,
    verification: Mapping[str, Any],
    parent_started_ns: int,
    parent_completed_ns: int,
) -> dict[str, Any]:
    authority = verification.get("timing_authority")
    if not isinstance(authority, Mapping):
        raise StageFailure(
            "implementation",
            "verifier_timing_authority_absent",
            str(verification_path),
        )
    module_started_ns = authority.get("module_wall_started_perf_counter_ns")
    summary = _canonical_record(
        log_path,
        schema="stda_f0_independent_frame_verification_v1",
    )
    internal_wall_ns = summary.get("internal_full_wall_through_output_rehash_ns")
    output_payload = verification_path.read_bytes()
    expected_boundary = (
        "verifier_module_start_through_output_fsync_and_independent_rehash"
    )
    integer_fields = (
        type(module_started_ns) is int
        and type(internal_wall_ns) is int
        and int(internal_wall_ns) > 0
    )
    internal_ended_ns = (
        int(module_started_ns) + int(internal_wall_ns) if integer_fields else -1
    )
    checks = {
        "report_parent_wall_authoritative": authority.get(
            "parent_process_wall_authoritative"
        )
        is True,
        "report_internal_wall_emitted_after_rehash": authority.get(
            "internal_process_wall_emitted_after_output_rehash"
        )
        is True,
        "summary_parent_wall_authoritative": summary.get(
            "parent_process_wall_authoritative"
        )
        is True,
        "summary_internal_boundary": summary.get("internal_full_wall_boundary")
        == expected_boundary,
        "summary_passed": summary.get("passed") is True
        and verification.get("passed") is True,
        "summary_output_path": summary.get("output")
        == str(verification_path.absolute()),
        "summary_output_size": type(summary.get("output_size_bytes")) is int
        and int(summary["output_size_bytes"]) == len(output_payload),
        "summary_output_sha256": summary.get("output_sha256")
        == sha256_bytes(output_payload),
        "integer_timing_fields": integer_fields,
        "child_module_started_inside_parent_wall": integer_fields
        and parent_started_ns <= int(module_started_ns) <= parent_completed_ns,
        "child_rehash_ended_inside_parent_wall": integer_fields
        and int(module_started_ns) <= internal_ended_ns <= parent_completed_ns,
    }
    if not all(checks.values()):
        raise StageFailure(
            "implementation",
            "verifier_timing_authority_invalid",
            str(checks),
        )
    return {
        "schema": "stda_f0_verifier_parent_timing_evidence_v1",
        "parent_process_wall_authoritative": True,
        "parent_started_perf_counter_ns": int(parent_started_ns),
        "parent_completed_perf_counter_ns": int(parent_completed_ns),
        "parent_wall_ns": int(parent_completed_ns - parent_started_ns),
        "child_module_started_perf_counter_ns": int(module_started_ns),
        "child_internal_full_wall_through_output_rehash_ns": int(
            internal_wall_ns
        ),
        "child_output_rehash_completed_perf_counter_ns": internal_ended_ns,
        "completion_receipt_sha256": sha256_file(log_path),
        "verification_output_sha256": sha256_bytes(output_payload),
        "checks": checks,
        "passed": True,
    }


def _copy_regular_file(source: Path, destination: Path) -> dict[str, Any]:
    source = _resolved_existing_file(source, label="copy_source")
    if destination.exists() or destination.is_symlink():
        raise StageFailure("bundle", "copy_destination_exists", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    exclusive_write_bytes(destination, payload)
    return {
        "path": str(destination),
        "size": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _target_cache_path(cache_root: Path, record: Mapping[str, Any]) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_radar_"
        f"{int(record['radar_index']):05d}.npz"
    )


def _frame_directory_name(record: Mapping[str, Any]) -> str:
    return (
        f"seq{int(record['sequence']):02d}_"
        f"radar{int(record['radar_index']):05d}"
    )


def _frame_key(record: Mapping[str, Any]) -> str:
    return (
        f"seq{int(record['sequence']):02d}/"
        f"radar{int(record['radar_index']):05d}"
    )


def _ensure_distinct_mounts(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shared = mount_information(args.shared_evidence_parent)
    local = mount_information(args.local_evidence_parent)
    checks = {
        "absolute_paths_differ": (
            args.shared_evidence_parent.resolve()
            != args.local_evidence_parent.resolve()
        ),
        "device_ids_differ": shared["device_id"] != local["device_id"],
        "shared_parent_below_mount": (
            args.shared_evidence_parent.resolve().is_relative_to(
                args.shared_mount_root.resolve()
            )
        ),
        "transaction_parent_below_shared_mount": (
            args.transaction_parent.resolve().is_relative_to(
                args.shared_mount_root.resolve()
            )
        ),
        "transaction_parent_on_shared_device": (
            args.transaction_parent.stat().st_dev == shared["device_id"]
        ),
        "local_parent_below_mirror": (
            args.local_evidence_parent.resolve().is_relative_to(
                Path("/home/wangning/stda_evidence_mirror").resolve()
            )
        ),
        "shared_is_sshfs": shared.get("fstype") == "fuse.sshfs",
        "shared_source_is_l40s": "10.254.30.44" in str(shared.get("source")),
        "local_is_not_sshfs": local.get("fstype") != "fuse.sshfs",
    }
    if not all(checks.values()):
        raise StageFailure("preflight", "dual_copy_mount_contract", str(checks))
    return ({**shared, "checks": checks}, local)


def _foreign_gpu_processes(physical_index: int) -> list[dict[str, Any]]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise StageFailure("preflight", "gpu_process_query", completed.stdout)
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        columns = [value.strip() for value in line.split(",", 2)]
        if len(columns) != 3:
            raise StageFailure("preflight", "gpu_process_parse", line)
        rows.append(
            {
                "pid": int(columns[0]),
                "process_name": columns[1],
                "used_memory_mib": int(columns[2]),
            }
        )
    return rows


def _require_gpu_idle_snapshot(physical_index: int, *, label: str) -> dict[str, Any]:
    processes = _foreign_gpu_processes(physical_index)
    snapshot = {
        "label": label,
        "observed_time_ns": time.time_ns(),
        "physical_gpu": physical_index,
        "foreign_processes": processes,
        "passed": not processes,
    }
    if processes:
        raise StageFailure("preflight", "gpu2_not_idle", str(snapshot))
    return snapshot


def _run_mandatory_pytest_group(
    *,
    repo: Path,
    name: str,
    test_paths: Sequence[str],
    gpu_uuid: str | None,
    source_hashes: Mapping[str, Any],
    extra_environment: Mapping[str, str] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    expected_test_hashes: dict[str, str] = {}
    for relative in test_paths:
        path = (repo / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(repo.resolve() / "code/tests"):
            raise StageFailure("preflight", "mandatory_test_path", str(path))
        expected = source_hashes.get(relative)
        observed = sha256_file(path)
        if not isinstance(expected, str) or observed != expected:
            raise StageFailure(
                "preflight",
                "mandatory_test_source_hash",
                f"{relative}: expected={expected}, observed={observed}",
            )
        expected_test_hashes[relative] = observed

    environment = child_environment(
        repo=repo,
        physical_gpu=gpu_uuid,
        phase=f"preflight_pytest_{name}",
    )
    environment.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "STDA_F0_SYNTHETIC_ONLY": "1",
        }
    )
    if extra_environment:
        environment.update({str(key): str(value) for key, value in extra_environment.items()})

    with tempfile.TemporaryDirectory(prefix=f"stda_f0_pytest_{name}_") as temporary:
        base_temp = Path(temporary) / "pytest"
        pytest_arguments = (
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(base_temp),
            *(str(repo / relative) for relative in test_paths),
        )
        statement = "import pytest;raise SystemExit(pytest.main(sys.argv[1:]))"
        command = (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _isolated_python_bootstrap(repo=repo, statement=statement),
            *pytest_arguments,
        )
        started_ns = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b""
            raise StageFailure(
                "preflight",
                "mandatory_tests_timeout",
                f"{name}: output_sha256={sha256_bytes(output)}",
            ) from error
        elapsed_ns = time.perf_counter_ns() - started_ns
    output = bytes(completed.stdout)
    report = {
        "name": name,
        "command": list(command),
        "command_sha256": sha256_bytes(canonical_json_bytes(list(command))),
        "test_paths": list(test_paths),
        "test_source_sha256": expected_test_hashes,
        "return_code": int(completed.returncode),
        "wall_ns": elapsed_ns,
        "stdout_bytes": len(output),
        "stdout_sha256": sha256_bytes(output),
        "stdout_tail": output[-4096:].decode("utf-8", errors="replace"),
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "python_isolated_no_site": command[1:3] == ("-I", "-S"),
        "pytest_cache_disabled": "no:cacheprovider" in command,
        "synthetic_only": environment["STDA_F0_SYNTHETIC_ONLY"] == "1",
        "real_target_or_cache_arguments": [],
        "passed": completed.returncode == 0,
    }
    if report["passed"] is not True:
        raise StageFailure(
            "preflight",
            "mandatory_tests_failed",
            f"{name}: {report['stdout_tail']}",
        )
    return report


def run_mandatory_synthetic_tests(
    *,
    repo: Path,
    physical_gpu: int,
    gpu_uuid: str,
    gpu_pci: str,
    gpu_name: str,
    source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    idle_snapshots: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    specifications = (
        (
            "cpu",
            MANDATORY_CPU_TEST_PATHS,
            None,
            {},
            900.0,
        ),
        (
            "candidate_700k_cuda",
            MANDATORY_CANDIDATE_TEST_PATHS,
            gpu_uuid,
            {},
            900.0,
        ),
        (
            "metric_explicit_cuda_smoke",
            MANDATORY_METRIC_TEST_PATHS,
            gpu_uuid,
            {
                "STDA_F0_CUDA_SMOKE": "1",
                "STDA_EXPECTED_GPU_UUID": gpu_uuid,
                "STDA_EXPECTED_GPU_PCI": gpu_pci,
                "STDA_EXPECTED_GPU_NAME": gpu_name,
            },
            600.0,
        ),
    )
    for name, paths, visible_uuid, extra, timeout_seconds in specifications:
        idle_snapshots.append(
            _require_gpu_idle_snapshot(physical_gpu, label=f"before_{name}")
        )
        groups.append(
            _run_mandatory_pytest_group(
                repo=repo,
                name=name,
                test_paths=paths,
                gpu_uuid=visible_uuid,
                source_hashes=source_hashes,
                extra_environment=extra,
                timeout_seconds=timeout_seconds,
            )
        )
        idle_snapshots.append(
            _require_gpu_idle_snapshot(physical_gpu, label=f"after_{name}")
        )
    document = {
        "schema": "stda_f0_mandatory_synthetic_tests_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "groups": groups,
        "gpu_idle_snapshots": idle_snapshots,
        "test_count": sum(len(group["test_paths"]) for group in groups),
        "candidate_700k_included": MANDATORY_CANDIDATE_TEST_PATHS[0]
        in groups[1]["test_paths"],
        "metric_isolated_import_test_included": MANDATORY_METRIC_TEST_PATHS[0]
        in groups[2]["test_paths"],
        "explicit_metric_smoke_enabled": groups[2]["cuda_visible_devices"]
        == gpu_uuid,
        "cache_root_opened": False,
        "passed": all(group["passed"] for group in groups)
        and all(snapshot["passed"] for snapshot in idle_snapshots),
    }
    document["report_sha256"] = sha256_bytes(canonical_json_bytes(document))
    return document


def _locked_source_paths(repo: Path) -> tuple[str, ...]:
    fixed_paths = {
        "docs/stda_f0_sparse_target_demand_assignment_protocol.md",
        "artifacts/idea/stda_f0_freeze_record.json",
        "artifacts/idea/stda_f0_prefreeze_audit_round6.md",
        "code/eval/dense_geometry.py",
        "code/eval/vrh_f0_metrics.py",
        "code/eval/vrh_f0_support.py",
        "code/eval/rald_wce_stage0.py",
        "code/models/rald_wce_field.py",
        "code/models/rald_matched.py",
        "code/models/cube_cycle.py",
        "code/cube_dense/kradar.py",
        "code/scripts/preflight_stda_f0_capacity.py",
        *SUPPORT_RUNTIME_SOURCE_PATHS,
        *ORACLE_CHILD_SOURCE_PATHS,
        *VERIFY_CHILD_SOURCE_PATHS,
        *MANDATORY_TEST_PATHS,
    }
    for pattern in (
        "code/eval/stda_f0_*.py",
        "code/scripts/stda_f0_*.py",
        "code/tests/test_stda_f0_*.py",
    ):
        fixed_paths.update(
            str(path.relative_to(repo)) for path in repo.glob(pattern) if path.is_file()
        )
    return tuple(sorted(fixed_paths))


def _source_binding(repo: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for relative in _locked_source_paths(repo):
        path = repo / relative
        if not path.is_file():
            raise StageFailure("preflight", "source_file_missing", relative)
        records[relative] = sha256_file(path)
    return records


def _require_no_runtime_caches(repo: Path, *, label: str) -> dict[str, Any]:
    forbidden: list[str] = []
    for root, directories, filenames in os.walk(repo, followlinks=False):
        directories[:] = [name for name in directories if name != ".git"]
        for name in tuple(directories):
            if name in {"__pycache__", ".pytest_cache"}:
                forbidden.append(str((Path(root) / name).relative_to(repo)))
                directories.remove(name)
        forbidden.extend(
            str((Path(root) / name).relative_to(repo))
            for name in filenames
            if name.endswith((".pyc", ".pyo"))
        )
    if forbidden:
        raise StageFailure(
            "preflight",
            "runtime_cache_present",
            f"{label}: {sorted(forbidden)}",
        )
    return {
        "label": label,
        "forbidden_cache_paths": [],
        "passed": True,
    }


def _assert_source_lock(
    *,
    repo: Path,
    source_commit: str,
    expected_hashes: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    try:
        verify_source_tree(repo, source_commit)
        observed = _source_binding(repo)
    except BaseException as error:
        code = error.code if isinstance(error, StageFailure) else type(error).__name__
        raise StageFailure(
            "implementation",
            f"source_lock_{code}",
            f"{label}: {error}",
        ) from error
    expected = {str(key): str(value) for key, value in expected_hashes.items()}
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        added = sorted(set(observed) - set(expected))
        changed = sorted(
            key
            for key in set(observed) & set(expected)
            if observed[key] != expected[key]
        )
        raise StageFailure(
            "implementation",
            "source_hash_lock_mismatch",
            f"{label}: missing={missing}, added={added}, changed={changed}",
        )
    return {
        "label": label,
        "verified_time_ns": time.time_ns(),
        "source_commit": source_commit,
        "source_file_count": len(observed),
        "source_hashes_sha256": sha256_bytes(canonical_json_bytes(observed)),
        "head_matches": True,
        "worktree_clean": True,
        "hashes_match": True,
    }


def _validate_child_source_evidence(
    *,
    child_name: str,
    observed_hashes: Any,
    expected_hashes: Mapping[str, Any],
    required_paths: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(observed_hashes, Mapping):
        raise StageFailure(
            "implementation",
            "child_source_evidence_absent",
            child_name,
        )
    observed = {str(key): str(value) for key, value in observed_hashes.items()}
    expected = {str(key): str(value) for key, value in expected_hashes.items()}
    required = set(required_paths)
    missing = sorted(required - set(observed))
    unknown = sorted(set(observed) - set(expected))
    changed = sorted(
        key for key in observed if key in expected and observed[key] != expected[key]
    )
    if missing or unknown or changed:
        raise StageFailure(
            "implementation",
            "child_source_evidence_mismatch",
            (
                f"{child_name}: missing={missing}, unknown={unknown}, "
                f"changed={changed}"
            ),
        )
    return {
        "child": child_name,
        "reported_source_count": len(observed),
        "required_source_count": len(required),
        "reported_source_hashes_sha256": sha256_bytes(
            canonical_json_bytes(observed)
        ),
        "passed": True,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    orchestrator_isolation = require_orchestrator_isolation()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise StageFailure(
            "preflight",
            "orchestrator_cuda_visibility",
            repr(os.environ.get("CUDA_VISIBLE_DEVICES")),
        )
    args.repo = _resolved_existing_directory(args.repo, label="repo")
    args.data_root = _resolved_existing_directory(args.data_root, label="data_root")
    args.cache_root = args.cache_root.expanduser().resolve(strict=False)
    args.shared_mount_root = _resolved_existing_directory(
        args.shared_mount_root,
        label="shared_mount_root",
    )
    args.shared_evidence_parent.mkdir(parents=True, exist_ok=True)
    args.local_evidence_parent.mkdir(parents=True, exist_ok=True)
    args.transaction_parent.mkdir(parents=True, exist_ok=True)
    args.shared_evidence_parent = _resolved_existing_directory(
        args.shared_evidence_parent,
        label="shared_evidence_parent",
    )
    args.local_evidence_parent = _resolved_existing_directory(
        args.local_evidence_parent,
        label="local_evidence_parent",
    )
    args.transaction_parent = _resolved_existing_directory(
        args.transaction_parent,
        label="transaction_parent",
    )
    for name in (
        "manifest",
        "scene_split",
        "normalization",
        "checkpoint",
        "formal_metrics",
        "formal_run_manifest",
        "candidate_hash_manifest",
    ):
        setattr(
            args,
            name,
            _resolved_existing_file(getattr(args, name), label=name),
        )
    source = verify_source_tree(args.repo, args.source_commit)
    input_hashes, records = verify_frozen_inputs(args)
    gpu = require_allowed_h200(args.physical_gpu)
    if int(args.physical_gpu) != 2:
        raise StageFailure("preflight", "gpu2_required_by_user", str(args.physical_gpu))
    args.gpu_uuid = str(gpu["uuid"])
    environment = environment_report(args, gpu)
    shared_mount, local_mount = _ensure_distinct_mounts(args)
    source_hashes = _source_binding(args.repo)
    runtime_cache_before_tests = _require_no_runtime_caches(
        args.repo,
        label="preflight_before_mandatory_tests",
    )
    source_lock_before_tests = _assert_source_lock(
        repo=args.repo,
        source_commit=args.source_commit,
        expected_hashes=source_hashes,
        label="preflight_before_mandatory_tests",
    )
    mandatory_tests = run_mandatory_synthetic_tests(
        repo=args.repo,
        physical_gpu=args.physical_gpu,
        gpu_uuid=args.gpu_uuid,
        gpu_pci=str(gpu["pci_bus_id"]),
        gpu_name=str(gpu["name"]),
        source_hashes=source_hashes,
    )
    runtime_cache_after_tests = _require_no_runtime_caches(
        args.repo,
        label="preflight_after_mandatory_tests",
    )
    source_lock_after_tests = _assert_source_lock(
        repo=args.repo,
        source_commit=args.source_commit,
        expected_hashes=source_hashes,
        label="preflight_after_mandatory_tests",
    )
    foreign_processes = _foreign_gpu_processes(args.physical_gpu)
    report = {
        "schema": "stda_f0_preflight_v1",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "source": source,
        "source_hashes": source_hashes,
        "source_execution_locks": [
            source_lock_before_tests,
            source_lock_after_tests,
        ],
        "runtime_cache_checks": [
            runtime_cache_before_tests,
            runtime_cache_after_tests,
        ],
        "input_hashes": input_hashes,
        "cohort": records,
        "gpu": gpu,
        "foreign_gpu_processes": foreign_processes,
        "mandatory_synthetic_tests": mandatory_tests,
        "environment": environment,
        "orchestrator_interpreter": orchestrator_isolation,
        "shared_mount": shared_mount,
        "local_mount": local_mount,
        "target_cache_opened": False,
        "orchestrator_cuda_visible_devices": "",
        "orchestrator_imports_cuda_runtime": False,
        "formal_launch_consumed": False,
        "passed": not foreign_processes
        and mandatory_tests["passed"] is True
        and orchestrator_isolation["passed"] is True,
    }
    if foreign_processes:
        raise StageFailure("preflight", "gpu2_not_idle", str(foreign_processes))
    return report


def _create_formal_launch_marker(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    marker = args.shared_evidence_parent / (
        f"stda_f0_formal_launch_{PROTOCOL_SHA256[:16]}.json"
    )
    log_path = args.shared_evidence_parent / (
        f"stda_f0_formal_log_{PROTOCOL_SHA256[:16]}.jsonl"
    )
    forbidden_patterns = (
        f"stda_f0_{args.source_commit[:8]}_{PROTOCOL_SHA256[:8]}_",
        f"stda_f0_transaction_{args.source_commit[:8]}_{PROTOCOL_SHA256[:8]}_",
        f"stda_f0_failure_{args.source_commit[:8]}_{PROTOCOL_SHA256[:8]}_",
    )
    prior = sorted(
        str(path)
        for parent in (
            args.shared_evidence_parent,
            args.local_evidence_parent,
            args.transaction_parent,
        )
        for path in parent.iterdir()
        if path.name == marker.name
        or path.name == log_path.name
        or (
            path.name.startswith("stda_f0_")
            and f"_{PROTOCOL_SHA256[:8]}_" in path.name
        )
        or any(path.name.startswith(prefix) for prefix in forbidden_patterns)
    )
    if prior:
        raise StageFailure("launch", "prior_formal_evidence", str(prior))
    document = {
        "schema": "stda_f0_formal_launch_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "source_commit": args.source_commit,
        "physical_gpu": args.physical_gpu,
        "gpu_uuid": preflight["gpu"]["uuid"],
        "cohort_frame_count": EXPECTED_TRAIN_FRAMES,
        "launch_time_ns": time.time_ns(),
        "one_target_conditioned_launch_consumed": True,
        "log_path": str(log_path.resolve()),
    }
    exclusive_write_bytes(marker, canonical_json_bytes(document))
    exclusive_write_bytes(log_path, b"")
    return marker, document


def _freeze_tree_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise StageFailure("support", "support_symlink", str(path))
        if path.is_file():
            os.chmod(path, 0o444)
        elif path.is_dir():
            os.chmod(path, 0o555)
            fsync_directory(path)
    os.chmod(root, 0o555)
    fsync_directory(root)
    fsync_directory(root.parent)


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_support_sidecar(
    *,
    support_root: Path,
    record: Mapping[str, Any],
    pointer: Any,
    filename: str,
    schema: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(pointer, Mapping):
        raise StageFailure(
            "support",
            "support_sidecar_record_absent",
            f"{_frame_key(record)}:{filename}",
        )
    expected_relative = str(
        PurePosixPath("frames") / _frame_directory_name(record) / filename
    )
    size_bytes = pointer.get("size_bytes")
    if (
        pointer.get("relative_path") != expected_relative
        or type(size_bytes) is not int
        or size_bytes < 0
        or not _is_lower_sha256(pointer.get("sha256"))
        or pointer.get("fsynced") is not True
        or pointer.get("full_rehash_passed") is not True
    ):
        raise StageFailure(
            "support",
            "support_sidecar_record_invalid",
            f"{_frame_key(record)}:{filename}:{dict(pointer)}",
        )
    path = support_root.joinpath(*PurePosixPath(expected_relative).parts)
    if path.is_symlink() or not path.is_file():
        raise StageFailure("support", "support_sidecar_missing", str(path))
    before = path.stat()
    document = _canonical_record(path, schema=schema)
    replay = path.read_bytes()
    after = path.stat()
    if (
        before != after
        or len(replay) != size_bytes
        or sha256_bytes(replay) != pointer.get("sha256")
        or canonical_json_bytes(document) != replay
    ):
        raise StageFailure("support", "support_sidecar_rehash", str(path))
    return document, dict(pointer)


def _validate_support_manifest(
    support_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    expected_source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = support_root / "support_manifest.json"
    manifest = _canonical_record(
        manifest_path,
        schema="stda_f0_support_manifest_v1",
    )
    if (
        manifest.get("protocol_sha256") != PROTOCOL_SHA256
        or manifest.get("protocol_freeze_commit") != PROTOCOL_FREEZE_COMMIT
        or manifest.get("source_commit") != source_commit
    ):
        raise StageFailure("support", "support_manifest_binding", str(manifest_path))
    _validate_child_source_evidence(
        child_name="support",
        observed_hashes=manifest.get("source_hashes"),
        expected_hashes=expected_source_hashes,
        required_paths=SUPPORT_RUNTIME_SOURCE_PATHS,
    )
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != EXPECTED_TRAIN_FRAMES:
        raise StageFailure("support", "support_frame_count", str(type(frames)))
    formal_contract = manifest.get("formal_contract")
    required_contract = {
        "partition": "train",
        "frame_count": EXPECTED_TRAIN_FRAMES,
        "candidate_count_per_frame": 700_000,
        "predecessor_hash_frames": 12,
        "fresh_hash_frames": 64,
        "single_cuda_visible_os_process": True,
        "descendant_processes_created": False,
        "target_loader_started": False,
    }
    if not isinstance(formal_contract, Mapping) or any(
        formal_contract.get(key) != value for key, value in required_contract.items()
    ):
        raise StageFailure("support", "support_formal_contract", str(formal_contract))
    summary = manifest.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("predecessor_hashes_verified") != 12
        or summary.get("fresh_candidate_hashes_recorded") != 64
        or summary.get("all_independent_support_verifications_passed") is not True
    ):
        raise StageFailure("support", "support_summary", str(summary))
    for expected, frame in zip(records, frames, strict=True):
        identity = (int(frame.get("sequence", -1)), int(frame.get("radar_index", -1)))
        if identity != (int(expected["sequence"]), int(expected["radar_index"])):
            raise StageFailure("support", "support_frame_order", str(identity))
        if frame.get("frame_key") != _frame_key(expected):
            raise StageFailure("support", "support_frame_key", str(identity))
        if (
            int(frame.get("position", -1)) != int(expected["position"])
            or frame.get("partition") != "train"
        ):
            raise StageFailure("support", "support_frame_identity", str(identity))
        support = frame.get("support")
        candidate = frame.get("candidate")
        cube = frame.get("cube")
        files = frame.get("files")
        if (
            not isinstance(support, Mapping)
            or not isinstance(candidate, Mapping)
            or not isinstance(cube, Mapping)
            or not isinstance(files, Mapping)
        ):
            raise StageFailure("support", "support_frame_schema", str(identity))
        if (
            candidate.get("passed") is not True
            or int(support.get("candidate_count", -1)) != 700_000
            or frame.get("support_target_input") is not False
            or frame.get("ground_truth_accessed") is not False
        ):
            raise StageFailure("support", "candidate_frame_contract", str(identity))
        if support.get("formal_independent_spacing_verified") is not True:
            raise StageFailure("support", "support_spacing_unverified", str(identity))
        independent = support.get("independent_verifier")
        if not isinstance(independent, Mapping) or independent.get("passed") is not True:
            raise StageFailure("support", "support_independent_replay", str(identity))
        for filename, row in files.items():
            if not isinstance(filename, str) or not isinstance(row, Mapping):
                raise StageFailure("support", "support_file_record", str(identity))
            relative = row.get("relative_path")
            expected_relative = str(
                PurePosixPath("frames")
                / _frame_directory_name(expected)
                / filename
            )
            size_bytes = row.get("size_bytes")
            if (
                relative != expected_relative
                or type(size_bytes) is not int
                or size_bytes < 0
                or not _is_lower_sha256(row.get("sha256"))
                or row.get("fsynced") is not True
                or row.get("full_rehash_passed") is not True
            ):
                raise StageFailure("support", "support_file_path", str(identity))
            path = support_root.joinpath(*PurePosixPath(relative).parts)
            if not path.is_file() or path.is_symlink():
                raise StageFailure("support", "support_file_missing", str(path))
            if (
                path.stat().st_size != size_bytes
                or sha256_file(path) != row.get("sha256")
            ):
                raise StageFailure("support", "support_file_hash", str(path))
        support_bin = files.get("support.bin")
        if not isinstance(support_bin, Mapping):
            raise StageFailure("support", "support_bin_record", str(identity))
        if support_bin.get("sha256") != support.get("support_sha256"):
            raise StageFailure("support", "support_commit_mismatch", str(identity))
        _validate_child_source_evidence(
            child_name=f"support_frame:{_frame_key(expected)}",
            observed_hashes=frame.get("runtime_source_sha256"),
            expected_hashes=expected_source_hashes,
            required_paths=SUPPORT_RUNTIME_SOURCE_PATHS,
        )
        commitment, commitment_record = _load_support_sidecar(
            support_root=support_root,
            record=expected,
            pointer=frame.get("frame_commitment"),
            filename="support_record.json",
            schema="stda_f0_support_frame_commitment_v1",
        )
        commitment_expected = {
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "position": int(expected["position"]),
            "sequence": int(expected["sequence"]),
            "radar_index": int(expected["radar_index"]),
            "cube_sha256": cube.get("sha256"),
            "cube": dict(cube),
            "candidate_hashes": candidate.get("hashes"),
            "candidate_predecessor_expected": candidate.get(
                "predecessor_expected"
            ),
            "candidate_predecessor_hashes_match": candidate.get(
                "predecessor_hashes_match"
            ),
            "candidate_field_sha256": support.get("candidate_field_sha256"),
            "support_sha256": support.get("support_sha256"),
            "support_count": support.get("support_count"),
            "selected_color_id": support.get("selected_color_id"),
            "color_cardinalities": support.get("color_cardinalities"),
            "files": dict(files),
        }
        commitment_mismatch = sorted(
            key
            for key, value in commitment_expected.items()
            if commitment.get(key) != value
        )
        if (
            commitment_mismatch
            or commitment.get("independent_support_verification_passed") is not True
            or commitment.get("support_target_input") is not False
            or commitment.get("ground_truth_accessed") is not False
        ):
            raise StageFailure(
                "support",
                "support_frame_commitment_binding",
                f"{_frame_key(expected)}:{commitment_mismatch}",
            )
        manifest_entry, _ = _load_support_sidecar(
            support_root=support_root,
            record=expected,
            pointer=frame.get("manifest_entry_evidence"),
            filename="support_manifest_entry.json",
            schema="stda_f0_support_manifest_entry_evidence_v1",
        )
        entry_expected = {
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "position": int(expected["position"]),
            "sequence": int(expected["sequence"]),
            "radar_index": int(expected["radar_index"]),
            "frame_key": _frame_key(expected),
            "partition": "train",
            "cube": dict(cube),
            "candidate": dict(candidate),
            "support": dict(support),
            "files": dict(files),
            "frame_commitment": commitment_record,
            "runtime_source_sha256": frame.get("runtime_source_sha256"),
            "critical_runtime_source_sha256": frame.get(
                "critical_runtime_source_sha256"
            ),
            "orchestrator_source_sha256": frame.get(
                "orchestrator_source_sha256"
            ),
        }
        entry_mismatch = sorted(
            key for key, value in entry_expected.items() if manifest_entry.get(key) != value
        )
        timing_boundary = manifest_entry.get("timing_boundary_contract")
        required_timing_boundary = {
            "support_frame_ns_in_final_manifest": True,
            "manifest_entry_write_fsync_rehash_inside_support_frame_ns": True,
            "cleanup_and_final_cuda_sync_inside_support_frame_ns": True,
            "elapsed_time_excluded_here_to_avoid_self_reference": True,
        }
        if (
            entry_mismatch
            or not isinstance(timing_boundary, Mapping)
            or any(
                timing_boundary.get(key) is not value
                for key, value in required_timing_boundary.items()
            )
            or manifest_entry.get("support_target_input") is not False
            or manifest_entry.get("ground_truth_accessed") is not False
        ):
            raise StageFailure(
                "support",
                "support_manifest_entry_binding",
                f"{_frame_key(expected)}:{entry_mismatch}",
            )
        _validate_child_source_evidence(
            child_name=f"support_manifest_entry:{_frame_key(expected)}",
            observed_hashes=manifest_entry.get("runtime_source_sha256"),
            expected_hashes=expected_source_hashes,
            required_paths=SUPPORT_RUNTIME_SOURCE_PATHS,
        )
        _validate_child_source_evidence(
            child_name=f"support_manifest_entry_critical:{_frame_key(expected)}",
            observed_hashes=manifest_entry.get("critical_runtime_source_sha256"),
            expected_hashes=expected_source_hashes,
            required_paths=SUPPORT_CHILD_SOURCE_PATHS,
        )
        _validate_child_source_evidence(
            child_name=f"support_manifest_entry_orchestrator:{_frame_key(expected)}",
            observed_hashes=manifest_entry.get("orchestrator_source_sha256"),
            expected_hashes=expected_source_hashes,
            required_paths=SUPPORT_ORCHESTRATOR_SOURCE_PATHS,
        )
    ledger = _canonical_record(
        support_root / "open_ledger.json",
        schema="stda_f0_support_open_ledger_v1",
    )
    if (
        ledger.get("source_commit") != source_commit
        or ledger.get("support_target_input") is not False
        or ledger.get("decision", {}).get("passed") is not True
    ):
        raise StageFailure("support", "support_open_ledger", str(ledger.get("decision")))
    _freeze_tree_readonly(support_root)
    return manifest


def _support_frame_directory(
    support_root: Path,
    support_frame: Mapping[str, Any],
) -> Path:
    files = support_frame.get("files")
    if not isinstance(files, Mapping):
        raise StageFailure("support", "support_files_absent", str(support_frame))
    row = files.get("support.bin")
    if not isinstance(row, Mapping) or not isinstance(row.get("relative_path"), str):
        raise StageFailure("support", "support_bin_absent", str(support_frame))
    path = support_root.joinpath(*PurePosixPath(row["relative_path"]).parts)
    return path.parent.resolve(strict=True)


def _load_oracle_report(
    output_dir: Path,
    *,
    expected_source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    complete = _canonical_record(
        output_dir / "ORACLE_COMPLETE.json",
        schema="stda_f0_oracle_frame_complete_v1",
    )
    report_path = output_dir / str(complete.get("oracle_report_path"))
    report = _canonical_record(
        report_path,
        schema="stda_f0_oracle_frame_v1",
    )
    if (
        report_path.stat().st_size != int(complete.get("oracle_report_bytes", -1))
        or sha256_file(report_path) != complete.get("oracle_report_sha256")
        or report.get("status") != complete.get("status")
    ):
        raise StageFailure("oracle", "oracle_completion_binding", str(output_dir))
    cpu_only = report.get("cpu_only")
    if (
        report.get("protocol_sha256") != PROTOCOL_SHA256
        or report.get("protocol_freeze_commit") != PROTOCOL_FREEZE_COMMIT
        or not isinstance(cpu_only, Mapping)
        or cpu_only.get("cuda_visible_devices") != ""
        or cpu_only.get("forbidden_cuda_modules_loaded") != []
        or cpu_only.get("geometry_evaluator_used") is not False
        or cpu_only.get("candidate_reconstruction_used") is not False
    ):
        raise StageFailure("oracle", "oracle_boundary_contract", str(output_dir))
    status = report.get("status")
    target = report.get("target")
    if status == "stda_f0_packed_support_capacity_no_go":
        stat_before = target.get("stat_before") if isinstance(target, Mapping) else None
        stat_after = target.get("stat_after") if isinstance(target, Mapping) else None
        cache_size = target.get("cache_size_bytes") if isinstance(target, Mapping) else None
        target_semantics = (
            isinstance(target, Mapping)
            and target.get("opened") is True
            and target.get("array_loader_called") is False
            and target.get("cache_arrays_read") == []
            and target.get("target_array_materialized") is False
            and target.get("path_read_calls") == 1
            and target.get("immutable_bytes_materialized") is True
            and target.get("hash_consumed_same_payload") is True
            and target.get("clock") == "time.perf_counter_ns"
            and type(target.get("read_started_perf_counter_ns")) is int
            and int(target["read_started_perf_counter_ns"]) > 0
            and _is_lower_sha256(target.get("cache_sha256"))
            and type(cache_size) is int
            and cache_size >= 0
            and isinstance(stat_before, Mapping)
            and stat_before == stat_after
            and stat_before.get("size_bytes") == cache_size
        )
    else:
        target_semantics = isinstance(target, Mapping) and target.get("opened") is True
    if not target_semantics:
        raise StageFailure("oracle", "oracle_target_boundary", str(status))
    _validate_child_source_evidence(
        child_name="oracle",
        observed_hashes=report.get("source_sha256"),
        expected_hashes=expected_source_hashes,
        required_paths=ORACLE_RUNTIME_SOURCE_PATHS,
    )
    return report


def _load_oracle_child_timing(
    output_dir: Path,
    *,
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    timing_path = output_dir / "ORACLE_FRAME_TIMING.json"
    timing = _canonical_record(
        timing_path,
        schema="stda_f0_oracle_child_timing_v2",
    )
    if (
        timing.get("protocol_sha256") != PROTOCOL_SHA256
        or timing.get("protocol_freeze_commit") != PROTOCOL_FREEZE_COMMIT
        or timing.get("status") != oracle.get("status")
        or timing.get("frame_key") != oracle.get("frame", {}).get("frame_key")
    ):
        raise StageFailure("oracle", "oracle_child_timing_binding", str(timing_path))
    started_ns = int(timing.get("oracle_frame_started_perf_counter_ns", -1))
    ended_ns = int(
        timing.get("oracle_child_pre_metric_ended_perf_counter_ns", -1)
    )
    pre_metric_ns = int(timing.get("oracle_child_pre_metric_ns", -1))
    if (
        started_ns <= 0
        or ended_ns < started_ns
        or pre_metric_ns != ended_ns - started_ns
        or timing.get("authoritative_oracle_frame_ns") is not False
        or timing.get("parent_is_sole_oracle_frame_authority") is not True
        or timing.get("support_reverification_excluded") is not True
        or timing.get("timing_receipt_publication_excluded") is not True
    ):
        raise StageFailure("oracle", "oracle_child_timing_arithmetic", str(timing))
    return {
        **timing,
        "timing_receipt_sha256": sha256_file(timing_path),
    }


def _class_metric_mapping(
    structure: Mapping[str, Any],
    *,
    target_applicability: Mapping[str, Any],
) -> dict[str, Any]:
    rows = structure.get("class_metrics")
    if not isinstance(rows, list):
        raise StageFailure("gate", "structure_class_rows", str(type(rows)))
    mapping: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("key"), str):
            raise StageFailure("gate", "structure_class_row", str(row))
        if row["key"] in mapping:
            raise StageFailure("gate", "structure_class_duplicate", str(row["key"]))
        mapping[row["key"]] = dict(row)
    expected_keys = {
        f"{prefix}_{suffix}"
        for prefix in RANGE_GATE_PREFIXES
        for suffix in RETURN_GATE_SUFFIXES
    }
    if set(target_applicability) != expected_keys:
        raise StageFailure(
            "gate",
            "target_class_applicability_keys",
            str(sorted(target_applicability)),
        )
    if mapping:
        if set(mapping) != expected_keys:
            raise StageFailure("gate", "structure_class_key_set", str(sorted(mapping)))
        observed = {
            key: row.get("applicable") is True for key, row in mapping.items()
        }
        expected = {key: value is True for key, value in target_applicability.items()}
        if observed != expected:
            raise StageFailure("gate", "structure_target_applicability", str(observed))
        return mapping
    if structure.get("structural_domain_valid") is True:
        raise StageFailure("gate", "valid_structure_has_no_classes", str(structure))
    for key in sorted(expected_keys):
        applicable = target_applicability[key] is True
        mapping[key] = {
            "key": key,
            "applicable": applicable,
            "completeness_mean_distance_m": 120.0 if applicable else None,
            "recall_1m": 0.0 if applicable else None,
            "passed": False if applicable else None,
        }
    return mapping


def _structure_gate_view(
    structure: Mapping[str, Any],
    *,
    target_applicability: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "structural_domain_valid": structure.get("structural_domain_valid") is True,
        "class_metrics": _class_metric_mapping(
            structure,
            target_applicability=target_applicability,
        ),
    }


ARM_FILE_NAMES = {
    "decision": "decision",
    "pointwise": "packed_pointwise",
    "greedy": "round_robin_greedy",
}


def _build_metric_command(
    *,
    args: argparse.Namespace,
    record: Mapping[str, Any],
    oracle_dir: Path,
    oracle: Mapping[str, Any],
    output: Path,
    gpu: Mapping[str, Any],
) -> list[str]:
    target = oracle_dir / "fit_evidence/target_xyz_confidence.npy"
    fit_binding = oracle.get("fit_binding")
    if not isinstance(fit_binding, Mapping):
        raise StageFailure("metric", "fit_binding_absent", _frame_key(record))
    fit_hashes = fit_binding.get("fit_evidence_files_sha256")
    if not isinstance(fit_hashes, Mapping):
        raise StageFailure("metric", "fit_hashes_absent", _frame_key(record))
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(args.repo / "code/scripts/stda_f0_metric_phase.py"),
        "--frame-key",
        _frame_key(record),
        "--source-commit",
        args.source_commit,
        "--target",
        str(target),
        "--target-sha256",
        str(fit_hashes.get("target_xyz_confidence.npy")),
    ]
    exports = oracle.get("exports")
    if not isinstance(exports, Mapping):
        raise StageFailure("metric", "exports_absent", _frame_key(record))
    for alias, oracle_name in ARM_FILE_NAMES.items():
        export = exports.get(oracle_name)
        if not isinstance(export, Mapping):
            raise StageFailure("metric", "export_absent", f"{_frame_key(record)}:{alias}")
        npy = export.get("npy")
        if not isinstance(npy, Mapping):
            raise StageFailure("metric", "export_npy_absent", alias)
        command.extend(
            (
                f"--{alias}-export",
                str(oracle_dir / "exports" / f"{oracle_name}.npy"),
                f"--{alias}-sha256",
                str(npy.get("sha256")),
            )
        )
    command.extend(
        (
            "--expected-gpu-uuid",
            str(gpu["uuid"]),
            "--expected-gpu-pci",
            str(gpu["pci_bus_id"]),
            "--expected-gpu-name",
            str(gpu["name"]),
            "--output",
            str(output),
        )
    )
    return command


def _normalize_scientific_frame(
    *,
    record: Mapping[str, Any],
    oracle: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
    metric: Mapping[str, Any] | None,
    oracle_frame_ns: int,
    oracle_child_pre_metric_ns: int,
    oracle_parent_pre_target_ns: int,
    oracle_parent_child_wall_ns: int,
    independent_verifier_wall_ns: int,
    independent_verifier_serialization_ns: int,
    metric_wall_ns: int,
    source_execution_locks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    support = oracle.get("support")
    if not isinstance(support, Mapping):
        raise StageFailure("frame", "oracle_support_absent", _frame_key(record))
    capacity = support.get("capacity_status")
    if not isinstance(capacity, Mapping):
        raise StageFailure("frame", "support_capacity_absent", _frame_key(record))
    measured_lower_bound_ns = (
        int(oracle_child_pre_metric_ns)
        + int(independent_verifier_wall_ns)
        + int(independent_verifier_serialization_ns)
        + int(metric_wall_ns)
    )
    if int(oracle_frame_ns) < measured_lower_bound_ns:
        raise StageFailure(
            "implementation",
            "oracle_frame_timing_under_count",
            (
                f"{_frame_key(record)}: frame={oracle_frame_ns}, "
                f"measured_lower_bound={measured_lower_bound_ns}"
            ),
        )
    normalized: dict[str, Any] = {
        "position": int(record["position"]),
        "sequence": int(record["sequence"]),
        "radar_index": int(record["radar_index"]),
        "frame_key": _frame_key(record),
        "oracle_status": oracle.get("status"),
        "source_execution_locks": [dict(row) for row in source_execution_locks],
        "support": {
            "capacity_sufficient": capacity.get("capacity_sufficient") is True,
            "support_count": int(support.get("support_count", -1)),
            "support_sha256": support.get("sha256"),
        },
        "graph": {
            "maximum_cardinality": int(
                oracle.get("cardinality", {}).get("transported_mass", 0)
            ),
        },
        "arms": {},
        "resources": {
            "oracle_report_allocation_ns": int(
                oracle.get("timings_ns", {}).get("allocation_core_ns", 0)
            ),
            "independent_verifier_wall_ns": int(independent_verifier_wall_ns),
            "independent_verifier_serialization_ns": int(
                independent_verifier_serialization_ns
            ),
            "oracle_allocation_ns": (
                int(oracle.get("timings_ns", {}).get("allocation_core_ns", 0))
                + int(independent_verifier_wall_ns)
                + int(independent_verifier_serialization_ns)
            ),
            "oracle_frame_ns": int(oracle_frame_ns),
            "oracle_child_pre_metric_ns": int(oracle_child_pre_metric_ns),
            "oracle_parent_pre_target_ns_excluded": int(
                oracle_parent_pre_target_ns
            ),
            "oracle_parent_child_wall_ns_reported": int(oracle_parent_child_wall_ns),
            "metric_wall_ns": int(metric_wall_ns),
            "oracle_frame_measured_lower_bound_ns": measured_lower_bound_ns,
            "oracle_frame_timing_lower_bound_valid": True,
            "torch_peak_allocated_bytes": 0,
            "torch_peak_reserved_bytes": 0,
        },
    }
    if oracle.get("status") != "stda_f0_oracle_ready_for_cuda_metrics":
        return normalized
    if verification is None or metric is None:
        raise StageFailure("frame", "full_frame_evidence_absent", _frame_key(record))
    if verification.get("passed") is not True:
        raise StageFailure("verify", "independent_frame_replay", _frame_key(record))
    metric_rows = metric.get("metrics")
    verification_arms = verification.get("arms")
    if not isinstance(metric_rows, Mapping) or not isinstance(verification_arms, Mapping):
        raise StageFailure("frame", "arm_reports_absent", _frame_key(record))
    vectors: dict[str, dict[str, dict[str, bool]]] = {}
    arms: dict[str, Any] = {}
    for alias in ARM_FILE_NAMES:
        metric_row = metric_rows.get(alias)
        verified_arm = verification_arms.get(alias)
        if not isinstance(metric_row, Mapping) or not isinstance(verified_arm, Mapping):
            raise StageFailure("frame", "arm_report_absent", f"{_frame_key(record)}:{alias}")
        spacing = verified_arm.get("spacing")
        structure = verified_arm.get("structural_replay", {}).get("computed_report")
        target_applicability = verified_arm.get("target_class_applicability")
        if (
            not isinstance(spacing, Mapping)
            or not isinstance(structure, Mapping)
            or not isinstance(target_applicability, Mapping)
        ):
            raise StageFailure("frame", "arm_replay_absent", f"{_frame_key(record)}:{alias}")
        export_view = {
            "point_count": int(spacing.get("point_count", -1)),
            "finite_xyz": spacing.get("finite_xyz") is True,
            "unique_xyz_bytes": spacing.get("unique_xyz_bytes") is True,
            "spacing": dict(spacing),
        }
        vector = build_shared_gate_vector(
            export_view,
            metric_row,
            _structure_gate_view(
                structure,
                target_applicability=target_applicability,
            ),
        )
        vectors[alias] = vector
        arms[alias] = {
            "passed": gate_vector_passed(vector),
            "gate_vector": vector,
            "export": export_view,
            "metric": dict(metric_row),
            "structure": _structure_gate_view(
                structure,
                target_applicability=target_applicability,
            ),
            "independent_replay_passed": verified_arm.get("passed") is True,
        }
    validate_cross_arm_applicability(vectors)
    normalized["arms"] = arms
    runtime = metric.get("runtime", {})
    normalized["resources"].update(
        {
            "torch_peak_allocated_bytes": int(
                runtime.get("cuda_peak_allocated_bytes", 0)
            ),
            "torch_peak_reserved_bytes": int(
                runtime.get("cuda_peak_reserved_bytes", 0)
            ),
            "metric_ns": int(runtime.get("metric_ns", 0)),
        }
    )
    return normalized


def _remaining_timeout_seconds(
    transaction_started_ns: int,
    requested_seconds: float,
) -> float:
    elapsed = time.perf_counter_ns() - transaction_started_ns
    remaining_ns = MAX_TRANSACTION_NS - elapsed
    if remaining_ns <= 0:
        raise StageFailure("resource", "transaction_timeout", str(elapsed))
    return max(0.1, min(float(requested_seconds), remaining_ns / 1_000_000_000))


def _run_support_child(
    *,
    args: argparse.Namespace,
    staging: Path,
    monitor: ProcessTreeMonitor,
    transaction_started_ns: int | None,
    transaction_clock: dict[str, int] | None,
    records: Sequence[Mapping[str, Any]],
    expected_source_hashes: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    support_root = staging / "support"
    support_root.mkdir(parents=False, exist_ok=False)
    fsync_directory(staging)
    log_path = staging / "logs/support_child.json"
    command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(args.repo / "code/scripts/stda_f0_support_phase.py"),
        "--data-root",
        str(args.data_root),
        "--manifest",
        str(args.manifest),
        "--scene-split",
        str(args.scene_split),
        "--normalization",
        str(args.normalization),
        "--checkpoint",
        str(args.checkpoint),
        "--candidate-hash-manifest",
        str(args.candidate_hash_manifest),
        "--repo",
        str(args.repo),
        "--output-root",
        str(support_root),
        "--source-commit",
        args.source_commit,
        "--device",
        "cuda:0",
    )
    event("support_started", output_root=str(support_root))
    source_locks = [
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=expected_source_hashes,
            label="before_support_child",
        )
    ]
    if transaction_started_ns is None:
        transaction_started_ns = time.perf_counter_ns()
        if transaction_clock is not None:
            transaction_clock["started_ns"] = transaction_started_ns
    run_child(
        command,
        environment=child_environment(
            repo=args.repo,
            physical_gpu=args.gpu_uuid,
            phase="support",
        ),
        log_path=log_path,
        phase="support",
        cuda_visible=True,
        monitor=monitor,
        timeout_seconds=_remaining_timeout_seconds(
            transaction_started_ns,
            args.support_timeout_seconds,
        ),
    )
    source_locks.append(
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=expected_source_hashes,
            label="after_support_child",
        )
    )
    manifest = _validate_support_manifest(
        support_root,
        records,
        source_commit=args.source_commit,
        expected_source_hashes=expected_source_hashes,
    )
    event(
        "support_completed",
        support_manifest_sha256=sha256_file(support_root / "support_manifest.json"),
        source_execution_locks=source_locks,
    )
    return support_root, manifest


def _verify_phase_command(
    *,
    args: argparse.Namespace,
    support_frame_dir: Path,
    oracle_dir: Path,
    target_cache: Path,
    expected_support_sha256: str,
    output: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(args.repo / "code/scripts/stda_f0_verify_phase.py"),
        "--support-frame-dir",
        str(support_frame_dir),
        "--oracle-dir",
        str(oracle_dir),
        "--target-cache",
        str(target_cache),
        "--expected-support-sha256",
        expected_support_sha256,
        "--output",
        str(output),
    )


def _run_one_oracle_frame(
    *,
    args: argparse.Namespace,
    staging: Path,
    support_root: Path,
    support_frame: Mapping[str, Any],
    record: Mapping[str, Any],
    monitor: ProcessTreeMonitor,
    transaction_started_ns: int,
    gpu: Mapping[str, Any],
    expected_source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    frame_root = staging / "frames" / _frame_directory_name(record)
    frame_root.mkdir(parents=True, exist_ok=False)
    fsync_directory(frame_root.parent)
    oracle_dir = frame_root / "oracle"
    support_frame_dir = _support_frame_directory(support_root, support_frame)
    expected_support_sha = str(support_frame["support"]["support_sha256"])
    target_cache = _target_cache_path(args.cache_root, record)
    source_execution_locks: list[dict[str, Any]] = []
    source_execution_locks.append(
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=expected_source_hashes,
            label=f"{_frame_key(record)}:before_oracle_child",
        )
    )
    oracle_parent_spawn_started_ns = time.perf_counter_ns()
    phase_suffix = f"{int(record['position']):02d}"
    oracle_command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(args.repo / "code/scripts/stda_f0_oracle_phase.py"),
        "--support-frame-dir",
        str(support_frame_dir),
        "--target-cache",
        str(target_cache),
        "--resources",
        str(args.data_root / "resources"),
        "--output-dir",
        str(oracle_dir),
        "--sequence",
        str(record["sequence"]),
        "--radar-index",
        str(record["radar_index"]),
        "--expected-support-sha256",
        expected_support_sha,
        "--replay-script",
        str(args.repo / "code/scripts/stda_f0_assignment_replay.py"),
        "--python-executable",
        sys.executable,
    )
    event("oracle_frame_started", frame_key=_frame_key(record))
    run_child(
        oracle_command,
        environment=child_environment(
            repo=args.repo,
            physical_gpu=None,
            phase=f"oracle_{phase_suffix}",
        ),
        log_path=frame_root / "oracle_child.log",
        phase=f"oracle_{phase_suffix}",
        cuda_visible=False,
        monitor=monitor,
        timeout_seconds=_remaining_timeout_seconds(
            transaction_started_ns,
            args.frame_child_timeout_seconds,
        ),
    )
    oracle_parent_child_completed_ns = time.perf_counter_ns()
    source_execution_locks.append(
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=expected_source_hashes,
            label=f"{_frame_key(record)}:after_oracle_child",
        )
    )
    oracle = _load_oracle_report(
        oracle_dir,
        expected_source_hashes=expected_source_hashes,
    )
    if oracle.get("frame", {}).get("frame_key") != _frame_key(record):
        raise StageFailure("oracle", "oracle_frame_identity", _frame_key(record))
    oracle_child_timing = _load_oracle_child_timing(oracle_dir, oracle=oracle)
    oracle_child_started_ns = int(
        oracle_child_timing["oracle_frame_started_perf_counter_ns"]
    )
    oracle_child_pre_metric_ended_ns = int(
        oracle_child_timing[
            "oracle_child_pre_metric_ended_perf_counter_ns"
        ]
    )
    if (
        oracle_child_started_ns < oracle_parent_spawn_started_ns
        or oracle_child_pre_metric_ended_ns > oracle_parent_child_completed_ns
    ):
        raise StageFailure(
            "implementation",
            "oracle_child_parent_clock_boundary",
            str(oracle_child_timing),
        )
    oracle_parent_pre_target_ns = (
        oracle_child_started_ns - oracle_parent_spawn_started_ns
    )
    oracle_parent_child_wall_ns = (
        oracle_parent_child_completed_ns - oracle_parent_spawn_started_ns
    )

    verification_path = frame_root / "independent_verification.json"
    source_execution_locks.append(
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=expected_source_hashes,
            label=f"{_frame_key(record)}:before_verify_child",
        )
    )
    verifier_started_ns = time.perf_counter_ns()
    verification_log_path = frame_root / "independent_verification.log"
    run_child(
        _verify_phase_command(
            args=args,
            support_frame_dir=support_frame_dir,
            oracle_dir=oracle_dir,
            target_cache=target_cache,
            expected_support_sha256=expected_support_sha,
            output=verification_path,
        ),
        environment=child_environment(
            repo=args.repo,
            physical_gpu=None,
            phase=f"verify_{phase_suffix}",
        ),
        log_path=verification_log_path,
        phase=f"verify_{phase_suffix}",
        cuda_visible=False,
        monitor=monitor,
        timeout_seconds=_remaining_timeout_seconds(
            transaction_started_ns,
            args.frame_child_timeout_seconds,
        ),
    )
    verifier_child_completed_ns = time.perf_counter_ns()
    source_execution_locks.append(
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=expected_source_hashes,
            label=f"{_frame_key(record)}:after_verify_child",
        )
    )
    verifier_serialization_started_ns = time.perf_counter_ns()
    verification = _canonical_record(
        verification_path,
        schema="stda_f0_independent_frame_verification_v1",
    )
    if (
        verification.get("passed") is not True
        or verification.get("protocol_sha256") != PROTOCOL_SHA256
        or verification.get("protocol_freeze_commit") != PROTOCOL_FREEZE_COMMIT
        or verification.get("frame", {}).get("frame_key") != _frame_key(record)
        or verification.get("oracle_status") != oracle.get("status")
    ):
        raise StageFailure("verify", "independent_frame_replay", _frame_key(record))
    verifier_parent_timing = _load_verifier_parent_timing_evidence(
        verification_path=verification_path,
        log_path=verification_log_path,
        verification=verification,
        parent_started_ns=verifier_started_ns,
        parent_completed_ns=verifier_child_completed_ns,
    )
    _validate_child_source_evidence(
        child_name="verify",
        observed_hashes=verification.get("source_sha256"),
        expected_hashes=expected_source_hashes,
        required_paths=VERIFY_CHILD_SOURCE_PATHS,
    )
    independent_verifier_serialization_ns = (
        time.perf_counter_ns() - verifier_serialization_started_ns
    )
    independent_verifier_wall_ns = (
        verifier_child_completed_ns - verifier_started_ns
    )

    metric: dict[str, Any] | None = None
    metric_wall_ns = 0
    if oracle.get("status") == "stda_f0_oracle_ready_for_cuda_metrics":
        metric_path = frame_root / "cuda_metrics.json"
        source_execution_locks.append(
            _assert_source_lock(
                repo=args.repo,
                source_commit=args.source_commit,
                expected_hashes=expected_source_hashes,
                label=f"{_frame_key(record)}:before_metric_child",
            )
        )
        metric_started_ns = time.perf_counter_ns()
        run_child(
            _build_metric_command(
                args=args,
                record=record,
                oracle_dir=oracle_dir,
                oracle=oracle,
                output=metric_path,
                gpu=gpu,
            ),
            environment=child_environment(
                repo=args.repo,
                physical_gpu=str(gpu["uuid"]),
                phase=f"metric_{phase_suffix}",
            ),
            log_path=frame_root / "cuda_metric_child.log",
            phase=f"metric_{phase_suffix}",
            cuda_visible=True,
            monitor=monitor,
            timeout_seconds=_remaining_timeout_seconds(
                transaction_started_ns,
                args.frame_child_timeout_seconds,
            ),
        )
        metric_wall_ns = time.perf_counter_ns() - metric_started_ns
        source_execution_locks.append(
            _assert_source_lock(
                repo=args.repo,
                source_commit=args.source_commit,
                expected_hashes=expected_source_hashes,
                label=f"{_frame_key(record)}:after_metric_child",
            )
        )
        metric = _canonical_record(
            metric_path,
            schema="stda_f0_metric_phase_v1",
        )
        if (
            metric.get("frame_key") != _frame_key(record)
            or metric.get("source_commit") != args.source_commit
            or metric.get("protocol_sha256") != PROTOCOL_SHA256
        ):
            raise StageFailure("metric", "metric_binding", _frame_key(record))

    provisional_boundary_ns = time.perf_counter_ns()
    provisional = _normalize_scientific_frame(
        record=record,
        oracle=oracle,
        verification=verification,
        metric=metric,
        oracle_frame_ns=provisional_boundary_ns - oracle_child_started_ns,
        oracle_child_pre_metric_ns=int(
            oracle_child_timing["oracle_child_pre_metric_ns"]
        ),
        oracle_parent_pre_target_ns=oracle_parent_pre_target_ns,
        oracle_parent_child_wall_ns=oracle_parent_child_wall_ns,
        independent_verifier_wall_ns=independent_verifier_wall_ns,
        independent_verifier_serialization_ns=(
            independent_verifier_serialization_ns
        ),
        metric_wall_ns=metric_wall_ns,
        source_execution_locks=source_execution_locks,
    )
    science_document = dict(provisional)
    science_document.pop("resources", None)
    science_document.update(
        {
            "schema": "stda_f0_frame_science_v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "source_commit": args.source_commit,
            "oracle_report_sha256": sha256_file(oracle_dir / "oracle_report.json"),
            "oracle_child_timing_sha256": oracle_child_timing[
                "timing_receipt_sha256"
            ],
            "independent_verification_sha256": sha256_file(verification_path),
            "independent_verification_log_sha256": verifier_parent_timing[
                "completion_receipt_sha256"
            ],
            "cuda_metric_sha256": (
                None if metric is None else sha256_file(frame_root / "cuda_metrics.json")
            ),
        }
    )
    atomic_write_json(frame_root / "frame_science.json", science_document)
    if sha256_file(frame_root / "frame_science.json") != sha256_bytes(
        canonical_json_bytes(science_document)
    ):
        raise StageFailure("frame", "frame_science_rehash", _frame_key(record))
    fsync_directory(frame_root)
    timing_started_ns = time.perf_counter_ns()
    timing_document = {
        "schema": "stda_f0_oracle_frame_timing_v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": args.source_commit,
        "frame_key": _frame_key(record),
        "oracle_frame_started_perf_counter_ns": oracle_child_started_ns,
        "oracle_child_pre_metric_ended_perf_counter_ns": (
            oracle_child_pre_metric_ended_ns
        ),
        "oracle_child_pre_metric_ns": int(
            oracle_child_timing["oracle_child_pre_metric_ns"]
        ),
        "oracle_child_timing_sha256": oracle_child_timing[
            "timing_receipt_sha256"
        ],
        "oracle_parent_pre_target_ns_excluded": oracle_parent_pre_target_ns,
        "oracle_parent_child_wall_ns_reported": oracle_parent_child_wall_ns,
        "independent_verifier_wall_ns": independent_verifier_wall_ns,
        "independent_verifier_serialization_ns": (
            independent_verifier_serialization_ns
        ),
        "independent_verifier_parent_timing": verifier_parent_timing,
        "metric_wall_ns": metric_wall_ns,
        "boundary": (
            "oracle_frame_ns is derived after this receipt is fsynced and fully "
            "rehashed as evidence_boundary_perf_counter_ns minus the child target "
            "first-byte-read counter; the final counter and derived duration are "
            "recorded in the enclosing resource report to avoid self-reference"
        ),
        "receipt_started_ns": timing_started_ns,
        "receipt_publication_included_in_oracle_frame_ns": True,
        "parent_spawn_and_pre_target_excluded": True,
        "self_referential_end_fields_omitted": True,
    }
    timing_path = frame_root / "oracle_frame_timing.json"
    atomic_write_json(timing_path, timing_document)
    fsync_directory(frame_root)
    replayed_timing = _canonical_record(
        timing_path,
        schema="stda_f0_oracle_frame_timing_v2",
    )
    timing_receipt_sha256 = sha256_file(timing_path)
    if (
        replayed_timing != timing_document
        or timing_receipt_sha256 != sha256_bytes(
            canonical_json_bytes(timing_document)
        )
    ):
        raise StageFailure("frame", "oracle_frame_timing_rehash", _frame_key(record))
    evidence_boundary_ns = time.perf_counter_ns()
    timing_receipt_publication_ns = evidence_boundary_ns - timing_started_ns
    oracle_frame_ns = evidence_boundary_ns - oracle_child_started_ns
    normalized = _normalize_scientific_frame(
        record=record,
        oracle=oracle,
        verification=verification,
        metric=metric,
        oracle_frame_ns=oracle_frame_ns,
        oracle_child_pre_metric_ns=int(
            oracle_child_timing["oracle_child_pre_metric_ns"]
        ),
        oracle_parent_pre_target_ns=oracle_parent_pre_target_ns,
        oracle_parent_child_wall_ns=oracle_parent_child_wall_ns,
        independent_verifier_wall_ns=independent_verifier_wall_ns,
        independent_verifier_serialization_ns=(
            independent_verifier_serialization_ns
        ),
        metric_wall_ns=metric_wall_ns,
        source_execution_locks=source_execution_locks,
    )
    normalized["resources"].update(
        {
            "oracle_evidence_boundary_perf_counter_ns": evidence_boundary_ns,
            "oracle_frame_timing_receipt_sha256": timing_receipt_sha256,
            "timing_receipt_publication_ns": timing_receipt_publication_ns,
            "timing_receipt_publication_included": True,
        }
    )
    event(
        "oracle_frame_completed",
        frame_key=_frame_key(record),
        oracle_status=oracle.get("status"),
        oracle_frame_ns=oracle_frame_ns,
    )
    return normalized


def _run_all_oracle_frames(
    *,
    args: argparse.Namespace,
    staging: Path,
    support_root: Path,
    support_manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    monitor: ProcessTreeMonitor,
    transaction_started_ns: int,
    gpu: Mapping[str, Any],
    expected_source_hashes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    support_frames = support_manifest.get("frames")
    if not isinstance(support_frames, list):
        raise StageFailure("oracle", "support_frames_absent", "manifest")
    frames: list[dict[str, Any]] = []
    for support_frame, record in zip(support_frames, records, strict=True):
        frames.append(
            _run_one_oracle_frame(
                args=args,
                staging=staging,
                support_root=support_root,
                support_frame=support_frame,
                record=record,
                monitor=monitor,
                transaction_started_ns=transaction_started_ns,
                gpu=gpu,
                expected_source_hashes=expected_source_hashes,
            )
        )
    if len(frames) != EXPECTED_TRAIN_FRAMES:
        raise StageFailure("oracle", "oracle_frame_count", str(len(frames)))
    return frames


def _prepare_shared_staging(
    *,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    launch_marker: Path,
) -> Path:
    staging = Path(
        tempfile.mkdtemp(
            prefix=(
                f".stda_f0_bundle_staging_{args.source_commit[:8]}_"
                f"{PROTOCOL_SHA256[:8]}_"
            ),
            dir=args.shared_evidence_parent,
        )
    )
    for name in ("global", "frozen", "tools", "logs", "frames"):
        (staging / name).mkdir(parents=False, exist_ok=False)
    atomic_write_json(staging / "global/preflight.json", dict(preflight))
    atomic_write_json(
        staging / "global/source_binding.json",
        {
            "schema": "stda_f0_source_binding_v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "source_commit": args.source_commit,
            "source_hashes": dict(preflight["source_hashes"]),
            "input_hashes": dict(preflight["input_hashes"]),
        },
    )
    _copy_regular_file(
        args.repo / "docs/stda_f0_sparse_target_demand_assignment_protocol.md",
        staging / "frozen/stda_f0_sparse_target_demand_assignment_protocol.md",
    )
    _copy_regular_file(
        args.repo / "artifacts/idea/stda_f0_freeze_record.json",
        staging / "frozen/stda_f0_freeze_record.json",
    )
    _copy_regular_file(
        args.repo / "artifacts/idea/stda_f0_prefreeze_audit_round6.md",
        staging / "frozen/stda_f0_prefreeze_audit_round6.md",
    )
    _copy_regular_file(
        launch_marker,
        staging / "global/formal_launch.json",
    )
    _copy_regular_file(
        args.repo / "code/scripts/stda_f0_bundle_verify.py",
        staging / "tools/stda_f0_bundle_verify.py",
    )
    _fsync_tree(staging)
    return staging


def _write_global_scientific_evidence(
    *,
    staging: Path,
    args: argparse.Namespace,
    support_manifest: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frame_index = {
        "schema": "stda_f0_frame_index_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "source_commit": args.source_commit,
        "frame_count": len(frames),
        "frames": [dict(frame) for frame in frames],
    }
    atomic_write_json(staging / "global/frame_index.json", frame_index)
    summary = {
        "schema": "stda_f0_scientific_evidence_summary_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": args.source_commit,
        "frame_count": len(frames),
        "support_manifest_sha256": sha256_file(
            staging / "support/support_manifest.json"
        ),
        "support_open_ledger_sha256": sha256_file(
            staging / "support/open_ledger.json"
        ),
        "oracle_status_counts": {
            status: sum(frame["oracle_status"] == status for frame in frames)
            for status in sorted({str(frame["oracle_status"]) for frame in frames})
        },
        "support_capacity_sufficient_all": all(
            frame["support"]["capacity_sufficient"] is True for frame in frames
        ),
        "full_graph_all": all(
            int(frame["graph"]["maximum_cardinality"]) == 10_000
            for frame in frames
        ),
        "all_available_arm_replays_passed": all(
            arm["independent_replay_passed"] is True
            for frame in frames
            for arm in frame["arms"].values()
        ),
        "all_cross_arm_applicability_checked": True,
        "scientific_status_pending_resource_and_publication_checks": True,
        "bundle_complete_claims_no_terminal_status": True,
        "support_summary": dict(support_manifest.get("summary", {})),
    }
    atomic_write_json(staging / "global/scientific_summary.json", summary)
    _fsync_tree(staging)
    return summary


def _verification_consensus(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    timestamps_ns: Mapping[str, int],
) -> dict[str, Any]:
    if not reports:
        raise StageFailure("bundle_verify", "no_verification_reports", "")
    first = next(iter(reports.values()))
    fields = (
        "payload_count",
        "payload_bytes",
        "complete_bundle_bytes",
        "bundle_root",
        "payload_manifest_sha256",
        "bundle_complete_sha256",
    )
    for name, report in reports.items():
        if report.get("passed") is not True:
            raise StageFailure("bundle_verify", "verifier_not_passed", name)
        differences = {
            field: (first.get(field), report.get(field))
            for field in fields
            if report.get(field) != first.get(field)
        }
        if differences:
            raise StageFailure("bundle_verify", "verifier_disagreement", str(differences))
    return {
        **{field: first[field] for field in fields},
        "timestamps_ns": dict(timestamps_ns),
        "reports": {name: dict(report) for name, report in reports.items()},
    }


def _publish_compact_result(
    *,
    args: argparse.Namespace,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    payload = canonical_json_bytes(dict(result))
    digest = sha256_bytes(payload)
    filename = (
        f"stda_f0_compact_{args.source_commit[:8]}_{PROTOCOL_SHA256[:8]}_"
        f"{digest[:16]}.json"
    )
    paths = (
        args.shared_evidence_parent / filename,
        args.local_evidence_parent / filename,
    )
    for path in paths:
        exclusive_write_bytes(path, payload)
    return {
        "sha256": digest,
        "size": len(payload),
        "paths": [str(path.resolve()) for path in paths],
    }


def _stop_transaction_monitor(
    monitor: ProcessTreeMonitor,
    transaction_started_ns: int,
) -> tuple[dict[str, Any], int, int]:
    report = monitor.stop()
    boundary_ns = time.perf_counter_ns()
    return report, boundary_ns - transaction_started_ns, boundary_ns


def _run_source_locked_child(
    *,
    args: argparse.Namespace,
    expected_source_hashes: Mapping[str, Any],
    label: str,
    operation: Callable[[], Any],
) -> Any:
    before = _assert_source_lock(
        repo=args.repo,
        source_commit=args.source_commit,
        expected_hashes=expected_source_hashes,
        label=f"before_{label}",
    )
    try:
        result = operation()
    finally:
        after = _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=expected_source_hashes,
            label=f"after_{label}",
        )
    event(
        "source_locked_child_completed",
        child=label,
        before=before,
        after=after,
    )
    return result


def _classify_execution_failure(
    error: BaseException,
    *,
    monitor_report: Mapping[str, Any] | None,
    resources: Mapping[str, Any] | None,
) -> str:
    if resources is not None and resources.get("implementation_valid") is False:
        return TERMINAL_STATUSES[0]
    if monitor_report is not None and not all(
        _monitor_implementation_checks(monitor_report).values()
    ):
        return TERMINAL_STATUSES[0]
    if isinstance(error, StageFailure) and error.code in RESOURCE_FAILURE_CODES:
        return TERMINAL_STATUSES[1]
    return TERMINAL_STATUSES[0]


def run_formal(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    global _EVENT_LOG_PATH

    if preflight.get("mandatory_synthetic_tests", {}).get("passed") is not True:
        raise StageFailure(
            "preflight",
            "mandatory_test_lock_absent",
            "formal launch requires a passed SHA-bound mandatory test report",
        )
    _assert_source_lock(
        repo=args.repo,
        source_commit=args.source_commit,
        expected_hashes=preflight["source_hashes"],
        label="before_formal_launch_commit",
    )
    marker, launch = _create_formal_launch_marker(args, preflight)
    _EVENT_LOG_PATH = Path(str(launch["log_path"]))
    event(
        "formal_launch_committed",
        launch_marker=str(marker),
        source_commit=args.source_commit,
    )
    staging: Path | None = None
    shared_final: Path | None = None
    local_final: Path | None = None
    monitor: ProcessTreeMonitor | None = None
    monitor_report: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []
    support_manifest: dict[str, Any] | None = None
    transaction_started_ns: int | None = None
    transaction_clock: dict[str, int] = {}
    transaction_work_ns: int | None = None
    bundle_root: str | None = None
    verification: dict[str, Any] | None = None
    transaction: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    try:
        staging = _prepare_shared_staging(
            args=args,
            preflight=preflight,
            launch_marker=marker,
        )
        monitor = ProcessTreeMonitor(os.getpid(), str(preflight["gpu"]["uuid"]))
        monitor.start()
        support_root, support_manifest = _run_support_child(
            args=args,
            staging=staging,
            monitor=monitor,
            transaction_started_ns=None,
            transaction_clock=transaction_clock,
            records=preflight["cohort"],
            expected_source_hashes=preflight["source_hashes"],
        )
        transaction_started_ns = transaction_clock["started_ns"]
        frames = _run_all_oracle_frames(
            args=args,
            staging=staging,
            support_root=support_root,
            support_manifest=support_manifest,
            records=preflight["cohort"],
            monitor=monitor,
            transaction_started_ns=transaction_started_ns,
            gpu=preflight["gpu"],
            expected_source_hashes=preflight["source_hashes"],
        )
        _write_global_scientific_evidence(
            staging=staging,
            args=args,
            support_manifest=support_manifest,
            frames=frames,
        )
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=preflight["source_hashes"],
            label="before_bundle_seal",
        )
        sealed = seal_bundle(staging, source_commit=args.source_commit)
        bundle_root = str(sealed["bundle_root"])
        cpu_environment = child_environment(
            repo=args.repo,
            physical_gpu=None,
            phase="bundle_verify",
        )
        verification_reports: dict[str, Mapping[str, Any]] = {}
        verification_timestamps: dict[str, int] = {}
        verification_reports["shared_staging_h200"] = _run_source_locked_child(
            args=args,
            expected_source_hashes=preflight["source_hashes"],
            label="shared_staging_h200_bundle_verifier",
            operation=lambda: run_local_bundle_verifier(
                staging / "tools/stda_f0_bundle_verify.py",
                staging,
                environment=cpu_environment,
            ),
        )
        verification_timestamps["shared_staging_h200"] = time.time_ns()
        if verification_reports["shared_staging_h200"]["bundle_root"] != bundle_root:
            raise StageFailure("bundle_verify", "sealed_root_mismatch", bundle_root)

        local_staging = Path(
            tempfile.mkdtemp(
                prefix=(
                    f".stda_f0_bundle_staging_{args.source_commit[:8]}_"
                    f"{PROTOCOL_SHA256[:8]}_"
                ),
                dir=args.local_evidence_parent,
            )
        )
        local_staging.rmdir()
        copy_bundle(staging, local_staging)
        verification_reports["local_staging_h200"] = _run_source_locked_child(
            args=args,
            expected_source_hashes=preflight["source_hashes"],
            label="local_staging_h200_bundle_verifier",
            operation=lambda: run_local_bundle_verifier(
                local_staging / "tools/stda_f0_bundle_verify.py",
                local_staging,
                environment=cpu_environment,
            ),
        )
        verification_timestamps["local_staging_h200"] = time.time_ns()
        verification_reports["shared_staging_l40s"] = _run_source_locked_child(
            args=args,
            expected_source_hashes=preflight["source_hashes"],
            label="shared_staging_l40s_bundle_verifier",
            operation=lambda: run_l40s_bundle_verifier(
                staging,
                shared_mount_root=args.shared_mount_root,
                l40s_mount_root=args.l40s_mount_root,
                l40s_host=args.l40s_host,
            ),
        )
        verification_timestamps["shared_staging_l40s"] = time.time_ns()
        _verification_consensus(
            verification_reports,
            timestamps_ns=verification_timestamps,
        )
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=preflight["source_hashes"],
            label="before_dual_bundle_rename",
        )

        final_basename = (
            f"stda_f0_{args.source_commit[:8]}_{PROTOCOL_SHA256[:8]}_"
            f"{bundle_root[:16]}"
        )
        shared_final = args.shared_evidence_parent / final_basename
        local_final = args.local_evidence_parent / final_basename
        rename_bundle(staging, shared_final)
        staging = None
        rename_bundle(local_staging, local_final)
        verification_reports["shared_final_h200"] = _run_source_locked_child(
            args=args,
            expected_source_hashes=preflight["source_hashes"],
            label="shared_final_h200_bundle_verifier",
            operation=lambda: run_local_bundle_verifier(
                shared_final / "tools/stda_f0_bundle_verify.py",
                shared_final,
                environment=cpu_environment,
            ),
        )
        verification_timestamps["shared_final_h200"] = time.time_ns()
        verification_reports["local_final_h200"] = _run_source_locked_child(
            args=args,
            expected_source_hashes=preflight["source_hashes"],
            label="local_final_h200_bundle_verifier",
            operation=lambda: run_local_bundle_verifier(
                local_final / "tools/stda_f0_bundle_verify.py",
                local_final,
                environment=cpu_environment,
            ),
        )
        verification_timestamps["local_final_h200"] = time.time_ns()
        verification_reports["shared_final_l40s"] = _run_source_locked_child(
            args=args,
            expected_source_hashes=preflight["source_hashes"],
            label="shared_final_l40s_bundle_verifier",
            operation=lambda: run_l40s_bundle_verifier(
                shared_final,
                shared_mount_root=args.shared_mount_root,
                l40s_mount_root=args.l40s_mount_root,
                l40s_host=args.l40s_host,
            ),
        )
        verification_timestamps["shared_final_l40s"] = time.time_ns()
        verification = _verification_consensus(
            verification_reports,
            timestamps_ns=verification_timestamps,
        )
        _assert_source_lock(
            repo=args.repo,
            source_commit=args.source_commit,
            expected_hashes=preflight["source_hashes"],
            label="after_final_bundle_reverification",
        )
        monitor_report, transaction_work_ns, record_started_ns = (
            _stop_transaction_monitor(monitor, transaction_started_ns)
        )
        monitor = None
        resources = summarize_resources(
            monitor_report=monitor_report,
            support_manifest=support_manifest,
            frames=frames,
            transaction_work_ns=transaction_work_ns,
        )
        if not resources["implementation_valid"]:
            terminal_status = TERMINAL_STATUSES[0]
            failed_stage = "implementation"
            failed_code = "monitor_or_timing_contract"
        elif not resources["resource_valid"]:
            terminal_status = TERMINAL_STATUSES[1]
            failed_stage = "resource"
            failed_code = "frozen_resource_ceiling"
        else:
            terminal_status = determine_scientific_status(frames)
            failed_stage = ""
            failed_code = ""
        if terminal_status in TERMINAL_STATUSES[:2]:
            failure = publish_failure_record(
                args=args,
                source_commit=args.source_commit,
                terminal_status=terminal_status,
                failed_stage=failed_stage,
                failed_code=failed_code,
                resources=resources,
                environment=preflight["environment"],
                input_hashes=preflight["input_hashes"],
                evidence_root=shared_final,
                bundle_root=bundle_root,
                bundle_paths=(str(shared_final), str(local_final)),
                transaction_root=None,
            )
            transaction = None
        else:
            transaction = publish_transaction_record(
                args=args,
                source_commit=args.source_commit,
                terminal_status=terminal_status,
                resources=resources,
                shared_final=shared_final,
                local_final=local_final,
                shared_mount=preflight["shared_mount"],
                local_mount=preflight["local_mount"],
                verification=verification,
            )
            failure = None
        record_publication_ns = time.perf_counter_ns() - record_started_ns
        event(
            "formal_terminal_status_published",
            terminal_status=terminal_status,
            bundle_root=bundle_root,
            transaction_root=(
                None if transaction is None else transaction["transaction_root"]
            ),
            failure_root=None if failure is None else failure["failure_root"],
            record_publication_ns=record_publication_ns,
        )
        result = {
            "schema": "stda_f0_formal_compact_result_v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "source_commit": args.source_commit,
            "terminal_status": terminal_status,
            "bundle_root": bundle_root,
            "bundle_paths": [str(shared_final), str(local_final)],
            "verification": verification,
            "resources": resources,
            "transaction": transaction,
            "failure": failure,
            "formal_launch_marker": str(marker),
            "formal_log_path": str(_EVENT_LOG_PATH),
            "formal_log_sha256_before_compact": sha256_file(_EVENT_LOG_PATH),
            "record_publication_ns": record_publication_ns,
        }
        result["compact_evidence"] = _publish_compact_result(
            args=args,
            result=result,
        )
        return result
    except BaseException as error:
        if transaction_started_ns is None:
            transaction_started_ns = transaction_clock.get("started_ns")
        if monitor is not None:
            if transaction_started_ns is None:
                monitor_report = monitor.stop()
                record_started_ns = time.perf_counter_ns()
            else:
                monitor_report, transaction_work_ns, record_started_ns = (
                    _stop_transaction_monitor(monitor, transaction_started_ns)
                )
            monitor = None
        else:
            record_started_ns = time.perf_counter_ns()
            if transaction_started_ns is not None and transaction_work_ns is None:
                transaction_work_ns = record_started_ns - transaction_started_ns
        partial_resources = {
            "transaction_work_ns": transaction_work_ns,
            "monitor": monitor_report,
            "completed_frame_count": len(frames),
            "complete_resource_summary": resources,
        }
        failed_stage = error.stage if isinstance(error, StageFailure) else "unhandled"
        failed_code = error.code if isinstance(error, StageFailure) else type(error).__name__
        terminal_status = _classify_execution_failure(
            error,
            monitor_report=monitor_report,
            resources=resources,
        )
        failure = publish_failure_record(
            args=args,
            source_commit=args.source_commit,
            terminal_status=terminal_status,
            failed_stage=failed_stage,
            failed_code=failed_code,
            resources=partial_resources,
            environment=preflight["environment"],
            input_hashes=preflight["input_hashes"],
            evidence_root=(shared_final if shared_final is not None else staging),
            bundle_root=bundle_root,
            bundle_paths=tuple(
                str(path)
                for path in (shared_final, local_final)
                if path is not None
            ),
            transaction_root=(
                None if transaction is None else transaction.get("transaction_root")
            ),
        )
        record_publication_ns = time.perf_counter_ns() - record_started_ns
        event(
            "formal_execution_failed",
            terminal_status=terminal_status,
            failed_stage=failed_stage,
            failed_code=failed_code,
            error_message=str(error),
            record_publication_ns=record_publication_ns,
        )
        return {
            "schema": "stda_f0_formal_compact_result_v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "source_commit": args.source_commit,
            "terminal_status": terminal_status,
            "failure": failure,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback_sha256": sha256_bytes(traceback.format_exc().encode("utf-8")),
            "formal_launch_marker": str(marker),
            "formal_log_path": str(_EVENT_LOG_PATH),
            "record_publication_ns": record_publication_ns,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the single frozen all-76 STDA-F0 formal transaction"
        )
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--physical-gpu", type=int, default=2, choices=(2,))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formal-metrics", type=Path, required=True)
    parser.add_argument("--formal-run-manifest", type=Path, required=True)
    parser.add_argument("--candidate-hash-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--shared-mount-root", type=Path, required=True)
    parser.add_argument("--l40s-mount-root", type=Path, required=True)
    parser.add_argument("--shared-evidence-parent", type=Path, required=True)
    parser.add_argument("--local-evidence-parent", type=Path, required=True)
    parser.add_argument("--transaction-parent", type=Path, required=True)
    parser.add_argument("--l40s-host", default="wangning@10.254.30.44")
    parser.add_argument("--support-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--frame-child-timeout-seconds", type=float, default=300.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--formal-launch-token")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.formal_launch_token is not None
        and args.formal_launch_token != FORMAL_LAUNCH_TOKEN
    ):
        parser.error(
            "formal launch token mismatch; preflight-only is the default safe workflow"
        )
    try:
        preflight = run_preflight(args)
    except BaseException as error:
        document = {
            "schema": "stda_f0_preflight_failure_v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "source_commit": args.source_commit,
            "formal_launch_consumed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_stage": (
                error.stage if isinstance(error, StageFailure) else "preflight"
            ),
            "failed_code": (
                error.code if isinstance(error, StageFailure) else type(error).__name__
            ),
        }
        sys.stdout.buffer.write(canonical_json_bytes(document))
        sys.stdout.buffer.flush()
        return 2
    if args.preflight_only:
        sys.stdout.buffer.write(canonical_json_bytes(preflight))
        sys.stdout.buffer.flush()
        return 0
    result = run_formal(args, preflight)
    summary = {
        "schema": "stda_f0_formal_terminal_summary_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": args.source_commit,
        "terminal_status": result["terminal_status"],
        "bundle_root": result.get("bundle_root"),
        "transaction_root": (
            None
            if result.get("transaction") is None
            else result["transaction"].get("transaction_root")
        ),
        "failure_root": (
            None
            if result.get("failure") is None
            else result["failure"].get("failure_root")
        ),
        "formal_log_path": result.get("formal_log_path"),
    }
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    sys.stdout.buffer.flush()
    return 0 if result["terminal_status"] in TERMINAL_STATUSES[2:] else 2


if __name__ == "__main__":
    raise SystemExit(main())
