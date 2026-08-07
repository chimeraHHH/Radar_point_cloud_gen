#!/usr/bin/env python3
"""Formal target-free support child for the frozen STDA-F0 gate.

The module deliberately installs its audit hook before importing any numerical
or project package.  Scientific imports happen only after the command-line,
environment, path, and open-file policy have been frozen.
"""

from __future__ import annotations

import sys


_ACTIVE_AUDITOR = None
_CAPTURE_BOOTSTRAP_OPENS = True
_BOOTSTRAP_OPEN_EVENTS: list[dict[str, object]] = []


def _stda_f0_early_audit_hook(event: str, args: tuple[object, ...]) -> None:
    auditor = _ACTIVE_AUDITOR
    if auditor is not None:
        auditor.handle(event, args)
    elif _CAPTURE_BOOTSTRAP_OPENS and event == "open":
        raw_path = args[0] if args else None
        _BOOTSTRAP_OPEN_EVENTS.append(
            {
                "event": "open",
                "path_repr": repr(raw_path),
                "phase": "interpreter_or_stdlib_bootstrap",
            }
        )


sys.addaudithook(_stda_f0_early_audit_hook)
AUDIT_HOOK_INSTALLED_BEFORE_SCIENTIFIC_IMPORTS = True


import argparse
import ast
import gc
import getpass
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import time
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping, Sequence


_CAPTURE_BOOTSTRAP_OPENS = False


PROTOCOL = "stda_f0_sparse_target_demand_assignment"
PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
PROTOCOL_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
FREEZE_RECORD_SHA256 = (
    "af0134038cb501b03fe1dd25937c46f9bdfacbca08fa5c6068afbc3dcfe42c7d"
)
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
FROZEN_NORMALIZATION_SHA256 = (
    "4d0bca7d027a1a9f457c526f21a034a406ce4973e2dccee55ddd2d7019b41b77"
)
FROZEN_CHECKPOINT_SHA256 = (
    "c0ebb4c1509c3d36b6834bd39977cb1cda2fcaff1747eabc97ebb41af0f06ce4"
)
FROZEN_CANDIDATE_HASH_MANIFEST_SHA256 = (
    "1d650272a021c36ee2d92219ebec3b1d91508aefdb5242c0e6fcd323d6bed2bf"
)
FROZEN_AXIS_SHA256 = {
    "info_arr.mat": (
        "53f72b22544aa11bc0057f9b8c2177a7a844fddd0e8ce3f753a989d07159767a"
    ),
    "arr_doppler.mat": (
        "f81e56889c2cedc98eb3eb8a4828e382845e3d4758a36f3b8fc0fce4839e0493"
    ),
}
FORMAL_SEED = 20260716
FORMAL_TRAIN_FRAME_COUNT = 76
FORMAL_VALIDATION_FRAME_COUNT = 24
FORMAL_CANDIDATE_COUNT = 700_000
PREDECESSOR_HASH_FRAME_COUNT = 12
FRESH_PARENT_PROTOCOL = "g1_ra1_rald_wce_stage0_v1"
FRESH_PARENT_SOURCE_COMMIT = "f2a9489d40323d1ef45d85de958f4aea8126e1c8"
SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_DEVICE_ARGUMENT = "cuda:0"
EXPECTED_GPU_NAME = "NVIDIA H200 NVL"

SUPPORT_FILE_SCHEMA = (
    ("support.bin", None, None),
    ("support_stable_candidate_id.npy", "<i8", 1),
    ("support_grid_cell.npy", "<i8", 2),
    ("support_xyz.npy", "<f4", 2),
    ("support_base_confidence.npy", "<f4", 1),
    ("support_color.npy", "<u1", 1),
)
CRITICAL_SOURCE_PATHS = (
    "code/scripts/stda_f0_support_phase.py",
    "code/eval/stda_f0_candidate.py",
    "code/eval/stda_f0_support.py",
    "code/eval/stda_f0_verify.py",
    "code/eval/rald_wce_stage0.py",
    "code/models/rald_wce_field.py",
    "code/models/rald_matched.py",
    "code/models/cube_cycle.py",
    "code/cube_dense/kradar.py",
)
BOUNDARY_FLAGS = {
    "support_target_input": False,
    "demand_target_conditioned": True,
    "graph_target_conditioned": True,
    "cost_target_conditioned": True,
    "control_score_target_conditioned": True,
    "assignment_target_conditioned": True,
    "solver_raw_target": False,
    "deployment_claim": False,
}

FORBIDDEN_OPTION_FRAGMENTS = (
    "target",
    "cache",
    "lidar",
    "label",
    "cfar",
    "future",
)
FORBIDDEN_ENVIRONMENT_FRAGMENTS = (
    "TARGET",
    "CACHE",
    "LIDAR",
    "LABEL",
    "CFAR",
    "FUTURE",
)
SANITIZED_ENVIRONMENT_KEYS = {
    "AWS_SHARED_CREDENTIALS_FILE",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "MPLCONFIGDIR",
    "PIP_CACHE_DIR",
    "PYTHONPYCACHEPREFIX",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
}
REQUIRED_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "LC_ALL": "C",
    "TZ": "UTC",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CONDA_DEFAULT_ENV": "hym_radar",
}
RETAINED_ENVIRONMENT_KEYS = {
    *REQUIRED_ENVIRONMENT,
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDNN_LOGDEST_DBG",
    "CUDNN_LOGERR_DBG",
    "CUDNN_LOGINFO_DBG",
    "HOME",
    "LANG",
    "LD_LIBRARY_PATH",
    "LOGNAME",
    "NVIDIA_TF32_OVERRIDE",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "SHELL",
    "USER",
}
SOLVER_ENVIRONMENT_PREFIXES = (
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "MKL_",
    "NVIDIA_",
    "NUMEXPR_",
    "OMP_",
    "OPENBLAS_",
    "TORCH_",
)
FORBIDDEN_DATA_COMPONENTS = {
    "cache",
    "caches",
    "cfar",
    "future",
    "futures",
    "label",
    "labels",
    "lidar",
    "lidar64",
    "lidar128",
    "os1-128",
    "os2-64",
    "target",
    "targets",
}
PROCESS_AUDIT_EVENTS = {
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.system",
    "pty.spawn",
    "subprocess.Popen",
}
MUTATION_EVENT_PATHS = {
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.link": (0, 1),
    "os.mkdir": (0,),
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.rmdir": (0,),
    "os.symlink": (1,),
    "os.truncate": (0,),
    "os.unlink": (0,),
}


class SupportPhaseContractError(RuntimeError):
    """Raised when the formal support-child contract is violated."""


class AuditAccessError(PermissionError):
    """Raised before a non-allowlisted read, write, process, or IPC action."""


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_hashed_bytes(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, sha256_bytes(payload)


def _hashed_value(value: str) -> str:
    return sha256_bytes(value.encode("utf-8", errors="surrogateescape"))


def _path_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _resolved_path(value: os.PathLike[str] | str) -> Path:
    return Path(os.path.realpath(os.path.abspath(os.fspath(value))))


def _path_has_forbidden_data_semantics(path: Path) -> bool:
    lowered = tuple(part.lower() for part in path.parts)
    if path.suffix.lower() == ".npz":
        return True
    for component in lowered:
        if component == "__pycache__":
            continue
        normalized = component.replace("_", "-")
        tokens = set(filter(None, normalized.split("-")))
        if component in FORBIDDEN_DATA_COMPONENTS:
            return True
        if tokens.intersection(FORBIDDEN_DATA_COMPONENTS):
            return True
    return False


def _value_has_forbidden_path(value: str) -> bool:
    if not value:
        return False
    lowered = value.lower().replace("\\", "/")
    if ".npz" in lowered:
        return True
    components = [part for part in lowered.split("/") if part]
    return any(
        component in FORBIDDEN_DATA_COMPONENTS
        for component in components
        if component != "__pycache__"
    )


def sanitize_environment(
    environment: MutableMapping[str, str],
    *,
    enforce_required: bool = True,
) -> dict[str, object]:
    """Remove ambient configuration and reject any GT-bearing environment input."""

    removed: list[dict[str, str]] = []
    for key in sorted(tuple(environment)):
        value = str(environment[key])
        upper = key.upper()
        if key in SANITIZED_ENVIRONMENT_KEYS:
            removed.append(
                {"key": key, "value_sha256": _hashed_value(value), "action": "removed"}
            )
            environment.pop(key, None)
            continue
        if any(fragment in upper for fragment in FORBIDDEN_ENVIRONMENT_FRAGMENTS):
            raise SupportPhaseContractError(
                f"forbidden support environment key: {key}"
            )
        if _value_has_forbidden_path(value):
            raise SupportPhaseContractError(
                f"forbidden support environment path in key: {key}"
            )
        solver_relevant = upper.startswith(SOLVER_ENVIRONMENT_PREFIXES)
        if solver_relevant and key not in RETAINED_ENVIRONMENT_KEYS:
            raise SupportPhaseContractError(
                f"undeclared solver-relevant environment key: {key}"
            )
        if key not in RETAINED_ENVIRONMENT_KEYS:
            removed.append(
                {"key": key, "value_sha256": _hashed_value(value), "action": "removed"}
            )
            environment.pop(key, None)

    required_checks: dict[str, bool] = {}
    for key, expected in REQUIRED_ENVIRONMENT.items():
        passed = environment.get(key) == expected
        required_checks[key] = passed
        if enforce_required and not passed:
            raise SupportPhaseContractError(
                f"support environment requires {key}={expected!r}"
            )
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    visible_ok = visible in ("0", "2")
    required_checks["CUDA_VISIBLE_DEVICES_is_one_allowed_H200_index"] = visible_ok
    if enforce_required and not visible_ok:
        raise SupportPhaseContractError(
            "support environment requires exactly physical H200 GPU 0 or GPU 2"
        )
    conda_prefix = environment.get("CONDA_PREFIX")
    prefix_ok = bool(conda_prefix) and _resolved_path(conda_prefix) == _resolved_path(
        sys.prefix
    )
    required_checks["CONDA_PREFIX_matches_interpreter"] = prefix_ok
    if enforce_required and not prefix_ok:
        raise SupportPhaseContractError(
            "support environment must execute from the hym_radar Conda prefix"
        )

    sys.dont_write_bytecode = True
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    retained = [
        {"key": key, "value_sha256": _hashed_value(str(environment[key]))}
        for key in sorted(environment)
    ]
    return {
        "required_checks": required_checks,
        "required_values": {
            key: environment.get(key) for key in sorted(REQUIRED_ENVIRONMENT)
        },
        "removed_entries": removed,
        "retained_entries": retained,
        "pycache_writes_disabled": bool(sys.dont_write_bytecode),
        "passed": all(required_checks.values()),
    }


class OpenAuditPolicy:
    """In-memory audit ledger and deny-by-default support-child path policy."""

    def __init__(
        self,
        *,
        interpreter_roots: Sequence[Path],
        source_roots: Sequence[Path],
        exact_read_paths: Sequence[Path],
        staging_root: Path,
    ) -> None:
        self.interpreter_roots = tuple(
            sorted({_resolved_path(path) for path in interpreter_roots}, key=str)
        )
        self.source_roots = tuple(
            sorted({_resolved_path(path) for path in source_roots}, key=str)
        )
        self.exact_read_paths = {
            _resolved_path(path) for path in exact_read_paths
        }
        self.git_metadata_roots: set[Path] = set()
        self.staging_root = _resolved_path(staging_root)
        self.current_cube: Path | None = None
        self.events: list[dict[str, object]] = []
        self.denied_count = 0
        self.process_attempt_count = 0
        self.ipc_attempt_count = 0
        self.sealed = False

    def add_git_metadata_root(self, path: Path) -> None:
        if self.sealed:
            raise RuntimeError("cannot extend a sealed STDA audit policy")
        self.git_metadata_roots.add(_resolved_path(path))

    def set_current_cube(self, path: Path) -> None:
        if self.current_cube is not None:
            raise RuntimeError("STDA support audit already has a current Cube")
        resolved = _resolved_path(path)
        if _path_has_forbidden_data_semantics(resolved):
            raise AuditAccessError("current Cube path has forbidden data semantics")
        self.current_cube = resolved

    def clear_current_cube(self, path: Path) -> None:
        resolved = _resolved_path(path)
        if self.current_cube != resolved:
            raise RuntimeError("STDA support audit current Cube changed")
        self.current_cube = None

    def classify_path(self, path: object, *, write: bool) -> dict[str, object]:
        if isinstance(path, int):
            return {
                "path": f"fd:{path}",
                "access": "write" if write else "read",
                "category": "existing_file_descriptor",
                "allowed": True,
                "reason": "descriptor_has_no_new_path_resolution",
            }
        if not isinstance(path, (str, bytes, os.PathLike)):
            return {
                "path": repr(path),
                "access": "write" if write else "read",
                "category": "invalid_path_type",
                "allowed": False,
                "reason": "file_open_path_is_not_pathlike",
            }
        resolved = _resolved_path(os.fsdecode(path))
        forbidden = _path_has_forbidden_data_semantics(resolved)
        category = "not_allowlisted"
        allowed = False
        reason = "path_is_outside_frozen_support_policy"

        if write:
            allowed = _path_within(resolved, self.staging_root)
            category = "staging" if allowed else category
            reason = (
                "write_is_inside_staging"
                if allowed
                else "write_is_outside_staging"
            )
        elif resolved in self.exact_read_paths:
            allowed = True
            category = "exact_frozen_input"
            reason = "path_matches_exact_frozen_read"
        elif self.current_cube is not None and resolved == self.current_cube:
            allowed = True
            category = "current_cube"
            reason = "path_matches_current_canonical_cube"
        elif _path_within(resolved, self.staging_root):
            allowed = True
            category = "staging"
            reason = "read_is_inside_staging"
        elif any(_path_within(resolved, root) for root in self.interpreter_roots):
            allowed = True
            category = "interpreter_environment"
            reason = "read_is_inside_interpreter_environment"
        elif any(_path_within(resolved, root) for root in self.source_roots):
            allowed = True
            category = "clean_source_snapshot"
            reason = "read_is_inside_clean_source_snapshot"
        elif any(_path_within(resolved, root) for root in self.git_metadata_roots):
            allowed = True
            category = "git_metadata"
            reason = "read_is_inside_source_git_metadata"

        if forbidden and category not in {
            "current_cube",
            "exact_frozen_input",
            "interpreter_environment",
        }:
            allowed = False
            category = "forbidden_data_path"
            reason = "path_has_target_cache_or_future_data_semantics"
        return {
            "path": str(resolved),
            "access": "write" if write else "read",
            "category": category,
            "allowed": allowed,
            "reason": reason,
        }

    @staticmethod
    def _open_is_write(args: tuple[object, ...]) -> bool:
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        if isinstance(mode, str) and any(token in mode for token in "wax+"):
            return True
        if isinstance(flags, int):
            write_flags = (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
            return bool(flags & write_flags)
        return False

    def _append_decision(
        self,
        *,
        event: str,
        decision: Mapping[str, object],
    ) -> None:
        record = {"sequence": len(self.events), "event": event, **decision}
        self.events.append(record)
        if not bool(decision.get("allowed")):
            self.denied_count += 1

    def handle(self, event: str, args: tuple[object, ...]) -> None:
        if self.sealed:
            if event == "open" or event in MUTATION_EVENT_PATHS:
                raise AuditAccessError("file access attempted after audit ledger seal")
            if self._is_process_event(event) or event.startswith("socket."):
                raise AuditAccessError("process or IPC attempted after audit ledger seal")
            return
        if event == "open":
            path = args[0] if args else None
            decision = self.classify_path(path, write=self._open_is_write(args))
            self._append_decision(event=event, decision=decision)
            if not decision["allowed"]:
                raise AuditAccessError(
                    f"STDA support denied {decision['access']} path: {decision['path']}"
                )
            return
        if self._is_process_event(event):
            self.process_attempt_count += 1
            decision = {
                "path": None,
                "access": "process_creation",
                "category": "forbidden_process",
                "allowed": False,
                "reason": "support_child_cannot_create_descendants",
            }
            self._append_decision(event=event, decision=decision)
            raise AuditAccessError("STDA support child cannot create descendants")
        if event.startswith("socket."):
            self.ipc_attempt_count += 1
            decision = {
                "path": None,
                "access": "ipc",
                "category": "forbidden_ipc",
                "allowed": False,
                "reason": "support_child_cannot_use_ipc_or_network",
            }
            self._append_decision(event=event, decision=decision)
            raise AuditAccessError("STDA support child cannot use IPC or network")
        indices = MUTATION_EVENT_PATHS.get(event)
        if indices is not None:
            for index in indices:
                path = args[index] if index < len(args) else None
                decision = self.classify_path(path, write=True)
                self._append_decision(event=event, decision=decision)
                if not decision["allowed"]:
                    raise AuditAccessError(
                        f"STDA support denied mutation path: {decision['path']}"
                    )

    @staticmethod
    def _is_process_event(event: str) -> bool:
        return (
            event in PROCESS_AUDIT_EVENTS
            or event.startswith("os.exec")
            or event.startswith("os.spawn")
            or event.startswith("subprocess.")
        )

    def seal(self) -> tuple[dict[str, object], ...]:
        if self.current_cube is not None:
            raise RuntimeError("cannot seal STDA audit while a Cube remains allowed")
        if self.sealed:
            raise RuntimeError("STDA audit policy is already sealed")
        self.sealed = True
        return tuple(dict(event) for event in self.events)

    def ledger_document(
        self,
        *,
        source_commit: str,
        environment_report: Mapping[str, object],
        support_manifest_record: Mapping[str, object],
        sealed_events: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        allowed_count = sum(bool(event["allowed"]) for event in sealed_events)
        return {
            "schema": "stda_f0_support_open_ledger_v1",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "source_commit": source_commit,
            "support_target_input": False,
            "bootstrap": {
                "audit_hook_installed_before_scientific_imports": (
                    AUDIT_HOOK_INSTALLED_BEFORE_SCIENTIFIC_IMPORTS
                ),
                "script_module_open_precedes_python_module_execution": True,
                "captured_interpreter_or_stdlib_open_events": list(
                    _BOOTSTRAP_OPEN_EVENTS
                ),
            },
            "policy": {
                "interpreter_roots": [str(path) for path in self.interpreter_roots],
                "source_roots": [str(path) for path in self.source_roots],
                "git_metadata_roots": sorted(
                    str(path) for path in self.git_metadata_roots
                ),
                "exact_read_paths": sorted(
                    str(path) for path in self.exact_read_paths
                ),
                "write_root": str(self.staging_root),
                "dynamic_cube_rule": "exactly_one_current_canonical_cube_path",
                "file_open_default": "deny",
                "descendant_processes": "deny",
                "ipc_and_network": "deny",
            },
            "environment": dict(environment_report),
            "support_manifest": dict(support_manifest_record),
            "events": list(sealed_events),
            "decision": {
                "open_and_mutation_event_count": len(sealed_events),
                "allowed_event_count": allowed_count,
                "denied_event_count": self.denied_count,
                "process_creation_attempt_count": self.process_attempt_count,
                "ipc_attempt_count": self.ipc_attempt_count,
                "all_reads_allowlisted": self.denied_count == 0,
                "all_writes_inside_staging": self.denied_count == 0,
                "no_descendant_process_created": self.process_attempt_count == 0,
                "no_ipc_or_network_used": self.ipc_attempt_count == 0,
                "passed": (
                    self.denied_count == 0
                    and self.process_attempt_count == 0
                    and self.ipc_attempt_count == 0
                ),
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen STDA-F0 target-free support child",
        allow_abbrev=False,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-hash-manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", required=True, choices=(EXPECTED_DEVICE_ARGUMENT,))
    return parser


def parser_option_strings(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    return tuple(
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in ("-h", "--help")
    )


def validate_parser_surface(parser: argparse.ArgumentParser) -> None:
    options = parser_option_strings(parser)
    expected = {
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
    if set(options) != expected or len(options) != len(expected):
        raise SupportPhaseContractError("support CLI surface changed")
    forbidden = [
        option
        for option in options
        if any(fragment in option.lower() for fragment in FORBIDDEN_OPTION_FRAGMENTS)
    ]
    if forbidden:
        raise SupportPhaseContractError(
            f"support CLI exposes a forbidden input surface: {forbidden}"
        )


def canonical_train_records(document: Mapping[str, object]) -> list[dict[str, int]]:
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise SupportPhaseContractError("frozen manifest has no frame list")
    partition_counts = {"train": 0, "validation": 0}
    identities: set[tuple[int, int]] = set()
    train: list[dict[str, int]] = []
    for raw in frames:
        if not isinstance(raw, Mapping):
            raise SupportPhaseContractError("frozen manifest frame is not an object")
        partition = raw.get("partition")
        if partition not in partition_counts:
            raise SupportPhaseContractError("frozen manifest contains another partition")
        sequence = raw.get("sequence")
        radar_index = raw.get("radar_index")
        if type(sequence) is not int or type(radar_index) is not int:
            raise SupportPhaseContractError("frozen frame identity is not integral")
        identity = (sequence, radar_index)
        if identity in identities:
            raise SupportPhaseContractError("frozen manifest repeats a frame identity")
        identities.add(identity)
        partition_counts[str(partition)] += 1
        if partition == "train":
            train.append({"sequence": sequence, "radar_index": radar_index})
    expected = {
        "train": FORMAL_TRAIN_FRAME_COUNT,
        "validation": FORMAL_VALIDATION_FRAME_COUNT,
    }
    if partition_counts != expected or len(train) != FORMAL_TRAIN_FRAME_COUNT:
        raise SupportPhaseContractError(
            f"support child requires canonical 76/24 manifest, got {partition_counts}"
        )
    return train


def candidate_hash_map(
    document: Mapping[str, object],
) -> dict[tuple[int, int], dict[str, str]]:
    if document.get("kind") != "stda_f0_target_free_candidate_hash_manifest_v1":
        raise SupportPhaseContractError("candidate-hash manifest kind changed")
    if document.get("schema_version") != 1:
        raise SupportPhaseContractError("candidate-hash manifest schema changed")
    rows = document.get("frames")
    if not isinstance(rows, list) or len(rows) != PREDECESSOR_HASH_FRAME_COUNT:
        raise SupportPhaseContractError("candidate-hash manifest must contain 12 frames")
    keys = (
        "q0_sha256",
        "q1_sha256",
        "candidate_xyz_sha256",
        "base_confidence_sha256",
    )
    result: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SupportPhaseContractError("candidate-hash frame is not an object")
        identity = (int(row["sequence"]), int(row["radar_index"]))
        hashes = {key: str(row.get(key, "")) for key in keys}
        if identity in result or any(
            not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()
        ):
            raise SupportPhaseContractError("candidate-hash frame is malformed")
        result[identity] = hashes
    return result


def _validate_cli_paths(args: argparse.Namespace) -> dict[str, Path]:
    if not SOURCE_COMMIT_PATTERN.fullmatch(args.source_commit):
        raise SupportPhaseContractError("source commit must be a full lowercase SHA")
    if args.device != EXPECTED_DEVICE_ARGUMENT:
        raise SupportPhaseContractError("support child requires cuda:0")
    paths = {
        "data_root": _resolved_path(args.data_root),
        "manifest": _resolved_path(args.manifest),
        "scene_split": _resolved_path(args.scene_split),
        "normalization": _resolved_path(args.normalization),
        "checkpoint": _resolved_path(args.checkpoint),
        "candidate_hash_manifest": _resolved_path(args.candidate_hash_manifest),
        "repo": _resolved_path(args.repo),
        "output_root": _resolved_path(args.output_root),
    }
    for name, path in paths.items():
        if _path_has_forbidden_data_semantics(path):
            raise SupportPhaseContractError(
                f"support argument {name} has forbidden data-path semantics"
            )
    for name in (
        "manifest",
        "scene_split",
        "normalization",
        "checkpoint",
        "candidate_hash_manifest",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    if not paths["data_root"].is_dir() or not paths["repo"].is_dir():
        raise NotADirectoryError("support data-root and repo must be directories")
    output_root = paths["output_root"]
    if not output_root.is_dir() or output_root.is_symlink():
        raise NotADirectoryError("support output-root must be an existing real directory")
    if any(output_root.iterdir()):
        raise FileExistsError("support output-root must be empty")
    for name, path in paths.items():
        if name != "output_root" and _path_within(path, output_root):
            raise SupportPhaseContractError("support read input cannot be inside staging")
    return paths


def _interpreter_roots(environment: Mapping[str, str]) -> tuple[Path, ...]:
    roots = {
        _resolved_path(sys.prefix),
        _resolved_path(sys.base_prefix),
        _resolved_path(Path(sys.executable).parent),
    }
    conda_prefix = environment.get("CONDA_PREFIX")
    if conda_prefix:
        roots.add(_resolved_path(conda_prefix))
    return tuple(sorted(roots, key=str))


def _git_directory(repo: Path, auditor: OpenAuditPolicy) -> Path:
    marker = repo / ".git"
    if marker.is_dir():
        git_dir = _resolved_path(marker)
    elif marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if not text.startswith(prefix):
            raise SupportPhaseContractError("source .git marker is malformed")
        candidate = Path(text[len(prefix) :])
        if not candidate.is_absolute():
            candidate = repo / candidate
        git_dir = _resolved_path(candidate)
    else:
        raise SupportPhaseContractError("support source snapshot has no Git metadata")
    if not git_dir.is_dir():
        raise SupportPhaseContractError("support source Git directory is absent")
    auditor.add_git_metadata_root(git_dir)
    return git_dir


def _git_head_commit(repo: Path, auditor: OpenAuditPolicy) -> str:
    git_dir = _git_directory(repo, auditor)
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if SOURCE_COMMIT_PATTERN.fullmatch(head):
        return head
    prefix = "ref: "
    if not head.startswith(prefix):
        raise SupportPhaseContractError("source Git HEAD is malformed")
    reference = head[len(prefix) :]
    reference_path = git_dir / reference
    if reference_path.is_file():
        commit = reference_path.read_text(encoding="ascii").strip()
    else:
        packed = (git_dir / "packed-refs").read_text(encoding="ascii")
        matches = [
            line.split(" ", 1)[0]
            for line in packed.splitlines()
            if line and not line.startswith(("#", "^")) and line.endswith(" " + reference)
        ]
        if len(matches) != 1:
            raise SupportPhaseContractError("source Git reference is unresolved")
        commit = matches[0]
    if not SOURCE_COMMIT_PATTERN.fullmatch(commit):
        raise SupportPhaseContractError("source Git commit is malformed")
    return commit


def _load_json_bytes(payload: bytes, name: str) -> dict[str, object]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupportPhaseContractError(f"{name} is not valid JSON") from error
    if not isinstance(document, dict):
        raise SupportPhaseContractError(f"{name} must contain a JSON object")
    return document


def _validate_input_contract(
    *,
    paths: Mapping[str, Path],
    source_commit: str,
    auditor: OpenAuditPolicy,
) -> dict[str, object]:
    observed_head = _git_head_commit(paths["repo"], auditor)
    if observed_head != source_commit:
        raise SupportPhaseContractError("source commit does not match checked-out HEAD")

    protocol_path = paths["repo"] / "docs/stda_f0_sparse_target_demand_assignment_protocol.md"
    freeze_path = paths["repo"] / "artifacts/idea/stda_f0_freeze_record.json"
    input_specs = (
        ("protocol", protocol_path, PROTOCOL_SHA256),
        ("freeze_record", freeze_path, FREEZE_RECORD_SHA256),
        ("manifest", paths["manifest"], FROZEN_MANIFEST_SHA256),
        ("scene_split", paths["scene_split"], FROZEN_SCENE_SPLIT_SHA256),
        ("normalization", paths["normalization"], FROZEN_NORMALIZATION_SHA256),
        (
            "candidate_hash_manifest",
            paths["candidate_hash_manifest"],
            FROZEN_CANDIDATE_HASH_MANIFEST_SHA256,
        ),
    )
    documents: dict[str, dict[str, object]] = {}
    input_records: dict[str, dict[str, object]] = {}
    for name, path, expected in input_specs:
        payload, observed = _read_hashed_bytes(path)
        if observed != expected:
            raise SupportPhaseContractError(f"frozen {name} SHA-256 changed")
        input_records[name] = {
            "path": str(path),
            "size_bytes": len(payload),
            "sha256": observed,
        }
        if name != "protocol":
            documents[name] = _load_json_bytes(payload, name)

    freeze = documents["freeze_record"]
    if (
        freeze.get("schema") != "stda_f0_freeze_record_v1"
        or freeze.get("verdict") != "FREEZE"
        or freeze.get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise SupportPhaseContractError("external STDA freeze activation changed")
    scene_split = documents["scene_split"]
    if scene_split.get("gate_pass") is not True:
        raise SupportPhaseContractError("scene split leakage gate is not passed")

    train_records = canonical_train_records(documents["manifest"])
    normalization = documents["normalization"]
    values = normalization.get("normalization", normalization)
    if not isinstance(values, Mapping):
        raise SupportPhaseContractError("normalization values are absent")
    log_center = float(values.get("center", math.nan))
    log_scale = float(values.get("scale", math.nan))
    if not math.isfinite(log_center) or not math.isfinite(log_scale) or log_scale <= 0:
        raise SupportPhaseContractError("normalization values are invalid")
    normalization_frames = normalization.get("frames")
    if isinstance(normalization_frames, list):
        normalization_order = [
            {
                "sequence": int(row["sequence"]),
                "radar_index": int(row["radar_index"]),
            }
            for row in normalization_frames
        ]
        if normalization_order != train_records:
            raise SupportPhaseContractError(
                "normalization and canonical train-frame order differ"
            )

    expected_candidates = candidate_hash_map(documents["candidate_hash_manifest"])
    train_identities = {
        (record["sequence"], record["radar_index"]) for record in train_records
    }
    if not set(expected_candidates).issubset(train_identities):
        raise SupportPhaseContractError(
            "candidate-hash manifest contains a non-train identity"
        )

    checkpoint_hash = sha256_file(paths["checkpoint"])
    if checkpoint_hash != FROZEN_CHECKPOINT_SHA256:
        raise SupportPhaseContractError("Fresh-WCE checkpoint SHA-256 changed")
    input_records["checkpoint"] = {
        "path": str(paths["checkpoint"]),
        "size_bytes": paths["checkpoint"].stat().st_size,
        "sha256": checkpoint_hash,
    }

    resources = paths["data_root"] / "resources"
    axis_records: dict[str, dict[str, object]] = {}
    for filename, expected in FROZEN_AXIS_SHA256.items():
        path = _resolved_path(resources / filename)
        observed = sha256_file(path)
        if observed != expected:
            raise SupportPhaseContractError(f"frozen axis changed: {filename}")
        axis_records[filename] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": observed,
        }

    source_hashes: dict[str, str] = {}
    for relative in CRITICAL_SOURCE_PATHS:
        path = paths["repo"] / relative
        source_hashes[relative] = sha256_file(path)
    own_source = (paths["repo"] / CRITICAL_SOURCE_PATHS[0]).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(own_source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    if any(name in imported for name in ("subprocess", "multiprocessing")):
        raise SupportPhaseContractError("support child imports a process launcher")

    return {
        "input_records": input_records,
        "axis_records": axis_records,
        "source_hashes": source_hashes,
        "train_records": train_records,
        "expected_candidate_hashes": expected_candidates,
        "log_center": log_center,
        "log_scale": log_scale,
        "git_head_matches_source_commit": True,
    }


def _load_scientific_runtime(
    repo: Path,
    interpreter_roots: Sequence[Path],
) -> SimpleNamespace:
    code_root = _resolved_path(repo / "code")
    retained_path: list[str] = [str(code_root)]
    for entry in sys.path:
        if not entry:
            continue
        resolved = _resolved_path(entry)
        if any(_path_within(resolved, root) for root in interpreter_roots):
            value = str(resolved)
            if value not in retained_path:
                retained_path.append(value)
    sys.path[:] = retained_path

    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    scipy = importlib.import_module("scipy")
    kradar = importlib.import_module("cube_dense.kradar")
    candidate = importlib.import_module("eval.stda_f0_candidate")
    support = importlib.import_module("eval.stda_f0_support")
    verify = importlib.import_module("eval.stda_f0_verify")
    return SimpleNamespace(
        np=np,
        torch=torch,
        scipy=scipy,
        kradar=kradar,
        candidate=candidate,
        support=support,
        verify=verify,
    )


def _validate_runtime_api(runtime: SimpleNamespace) -> dict[str, object]:
    modules = (runtime.candidate, runtime.support, runtime.verify)
    for module in modules:
        if module.PROTOCOL_SHA256 != PROTOCOL_SHA256:
            raise SupportPhaseContractError("STDA module protocol SHA changed")
        if module.PROTOCOL_FREEZE_COMMIT != PROTOCOL_FREEZE_COMMIT:
            raise SupportPhaseContractError("STDA module freeze commit changed")
    signatures = {
        "load_tesseract": tuple(
            inspect.signature(runtime.kradar.load_tesseract).parameters
        ),
        "reconstruct_candidate_field": tuple(
            inspect.signature(
                runtime.candidate.reconstruct_candidate_field
            ).parameters
        ),
        "verify_candidate_only": tuple(
            inspect.signature(runtime.candidate.verify_candidate_only).parameters
        ),
        "build_packed_support": tuple(
            inspect.signature(runtime.support.build_packed_support).parameters
        ),
        "verify_packed_support_bytes": tuple(
            inspect.signature(runtime.verify.verify_packed_support_bytes).parameters
        ),
    }
    expected = {
        "load_tesseract": ("path", "reverse_angular_axes"),
        "reconstruct_candidate_field": (
            "base_model",
            "cube_drae",
            "range_m",
            "azimuth_rad",
            "elevation_rad",
            "inference_config",
        ),
        "verify_candidate_only": (
            "field",
            "sequence",
            "radar_index",
            "expected_hashes",
        ),
        "build_packed_support": (
            "candidate_xyz_m",
            "base_confidence",
            "candidate_ids",
        ),
        "verify_packed_support_bytes": ("serialized",),
    }
    if signatures != expected:
        raise SupportPhaseContractError(f"STDA target-free API changed: {signatures}")
    forbidden_modules = {
        "scripts.train_rald_wce_pilot",
        "eval.stda_f0_fit",
        "eval.stda_f0_round",
        "eval.stda_f0_structure",
        "eval.dense_geometry",
    }
    loaded_forbidden = sorted(forbidden_modules.intersection(sys.modules))
    if loaded_forbidden:
        raise SupportPhaseContractError(
            f"support child loaded a GT-bearing module: {loaded_forbidden}"
        )
    return {
        "signatures": {key: list(value) for key, value in signatures.items()},
        "forbidden_modules_loaded": loaded_forbidden,
        "passed": True,
    }


def _require_h200(runtime: SimpleNamespace, device_argument: str) -> tuple[Any, dict[str, object]]:
    torch = runtime.torch
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SupportPhaseContractError("CUDA device order changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "2"):
        raise SupportPhaseContractError("support child has a forbidden physical GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SupportPhaseContractError("support child requires one visible CUDA device")
    device = torch.device(device_argument)
    if device.type != "cuda" or device.index not in (None, 0):
        raise SupportPhaseContractError("support child requires cuda:0")
    name = torch.cuda.get_device_name(device)
    if name != EXPECTED_GPU_NAME or not torch.cuda.is_bf16_supported():
        raise SupportPhaseContractError(f"support child requires BF16 H200, got {name}")
    properties = torch.cuda.get_device_properties(device)
    return device, {
        "device_argument": device_argument,
        "visible_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "name": name,
        "total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _npy_bytes(runtime: SimpleNamespace, values: Any, dtype: str, ndim: int) -> bytes:
    np = runtime.np
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    if array.dtype != np.dtype(dtype) or array.ndim != ndim:
        raise SupportPhaseContractError("support sidecar dtype or rank changed")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    replay = np.load(io.BytesIO(payload), allow_pickle=False)
    if replay.dtype != np.dtype(dtype) or replay.shape != array.shape:
        raise SupportPhaseContractError("support sidecar NumPy replay changed")
    if not np.array_equal(replay, array):
        raise SupportPhaseContractError("support sidecar NumPy values changed")
    return payload


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsync_rehash(path: Path, payload: bytes) -> dict[str, object]:
    expected = sha256_bytes(payload)
    with path.open("x+b") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        replay = handle.read()
        if replay != payload or sha256_bytes(replay) != expected:
            raise SupportPhaseContractError(f"written payload replay changed: {path}")
    _fsync_directory(path.parent)
    return {
        "relative_path": None,
        "size_bytes": len(payload),
        "sha256": expected,
        "fsynced": True,
        "full_rehash_passed": True,
    }


def _support_sidecar_payloads(runtime: SimpleNamespace, support: Any) -> dict[str, bytes]:
    return {
        "support_stable_candidate_id.npy": _npy_bytes(
            runtime, support.stable_candidate_id, "<i8", 1
        ),
        "support_grid_cell.npy": _npy_bytes(runtime, support.cell_xyz, "<i8", 2),
        "support_xyz.npy": _npy_bytes(runtime, support.xyz_m, "<f4", 2),
        "support_base_confidence.npy": _npy_bytes(
            runtime, support.base_confidence, "<f4", 1
        ),
        "support_color.npy": _npy_bytes(runtime, support.color_id, "<u1", 1),
    }


def _frame_directory_name(sequence: int, radar_index: int) -> str:
    return f"seq{sequence:02d}_radar{radar_index:05d}"


def _cube_path(data_root: Path, sequence: int, radar_index: int) -> Path:
    path = _resolved_path(
        data_root
        / str(sequence)
        / "radar_tesseract"
        / f"tesseract_{radar_index:05d}.mat"
    )
    if not _path_within(path, data_root):
        raise SupportPhaseContractError("canonical Cube path escaped data root")
    return path


def _process_frame(
    *,
    runtime: SimpleNamespace,
    auditor: OpenAuditPolicy,
    data_root: Path,
    output_root: Path,
    position: int,
    record: Mapping[str, int],
    expected_hashes: Mapping[str, str] | None,
    base_model: Any,
    axes_tensors: tuple[Any, Any, Any],
    inference_config: Any,
    device: Any,
) -> dict[str, object]:
    np = runtime.np
    torch = runtime.torch
    sequence = int(record["sequence"])
    radar_index = int(record["radar_index"])
    cube_path = _cube_path(data_root, sequence, radar_index)
    frame_root = output_root / "frames" / _frame_directory_name(sequence, radar_index)
    frame_root.mkdir(mode=0o750)
    _fsync_directory(frame_root.parent)
    _fsync_directory(output_root)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    support_frame_started = time.perf_counter_ns()
    auditor.set_current_cube(cube_path)
    try:
        cube_sha256 = sha256_file(cube_path)
        cube_numpy = runtime.kradar.load_tesseract(cube_path).astype(
            np.float32, copy=False
        )
    finally:
        auditor.clear_current_cube(cube_path)
    if cube_numpy.shape != (64, 256, 107, 37):
        raise SupportPhaseContractError("canonical Cube shape changed")
    cube = torch.from_numpy(cube_numpy).unsqueeze(0).to(device)
    range_m, azimuth_rad, elevation_rad = axes_tensors

    torch.cuda.synchronize(device)
    candidate_started = time.perf_counter_ns()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        field = runtime.candidate.reconstruct_candidate_field(
            base_model,
            cube,
            range_m,
            azimuth_rad,
            elevation_rad,
            inference_config,
        )
    candidate_verification = runtime.candidate.verify_candidate_only(
        field,
        sequence=sequence,
        radar_index=radar_index,
        expected_hashes=expected_hashes,
    )
    torch.cuda.synchronize(device)
    candidate_ended = time.perf_counter_ns()

    torch.cuda.synchronize(device)
    allocation_started = time.perf_counter_ns()
    xyz = field["xyz_m"].detach().float().cpu().contiguous().numpy()
    confidence = (
        field["base_confidence"].detach().float().cpu().contiguous().numpy()
    )
    candidate_ids = runtime.support.stable_candidate_ids(FORMAL_CANDIDATE_COUNT)
    packed = runtime.support.build_packed_support(xyz, confidence, candidate_ids)
    serialized = runtime.support.serialize_packed_support(packed)
    builder_commit = runtime.support.commit_packed_support(
        serialized, expected_sha256=packed.digest_sha256
    )
    independent = runtime.verify.verify_packed_support_bytes(serialized)
    if not independent.get("passed"):
        raise SupportPhaseContractError("independent packed-support verifier failed")
    if independent.get("support_sha256") != packed.digest_sha256:
        raise SupportPhaseContractError("independent packed-support hash changed")

    payloads = {"support.bin": serialized, **_support_sidecar_payloads(runtime, packed)}
    files: dict[str, dict[str, object]] = {}
    for filename, _, _ in SUPPORT_FILE_SCHEMA:
        payload = payloads[filename]
        file_record = _write_fsync_rehash(frame_root / filename, payload)
        file_record["relative_path"] = str(
            (frame_root / filename).relative_to(output_root)
        )
        files[filename] = file_record
    frame_commitment_payload = canonical_json_bytes(
        {
            "schema": "stda_f0_support_frame_commitment_v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "position": position,
            "sequence": sequence,
            "radar_index": radar_index,
            "cube_sha256": cube_sha256,
            "candidate_hashes": candidate_verification.hashes,
            "candidate_predecessor_expected": (
                candidate_verification.predecessor_expected
            ),
            "candidate_predecessor_hashes_match": (
                candidate_verification.predecessor_hashes_match
            ),
            "candidate_field_sha256": packed.candidate_field_sha256,
            "support_sha256": packed.digest_sha256,
            "support_count": int(packed.support_count),
            "selected_color_id": int(packed.selected_color_id),
            "color_cardinalities": list(packed.color_cardinalities),
            "files": files,
            "independent_support_verification_passed": bool(
                independent["passed"]
            ),
            "support_target_input": False,
            "ground_truth_accessed": False,
        }
    )
    frame_commitment = _write_fsync_rehash(
        frame_root / "support_record.json", frame_commitment_payload
    )
    frame_commitment["relative_path"] = str(
        (frame_root / "support_record.json").relative_to(output_root)
    )
    _fsync_directory(frame_root)
    _fsync_directory(frame_root.parent)
    _fsync_directory(output_root)
    allocation_ended = time.perf_counter_ns()

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    capacity = packed.capacity_status.as_dict()
    verification_report = candidate_verification.report()
    del field, cube, cube_numpy, xyz, confidence, candidate_ids, payloads
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    support_frame_ended = time.perf_counter_ns()
    return {
        "position": position,
        "sequence": sequence,
        "radar_index": radar_index,
        "frame_key": f"seq{sequence:02d}/radar{radar_index:05d}",
        "partition": "train",
        "cube": {
            "path": str(cube_path),
            "sha256": cube_sha256,
            "loaded_directly_with": "cube_dense.kradar.load_tesseract",
        },
        "candidate": verification_report,
        "support": {
            "candidate_count": int(packed.candidate_count),
            "support_count": int(packed.support_count),
            "selected_color_id": int(packed.selected_color_id),
            "color_cardinalities": list(packed.color_cardinalities),
            "candidate_field_sha256": packed.candidate_field_sha256,
            "support_sha256": packed.digest_sha256,
            "capacity": capacity,
            "builder_commit": {
                "support_sha256": builder_commit.support_sha256,
                "builder_spacing_self_check_passed": (
                    builder_commit.builder_spacing_self_check_passed
                ),
                "builder_is_not_formal_independent_verifier": True,
            },
            "formal_independent_spacing_verified": bool(independent["passed"]),
            "independent_verifier": independent,
        },
        "files": files,
        "frame_commitment": frame_commitment,
        "timing": {
            "support_frame_ns": support_frame_ended - support_frame_started,
            "candidate_ns": candidate_ended - candidate_started,
            "support_allocation_ns": allocation_ended - allocation_started,
            "clock": "time.perf_counter_ns",
            "candidate_cuda_synchronized_boundaries": True,
            "allocation_cuda_synchronized_before_boundary": True,
            "support_frame_boundary": (
                "before_cube_open_to_after_support_record_fsync_rehash_cleanup_and_sync"
            ),
        },
        "torch_memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "support_target_input": False,
        "ground_truth_accessed": False,
    }


def _write_final_ledger(
    *,
    auditor: OpenAuditPolicy,
    output_root: Path,
    source_commit: str,
    environment_report: Mapping[str, object],
    support_manifest_record: Mapping[str, object],
) -> dict[str, object]:
    ledger_path = output_root / "open_ledger.json"
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(output_root, directory_flags)
    try:
        with ledger_path.open("x+b") as handle:
            sealed_events = auditor.seal()
            document = auditor.ledger_document(
                source_commit=source_commit,
                environment_report=environment_report,
                support_manifest_record=support_manifest_record,
                sealed_events=sealed_events,
            )
            if document["decision"]["passed"] is not True:
                raise SupportPhaseContractError("support open ledger decision failed")
            payload = canonical_json_bytes(document)
            expected = sha256_bytes(payload)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            replay = handle.read()
            if replay != payload or sha256_bytes(replay) != expected:
                raise SupportPhaseContractError("support open ledger replay changed")
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "relative_path": "open_ledger.json",
        "size_bytes": len(payload),
        "sha256": expected,
        "fsynced": True,
        "full_rehash_passed": True,
        "decision_passed": True,
    }


def run_support_phase(args: argparse.Namespace) -> dict[str, object]:
    global _ACTIVE_AUDITOR

    parser = build_parser()
    validate_parser_surface(parser)
    paths = _validate_cli_paths(args)
    environment_report = sanitize_environment(os.environ, enforce_required=True)
    if getpass.getuser() != "wangning":
        raise SupportPhaseContractError("formal support child must run as wangning")

    resources = paths["data_root"] / "resources"
    protocol_path = paths["repo"] / "docs/stda_f0_sparse_target_demand_assignment_protocol.md"
    freeze_path = paths["repo"] / "artifacts/idea/stda_f0_freeze_record.json"
    exact_reads = (
        paths["manifest"],
        paths["scene_split"],
        paths["normalization"],
        paths["checkpoint"],
        paths["candidate_hash_manifest"],
        protocol_path,
        freeze_path,
        resources / "info_arr.mat",
        resources / "arr_doppler.mat",
        paths["repo"] / ".git",
    )
    interpreter_roots = _interpreter_roots(os.environ)
    auditor = OpenAuditPolicy(
        interpreter_roots=interpreter_roots,
        source_roots=(paths["repo"] / "code",),
        exact_read_paths=exact_reads,
        staging_root=paths["output_root"],
    )
    if _ACTIVE_AUDITOR is not None:
        raise RuntimeError("another STDA support auditor is already active")
    _ACTIVE_AUDITOR = auditor

    contract = _validate_input_contract(
        paths=paths,
        source_commit=args.source_commit,
        auditor=auditor,
    )
    runtime = _load_scientific_runtime(paths["repo"], interpreter_roots)
    api_audit = _validate_runtime_api(runtime)
    device, gpu = _require_h200(runtime, args.device)
    if (
        str(runtime.np.__version__) != "2.2.6"
        or str(runtime.scipy.__version__) != "1.15.3"
        or str(runtime.torch.__version__) != "2.12.1+cu130"
    ):
        raise SupportPhaseContractError("formal scientific package versions changed")

    runtime.torch.manual_seed(FORMAL_SEED)
    runtime.torch.cuda.manual_seed_all(FORMAL_SEED)
    runtime.np.random.seed(FORMAL_SEED)
    axes = runtime.kradar.load_axes(resources)
    axes_tensors = runtime.candidate.axes_tensors(axes, device)
    inference_config = runtime.candidate.frozen_inference_config()
    checkpoint = runtime.torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("protocol") != FRESH_PARENT_PROTOCOL
        or checkpoint.get("source_commit") != FRESH_PARENT_SOURCE_COMMIT
        or int(checkpoint.get("epoch", -1)) != 20
    ):
        raise SupportPhaseContractError("Fresh-WCE checkpoint identity changed")
    base_model = runtime.candidate.build_frozen_base_model(
        checkpoint,
        log_center=float(contract["log_center"]),
        log_scale=float(contract["log_scale"]),
        device=device,
    )
    del checkpoint
    gc.collect()
    runtime.torch.cuda.empty_cache()
    runtime.torch.cuda.synchronize(device)

    frames_root = paths["output_root"] / "frames"
    frames_root.mkdir(mode=0o750)
    _fsync_directory(frames_root)
    _fsync_directory(paths["output_root"])
    expected_candidate_hashes = contract["expected_candidate_hashes"]
    frames: list[dict[str, object]] = []
    for position, record in enumerate(contract["train_records"]):
        identity = (record["sequence"], record["radar_index"])
        frames.append(
            _process_frame(
                runtime=runtime,
                auditor=auditor,
                data_root=paths["data_root"],
                output_root=paths["output_root"],
                position=position,
                record=record,
                expected_hashes=expected_candidate_hashes.get(identity),
                base_model=base_model,
                axes_tensors=axes_tensors,
                inference_config=inference_config,
                device=device,
            )
        )
    if len(frames) != FORMAL_TRAIN_FRAME_COUNT:
        raise SupportPhaseContractError("support child did not process exactly 76 frames")
    predecessor_verified = sum(
        bool(frame["candidate"]["predecessor_expected"]) for frame in frames
    )
    fresh_hash_frames = FORMAL_TRAIN_FRAME_COUNT - predecessor_verified
    if (
        predecessor_verified != PREDECESSOR_HASH_FRAME_COUNT
        or fresh_hash_frames != 64
        or not all(frame["candidate"]["passed"] for frame in frames)
    ):
        raise SupportPhaseContractError("candidate hash verification coverage changed")

    runtime.torch.cuda.synchronize(device)
    del base_model, axes_tensors
    gc.collect()
    runtime.torch.cuda.empty_cache()
    runtime.torch.cuda.synchronize(device)

    support_counts = [int(frame["support"]["support_count"]) for frame in frames]
    support_manifest = {
        "schema": "stda_f0_support_manifest_v1",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "source_commit": args.source_commit,
        "boundary": BOUNDARY_FLAGS,
        "formal_contract": {
            "partition": "train",
            "frame_count": FORMAL_TRAIN_FRAME_COUNT,
            "candidate_count_per_frame": FORMAL_CANDIDATE_COUNT,
            "predecessor_hash_frames": PREDECESSOR_HASH_FRAME_COUNT,
            "fresh_hash_frames": 64,
            "single_cuda_visible_os_process": True,
            "descendant_processes_created": False,
            "target_loader_started": False,
        },
        "inputs": contract["input_records"],
        "axes": contract["axis_records"],
        "source_hashes": contract["source_hashes"],
        "source_checks": {
            "git_head_matches_source_commit": contract[
                "git_head_matches_source_commit"
            ],
            "critical_source_hash_count": len(contract["source_hashes"]),
            "clean_worktree_verified_by_parent_orchestrator": True,
        },
        "api_audit": api_audit,
        "runtime": {
            "python": platform.python_version(),
            "numpy": str(runtime.np.__version__),
            "scipy": str(runtime.scipy.__version__),
            "torch": str(runtime.torch.__version__),
            "torch_cuda": str(runtime.torch.version.cuda),
            "hostname": platform.node(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "gpu": gpu,
            "bf16_autocast": True,
        },
        "environment": environment_report,
        "frames": frames,
        "summary": {
            "frame_count": len(frames),
            "predecessor_hashes_verified": predecessor_verified,
            "fresh_candidate_hashes_recorded": fresh_hash_frames,
            "minimum_support_count": min(support_counts),
            "maximum_support_count": max(support_counts),
            "packed_support_capacity_no_go_frame_count": sum(
                count < 10_000 for count in support_counts
            ),
            "all_independent_support_verifications_passed": all(
                frame["support"]["independent_verifier"]["passed"]
                for frame in frames
            ),
            "maximum_torch_peak_allocated_bytes": max(
                int(frame["torch_memory"]["peak_allocated_bytes"])
                for frame in frames
            ),
            "maximum_torch_peak_reserved_bytes": max(
                int(frame["torch_memory"]["peak_reserved_bytes"])
                for frame in frames
            ),
        },
        "audit": {
            "open_ledger_relative_path": "open_ledger.json",
            "open_ledger_written_after_manifest": True,
            "open_ledger_hash_is_external_to_avoid_self_reference": True,
        },
    }
    manifest_payload = canonical_json_bytes(support_manifest)
    manifest_record = _write_fsync_rehash(
        paths["output_root"] / "support_manifest.json", manifest_payload
    )
    manifest_record["relative_path"] = "support_manifest.json"
    _fsync_directory(paths["output_root"])
    ledger_record = _write_final_ledger(
        auditor=auditor,
        output_root=paths["output_root"],
        source_commit=args.source_commit,
        environment_report=environment_report,
        support_manifest_record=manifest_record,
    )
    return {
        "schema": "stda_f0_support_child_completion_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "source_commit": args.source_commit,
        "frame_count": len(frames),
        "support_manifest": manifest_record,
        "open_ledger": ledger_record,
        "support_target_input": False,
        "ground_truth_accessed": False,
        "completed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    validate_parser_surface(parser)
    args = parser.parse_args(argv)
    completion = run_support_phase(args)
    sys.stdout.write(canonical_json_bytes(completion).decode("ascii"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
