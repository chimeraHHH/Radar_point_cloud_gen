#!/usr/bin/env python3
"""Independently replay one immutable STDA-F0 oracle frame on CPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eval.stda_f0_verify import (  # noqa: E402
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_SHA256,
    verify_csr,
    verify_hall_certificate,
    verify_matching,
    verify_packed_support_bytes,
    verify_control_export_replay,
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


def _canonical_document(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"STDA verifier JSON is invalid: {path}") from error
    if not isinstance(document, dict) or canonical_json_bytes(document) != payload:
        raise ValueError(f"STDA verifier JSON is not canonical: {path}")
    return document, payload


def _regular_bytes(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
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


def _directory_payloads(path: Path, *, suffix: str = ".npy") -> dict[str, bytes]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"STDA verifier directory is invalid: {path}")
    payloads = {
        child.name: _regular_bytes(child)
        for child in sorted(path.iterdir())
        if child.is_file() and child.suffix == suffix
    }
    if not payloads:
        raise ValueError(f"STDA verifier directory has no {suffix} inputs: {path}")
    return payloads


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
    if complete.get("schema") != "stda_f0_oracle_frame_complete_v1":
        raise ValueError("STDA oracle completion schema changed")
    report_path = oracle_dir / str(complete.get("oracle_report_path"))
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
) -> dict[str, Any]:
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


def _graph_replay(oracle_dir: Path, oracle: Mapping[str, Any]) -> dict[str, Any]:
    solver_payloads = _directory_payloads(oracle_dir / "solver_inputs")
    result_payloads = _directory_payloads(oracle_dir / "round_results")
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
    csr = verify_csr(
        indptr,
        indices,
        data,
        support_cardinality=int(support_id.size),
    )
    custom_replay = verify_matching(
        custom,
        indptr,
        indices,
        support_cardinality=int(support_id.size),
    )
    scipy_replay = verify_matching(
        scipy,
        indptr,
        indices,
        support_cardinality=int(support_id.size),
    )
    cardinality = oracle.get("cardinality")
    cardinality_checks = {
        "mapping": isinstance(cardinality, Mapping),
        "custom_equals_scipy": custom_replay["cardinality"]
        == scipy_replay["cardinality"],
        "oracle_transported_mass": isinstance(cardinality, Mapping)
        and cardinality.get("transported_mass") == custom_replay["cardinality"],
        "oracle_custom": isinstance(cardinality, Mapping)
        and cardinality.get("custom_matching", {}).get("cardinality")
        == custom_replay["cardinality"],
        "oracle_scipy": isinstance(cardinality, Mapping)
        and cardinality.get("scipy_matching", {}).get("cardinality")
        == scipy_replay["cardinality"],
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = oracle.get("frame")
    if not isinstance(frame, Mapping):
        raise ValueError("STDA oracle frame identity is absent")
    solver = _directory_payloads(oracle_dir / "solver_inputs")
    controls = _directory_payloads(oracle_dir / "control_inputs")
    results = _directory_payloads(oracle_dir / "round_results")
    exports = {
        path.name: _regular_bytes(path)
        for path in sorted((oracle_dir / "exports").iterdir())
        if path.is_file() and path.suffix in (".bin", ".npy")
    }
    control_replay = verify_control_export_replay(
        sequence=int(frame["sequence"]),
        radar_index=int(frame["radar_index"]),
        solver_inputs=solver,
        control_inputs=controls,
        round_results=results,
        exports=exports,
        oracle_report=oracle,
    )
    target_bytes = _regular_bytes(oracle_dir / "fit_evidence/target_xyz_confidence.bin")
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


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("STDA independent verifier requires CUDA_VISIBLE_DEVICES empty")
    forbidden = sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in FORBIDDEN_MODULE_PREFIXES
    )
    if forbidden:
        raise RuntimeError(f"STDA independent verifier loaded forbidden modules: {forbidden}")
    support_frame_dir = arguments.support_frame_dir.resolve(strict=True)
    oracle_dir = arguments.oracle_dir.resolve(strict=True)
    output = arguments.output.absolute()
    if output == oracle_dir or oracle_dir in output.parents:
        raise ValueError("STDA independent verifier cannot write inside oracle evidence")
    oracle, complete = _load_oracle(oracle_dir)
    status = oracle.get("status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"STDA oracle status is not independently replayable: {status}")
    support = _support_replay(
        support_frame_dir,
        expected_support_sha256=arguments.expected_support_sha256,
        oracle=oracle,
    )
    graph: dict[str, Any] | None = None
    controls: dict[str, Any] | None = None
    arms: dict[str, Any] = {}
    if status in (GRAPH_NO_GO_STATUS, READY_STATUS):
        graph = _graph_replay(oracle_dir, oracle)
    if status == READY_STATUS:
        controls, arms = _full_arm_replay(oracle_dir, oracle)
    status_checks = {
        "support_no_go_has_no_graph": status != SUPPORT_NO_GO_STATUS
        or graph is None,
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
    }
    checks = {
        "protocol_sha256": oracle.get("protocol_sha256") == PROTOCOL_SHA256,
        "protocol_freeze_commit": oracle.get("protocol_freeze_commit")
        == PROTOCOL_FREEZE_COMMIT,
        "oracle_completion_status": complete.get("status") == status,
        "support": support["passed"],
        "graph_when_present": graph is None or graph["passed"],
        "controls_when_present": controls is None or controls["passed"],
        "arms_when_present": not arms or all(arm["passed"] for arm in arms.values()),
        "status_semantics": all(status_checks.values()),
        "cpu_only": True,
        "forbidden_modules_absent": not forbidden,
    }
    return {
        "schema": SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "oracle_status": status,
        "frame": oracle.get("frame"),
        "support": support,
        "graph": graph,
        "controls": controls,
        "arms": arms,
        "status_checks": status_checks,
        "checks": checks,
        "source_sha256": {
            "code/eval/stda_f0_verify.py": sha256_file(
                CODE_ROOT / "eval/stda_f0_verify.py"
            ),
            "code/scripts/stda_f0_verify_phase.py": sha256_file(Path(__file__)),
        },
        "passed": all(checks.values()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-frame-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--expected-support-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = run(arguments)
    output_sha256, output_size = _exclusive_json(arguments.output, report)
    summary = {
        "schema": SCHEMA,
        "output": str(arguments.output.absolute()),
        "output_sha256": output_sha256,
        "output_size_bytes": output_size,
        "passed": report["passed"],
    }
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    sys.stdout.buffer.flush()
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
