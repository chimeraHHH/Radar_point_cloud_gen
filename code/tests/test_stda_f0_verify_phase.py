from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import pytest

import eval.stda_f0_fit as fit_module
import eval.stda_f0_round as round_module
import scripts.stda_f0_assignment_replay as replay_module
import scripts.stda_f0_oracle_phase as oracle_module
import scripts.stda_f0_verify_phase as verify_phase
from eval.stda_f0_fit import (
    build_midpoint_demand_slots,
    build_packed_pointwise_sidecar,
    build_sparse_assignment_graph,
    canonicalize_support,
    canonicalize_target_atoms,
)
from eval.stda_f0_round import (
    AssignmentGraph,
    GreedySidecar,
    PackedSupport as RoundPackedSupport,
    PointwiseSidecar,
    packed_pointwise_control,
    target_atom_round_robin_greedy,
)
from eval.stda_f0_structure import (
    StructuralDomain,
    evaluate_structure,
    freeze_structural_inputs,
)
from eval.stda_f0_support import (
    PackedSupport as TargetFreeSupport,
    build_packed_support,
    serialize_packed_support,
)
from eval.stda_f0_verify import verify_hall_certificate


SEED = 20260808
REQUIRED_COUNT = 10_000


@dataclass(frozen=True)
class SyntheticFrame:
    support_dir: Path
    oracle_dir: Path
    target_cache: Path
    support_sha256: str
    status: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(document: Any) -> bytes:
    return verify_phase.canonical_json_bytes(document)


def _read_json(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    document = json.loads(payload.decode("ascii"))
    assert _canonical_json(document) == payload
    return document


def _false_boolean_paths(value: Any, prefix: str = "") -> list[str]:
    failed: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            failed.extend(_false_boolean_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failed.extend(_false_boolean_paths(child, f"{prefix}[{index}]"))
    elif value is False:
        failed.append(prefix)
    return failed


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
    return stream.getvalue()


def _load_npy(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def _formal_support() -> tuple[TargetFreeSupport, np.ndarray]:
    rng = np.random.default_rng(SEED)
    cells_list: list[tuple[int, int, int]] = []
    for cluster in range(40):
        cluster_x = cluster % 8
        cluster_y = cluster // 8
        base_x = 100 + 200 * cluster_x
        base_y = -400 + 200 * cluster_y
        for local_x in range(25):
            for local_y in range(10):
                cells_list.append(
                    (base_x + 2 * local_x, base_y + 2 * local_y, 2)
                )
    cells = np.asarray(cells_list, dtype="<i8")
    assert cells.shape == (REQUIRED_COUNT, 3)
    xyz = ((cells.astype("<f8") + 0.5) / 20.0).astype("<f4")
    confidence = rng.uniform(0.25, 0.95, REQUIRED_COUNT).astype("<f4")
    candidate_id = np.arange(REQUIRED_COUNT, dtype="<i8")
    support = build_packed_support(xyz, confidence, candidate_id)
    assert support.support_count == REQUIRED_COUNT
    assert support.color_cardinalities == (REQUIRED_COUNT, 0, 0, 0, 0, 0, 0, 0)
    assert support.selected_color_id == 0
    return support, np.ascontiguousarray(support.xyz_m, dtype="<f4")


def _target_cache_bytes(target: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, target_xyz_confidence=np.ascontiguousarray(target, dtype="<f4"))
    return stream.getvalue()


def _write_target_cache(root: Path, name: str, payload: bytes) -> Path:
    path = root / f"{name}.npz"
    path.write_bytes(payload)
    return path.resolve(strict=True)


def _small_support(xyz: np.ndarray) -> TargetFreeSupport:
    rng = np.random.default_rng(SEED + 1)
    count = 32
    return build_packed_support(
        xyz[:count],
        rng.uniform(0.25, 0.95, count).astype("<f4"),
        np.arange(count, dtype="<i8"),
    )


def _synthetic_domain() -> tuple[StructuralDomain, dict[str, Any]]:
    azimuth = tuple(np.linspace(-1.5, 1.5, 3073, dtype="<f8").tolist())
    elevation = (-0.2, 0.2)
    domain = StructuralDomain.from_edges(azimuth, elevation)
    azimuth_bytes = np.asarray(azimuth, dtype="<f8").tobytes(order="C")
    elevation_bytes = np.asarray(elevation, dtype="<f8").tobytes(order="C")
    return domain, {
        "resources_dir": "synthetic://stda-f0-verify-phase",
        "resource_sha256": {
            "info_arr.mat": _sha256(b"synthetic-info-arr"),
            "arr_doppler.mat": _sha256(b"synthetic-arr-doppler"),
        },
        "vrh_support_sha256": _sha256(b"synthetic-vrh-support"),
        "vrh_schema_header_sha256": _sha256(b"synthetic-vrh-schema"),
        "azimuth_edge_count": len(azimuth),
        "elevation_edge_count": len(elevation),
        "azimuth_edges_rad": list(azimuth),
        "elevation_edges_rad": list(elevation),
        "azimuth_edges_sha256": _sha256(azimuth_bytes),
        "elevation_edges_sha256": _sha256(elevation_bytes),
        "structural_domain_sha256": domain.digest_sha256,
        "synthetic_test_domain": True,
    }


def _write_support(
    root: Path,
    name: str,
    support: TargetFreeSupport,
) -> tuple[Path, str, bytes]:
    support_dir = root / name
    support_dir.mkdir()
    payload = serialize_packed_support(support)
    (support_dir / "support.bin").write_bytes(payload)
    return support_dir, _sha256(payload), payload


def _support_report(
    support: TargetFreeSupport,
    payload: bytes,
) -> dict[str, Any]:
    return {
        "sha256": _sha256(payload),
        "bytes": len(payload),
        "candidate_field_sha256": support.candidate_field_sha256,
        "candidate_count": support.candidate_count,
        "support_count": support.support_count,
        "selected_color_id": support.selected_color_id,
        "color_cardinalities": list(support.color_cardinalities),
    }


def _round_support(support: TargetFreeSupport) -> RoundPackedSupport:
    return RoundPackedSupport(
        stable_candidate_id=support.stable_candidate_id,
        grid_cell=support.cell_xyz,
        xyz=support.xyz_m,
        base_confidence=support.base_confidence,
        color=support.color_id,
    )


def _support_arrays(support: TargetFreeSupport) -> dict[str, np.ndarray]:
    return {
        "support_stable_candidate_id.npy": support.stable_candidate_id,
        "support_grid_cell.npy": support.cell_xyz,
        "support_xyz.npy": support.xyz_m,
        "support_base_confidence.npy": support.base_confidence,
        "support_color.npy": support.color_id,
    }


def _fit_artifacts(
    *,
    support: TargetFreeSupport,
    support_payload: bytes,
    target: np.ndarray,
    target_cache_payload: bytes,
) -> dict[str, Any]:
    target = np.ascontiguousarray(target, dtype="<f4")
    atoms = canonicalize_target_atoms(target)
    demand = build_midpoint_demand_slots(atoms)
    canonical_support = canonicalize_support(
        support.xyz_m, support.stable_candidate_id
    )
    fitted_graph = build_sparse_assignment_graph(
        demand,
        canonical_support.xyz_float32,
        canonical_support.stable_candidate_id,
    )
    fitted_pointwise = build_packed_pointwise_sidecar(
        atoms,
        canonical_support.xyz_float32,
        canonical_support.stable_candidate_id,
    )
    packed = _round_support(support)
    graph = AssignmentGraph(
        indptr=fitted_graph.indptr,
        indices=fitted_graph.indices,
        data=fitted_graph.data,
        edge_squared_distance_m2=fitted_graph.squared_distance_m2,
        edge_distance_m=fitted_graph.distance_m,
        slot_id=demand.slot_id,
        support_cardinality=support.support_count,
    )
    greedy_sidecar = GreedySidecar(
        slot_atom_id=demand.canonical_atom_id,
        atom_id=atoms.canonical_atom_id,
        atom_xyz=atoms.xyz_float64,
        atom_weight=atoms.aggregate_weight,
    )
    pointwise_sidecar = PointwiseSidecar(
        nearest_atom_id=fitted_pointwise.nearest_atom_id,
        squared_distance=fitted_pointwise.squared_distance_m2,
        distance_m=fitted_pointwise.distance_m,
    )
    solver_arrays = {
        **_support_arrays(support),
        "graph_indptr.npy": fitted_graph.indptr,
        "graph_indices.npy": fitted_graph.indices,
        "graph_data.npy": fitted_graph.data,
        "graph_edge_squared_distance_m2.npy": fitted_graph.squared_distance_m2,
        "graph_edge_distance_m.npy": fitted_graph.distance_m,
        "demand_slot_id.npy": demand.slot_id,
    }
    control_arrays = {
        "demand_slot_atom_id.npy": demand.canonical_atom_id,
        "demand_atom_id.npy": atoms.canonical_atom_id,
        "demand_atom_xyz.npy": atoms.xyz_float64,
        "demand_atom_weight.npy": atoms.aggregate_weight,
        "pointwise_nearest_atom_id.npy": fitted_pointwise.nearest_atom_id,
        "pointwise_squared_distance.npy": fitted_pointwise.squared_distance_m2,
        "pointwise_distance_m.npy": fitted_pointwise.distance_m,
    }
    fit_arrays = {
        "target_xyz_confidence.npy": target,
        "target_atom_xyz_float32.npy": atoms.xyz_float32,
        "target_atom_polar_rae.npy": atoms.polar_rae,
        "target_atom_cdf.npy": atoms.cdf,
        "demand_slot_xyz.npy": demand.slot_xyz,
        "canonical_support_xyz_float64.npy": canonical_support.xyz_float64,
    }
    solver_payloads = _array_payloads(solver_arrays)
    control_payloads = _array_payloads(control_arrays)
    fit_payloads = {
        **_array_payloads(fit_arrays),
        "target_xyz_confidence.bin": target.tobytes(order="C"),
    }
    fit_binding = {
        "schema": "stda_f0_fit_binding_v1",
        "protocol_sha256": verify_phase.PROTOCOL_SHA256,
        "protocol_freeze_commit": verify_phase.PROTOCOL_FREEZE_COMMIT,
        "support_bin_sha256": _sha256(support_payload),
        "target_cache_sha256": _sha256(target_cache_payload),
        "target_tensor_sha256": _sha256(target.tobytes(order="C")),
        "canonical_target_atoms_sha256": atoms.digest_sha256,
        "demand_slots_sha256": demand.digest_sha256,
        "canonical_support_sha256": canonical_support.digest_sha256,
        "sparse_graph_sha256": fitted_graph.digest_sha256,
        "packed_pointwise_sidecar_sha256": fitted_pointwise.digest_sha256,
        "solver_input_files_sha256": _payload_hashes(solver_payloads),
        "control_input_files_sha256": _payload_hashes(control_payloads),
        "fit_evidence_files_sha256": _payload_hashes(fit_payloads),
    }
    fit_digests = {
        name: fit_binding[name]
        for name in (
            "canonical_target_atoms_sha256",
            "demand_slots_sha256",
            "canonical_support_sha256",
            "sparse_graph_sha256",
            "packed_pointwise_sidecar_sha256",
        )
    }
    round_binding = {
        "solver_input_files_sha256": _payload_hashes(solver_payloads),
        "control_input_files_sha256": _payload_hashes(control_payloads),
        "fit_evidence_files_sha256": _payload_hashes(fit_payloads),
        "round_support_sha256": packed.digest_sha256,
        "round_graph_sha256": graph.digest_sha256,
        "round_greedy_sidecar_sha256": greedy_sidecar.digest_sha256,
        "round_pointwise_sidecar_sha256": pointwise_sidecar.digest_sha256,
    }
    return {
        "atoms": atoms,
        "demand": demand,
        "packed": packed,
        "graph": graph,
        "greedy_sidecar": greedy_sidecar,
        "pointwise_sidecar": pointwise_sidecar,
        "solver_payloads": solver_payloads,
        "control_payloads": control_payloads,
        "fit_payloads": fit_payloads,
        "fit_binding": fit_binding,
        "fit_digests": fit_digests,
        "round_binding": round_binding,
    }


def _payload_hashes(payloads: dict[str, bytes]) -> dict[str, str]:
    return {name: _sha256(payload) for name, payload in sorted(payloads.items())}


def _array_payloads(arrays: dict[str, np.ndarray]) -> dict[str, bytes]:
    return {name: _npy_bytes(values) for name, values in arrays.items()}


def _write_payloads(
    oracle_dir: Path,
    relative_dir: str,
    payloads: dict[str, bytes],
    records: list[dict[str, Any]],
) -> None:
    directory = oracle_dir / relative_dir
    directory.mkdir(parents=True, exist_ok=False)
    for name, payload in sorted(payloads.items()):
        (directory / name).write_bytes(payload)
        records.append(
            {
                "path": f"{relative_dir}/{name}",
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )


def _write_root_payload(
    oracle_dir: Path,
    relative_path: str,
    payload: bytes,
    records: list[dict[str, Any]],
) -> None:
    path = oracle_dir / relative_path
    path.write_bytes(payload)
    records.append(
        {
            "path": relative_path,
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
    )


def _write_assignment_replay(
    *,
    oracle_dir: Path,
    solver_payloads: dict[str, bytes],
    decision: Any,
    packed: RoundPackedSupport,
    graph: AssignmentGraph,
    records: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, bool]]:
    replay_dir = oracle_dir / "assignment_replay"
    replay_dir.mkdir(parents=False, exist_ok=False)
    input_hashes = _payload_hashes(solver_payloads)
    replay_script = Path(replay_module.__file__).resolve(strict=True)
    request = oracle_module._build_replay_request(
        support_cardinality=packed.count,
        input_hashes=input_hashes,
        replay_script_sha256=replay_module.sha256_file(replay_script),
    )
    request_path = replay_dir / "request.json"
    result_path = replay_dir / "result.json"
    export_path = replay_dir / "export.bin"
    assignment_path = replay_dir / "assignment.bin"
    request_payload = _canonical_json(request)
    request_path.write_bytes(request_payload)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(replay_script),
            "--input-root",
            str(oracle_dir / "solver_inputs"),
            "--request",
            str(request_path),
            "--result",
            str(result_path),
            "--export",
            str(export_path),
            "--assignment",
            str(assignment_path),
        ],
        cwd=oracle_dir / "solver_inputs",
        env=oracle_module._clean_replay_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert completed.stdout == b""
    assert completed.stderr == b""
    result_payload = result_path.read_bytes()
    result = json.loads(result_payload.decode("ascii"))
    assert _canonical_json(result) == result_payload
    expected_source_hashes = replay_module._runtime_source_hashes()
    clean_checks = oracle_module._verify_subprocess_assignment(
        report=result,
        report_payload=result_payload,
        request=request,
        reference=decision,
        replay_export_bytes=export_path.read_bytes(),
        replay_assignment_bytes=assignment_path.read_bytes(),
        expected_input_hashes=input_hashes,
        expected_source_hashes=expected_source_hashes,
    )
    assert all(clean_checks.values()), clean_checks
    inprocess = round_module.verify_full_assignment(
        packed,
        graph,
        decision,
        reference=decision,
    )
    inprocess_checks = oracle_module._checks_dict(inprocess)
    assert all(inprocess_checks.values()), inprocess_checks
    for name in ("request.json", "result.json", "export.bin", "assignment.bin"):
        payload = (replay_dir / name).read_bytes()
        records.append(
            {
                "path": f"assignment_replay/{name}",
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    return inprocess_checks, clean_checks


def _base_report(
    *,
    status: str,
    sequence: int,
    radar_index: int,
    support: TargetFreeSupport,
    support_payload: bytes,
) -> dict[str, Any]:
    return {
        "schema": "stda_f0_oracle_frame_v1",
        "protocol_sha256": verify_phase.PROTOCOL_SHA256,
        "protocol_freeze_commit": verify_phase.PROTOCOL_FREEZE_COMMIT,
        "status": status,
        "frame": {
            "sequence": sequence,
            "radar_index": radar_index,
            "frame_key": f"seq{sequence:02d}/radar{radar_index:05d}",
        },
        "support": _support_report(support, support_payload),
        "payload_files": [],
        "synthetic_seed": SEED,
    }


def _finalize_oracle(oracle_dir: Path, report: dict[str, Any]) -> None:
    report["payload_files"] = sorted(
        report["payload_files"], key=lambda record: str(record["path"])
    )
    report_payload = _canonical_json(report)
    (oracle_dir / "oracle_report.json").write_bytes(report_payload)
    complete = {
        "schema": "stda_f0_oracle_frame_complete_v1",
        "protocol_sha256": verify_phase.PROTOCOL_SHA256,
        "protocol_freeze_commit": verify_phase.PROTOCOL_FREEZE_COMMIT,
        "status": report["status"],
        "oracle_report_path": "oracle_report.json",
        "oracle_report_bytes": len(report_payload),
        "oracle_report_sha256": _sha256(report_payload),
    }
    (oracle_dir / "ORACLE_COMPLETE.json").write_bytes(_canonical_json(complete))


def _frame(
    *,
    support_dir: Path,
    oracle_dir: Path,
    target_cache: Path,
    support_sha256: str,
    status: str,
) -> SyntheticFrame:
    return SyntheticFrame(
        support_dir=support_dir,
        oracle_dir=oracle_dir,
        target_cache=target_cache,
        support_sha256=support_sha256,
        status=status,
    )


def _capacity_no_go_frame(
    root: Path,
    support: TargetFreeSupport,
) -> SyntheticFrame:
    support_dir, support_sha256, payload = _write_support(
        root, "support_no_go_support", support
    )
    oracle_dir = root / "support_no_go_oracle"
    oracle_dir.mkdir()
    target_cache_payload = b"synthetic-invalid-npz-must-not-be-parsed\n"
    target_cache = _write_target_cache(
        root, "support_no_go_target_cache", target_cache_payload
    )
    report = _base_report(
        status=verify_phase.SUPPORT_NO_GO_STATUS,
        sequence=1,
        radar_index=11,
        support=support,
        support_payload=payload,
    )
    report["target"] = {
        "opened": True,
        "array_loader_called": False,
        "cache_arrays_read": [],
        "target_array_materialized": False,
        "synthetic": True,
        "path": str(target_cache),
        "cache_sha256": _sha256(target_cache_payload),
    }
    _finalize_oracle(oracle_dir, report)
    return _frame(
        support_dir=support_dir,
        oracle_dir=oracle_dir,
        target_cache=target_cache,
        support_sha256=support_sha256,
        status=verify_phase.SUPPORT_NO_GO_STATUS,
    )


def _matching_arrays(
    slot_to_support: np.ndarray,
    support_to_slot: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "matching_custom_slot_to_support_rank.npy": slot_to_support,
        "matching_custom_support_to_slot_row.npy": support_to_slot,
        "matching_scipy_slot_to_support_rank.npy": slot_to_support,
        "matching_scipy_support_to_slot_row.npy": support_to_slot,
    }


def _graph_no_go_frame(
    root: Path,
    support: TargetFreeSupport,
) -> SyntheticFrame:
    support_dir, support_sha256, support_payload = _write_support(
        root, "graph_no_go_support", support
    )
    oracle_dir = root / "graph_no_go_oracle"
    oracle_dir.mkdir()
    records: list[dict[str, Any]] = []
    target = np.concatenate(
        (
            np.ascontiguousarray(support.xyz_m[:1], dtype="<f4"),
            np.ones((1, 1), dtype="<f4"),
        ),
        axis=1,
    )
    target_cache_payload = _target_cache_bytes(target)
    target_cache = _write_target_cache(
        root, "graph_no_go_target_cache", target_cache_payload
    )
    fitted = _fit_artifacts(
        support=support,
        support_payload=support_payload,
        target=target,
        target_cache_payload=target_cache_payload,
    )
    graph = fitted["graph"]
    packed = fitted["packed"]
    solver_payloads = fitted["solver_payloads"]
    control_payloads = fitted["control_payloads"]
    fit_payloads = fitted["fit_payloads"]
    certificate = round_module.maximum_cardinality_certificate(packed, graph)
    assert certificate.transported_mass == 256
    assert certificate.hall_witness is not None
    witness = certificate.hall_witness
    custom_checks = oracle_module._checks_dict(
        round_module.verify_matching(graph, certificate.custom_matching)
    )
    scipy_checks = oracle_module._checks_dict(
        round_module.verify_matching(graph, certificate.scipy_matching)
    )
    assert all(custom_checks.values())
    assert all(scipy_checks.values())
    hall = verify_hall_certificate(
        witness.slot_rows,
        witness.support_ranks,
        certificate.custom_matching.slot_to_support_rank,
        graph.indptr,
        graph.indices,
        support_cardinality=REQUIRED_COUNT,
    )
    assert hall["passed"] is True
    result_arrays = {
        "matching_custom_slot_to_support_rank.npy": (
            certificate.custom_matching.slot_to_support_rank
        ),
        "matching_custom_support_to_slot_row.npy": (
            certificate.custom_matching.support_to_slot_row
        ),
        "matching_scipy_slot_to_support_rank.npy": (
            certificate.scipy_matching.slot_to_support_rank
        ),
        "matching_scipy_support_to_slot_row.npy": (
            certificate.scipy_matching.support_to_slot_row
        ),
        "hall_reachable_slot_rows.npy": witness.slot_rows,
        "hall_reachable_support_ranks.npy": witness.support_ranks,
        "hall_reachable_slot_ids.npy": witness.slot_ids,
        "hall_reachable_support_ids.npy": witness.support_ids,
    }
    result_payloads = _array_payloads(result_arrays)
    _write_payloads(oracle_dir, "solver_inputs", solver_payloads, records)
    _write_payloads(oracle_dir, "control_inputs", control_payloads, records)
    _write_payloads(oracle_dir, "fit_evidence", fit_payloads, records)
    _write_payloads(oracle_dir, "round_results", result_payloads, records)
    fit_binding_payload = _canonical_json(fitted["fit_binding"])
    _write_root_payload(
        oracle_dir, "fit_binding.json", fit_binding_payload, records
    )

    report = _base_report(
        status=verify_phase.GRAPH_NO_GO_STATUS,
        sequence=2,
        radar_index=22,
        support=support,
        support_payload=support_payload,
    )
    report.update(
        {
            "target": {
                "opened": True,
                "synthetic": True,
                "path": str(target_cache),
                "cache_sha256": _sha256(target_cache_payload),
                "target_tensor_sha256": _sha256(target.tobytes(order="C")),
                "target_shape": [1, 4],
                "target_dtype": "<f4",
                "cache_arrays_read": ["target_xyz_confidence"],
            },
            "fit_binding": fitted["fit_binding"],
            "fit_digests": fitted["fit_digests"],
            "round_input_binding": fitted["round_binding"],
            "result_array_files_sha256": _payload_hashes(result_payloads),
            "cardinality": {
                "transported_mass": certificate.transported_mass,
                "custom_matching": {
                    "cardinality": certificate.custom_matching.cardinality,
                    "digest_sha256": certificate.custom_matching.digest_sha256,
                },
                "scipy_matching": {
                    "cardinality": certificate.scipy_matching.cardinality,
                    "digest_sha256": certificate.scipy_matching.digest_sha256,
                },
                "hall_witness": {
                    "reachable_slot_count": hall["reachable_slot_count"],
                    "reachable_support_count": hall["reachable_support_count"],
                    "hall_deficit": hall["hall_deficit"],
                    "slot_set_sha256": hall["reachable_slots_sha256"],
                    "support_set_sha256": hall["reachable_support_sha256"],
                },
            },
            "matching_replay": {
                "custom": custom_checks,
                "scipy": scipy_checks,
                "hall": oracle_module._checks_dict(
                    round_module.verify_hall_witness(
                        packed,
                        graph,
                        certificate.custom_matching,
                        witness,
                    )
                ),
            },
            "payload_files": records,
        }
    )
    _finalize_oracle(oracle_dir, report)
    return _frame(
        support_dir=support_dir,
        oracle_dir=oracle_dir,
        target_cache=target_cache,
        support_sha256=support_sha256,
        status=verify_phase.GRAPH_NO_GO_STATUS,
    )


def _export_pair(arm: str, xyz: np.ndarray) -> tuple[dict[str, bytes], dict[str, Any]]:
    export = np.ascontiguousarray(xyz, dtype="<f4")
    binary = export.tobytes(order="C")
    npy = _npy_bytes(export)
    return {
        f"{arm}.bin": binary,
        f"{arm}.npy": npy,
    }, {
        "arm": arm,
        "count": int(export.shape[0]),
        "dtype": "<f4",
        "bin": {
            "path": f"exports/{arm}.bin",
            "bytes": len(binary),
            "sha256": _sha256(binary),
        },
        "npy": {
            "path": f"exports/{arm}.npy",
            "bytes": len(npy),
            "sha256": _sha256(npy),
        },
        "raw_xyz_sha256": _sha256(binary),
        "formal_spacing_verification_pending": True,
    }


def _full_ready_frame(
    root: Path,
    support: TargetFreeSupport,
) -> SyntheticFrame:
    sequence, radar_index = 3, 33
    support_dir, support_sha256, support_payload = _write_support(
        root, "full_ready_support", support
    )
    oracle_dir = root / "full_ready_oracle"
    oracle_dir.mkdir()
    records: list[dict[str, Any]] = []
    target_xyz = np.ascontiguousarray(support.xyz_m[::250], dtype="<f4")
    assert target_xyz.shape == (40, 3)
    target = np.concatenate(
        (target_xyz, np.ones((target_xyz.shape[0], 1), dtype="<f4")), axis=1
    )
    target_cache_payload = _target_cache_bytes(target)
    target_cache = _write_target_cache(
        root, "full_ready_target_cache", target_cache_payload
    )
    fitted = _fit_artifacts(
        support=support,
        support_payload=support_payload,
        target=target,
        target_cache_payload=target_cache_payload,
    )
    packed = fitted["packed"]
    graph = fitted["graph"]
    greedy_sidecar = fitted["greedy_sidecar"]
    pointwise_sidecar = fitted["pointwise_sidecar"]
    certificate = round_module.maximum_cardinality_certificate(packed, graph)
    assert certificate.transported_mass == REQUIRED_COUNT
    assert certificate.hall_witness is None
    custom_matching_checks = oracle_module._checks_dict(
        round_module.verify_matching(graph, certificate.custom_matching)
    )
    scipy_matching_checks = oracle_module._checks_dict(
        round_module.verify_matching(graph, certificate.scipy_matching)
    )
    assert all(custom_matching_checks.values())
    assert all(scipy_matching_checks.values())
    decision = round_module.solve_full_assignment(packed, graph)
    pointwise = packed_pointwise_control(
        packed,
        pointwise_sidecar,
        output_count=REQUIRED_COUNT,
    )
    greedy = target_atom_round_robin_greedy(
        packed,
        graph,
        greedy_sidecar,
        sequence=sequence,
        radar_index=radar_index,
    )
    assert greedy.failed_slot_count == 0

    solver_payloads = fitted["solver_payloads"]
    control_payloads = fitted["control_payloads"]
    fit_payloads = fitted["fit_payloads"]
    result_payloads = _array_payloads(
        {
            "matching_custom_slot_to_support_rank.npy": (
                certificate.custom_matching.slot_to_support_rank
            ),
            "matching_custom_support_to_slot_row.npy": (
                certificate.custom_matching.support_to_slot_row
            ),
            "matching_scipy_slot_to_support_rank.npy": (
                certificate.scipy_matching.slot_to_support_rank
            ),
            "matching_scipy_support_to_slot_row.npy": (
                certificate.scipy_matching.support_to_slot_row
            ),
            "decision_slot_row.npy": decision.slot_row,
            "decision_slot_id.npy": decision.slot_id,
            "decision_support_rank.npy": decision.support_rank,
            "decision_support_id.npy": decision.support_id,
            "decision_edge_cost.npy": decision.edge_cost,
            "decision_selected_support_rank.npy": decision.selected_support_rank,
            "decision_selected_support_id.npy": decision.selected_support_id,
            "packed_pointwise_selected_support_rank.npy": pointwise.selected_support_rank,
            "packed_pointwise_selected_support_id.npy": pointwise.selected_support_id,
            "packed_pointwise_nearest_atom_id.npy": pointwise.nearest_atom_id,
            "packed_pointwise_squared_distance.npy": pointwise.squared_distance,
            "packed_pointwise_distance_m.npy": pointwise.distance_m,
            "round_robin_greedy_round_index.npy": greedy.round_index,
            "round_robin_greedy_atom_id.npy": greedy.atom_id,
            "round_robin_greedy_slot_row.npy": greedy.slot_row,
            "round_robin_greedy_slot_id.npy": greedy.slot_id,
            "round_robin_greedy_support_rank.npy": greedy.support_rank,
            "round_robin_greedy_support_id.npy": greedy.support_id,
            "round_robin_greedy_edge_cost.npy": greedy.edge_cost,
            "round_robin_greedy_selected_support_rank.npy": greedy.selected_support_rank,
            "round_robin_greedy_selected_support_id.npy": greedy.selected_support_id,
            "round_robin_greedy_ordered_atom_id.npy": greedy.atom_order.ordered_atom_id,
            "round_robin_greedy_order_digest_bytes.npy": (
                greedy.atom_order.ordered_digest_bytes
            ),
        }
    )
    export_payloads: dict[str, bytes] = {}
    export_reports: dict[str, Any] = {}
    arm_xyz = {
        "decision": decision.export_xyz,
        "packed_pointwise": pointwise.export_xyz,
        "round_robin_greedy": greedy.export_xyz,
    }
    for arm, xyz in arm_xyz.items():
        pair, arm_report = _export_pair(arm, xyz)
        export_payloads.update(pair)
        export_reports[arm] = arm_report

    _write_payloads(oracle_dir, "solver_inputs", solver_payloads, records)
    _write_payloads(oracle_dir, "control_inputs", control_payloads, records)
    _write_payloads(oracle_dir, "fit_evidence", fit_payloads, records)
    _write_payloads(oracle_dir, "round_results", result_payloads, records)
    _write_payloads(oracle_dir, "exports", export_payloads, records)
    inprocess_replay_checks, clean_replay_checks = _write_assignment_replay(
        oracle_dir=oracle_dir,
        solver_payloads=solver_payloads,
        decision=decision,
        packed=packed,
        graph=graph,
        records=records,
    )
    fit_binding_payload = _canonical_json(fitted["fit_binding"])
    _write_root_payload(
        oracle_dir, "fit_binding.json", fit_binding_payload, records
    )
    target_bytes = target.tobytes(order="C")
    domain, domain_report = _synthetic_domain()
    structure_payloads: dict[str, bytes] = {}
    structure_reports: dict[str, Any] = {}
    for arm, xyz in arm_xyz.items():
        frozen = freeze_structural_inputs(xyz, target, domain=domain)
        result = evaluate_structure(frozen)
        document = oracle_module._plain_structure_report(result, domain)
        payload = _canonical_json(document)
        structure_payloads[f"{arm}.json"] = payload
        structure_reports[arm] = {
            "report": {
                "path": f"structure/{arm}.json",
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        }
    _write_payloads(oracle_dir, "structure", structure_payloads, records)

    report = _base_report(
        status=verify_phase.READY_STATUS,
        sequence=sequence,
        radar_index=radar_index,
        support=support,
        support_payload=support_payload,
    )
    report.update(
        {
            "target": {
                "opened": True,
                "synthetic": True,
                "path": str(target_cache),
                "cache_sha256": _sha256(target_cache_payload),
                "target_tensor_sha256": _sha256(target_bytes),
                "target_shape": [40, 4],
                "target_dtype": "<f4",
                "cache_arrays_read": ["target_xyz_confidence"],
            },
            "fit_binding": fitted["fit_binding"],
            "fit_digests": fitted["fit_digests"],
            "round_input_binding": fitted["round_binding"],
            "result_array_files_sha256": _payload_hashes(result_payloads),
            "cardinality": {
                "transported_mass": certificate.transported_mass,
                "custom_matching": {
                    "cardinality": certificate.custom_matching.cardinality,
                    "digest_sha256": certificate.custom_matching.digest_sha256,
                },
                "scipy_matching": {
                    "cardinality": certificate.scipy_matching.cardinality,
                    "digest_sha256": certificate.scipy_matching.digest_sha256,
                },
                "hall_witness": None,
            },
            "matching_replay": {
                "custom": custom_matching_checks,
                "scipy": scipy_matching_checks,
                "hall": None,
            },
            "decision": {
                "objective": decision.objective,
                "assignment_sha256": decision.assignment_sha256,
                "selected_id_sha256": decision.selected_id_sha256,
                "export_sha256": decision.export_sha256,
                "objective_sha256": decision.objective_sha256,
                "digest_sha256": decision.digest_sha256,
                "second_inprocess_replay": inprocess_replay_checks,
                "clean_subprocess_replay": clean_replay_checks,
                "three_solver_runs_identical": True,
            },
            "packed_pointwise": {
                "selection_sha256": pointwise.selection_sha256,
                "selected_id_sha256": pointwise.selected_id_sha256,
                "export_sha256": pointwise.export_sha256,
                "digest_sha256": pointwise.digest_sha256,
            },
            "round_robin_greedy": {
                "failed_slot_count": greedy.failed_slot_count,
                "full_capacity": greedy.full_capacity,
                "atom_order_sha256": greedy.atom_order.digest_sha256,
                "assignment_sha256": greedy.assignment_sha256,
                "selected_id_sha256": greedy.selected_id_sha256,
                "export_sha256": greedy.export_sha256,
                "digest_sha256": greedy.digest_sha256,
            },
            "exports": export_reports,
            "structural_domain": domain_report,
            "structure": structure_reports,
            "timings_ns": {
                "decision_solver_inprocess_1_ns": 1,
                "decision_solver_inprocess_2_ns": 1,
                "decision_solver_inprocess_replay_ns": 1,
                "decision_solver_subprocess_3_ns": 1,
                "decision_solver_subprocess_replay_ns": 1,
            },
            "payload_files": records,
        }
    )
    _finalize_oracle(oracle_dir, report)
    return _frame(
        support_dir=support_dir,
        oracle_dir=oracle_dir,
        target_cache=target_cache,
        support_sha256=support_sha256,
        status=verify_phase.READY_STATUS,
    )


@pytest.fixture(scope="module")
def synthetic_frames(tmp_path_factory: pytest.TempPathFactory) -> dict[str, SyntheticFrame]:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    root = tmp_path_factory.mktemp("stda_f0_verify_phase")
    formal_support, xyz = _formal_support()
    capacity_support = _small_support(xyz)
    frames = {
        "support_no_go": _capacity_no_go_frame(root, capacity_support),
        "graph_no_go": _graph_no_go_frame(root, formal_support),
        "full_ready": _full_ready_frame(root, formal_support),
    }

    assert frames["support_no_go"].status == verify_phase.SUPPORT_NO_GO_STATUS
    assert frames["graph_no_go"].status == verify_phase.GRAPH_NO_GO_STATUS
    assert frames["full_ready"].status == verify_phase.READY_STATUS
    return frames


def _arguments(frame: SyntheticFrame, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        support_frame_dir=frame.support_dir,
        oracle_dir=frame.oracle_dir,
        target_cache=frame.target_cache,
        expected_support_sha256=frame.support_sha256,
        output=output,
    )


def _copy_frame(frame: SyntheticFrame, root: Path, name: str) -> SyntheticFrame:
    oracle_dir = root / name
    shutil.copytree(frame.oracle_dir, oracle_dir)
    return SyntheticFrame(
        support_dir=frame.support_dir,
        oracle_dir=oracle_dir,
        target_cache=frame.target_cache,
        support_sha256=frame.support_sha256,
        status=frame.status,
    )


def _update_payload_record(
    report: dict[str, Any],
    relative_path: str,
    payload: bytes,
) -> None:
    records = [
        record
        for record in report["payload_files"]
        if record.get("path") == relative_path
    ]
    assert len(records) == 1
    records[0].update(bytes=len(payload), sha256=_sha256(payload))


def _rewrite_oracle_report(
    oracle_dir: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    report_path = oracle_dir / "oracle_report.json"
    report = _read_json(report_path)
    mutate(report)
    report_payload = _canonical_json(report)
    report_path.write_bytes(report_payload)

    complete_path = oracle_dir / "ORACLE_COMPLETE.json"
    complete = _read_json(complete_path)
    complete["oracle_report_bytes"] = len(report_payload)
    complete["oracle_report_sha256"] = _sha256(report_payload)
    complete_path.write_bytes(_canonical_json(complete))


def test_support_capacity_no_go_replays_without_graph_or_target(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = synthetic_frames["support_no_go"]
    report = verify_phase.run(_arguments(frame, tmp_path / "support_report.json"))

    assert report["passed"] is True, {
        "checks": _false_boolean_paths(report["checks"]),
        "status_checks": _false_boolean_paths(report["status_checks"]),
    }
    assert report["oracle_status"] == verify_phase.SUPPORT_NO_GO_STATUS
    assert report["support"]["passed"] is True
    assert report["graph"] is None
    assert report["controls"] is None
    assert report["arms"] == {}
    assert report["status_checks"]["support_no_go_has_no_graph"] is True
    assert report["target_cache_provenance"]["passed"] is True
    assert report["target_cache_provenance"]["array"]["parsed"] is False
    assert report["fit_semantic_replay"] is None


def test_graph_cardinality_no_go_replays_verified_hall_witness(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = synthetic_frames["graph_no_go"]
    report = verify_phase.run(_arguments(frame, tmp_path / "graph_report.json"))
    graph = report["graph"]

    assert report["passed"] is True, {
        "checks": _false_boolean_paths(report["checks"]),
        "status_checks": _false_boolean_paths(report["status_checks"]),
        "graph_checks": _false_boolean_paths(report["graph"]["checks"]),
        "semantic_checks": _false_boolean_paths(
            report["fit_semantic_replay"]["checks"]
        ),
    }
    assert report["oracle_status"] == verify_phase.GRAPH_NO_GO_STATUS
    assert graph is not None and graph["passed"] is True
    assert graph["custom_matching"]["cardinality"] == 256
    assert graph["scipy_matching"]["cardinality"] == 256
    assert graph["hall"] is not None
    assert graph["hall"]["passed"] is True
    assert graph["hall"]["hall_deficit"] == REQUIRED_COUNT - 256
    assert report["status_checks"]["graph_no_go_has_verified_deficit"] is True
    assert report["controls"] is None
    assert report["arms"] == {}
    semantic = report["fit_semantic_replay"]
    assert semantic is not None and semantic["passed"] is True
    assert semantic["arithmetic"]["graph_edge_count"] == 2_560_000
    assert semantic["arithmetic"]["graph_array_replay"]["passed"] is True
    assert semantic["arithmetic"]["control_array_replay"]["passed"] is True


def test_full_ready_replays_all_controls_exports_and_structures(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = synthetic_frames["full_ready"]
    report = verify_phase.run(_arguments(frame, tmp_path / "ready_report.json"))

    assert report["passed"] is True, {
        "checks": _false_boolean_paths(report["checks"]),
        "status_checks": _false_boolean_paths(report["status_checks"]),
        "graph_checks": _false_boolean_paths(report["graph"]["checks"]),
        "semantic_checks": _false_boolean_paths(
            report["fit_semantic_replay"]["checks"]
        ),
        "control_checks": _false_boolean_paths(report["controls"]["checks"]),
    }
    assert report["oracle_status"] == verify_phase.READY_STATUS
    assert report["graph"]["custom_matching"]["cardinality"] == REQUIRED_COUNT
    assert report["graph"]["hall"] is None
    assert report["controls"]["passed"] is True
    assert report["fit_semantic_replay"]["passed"] is True
    semantic_timings = report["fit_semantic_replay"]["arithmetic"]["timings_ns"]
    assert semantic_timings["ckdtree_k256_graph_and_integer_cost_ns"] > 0
    assert semantic_timings["ckdtree_pointwise_tie_replay_ns"] > 0
    assert report["timings_ns"]["independent_fit_semantic_reconstruction_ns"] > 0
    assert set(report["arms"]) == {"decision", "pointwise", "greedy"}
    for arm in report["arms"].values():
        assert arm["passed"] is True
        assert arm["export_replay"]["passed"] is True
        assert arm["spacing"]["passed"] is True
        assert arm["structural_replay"]["passed"] is True
        assert arm["structural_replay"]["evaluation_complete"] is True
        assert arm["structural_replay"]["computed_report"]["schema"] == (
            "stda_f0_structural_report_v1"
        )


def test_rebound_self_consistent_graph_cost_fails_semantic_reconstruction(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = _copy_frame(
        synthetic_frames["graph_no_go"], tmp_path, "semantic_graph_tamper"
    )
    relative = "solver_inputs/graph_data.npy"
    graph_data_path = frame.oracle_dir / relative
    changed_data = _load_npy(graph_data_path).copy()
    changed_data[0] += np.int64(REQUIRED_COUNT + 1)
    changed_payload = _npy_bytes(changed_data.astype("<i8", copy=False))
    graph_data_path.write_bytes(changed_payload)

    solver_dir = frame.oracle_dir / "solver_inputs"
    indptr = _load_npy(solver_dir / "graph_indptr.npy")
    indices = _load_npy(solver_dir / "graph_indices.npy")
    squared = _load_npy(solver_dir / "graph_edge_squared_distance_m2.npy")
    distance = _load_npy(solver_dir / "graph_edge_distance_m.npy")
    slot_id = _load_npy(solver_dir / "demand_slot_id.npy")
    support_id = _load_npy(solver_dir / "support_stable_candidate_id.npy")
    binding_path = frame.oracle_dir / "fit_binding.json"
    fit_binding = _read_json(binding_path)
    fit_graph_digest = fit_module._canonical_digest(
        "stda_f0_sparse_assignment_graph_v1",
        fit_module.GRAPH_SCHEMA,
        {
            "indptr": indptr,
            "indices": indices,
            "data": changed_data,
            "squared_distance_m2": squared,
            "distance_m": distance,
        },
        extra_header={
            "demand_digest_sha256": fit_binding["demand_slots_sha256"],
            "shape": [REQUIRED_COUNT, int(support_id.size)],
            "support_digest_sha256": fit_binding["canonical_support_sha256"],
        },
    )
    rebound_round_graph = AssignmentGraph(
        indptr=indptr,
        indices=indices,
        data=changed_data,
        edge_squared_distance_m2=squared,
        edge_distance_m=distance,
        slot_id=slot_id,
        support_cardinality=int(support_id.size),
    )
    changed_sha256 = _sha256(changed_payload)
    fit_binding["solver_input_files_sha256"]["graph_data.npy"] = changed_sha256
    fit_binding["sparse_graph_sha256"] = fit_graph_digest
    fit_binding_payload = _canonical_json(fit_binding)
    binding_path.write_bytes(fit_binding_payload)

    def rebind(report: dict[str, Any]) -> None:
        report["fit_binding"] = fit_binding
        report["fit_digests"]["sparse_graph_sha256"] = fit_graph_digest
        report["round_input_binding"]["solver_input_files_sha256"][
            "graph_data.npy"
        ] = changed_sha256
        report["round_input_binding"][
            "round_graph_sha256"
        ] = rebound_round_graph.digest_sha256
        _update_payload_record(report, relative, changed_payload)
        _update_payload_record(report, "fit_binding.json", fit_binding_payload)

    _rewrite_oracle_report(frame.oracle_dir, rebind)
    report = verify_phase.run(_arguments(frame, tmp_path / "semantic_graph.json"))

    semantic = report["fit_semantic_replay"]["arithmetic"]
    assert report["graph"]["passed"] is True
    assert semantic["file_bindings"]["fit_solver"]["passed"] is True
    assert semantic["fit_digest_checks"]["sparse_graph_sha256"] is False
    assert semantic["round_digest_checks"]["round_graph_sha256"] is False
    assert semantic["graph_array_replay"]["arrays"]["graph_data.npy"][
        "passed"
    ] is False
    assert semantic["passed"] is False
    assert report["passed"] is False


def test_rebound_solver_support_column_fails_support_bin_byte_replay(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = _copy_frame(
        synthetic_frames["graph_no_go"], tmp_path, "semantic_support_tamper"
    )
    relative = "solver_inputs/support_base_confidence.npy"
    path = frame.oracle_dir / relative
    changed = _load_npy(path).copy()
    changed[0] += np.float32(0.125)
    changed_payload = _npy_bytes(changed.astype("<f4", copy=False))
    path.write_bytes(changed_payload)

    solver_dir = frame.oracle_dir / "solver_inputs"
    rebound_support = RoundPackedSupport(
        stable_candidate_id=_load_npy(
            solver_dir / "support_stable_candidate_id.npy"
        ),
        grid_cell=_load_npy(solver_dir / "support_grid_cell.npy"),
        xyz=_load_npy(solver_dir / "support_xyz.npy"),
        base_confidence=changed,
        color=_load_npy(solver_dir / "support_color.npy"),
    )
    changed_sha256 = _sha256(changed_payload)
    binding_path = frame.oracle_dir / "fit_binding.json"
    fit_binding = _read_json(binding_path)
    fit_binding["solver_input_files_sha256"][
        "support_base_confidence.npy"
    ] = changed_sha256
    fit_binding_payload = _canonical_json(fit_binding)
    binding_path.write_bytes(fit_binding_payload)

    def rebind(report: dict[str, Any]) -> None:
        report["fit_binding"] = fit_binding
        report["round_input_binding"]["solver_input_files_sha256"][
            "support_base_confidence.npy"
        ] = changed_sha256
        report["round_input_binding"][
            "round_support_sha256"
        ] = rebound_support.digest_sha256
        _update_payload_record(report, relative, changed_payload)
        _update_payload_record(report, "fit_binding.json", fit_binding_payload)

    _rewrite_oracle_report(frame.oracle_dir, rebind)
    report = verify_phase.run(_arguments(frame, tmp_path / "semantic_support.json"))

    semantic = report["fit_semantic_replay"]["arithmetic"]
    assert report["graph"]["passed"] is True
    assert semantic["file_bindings"]["fit_solver"]["passed"] is True
    assert semantic["support_array_replay"]["arrays"][
        "support_base_confidence.npy"
    ]["passed"] is False
    assert semantic["passed"] is False
    assert report["passed"] is False


def test_authoritative_cache_rejects_self_consistent_wrong_snapshot(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = _copy_frame(
        synthetic_frames["graph_no_go"], tmp_path, "wrong_target_snapshot"
    )
    with np.load(frame.target_cache, allow_pickle=False) as cache:
        target = np.array(cache["target_xyz_confidence"], dtype="<f4", copy=True)
    target[0, 0] += np.float32(0.2)
    authoritative_payload = _target_cache_bytes(target)
    authoritative_cache = _write_target_cache(
        tmp_path, "authoritative_changed_target", authoritative_payload
    )
    rebound_frame = SyntheticFrame(
        support_dir=frame.support_dir,
        oracle_dir=frame.oracle_dir,
        target_cache=authoritative_cache,
        support_sha256=frame.support_sha256,
        status=frame.status,
    )

    def rebind_target_provenance(report: dict[str, Any]) -> None:
        report["target"].update(
            path=str(authoritative_cache),
            cache_sha256=_sha256(authoritative_payload),
            target_tensor_sha256=_sha256(target.tobytes(order="C")),
            target_shape=[int(value) for value in target.shape],
            target_dtype="<f4",
            cache_arrays_read=["target_xyz_confidence"],
        )

    _rewrite_oracle_report(frame.oracle_dir, rebind_target_provenance)
    report = verify_phase.run(
        _arguments(rebound_frame, tmp_path / "wrong_target_report.json")
    )

    assert report["target_cache_provenance"]["passed"] is True
    assert report["graph"]["passed"] is True
    semantic = report["fit_semantic_replay"]["arithmetic"]
    assert semantic["binding_checks"]["fit_target_bin_matches_authoritative_cache_array"] is False
    assert semantic["fit_array_replay"]["arrays"]["target_xyz_confidence.npy"][
        "passed"
    ] is False
    assert report["passed"] is False


def test_file_hash_tamper_is_detected(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = _copy_frame(synthetic_frames["graph_no_go"], tmp_path, "hash_tamper")
    path = frame.oracle_dir / "solver_inputs/graph_edge_distance_m.npy"
    changed = _load_npy(path).copy()
    changed[0] = np.nextafter(changed[0], np.float64(math.inf))
    path.write_bytes(_npy_bytes(changed.astype("<f8", copy=False)))

    report = verify_phase.run(_arguments(frame, tmp_path / "hash_report.json"))

    assert report["passed"] is False
    assert report["graph"]["solver_file_binding"]["passed"] is False
    assert report["graph"]["solver_file_binding"]["checks"][
        "exact_file_names"
    ] is True
    assert report["graph"]["solver_file_binding"]["checks"][
        "all_hashes_match"
    ] is False


@pytest.mark.parametrize(
    ("relative_dir", "binding_key"),
    (
        ("solver_inputs", "solver_input_files_sha256"),
        ("control_inputs", "control_input_files_sha256"),
        ("fit_evidence", "fit_evidence_files_sha256"),
    ),
)
def test_rebound_extra_input_file_is_rejected(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
    relative_dir: str,
    binding_key: str,
) -> None:
    frame = _copy_frame(
        synthetic_frames["graph_no_go"],
        tmp_path,
        f"extra_{relative_dir}",
    )
    payload = _npy_bytes(np.asarray([1], dtype="<i8"))
    extra = frame.oracle_dir / relative_dir / "unexpected.npy"
    extra.write_bytes(payload)

    def rebind(report: dict[str, Any]) -> None:
        report["fit_binding"][binding_key]["unexpected.npy"] = _sha256(payload)
        report["round_input_binding"][binding_key]["unexpected.npy"] = _sha256(
            payload
        )
        report["payload_files"].append(
            {
                "path": f"{relative_dir}/unexpected.npy",
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )

    _rewrite_oracle_report(frame.oracle_dir, rebind)
    with pytest.raises(ValueError, match="directory file set changed"):
        verify_phase.run(_arguments(frame, tmp_path / f"{relative_dir}.json"))


def test_rebound_structural_report_tamper_fails_independent_replay(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = _copy_frame(synthetic_frames["full_ready"], tmp_path, "structure_tamper")
    relative = "structure/decision.json"
    path = frame.oracle_dir / relative
    structure = _read_json(path)
    structure["matched_count"] = int(structure["matched_count"]) - 1
    payload = _canonical_json(structure)
    path.write_bytes(payload)

    def rebind(report: dict[str, Any]) -> None:
        report["structure"]["decision"]["report"].update(
            bytes=len(payload), sha256=_sha256(payload)
        )
        _update_payload_record(report, relative, payload)

    _rewrite_oracle_report(frame.oracle_dir, rebind)
    report = verify_phase.run(_arguments(frame, tmp_path / "structure_report.json"))

    replay = report["arms"]["decision"]["structural_replay"]
    assert report["passed"] is False
    assert replay["passed"] is False
    assert replay["checks"]["matched_count"] is False


def test_rebound_export_npy_bin_mismatch_is_detected(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = _copy_frame(synthetic_frames["full_ready"], tmp_path, "export_tamper")
    relative = "exports/decision.npy"
    path = frame.oracle_dir / relative
    changed = _load_npy(path).copy()
    changed[0, 1] += np.float32(0.25)
    payload = _npy_bytes(changed.astype("<f4", copy=False))
    path.write_bytes(payload)

    def rebind(report: dict[str, Any]) -> None:
        report["exports"]["decision"]["npy"].update(
            bytes=len(payload), sha256=_sha256(payload)
        )
        _update_payload_record(report, relative, payload)

    _rewrite_oracle_report(frame.oracle_dir, rebind)
    report = verify_phase.run(_arguments(frame, tmp_path / "export_report.json"))
    export = report["controls"]["exports"]["decision"]

    assert report["passed"] is False
    assert export["checks"]["npy_file_record"] is True
    assert export["checks"]["npy_bin_raw_bytes_identical"] is False


@pytest.mark.parametrize(
    "target",
    (
        np.ones((2, 4), dtype="<f8"),
        np.ones((2, 3), dtype="<f4"),
        np.asfortranarray(np.ones((2, 4), dtype="<f4")),
    ),
    ids=("wrong-dtype", "wrong-shape", "fortran-order"),
)
def test_authoritative_target_rejects_noncanonical_member(
    target: np.ndarray,
    tmp_path: Path,
) -> None:
    stream = io.BytesIO()
    np.savez(stream, target_xyz_confidence=target)
    payload = stream.getvalue()
    cache = tmp_path / "noncanonical_target.npz"
    cache.write_bytes(payload)
    oracle = {
        "target": {
            "opened": True,
            "path": str(cache.resolve(strict=True)),
            "cache_sha256": _sha256(payload),
            "cache_arrays_read": ["target_xyz_confidence"],
        }
    }

    with pytest.raises(ValueError, match="target cache is invalid"):
        verify_phase._target_cache_provenance(
            cache,
            oracle=oracle,
            parse_array=True,
        )


def test_verifier_paths_reject_parent_traversal_and_symlink_components(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = evidence / "target.bin"
    target.write_bytes(b"target")
    alias = tmp_path / "alias"
    alias.symlink_to(evidence, target_is_directory=True)

    with pytest.raises(ValueError, match="parent traversal"):
        verify_phase._resolve_input_path(
            evidence / ".." / "evidence" / "target.bin",
            directory=False,
            label="target-cache",
        )
    with pytest.raises(ValueError, match="symlink"):
        verify_phase._resolve_input_path(
            alias / "target.bin",
            directory=False,
            label="target-cache",
        )
    with pytest.raises(ValueError, match="parent traversal"):
        verify_phase._resolve_output_path(
            tmp_path / "scratch" / ".." / "result.json",
            forbidden_roots=(evidence,),
            forbidden_files=(target,),
        )


def test_cuda_visible_devices_must_be_explicitly_empty(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    frame = synthetic_frames["support_no_go"]

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES empty"):
        verify_phase.run(_arguments(frame, tmp_path / "cuda_report.json"))


def test_formal_script_bootstraps_numpy_and_scipy_under_isolated_no_site() -> None:
    script = Path(verify_phase.__file__).resolve(strict=True)
    environment = dict(os.environ)
    environment.update(verify_phase.PINNED_ENVIRONMENT)
    environment.update(
        {
            "CONDA_PREFIX": sys.prefix,
            "CONDA_DEFAULT_ENV": Path(sys.prefix).name,
            "PYTHONPATH": str(verify_phase.CODE_ROOT),
            "STDA_F0_PHASE": "verify_999",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "--help"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--target-cache" in completed.stdout


def test_main_publishes_once_and_cannot_overwrite_output(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = synthetic_frames["support_no_go"]
    output = tmp_path / "immutable_output.json"
    argv = [
        "--support-frame-dir",
        str(frame.support_dir),
        "--oracle-dir",
        str(frame.oracle_dir),
        "--target-cache",
        str(frame.target_cache),
        "--expected-support-sha256",
        frame.support_sha256,
        "--output",
        str(output),
    ]
    environment = dict(os.environ)
    environment.update(verify_phase.PINNED_ENVIRONMENT)
    environment.update(
        {
            "CONDA_PREFIX": sys.prefix,
            "CONDA_DEFAULT_ENV": Path(sys.prefix).name,
            "PYTHONPATH": str(verify_phase.CODE_ROOT),
            "STDA_F0_PHASE": "verify_998",
        }
    )
    command = [
        sys.executable,
        "-I",
        "-S",
        str(Path(verify_phase.__file__).resolve(strict=True)),
        *argv,
    ]

    first = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert first.returncode == 0, first.stderr.decode("utf-8", errors="replace")
    original = output.read_bytes()
    assert _canonical_json(json.loads(original.decode("ascii"))) == original
    second = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert second.returncode != 0
    assert output.read_bytes() == original
