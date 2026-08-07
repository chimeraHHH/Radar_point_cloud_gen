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
from typing import Any, Callable

import numpy as np
import pytest

import eval.stda_f0_round as round_module
import scripts.stda_f0_oracle_phase as oracle_module
import scripts.stda_f0_verify_phase as verify_phase
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


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
    return stream.getvalue()


def _load_npy(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def _formal_support() -> tuple[TargetFreeSupport, np.ndarray]:
    rng = np.random.default_rng(SEED)
    cell_x, cell_y = np.meshgrid(
        np.arange(100, 300, 2, dtype="<i8"),
        np.arange(-100, 100, 2, dtype="<i8"),
        indexing="ij",
    )
    cells = np.column_stack(
        (
            cell_x.reshape(-1),
            cell_y.reshape(-1),
            np.full(REQUIRED_COUNT, 2, dtype="<i8"),
        )
    )
    xyz = ((cells.astype("<f8") + 0.5) / 20.0).astype("<f4")
    confidence = rng.uniform(0.25, 0.95, REQUIRED_COUNT).astype("<f4")
    candidate_id = np.arange(REQUIRED_COUNT, dtype="<i8")
    support = build_packed_support(xyz, confidence, candidate_id)
    assert support.support_count == REQUIRED_COUNT
    assert support.color_cardinalities == (REQUIRED_COUNT, 0, 0, 0, 0, 0, 0, 0)
    assert support.selected_color_id == 0
    return support, np.ascontiguousarray(support.xyz_m, dtype="<f4")


def _small_support(xyz: np.ndarray) -> TargetFreeSupport:
    rng = np.random.default_rng(SEED + 1)
    count = 32
    return build_packed_support(
        xyz[:count],
        rng.uniform(0.25, 0.95, count).astype("<f4"),
        np.arange(count, dtype="<i8"),
    )


def _synthetic_domain() -> tuple[StructuralDomain, dict[str, Any]]:
    azimuth = tuple(np.linspace(-1.0, 1.0, 2049, dtype="<f8").tolist())
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


def _graph(
    *,
    deficient: bool,
) -> tuple[AssignmentGraph, dict[str, np.ndarray]]:
    rows = np.arange(REQUIRED_COUNT, dtype="<i8")[:, None]
    offsets = np.arange(256, dtype="<i8")[None, :]
    if deficient:
        columns = np.broadcast_to(offsets, (REQUIRED_COUNT, 256)).copy()
        costs = np.broadcast_to(offsets + 1, columns.shape).copy()
    else:
        columns = np.sort(np.mod(rows + offsets, REQUIRED_COUNT), axis=1)
        costs = np.mod(columns - rows, REQUIRED_COUNT) + 1
    indices = np.ascontiguousarray(columns.reshape(-1), dtype="<i4")
    data = np.ascontiguousarray(costs.reshape(-1), dtype="<i8")
    squared = np.ascontiguousarray(data.astype("<f8") / 1_000_000.0)
    distance = np.fromiter(
        (math.sqrt(float(value)) for value in squared),
        dtype="<f8",
        count=squared.size,
    )
    indptr = np.arange(
        0,
        (REQUIRED_COUNT + 1) * 256,
        256,
        dtype="<i8",
    )
    slot_id = np.arange(REQUIRED_COUNT, dtype="<i8")
    graph = AssignmentGraph(
        indptr=indptr,
        indices=indices,
        data=data,
        edge_squared_distance_m2=squared,
        edge_distance_m=distance,
        slot_id=slot_id,
        support_cardinality=REQUIRED_COUNT,
    )
    return graph, {
        "graph_indptr.npy": indptr,
        "graph_indices.npy": indices,
        "graph_data.npy": data,
        "graph_edge_squared_distance_m2.npy": squared,
        "graph_edge_distance_m.npy": distance,
        "demand_slot_id.npy": slot_id,
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
    support_sha256: str,
    status: str,
) -> SyntheticFrame:
    return SyntheticFrame(
        support_dir=support_dir,
        oracle_dir=oracle_dir,
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
    report = _base_report(
        status=verify_phase.SUPPORT_NO_GO_STATUS,
        sequence=1,
        radar_index=11,
        support=support,
        support_payload=payload,
    )
    report["target"] = {"opened": False, "synthetic": True}
    _finalize_oracle(oracle_dir, report)
    return _frame(
        support_dir=support_dir,
        oracle_dir=oracle_dir,
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
    graph, graph_arrays = _graph(deficient=True)
    solver_payloads = _array_payloads({**_support_arrays(support), **graph_arrays})
    slot_to_support = np.full(REQUIRED_COUNT, -1, dtype="<i8")
    slot_to_support[:256] = np.arange(256, dtype="<i8")
    support_to_slot = np.full(REQUIRED_COUNT, -1, dtype="<i8")
    support_to_slot[:256] = np.arange(256, dtype="<i8")
    reachable_slots = np.arange(REQUIRED_COUNT, dtype="<i8")
    reachable_support = np.arange(256, dtype="<i8")
    hall = verify_hall_certificate(
        reachable_slots,
        reachable_support,
        slot_to_support,
        graph.indptr,
        graph.indices,
        support_cardinality=REQUIRED_COUNT,
    )
    assert hall["passed"] is True
    result_arrays = {
        **_matching_arrays(slot_to_support, support_to_slot),
        "hall_reachable_slot_rows.npy": reachable_slots,
        "hall_reachable_support_ranks.npy": reachable_support,
        "hall_reachable_slot_ids.npy": reachable_slots,
        "hall_reachable_support_ids.npy": reachable_support,
    }
    result_payloads = _array_payloads(result_arrays)
    _write_payloads(oracle_dir, "solver_inputs", solver_payloads, records)
    _write_payloads(oracle_dir, "round_results", result_payloads, records)

    report = _base_report(
        status=verify_phase.GRAPH_NO_GO_STATUS,
        sequence=2,
        radar_index=22,
        support=support,
        support_payload=support_payload,
    )
    report.update(
        {
            "target": {"opened": True, "synthetic": True},
            "round_input_binding": {
                "solver_input_files_sha256": _payload_hashes(solver_payloads),
            },
            "result_array_files_sha256": _payload_hashes(result_payloads),
            "cardinality": {
                "transported_mass": 256,
                "custom_matching": {"cardinality": 256},
                "scipy_matching": {"cardinality": 256},
                "hall_witness": {
                    "reachable_slot_count": hall["reachable_slot_count"],
                    "reachable_support_count": hall["reachable_support_count"],
                    "hall_deficit": hall["hall_deficit"],
                    "slot_set_sha256": hall["reachable_slots_sha256"],
                    "support_set_sha256": hall["reachable_support_sha256"],
                },
            },
            "payload_files": records,
        }
    )
    _finalize_oracle(oracle_dir, report)
    return _frame(
        support_dir=support_dir,
        oracle_dir=oracle_dir,
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
    packed = _round_support(support)
    graph, graph_arrays = _graph(deficient=False)
    identity = np.arange(REQUIRED_COUNT, dtype="<i8")
    decision = round_module._full_assignment_from_columns(
        packed,
        graph,
        identity,
        identity,
    )

    atom_id = np.asarray([0], dtype="<i8")
    atom_xyz = packed.xyz[:1].astype("<f8")
    atom_weight = np.asarray([1.0], dtype="<f8")
    slot_atom_id = np.zeros(REQUIRED_COUNT, dtype="<i8")
    greedy_sidecar = GreedySidecar(
        slot_atom_id=slot_atom_id,
        atom_id=atom_id,
        atom_xyz=atom_xyz,
        atom_weight=atom_weight,
    )
    delta = packed.xyz.astype("<f8") - atom_xyz[0]
    point_squared = np.einsum("ij,ij->i", delta, delta).astype("<f8")
    point_distance = np.fromiter(
        (math.sqrt(float(value)) for value in point_squared),
        dtype="<f8",
        count=REQUIRED_COUNT,
    )
    pointwise_sidecar = PointwiseSidecar(
        nearest_atom_id=np.zeros(REQUIRED_COUNT, dtype="<i8"),
        squared_distance=point_squared,
        distance_m=point_distance,
    )
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

    solver_payloads = _array_payloads({**_support_arrays(support), **graph_arrays})
    control_payloads = _array_payloads(
        {
            "demand_slot_atom_id.npy": slot_atom_id,
            "demand_atom_id.npy": atom_id,
            "demand_atom_xyz.npy": atom_xyz,
            "demand_atom_weight.npy": atom_weight,
            "pointwise_nearest_atom_id.npy": pointwise_sidecar.nearest_atom_id,
            "pointwise_squared_distance.npy": pointwise_sidecar.squared_distance,
            "pointwise_distance_m.npy": pointwise_sidecar.distance_m,
        }
    )
    result_payloads = _array_payloads(
        {
            **_matching_arrays(identity, identity),
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
    _write_payloads(oracle_dir, "round_results", result_payloads, records)
    _write_payloads(oracle_dir, "exports", export_payloads, records)

    target = np.asarray(
        [[packed.xyz[0, 0], packed.xyz[0, 1], packed.xyz[0, 2], 1.0]],
        dtype="<f4",
    )
    target_bytes = target.tobytes(order="C")
    _write_payloads(
        oracle_dir,
        "fit_evidence",
        {"target_xyz_confidence.bin": target_bytes},
        records,
    )
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
                "target_tensor_sha256": _sha256(target_bytes),
            },
            "round_input_binding": {
                "solver_input_files_sha256": _payload_hashes(solver_payloads),
                "control_input_files_sha256": _payload_hashes(control_payloads),
                "round_support_sha256": packed.digest_sha256,
                "round_graph_sha256": graph.digest_sha256,
                "round_greedy_sidecar_sha256": greedy_sidecar.digest_sha256,
                "round_pointwise_sidecar_sha256": pointwise_sidecar.digest_sha256,
            },
            "result_array_files_sha256": _payload_hashes(result_payloads),
            "cardinality": {
                "transported_mass": REQUIRED_COUNT,
                "custom_matching": {"cardinality": REQUIRED_COUNT},
                "scipy_matching": {"cardinality": REQUIRED_COUNT},
                "hall_witness": None,
            },
            "decision": {
                "objective": decision.objective,
                "assignment_sha256": decision.assignment_sha256,
                "selected_id_sha256": decision.selected_id_sha256,
                "export_sha256": decision.export_sha256,
                "objective_sha256": decision.objective_sha256,
                "digest_sha256": decision.digest_sha256,
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
            "payload_files": records,
        }
    )
    _finalize_oracle(oracle_dir, report)
    return _frame(
        support_dir=support_dir,
        oracle_dir=oracle_dir,
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
        expected_support_sha256=frame.support_sha256,
        output=output,
    )


def _copy_frame(frame: SyntheticFrame, root: Path, name: str) -> SyntheticFrame:
    oracle_dir = root / name
    shutil.copytree(frame.oracle_dir, oracle_dir)
    return SyntheticFrame(
        support_dir=frame.support_dir,
        oracle_dir=oracle_dir,
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

    assert report["passed"] is True
    assert report["oracle_status"] == verify_phase.SUPPORT_NO_GO_STATUS
    assert report["support"]["passed"] is True
    assert report["graph"] is None
    assert report["controls"] is None
    assert report["arms"] == {}
    assert report["status_checks"]["support_no_go_has_no_graph"] is True


def test_graph_cardinality_no_go_replays_verified_hall_witness(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = synthetic_frames["graph_no_go"]
    report = verify_phase.run(_arguments(frame, tmp_path / "graph_report.json"))
    graph = report["graph"]

    assert report["passed"] is True
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


def test_full_ready_replays_all_controls_exports_and_structures(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
) -> None:
    frame = synthetic_frames["full_ready"]
    report = verify_phase.run(_arguments(frame, tmp_path / "ready_report.json"))

    assert report["passed"] is True
    assert report["oracle_status"] == verify_phase.READY_STATUS
    assert report["graph"]["custom_matching"]["cardinality"] == REQUIRED_COUNT
    assert report["graph"]["hall"] is None
    assert report["controls"]["passed"] is True
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


def test_cuda_visible_devices_must_be_explicitly_empty(
    synthetic_frames: dict[str, SyntheticFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    frame = synthetic_frames["support_no_go"]

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES empty"):
        verify_phase.run(_arguments(frame, tmp_path / "cuda_report.json"))


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
        "--expected-support-sha256",
        frame.support_sha256,
        "--output",
        str(output),
    ]

    assert verify_phase.main(argv) == 0
    original = output.read_bytes()
    assert _canonical_json(json.loads(original.decode("ascii"))) == original
    with pytest.raises(FileExistsError):
        verify_phase.main(argv)
    assert output.read_bytes() == original
