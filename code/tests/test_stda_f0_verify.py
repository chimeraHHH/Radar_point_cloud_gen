from __future__ import annotations

import ast
import copy
import hashlib
import io
import inspect
import json
import math
import struct
from typing import Any

import numpy as np
import pytest

import eval.stda_f0_verify as verify_module
from eval.stda_f0_structure import (
    OUTPUT_RANGE_UPPER_M as EVALUATOR_OUTPUT_RANGE_UPPER_M,
    PROTOCOL_FREEZE_COMMIT as EVALUATOR_FREEZE_COMMIT,
    PROTOCOL_SHA256 as EVALUATOR_PROTOCOL_SHA256,
    StructuralDomain,
    StructuralEvaluation,
    StructuralInputs,
    evaluate_structure,
    freeze_structural_inputs,
)
from eval.stda_f0_round import (
    AssignmentGraph,
    GreedySidecar,
    PackedSupport,
    PointwiseSidecar,
    packed_pointwise_control,
    solve_full_assignment,
    target_atom_round_robin_greedy,
)
from eval.stda_f0_verify import (
    CONTROL_EXPORT_REPLAY_SCHEMA,
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_SHA256,
    STRUCTURAL_REPORT_SCHEMA,
    UNMATCHED_SENTINEL,
    verify_exact_spacing_bytes,
    verify_control_export_replay,
    verify_export_pair,
    verify_packed_support_bytes,
    verify_structural_replay,
)


EXPECTED_PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
EXPECTED_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"


def _point(
    radius: float,
    azimuth: float = 0.0,
    elevation: float = 0.0,
) -> np.ndarray:
    cosine = math.cos(elevation)
    return np.asarray(
        [
            radius * cosine * math.cos(azimuth),
            radius * cosine * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype="<f4",
    )


def _target(points: list[np.ndarray], confidence: float = 1.0) -> np.ndarray:
    xyz = np.asarray(points, dtype="<f4")
    weight = np.full((len(points), 1), confidence, dtype="<f4")
    return np.concatenate((xyz, weight), axis=1)


def _objective_key_plain(result: StructuralEvaluation) -> list[Any] | None:
    if result.objective_key is None:
        return None
    return [
        result.objective_key[0],
        result.objective_key[1],
        list(result.objective_key[2]),
    ]


def _plain_report(
    inputs: StructuralInputs,
    result: StructuralEvaluation,
) -> dict[str, Any]:
    return {
        "schema": STRUCTURAL_REPORT_SCHEMA,
        "protocol_sha256": result.protocol_sha256,
        "protocol_freeze_commit": result.protocol_freeze_commit,
        "domain_sha256": inputs.domain.digest_sha256,
        "raw_inputs_sha256": result.raw_inputs_sha256,
        "canonical_inputs_sha256": result.canonical_inputs_sha256,
        "output": {
            "source_row_count": result.output.source_row_count,
            "duplicate_row_count": result.output.duplicate_row_count,
            "invalid_event_ids": list(result.output.invalid_event_ids),
            "digest_sha256": result.output.digest_sha256,
        },
        "target": {
            "source_row_count": result.target.source_row_count,
            "zero_confidence_row_count": result.target.zero_confidence_row_count,
            "invalid_target_ids": list(result.target.invalid_target_ids),
            "digest_sha256": result.target.digest_sha256,
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
        "objective_key": _objective_key_plain(result),
        "groups_sha256": result.groups_sha256,
        "assignments_sha256": result.assignments_sha256,
        "complete_mapping_sha256": result.complete_mapping_sha256,
        "class_metrics_sha256": result.class_metrics_sha256,
        "result_sha256": result.result_sha256,
    }


def _evaluate_and_replay(
    output: np.ndarray,
    target: np.ndarray,
    *,
    azimuth_edges: tuple[float, ...] = (-0.2, 0.2),
    elevation_edges: tuple[float, ...] = (-0.2, 0.2),
) -> tuple[StructuralInputs, StructuralEvaluation, dict[str, Any], dict[str, Any]]:
    domain = StructuralDomain.from_edges(azimuth_edges, elevation_edges)
    inputs = freeze_structural_inputs(output, target, domain=domain)
    evaluation = evaluate_structure(inputs)
    expected = _plain_report(inputs, evaluation)
    replay = verify_structural_replay(
        inputs.output_xyz_le_f4,
        inputs.target_xyz_confidence_le_f4,
        azimuth_edges,
        elevation_edges,
        expected,
    )
    return inputs, evaluation, expected, replay


def _canonical_json_line(document: dict[str, Any]) -> bytes:
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


def _valid_packed_support_bytes() -> bytes:
    stable_id = np.asarray([3, 7], dtype="<i8")
    cell_x = np.asarray([0, 2], dtype="<i8")
    cell_y = np.asarray([0, 0], dtype="<i8")
    cell_z = np.asarray([0, 0], dtype="<i8")
    x_m = np.asarray([0.0, 0.1], dtype="<f4")
    y_m = np.asarray([0.0, 0.0], dtype="<f4")
    z_m = np.asarray([0.0, 0.0], dtype="<f4")
    confidence = np.asarray([0.9, 0.8], dtype="<f4")
    color = np.asarray([0, 0], dtype="<u1")
    payload = b"".join(
        array.tobytes(order="C")
        for array in (
            stable_id,
            cell_x,
            cell_y,
            cell_z,
            x_m,
            y_m,
            z_m,
            confidence,
            color,
        )
    )
    columns = [
        {"name": name, "dtype": dtype}
        for name, dtype in (
            ("stable_candidate_id", "<i8"),
            ("cell_x", "<i8"),
            ("cell_y", "<i8"),
            ("cell_z", "<i8"),
            ("x_m", "<f4"),
            ("y_m", "<f4"),
            ("z_m", "<f4"),
            ("base_confidence", "<f4"),
            ("color_id", "<u1"),
        )
    ]
    header = {
        "schema": "stda_f0_packed_support_v1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_freeze_commit": EXPECTED_FREEZE_COMMIT,
        "support_count": 2,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_field_sha256": "1" * 64,
        "selected_color_id": 0,
        "color_cardinalities": [2, 0, 0, 0, 0, 0, 0, 0],
        "columns": columns,
    }
    return _canonical_json_line(header) + payload


def _rehash_packed_payload(serialized: bytes, payload: bytes) -> bytes:
    header_bytes, _ = serialized.split(b"\n", 1)
    header = json.loads(header_bytes.decode("ascii"))
    header["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    return _canonical_json_line(header) + payload


def _npy_payload(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
    return stream.getvalue()


def _npy_payloads(values: dict[str, np.ndarray]) -> dict[str, bytes]:
    return {name: _npy_payload(array) for name, array in values.items()}


def _payload_hashes(payloads: dict[str, bytes]) -> dict[str, str]:
    return {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in payloads.items()
    }


def _export_artifacts(
    exports_xyz: dict[str, np.ndarray],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    payloads: dict[str, bytes] = {}
    reports: dict[str, dict[str, Any]] = {}
    for arm, xyz in exports_xyz.items():
        export = np.ascontiguousarray(xyz, dtype="<f4")
        binary = export.tobytes(order="C")
        npy = _npy_payload(export)
        payloads[f"{arm}.bin"] = binary
        payloads[f"{arm}.npy"] = npy
        reports[arm] = {
            "arm": arm,
            "count": int(export.shape[0]),
            "dtype": "<f4",
            "bin": {
                "path": f"exports/{arm}.bin",
                "bytes": len(binary),
                "sha256": hashlib.sha256(binary).hexdigest(),
            },
            "npy": {
                "path": f"exports/{arm}.npy",
                "bytes": len(npy),
                "sha256": hashlib.sha256(npy).hexdigest(),
            },
            "raw_xyz_sha256": hashlib.sha256(binary).hexdigest(),
            "formal_spacing_verification_pending": True,
        }
    return payloads, reports


def _control_artifact_bundle(
    *,
    exhaust_greedy: bool = False,
    formal_count: bool = False,
) -> tuple[dict[str, Any], int | None]:
    rng = np.random.default_rng(20260808)
    slot_count = 10_000 if formal_count else 6
    support_count = slot_count + 6
    stable_id = (
        np.arange(support_count, dtype="<i8") * 17
        + int(rng.integers(100, 500))
    ).astype("<i8")
    xyz = np.zeros((support_count, 3), dtype="<f4")
    xyz[:, 0] = np.arange(support_count, dtype="<f4") * np.float32(0.5)
    grid_cell = np.floor(xyz.astype("<f8") * 20.0).astype("<i8")
    confidence = rng.uniform(0.2, 0.9, support_count).astype("<f4")
    color = (
        4 * np.mod(grid_cell[:, 0], 2)
        + 2 * np.mod(grid_cell[:, 1], 2)
        + np.mod(grid_cell[:, 2], 2)
    ).astype("<u1")
    support = PackedSupport(
        stable_candidate_id=stable_id,
        grid_cell=grid_cell,
        xyz=xyz,
        base_confidence=confidence,
        color=color,
    )

    if exhaust_greedy:
        graph_rows = tuple(
            [0, 1] if row == 0 else [0] if row == 1 else [row]
            for row in range(slot_count)
        )
        graph_costs = tuple(
            [1, 2] if row == 0 else [1] for row in range(slot_count)
        )
        slot_atom_id = np.full(slot_count, 10, dtype="<i8")
        expected_graph_k: int | None = None
    else:
        graph_rows = tuple(
            sorted({row, row + 1, row + 6}) for row in range(slot_count)
        )
        graph_costs = tuple(
            [1 + ((row * 11 + rank * 7) % 31) for rank in neighbors]
            for row, neighbors in enumerate(graph_rows)
        )
        slot_atom_id = np.asarray([10, 20, 30, 10, 20, 30], dtype="<i8")
        expected_graph_k = 3
    indptr_values = [0]
    indices_values: list[int] = []
    cost_values: list[int] = []
    for neighbors, costs in zip(graph_rows, graph_costs, strict=True):
        indices_values.extend(neighbors)
        cost_values.extend(costs)
        indptr_values.append(len(indices_values))
    indptr = np.asarray(indptr_values, dtype="<i8")
    indices = np.asarray(indices_values, dtype="<i4")
    data = np.asarray(cost_values, dtype="<i8")
    squared_distance = data.astype("<f8") / 100.0
    edge_distance = np.fromiter(
        (math.sqrt(float(value)) for value in squared_distance),
        dtype="<f8",
        count=squared_distance.size,
    )
    slot_id = np.arange(slot_count, dtype="<i8")
    graph = AssignmentGraph(
        indptr=indptr,
        indices=indices,
        data=data,
        edge_squared_distance_m2=squared_distance,
        edge_distance_m=edge_distance,
        slot_id=slot_id,
        support_cardinality=support_count,
    )
    atom_id = np.asarray([10, 20, 30], dtype="<i8")
    atom_xyz = np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.0, 0.5]],
        dtype="<f4",
    ).astype("<f8")
    atom_weight = np.asarray([0.5, 1.0, 2.0], dtype="<f8")
    greedy_sidecar = GreedySidecar(
        slot_atom_id=slot_atom_id,
        atom_id=atom_id,
        atom_xyz=atom_xyz,
        atom_weight=atom_weight,
    )
    nearest_atom_id = rng.choice(atom_id, support_count).astype("<i8")
    pointwise_squared = (
        rng.uniform(0.05, 4.0, support_count)
        + np.arange(support_count, dtype="<f8") * 1e-6
    ).astype("<f8")
    pointwise_distance = np.fromiter(
        (math.sqrt(float(value)) for value in pointwise_squared),
        dtype="<f8",
        count=support_count,
    )
    pointwise_sidecar = PointwiseSidecar(
        nearest_atom_id=nearest_atom_id,
        squared_distance=pointwise_squared,
        distance_m=pointwise_distance,
    )

    decision = solve_full_assignment(support, graph)
    pointwise = packed_pointwise_control(
        support, pointwise_sidecar, output_count=slot_count
    )
    greedy = target_atom_round_robin_greedy(
        support,
        graph,
        greedy_sidecar,
        sequence=3,
        radar_index=17,
    )
    solver_values = {
        "support_stable_candidate_id.npy": stable_id,
        "support_grid_cell.npy": grid_cell,
        "support_xyz.npy": xyz,
        "support_base_confidence.npy": confidence,
        "support_color.npy": color,
        "graph_indptr.npy": indptr,
        "graph_indices.npy": indices,
        "graph_data.npy": data,
        "graph_edge_squared_distance_m2.npy": squared_distance,
        "graph_edge_distance_m.npy": edge_distance,
        "demand_slot_id.npy": slot_id,
    }
    control_values = {
        "demand_slot_atom_id.npy": slot_atom_id,
        "demand_atom_id.npy": atom_id,
        "demand_atom_xyz.npy": atom_xyz,
        "demand_atom_weight.npy": atom_weight,
        "pointwise_nearest_atom_id.npy": nearest_atom_id,
        "pointwise_squared_distance.npy": pointwise_squared,
        "pointwise_distance_m.npy": pointwise_distance,
    }
    result_values = {
        "decision_slot_row.npy": decision.slot_row,
        "decision_slot_id.npy": decision.slot_id,
        "decision_support_rank.npy": decision.support_rank,
        "decision_support_id.npy": decision.support_id,
        "decision_edge_cost.npy": decision.edge_cost,
        "decision_selected_support_rank.npy": decision.selected_support_rank,
        "decision_selected_support_id.npy": decision.selected_support_id,
        "packed_pointwise_selected_support_rank.npy": (
            pointwise.selected_support_rank
        ),
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
        "round_robin_greedy_selected_support_rank.npy": (
            greedy.selected_support_rank
        ),
        "round_robin_greedy_selected_support_id.npy": greedy.selected_support_id,
        "round_robin_greedy_ordered_atom_id.npy": greedy.atom_order.ordered_atom_id,
        "round_robin_greedy_order_digest_bytes.npy": (
            greedy.atom_order.ordered_digest_bytes
        ),
    }
    solver_payloads = _npy_payloads(solver_values)
    control_payloads = _npy_payloads(control_values)
    result_payloads = _npy_payloads(result_values)
    export_payloads, export_reports = _export_artifacts(
        {
            "decision": decision.export_xyz,
            "packed_pointwise": pointwise.export_xyz,
            "round_robin_greedy": greedy.export_xyz,
        }
    )
    oracle_report = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_freeze_commit": EXPECTED_FREEZE_COMMIT,
        "frame": {
            "sequence": 3,
            "radar_index": 17,
            "frame_key": "seq03/radar00017",
        },
        "round_input_binding": {
            "solver_input_files_sha256": _payload_hashes(solver_payloads),
            "control_input_files_sha256": _payload_hashes(control_payloads),
            "round_support_sha256": support.digest_sha256,
            "round_graph_sha256": graph.digest_sha256,
            "round_greedy_sidecar_sha256": greedy_sidecar.digest_sha256,
            "round_pointwise_sidecar_sha256": pointwise_sidecar.digest_sha256,
        },
        "result_array_files_sha256": _payload_hashes(result_payloads),
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
    }
    return (
        {
            "sequence": 3,
            "radar_index": 17,
            "solver_inputs": solver_payloads,
            "control_inputs": control_payloads,
            "round_results": result_payloads,
            "exports": export_payloads,
            "oracle_report": oracle_report,
            "required_export_count": slot_count,
        },
        expected_graph_k,
    )


def _run_control_bundle(
    bundle: dict[str, Any], expected_graph_k: int | None
) -> dict[str, Any]:
    return verify_control_export_replay(
        sequence=bundle["sequence"],
        radar_index=bundle["radar_index"],
        solver_inputs=bundle["solver_inputs"],
        control_inputs=bundle["control_inputs"],
        round_results=bundle["round_results"],
        exports=bundle["exports"],
        oracle_report=bundle["oracle_report"],
        required_export_count=bundle["required_export_count"],
        expected_graph_k=expected_graph_k,
    )


def _replace_result_array(
    bundle: dict[str, Any], filename: str, values: np.ndarray
) -> None:
    payload = _npy_payload(values)
    bundle["round_results"][filename] = payload
    bundle["oracle_report"]["result_array_files_sha256"][filename] = (
        hashlib.sha256(payload).hexdigest()
    )


def _load_test_npy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def test_verifier_is_bound_and_has_no_forbidden_predecessor_imports() -> None:
    assert PROTOCOL_SHA256 == EXPECTED_PROTOCOL_SHA256
    assert PROTOCOL_FREEZE_COMMIT == EXPECTED_FREEZE_COMMIT
    assert EVALUATOR_PROTOCOL_SHA256 == EXPECTED_PROTOCOL_SHA256
    assert EVALUATOR_FREEZE_COMMIT == EXPECTED_FREEZE_COMMIT
    assert UNMATCHED_SENTINEL == 2**63 - 1

    tree = ast.parse(inspect.getsource(verify_module))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden = {
        "eval.stda_f0_metric",
        "eval.stda_f0_structure",
        "eval.stda_f0_support",
        "eval.stda_f0_fit",
        "eval.stda_f0_round",
        "eval.vrh_f0_metrics",
        "eval.dense_geometry",
    }
    assert imported_modules.isdisjoint(forbidden)

    signature = inspect.signature(verify_control_export_replay)
    assert tuple(signature.parameters) == (
        "sequence",
        "radar_index",
        "solver_inputs",
        "control_inputs",
        "round_results",
        "exports",
        "oracle_report",
        "required_export_count",
        "expected_graph_k",
    )
    assert signature.parameters["required_export_count"].default == 10_000
    assert signature.parameters["expected_graph_k"].default == 256


def test_independent_control_and_three_export_replay_passes_fixed_seed() -> None:
    bundle, expected_graph_k = _control_artifact_bundle()
    replay = _run_control_bundle(bundle, expected_graph_k)

    assert replay["schema"] == CONTROL_EXPORT_REPLAY_SCHEMA
    assert replay["passed"] is True
    assert replay["decision"]["checks"]["edge_membership"] is True
    assert replay["packed_pointwise"]["checks"][
        "frozen_squared_distance_then_id_order"
    ] is True
    assert replay["round_robin_greedy"]["checks"][
        "minimum_unused_edge_choice"
    ] is True
    assert replay["round_robin_greedy"]["checks"]["support_capacity_one"] is True
    for arm in ("decision", "packed_pointwise", "round_robin_greedy"):
        export = replay["exports"][arm]
        assert export["point_count"] == 6
        assert export["checks"]["finite_xyz"] is True
        assert export["checks"]["unique_xyz_bytes"] is True
        assert export["checks"]["strict_spacing_5cm"] is True
        assert export["checks"]["npy_bin_raw_bytes_identical"] is True
        assert export["checks"]["bound_to_replayed_selection"] is True


def test_packed_pointwise_tamper_changes_frozen_selection_order() -> None:
    bundle, expected_graph_k = _control_artifact_bundle()
    filename = "packed_pointwise_selected_support_rank.npy"
    selected = _load_test_npy(bundle["round_results"][filename]).copy()
    selected[[0, 1]] = selected[[1, 0]]
    _replace_result_array(bundle, filename, selected)

    replay = _run_control_bundle(bundle, expected_graph_k)
    assert replay["passed"] is False
    assert replay["file_bindings"]["round_results"]["passed"] is True
    assert replay["packed_pointwise"]["checks"][
        "frozen_squared_distance_then_id_order"
    ] is False
    assert replay["packed_pointwise"]["checks"]["hashes_match"] is False


def test_round_robin_greedy_order_change_is_rejected() -> None:
    bundle, expected_graph_k = _control_artifact_bundle()
    filename = "round_robin_greedy_ordered_atom_id.npy"
    ordered = _load_test_npy(bundle["round_results"][filename]).copy()
    ordered[[0, 1]] = ordered[[1, 0]]
    _replace_result_array(bundle, filename, ordered)

    replay = _run_control_bundle(bundle, expected_graph_k)
    assert replay["passed"] is False
    assert replay["round_robin_greedy"]["checks"][
        "atom_order_and_digest_bytes"
    ] is False
    assert replay["round_robin_greedy"]["checks"][
        "forward_rotation_and_slot_order"
    ] is True


def test_round_robin_greedy_rejects_support_reuse() -> None:
    bundle, expected_graph_k = _control_artifact_bundle()
    rank_name = "round_robin_greedy_support_rank.npy"
    id_name = "round_robin_greedy_support_id.npy"
    ranks = _load_test_npy(bundle["round_results"][rank_name]).copy()
    support_ids = _load_test_npy(bundle["round_results"][id_name]).copy()
    ranks[1] = ranks[0]
    support_ids[1] = support_ids[0]
    _replace_result_array(bundle, rank_name, ranks)
    _replace_result_array(bundle, id_name, support_ids)

    replay = _run_control_bundle(bundle, expected_graph_k)
    assert replay["passed"] is False
    assert replay["round_robin_greedy"]["checks"]["support_capacity_one"] is False
    assert replay["round_robin_greedy"]["checks"][
        "minimum_unused_edge_choice"
    ] is False


def test_round_robin_greedy_rejects_nonedge_tamper() -> None:
    bundle, expected_graph_k = _control_artifact_bundle()
    rank_name = "round_robin_greedy_support_rank.npy"
    id_name = "round_robin_greedy_support_id.npy"
    cost_name = "round_robin_greedy_edge_cost.npy"
    slot_name = "round_robin_greedy_slot_row.npy"
    ranks = _load_test_npy(bundle["round_results"][rank_name]).copy()
    support_ids = _load_test_npy(bundle["round_results"][id_name]).copy()
    costs = _load_test_npy(bundle["round_results"][cost_name]).copy()
    slot_rows = _load_test_npy(bundle["round_results"][slot_name])
    graph_indices = _load_test_npy(
        bundle["solver_inputs"]["graph_indices.npy"]
    )
    graph_indptr = _load_test_npy(
        bundle["solver_inputs"]["graph_indptr.npy"]
    )
    stable_ids = _load_test_npy(
        bundle["solver_inputs"]["support_stable_candidate_id.npy"]
    )
    row = int(slot_rows[0])
    neighbors = set(
        int(value)
        for value in graph_indices[graph_indptr[row] : graph_indptr[row + 1]]
    )
    forged_rank = next(
        rank for rank in range(stable_ids.size) if rank not in neighbors
    )
    ranks[0] = forged_rank
    support_ids[0] = stable_ids[forged_rank]
    costs[0] = 1
    _replace_result_array(bundle, rank_name, ranks)
    _replace_result_array(bundle, id_name, support_ids)
    _replace_result_array(bundle, cost_name, costs)

    replay = _run_control_bundle(bundle, expected_graph_k)
    assert replay["passed"] is False
    assert replay["round_robin_greedy"]["checks"][
        "reported_edge_membership_and_cost"
    ] is False


def test_round_robin_greedy_capacity_exhaustion_replays_without_repair() -> None:
    bundle, expected_graph_k = _control_artifact_bundle(
        exhaust_greedy=True,
        formal_count=True,
    )
    replay = _run_control_bundle(bundle, expected_graph_k)

    assert replay["round_robin_greedy"]["passed"] is True
    assert replay["round_robin_greedy"]["failed_slot_count"] == 1
    assert replay["round_robin_greedy"]["full_capacity"] is False
    assert replay["round_robin_greedy"]["checks"][
        "failed_slots_preserved"
    ] is True
    assert replay["exports"]["decision"]["point_count"] == 10_000
    assert replay["exports"]["packed_pointwise"]["point_count"] == 10_000
    assert replay["exports"]["round_robin_greedy"]["point_count"] == 9_999
    assert replay["exports"]["round_robin_greedy"]["point_count"] < 10_000
    assert replay["exports"]["round_robin_greedy"]["checks"][
        "required_count"
    ] is True
    assert replay["exports"]["round_robin_greedy"][
        "scientific_exact_required_count"
    ] is False
    assert replay["passed"] is True


def test_export_npy_bin_mismatch_is_rejected_after_file_rebinding() -> None:
    bundle, expected_graph_k = _control_artifact_bundle()
    arm = "packed_pointwise"
    filename = f"{arm}.npy"
    changed = _load_test_npy(bundle["exports"][filename]).copy()
    changed[0, 1] += np.float32(0.25)
    payload = _npy_payload(changed)
    bundle["exports"][filename] = payload
    record = bundle["oracle_report"]["exports"][arm]["npy"]
    record["bytes"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()

    replay = _run_control_bundle(bundle, expected_graph_k)
    export = replay["exports"][arm]
    assert replay["passed"] is False
    assert export["checks"]["npy_file_record"] is True
    assert export["checks"]["npy_bin_raw_bytes_identical"] is False


def test_export_duplicate_bytes_fail_unique_and_strict_spacing() -> None:
    bundle, expected_graph_k = _control_artifact_bundle()
    arm = "decision"
    changed = _load_test_npy(bundle["exports"][f"{arm}.npy"]).copy()
    changed[1] = changed[0]
    binary = changed.astype("<f4", copy=False).tobytes(order="C")
    npy = _npy_payload(changed)
    bundle["exports"][f"{arm}.bin"] = binary
    bundle["exports"][f"{arm}.npy"] = npy
    report = bundle["oracle_report"]["exports"][arm]
    report["bin"].update(
        bytes=len(binary), sha256=hashlib.sha256(binary).hexdigest()
    )
    report["npy"].update(bytes=len(npy), sha256=hashlib.sha256(npy).hexdigest())
    report["raw_xyz_sha256"] = hashlib.sha256(binary).hexdigest()

    replay = _run_control_bundle(bundle, expected_graph_k)
    export = replay["exports"][arm]
    assert replay["passed"] is False
    assert export["checks"]["npy_bin_raw_bytes_identical"] is True
    assert export["checks"]["unique_xyz_bytes"] is False
    assert export["checks"]["strict_spacing_5cm"] is False


def test_exact_spacing_handles_negative_cells_ulps_and_nominal_equality() -> None:
    zero = np.float32(0.0)
    five_cm = np.float32(0.05)
    below = np.nextafter(five_cm, np.float32(-np.inf))
    above = np.nextafter(five_cm, np.float32(np.inf))
    negative_five_cm = np.float32(-0.05)
    negative_inside = np.nextafter(negative_five_cm, zero)

    def replay(second_x: np.float32) -> dict[str, Any]:
        xyz = np.asarray([[zero, zero, zero], [second_x, zero, zero]], dtype="<f4")
        return verify_exact_spacing_bytes(xyz.tobytes(order="C"), point_count=2)

    assert replay(five_cm)["passed"] is True
    assert replay(above)["passed"] is True
    assert replay(below)["strict_spacing_5cm"] is False
    assert replay(negative_five_cm)["passed"] is True
    assert replay(negative_inside)["strict_spacing_5cm"] is False

    duplicate = np.asarray([[zero, zero, zero], [zero, zero, zero]], dtype="<f4")
    duplicate_report = verify_exact_spacing_bytes(duplicate.tobytes(order="C"))
    assert duplicate_report["unique_xyz_bytes"] is False
    assert duplicate_report["violation_count"] == 1


def test_packed_support_rejects_hash_and_semantic_byte_corruption() -> None:
    serialized = _valid_packed_support_bytes()
    assert verify_packed_support_bytes(serialized)["passed"] is True

    header_bytes, payload = serialized.split(b"\n", 1)
    hash_corruption = bytearray(payload)
    hash_corruption[-1] ^= 1
    with pytest.raises(ValueError, match="payload hash"):
        verify_packed_support_bytes(header_bytes + b"\n" + bytes(hash_corruption))

    duplicated_id = bytearray(payload)
    duplicated_id[8:16] = struct.pack("<q", 3)
    self_consistent_corruption = _rehash_packed_payload(
        serialized,
        bytes(duplicated_id),
    )
    with pytest.raises(ValueError, match="IDs are not strictly increasing"):
        verify_packed_support_bytes(self_consistent_corruption)


def test_independent_replay_covers_all_six_range_return_classes() -> None:
    azimuth_edges = (-0.8, -0.2, 0.2, 0.8)
    specifications = (
        (-0.5, 10.0, 12.0),
        (0.0, 40.0, 42.0),
        (0.5, 70.0, 72.0),
    )
    points = [
        _point(radius, azimuth)
        for azimuth, first, later in specifications
        for radius in (first, later)
    ]
    _, evaluation, _, replay = _evaluate_and_replay(
        np.asarray(points, dtype="<f4"),
        _target(points),
        azimuth_edges=azimuth_edges,
    )

    expected_classes = {
        "range_0_30_first",
        "range_0_30_later",
        "range_30_60_first",
        "range_30_60_later",
        "range_60_120_first",
        "range_60_120_later",
    }
    assert replay["passed"] is True
    assert replay["checks"]["mapping_is_independent_optimum"] is True
    assert replay["checks"]["all_input_output_hashes"] is True
    assert replay["computed_report"]["complete_mapping_sha256"] == (
        evaluation.complete_mapping_sha256
    )
    metrics = replay["computed_report"]["class_metrics"]
    assert {metric["key"] for metric in metrics} == expected_classes
    assert all(metric["applicable"] for metric in metrics)
    assert all(metric["passed"] is True for metric in metrics)


def test_structural_replay_is_row_permutation_neutral_after_raw_hash() -> None:
    output = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype="<f4",
    )
    target = np.asarray(
        [
            [1.0, -0.0, 0.0, 0.25],
            [1.0, 0.0, -0.0, 0.75],
            [2.0, 0.0, 0.0, 1.0],
            [3.0, -0.0, -0.0, 0.0],
        ],
        dtype="<f4",
    )
    _, _, _, first = _evaluate_and_replay(output, target)
    _, _, _, second = _evaluate_and_replay(
        output[[3, 2, 0, 1]],
        target[[2, 0, 3, 1]],
    )

    assert first["passed"] is True
    assert second["passed"] is True
    first_report = first["computed_report"]
    second_report = second["computed_report"]
    assert first_report["raw_inputs_sha256"] != second_report["raw_inputs_sha256"]
    assert first_report["canonical_inputs_sha256"] == (
        second_report["canonical_inputs_sha256"]
    )
    assert first_report["complete_mapping_sha256"] == (
        second_report["complete_mapping_sha256"]
    )
    assert first_report["result_sha256"] == second_report["result_sha256"]


def test_equal_range_tie_uses_complete_canonical_assignment_tuple() -> None:
    output = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [6.0, 8.0, 0.0],
            [8.0, 6.0, 0.0],
        ],
        dtype="<f4",
    )
    target = np.asarray(
        [
            [3.0, 4.0, 0.0, 1.0],
            [6.0, 8.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    _, evaluation, _, replay = _evaluate_and_replay(
        output,
        target,
        azimuth_edges=(0.0, 1.5),
    )

    ten_meter_events = sorted(
        event["event_id"]
        for event in replay["canonical_output_events"]
        if event["range_m"] == 10.0
    )
    assert replay["passed"] is True
    assert len(ten_meter_events) == 2
    assert replay["computed_report"]["complete_assignment_tuple"][1] == (
        ten_meter_events[0]
    )
    assert replay["computed_report"]["objective_key"] == [
        evaluation.integer_cost,
        -evaluation.matched_count,
        list(evaluation.complete_assignment_tuple),
    ]


def test_replay_handles_skipped_events_and_unmatched_groups() -> None:
    output = np.asarray(
        [[4.0, 3.0, 0.0], [8.0, 6.0, 0.0], [16.0, 12.0, 0.0]],
        dtype="<f4",
    )
    target = np.asarray(
        [[4.0, 3.0, 0.0, 1.0], [16.0, 12.0, 0.0, 1.0]],
        dtype="<f4",
    )
    _, _, _, replay = _evaluate_and_replay(
        output,
        target,
        azimuth_edges=(0.0, 1.5),
    )
    assignments = replay["computed_report"]["assignments"]
    skipped = next(
        event
        for event in replay["canonical_output_events"]
        if event["inferred_depth"] == 2
    )

    assert replay["passed"] is True
    assert [row["matched_depth"] for row in assignments] == [1, 3]
    assert skipped["event_id"] not in replay["computed_report"][
        "complete_assignment_tuple"
    ]

    _, _, _, short_replay = _evaluate_and_replay(
        output[:1],
        target,
        azimuth_edges=(0.0, 1.5),
    )
    short_assignments = short_replay["computed_report"]["assignments"]
    assert short_replay["passed"] is True
    assert short_assignments[1]["matched_event_id"] == UNMATCHED_SENTINEL
    assert short_assignments[1]["assigned_distance_m"] == 120.0


def test_consecutive_target_grouping_is_strict_at_five_centimeters() -> None:
    empty_output = np.empty((0, 3), dtype="<f4")
    five_cm = np.float32(0.05)
    below = np.nextafter(five_cm, np.float32(0.0))
    separate_target = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [five_cm, 0.0, 0.0, 1.0]],
        dtype="<f4",
    )
    grouped_target = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [below, 0.0, 0.0, 1.0]],
        dtype="<f4",
    )
    _, _, _, separate = _evaluate_and_replay(empty_output, separate_target)
    _, _, _, grouped = _evaluate_and_replay(empty_output, grouped_target)

    assert separate["passed"] is True
    assert grouped["passed"] is True
    assert len(separate["computed_report"]["groups"]) == 2
    assert len(grouped["computed_report"]["groups"]) == 1


def test_output_target_range_and_angular_boundaries_fail_closed() -> None:
    azimuth_edges = (-0.5 * math.pi, 0.5 * math.pi)
    elevation_edges = (-0.5 * math.pi, 0.5 * math.pi)
    valid_output = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [EVALUATOR_OUTPUT_RANGE_UPPER_M, 0.0, 0.0],
        ],
        dtype="<f4",
    )
    below_120 = np.nextafter(np.float32(120.0), np.float32(0.0))
    valid_target = np.asarray([[below_120, 0.0, 0.0, 1.0]], dtype="<f4")
    _, _, _, valid = _evaluate_and_replay(
        valid_output,
        valid_target,
        azimuth_edges=azimuth_edges,
        elevation_edges=elevation_edges,
    )
    assert valid["passed"] is True
    assert valid["structural_domain_valid"] is True

    above_output = np.nextafter(
        np.float32(EVALUATOR_OUTPUT_RANGE_UPPER_M),
        np.float32(np.inf),
    )
    invalid_output = np.asarray(
        [[above_output, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype="<f4",
    )
    invalid_target = np.asarray([[120.0, 0.0, 0.0, 1.0]], dtype="<f4")
    _, evaluation, _, invalid = _evaluate_and_replay(
        invalid_output,
        invalid_target,
        azimuth_edges=azimuth_edges,
        elevation_edges=elevation_edges,
    )

    assert invalid["passed"] is True
    assert invalid["structural_domain_valid"] is False
    assert invalid["evaluation_complete"] is False
    assert evaluation.failure_reasons == (
        "output_event_out_of_structural_domain",
        "positive_target_out_of_structural_domain",
    )
    assert invalid["computed_report"]["failure_reasons"] == list(
        evaluation.failure_reasons
    )


def test_corrupted_expected_mapping_report_fails_each_replay_layer() -> None:
    output = np.asarray([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype="<f4")
    target = np.asarray(
        [[5.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]],
        dtype="<f4",
    )
    inputs, _, expected, replay = _evaluate_and_replay(output, target)
    assert replay["passed"] is True

    tuple_corruption = copy.deepcopy(expected)
    tuple_corruption["complete_assignment_tuple"][0] = UNMATCHED_SENTINEL
    tuple_result = verify_structural_replay(
        inputs.output_xyz_le_f4,
        inputs.target_xyz_confidence_le_f4,
        (-0.2, 0.2),
        (-0.2, 0.2),
        tuple_corruption,
    )
    assert tuple_result["passed"] is False
    assert tuple_result["checks"]["mapping_tuple_consistent"] is False

    distance_corruption = copy.deepcopy(expected)
    distance_corruption["assignments"][0]["assigned_distance_m"] += 0.5
    distance_result = verify_structural_replay(
        inputs.output_xyz_le_f4,
        inputs.target_xyz_confidence_le_f4,
        (-0.2, 0.2),
        (-0.2, 0.2),
        distance_corruption,
    )
    assert distance_result["passed"] is False
    assert distance_result["checks"]["mapping_distances"] is False

    objective_corruption = copy.deepcopy(expected)
    objective_corruption["assignments"][0]["integer_cost"] += 1
    objective_result = verify_structural_replay(
        inputs.output_xyz_le_f4,
        inputs.target_xyz_confidence_le_f4,
        (-0.2, 0.2),
        (-0.2, 0.2),
        objective_corruption,
    )
    assert objective_result["passed"] is False
    assert objective_result["checks"]["mapping_entry_objectives"] is False

    reuse_corruption = copy.deepcopy(expected)
    first_event_id = reuse_corruption["assignments"][0]["matched_event_id"]
    reuse_corruption["assignments"][1]["matched_event_id"] = first_event_id
    reuse_corruption["complete_assignment_tuple"][1] = first_event_id
    reuse_corruption["objective_key"][2][1] = first_event_id
    reuse_result = verify_structural_replay(
        inputs.output_xyz_le_f4,
        inputs.target_xyz_confidence_le_f4,
        (-0.2, 0.2),
        (-0.2, 0.2),
        reuse_corruption,
    )
    assert reuse_result["passed"] is False
    assert reuse_result["checks"]["mapping_event_nonreuse"] is False
    assert reuse_result["checks"]["mapping_eligibility"] is False
    assert reuse_result["checks"]["mapping_event_order"] is False
    assert reuse_result["checks"]["mapping_hash_matches_rows"] is False

    order_corruption = copy.deepcopy(expected)
    order_corruption["assignments"].reverse()
    order_result = verify_structural_replay(
        inputs.output_xyz_le_f4,
        inputs.target_xyz_confidence_le_f4,
        (-0.2, 0.2),
        (-0.2, 0.2),
        order_corruption,
    )
    assert order_result["passed"] is False
    assert order_result["checks"]["mapping_canonical_group_order"] is False
