from __future__ import annotations

import ast
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import eval.stda_f0_round as stda_round
from eval.stda_f0_round import (
    AssignmentGraph,
    GreedySidecar,
    PackedSupport,
    PointwiseSidecar,
    build_hall_witness,
    deterministic_hopcroft_karp,
    greedy_atom_order,
    load_assignment_graph,
    load_greedy_sidecar,
    load_packed_support,
    load_pointwise_sidecar,
    maximum_cardinality_certificate,
    packed_pointwise_control,
    scipy_maximum_matching,
    solve_full_assignment,
    target_atom_round_robin_greedy,
    verify_full_assignment,
    verify_greedy_control,
    verify_hall_witness,
    verify_matching,
    verify_packed_pointwise,
)


def synthetic_support(count: int) -> PackedSupport:
    xyz = np.zeros((count, 3), dtype="<f4")
    xyz[:, 0] = np.arange(count, dtype=np.float32) * np.float32(0.10)
    return PackedSupport(
        stable_candidate_id=np.arange(100, 100 + count, dtype="<i8"),
        grid_cell=np.column_stack(
            (
                2 * np.arange(count, dtype="<i8"),
                np.zeros(count, dtype="<i8"),
                np.zeros(count, dtype="<i8"),
            )
        ),
        xyz=xyz,
        base_confidence=np.ones(count, dtype="<f4"),
        color=np.zeros(count, dtype="<u1"),
    )


def synthetic_graph(
    rows: list[list[int]],
    *,
    support_count: int,
    costs: list[list[int]] | None = None,
) -> AssignmentGraph:
    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    squared_distances: list[float] = []
    distances: list[float] = []
    for row_index, neighbors in enumerate(rows):
        row_costs = costs[row_index] if costs is not None else list(
            range(1, len(neighbors) + 1)
        )
        paired = sorted(zip(neighbors, row_costs), key=lambda item: item[0])
        indices.extend(item[0] for item in paired)
        data.extend(item[1] for item in paired)
        row_squared = [
            (float(item[1]) / 1_000_000.0) ** 2 for item in paired
        ]
        squared_distances.extend(row_squared)
        distances.extend(math.sqrt(value) for value in row_squared)
        indptr.append(len(indices))
    return AssignmentGraph(
        indptr=np.asarray(indptr, dtype="<i8"),
        indices=np.asarray(indices, dtype="<i4"),
        data=np.asarray(data, dtype="<i8"),
        edge_squared_distance_m2=np.asarray(squared_distances, dtype="<f8"),
        edge_distance_m=np.asarray(distances, dtype="<f8"),
        slot_id=np.arange(len(rows), dtype="<i8"),
        support_cardinality=support_count,
    )


def synthetic_greedy_sidecar(
    slot_atom_id: list[int],
    *,
    atom_count: int = 3,
) -> GreedySidecar:
    xyz = np.zeros((atom_count, 3), dtype="<f8")
    xyz[:, 0] = np.arange(1, atom_count + 1, dtype=np.float64)
    return GreedySidecar(
        slot_atom_id=np.asarray(slot_atom_id, dtype="<i8"),
        atom_id=np.arange(atom_count, dtype="<i8"),
        atom_xyz=xyz,
        atom_weight=np.arange(1, atom_count + 1, dtype="<f8"),
    )


def test_full_graph_custom_and_scipy_cardinality_agree() -> None:
    support = synthetic_support(4)
    graph = synthetic_graph(
        [[0, 1], [1, 2], [0, 2, 3]],
        support_count=support.count,
    )

    custom = deterministic_hopcroft_karp(graph)
    scipy = scipy_maximum_matching(graph)
    certificate = maximum_cardinality_certificate(support, graph)

    assert custom.cardinality == scipy.cardinality == 3
    assert verify_matching(graph, custom).passed
    assert verify_matching(graph, scipy).passed
    assert certificate.transported_mass == 3
    assert certificate.unmatched_target_mass == 0
    assert certificate.unused_source_mass == 1
    assert certificate.hall_witness is None


def test_deficient_graph_emits_verified_hall_witness() -> None:
    support = synthetic_support(3)
    graph = synthetic_graph([[0], [0], [1]], support_count=support.count)

    certificate = maximum_cardinality_certificate(support, graph)

    assert certificate.transported_mass == 2
    assert certificate.unmatched_target_mass == 1
    assert certificate.hall_witness is not None
    witness = certificate.hall_witness
    assert witness.hall_deficit == 1
    assert witness.global_deficit == 1
    np.testing.assert_array_equal(witness.slot_rows, [0, 1])
    np.testing.assert_array_equal(witness.support_ranks, [0])
    assert verify_hall_witness(
        support,
        graph,
        certificate.custom_matching,
        witness,
    ).passed


def test_hall_replay_rejects_a_truncated_full_neighborhood() -> None:
    support = synthetic_support(3)
    graph = synthetic_graph([[0], [0], [1]], support_count=support.count)
    matching = deterministic_hopcroft_karp(graph)
    witness = build_hall_witness(support, graph, matching)

    forged = stda_round.HallWitness(
        slot_rows=witness.slot_rows,
        support_ranks=np.asarray([], dtype="<i8"),
        slot_ids=witness.slot_ids,
        support_ids=np.asarray([], dtype="<i8"),
        hall_deficit=2,
        global_deficit=1,
        slot_set_sha256=witness.slot_set_sha256,
        support_set_sha256=witness.support_set_sha256,
        matching_sha256=witness.matching_sha256,
        digest_sha256=witness.digest_sha256,
    )

    checks = verify_hall_witness(support, graph, matching, forged)
    assert not checks.passed
    assert not checks.full_neighborhood_equal
    assert not checks.global_deficit_bounds_hall


def test_sparse_full_assignment_is_repeatable_and_replayable() -> None:
    support = synthetic_support(4)
    graph = synthetic_graph(
        [[0, 1], [0, 1], [2, 3]],
        support_count=support.count,
        costs=[[1, 100], [100, 1], [1, 2]],
    )

    first = solve_full_assignment(support, graph)
    second = solve_full_assignment(support, graph)

    np.testing.assert_array_equal(first.support_rank, [0, 1, 2])
    assert first.objective == 3
    assert first.assignment_sha256 == second.assignment_sha256
    assert first.selected_id_sha256 == second.selected_id_sha256
    assert first.export_sha256 == second.export_sha256
    assert first.objective_sha256 == second.objective_sha256
    assert verify_full_assignment(
        support,
        graph,
        second,
        reference=first,
    ).passed


def test_packed_pointwise_orders_exact_ties_by_stable_candidate_id() -> None:
    support = synthetic_support(5)
    squared = np.asarray([2.0, 1.0, 1.0, 0.5, 1.0], dtype="<f8")
    sidecar = PointwiseSidecar(
        nearest_atom_id=np.asarray([4, 3, 2, 1, 0], dtype="<i8"),
        squared_distance=squared,
        distance_m=np.asarray(
            [math.sqrt(float(value)) for value in squared], dtype="<f8"
        ),
    )

    first = packed_pointwise_control(support, sidecar, output_count=3)
    second = packed_pointwise_control(support, sidecar, output_count=3)

    np.testing.assert_array_equal(first.selected_support_rank, [3, 1, 2])
    np.testing.assert_array_equal(first.selected_support_id, [103, 101, 102])
    assert verify_packed_pointwise(
        support,
        sidecar,
        second,
        reference=first,
    ).passed


def test_greedy_atom_order_matches_exact_domain_separated_hash() -> None:
    sidecar = synthetic_greedy_sidecar([0, 1, 2])
    result = greedy_atom_order(sidecar, sequence=1, radar_index=232)
    frame_key = b"seq01/radar00232"
    keyed = []
    for atom_id in range(3):
        atom_bytes = (
            np.asarray(sidecar.atom_xyz[atom_id], dtype="<f4").tobytes()
            + np.asarray([sidecar.atom_weight[atom_id]], dtype="<f8").tobytes()
        )
        digest = hashlib.sha256(
            b"stda_f0_greedy_atom_order_v1\0"
            + frame_key
            + b"\0"
            + atom_bytes
        ).digest()
        keyed.append((digest, atom_id))
    keyed.sort(key=lambda item: (item[0], item[1]))

    assert result.frame_key == frame_key
    np.testing.assert_array_equal(
        result.ordered_atom_id,
        [item[1] for item in keyed],
    )
    assert result.ordered_digest_bytes[0].tobytes() == keyed[0][0]


def test_greedy_starts_at_r_zero_rotates_forward_and_skips_exhausted_atoms() -> None:
    seed_sidecar = synthetic_greedy_sidecar([0, 1, 2])
    order = greedy_atom_order(seed_sidecar, sequence=1, radar_index=232)
    first, second, third = [int(value) for value in order.ordered_atom_id]
    sidecar = synthetic_greedy_sidecar([first, second, third, third, first])
    support = synthetic_support(5)
    graph = synthetic_graph(
        [[0], [1], [2], [3], [4]],
        support_count=support.count,
    )

    result = target_atom_round_robin_greedy(
        support,
        graph,
        sidecar,
        sequence=1,
        radar_index=232,
    )

    np.testing.assert_array_equal(result.round_index, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(
        result.atom_id,
        [first, second, third, third, first],
    )
    np.testing.assert_array_equal(result.slot_id, [0, 1, 2, 3, 4])
    assert result.full_capacity
    assert verify_greedy_control(
        support,
        graph,
        sidecar,
        result,
        sequence=1,
        radar_index=232,
    ).passed


def test_greedy_failed_slot_is_consumed_and_forward_visit_continues() -> None:
    seed_sidecar = synthetic_greedy_sidecar([0, 1], atom_count=2)
    order = greedy_atom_order(seed_sidecar, sequence=4, radar_index=17)
    first, second = [int(value) for value in order.ordered_atom_id]
    sidecar = synthetic_greedy_sidecar([first, second], atom_count=2)
    support = synthetic_support(2)
    graph = synthetic_graph([[], [0]], support_count=support.count)

    result = target_atom_round_robin_greedy(
        support,
        graph,
        sidecar,
        sequence=4,
        radar_index=17,
    )

    np.testing.assert_array_equal(result.atom_id, [first, second])
    np.testing.assert_array_equal(result.support_rank, [-1, 0])
    assert result.failed_slot_count == 1
    assert result.slot_row.size == 2
    assert not result.full_capacity
    assert verify_greedy_control(
        support,
        graph,
        sidecar,
        result,
        sequence=4,
        radar_index=17,
    ).passed


def test_greedy_records_candidate_exhaustion_without_repair() -> None:
    seed_sidecar = synthetic_greedy_sidecar([0, 1], atom_count=2)
    order = greedy_atom_order(seed_sidecar, sequence=2, radar_index=8)
    first, second = [int(value) for value in order.ordered_atom_id]
    sidecar = synthetic_greedy_sidecar([first, second], atom_count=2)
    support = synthetic_support(2)
    graph = synthetic_graph([[0], [0]], support_count=support.count)

    result = target_atom_round_robin_greedy(
        support,
        graph,
        sidecar,
        sequence=2,
        radar_index=8,
    )

    np.testing.assert_array_equal(result.support_rank, [0, -1])
    assert result.failed_slot_count == 1
    assert result.selected_support_rank.size == 1


def _write_npy(path: Path, values: np.ndarray) -> str:
    np.save(path, values, allow_pickle=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_file_loaders_return_hash_bound_read_only_arrays(tmp_path: Path) -> None:
    support = synthetic_support(3)
    support_values = {
        "support_stable_candidate_id.npy": support.stable_candidate_id,
        "support_grid_cell.npy": support.grid_cell,
        "support_xyz.npy": support.xyz,
        "support_base_confidence.npy": support.base_confidence,
        "support_color.npy": support.color,
    }
    support_hashes = {
        name: _write_npy(tmp_path / name, value)
        for name, value in support_values.items()
    }
    graph = synthetic_graph([[0], [1]], support_count=support.count)
    graph_values = {
        "graph_indptr.npy": graph.indptr,
        "graph_indices.npy": graph.indices,
        "graph_data.npy": graph.data,
        "graph_edge_squared_distance_m2.npy": graph.edge_squared_distance_m2,
        "graph_edge_distance_m.npy": graph.edge_distance_m,
        "demand_slot_id.npy": graph.slot_id,
    }
    graph_hashes = {
        name: _write_npy(tmp_path / name, value)
        for name, value in graph_values.items()
    }
    greedy = synthetic_greedy_sidecar([0, 1], atom_count=2)
    greedy_values = {
        "demand_slot_atom_id.npy": greedy.slot_atom_id,
        "demand_atom_id.npy": greedy.atom_id,
        "demand_atom_xyz.npy": greedy.atom_xyz,
        "demand_atom_weight.npy": greedy.atom_weight,
    }
    greedy_hashes = {
        name: _write_npy(tmp_path / name, value)
        for name, value in greedy_values.items()
    }
    squared = np.asarray([0.0, 1.0, 4.0], dtype="<f8")
    pointwise_values = {
        "pointwise_nearest_atom_id.npy": np.asarray([0, 0, 1], dtype="<i8"),
        "pointwise_squared_distance.npy": squared,
        "pointwise_distance_m.npy": np.asarray([0.0, 1.0, 2.0], dtype="<f8"),
    }
    pointwise_hashes = {
        name: _write_npy(tmp_path / name, value)
        for name, value in pointwise_values.items()
    }

    loaded_support = load_packed_support(
        tmp_path, expected_sha256=support_hashes
    )
    loaded_graph = load_assignment_graph(
        tmp_path,
        support_cardinality=loaded_support.count,
        expected_sha256=graph_hashes,
    )
    loaded_greedy = load_greedy_sidecar(
        tmp_path, expected_sha256=greedy_hashes
    )
    loaded_pointwise = load_pointwise_sidecar(
        tmp_path, expected_sha256=pointwise_hashes
    )

    assert not loaded_support.xyz.flags.writeable
    assert not loaded_graph.indices.flags.writeable
    assert not loaded_greedy.atom_weight.flags.writeable
    assert not loaded_pointwise.squared_distance.flags.writeable
    with pytest.raises(ValueError):
        loaded_support.xyz[0, 0] = 99.0
    forged = dict(support_hashes)
    forged["support_xyz.npy"] = "00" * 32
    with pytest.raises(ValueError):
        load_packed_support(tmp_path, expected_sha256=forged)


def test_round_module_has_no_fitter_loader_metric_or_structure_dependency() -> None:
    assert stda_round.PROTOCOL_FREEZE_COMMIT == (
        "eb839e1e806c44dd1668051085a307faf4fe83a0"
    )
    assert stda_round.PROTOCOL_SHA256 == (
        "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
    )
    assert stda_round.BASE_FREEZE_COMMIT == stda_round.PROTOCOL_FREEZE_COMMIT
    assert stda_round.FROZEN_PROTOCOL_SHA256 == stda_round.PROTOCOL_SHA256
    source = inspect.getsource(stda_round)
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    forbidden_import_fragments = (
        "stda_f0_fit",
        "stda_f0_structure",
        "stda_f0_verify",
        "vrh_f0_fit",
        "vrh_f0_metrics",
        "dense_geometry",
        "cube_dense",
    )
    assert not any(
        fragment in module
        for module in imported
        for fragment in forbidden_import_fragments
    )
    assert "load_target" not in source
    assert "target_xyz" not in source
    decision_functions = (
        deterministic_hopcroft_karp,
        scipy_maximum_matching,
        maximum_cardinality_certificate,
        solve_full_assignment,
    )
    for function in decision_functions:
        parameter_names = set(inspect.signature(function).parameters)
        assert not any("target" in name for name in parameter_names)
        assert not any("metric" in name for name in parameter_names)
