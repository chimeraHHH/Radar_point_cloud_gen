from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from eval.stda_f0_fit import (
    BASE_FREEZE_COMMIT,
    DEMAND_SLOT_COUNT,
    EDGE_COST_EXACT_LIMIT,
    FROZEN_PROTOCOL_SHA256,
    GRAPH_EDGE_COUNT,
    GRAPH_NEIGHBOR_COUNT,
    build_midpoint_demand_slots,
    build_packed_pointwise_sidecar,
    build_sparse_assignment_graph,
    canonicalize_support,
    canonicalize_target_atoms,
    integer_edge_cost,
    load_target_xyz_confidence,
    load_target_xyz_confidence_bytes,
    select_frozen_neighbors,
    validate_full_assignment_objective_bound,
)


def _single_atom_target(confidence: float = 1.0) -> np.ndarray:
    return np.asarray([[0.0, 0.0, 0.0, confidence]], dtype="<f4")


def _duplicate_support(
    count: int = GRAPH_NEIGHBOR_COUNT + 44,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.zeros((count, 3), dtype="<f4")
    stable_id = np.arange(count, dtype="<i8")
    return xyz, stable_id


def test_source_binds_exact_freeze_constants_and_one_explicit_target_loader() -> None:
    assert BASE_FREEZE_COMMIT == "eb839e1e806c44dd1668051085a307faf4fe83a0"
    assert FROZEN_PROTOCOL_SHA256 == (
        "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
    )
    source = inspect.getsource(inspect.getmodule(canonicalize_target_atoms))
    assert source is not None
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("round" in name for name in imported)
    assert not any("metric" in name for name in imported)
    loader_source = inspect.getsource(load_target_xyz_confidence)
    byte_loader_source = inspect.getsource(load_target_xyz_confidence_bytes)
    assert "load_target_xyz_confidence_bytes" in loader_source
    assert 'cache["target_xyz_confidence"]' in byte_loader_source
    assert "target_rae_index" not in byte_loader_source
    assert "cfar" not in byte_loader_source.lower()
    assert "cube" not in byte_loader_source.lower()


def test_target_loader_is_support_gated_and_reads_only_approved_array(
    tmp_path: Path,
) -> None:
    path = tmp_path / "target.npz"
    target = np.asarray([[1.0, -0.0, 0.0, 0.75]], dtype="<f4")
    np.savez(
        path,
        target_xyz_confidence=target,
        target_rae_index=np.zeros((1, 3), dtype="<i8"),
        cfar_xyzd_power_snr=np.ones((1, 6), dtype="<f4"),
    )

    loaded = load_target_xyz_confidence(
        path,
        support_commit_sha256="ab" * 32,
    )

    np.testing.assert_array_equal(loaded.target_xyz_confidence, target)
    assert not loaded.target_xyz_confidence.flags.writeable
    assert loaded.cache_arrays_read == ("target_xyz_confidence",)
    assert loaded.cache_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="support commitment"):
        load_target_xyz_confidence(path, support_commit_sha256="invalid")


def test_target_and_demand_hashes_ignore_row_permutation() -> None:
    target = np.asarray(
        [
            [3.0, 0.0, 0.0, 0.25],
            [1.0, 0.0, 0.0, 0.50],
            [3.0, 0.0, 0.0, 0.75],
            [2.0, 0.0, 0.0, 0.25],
        ],
        dtype="<f4",
    )
    permutation = np.asarray([2, 0, 3, 1])

    first_atoms = canonicalize_target_atoms(target)
    second_atoms = canonicalize_target_atoms(target[permutation])
    first_demand = build_midpoint_demand_slots(first_atoms)
    second_demand = build_midpoint_demand_slots(second_atoms)

    assert first_atoms.digest_sha256 == second_atoms.digest_sha256
    assert first_atoms.canonical_bytes() == second_atoms.canonical_bytes()
    assert first_demand.digest_sha256 == second_demand.digest_sha256
    assert first_demand.canonical_bytes() == second_demand.canonical_bytes()
    np.testing.assert_array_equal(
        first_demand.slot_id,
        np.arange(DEMAND_SLOT_COUNT, dtype="<i8"),
    )


def test_midpoint_cdf_creates_exact_mass_proportional_slots() -> None:
    target = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 3.0],
        ],
        dtype="<f4",
    )

    atoms = canonicalize_target_atoms(target)
    demand = build_midpoint_demand_slots(atoms)

    np.testing.assert_array_equal(
        np.bincount(demand.canonical_atom_id),
        [2_500, 7_500],
    )
    assert demand.slot_id.dtype.str == "<i8"
    assert demand.canonical_atom_id.dtype.str == "<i8"
    assert demand.slot_xyz.dtype.str == "<f8"
    assert not demand.slot_xyz.flags.writeable


def test_signed_zero_and_zero_confidence_rows_are_hash_neutral() -> None:
    positive_zero = np.asarray(
        [[0.0, 0.0, 0.0, 1.0]],
        dtype="<f4",
    )
    signed_and_zero_mass = np.asarray(
        [
            [-0.0, 0.0, -0.0, 1.0],
            [12.0, -0.0, 4.0, -0.0],
            [-7.0, 8.0, -9.0, 0.0],
        ],
        dtype="<f4",
    )

    reference = canonicalize_target_atoms(positive_zero)
    observed = canonicalize_target_atoms(signed_and_zero_mass)

    assert observed.zero_confidence_row_count == 2
    assert observed.digest_sha256 == reference.digest_sha256
    assert observed.canonical_bytes() == reference.canonical_bytes()
    assert not np.signbit(observed.xyz_float32).any()


def test_bit_identical_fsum_split_merge_preserves_atoms_and_demand() -> None:
    merged = np.asarray([[4.0, 5.0, 6.0, 1.0]], dtype="<f4")
    split = np.asarray(
        [
            [4.0, 5.0, 6.0, 0.75],
            [4.0, 5.0, 6.0, 0.25],
        ],
        dtype="<f4",
    )

    merged_atoms = canonicalize_target_atoms(merged)
    split_atoms = canonicalize_target_atoms(split[::-1].copy())
    merged_demand = build_midpoint_demand_slots(merged_atoms)
    split_demand = build_midpoint_demand_slots(split_atoms)

    assert merged_atoms.aggregate_weight[0] == split_atoms.aggregate_weight[0]
    assert merged_atoms.digest_sha256 == split_atoms.digest_sha256
    assert merged_demand.digest_sha256 == split_demand.digest_sha256


@pytest.mark.parametrize(
    "target",
    [
        np.asarray([[0.0, 0.0, 0.0, -1.0]], dtype="<f4"),
        np.asarray([[0.0, 0.0, 0.0, np.nan]], dtype="<f4"),
        np.asarray([[0.0, 0.0, 0.0, np.inf]], dtype="<f4"),
        np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype="<f4"),
    ],
)
def test_invalid_target_mass_is_rejected(target: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonicalize_target_atoms(target)


def test_more_than_k_exact_ties_use_smallest_stable_candidate_ids() -> None:
    xyz, stable_id = _duplicate_support()
    permutation = np.arange(stable_id.size)[::-1]

    selected = select_frozen_neighbors(
        np.zeros(3, dtype="<f8"),
        xyz[permutation],
        stable_id[permutation],
    )

    np.testing.assert_array_equal(
        selected.stable_candidate_id,
        np.arange(GRAPH_NEIGHBOR_COUNT, dtype="<i8"),
    )
    np.testing.assert_array_equal(
        selected.squared_distance_m2,
        np.zeros(GRAPH_NEIGHBOR_COUNT, dtype="<f8"),
    )


def test_exact_ties_and_ulp_separation_have_no_tolerance() -> None:
    next_float32 = np.nextafter(np.float32(1.0), np.float32(np.inf))
    filler_count = GRAPH_NEIGHBOR_COUNT - 3
    xyz = np.vstack(
        (
            np.asarray(
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [next_float32, 0.0, 0.0]],
                dtype="<f4",
            ),
            np.full((filler_count, 3), 2.0, dtype="<f4"),
        )
    )
    stable_id = np.concatenate(
        (
            np.asarray([100, 3, 1], dtype="<i8"),
            np.arange(1_000, 1_000 + filler_count, dtype="<i8"),
        )
    )

    selected = select_frozen_neighbors(
        np.zeros(3, dtype="<f8"),
        xyz,
        stable_id,
    )

    np.testing.assert_array_equal(selected.stable_candidate_id[:3], [3, 100, 1])
    assert selected.squared_distance_m2[0] == selected.squared_distance_m2[1]
    assert selected.squared_distance_m2[2] > selected.squared_distance_m2[1]


def test_canonical_support_is_permutation_and_byte_order_invariant() -> None:
    xyz = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=">f4",
    )
    xyz = np.repeat(xyz, GRAPH_NEIGHBOR_COUNT // 3 + 1, axis=0)[
        :GRAPH_NEIGHBOR_COUNT
    ]
    stable_id = np.arange(GRAPH_NEIGHBOR_COUNT, dtype=">i8")
    permutation = np.roll(np.arange(GRAPH_NEIGHBOR_COUNT), 73)

    first = canonicalize_support(xyz, stable_id)
    second = canonicalize_support(xyz[permutation], stable_id[permutation])

    assert first.digest_sha256 == second.digest_sha256
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.xyz_float32.dtype.str == "<f4"
    assert first.stable_candidate_id.dtype.str == "<i8"


def test_csr_bytes_are_deterministic_under_support_permutation() -> None:
    atoms = canonicalize_target_atoms(_single_atom_target())
    demand = build_midpoint_demand_slots(atoms)
    xyz, stable_id = _duplicate_support(GRAPH_NEIGHBOR_COUNT)
    permutation = np.roll(np.arange(GRAPH_NEIGHBOR_COUNT), 91)

    first = build_sparse_assignment_graph(demand, xyz, stable_id)
    second = build_sparse_assignment_graph(
        demand,
        xyz[permutation],
        stable_id[permutation],
    )

    assert first.shape == (DEMAND_SLOT_COUNT, GRAPH_NEIGHBOR_COUNT)
    assert first.edge_count == GRAPH_EDGE_COUNT
    assert first.indptr.dtype.str == "<i8"
    assert first.indices.dtype.str == "<i4"
    assert first.data.dtype.str == "<i8"
    assert first.digest_sha256 == second.digest_sha256
    assert first.canonical_bytes() == second.canonical_bytes()


def test_integer_cost_rejects_edge_and_objective_overflow_inputs() -> None:
    assert integer_edge_cost(
        0.0,
        support_cardinality=GRAPH_NEIGHBOR_COUNT,
        support_rank=0,
    ) == 1
    with pytest.raises(OverflowError):
        integer_edge_cost(
            float(EDGE_COST_EXACT_LIMIT),
            support_cardinality=GRAPH_NEIGHBOR_COUNT,
            support_rank=0,
        )
    with pytest.raises(ValueError):
        integer_edge_cost(
            -1.0,
            support_cardinality=GRAPH_NEIGHBOR_COUNT,
            support_rank=0,
        )
    with pytest.raises(OverflowError):
        validate_full_assignment_objective_bound(
            [EDGE_COST_EXACT_LIMIT - 1] * DEMAND_SLOT_COUNT
        )


def test_pointwise_sidecar_uses_exact_tie_break_and_canonical_bytes() -> None:
    target = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    atoms = canonicalize_target_atoms(target)
    support_xyz, stable_id = _duplicate_support(DEMAND_SLOT_COUNT)
    permutation = np.roll(np.arange(DEMAND_SLOT_COUNT), 317)

    first = build_packed_pointwise_sidecar(atoms, support_xyz, stable_id)
    second = build_packed_pointwise_sidecar(
        atoms,
        support_xyz[permutation],
        stable_id[permutation],
    )

    assert np.all(first.nearest_atom_id == 0)
    assert np.all(first.squared_distance_m2 == 1.0)
    assert np.all(first.distance_m == 1.0)
    np.testing.assert_array_equal(
        first.selected_support_ranks(),
        np.arange(DEMAND_SLOT_COUNT, dtype="<i8"),
    )
    assert first.digest_sha256 == second.digest_sha256
    assert first.canonical_bytes() == second.canonical_bytes()
