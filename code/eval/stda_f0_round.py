"""Deterministic solver and control arms for the frozen STDA-F0 gate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import struct
from typing import Mapping

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    maximum_bipartite_matching,
    min_weight_full_bipartite_matching,
)


BASE_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
FROZEN_PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
PROTOCOL_FREEZE_COMMIT = BASE_FREEZE_COMMIT
PROTOCOL_SHA256 = FROZEN_PROTOCOL_SHA256

FORMAL_SLOT_COUNT = 10_000
FORMAL_NEIGHBOR_COUNT = 256
MAX_EXACT_INTEGER = 1 << 53
MAX_SIGNED_INT64 = 1 << 63

SUPPORT_ARRAY_FILES = (
    ("stable_candidate_id", "support_stable_candidate_id.npy", "<i8", 1),
    ("grid_cell", "support_grid_cell.npy", "<i8", 2),
    ("xyz", "support_xyz.npy", "<f4", 2),
    ("base_confidence", "support_base_confidence.npy", "<f4", 1),
    ("color", "support_color.npy", "<u1", 1),
)
GRAPH_ARRAY_FILES = (
    ("indptr", "graph_indptr.npy", "<i8", 1),
    ("indices", "graph_indices.npy", "<i4", 1),
    ("data", "graph_data.npy", "<i8", 1),
    (
        "edge_squared_distance_m2",
        "graph_edge_squared_distance_m2.npy",
        "<f8",
        1,
    ),
    ("edge_distance_m", "graph_edge_distance_m.npy", "<f8", 1),
    ("slot_id", "demand_slot_id.npy", "<i8", 1),
)
GREEDY_ARRAY_FILES = (
    ("slot_atom_id", "demand_slot_atom_id.npy", "<i8", 1),
    ("atom_id", "demand_atom_id.npy", "<i8", 1),
    ("atom_xyz", "demand_atom_xyz.npy", "<f8", 2),
    ("atom_weight", "demand_atom_weight.npy", "<f8", 1),
)
POINTWISE_ARRAY_FILES = (
    ("nearest_atom_id", "pointwise_nearest_atom_id.npy", "<i8", 1),
    ("squared_distance", "pointwise_squared_distance.npy", "<f8", 1),
    ("distance_m", "pointwise_distance_m.npy", "<f8", 1),
)


def _immutable_array(values: np.ndarray, dtype: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.dtype(dtype)).reshape(
        array.shape
    )


def _hash_arrays(domain: bytes, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(domain + b"\0")
    digest.update(struct.pack("<I", len(arrays)))
    for values in arrays:
        array = np.ascontiguousarray(values)
        dtype = array.dtype.str.encode("ascii")
        digest.update(struct.pack("<I", len(dtype)))
        digest.update(dtype)
        digest.update(struct.pack("<I", array.ndim))
        digest.update(struct.pack("<" + "Q" * array.ndim, *array.shape))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _hash_hex_fields(domain: bytes, *fields: str) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for field in fields:
        encoded = field.encode("ascii")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _hash_file_set(domain: bytes, hashes: tuple[tuple[str, str], ...]) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for name, value in hashes:
        encoded_name = name.encode("ascii")
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _load_array_set(
    root: Path,
    schema: tuple[tuple[str, str, str, int], ...],
    *,
    expected_sha256: Mapping[str, str] | None,
) -> tuple[dict[str, np.ndarray], tuple[tuple[str, str], ...]]:
    arrays: dict[str, np.ndarray] = {}
    hashes: list[tuple[str, str]] = []
    for field, filename, dtype, ndim in schema:
        path = root / filename
        payload = path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            expected = expected_sha256.get(filename)
            if expected is None:
                raise ValueError(f"Missing expected SHA-256 for {filename}")
            if observed != expected:
                raise ValueError(f"Immutable sidecar hash changed: {filename}")
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
        if not isinstance(loaded, np.ndarray):
            raise TypeError(f"{filename} is not a NumPy array")
        if loaded.dtype != np.dtype(dtype):
            raise ValueError(
                f"{filename} dtype {loaded.dtype.str} does not equal {dtype}"
            )
        if loaded.ndim != ndim:
            raise ValueError(f"{filename} rank {loaded.ndim} does not equal {ndim}")
        arrays[field] = _immutable_array(loaded, dtype)
        hashes.append((filename, observed))
    return arrays, tuple(hashes)


@dataclass(frozen=True)
class PackedSupport:
    stable_candidate_id: np.ndarray
    grid_cell: np.ndarray
    xyz: np.ndarray
    base_confidence: np.ndarray
    color: np.ndarray
    source_hashes: tuple[tuple[str, str], ...] = ()
    digest_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stable_candidate_id",
            _immutable_array(self.stable_candidate_id, "<i8"),
        )
        object.__setattr__(self, "grid_cell", _immutable_array(self.grid_cell, "<i8"))
        object.__setattr__(self, "xyz", _immutable_array(self.xyz, "<f4"))
        object.__setattr__(
            self,
            "base_confidence",
            _immutable_array(self.base_confidence, "<f4"),
        )
        object.__setattr__(self, "color", _immutable_array(self.color, "<u1"))
        count = self.stable_candidate_id.size
        if count == 0:
            raise ValueError("Packed support cannot be empty")
        if self.stable_candidate_id.shape != (count,):
            raise ValueError("Packed support candidate IDs must be one-dimensional")
        if self.grid_cell.shape != (count, 3):
            raise ValueError("Packed support grid cells must have shape (N,3)")
        if self.xyz.shape != (count, 3):
            raise ValueError("Packed support XYZ must have shape (N,3)")
        if self.base_confidence.shape != (count,) or self.color.shape != (count,):
            raise ValueError("Packed support sidecars must align with candidate IDs")
        if np.any(np.diff(self.stable_candidate_id) <= 0):
            raise ValueError("Packed support candidate IDs must be strictly increasing")
        if not np.isfinite(self.xyz).all() or not np.isfinite(
            self.base_confidence
        ).all():
            raise ValueError("Packed support contains a non-finite value")
        if np.any(self.color > 7) or np.unique(self.color).size != 1:
            raise ValueError("Packed support must contain one valid parity color")
        observed = _hash_arrays(
            b"stda_f0_packed_support_v1",
            self.stable_candidate_id,
            self.grid_cell,
            self.xyz,
            self.base_confidence,
            self.color,
        )
        if self.digest_sha256 and self.digest_sha256 != observed:
            raise ValueError("Packed support digest changed")
        object.__setattr__(self, "digest_sha256", observed)

    @property
    def count(self) -> int:
        return int(self.stable_candidate_id.size)


@dataclass(frozen=True)
class AssignmentGraph:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    edge_squared_distance_m2: np.ndarray
    edge_distance_m: np.ndarray
    slot_id: np.ndarray
    support_cardinality: int
    source_hashes: tuple[tuple[str, str], ...] = ()
    digest_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "indptr", _immutable_array(self.indptr, "<i8"))
        object.__setattr__(self, "indices", _immutable_array(self.indices, "<i4"))
        object.__setattr__(self, "data", _immutable_array(self.data, "<i8"))
        object.__setattr__(
            self,
            "edge_squared_distance_m2",
            _immutable_array(self.edge_squared_distance_m2, "<f8"),
        )
        object.__setattr__(
            self,
            "edge_distance_m",
            _immutable_array(self.edge_distance_m, "<f8"),
        )
        object.__setattr__(self, "slot_id", _immutable_array(self.slot_id, "<i8"))
        if self.support_cardinality <= 0:
            raise ValueError("Graph support cardinality must be positive")
        if self.indptr.shape != (self.slot_id.size + 1,):
            raise ValueError("Graph indptr does not align with slot IDs")
        if self.indptr[0] != 0 or np.any(np.diff(self.indptr) < 0):
            raise ValueError("Graph indptr is not a valid cumulative index")
        edge_count = int(self.indptr[-1])
        if not (
            self.indices.shape
            == self.data.shape
            == self.edge_squared_distance_m2.shape
            == self.edge_distance_m.shape
            == (edge_count,)
        ):
            raise ValueError("Graph edge arrays have inconsistent lengths")
        if np.any(np.diff(self.slot_id) <= 0):
            raise ValueError("Graph slot IDs must be strictly increasing")
        if edge_count:
            if np.any(self.indices < 0) or np.any(
                self.indices >= self.support_cardinality
            ):
                raise ValueError("Graph support rank is outside packed support")
            if np.any(self.data <= 0) or np.any(self.data >= MAX_EXACT_INTEGER):
                raise ValueError("Graph costs must be positive and below 2^53")
            if not np.isfinite(self.edge_squared_distance_m2).all() or np.any(
                self.edge_squared_distance_m2 < 0.0
            ):
                raise ValueError("Graph squared edge distance is invalid")
            if not np.isfinite(self.edge_distance_m).all() or np.any(
                self.edge_distance_m < 0.0
            ):
                raise ValueError("Graph edge distance is invalid")
            replayed_distance = np.fromiter(
                (
                    math.sqrt(float(value))
                    for value in self.edge_squared_distance_m2
                ),
                dtype="<f8",
                count=edge_count,
            )
            if not np.array_equal(replayed_distance, self.edge_distance_m):
                raise ValueError("Graph edge distance is not the frozen sqrt replay")
        row_maximum_sum = 0
        for row in range(self.slot_count):
            start = int(self.indptr[row])
            stop = int(self.indptr[row + 1])
            columns = self.indices[start:stop]
            if columns.size and np.any(np.diff(columns) <= 0):
                raise ValueError("Graph CSR rows must be strictly column sorted")
            if stop > start:
                row_maximum_sum += int(self.data[start:stop].max())
        if row_maximum_sum >= MAX_SIGNED_INT64:
            raise ValueError("Graph row-maximum objective bound exceeds int64")
        observed = _hash_arrays(
            b"stda_f0_assignment_graph_v1",
            self.indptr,
            self.indices,
            self.data,
            self.edge_squared_distance_m2,
            self.edge_distance_m,
            self.slot_id,
            np.asarray([self.support_cardinality], dtype="<i8"),
        )
        if self.digest_sha256 and self.digest_sha256 != observed:
            raise ValueError("Assignment graph digest changed")
        object.__setattr__(self, "digest_sha256", observed)

    @property
    def slot_count(self) -> int:
        return int(self.slot_id.size)

    @property
    def edge_count(self) -> int:
        return int(self.indices.size)

    def adjacency_csr(self) -> csr_matrix:
        values = np.ones(self.edge_count, dtype=np.uint8)
        return csr_matrix(
            (values, self.indices, self.indptr),
            shape=(self.slot_count, self.support_cardinality),
        )

    def cost_csr(self) -> csr_matrix:
        return csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=(self.slot_count, self.support_cardinality),
        )


@dataclass(frozen=True)
class GreedySidecar:
    slot_atom_id: np.ndarray
    atom_id: np.ndarray
    atom_xyz: np.ndarray
    atom_weight: np.ndarray
    source_hashes: tuple[tuple[str, str], ...] = ()
    digest_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "slot_atom_id", _immutable_array(self.slot_atom_id, "<i8")
        )
        object.__setattr__(self, "atom_id", _immutable_array(self.atom_id, "<i8"))
        object.__setattr__(self, "atom_xyz", _immutable_array(self.atom_xyz, "<f8"))
        object.__setattr__(
            self, "atom_weight", _immutable_array(self.atom_weight, "<f8")
        )
        atom_count = self.atom_id.size
        if atom_count == 0:
            raise ValueError("Greedy sidecar must contain canonical atoms")
        if self.atom_xyz.shape != (atom_count, 3):
            raise ValueError("Greedy atom XYZ must have shape (A,3)")
        if self.atom_weight.shape != (atom_count,):
            raise ValueError("Greedy atom weights must align with atom IDs")
        if np.any(np.diff(self.atom_id) <= 0):
            raise ValueError("Canonical atom IDs must be strictly increasing")
        if not np.isfinite(self.atom_xyz).all() or not np.isfinite(
            self.atom_weight
        ).all():
            raise ValueError("Greedy sidecar contains a non-finite value")
        if np.any(self.atom_weight <= 0.0):
            raise ValueError("Canonical atom weights must be positive")
        positions = np.searchsorted(self.atom_id, self.slot_atom_id)
        valid = positions < atom_count
        if bool(valid.any()):
            valid[valid] &= self.atom_id[positions[valid]] == self.slot_atom_id[valid]
        if not bool(valid.all()):
            raise ValueError("A demand slot refers to an unknown canonical atom")
        xyz_f4 = self.atom_xyz.astype("<f4")
        if not np.array_equal(xyz_f4.astype("<f8"), self.atom_xyz):
            raise ValueError("Greedy atom XYZ is not an exact float32 promotion")
        observed = _hash_arrays(
            b"stda_f0_greedy_sidecar_v1",
            self.slot_atom_id,
            self.atom_id,
            self.atom_xyz,
            self.atom_weight,
        )
        if self.digest_sha256 and self.digest_sha256 != observed:
            raise ValueError("Greedy sidecar digest changed")
        object.__setattr__(self, "digest_sha256", observed)


@dataclass(frozen=True)
class PointwiseSidecar:
    nearest_atom_id: np.ndarray
    squared_distance: np.ndarray
    distance_m: np.ndarray
    source_hashes: tuple[tuple[str, str], ...] = ()
    digest_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nearest_atom_id",
            _immutable_array(self.nearest_atom_id, "<i8"),
        )
        object.__setattr__(
            self,
            "squared_distance",
            _immutable_array(self.squared_distance, "<f8"),
        )
        object.__setattr__(self, "distance_m", _immutable_array(self.distance_m, "<f8"))
        count = self.nearest_atom_id.size
        if not (
            self.nearest_atom_id.shape
            == self.squared_distance.shape
            == self.distance_m.shape
            == (count,)
        ):
            raise ValueError("Pointwise sidecar columns have inconsistent shapes")
        if np.any(self.nearest_atom_id < 0):
            raise ValueError("Pointwise nearest atom ID cannot be negative")
        if not np.isfinite(self.squared_distance).all() or not np.isfinite(
            self.distance_m
        ).all():
            raise ValueError("Pointwise sidecar contains a non-finite value")
        if np.any(self.squared_distance < 0.0) or np.any(self.distance_m < 0.0):
            raise ValueError("Pointwise distances cannot be negative")
        replayed = np.fromiter(
            (math.sqrt(float(value)) for value in self.squared_distance),
            dtype="<f8",
            count=count,
        )
        if not np.array_equal(replayed, self.distance_m):
            raise ValueError("Pointwise distance is not the frozen sqrt replay")
        observed = _hash_arrays(
            b"stda_f0_pointwise_sidecar_v1",
            self.nearest_atom_id,
            self.squared_distance,
            self.distance_m,
        )
        if self.digest_sha256 and self.digest_sha256 != observed:
            raise ValueError("Pointwise sidecar digest changed")
        object.__setattr__(self, "digest_sha256", observed)


def load_packed_support(
    root: Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> PackedSupport:
    arrays, hashes = _load_array_set(
        root, SUPPORT_ARRAY_FILES, expected_sha256=expected_sha256
    )
    return PackedSupport(**arrays, source_hashes=hashes)


def load_assignment_graph(
    root: Path,
    *,
    support_cardinality: int,
    expected_sha256: Mapping[str, str] | None = None,
    require_formal_shape: bool = False,
) -> AssignmentGraph:
    arrays, hashes = _load_array_set(
        root, GRAPH_ARRAY_FILES, expected_sha256=expected_sha256
    )
    graph = AssignmentGraph(
        **arrays,
        support_cardinality=support_cardinality,
        source_hashes=hashes,
    )
    if require_formal_shape:
        if graph.slot_count != FORMAL_SLOT_COUNT:
            raise ValueError("Formal graph must contain exactly 10,000 slots")
        degrees = np.diff(graph.indptr)
        if not np.all(degrees == FORMAL_NEIGHBOR_COUNT):
            raise ValueError("Formal graph must contain exactly 256 edges per slot")
        if graph.edge_count != FORMAL_SLOT_COUNT * FORMAL_NEIGHBOR_COUNT:
            raise ValueError("Formal graph edge count changed")
        if not np.array_equal(
            graph.slot_id, np.arange(FORMAL_SLOT_COUNT, dtype="<i8")
        ):
            raise ValueError("Formal graph slot IDs must be 0 through 9,999")
    return graph


def load_greedy_sidecar(
    root: Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> GreedySidecar:
    arrays, hashes = _load_array_set(
        root, GREEDY_ARRAY_FILES, expected_sha256=expected_sha256
    )
    return GreedySidecar(**arrays, source_hashes=hashes)


def load_pointwise_sidecar(
    root: Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> PointwiseSidecar:
    arrays, hashes = _load_array_set(
        root, POINTWISE_ARRAY_FILES, expected_sha256=expected_sha256
    )
    return PointwiseSidecar(**arrays, source_hashes=hashes)


@dataclass(frozen=True)
class MatchingResult:
    method: str
    slot_to_support_rank: np.ndarray
    support_to_slot_row: np.ndarray
    cardinality: int
    digest_sha256: str


@dataclass(frozen=True)
class MatchingChecks:
    shape_valid: bool
    sentinels_and_bounds_valid: bool
    cardinality_valid: bool
    reciprocal: bool
    support_capacity: bool
    edge_membership: bool
    digest_valid: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.shape_valid,
                self.sentinels_and_bounds_valid,
                self.cardinality_valid,
                self.reciprocal,
                self.support_capacity,
                self.edge_membership,
                self.digest_valid,
            )
        )


def _matching_result(
    method: str,
    slot_to_support_rank: np.ndarray,
    support_to_slot_row: np.ndarray,
) -> MatchingResult:
    slot_to = _immutable_array(slot_to_support_rank, "<i8")
    support_to = _immutable_array(support_to_slot_row, "<i8")
    cardinality = int((slot_to >= 0).sum())
    digest = _hash_arrays(b"stda_f0_matching_v1", slot_to, support_to)
    return MatchingResult(
        method=method,
        slot_to_support_rank=slot_to,
        support_to_slot_row=support_to,
        cardinality=cardinality,
        digest_sha256=digest,
    )


def _edge_position(graph: AssignmentGraph, slot_row: int, support_rank: int) -> int:
    start = int(graph.indptr[slot_row])
    stop = int(graph.indptr[slot_row + 1])
    relative = int(np.searchsorted(graph.indices[start:stop], support_rank))
    position = start + relative
    if position >= stop or int(graph.indices[position]) != support_rank:
        return -1
    return position


def verify_matching(
    graph: AssignmentGraph,
    result: MatchingResult,
) -> MatchingChecks:
    slot_to = result.slot_to_support_rank
    support_to = result.support_to_slot_row
    shape_valid = slot_to.shape == (graph.slot_count,) and support_to.shape == (
        graph.support_cardinality,
    )
    if not shape_valid:
        return MatchingChecks(False, False, False, False, False, False, False)
    sentinels_and_bounds_valid = bool(
        np.all(slot_to >= -1)
        and np.all(slot_to < graph.support_cardinality)
        and np.all(support_to >= -1)
        and np.all(support_to < graph.slot_count)
    )
    matched_rows = np.flatnonzero(slot_to >= 0)
    matched_support = slot_to[matched_rows]
    cardinality_valid = (
        result.cardinality == matched_rows.size == int((support_to >= 0).sum())
    )
    support_capacity = (
        matched_support.size == np.unique(matched_support).size
        and bool(np.all(matched_support < graph.support_cardinality))
    )
    reciprocal = support_capacity
    edge_membership = True
    if support_capacity:
        for slot_row, support_rank in zip(matched_rows, matched_support):
            reciprocal &= int(support_to[int(support_rank)]) == int(slot_row)
            edge_membership &= (
                _edge_position(graph, int(slot_row), int(support_rank)) >= 0
            )
    else:
        reciprocal = False
        edge_membership = False
    digest_valid = result.digest_sha256 == _hash_arrays(
        b"stda_f0_matching_v1", slot_to, support_to
    )
    return MatchingChecks(
        shape_valid=shape_valid,
        sentinels_and_bounds_valid=sentinels_and_bounds_valid,
        cardinality_valid=cardinality_valid,
        reciprocal=bool(reciprocal),
        support_capacity=bool(support_capacity),
        edge_membership=bool(edge_membership),
        digest_valid=digest_valid,
    )


def _hopcroft_bfs(
    graph: AssignmentGraph,
    slot_to: np.ndarray,
    support_to: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    infinity = graph.slot_count + 1
    distance = np.full(graph.slot_count, infinity, dtype=np.int64)
    queue: deque[int] = deque()
    for slot_row in range(graph.slot_count):
        if slot_to[slot_row] < 0:
            distance[slot_row] = 0
            queue.append(slot_row)
    shortest: int | None = None
    while queue:
        slot_row = queue.popleft()
        next_distance = int(distance[slot_row]) + 1
        if shortest is not None and next_distance > shortest:
            continue
        start = int(graph.indptr[slot_row])
        stop = int(graph.indptr[slot_row + 1])
        for support_rank in graph.indices[start:stop]:
            matched_slot = int(support_to[int(support_rank)])
            if matched_slot < 0:
                shortest = next_distance if shortest is None else min(
                    shortest, next_distance
                )
            elif distance[matched_slot] == infinity:
                distance[matched_slot] = next_distance
                queue.append(matched_slot)
    return distance, shortest


def _hopcroft_augment(
    graph: AssignmentGraph,
    root: int,
    slot_to: np.ndarray,
    support_to: np.ndarray,
    distance: np.ndarray,
    shortest: int,
) -> bool:
    infinity = graph.slot_count + 1
    rows = [root]
    cursors = [int(graph.indptr[root])]
    via_support: list[int] = []
    while rows:
        slot_row = rows[-1]
        stop = int(graph.indptr[slot_row + 1])
        descended = False
        while cursors[-1] < stop:
            position = cursors[-1]
            cursors[-1] += 1
            support_rank = int(graph.indices[position])
            matched_slot = int(support_to[support_rank])
            if matched_slot < 0:
                if int(distance[slot_row]) + 1 != shortest:
                    continue
                slot_to[slot_row] = support_rank
                support_to[support_rank] = slot_row
                for level in range(len(via_support) - 1, -1, -1):
                    parent_slot = rows[level]
                    parent_support = via_support[level]
                    slot_to[parent_slot] = parent_support
                    support_to[parent_support] = parent_slot
                return True
            if distance[matched_slot] == distance[slot_row] + 1:
                rows.append(matched_slot)
                cursors.append(int(graph.indptr[matched_slot]))
                via_support.append(support_rank)
                descended = True
                break
        if descended:
            continue
        distance[slot_row] = infinity
        rows.pop()
        cursors.pop()
        if via_support and len(via_support) >= len(rows):
            via_support.pop()
    return False


def deterministic_hopcroft_karp(graph: AssignmentGraph) -> MatchingResult:
    """Return a deterministic maximum matching using sorted CSR neighbors."""

    slot_to = np.full(graph.slot_count, -1, dtype=np.int64)
    support_to = np.full(graph.support_cardinality, -1, dtype=np.int64)
    while True:
        distance, shortest = _hopcroft_bfs(graph, slot_to, support_to)
        if shortest is None:
            break
        augmented = 0
        for slot_row in range(graph.slot_count):
            if slot_to[slot_row] < 0 and distance[slot_row] == 0:
                augmented += int(
                    _hopcroft_augment(
                        graph,
                        slot_row,
                        slot_to,
                        support_to,
                        distance,
                        shortest,
                    )
                )
        if augmented == 0:
            raise AssertionError("Hopcroft-Karp BFS found an unusable augmenting layer")
    result = _matching_result("deterministic_hopcroft_karp", slot_to, support_to)
    if not verify_matching(graph, result).passed:
        raise AssertionError("Custom Hopcroft-Karp replay failed")
    return result


def scipy_maximum_matching(graph: AssignmentGraph) -> MatchingResult:
    slot_to = maximum_bipartite_matching(
        graph.adjacency_csr(), perm_type="column"
    ).astype(np.int64, copy=False)
    support_to = np.full(graph.support_cardinality, -1, dtype=np.int64)
    for slot_row, support_rank in enumerate(slot_to):
        if support_rank >= 0:
            if support_to[int(support_rank)] >= 0:
                raise AssertionError("SciPy maximum matching reused support")
            support_to[int(support_rank)] = slot_row
    result = _matching_result("scipy_maximum_bipartite_matching", slot_to, support_to)
    if not verify_matching(graph, result).passed:
        raise AssertionError("SciPy maximum matching replay failed")
    return result


def _alternating_reachable(
    graph: AssignmentGraph,
    matching: MatchingResult,
) -> tuple[np.ndarray, np.ndarray]:
    reachable_slot = matching.slot_to_support_rank < 0
    reachable_support = np.zeros(graph.support_cardinality, dtype=bool)
    queue: deque[int] = deque(int(value) for value in np.flatnonzero(reachable_slot))
    while queue:
        slot_row = queue.popleft()
        matched_support = int(matching.slot_to_support_rank[slot_row])
        start = int(graph.indptr[slot_row])
        stop = int(graph.indptr[slot_row + 1])
        for support_rank_value in graph.indices[start:stop]:
            support_rank = int(support_rank_value)
            if support_rank == matched_support or reachable_support[support_rank]:
                continue
            reachable_support[support_rank] = True
            matched_slot = int(matching.support_to_slot_row[support_rank])
            if matched_slot >= 0 and not reachable_slot[matched_slot]:
                reachable_slot[matched_slot] = True
                queue.append(matched_slot)
    return np.flatnonzero(reachable_slot), np.flatnonzero(reachable_support)


@dataclass(frozen=True)
class HallWitness:
    slot_rows: np.ndarray
    support_ranks: np.ndarray
    slot_ids: np.ndarray
    support_ids: np.ndarray
    hall_deficit: int
    global_deficit: int
    slot_set_sha256: str
    support_set_sha256: str
    matching_sha256: str
    digest_sha256: str


@dataclass(frozen=True)
class HallChecks:
    matching_valid: bool
    graph_deficient: bool
    alternating_slot_set_equal: bool
    alternating_support_set_equal: bool
    full_neighborhood_equal: bool
    slot_ids_equal: bool
    support_ids_equal: bool
    strict_hall_inequality: bool
    global_deficit_bounds_hall: bool
    hashes_valid: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.matching_valid,
                self.graph_deficient,
                self.alternating_slot_set_equal,
                self.alternating_support_set_equal,
                self.full_neighborhood_equal,
                self.slot_ids_equal,
                self.support_ids_equal,
                self.strict_hall_inequality,
                self.global_deficit_bounds_hall,
                self.hashes_valid,
            )
        )


def build_hall_witness(
    support: PackedSupport,
    graph: AssignmentGraph,
    matching: MatchingResult,
) -> HallWitness:
    if matching.cardinality >= graph.slot_count:
        raise ValueError("A full matching has no deficient Hall witness")
    if support.count != graph.support_cardinality:
        raise ValueError("Support and graph cardinalities differ")
    if not verify_matching(graph, matching).passed:
        raise ValueError("Cannot derive Hall witness from an invalid matching")
    slot_rows, alternating_support = _alternating_reachable(graph, matching)
    neighborhood = np.zeros(graph.support_cardinality, dtype=bool)
    for slot_row in slot_rows:
        start = int(graph.indptr[int(slot_row)])
        stop = int(graph.indptr[int(slot_row) + 1])
        neighborhood[graph.indices[start:stop]] = True
    support_ranks = np.flatnonzero(neighborhood)
    if not np.array_equal(alternating_support, support_ranks):
        raise AssertionError("Alternating support set is not the full Hall neighborhood")
    slot_rows = _immutable_array(slot_rows, "<i8")
    support_ranks = _immutable_array(support_ranks, "<i8")
    slot_ids = _immutable_array(graph.slot_id[slot_rows], "<i8")
    support_ids = _immutable_array(
        support.stable_candidate_id[support_ranks], "<i8"
    )
    hall_deficit = int(slot_rows.size - support_ranks.size)
    global_deficit = int(graph.slot_count - matching.cardinality)
    if hall_deficit <= 0 or global_deficit < hall_deficit:
        raise AssertionError("Terminal alternating sets do not certify Hall deficit")
    slot_hash = _hash_arrays(b"stda_f0_hall_slot_set_v1", slot_rows)
    support_hash = _hash_arrays(
        b"stda_f0_hall_support_set_v1", support_ranks
    )
    digest = _hash_arrays(
        b"stda_f0_hall_witness_v1",
        slot_rows,
        support_ranks,
        slot_ids,
        support_ids,
        np.asarray([hall_deficit, global_deficit], dtype="<i8"),
    )
    return HallWitness(
        slot_rows=slot_rows,
        support_ranks=support_ranks,
        slot_ids=slot_ids,
        support_ids=support_ids,
        hall_deficit=hall_deficit,
        global_deficit=global_deficit,
        slot_set_sha256=slot_hash,
        support_set_sha256=support_hash,
        matching_sha256=matching.digest_sha256,
        digest_sha256=digest,
    )


def verify_hall_witness(
    support: PackedSupport,
    graph: AssignmentGraph,
    matching: MatchingResult,
    witness: HallWitness,
) -> HallChecks:
    matching_valid = verify_matching(graph, matching).passed
    graph_deficient = matching.cardinality < graph.slot_count
    if matching_valid and graph_deficient:
        alternating_slots, alternating_support = _alternating_reachable(
            graph, matching
        )
    else:
        alternating_slots = np.asarray([], dtype=np.int64)
        alternating_support = np.asarray([], dtype=np.int64)
    neighborhood = np.zeros(graph.support_cardinality, dtype=bool)
    for slot_row in witness.slot_rows:
        if slot_row < 0 or slot_row >= graph.slot_count:
            continue
        start = int(graph.indptr[int(slot_row)])
        stop = int(graph.indptr[int(slot_row) + 1])
        neighborhood[graph.indices[start:stop]] = True
    full_neighborhood = np.flatnonzero(neighborhood)
    expected_slot_ids = (
        graph.slot_id[witness.slot_rows]
        if bool(
            np.all(
                (witness.slot_rows >= 0) & (witness.slot_rows < graph.slot_count)
            )
        )
        else np.asarray([], dtype=np.int64)
    )
    expected_support_ids = (
        support.stable_candidate_id[witness.support_ranks]
        if bool(
            np.all(
                (witness.support_ranks >= 0)
                & (witness.support_ranks < support.count)
            )
        )
        else np.asarray([], dtype=np.int64)
    )
    hall_deficit = int(witness.slot_rows.size - witness.support_ranks.size)
    global_deficit = int(graph.slot_count - matching.cardinality)
    slot_hash = _hash_arrays(b"stda_f0_hall_slot_set_v1", witness.slot_rows)
    support_hash = _hash_arrays(
        b"stda_f0_hall_support_set_v1", witness.support_ranks
    )
    digest = _hash_arrays(
        b"stda_f0_hall_witness_v1",
        witness.slot_rows,
        witness.support_ranks,
        witness.slot_ids,
        witness.support_ids,
        np.asarray([witness.hall_deficit, witness.global_deficit], dtype="<i8"),
    )
    hashes_valid = all(
        (
            witness.slot_set_sha256 == slot_hash,
            witness.support_set_sha256 == support_hash,
            witness.matching_sha256 == matching.digest_sha256,
            witness.digest_sha256 == digest,
            witness.hall_deficit == hall_deficit,
            witness.global_deficit == global_deficit,
        )
    )
    return HallChecks(
        matching_valid=matching_valid,
        graph_deficient=graph_deficient,
        alternating_slot_set_equal=np.array_equal(
            witness.slot_rows, alternating_slots
        ),
        alternating_support_set_equal=np.array_equal(
            witness.support_ranks, alternating_support
        ),
        full_neighborhood_equal=np.array_equal(
            witness.support_ranks, full_neighborhood
        ),
        slot_ids_equal=np.array_equal(witness.slot_ids, expected_slot_ids),
        support_ids_equal=np.array_equal(
            witness.support_ids, expected_support_ids
        ),
        strict_hall_inequality=witness.support_ranks.size < witness.slot_rows.size,
        global_deficit_bounds_hall=global_deficit >= hall_deficit > 0,
        hashes_valid=hashes_valid,
    )


@dataclass(frozen=True)
class CardinalityCertificate:
    source_capacity: int
    requested_target_mass: int
    transported_mass: int
    unmatched_target_mass: int
    unused_source_mass: int
    custom_matching: MatchingResult
    scipy_matching: MatchingResult
    hall_witness: HallWitness | None
    digest_sha256: str


def maximum_cardinality_certificate(
    support: PackedSupport,
    graph: AssignmentGraph,
) -> CardinalityCertificate:
    if support.count != graph.support_cardinality:
        raise ValueError("Support and graph cardinalities differ")
    custom = deterministic_hopcroft_karp(graph)
    scipy = scipy_maximum_matching(graph)
    if custom.cardinality != scipy.cardinality:
        raise AssertionError("Custom and SciPy maximum cardinalities differ")
    witness = (
        build_hall_witness(support, graph, custom)
        if custom.cardinality < graph.slot_count
        else None
    )
    transported = custom.cardinality
    fields = np.asarray(
        [
            support.count,
            graph.slot_count,
            transported,
            graph.slot_count - transported,
            support.count - transported,
        ],
        dtype="<i8",
    )
    digest = _hash_hex_fields(
        b"stda_f0_cardinality_certificate_v1",
        _hash_arrays(b"stda_f0_cardinality_fields_v1", fields),
        custom.digest_sha256,
        scipy.digest_sha256,
        witness.digest_sha256 if witness is not None else "",
    )
    return CardinalityCertificate(
        source_capacity=support.count,
        requested_target_mass=graph.slot_count,
        transported_mass=transported,
        unmatched_target_mass=graph.slot_count - transported,
        unused_source_mass=support.count - transported,
        custom_matching=custom,
        scipy_matching=scipy,
        hall_witness=witness,
        digest_sha256=digest,
    )


@dataclass(frozen=True)
class FullAssignmentResult:
    slot_row: np.ndarray
    slot_id: np.ndarray
    support_rank: np.ndarray
    support_id: np.ndarray
    edge_cost: np.ndarray
    selected_support_rank: np.ndarray
    selected_support_id: np.ndarray
    export_xyz: np.ndarray
    objective: int
    assignment_sha256: str
    selected_id_sha256: str
    export_sha256: str
    objective_sha256: str
    digest_sha256: str


@dataclass(frozen=True)
class AssignmentReplayChecks:
    shape_valid: bool
    all_slots_once: bool
    support_capacity: bool
    assignment_sidecars_equal: bool
    edge_membership: bool
    edge_cost_equal: bool
    objective_equal: bool
    export_finite: bool
    export_unique_bytes: bool
    canonical_export_order: bool
    hashes_valid: bool
    reference_equal: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.shape_valid,
                self.all_slots_once,
                self.support_capacity,
                self.assignment_sidecars_equal,
                self.edge_membership,
                self.edge_cost_equal,
                self.objective_equal,
                self.export_finite,
                self.export_unique_bytes,
                self.canonical_export_order,
                self.hashes_valid,
                self.reference_equal,
            )
        )


def _unique_xyz_bytes(xyz: np.ndarray) -> bool:
    rows = np.ascontiguousarray(xyz).view(
        np.dtype((np.void, xyz.dtype.itemsize * xyz.shape[1]))
    )
    return np.unique(rows.reshape(-1)).size == xyz.shape[0]


def _full_assignment_from_columns(
    support: PackedSupport,
    graph: AssignmentGraph,
    slot_row: np.ndarray,
    support_rank: np.ndarray,
) -> FullAssignmentResult:
    order = np.argsort(slot_row, kind="stable")
    slot_row = _immutable_array(slot_row[order], "<i8")
    support_rank = _immutable_array(support_rank[order], "<i8")
    if not np.array_equal(slot_row, np.arange(graph.slot_count, dtype="<i8")):
        raise ValueError("Full assignment did not cover every slot")
    positions = np.empty(graph.slot_count, dtype=np.int64)
    for index, (row, rank) in enumerate(zip(slot_row, support_rank)):
        positions[index] = _edge_position(graph, int(row), int(rank))
    if np.any(positions < 0):
        raise ValueError("Full assignment selected a non-edge")
    slot_id = _immutable_array(graph.slot_id[slot_row], "<i8")
    support_id = _immutable_array(
        support.stable_candidate_id[support_rank], "<i8"
    )
    edge_cost = _immutable_array(graph.data[positions], "<i8")
    objective = sum(int(value) for value in edge_cost)
    if objective >= MAX_SIGNED_INT64:
        raise ValueError("Full assignment objective exceeds signed int64")
    selected_rank = _immutable_array(np.sort(support_rank), "<i8")
    selected_id = _immutable_array(
        support.stable_candidate_id[selected_rank], "<i8"
    )
    export_xyz = _immutable_array(support.xyz[selected_rank], "<f4")
    assignment_hash = _hash_arrays(
        b"stda_f0_decision_assignment_v1",
        slot_row,
        slot_id,
        support_rank,
        support_id,
        edge_cost,
    )
    selected_hash = _hash_arrays(
        b"stda_f0_decision_selected_ids_v1", selected_id
    )
    export_hash = _hash_arrays(b"stda_f0_decision_export_v1", export_xyz)
    objective_hash = _hash_arrays(
        b"stda_f0_decision_objective_v1",
        np.asarray([objective], dtype="<i8"),
    )
    digest = _hash_hex_fields(
        b"stda_f0_decision_result_v1",
        assignment_hash,
        selected_hash,
        export_hash,
        objective_hash,
    )
    return FullAssignmentResult(
        slot_row=slot_row,
        slot_id=slot_id,
        support_rank=support_rank,
        support_id=support_id,
        edge_cost=edge_cost,
        selected_support_rank=selected_rank,
        selected_support_id=selected_id,
        export_xyz=export_xyz,
        objective=objective,
        assignment_sha256=assignment_hash,
        selected_id_sha256=selected_hash,
        export_sha256=export_hash,
        objective_sha256=objective_hash,
        digest_sha256=digest,
    )


def solve_full_assignment(
    support: PackedSupport,
    graph: AssignmentGraph,
) -> FullAssignmentResult:
    """Run one frozen SciPy sparse full-assignment execution."""

    if support.count != graph.support_cardinality:
        raise ValueError("Support and graph cardinalities differ")
    slot_row, support_rank = min_weight_full_bipartite_matching(graph.cost_csr())
    result = _full_assignment_from_columns(
        support,
        graph,
        np.asarray(slot_row, dtype=np.int64),
        np.asarray(support_rank, dtype=np.int64),
    )
    if not verify_full_assignment(support, graph, result).passed:
        raise AssertionError("Full-assignment replay failed")
    return result


def verify_full_assignment(
    support: PackedSupport,
    graph: AssignmentGraph,
    result: FullAssignmentResult,
    *,
    reference: FullAssignmentResult | None = None,
) -> AssignmentReplayChecks:
    count = graph.slot_count
    shape_valid = all(
        values.shape == (count,)
        for values in (
            result.slot_row,
            result.slot_id,
            result.support_rank,
            result.support_id,
            result.edge_cost,
            result.selected_support_rank,
            result.selected_support_id,
        )
    ) and result.export_xyz.shape == (count, 3)
    all_slots_once = shape_valid and np.array_equal(
        result.slot_row, np.arange(count, dtype="<i8")
    ) and np.array_equal(result.slot_id, graph.slot_id)
    support_capacity = shape_valid and (
        np.unique(result.support_rank).size == count
        and bool(np.all(result.support_rank >= 0))
        and bool(np.all(result.support_rank < support.count))
    )
    assignment_sidecars_equal = all_slots_once and support_capacity and all(
        (
            np.array_equal(
                result.support_id,
                support.stable_candidate_id[result.support_rank],
            ),
            np.array_equal(result.slot_id, graph.slot_id[result.slot_row]),
        )
    )
    edge_membership = bool(all_slots_once and support_capacity)
    edge_cost_equal = bool(all_slots_once and support_capacity)
    expected_cost: list[int] = []
    if all_slots_once and support_capacity:
        for row, rank in zip(result.slot_row, result.support_rank):
            position = _edge_position(graph, int(row), int(rank))
            edge_membership &= position >= 0
            if position >= 0:
                expected_cost.append(int(graph.data[position]))
            else:
                expected_cost.append(-1)
        edge_cost_equal = np.array_equal(
            result.edge_cost, np.asarray(expected_cost, dtype="<i8")
        )
    objective = sum(int(value) for value in result.edge_cost)
    objective_equal = shape_valid and result.objective == objective
    expected_rank = np.sort(result.support_rank) if shape_valid else np.asarray([])
    canonical_export_order = (
        shape_valid
        and support_capacity
        and np.array_equal(result.selected_support_rank, expected_rank)
        and np.array_equal(
            result.selected_support_id,
            support.stable_candidate_id[result.selected_support_rank],
        )
        and np.array_equal(result.export_xyz, support.xyz[result.selected_support_rank])
    )
    export_finite = bool(np.isfinite(result.export_xyz).all())
    export_unique_bytes = shape_valid and _unique_xyz_bytes(result.export_xyz)
    assignment_hash = _hash_arrays(
        b"stda_f0_decision_assignment_v1",
        result.slot_row,
        result.slot_id,
        result.support_rank,
        result.support_id,
        result.edge_cost,
    )
    selected_hash = _hash_arrays(
        b"stda_f0_decision_selected_ids_v1", result.selected_support_id
    )
    export_hash = _hash_arrays(b"stda_f0_decision_export_v1", result.export_xyz)
    objective_hash = _hash_arrays(
        b"stda_f0_decision_objective_v1",
        np.asarray([result.objective], dtype="<i8"),
    )
    digest = _hash_hex_fields(
        b"stda_f0_decision_result_v1",
        assignment_hash,
        selected_hash,
        export_hash,
        objective_hash,
    )
    hashes_valid = all(
        (
            result.assignment_sha256 == assignment_hash,
            result.selected_id_sha256 == selected_hash,
            result.export_sha256 == export_hash,
            result.objective_sha256 == objective_hash,
            result.digest_sha256 == digest,
        )
    )
    reference_equal = reference is None or all(
        (
            result.assignment_sha256 == reference.assignment_sha256,
            result.selected_id_sha256 == reference.selected_id_sha256,
            result.export_sha256 == reference.export_sha256,
            result.objective_sha256 == reference.objective_sha256,
            result.digest_sha256 == reference.digest_sha256,
        )
    )
    return AssignmentReplayChecks(
        shape_valid=shape_valid,
        all_slots_once=bool(all_slots_once),
        support_capacity=bool(support_capacity),
        assignment_sidecars_equal=bool(assignment_sidecars_equal),
        edge_membership=bool(edge_membership),
        edge_cost_equal=bool(edge_cost_equal),
        objective_equal=objective_equal,
        export_finite=export_finite,
        export_unique_bytes=bool(export_unique_bytes),
        canonical_export_order=bool(canonical_export_order),
        hashes_valid=hashes_valid,
        reference_equal=reference_equal,
    )


@dataclass(frozen=True)
class PackedPointwiseResult:
    selected_support_rank: np.ndarray
    selected_support_id: np.ndarray
    nearest_atom_id: np.ndarray
    squared_distance: np.ndarray
    distance_m: np.ndarray
    export_xyz: np.ndarray
    selection_sha256: str
    selected_id_sha256: str
    export_sha256: str
    digest_sha256: str


@dataclass(frozen=True)
class PointwiseReplayChecks:
    count_valid: bool
    frozen_order_equal: bool
    sidecar_columns_equal: bool
    export_equal: bool
    hashes_valid: bool
    reference_equal: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.count_valid,
                self.frozen_order_equal,
                self.sidecar_columns_equal,
                self.export_equal,
                self.hashes_valid,
                self.reference_equal,
            )
        )


def packed_pointwise_control(
    support: PackedSupport,
    sidecar: PointwiseSidecar,
    *,
    output_count: int = FORMAL_SLOT_COUNT,
) -> PackedPointwiseResult:
    if sidecar.nearest_atom_id.size != support.count:
        raise ValueError("Pointwise sidecar does not align with packed support")
    if output_count <= 0 or output_count > support.count:
        raise ValueError("Pointwise output count is outside packed support")
    order = np.lexsort((support.stable_candidate_id, sidecar.squared_distance))
    selected_rank = _immutable_array(order[:output_count], "<i8")
    selected_id = _immutable_array(
        support.stable_candidate_id[selected_rank], "<i8"
    )
    nearest = _immutable_array(sidecar.nearest_atom_id[selected_rank], "<i8")
    squared = _immutable_array(sidecar.squared_distance[selected_rank], "<f8")
    distance = _immutable_array(sidecar.distance_m[selected_rank], "<f8")
    export_xyz = _immutable_array(support.xyz[selected_rank], "<f4")
    selection_hash = _hash_arrays(
        b"stda_f0_packed_pointwise_selection_v1",
        selected_rank,
        selected_id,
        nearest,
        squared,
        distance,
    )
    selected_hash = _hash_arrays(
        b"stda_f0_packed_pointwise_selected_ids_v1", selected_id
    )
    export_hash = _hash_arrays(
        b"stda_f0_packed_pointwise_export_v1", export_xyz
    )
    digest = _hash_hex_fields(
        b"stda_f0_packed_pointwise_result_v1",
        selection_hash,
        selected_hash,
        export_hash,
    )
    return PackedPointwiseResult(
        selected_support_rank=selected_rank,
        selected_support_id=selected_id,
        nearest_atom_id=nearest,
        squared_distance=squared,
        distance_m=distance,
        export_xyz=export_xyz,
        selection_sha256=selection_hash,
        selected_id_sha256=selected_hash,
        export_sha256=export_hash,
        digest_sha256=digest,
    )


def verify_packed_pointwise(
    support: PackedSupport,
    sidecar: PointwiseSidecar,
    result: PackedPointwiseResult,
    *,
    reference: PackedPointwiseResult | None = None,
) -> PointwiseReplayChecks:
    count = result.selected_support_rank.size
    count_valid = 0 < count <= support.count
    expected_order = np.lexsort(
        (support.stable_candidate_id, sidecar.squared_distance)
    )[:count]
    frozen_order_equal = np.array_equal(result.selected_support_rank, expected_order)
    sidecar_columns_equal = frozen_order_equal and all(
        (
            np.array_equal(
                result.selected_support_id,
                support.stable_candidate_id[result.selected_support_rank],
            ),
            np.array_equal(
                result.nearest_atom_id,
                sidecar.nearest_atom_id[result.selected_support_rank],
            ),
            np.array_equal(
                result.squared_distance,
                sidecar.squared_distance[result.selected_support_rank],
            ),
            np.array_equal(
                result.distance_m,
                sidecar.distance_m[result.selected_support_rank],
            ),
        )
    )
    export_equal = frozen_order_equal and np.array_equal(
        result.export_xyz, support.xyz[result.selected_support_rank]
    )
    selection_hash = _hash_arrays(
        b"stda_f0_packed_pointwise_selection_v1",
        result.selected_support_rank,
        result.selected_support_id,
        result.nearest_atom_id,
        result.squared_distance,
        result.distance_m,
    )
    selected_hash = _hash_arrays(
        b"stda_f0_packed_pointwise_selected_ids_v1",
        result.selected_support_id,
    )
    export_hash = _hash_arrays(
        b"stda_f0_packed_pointwise_export_v1", result.export_xyz
    )
    digest = _hash_hex_fields(
        b"stda_f0_packed_pointwise_result_v1",
        selection_hash,
        selected_hash,
        export_hash,
    )
    hashes_valid = all(
        (
            result.selection_sha256 == selection_hash,
            result.selected_id_sha256 == selected_hash,
            result.export_sha256 == export_hash,
            result.digest_sha256 == digest,
        )
    )
    reference_equal = reference is None or result.digest_sha256 == reference.digest_sha256
    return PointwiseReplayChecks(
        count_valid=count_valid,
        frozen_order_equal=frozen_order_equal,
        sidecar_columns_equal=bool(sidecar_columns_equal),
        export_equal=bool(export_equal),
        hashes_valid=hashes_valid,
        reference_equal=reference_equal,
    )


@dataclass(frozen=True)
class GreedyAtomOrder:
    frame_key: bytes
    ordered_atom_id: np.ndarray
    ordered_digest_bytes: np.ndarray
    digest_sha256: str


def greedy_atom_order(
    sidecar: GreedySidecar,
    *,
    sequence: int,
    radar_index: int,
) -> GreedyAtomOrder:
    if sequence < 0 or radar_index < 0:
        raise ValueError("Frame key indices cannot be negative")
    frame_key = f"seq{sequence:02d}/radar{radar_index:05d}".encode("ascii")
    active = np.unique(sidecar.slot_atom_id)
    keyed: list[tuple[bytes, int]] = []
    for atom_id_value in active:
        atom_id = int(atom_id_value)
        row = int(np.searchsorted(sidecar.atom_id, atom_id))
        xyz_bytes = np.asarray(sidecar.atom_xyz[row], dtype="<f4").tobytes(order="C")
        weight_bytes = np.asarray([sidecar.atom_weight[row]], dtype="<f8").tobytes(
            order="C"
        )
        atom_byte_key = xyz_bytes + weight_bytes
        order_digest = hashlib.sha256(
            b"stda_f0_greedy_atom_order_v1\0"
            + frame_key
            + b"\0"
            + atom_byte_key
        ).digest()
        keyed.append((order_digest, atom_id))
    keyed.sort(key=lambda item: (item[0], item[1]))
    ordered_atom_id = _immutable_array(
        np.asarray([item[1] for item in keyed]), "<i8"
    )
    digest_bytes = _immutable_array(
        np.frombuffer(b"".join(item[0] for item in keyed), dtype=np.uint8).reshape(
            len(keyed), 32
        ),
        "<u1",
    )
    digest = _hash_arrays(
        b"stda_f0_greedy_atom_order_commitment_v1",
        np.frombuffer(frame_key, dtype=np.uint8),
        ordered_atom_id,
        digest_bytes,
    )
    return GreedyAtomOrder(
        frame_key=frame_key,
        ordered_atom_id=ordered_atom_id,
        ordered_digest_bytes=digest_bytes,
        digest_sha256=digest,
    )


@dataclass(frozen=True)
class GreedyResult:
    atom_order: GreedyAtomOrder
    round_index: np.ndarray
    atom_id: np.ndarray
    slot_row: np.ndarray
    slot_id: np.ndarray
    support_rank: np.ndarray
    support_id: np.ndarray
    edge_cost: np.ndarray
    selected_support_rank: np.ndarray
    selected_support_id: np.ndarray
    export_xyz: np.ndarray
    failed_slot_count: int
    assignment_sha256: str
    selected_id_sha256: str
    export_sha256: str
    digest_sha256: str

    @property
    def full_capacity(self) -> bool:
        return self.failed_slot_count == 0 and bool(np.all(self.support_rank >= 0))


@dataclass(frozen=True)
class GreedyReplayChecks:
    all_slots_consumed_once: bool
    atom_and_slot_order_equal: bool
    forward_rotation_equal: bool
    candidate_capacity: bool
    edge_choice_equal: bool
    failed_slots_preserved: bool
    export_equal: bool
    hashes_valid: bool
    reference_equal: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.all_slots_consumed_once,
                self.atom_and_slot_order_equal,
                self.forward_rotation_equal,
                self.candidate_capacity,
                self.edge_choice_equal,
                self.failed_slots_preserved,
                self.export_equal,
                self.hashes_valid,
                self.reference_equal,
            )
        )


def _run_greedy_control(
    support: PackedSupport,
    graph: AssignmentGraph,
    sidecar: GreedySidecar,
    *,
    sequence: int,
    radar_index: int,
) -> GreedyResult:
    if support.count != graph.support_cardinality:
        raise ValueError("Support and graph cardinalities differ")
    if sidecar.slot_atom_id.shape != (graph.slot_count,):
        raise ValueError("Greedy slot-to-atom sidecar does not align with graph")
    atom_order = greedy_atom_order(
        sidecar, sequence=sequence, radar_index=radar_index
    )
    ordered = [int(value) for value in atom_order.ordered_atom_id]
    if graph.slot_count and not ordered:
        raise ValueError("Greedy control has demand slots but no active atoms")
    rows_by_atom: dict[int, np.ndarray] = {}
    cursor: dict[int, int] = {}
    for atom_id in ordered:
        rows = np.flatnonzero(sidecar.slot_atom_id == atom_id)
        rows = rows[np.argsort(graph.slot_id[rows], kind="stable")]
        rows_by_atom[atom_id] = rows
        cursor[atom_id] = 0
    unused = np.ones(support.count, dtype=bool)
    trace_round: list[int] = []
    trace_atom: list[int] = []
    trace_slot_row: list[int] = []
    trace_slot_id: list[int] = []
    trace_support: list[int] = []
    trace_support_id: list[int] = []
    trace_cost: list[int] = []
    consumed = 0
    round_index = 0
    atom_count = len(ordered)
    while consumed < graph.slot_count:
        consumed_before = consumed
        for offset in range(atom_count):
            atom_id = ordered[(round_index + offset) % atom_count]
            atom_rows = rows_by_atom[atom_id]
            atom_cursor = cursor[atom_id]
            if atom_cursor >= atom_rows.size:
                continue
            slot_row = int(atom_rows[atom_cursor])
            cursor[atom_id] = atom_cursor + 1
            consumed += 1
            start = int(graph.indptr[slot_row])
            stop = int(graph.indptr[slot_row + 1])
            choices = [
                position
                for position in range(start, stop)
                if unused[int(graph.indices[position])]
            ]
            if choices:
                position = min(
                    choices,
                    key=lambda value: (
                        int(graph.data[value]),
                        int(graph.indices[value]),
                    ),
                )
                support_rank = int(graph.indices[position])
                edge_cost = int(graph.data[position])
                unused[support_rank] = False
                support_id = int(support.stable_candidate_id[support_rank])
            else:
                support_rank = -1
                support_id = -1
                edge_cost = -1
            trace_round.append(round_index)
            trace_atom.append(atom_id)
            trace_slot_row.append(slot_row)
            trace_slot_id.append(int(graph.slot_id[slot_row]))
            trace_support.append(support_rank)
            trace_support_id.append(support_id)
            trace_cost.append(edge_cost)
        if consumed == consumed_before:
            raise AssertionError("Greedy rounds stopped before consuming every slot")
        round_index += 1
    rounds = _immutable_array(np.asarray(trace_round), "<i8")
    atoms = _immutable_array(np.asarray(trace_atom), "<i8")
    slot_rows = _immutable_array(np.asarray(trace_slot_row), "<i8")
    slot_ids = _immutable_array(np.asarray(trace_slot_id), "<i8")
    support_ranks = _immutable_array(np.asarray(trace_support), "<i8")
    support_ids = _immutable_array(np.asarray(trace_support_id), "<i8")
    edge_costs = _immutable_array(np.asarray(trace_cost), "<i8")
    selected_rank = _immutable_array(np.sort(support_ranks[support_ranks >= 0]), "<i8")
    selected_id = _immutable_array(
        support.stable_candidate_id[selected_rank], "<i8"
    )
    export_xyz = _immutable_array(support.xyz[selected_rank], "<f4")
    failed = int((support_ranks < 0).sum())
    assignment_hash = _hash_arrays(
        b"stda_f0_greedy_assignment_v1",
        rounds,
        atoms,
        slot_rows,
        slot_ids,
        support_ranks,
        support_ids,
        edge_costs,
    )
    selected_hash = _hash_arrays(
        b"stda_f0_greedy_selected_ids_v1", selected_id
    )
    export_hash = _hash_arrays(b"stda_f0_greedy_export_v1", export_xyz)
    digest = _hash_hex_fields(
        b"stda_f0_greedy_result_v1",
        atom_order.digest_sha256,
        assignment_hash,
        selected_hash,
        export_hash,
    )
    return GreedyResult(
        atom_order=atom_order,
        round_index=rounds,
        atom_id=atoms,
        slot_row=slot_rows,
        slot_id=slot_ids,
        support_rank=support_ranks,
        support_id=support_ids,
        edge_cost=edge_costs,
        selected_support_rank=selected_rank,
        selected_support_id=selected_id,
        export_xyz=export_xyz,
        failed_slot_count=failed,
        assignment_sha256=assignment_hash,
        selected_id_sha256=selected_hash,
        export_sha256=export_hash,
        digest_sha256=digest,
    )


def target_atom_round_robin_greedy(
    support: PackedSupport,
    graph: AssignmentGraph,
    sidecar: GreedySidecar,
    *,
    sequence: int,
    radar_index: int,
) -> GreedyResult:
    """Run the frozen r=0 forward-rotation greedy control without repair."""

    return _run_greedy_control(
        support,
        graph,
        sidecar,
        sequence=sequence,
        radar_index=radar_index,
    )


def verify_greedy_control(
    support: PackedSupport,
    graph: AssignmentGraph,
    sidecar: GreedySidecar,
    result: GreedyResult,
    *,
    sequence: int,
    radar_index: int,
    reference: GreedyResult | None = None,
) -> GreedyReplayChecks:
    expected = _run_greedy_control(
        support,
        graph,
        sidecar,
        sequence=sequence,
        radar_index=radar_index,
    )
    all_slots_consumed_once = (
        result.slot_row.size == graph.slot_count
        and np.unique(result.slot_row).size == graph.slot_count
        and set(int(value) for value in result.slot_row)
        == set(range(graph.slot_count))
    )
    atom_and_slot_order_equal = all(
        (
            np.array_equal(result.atom_id, expected.atom_id),
            np.array_equal(result.slot_row, expected.slot_row),
            np.array_equal(result.slot_id, expected.slot_id),
        )
    )
    forward_rotation_equal = all(
        (
            np.array_equal(result.round_index, expected.round_index),
            result.atom_order.frame_key == expected.atom_order.frame_key,
            np.array_equal(
                result.atom_order.ordered_atom_id,
                expected.atom_order.ordered_atom_id,
            ),
            np.array_equal(
                result.atom_order.ordered_digest_bytes,
                expected.atom_order.ordered_digest_bytes,
            ),
            result.atom_order.digest_sha256 == expected.atom_order.digest_sha256,
        )
    )
    selected = result.support_rank[result.support_rank >= 0]
    candidate_capacity = np.unique(selected).size == selected.size
    edge_choice_equal = all(
        (
            np.array_equal(result.support_rank, expected.support_rank),
            np.array_equal(result.support_id, expected.support_id),
            np.array_equal(result.edge_cost, expected.edge_cost),
        )
    )
    failed_slots_preserved = (
        result.failed_slot_count == int((result.support_rank < 0).sum())
        and result.failed_slot_count == expected.failed_slot_count
    )
    export_equal = all(
        (
            np.array_equal(
                result.selected_support_rank, expected.selected_support_rank
            ),
            np.array_equal(result.selected_support_id, expected.selected_support_id),
            np.array_equal(result.export_xyz, expected.export_xyz),
        )
    )
    assignment_hash = _hash_arrays(
        b"stda_f0_greedy_assignment_v1",
        result.round_index,
        result.atom_id,
        result.slot_row,
        result.slot_id,
        result.support_rank,
        result.support_id,
        result.edge_cost,
    )
    selected_hash = _hash_arrays(
        b"stda_f0_greedy_selected_ids_v1", result.selected_support_id
    )
    export_hash = _hash_arrays(b"stda_f0_greedy_export_v1", result.export_xyz)
    digest = _hash_hex_fields(
        b"stda_f0_greedy_result_v1",
        result.atom_order.digest_sha256,
        assignment_hash,
        selected_hash,
        export_hash,
    )
    hashes_valid = all(
        (
            result.assignment_sha256 == assignment_hash,
            result.selected_id_sha256 == selected_hash,
            result.export_sha256 == export_hash,
            result.digest_sha256 == digest,
        )
    )
    reference_equal = reference is None or result.digest_sha256 == reference.digest_sha256
    return GreedyReplayChecks(
        all_slots_consumed_once=all_slots_consumed_once,
        atom_and_slot_order_equal=atom_and_slot_order_equal,
        forward_rotation_equal=forward_rotation_equal,
        candidate_capacity=bool(candidate_capacity),
        edge_choice_equal=edge_choice_equal,
        failed_slots_preserved=failed_slots_preserved,
        export_equal=export_equal,
        hashes_valid=hashes_valid,
        reference_equal=reference_equal,
    )


def source_file_set_sha256(
    domain: bytes,
    hashes: tuple[tuple[str, str], ...],
) -> str:
    """Expose the canonical fixed-file commitment for orchestration records."""

    return _hash_file_set(domain, hashes)
