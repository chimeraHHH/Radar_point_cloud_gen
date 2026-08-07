"""Independent byte-level verifier for the frozen STDA-F0 protocol.

This module deliberately does not import the support builder, target fitter,
rounding solver, or structural evaluator.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import math
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
PROTOCOL_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
DEMAND_COUNT = 10_000
GRAPH_K = 256
UNMATCHED_SENTINEL = 2**63 - 1

CONTROL_EXPORT_REPLAY_SCHEMA = "stda_f0_independent_control_export_replay_v1"

_SOLVER_INPUT_SCHEMA = {
    "support_stable_candidate_id.npy": ("<i8", 1),
    "support_grid_cell.npy": ("<i8", 2),
    "support_xyz.npy": ("<f4", 2),
    "support_base_confidence.npy": ("<f4", 1),
    "support_color.npy": ("<u1", 1),
    "graph_indptr.npy": ("<i8", 1),
    "graph_indices.npy": ("<i4", 1),
    "graph_data.npy": ("<i8", 1),
    "graph_edge_squared_distance_m2.npy": ("<f8", 1),
    "graph_edge_distance_m.npy": ("<f8", 1),
    "demand_slot_id.npy": ("<i8", 1),
}
_CONTROL_INPUT_SCHEMA = {
    "demand_slot_atom_id.npy": ("<i8", 1),
    "demand_atom_id.npy": ("<i8", 1),
    "demand_atom_xyz.npy": ("<f8", 2),
    "demand_atom_weight.npy": ("<f8", 1),
    "pointwise_nearest_atom_id.npy": ("<i8", 1),
    "pointwise_squared_distance.npy": ("<f8", 1),
    "pointwise_distance_m.npy": ("<f8", 1),
}
_CONTROL_RESULT_SCHEMA = {
    "decision_slot_row.npy": ("<i8", 1),
    "decision_slot_id.npy": ("<i8", 1),
    "decision_support_rank.npy": ("<i8", 1),
    "decision_support_id.npy": ("<i8", 1),
    "decision_edge_cost.npy": ("<i8", 1),
    "decision_selected_support_rank.npy": ("<i8", 1),
    "decision_selected_support_id.npy": ("<i8", 1),
    "packed_pointwise_selected_support_rank.npy": ("<i8", 1),
    "packed_pointwise_selected_support_id.npy": ("<i8", 1),
    "packed_pointwise_nearest_atom_id.npy": ("<i8", 1),
    "packed_pointwise_squared_distance.npy": ("<f8", 1),
    "packed_pointwise_distance_m.npy": ("<f8", 1),
    "round_robin_greedy_round_index.npy": ("<i8", 1),
    "round_robin_greedy_atom_id.npy": ("<i8", 1),
    "round_robin_greedy_slot_row.npy": ("<i8", 1),
    "round_robin_greedy_slot_id.npy": ("<i8", 1),
    "round_robin_greedy_support_rank.npy": ("<i8", 1),
    "round_robin_greedy_support_id.npy": ("<i8", 1),
    "round_robin_greedy_edge_cost.npy": ("<i8", 1),
    "round_robin_greedy_selected_support_rank.npy": ("<i8", 1),
    "round_robin_greedy_selected_support_id.npy": ("<i8", 1),
    "round_robin_greedy_ordered_atom_id.npy": ("<i8", 1),
    "round_robin_greedy_order_digest_bytes.npy": ("<u1", 2),
}
_EXPORT_ARMS = ("decision", "packed_pointwise", "round_robin_greedy")

STRUCTURAL_REPORT_SCHEMA = "stda_f0_structural_report_v1"
STRUCTURAL_REPLAY_SCHEMA = "stda_f0_independent_structural_replay_v1"
OUTPUT_RANGE_UPPER_M = 118.2685546875
TARGET_RANGE_UPPER_M = 120.0
UNMATCHED_DISTANCE_M = 120.0
GROUP_SPAN_M = 0.05
COMPLETENESS_GATE_M = 1.0
RECALL_GATE = 0.80
RETURN_FIRST = 0
RETURN_LATER = 1
RETURN_LABELS = ("first", "later")
RANGE_BOUNDS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
RANGE_LABELS = ("range_0_30", "range_30_60", "range_60_120")

_STRUCTURAL_HASH_NAMESPACE = b"stda_f0_structure_v1\0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _as_le_array(value: np.ndarray, dtype: str) -> np.ndarray:
    array = np.asarray(value)
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise TypeError(f"STDA-F0 expected {expected.str}, got {array.dtype.str}")
    if not array.flags.c_contiguous:
        raise ValueError("STDA-F0 canonical array must be C-contiguous")
    return array


def _float32_rows_from_bytes(payload: bytes, point_count: int | None) -> np.ndarray:
    if len(payload) % 12 != 0:
        raise ValueError("STDA-F0 XYZ payload is not a whole number of float32 rows")
    inferred = len(payload) // 12
    if point_count is not None and inferred != point_count:
        raise ValueError("STDA-F0 XYZ payload cardinality changed")
    xyz = np.frombuffer(payload, dtype="<f4").reshape(inferred, 3)
    if not bool(np.isfinite(xyz).all()):
        raise ValueError("STDA-F0 XYZ payload contains non-finite values")
    return xyz


def _ratio(value: np.float32) -> tuple[int, int]:
    return float(value).as_integer_ratio()


def _exact_cell(value: np.float32) -> int:
    numerator, denominator = _ratio(value)
    return (20 * numerator) // denominator


def _exact_squared_below_5cm(
    first: Sequence[np.float32],
    second: Sequence[np.float32],
) -> bool:
    differences: list[tuple[int, int]] = []
    for left, right in zip(first, second, strict=True):
        left_numerator, left_denominator = _ratio(left)
        right_numerator, right_denominator = _ratio(right)
        denominator = max(left_denominator, right_denominator)
        numerator = (
            left_numerator * (denominator // left_denominator)
            - right_numerator * (denominator // right_denominator)
        )
        differences.append((numerator, denominator))
    common_denominator = max(denominator for _, denominator in differences)
    squared_numerator = sum(
        numerator * numerator * (common_denominator // denominator) ** 2
        for numerator, denominator in differences
    )
    return 400 * squared_numerator < common_denominator * common_denominator


def verify_exact_spacing_bytes(
    payload: bytes,
    *,
    point_count: int | None = None,
) -> dict[str, Any]:
    """Exhaustively reject every serialized pair with distance below 5 cm."""

    xyz = _float32_rows_from_bytes(payload, point_count)
    row_bytes = [payload[offset : offset + 12] for offset in range(0, len(payload), 12)]
    duplicate_rows = len(row_bytes) - len(set(row_bytes))
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    compared_pairs = 0
    violations: list[tuple[int, int]] = []
    offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    )
    for row, point in enumerate(xyz):
        cell = tuple(_exact_cell(value) for value in point)
        for dx, dy, dz in offsets:
            neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            for prior in cells.get(neighbor, ()):
                compared_pairs += 1
                if _exact_squared_below_5cm(point, xyz[prior]):
                    violations.append((prior, row))
        cells[cell].append(row)
    return {
        "point_count": int(xyz.shape[0]),
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "finite_xyz": True,
        "unique_xyz_bytes": duplicate_rows == 0,
        "duplicate_xyz_byte_rows": duplicate_rows,
        "occupied_exact_cells": len(cells),
        "enumerated_neighbor_pairs": compared_pairs,
        "violation_count": len(violations),
        "first_violation_rows": list(violations[0]) if violations else None,
        "strict_spacing_5cm": not violations,
        "passed": duplicate_rows == 0 and not violations,
        "arithmetic": "float32_as_integer_ratio_integer_cross_multiplication",
    }


def verify_packed_support_bytes(serialized: bytes) -> dict[str, Any]:
    """Parse and verify support bytes without importing the support builder."""

    if not isinstance(serialized, bytes):
        raise TypeError("STDA-F0 support verification requires immutable bytes")
    newline = serialized.find(b"\n")
    if newline <= 0 or newline >= 64 * 1024:
        raise ValueError("STDA-F0 packed-support header boundary is invalid")
    header_bytes = serialized[: newline + 1]
    payload = serialized[newline + 1 :]
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("STDA-F0 packed-support header is invalid") from error
    if not isinstance(header, dict):
        raise ValueError("STDA-F0 packed-support header must be an object")
    canonical = (
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    if canonical != header_bytes:
        raise ValueError("STDA-F0 packed-support header is not canonical JSON")
    if header.get("schema") != "stda_f0_packed_support_v1":
        raise ValueError("STDA-F0 packed-support schema changed")
    if header.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("STDA-F0 packed-support protocol hash changed")
    if header.get("protocol_freeze_commit") != PROTOCOL_FREEZE_COMMIT:
        raise ValueError("STDA-F0 packed-support freeze commit changed")
    count = header.get("support_count")
    if type(count) is not int or count < 0:
        raise ValueError("STDA-F0 packed-support cardinality is invalid")
    if header.get("payload_sha256") != sha256_bytes(payload):
        raise ValueError("STDA-F0 packed-support payload hash changed")
    schema = (
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
    expected_columns = [
        {"name": name, "dtype": dtype} for name, dtype in schema
    ]
    if header.get("columns") != expected_columns:
        raise ValueError("STDA-F0 packed-support columns changed")
    bytes_per_row = sum(np.dtype(dtype).itemsize for _, dtype in schema)
    if len(payload) != count * bytes_per_row:
        raise ValueError("STDA-F0 packed-support payload size changed")
    columns: dict[str, np.ndarray] = {}
    offset = 0
    for name, dtype in schema:
        stop = offset + count * np.dtype(dtype).itemsize
        columns[name] = np.frombuffer(payload[offset:stop], dtype=dtype)
        offset = stop
    stable_id = columns["stable_candidate_id"]
    cells = np.column_stack(
        (columns["cell_x"], columns["cell_y"], columns["cell_z"])
    ).astype("<i8", copy=False)
    xyz = np.column_stack(
        (columns["x_m"], columns["y_m"], columns["z_m"])
    ).astype("<f4", copy=False)
    confidence = columns["base_confidence"]
    color = columns["color_id"]
    if count > 1 and not bool(np.all(stable_id[1:] > stable_id[:-1])):
        raise ValueError("STDA-F0 packed-support IDs are not strictly increasing")
    if not bool(np.isfinite(xyz).all() and np.isfinite(confidence).all()):
        raise ValueError("STDA-F0 packed-support values are non-finite")
    if np.unique(cells, axis=0).shape[0] != count:
        raise ValueError("STDA-F0 packed-support contains duplicate cells")
    replay_cells = np.asarray(
        [[_exact_cell(value) for value in point] for point in xyz],
        dtype="<i8",
    )
    if not np.array_equal(cells, replay_cells):
        raise ValueError("STDA-F0 packed-support exact cells changed")
    replay_color = (
        4 * np.mod(cells[:, 0], 2)
        + 2 * np.mod(cells[:, 1], 2)
        + np.mod(cells[:, 2], 2)
    ).astype("<u1", copy=False)
    selected_color = header.get("selected_color_id")
    color_cardinalities = header.get("color_cardinalities")
    if (
        type(selected_color) is not int
        or not isinstance(color_cardinalities, list)
        or len(color_cardinalities) != 8
        or any(type(value) is not int or value < 0 for value in color_cardinalities)
    ):
        raise ValueError("STDA-F0 packed-support color metadata is invalid")
    expected_color = min(
        range(8), key=lambda value: (-color_cardinalities[value], value)
    )
    if (
        selected_color != expected_color
        or color_cardinalities[selected_color] != count
        or not np.array_equal(color, replay_color)
        or (count and not bool(np.all(color == selected_color)))
    ):
        raise ValueError("STDA-F0 packed-support color replay changed")
    xyz_bytes = np.ascontiguousarray(xyz, dtype="<f4").tobytes(order="C")
    spacing = verify_exact_spacing_bytes(xyz_bytes, point_count=count)
    checks = {
        "canonical_header": True,
        "payload_hash": True,
        "stable_ids": True,
        "finite_values": True,
        "unique_cells": True,
        "exact_cells": True,
        "largest_color": True,
        "strict_spacing_5cm": bool(spacing["strict_spacing_5cm"]),
        "unique_xyz_bytes": bool(spacing["unique_xyz_bytes"]),
    }
    return {
        "support_count": count,
        "support_sha256": sha256_bytes(serialized),
        "candidate_field_sha256": header.get("candidate_field_sha256"),
        "selected_color_id": selected_color,
        "color_cardinalities": color_cardinalities,
        "xyz_sha256": sha256_bytes(xyz_bytes),
        "spacing": spacing,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    *,
    support_cardinality: int,
    expected_rows: int = DEMAND_COUNT,
    expected_k: int = GRAPH_K,
) -> dict[str, Any]:
    indptr = _as_le_array(indptr, "<i8")
    indices = _as_le_array(indices, "<i4")
    data = _as_le_array(data, "<i8")
    if indptr.ndim != 1 or indptr.shape != (expected_rows + 1,):
        raise ValueError("STDA-F0 CSR indptr shape changed")
    expected_edges = expected_rows * expected_k
    if indices.ndim != 1 or data.ndim != 1:
        raise ValueError("STDA-F0 CSR edge arrays must be one-dimensional")
    if indices.size != expected_edges or data.size != expected_edges:
        raise ValueError("STDA-F0 CSR edge cardinality changed")
    if int(indptr[0]) != 0 or int(indptr[-1]) != expected_edges:
        raise ValueError("STDA-F0 CSR endpoints changed")
    row_sizes = np.diff(indptr)
    exact_k = bool(np.all(row_sizes == expected_k))
    in_bounds = bool(
        np.all(indices >= 0) and np.all(indices < int(support_cardinality))
    )
    positive_cost = bool(np.all(data > 0))
    below_2p53 = bool(np.all(data < 2**53))
    sorted_unique = True
    maximum_sum = 0
    for row in range(expected_rows):
        begin = int(indptr[row])
        end = int(indptr[row + 1])
        columns = indices[begin:end]
        if columns.size != expected_k or not bool(np.all(columns[1:] > columns[:-1])):
            sorted_unique = False
        if end > begin:
            maximum_sum += int(np.max(data[begin:end]))
    checks = {
        "exact_graph_entries": indices.size == expected_edges,
        "exact_k_per_row": exact_k,
        "indices_in_support": in_bounds,
        "indices_strictly_increasing": sorted_unique,
        "positive_int64_cost": positive_cost,
        "edge_cost_below_2p53": below_2p53,
        "row_maximum_sum_below_int64": maximum_sum < 2**63,
    }
    return {
        "shape": [expected_rows, int(support_cardinality)],
        "edge_count": int(indices.size),
        "row_maximum_cost_sum": maximum_sum,
        "checks": checks,
        "passed": all(checks.values()),
        "hashes": {
            "indptr_sha256": sha256_bytes(indptr.tobytes()),
            "indices_sha256": sha256_bytes(indices.tobytes()),
            "data_sha256": sha256_bytes(data.tobytes()),
        },
    }


def _edge_lookup(
    row: int,
    column: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> int | None:
    begin = int(indptr[row])
    end = int(indptr[row + 1])
    position = int(np.searchsorted(indices[begin:end], column, side="left")) + begin
    if position >= end or int(indices[position]) != column:
        return None
    return int(data[position])


def verify_full_assignment(
    row_to_support: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    *,
    support_cardinality: int,
    expected_objective: int | None = None,
) -> dict[str, Any]:
    assignment = _as_le_array(row_to_support, "<i8")
    indptr = _as_le_array(indptr, "<i8")
    indices = _as_le_array(indices, "<i4")
    data = _as_le_array(data, "<i8")
    if assignment.shape != (DEMAND_COUNT,):
        raise ValueError("STDA-F0 assignment must cover every demand slot")
    in_bounds = bool(
        np.all(assignment >= 0) and np.all(assignment < support_cardinality)
    )
    unique = int(np.unique(assignment).size) == DEMAND_COUNT
    objective = 0
    all_edges_present = in_bounds
    if in_bounds:
        for row, column in enumerate(assignment.tolist()):
            cost = _edge_lookup(row, int(column), indptr, indices, data)
            if cost is None:
                all_edges_present = False
                break
            objective += cost
    checks = {
        "all_slots_assigned": assignment.size == DEMAND_COUNT,
        "support_indices_in_bounds": in_bounds,
        "support_capacity_one": unique,
        "all_assignments_are_graph_edges": all_edges_present,
        "objective_matches": expected_objective is None
        or objective == int(expected_objective),
        "objective_below_int64": objective < 2**63,
    }
    return {
        "objective_python_int": objective,
        "assignment_sha256": sha256_bytes(assignment.tobytes()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _matching_cardinality(row_to_support: np.ndarray) -> int:
    return sum(int(column) >= 0 for column in row_to_support.tolist())


def verify_matching(
    row_to_support: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    *,
    support_cardinality: int,
) -> dict[str, Any]:
    matching = _as_le_array(row_to_support, "<i8")
    indptr = _as_le_array(indptr, "<i8")
    indices = _as_le_array(indices, "<i4")
    if matching.shape != (DEMAND_COUNT,):
        raise ValueError("STDA-F0 maximum matching row vector changed")
    selected = matching[matching >= 0]
    in_bounds = bool(np.all(selected < support_cardinality))
    capacity_one = int(np.unique(selected).size) == int(selected.size)
    edge_membership = in_bounds
    if in_bounds:
        for row, column in enumerate(matching.tolist()):
            if column < 0:
                continue
            begin = int(indptr[row])
            end = int(indptr[row + 1])
            position = int(np.searchsorted(indices[begin:end], column)) + begin
            if position >= end or int(indices[position]) != column:
                edge_membership = False
                break
    checks = {
        "matched_support_in_bounds": in_bounds,
        "support_capacity_one": capacity_one,
        "all_matches_are_graph_edges": edge_membership,
    }
    return {
        "cardinality": _matching_cardinality(matching),
        "matching_sha256": sha256_bytes(matching.tobytes()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_hall_certificate(
    reachable_slots: np.ndarray,
    reachable_support: np.ndarray,
    row_to_support: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    *,
    support_cardinality: int,
) -> dict[str, Any]:
    slots = _as_le_array(reachable_slots, "<i8")
    support = _as_le_array(reachable_support, "<i8")
    matching = _as_le_array(row_to_support, "<i8")
    indptr = _as_le_array(indptr, "<i8")
    indices = _as_le_array(indices, "<i4")
    if slots.ndim != 1 or support.ndim != 1:
        raise ValueError("STDA-F0 Hall sets must be one-dimensional")
    canonical_slots = np.unique(slots)
    canonical_support = np.unique(support)
    slot_bounds = bool(np.all(canonical_slots >= 0) and np.all(canonical_slots < DEMAND_COUNT))
    support_bounds = bool(
        np.all(canonical_support >= 0)
        and np.all(canonical_support < support_cardinality)
    )
    neighbors: set[int] = set()
    if slot_bounds:
        for row in canonical_slots.tolist():
            begin = int(indptr[row])
            end = int(indptr[row + 1])
            neighbors.update(int(value) for value in indices[begin:end])
    reconstructed = np.asarray(sorted(neighbors), dtype="<i8")
    cardinality = _matching_cardinality(matching)
    deficit = int(canonical_slots.size - reconstructed.size)
    checks = {
        "slot_set_canonical": np.array_equal(slots, canonical_slots),
        "support_set_canonical": np.array_equal(support, canonical_support),
        "slot_ids_in_bounds": slot_bounds,
        "support_ids_in_bounds": support_bounds,
        "neighbor_set_exact": np.array_equal(canonical_support, reconstructed),
        "strict_hall_inequality": reconstructed.size < canonical_slots.size,
        "deficit_bounded_by_unmatched": DEMAND_COUNT - cardinality >= deficit,
    }
    return {
        "matching_cardinality": cardinality,
        "reachable_slot_count": int(canonical_slots.size),
        "reachable_support_count": int(reconstructed.size),
        "hall_deficit": deficit,
        "reachable_slots_sha256": sha256_bytes(canonical_slots.tobytes()),
        "reachable_support_sha256": sha256_bytes(reconstructed.tobytes()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _round_hash_arrays(domain: bytes, *arrays: np.ndarray) -> str:
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


def _round_hash_hex_fields(domain: bytes, *fields: str) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for field in fields:
        encoded = field.encode("ascii")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _load_npy_bytes(
    payload: bytes,
    *,
    filename: str,
    dtype: str,
    ndim: int,
) -> np.ndarray:
    if not isinstance(payload, bytes):
        raise TypeError(f"STDA-F0 {filename} must be immutable bytes")
    stream = io.BytesIO(payload)
    try:
        loaded = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"STDA-F0 {filename} is not a safe NPY array") from error
    if stream.read(1) != b"":
        raise ValueError(f"STDA-F0 {filename} has trailing bytes")
    if not isinstance(loaded, np.ndarray):
        raise TypeError(f"STDA-F0 {filename} did not decode to an array")
    if loaded.dtype != np.dtype(dtype):
        raise ValueError(
            f"STDA-F0 {filename} dtype {loaded.dtype.str} does not equal {dtype}"
        )
    if loaded.ndim != ndim:
        raise ValueError(
            f"STDA-F0 {filename} rank {loaded.ndim} does not equal {ndim}"
        )
    if not loaded.flags.c_contiguous:
        raise ValueError(f"STDA-F0 {filename} is not C-contiguous")
    return np.frombuffer(
        loaded.tobytes(order="C"), dtype=np.dtype(dtype)
    ).reshape(loaded.shape)


def _load_required_arrays(
    payloads: Mapping[str, bytes],
    schema: Mapping[str, tuple[str, int]],
    *,
    label: str,
) -> dict[str, np.ndarray]:
    missing = sorted(set(schema) - set(payloads))
    if missing:
        raise ValueError(f"STDA-F0 {label} is missing files: {missing}")
    return {
        filename: _load_npy_bytes(
            payloads[filename],
            filename=filename,
            dtype=dtype,
            ndim=ndim,
        )
        for filename, (dtype, ndim) in schema.items()
    }


def _file_set_binding(
    payloads: Mapping[str, bytes],
    expected_sha256: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(expected_sha256, Mapping):
        raise ValueError(f"STDA-F0 {label} SHA binding must be a mapping")
    expected_names = set(expected_sha256)
    observed_names = set(payloads)
    file_reports: dict[str, dict[str, Any]] = {}
    hashes_valid = True
    for filename in sorted(observed_names | expected_names):
        payload = payloads.get(filename)
        expected = expected_sha256.get(filename)
        observed = sha256_bytes(payload) if isinstance(payload, bytes) else None
        valid_expected = (
            isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected)
        )
        matched = isinstance(payload, bytes) and valid_expected and observed == expected
        hashes_valid &= matched
        file_reports[filename] = {
            "bytes": len(payload) if isinstance(payload, bytes) else None,
            "observed_sha256": observed,
            "expected_sha256": expected,
            "matched": matched,
        }
    checks = {
        "exact_file_names": observed_names == expected_names,
        "all_payloads_are_bytes": all(
            isinstance(value, bytes) for value in payloads.values()
        ),
        "all_file_hashes_match": hashes_valid,
    }
    return {
        "label": label,
        "checks": checks,
        "files": file_reports,
        "passed": all(checks.values()),
    }


def _validate_replay_inputs(
    solver: Mapping[str, np.ndarray],
    controls: Mapping[str, np.ndarray],
    *,
    required_export_count: int,
    expected_graph_k: int | None,
) -> dict[str, Any]:
    stable_id = solver["support_stable_candidate_id.npy"]
    grid_cell = solver["support_grid_cell.npy"]
    xyz = solver["support_xyz.npy"]
    confidence = solver["support_base_confidence.npy"]
    color = solver["support_color.npy"]
    support_count = int(stable_id.size)
    support_row_bytes = [
        xyz[row].tobytes(order="C") for row in range(support_count)
    ]
    support_checks = {
        "positive_required_count": type(required_export_count) is int
        and required_export_count > 0,
        "support_has_required_capacity": support_count >= required_export_count,
        "support_shapes": (
            stable_id.shape == (support_count,)
            and grid_cell.shape == (support_count, 3)
            and xyz.shape == (support_count, 3)
            and confidence.shape == (support_count,)
            and color.shape == (support_count,)
        ),
        "stable_ids_strictly_increasing": support_count > 0
        and bool(np.all(stable_id[1:] > stable_id[:-1])),
        "support_finite": bool(
            np.isfinite(xyz).all() and np.isfinite(confidence).all()
        ),
        "support_unique_xyz_bytes": len(set(support_row_bytes)) == support_count,
        "support_unique_grid_cells": (
            np.unique(grid_cell, axis=0).shape[0] == support_count
        ),
        "support_exact_grid_cells": np.array_equal(
            grid_cell,
            np.asarray(
                [[_exact_cell(value) for value in point] for point in xyz],
                dtype="<i8",
            ),
        ),
        "support_one_parity_color": (
            support_count > 0
            and np.unique(color).size == 1
            and bool(np.all(color <= 7))
            and np.array_equal(
                color,
                (
                    4 * np.mod(grid_cell[:, 0], 2)
                    + 2 * np.mod(grid_cell[:, 1], 2)
                    + np.mod(grid_cell[:, 2], 2)
                ).astype("<u1"),
            )
        ),
    }

    indptr = solver["graph_indptr.npy"]
    indices = solver["graph_indices.npy"]
    data = solver["graph_data.npy"]
    squared = solver["graph_edge_squared_distance_m2.npy"]
    distance = solver["graph_edge_distance_m.npy"]
    slot_id = solver["demand_slot_id.npy"]
    slot_count = int(slot_id.size)
    valid_indptr = (
        indptr.shape == (slot_count + 1,)
        and int(indptr[0]) == 0
        and bool(np.all(np.diff(indptr) >= 0))
    )
    edge_count = int(indptr[-1]) if valid_indptr else -1
    edge_shapes = (
        edge_count >= 0
        and indices.shape
        == data.shape
        == squared.shape
        == distance.shape
        == (edge_count,)
    )
    row_columns_sorted = valid_indptr and edge_shapes
    if row_columns_sorted:
        for row in range(slot_count):
            start = int(indptr[row])
            stop = int(indptr[row + 1])
            columns = indices[start:stop]
            if not bool(np.all(columns[1:] > columns[:-1])):
                row_columns_sorted = False
                break
    expected_degrees = (
        expected_graph_k is None
        or (
            type(expected_graph_k) is int
            and expected_graph_k > 0
            and valid_indptr
            and bool(np.all(np.diff(indptr) == expected_graph_k))
        )
    )
    sqrt_replay = False
    if edge_shapes and bool(np.isfinite(squared).all()) and bool(np.all(squared >= 0.0)):
        replayed = np.fromiter(
            (math.sqrt(float(value)) for value in squared),
            dtype="<f8",
            count=edge_count,
        )
        sqrt_replay = np.array_equal(replayed, distance)
    graph_checks = {
        "slot_count": slot_count == required_export_count,
        "slot_ids_canonical": np.array_equal(
            slot_id, np.arange(slot_count, dtype="<i8")
        ),
        "valid_indptr": valid_indptr,
        "edge_array_shapes": edge_shapes,
        "expected_degree": expected_degrees,
        "support_indices_in_bounds": edge_shapes
        and bool(np.all(indices >= 0))
        and bool(np.all(indices < support_count)),
        "row_columns_strictly_increasing": row_columns_sorted,
        "positive_exact_integer_costs": edge_shapes
        and bool(np.all(data > 0))
        and bool(np.all(data < 2**53)),
        "edge_distance_sqrt_replay": sqrt_replay,
    }

    slot_atom = controls["demand_slot_atom_id.npy"]
    atom_id = controls["demand_atom_id.npy"]
    atom_xyz = controls["demand_atom_xyz.npy"]
    atom_weight = controls["demand_atom_weight.npy"]
    nearest_atom = controls["pointwise_nearest_atom_id.npy"]
    point_squared = controls["pointwise_squared_distance.npy"]
    point_distance = controls["pointwise_distance_m.npy"]
    atom_count = int(atom_id.size)
    atom_positions = np.searchsorted(atom_id, slot_atom)
    known_slot_atoms = bool(np.all(atom_positions < atom_count))
    if known_slot_atoms and slot_atom.size:
        known_slot_atoms = np.array_equal(atom_id[atom_positions], slot_atom)
    nearest_positions = np.searchsorted(atom_id, nearest_atom)
    known_nearest_atoms = bool(np.all(nearest_positions < atom_count))
    if known_nearest_atoms and nearest_atom.size:
        known_nearest_atoms = np.array_equal(atom_id[nearest_positions], nearest_atom)
    point_sqrt = False
    if (
        point_squared.shape == (support_count,)
        and bool(np.isfinite(point_squared).all())
        and bool(np.all(point_squared >= 0.0))
    ):
        replayed_point_distance = np.fromiter(
            (math.sqrt(float(value)) for value in point_squared),
            dtype="<f8",
            count=support_count,
        )
        point_sqrt = np.array_equal(replayed_point_distance, point_distance)
    control_checks = {
        "greedy_shapes": (
            slot_atom.shape == (slot_count,)
            and atom_count > 0
            and atom_xyz.shape == (atom_count, 3)
            and atom_weight.shape == (atom_count,)
        ),
        "atom_ids_strictly_increasing": atom_count > 0
        and bool(np.all(atom_id[1:] > atom_id[:-1])),
        "slot_atoms_known": known_slot_atoms,
        "atom_values_valid": bool(
            np.isfinite(atom_xyz).all()
            and np.isfinite(atom_weight).all()
            and np.all(atom_weight > 0.0)
        ),
        "atom_xyz_exact_float32_promotion": np.array_equal(
            atom_xyz.astype("<f4").astype("<f8"), atom_xyz
        ),
        "pointwise_shapes": (
            nearest_atom.shape
            == point_squared.shape
            == point_distance.shape
            == (support_count,)
        ),
        "pointwise_atoms_known": known_nearest_atoms,
        "pointwise_values_valid": bool(
            np.all(nearest_atom >= 0)
            and np.isfinite(point_squared).all()
            and np.isfinite(point_distance).all()
            and np.all(point_squared >= 0.0)
            and np.all(point_distance >= 0.0)
        ),
        "pointwise_distance_sqrt_replay": point_sqrt,
    }
    digests = {
        "round_support_sha256": _round_hash_arrays(
            b"stda_f0_packed_support_v1",
            stable_id,
            grid_cell,
            xyz,
            confidence,
            color,
        ),
        "round_graph_sha256": _round_hash_arrays(
            b"stda_f0_assignment_graph_v1",
            indptr,
            indices,
            data,
            squared,
            distance,
            slot_id,
            np.asarray([support_count], dtype="<i8"),
        ),
        "round_greedy_sidecar_sha256": _round_hash_arrays(
            b"stda_f0_greedy_sidecar_v1",
            slot_atom,
            atom_id,
            atom_xyz,
            atom_weight,
        ),
        "round_pointwise_sidecar_sha256": _round_hash_arrays(
            b"stda_f0_pointwise_sidecar_v1",
            nearest_atom,
            point_squared,
            point_distance,
        ),
    }
    checks = {
        "support": all(support_checks.values()),
        "graph": all(graph_checks.values()),
        "controls": all(control_checks.values()),
    }
    return {
        "support_count": support_count,
        "slot_count": slot_count,
        "support_checks": support_checks,
        "graph_checks": graph_checks,
        "control_checks": control_checks,
        "digests": digests,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _expected_report_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"STDA-F0 {label} report must be a mapping")
    return value


def _decision_replay(
    solver: Mapping[str, np.ndarray],
    results: Mapping[str, np.ndarray],
    expected_report: Mapping[str, Any],
    *,
    required_count: int,
) -> tuple[dict[str, Any], np.ndarray]:
    stable_id = solver["support_stable_candidate_id.npy"]
    support_xyz = solver["support_xyz.npy"]
    indptr = solver["graph_indptr.npy"]
    indices = solver["graph_indices.npy"]
    data = solver["graph_data.npy"]
    graph_slot_id = solver["demand_slot_id.npy"]
    slot_row = results["decision_slot_row.npy"]
    slot_id = results["decision_slot_id.npy"]
    support_rank = results["decision_support_rank.npy"]
    support_id = results["decision_support_id.npy"]
    edge_cost = results["decision_edge_cost.npy"]
    selected_rank = results["decision_selected_support_rank.npy"]
    selected_id = results["decision_selected_support_id.npy"]
    trace_shape = all(
        array.shape == (required_count,)
        for array in (slot_row, slot_id, support_rank, support_id, edge_cost)
    )
    selected_shape = selected_rank.shape == selected_id.shape == (required_count,)
    all_slots_once = trace_shape and np.array_equal(
        slot_row, np.arange(required_count, dtype="<i8")
    ) and np.array_equal(slot_id, graph_slot_id)
    ranks_in_bounds = trace_shape and bool(
        np.all(support_rank >= 0) and np.all(support_rank < stable_id.size)
    )
    candidate_capacity = ranks_in_bounds and (
        np.unique(support_rank).size == required_count
    )
    support_id_binding = ranks_in_bounds and np.array_equal(
        support_id, stable_id[support_rank]
    )
    edge_membership = bool(all_slots_once and ranks_in_bounds)
    edge_cost_binding = bool(all_slots_once and ranks_in_bounds)
    replayed_cost: list[int] = []
    if edge_membership:
        for row, rank in zip(slot_row.tolist(), support_rank.tolist(), strict=True):
            cost = _edge_lookup(int(row), int(rank), indptr, indices, data)
            if cost is None:
                edge_membership = False
                replayed_cost.append(-1)
            else:
                replayed_cost.append(cost)
        edge_cost_binding = np.array_equal(
            edge_cost, np.asarray(replayed_cost, dtype="<i8")
        )
    canonical_rank = (
        np.sort(support_rank) if trace_shape else np.empty(0, dtype="<i8")
    )
    canonical_selection = (
        selected_shape
        and candidate_capacity
        and np.array_equal(selected_rank, canonical_rank)
        and np.array_equal(selected_id, stable_id[selected_rank])
    )
    export_xyz = (
        np.ascontiguousarray(support_xyz[selected_rank], dtype="<f4")
        if canonical_selection
        else np.empty((0, 3), dtype="<f4")
    )
    objective = sum(int(value) for value in edge_cost.tolist()) if trace_shape else -1
    assignment_sha256 = _round_hash_arrays(
        b"stda_f0_decision_assignment_v1",
        slot_row,
        slot_id,
        support_rank,
        support_id,
        edge_cost,
    )
    selected_id_sha256 = _round_hash_arrays(
        b"stda_f0_decision_selected_ids_v1", selected_id
    )
    export_sha256 = _round_hash_arrays(
        b"stda_f0_decision_export_v1", export_xyz
    )
    objective_sha256 = _round_hash_arrays(
        b"stda_f0_decision_objective_v1",
        np.asarray([objective], dtype="<i8"),
    )
    digest_sha256 = _round_hash_hex_fields(
        b"stda_f0_decision_result_v1",
        assignment_sha256,
        selected_id_sha256,
        export_sha256,
        objective_sha256,
    )
    hashes = {
        "assignment_sha256": assignment_sha256,
        "selected_id_sha256": selected_id_sha256,
        "export_sha256": export_sha256,
        "objective_sha256": objective_sha256,
        "digest_sha256": digest_sha256,
    }
    hash_checks = {
        key: expected_report.get(key) == value for key, value in hashes.items()
    }
    checks = {
        "trace_shape": trace_shape,
        "selected_shape": selected_shape,
        "all_slots_once": all_slots_once,
        "support_ranks_in_bounds": ranks_in_bounds,
        "support_capacity_one": candidate_capacity,
        "support_id_binding": support_id_binding,
        "edge_membership": edge_membership,
        "edge_cost_binding": edge_cost_binding,
        "canonical_selection": canonical_selection,
        "objective_matches": expected_report.get("objective") == objective,
        "hashes_match": all(hash_checks.values()),
    }
    return (
        {
            "checks": checks,
            "hash_checks": hash_checks,
            "computed_hashes": hashes,
            "objective": objective,
            "selected_count": int(selected_rank.size),
            "passed": all(checks.values()),
        },
        export_xyz,
    )


def _packed_pointwise_replay(
    solver: Mapping[str, np.ndarray],
    controls: Mapping[str, np.ndarray],
    results: Mapping[str, np.ndarray],
    expected_report: Mapping[str, Any],
    *,
    required_count: int,
) -> tuple[dict[str, Any], np.ndarray]:
    stable_id = solver["support_stable_candidate_id.npy"]
    support_xyz = solver["support_xyz.npy"]
    nearest_sidecar = controls["pointwise_nearest_atom_id.npy"]
    squared_sidecar = controls["pointwise_squared_distance.npy"]
    distance_sidecar = controls["pointwise_distance_m.npy"]
    selected_rank = results["packed_pointwise_selected_support_rank.npy"]
    selected_id = results["packed_pointwise_selected_support_id.npy"]
    nearest_atom = results["packed_pointwise_nearest_atom_id.npy"]
    squared_distance = results["packed_pointwise_squared_distance.npy"]
    distance_m = results["packed_pointwise_distance_m.npy"]
    shapes = all(
        array.shape == (required_count,)
        for array in (
            selected_rank,
            selected_id,
            nearest_atom,
            squared_distance,
            distance_m,
        )
    )
    ranks_in_bounds = shapes and bool(
        np.all(selected_rank >= 0) and np.all(selected_rank < stable_id.size)
    )
    candidate_capacity = ranks_in_bounds and (
        np.unique(selected_rank).size == required_count
    )
    expected_order = np.lexsort((stable_id, squared_sidecar))[:required_count]
    frozen_order = shapes and np.array_equal(selected_rank, expected_order)
    sidecar_binding = ranks_in_bounds and all(
        (
            np.array_equal(selected_id, stable_id[selected_rank]),
            np.array_equal(nearest_atom, nearest_sidecar[selected_rank]),
            np.array_equal(squared_distance, squared_sidecar[selected_rank]),
            np.array_equal(distance_m, distance_sidecar[selected_rank]),
        )
    )
    export_xyz = (
        np.ascontiguousarray(support_xyz[selected_rank], dtype="<f4")
        if ranks_in_bounds
        else np.empty((0, 3), dtype="<f4")
    )
    selection_sha256 = _round_hash_arrays(
        b"stda_f0_packed_pointwise_selection_v1",
        selected_rank,
        selected_id,
        nearest_atom,
        squared_distance,
        distance_m,
    )
    selected_id_sha256 = _round_hash_arrays(
        b"stda_f0_packed_pointwise_selected_ids_v1", selected_id
    )
    export_sha256 = _round_hash_arrays(
        b"stda_f0_packed_pointwise_export_v1", export_xyz
    )
    digest_sha256 = _round_hash_hex_fields(
        b"stda_f0_packed_pointwise_result_v1",
        selection_sha256,
        selected_id_sha256,
        export_sha256,
    )
    hashes = {
        "selection_sha256": selection_sha256,
        "selected_id_sha256": selected_id_sha256,
        "export_sha256": export_sha256,
        "digest_sha256": digest_sha256,
    }
    hash_checks = {
        key: expected_report.get(key) == value for key, value in hashes.items()
    }
    checks = {
        "exact_count": shapes,
        "support_ranks_in_bounds": ranks_in_bounds,
        "support_capacity_one": candidate_capacity,
        "frozen_squared_distance_then_id_order": frozen_order,
        "selected_sidecars_bound": sidecar_binding,
        "export_bound_to_shared_support": ranks_in_bounds,
        "hashes_match": all(hash_checks.values()),
    }
    return (
        {
            "checks": checks,
            "hash_checks": hash_checks,
            "computed_hashes": hashes,
            "selected_count": int(selected_rank.size),
            "passed": all(checks.values()),
        },
        export_xyz,
    )


def _independent_greedy_atom_order(
    slot_atom_id: np.ndarray,
    atom_id: np.ndarray,
    atom_xyz: np.ndarray,
    atom_weight: np.ndarray,
    *,
    sequence: int,
    radar_index: int,
) -> tuple[bytes, np.ndarray, np.ndarray, str]:
    if sequence < 0 or radar_index < 0:
        raise ValueError("STDA-F0 frame indices must be nonnegative")
    frame_key = f"seq{sequence:02d}/radar{radar_index:05d}".encode("ascii")
    keyed: list[tuple[bytes, int]] = []
    for atom_value in np.unique(slot_atom_id).tolist():
        canonical_atom_id = int(atom_value)
        row = int(np.searchsorted(atom_id, canonical_atom_id))
        if row >= atom_id.size or int(atom_id[row]) != canonical_atom_id:
            raise ValueError("STDA-F0 greedy slot refers to an unknown atom")
        atom_byte_key = (
            np.asarray(atom_xyz[row], dtype="<f4").tobytes(order="C")
            + np.asarray([atom_weight[row]], dtype="<f8").tobytes(order="C")
        )
        order_digest = hashlib.sha256(
            b"stda_f0_greedy_atom_order_v1\0"
            + frame_key
            + b"\0"
            + atom_byte_key
        ).digest()
        keyed.append((order_digest, canonical_atom_id))
    keyed.sort(key=lambda item: (item[0], item[1]))
    ordered_atom_id = np.asarray([item[1] for item in keyed], dtype="<i8")
    ordered_digest_bytes = np.frombuffer(
        b"".join(item[0] for item in keyed), dtype="<u1"
    ).reshape(len(keyed), 32)
    digest_sha256 = _round_hash_arrays(
        b"stda_f0_greedy_atom_order_commitment_v1",
        np.frombuffer(frame_key, dtype="<u1"),
        ordered_atom_id,
        ordered_digest_bytes,
    )
    return frame_key, ordered_atom_id, ordered_digest_bytes, digest_sha256


def _simulate_round_robin_greedy(
    stable_id: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    graph_slot_id: np.ndarray,
    slot_atom_id: np.ndarray,
    ordered_atom_id: np.ndarray,
) -> dict[str, np.ndarray]:
    rows_by_atom: dict[int, np.ndarray] = {}
    cursors: dict[int, int] = {}
    for atom_value in ordered_atom_id.tolist():
        atom = int(atom_value)
        rows = np.flatnonzero(slot_atom_id == atom)
        rows = rows[np.argsort(graph_slot_id[rows], kind="stable")]
        rows_by_atom[atom] = rows
        cursors[atom] = 0
    unused = np.ones(stable_id.size, dtype=bool)
    trace_round: list[int] = []
    trace_atom: list[int] = []
    trace_slot_row: list[int] = []
    trace_slot_id: list[int] = []
    trace_support_rank: list[int] = []
    trace_support_id: list[int] = []
    trace_edge_cost: list[int] = []
    consumed = 0
    round_index = 0
    atom_count = int(ordered_atom_id.size)
    if graph_slot_id.size and atom_count == 0:
        raise ValueError("STDA-F0 greedy replay has slots but no active atoms")
    while consumed < graph_slot_id.size:
        consumed_before = consumed
        for offset in range(atom_count):
            atom = int(ordered_atom_id[(round_index + offset) % atom_count])
            atom_rows = rows_by_atom[atom]
            cursor = cursors[atom]
            if cursor >= atom_rows.size:
                continue
            slot_row = int(atom_rows[cursor])
            cursors[atom] = cursor + 1
            consumed += 1
            start = int(indptr[slot_row])
            stop = int(indptr[slot_row + 1])
            best_position: int | None = None
            best_key: tuple[int, int] | None = None
            for position in range(start, stop):
                support_rank = int(indices[position])
                if not unused[support_rank]:
                    continue
                key = (int(data[position]), support_rank)
                if best_key is None or key < best_key:
                    best_key = key
                    best_position = position
            if best_position is None:
                support_rank = -1
                support_identifier = -1
                edge_cost = -1
            else:
                support_rank = int(indices[best_position])
                support_identifier = int(stable_id[support_rank])
                edge_cost = int(data[best_position])
                unused[support_rank] = False
            trace_round.append(round_index)
            trace_atom.append(atom)
            trace_slot_row.append(slot_row)
            trace_slot_id.append(int(graph_slot_id[slot_row]))
            trace_support_rank.append(support_rank)
            trace_support_id.append(support_identifier)
            trace_edge_cost.append(edge_cost)
        if consumed == consumed_before:
            raise AssertionError("STDA-F0 greedy replay stopped before all slots")
        round_index += 1
    return {
        "round_index": np.asarray(trace_round, dtype="<i8"),
        "atom_id": np.asarray(trace_atom, dtype="<i8"),
        "slot_row": np.asarray(trace_slot_row, dtype="<i8"),
        "slot_id": np.asarray(trace_slot_id, dtype="<i8"),
        "support_rank": np.asarray(trace_support_rank, dtype="<i8"),
        "support_id": np.asarray(trace_support_id, dtype="<i8"),
        "edge_cost": np.asarray(trace_edge_cost, dtype="<i8"),
    }


def _round_robin_greedy_replay(
    solver: Mapping[str, np.ndarray],
    controls: Mapping[str, np.ndarray],
    results: Mapping[str, np.ndarray],
    expected_report: Mapping[str, Any],
    *,
    sequence: int,
    radar_index: int,
) -> tuple[dict[str, Any], np.ndarray]:
    stable_id = solver["support_stable_candidate_id.npy"]
    support_xyz = solver["support_xyz.npy"]
    indptr = solver["graph_indptr.npy"]
    indices = solver["graph_indices.npy"]
    data = solver["graph_data.npy"]
    graph_slot_id = solver["demand_slot_id.npy"]
    slot_atom_id = controls["demand_slot_atom_id.npy"]
    atom_id = controls["demand_atom_id.npy"]
    atom_xyz = controls["demand_atom_xyz.npy"]
    atom_weight = controls["demand_atom_weight.npy"]
    _, ordered_atom_id, ordered_digest_bytes, atom_order_sha256 = (
        _independent_greedy_atom_order(
            slot_atom_id,
            atom_id,
            atom_xyz,
            atom_weight,
            sequence=sequence,
            radar_index=radar_index,
        )
    )
    expected_trace = _simulate_round_robin_greedy(
        stable_id,
        indptr,
        indices,
        data,
        graph_slot_id,
        slot_atom_id,
        ordered_atom_id,
    )
    observed_trace = {
        "round_index": results["round_robin_greedy_round_index.npy"],
        "atom_id": results["round_robin_greedy_atom_id.npy"],
        "slot_row": results["round_robin_greedy_slot_row.npy"],
        "slot_id": results["round_robin_greedy_slot_id.npy"],
        "support_rank": results["round_robin_greedy_support_rank.npy"],
        "support_id": results["round_robin_greedy_support_id.npy"],
        "edge_cost": results["round_robin_greedy_edge_cost.npy"],
    }
    trace_shape = all(
        values.shape == (graph_slot_id.size,) for values in observed_trace.values()
    )
    all_slots_once = trace_shape and (
        np.unique(observed_trace["slot_row"]).size == graph_slot_id.size
        and set(int(value) for value in observed_trace["slot_row"].tolist())
        == set(range(graph_slot_id.size))
    )
    order_binding = (
        np.array_equal(
            results["round_robin_greedy_ordered_atom_id.npy"], ordered_atom_id
        )
        and np.array_equal(
            results["round_robin_greedy_order_digest_bytes.npy"],
            ordered_digest_bytes,
        )
    )
    trace_checks = {
        key: np.array_equal(observed_trace[key], expected_trace[key])
        for key in expected_trace
    }
    selected_success = observed_trace["support_rank"][
        observed_trace["support_rank"] >= 0
    ]
    support_capacity = np.unique(selected_success).size == selected_success.size
    reported_edges_valid = trace_shape
    if reported_edges_valid:
        for slot_row, support_rank, support_identifier, edge_cost in zip(
            observed_trace["slot_row"].tolist(),
            observed_trace["support_rank"].tolist(),
            observed_trace["support_id"].tolist(),
            observed_trace["edge_cost"].tolist(),
            strict=True,
        ):
            if support_rank == -1:
                if support_identifier != -1 or edge_cost != -1:
                    reported_edges_valid = False
                    break
                continue
            if support_rank < 0 or support_rank >= stable_id.size:
                reported_edges_valid = False
                break
            expected_cost = _edge_lookup(
                int(slot_row), int(support_rank), indptr, indices, data
            )
            if (
                expected_cost is None
                or int(stable_id[support_rank]) != support_identifier
                or expected_cost != edge_cost
            ):
                reported_edges_valid = False
                break
    expected_selected_rank = np.sort(selected_success).astype("<i8", copy=False)
    selected_rank = results["round_robin_greedy_selected_support_rank.npy"]
    selected_id = results["round_robin_greedy_selected_support_id.npy"]
    canonical_selection = (
        np.array_equal(selected_rank, expected_selected_rank)
        and bool(np.all(selected_rank >= 0))
        and bool(np.all(selected_rank < stable_id.size))
        and np.array_equal(selected_id, stable_id[selected_rank])
    )
    export_xyz = (
        np.ascontiguousarray(support_xyz[selected_rank], dtype="<f4")
        if canonical_selection
        else np.empty((0, 3), dtype="<f4")
    )
    failed_slot_count = int((observed_trace["support_rank"] < 0).sum())
    full_capacity = failed_slot_count == 0 and bool(
        np.all(observed_trace["support_rank"] >= 0)
    )
    assignment_sha256 = _round_hash_arrays(
        b"stda_f0_greedy_assignment_v1",
        observed_trace["round_index"],
        observed_trace["atom_id"],
        observed_trace["slot_row"],
        observed_trace["slot_id"],
        observed_trace["support_rank"],
        observed_trace["support_id"],
        observed_trace["edge_cost"],
    )
    selected_id_sha256 = _round_hash_arrays(
        b"stda_f0_greedy_selected_ids_v1", selected_id
    )
    export_sha256 = _round_hash_arrays(
        b"stda_f0_greedy_export_v1", export_xyz
    )
    digest_sha256 = _round_hash_hex_fields(
        b"stda_f0_greedy_result_v1",
        atom_order_sha256,
        assignment_sha256,
        selected_id_sha256,
        export_sha256,
    )
    hashes = {
        "atom_order_sha256": atom_order_sha256,
        "assignment_sha256": assignment_sha256,
        "selected_id_sha256": selected_id_sha256,
        "export_sha256": export_sha256,
        "digest_sha256": digest_sha256,
    }
    hash_checks = {
        key: expected_report.get(key) == value for key, value in hashes.items()
    }
    checks = {
        "trace_shape": trace_shape,
        "all_slots_consumed_once": all_slots_once,
        "atom_order_and_digest_bytes": order_binding,
        "forward_rotation_and_slot_order": all(
            trace_checks[key]
            for key in ("round_index", "atom_id", "slot_row", "slot_id")
        ),
        "minimum_unused_edge_choice": all(
            trace_checks[key]
            for key in ("support_rank", "support_id", "edge_cost")
        ),
        "reported_edge_membership_and_cost": reported_edges_valid,
        "support_capacity_one": support_capacity,
        "canonical_selection": canonical_selection,
        "failed_slots_preserved": (
            expected_report.get("failed_slot_count") == failed_slot_count
        ),
        "full_capacity_reported_exactly": (
            expected_report.get("full_capacity") is full_capacity
        ),
        "hashes_match": all(hash_checks.values()),
    }
    return (
        {
            "checks": checks,
            "trace_checks": trace_checks,
            "hash_checks": hash_checks,
            "computed_hashes": hashes,
            "failed_slot_count": failed_slot_count,
            "full_capacity": full_capacity,
            "selected_count": int(selected_rank.size),
            "passed": all(checks.values()),
        },
        export_xyz,
    )


def verify_export_pair(
    binary_payload: bytes,
    npy_payload: bytes,
    *,
    arm: str,
    expected_xyz: np.ndarray,
    expected_report: Mapping[str, Any],
    required_count: int = DEMAND_COUNT,
) -> dict[str, Any]:
    """Verify one immutable XYZ NPY/BIN pair and its oracle file records."""

    if arm not in _EXPORT_ARMS:
        raise ValueError(f"STDA-F0 export arm is not frozen: {arm}")
    if not isinstance(binary_payload, bytes) or not isinstance(npy_payload, bytes):
        raise TypeError("STDA-F0 export artifacts must be immutable bytes")
    expected = _as_le_array(expected_xyz, "<f4")
    if expected.ndim != 2 or expected.shape[1:] != (3,):
        raise ValueError("STDA-F0 expected export must have shape (N,3)")
    binary_xyz = _float32_rows_from_bytes(binary_payload, None)
    npy_xyz = _load_npy_bytes(
        npy_payload,
        filename=f"{arm}.npy",
        dtype="<f4",
        ndim=2,
    )
    npy_shape_valid = npy_xyz.shape[1:] == (3,)
    npy_raw = npy_xyz.tobytes(order="C") if npy_shape_valid else b""
    binary_sha256 = sha256_bytes(binary_payload)
    npy_sha256 = sha256_bytes(npy_payload)
    spacing = verify_exact_spacing_bytes(binary_payload)
    bin_record = expected_report.get("bin")
    npy_record = expected_report.get("npy")
    bin_record_valid = isinstance(bin_record, Mapping) and all(
        (
            bin_record.get("path") == f"exports/{arm}.bin",
            bin_record.get("bytes") == len(binary_payload),
            bin_record.get("sha256") == binary_sha256,
        )
    )
    npy_record_valid = isinstance(npy_record, Mapping) and all(
        (
            npy_record.get("path") == f"exports/{arm}.npy",
            npy_record.get("bytes") == len(npy_payload),
            npy_record.get("sha256") == npy_sha256,
        )
    )
    checks = {
        "arm_name": expected_report.get("arm") == arm,
        "dtype": expected_report.get("dtype") == "<f4",
        "binary_xyz_shape": binary_xyz.ndim == 2 and binary_xyz.shape[1:] == (3,),
        "npy_xyz_shape": npy_shape_valid,
        "required_count": int(binary_xyz.shape[0]) == required_count,
        "report_count": expected_report.get("count") == int(binary_xyz.shape[0]),
        "finite_xyz": bool(
            np.isfinite(binary_xyz).all() and np.isfinite(npy_xyz).all()
        ),
        "unique_xyz_bytes": bool(spacing["unique_xyz_bytes"]),
        "strict_spacing_5cm": bool(spacing["strict_spacing_5cm"]),
        "npy_bin_raw_bytes_identical": npy_shape_valid and npy_raw == binary_payload,
        "bound_to_replayed_selection": binary_payload
        == expected.tobytes(order="C"),
        "binary_file_record": bin_record_valid,
        "npy_file_record": npy_record_valid,
        "raw_xyz_sha256": expected_report.get("raw_xyz_sha256")
        == binary_sha256,
    }
    return {
        "arm": arm,
        "point_count": int(binary_xyz.shape[0]),
        "expected_replay_count": required_count,
        "binary_sha256": binary_sha256,
        "npy_sha256": npy_sha256,
        "spacing": spacing,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_control_export_replay(
    *,
    sequence: int,
    radar_index: int,
    solver_inputs: Mapping[str, bytes],
    control_inputs: Mapping[str, bytes],
    round_results: Mapping[str, bytes],
    exports: Mapping[str, bytes],
    oracle_report: Mapping[str, Any],
    required_export_count: int = DEMAND_COUNT,
    expected_graph_k: int | None = GRAPH_K,
) -> dict[str, Any]:
    """Independently replay the two controls and bind all three exports.

    Inputs are immutable file payloads keyed by the filenames serialized by the
    oracle. The formal call uses the default 10,000 slots and K=256; smaller
    explicit values exist only for deterministic synthetic protocol tests.
    """

    if not all(
        isinstance(value, Mapping)
        for value in (
            solver_inputs,
            control_inputs,
            round_results,
            exports,
            oracle_report,
        )
    ):
        raise TypeError("STDA-F0 replay inputs and report must be mappings")
    if sequence < 0 or radar_index < 0:
        raise ValueError("STDA-F0 frame indices must be nonnegative")
    if type(required_export_count) is not int or required_export_count <= 0:
        raise ValueError("STDA-F0 required export count must be positive")
    if expected_graph_k is not None and (
        type(expected_graph_k) is not int or expected_graph_k <= 0
    ):
        raise ValueError("STDA-F0 expected graph degree must be positive or None")

    round_binding = _expected_report_mapping(
        oracle_report.get("round_input_binding"), label="round input binding"
    )
    solver_file_binding = _file_set_binding(
        solver_inputs,
        round_binding.get("solver_input_files_sha256"),
        label="solver_inputs",
    )
    control_file_binding = _file_set_binding(
        control_inputs,
        round_binding.get("control_input_files_sha256"),
        label="control_inputs",
    )
    result_file_binding = _file_set_binding(
        round_results,
        oracle_report.get("result_array_files_sha256"),
        label="round_results",
    )
    solver = _load_required_arrays(
        solver_inputs, _SOLVER_INPUT_SCHEMA, label="solver_inputs"
    )
    controls = _load_required_arrays(
        control_inputs, _CONTROL_INPUT_SCHEMA, label="control_inputs"
    )
    results = _load_required_arrays(
        round_results, _CONTROL_RESULT_SCHEMA, label="round_results"
    )
    input_replay = _validate_replay_inputs(
        solver,
        controls,
        required_export_count=required_export_count,
        expected_graph_k=expected_graph_k,
    )
    input_digest_checks = {
        name: round_binding.get(name) == value
        for name, value in input_replay["digests"].items()
    }
    decision_report = _expected_report_mapping(
        oracle_report.get("decision"), label="decision"
    )
    pointwise_report = _expected_report_mapping(
        oracle_report.get("packed_pointwise"), label="packed pointwise"
    )
    greedy_report = _expected_report_mapping(
        oracle_report.get("round_robin_greedy"), label="round-robin greedy"
    )
    decision_replay, decision_xyz = _decision_replay(
        solver,
        results,
        decision_report,
        required_count=required_export_count,
    )
    pointwise_replay, pointwise_xyz = _packed_pointwise_replay(
        solver,
        controls,
        results,
        pointwise_report,
        required_count=required_export_count,
    )
    greedy_replay, greedy_xyz = _round_robin_greedy_replay(
        solver,
        controls,
        results,
        greedy_report,
        sequence=sequence,
        radar_index=radar_index,
    )

    expected_export_names = {
        f"{arm}.{suffix}" for arm in _EXPORT_ARMS for suffix in ("bin", "npy")
    }
    missing_exports = sorted(expected_export_names - set(exports))
    if missing_exports:
        raise ValueError(f"STDA-F0 exports are missing files: {missing_exports}")
    export_reports = _expected_report_mapping(
        oracle_report.get("exports"), label="exports"
    )
    expected_xyz = {
        "decision": decision_xyz,
        "packed_pointwise": pointwise_xyz,
        "round_robin_greedy": greedy_xyz,
    }
    export_replays: dict[str, dict[str, Any]] = {}
    for arm in _EXPORT_ARMS:
        arm_report = _expected_report_mapping(
            export_reports.get(arm), label=f"{arm} export"
        )
        replay_count = (
            greedy_replay["selected_count"]
            if arm == "round_robin_greedy"
            else required_export_count
        )
        export_replays[arm] = verify_export_pair(
            exports[f"{arm}.bin"],
            exports[f"{arm}.npy"],
            arm=arm,
            expected_xyz=expected_xyz[arm],
            expected_report=arm_report,
            required_count=replay_count,
        )
        export_replays[arm]["scientific_exact_required_count"] = (
            export_replays[arm]["point_count"] == required_export_count
        )

    frame_report = oracle_report.get("frame")
    frame_checks = {
        "mapping": isinstance(frame_report, Mapping),
        "sequence": isinstance(frame_report, Mapping)
        and frame_report.get("sequence") == sequence,
        "radar_index": isinstance(frame_report, Mapping)
        and frame_report.get("radar_index") == radar_index,
        "frame_key": isinstance(frame_report, Mapping)
        and frame_report.get("frame_key")
        == f"seq{sequence:02d}/radar{radar_index:05d}",
    }
    checks = {
        "protocol_sha256": oracle_report.get("protocol_sha256")
        == PROTOCOL_SHA256,
        "protocol_freeze_commit": oracle_report.get("protocol_freeze_commit")
        == PROTOCOL_FREEZE_COMMIT,
        "frame": all(frame_checks.values()),
        "exact_export_file_names": set(exports) == expected_export_names,
        "solver_file_binding": solver_file_binding["passed"],
        "control_file_binding": control_file_binding["passed"],
        "result_file_binding": result_file_binding["passed"],
        "input_semantics": input_replay["passed"],
        "input_digests": all(input_digest_checks.values()),
        "decision_replay": decision_replay["passed"],
        "packed_pointwise_replay": pointwise_replay["passed"],
        "round_robin_greedy_replay": greedy_replay["passed"],
        "all_export_pairs": all(
            report["passed"] for report in export_replays.values()
        ),
    }
    return {
        "schema": CONTROL_EXPORT_REPLAY_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "frame": {
            "sequence": sequence,
            "radar_index": radar_index,
            "frame_key": f"seq{sequence:02d}/radar{radar_index:05d}",
        },
        "required_export_count": required_export_count,
        "expected_graph_k": expected_graph_k,
        "file_bindings": {
            "solver_inputs": solver_file_binding,
            "control_inputs": control_file_binding,
            "round_results": result_file_binding,
        },
        "input_replay": input_replay,
        "input_digest_checks": input_digest_checks,
        "decision": decision_replay,
        "packed_pointwise": pointwise_replay,
        "round_robin_greedy": greedy_replay,
        "exports": export_replays,
        "frame_checks": frame_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def stable_hash_rows(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        if not array.flags.c_contiguous:
            raise ValueError("STDA-F0 hash input must be C-contiguous")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _structural_hash_parts(kind: str, parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(_STRUCTURAL_HASH_NAMESPACE)
    digest.update(PROTOCOL_FREEZE_COMMIT.encode("ascii"))
    digest.update(b"\0")
    digest.update(PROTOCOL_SHA256.encode("ascii"))
    digest.update(b"\0")
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    for part in parts:
        value = bytes(part)
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)
    return digest.hexdigest()


def _structural_hash_records(
    kind: str,
    records: Iterable[bytes],
    *,
    metadata: Iterable[bytes] = (),
) -> str:
    material = list(metadata)
    material.extend(records)
    return _structural_hash_parts(kind, material)


def _pack_nonnegative_integer(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ValueError("STDA structural integer encoding requires nonnegative int")
    width = max(1, (value.bit_length() + 7) // 8)
    return struct.pack("<Q", width) + value.to_bytes(
        width,
        byteorder="little",
        signed=False,
    )


def _pack_u64_tuple(values: Sequence[int]) -> bytes:
    material = tuple(values)
    if any(type(value) is not int or not 0 <= value < 2**64 for value in material):
        raise ValueError("STDA structural tuple contains a non-u64 value")
    return struct.pack("<Q", len(material)) + b"".join(
        struct.pack("<Q", value) for value in material
    )


def _require_raw_bytes(payload: bytes, columns: int, label: str) -> np.ndarray:
    if not isinstance(payload, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    row_bytes = columns * 4
    if len(payload) % row_bytes:
        raise ValueError(f"{label} is not a whole number of float32 rows")
    array = np.frombuffer(payload, dtype="<f4").reshape(-1, columns)
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} contains a non-finite value")
    return array


def _polar_from_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    radius = math.sqrt((x * x + y * y) + z * z)
    azimuth = math.atan2(y, x)
    if azimuth == math.pi:
        azimuth = -math.pi
    ratio = 0.0 if radius == 0.0 else max(-1.0, min(1.0, z / radius))
    return radius, azimuth, math.asin(ratio)


def _axis_index(edges: tuple[float, ...], value: float) -> int:
    if value < edges[0] or value > edges[-1]:
        return -1
    if value == edges[-1]:
        return len(edges) - 2
    index = bisect_right(edges, value) - 1
    return index if 0 <= index < len(edges) - 1 else -1


@dataclass(frozen=True)
class _IndependentDomain:
    azimuth_edges_rad: tuple[float, ...]
    elevation_edges_rad: tuple[float, ...]

    @classmethod
    def from_edges(
        cls,
        azimuth_edges_rad: Sequence[float],
        elevation_edges_rad: Sequence[float],
    ) -> _IndependentDomain:
        azimuth = tuple(float(value) for value in azimuth_edges_rad)
        elevation = tuple(float(value) for value in elevation_edges_rad)
        for label, edges in (("azimuth", azimuth), ("elevation", elevation)):
            if len(edges) < 2 or not all(math.isfinite(value) for value in edges):
                raise ValueError(f"STDA {label} edges must be a finite vector")
            if not all(left < right for left, right in zip(edges, edges[1:])):
                raise ValueError(f"STDA {label} edges must be strictly increasing")
        if azimuth[0] < -math.pi or azimuth[-1] > math.pi:
            raise ValueError("STDA azimuth edges must lie inside [-pi,pi]")
        if azimuth[-1] - azimuth[0] >= 2.0 * math.pi:
            raise ValueError("STDA azimuth support must be non-wrapping")
        if elevation[0] < -0.5 * math.pi or elevation[-1] > 0.5 * math.pi:
            raise ValueError("STDA elevation edges must lie inside [-pi/2,pi/2]")
        return cls(azimuth, elevation)

    @property
    def elevation_count(self) -> int:
        return len(self.elevation_edges_rad) - 1

    @property
    def digest_sha256(self) -> str:
        return _structural_hash_parts(
            "domain",
            (
                struct.pack("<Q", len(self.azimuth_edges_rad)),
                b"".join(struct.pack("<d", value) for value in self.azimuth_edges_rad),
                struct.pack("<Q", len(self.elevation_edges_rad)),
                b"".join(
                    struct.pack("<d", value) for value in self.elevation_edges_rad
                ),
                struct.pack(
                    "<dddd",
                    OUTPUT_RANGE_UPPER_M,
                    TARGET_RANGE_UPPER_M,
                    UNMATCHED_DISTANCE_M,
                    GROUP_SPAN_M,
                ),
            ),
        )

    def ray_id(self, azimuth: float, elevation: float) -> int:
        azimuth_index = _axis_index(self.azimuth_edges_rad, azimuth)
        elevation_index = _axis_index(self.elevation_edges_rad, elevation)
        if azimuth_index < 0 or elevation_index < 0:
            return -1
        return azimuth_index * self.elevation_count + elevation_index


@dataclass(frozen=True)
class _OutputEvent:
    event_id: int
    ray_id: int
    inferred_depth: int
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    x: float
    y: float
    z: float
    xyz_bytes: bytes


@dataclass(frozen=True)
class _TargetAtom:
    target_id: int
    ray_id: int
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    x: float
    y: float
    z: float
    xyz_bytes: bytes
    weight: float


@dataclass(frozen=True)
class _TargetGroup:
    ray_id: int
    group_index: int
    return_class: int
    range_stratum: int
    representative_range_m: float
    group_weight: float
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class _Assignment:
    ray_id: int
    group_index: int
    return_class: int
    range_stratum: int
    representative_range_m: float
    group_weight: float
    assigned_distance_m: float
    matched_event_id: int
    matched_depth: int
    matched_event_range_m: float | None
    integer_cost: int


@dataclass(frozen=True)
class _DPSolution:
    integer_cost: int
    matched_count: int
    assignment_tuple: tuple[int, ...]

    @property
    def key(self) -> tuple[int, int, tuple[int, ...]]:
        return self.integer_cost, -self.matched_count, self.assignment_tuple


def _event_record(event: _OutputEvent) -> bytes:
    return b"".join(
        (
            struct.pack(
                "<QqQdddddd",
                event.event_id,
                event.ray_id,
                event.inferred_depth,
                event.range_m,
                event.azimuth_rad,
                event.elevation_rad,
                event.x,
                event.y,
                event.z,
            ),
            event.xyz_bytes,
        )
    )


def _canonical_output_events(
    output_bytes: bytes,
    domain: _IndependentDomain,
) -> tuple[tuple[_OutputEvent, ...], int, tuple[int, ...], str]:
    xyz = _require_raw_bytes(output_bytes, 3, "STDA output XYZ bytes")
    unique_rows = {output_bytes[offset : offset + 12] for offset in range(0, len(output_bytes), 12)}
    provisional: list[tuple[float, float, float, float, float, float, bytes, int]] = []
    for xyz_bytes in unique_rows:
        x32, y32, z32 = struct.unpack("<fff", xyz_bytes)
        x, y, z = float(x32), float(y32), float(z32)
        radius, azimuth, elevation = _polar_from_xyz(x, y, z)
        ray_id = domain.ray_id(azimuth, elevation)
        if not 0.0 <= radius <= OUTPUT_RANGE_UPPER_M:
            ray_id = -1
        provisional.append((radius, azimuth, elevation, x, y, z, xyz_bytes, ray_id))
    provisional.sort(key=lambda row: row[:7])

    depth_by_id: dict[int, int] = {}
    rows_by_ray: dict[int, list[tuple[float, int]]] = {}
    for event_id, row in enumerate(provisional):
        if row[7] >= 0:
            rows_by_ray.setdefault(row[7], []).append((row[0], event_id))
    for ray_rows in rows_by_ray.values():
        ray_rows.sort(key=lambda row: (row[0], row[1]))
        for depth, (_, event_id) in enumerate(ray_rows, start=1):
            depth_by_id[event_id] = depth

    events = tuple(
        _OutputEvent(
            event_id=event_id,
            ray_id=row[7],
            inferred_depth=depth_by_id.get(event_id, 0),
            range_m=row[0],
            azimuth_rad=row[1],
            elevation_rad=row[2],
            x=row[3],
            y=row[4],
            z=row[5],
            xyz_bytes=row[6],
        )
        for event_id, row in enumerate(provisional)
    )
    invalid_ids = tuple(event.event_id for event in events if event.ray_id < 0)
    digest = _structural_hash_records(
        "canonical_output_events",
        map(_event_record, events),
    )
    return events, int(xyz.shape[0]) - len(events), invalid_ids, digest


def _target_atom_record(atom: _TargetAtom) -> bytes:
    return b"".join(
        (
            struct.pack(
                "<Qqddddddd",
                atom.target_id,
                atom.ray_id,
                atom.range_m,
                atom.azimuth_rad,
                atom.elevation_rad,
                atom.x,
                atom.y,
                atom.z,
                atom.weight,
            ),
            atom.xyz_bytes,
        )
    )


def _canonical_positive_target_atoms(
    target_bytes: bytes,
    domain: _IndependentDomain,
) -> tuple[tuple[_TargetAtom, ...], int, tuple[int, ...], str]:
    target = _require_raw_bytes(
        target_bytes,
        4,
        "STDA target XYZ-confidence bytes",
    )
    if bool(np.any(target[:, 3] < 0.0)):
        raise ValueError("STDA target confidence must be nonnegative")

    confidence_by_xyz: dict[bytes, list[tuple[int, float]]] = {}
    zero_count = 0
    for row in target:
        canonical = tuple(np.float32(0.0) if value == 0.0 else value for value in row)
        confidence = canonical[3]
        if confidence == 0.0:
            zero_count += 1
            continue
        xyz_bytes = np.asarray(canonical[:3], dtype="<f4").tobytes(order="C")
        confidence_bytes = np.asarray([confidence], dtype="<f4").tobytes(order="C")
        confidence_bits = struct.unpack("<I", confidence_bytes)[0]
        confidence_by_xyz.setdefault(xyz_bytes, []).append(
            (confidence_bits, float(confidence))
        )

    provisional: list[
        tuple[float, float, float, float, float, float, bytes, float, int]
    ] = []
    for xyz_bytes, confidence_values in confidence_by_xyz.items():
        confidence_values.sort(key=lambda item: item[0])
        weight = math.fsum(value for _, value in confidence_values)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("STDA canonical target atom has invalid positive weight")
        x32, y32, z32 = struct.unpack("<fff", xyz_bytes)
        x, y, z = float(x32), float(y32), float(z32)
        radius, azimuth, elevation = _polar_from_xyz(x, y, z)
        ray_id = domain.ray_id(azimuth, elevation)
        if not 0.0 <= radius < TARGET_RANGE_UPPER_M:
            ray_id = -1
        provisional.append(
            (radius, azimuth, elevation, x, y, z, xyz_bytes, weight, ray_id)
        )
    provisional.sort(key=lambda row: row[:7])

    atoms = tuple(
        _TargetAtom(
            target_id=target_id,
            ray_id=row[8],
            range_m=row[0],
            azimuth_rad=row[1],
            elevation_rad=row[2],
            x=row[3],
            y=row[4],
            z=row[5],
            xyz_bytes=row[6],
            weight=row[7],
        )
        for target_id, row in enumerate(provisional)
    )
    invalid_ids = tuple(atom.target_id for atom in atoms if atom.ray_id < 0)
    digest = _structural_hash_records(
        "canonical_target_atoms",
        map(_target_atom_record, atoms),
    )
    return atoms, zero_count, invalid_ids, digest


def _range_stratum(radius: float) -> int:
    for index, (lower, upper) in enumerate(RANGE_BOUNDS_M):
        if lower <= radius < upper:
            return index
    raise ValueError(f"STDA target group range is outside [0,120): {radius!r}")


def _weighted_integer_cost(weight: float, distance_m: float) -> int:
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("STDA structural group weight must be finite and positive")
    if not math.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("STDA structural distance must be finite and nonnegative")
    weight_key = int(np.rint(np.float64(1.0e9 * weight)))
    distance_key = int(np.rint(np.float64(1.0e6 * distance_m)))
    if weight_key < 0 or distance_key < 0:
        raise ValueError("STDA structural integer cost quantization is invalid")
    return weight_key * distance_key


def _group_record(group: _TargetGroup) -> bytes:
    return b"".join(
        (
            struct.pack(
                "<qQBBdd",
                group.ray_id,
                group.group_index,
                group.return_class,
                group.range_stratum,
                group.representative_range_m,
                group.group_weight,
            ),
            _pack_u64_tuple(group.target_ids),
        )
    )


def _build_target_groups(atoms: tuple[_TargetAtom, ...]) -> tuple[_TargetGroup, ...]:
    atoms_by_ray: dict[int, list[_TargetAtom]] = {}
    for atom in atoms:
        atoms_by_ray.setdefault(atom.ray_id, []).append(atom)

    groups: list[_TargetGroup] = []
    for ray_id in sorted(atoms_by_ray):
        ordered = sorted(
            atoms_by_ray[ray_id],
            key=lambda atom: (atom.range_m, atom.target_id),
        )
        start = 0
        group_index = 0
        while start < len(ordered):
            stop = start + 1
            first_range = ordered[start].range_m
            while (
                stop < len(ordered)
                and ordered[stop].range_m - first_range < GROUP_SPAN_M
            ):
                stop += 1
            members = ordered[start:stop]
            weight = math.fsum(atom.weight for atom in members)
            representative = math.fsum(
                atom.range_m * atom.weight for atom in members
            ) / weight
            groups.append(
                _TargetGroup(
                    ray_id=ray_id,
                    group_index=group_index,
                    return_class=RETURN_FIRST if group_index == 0 else RETURN_LATER,
                    range_stratum=_range_stratum(representative),
                    representative_range_m=representative,
                    group_weight=weight,
                    target_ids=tuple(atom.target_id for atom in members),
                )
            )
            start = stop
            group_index += 1
    return tuple(groups)


def _solve_one_ray_independently(
    groups: Sequence[_TargetGroup],
    events: Sequence[_OutputEvent],
) -> _DPSolution:
    ordered_events = tuple(
        sorted(events, key=lambda event: (event.range_m, event.event_id))
    )
    group_count = len(groups)
    event_count = len(ordered_events)
    table: dict[tuple[int, int, bool], _DPSolution] = {}
    terminal = _DPSolution(0, 0, ())
    for event_index in range(event_count + 1):
        table[(group_count, event_index, False)] = terminal
        table[(group_count, event_index, True)] = terminal

    for group_index in range(group_count - 1, -1, -1):
        group = groups[group_index]
        for event_index in range(event_count, -1, -1):
            for previously_matched in (False, True):
                unmatched_suffix = table[
                    (group_index + 1, event_index, previously_matched)
                ]
                candidates = [
                    _DPSolution(
                        integer_cost=(
                            _weighted_integer_cost(
                                group.group_weight,
                                UNMATCHED_DISTANCE_M,
                            )
                            + unmatched_suffix.integer_cost
                        ),
                        matched_count=unmatched_suffix.matched_count,
                        assignment_tuple=(UNMATCHED_SENTINEL,)
                        + unmatched_suffix.assignment_tuple,
                    )
                ]
                if event_index < event_count:
                    candidates.append(
                        table[(group_index, event_index + 1, previously_matched)]
                    )
                    event = ordered_events[event_index]
                    eligible = (
                        group.return_class == RETURN_FIRST
                        and event.inferred_depth == 1
                    ) or (
                        group.return_class == RETURN_LATER
                        and event.inferred_depth >= 2
                        and previously_matched
                    )
                    if eligible:
                        distance = abs(group.representative_range_m - event.range_m)
                        matched_suffix = table[
                            (group_index + 1, event_index + 1, True)
                        ]
                        candidates.append(
                            _DPSolution(
                                integer_cost=(
                                    _weighted_integer_cost(
                                        group.group_weight,
                                        distance,
                                    )
                                    + matched_suffix.integer_cost
                                ),
                                matched_count=matched_suffix.matched_count + 1,
                                assignment_tuple=(event.event_id,)
                                + matched_suffix.assignment_tuple,
                            )
                        )
                table[(group_index, event_index, previously_matched)] = min(
                    candidates,
                    key=lambda solution: solution.key,
                )
    return table[(0, 0, False)]


def _assignment_record(assignment: _Assignment) -> bytes:
    matched_range = (
        b"\0"
        if assignment.matched_event_range_m is None
        else b"\1" + struct.pack("<d", assignment.matched_event_range_m)
    )
    return b"".join(
        (
            struct.pack(
                "<qQBBdddQq",
                assignment.ray_id,
                assignment.group_index,
                assignment.return_class,
                assignment.range_stratum,
                assignment.representative_range_m,
                assignment.group_weight,
                assignment.assigned_distance_m,
                assignment.matched_event_id,
                assignment.matched_depth,
            ),
            matched_range,
            _pack_nonnegative_integer(assignment.integer_cost),
        )
    )


def _build_assignments_independently(
    groups: tuple[_TargetGroup, ...],
    events: tuple[_OutputEvent, ...],
) -> tuple[tuple[_Assignment, ...], int, int, tuple[int, ...]]:
    event_lookup = {event.event_id: event for event in events}
    groups_by_ray: dict[int, list[_TargetGroup]] = {}
    events_by_ray: dict[int, list[_OutputEvent]] = {}
    for group in groups:
        groups_by_ray.setdefault(group.ray_id, []).append(group)
    for event in events:
        events_by_ray.setdefault(event.ray_id, []).append(event)

    assignments: list[_Assignment] = []
    total_cost = 0
    total_matched = 0
    complete_mapping: list[int] = []
    for ray_id in sorted(groups_by_ray):
        ray_groups = groups_by_ray[ray_id]
        solution = _solve_one_ray_independently(
            ray_groups,
            events_by_ray.get(ray_id, ()),
        )
        if len(solution.assignment_tuple) != len(ray_groups):
            raise AssertionError("independent STDA DP did not assign every group")
        total_cost += solution.integer_cost
        total_matched += solution.matched_count
        complete_mapping.extend(solution.assignment_tuple)
        for group, event_id in zip(
            ray_groups,
            solution.assignment_tuple,
            strict=True,
        ):
            if event_id == UNMATCHED_SENTINEL:
                distance = UNMATCHED_DISTANCE_M
                depth = -1
                event_range = None
            else:
                event = event_lookup[event_id]
                distance = abs(group.representative_range_m - event.range_m)
                depth = event.inferred_depth
                event_range = event.range_m
            assignments.append(
                _Assignment(
                    ray_id=group.ray_id,
                    group_index=group.group_index,
                    return_class=group.return_class,
                    range_stratum=group.range_stratum,
                    representative_range_m=group.representative_range_m,
                    group_weight=group.group_weight,
                    assigned_distance_m=distance,
                    matched_event_id=event_id,
                    matched_depth=depth,
                    matched_event_range_m=event_range,
                    integer_cost=_weighted_integer_cost(group.group_weight, distance),
                )
            )
    assignments.sort(key=lambda item: (item.ray_id, item.group_index))
    mapping = tuple(item.matched_event_id for item in assignments)
    if mapping != tuple(complete_mapping):
        raise AssertionError("independent STDA global mapping order changed")
    if sum(item.integer_cost for item in assignments) != total_cost:
        raise AssertionError("independent STDA integer objective changed")
    return tuple(assignments), total_cost, total_matched, mapping


def _mapping_record(assignment: _Assignment) -> bytes:
    return struct.pack(
        "<qQQ",
        assignment.ray_id,
        assignment.group_index,
        assignment.matched_event_id,
    )


def _class_metric_rows(
    assignments: tuple[_Assignment, ...],
) -> tuple[dict[str, Any], ...]:
    metrics: list[dict[str, Any]] = []
    for range_stratum, range_label in enumerate(RANGE_LABELS):
        for return_class, return_label in enumerate(RETURN_LABELS):
            selected = tuple(
                assignment
                for assignment in assignments
                if assignment.range_stratum == range_stratum
                and assignment.return_class == return_class
            )
            key = f"{range_label}_{return_label}"
            if not selected:
                metrics.append(
                    {
                        "key": key,
                        "range_stratum": range_stratum,
                        "return_class": return_class,
                        "applicable": False,
                        "group_count": 0,
                        "effective_weight": 0.0,
                        "completeness_mean_distance_m": None,
                        "recall_1m": None,
                        "passed": None,
                    }
                )
                continue
            weight_sum = math.fsum(item.group_weight for item in selected)
            completeness = math.fsum(
                item.group_weight * item.assigned_distance_m for item in selected
            ) / weight_sum
            recall = math.fsum(
                item.group_weight
                for item in selected
                if item.assigned_distance_m <= COMPLETENESS_GATE_M
            ) / weight_sum
            metrics.append(
                {
                    "key": key,
                    "range_stratum": range_stratum,
                    "return_class": return_class,
                    "applicable": True,
                    "group_count": len(selected),
                    "effective_weight": weight_sum,
                    "completeness_mean_distance_m": completeness,
                    "recall_1m": recall,
                    "passed": (
                        completeness <= COMPLETENESS_GATE_M
                        and recall >= RECALL_GATE
                    ),
                }
            )
    return tuple(metrics)


def _class_metric_record(metric: Mapping[str, Any]) -> bytes:
    optional = b""
    if metric["applicable"]:
        optional = struct.pack(
            "<ddB",
            metric["completeness_mean_distance_m"],
            metric["recall_1m"],
            int(metric["passed"]),
        )
    return b"".join(
        (
            metric["key"].encode("ascii") + b"\0",
            struct.pack(
                "<BBBQd",
                metric["range_stratum"],
                metric["return_class"],
                int(metric["applicable"]),
                metric["group_count"],
                metric["effective_weight"],
            ),
            optional,
        )
    )


def _canonical_input_digest(
    domain_sha256: str,
    output_sha256: str,
    target_sha256: str,
) -> str:
    return _structural_hash_parts(
        "canonical_inputs",
        (
            bytes.fromhex(domain_sha256),
            bytes.fromhex(output_sha256),
            bytes.fromhex(target_sha256),
        ),
    )


def _result_digest(
    *,
    canonical_inputs_sha256: str,
    structural_domain_valid: bool,
    evaluation_complete: bool,
    failure_reasons: tuple[str, ...],
    groups_sha256: str,
    assignments_sha256: str,
    mapping_sha256: str,
    class_metrics_sha256: str,
    integer_cost: int | None,
    matched_count: int,
    mapping: tuple[int, ...],
) -> str:
    cost_bytes = (
        b"none"
        if integer_cost is None
        else _pack_nonnegative_integer(integer_cost)
    )
    return _structural_hash_parts(
        "evaluation_result",
        (
            bytes.fromhex(canonical_inputs_sha256),
            bytes((int(structural_domain_valid), int(evaluation_complete))),
            b"\0".join(reason.encode("ascii") for reason in failure_reasons),
            bytes.fromhex(groups_sha256),
            bytes.fromhex(assignments_sha256),
            bytes.fromhex(mapping_sha256),
            bytes.fromhex(class_metrics_sha256),
            cost_bytes,
            struct.pack("<Q", matched_count),
            _pack_u64_tuple(mapping),
        ),
    )


def _group_plain(group: _TargetGroup) -> dict[str, Any]:
    return {
        "ray_id": group.ray_id,
        "group_index": group.group_index,
        "return_class": group.return_class,
        "range_stratum": group.range_stratum,
        "representative_range_m": group.representative_range_m,
        "group_weight": group.group_weight,
        "canonical_target_ids": list(group.target_ids),
    }


def _assignment_plain(assignment: _Assignment) -> dict[str, Any]:
    return {
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


def _event_plain(event: _OutputEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "ray_id": event.ray_id,
        "inferred_depth": event.inferred_depth,
        "range_m": event.range_m,
        "xyz_sha256": sha256_bytes(event.xyz_bytes),
    }


def _atom_plain(atom: _TargetAtom) -> dict[str, Any]:
    return {
        "canonical_target_id": atom.target_id,
        "ray_id": atom.ray_id,
        "range_m": atom.range_m,
        "weight": atom.weight,
        "xyz_sha256": sha256_bytes(atom.xyz_bytes),
    }


def _plain_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _plain_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _plain_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return bool(left == right)


def _mapping_hash_from_plain_rows(rows: Sequence[Mapping[str, Any]]) -> str | None:
    records: list[bytes] = []
    try:
        for row in rows:
            ray_id = row["ray_id"]
            group_index = row["group_index"]
            event_id = row["matched_event_id"]
            if (
                type(ray_id) is not int
                or type(group_index) is not int
                or type(event_id) is not int
                or not -(2**63) <= ray_id < 2**63
                or not 0 <= group_index < 2**64
                or not 0 <= event_id < 2**64
            ):
                return None
            records.append(struct.pack("<qQQ", ray_id, group_index, event_id))
    except (KeyError, TypeError):
        return None
    return _structural_hash_records("complete_group_event_mapping", records)


def _numeric_field_equal(value: Any, expected: float) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value) == expected
    )


def _replay_expected_mapping(
    expected_report: Mapping[str, Any],
    groups: tuple[_TargetGroup, ...],
    events: tuple[_OutputEvent, ...],
    independent_assignments: tuple[_Assignment, ...],
    independent_cost: int | None,
    independent_matched_count: int,
    independent_mapping: tuple[int, ...],
) -> dict[str, bool]:
    rows = expected_report.get("assignments")
    expected_tuple = expected_report.get("complete_assignment_tuple")
    expected_key = expected_report.get("objective_key")
    if not isinstance(rows, list) or not isinstance(expected_tuple, list):
        return {
            "mapping_assignment_count": False,
            "mapping_canonical_group_order": False,
            "mapping_group_metadata": False,
            "mapping_tuple_consistent": False,
            "mapping_event_ids_valid": False,
            "mapping_eligibility": False,
            "mapping_event_order": False,
            "mapping_event_nonreuse": False,
            "mapping_distances": False,
            "mapping_depth_and_range": False,
            "mapping_entry_objectives": False,
            "mapping_total_objective": False,
            "mapping_matched_count": False,
            "mapping_global_key": False,
            "mapping_is_independent_optimum": False,
            "mapping_hash_matches_rows": False,
        }

    assignment_count = len(rows) == len(groups)
    canonical_order = assignment_count
    group_metadata = assignment_count
    tuple_consistent = len(expected_tuple) == len(rows)
    event_ids_valid = assignment_count
    eligibility = assignment_count
    event_order = assignment_count
    event_nonreuse = assignment_count
    distances = assignment_count
    depth_and_range = assignment_count
    entry_objectives = assignment_count
    matched_ids: set[int] = set()
    replay_mapping: list[int] = []
    replay_cost = 0
    replay_matched = 0
    previous_ray: int | None = None
    previous_event_rank = -1
    previously_matched = False
    events_by_id = {event.event_id: event for event in events}
    event_rank = {
        event.event_id: rank
        for ray_id in sorted({event.ray_id for event in events if event.ray_id >= 0})
        for rank, event in enumerate(
            sorted(
                (candidate for candidate in events if candidate.ray_id == ray_id),
                key=lambda candidate: (candidate.range_m, candidate.event_id),
            )
        )
    }

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or index >= len(groups):
            canonical_order = False
            group_metadata = False
            tuple_consistent = False
            event_ids_valid = False
            eligibility = False
            event_order = False
            event_nonreuse = False
            distances = False
            depth_and_range = False
            entry_objectives = False
            continue
        group = groups[index]
        ray_id = row.get("ray_id")
        group_index = row.get("group_index")
        canonical_order = canonical_order and (
            type(ray_id) is int
            and type(group_index) is int
            and (ray_id, group_index) == (group.ray_id, group.group_index)
        )
        group_metadata = group_metadata and (
            row.get("return_class") == group.return_class
            and row.get("range_stratum") == group.range_stratum
            and _numeric_field_equal(
                row.get("representative_range_m"),
                group.representative_range_m,
            )
            and _numeric_field_equal(row.get("group_weight"), group.group_weight)
        )
        if previous_ray != group.ray_id:
            previous_ray = group.ray_id
            previous_event_rank = -1
            previously_matched = False

        event_id = row.get("matched_event_id")
        if type(event_id) is not int:
            event_ids_valid = False
            eligibility = False
            event_order = False
            event_nonreuse = False
            distances = False
            depth_and_range = False
            entry_objectives = False
            continue
        replay_mapping.append(event_id)
        if index >= len(expected_tuple) or expected_tuple[index] != event_id:
            tuple_consistent = False

        if event_id == UNMATCHED_SENTINEL:
            expected_distance = UNMATCHED_DISTANCE_M
            expected_depth = -1
            expected_range = None
        else:
            event = events_by_id.get(event_id)
            if event is None:
                event_ids_valid = False
                eligibility = False
                event_order = False
                distances = False
                depth_and_range = False
                entry_objectives = False
                continue
            if event_id in matched_ids:
                event_nonreuse = False
            matched_ids.add(event_id)
            if event.ray_id != group.ray_id:
                eligibility = False
            eligible = (
                group.return_class == RETURN_FIRST and event.inferred_depth == 1
            ) or (
                group.return_class == RETURN_LATER
                and event.inferred_depth >= 2
                and previously_matched
            )
            eligibility = eligibility and eligible
            rank = event_rank[event_id]
            if rank <= previous_event_rank:
                event_order = False
            previous_event_rank = rank
            previously_matched = True
            replay_matched += 1
            expected_distance = abs(
                group.representative_range_m - event.range_m
            )
            expected_depth = event.inferred_depth
            expected_range = event.range_m

        distances = distances and _numeric_field_equal(
            row.get("assigned_distance_m"),
            expected_distance,
        )
        depth_and_range = depth_and_range and row.get("matched_depth") == expected_depth
        if expected_range is None:
            depth_and_range = depth_and_range and row.get("matched_event_range_m") is None
        else:
            depth_and_range = depth_and_range and _numeric_field_equal(
                row.get("matched_event_range_m"),
                expected_range,
            )
        expected_entry_cost = _weighted_integer_cost(
            group.group_weight,
            expected_distance,
        )
        entry_cost = row.get("integer_cost")
        entry_objectives = entry_objectives and (
            type(entry_cost) is int and entry_cost == expected_entry_cost
        )
        replay_cost += expected_entry_cost

    if len(replay_mapping) != len(rows):
        tuple_consistent = False
    report_cost = expected_report.get("integer_cost")
    report_matched = expected_report.get("matched_count")
    total_objective = (
        independent_cost is None
        if not groups and report_cost is None
        else type(report_cost) is int and report_cost == replay_cost
    )
    matched_count = type(report_matched) is int and report_matched == replay_matched
    expected_objective_key = (
        None
        if report_cost is None
        else [report_cost, -report_matched, list(expected_tuple)]
        if type(report_matched) is int
        else None
    )
    global_key = _plain_equal(expected_key, expected_objective_key)
    independent_key = (
        None
        if independent_cost is None
        else [
            independent_cost,
            -independent_matched_count,
            list(independent_mapping),
        ]
    )
    mapping_is_optimum = (
        list(replay_mapping) == list(independent_mapping)
        and _plain_equal(expected_key, independent_key)
        and rows == [_assignment_plain(item) for item in independent_assignments]
    )
    mapping_hash = _mapping_hash_from_plain_rows(
        tuple(row for row in rows if isinstance(row, Mapping))
    )
    return {
        "mapping_assignment_count": assignment_count,
        "mapping_canonical_group_order": canonical_order,
        "mapping_group_metadata": group_metadata,
        "mapping_tuple_consistent": tuple_consistent,
        "mapping_event_ids_valid": event_ids_valid,
        "mapping_eligibility": eligibility,
        "mapping_event_order": event_order,
        "mapping_event_nonreuse": event_nonreuse,
        "mapping_distances": distances,
        "mapping_depth_and_range": depth_and_range,
        "mapping_entry_objectives": entry_objectives,
        "mapping_total_objective": total_objective,
        "mapping_matched_count": matched_count,
        "mapping_global_key": global_key,
        "mapping_is_independent_optimum": mapping_is_optimum,
        "mapping_hash_matches_rows": (
            mapping_hash is not None
            and mapping_hash == expected_report.get("complete_mapping_sha256")
        ),
    }


def verify_structural_replay(
    output_xyz_le_f4: bytes,
    target_xyz_confidence_le_f4: bytes,
    azimuth_edges_rad: Sequence[float],
    elevation_edges_rad: Sequence[float],
    expected_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay a plain STDA structural mapping and report.

    ``expected_report`` is a JSON-compatible mapping with schema
    ``stda_f0_structural_report_v1``. It contains the evaluator's plain output
    and target summaries, canonical groups, full assignment rows, six class
    metrics, global objective key, and all structure hashes. The verifier reads
    only immutable raw float32 bytes and angular edges; it reconstructs every
    identity and decision represented by the report.

    ``passed`` means that the independent replay agrees with the supplied
    report. A correctly reported out-of-domain arm therefore has ``passed``
    true while ``structural_domain_valid`` remains false.
    """

    if not isinstance(expected_report, Mapping):
        raise TypeError("STDA expected structural report must be a plain mapping")
    if not isinstance(output_xyz_le_f4, bytes):
        raise TypeError("STDA output XYZ must be immutable bytes")
    if not isinstance(target_xyz_confidence_le_f4, bytes):
        raise TypeError("STDA target XYZ-confidence must be immutable bytes")
    domain = _IndependentDomain.from_edges(
        azimuth_edges_rad,
        elevation_edges_rad,
    )
    output_count = len(output_xyz_le_f4) // 12
    target_count = len(target_xyz_confidence_le_f4) // 16
    events, duplicate_count, invalid_event_ids, output_sha256 = (
        _canonical_output_events(output_xyz_le_f4, domain)
    )
    atoms, zero_count, invalid_target_ids, target_sha256 = (
        _canonical_positive_target_atoms(target_xyz_confidence_le_f4, domain)
    )
    raw_inputs_sha256 = _structural_hash_parts(
        "raw_inputs",
        (
            bytes.fromhex(domain.digest_sha256),
            struct.pack("<Q", output_count),
            output_xyz_le_f4,
            struct.pack("<Q", target_count),
            target_xyz_confidence_le_f4,
        ),
    )
    canonical_inputs_sha256 = _canonical_input_digest(
        domain.digest_sha256,
        output_sha256,
        target_sha256,
    )
    failure_reasons: list[str] = []
    if invalid_event_ids:
        failure_reasons.append("output_event_out_of_structural_domain")
    if invalid_target_ids:
        failure_reasons.append("positive_target_out_of_structural_domain")
    structural_domain_valid = not failure_reasons
    if structural_domain_valid:
        groups = _build_target_groups(atoms)
        assignments, integer_cost, matched_count, mapping = (
            _build_assignments_independently(groups, events)
        )
        class_metrics = _class_metric_rows(assignments)
        evaluation_complete = True
    else:
        groups = ()
        assignments = ()
        class_metrics = ()
        integer_cost = None
        matched_count = 0
        mapping = ()
        evaluation_complete = False

    groups_sha256 = _structural_hash_records(
        "target_return_groups",
        map(_group_record, groups),
    )
    assignments_sha256 = _structural_hash_records(
        "group_assignments",
        map(_assignment_record, assignments),
    )
    mapping_sha256 = _structural_hash_records(
        "complete_group_event_mapping",
        map(_mapping_record, assignments),
    )
    class_metrics_sha256 = _structural_hash_records(
        "range_return_class_metrics",
        map(_class_metric_record, class_metrics),
    )
    result_sha256 = _result_digest(
        canonical_inputs_sha256=canonical_inputs_sha256,
        structural_domain_valid=structural_domain_valid,
        evaluation_complete=evaluation_complete,
        failure_reasons=tuple(failure_reasons),
        groups_sha256=groups_sha256,
        assignments_sha256=assignments_sha256,
        mapping_sha256=mapping_sha256,
        class_metrics_sha256=class_metrics_sha256,
        integer_cost=integer_cost,
        matched_count=matched_count,
        mapping=mapping,
    )
    objective_key = (
        None
        if integer_cost is None
        else [integer_cost, -matched_count, list(mapping)]
    )
    output_report = {
        "source_row_count": output_count,
        "duplicate_row_count": duplicate_count,
        "invalid_event_ids": list(invalid_event_ids),
        "digest_sha256": output_sha256,
    }
    target_report = {
        "source_row_count": target_count,
        "zero_confidence_row_count": zero_count,
        "invalid_target_ids": list(invalid_target_ids),
        "digest_sha256": target_sha256,
    }
    computed_report = {
        "schema": STRUCTURAL_REPORT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "domain_sha256": domain.digest_sha256,
        "raw_inputs_sha256": raw_inputs_sha256,
        "canonical_inputs_sha256": canonical_inputs_sha256,
        "output": output_report,
        "target": target_report,
        "structural_domain_valid": structural_domain_valid,
        "evaluation_complete": evaluation_complete,
        "failure_reasons": list(failure_reasons),
        "groups": [_group_plain(group) for group in groups],
        "assignments": [_assignment_plain(item) for item in assignments],
        "class_metrics": [dict(metric) for metric in class_metrics],
        "integer_cost": integer_cost,
        "matched_count": matched_count,
        "complete_assignment_tuple": list(mapping),
        "objective_key": objective_key,
        "groups_sha256": groups_sha256,
        "assignments_sha256": assignments_sha256,
        "complete_mapping_sha256": mapping_sha256,
        "class_metrics_sha256": class_metrics_sha256,
        "result_sha256": result_sha256,
    }
    expected_output = expected_report.get("output")
    expected_target = expected_report.get("target")
    checks = {
        "report_schema": expected_report.get("schema") == STRUCTURAL_REPORT_SCHEMA,
        "protocol_sha256": expected_report.get("protocol_sha256") == PROTOCOL_SHA256,
        "protocol_freeze_commit": (
            expected_report.get("protocol_freeze_commit")
            == PROTOCOL_FREEZE_COMMIT
        ),
        "domain_sha256": expected_report.get("domain_sha256") == domain.digest_sha256,
        "raw_inputs_sha256": (
            expected_report.get("raw_inputs_sha256") == raw_inputs_sha256
        ),
        "canonical_inputs_sha256": (
            expected_report.get("canonical_inputs_sha256")
            == canonical_inputs_sha256
        ),
        "output_report": (
            isinstance(expected_output, dict)
            and _plain_equal(expected_output, output_report)
        ),
        "target_report": (
            isinstance(expected_target, dict)
            and _plain_equal(expected_target, target_report)
        ),
        "structural_domain_valid": (
            expected_report.get("structural_domain_valid")
            is structural_domain_valid
        ),
        "evaluation_complete": (
            expected_report.get("evaluation_complete") is evaluation_complete
        ),
        "failure_reasons": _plain_equal(
            expected_report.get("failure_reasons"),
            list(failure_reasons),
        ),
        "groups": _plain_equal(
            expected_report.get("groups"),
            computed_report["groups"],
        ),
        "assignments": _plain_equal(
            expected_report.get("assignments"),
            computed_report["assignments"],
        ),
        "class_metrics": _plain_equal(
            expected_report.get("class_metrics"),
            computed_report["class_metrics"],
        ),
        "integer_cost": expected_report.get("integer_cost") == integer_cost,
        "matched_count": expected_report.get("matched_count") == matched_count,
        "complete_assignment_tuple": _plain_equal(
            expected_report.get("complete_assignment_tuple"),
            list(mapping),
        ),
        "global_objective_key": _plain_equal(
            expected_report.get("objective_key"),
            objective_key,
        ),
        "groups_sha256": expected_report.get("groups_sha256") == groups_sha256,
        "assignments_sha256": (
            expected_report.get("assignments_sha256") == assignments_sha256
        ),
        "complete_mapping_sha256": (
            expected_report.get("complete_mapping_sha256") == mapping_sha256
        ),
        "class_metrics_sha256": (
            expected_report.get("class_metrics_sha256") == class_metrics_sha256
        ),
        "result_sha256": expected_report.get("result_sha256") == result_sha256,
    }
    checks.update(
        _replay_expected_mapping(
            expected_report,
            groups,
            events,
            assignments,
            integer_cost,
            matched_count,
            mapping,
        )
    )
    checks["all_input_output_hashes"] = all(
        checks[key]
        for key in (
            "domain_sha256",
            "raw_inputs_sha256",
            "canonical_inputs_sha256",
            "groups_sha256",
            "assignments_sha256",
            "complete_mapping_sha256",
            "class_metrics_sha256",
            "result_sha256",
        )
    )
    return {
        "schema": STRUCTURAL_REPLAY_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "passed": all(checks.values()),
        "structural_domain_valid": structural_domain_valid,
        "evaluation_complete": evaluation_complete,
        "checks": checks,
        "computed_report": computed_report,
        "canonical_output_events": [_event_plain(event) for event in events],
        "canonical_positive_targets": [_atom_plain(atom) for atom in atoms],
    }
