"""Independent byte-level verifier for the frozen STDA-F0 protocol.

This module deliberately does not import the support builder, target fitter,
rounding solver, or structural evaluator.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import hashlib
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
