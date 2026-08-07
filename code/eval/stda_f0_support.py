"""Target-free support construction and byte commitments for STDA-F0."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


PROTOCOL = "stda_f0_sparse_target_demand_assignment"
PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
PROTOCOL_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"

SUPPORT_SCHEMA = "stda_f0_packed_support_v1"
CANDIDATE_FIELD_SCHEMA = "stda_f0_candidate_field_v1"
CAPACITY_NO_GO_STATUS = "stda_f0_packed_support_capacity_no_go"
REQUIRED_EXPORT_COUNT = 10_000
GRID_SCALE_PER_M = 20
COLOR_COUNT = 8
MINIMUM_DISTANCE_SQUARED_NUMERATOR = 1
MINIMUM_DISTANCE_SQUARED_DENOMINATOR = 400
MAX_HEADER_BYTES = 64 * 1024
BUILDER_SPACING_SELF_CHECK_AUTHORITATIVE = False
FORMAL_SPACING_VERIFIER_MODULE = "eval.stda_f0_verify"
SUPPORT_SERIALIZATION_FRAMING = (
    "canonical_ascii_json_line_then_column_major_payload"
)

SUPPORT_COLUMN_SCHEMA = (
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

CANDIDATE_COLUMN_SCHEMA = (
    ("stable_candidate_id", "<i8"),
    ("x_m", "<f4"),
    ("y_m", "<f4"),
    ("z_m", "<f4"),
    ("base_confidence", "<f4"),
)


class SupportSerializationError(ValueError):
    """Raised when support bytes do not match the frozen schema."""


class SpacingViolationError(ValueError):
    """Raised by the non-authoritative builder spacing self-check."""

    def __init__(self, first_row: int, second_row: int) -> None:
        self.first_row = first_row
        self.second_row = second_row
        super().__init__(
            "serialized XYZ rows "
            f"{first_row} and {second_row} are strictly closer than 5 cm"
        )


@dataclass(frozen=True)
class SupportCapacityStatus:
    """Section 5 capacity outcome without making a graph-level conclusion."""

    required_count: int
    support_count: int
    selected_color_id: int
    color_cardinalities: tuple[int, ...]
    capacity_sufficient: bool
    graph_construction_allowed: bool
    terminal_status: str | None
    conclusion_scope: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "required_count": self.required_count,
            "support_count": self.support_count,
            "selected_color_id": self.selected_color_id,
            "color_cardinalities": list(self.color_cardinalities),
            "capacity_sufficient": self.capacity_sufficient,
            "graph_construction_allowed": self.graph_construction_allowed,
            "terminal_status": self.terminal_status,
            "conclusion_scope": self.conclusion_scope,
        }


@dataclass(frozen=True)
class PackedSupport:
    """One target-independent largest-color parity support."""

    candidate_count: int
    candidate_field_sha256: str
    stable_candidate_id: np.ndarray
    cell_xyz: np.ndarray
    xyz_m: np.ndarray
    base_confidence: np.ndarray
    color_id: np.ndarray
    selected_color_id: int
    color_cardinalities: tuple[int, ...]
    digest_sha256: str

    @property
    def support_count(self) -> int:
        return int(self.stable_candidate_id.size)

    @property
    def retained_cell_count(self) -> int:
        return sum(self.color_cardinalities)

    @property
    def capacity_status(self) -> SupportCapacityStatus:
        return support_capacity_status(self)


@dataclass(frozen=True)
class BuilderSpacingSelfCheck:
    """Non-authoritative exact builder-side replay of serialized spacing."""

    input_sha256: str
    point_count: int
    checked_neighbor_pair_count: int
    threshold_squared_numerator: int = MINIMUM_DISTANCE_SQUARED_NUMERATOR
    threshold_squared_denominator: int = MINIMUM_DISTANCE_SQUARED_DENOMINATOR
    passed: bool = True
    authoritative: bool = BUILDER_SPACING_SELF_CHECK_AUTHORITATIVE


@dataclass(frozen=True)
class SupportCommitment:
    """Metadata obtained by reloading bytes and running builder-side checks."""

    protocol_sha256: str
    protocol_freeze_commit: str
    support_sha256: str
    candidate_field_sha256: str
    candidate_count: int
    support_count: int
    selected_color_id: int
    color_cardinalities: tuple[int, ...]
    serialized_size_bytes: int
    capacity_status: SupportCapacityStatus
    builder_spacing_self_check_passed: bool
    builder_spacing_self_check_checked_neighbor_pair_count: int
    formal_independent_spacing_verified: bool = False


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
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


def support_serialization_schema() -> dict[str, object]:
    """Expose the frozen byte layout for a separately implemented parser."""

    return {
        "schema": SUPPORT_SCHEMA,
        "framing": SUPPORT_SERIALIZATION_FRAMING,
        "columns": [
            {"name": name, "dtype": dtype}
            for name, dtype in SUPPORT_COLUMN_SCHEMA
        ],
        "bytes_per_row": sum(
            np.dtype(dtype).itemsize for _, dtype in SUPPORT_COLUMN_SCHEMA
        ),
        "canonical_order": "stable_candidate_id_ascending",
    }


def _immutable_array(values: np.ndarray, dtype: str) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype(dtype)).reshape(
        contiguous.shape
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _same_json_tree(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_json_tree(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_tree(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return bool(left == right)


def stable_candidate_ids(candidate_count: int) -> np.ndarray:
    """Return the frozen row IDs for a canonically ordered candidate field."""

    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError("candidate_count must be a nonnegative integer")
    return _immutable_array(np.arange(candidate_count, dtype="<i8"), "<i8")


def exact_float32_floor_20(value: float | np.float32) -> int:
    """Compute floor(20*x) from the exact dyadic value of one float32."""

    value_f32 = np.float32(value)
    value_float = float(value_f32)
    if not np.isfinite(value_f32):
        raise ValueError("grid coordinate must be a finite float32")
    numerator, denominator = value_float.as_integer_ratio()
    cell = (GRID_SCALE_PER_M * numerator) // denominator
    if not np.iinfo(np.int64).min <= cell <= np.iinfo(np.int64).max:
        raise OverflowError("exact grid cell does not fit little-endian int64")
    return cell


def exact_float32_cells(xyz_m: np.ndarray) -> np.ndarray:
    """Map float32 XYZ rows to exact 5 cm grid cells."""

    xyz = np.ascontiguousarray(np.asarray(xyz_m, dtype="<f4"))
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("candidate XYZ must have shape (N,3)")
    if not np.isfinite(xyz).all():
        raise ValueError("candidate XYZ must contain only finite float32 values")
    cells = np.empty(xyz.shape, dtype="<i8")
    for row in range(xyz.shape[0]):
        for axis in range(3):
            cells[row, axis] = exact_float32_floor_20(xyz[row, axis])
    return _immutable_array(cells, "<i8")


def _canonical_candidate_inputs(
    candidate_xyz_m: np.ndarray,
    base_confidence: np.ndarray,
    candidate_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.ascontiguousarray(np.asarray(candidate_xyz_m, dtype="<f4"))
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("STDA-F0 candidate XYZ must have shape (N,3)")
    count = int(xyz.shape[0])
    confidence = np.ascontiguousarray(np.asarray(base_confidence, dtype="<f4"))
    if confidence.shape != (count,):
        raise ValueError("STDA-F0 base confidence must match candidate rows")
    if not np.isfinite(xyz).all() or not np.isfinite(confidence).all():
        raise ValueError("STDA-F0 candidate XYZ and confidence must be finite")

    if candidate_ids is None:
        ids = stable_candidate_ids(count)
    else:
        raw_ids = np.asarray(candidate_ids)
        if raw_ids.shape != (count,):
            raise ValueError("STDA-F0 stable candidate IDs must match rows")
        if raw_ids.dtype.kind not in "iu":
            raise TypeError("STDA-F0 stable candidate IDs must be integers")
        if raw_ids.dtype.kind == "u" and raw_ids.size:
            if int(raw_ids.max()) > np.iinfo(np.int64).max:
                raise OverflowError("stable candidate ID does not fit int64")
        ids = np.ascontiguousarray(raw_ids, dtype="<i8")
    if ids.size and np.any(ids < 0):
        raise ValueError("STDA-F0 stable candidate IDs must be nonnegative")
    if np.unique(ids).size != count:
        raise ValueError("STDA-F0 stable candidate IDs must be unique")
    return xyz, confidence, ids


def _candidate_columns(
    xyz_m: np.ndarray,
    base_confidence: np.ndarray,
    candidate_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(candidate_ids, kind="stable")
    return {
        "stable_candidate_id": candidate_ids[order],
        "x_m": xyz_m[order, 0],
        "y_m": xyz_m[order, 1],
        "z_m": xyz_m[order, 2],
        "base_confidence": base_confidence[order],
    }


def _columns_payload(
    schema: Sequence[tuple[str, str]],
    columns: Mapping[str, np.ndarray],
) -> bytes:
    if tuple(columns) != tuple(name for name, _ in schema):
        raise ValueError("serialized columns do not match the frozen schema")
    return b"".join(
        np.ascontiguousarray(
            np.asarray(columns[name], dtype=np.dtype(dtype))
        ).tobytes(order="C")
        for name, dtype in schema
    )


def candidate_field_digest_sha256(
    candidate_xyz_m: np.ndarray,
    base_confidence: np.ndarray,
    candidate_ids: np.ndarray | None = None,
) -> str:
    """Hash candidate rows canonically by stable ID, independent of row order."""

    xyz, confidence, ids = _canonical_candidate_inputs(
        candidate_xyz_m,
        base_confidence,
        candidate_ids,
    )
    header = _canonical_json_bytes(
        {
            "schema": CANDIDATE_FIELD_SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "candidate_count": int(ids.size),
            "canonical_order": "stable_candidate_id_ascending",
            "columns": [
                {"name": name, "dtype": dtype}
                for name, dtype in CANDIDATE_COLUMN_SCHEMA
            ],
        }
    )
    payload = _columns_payload(
        CANDIDATE_COLUMN_SCHEMA,
        _candidate_columns(xyz, confidence, ids),
    )
    return hashlib.sha256(header + payload).hexdigest()


def _color_id(cell: tuple[int, int, int]) -> int:
    return 4 * (cell[0] % 2) + 2 * (cell[1] % 2) + (cell[2] % 2)


def _support_columns(support: PackedSupport) -> dict[str, np.ndarray]:
    return {
        "stable_candidate_id": support.stable_candidate_id,
        "cell_x": support.cell_xyz[:, 0],
        "cell_y": support.cell_xyz[:, 1],
        "cell_z": support.cell_xyz[:, 2],
        "x_m": support.xyz_m[:, 0],
        "y_m": support.xyz_m[:, 1],
        "z_m": support.xyz_m[:, 2],
        "base_confidence": support.base_confidence,
        "color_id": support.color_id,
    }


def _capacity_from_counts(
    support_count: int,
    selected_color_id: int,
    color_cardinalities: tuple[int, ...],
) -> SupportCapacityStatus:
    sufficient = support_count >= REQUIRED_EXPORT_COUNT
    return SupportCapacityStatus(
        required_count=REQUIRED_EXPORT_COUNT,
        support_count=support_count,
        selected_color_id=selected_color_id,
        color_cardinalities=color_cardinalities,
        capacity_sufficient=sufficient,
        graph_construction_allowed=sufficient,
        terminal_status=None if sufficient else CAPACITY_NO_GO_STATUS,
        conclusion_scope=(
            None
            if sufficient
            else "exact_section_5_largest_color_parity_support_only"
        ),
    )


def support_capacity_status(support: PackedSupport) -> SupportCapacityStatus:
    """Return the frozen packed-support capacity status for one frame."""

    return _capacity_from_counts(
        support.support_count,
        support.selected_color_id,
        support.color_cardinalities,
    )


def _support_header(
    support: PackedSupport,
    payload_sha256: str,
) -> dict[str, object]:
    capacity = support_capacity_status(support)
    return {
        "schema": SUPPORT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "candidate_count": support.candidate_count,
        "candidate_field_sha256": support.candidate_field_sha256,
        "retained_cell_count": support.retained_cell_count,
        "support_count": support.support_count,
        "selected_color_id": support.selected_color_id,
        "color_cardinalities": list(support.color_cardinalities),
        "required_export_count": REQUIRED_EXPORT_COUNT,
        "capacity_sufficient": capacity.capacity_sufficient,
        "graph_construction_allowed": capacity.graph_construction_allowed,
        "terminal_status": capacity.terminal_status,
        "canonical_order": "stable_candidate_id_ascending",
        "framing": SUPPORT_SERIALIZATION_FRAMING,
        "grid_scale_per_m": GRID_SCALE_PER_M,
        "minimum_distance_squared": {
            "numerator": MINIMUM_DISTANCE_SQUARED_NUMERATOR,
            "denominator": MINIMUM_DISTANCE_SQUARED_DENOMINATOR,
        },
        "columns": [
            {"name": name, "dtype": dtype}
            for name, dtype in SUPPORT_COLUMN_SCHEMA
        ],
        "payload_sha256": payload_sha256,
    }


def _validate_support(support: PackedSupport) -> None:
    count = support.support_count
    if type(support.candidate_count) is not int or support.candidate_count < 0:
        raise ValueError("packed support candidate_count is invalid")
    if not _is_sha256(support.candidate_field_sha256):
        raise ValueError("packed support candidate digest is malformed")
    if support.stable_candidate_id.shape != (count,):
        raise ValueError("packed support stable ID shape changed")
    if support.cell_xyz.shape != (count, 3):
        raise ValueError("packed support cell shape changed")
    if support.xyz_m.shape != (count, 3):
        raise ValueError("packed support XYZ shape changed")
    if support.base_confidence.shape != (count,):
        raise ValueError("packed support confidence shape changed")
    if support.color_id.shape != (count,):
        raise ValueError("packed support color shape changed")
    expected_dtypes = (
        (support.stable_candidate_id, "<i8"),
        (support.cell_xyz, "<i8"),
        (support.xyz_m, "<f4"),
        (support.base_confidence, "<f4"),
        (support.color_id, "<u1"),
    )
    for values, dtype in expected_dtypes:
        if values.dtype != np.dtype(dtype):
            raise ValueError("packed support column dtype changed")
    if not np.isfinite(support.xyz_m).all() or not np.isfinite(
        support.base_confidence
    ).all():
        raise ValueError("packed support contains a non-finite value")
    if count and np.any(support.stable_candidate_id < 0):
        raise ValueError("packed support stable IDs must be nonnegative")
    if count > 1 and not np.all(
        support.stable_candidate_id[1:] > support.stable_candidate_id[:-1]
    ):
        raise ValueError("packed support must be ordered by unique stable ID")
    if len(support.color_cardinalities) != COLOR_COUNT or any(
        type(value) is not int or value < 0
        for value in support.color_cardinalities
    ):
        raise ValueError("packed support must record eight color cardinalities")
    expected_color = min(
        range(COLOR_COUNT),
        key=lambda color: (-support.color_cardinalities[color], color),
    )
    if support.selected_color_id != expected_color:
        raise ValueError("packed support largest-color tie rule changed")
    if support.color_cardinalities[expected_color] != count:
        raise ValueError("packed support count differs from selected color")
    if support.retained_cell_count > support.candidate_count:
        raise ValueError("packed support retained more cells than candidates")
    if count and not np.all(support.color_id == support.selected_color_id):
        raise ValueError("packed support contains a row from another color")
    if np.unique(support.cell_xyz, axis=0).shape[0] != count:
        raise ValueError("packed support contains duplicate retained cells")

    computed_cells = exact_float32_cells(support.xyz_m)
    if not np.array_equal(computed_cells, support.cell_xyz):
        raise ValueError("packed support cell metadata differs from exact XYZ")
    computed_colors = np.asarray(
        [_color_id(tuple(int(value) for value in row)) for row in computed_cells],
        dtype="<u1",
    )
    if not np.array_equal(computed_colors, support.color_id):
        raise ValueError("packed support color metadata differs from exact cells")


def build_packed_support(
    candidate_xyz_m: np.ndarray,
    base_confidence: np.ndarray,
    candidate_ids: np.ndarray | None = None,
) -> PackedSupport:
    """Build the target-independent Section 5 largest-color support."""

    xyz, confidence, ids = _canonical_candidate_inputs(
        candidate_xyz_m,
        base_confidence,
        candidate_ids,
    )
    cells = exact_float32_cells(xyz)
    winner_by_cell: dict[tuple[int, int, int], int] = {}
    for row in range(ids.size):
        cell = tuple(int(value) for value in cells[row])
        incumbent = winner_by_cell.get(cell)
        if incumbent is None:
            winner_by_cell[cell] = row
            continue
        if confidence[row] > confidence[incumbent] or (
            confidence[row] == confidence[incumbent]
            and ids[row] < ids[incumbent]
        ):
            winner_by_cell[cell] = row

    rows_by_color: list[list[int]] = [[] for _ in range(COLOR_COUNT)]
    for cell, row in winner_by_cell.items():
        rows_by_color[_color_id(cell)].append(row)
    color_cardinalities = tuple(len(rows) for rows in rows_by_color)
    selected_color_id = min(
        range(COLOR_COUNT),
        key=lambda color: (-color_cardinalities[color], color),
    )
    selected_rows = sorted(
        rows_by_color[selected_color_id],
        key=lambda row: int(ids[row]),
    )
    selected = np.asarray(selected_rows, dtype=np.int64)

    candidate_digest = candidate_field_digest_sha256(xyz, confidence, ids)
    provisional = PackedSupport(
        candidate_count=int(ids.size),
        candidate_field_sha256=candidate_digest,
        stable_candidate_id=_immutable_array(ids[selected], "<i8"),
        cell_xyz=_immutable_array(cells[selected], "<i8"),
        xyz_m=_immutable_array(xyz[selected], "<f4"),
        base_confidence=_immutable_array(confidence[selected], "<f4"),
        color_id=_immutable_array(
            np.full(selected.size, selected_color_id, dtype="<u1"),
            "<u1",
        ),
        selected_color_id=selected_color_id,
        color_cardinalities=color_cardinalities,
        digest_sha256="",
    )
    _validate_support(provisional)
    serialized = serialize_packed_support(provisional)
    digest = hashlib.sha256(serialized).hexdigest()
    return PackedSupport(
        candidate_count=provisional.candidate_count,
        candidate_field_sha256=provisional.candidate_field_sha256,
        stable_candidate_id=provisional.stable_candidate_id,
        cell_xyz=provisional.cell_xyz,
        xyz_m=provisional.xyz_m,
        base_confidence=provisional.base_confidence,
        color_id=provisional.color_id,
        selected_color_id=provisional.selected_color_id,
        color_cardinalities=provisional.color_cardinalities,
        digest_sha256=digest,
    )


def serialize_packed_support(support: PackedSupport) -> bytes:
    """Serialize one support as canonical header plus little-endian columns."""

    _validate_support(support)
    payload = _columns_payload(SUPPORT_COLUMN_SCHEMA, _support_columns(support))
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    serialized = _canonical_json_bytes(
        _support_header(support, payload_sha256)
    ) + payload
    observed = hashlib.sha256(serialized).hexdigest()
    if support.digest_sha256 and support.digest_sha256 != observed:
        raise SupportSerializationError("packed support digest changed")
    return serialized


def packed_support_digest_sha256(serialized: bytes) -> str:
    """Hash exact serialized support bytes without interpreting them."""

    if not isinstance(serialized, bytes):
        raise TypeError("serialized support must be immutable bytes")
    return hashlib.sha256(serialized).hexdigest()


def _read_plain_int(document: Mapping[str, object], key: str) -> int:
    value = document.get(key)
    if type(value) is not int:
        raise SupportSerializationError(f"support header {key} is not an integer")
    return value


def _parse_support_header(serialized: bytes) -> tuple[dict[str, object], bytes, bytes]:
    if not isinstance(serialized, bytes):
        raise TypeError("serialized support must be immutable bytes")
    newline = serialized.find(b"\n")
    if newline <= 0 or newline + 1 > MAX_HEADER_BYTES:
        raise SupportSerializationError("support header boundary is invalid")
    header_bytes = serialized[: newline + 1]
    payload = serialized[newline + 1 :]
    try:
        decoded = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupportSerializationError(
            "support header is not canonical JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise SupportSerializationError("support header must be a JSON object")
    header = dict(decoded)
    if _canonical_json_bytes(header) != header_bytes:
        raise SupportSerializationError("support header bytes are not canonical")
    return header, header_bytes, payload


def deserialize_packed_support(
    serialized: bytes,
    *,
    expected_sha256: str | None = None,
) -> PackedSupport:
    """Reload and validate every byte of a frozen packed support."""

    if expected_sha256 is not None and not _is_sha256(expected_sha256):
        raise ValueError("expected support SHA-256 is malformed")
    observed_sha256 = packed_support_digest_sha256(serialized)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise SupportSerializationError("serialized support SHA-256 mismatch")
    header, _, payload = _parse_support_header(serialized)
    if header.get("schema") != SUPPORT_SCHEMA:
        raise SupportSerializationError("support schema changed")
    if header.get("protocol_sha256") != PROTOCOL_SHA256:
        raise SupportSerializationError("support protocol SHA-256 changed")
    if header.get("protocol_freeze_commit") != PROTOCOL_FREEZE_COMMIT:
        raise SupportSerializationError("support freeze commit changed")

    candidate_count = _read_plain_int(header, "candidate_count")
    support_count = _read_plain_int(header, "support_count")
    selected_color_id = _read_plain_int(header, "selected_color_id")
    if candidate_count < 0 or support_count < 0:
        raise SupportSerializationError("support header contains a negative count")
    raw_colors = header.get("color_cardinalities")
    if not isinstance(raw_colors, list) or len(raw_colors) != COLOR_COUNT:
        raise SupportSerializationError("support header color counts changed")
    if any(type(value) is not int or value < 0 for value in raw_colors):
        raise SupportSerializationError("support header color count is invalid")
    color_cardinalities = tuple(int(value) for value in raw_colors)
    candidate_digest = header.get("candidate_field_sha256")
    if not _is_sha256(candidate_digest):
        raise SupportSerializationError("support candidate digest is malformed")
    payload_digest = header.get("payload_sha256")
    if not _is_sha256(payload_digest):
        raise SupportSerializationError("support payload digest is malformed")
    if hashlib.sha256(payload).hexdigest() != payload_digest:
        raise SupportSerializationError("support payload SHA-256 mismatch")

    bytes_per_row = sum(np.dtype(dtype).itemsize for _, dtype in SUPPORT_COLUMN_SCHEMA)
    if len(payload) != support_count * bytes_per_row:
        raise SupportSerializationError("support payload size differs from schema")
    columns: dict[str, np.ndarray] = {}
    offset = 0
    for name, dtype in SUPPORT_COLUMN_SCHEMA:
        itemsize = np.dtype(dtype).itemsize
        stop = offset + support_count * itemsize
        columns[name] = np.frombuffer(payload[offset:stop], dtype=np.dtype(dtype))
        offset = stop
    if offset != len(payload):
        raise SupportSerializationError("support payload has trailing bytes")

    support = PackedSupport(
        candidate_count=candidate_count,
        candidate_field_sha256=str(candidate_digest),
        stable_candidate_id=columns["stable_candidate_id"],
        cell_xyz=_immutable_array(
            np.column_stack(
                (columns["cell_x"], columns["cell_y"], columns["cell_z"])
            ),
            "<i8",
        ),
        xyz_m=_immutable_array(
            np.column_stack((columns["x_m"], columns["y_m"], columns["z_m"])),
            "<f4",
        ),
        base_confidence=columns["base_confidence"],
        color_id=columns["color_id"],
        selected_color_id=selected_color_id,
        color_cardinalities=color_cardinalities,
        digest_sha256=observed_sha256,
    )
    _validate_support(support)
    expected_header = _support_header(support, str(payload_digest))
    if not _same_json_tree(header, expected_header):
        raise SupportSerializationError("support header semantics changed")
    return support


def serialize_xyz_float32(xyz_m: np.ndarray) -> bytes:
    """Serialize finite XYZ rows as contiguous little-endian float32 triples."""

    xyz = np.ascontiguousarray(np.asarray(xyz_m, dtype="<f4"))
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("serialized XYZ must have shape (N,3)")
    if not np.isfinite(xyz).all():
        raise ValueError("serialized XYZ must contain only finite values")
    return xyz.tobytes(order="C")


def deserialize_xyz_float32(serialized_xyz: bytes) -> np.ndarray:
    """Reload immutable little-endian float32 XYZ triples."""

    if not isinstance(serialized_xyz, bytes):
        raise TypeError("serialized XYZ must be immutable bytes")
    row_bytes = 3 * np.dtype("<f4").itemsize
    if len(serialized_xyz) % row_bytes:
        raise SupportSerializationError("serialized XYZ byte count is not row-aligned")
    return np.frombuffer(serialized_xyz, dtype="<f4").reshape(-1, 3)


def _exact_squared_distance_below_minimum(
    left: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> bool:
    squared_numerators: list[int] = []
    squared_denominators: list[int] = []
    for (left_numerator, left_denominator), (
        right_numerator,
        right_denominator,
    ) in zip(left, right, strict=True):
        difference_numerator = (
            left_numerator * right_denominator
            - right_numerator * left_denominator
        )
        difference_denominator = left_denominator * right_denominator
        squared_numerators.append(difference_numerator * difference_numerator)
        squared_denominators.append(
            difference_denominator * difference_denominator
        )
    common_denominator = (
        squared_denominators[0]
        * squared_denominators[1]
        * squared_denominators[2]
    )
    squared_numerator = sum(
        squared_numerators[axis]
        * squared_denominators[(axis + 1) % 3]
        * squared_denominators[(axis + 2) % 3]
        for axis in range(3)
    )
    return (
        squared_numerator * MINIMUM_DISTANCE_SQUARED_DENOMINATOR
        < common_denominator * MINIMUM_DISTANCE_SQUARED_NUMERATOR
    )


def _builder_self_check_xyz_array_spacing(
    xyz: np.ndarray,
    *,
    input_sha256: str,
) -> BuilderSpacingSelfCheck:
    if xyz.dtype != np.dtype("<f4") or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise SupportSerializationError("spacing verifier requires little-endian XYZ")
    if not np.isfinite(xyz).all():
        raise SupportSerializationError("spacing verifier found non-finite XYZ")

    exact_rows: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = []
    cells: list[tuple[int, int, int]] = []
    for row in xyz:
        ratios = tuple(float(value).as_integer_ratio() for value in row)
        exact_rows.append(ratios)  # type: ignore[arg-type]
        cell_values = tuple(
            (GRID_SCALE_PER_M * numerator) // denominator
            for numerator, denominator in ratios
        )
        if any(
            value < np.iinfo(np.int64).min or value > np.iinfo(np.int64).max
            for value in cell_values
        ):
            raise SupportSerializationError("spacing hash cell does not fit int64")
        cells.append(cell_values)  # type: ignore[arg-type]

    prior_by_cell: dict[tuple[int, int, int], list[int]] = {}
    checked_pairs = 0
    for row_index, cell in enumerate(cells):
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    prior_rows = prior_by_cell.get(
                        (
                            cell[0] + delta_x,
                            cell[1] + delta_y,
                            cell[2] + delta_z,
                        ),
                        (),
                    )
                    for prior_index in prior_rows:
                        checked_pairs += 1
                        if _exact_squared_distance_below_minimum(
                            exact_rows[prior_index],
                            exact_rows[row_index],
                        ):
                            raise SpacingViolationError(prior_index, row_index)
        prior_by_cell.setdefault(cell, []).append(row_index)
    return BuilderSpacingSelfCheck(
        input_sha256=input_sha256,
        point_count=int(xyz.shape[0]),
        checked_neighbor_pair_count=checked_pairs,
    )


def builder_self_check_serialized_xyz_spacing(
    serialized_xyz: bytes,
    *,
    expected_sha256: str | None = None,
) -> BuilderSpacingSelfCheck:
    """Run a non-authoritative exact self-check on serialized XYZ bytes."""

    if not isinstance(serialized_xyz, bytes):
        raise TypeError("serialized XYZ must be immutable bytes")
    observed_sha256 = hashlib.sha256(serialized_xyz).hexdigest()
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ValueError("expected XYZ SHA-256 is malformed")
        if observed_sha256 != expected_sha256:
            raise SupportSerializationError("serialized XYZ SHA-256 mismatch")
    xyz = deserialize_xyz_float32(serialized_xyz)
    return _builder_self_check_xyz_array_spacing(
        xyz,
        input_sha256=observed_sha256,
    )


def builder_self_check_serialized_support_spacing(
    serialized: bytes,
    *,
    expected_sha256: str | None = None,
) -> BuilderSpacingSelfCheck:
    """Run the non-authoritative builder spacing self-check after reload."""

    support = deserialize_packed_support(
        serialized,
        expected_sha256=expected_sha256,
    )
    exact_cells: list[tuple[int, int, int]] = []
    for row in support.xyz_m:
        ratios = tuple(float(value).as_integer_ratio() for value in row)
        exact_cells.append(
            tuple(
                (GRID_SCALE_PER_M * numerator) // denominator
                for numerator, denominator in ratios
            )
        )
    serialized_cells = [
        tuple(int(value) for value in row) for row in support.cell_xyz
    ]
    if exact_cells != serialized_cells:
        raise SupportSerializationError(
            "serialized support cells differ from builder exact self-check"
        )
    return _builder_self_check_xyz_array_spacing(
        support.xyz_m,
        input_sha256=support.digest_sha256,
    )


def commit_packed_support(
    serialized: bytes,
    *,
    expected_sha256: str | None = None,
) -> SupportCommitment:
    """Rehash bytes and run non-authoritative builder-side self-checks."""

    support = deserialize_packed_support(
        serialized,
        expected_sha256=expected_sha256,
    )
    spacing = builder_self_check_serialized_support_spacing(
        serialized,
        expected_sha256=support.digest_sha256,
    )
    return SupportCommitment(
        protocol_sha256=PROTOCOL_SHA256,
        protocol_freeze_commit=PROTOCOL_FREEZE_COMMIT,
        support_sha256=support.digest_sha256,
        candidate_field_sha256=support.candidate_field_sha256,
        candidate_count=support.candidate_count,
        support_count=support.support_count,
        selected_color_id=support.selected_color_id,
        color_cardinalities=support.color_cardinalities,
        serialized_size_bytes=len(serialized),
        capacity_status=support.capacity_status,
        builder_spacing_self_check_passed=spacing.passed,
        builder_spacing_self_check_checked_neighbor_pair_count=(
            spacing.checked_neighbor_pair_count
        ),
    )
