"""Deterministic target-conditioned fitting for the frozen STDA-F0 gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
from typing import Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree

from cube_dense.kradar import cartesian_to_polar
from eval.vrh_f0_support import canonicalize_azimuth


BASE_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
FROZEN_PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
PROTOCOL_FREEZE_COMMIT = BASE_FREEZE_COMMIT
PROTOCOL_SHA256 = FROZEN_PROTOCOL_SHA256

DEMAND_SLOT_COUNT = 10_000
GRAPH_NEIGHBOR_COUNT = 256
GRAPH_EDGE_COUNT = DEMAND_SLOT_COUNT * GRAPH_NEIGHBOR_COUNT
EDGE_COST_EXACT_LIMIT = 1 << 53
OBJECTIVE_INT64_LIMIT = 1 << 63
CANONICAL_BUNDLE_DOMAIN = b"stda_f0_canonical_array_bundle_v1\0"

ATOM_SCHEMA = (
    ("canonical_atom_id", "<i8"),
    ("atom_xyz", "<f8"),
    ("polar_rae", "<f8"),
    ("aggregate_weight", "<f8"),
    ("cdf", "<f8"),
)
DEMAND_SCHEMA = (
    ("slot_id", "<i8"),
    ("canonical_atom_id", "<i8"),
    ("slot_xyz", "<f8"),
)
SUPPORT_SCHEMA = (
    ("stable_candidate_id", "<i8"),
    ("support_xyz", "<f4"),
)
GRAPH_SCHEMA = (
    ("indptr", "<i8"),
    ("indices", "<i4"),
    ("data", "<i8"),
    ("squared_distance_m2", "<f8"),
    ("distance_m", "<f8"),
)
POINTWISE_SCHEMA = (
    ("nearest_atom_id", "<i8"),
    ("squared_distance_m2", "<f8"),
    ("distance_m", "<f8"),
)


@dataclass(frozen=True)
class TargetLoad:
    """One immutable, support-gated read of the approved target cache array."""

    target_xyz_confidence: np.ndarray
    cache_sha256: str
    target_tensor_sha256: str
    support_commit_sha256: str
    cache_arrays_read: tuple[str, ...] = ("target_xyz_confidence",)


def _validate_support_commit_sha256(support_commit_sha256: str) -> None:
    if (
        len(support_commit_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in support_commit_sha256
        )
    ):
        raise ValueError("STDA target loading requires a valid support commitment")


def _read_regular_file_snapshot(path: Path) -> bytes:
    """Read one regular file through one no-follow descriptor."""

    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("STDA target cache must be one regular non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("STDA target cache must be one regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_mode != after.st_mode
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or len(payload) != before.st_size
        ):
            raise ValueError("STDA target cache changed during its sole approved read")
        return payload
    finally:
        os.close(descriptor)


def load_target_xyz_confidence_bytes(
    payload: bytes,
    *,
    support_commit_sha256: str,
) -> TargetLoad:
    """Parse only target_xyz_confidence from one already-bound cache payload."""

    _validate_support_commit_sha256(support_commit_sha256)
    if not isinstance(payload, bytes):
        raise TypeError("STDA target cache payload must be immutable bytes")
    with np.load(io.BytesIO(payload), allow_pickle=False) as cache:
        if "target_xyz_confidence" not in cache:
            raise ValueError("STDA target cache lacks target_xyz_confidence")
        target = np.array(
            cache["target_xyz_confidence"],
            dtype="<f4",
            order="C",
            copy=True,
        )
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError("STDA target cache array must have shape (N,4)")
    if not np.isfinite(target).all() or np.any(target[:, 3] < 0.0):
        raise ValueError("STDA target cache contains invalid XYZ/confidence")
    immutable = _little_endian_array(target, "<f4")
    return TargetLoad(
        target_xyz_confidence=immutable,
        cache_sha256=hashlib.sha256(payload).hexdigest(),
        target_tensor_sha256=hashlib.sha256(immutable.tobytes(order="C")).hexdigest(),
        support_commit_sha256=support_commit_sha256,
    )


def load_target_xyz_confidence(
    cache_path: Path,
    *,
    support_commit_sha256: str,
) -> TargetLoad:
    """Compatibility path loader using the same descriptor-bound byte parser."""

    _validate_support_commit_sha256(support_commit_sha256)
    payload = _read_regular_file_snapshot(cache_path)
    return load_target_xyz_confidence_bytes(
        payload,
        support_commit_sha256=support_commit_sha256,
    )


def _little_endian_array(values: np.ndarray, dtype: str) -> np.ndarray:
    array = np.array(values, dtype=np.dtype(dtype), order="C", copy=True)
    array.flags.writeable = False
    return array


def _require_canonical_array(
    name: str,
    array: np.ndarray,
    dtype: str,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"{name} dtype {array.dtype.str} does not equal {dtype}")
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not equal {shape}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if array.flags.writeable:
        raise ValueError(f"{name} must be immutable")


def _canonical_fields(
    schema: tuple[tuple[str, str], ...],
    arrays: Mapping[str, np.ndarray],
) -> tuple[tuple[str, str, np.ndarray], ...]:
    expected_names = tuple(name for name, _ in schema)
    if tuple(arrays) != expected_names:
        raise ValueError(
            f"canonical fields {tuple(arrays)} do not match {expected_names}"
        )
    fields = []
    for name, dtype in schema:
        fields.append((name, dtype, _little_endian_array(arrays[name], dtype)))
    return tuple(fields)


def _canonical_header(
    kind: str,
    fields: Iterable[tuple[str, str, np.ndarray]],
    *,
    extra_header: Mapping[str, object] | None = None,
) -> bytes:
    field_list = tuple(fields)
    document = {
        "base_freeze_commit": BASE_FREEZE_COMMIT,
        "fields": [
            {
                "dtype": dtype,
                "name": name,
                "nbytes": int(array.nbytes),
                "shape": [int(value) for value in array.shape],
            }
            for name, dtype, array in field_list
        ],
        "kind": kind,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        **dict(extra_header or {}),
    }
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


def _canonical_bytes(
    kind: str,
    schema: tuple[tuple[str, str], ...],
    arrays: Mapping[str, np.ndarray],
    *,
    extra_header: Mapping[str, object] | None = None,
) -> bytes:
    fields = _canonical_fields(schema, arrays)
    chunks = [
        CANONICAL_BUNDLE_DOMAIN,
        _canonical_header(kind, fields, extra_header=extra_header),
    ]
    chunks.extend(array.tobytes(order="C") for _, _, array in fields)
    return b"".join(chunks)


def _canonical_digest(
    kind: str,
    schema: tuple[tuple[str, str], ...],
    arrays: Mapping[str, np.ndarray],
    *,
    extra_header: Mapping[str, object] | None = None,
) -> str:
    fields = _canonical_fields(schema, arrays)
    digest = hashlib.sha256(CANONICAL_BUNDLE_DOMAIN)
    digest.update(_canonical_header(kind, fields, extra_header=extra_header))
    for _, _, array in fields:
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _normalize_float32_signed_zero(values: np.ndarray) -> np.ndarray:
    normalized = np.array(values, dtype="<f4", order="C", copy=True)
    normalized[normalized == np.float32(0.0)] = np.float32(0.0)
    return normalized


def _float32_from_bits(bits: int) -> float:
    encoded = int(bits).to_bytes(4, byteorder="little", signed=False)
    return float(np.frombuffer(encoded, dtype="<f4", count=1)[0])


def _validate_digest(
    observed: str,
    kind: str,
    schema: tuple[tuple[str, str], ...],
    arrays: Mapping[str, np.ndarray],
    *,
    extra_header: Mapping[str, object] | None = None,
) -> None:
    expected = _canonical_digest(
        kind,
        schema,
        arrays,
        extra_header=extra_header,
    )
    if observed != expected:
        raise ValueError(f"{kind} canonical bytes changed after construction")


@dataclass(frozen=True)
class CanonicalTargetAtoms:
    canonical_atom_id: np.ndarray
    xyz_float32: np.ndarray
    xyz_float64: np.ndarray
    polar_rae: np.ndarray
    aggregate_weight: np.ndarray
    cdf: np.ndarray
    zero_confidence_row_count: int
    digest_sha256: str

    @property
    def count(self) -> int:
        return int(self.canonical_atom_id.size)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "canonical_atom_id": self.canonical_atom_id,
            "atom_xyz": self.xyz_float64,
            "polar_rae": self.polar_rae,
            "aggregate_weight": self.aggregate_weight,
            "cdf": self.cdf,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes("stda_f0_target_atoms_v1", ATOM_SCHEMA, self.arrays())

    def validate(self) -> None:
        expected = (self.count,)
        if self.count == 0:
            raise ValueError("STDA target atoms must be non-empty")
        _require_canonical_array(
            "canonical_atom_id", self.canonical_atom_id, "<i8", expected
        )
        _require_canonical_array(
            "atom_xyz_float32", self.xyz_float32, "<f4", (self.count, 3)
        )
        _require_canonical_array(
            "atom_xyz_float64", self.xyz_float64, "<f8", (self.count, 3)
        )
        _require_canonical_array(
            "atom_polar_rae", self.polar_rae, "<f8", (self.count, 3)
        )
        _require_canonical_array(
            "aggregate_weight", self.aggregate_weight, "<f8", expected
        )
        _require_canonical_array("atom_cdf", self.cdf, "<f8", expected)
        if not np.array_equal(
            self.canonical_atom_id,
            np.arange(self.count, dtype="<i8"),
        ):
            raise ValueError("STDA canonical atom IDs must be contiguous from zero")
        if not np.isfinite(self.xyz_float64).all():
            raise ValueError("STDA atom XYZ must be finite")
        zero = self.xyz_float32 == np.float32(0.0)
        if np.any(np.signbit(self.xyz_float32[zero])):
            raise ValueError("STDA atom XYZ contains a negative signed zero")
        promoted = np.ascontiguousarray(self.xyz_float32, dtype="<f8")
        if not np.array_equal(promoted.view("<u8"), self.xyz_float64.view("<u8")):
            raise ValueError("STDA float64 atom XYZ is not an exact float32 promotion")
        if not np.isfinite(self.polar_rae).all():
            raise ValueError("STDA atom RAE must be finite")
        if not np.isfinite(self.aggregate_weight).all() or np.any(
            self.aggregate_weight <= 0.0
        ):
            raise ValueError("STDA aggregate atom weights must be finite and positive")
        if not np.isfinite(self.cdf).all() or np.any(np.diff(self.cdf) < 0.0):
            raise ValueError("STDA atom CDF must be finite and nondecreasing")
        if self.cdf[-1] != 1.0:
            raise ValueError("STDA atom CDF endpoint must be bit-exact one")
        _validate_digest(
            self.digest_sha256,
            "stda_f0_target_atoms_v1",
            ATOM_SCHEMA,
            self.arrays(),
        )


@dataclass(frozen=True)
class DemandSlots:
    slot_id: np.ndarray
    canonical_atom_id: np.ndarray
    slot_xyz: np.ndarray
    atom_digest_sha256: str
    digest_sha256: str

    @property
    def count(self) -> int:
        return int(self.slot_id.size)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "slot_id": self.slot_id,
            "canonical_atom_id": self.canonical_atom_id,
            "slot_xyz": self.slot_xyz,
        }

    def _extra_header(self) -> dict[str, object]:
        return {"atom_digest_sha256": self.atom_digest_sha256}

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(
            "stda_f0_demand_slots_v1",
            DEMAND_SCHEMA,
            self.arrays(),
            extra_header=self._extra_header(),
        )

    def validate(self) -> None:
        if self.count != DEMAND_SLOT_COUNT:
            raise ValueError("STDA demand must contain exactly 10,000 slots")
        _require_canonical_array(
            "demand_slot_id", self.slot_id, "<i8", (DEMAND_SLOT_COUNT,)
        )
        _require_canonical_array(
            "demand_atom_id",
            self.canonical_atom_id,
            "<i8",
            (DEMAND_SLOT_COUNT,),
        )
        _require_canonical_array(
            "demand_slot_xyz", self.slot_xyz, "<f8", (DEMAND_SLOT_COUNT, 3)
        )
        if not np.array_equal(
            self.slot_id,
            np.arange(DEMAND_SLOT_COUNT, dtype="<i8"),
        ):
            raise ValueError("STDA slot IDs must be contiguous from zero")
        if self.canonical_atom_id.shape != (DEMAND_SLOT_COUNT,):
            raise ValueError("STDA demand atom IDs have invalid shape")
        if self.slot_xyz.shape != (DEMAND_SLOT_COUNT, 3):
            raise ValueError("STDA demand XYZ has invalid shape")
        if np.any(self.canonical_atom_id < 0) or not np.isfinite(
            self.slot_xyz
        ).all():
            raise ValueError("STDA demand contains invalid atom IDs or XYZ")
        _validate_digest(
            self.digest_sha256,
            "stda_f0_demand_slots_v1",
            DEMAND_SCHEMA,
            self.arrays(),
            extra_header=self._extra_header(),
        )


@dataclass(frozen=True)
class CanonicalSupport:
    stable_candidate_id: np.ndarray
    xyz_float32: np.ndarray
    xyz_float64: np.ndarray
    digest_sha256: str

    @property
    def count(self) -> int:
        return int(self.stable_candidate_id.size)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "stable_candidate_id": self.stable_candidate_id,
            "support_xyz": self.xyz_float32,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(
            "stda_f0_canonical_support_view_v1",
            SUPPORT_SCHEMA,
            self.arrays(),
        )

    def validate(self) -> None:
        if self.count < GRAPH_NEIGHBOR_COUNT:
            raise ValueError("STDA support has fewer than 256 points")
        _require_canonical_array(
            "support_stable_candidate_id",
            self.stable_candidate_id,
            "<i8",
            (self.count,),
        )
        _require_canonical_array(
            "support_xyz_float32", self.xyz_float32, "<f4", (self.count, 3)
        )
        _require_canonical_array(
            "support_xyz_float64", self.xyz_float64, "<f8", (self.count, 3)
        )
        if np.any(self.stable_candidate_id < 0) or np.any(
            np.diff(self.stable_candidate_id) <= 0
        ):
            raise ValueError("STDA stable candidate IDs must increase strictly")
        if not np.isfinite(self.xyz_float32).all():
            raise ValueError("STDA support XYZ must be finite")
        promoted = np.ascontiguousarray(self.xyz_float32, dtype="<f8")
        if not np.array_equal(promoted.view("<u8"), self.xyz_float64.view("<u8")):
            raise ValueError("STDA float64 support XYZ is not an exact float32 promotion")
        _validate_digest(
            self.digest_sha256,
            "stda_f0_canonical_support_view_v1",
            SUPPORT_SCHEMA,
            self.arrays(),
        )


@dataclass(frozen=True)
class FrozenNeighborRow:
    support_rank: np.ndarray
    stable_candidate_id: np.ndarray
    squared_distance_m2: np.ndarray
    distance_m: np.ndarray


@dataclass(frozen=True)
class SparseAssignmentGraph:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    squared_distance_m2: np.ndarray
    distance_m: np.ndarray
    shape: tuple[int, int]
    support_digest_sha256: str
    demand_digest_sha256: str
    digest_sha256: str

    @property
    def edge_count(self) -> int:
        return int(self.indices.size)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "indptr": self.indptr,
            "indices": self.indices,
            "data": self.data,
            "squared_distance_m2": self.squared_distance_m2,
            "distance_m": self.distance_m,
        }

    def _extra_header(self) -> dict[str, object]:
        return {
            "demand_digest_sha256": self.demand_digest_sha256,
            "shape": [int(value) for value in self.shape],
            "support_digest_sha256": self.support_digest_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(
            "stda_f0_sparse_assignment_graph_v1",
            GRAPH_SCHEMA,
            self.arrays(),
            extra_header=self._extra_header(),
        )

    def validate(self) -> None:
        if self.shape[0] != DEMAND_SLOT_COUNT:
            raise ValueError("STDA graph must have exactly 10,000 rows")
        if self.shape[1] < GRAPH_NEIGHBOR_COUNT:
            raise ValueError("STDA graph support cardinality is too small")
        _require_canonical_array(
            "graph_indptr", self.indptr, "<i8", (DEMAND_SLOT_COUNT + 1,)
        )
        if self.edge_count != GRAPH_EDGE_COUNT:
            raise ValueError("STDA graph must have exactly 2,560,000 edges")
        if any(
            array.shape != (GRAPH_EDGE_COUNT,)
            for array in (
                self.indices,
                self.data,
                self.squared_distance_m2,
                self.distance_m,
            )
        ):
            raise ValueError("STDA graph edge arrays have invalid shape")
        _require_canonical_array(
            "graph_indices", self.indices, "<i4", (GRAPH_EDGE_COUNT,)
        )
        _require_canonical_array(
            "graph_data", self.data, "<i8", (GRAPH_EDGE_COUNT,)
        )
        _require_canonical_array(
            "graph_squared_distance_m2",
            self.squared_distance_m2,
            "<f8",
            (GRAPH_EDGE_COUNT,),
        )
        _require_canonical_array(
            "graph_distance_m", self.distance_m, "<f8", (GRAPH_EDGE_COUNT,)
        )
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != GRAPH_EDGE_COUNT:
            raise ValueError("STDA graph indptr endpoints are invalid")
        if np.any(np.diff(self.indptr) != GRAPH_NEIGHBOR_COUNT):
            raise ValueError("STDA graph rows must contain exactly 256 edges")
        if np.any(self.indices < 0) or np.any(self.indices >= self.shape[1]):
            raise ValueError("STDA graph contains an invalid support column")
        for row in range(DEMAND_SLOT_COUNT):
            start = int(self.indptr[row])
            stop = int(self.indptr[row + 1])
            if np.any(np.diff(self.indices[start:stop]) <= 0):
                raise ValueError("STDA CSR columns must increase within each row")
        if np.any(self.data <= 0) or np.any(self.data >= EDGE_COST_EXACT_LIMIT):
            raise ValueError("STDA graph contains an invalid integer edge cost")
        if not np.isfinite(self.squared_distance_m2).all() or np.any(
            self.squared_distance_m2 < 0.0
        ):
            raise ValueError("STDA graph squared distances are invalid")
        if not np.isfinite(self.distance_m).all() or np.any(self.distance_m < 0.0):
            raise ValueError("STDA graph distances are invalid")
        row_maxima = []
        for row in range(DEMAND_SLOT_COUNT):
            start = int(self.indptr[row])
            stop = int(self.indptr[row + 1])
            row_maxima.append(max(int(value) for value in self.data[start:stop]))
        validate_full_assignment_objective_bound(row_maxima)
        _validate_digest(
            self.digest_sha256,
            "stda_f0_sparse_assignment_graph_v1",
            GRAPH_SCHEMA,
            self.arrays(),
            extra_header=self._extra_header(),
        )


@dataclass(frozen=True)
class PackedPointwiseSidecar:
    stable_candidate_id: np.ndarray
    nearest_atom_id: np.ndarray
    squared_distance_m2: np.ndarray
    distance_m: np.ndarray
    support_digest_sha256: str
    atom_digest_sha256: str
    digest_sha256: str

    @property
    def count(self) -> int:
        return int(self.nearest_atom_id.size)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "nearest_atom_id": self.nearest_atom_id,
            "squared_distance_m2": self.squared_distance_m2,
            "distance_m": self.distance_m,
        }

    def _extra_header(self) -> dict[str, object]:
        return {
            "atom_digest_sha256": self.atom_digest_sha256,
            "support_digest_sha256": self.support_digest_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(
            "stda_f0_packed_pointwise_sidecar_v1",
            POINTWISE_SCHEMA,
            self.arrays(),
            extra_header=self._extra_header(),
        )

    def selected_support_ranks(self) -> np.ndarray:
        if self.count < DEMAND_SLOT_COUNT:
            raise ValueError("packed pointwise support has fewer than 10,000 points")
        order = sorted(
            range(self.count),
            key=lambda row: (
                float(self.squared_distance_m2[row]),
                int(self.stable_candidate_id[row]),
            ),
        )
        return _little_endian_array(order[:DEMAND_SLOT_COUNT], "<i8")

    def validate(self) -> None:
        if self.count != self.stable_candidate_id.size:
            raise ValueError("STDA pointwise stable IDs have invalid shape")
        _require_canonical_array(
            "pointwise_stable_candidate_id",
            self.stable_candidate_id,
            "<i8",
            (self.count,),
        )
        _require_canonical_array(
            "pointwise_nearest_atom_id",
            self.nearest_atom_id,
            "<i8",
            (self.count,),
        )
        _require_canonical_array(
            "pointwise_squared_distance_m2",
            self.squared_distance_m2,
            "<f8",
            (self.count,),
        )
        _require_canonical_array(
            "pointwise_distance_m", self.distance_m, "<f8", (self.count,)
        )
        if np.any(np.diff(self.stable_candidate_id) <= 0):
            raise ValueError("STDA pointwise stable IDs must increase strictly")
        if np.any(self.nearest_atom_id < 0):
            raise ValueError("STDA pointwise nearest atom IDs must be nonnegative")
        if not np.isfinite(self.squared_distance_m2).all() or np.any(
            self.squared_distance_m2 < 0.0
        ):
            raise ValueError("STDA pointwise squared distances are invalid")
        if not np.isfinite(self.distance_m).all() or np.any(self.distance_m < 0.0):
            raise ValueError("STDA pointwise distances are invalid")
        _validate_digest(
            self.digest_sha256,
            "stda_f0_packed_pointwise_sidecar_v1",
            POINTWISE_SCHEMA,
            self.arrays(),
            extra_header=self._extra_header(),
        )


def canonicalize_target_atoms(
    target_xyz_confidence: np.ndarray,
) -> CanonicalTargetAtoms:
    """Canonicalize explicit float32 XYZ-confidence input into weighted atoms."""

    target = _normalize_float32_signed_zero(target_xyz_confidence)
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError("STDA target input must be non-empty with shape (N,4)")
    if not np.isfinite(target).all():
        raise ValueError("STDA target input must be finite after float32 conversion")
    confidence = target[:, 3]
    if np.any(confidence < 0.0):
        raise ValueError("STDA target confidence must be nonnegative")

    positive = confidence > 0.0
    zero_count = int((~positive).sum())
    if not bool(positive.any()):
        raise ValueError("STDA target input has no positive-confidence rows")

    grouped_confidence_bits: dict[bytes, list[int]] = {}
    positive_target = np.ascontiguousarray(target[positive], dtype="<f4")
    confidence_bits = np.ascontiguousarray(
        positive_target[:, 3],
        dtype="<f4",
    ).view("<u4")
    for row in range(positive_target.shape[0]):
        xyz_key = positive_target[row, :3].tobytes(order="C")
        grouped_confidence_bits.setdefault(xyz_key, []).append(
            int(confidence_bits[row])
        )

    grouped_rows: list[tuple[bytes, np.ndarray, float]] = []
    for xyz_key, bit_values in grouped_confidence_bits.items():
        weight = math.fsum(
            _float32_from_bits(bits) for bits in sorted(bit_values)
        )
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("STDA aggregate atom weight must be finite and positive")
        xyz = np.frombuffer(xyz_key, dtype="<f4", count=3).copy()
        grouped_rows.append((xyz_key, xyz, weight))

    grouped_xyz_float32 = np.ascontiguousarray(
        np.stack([row[1] for row in grouped_rows], axis=0),
        dtype="<f4",
    )
    grouped_xyz_float64 = np.ascontiguousarray(grouped_xyz_float32, dtype="<f8")
    polar_rae = np.ascontiguousarray(
        cartesian_to_polar(grouped_xyz_float64),
        dtype="<f8",
    )
    polar_rae[:, 1] = canonicalize_azimuth(polar_rae[:, 1])
    if not np.isfinite(polar_rae).all():
        raise ValueError("STDA target RAE conversion produced a non-finite value")

    order = sorted(
        range(len(grouped_rows)),
        key=lambda row: (
            float(polar_rae[row, 0]),
            float(polar_rae[row, 1]),
            float(polar_rae[row, 2]),
            float(grouped_xyz_float64[row, 0]),
            float(grouped_xyz_float64[row, 1]),
            float(grouped_xyz_float64[row, 2]),
            grouped_rows[row][0],
        ),
    )
    xyz_float32 = _little_endian_array(grouped_xyz_float32[order], "<f4")
    xyz_float64 = _little_endian_array(grouped_xyz_float64[order], "<f8")
    ordered_polar = _little_endian_array(polar_rae[order], "<f8")
    ordered_weights_list = [float(grouped_rows[row][2]) for row in order]
    aggregate_weight = _little_endian_array(ordered_weights_list, "<f8")
    total = math.fsum(ordered_weights_list)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("STDA total target confidence must be finite and positive")
    cdf_values = [
        math.fsum(ordered_weights_list[: stop + 1]) / total
        for stop in range(len(ordered_weights_list))
    ]
    cdf = _little_endian_array(cdf_values, "<f8")
    if not np.isfinite(cdf).all() or np.any(np.diff(cdf) < 0.0):
        raise ValueError("STDA target CDF must be finite and nondecreasing")
    if cdf[-1] != 1.0:
        raise ValueError("STDA target CDF endpoint is not bit-exact one")

    atom_id = _little_endian_array(np.arange(len(order)), "<i8")
    arrays = {
        "canonical_atom_id": atom_id,
        "atom_xyz": xyz_float64,
        "polar_rae": ordered_polar,
        "aggregate_weight": aggregate_weight,
        "cdf": cdf,
    }
    digest = _canonical_digest("stda_f0_target_atoms_v1", ATOM_SCHEMA, arrays)
    atoms = CanonicalTargetAtoms(
        canonical_atom_id=atom_id,
        xyz_float32=xyz_float32,
        xyz_float64=xyz_float64,
        polar_rae=ordered_polar,
        aggregate_weight=aggregate_weight,
        cdf=cdf,
        zero_confidence_row_count=zero_count,
        digest_sha256=digest,
    )
    atoms.validate()
    return atoms


def build_midpoint_demand_slots(atoms: CanonicalTargetAtoms) -> DemandSlots:
    """Discretize canonical target mass into the frozen 10,000 slots."""

    atoms.validate()
    midpoint = np.fromiter(
        ((2 * slot + 1) / 20_000 for slot in range(DEMAND_SLOT_COUNT)),
        dtype="<f8",
        count=DEMAND_SLOT_COUNT,
    )
    atom_index = np.searchsorted(
        np.asarray(atoms.cdf, dtype="<f8"),
        midpoint,
        side="right",
    )
    if np.any(atom_index < 0) or np.any(atom_index >= atoms.count):
        raise ValueError("STDA demand CDF search produced an invalid atom index")
    slot_id = _little_endian_array(np.arange(DEMAND_SLOT_COUNT), "<i8")
    atom_id = _little_endian_array(atoms.canonical_atom_id[atom_index], "<i8")
    slot_xyz = _little_endian_array(atoms.xyz_float64[atom_index], "<f8")
    arrays = {
        "slot_id": slot_id,
        "canonical_atom_id": atom_id,
        "slot_xyz": slot_xyz,
    }
    extra = {"atom_digest_sha256": atoms.digest_sha256}
    digest = _canonical_digest(
        "stda_f0_demand_slots_v1",
        DEMAND_SCHEMA,
        arrays,
        extra_header=extra,
    )
    demand = DemandSlots(
        slot_id=slot_id,
        canonical_atom_id=atom_id,
        slot_xyz=slot_xyz,
        atom_digest_sha256=atoms.digest_sha256,
        digest_sha256=digest,
    )
    demand.validate()
    return demand


def canonicalize_support(
    support_xyz: np.ndarray,
    stable_candidate_id: np.ndarray,
) -> CanonicalSupport:
    """Order an immutable packed support by original stable candidate ID."""

    xyz = np.array(support_xyz, dtype="<f4", order="C", copy=True)
    raw_id = np.asarray(stable_candidate_id)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < GRAPH_NEIGHBOR_COUNT:
        raise ValueError("STDA support must have shape (N,3) with N>=256")
    if raw_id.ndim != 1 or raw_id.shape[0] != xyz.shape[0]:
        raise ValueError("STDA stable candidate IDs must align with support XYZ")
    if raw_id.dtype.kind not in "iu":
        raise ValueError("STDA stable candidate IDs must have integer dtype")
    if raw_id.dtype.kind == "u" and np.any(raw_id > np.iinfo(np.int64).max):
        raise ValueError("STDA stable candidate ID exceeds signed int64")
    candidate_id = np.asarray(raw_id, dtype="<i8")
    if np.any(candidate_id < 0):
        raise ValueError("STDA stable candidate IDs must be nonnegative")
    if np.unique(candidate_id).size != candidate_id.size:
        raise ValueError("STDA stable candidate IDs must be unique")
    if candidate_id.size > np.iinfo(np.int32).max:
        raise ValueError("STDA support cardinality exceeds CSR int32 columns")
    if not np.isfinite(xyz).all():
        raise ValueError("STDA support XYZ must be finite after float32 conversion")

    order = np.argsort(candidate_id, kind="stable")
    ordered_id = _little_endian_array(candidate_id[order], "<i8")
    ordered_xyz_float32 = _little_endian_array(xyz[order], "<f4")
    ordered_xyz_float64 = _little_endian_array(ordered_xyz_float32, "<f8")
    arrays = {
        "stable_candidate_id": ordered_id,
        "support_xyz": ordered_xyz_float32,
    }
    digest = _canonical_digest(
        "stda_f0_canonical_support_view_v1",
        SUPPORT_SCHEMA,
        arrays,
    )
    support = CanonicalSupport(
        stable_candidate_id=ordered_id,
        xyz_float32=ordered_xyz_float32,
        xyz_float64=ordered_xyz_float64,
        digest_sha256=digest,
    )
    support.validate()
    return support


def _support_tree(support: CanonicalSupport) -> cKDTree:
    return cKDTree(
        support.xyz_float64,
        leafsize=16,
        compact_nodes=True,
        balanced_tree=True,
        copy_data=True,
    )


def _exact_squared_distance(left: np.ndarray, right: np.ndarray) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    dz = float(left[2]) - float(right[2])
    return (dx * dx + dy * dy) + dz * dz


def _select_frozen_neighbors_from_tree(
    tree: cKDTree,
    support: CanonicalSupport,
    query_xyz: np.ndarray,
) -> FrozenNeighborRow:
    query = np.asarray(query_xyz, dtype="<f8")
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("STDA graph query must be one finite float64 XYZ row")
    provisional_distance, _ = tree.query(
        query,
        k=GRAPH_NEIGHBOR_COUNT,
        p=2,
        eps=0,
        workers=1,
    )
    boundary = float(np.asarray(provisional_distance)[GRAPH_NEIGHBOR_COUNT - 1])
    if not math.isfinite(boundary):
        raise ValueError("STDA provisional 256th-neighbor distance is non-finite")
    candidate_rows = tree.query_ball_point(
        query,
        r=math.nextafter(boundary, math.inf),
        p=2,
        eps=0,
        workers=1,
        return_sorted=True,
    )
    candidate_rank = [int(value) for value in candidate_rows]
    if len(candidate_rank) < GRAPH_NEIGHBOR_COUNT:
        raise ValueError("STDA KNN boundary expansion returned fewer than 256 rows")
    if len(set(candidate_rank)) != len(candidate_rank):
        raise ValueError("STDA KNN boundary expansion returned duplicate rows")

    resolved = []
    for rank in candidate_rank:
        squared_distance = _exact_squared_distance(
            support.xyz_float64[rank],
            query,
        )
        if not math.isfinite(squared_distance) or squared_distance < 0.0:
            raise ValueError("STDA exact edge squared distance is invalid")
        resolved.append(
            (
                squared_distance,
                int(support.stable_candidate_id[rank]),
                rank,
                math.sqrt(squared_distance),
            )
        )
    resolved.sort(key=lambda row: (row[0], row[1]))
    retained = resolved[:GRAPH_NEIGHBOR_COUNT]
    return FrozenNeighborRow(
        support_rank=_little_endian_array([row[2] for row in retained], "<i8"),
        stable_candidate_id=_little_endian_array(
            [row[1] for row in retained], "<i8"
        ),
        squared_distance_m2=_little_endian_array(
            [row[0] for row in retained], "<f8"
        ),
        distance_m=_little_endian_array([row[3] for row in retained], "<f8"),
    )


def select_frozen_neighbors(
    query_xyz: np.ndarray,
    support_xyz: np.ndarray,
    stable_candidate_id: np.ndarray,
) -> FrozenNeighborRow:
    """Return the exact frozen KNN row for one explicit query point."""

    support = canonicalize_support(support_xyz, stable_candidate_id)
    return _select_frozen_neighbors_from_tree(
        _support_tree(support),
        support,
        query_xyz,
    )


def integer_edge_cost(
    distance_m: float,
    *,
    support_cardinality: int,
    support_rank: int,
) -> int:
    """Compute and validate the frozen positive int64 composite edge cost."""

    if not isinstance(support_cardinality, (int, np.integer)):
        raise ValueError("STDA support cardinality must be an integer")
    if not isinstance(support_rank, (int, np.integer)):
        raise ValueError("STDA support rank must be an integer")
    support_cardinality = int(support_cardinality)
    support_rank = int(support_rank)
    distance = float(distance_m)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("STDA edge distance must be finite and nonnegative")
    if support_cardinality < GRAPH_NEIGHBOR_COUNT:
        raise ValueError("STDA support cardinality must be at least 256")
    if support_rank < 0 or support_rank >= support_cardinality:
        raise ValueError("STDA support rank is out of range")
    rounded = np.rint(np.float64(distance * 1.0e6))
    if not np.isfinite(rounded):
        raise OverflowError("STDA micrometer distance is non-finite")
    distance_um = int(rounded)
    edge_cost = distance_um * (support_cardinality + 1) + support_rank + 1
    if edge_cost <= 0 or edge_cost >= EDGE_COST_EXACT_LIMIT:
        raise OverflowError("STDA edge cost is outside the positive exact-int64 range")
    return edge_cost


def validate_full_assignment_objective_bound(
    row_maximum_costs: Iterable[int],
) -> int:
    """Return the Python-int row-max sum, rejecting a signed-int64 overflow."""

    maxima = tuple(int(value) for value in row_maximum_costs)
    if len(maxima) != DEMAND_SLOT_COUNT:
        raise ValueError("STDA objective bound requires 10,000 row maxima")
    if any(value <= 0 or value >= EDGE_COST_EXACT_LIMIT for value in maxima):
        raise ValueError("STDA objective bound received an invalid row maximum")
    total = sum(maxima)
    if total >= OBJECTIVE_INT64_LIMIT:
        raise OverflowError("STDA full-assignment objective can overflow int64")
    return total


def build_sparse_assignment_graph(
    demand: DemandSlots,
    support_xyz: np.ndarray,
    stable_candidate_id: np.ndarray,
) -> SparseAssignmentGraph:
    """Build the frozen 10,000 by support-cardinality graph and edge metadata."""

    demand.validate()
    support = canonicalize_support(support_xyz, stable_candidate_id)
    tree = _support_tree(support)
    indptr = _little_endian_array(
        np.arange(
            0,
            GRAPH_EDGE_COUNT + GRAPH_NEIGHBOR_COUNT,
            GRAPH_NEIGHBOR_COUNT,
        ),
        "<i8",
    )
    indices = np.empty(GRAPH_EDGE_COUNT, dtype="<i4")
    data = np.empty(GRAPH_EDGE_COUNT, dtype="<i8")
    squared_distance = np.empty(GRAPH_EDGE_COUNT, dtype="<f8")
    distance = np.empty(GRAPH_EDGE_COUNT, dtype="<f8")
    row_maxima: list[int] = []
    neighbor_cache: dict[int, tuple[bytes, FrozenNeighborRow]] = {}

    for slot in range(DEMAND_SLOT_COUNT):
        atom_id = int(demand.canonical_atom_id[slot])
        cached = neighbor_cache.get(atom_id)
        slot_xyz_bytes = demand.slot_xyz[slot].tobytes(order="C")
        if cached is None:
            neighbor_row = _select_frozen_neighbors_from_tree(
                tree,
                support,
                demand.slot_xyz[slot],
            )
            neighbor_cache[atom_id] = (slot_xyz_bytes, neighbor_row)
        else:
            cached_xyz_bytes, neighbor_row = cached
            if slot_xyz_bytes != cached_xyz_bytes:
                raise ValueError("STDA slots sharing an atom ID have different XYZ")

        order = np.argsort(neighbor_row.support_rank, kind="stable")
        start = slot * GRAPH_NEIGHBOR_COUNT
        stop = start + GRAPH_NEIGHBOR_COUNT
        row_rank = neighbor_row.support_rank[order]
        row_squared = neighbor_row.squared_distance_m2[order]
        row_distance = neighbor_row.distance_m[order]
        row_cost = [
            integer_edge_cost(
                float(row_distance[offset]),
                support_cardinality=support.count,
                support_rank=int(row_rank[offset]),
            )
            for offset in range(GRAPH_NEIGHBOR_COUNT)
        ]
        indices[start:stop] = row_rank.astype("<i4", copy=False)
        data[start:stop] = np.asarray(row_cost, dtype="<i8")
        squared_distance[start:stop] = row_squared
        distance[start:stop] = row_distance
        row_maxima.append(max(row_cost))

    validate_full_assignment_objective_bound(row_maxima)

    frozen_indices = _little_endian_array(indices, "<i4")
    frozen_data = _little_endian_array(data, "<i8")
    frozen_squared = _little_endian_array(squared_distance, "<f8")
    frozen_distance = _little_endian_array(distance, "<f8")
    arrays = {
        "indptr": indptr,
        "indices": frozen_indices,
        "data": frozen_data,
        "squared_distance_m2": frozen_squared,
        "distance_m": frozen_distance,
    }
    shape = (DEMAND_SLOT_COUNT, support.count)
    extra = {
        "demand_digest_sha256": demand.digest_sha256,
        "shape": [int(value) for value in shape],
        "support_digest_sha256": support.digest_sha256,
    }
    digest = _canonical_digest(
        "stda_f0_sparse_assignment_graph_v1",
        GRAPH_SCHEMA,
        arrays,
        extra_header=extra,
    )
    graph = SparseAssignmentGraph(
        indptr=indptr,
        indices=frozen_indices,
        data=frozen_data,
        squared_distance_m2=frozen_squared,
        distance_m=frozen_distance,
        shape=shape,
        support_digest_sha256=support.digest_sha256,
        demand_digest_sha256=demand.digest_sha256,
        digest_sha256=digest,
    )
    graph.validate()
    return graph


def _nearest_atom(
    tree: cKDTree,
    atoms: CanonicalTargetAtoms,
    query_xyz: np.ndarray,
) -> tuple[int, float, float]:
    provisional_distance, _ = tree.query(
        query_xyz,
        k=1,
        p=2,
        eps=0,
        workers=1,
    )
    boundary = float(provisional_distance)
    if not math.isfinite(boundary):
        raise ValueError("STDA pointwise nearest-target distance is non-finite")
    candidate_rows = tree.query_ball_point(
        query_xyz,
        r=math.nextafter(boundary, math.inf),
        p=2,
        eps=0,
        workers=1,
        return_sorted=True,
    )
    if not candidate_rows:
        raise ValueError("STDA pointwise boundary expansion returned no target")
    resolved = []
    for candidate in candidate_rows:
        atom_id = int(atoms.canonical_atom_id[int(candidate)])
        squared_distance = _exact_squared_distance(
            atoms.xyz_float64[int(candidate)],
            query_xyz,
        )
        if not math.isfinite(squared_distance) or squared_distance < 0.0:
            raise ValueError("STDA pointwise squared distance is invalid")
        resolved.append((squared_distance, atom_id))
    squared_distance, atom_id = min(resolved, key=lambda row: (row[0], row[1]))
    return atom_id, squared_distance, math.sqrt(squared_distance)


def build_packed_pointwise_sidecar(
    atoms: CanonicalTargetAtoms,
    support_xyz: np.ndarray,
    stable_candidate_id: np.ndarray,
) -> PackedPointwiseSidecar:
    """Build the immutable GT-conditioned packed-pointwise control sidecar."""

    atoms.validate()
    support = canonicalize_support(support_xyz, stable_candidate_id)
    tree = cKDTree(
        atoms.xyz_float64,
        leafsize=16,
        compact_nodes=True,
        balanced_tree=True,
        copy_data=True,
    )
    nearest_atom_id = np.empty(support.count, dtype="<i8")
    squared_distance = np.empty(support.count, dtype="<f8")
    distance = np.empty(support.count, dtype="<f8")
    for row in range(support.count):
        atom_id, distance_squared, distance_m = _nearest_atom(
            tree,
            atoms,
            support.xyz_float64[row],
        )
        nearest_atom_id[row] = atom_id
        squared_distance[row] = distance_squared
        distance[row] = distance_m

    frozen_atom_id = _little_endian_array(nearest_atom_id, "<i8")
    frozen_squared = _little_endian_array(squared_distance, "<f8")
    frozen_distance = _little_endian_array(distance, "<f8")
    arrays = {
        "nearest_atom_id": frozen_atom_id,
        "squared_distance_m2": frozen_squared,
        "distance_m": frozen_distance,
    }
    extra = {
        "atom_digest_sha256": atoms.digest_sha256,
        "support_digest_sha256": support.digest_sha256,
    }
    digest = _canonical_digest(
        "stda_f0_packed_pointwise_sidecar_v1",
        POINTWISE_SCHEMA,
        arrays,
        extra_header=extra,
    )
    sidecar = PackedPointwiseSidecar(
        stable_candidate_id=support.stable_candidate_id,
        nearest_atom_id=frozen_atom_id,
        squared_distance_m2=frozen_squared,
        distance_m=frozen_distance,
        support_digest_sha256=support.digest_sha256,
        atom_digest_sha256=atoms.digest_sha256,
        digest_sha256=digest,
    )
    sidecar.validate()
    return sidecar
