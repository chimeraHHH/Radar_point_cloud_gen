"""Target-independent support and binary schemas for the frozen VRH-F0 gate."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from cube_dense.kradar import KRadarAxes, polar_to_cartesian


R2_COUNT = 512
A2_COUNT = 214
E2_COUNT = 74
SUBRAY_COUNT = A2_COUNT * E2_COUNT
CELL_COUNT = R2_COUNT * SUBRAY_COUNT
NATURAL_RANGE_UPPER_M = 118.2685546875
MINIMUM_DISTANCE_M = 0.05
HAZARD_TAU_M = 0.20
HAZARD_EPSILON = 2.0**-24
_SUPPORT_COMMIT_SEAL = object()

MODEL_MARK_SCHEMA = (
    ("base_hazard", "<f8"),
    ("delta_r", "<f8"),
    ("delta_a", "<f8"),
    ("delta_e", "<f8"),
    ("opaque_distance_priority_key", "<i8"),
    ("confidence", "<f8"),
)


def _as_little_endian_contiguous(
    values: np.ndarray,
    dtype: str,
) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))


def _schema_header(kind: str, schema: Iterable[tuple[str, str]], **extra: object) -> bytes:
    document = {
        "kind": kind,
        "schema": [{"name": name, "dtype": dtype} for name, dtype in schema],
        **extra,
    }
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_columns_digest(
    *,
    kind: str,
    schema: tuple[tuple[str, str], ...],
    columns: Mapping[str, np.ndarray],
    extra_header: Mapping[str, object] | None = None,
) -> str:
    """Hash one schema header followed by little-endian column arrays."""

    expected = tuple(name for name, _ in schema)
    if tuple(columns) != expected:
        raise ValueError(f"{kind} columns differ from schema: {tuple(columns)}")
    lengths = {np.asarray(values).size for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"{kind} columns have inconsistent lengths: {lengths}")
    header = _schema_header(
        kind,
        schema,
        count=next(iter(lengths), 0),
        **dict(extra_header or {}),
    )
    digest = hashlib.sha256(header)
    for name, dtype in schema:
        digest.update(_as_little_endian_contiguous(columns[name], dtype).tobytes(order="C"))
    return digest.hexdigest()


def write_canonical_columns(
    path: Path,
    *,
    kind: str,
    schema: tuple[tuple[str, str], ...],
    columns: Mapping[str, np.ndarray],
    extra_header: Mapping[str, object] | None = None,
) -> str:
    """Write canonical binary columns atomically and return the file digest."""

    expected = tuple(name for name, _ in schema)
    if tuple(columns) != expected:
        raise ValueError(f"{kind} columns differ from schema: {tuple(columns)}")
    lengths = {np.asarray(values).size for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"{kind} columns have inconsistent lengths: {lengths}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    header = _schema_header(
        kind,
        schema,
        count=next(iter(lengths), 0),
        **dict(extra_header or {}),
    )
    with temporary.open("wb") as handle:
        handle.write(header)
        for name, dtype in schema:
            handle.write(
                _as_little_endian_contiguous(columns[name], dtype).tobytes(order="C")
            )
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitAxis:
    centers: np.ndarray
    edges: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lower_interior: np.ndarray
    upper_interior: np.ndarray

    @property
    def count(self) -> int:
        return int(self.centers.size)


@dataclass(frozen=True)
class VRHSupport:
    range_axis: SplitAxis
    azimuth_axis: SplitAxis
    elevation_axis: SplitAxis
    digest_sha256: str
    schema_header_sha256: str

    @property
    def shape_rae(self) -> tuple[int, int, int]:
        return (self.range_axis.count, self.azimuth_axis.count, self.elevation_axis.count)

    @property
    def ray_count(self) -> int:
        return self.azimuth_axis.count * self.elevation_axis.count

    @property
    def cell_count(self) -> int:
        return self.ray_count * self.range_axis.count


@dataclass(frozen=True, init=False)
class SupportCommit:
    digest_sha256: str
    cell_count: int
    ray_count: int
    _seal: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ModelMarks:
    base_hazard: np.ndarray
    delta_r: np.ndarray
    delta_a: np.ndarray
    delta_e: np.ndarray
    opaque_distance_priority_key: np.ndarray
    confidence: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.base_hazard.shape)  # type: ignore[return-value]

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "base_hazard": self.base_hazard.reshape(-1),
            "delta_r": self.delta_r.reshape(-1),
            "delta_a": self.delta_a.reshape(-1),
            "delta_e": self.delta_e.reshape(-1),
            "opaque_distance_priority_key": self.opaque_distance_priority_key.reshape(-1),
            "confidence": self.confidence.reshape(-1),
        }

    def validate(self, support: VRHSupport) -> None:
        expected = (support.ray_count, support.range_axis.count)
        for name, dtype in MODEL_MARK_SCHEMA:
            values = getattr(self, name)
            if values.shape != expected:
                raise ValueError(f"ModelMarks.{name} shape {values.shape} != {expected}")
            if np.dtype(values.dtype) != np.dtype(dtype):
                raise ValueError(f"ModelMarks.{name} dtype {values.dtype} != {dtype}")
        for name in ("base_hazard", "delta_r", "delta_a", "delta_e", "confidence"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"ModelMarks.{name} contains a non-finite value")
        if not np.all(
            (self.base_hazard >= HAZARD_EPSILON)
            & (self.base_hazard <= 1.0 - HAZARD_EPSILON)
        ):
            raise ValueError("ModelMarks hazard leaves the frozen finite interval")
        if np.any(self.confidence < 0.0):
            raise ValueError("ModelMarks confidence must be nonnegative")


def _native_edges(values: np.ndarray, *, range_axis: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("VRH native axis must be a finite vector")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("VRH native axis must be strictly increasing")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if range_axis:
        edges[0] = 0.0
    return edges


def _split_axis(values: np.ndarray, *, range_axis: bool) -> SplitAxis:
    native_edges = _native_edges(values, range_axis=range_axis)
    split_edges = np.empty(2 * values.size + 1, dtype=np.float64)
    split_edges[0::2] = native_edges
    split_edges[1::2] = 0.5 * (native_edges[:-1] + native_edges[1:])
    lower = split_edges[:-1].copy()
    upper = split_edges[1:].copy()
    centers = 0.5 * (lower + upper)
    lower_interior = np.nextafter(lower, upper)
    upper_interior = np.nextafter(upper, lower)
    if not np.all(lower_interior < upper_interior):
        raise ValueError("VRH split cell has no open float64 interior")
    if not np.all(np.nextafter(lower_interior, upper_interior) < upper_interior):
        raise ValueError("VRH split cell has fewer than three interior values")
    return SplitAxis(
        centers=np.ascontiguousarray(centers),
        edges=np.ascontiguousarray(split_edges),
        lower=np.ascontiguousarray(lower),
        upper=np.ascontiguousarray(upper),
        lower_interior=np.ascontiguousarray(lower_interior),
        upper_interior=np.ascontiguousarray(upper_interior),
    )


def _immutable_array(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _freeze_axis(axis: SplitAxis) -> SplitAxis:
    return SplitAxis(
        centers=_immutable_array(axis.centers),
        edges=_immutable_array(axis.edges),
        lower=_immutable_array(axis.lower),
        upper=_immutable_array(axis.upper),
        lower_interior=_immutable_array(axis.lower_interior),
        upper_interior=_immutable_array(axis.upper_interior),
    )


def _support_columns(
    range_axis: SplitAxis,
    azimuth_axis: SplitAxis,
    elevation_axis: SplitAxis,
) -> tuple[tuple[tuple[str, str], ...], dict[str, np.ndarray]]:
    schema: list[tuple[str, str]] = []
    columns: dict[str, np.ndarray] = {}
    for prefix, axis in (
        ("range", range_axis),
        ("azimuth", azimuth_axis),
        ("elevation", elevation_axis),
    ):
        for suffix, values in (
            ("centers", axis.centers),
            ("edges", axis.edges),
            ("lower", axis.lower),
            ("upper", axis.upper),
            ("lower_interior", axis.lower_interior),
            ("upper_interior", axis.upper_interior),
        ):
            name = f"{prefix}_{suffix}"
            schema.append((name, "<f8"))
            columns[name] = values
    return tuple(schema), columns


def _support_digest(
    range_axis: SplitAxis,
    azimuth_axis: SplitAxis,
    elevation_axis: SplitAxis,
) -> tuple[str, str]:
    schema, columns = _support_columns(range_axis, azimuth_axis, elevation_axis)
    header = _schema_header(
        "vrh_f0_factorized_support_v1",
        schema,
        shape_rae=[range_axis.count, azimuth_axis.count, elevation_axis.count],
        layout="cell_id=((a2*E2)+e2)*R2+r2",
        stable_cell_id_count=CELL_COUNT,
        legal_range_interval_m=[0.0, NATURAL_RANGE_UPPER_M],
        legal_range_upper_closed=False,
    )
    header_digest = hashlib.sha256(header).hexdigest()
    digest = hashlib.sha256(header)
    for name, dtype in schema:
        digest.update(_as_little_endian_contiguous(columns[name], dtype).tobytes(order="C"))
    for start in range(0, CELL_COUNT, 1_000_000):
        stop = min(start + 1_000_000, CELL_COUNT)
        stable_ids = np.arange(start, stop, dtype="<u4")
        digest.update(stable_ids.tobytes(order="C"))
    return digest.hexdigest(), header_digest


def build_vrh_support(axes: KRadarAxes) -> VRHSupport:
    range_axis = _freeze_axis(_split_axis(axes.range_m, range_axis=True))
    azimuth_axis = _freeze_axis(_split_axis(axes.azimuth_rad, range_axis=False))
    elevation_axis = _freeze_axis(_split_axis(axes.elevation_rad, range_axis=False))
    counts = (range_axis.count, azimuth_axis.count, elevation_axis.count)
    if counts != (R2_COUNT, A2_COUNT, E2_COUNT):
        raise ValueError(f"VRH split support changed: {counts}")
    if range_axis.edges[-1] != NATURAL_RANGE_UPPER_M:
        raise ValueError(
            f"VRH natural range edge changed: {range_axis.edges[-1]!r}"
        )
    if azimuth_axis.edges[-1] - azimuth_axis.edges[0] >= 2.0 * np.pi:
        raise ValueError("VRH angular support must be non-wrapping")
    digest, header_digest = _support_digest(
        range_axis,
        azimuth_axis,
        elevation_axis,
    )
    return VRHSupport(
        range_axis=range_axis,
        azimuth_axis=azimuth_axis,
        elevation_axis=elevation_axis,
        digest_sha256=digest,
        schema_header_sha256=header_digest,
    )


def write_vrh_support(path: Path, support: VRHSupport) -> str:
    """Write the exact factorized support commitment plus stable-ID stream."""

    schema, columns = _support_columns(
        support.range_axis,
        support.azimuth_axis,
        support.elevation_axis,
    )
    header = _schema_header(
        "vrh_f0_factorized_support_v1",
        schema,
        shape_rae=list(support.shape_rae),
        layout="cell_id=((a2*E2)+e2)*R2+r2",
        stable_cell_id_count=CELL_COUNT,
        legal_range_interval_m=[0.0, NATURAL_RANGE_UPPER_M],
        legal_range_upper_closed=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(header)
        for name, dtype in schema:
            handle.write(
                _as_little_endian_contiguous(columns[name], dtype).tobytes(order="C")
            )
        for start in range(0, CELL_COUNT, 1_000_000):
            stop = min(start + 1_000_000, CELL_COUNT)
            handle.write(np.arange(start, stop, dtype="<u4").tobytes(order="C"))
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != support.digest_sha256:
        raise AssertionError("VRH support serialization changed its commitment")
    return observed


def commit_support(support: VRHSupport) -> SupportCommit:
    if support.cell_count != CELL_COUNT or support.ray_count != SUBRAY_COUNT:
        raise ValueError("VRH support cannot be committed with changed cardinality")
    observed, header = _support_digest(
        support.range_axis,
        support.azimuth_axis,
        support.elevation_axis,
    )
    if observed != support.digest_sha256 or header != support.schema_header_sha256:
        raise ValueError("VRH support bytes changed before commitment")
    arrays = (
        values
        for axis in (support.range_axis, support.azimuth_axis, support.elevation_axis)
        for values in (
            axis.centers,
            axis.edges,
            axis.lower,
            axis.upper,
            axis.lower_interior,
            axis.upper_interior,
        )
    )
    if any(values.flags.writeable for values in arrays):
        raise ValueError("VRH support commitment requires immutable axis arrays")
    commitment = object.__new__(SupportCommit)
    object.__setattr__(commitment, "digest_sha256", support.digest_sha256)
    object.__setattr__(commitment, "cell_count", support.cell_count)
    object.__setattr__(commitment, "ray_count", support.ray_count)
    object.__setattr__(commitment, "_seal", _SUPPORT_COMMIT_SEAL)
    return commitment


def validate_support_commit(commitment: SupportCommit) -> None:
    if not isinstance(commitment, SupportCommit):
        raise TypeError("VRH target loading requires a support commitment")
    if commitment._seal is not _SUPPORT_COMMIT_SEAL:
        raise ValueError("VRH support commitment capability is forged")
    if commitment.cell_count != CELL_COUNT or commitment.ray_count != SUBRAY_COUNT:
        raise ValueError("VRH support commitment cardinality changed")
    if len(commitment.digest_sha256) != 64:
        raise ValueError("VRH support commitment digest is malformed")


def stable_cell_ids(
    a2: np.ndarray | int,
    e2: np.ndarray | int,
    r2: np.ndarray | int,
) -> np.ndarray:
    a_value, e_value, r_value = np.broadcast_arrays(
        np.asarray(a2, dtype=np.int64),
        np.asarray(e2, dtype=np.int64),
        np.asarray(r2, dtype=np.int64),
    )
    if np.any((a_value < 0) | (a_value >= A2_COUNT)):
        raise ValueError("VRH azimuth subcell index is outside support")
    if np.any((e_value < 0) | (e_value >= E2_COUNT)):
        raise ValueError("VRH elevation subcell index is outside support")
    if np.any((r_value < 0) | (r_value >= R2_COUNT)):
        raise ValueError("VRH range subcell index is outside support")
    return ((a_value * E2_COUNT + e_value) * R2_COUNT + r_value).astype(np.uint32)


def unravel_cell_ids(cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(cell_ids, dtype=np.uint64)
    if np.any(ids >= CELL_COUNT):
        raise ValueError("VRH stable cell ID is outside support")
    r2 = (ids % R2_COUNT).astype(np.int64)
    ray = ids // R2_COUNT
    e2 = (ray % E2_COUNT).astype(np.int64)
    a2 = (ray // E2_COUNT).astype(np.int64)
    return a2, e2, r2


def cell_center_rae(
    support: VRHSupport,
    cell_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a2, e2, r2 = unravel_cell_ids(cell_ids)
    return (
        support.range_axis.centers[r2],
        support.azimuth_axis.centers[a2],
        support.elevation_axis.centers[e2],
    )


def cell_center_xyz(support: VRHSupport, cell_ids: np.ndarray) -> np.ndarray:
    radius, azimuth, elevation = cell_center_rae(support, cell_ids)
    return polar_to_cartesian(radius, azimuth, elevation).astype(np.float64, copy=False)


def canonicalize_azimuth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    canonical = values.copy()
    outside = (values < -np.pi) | (values >= np.pi)
    canonical[outside] = (
        np.remainder(values[outside] + np.pi, 2.0 * np.pi) - np.pi
    )
    return np.where(canonical == np.pi, -np.pi, canonical)


def assign_half_open(
    values: np.ndarray,
    axis: SplitAxis,
    *,
    final_edge_closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    indices = np.searchsorted(axis.edges, values, side="right") - 1
    in_support = (values >= axis.edges[0]) & (values < axis.edges[-1])
    if final_edge_closed:
        final = values.view(np.uint64) == np.asarray(axis.edges[-1]).view(np.uint64)
        indices = np.where(final, axis.count - 1, indices)
        in_support |= final
    return np.clip(indices, 0, axis.count - 1).astype(np.int64), in_support


def assign_polar_to_support(
    support: VRHSupport,
    polar_rae: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(polar_rae, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("VRH polar assignment expects (N,3)")
    r2, range_ok = assign_half_open(
        values[:, 0], support.range_axis, final_edge_closed=False
    )
    a2, azimuth_ok = assign_half_open(
        canonicalize_azimuth(values[:, 1]),
        support.azimuth_axis,
        final_edge_closed=True,
    )
    e2, elevation_ok = assign_half_open(
        values[:, 2], support.elevation_axis, final_edge_closed=True
    )
    return r2, a2, e2, range_ok & azimuth_ok & elevation_ok


def fitted_rae_from_marks(
    support: VRHSupport,
    marks: ModelMarks,
    cell_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marks.validate(support)
    ids = np.asarray(cell_ids, dtype=np.uint64)
    a2, e2, r2 = unravel_cell_ids(ids)
    ray = a2 * E2_COUNT + e2
    return (
        support.range_axis.centers[r2] + marks.delta_r[ray, r2],
        support.azimuth_axis.centers[a2] + marks.delta_a[ray, r2],
        support.elevation_axis.centers[e2] + marks.delta_e[ray, r2],
    )


def fitted_xyz_from_marks(
    support: VRHSupport,
    marks: ModelMarks,
    cell_ids: np.ndarray,
) -> np.ndarray:
    radius, azimuth, elevation = fitted_rae_from_marks(support, marks, cell_ids)
    return polar_to_cartesian(radius, azimuth, elevation).astype(np.float64, copy=False)


def model_marks_interior_report(
    support: VRHSupport,
    marks: ModelMarks,
    *,
    ray_chunk_size: int = 1_024,
) -> dict[str, int | bool]:
    """Independently verify every fitted coordinate against its source cell."""

    marks.validate(support)
    if ray_chunk_size <= 0:
        raise ValueError("VRH mark-interior chunk size must be positive")
    range_violations = 0
    azimuth_violations = 0
    elevation_violations = 0
    range_center = support.range_axis.centers.reshape(1, -1)
    for start in range(0, support.ray_count, ray_chunk_size):
        stop = min(start + ray_chunk_size, support.ray_count)
        ray = np.arange(start, stop, dtype=np.int64)
        a2 = ray // support.elevation_axis.count
        e2 = ray % support.elevation_axis.count
        fitted_r = range_center + marks.delta_r[start:stop]
        fitted_a = (
            support.azimuth_axis.centers[a2, None]
            + marks.delta_a[start:stop]
        )
        fitted_e = (
            support.elevation_axis.centers[e2, None]
            + marks.delta_e[start:stop]
        )
        range_violations += int(
            (
                (fitted_r < support.range_axis.lower_interior.reshape(1, -1))
                | (fitted_r > support.range_axis.upper_interior.reshape(1, -1))
            ).sum()
        )
        azimuth_violations += int(
            (
                (fitted_a < support.azimuth_axis.lower_interior[a2, None])
                | (fitted_a > support.azimuth_axis.upper_interior[a2, None])
            ).sum()
        )
        elevation_violations += int(
            (
                (fitted_e < support.elevation_axis.lower_interior[e2, None])
                | (fitted_e > support.elevation_axis.upper_interior[e2, None])
            ).sum()
        )
    total = range_violations + azimuth_violations + elevation_violations
    return {
        "checked_cell_count": support.cell_count,
        "range_violation_count": range_violations,
        "azimuth_violation_count": azimuth_violations,
        "elevation_violation_count": elevation_violations,
        "total_violation_count": total,
        "all_marks_inside_source_cell_interiors": total == 0,
    }


def model_marks_digest(marks: ModelMarks, support: VRHSupport) -> str:
    marks.validate(support)
    return canonical_columns_digest(
        kind="vrh_f0_model_marks_v1",
        schema=MODEL_MARK_SCHEMA,
        columns=marks.columns(),
        extra_header={"shape": list(marks.shape), "support_sha256": support.digest_sha256},
    )


def write_model_marks(path: Path, marks: ModelMarks, support: VRHSupport) -> str:
    marks.validate(support)
    return write_canonical_columns(
        path,
        kind="vrh_f0_model_marks_v1",
        schema=MODEL_MARK_SCHEMA,
        columns=marks.columns(),
        extra_header={"shape": list(marks.shape), "support_sha256": support.digest_sha256},
    )
