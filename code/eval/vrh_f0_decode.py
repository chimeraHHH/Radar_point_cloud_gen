"""Target-free renewal decoding and export for the frozen VRH-F0 gate."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from cube_dense.kradar import polar_to_cartesian
from eval.vrh_f0_support import (
    MINIMUM_DISTANCE_M,
    ModelMarks,
    VRHSupport,
    canonical_columns_digest,
    model_marks_interior_report,
    write_canonical_columns,
)


LOG_QUANTIZATION_SCALE = 1.0e12
CONFIDENCE_QUANTIZATION_SCALE = 1.0e9
CANONICAL_STREAM_SCHEMA = (
    ("ray_id", "<u2"),
    ("stream_index", "<u2"),
    ("cell_id", "<u4"),
    ("depth", "<u2"),
    ("q_log_probability", "<i8"),
    ("opaque_distance_priority_key", "<i8"),
    ("confidence", "<f8"),
    ("r", "<f8"),
    ("a", "<f8"),
    ("e", "<f8"),
)
EXPOSURE_TRACE_SCHEMA = (
    ("pop_index", "<u8"),
    ("ray_id", "<u2"),
    ("stream_index", "<u2"),
    ("cell_id", "<u4"),
    ("accepted", "<u1"),
)
SELECTED_EVENT_SCHEMA = (
    ("cell_id", "<u4"),
    ("depth", "<u2"),
    ("r", "<f8"),
    ("a", "<f8"),
    ("e", "<f8"),
    ("x", "<f8"),
    ("y", "<f8"),
    ("z", "<f8"),
    ("confidence", "<f8"),
)
ACCEPTED_CELL_ID_SCHEMA = (("cell_id", "<u4"),)


@dataclass(frozen=True)
class CanonicalStreams:
    ray_id: np.ndarray
    stream_index: np.ndarray
    cell_id: np.ndarray
    depth: np.ndarray
    q_log_probability: np.ndarray
    opaque_distance_priority_key: np.ndarray
    confidence: np.ndarray
    r: np.ndarray
    a: np.ndarray
    e: np.ndarray
    ray_offsets: np.ndarray
    range_count: int
    elevation_count: int
    support_sha256: str
    digest_sha256: str

    @property
    def event_count(self) -> int:
        return int(self.cell_id.size)

    @property
    def ray_count(self) -> int:
        return int(self.ray_offsets.size - 1)

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "ray_id": self.ray_id,
            "stream_index": self.stream_index,
            "cell_id": self.cell_id,
            "depth": self.depth,
            "q_log_probability": self.q_log_probability,
            "opaque_distance_priority_key": self.opaque_distance_priority_key,
            "confidence": self.confidence,
            "r": self.r,
            "a": self.a,
            "e": self.e,
        }

    def ray_slice(self, ray_id: int) -> slice:
        if ray_id < 0 or ray_id >= self.ray_count:
            raise IndexError(ray_id)
        return slice(int(self.ray_offsets[ray_id]), int(self.ray_offsets[ray_id + 1]))

    def validate(self, support: VRHSupport) -> None:
        if self.range_count != support.range_axis.count:
            raise ValueError("CanonicalStreams range count changed")
        if self.elevation_count != support.elevation_axis.count:
            raise ValueError("CanonicalStreams elevation count changed")
        if self.support_sha256 != support.digest_sha256:
            raise ValueError("CanonicalStreams support commitment changed")
        expected_dtypes = dict(CANONICAL_STREAM_SCHEMA)
        lengths = set()
        for name in expected_dtypes:
            values = getattr(self, name)
            lengths.add(values.size)
            if values.ndim != 1:
                raise ValueError(f"CanonicalStreams.{name} must be one-dimensional")
            if np.dtype(values.dtype) != np.dtype(expected_dtypes[name]):
                raise ValueError(
                    f"CanonicalStreams.{name} dtype {values.dtype} != {expected_dtypes[name]}"
                )
        if len(lengths) != 1:
            raise ValueError("CanonicalStreams columns have different lengths")
        if self.ray_offsets.shape != (support.ray_count + 1,):
            raise ValueError("CanonicalStreams ray offsets changed")
        if self.ray_offsets[0] != 0 or self.ray_offsets[-1] != self.event_count:
            raise ValueError("CanonicalStreams ray offsets do not span events")
        if np.any(np.diff(self.ray_offsets) < 0):
            raise ValueError("CanonicalStreams ray offsets are not monotone")
        for ray in range(support.ray_count):
            section = self.ray_slice(ray)
            count = section.stop - section.start
            if count == 0:
                continue
            if not np.all(self.ray_id[section] == ray):
                raise ValueError("CanonicalStreams ray order changed")
            expected_index = np.arange(count, dtype=np.uint16)
            if not np.array_equal(self.stream_index[section], expected_index):
                raise ValueError("CanonicalStreams stream index changed")
            if not np.array_equal(self.depth[section], expected_index + 1):
                raise ValueError("CanonicalStreams depth changed")
            if not np.all(np.diff(self.cell_id[section].astype(np.int64)) > 0):
                raise ValueError("CanonicalStreams cell IDs are not range ordered")
        for name in ("confidence", "r", "a", "e"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"CanonicalStreams.{name} contains non-finite values")


@dataclass(frozen=True)
class ExposureTrace:
    pop_index: np.ndarray
    ray_id: np.ndarray
    stream_index: np.ndarray
    cell_id: np.ndarray
    accepted: np.ndarray
    digest_sha256: str

    @property
    def count(self) -> int:
        return int(self.pop_index.size)

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "pop_index": self.pop_index,
            "ray_id": self.ray_id,
            "stream_index": self.stream_index,
            "cell_id": self.cell_id,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class ExportResult:
    arm: str
    streams_sha256: str
    range_count: int
    selected_event_rows: np.ndarray
    selected_cell_id: np.ndarray
    selected_depth: np.ndarray
    selected_rae: np.ndarray
    selected_xyz: np.ndarray
    selected_confidence: np.ndarray
    trace: ExposureTrace
    accepted_cell_id_sha256: str
    selected_set_sha256: str
    report: dict[str, Any]

    @property
    def selected_count(self) -> int:
        return int(self.selected_cell_id.size)

    def selected_columns(self) -> dict[str, np.ndarray]:
        return {
            "cell_id": self.selected_cell_id,
            "depth": self.selected_depth,
            "r": self.selected_rae[:, 0],
            "a": self.selected_rae[:, 1],
            "e": self.selected_rae[:, 2],
            "x": self.selected_xyz[:, 0],
            "y": self.selected_xyz[:, 1],
            "z": self.selected_xyz[:, 2],
            "confidence": self.selected_confidence,
        }


def _quantize_finite(value: float, scale: float, label: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f"VRH {label} quantization received a non-finite value")
    quantized = float(np.rint(scale * value))
    if not math.isfinite(quantized) or not (
        np.iinfo(np.int64).min <= quantized <= np.iinfo(np.int64).max
    ):
        raise ValueError(f"VRH {label} quantization exceeds signed int64")
    return int(quantized)


def canonical_streams_digest(streams: CanonicalStreams) -> str:
    return canonical_columns_digest(
        kind="vrh_f0_canonical_streams_v1",
        schema=CANONICAL_STREAM_SCHEMA,
        columns=streams.columns(),
        extra_header={
            "ray_count": streams.ray_count,
            "range_count": streams.range_count,
            "elevation_count": streams.elevation_count,
            "support_sha256": streams.support_sha256,
        },
    )


def _verify_stream_commitment(streams: CanonicalStreams) -> None:
    observed = canonical_streams_digest(streams)
    if observed != streams.digest_sha256:
        raise ValueError("VRH canonical stream bytes changed after decoding")


def _decode_one_ray(
    support: VRHSupport,
    marks: ModelMarks,
    ray_id: int,
) -> list[tuple[int, int, int, int, int, int, float, float, float, float]]:
    hazards = marks.base_hazard[ray_id]
    radius = support.range_axis.centers + marks.delta_r[ray_id]
    range_count = support.range_axis.count
    elevation_count = support.elevation_axis.count
    a2 = ray_id // elevation_count
    e2 = ray_id % elevation_count
    azimuth = support.azimuth_axis.centers[a2] + marks.delta_a[ray_id]
    elevation = support.elevation_axis.centers[e2] + marks.delta_e[ray_id]
    if not (
        np.isfinite(hazards).all()
        and np.isfinite(radius).all()
        and np.isfinite(azimuth).all()
        and np.isfinite(elevation).all()
    ):
        raise ValueError("VRH decoder received NaN or infinity")
    cursor = 0
    previous_radius: float | None = None
    events: list[tuple[int, int, int, int, int, int, float, float, float, float]] = []
    while cursor < range_count:
        effective = hazards[cursor:].copy()
        if previous_radius is not None:
            effective[radius[cursor:] < previous_radius + MINIMUM_DISTANCE_M] = 0.0
        positive = effective > 0.0
        survival_term = np.zeros_like(effective)
        survival_term[positive] = np.log1p(-effective[positive])
        if not np.isfinite(survival_term).all():
            raise ValueError("VRH STOP survival became non-finite")
        prefix = np.empty_like(effective)
        prefix[0] = 0.0
        if prefix.size > 1:
            prefix[1:] = np.cumsum(survival_term[:-1])
        event_log = np.full_like(effective, -np.inf)
        event_log[positive] = np.log(effective[positive]) + prefix[positive]
        stop_log = float(survival_term.sum())
        q_stop = _quantize_finite(stop_log, LOG_QUANTIZATION_SCALE, "STOP log")
        finite_rows = np.flatnonzero(np.isfinite(event_log))
        if finite_rows.size == 0:
            break
        quantized = np.rint(LOG_QUANTIZATION_SCALE * event_log[finite_rows])
        if not np.isfinite(quantized).all() or np.any(
            (quantized < np.iinfo(np.int64).min)
            | (quantized > np.iinfo(np.int64).max)
        ):
            raise ValueError("VRH event log quantization exceeds signed int64")
        quantized = quantized.astype(np.int64)
        best_value = int(quantized.max())
        best_relative = int(finite_rows[np.flatnonzero(quantized == best_value)[0]])
        if best_value < q_stop:
            break
        r2 = cursor + best_relative
        depth = len(events) + 1
        cell_id = ray_id * range_count + r2
        events.append(
            (
                ray_id,
                depth - 1,
                cell_id,
                depth,
                best_value,
                int(marks.opaque_distance_priority_key[ray_id, r2]),
                float(marks.confidence[ray_id, r2]),
                float(radius[r2]),
                float(azimuth[r2]),
                float(elevation[r2]),
            )
        )
        previous_radius = float(radius[r2])
        cursor = r2 + 1
    if len(events) > range_count:
        raise AssertionError("VRH decoder exceeded finite support")
    return events


def decode_canonical_streams(
    support: VRHSupport,
    model_marks: ModelMarks,
) -> CanonicalStreams:
    """Fully unroll every ray to STOP before either exporter runs."""

    model_marks.validate(support)
    interior = model_marks_interior_report(support, model_marks)
    if not interior["all_marks_inside_source_cell_interiors"]:
        raise ValueError("VRH decoder received an out-of-cell fitted mark")
    buffers = {
        name: np.empty(support.cell_count, dtype=dtype)
        for name, dtype in CANONICAL_STREAM_SCHEMA
    }
    event_count = 0
    offsets = np.zeros(support.ray_count + 1, dtype="<u8")
    for ray_id in range(support.ray_count):
        rows = _decode_one_ray(support, model_marks, ray_id)
        if rows:
            stop = event_count + len(rows)
            columns = list(zip(*rows))
            for (name, _), values in zip(CANONICAL_STREAM_SCHEMA, columns):
                buffers[name][event_count:stop] = values
            event_count = stop
        offsets[ray_id + 1] = event_count
    arrays = {
        name: np.array(values[:event_count], copy=True, order="C")
        for name, values in buffers.items()
    }
    streams = CanonicalStreams(
        **arrays,
        ray_offsets=offsets,
        range_count=support.range_axis.count,
        elevation_count=support.elevation_axis.count,
        support_sha256=support.digest_sha256,
        digest_sha256="",
    )
    digest = canonical_streams_digest(streams)
    object.__setattr__(streams, "digest_sha256", digest)
    for values in (*streams.columns().values(), streams.ray_offsets):
        values.flags.writeable = False
    streams.validate(support)
    return streams


class _SpatialExclusion:
    def __init__(self, minimum_distance_m: float) -> None:
        if minimum_distance_m <= 0.0:
            raise ValueError("VRH spacing must be positive")
        self.minimum_distance_m = float(minimum_distance_m)
        self.minimum_squared = float(minimum_distance_m**2)
        self.cells: dict[tuple[int, int, int], list[np.ndarray]] = {}

    def _cell(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(math.floor(float(value) / self.minimum_distance_m) for value in point)

    def conflicts(self, point: np.ndarray) -> bool:
        cell = self._cell(point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in self.cells.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz), ()
                    ):
                        if float(np.sum((point - other) ** 2)) < self.minimum_squared:
                            return True
        return False

    def add(self, point: np.ndarray) -> None:
        self.cells.setdefault(self._cell(point), []).append(point.copy())


def _event_heap_key(streams: CanonicalStreams, event_row: int) -> tuple[int, int, int, int]:
    confidence_key = _quantize_finite(
        float(streams.confidence[event_row]),
        CONFIDENCE_QUANTIZATION_SCALE,
        "confidence",
    )
    return (
        -int(streams.q_log_probability[event_row]),
        int(streams.opaque_distance_priority_key[event_row]),
        -confidence_key,
        int(streams.cell_id[event_row]),
    )


def _finalize_trace(rows: list[tuple[int, int, int, int, int]], arm: str) -> ExposureTrace:
    if rows:
        values = list(zip(*rows))
    else:
        values = [()] * len(EXPOSURE_TRACE_SCHEMA)
    columns = {
        name: np.ascontiguousarray(column, dtype=dtype)
        for (name, dtype), column in zip(EXPOSURE_TRACE_SCHEMA, values)
    }
    digest = canonical_columns_digest(
        kind="vrh_f0_exposure_trace_v1",
        schema=EXPOSURE_TRACE_SCHEMA,
        columns=columns,
        extra_header={"arm": arm},
    )
    return ExposureTrace(**columns, digest_sha256=digest)


def _export(
    streams: CanonicalStreams,
    *,
    arm: str,
    initialize: Callable[[list[tuple[tuple[int, int, int, int], int]]], None],
    reveal_next: bool,
    ordered_event_rows: np.ndarray | None,
    exact_count: int,
    minimum_distance_m: float,
) -> ExportResult:
    if exact_count <= 0:
        raise ValueError("VRH exact count must be positive")
    _verify_stream_commitment(streams)
    heap: list[tuple[tuple[int, int, int, int], int]] = []
    if ordered_event_rows is None:
        initialize(heap)
    else:
        ordered_event_rows = np.asarray(ordered_event_rows, dtype=np.int64)
        if not np.array_equal(
            np.sort(ordered_event_rows),
            np.arange(streams.event_count, dtype=np.int64),
        ):
            raise ValueError("VRH flat exposure order is not a full event permutation")
    flat_cursor = 0
    exclusion = _SpatialExclusion(minimum_distance_m)
    selected_rows: list[int] = []
    trace_rows: list[tuple[int, int, int, int, int]] = []
    rejected_spacing = 0
    while len(selected_rows) < exact_count and (
        heap
        or (
            ordered_event_rows is not None
            and flat_cursor < ordered_event_rows.size
        )
    ):
        if ordered_event_rows is None:
            _, event_row = heapq.heappop(heap)
        else:
            event_row = int(ordered_event_rows[flat_cursor])
            flat_cursor += 1
        point = polar_to_cartesian(
            streams.r[event_row : event_row + 1],
            streams.a[event_row : event_row + 1],
            streams.e[event_row : event_row + 1],
        )[0]
        accepted = not exclusion.conflicts(point)
        trace_rows.append(
            (
                len(trace_rows),
                int(streams.ray_id[event_row]),
                int(streams.stream_index[event_row]),
                int(streams.cell_id[event_row]),
                int(accepted),
            )
        )
        if accepted:
            exclusion.add(point)
            selected_rows.append(event_row)
        else:
            rejected_spacing += 1
        if reveal_next:
            ray_id = int(streams.ray_id[event_row])
            next_index = int(streams.stream_index[event_row]) + 1
            section = streams.ray_slice(ray_id)
            if next_index < section.stop - section.start:
                next_row = section.start + next_index
                heapq.heappush(heap, (_event_heap_key(streams, next_row), next_row))

    selected = np.asarray(selected_rows, dtype=np.int64)
    cell_id = np.ascontiguousarray(streams.cell_id[selected], dtype="<u4")
    depth = np.ascontiguousarray(streams.depth[selected], dtype="<u2")
    selected_rae = np.ascontiguousarray(
        np.column_stack((streams.r[selected], streams.a[selected], streams.e[selected])),
        dtype=np.float64,
    )
    selected_xyz = np.ascontiguousarray(
        polar_to_cartesian(
            selected_rae[:, 0], selected_rae[:, 1], selected_rae[:, 2]
        ),
        dtype=np.float64,
    )
    selected_confidence = np.ascontiguousarray(streams.confidence[selected], dtype="<f8")
    selected_columns = {
        "cell_id": cell_id,
        "depth": depth,
        "r": selected_rae[:, 0],
        "a": selected_rae[:, 1],
        "e": selected_rae[:, 2],
        "x": selected_xyz[:, 0],
        "y": selected_xyz[:, 1],
        "z": selected_xyz[:, 2],
        "confidence": selected_confidence,
    }
    selected_digest = canonical_columns_digest(
        kind="vrh_f0_selected_events_v1",
        schema=SELECTED_EVENT_SCHEMA,
        columns=selected_columns,
        extra_header={"arm": arm, "streams_sha256": streams.digest_sha256},
    )
    accepted_cell_digest = canonical_columns_digest(
        kind="vrh_f0_accepted_cell_ids_v1",
        schema=ACCEPTED_CELL_ID_SCHEMA,
        columns={"cell_id": cell_id},
        extra_header={"streams_sha256": streams.digest_sha256},
    )
    trace = _finalize_trace(trace_rows, arm)
    if selected_xyz.shape[0] >= 2:
        nearest_other = cKDTree(selected_xyz).query(
            selected_xyz, k=2, workers=1
        )[0][:, 1]
        observed_minimum = float(nearest_other.min())
    else:
        observed_minimum = math.inf
    finite_xyz = bool(np.isfinite(selected_xyz).all())
    unique_cells = int(np.unique(cell_id).size)
    unique_xyz = int(np.unique(selected_xyz, axis=0).shape[0]) if finite_xyz else 0
    spacing_passed = observed_minimum >= minimum_distance_m - 1e-12
    report = {
        "arm": arm,
        "method": (
            "one_next_event_per_ray_sequential_frontier"
            if reveal_next
            else "same_stream_all_events_flat_exposure_control"
        ),
        "canonical_stream_sha256": streams.digest_sha256,
        "exposure_trace_sha256": trace.digest_sha256,
        "selected_set_sha256": selected_digest,
        "canonical_event_count": streams.event_count,
        "popped_event_count": trace.count,
        "selected_count": int(selected.size),
        "exact_count_reached": int(selected.size) == exact_count,
        "frontier_exhausted": (
            not heap
            and (
                ordered_event_rows is None
                or flat_cursor >= ordered_event_rows.size
            )
            and int(selected.size) < exact_count
        ),
        "rejected_minimum_distance_count": rejected_spacing,
        "observed_minimum_pair_distance_m": (
            observed_minimum if math.isfinite(observed_minimum) else None
        ),
        "independent_kdtree_spacing_passed": spacing_passed,
        "finite_xyz": finite_xyz,
        "unique_selected_cell_count": unique_cells,
        "unique_selected_xyz_count": unique_xyz,
        "accepted_depth_ge_2_count": int((depth >= 2).sum()),
        "accepted_cell_id_sha256": accepted_cell_digest,
        "target_input": False,
        "sidecar_input": False,
        "range_quota_used": False,
        "padding_used": False,
        "random_jitter_used": False,
        "duplicate_fallback_used": False,
    }
    return ExportResult(
        arm=arm,
        streams_sha256=streams.digest_sha256,
        range_count=streams.range_count,
        selected_event_rows=selected,
        selected_cell_id=cell_id,
        selected_depth=depth,
        selected_rae=selected_rae,
        selected_xyz=selected_xyz,
        selected_confidence=selected_confidence,
        trace=trace,
        accepted_cell_id_sha256=accepted_cell_digest,
        selected_set_sha256=selected_digest,
        report=report,
    )


def export_sequential_frontier(
    canonical_streams: CanonicalStreams,
    *,
    exact_count: int = 10_000,
    minimum_distance_m: float = MINIMUM_DISTANCE_M,
) -> ExportResult:
    """Expose only the current next event from each nonempty ray."""

    def initialize(heap: list[tuple[tuple[int, int, int, int], int]]) -> None:
        for ray_id in range(canonical_streams.ray_count):
            section = canonical_streams.ray_slice(ray_id)
            if section.start < section.stop:
                heapq.heappush(
                    heap,
                    (_event_heap_key(canonical_streams, section.start), section.start),
                )

    return _export(
        canonical_streams,
        arm="decision_sequential_frontier",
        initialize=initialize,
        reveal_next=True,
        ordered_event_rows=None,
        exact_count=exact_count,
        minimum_distance_m=minimum_distance_m,
    )


def export_flat_control(
    canonical_streams: CanonicalStreams,
    *,
    exact_count: int = 10_000,
    minimum_distance_m: float = MINIMUM_DISTANCE_M,
) -> ExportResult:
    """Expose all events from the same frozen canonical streams at once."""

    confidence = np.rint(
        CONFIDENCE_QUANTIZATION_SCALE * canonical_streams.confidence
    )
    if not np.isfinite(confidence).all() or np.any(
        (confidence < np.iinfo(np.int64).min)
        | (confidence > np.iinfo(np.int64).max)
    ):
        raise ValueError("VRH flat confidence quantization exceeds signed int64")
    confidence = confidence.astype(np.int64)
    order = np.lexsort(
        (
            canonical_streams.cell_id.astype(np.int64),
            -confidence,
            canonical_streams.opaque_distance_priority_key,
            -canonical_streams.q_log_probability,
        )
    )

    def initialize(_heap: list[tuple[tuple[int, int, int, int], int]]) -> None:
        return None

    return _export(
        canonical_streams,
        arm="control_flat_exposure",
        initialize=initialize,
        reveal_next=False,
        ordered_event_rows=order,
        exact_count=exact_count,
        minimum_distance_m=minimum_distance_m,
    )


def write_canonical_streams(path: Path, streams: CanonicalStreams) -> str:
    return write_canonical_columns(
        path,
        kind="vrh_f0_canonical_streams_v1",
        schema=CANONICAL_STREAM_SCHEMA,
        columns=streams.columns(),
        extra_header={
            "ray_count": streams.ray_count,
            "range_count": streams.range_count,
            "elevation_count": streams.elevation_count,
            "support_sha256": streams.support_sha256,
        },
    )


def write_exposure_trace(path: Path, trace: ExposureTrace, arm: str) -> str:
    return write_canonical_columns(
        path,
        kind="vrh_f0_exposure_trace_v1",
        schema=EXPOSURE_TRACE_SCHEMA,
        columns=trace.columns(),
        extra_header={"arm": arm},
    )


def write_selected_events(path: Path, export: ExportResult) -> str:
    return write_canonical_columns(
        path,
        kind="vrh_f0_selected_events_v1",
        schema=SELECTED_EVENT_SCHEMA,
        columns=export.selected_columns(),
        extra_header={"arm": export.arm, "streams_sha256": export.streams_sha256},
    )
