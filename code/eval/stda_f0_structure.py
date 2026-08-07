"""Representation-neutral structural evaluator for frozen STDA-F0."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Iterable, Sequence

import numpy as np


PROTOCOL_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)

OUTPUT_RANGE_UPPER_M = 118.2685546875
TARGET_RANGE_UPPER_M = 120.0
UNMATCHED_DISTANCE_M = 120.0
GROUP_SPAN_M = 0.05
COMPLETENESS_GATE_M = 1.0
RECALL_GATE = 0.80
UNMATCHED_EVENT_ID = (1 << 63) - 1

RETURN_FIRST = 0
RETURN_LATER = 1
RETURN_LABELS = ("first", "later")
RANGE_BOUNDS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
RANGE_LABELS = ("range_0_30", "range_30_60", "range_60_120")

_HASH_NAMESPACE = b"stda_f0_structure_v1\0"


def _hash_parts(kind: str, parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_NAMESPACE)
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


def _hash_records(
    kind: str,
    records: Iterable[bytes],
    *,
    metadata: Iterable[bytes] = (),
) -> str:
    material = list(metadata)
    material.extend(records)
    return _hash_parts(kind, material)


def _pack_nonnegative_integer(value: int) -> bytes:
    if value < 0:
        raise ValueError("STDA structural integer encoding requires nonnegative input")
    width = max(1, (value.bit_length() + 7) // 8)
    encoded = value.to_bytes(width, byteorder="little", signed=False)
    return struct.pack("<Q", width) + encoded


def _pack_u64_tuple(values: Sequence[int]) -> bytes:
    return struct.pack("<Q", len(values)) + b"".join(
        struct.pack("<Q", value) for value in values
    )


def _immutable_f32_bytes(values: np.ndarray, columns: int, label: str) -> tuple[bytes, int]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{label} must have shape (N,{columns})")
    frozen = np.ascontiguousarray(array, dtype="<f4")
    return frozen.tobytes(order="C"), int(frozen.shape[0])


def _polar_from_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    radius = math.sqrt((x * x + y * y) + z * z)
    azimuth = math.atan2(y, x)
    if azimuth == math.pi:
        azimuth = -math.pi
    ratio = 0.0 if radius == 0.0 else max(-1.0, min(1.0, z / radius))
    elevation = math.asin(ratio)
    return radius, azimuth, elevation


def _axis_index(edges: tuple[float, ...], value: float) -> int:
    if value < edges[0] or value > edges[-1]:
        return -1
    if value == edges[-1]:
        return len(edges) - 2
    index = bisect_right(edges, value) - 1
    return index if 0 <= index < len(edges) - 1 else -1


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


@dataclass(frozen=True)
class StructuralDomain:
    """Immutable angular support used by the representation-neutral evaluator."""

    azimuth_edges_rad: tuple[float, ...]
    elevation_edges_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        azimuth = tuple(float(value) for value in self.azimuth_edges_rad)
        elevation = tuple(float(value) for value in self.elevation_edges_rad)
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
        object.__setattr__(self, "azimuth_edges_rad", azimuth)
        object.__setattr__(self, "elevation_edges_rad", elevation)

    @classmethod
    def from_edges(
        cls,
        azimuth_edges_rad: Sequence[float],
        elevation_edges_rad: Sequence[float],
    ) -> StructuralDomain:
        return cls(tuple(azimuth_edges_rad), tuple(elevation_edges_rad))

    @property
    def azimuth_count(self) -> int:
        return len(self.azimuth_edges_rad) - 1

    @property
    def elevation_count(self) -> int:
        return len(self.elevation_edges_rad) - 1

    @property
    def ray_count(self) -> int:
        return self.azimuth_count * self.elevation_count

    @property
    def digest_sha256(self) -> str:
        return _hash_parts(
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
class StructuralInputs:
    """Exact immutable float32 snapshots consumed by the evaluator."""

    domain: StructuralDomain
    output_xyz_le_f4: bytes
    output_count: int
    target_xyz_confidence_le_f4: bytes
    target_count: int

    def __post_init__(self) -> None:
        output = bytes(self.output_xyz_le_f4)
        target = bytes(self.target_xyz_confidence_le_f4)
        if self.output_count < 0 or len(output) != self.output_count * 3 * 4:
            raise ValueError("STDA output byte count does not match output_count")
        if self.target_count < 0 or len(target) != self.target_count * 4 * 4:
            raise ValueError("STDA target byte count does not match target_count")
        object.__setattr__(self, "output_xyz_le_f4", output)
        object.__setattr__(self, "target_xyz_confidence_le_f4", target)

    @property
    def raw_sha256(self) -> str:
        return _hash_parts(
            "raw_inputs",
            (
                bytes.fromhex(self.domain.digest_sha256),
                struct.pack("<Q", self.output_count),
                self.output_xyz_le_f4,
                struct.pack("<Q", self.target_count),
                self.target_xyz_confidence_le_f4,
            ),
        )


@dataclass(frozen=True)
class CanonicalOutputEvent:
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

    @property
    def domain_valid(self) -> bool:
        return self.ray_id >= 0


@dataclass(frozen=True)
class CanonicalOutputSet:
    events: tuple[CanonicalOutputEvent, ...]
    source_row_count: int
    duplicate_row_count: int
    invalid_event_ids: tuple[int, ...]
    digest_sha256: str

    @property
    def domain_valid(self) -> bool:
        return not self.invalid_event_ids


@dataclass(frozen=True)
class CanonicalTargetAtom:
    canonical_target_id: int
    ray_id: int
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    x: float
    y: float
    z: float
    xyz_bytes: bytes
    weight: float

    @property
    def domain_valid(self) -> bool:
        return self.ray_id >= 0


@dataclass(frozen=True)
class CanonicalTargetSet:
    atoms: tuple[CanonicalTargetAtom, ...]
    source_row_count: int
    zero_confidence_row_count: int
    invalid_target_ids: tuple[int, ...]
    digest_sha256: str

    @property
    def domain_valid(self) -> bool:
        return not self.invalid_target_ids


@dataclass(frozen=True)
class TargetReturnGroup:
    ray_id: int
    group_index: int
    return_class: int
    range_stratum: int
    representative_range_m: float
    group_weight: float
    canonical_target_ids: tuple[int, ...]

    @property
    def return_label(self) -> str:
        return RETURN_LABELS[self.return_class]

    @property
    def class_key(self) -> str:
        return f"{RANGE_LABELS[self.range_stratum]}_{self.return_label}"


@dataclass(frozen=True)
class GroupAssignment:
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

    @property
    def matched(self) -> bool:
        return self.matched_event_id != UNMATCHED_EVENT_ID


@dataclass(frozen=True)
class ClassMetric:
    key: str
    range_stratum: int
    return_class: int
    applicable: bool
    group_count: int
    effective_weight: float
    completeness_mean_distance_m: float | None
    recall_1m: float | None
    passed: bool | None


@dataclass(frozen=True)
class StructuralEvaluation:
    raw_inputs_sha256: str
    canonical_inputs_sha256: str
    output: CanonicalOutputSet
    target: CanonicalTargetSet
    structural_domain_valid: bool
    evaluation_complete: bool
    failure_reasons: tuple[str, ...]
    groups: tuple[TargetReturnGroup, ...]
    assignments: tuple[GroupAssignment, ...]
    class_metrics: tuple[ClassMetric, ...]
    integer_cost: int | None
    matched_count: int
    complete_assignment_tuple: tuple[int, ...]
    groups_sha256: str
    assignments_sha256: str
    complete_mapping_sha256: str
    class_metrics_sha256: str
    result_sha256: str

    @property
    def objective_key(self) -> tuple[int, int, tuple[int, ...]] | None:
        if self.integer_cost is None:
            return None
        return (
            self.integer_cost,
            -self.matched_count,
            self.complete_assignment_tuple,
        )

    @property
    def protocol_freeze_commit(self) -> str:
        return PROTOCOL_FREEZE_COMMIT

    @property
    def protocol_sha256(self) -> str:
        return PROTOCOL_SHA256


@dataclass(frozen=True)
class _DPSolution:
    integer_cost: int
    matched_count: int
    assignment_tuple: tuple[int, ...]

    @property
    def key(self) -> tuple[int, int, tuple[int, ...]]:
        return self.integer_cost, -self.matched_count, self.assignment_tuple


def freeze_structural_inputs(
    output_xyz: np.ndarray,
    target_xyz_confidence: np.ndarray,
    *,
    domain: StructuralDomain,
) -> StructuralInputs:
    """Serialize evaluator inputs once as immutable little-endian float32 bytes."""

    output_bytes, output_count = _immutable_f32_bytes(output_xyz, 3, "STDA output")
    target_bytes, target_count = _immutable_f32_bytes(
        target_xyz_confidence,
        4,
        "STDA target",
    )
    return StructuralInputs(
        domain=domain,
        output_xyz_le_f4=output_bytes,
        output_count=output_count,
        target_xyz_confidence_le_f4=target_bytes,
        target_count=target_count,
    )


def _event_record(event: CanonicalOutputEvent) -> bytes:
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


def canonicalize_output_events(inputs: StructuralInputs) -> CanonicalOutputSet:
    """Assign row-permutation-neutral event IDs to unique float32 XYZ bytes."""

    xyz = np.frombuffer(inputs.output_xyz_le_f4, dtype="<f4").reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError("STDA output XYZ contains a non-finite value")

    unique_bytes = {xyz[row].tobytes(order="C") for row in range(xyz.shape[0])}
    provisional: list[tuple[float, float, float, float, float, float, bytes, int]] = []
    for xyz_bytes in unique_bytes:
        x32, y32, z32 = struct.unpack("<fff", xyz_bytes)
        x, y, z = float(x32), float(y32), float(z32)
        radius, azimuth, elevation = _polar_from_xyz(x, y, z)
        ray_id = inputs.domain.ray_id(azimuth, elevation)
        if not (0.0 <= radius <= OUTPUT_RANGE_UPPER_M):
            ray_id = -1
        provisional.append(
            (radius, azimuth, elevation, x, y, z, xyz_bytes, ray_id)
        )
    provisional.sort(key=lambda row: row[:7])

    depth_by_id: dict[int, int] = {}
    rows_by_ray: dict[int, list[tuple[float, int]]] = {}
    for event_id, row in enumerate(provisional):
        ray_id = row[7]
        if ray_id >= 0:
            rows_by_ray.setdefault(ray_id, []).append((row[0], event_id))
    for ray_rows in rows_by_ray.values():
        ray_rows.sort(key=lambda row: (row[0], row[1]))
        for depth, (_, event_id) in enumerate(ray_rows, start=1):
            depth_by_id[event_id] = depth

    events = tuple(
        CanonicalOutputEvent(
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
    invalid = tuple(event.event_id for event in events if not event.domain_valid)
    digest = _hash_records("canonical_output_events", map(_event_record, events))
    return CanonicalOutputSet(
        events=events,
        source_row_count=inputs.output_count,
        duplicate_row_count=inputs.output_count - len(events),
        invalid_event_ids=invalid,
        digest_sha256=digest,
    )


def _canonical_positive_zero(value: np.float32) -> np.float32:
    return np.float32(0.0) if value == 0.0 else value


def _target_atom_record(atom: CanonicalTargetAtom) -> bytes:
    return b"".join(
        (
            struct.pack(
                "<Qqddddddd",
                atom.canonical_target_id,
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


def canonicalize_positive_target_atoms(inputs: StructuralInputs) -> CanonicalTargetSet:
    """Aggregate positive target mass by canonical float32 XYZ bytes."""

    target = np.frombuffer(
        inputs.target_xyz_confidence_le_f4,
        dtype="<f4",
    ).reshape(-1, 4)
    if not np.isfinite(target).all():
        raise ValueError("STDA target contains a non-finite value")
    if np.any(target[:, 3] < 0.0):
        raise ValueError("STDA target confidence must be nonnegative")

    confidence_by_xyz: dict[bytes, list[tuple[int, float]]] = {}
    zero_count = 0
    for row in target:
        canonical = tuple(_canonical_positive_zero(value) for value in row)
        confidence = canonical[3]
        if confidence == 0.0:
            zero_count += 1
            continue
        xyz_array = np.asarray(canonical[:3], dtype="<f4")
        xyz_bytes = xyz_array.tobytes(order="C")
        confidence_bytes = np.asarray([confidence], dtype="<f4").tobytes(order="C")
        confidence_bits = struct.unpack("<I", confidence_bytes)[0]
        confidence_by_xyz.setdefault(xyz_bytes, []).append(
            (confidence_bits, float(confidence))
        )

    provisional: list[tuple[float, float, float, float, float, float, bytes, float, int]] = []
    for xyz_bytes, confidence_values in confidence_by_xyz.items():
        confidence_values.sort(key=lambda item: item[0])
        weight = math.fsum(value for _, value in confidence_values)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("STDA canonical target atom has invalid positive weight")
        x32, y32, z32 = struct.unpack("<fff", xyz_bytes)
        x, y, z = float(x32), float(y32), float(z32)
        radius, azimuth, elevation = _polar_from_xyz(x, y, z)
        ray_id = inputs.domain.ray_id(azimuth, elevation)
        if not (0.0 <= radius < TARGET_RANGE_UPPER_M):
            ray_id = -1
        provisional.append(
            (radius, azimuth, elevation, x, y, z, xyz_bytes, weight, ray_id)
        )
    provisional.sort(key=lambda row: row[:7])

    atoms = tuple(
        CanonicalTargetAtom(
            canonical_target_id=target_id,
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
    invalid = tuple(
        atom.canonical_target_id for atom in atoms if not atom.domain_valid
    )
    digest = _hash_records("canonical_target_atoms", map(_target_atom_record, atoms))
    return CanonicalTargetSet(
        atoms=atoms,
        source_row_count=inputs.target_count,
        zero_confidence_row_count=zero_count,
        invalid_target_ids=invalid,
        digest_sha256=digest,
    )


def _group_record(group: TargetReturnGroup) -> bytes:
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
            _pack_u64_tuple(group.canonical_target_ids),
        )
    )


def build_target_return_groups(
    targets: CanonicalTargetSet,
) -> tuple[TargetReturnGroup, ...]:
    """Form exact consecutive subray groups using the first range as anchor."""

    if not targets.domain_valid:
        raise ValueError("STDA cannot group out-of-domain target atoms")
    atoms_by_ray: dict[int, list[CanonicalTargetAtom]] = {}
    for atom in targets.atoms:
        atoms_by_ray.setdefault(atom.ray_id, []).append(atom)

    groups: list[TargetReturnGroup] = []
    for ray_id in sorted(atoms_by_ray):
        atoms = sorted(
            atoms_by_ray[ray_id],
            key=lambda atom: (atom.range_m, atom.canonical_target_id),
        )
        start = 0
        group_index = 0
        while start < len(atoms):
            stop = start + 1
            first_range = atoms[start].range_m
            while (
                stop < len(atoms)
                and atoms[stop].range_m - first_range < GROUP_SPAN_M
            ):
                stop += 1
            members = atoms[start:stop]
            group_weight = math.fsum(atom.weight for atom in members)
            representative = math.fsum(
                atom.range_m * atom.weight for atom in members
            ) / group_weight
            groups.append(
                TargetReturnGroup(
                    ray_id=ray_id,
                    group_index=group_index,
                    return_class=(
                        RETURN_FIRST if group_index == 0 else RETURN_LATER
                    ),
                    range_stratum=_range_stratum(representative),
                    representative_range_m=representative,
                    group_weight=group_weight,
                    canonical_target_ids=tuple(
                        atom.canonical_target_id for atom in members
                    ),
                )
            )
            start = stop
            group_index += 1
    return tuple(groups)


def _solve_one_ray(
    groups: Sequence[TargetReturnGroup],
    events: Sequence[CanonicalOutputEvent],
) -> _DPSolution:
    ordered_events = tuple(sorted(events, key=lambda event: (event.range_m, event.event_id)))
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
                        assignment_tuple=(UNMATCHED_EVENT_ID,)
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
                        distance = abs(
                            group.representative_range_m - event.range_m
                        )
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


def _assignment_record(assignment: GroupAssignment) -> bytes:
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


def _build_assignments(
    groups: tuple[TargetReturnGroup, ...],
    events: tuple[CanonicalOutputEvent, ...],
) -> tuple[tuple[GroupAssignment, ...], int, int, tuple[int, ...]]:
    event_lookup = {event.event_id: event for event in events}
    groups_by_ray: dict[int, list[TargetReturnGroup]] = {}
    events_by_ray: dict[int, list[CanonicalOutputEvent]] = {}
    for group in groups:
        groups_by_ray.setdefault(group.ray_id, []).append(group)
    for event in events:
        events_by_ray.setdefault(event.ray_id, []).append(event)

    assignments: list[GroupAssignment] = []
    total_cost = 0
    total_matched = 0
    complete_mapping: list[int] = []
    for ray_id in sorted(groups_by_ray):
        ray_groups = groups_by_ray[ray_id]
        solution = _solve_one_ray(ray_groups, events_by_ray.get(ray_id, ()))
        if len(solution.assignment_tuple) != len(ray_groups):
            raise AssertionError("STDA structural DP did not assign every target group")
        total_cost += solution.integer_cost
        total_matched += solution.matched_count
        complete_mapping.extend(solution.assignment_tuple)
        for group, event_id in zip(ray_groups, solution.assignment_tuple):
            if event_id == UNMATCHED_EVENT_ID:
                event = None
                distance = UNMATCHED_DISTANCE_M
                depth = -1
                event_range = None
            else:
                event = event_lookup[event_id]
                distance = abs(group.representative_range_m - event.range_m)
                depth = event.inferred_depth
                event_range = event.range_m
            assignments.append(
                GroupAssignment(
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
    mapping = tuple(assignment.matched_event_id for assignment in assignments)
    if mapping != tuple(complete_mapping):
        raise AssertionError("STDA structural mapping order changed")
    if sum(item.integer_cost for item in assignments) != total_cost:
        raise AssertionError("STDA structural integer objective changed")
    return tuple(assignments), total_cost, total_matched, mapping


def _class_metric_record(metric: ClassMetric) -> bytes:
    optional = b""
    if metric.applicable:
        if (
            metric.completeness_mean_distance_m is None
            or metric.recall_1m is None
            or metric.passed is None
        ):
            raise ValueError("STDA applicable class metric is incomplete")
        optional = struct.pack(
            "<ddB",
            metric.completeness_mean_distance_m,
            metric.recall_1m,
            int(metric.passed),
        )
    return b"".join(
        (
            metric.key.encode("ascii") + b"\0",
            struct.pack(
                "<BBBQd",
                metric.range_stratum,
                metric.return_class,
                int(metric.applicable),
                metric.group_count,
                metric.effective_weight,
            ),
            optional,
        )
    )


def _class_metrics(
    assignments: tuple[GroupAssignment, ...],
) -> tuple[ClassMetric, ...]:
    metrics: list[ClassMetric] = []
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
                    ClassMetric(
                        key=key,
                        range_stratum=range_stratum,
                        return_class=return_class,
                        applicable=False,
                        group_count=0,
                        effective_weight=0.0,
                        completeness_mean_distance_m=None,
                        recall_1m=None,
                        passed=None,
                    )
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
            passed = completeness <= COMPLETENESS_GATE_M and recall >= RECALL_GATE
            metrics.append(
                ClassMetric(
                    key=key,
                    range_stratum=range_stratum,
                    return_class=return_class,
                    applicable=True,
                    group_count=len(selected),
                    effective_weight=weight_sum,
                    completeness_mean_distance_m=completeness,
                    recall_1m=recall,
                    passed=passed,
                )
            )
    return tuple(metrics)


def _mapping_record(assignment: GroupAssignment) -> bytes:
    return struct.pack(
        "<qQQ",
        assignment.ray_id,
        assignment.group_index,
        assignment.matched_event_id,
    )


def _canonical_input_digest(
    inputs: StructuralInputs,
    output: CanonicalOutputSet,
    target: CanonicalTargetSet,
) -> str:
    return _hash_parts(
        "canonical_inputs",
        (
            bytes.fromhex(inputs.domain.digest_sha256),
            bytes.fromhex(output.digest_sha256),
            bytes.fromhex(target.digest_sha256),
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
    cost_bytes = b"none" if integer_cost is None else _pack_nonnegative_integer(integer_cost)
    return _hash_parts(
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


def evaluate_structure(inputs: StructuralInputs) -> StructuralEvaluation:
    """Evaluate frozen first/later structure without solver-specific identities."""

    output = canonicalize_output_events(inputs)
    target = canonicalize_positive_target_atoms(inputs)
    canonical_inputs_sha256 = _canonical_input_digest(inputs, output, target)
    reasons: list[str] = []
    if not output.domain_valid:
        reasons.append("output_event_out_of_structural_domain")
    if not target.domain_valid:
        reasons.append("positive_target_out_of_structural_domain")

    structural_domain_valid = not reasons
    if structural_domain_valid:
        groups = build_target_return_groups(target)
        assignments, integer_cost, matched_count, mapping = _build_assignments(
            groups,
            output.events,
        )
        class_metrics = _class_metrics(assignments)
        evaluation_complete = True
    else:
        groups = ()
        assignments = ()
        class_metrics = ()
        integer_cost = None
        matched_count = 0
        mapping = ()
        evaluation_complete = False

    groups_sha256 = _hash_records("target_return_groups", map(_group_record, groups))
    assignments_sha256 = _hash_records(
        "group_assignments",
        map(_assignment_record, assignments),
    )
    mapping_sha256 = _hash_records(
        "complete_group_event_mapping",
        map(_mapping_record, assignments),
    )
    class_metrics_sha256 = _hash_records(
        "range_return_class_metrics",
        map(_class_metric_record, class_metrics),
    )
    result_sha256 = _result_digest(
        canonical_inputs_sha256=canonical_inputs_sha256,
        structural_domain_valid=structural_domain_valid,
        evaluation_complete=evaluation_complete,
        failure_reasons=tuple(reasons),
        groups_sha256=groups_sha256,
        assignments_sha256=assignments_sha256,
        mapping_sha256=mapping_sha256,
        class_metrics_sha256=class_metrics_sha256,
        integer_cost=integer_cost,
        matched_count=matched_count,
        mapping=mapping,
    )
    return StructuralEvaluation(
        raw_inputs_sha256=inputs.raw_sha256,
        canonical_inputs_sha256=canonical_inputs_sha256,
        output=output,
        target=target,
        structural_domain_valid=structural_domain_valid,
        evaluation_complete=evaluation_complete,
        failure_reasons=tuple(reasons),
        groups=groups,
        assignments=assignments,
        class_metrics=class_metrics,
        integer_cost=integer_cost,
        matched_count=matched_count,
        complete_assignment_tuple=mapping,
        groups_sha256=groups_sha256,
        assignments_sha256=assignments_sha256,
        complete_mapping_sha256=mapping_sha256,
        class_metrics_sha256=class_metrics_sha256,
        result_sha256=result_sha256,
    )
