"""GT-visible evaluators for the frozen VRH-F0 capacity gate."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from eval.vrh_f0_decode import ExportResult
from eval.vrh_f0_fit import CanonicalTargets
from eval.vrh_f0_support import (
    VRHSupport,
    assign_polar_to_support,
    canonical_columns_digest,
    write_canonical_columns,
)


UNMATCHED_DISTANCE_M = 120.0
RETURN_COMPLETENESS_GATE_M = 1.0
RETURN_RECALL_GATE = 0.80
GROUP_SPAN_M = 0.05
RANGE_BOUNDS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
RANGE_LABELS = ("range_0_30m", "range_30_60m", "range_60_120m")
RETURN_ASSIGNMENT_SCHEMA = (
    ("ray_id", "<u2"),
    ("group_index", "<u2"),
    ("return_class", "<u1"),
    ("range_stratum", "<u1"),
    ("representative_range_m", "<f8"),
    ("group_weight", "<f8"),
    ("assigned_distance_m", "<f8"),
    ("matched_cell_id", "<i8"),
    ("matched_depth", "<i2"),
)


@dataclass(frozen=True)
class TargetReturnGroup:
    ray_id: int
    group_index: int
    return_class: int
    range_stratum: int
    representative_range_m: float
    group_weight: float
    canonical_target_ids: tuple[int, ...]


@dataclass(frozen=True)
class _DPSolution:
    integer_cost: int
    matched_count: int
    matched_cell_sequence: tuple[int, ...]
    assignments: tuple[tuple[float, int], ...]

    @property
    def ordering_key(self) -> tuple[int, int, tuple[int, ...]]:
        return (
            self.integer_cost,
            -self.matched_count,
            self.matched_cell_sequence,
        )


@dataclass(frozen=True)
class ReturnMatchResult:
    ray_id: np.ndarray
    group_index: np.ndarray
    return_class: np.ndarray
    range_stratum: np.ndarray
    representative_range_m: np.ndarray
    group_weight: np.ndarray
    assigned_distance_m: np.ndarray
    matched_cell_id: np.ndarray
    matched_depth: np.ndarray
    selected_set_sha256: str
    digest_sha256: str
    report: dict[str, Any]

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "ray_id": self.ray_id,
            "group_index": self.group_index,
            "return_class": self.return_class,
            "range_stratum": self.range_stratum,
            "representative_range_m": self.representative_range_m,
            "group_weight": self.group_weight,
            "assigned_distance_m": self.assigned_distance_m,
            "matched_cell_id": self.matched_cell_id,
            "matched_depth": self.matched_depth,
        }


def _range_stratum(radius: float) -> int:
    for index, (lower, upper) in enumerate(RANGE_BOUNDS_M):
        if lower <= radius < upper:
            return index
    raise ValueError(f"VRH return-group range is outside metrics: {radius}")


def build_target_return_groups(
    support: VRHSupport,
    targets: CanonicalTargets,
) -> list[TargetReturnGroup]:
    """Build deterministic same-subray first/later target groups."""

    _, a2, e2, in_support = assign_polar_to_support(support, targets.polar_rae)
    ray_id = a2 * support.elevation_axis.count + e2
    groups: list[TargetReturnGroup] = []
    for ray in np.unique(ray_id[in_support]):
        rows = np.flatnonzero(in_support & (ray_id == ray))
        order = np.lexsort(
            (targets.canonical_target_id[rows], targets.polar_rae[rows, 0])
        )
        rows = rows[order]
        start = 0
        group_index = 0
        while start < rows.size:
            stop = start + 1
            first_range = float(targets.polar_rae[rows[start], 0])
            while stop < rows.size:
                next_range = float(targets.polar_rae[rows[stop], 0])
                if next_range - first_range >= GROUP_SPAN_M:
                    break
                stop += 1
            members = rows[start:stop]
            weight = np.maximum(targets.confidence[members], 0.0)
            weight_sum = float(weight.sum())
            ranges = targets.polar_rae[members, 0]
            representative = (
                float(np.average(ranges, weights=weight))
                if weight_sum > 0.0
                else float(ranges.mean())
            )
            groups.append(
                TargetReturnGroup(
                    ray_id=int(ray),
                    group_index=group_index,
                    return_class=0 if group_index == 0 else 1,
                    range_stratum=_range_stratum(representative),
                    representative_range_m=representative,
                    group_weight=weight_sum,
                    canonical_target_ids=tuple(
                        int(value) for value in targets.canonical_target_id[members]
                    ),
                )
            )
            group_index += 1
            start = stop
    groups.sort(key=lambda item: (item.ray_id, item.group_index))
    return groups


def _weighted_integer_cost(weight: float, distance_m: float) -> int:
    weight_key = int(np.rint(1.0e9 * weight))
    distance_key = int(np.rint(1.0e6 * distance_m))
    if weight_key < 0 or distance_key < 0:
        raise ValueError("VRH matching cost cannot be negative")
    return weight_key * distance_key


def _match_one_ray(
    groups: list[TargetReturnGroup],
    event_rows: np.ndarray,
    export: ExportResult,
) -> _DPSolution:
    event_rows = np.asarray(event_rows, dtype=np.int64)
    if event_rows.size:
        order = np.lexsort(
            (export.selected_cell_id[event_rows], export.selected_rae[event_rows, 0])
        )
        event_rows = event_rows[order]

    @lru_cache(maxsize=None)
    def solve(group_index: int, event_index: int, previously_matched: bool) -> _DPSolution:
        if group_index == len(groups):
            return _DPSolution(0, 0, (), ())
        group = groups[group_index]
        unmatched_suffix = solve(group_index + 1, event_index, previously_matched)
        unmatched = _DPSolution(
            integer_cost=unmatched_suffix.integer_cost
            + _weighted_integer_cost(group.group_weight, UNMATCHED_DISTANCE_M),
            matched_count=unmatched_suffix.matched_count,
            matched_cell_sequence=unmatched_suffix.matched_cell_sequence,
            assignments=((UNMATCHED_DISTANCE_M, -1),) + unmatched_suffix.assignments,
        )
        candidates = [unmatched]
        if event_index < event_rows.size:
            candidates.append(solve(group_index, event_index + 1, previously_matched))
            row = int(event_rows[event_index])
            depth = int(export.selected_depth[row])
            eligible = (
                (group.return_class == 0 and depth == 1)
                or (group.return_class == 1 and depth >= 2 and previously_matched)
            )
            if eligible:
                distance = abs(
                    group.representative_range_m
                    - float(export.selected_rae[row, 0])
                )
                suffix = solve(group_index + 1, event_index + 1, True)
                cell_id = int(export.selected_cell_id[row])
                candidates.append(
                    _DPSolution(
                        integer_cost=suffix.integer_cost
                        + _weighted_integer_cost(group.group_weight, distance),
                        matched_count=suffix.matched_count + 1,
                        matched_cell_sequence=(cell_id,)
                        + suffix.matched_cell_sequence,
                        assignments=((distance, row),) + suffix.assignments,
                    )
                )
        return min(candidates, key=lambda item: item.ordering_key)

    return solve(0, 0, False)


def _verify_assignments_independently(
    groups: list[TargetReturnGroup],
    columns: dict[str, np.ndarray],
    export: ExportResult,
    expected_integer_cost: int,
) -> dict[str, bool]:
    group_lookup = {(group.ray_id, group.group_index): group for group in groups}
    event_lookup = {
        int(cell_id): index
        for index, cell_id in enumerate(export.selected_cell_id)
    }
    used_cells: set[int] = set()
    prior_match: dict[int, bool] = {}
    prior_event_range: dict[int, float] = {}
    observed_cost = 0
    checks = {
        "group_identity": True,
        "same_ray": True,
        "depth_eligibility": True,
        "prior_match_for_later": True,
        "strict_event_range_order": True,
        "event_nonreuse": True,
        "assigned_distance": True,
        "integer_weighted_cost": True,
    }
    for row in range(columns["ray_id"].size):
        ray = int(columns["ray_id"][row])
        group_index = int(columns["group_index"][row])
        group = group_lookup.get((ray, group_index))
        if group is None:
            checks["group_identity"] = False
            continue
        distance = float(columns["assigned_distance_m"][row])
        cell_id = int(columns["matched_cell_id"][row])
        depth = int(columns["matched_depth"][row])
        if cell_id < 0:
            checks["assigned_distance"] &= distance == UNMATCHED_DISTANCE_M
            checks["depth_eligibility"] &= depth == -1
        else:
            event_row = event_lookup.get(cell_id)
            if event_row is None:
                checks["same_ray"] = False
                continue
            event_ray = cell_id // export.range_count
            checks["same_ray"] &= event_ray == ray
            event_depth = int(export.selected_depth[event_row])
            checks["depth_eligibility"] &= depth == event_depth
            if group.return_class == 0:
                checks["depth_eligibility"] &= event_depth == 1
            else:
                checks["depth_eligibility"] &= event_depth >= 2
                checks["prior_match_for_later"] &= prior_match.get(ray, False)
            event_range = float(export.selected_rae[event_row, 0])
            if ray in prior_event_range:
                checks["strict_event_range_order"] &= (
                    event_range > prior_event_range[ray]
                )
            expected_distance = abs(group.representative_range_m - event_range)
            checks["assigned_distance"] &= distance == expected_distance
            checks["event_nonreuse"] &= cell_id not in used_cells
            used_cells.add(cell_id)
            prior_match[ray] = True
            prior_event_range[ray] = event_range
        observed_cost += _weighted_integer_cost(group.group_weight, distance)
    checks["integer_weighted_cost"] = observed_cost == expected_integer_cost
    return checks


def evaluate_first_later_returns(
    groups: list[TargetReturnGroup],
    export: ExportResult,
) -> ReturnMatchResult:
    """Run the frozen same-ray, order-preserving first/later DP."""

    event_ray = export.selected_cell_id.astype(np.int64) // export.range_count
    rows: list[tuple[int, int, int, int, float, float, float, int, int]] = []
    assignments_by_ray: dict[str, dict[str, Any]] = {}
    total_integer_cost = 0
    for ray in sorted({group.ray_id for group in groups}):
        ray_groups = [group for group in groups if group.ray_id == ray]
        event_rows = np.flatnonzero(event_ray == ray)
        solution = _match_one_ray(ray_groups, event_rows, export)
        total_integer_cost += solution.integer_cost
        if len(solution.assignments) != len(ray_groups):
            raise AssertionError("VRH DP did not assign every target group")
        matched_cells: list[int] = []
        matched_ranges: list[float] = []
        assignment_report: list[dict[str, Any]] = []
        for group, (distance, event_row) in zip(ray_groups, solution.assignments):
            if event_row >= 0:
                cell_id = int(export.selected_cell_id[event_row])
                depth = int(export.selected_depth[event_row])
                event_range = float(export.selected_rae[event_row, 0])
                matched_cells.append(cell_id)
                matched_ranges.append(event_range)
            else:
                cell_id = -1
                depth = -1
                event_range = None
            rows.append(
                (
                    group.ray_id,
                    group.group_index,
                    group.return_class,
                    group.range_stratum,
                    group.representative_range_m,
                    group.group_weight,
                    float(distance),
                    cell_id,
                    depth,
                )
            )
            assignment_report.append(
                {
                    "group_index": group.group_index,
                    "return_class": "first" if group.return_class == 0 else "later",
                    "representative_range_m": group.representative_range_m,
                    "group_weight": group.group_weight,
                    "assigned_distance_m": float(distance),
                    "matched_cell_id": cell_id,
                    "matched_depth": depth,
                    "matched_event_range_m": event_range,
                }
            )
        if len(matched_cells) != len(set(matched_cells)):
            raise AssertionError("VRH DP reused an accepted event")
        if matched_ranges and not np.all(np.diff(matched_ranges) > 0.0):
            raise AssertionError("VRH DP matched events out of strict range order")
        assignments_by_ray[str(ray)] = {
            "target_group_count": len(ray_groups),
            "accepted_event_count": int(event_rows.size),
            "integer_weighted_cost": solution.integer_cost,
            "matched_group_count": solution.matched_count,
            "assignments": assignment_report,
        }

    if rows:
        values = list(zip(*rows))
    else:
        values = [()] * len(RETURN_ASSIGNMENT_SCHEMA)
    columns = {
        name: np.ascontiguousarray(value, dtype=dtype)
        for (name, dtype), value in zip(RETURN_ASSIGNMENT_SCHEMA, values)
    }
    digest = canonical_columns_digest(
        kind="vrh_f0_return_assignments_v1",
        schema=RETURN_ASSIGNMENT_SCHEMA,
        columns=columns,
        extra_header={"arm": export.arm, "selected_set_sha256": export.selected_set_sha256},
    )
    assignment_checks = _verify_assignments_independently(
        groups,
        columns,
        export,
        total_integer_cost,
    )
    if not all(assignment_checks.values()):
        raise AssertionError(f"VRH independent assignment replay failed: {assignment_checks}")
    class_metrics: dict[str, dict[str, Any]] = {}
    all_classes_passed = True
    for stratum, range_label in enumerate(RANGE_LABELS):
        for return_class, return_label in ((0, "first"), (1, "later")):
            mask = (columns["range_stratum"] == stratum) & (
                columns["return_class"] == return_class
            )
            if not bool(mask.any()):
                continue
            weights = columns["group_weight"][mask]
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                raise ValueError(
                    f"VRH nonempty {range_label}/{return_label} class has zero weight"
                )
            distance = columns["assigned_distance_m"][mask]
            completeness = float(np.average(distance, weights=weights))
            recall = float(np.average(distance <= 1.0, weights=weights))
            passed = (
                completeness <= RETURN_COMPLETENESS_GATE_M
                and recall >= RETURN_RECALL_GATE
            )
            all_classes_passed &= passed
            class_metrics[f"{range_label}_{return_label}"] = {
                "group_count": int(mask.sum()),
                "effective_weight": weight_sum,
                "completeness_mean_distance_m": completeness,
                "recall_1m": recall,
                "passed": passed,
            }
    report = {
        "assignment_sha256": digest,
        "target_group_count": len(rows),
        "first_group_count": int((columns["return_class"] == 0).sum()),
        "later_group_count": int((columns["return_class"] == 1).sum()),
        "matched_group_count": int((columns["matched_cell_id"] >= 0).sum()),
        "unmatched_group_count": int((columns["matched_cell_id"] < 0).sum()),
        "integer_weighted_cost": total_integer_cost,
        "assignment_replay_checks": assignment_checks,
        "monotone": assignment_checks["strict_event_range_order"],
        "event_nonreuse": assignment_checks["event_nonreuse"],
        "unmatched_distance_m": UNMATCHED_DISTANCE_M,
        "class_metrics": class_metrics,
        "all_nonempty_classes_passed": all_classes_passed,
        "assignments_by_ray": assignments_by_ray,
    }
    return ReturnMatchResult(
        **columns,
        selected_set_sha256=export.selected_set_sha256,
        digest_sha256=digest,
        report=report,
    )


def selected_events_inside_cell_interiors(
    support: VRHSupport,
    export: ExportResult,
) -> bool:
    cell_id = export.selected_cell_id.astype(np.int64)
    range_count = support.range_axis.count
    elevation_count = support.elevation_axis.count
    r2 = cell_id % range_count
    ray = cell_id // range_count
    e2 = ray % elevation_count
    a2 = ray // elevation_count
    return bool(
        np.all(
            (export.selected_rae[:, 0] >= support.range_axis.lower_interior[r2])
            & (export.selected_rae[:, 0] <= support.range_axis.upper_interior[r2])
            & (export.selected_rae[:, 1] >= support.azimuth_axis.lower_interior[a2])
            & (export.selected_rae[:, 1] <= support.azimuth_axis.upper_interior[a2])
            & (export.selected_rae[:, 2] >= support.elevation_axis.lower_interior[e2])
            & (export.selected_rae[:, 2] <= support.elevation_axis.upper_interior[e2])
        )
    )


def write_return_assignments(path: Path, result: ReturnMatchResult, arm: str) -> str:
    return write_canonical_columns(
        path,
        kind="vrh_f0_return_assignments_v1",
        schema=RETURN_ASSIGNMENT_SCHEMA,
        columns=result.columns(),
        extra_header={
            "arm": arm,
            "selected_set_sha256": result.selected_set_sha256,
        },
    )
