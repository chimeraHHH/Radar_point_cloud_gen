from __future__ import annotations

import math

import numpy as np

from eval.stda_f0_structure import (
    OUTPUT_RANGE_UPPER_M,
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_SHA256,
    RETURN_FIRST,
    RETURN_LATER,
    UNMATCHED_EVENT_ID,
    StructuralDomain,
    build_target_return_groups,
    canonicalize_positive_target_atoms,
    evaluate_structure,
    freeze_structural_inputs,
)


def _point(radius: float, azimuth: float = 0.0, elevation: float = 0.0) -> np.ndarray:
    cosine = math.cos(elevation)
    return np.asarray(
        [
            radius * cosine * math.cos(azimuth),
            radius * cosine * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype="<f4",
    )


def _target(points: list[np.ndarray], confidence: float = 1.0) -> np.ndarray:
    xyz = np.asarray(points, dtype="<f4")
    weights = np.full((len(points), 1), confidence, dtype="<f4")
    return np.concatenate((xyz, weights), axis=1)


def _one_ray_domain() -> StructuralDomain:
    return StructuralDomain.from_edges((0.0, 1.5), (-0.2, 0.2))


def test_source_binds_frozen_protocol_and_commit() -> None:
    assert PROTOCOL_FREEZE_COMMIT == "eb839e1e806c44dd1668051085a307faf4fe83a0"
    assert PROTOCOL_SHA256 == (
        "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
    )


def test_multiple_events_in_one_subray_keep_ids_and_equal_range_depth_order() -> None:
    output = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [4.0, 3.0, 0.0],
            [6.0, 8.0, 0.0],
        ],
        dtype="<f4",
    )
    target = np.asarray(
        [
            [3.0, 4.0, 0.0, 1.0],
            [6.0, 8.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    result = evaluate_structure(
        freeze_structural_inputs(output, target, domain=_one_ray_domain())
    )

    assert result.structural_domain_valid is True
    assert len(result.output.events) == 3
    assert {event.ray_id for event in result.output.events} == {0}
    equal_range = [event for event in result.output.events if event.range_m == 5.0]
    assert len(equal_range) == 2
    assert [event.inferred_depth for event in equal_range] == [1, 2]
    assert len({event.event_id for event in equal_range}) == 2


def test_complete_assignment_tuple_breaks_equal_distance_event_tie() -> None:
    output = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [6.0, 8.0, 0.0],
            [8.0, 6.0, 0.0],
        ],
        dtype="<f4",
    )
    target = np.asarray(
        [
            [3.0, 4.0, 0.0, 1.0],
            [6.0, 8.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    result = evaluate_structure(
        freeze_structural_inputs(output, target, domain=_one_ray_domain())
    )

    ten_meter_events = sorted(
        event.event_id for event in result.output.events if event.range_m == 10.0
    )
    assert len(ten_meter_events) == 2
    assert result.assignments[1].matched_event_id == ten_meter_events[0]
    assert result.complete_assignment_tuple == tuple(
        assignment.matched_event_id for assignment in result.assignments
    )
    assert result.objective_key == (
        result.integer_cost,
        -result.matched_count,
        result.complete_assignment_tuple,
    )


def test_ordered_dp_skips_an_event_and_records_an_unmatched_group() -> None:
    output = np.asarray(
        [
            [4.0, 3.0, 0.0],
            [8.0, 6.0, 0.0],
            [16.0, 12.0, 0.0],
        ],
        dtype="<f4",
    )
    target = np.asarray(
        [
            [4.0, 3.0, 0.0, 1.0],
            [16.0, 12.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    result = evaluate_structure(
        freeze_structural_inputs(output, target, domain=_one_ray_domain())
    )

    assert [item.matched_depth for item in result.assignments] == [1, 3]
    skipped_event = next(event for event in result.output.events if event.inferred_depth == 2)
    assert skipped_event.event_id not in result.complete_assignment_tuple

    short_result = evaluate_structure(
        freeze_structural_inputs(output[:1], target, domain=_one_ray_domain())
    )
    assert short_result.assignments[0].matched_depth == 1
    assert short_result.assignments[1].matched_event_id == UNMATCHED_EVENT_ID
    assert short_result.assignments[1].matched_depth == -1
    assert short_result.assignments[1].assigned_distance_m == 120.0


def test_later_group_requires_a_previously_matched_first_depth() -> None:
    output = np.asarray([[16.0, 12.0, 0.0]], dtype="<f4")
    target = np.asarray(
        [
            [4.0, 3.0, 0.0, 1.0],
            [16.0, 12.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    result = evaluate_structure(
        freeze_structural_inputs(output, target, domain=_one_ray_domain())
    )

    assert result.assignments[0].return_class == RETURN_FIRST
    assert result.assignments[0].matched_depth == 1
    assert result.assignments[1].return_class == RETURN_LATER
    assert result.assignments[1].matched_event_id == UNMATCHED_EVENT_ID


def test_exact_first_anchored_grouping_uses_strict_five_centimeters() -> None:
    domain = StructuralDomain.from_edges((-0.2, 0.2), (-0.2, 0.2))
    target = np.asarray(
        [
            [10.00, 0.0, 0.0, 1.0],
            [10.04, 0.0, 0.0, 1.0],
            [10.08, 0.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    inputs = freeze_structural_inputs(
        np.empty((0, 3), dtype="<f4"),
        target,
        domain=domain,
    )
    groups = build_target_return_groups(canonicalize_positive_target_atoms(inputs))

    assert len(groups) == 2
    assert groups[0].canonical_target_ids == (0, 1)
    assert groups[0].return_class == RETURN_FIRST
    assert groups[1].canonical_target_ids == (2,)
    assert groups[1].return_class == RETURN_LATER

    exact_boundary_target = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.05, 0.0, 0.0, 1.0]],
        dtype="<f4",
    )
    boundary_inputs = freeze_structural_inputs(
        np.empty((0, 3), dtype="<f4"),
        exact_boundary_target,
        domain=domain,
    )
    boundary_groups = build_target_return_groups(
        canonicalize_positive_target_atoms(boundary_inputs)
    )
    assert len(boundary_groups) == 2


def test_all_six_range_by_return_classes_are_reported() -> None:
    domain = StructuralDomain.from_edges(
        (-0.8, -0.2, 0.2, 0.8),
        (-0.2, 0.2),
    )
    specifications = (
        (-0.5, 10.0, 12.0),
        (0.0, 40.0, 42.0),
        (0.5, 70.0, 72.0),
    )
    points = [
        _point(radius, azimuth)
        for azimuth, first, later in specifications
        for radius in (first, later)
    ]
    output = np.asarray(points, dtype="<f4")
    target = _target(points)
    result = evaluate_structure(
        freeze_structural_inputs(output, target, domain=domain)
    )

    expected = {
        "range_0_30_first",
        "range_0_30_later",
        "range_30_60_first",
        "range_30_60_later",
        "range_60_120_first",
        "range_60_120_later",
    }
    assert {metric.key for metric in result.class_metrics} == expected
    assert all(metric.applicable for metric in result.class_metrics)
    assert all(metric.group_count == 1 for metric in result.class_metrics)
    assert all(metric.recall_1m == 1.0 for metric in result.class_metrics)
    assert all(metric.passed is True for metric in result.class_metrics)
    assert result.matched_count == 6
    for digest in (
        result.output.digest_sha256,
        result.target.digest_sha256,
        result.groups_sha256,
        result.assignments_sha256,
        result.complete_mapping_sha256,
        result.class_metrics_sha256,
        result.result_sha256,
    ):
        assert len(digest) == 64


def test_output_and_target_range_and_angular_boundaries_are_fail_closed() -> None:
    domain = StructuralDomain.from_edges(
        (-0.5 * math.pi, 0.5 * math.pi),
        (-0.5 * math.pi, 0.5 * math.pi),
    )
    output = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [OUTPUT_RANGE_UPPER_M, 0.0, 0.0],
        ],
        dtype="<f4",
    )
    below_120 = np.nextafter(np.float32(120.0), np.float32(0.0))
    target = np.asarray([[below_120, 0.0, 0.0, 1.0]], dtype="<f4")
    valid = evaluate_structure(freeze_structural_inputs(output, target, domain=domain))

    assert valid.structural_domain_valid is True
    assert valid.evaluation_complete is True
    assert not valid.output.invalid_event_ids
    assert not valid.target.invalid_target_ids

    above_output = np.nextafter(
        np.float32(OUTPUT_RANGE_UPPER_M),
        np.float32(np.inf),
    )
    invalid_output = evaluate_structure(
        freeze_structural_inputs(
            np.asarray([[above_output, 0.0, 0.0]], dtype="<f4"),
            target,
            domain=domain,
        )
    )
    assert invalid_output.structural_domain_valid is False
    assert invalid_output.evaluation_complete is False
    assert invalid_output.output.invalid_event_ids == (0,)

    invalid_angle = evaluate_structure(
        freeze_structural_inputs(
            np.asarray([[-1.0, 0.0, 0.0]], dtype="<f4"),
            target,
            domain=domain,
        )
    )
    assert invalid_angle.structural_domain_valid is False
    assert invalid_angle.output.invalid_event_ids == (0,)

    invalid_target = evaluate_structure(
        freeze_structural_inputs(
            np.asarray([[1.0, 0.0, 0.0]], dtype="<f4"),
            np.asarray([[120.0, 0.0, 0.0, 1.0]], dtype="<f4"),
            domain=domain,
        )
    )
    assert invalid_target.structural_domain_valid is False
    assert invalid_target.target.invalid_target_ids == (0,)
    assert invalid_target.failure_reasons == (
        "positive_target_out_of_structural_domain",
    )


def test_target_and_export_row_permutations_preserve_canonical_result_hashes() -> None:
    domain = StructuralDomain.from_edges((-0.2, 0.2), (-0.2, 0.2))
    output = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype="<f4",
    )
    target = np.asarray(
        [
            [1.0, -0.0, 0.0, 0.25],
            [1.0, 0.0, -0.0, 0.75],
            [2.0, 0.0, 0.0, 1.0],
            [3.0, -0.0, -0.0, 0.0],
        ],
        dtype="<f4",
    )
    first = evaluate_structure(
        freeze_structural_inputs(output, target, domain=domain)
    )
    second = evaluate_structure(
        freeze_structural_inputs(
            output[[3, 2, 0, 1]],
            target[[2, 0, 3, 1]],
            domain=domain,
        )
    )

    assert first.raw_inputs_sha256 != second.raw_inputs_sha256
    assert first.output.duplicate_row_count == 1
    assert first.output.digest_sha256 == second.output.digest_sha256
    assert first.target.digest_sha256 == second.target.digest_sha256
    assert [atom.weight for atom in first.target.atoms] == [1.0, 1.0]
    assert first.canonical_inputs_sha256 == second.canonical_inputs_sha256
    assert first.groups_sha256 == second.groups_sha256
    assert first.assignments_sha256 == second.assignments_sha256
    assert first.complete_mapping_sha256 == second.complete_mapping_sha256
    assert first.class_metrics_sha256 == second.class_metrics_sha256
    assert first.result_sha256 == second.result_sha256
