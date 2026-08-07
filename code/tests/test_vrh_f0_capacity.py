from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from cube_dense.kradar import KRadarAxes, load_axes, polar_to_cartesian
from eval.vrh_f0_decode import (
    CANONICAL_STREAM_SCHEMA,
    CanonicalStreams,
    ExposureTrace,
    ExportResult,
    decode_canonical_streams,
    canonical_streams_digest,
    export_flat_control,
    export_sequential_frontier,
)
from eval.vrh_f0_fit import (
    canonicalize_targets,
    fit_gt_oracle_marks,
    load_target_only,
    write_fit_sidecar,
)
from eval.vrh_f0_metrics import (
    TargetReturnGroup,
    build_target_return_groups,
    evaluate_first_later_returns,
)
from eval.vrh_f0_support import (
    A2_COUNT,
    CELL_COUNT,
    E2_COUNT,
    HAZARD_EPSILON,
    HAZARD_TAU_M,
    NATURAL_RANGE_UPPER_M,
    R2_COUNT,
    SUBRAY_COUNT,
    ModelMarks,
    SplitAxis,
    SupportCommit,
    VRHSupport,
    assign_half_open,
    build_vrh_support,
    canonicalize_azimuth,
    commit_support,
    model_marks_interior_report,
    stable_cell_ids,
    write_model_marks,
)
from scripts.preflight_vrh_f0_capacity import (
    begin_transaction,
    publish_transaction,
    terminal_status,
)


def split_axis(edges: list[float]) -> SplitAxis:
    edge = np.asarray(edges, dtype=np.float64)
    lower = edge[:-1].copy()
    upper = edge[1:].copy()
    return SplitAxis(
        centers=0.5 * (lower + upper),
        edges=edge,
        lower=lower,
        upper=upper,
        lower_interior=np.nextafter(lower, upper),
        upper_interior=np.nextafter(upper, lower),
    )


def synthetic_support(
    range_edges: list[float],
    azimuth_edges: list[float] | None = None,
    elevation_edges: list[float] | None = None,
) -> VRHSupport:
    return VRHSupport(
        range_axis=split_axis(range_edges),
        azimuth_axis=split_axis(azimuth_edges or [-0.1, 0.1]),
        elevation_axis=split_axis(elevation_edges or [-0.1, 0.1]),
        digest_sha256="ab" * 32,
        schema_header_sha256="cd" * 32,
    )


def model_marks(
    support: VRHSupport,
    hazards: np.ndarray,
    *,
    fitted_radius: np.ndarray | None = None,
) -> ModelMarks:
    hazards = np.asarray(hazards, dtype="<f8")
    shape = (support.ray_count, support.range_axis.count)
    hazards = np.broadcast_to(hazards, shape).copy()
    delta_r = np.zeros(shape, dtype="<f8")
    if fitted_radius is not None:
        delta_r[:] = np.asarray(fitted_radius, dtype=np.float64) - support.range_axis.centers
    return ModelMarks(
        base_hazard=hazards,
        delta_r=delta_r,
        delta_a=np.zeros(shape, dtype="<f8"),
        delta_e=np.zeros(shape, dtype="<f8"),
        opaque_distance_priority_key=np.zeros(shape, dtype="<i8"),
        confidence=np.ones(shape, dtype="<f8"),
    )


def manual_streams(
    *,
    range_count: int,
    ray_count: int,
    rows: list[tuple[int, int, int, int, int, int, float, float, float, float]],
) -> CanonicalStreams:
    rows = sorted(rows, key=lambda row: (row[0], row[1]))
    values = list(zip(*rows)) if rows else [()] * len(CANONICAL_STREAM_SCHEMA)
    arrays = {
        name: np.asarray(value, dtype=dtype)
        for (name, dtype), value in zip(CANONICAL_STREAM_SCHEMA, values)
    }
    offsets = np.zeros(ray_count + 1, dtype="<u8")
    for ray in range(ray_count):
        offsets[ray + 1] = offsets[ray] + int((arrays["ray_id"] == ray).sum())
    streams = CanonicalStreams(
        **arrays,
        ray_offsets=offsets,
        range_count=range_count,
        elevation_count=1,
        support_sha256="ab" * 32,
        digest_sha256="",
    )
    object.__setattr__(streams, "digest_sha256", canonical_streams_digest(streams))
    for values in (*streams.columns().values(), streams.ray_offsets):
        values.flags.writeable = False
    return streams


def manual_export(
    radius: list[float],
    depth: list[int],
    *,
    cell_id: list[int] | None = None,
    range_count: int = 128,
) -> ExportResult:
    count = len(radius)
    cells = np.asarray(cell_id or list(range(count)), dtype="<u4")
    radii = np.asarray(radius, dtype=np.float64)
    rae = np.column_stack((radii, np.zeros(count), np.zeros(count)))
    xyz = polar_to_cartesian(rae[:, 0], rae[:, 1], rae[:, 2])
    trace = ExposureTrace(
        pop_index=np.asarray([], dtype="<u8"),
        ray_id=np.asarray([], dtype="<u2"),
        stream_index=np.asarray([], dtype="<u2"),
        cell_id=np.asarray([], dtype="<u4"),
        accepted=np.asarray([], dtype="<u1"),
        digest_sha256="11" * 32,
    )
    return ExportResult(
        arm="unit",
        streams_sha256="22" * 32,
        range_count=range_count,
        selected_event_rows=np.arange(count, dtype=np.int64),
        selected_cell_id=cells,
        selected_depth=np.asarray(depth, dtype="<u2"),
        selected_rae=rae,
        selected_xyz=xyz,
        selected_confidence=np.ones(count, dtype="<f8"),
        trace=trace,
        accepted_cell_id_sha256="44" * 32,
        selected_set_sha256="33" * 32,
        report={},
    )


def target_from_ranges(ranges: list[float], confidence: float = 1.0) -> np.ndarray:
    radius = np.asarray(ranges, dtype=np.float64)
    xyz = polar_to_cartesian(radius, np.zeros_like(radius), np.zeros_like(radius))
    return np.column_stack((xyz, np.full(radius.size, confidence)))


@pytest.fixture(scope="session")
def formal_support() -> VRHSupport:
    resources = os.environ.get("VRH_KRADAR_RESOURCES")
    if resources is None:
        pytest.skip("formal K-Radar resources are bound only on the H200 run")
    return build_vrh_support(load_axes(Path(resources)))


def test_formal_support_uses_real_kradar_axes_when_bound(
    formal_support: VRHSupport,
) -> None:
    assert formal_support.shape_rae == (R2_COUNT, A2_COUNT, E2_COUNT)
    assert formal_support.ray_count == SUBRAY_COUNT
    assert formal_support.cell_count == CELL_COUNT
    assert formal_support.range_axis.edges[-1] == NATURAL_RANGE_UPPER_M
    assert formal_support.range_axis.edges[0] == 0.0
    assert np.all(
        formal_support.range_axis.lower_interior > formal_support.range_axis.lower
    )
    assert np.all(
        formal_support.range_axis.upper_interior < formal_support.range_axis.upper
    )
    assert not formal_support.range_axis.centers.flags.writeable


def test_stable_cell_id_order_is_ray_major_then_range() -> None:
    assert int(stable_cell_ids(0, 0, 0)) == 0
    assert int(stable_cell_ids(0, 0, 1)) == 1
    assert int(stable_cell_ids(0, 1, 0)) == R2_COUNT
    assert int(stable_cell_ids(1, 0, 0)) == E2_COUNT * R2_COUNT
    assert int(stable_cell_ids(A2_COUNT - 1, E2_COUNT - 1, R2_COUNT - 1)) == CELL_COUNT - 1
    with pytest.raises(ValueError):
        stable_cell_ids(-1, 0, 0)
    with pytest.raises(ValueError):
        stable_cell_ids(0, E2_COUNT, 0)


def test_canonical_azimuth_preserves_in_range_float64_bits() -> None:
    values = np.asarray([-1.0, -0.0, 0.0, 0.7], dtype=np.float64)
    canonical = canonicalize_azimuth(values)

    np.testing.assert_array_equal(canonical.view(np.uint64), values.view(np.uint64))


def test_half_open_assignment_and_final_angular_edge() -> None:
    axis = split_axis([-1.0, 0.0, 1.0])
    index, inside = assign_half_open(
        np.asarray([-1.0, 0.0, 1.0]), axis, final_edge_closed=True
    )
    np.testing.assert_array_equal(index, [0, 1, 1])
    np.testing.assert_array_equal(inside, [True, True, True])
    _, range_inside = assign_half_open(
        np.asarray([-1.0, 1.0]), axis, final_edge_closed=False
    )
    np.testing.assert_array_equal(range_inside, [True, False])


def test_target_loader_rejects_forged_commit_and_has_target_only_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "target.npz"
    target = target_from_ranges([1.0, 2.0]).astype(np.float32)
    np.savez(
        path,
        target_xyz_confidence=target,
        target_rae_index=np.zeros((2, 3), dtype=np.int64),
        cfar_xyzd_power_snr=np.ones((3, 6), dtype=np.float32),
        cube_drae=np.ones((2,), dtype=np.float32),
    )
    source = inspect.getsource(load_target_only)
    assert 'cache["target_xyz_confidence"]' in source
    assert 'cache["cfar_xyzd_power_snr"]' not in source
    assert 'cache["cube_drae"]' not in source
    with pytest.raises(TypeError):
        SupportCommit(
            digest_sha256="ab" * 32,
            cell_count=CELL_COUNT,
            ray_count=SUBRAY_COUNT,
        )
    with pytest.raises(TypeError):
        load_target_only(path, support_commit=None)  # type: ignore[arg-type]


def test_target_loader_reads_after_real_support_commit(
    tmp_path: Path,
    formal_support: VRHSupport,
) -> None:
    path = tmp_path / "target.npz"
    target = target_from_ranges([1.0, 2.0]).astype(np.float32)
    np.savez(path, target_xyz_confidence=target)

    loaded = load_target_only(path, support_commit=commit_support(formal_support))

    np.testing.assert_array_equal(loaded, target.astype(np.float64))


def test_fitter_is_target_permutation_and_exact_duplicate_invariant() -> None:
    support = synthetic_support([0.0, 1.0, 2.0], [-0.2, 0.0, 0.2])
    target = target_from_ranges([0.6, 1.6])
    first = fit_gt_oracle_marks(support, target, query_chunk_size=2)
    second = fit_gt_oracle_marks(support, target[::-1].copy(), query_chunk_size=3)
    duplicate = fit_gt_oracle_marks(
        support, np.concatenate((target, target[:1]), axis=0), query_chunk_size=4
    )

    assert first.report["model_marks_sha256"] == second.report["model_marks_sha256"]
    assert first.report["model_marks_sha256"] == duplicate.report["model_marks_sha256"]
    np.testing.assert_array_equal(
        first.model_marks.base_hazard, second.model_marks.base_hazard
    )


def test_fitter_tie_uses_smallest_canonical_target_id() -> None:
    support = synthetic_support([0.5, 1.5], [-0.2, 0.2], [-0.1, 0.1])
    target = np.asarray(
        [[1.0, -0.1, 0.0, 0.7], [1.0, 0.1, 0.0, 0.8]], dtype=np.float64
    )
    result = fit_gt_oracle_marks(support, target, query_chunk_size=1)

    assert int(result.fit_sidecar.nearest_target_id[0, 0]) == 0
    expected = np.clip(
        np.exp(-result.fit_sidecar.fitted_distance_m[0, 0] / HAZARD_TAU_M),
        HAZARD_EPSILON,
        1.0 - HAZARD_EPSILON,
    )
    assert result.model_marks.base_hazard[0, 0] == expected


def test_model_marks_and_sidecar_are_distinct_canonical_files(tmp_path: Path) -> None:
    support = synthetic_support([0.0, 1.0, 2.0])
    result = fit_gt_oracle_marks(support, target_from_ranges([0.5]), query_chunk_size=2)
    marks_path = tmp_path / "marks.bin"
    sidecar_path = tmp_path / "sidecar.bin"
    marks_hash = write_model_marks(marks_path, result.model_marks, support)
    sidecar_hash = write_fit_sidecar(sidecar_path, result.fit_sidecar, support)

    assert marks_hash == result.report["model_marks_sha256"]
    assert sidecar_hash == result.report["fit_sidecar_sha256"]
    assert marks_hash != sidecar_hash
    assert marks_path.read_bytes()[:80] != sidecar_path.read_bytes()[:80]
    assert model_marks_interior_report(support, result.model_marks)[
        "all_marks_inside_source_cell_interiors"
    ]


@pytest.mark.parametrize("return_count", (0, 1, 2, 7, 11))
def test_decoder_supports_frozen_variable_return_counts(return_count: int) -> None:
    support = synthetic_support(list(np.arange(13, dtype=np.float64)))
    hazards = np.full((1, 12), HAZARD_EPSILON, dtype=np.float64)
    hazards[0, :return_count] = 0.9
    streams = decode_canonical_streams(support, model_marks(support, hazards))

    assert streams.event_count == return_count
    np.testing.assert_array_equal(
        streams.depth, np.arange(1, return_count + 1, dtype=np.uint16)
    )


def test_last_bin_ties_stop_and_event_wins() -> None:
    support = synthetic_support([0.0, 1.0, 2.0])
    hazards = np.asarray([[HAZARD_EPSILON, 0.5]], dtype=np.float64)
    streams = decode_canonical_streams(support, model_marks(support, hazards))

    assert streams.event_count == 1
    assert int(streams.cell_id[0]) == 1


def test_refractory_suppresses_4p999cm_and_allows_5cm() -> None:
    support = synthetic_support([0.9, 1.02, 1.08, 1.2])
    hazards = np.asarray([[0.9, 0.9, HAZARD_EPSILON]], dtype=np.float64)
    suppressed = decode_canonical_streams(
        support,
        model_marks(support, hazards, fitted_radius=np.asarray([1.0, 1.04999, 1.15])),
    )
    allowed = decode_canonical_streams(
        support,
        model_marks(support, hazards, fitted_radius=np.asarray([1.0, 1.05, 1.15])),
    )

    assert suppressed.event_count == 1
    assert allowed.event_count == 2


def test_decoder_module_does_not_import_fitter_or_sidecar() -> None:
    source = inspect.getsourcefile(decode_canonical_streams)
    assert source is not None
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)

    assert not any(name.endswith("vrh_f0_fit") for name in imports)
    assert not any(name.endswith("vrh_f0_metrics") for name in imports)
    assert tuple(inspect.signature(decode_canonical_streams).parameters) == (
        "support",
        "model_marks",
    )


def test_exporter_rejects_mutated_canonical_stream_bytes() -> None:
    streams = manual_streams(
        range_count=1,
        ray_count=1,
        rows=[(0, 0, 0, 1, -1, 0, 1.0, 1.0, 0.0, 0.0)],
    )
    streams.r.flags.writeable = True
    streams.r[0] = 2.0

    with pytest.raises(ValueError, match="changed after decoding"):
        export_sequential_frontier(streams, exact_count=1)


def test_accepted_cell_hash_is_arm_independent_for_identical_selection() -> None:
    streams = manual_streams(
        range_count=1,
        ray_count=1,
        rows=[(0, 0, 0, 1, -1, 0, 1.0, 1.0, 0.0, 0.0)],
    )
    decision = export_sequential_frontier(streams, exact_count=1)
    control = export_flat_control(streams, exact_count=1)

    assert decision.accepted_cell_id_sha256 == control.accepted_cell_id_sha256
    assert decision.selected_set_sha256 != control.selected_set_sha256


def test_frontier_blocking_changes_selected_hash_against_flat_control() -> None:
    streams = manual_streams(
        range_count=2,
        ray_count=2,
        rows=[
            (0, 0, 0, 1, -100, 0, 1.0, 1.0, 0.0, 0.0),
            (0, 1, 1, 2, -1, 0, 1.0, 2.0, 0.0, 0.0),
            (1, 0, 2, 1, -50, 0, 1.0, 3.0, 0.0, 0.0),
        ],
    )
    decision = export_sequential_frontier(streams, exact_count=1)
    control = export_flat_control(streams, exact_count=1)

    assert int(decision.selected_cell_id[0]) == 2
    assert int(control.selected_cell_id[0]) == 1
    assert decision.accepted_cell_id_sha256 != control.accepted_cell_id_sha256


def test_rejected_frontier_event_is_consumed_and_reveals_next() -> None:
    streams = manual_streams(
        range_count=2,
        ray_count=2,
        rows=[
            (0, 0, 0, 1, -10, 0, 1.0, 1.0, 0.0, 0.0),
            (0, 1, 1, 2, -2, 0, 1.0, 2.0, 0.0, 0.0),
            (1, 0, 2, 1, -1, 0, 1.0, 1.0, 0.0, 0.0),
        ],
    )
    result = export_sequential_frontier(streams, exact_count=2)

    np.testing.assert_array_equal(result.selected_cell_id, [2, 1])
    assert result.report["rejected_minimum_distance_count"] == 1
    assert result.trace.accepted.tolist() == [1, 0, 1]


def test_online_spacing_rejects_4p999cm_and_accepts_5cm() -> None:
    too_close = manual_streams(
        range_count=2,
        ray_count=1,
        rows=[
            (0, 0, 0, 1, -1, 0, 1.0, 1.0, 0.0, 0.0),
            (0, 1, 1, 2, -2, 0, 1.0, 1.04999, 0.0, 0.0),
        ],
    )
    exact = manual_streams(
        range_count=2,
        ray_count=1,
        rows=[
            (0, 0, 0, 1, -1, 0, 1.0, 1.0, 0.0, 0.0),
            (0, 1, 1, 2, -2, 0, 1.0, 1.05, 0.0, 0.0),
        ],
    )

    assert export_flat_control(too_close, exact_count=2).selected_count == 1
    assert export_flat_control(exact, exact_count=2).selected_count == 2


def test_span_grouping_for_zero_four_and_eight_centimeters() -> None:
    support = synthetic_support([0.0, 20.0])
    targets = canonicalize_targets(target_from_ranges([10.0, 10.04, 10.08]))
    groups = build_target_return_groups(support, targets)

    assert len(groups) == 2
    assert groups[0].canonical_target_ids == (0, 1)
    assert groups[0].return_class == 0
    assert groups[1].canonical_target_ids == (2,)
    assert groups[1].return_class == 1


def test_first_later_dp_requires_depth_and_same_path_predecessor() -> None:
    groups = [
        TargetReturnGroup(0, 0, 0, 0, 10.0, 1.0, (0,)),
        TargetReturnGroup(0, 1, 1, 0, 20.0, 1.0, (1,)),
    ]
    passed = evaluate_first_later_returns(
        groups,
        manual_export([10.1, 20.1], [1, 2]),
    )
    first_only = evaluate_first_later_returns(
        groups,
        manual_export([10.1], [1]),
    )
    unrelated_depth_two = evaluate_first_later_returns(
        groups,
        manual_export([20.1], [2]),
    )

    assert passed.report["all_nonempty_classes_passed"] is True
    assert passed.report["event_nonreuse"] is True
    assert first_only.report["class_metrics"]["range_0_30m_later"]["recall_1m"] == 0.0
    assert unrelated_depth_two.report["matched_group_count"] == 0


def test_nonempty_zero_weight_return_class_is_implementation_invalid() -> None:
    groups = [TargetReturnGroup(0, 0, 0, 0, 10.0, 0.0, (0,))]
    with pytest.raises(ValueError, match="zero weight"):
        evaluate_first_later_returns(groups, manual_export([10.0], [1]))


def test_terminal_routing_separates_capacity_lattice_implementation_and_resource() -> None:
    assert terminal_status(
        implementation_valid=True,
        resource_valid=True,
        all_decision_frames_passed=True,
        renewal_activity=True,
        renewal_utility=True,
    ) == "vrh_f0_capacity_passed"
    assert terminal_status(
        implementation_valid=True,
        resource_valid=True,
        all_decision_frames_passed=True,
        renewal_activity=False,
        renewal_utility=False,
    ) == "vrh_f0_lattice_only"
    assert terminal_status(
        implementation_valid=True,
        resource_valid=True,
        all_decision_frames_passed=False,
        renewal_activity=True,
        renewal_utility=False,
    ) == "vrh_f0_capacity_no_go"
    assert terminal_status(
        implementation_valid=False,
        resource_valid=True,
        all_decision_frames_passed=False,
        renewal_activity=False,
        renewal_utility=False,
    ) == "vrh_f0_implementation_invalid"
    assert terminal_status(
        implementation_valid=False,
        resource_valid=False,
        all_decision_frames_passed=False,
        renewal_activity=False,
        renewal_utility=False,
    ) == "vrh_f0_resource_invalid"


def test_atomic_transaction_refuses_overwrite_and_publishes_once(tmp_path: Path) -> None:
    output = tmp_path / "terminal"
    staging = begin_transaction(output)
    path = publish_transaction(
        staging,
        output,
        {"status": "unit", "passed": False, "runtime": {"elapsed_seconds": 0.0}},
    )

    assert path.is_file()
    assert path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        begin_transaction(output)


def test_formal_builder_rejects_wrong_axis_cardinality() -> None:
    axes = KRadarAxes(
        doppler_mps=np.linspace(-1.0, 1.0, 64),
        range_m=np.linspace(0.5, 100.0, 8),
        azimuth_rad=np.linspace(-0.5, 0.5, 7),
        elevation_rad=np.linspace(-0.2, 0.2, 5),
    )
    with pytest.raises(ValueError, match="support changed"):
        build_vrh_support(axes)
