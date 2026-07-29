import inspect

import numpy as np
import pytest

from eval.rb1_range_echo_oracle import (
    ELEVATION_RAY_COUNT,
    ORACLE_LABEL,
    RangeEchoCapacityError,
    RangeEchoOracleConfig,
    build_gt_aided_range_echo_candidates,
    geometry_endpoints,
    select_exact_range_echoes,
)
from scripts.preflight_rb1_range_echo import load_target_only


def axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.linspace(0.5, 127.5, 256, dtype=np.float64),
        np.linspace(-0.8, 0.8, 107, dtype=np.float64),
        np.linspace(-0.25, 0.25, 37, dtype=np.float64),
    )


def polar_point(
    radius: float,
    azimuth_index: int,
    elevation_index: int,
    *,
    confidence: float = 1.0,
) -> np.ndarray:
    _, azimuth_axis, elevation_axis = axes()
    azimuth = azimuth_axis[azimuth_index]
    elevation = elevation_axis[elevation_index]
    cosine = np.cos(elevation)
    return np.asarray(
        [
            radius * cosine * np.cos(azimuth),
            radius * cosine * np.sin(azimuth),
            radius * np.sin(elevation),
            confidence,
        ],
        dtype=np.float32,
    )


def quota_target() -> np.ndarray:
    rows = []
    for index in range(12):
        rows.append(polar_point(10.0, 20 + index, 18))
    for index in range(8):
        rows.append(polar_point(40.0, 20 + index, 18))
    for index in range(4):
        rows.append(polar_point(70.0, 20 + index, 18))
    return np.stack(rows)


def build(
    target: np.ndarray,
    config: RangeEchoOracleConfig,
):
    range_axis, azimuth_axis, elevation_axis = axes()
    return build_gt_aided_range_echo_candidates(
        target,
        range_axis_m=range_axis,
        azimuth_axis_rad=azimuth_axis,
        elevation_axis_rad=elevation_axis,
        config=config,
    )


def test_config_rejects_fewer_than_four_peaks_per_ray() -> None:
    with pytest.raises(ValueError, match="at least four"):
        RangeEchoOracleConfig(k=3)


def test_each_ray_is_capped_at_k_and_sub_bin_offsets_are_bounded() -> None:
    target = np.stack(
        [
            polar_point(5.0 + 2.0 * index, 53, 18)
            for index in range(10)
        ]
    )
    candidates = build(target, RangeEchoOracleConfig(k=4))
    ray_linear = (
        candidates.ray_indices_ae[:, 0] * ELEVATION_RAY_COUNT
        + candidates.ray_indices_ae[:, 1]
    )
    _, counts = np.unique(ray_linear, return_counts=True)

    assert counts.max() == 4
    assert candidates.report["occupied_ray_count"] == 1
    assert candidates.report["k_truncation"][
        "raw_gt_aided_echo_group_count"
    ] == 10
    assert candidates.report["k_truncation"]["rays_truncated"] == 1
    assert np.all(
        candidates.sub_bin_offsets_m
        >= candidates.sub_bin_lower_bounds_m - 1e-7
    )
    assert np.all(
        candidates.sub_bin_offsets_m
        <= candidates.sub_bin_upper_bounds_m + 1e-7
    )


def test_gt_aided_candidate_and_selection_are_stable_under_input_order() -> None:
    target = quota_target()
    config = RangeEchoOracleConfig(
        k=4,
        exact_count=24,
        range_quotas=(12, 8, 4),
    )
    first = build(target, config)
    second = build(target[::-1].copy(), config)
    first_selection = select_exact_range_echoes(first, config=config)
    second_selection = select_exact_range_echoes(second, config=config)

    np.testing.assert_array_equal(first.xyz_m, second.xyz_m)
    np.testing.assert_array_equal(
        first.ray_indices_ae,
        second.ray_indices_ae,
    )
    assert first.report["hashes"] == second.report["hashes"]
    np.testing.assert_array_equal(
        first_selection.xyz_m,
        second_selection.xyz_m,
    )
    assert (
        first_selection.report["hashes"]
        == second_selection.report["hashes"]
    )


def test_exact_selection_satisfies_count_quotas_and_true_5cm() -> None:
    target = quota_target()
    config = RangeEchoOracleConfig(
        k=4,
        exact_count=24,
        range_quotas=(12, 8, 4),
    )
    selection = select_exact_range_echoes(build(target, config), config=config)

    assert selection.xyz_m.shape == (24, 3)
    assert selection.report["selected_by_range"] == {
        "range_0_30m": 12,
        "range_30_60m": 8,
        "range_60_120m": 4,
    }
    distance = np.linalg.norm(
        selection.xyz_m[:, None, :] - selection.xyz_m[None, :, :],
        axis=2,
    )
    np.fill_diagonal(distance, np.inf)
    assert float(distance.min()) >= 0.05 - 1e-6
    assert selection.report["copy_padding_jitter_duplicate"] is False


def test_capacity_failure_is_hard_and_never_fills_missing_points() -> None:
    target = quota_target()[:-1]
    config = RangeEchoOracleConfig(
        k=4,
        exact_count=24,
        range_quotas=(12, 8, 4),
    )
    with pytest.raises(RangeEchoCapacityError) as caught:
        select_exact_range_echoes(build(target, config), config=config)

    report = caught.value.report
    assert report["hard_capacity_failure"] is True
    assert report["copy_padding_jitter_duplicate"] is False
    assert report["best_failed_attempt"]["selected_count"] == 23
    assert report["best_failed_attempt"]["quota_deficit_by_range"][
        "range_60_120m"
    ] == 1


def test_reports_label_gt_diagnostic_as_unattainable_not_upper_bound() -> None:
    config = RangeEchoOracleConfig(
        k=4,
        exact_count=24,
        range_quotas=(12, 8, 4),
    )
    candidates = build(quota_target(), config)
    selection = select_exact_range_echoes(candidates, config=config)

    for report in (candidates.report, selection.report):
        assert report["oracle_label"] == ORACLE_LABEL
        assert report["deployable_method"] is False
        assert report["strict_mathematical_upper_bound"] is False


def test_far_frame_semantics_use_target_points_in_60_to_120m_only() -> None:
    near_target = np.stack(
        [polar_point(10.0, 30, 18), polar_point(20.0, 31, 18)]
    )
    far_target = np.concatenate(
        (near_target, polar_point(70.0, 32, 18)[None, :]),
        axis=0,
    )

    near_report = geometry_endpoints(near_target[:, :3], near_target)
    far_report = geometry_endpoints(far_target[:, :3], far_target)

    assert near_report["far_target_frame"] is False
    assert "range_60_120m_completeness_mean_distance_m" not in near_report
    assert far_report["far_target_frame"] is True
    assert far_report["range_60_120m_target_count"] == 1
    assert far_report["range_60_120m_completeness_mean_distance_m"] == 0.0


def test_target_loader_accesses_no_cube_or_cfar_array(tmp_path) -> None:
    path = tmp_path / "target.npz"
    target = quota_target()
    np.savez(
        path,
        target_xyz_confidence=target,
        cfar_xyzd_power_snr=np.ones((2, 6), dtype=np.float32),
        cube_drae=np.ones((1,), dtype=np.float32),
    )
    loaded = load_target_only(path)
    source = inspect.getsource(load_target_only)

    np.testing.assert_array_equal(loaded, target)
    assert 'cache["target_xyz_confidence"]' in source
    assert 'cache["cfar_xyzd_power_snr"]' not in source
    assert 'cache["cube_drae"]' not in source
