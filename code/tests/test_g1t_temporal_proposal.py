import inspect

import torch

from cube_dense.parent_prediction import PointPrediction
from eval.g1t_temporal_proposal import (
    HistoryProposalSource,
    build_temporal_proposal_arms,
    select_current_rescored,
    warp_history_source,
)
from models.cube_doppler import circular_mean
from scripts.eval_g1t_temporal_proposal import range_slice_geometry


def axes() -> tuple[torch.Tensor, ...]:
    doppler = torch.linspace(-8.0, 8.0, 64)
    ranges = torch.linspace(0.0, 30.0, 16)
    azimuth = torch.linspace(-1.0, 1.0, 9)
    elevation = torch.linspace(-0.4, 0.4, 7)
    return doppler, ranges, azimuth, elevation


def prediction(
    xyz_m: torch.Tensor,
    coordinates_rae: torch.Tensor,
    doppler_bin: int,
    static_center_mps: torch.Tensor | None = None,
) -> PointPrediction:
    count = xyz_m.shape[0]
    probability = torch.nn.functional.one_hot(
        torch.full((count,), doppler_bin), 64
    ).float()
    doppler, _, _, _ = axes()
    period = torch.median(torch.diff(doppler)) * doppler.numel()
    if static_center_mps is None:
        static_center_mps = torch.zeros(count)
    return PointPrediction(
        xyz_m=xyz_m,
        coordinates_rae=coordinates_rae,
        probability=probability,
        confidence=torch.ones(count),
        static_center_mps=torch.remainder(
            static_center_mps - doppler[0], period
        )
        + doppler[0],
    )


def cube() -> torch.Tensor:
    values = torch.arange(64 * 16 * 9 * 7, dtype=torch.float32)
    return values.reshape(1, 64, 16, 9, 7) / values.numel()


def test_doppler_warp_uses_positive_time_then_previous_to_current_transform() -> None:
    doppler, ranges, azimuth, elevation = axes()
    positive_bin = int(torch.argmin((doppler - 4.0).abs()))
    state = prediction(
        torch.tensor([[10.0, 0.0, 0.0]]),
        torch.tensor([[5.0, 4.0, 3.0]]),
        positive_bin,
    )
    transform = torch.eye(4)
    transform[:2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    transform[0, 3] = 2.0
    source = HistoryProposalSource(
        prediction=state,
        current_from_source=transform,
        age_seconds=0.5,
        age_frames=1,
    )
    period = torch.median(torch.diff(doppler)) * doppler.numel()

    ego = warp_history_source(
        source,
        doppler,
        doppler[0],
        period,
        ranges,
        azimuth,
        elevation,
        0.0,
        "zero_centered",
        apply_doppler_displacement=False,
        dynamic_threshold_mps=1.0,
    )
    displaced = warp_history_source(
        source,
        doppler,
        doppler[0],
        period,
        ranges,
        azimuth,
        elevation,
        0.0,
        "zero_centered",
        apply_doppler_displacement=True,
        dynamic_threshold_mps=1.0,
    )

    torch.testing.assert_close(
        ego.xyz_m, torch.tensor([[2.0, 10.0, 0.0]])
    )
    expected = 10.0 + float(doppler[positive_bin]) * 0.5
    torch.testing.assert_close(
        displaced.xyz_m, torch.tensor([[2.0, expected, 0.0]])
    )


def test_all_arms_keep_identical_output_count() -> None:
    doppler, ranges, azimuth, elevation = axes()
    period = torch.median(torch.diff(doppler)) * doppler.numel()
    current = prediction(
        torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [16.0, 0.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [2.0, 4.0, 3.0],
                [4.0, 4.0, 3.0],
                [6.0, 4.0, 3.0],
                [8.0, 4.0, 3.0],
            ]
        ),
        32,
    )
    history = HistoryProposalSource(
        prediction=current,
        current_from_source=torch.eye(4),
        age_seconds=0.1,
        age_frames=1,
    )

    arms = build_temporal_proposal_arms(
        cube(),
        current,
        [history],
        ranges,
        azimuth,
        elevation,
        doppler,
        doppler[0],
        period,
        0.0,
        "zero_centered",
        output_count=4,
        nms_kernel=(1, 1, 1),
    )

    assert {
        arm.prediction.xyz_m.shape[0] for arm in arms.values()
    } == {4}


def test_fixed_dedup_is_deterministic() -> None:
    doppler, ranges, azimuth, elevation = axes()
    period = torch.median(torch.diff(doppler)) * doppler.numel()
    state = prediction(
        torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [4.1, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [2.0, 4.0, 3.0],
                [2.1, 4.0, 3.0],
                [4.0, 4.0, 3.0],
                [6.0, 4.0, 3.0],
            ]
        ),
        32,
    )
    arguments = {
        "current_cube_drae": cube(),
        "candidate_states": [state],
        "source_age_seconds": [0.0],
        "source_age_frames": [0],
        "range_m": ranges,
        "azimuth_rad": azimuth,
        "elevation_rad": elevation,
        "doppler_mps": doppler,
        "doppler_lower_mps": doppler[0],
        "doppler_period_mps": period,
        "current_ego_speed_mps": 0.0,
        "static_hypothesis": "zero_centered",
        "output_count": 3,
        "nms_kernel": (1, 1, 1),
    }

    first = select_current_rescored(**arguments)
    second = select_current_rescored(**arguments)

    torch.testing.assert_close(
        first.prediction.coordinates_rae,
        second.prediction.coordinates_rae,
    )
    torch.testing.assert_close(
        first.selected_candidate_index,
        second.selected_candidate_index,
    )
    assert first.deduplicated_count == second.deduplicated_count
    assert first.fill_count == second.fill_count


def test_selection_api_cannot_receive_gt_or_lidar() -> None:
    for function in (select_current_rescored, build_temporal_proposal_arms):
        parameter_names = {
            name.lower() for name in inspect.signature(function).parameters
        }
        assert not any(
            token in name
            for name in parameter_names
            for token in ("target", "lidar", "ground_truth", "gt_")
        )


def test_t1_equals_t2_when_residual_doppler_is_zero() -> None:
    doppler, ranges, azimuth, elevation = axes()
    period = torch.median(torch.diff(doppler)) * doppler.numel()
    zero_bin = int(torch.argmin(doppler.abs()))
    xyz = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [16.0, 0.0, 0.0],
        ]
    )
    coordinates = torch.tensor(
        [
            [2.0, 4.0, 3.0],
            [4.0, 4.0, 3.0],
            [6.0, 4.0, 3.0],
            [8.0, 4.0, 3.0],
        ]
    )
    raw = prediction(xyz, coordinates, zero_bin)
    scalar = circular_mean(
        raw.probability, doppler, doppler[0], period
    )
    zero_residual = PointPrediction(
        xyz_m=raw.xyz_m,
        coordinates_rae=raw.coordinates_rae,
        probability=raw.probability,
        confidence=raw.confidence,
        static_center_mps=scalar,
    )
    history = HistoryProposalSource(
        prediction=zero_residual,
        current_from_source=torch.eye(4),
        age_seconds=0.1,
        age_frames=1,
    )

    arms = build_temporal_proposal_arms(
        cube(),
        zero_residual,
        [history],
        ranges,
        azimuth,
        elevation,
        doppler,
        doppler[0],
        period,
        0.0,
        "zero_centered",
        output_count=4,
        nms_kernel=(1, 1, 1),
    )

    torch.testing.assert_close(
        arms["t1_ego"].prediction.coordinates_rae,
        arms["t2_doppler"].prediction.coordinates_rae,
    )
    torch.testing.assert_close(
        arms["t1_ego"].source_age_seconds,
        arms["t2_doppler"].source_age_seconds,
    )


def test_range_slice_keeps_far_targets_when_prediction_has_no_far_points() -> None:
    report = range_slice_geometry(
        torch.tensor([[20.0, 0.0, 0.0]]),
        torch.tensor([[80.0, 0.0, 0.0]]),
        torch.ones(1),
        60.0,
    )

    assert report["far_completeness_mean_distance_m"] == 60.0
