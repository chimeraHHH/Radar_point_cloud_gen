import torch

from cube_dense.rald_prediction import RaLDPointPrediction
from models.temporal_baselines import (
    ego_warp_rald_prediction,
    rald_history_aggregate,
)


def state(xyz: torch.Tensor, confidence: torch.Tensor) -> RaLDPointPrediction:
    count = xyz.shape[0]
    probability = torch.nn.functional.one_hot(
        torch.arange(count) % 64, 64
    ).float()
    return RaLDPointPrediction(
        xyz_m=xyz,
        coordinates_rae=xyz.clone(),
        probability=probability,
        confidence=confidence,
    )


def test_rald_ego_warp_preserves_distribution() -> None:
    prediction = state(torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([0.8]))
    transform = torch.eye(4)
    transform[0, 3] = 2.0
    doppler = torch.linspace(-8.0, 8.0, 64)

    warped = ego_warp_rald_prediction(
        prediction,
        transform,
        doppler,
        doppler[0],
        torch.median(torch.diff(doppler)) * doppler.numel(),
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
    )

    torch.testing.assert_close(warped.xyz_m, torch.tensor([[12.0, 0.0, 0.0]]))
    torch.testing.assert_close(warped.probability, prediction.probability)


def test_rald_history_aggregate_keeps_exact_point_budget() -> None:
    current = state(
        torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        torch.tensor([0.9, 0.8]),
    )
    history = state(
        torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
        torch.tensor([0.7, 0.6]),
    )

    result, diagnostics = rald_history_aggregate(
        current, [history], point_count=3, spatial_shape=(8, 8, 8)
    )

    assert result.xyz_m.shape == (3, 3)
    assert diagnostics.history_count == 1
