import torch

from models.temporal_prior import ego_pose_warp, prior_distribution_features


def test_ego_pose_warp_does_not_apply_doppler_displacement() -> None:
    xyz = torch.tensor([[10.0, 0.0, 0.0]])
    probability = torch.nn.functional.one_hot(torch.tensor([63]), 64).float()
    confidence = torch.tensor([0.8])
    transform = torch.eye(4)
    transform[0, 3] = 2.0
    doppler = torch.linspace(-8.0, 8.0, 64)

    prior = ego_pose_warp(
        xyz,
        probability,
        confidence,
        transform,
        doppler,
        doppler[0],
        torch.tensor(16.0 + 16.0 / 63.0),
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
    )

    torch.testing.assert_close(prior.xyz_m, torch.tensor([[12.0, 0.0, 0.0]]))
    assert prior.dynamic_gate.tolist() == [False]
    torch.testing.assert_close(prior.confidence, confidence)


def test_prior_distribution_features_are_finite_without_static_reference() -> None:
    doppler = torch.linspace(-8.0, 8.0, 64)
    probability = torch.softmax(torch.randn(3, 64), dim=1)
    prior = ego_pose_warp(
        torch.tensor([[10.0, 0.0, 0.0], [20.0, 1.0, 0.0], [30.0, 2.0, 0.0]]),
        probability,
        torch.ones(3),
        torch.eye(4),
        doppler,
        doppler[0],
        torch.tensor(16.0 + 16.0 / 63.0),
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
    )

    features = prior_distribution_features(
        prior, doppler, doppler[0], torch.tensor(16.0 + 16.0 / 63.0)
    )

    assert features.shape == (3, 4)
    assert torch.isfinite(features).all()

