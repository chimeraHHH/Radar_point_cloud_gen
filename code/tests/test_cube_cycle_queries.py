from __future__ import annotations

import torch

from eval.cube_cycle import binary_ece
from losses.cube_cycle import (
    covered_spectrum_kl,
    existence_confidence_loss,
    normalized_cube_spectrum,
)
from models.cube_cycle import CubeCycleNet
from models.cube_doppler import query_cube_spectrum
from models.point_to_cube import soft_splat_raed, trilinear_query_features


def test_trilinear_query_matches_integer_gather() -> None:
    features = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(
        2, 3, 4, 5, 6
    )
    coordinates = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
    batch = torch.tensor([0, 1])

    queried = trilinear_query_features(features, coordinates, batch)
    expected = torch.stack((features[0, :, 1, 2, 3], features[1, :, 2, 1, 4]))

    torch.testing.assert_close(queried, expected)


def test_continuous_cube_spectrum_interpolates_and_has_coordinate_gradient() -> None:
    cube = torch.zeros(1, 64, 2, 2, 2)
    cube[0, 5, 0, 0, 0] = 9.0
    cube[0, 11, 1, 0, 0] = 9.0
    coordinate = torch.tensor([[0.25, 0.0, 0.0]], requires_grad=True)

    probability = query_cube_spectrum(cube, coordinate, smoothing=1e-8)

    assert probability[0, 5] > probability[0, 11] > 0
    probability[0, 11].backward()
    assert coordinate.grad is not None
    assert coordinate.grad[0, 0] > 0


def test_cycle_doppler_query_backpropagates_through_final_position() -> None:
    torch.manual_seed(7)
    model = CubeCycleNet(
        "distribution",
        torch.linspace(-8.0, 8.0, 64),
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
        base_channels=2,
    )
    with torch.no_grad():
        for parameter in model.offset_head.parameters():
            parameter.zero_()
        model.offset_head[-1].bias.fill_(0.54930615)
    features = torch.arange(1 * 2 * 4 * 4 * 4, dtype=torch.float32).reshape(
        1, 2, 4, 4, 4
    )
    features.requires_grad_(True)

    prediction = model.query_cycle(
        features, torch.tensor([[1, 1, 1]]), torch.tensor([0.0])
    )

    torch.testing.assert_close(
        prediction["coordinates_rae"], torch.full((1, 3), 1.25)
    )
    prediction["probability"][0, 0].backward()
    assert model.offset_head[-1].bias.grad is not None
    assert model.offset_head[-1].bias.grad.abs().sum() > 0


def test_target_support_local_kl_penalizes_missing_prediction_coverage() -> None:
    cube = torch.zeros(64, 2, 1, 1)
    cube[3, 0, 0, 0] = 10.0
    cube[17, 1, 0, 0] = 10.0
    target_probability, target_energy = normalized_cube_spectrum(cube)
    rendered = soft_splat_raed(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.nn.functional.one_hot(torch.tensor([3]), 64).float(),
        torch.ones(1),
        spatial_shape=(2, 1, 1),
    )

    target_loss = covered_spectrum_kl(
        rendered,
        target_probability,
        target_energy,
        support_mode="target",
        target_peak_count=2,
    )
    legacy_loss = covered_spectrum_kl(
        rendered,
        target_probability,
        target_energy,
        support_mode="prediction",
        target_peak_count=2,
    )

    assert target_loss > legacy_loss


def test_soft_splat_duplicate_and_circular_shift_invariants() -> None:
    coordinates = torch.tensor([[0.25, 0.5, 0.75], [1.5, 1.0, 0.25]])
    logits = torch.randn(2, 64)
    probability = torch.softmax(logits, dim=1)
    confidence = torch.tensor([0.4, 0.8])
    reference = soft_splat_raed(
        coordinates, probability, confidence, spatial_shape=(3, 3, 2)
    )
    duplicated = soft_splat_raed(
        coordinates.repeat_interleave(2, dim=0),
        probability.repeat_interleave(2, dim=0),
        (confidence / 2).repeat_interleave(2, dim=0),
        spatial_shape=(3, 3, 2),
    )
    shifted = soft_splat_raed(
        coordinates,
        probability.roll(7, dims=1),
        confidence,
        spatial_shape=(3, 3, 2),
    )

    torch.testing.assert_close(duplicated.energy_drae, reference.energy_drae)
    torch.testing.assert_close(
        shifted.energy_drae, reference.energy_drae.roll(7, dims=0)
    )


def test_existence_confidence_uses_matched_and_unmatched_bernoulli_targets() -> None:
    confidence = torch.tensor([0.8, 0.2])
    distance = torch.tensor([0.5, 1.5])

    loss, target = existence_confidence_loss(confidence, distance)

    assert target.tolist() == [1.0, 0.0]
    torch.testing.assert_close(loss, -torch.log(torch.tensor(0.8)))
    torch.testing.assert_close(binary_ece(confidence, target, bins=2), torch.tensor(0.2))
