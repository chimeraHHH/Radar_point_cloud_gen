import numpy as np
import pytest
import torch

from losses.polar_rectified_flow import (
    polar_rectified_flow_loss,
    validate_source_to_target,
)
from models.polar_rectified_flow import PolarRectifiedFlow


POINT_COUNT = 32
SPATIAL_SHAPE = (4, 4, 2)


def cube(seed: int, *, requires_grad: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(
        (1, 64, *SPATIAL_SHAPE),
        generator=generator,
    )
    return values.requires_grad_(requires_grad)


def tiny_model() -> PolarRectifiedFlow:
    return PolarRectifiedFlow(
        range_knots_m=np.array([0.0, 10.0, 30.0, 60.0]),
        range_cdf=np.array([0.0, 0.25, 0.75, 1.0]),
        azimuth_bounds_rad=(-0.8, 0.8),
        elevation_bounds_rad=(-0.25, 0.25),
        doppler_axis_mps=np.linspace(-8.0, 8.0, 64),
        log_center=0.0,
        log_scale=1.0,
        point_count=POINT_COUNT,
        model_dim=32,
        latent_count=8,
        condition_depth=1,
        particle_depth=1,
        heads=4,
        frequency_count=2,
        radar_spectral_channels=4,
        radar_encoded_shape=SPATIAL_SHAPE,
        radar_encoded_channels=4,
        radar_base_channels=4,
        radar_channel_multipliers=(1,),
        radar_blocks_per_level=1,
    )


def test_nfe4_inference_outputs_exact_particles_without_postprocessing() -> None:
    torch.manual_seed(3)
    model = tiny_model().eval()

    with torch.inference_mode():
        output = model.sample(cube(5), chunk_size=11)

    assert output["state"].shape == (1, POINT_COUNT, 3)
    assert output["coordinates_rae"].shape == (1, POINT_COUNT, 3)
    assert output["xyz_m"].shape == (1, POINT_COUNT, 3)
    assert output["doppler_logits"].shape == (1, POINT_COUNT, 64)
    assert output["doppler_probability"].shape == (1, POINT_COUNT, 64)
    assert output["doppler_mps"].shape == (1, POINT_COUNT)
    assert output["confidence"].shape == (1, POINT_COUNT)
    assert output["integration"] == {
        "method": "euler",
        "nfe": 4,
        "postprocessing": [],
    }
    assert output["architecture_metadata"]["inference_postprocessing"] == []
    assert all(
        torch.isfinite(output[key]).all()
        for key in (
            "state",
            "coordinates_rae",
            "xyz_m",
            "doppler_logits",
            "confidence",
        )
    )
    torch.testing.assert_close(
        output["doppler_probability"].sum(dim=-1),
        torch.ones(1, POINT_COUNT),
    )


def test_protocol_rejects_non_nfe4_inference() -> None:
    with pytest.raises(ValueError, match="frozen at NFE=4"):
        tiny_model().sample(cube(7), nfe=8)


def test_fixed_source_does_not_depend_on_cube() -> None:
    model = tiny_model()
    before = model.fixed_source_state(1).clone()
    model.encode_condition(cube(11))
    after = model.fixed_source_state(1)

    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_velocity_and_attributes_backpropagate_to_full_cube_condition() -> None:
    torch.manual_seed(13)
    model = tiny_model()
    measured = cube(17, requires_grad=True)
    condition = model.encode_condition(measured)
    source = model.fixed_source_state(1)
    velocity = model.velocity(source, 0.5, condition, chunk_size=9)
    attributes = model.endpoint_attributes(source, condition, chunk_size=9)
    loss = (
        velocity.square().mean()
        + attributes["doppler_logits"].square().mean()
        + attributes["confidence_logit"].square().mean()
    )

    loss.backward()

    assert measured.grad is not None
    assert torch.isfinite(measured.grad).all()
    assert torch.count_nonzero(measured.grad) > 0
    channel_gradient = measured.grad.abs().sum(dim=(0, 2, 3, 4))
    assert torch.count_nonzero(channel_gradient) == 64


def test_wrong_cube_changes_direct_transport_with_same_source() -> None:
    torch.manual_seed(19)
    model = tiny_model().eval()
    source = model.fixed_source_state(1)
    with torch.inference_mode():
        first_condition = model.encode_condition(cube(23))
        second_condition = model.encode_condition(cube(29))
        first = model.velocity(source, 0.5, first_condition, chunk_size=8)
        second = model.velocity(source, 0.5, second_condition, chunk_size=8)

    assert not torch.allclose(first_condition, second_condition)
    assert not torch.allclose(first, second)


def test_flow_loss_uses_one_to_one_transport_and_wrong_condition() -> None:
    torch.manual_seed(31)
    model = tiny_model()
    source = model.fixed_source_state(1)
    target = source.roll(3, dims=1) + 0.05
    permutation = torch.roll(torch.arange(POINT_COUNT), shifts=-3)
    confidence = torch.linspace(0.1, 1.0, POINT_COUNT).unsqueeze(0)
    doppler_distribution = torch.softmax(
        torch.randn(POINT_COUNT, 64),
        dim=-1,
    )

    result = polar_rectified_flow_loss(
        model,
        cube(37),
        source,
        target,
        permutation,
        time=torch.tensor([0.4]),
        target_doppler_distribution=doppler_distribution,
        target_confidence=confidence,
        wrong_cube_drae=cube(41),
        chunk_size=10,
    )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert {
        "flow_matching",
        "endpoint_geometry",
        "wrong_condition_margin",
        "doppler_cross_entropy",
        "doppler_circular_w1",
        "confidence_bce",
        "total",
    }.issubset(result.components)
    doppler_gradient = model.particle_decoder.doppler_head.weight.grad
    assert doppler_gradient is not None
    assert torch.isfinite(doppler_gradient).all()
    assert torch.count_nonzero(doppler_gradient.abs().sum(dim=1)) == 64


def test_transport_validation_rejects_duplicate_target_indices() -> None:
    invalid = torch.arange(POINT_COUNT)
    invalid[-1] = 0

    with pytest.raises(ValueError, match="one-to-one"):
        validate_source_to_target(invalid, POINT_COUNT)
