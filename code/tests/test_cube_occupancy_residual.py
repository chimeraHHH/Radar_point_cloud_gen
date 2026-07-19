import torch

from models.cube_occupancy import (
    CubeOccupancyNet,
    parameter_count,
    spectral_diagnostics,
    spectral_gradient_norm,
)


def make_model(mode: str, seed: int = 17) -> CubeOccupancyNet:
    torch.manual_seed(seed)
    return CubeOccupancyNet(mode, torch.linspace(-8.0, 8.0, 64), base_channels=8)


def test_full_raed_starts_as_exact_rae_max_function() -> None:
    rae_max = make_model("rae_max")
    full_raed = make_model("full_raed")
    cube = torch.rand(1, 64, 8, 8, 8)

    with torch.inference_mode():
        rae_logits = rae_max(cube)
        full_logits = full_raed(cube)

    torch.testing.assert_close(full_logits, rae_logits, rtol=0.0, atol=0.0)


def test_spatial_features_are_the_features_consumed_by_the_head() -> None:
    model = make_model("full_raed")
    cube = torch.rand(1, 64, 8, 8, 8)

    features = model.spatial_features(cube)

    torch.testing.assert_close(model(cube), model.head(features).squeeze(1))


def test_full_raed_residual_is_zero_initialized_and_bounded_in_size() -> None:
    rae_max = make_model("rae_max")
    full_raed = make_model("full_raed")

    residual = full_raed.spectral_residual_projection
    assert residual is not None
    assert torch.count_nonzero(residual.weight).item() == 0
    assert torch.count_nonzero(residual.bias).item() == 0
    relative_increase = (
        parameter_count(full_raed) - parameter_count(rae_max)
    ) / parameter_count(rae_max)
    assert relative_increase <= 0.01


def test_full_raed_residual_receives_gradient_at_initialization() -> None:
    model = make_model("full_raed")
    cube = torch.rand(1, 64, 8, 8, 8)

    model(cube).square().mean().backward()

    residual = model.spectral_residual_projection
    assert residual is not None
    assert residual.weight.grad is not None
    assert torch.count_nonzero(residual.weight.grad).item() > 0


def test_rank2_full_raed_starts_as_exact_rae_max_function() -> None:
    rae_max = make_model("rae_max")
    rank2 = make_model("full_raed_rank2")
    cube = torch.rand(1, 64, 8, 8, 8)

    with torch.inference_mode():
        expected = rae_max(cube)
        actual = rank2(cube)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_rank2_residual_is_bounded_and_receives_output_gradient() -> None:
    rae_max = make_model("rae_max")
    rank2 = make_model("full_raed_rank2")
    cube = torch.rand(1, 64, 8, 8, 8)

    rank2(cube).square().mean().backward()

    residual = rank2.spectral_rank_projection
    assert residual is not None
    assert torch.count_nonzero(residual[-1].weight).item() == 0
    assert residual[-1].weight.grad is not None
    assert torch.count_nonzero(residual[-1].weight.grad).item() > 0
    relative_increase = (
        parameter_count(rank2) - parameter_count(rae_max)
    ) / parameter_count(rae_max)
    assert relative_increase <= 0.01


def test_circular_harmonics_distinguish_matched_linear_moments() -> None:
    model = make_model("rae_circular_harmonics")
    first = torch.zeros(1, 64, 1, 1, 1)
    second = torch.zeros_like(first)
    first[:, [24, 40]] = 1.0
    second[:, [16, 48]] = 1.0 / 6.0
    second[:, 32] = 1.0

    first_encoded = model.encode_cube(first)
    second_encoded = model.encode_cube(second)

    assert not torch.allclose(first_encoded, second_encoded)


def test_spectral_diagnostics_separate_branch_from_peak_channel() -> None:
    model = make_model("rae_circular_harmonics")
    cube = torch.rand(1, 64, 8, 8, 8)

    model(cube).square().mean().backward()
    diagnostics = spectral_diagnostics(model)

    assert diagnostics["spectral_branch_weight_rms"] is not None
    assert diagnostics["trunk_input_weight_rms"] is not None
    assert diagnostics["spectral_to_trunk_weight_rms_ratio"] is not None
    assert spectral_gradient_norm(model) is not None
    assert spectral_gradient_norm(model) > 0.0


def test_rae_max_has_no_spectral_branch_diagnostics() -> None:
    diagnostics = spectral_diagnostics(make_model("rae_max"))

    assert diagnostics["spectral_branch_weight_rms"] is None
    assert diagnostics["spectral_to_trunk_weight_rms_ratio"] is None
