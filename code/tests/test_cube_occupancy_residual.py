import torch

from models.cube_occupancy import CubeOccupancyNet, parameter_count


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
