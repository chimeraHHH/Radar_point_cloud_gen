import torch

from scripts.verify_rald_matched import finite_nonzero, gradient_norm


def test_finite_nonzero_rejects_missing_zero_and_nonfinite_values() -> None:
    assert finite_nonzero(None) is False
    assert finite_nonzero(0.0) is False
    assert finite_nonzero(float("nan")) is False
    assert finite_nonzero(0.25) is True


def test_gradient_norm_reports_only_available_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    (first.square().sum()).backward()

    assert gradient_norm([second]) is None
    assert gradient_norm([first, second]) == torch.linalg.vector_norm(
        first.grad
    ).item()
