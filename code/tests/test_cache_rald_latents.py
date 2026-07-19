import numpy as np
import torch

from scripts.cache_rald_matched_latents import (
    build_model,
    frame_generator,
    valid_existing,
)


def config() -> dict:
    return {
        "ae_point_count": 16,
        "latent_count": 8,
        "model_dim": 32,
        "latent_dim": 4,
        "depth": 2,
        "heads": 4,
        "head_dim": 8,
    }


def test_cache_model_reconstruction_uses_frozen_architecture() -> None:
    model = build_model(config())

    assert model.point_count == 16
    assert model.latent_count == 8
    assert model.mean.out_features == 4


def test_frame_generator_is_key_deterministic() -> None:
    first = frame_generator(torch.device("cpu"), 17, 3, 41)
    second = frame_generator(torch.device("cpu"), 17, 3, 41)

    torch.testing.assert_close(
        torch.randn(11, generator=first), torch.randn(11, generator=second)
    )


def test_existing_cache_requires_shape_and_finite_values(tmp_path) -> None:
    path = tmp_path / "latent.npz"
    np.savez(path, latent_mean=np.zeros((8, 4)), posterior_kl=np.float32(0.2))
    assert valid_existing(path, (8, 4)) is True
    assert valid_existing(path, (4, 8)) is False

    np.savez(path, latent_mean=np.full((8, 4), np.nan), posterior_kl=0.2)
    assert valid_existing(path, (8, 4)) is False
