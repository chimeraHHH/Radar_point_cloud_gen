import torch

from models.rald_matched import (
    FourierPointEmbedding,
    RaLDEDMPreconditioner,
    RaLDPointAutoencoder,
    RadarTokenEncoder,
    edm_loss,
    edm_sample,
)


def small_autoencoder() -> RaLDPointAutoencoder:
    torch.manual_seed(17)
    return RaLDPointAutoencoder(
        point_count=16,
        latent_count=8,
        model_dim=32,
        latent_dim=4,
        depth=2,
        heads=4,
        head_dim=8,
    )


def small_edm() -> RaLDEDMPreconditioner:
    radar_encoder = RadarTokenEncoder(
        encoded_shape=(2, 2, 2),
        encoded_channels=4,
        token_dim=32,
        base_channels=4,
        channel_multipliers=(1, 1, 2),
        blocks_per_level=1,
    )
    return RaLDEDMPreconditioner(
        latent_count=8,
        latent_dim=4,
        model_dim=32,
        depth=2,
        heads=4,
        head_dim=8,
        radar_encoder=radar_encoder,
    )


def test_fourier_point_embedding_validates_and_preserves_shape() -> None:
    embedding = FourierPointEmbedding(output_dim=32)
    values = embedding(torch.rand(2, 11, 3))

    assert values.shape == (2, 11, 32)


def test_point_autoencoder_is_permutation_invariant_in_mean_latent() -> None:
    model = small_autoencoder().eval()
    points = torch.rand(1, 16, 3) * 2.0 - 1.0
    permutation = torch.randperm(points.shape[1])

    with torch.inference_mode():
        first = model.encode(points).mean
        second = model.encode(points[:, permutation]).mean

    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_point_autoencoder_has_finite_kl_and_decoder_gradients() -> None:
    model = small_autoencoder()
    points = torch.rand(1, 16, 3) * 2.0 - 1.0
    queries = torch.rand(1, 23, 3) * 2.0 - 1.0

    logits, posterior = model(points, queries, sample_posterior=False)
    loss = logits.square().mean() + 1e-3 * posterior.kl().mean()
    loss.backward()

    assert logits.shape == (1, 23)
    assert torch.isfinite(posterior.kl()).all()
    assert model.occupancy.weight.grad is not None
    assert torch.count_nonzero(model.occupancy.weight.grad).item() > 0


def test_radar_encoder_uses_native_spatial_shape() -> None:
    encoder = RadarTokenEncoder(
        encoded_shape=(4, 4, 2),
        encoded_channels=4,
        token_dim=32,
        base_channels=4,
        channel_multipliers=(1, 1, 2),
        blocks_per_level=1,
    )

    tokens = encoder(torch.rand(1, 1, 16, 16, 8))

    assert tokens.shape == (1, 32, 32)


def test_edm_zero_output_learns_then_backpropagates_to_radar_condition() -> None:
    torch.manual_seed(23)
    model = small_edm()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    latent = torch.randn(1, 8, 4)
    radar = torch.randn(1, 1, 8, 8, 8)

    loss = edm_loss(model, latent, radar)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.denoiser.output.weight.grad is not None
    assert torch.count_nonzero(model.denoiser.output.weight.grad).item() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    edm_loss(model, latent, radar).backward()
    first_weight = model.radar_encoder.encoder.input.weight
    assert first_weight.grad is not None
    assert torch.count_nonzero(first_weight.grad).item() > 0


def test_edm_sampling_is_seed_deterministic() -> None:
    torch.manual_seed(29)
    model = small_edm().eval()
    radar = torch.randn(1, 1, 8, 8, 8)

    first = edm_sample(model, radar, [31], steps=3)
    second = edm_sample(model, radar, [31], steps=3)

    assert first.shape == (1, 8, 4)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
