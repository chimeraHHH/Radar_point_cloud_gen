import torch

from models.rald_matched import (
    FourierPointEmbedding,
    FullRAEDRadarTokenEncoder,
    RaLDAnchorLatentRefiner,
    RaLDEDMPreconditioner,
    RaLDPointAutoencoder,
    RaLDPhysicalQueryHead,
    RadarTokenEncoder,
    edm_loss,
    edm_sample,
)


def test_full_raed_tokens_use_complete_spectrum() -> None:
    encoder = FullRAEDRadarTokenEncoder(
        log_center=2.0,
        log_scale=0.5,
        spectral_channels=4,
        encoded_shape=(4, 4, 2),
        encoded_channels=3,
        token_dim=8,
        base_channels=4,
        channel_multipliers=(1, 1, 1),
        blocks_per_level=1,
    )
    cube = torch.rand(1, 64, 16, 16, 8, requires_grad=True)

    tokens = encoder(cube)
    tokens.square().mean().backward()

    assert tokens.shape == (1, 32, 8)
    assert encoder.spectral_projection.weight.grad is not None
    assert torch.count_nonzero(encoder.spectral_projection.weight.grad).item() > 0


def test_physical_query_head_starts_from_measured_spectrum() -> None:
    head = RaLDPhysicalQueryHead(query_dim=8, spectrum_bins=6, hidden_dim=12)
    query_features = torch.randn(2, 5, 8)
    spectrum = torch.rand(2, 5, 6)
    expected = spectrum / spectrum.sum(dim=-1, keepdim=True)

    output = head(query_features, spectrum)

    torch.testing.assert_close(output["doppler_probability"], expected)
    torch.testing.assert_close(output["offset_bins"], torch.zeros(2, 5, 3))
    torch.testing.assert_close(output["confidence_logit"], torch.zeros(2, 5))


def test_anchor_refiner_is_permutation_invariant_and_equivariant() -> None:
    model = RaLDAnchorLatentRefiner(
        anchor_feature_dim=4,
        latent_count=8,
        model_dim=32,
        depth=2,
        heads=4,
        head_dim=8,
        spectrum_bins=6,
    ).eval()
    coordinates = torch.rand(1, 17, 3) * 2.0 - 1.0
    features = torch.randn(1, 17, 4)
    spectrum = torch.rand(1, 17, 6)
    permutation = torch.randperm(17)

    with torch.inference_mode():
        first = model(coordinates, features, spectrum)
        second = model(
            coordinates[:, permutation],
            features[:, permutation],
            spectrum[:, permutation],
        )

    torch.testing.assert_close(first["latent"], second["latent"], atol=1e-5, rtol=1e-5)
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        first["doppler_probability"],
        second["doppler_probability"][:, inverse],
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


def test_prepared_latent_matches_direct_decode() -> None:
    model = small_autoencoder().eval()
    latent = torch.randn(1, 8, 4)
    queries = torch.rand(1, 13, 3) * 2.0 - 1.0

    with torch.inference_mode():
        direct = model.decode(latent, queries)
        prepared = model.prepare_decoder_latent(latent)
        chunked = torch.cat(
            (
                model.decode_queries(prepared, queries[:, :5]),
                model.decode_queries(prepared, queries[:, 5:]),
            ),
            dim=1,
        )

    torch.testing.assert_close(direct, chunked, rtol=1e-6, atol=1e-6)


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
