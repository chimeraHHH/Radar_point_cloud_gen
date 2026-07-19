import inspect

import torch

from models.rald_anchor_ldm import RaLDAnchorLDM
from models.rald_matched import (
    FullRAEDRadarTokenEncoder,
    RaLDPhysicalQueryHead,
    edm_loss,
)


def tiny_full_raed_encoder() -> FullRAEDRadarTokenEncoder:
    return FullRAEDRadarTokenEncoder(
        log_center=2.0,
        log_scale=0.5,
        spectral_channels=4,
        encoded_shape=(2, 2, 2),
        encoded_channels=4,
        token_dim=32,
        base_channels=4,
        channel_multipliers=(1, 1, 1),
        blocks_per_level=1,
    )


def small_model() -> RaLDAnchorLDM:
    torch.manual_seed(17)
    return RaLDAnchorLDM(
        anchor_feature_dim=5,
        latent_count=8,
        latent_dim=4,
        model_dim=32,
        decoder_depth=2,
        denoiser_depth=2,
        heads=4,
        head_dim=8,
        edm_steps=3,
        radar_encoder=tiny_full_raed_encoder(),
    )


def point_state(batch: int = 1, count: int = 13) -> tuple[torch.Tensor, ...]:
    normalized_rae = torch.rand(batch, count, 3) * 2.0 - 1.0
    doppler_probability = torch.rand(batch, count, 64).softmax(dim=-1)
    confidence = torch.rand(batch, count)
    return normalized_rae, doppler_probability, confidence


def anchors(batch: int = 1, count: int = 11) -> tuple[torch.Tensor, ...]:
    normalized_rae = torch.rand(batch, count, 3) * 2.0 - 1.0
    features = torch.randn(batch, count, 5)
    return normalized_rae, features


def test_default_contract_matches_rald_g3l_scale() -> None:
    signature = inspect.signature(RaLDAnchorLDM.__init__)

    assert signature.parameters["latent_count"].default == 512
    assert signature.parameters["latent_dim"].default == 32
    assert signature.parameters["decoder_depth"].default == 24
    assert signature.parameters["denoiser_depth"].default == 24
    assert signature.parameters["edm_steps"].default == 18
    assert RaLDAnchorLDM.OFFICIAL_RALD_COMMIT.startswith("ffec4b4")


def test_posterior_is_invariant_and_anchor_decode_is_equivariant() -> None:
    model = small_model().eval()
    state = point_state()
    anchor_rae, anchor_features = anchors()
    point_permutation = torch.randperm(state[0].shape[1])
    anchor_permutation = torch.randperm(anchor_rae.shape[1])

    with torch.inference_mode():
        first = model.posterior_mean_path(*state, anchor_rae, anchor_features)
        second = model.posterior_mean_path(
            state[0][:, point_permutation],
            state[1][:, point_permutation],
            state[2][:, point_permutation],
            anchor_rae[:, anchor_permutation],
            anchor_features[:, anchor_permutation],
        )

    torch.testing.assert_close(
        first.posterior.mean, second.posterior.mean, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        first.posterior.log_variance,
        second.posterior.log_variance,
        rtol=1e-5,
        atol=1e-6,
    )
    inverse = torch.argsort(anchor_permutation)
    torch.testing.assert_close(
        first.query_features,
        second.query_features[:, inverse],
        rtol=1e-5,
        atol=1e-6,
    )


def test_shapes_and_physical_query_head_compatibility() -> None:
    model = small_model().eval()
    state = point_state(batch=2, count=13)
    anchor_rae, anchor_features = anchors(batch=2, count=11)
    local_spectrum = torch.rand(2, 11, 64)

    output = model.posterior_mean_path(*state, anchor_rae, anchor_features)
    physical = RaLDPhysicalQueryHead(
        query_dim=32, spectrum_bins=64, hidden_dim=32
    )(output.query_features, local_spectrum)

    assert output.latent.shape == (2, 8, 4)
    assert output.query_features.shape == (2, 11, 32)
    assert output.posterior.kl().shape == (2,)
    assert physical["doppler_probability"].shape == (2, 11, 64)
    assert physical["confidence_logit"].shape == (2, 11)
    torch.testing.assert_close(
        physical["offset_bins"], torch.zeros_like(physical["offset_bins"])
    )


def test_posterior_uses_doppler_distribution_and_confidence() -> None:
    model = small_model().eval()
    normalized_rae, probability, confidence = point_state()

    with torch.inference_mode():
        baseline = model.posterior_encoder(
            normalized_rae, probability, confidence
        ).mean
        shifted_doppler = model.posterior_encoder(
            normalized_rae, probability.roll(7, dims=-1), confidence
        ).mean
        changed_confidence = model.posterior_encoder(
            normalized_rae, probability, 1.0 - confidence
        ).mean

    assert not torch.allclose(baseline, shifted_doppler)
    assert not torch.allclose(baseline, changed_confidence)


def test_edm_first_and_second_step_gradients_reach_full_raed_condition() -> None:
    torch.manual_seed(23)
    model = small_model()
    optimizer = torch.optim.SGD(model.edm.parameters(), lr=1e-2)
    latent = torch.randn(1, 8, 4)
    cube = torch.rand(1, 64, 8, 8, 8)

    first_loss = edm_loss(model.edm, latent, cube)
    first_loss.backward()

    output_weight = model.edm.denoiser.output.weight
    assert output_weight.grad is not None
    assert torch.count_nonzero(output_weight.grad).item() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    edm_loss(model.edm, latent, cube).backward()
    condition_weight = model.edm.radar_encoder.spectral_projection.weight
    assert condition_weight.grad is not None
    assert torch.count_nonzero(condition_weight.grad).item() > 0


def test_seeded_edm_path_is_deterministic() -> None:
    torch.manual_seed(29)
    model = small_model().eval()
    cube = torch.rand(1, 64, 8, 8, 8)
    anchor_rae, anchor_features = anchors()

    first = model.sampled_edm_path(
        cube, anchor_rae, anchor_features, [31], steps=3
    )
    second = model.sampled_edm_path(
        cube, anchor_rae, anchor_features, [31], steps=3
    )

    torch.testing.assert_close(first.latent, second.latent, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.query_features, second.query_features, rtol=0.0, atol=0.0
    )


def test_decoder_uses_only_supplied_anchors_without_space_scan(monkeypatch) -> None:
    model = small_model().eval()
    latent = torch.randn(1, 8, 4)
    anchor_rae, anchor_features = anchors(count=7)

    def reject_space_scan(*args, **kwargs):
        raise AssertionError("G3L must not construct a full-space query grid")

    monkeypatch.setattr(torch, "meshgrid", reject_space_scan)
    with torch.inference_mode():
        query_features = model.decoder(latent, anchor_rae, anchor_features)

    assert query_features.shape == (1, 7, 32)
    assert not hasattr(model.decoder, "occupancy")
