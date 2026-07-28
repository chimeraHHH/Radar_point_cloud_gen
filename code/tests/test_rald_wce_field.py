import inspect

import torch

from losses.rald_wce import rald_wce_stage0_loss
from models.rald_wce_field import RaLDWCEField


SPATIAL_SHAPE = (16, 7, 3)


def cube(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(
        (1, 64, *SPATIAL_SHAPE),
        generator=generator,
    )


def queries(count: int = 23) -> torch.Tensor:
    generator = torch.Generator().manual_seed(101)
    return torch.rand((1, count, 3), generator=generator) * 2.0 - 1.0


def tiny_model() -> RaLDWCEField:
    return RaLDWCEField(
        log_center=0.0,
        log_scale=1.0,
        spatial_shape=SPATIAL_SHAPE,
        latent_count=12,
        model_dim=24,
        depth=2,
        heads=3,
        head_dim=8,
        fourier_frequency_dim=12,
        radar_base_channels=4,
        radar_spectral_channels=4,
        radar_encoded_shape=SPATIAL_SHAPE,
        radar_encoded_channels=4,
        radar_channel_multipliers=(1,),
        radar_blocks_per_level=1,
        offset_bounds_bins=(2.0, 1.0, 0.5),
        decode_chunk_size=7,
    )


def test_condition_exclusive_metadata_and_decoder_signature() -> None:
    model = tiny_model()
    metadata = model.architecture_metadata()
    signature = inspect.signature(type(model).decode_queries)

    assert metadata["geometry_decoder_inputs"] == [
        "normalized_rae_fourier_features",
        "condition_latents",
    ]
    assert metadata["pre_selection_local_cube_access"] is False
    assert metadata["low_threshold_radar_query_path_implemented"] is False
    assert metadata["low_threshold_radar_role_if_added"] == "coordinates_only"
    assert metadata["condition_latents_are_only_cube_path"] is True
    assert metadata["same_query_wrong_condition_supported"] is True
    assert metadata["confidence_source"] == "sigmoid_occupancy_logit"
    assert metadata["exact_10000_selection_implemented"] is False
    assert metadata["doppler_head_implemented"] is False
    assert metadata["edm_implemented"] is False
    assert set(signature.parameters) == {
        "self",
        "normalized_rae",
        "condition_latents",
        "chunk_size",
    }
    model.assert_condition_exclusive_contract()


def test_same_query_wrong_condition_field_contract() -> None:
    torch.manual_seed(3)
    model = tiny_model().eval()
    query = queries()
    output = model(
        cube(5),
        query,
        wrong_condition_cube_drae=cube(7),
    )

    assert output["matched_radar_tokens"].shape == (1, 336, 24)
    assert output["matched_condition_latents"].shape == (1, 12, 24)
    assert output["matched"]["occupancy_logit"].shape == (1, 23)
    assert output["matched"]["confidence"].shape == (1, 23)
    assert output["matched"]["residual_bins"].shape == (1, 23, 3)
    assert output["matched"]["refined_normalized_rae"].shape == (1, 23, 3)
    assert output["same_query_wrong_condition"] is True
    assert (
        output["matched"]["query_normalized_rae"].data_ptr()
        == output["wrong"]["query_normalized_rae"].data_ptr()
        == query.data_ptr()
    )
    torch.testing.assert_close(
        output["matched"]["confidence"],
        torch.sigmoid(output["matched"]["occupancy_logit"]),
    )
    assert not torch.allclose(
        output["matched"]["occupancy_logit"],
        output["wrong"]["occupancy_logit"],
    )


def test_decode_is_chunk_invariant() -> None:
    torch.manual_seed(11)
    model = tiny_model().eval()
    query = queries(29)
    condition = model.encode_condition(cube(13))["condition_latents"]
    first = model.decode_queries(query, condition, chunk_size=5)
    second = model.decode_queries(query, condition, chunk_size=29)

    for key in (
        "occupancy_logit",
        "confidence",
        "residual_bins",
        "residual_normalized",
        "refined_normalized_rae",
    ):
        torch.testing.assert_close(first[key], second[key])


def test_stage0_loss_backpropagates_through_condition_and_decoder() -> None:
    torch.manual_seed(17)
    model = tiny_model()
    query = queries(24)
    output = model(
        cube(19),
        query,
        wrong_condition_cube_drae=cube(23),
        chunk_size=6,
    )
    occupancy_target = torch.zeros(1, 24)
    occupancy_target[:, ::3] = 1.0
    residual_target = torch.zeros(1, 24, 3)
    residual_target[:, ::3] = 0.25
    loss = rald_wce_stage0_loss(
        output["matched"],
        output["wrong"],
        query,
        occupancy_target,
        residual_target,
    )
    loss.total.backward()

    assert torch.isfinite(loss.total)
    encoder_gradients = [
        parameter.grad
        for parameter in model.radar_encoder.parameters()
        if parameter.grad is not None
    ]
    decoder_gradients = [
        parameter.grad
        for parameter in model.decoder_attention.parameters()
        if parameter.grad is not None
    ]
    assert encoder_gradients
    assert decoder_gradients
    assert all(torch.isfinite(value).all() for value in encoder_gradients)
    assert all(torch.isfinite(value).all() for value in decoder_gradients)
    assert any(torch.count_nonzero(value) > 0 for value in encoder_gradients)
    assert any(torch.count_nonzero(value) > 0 for value in decoder_gradients)


def test_residual_is_bounded_and_refined_query_stays_in_domain() -> None:
    model = tiny_model()
    output = model(cube(29), queries(31))["matched"]

    assert torch.all(
        output["residual_bins"].abs()
        <= model.offset_bounds_bins + 1e-6
    )
    assert torch.all(output["refined_normalized_rae"].abs() <= 1.0)
