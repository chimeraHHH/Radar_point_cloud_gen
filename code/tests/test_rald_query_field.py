import torch

from models.rald_query_field import (
    RaLDQueryField,
    coarse_query_templates,
    stable_radar_proposals,
    tetrahedral_local_templates,
)


SPATIAL_SHAPE = (5, 5, 3)


def axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(0.0, 8.0, SPATIAL_SHAPE[0]),
        torch.linspace(-0.5, 0.5, SPATIAL_SHAPE[1]),
        torch.linspace(-0.2, 0.2, SPATIAL_SHAPE[2]),
    )


def cube(scale: float = 1.0) -> torch.Tensor:
    measured = torch.full((1, 64, *SPATIAL_SHAPE), 1e-3)
    measured[:, :, 1, 1, 1] = scale
    measured[:, :, 3, 3, 1] = 0.8 * scale
    measured[:, 7, 1, 1, 1] = 2.0 * scale
    measured[:, 41, 3, 3, 1] = 1.6 * scale
    return measured


def tiny_model(*, depth: int = 2) -> RaLDQueryField:
    range_m, azimuth, elevation = axes()
    return RaLDQueryField(
        range_m,
        azimuth,
        elevation,
        log_center=0.0,
        log_scale=1.0,
        base_seed_count=2,
        selected_coarse_count=3,
        latent_count=4,
        model_dim=12,
        depth=depth,
        heads=3,
        head_dim=4,
        fourier_frequency_dim=12,
        radar_base_channels=4,
        radar_spectral_channels=4,
        radar_encoded_shape=SPATIAL_SHAPE,
        radar_encoded_channels=4,
        radar_channel_multipliers=(1,),
        radar_blocks_per_level=1,
        offset_bounds_bins=(1.0, 0.5, 0.5),
        nms_kernel=(3, 3, 3),
        decode_chunk_size=7,
    )


def has_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in module.parameters()
    )


def test_exact_mixed_latent_topology_paths_receive_gradient() -> None:
    torch.manual_seed(7)
    model = tiny_model(depth=1)
    proposal_tokens = torch.randn(1, 2, model.model_dim)
    radar_tokens = torch.randn(
        1, model.expected_radar_token_count, model.model_dim
    )
    latent = model.encode_latent(proposal_tokens, radar_tokens)
    latent.square().mean().backward()

    assert model.static_latents.weight.grad is not None
    assert torch.count_nonzero(model.static_latents.weight.grad) > 0
    assert model.dynamic_latents.weight.grad is not None
    assert torch.count_nonzero(model.dynamic_latents.weight.grad) > 0
    assert has_gradient(model.dynamic_proposal_attention)
    assert has_gradient(model.mixed_projection)
    assert has_gradient(model.post_mix_proposal_attention)
    assert has_gradient(model.post_mix_feed_forward)


def test_every_latent_block_calls_and_trains_full_raed_condition_attention() -> None:
    torch.manual_seed(11)
    model = tiny_model(depth=3)
    calls = [0 for _ in model.latent_blocks]
    handles = []

    for index, block in enumerate(model.latent_blocks):
        def count_call(_module, _inputs, _output, *, block_index=index):
            calls[block_index] += 1

        handles.append(block.radar_attention.register_forward_hook(count_call))

    proposal_tokens = torch.randn(1, 2, model.model_dim)
    radar_tokens = torch.randn(
        1, model.expected_radar_token_count, model.model_dim
    )
    model.encode_latent(proposal_tokens, radar_tokens).sum().backward()
    for handle in handles:
        handle.remove()

    assert calls == [1, 1, 1]
    assert all(has_gradient(block.radar_attention) for block in model.latent_blocks)


def test_coarse_top_selection_expands_to_fixed_count_and_training_queries() -> None:
    torch.manual_seed(13)
    model = tiny_model()
    occupancy_queries = torch.tensor(
        [[[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
    )
    output = model(cube(), occupancy_queries_rae=occupancy_queries)
    provenance = output["coarse_selection_provenance"]

    assert coarse_query_templates().shape == (32, 3)
    assert tetrahedral_local_templates().shape == (4, 3)
    assert output["coordinates_rae"].shape == (1, 12, 3)
    assert output["xyz_m"].shape == (1, 12, 3)
    assert output["confidence"].shape == (1, 12)
    assert output["point_cube_spectrum"].shape == (1, 12, 64)
    assert output["zero_offset_xyz_m"].shape == (1, 12, 3)
    assert provenance["selected_coarse_index"].shape == (1, 3)
    assert provenance["selected_seed_index"].max() < 2
    assert provenance["selected_coarse_template_index"].max() < 32
    assert provenance["point_selected_coarse_index"].shape == (12,)
    assert provenance["point_tetrahedral_template_index"].tolist() == [
        0,
        1,
        2,
        3,
    ] * 3
    assert output["training_query_occupancy_logit"].shape == (1, 3)
    assert output["training_query_coordinates_rae"].shape == (1, 3, 3)
    assert output["training_query_point_cube_spectrum"].shape == (1, 3, 64)
    assert int(output["radar_token_count"]) == 75
    assert int(output["coarse_query_count"]) == 64
    assert int(output["selected_coarse_count"]) == 3
    assert int(output["final_point_count"]) == 12


def test_rald_occupancy_init_preserves_zero_offset_control() -> None:
    model = tiny_model()
    output = model(cube())

    torch.testing.assert_close(output["xyz_m"], output["zero_offset_xyz_m"])
    assert torch.count_nonzero(model.occupancy_head.weight) > 0
    assert torch.count_nonzero(output["occupancy_logit"]) > 0
    assert torch.count_nonzero(output["confidence_logit"]) == 0
    assert torch.count_nonzero(model.offset_head.weight) == 0
    assert torch.count_nonzero(model.offset_head.bias) == 0


def test_zero_offset_control_disables_coarse_and_final_residuals() -> None:
    model = tiny_model()
    with torch.no_grad():
        model.offset_head.bias.copy_(torch.tensor([0.5, -0.25, 0.25]))
    output = model(cube())
    provenance = output["coarse_selection_provenance"]
    expected_queries = (
        provenance["selected_zero_offset_center_coordinates_rae"][:, :, None, :]
        + model.local_templates[None, None, :, :]
    )
    expected_queries = model._clamp_coordinates(
        expected_queries.reshape(1, model.point_count, 3)
    )

    torch.testing.assert_close(
        output["zero_offset_coordinates_rae"],
        expected_queries,
    )
    assert not torch.allclose(
        output["coordinates_rae"],
        output["zero_offset_coordinates_rae"],
    )


def test_condition_shuffle_changes_only_conditioned_latent_path() -> None:
    torch.manual_seed(19)
    model = tiny_model(depth=2)
    measured = cube()
    shuffled = cube(scale=3.0)
    current = model(measured)
    cross_scene = model(measured, condition_cube_drae=shuffled)

    torch.testing.assert_close(
        current["proposal_coordinates_rae"],
        cross_scene["proposal_coordinates_rae"],
    )
    torch.testing.assert_close(
        current["point_cube_spectrum"],
        cross_scene["point_cube_spectrum"],
    )
    assert not torch.allclose(current["latent"], cross_scene["latent"])


def test_absolute_energy_intervention_changes_query_token() -> None:
    torch.manual_seed(17)
    model = tiny_model(depth=1)
    coordinate = torch.tensor([[[1.0, 1.0, 1.0]]])
    low = torch.zeros((1, 64, *SPATIAL_SHAPE))
    high = torch.zeros_like(low)
    low[:, :, 1, 1, 1] = 1.0
    high[:, :, 1, 1, 1] = 8.0

    low_evidence = model.query_tokens(low, coordinate)
    high_evidence = model.query_tokens(high, coordinate)

    torch.testing.assert_close(
        low_evidence["local_spectrum"], high_evidence["local_spectrum"]
    )
    assert torch.all(
        high_evidence["absolute_log_energy"]
        > low_evidence["absolute_log_energy"]
    )
    torch.testing.assert_close(
        low_evidence["normalized_log_energy"],
        torch.full(
            (1, 1, 1),
            torch.log10(torch.tensor(2.0)).item(),
        ),
    )
    torch.testing.assert_close(
        high_evidence["normalized_log_energy"],
        torch.full(
            (1, 1, 1),
            torch.log10(torch.tensor(9.0)).item(),
        ),
    )
    assert float(high_evidence["normalized_log_energy"].abs().max()) <= 4.0
    assert not torch.allclose(low_evidence["tokens"], high_evidence["tokens"])


def test_plateau_nms_is_deterministic_and_respects_local_suppression() -> None:
    plateau = torch.ones((1, 64, *SPATIAL_SHAPE))
    first = stable_radar_proposals(
        plateau, seed_count=4, nms_kernel=(3, 3, 3)
    )
    second = stable_radar_proposals(
        plateau, seed_count=4, nms_kernel=(3, 3, 3)
    )

    torch.testing.assert_close(first.coordinates_rae, second.coordinates_rae)
    torch.testing.assert_close(first.flat_index, second.flat_index)
    assert int(first.flat_index[0, 0]) == 0
    distance = (
        first.coordinates_rae[:, :, None, :]
        - first.coordinates_rae[:, None, :, :]
    ).abs()
    identity = torch.eye(4, dtype=torch.bool).unsqueeze(0)
    adjacent = (distance <= 1.0).all(dim=-1) & ~identity
    assert not adjacent.any()


def test_cached_proposal_indices_are_forward_equivalent() -> None:
    torch.manual_seed(23)
    model = tiny_model()
    measured = cube()
    proposals = stable_radar_proposals(
        measured,
        seed_count=model.base_seed_count,
        nms_kernel=model.nms_kernel,
    )

    direct = model(measured)
    cached = model(measured, proposal_flat_index=proposals.flat_index)

    for name in (
        "proposal_coordinates_rae",
        "proposal_integrated_log_energy",
        "latent",
        "occupancy_logit",
        "confidence_logit",
        "offset_bins",
        "coordinates_rae",
        "xyz_m",
        "point_cube_spectrum",
    ):
        torch.testing.assert_close(direct[name], cached[name])
    assert bool(direct["proposal_cache_used"].item()) is False
    assert bool(cached["proposal_cache_used"].item()) is True
