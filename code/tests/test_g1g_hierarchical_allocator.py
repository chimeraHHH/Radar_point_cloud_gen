import torch

from models.g1g_hierarchical_allocator import (
    G1GConditionExclusiveHierarchy,
)


SPATIAL_SHAPE = (16, 7, 3)


def axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(0.0, 60.0, SPATIAL_SHAPE[0]),
        torch.linspace(-0.8, 0.8, SPATIAL_SHAPE[1]),
        torch.linspace(-0.25, 0.25, SPATIAL_SHAPE[2]),
    )


def cube(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(
        (1, 64, *SPATIAL_SHAPE),
        generator=generator,
    )


def tiny_model() -> G1GConditionExclusiveHierarchy:
    range_m, azimuth, elevation = axes()
    return G1GConditionExclusiveHierarchy(
        range_m,
        azimuth,
        elevation,
        log_center=0.0,
        log_scale=1.0,
        model_dim=16,
        depth=2,
        heads=4,
        head_dim=4,
        fourier_frequency_dim=12,
        radar_base_channels=4,
        radar_spectral_channels=4,
        radar_encoded_shape=SPATIAL_SHAPE,
        radar_encoded_channels=4,
        radar_channel_multipliers=(1,),
        radar_blocks_per_level=1,
    )


def has_nonzero_finite_gradient(module: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return bool(
        gradients
        and all(torch.isfinite(gradient).all() for gradient in gradients)
        and any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    )


def test_exact_global_center_child_and_point_contract() -> None:
    model = tiny_model()
    output = model(cube(1))
    metadata = output["architecture_metadata"]

    assert output["radar_tokens"].shape == (1, 336, 16)
    assert output["center_coordinates_rae"].shape == (1, 2_500, 3)
    assert output["center_cube_spectrum"].shape == (1, 2_500, 64)
    assert output["child_coordinates_rae"].shape == (1, 2_500, 4, 3)
    assert output["coordinates_rae"].shape == (1, 10_000, 3)
    assert output["xyz_m"].shape == (1, 10_000, 3)
    assert output["point_cube_spectrum"].shape == (1, 10_000, 64)
    assert int(output["radar_token_count"]) == 336
    assert int(output["center_count"]) == 2_500
    assert int(output["children_per_center"]) == 4
    assert int(output["final_point_count"]) == 10_000
    assert metadata["global_radar_token_count"] == 336
    assert metadata["center_query_count"] == 2_500
    assert metadata["children_per_center"] == 4
    assert metadata["final_point_count"] == 10_000

    parent_counts = torch.bincount(
        output["point_parent_center_index"],
        minlength=2_500,
    )
    assert torch.all(parent_counts == 4)
    assert output["point_child_index"].tolist() == [0, 1, 2, 3] * 2_500


def test_center_allocation_has_no_measured_local_cube_dependency() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    measured = cube(5).requires_grad_(True)
    condition = cube(7).requires_grad_(True)
    output = model(measured, condition_cube_drae=condition)
    allocation_loss = (
        output["center_coordinates_rae"].square().mean()
        + output["center_score_logit"].square().mean()
    )
    measured_gradient, condition_gradient = torch.autograd.grad(
        allocation_loss,
        (measured, condition),
        allow_unused=True,
    )

    assert measured_gradient is None
    assert condition_gradient is not None
    assert torch.isfinite(condition_gradient).all()
    assert torch.count_nonzero(condition_gradient) > 0
    metadata = output["architecture_metadata"]
    assert metadata["pre_allocation_sources"] == [
        "learned_center_queries",
        "normalized_center_templates",
        "global_full_raed_tokens",
    ]
    assert metadata["forbidden_pre_allocation_sources"] == [
        "local_cube_spectrum",
        "local_cube_energy",
        "local_cube_neighborhood",
        "proposal_score",
    ]
    assert metadata["local_cube_sampling_stage"] == "after_center_allocation"


def test_local_cube_spectrum_enters_only_post_allocation_patch_path() -> None:
    torch.manual_seed(11)
    model = tiny_model()
    measured = cube(13).requires_grad_(True)
    condition = cube(17).requires_grad_(True)
    output = model(measured, condition_cube_drae=condition)
    local_loss = (
        output["center_cube_spectrum"].square().mean()
        + output["coordinates_rae"].square().mean()
        + output["point_cube_spectrum"].square().mean()
    )
    measured_gradient = torch.autograd.grad(local_loss, measured)[0]

    assert torch.isfinite(measured_gradient).all()
    assert torch.count_nonzero(measured_gradient) > 0
    torch.testing.assert_close(
        output["center_cube_spectrum"].sum(dim=-1),
        torch.ones(1, 2_500),
    )
    torch.testing.assert_close(
        output["point_cube_spectrum"].sum(dim=-1),
        torch.ones(1, 10_000),
    )


def test_child_offsets_obey_bin_and_physical_cell_diagonal_bounds() -> None:
    torch.manual_seed(19)
    model = tiny_model()
    output = model(cube(23))

    assert torch.all(
        output["child_offset_bins"].abs()
        <= model.child_offset_bound_bins + 1e-6
    )
    physical_distance = torch.linalg.vector_norm(
        output["child_physical_offset_m"],
        dim=-1,
    )
    assert torch.all(
        physical_distance
        <= output["center_cell_diagonal_m"] + 1e-5
    )


def test_condition_shuffle_hooks_change_global_allocation() -> None:
    torch.manual_seed(29)
    model = tiny_model().eval()
    measured = cube(31)
    first = model(measured, condition_cube_drae=cube(37))
    second = model(measured, condition_cube_drae=cube(41))

    assert not torch.allclose(
        first["radar_tokens"],
        second["radar_tokens"],
    )
    assert not torch.allclose(
        first["center_query_features"],
        second["center_query_features"],
    )
    assert not torch.allclose(
        first["center_coordinates_rae"],
        second["center_coordinates_rae"],
    )


def test_full_scaffold_has_finite_gradients_through_both_stages() -> None:
    torch.manual_seed(43)
    model = tiny_model()
    output = model(cube(47))
    loss = (
        output["center_score_logit"].square().mean()
        + output["coordinates_rae"].square().mean()
        + output["confidence_logit"].square().mean()
        + output["center_cube_spectrum"].square().mean()
        + output["point_cube_spectrum"].square().mean()
    )
    loss.backward()

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert has_nonzero_finite_gradient(model.radar_encoder)
    assert has_nonzero_finite_gradient(model.center_queries)
    assert has_nonzero_finite_gradient(model.allocation_blocks)
    assert has_nonzero_finite_gradient(model.center_coordinate_head)
    assert has_nonzero_finite_gradient(model.center_spectrum_projection)
    assert has_nonzero_finite_gradient(model.patch_decoder)
    assert has_nonzero_finite_gradient(model.child_offset_head)
