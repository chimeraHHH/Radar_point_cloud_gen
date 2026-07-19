import torch

from models.cube_doppler import query_cube_spectrum
from models.cube_occupancy import CubeOccupancyNet
from models.rald_anchor import (
    FrozenParentRaLDRefiner,
    frozen_parent_anchors,
    normalize_rae_coordinates,
)


def test_normalize_rae_coordinates_maps_endpoints() -> None:
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [255.0, 106.0, 36.0]])

    normalized = normalize_rae_coordinates(coordinates, (256, 107, 37))

    torch.testing.assert_close(
        normalized, torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    )


def test_frozen_parent_anchors_gather_ranked_features_and_spectra() -> None:
    parent = CubeOccupancyNet(
        "rae_max", torch.linspace(-8.0, 8.0, 64), base_channels=4
    ).eval()
    cube = torch.rand(1, 64, 8, 8, 8)

    with torch.inference_mode():
        anchors = frozen_parent_anchors(parent, cube, point_count=19)

    assert anchors.indices_rae.shape == (1, 19, 3)
    assert anchors.parent_features.shape == (1, 19, 4)
    assert anchors.local_cube_spectrum.shape == (1, 19, 64)
    torch.testing.assert_close(
        anchors.local_cube_spectrum.sum(dim=-1), torch.ones(1, 19)
    )
    assert torch.all(anchors.parent_confidence[:, :-1] >= anchors.parent_confidence[:, 1:])


def test_refiner_initialization_preserves_parent_anchor_state() -> None:
    parent = CubeOccupancyNet(
        "full_raed", torch.linspace(-8.0, 8.0, 64), base_channels=4
    )
    model = FrozenParentRaLDRefiner(
        parent,
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
        point_count=23,
        latent_count=8,
        model_dim=32,
        depth=2,
        heads=4,
        head_dim=8,
    ).eval()
    cube = torch.rand(1, 64, 8, 8, 8)

    with torch.inference_mode():
        output = model(cube)

    torch.testing.assert_close(
        output["coordinates_rae"], output["anchor_indices_rae"].to(torch.float32)
    )
    torch.testing.assert_close(output["xyz_m"], output["anchor_xyz_m"])
    torch.testing.assert_close(
        output["doppler_probability"], output["anchor_cube_spectrum"]
    )
    torch.testing.assert_close(output["confidence"], output["anchor_parent_confidence"])
    assert all(not parameter.requires_grad for parameter in model.parent.parameters())
    model.train()
    assert model.parent.training is False


def test_scalar_doppler_head_produces_wrapped_unimodal_distribution() -> None:
    parent = CubeOccupancyNet(
        "rae_max", torch.linspace(-8.0, 8.0, 64), base_channels=4
    )
    model = FrozenParentRaLDRefiner(
        parent,
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
        point_count=11,
        latent_count=8,
        model_dim=32,
        depth=1,
        heads=4,
        head_dim=8,
        doppler_head_mode="scalar",
    ).eval()
    cube = torch.rand(1, 64, 8, 8, 8)

    with torch.inference_mode():
        output = model(cube)

    assert output["doppler_scalar_bin"].shape == (1, 11)
    assert torch.all(output["doppler_scalar_bin"] >= 0.0)
    assert torch.all(output["doppler_scalar_bin"] < 64.0)
    torch.testing.assert_close(
        output["doppler_probability"].sum(dim=-1), torch.ones(1, 11)
    )


def test_distribution_head_queries_cube_at_final_continuous_position() -> None:
    parent = CubeOccupancyNet(
        "rae_max", torch.linspace(-8.0, 8.0, 64), base_channels=4
    )
    model = FrozenParentRaLDRefiner(
        parent,
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
        point_count=7,
        latent_count=8,
        model_dim=32,
        depth=1,
        heads=4,
        head_dim=8,
    ).eval()
    model.refiner.physical_head.offset.bias.data.fill_(1.0)
    cube = torch.rand(1, 64, 8, 8, 8)

    with torch.inference_mode():
        output = model(cube)
        expected = query_cube_spectrum(cube, output["coordinates_rae"][0])

    torch.testing.assert_close(output["point_cube_spectrum"][0], expected)
    torch.testing.assert_close(
        output["doppler_probability"], output["point_cube_spectrum"]
    )


def test_second_step_propagates_beyond_zero_initialized_physical_head() -> None:
    parent = CubeOccupancyNet(
        "rae_max", torch.linspace(-8.0, 8.0, 64), base_channels=4
    )
    model = FrozenParentRaLDRefiner(
        parent,
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
        point_count=17,
        latent_count=8,
        model_dim=32,
        depth=1,
        heads=4,
        head_dim=8,
    )
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1e-3)
    cube = torch.rand(1, 64, 8, 8, 8)

    first = model(cube)
    first_loss = first["offset_bins"].sum() + first["confidence_logit"].sum()
    first_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = model(cube)
    second["confidence_logit"].sum().backward()

    gradient = model.refiner.decoder_attention.attention.query.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0
