from copy import deepcopy
from math import prod

import pytest
import torch
import torch.nn as nn

from models.cube_doppler import query_cube_spectrum
from models.cube_occupancy import CubeOccupancyNet
from models.rald_anchor import FrozenParentRaLDRefiner
from models.rald_anchor_temporal import (
    FrozenParentRaLDTemporalRefiner,
    RaLDTemporalAnchorLatentRefiner,
)
from models.temporal_prior import WarpedPrior, ego_pose_warp


class TinyCurrentRadarEncoder(nn.Module):
    def __init__(self, token_dim: int = 32, token_count: int = 8) -> None:
        super().__init__()
        self.token_count = token_count
        self.projection = nn.Linear(64, token_dim)

    def forward(self, cube_drae: torch.Tensor) -> torch.Tensor:
        pooled = cube_drae.mean(dim=(2, 3, 4))
        token = self.projection(pooled)
        return token[:, None].expand(-1, self.token_count, -1)


class TinyPriorRadarEncoder(nn.Module):
    def __init__(self, token_dim: int = 32, token_count: int = 8) -> None:
        super().__init__()
        self.projection = nn.Linear(5, token_dim)
        self.token_count = token_count

    def forward(self, prior_crae: torch.Tensor) -> torch.Tensor:
        pooled = prior_crae.mean(dim=(2, 3, 4))
        token = self.projection(pooled)
        return token[:, None].expand(-1, self.token_count, -1)


def axes() -> tuple[torch.Tensor, ...]:
    return (
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
        torch.linspace(-8.0, 8.0, 64),
    )


def prior() -> WarpedPrior:
    range_m, azimuth_rad, elevation_rad, doppler_mps = axes()
    probability = torch.nn.functional.one_hot(
        torch.tensor([5, 21, 58]), 64
    ).float()
    step = torch.median(torch.diff(doppler_mps))
    return ego_pose_warp(
        torch.tensor([[10.0, 0.0, 0.0], [20.0, 1.0, 0.0], [35.0, -2.0, 1.0]]),
        probability,
        torch.tensor([0.9, 0.7, 0.8]),
        torch.eye(4),
        doppler_mps,
        doppler_mps[0],
        step * doppler_mps.numel(),
        range_m,
        azimuth_rad,
        elevation_rad,
    )


def paired_models(mode: str):
    torch.manual_seed(17)
    range_m, azimuth_rad, elevation_rad, doppler_mps = axes()
    parent = CubeOccupancyNet(
        "rae_max", doppler_mps, base_channels=4
    )
    base = FrozenParentRaLDRefiner(
        parent,
        range_m,
        azimuth_rad,
        elevation_rad,
        point_count=13,
        latent_count=8,
        model_dim=32,
        depth=1,
        heads=4,
        head_dim=8,
        radar_encoder=TinyCurrentRadarEncoder(),
        radar_token_dim=32,
    ).eval()
    temporal = FrozenParentRaLDTemporalRefiner(
        deepcopy(parent),
        range_m,
        azimuth_rad,
        elevation_rad,
        doppler_mps,
        TinyCurrentRadarEncoder(),
        temporal_fusion_mode=mode,
        point_count=13,
        latent_count=8,
        model_dim=32,
        depth=1,
        heads=4,
        head_dim=8,
        prior_radar_encoder=(
            TinyPriorRadarEncoder() if mode == "token" else None
        ),
    ).eval()
    temporal.load_single_frame_refiner(
        base.refiner.state_dict(), base.radar_encoder.state_dict()
    )
    return base, temporal


@pytest.mark.parametrize("mode", ("token", "latent", "query"))
def test_zero_gate_exactly_preserves_single_frame_rald(mode: str) -> None:
    base, temporal = paired_models(mode)
    cube = torch.rand(1, 64, 8, 8, 8)

    with torch.inference_mode():
        expected = base(cube)
        actual = temporal(cube, prior())

    for key in (
        "coordinates_rae",
        "xyz_m",
        "doppler_probability",
        "confidence",
        "query_features",
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)
    assert actual["temporal_prior_count"].item() == 3


def test_default_token_fusion_matches_rald_336_token_hierarchy() -> None:
    range_m, azimuth_rad, elevation_rad, doppler_mps = axes()
    production = FrozenParentRaLDTemporalRefiner(
        CubeOccupancyNet("rae_max", doppler_mps, base_channels=4),
        range_m,
        azimuth_rad,
        elevation_rad,
        doppler_mps,
        TinyCurrentRadarEncoder(token_count=336),
        temporal_fusion_mode="token",
        point_count=13,
        latent_count=8,
        model_dim=32,
        depth=1,
        heads=4,
        head_dim=8,
        prior_base_channels=4,
    )
    assert prod(production.prior_radar_encoder.encoded_shape) == 336


def test_temporal_refiner_queries_current_cube_at_final_position() -> None:
    _, temporal = paired_models("query")
    temporal.refiner.physical_head.offset.bias.data.fill_(1.0)
    cube = torch.rand(1, 64, 8, 8, 8)

    with torch.inference_mode():
        output = temporal(cube, prior())
        expected = query_cube_spectrum(cube, output["coordinates_rae"][0])

    torch.testing.assert_close(output["point_cube_spectrum"][0], expected)


@pytest.mark.parametrize(
    ("mode", "temporal_input", "projection_name"),
    (
        ("latent", "prior_tokens", "prior_latent_projection"),
        ("query", "draft_features", "prior_query_projection"),
    ),
)
def test_second_step_reaches_rald_temporal_adapter(
    mode: str, temporal_input: str, projection_name: str
) -> None:
    torch.manual_seed(23)
    model = RaLDTemporalAnchorLatentRefiner(
        anchor_feature_dim=4,
        temporal_fusion_mode=mode,
        latent_count=8,
        model_dim=32,
        depth=1,
        heads=4,
        head_dim=8,
        spectrum_bins=64,
        radar_token_dim=32,
    )
    optimizer = torch.optim.SGD(model.temporal_parameters(), lr=0.1)
    inputs = {
        "anchor_normalized_rae": torch.rand(1, 11, 3) * 2.0 - 1.0,
        "anchor_features": torch.randn(1, 11, 4),
        "local_cube_spectrum": torch.rand(1, 11, 64),
        "radar_tokens": torch.randn(1, 8, 32),
        temporal_input: torch.randn(
            1,
            11 if mode == "query" else 7,
            11 if mode == "query" else 8,
        ),
    }

    model(**inputs)["query_features"].square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(**inputs)["query_features"].square().mean().backward()

    projection = getattr(model, projection_name)[0]
    assert projection.weight.grad is not None
    assert torch.count_nonzero(projection.weight.grad).item() > 0


@pytest.mark.parametrize("mode", ("token", "latent", "query"))
def test_temporal_parameter_partition_is_complete(mode: str) -> None:
    _, model = paired_models(mode)
    temporal_ids = {id(parameter) for parameter in model.temporal_parameters()}
    refinement_ids = {id(parameter) for parameter in model.refinement_parameters()}
    base_ids = {id(parameter) for parameter in model.base_refinement_parameters()}

    assert temporal_ids
    assert temporal_ids <= refinement_ids
    assert not temporal_ids & base_ids
    assert temporal_ids | base_ids == refinement_ids
