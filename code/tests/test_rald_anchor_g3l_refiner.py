import math

import pytest
import torch
import torch.nn as nn

from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.rald_anchor_ldm_refiner import RaLDAnchorLDMRefiner
from models.rald_anchor_ldm import RaLDAnchorLDM
from models.rald_matched import FullRAEDRadarTokenEncoder


MODEL_DIM = 16
ANCHOR_FEATURE_DIM = 6
SPATIAL_SHAPE = (4, 3, 2)


class TinyFrozenG3R(nn.Module):
    def __init__(self, range_m, azimuth_rad, elevation_rad) -> None:
        super().__init__()
        self.query_state = nn.Parameter(torch.randn(3, MODEL_DIM))
        self.anchor_state = nn.Parameter(torch.randn(3, ANCHOR_FEATURE_DIM))
        self.logit = nn.Parameter(torch.tensor([0.2, -0.1, 0.4]))
        self.register_buffer(
            "coordinates",
            torch.tensor(
                [[1.0, 1.0, 0.4], [2.0, 0.8, 0.6], [2.5, 1.3, 0.5]],
                dtype=torch.float32,
            ),
        )
        self.register_buffer("range_m", range_m)
        self.register_buffer("azimuth_rad", azimuth_rad)
        self.register_buffer("elevation_rad", elevation_rad)

    def forward(self, cube_drae: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = cube_drae.shape[0]
        coordinates = self.coordinates.unsqueeze(0).expand(batch, -1, -1)
        query_features = self.query_state.unsqueeze(0).expand(batch, -1, -1)
        anchor_features = self.anchor_state.unsqueeze(0).expand(batch, -1, -1)
        confidence_logit = self.logit.unsqueeze(0).expand(batch, -1)
        xyz = continuous_rae_to_xyz(
            coordinates.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        ).reshape(batch, coordinates.shape[1], 3)
        return {
            "coordinates_rae": coordinates,
            "query_features": query_features,
            "anchor_features": anchor_features,
            "xyz_m": xyz,
            "confidence_logit": confidence_logit,
            "confidence": torch.sigmoid(confidence_logit),
        }


def axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(0.0, 3.0, SPATIAL_SHAPE[0]),
        torch.linspace(-0.3, 0.3, SPATIAL_SHAPE[1]),
        torch.linspace(-0.1, 0.1, SPATIAL_SHAPE[2]),
    )


def tiny_encoder() -> FullRAEDRadarTokenEncoder:
    return FullRAEDRadarTokenEncoder(
        log_center=1.0,
        log_scale=0.5,
        spectral_channels=4,
        encoded_shape=SPATIAL_SHAPE,
        encoded_channels=4,
        token_dim=MODEL_DIM,
        base_channels=4,
        channel_multipliers=(1,),
        blocks_per_level=1,
    )


def build_model() -> tuple[RaLDAnchorLDMRefiner, TinyFrozenG3R]:
    torch.manual_seed(71)
    range_m, azimuth_rad, elevation_rad = axes()
    parent = TinyFrozenG3R(range_m, azimuth_rad, elevation_rad)
    ldm = RaLDAnchorLDM(
        anchor_feature_dim=ANCHOR_FEATURE_DIM,
        latent_count=5,
        latent_dim=4,
        model_dim=MODEL_DIM,
        decoder_depth=1,
        denoiser_depth=1,
        heads=4,
        head_dim=4,
        edm_steps=2,
        radar_encoder=tiny_encoder(),
    )
    return (
        RaLDAnchorLDMRefiner(
            parent,
            ldm,
            range_m,
            azimuth_rad,
            elevation_rad,
            model_dim=MODEL_DIM,
        ),
        parent,
    )


def cube() -> torch.Tensor:
    doppler = torch.arange(64, dtype=torch.float32).view(1, 64, 1, 1, 1)
    radius = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1)
    azimuth = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 3, 1)
    elevation = torch.arange(2, dtype=torch.float32).view(1, 1, 1, 1, 2)
    center = torch.remainder(7.0 * radius + 3.0 * azimuth + elevation, 64.0)
    delta = torch.remainder(doppler - center + 32.0, 64.0) - 32.0
    return torch.exp(-0.5 * (delta / 2.5).square()) + 1e-3


def target_state() -> tuple[torch.Tensor, torch.Tensor]:
    target_rae = torch.tensor(
        [[0, 1, 0], [1, 2, 1], [2, 0, 1], [3, 1, 0]], dtype=torch.long
    )
    confidence = torch.tensor([0.9, 0.6, 0.8, 0.4])
    return target_rae, confidence


def test_identity_initialization_preserves_frozen_g3r_geometry() -> None:
    model, parent = build_model()
    target_rae, confidence = target_state()

    output = model(
        cube(), target_rae, confidence, sample_posterior=False
    )

    expected_coordinates = parent.coordinates.unsqueeze(0)
    torch.testing.assert_close(output["offset_bins"], torch.zeros_like(output["offset_bins"]))
    torch.testing.assert_close(output["coordinates_rae"], expected_coordinates)
    torch.testing.assert_close(output["xyz_m"], output["anchor_xyz_m"])
    torch.testing.assert_close(output["confidence"], torch.sigmoid(parent.logit)[None])
    torch.testing.assert_close(
        output["doppler_probability"], output["point_cube_spectrum"]
    )


def test_target_permutation_leaves_posterior_and_mean_decode_unchanged() -> None:
    model, _ = build_model()
    target_rae, confidence = target_state()
    permutation = torch.tensor([2, 0, 3, 1])

    first = model(cube(), target_rae, confidence, sample_posterior=False)
    second = model(
        cube(),
        target_rae[permutation],
        confidence[permutation],
        sample_posterior=False,
    )

    for key in (
        "posterior_mean",
        "posterior_log_variance",
        "coordinates_rae",
        "doppler_probability",
        "confidence",
    ):
        torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=1e-6)


def reconstruction_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    desired_probability = output["anchor_cube_spectrum"].roll(5, dims=-1).detach()
    return (
        (output["offset_bins"] - 0.2).square().mean()
        - (desired_probability * output["doppler_probability"].clamp_min(1e-8).log())
        .sum(dim=-1)
        .mean()
        + (output["confidence"] - 0.8).square().mean()
    )


def test_physical_loss_reaches_vae_after_zero_head_first_update() -> None:
    model, parent = build_model()
    target_rae, confidence = target_state()
    optimizer = torch.optim.SGD(model.vae_parameters(), lr=0.2)

    first = model(cube(), target_rae, confidence, sample_posterior=False)
    reconstruction_loss(first).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.physical_head.parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = model(cube(), target_rae, confidence, sample_posterior=False)
    (reconstruction_loss(second) + 1e-3 * second["posterior_kl"].mean()).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.ldm.posterior_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.ldm.decoder.parameters()
    )
    assert all(parameter.grad is None for parameter in parent.parameters())


def test_final_position_requeries_cube_spectrum() -> None:
    model, _ = build_model()
    target_rae, confidence = target_state()
    with torch.no_grad():
        model.physical_head.offset.bias.copy_(
            torch.tensor([math.atanh(0.8), 0.0, 0.0])
        )

    measured = cube()
    output = model(measured, target_rae, confidence, sample_posterior=False)
    coordinates = output["coordinates_rae"]
    batch = torch.zeros(coordinates.shape[1], 1)
    direct = query_cube_spectrum(
        measured, torch.cat((batch, coordinates[0]), dim=1)
    ).unsqueeze(0)

    torch.testing.assert_close(output["point_cube_spectrum"], direct)
    torch.testing.assert_close(output["doppler_probability"], direct)
    assert not torch.allclose(output["anchor_cube_spectrum"], direct)


def test_geometry_parent_and_edm_remain_frozen_in_train_mode() -> None:
    model, parent = build_model()
    model.train()
    target_rae, confidence = target_state()
    output = model(cube(), target_rae, confidence, sample_posterior=True)
    (reconstruction_loss(output) + output["posterior_kl"].mean()).backward()

    assert not parent.training
    assert not model.ldm.edm.training
    assert all(not parameter.requires_grad for parameter in parent.parameters())
    assert all(parameter.grad is None for parameter in parent.parameters())
    assert all(not parameter.requires_grad for parameter in model.ldm.edm.parameters())
    assert all(parameter.grad is None for parameter in model.ldm.edm.parameters())


def test_edm_uses_one_fixed_seed_and_isolates_shuffled_condition() -> None:
    model, _ = build_model()
    measured = cube()
    first = model.sample_edm(measured, [718], steps=2)
    second = model.sample_edm(measured, [718], steps=2)
    shuffled = model.sample_edm(
        measured,
        [718],
        condition_cube_drae=measured.roll(1, dims=2),
        steps=2,
    )

    torch.testing.assert_close(first["latent"], second["latent"], rtol=0, atol=0)
    torch.testing.assert_close(first["xyz_m"], second["xyz_m"], rtol=0, atol=0)
    torch.testing.assert_close(
        first["anchor_cube_spectrum"], shuffled["anchor_cube_spectrum"]
    )
    assert first["condition_is_measured_cube"] is True
    assert shuffled["condition_is_measured_cube"] is False
    with pytest.raises(ValueError, match="one fixed seed per frame"):
        model.sample_edm(measured, [], steps=2)
