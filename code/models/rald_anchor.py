"""RaLD set-latent refinement of frozen occupancy-parent anchors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn as nn

from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.cube_occupancy import CubeOccupancyNet
from models.rald_matched import RaLDAnchorLatentRefiner


@dataclass(frozen=True)
class FrozenAnchorBatch:
    indices_rae: torch.Tensor
    normalized_rae: torch.Tensor
    parent_features: torch.Tensor
    parent_logits: torch.Tensor
    parent_confidence: torch.Tensor
    local_cube_spectrum: torch.Tensor


def normalize_rae_coordinates(
    coordinates_rae: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Map RAE bin coordinates to the RaLD Fourier-embedding interval [-1, 1]."""

    if coordinates_rae.shape[-1] != 3:
        raise ValueError("RAE coordinates must end in three values")
    if len(spatial_shape) != 3 or any(size <= 1 for size in spatial_shape):
        raise ValueError(f"Invalid RAE spatial shape {spatial_shape}")
    maximum = coordinates_rae.new_tensor([size - 1 for size in spatial_shape])
    return 2.0 * coordinates_rae / maximum - 1.0


def frozen_parent_anchors(
    parent: CubeOccupancyNet,
    cube_drae: torch.Tensor,
    point_count: int = 10_000,
) -> FrozenAnchorBatch:
    """Decode confidence-ranked parent cells and their local physical evidence."""

    if point_count <= 0:
        raise ValueError("Anchor point count must be positive")
    if cube_drae.ndim != 5 or cube_drae.shape[1] != 64:
        raise ValueError(f"Expected Cube shape (B,64,R,A,E), got {cube_drae.shape}")
    spatial_shape = tuple(int(size) for size in cube_drae.shape[2:])
    features = parent.spatial_features(cube_drae)
    logits = parent.head(features).squeeze(1)
    batch_size = logits.shape[0]
    count = min(point_count, logits[0].numel())
    parent_logits, flat_index = torch.topk(
        logits.flatten(start_dim=1), count, dim=1, sorted=True
    )
    range_count, azimuth_count, elevation_count = spatial_shape
    radius = flat_index // (azimuth_count * elevation_count)
    remainder = flat_index % (azimuth_count * elevation_count)
    azimuth = remainder // elevation_count
    elevation = remainder % elevation_count
    indices = torch.stack((radius, azimuth, elevation), dim=-1)
    flattened_features = features.flatten(start_dim=2).transpose(1, 2)
    gather_index = flat_index[..., None].expand(-1, -1, features.shape[1])
    parent_features = flattened_features.gather(1, gather_index)

    batch_index = torch.arange(batch_size, device=indices.device)
    batch_index = batch_index[:, None].expand(-1, count)
    spectrum_query = torch.cat(
        (batch_index.reshape(-1, 1), indices.reshape(-1, 3)), dim=1
    )
    spectrum = query_cube_spectrum(cube_drae, spectrum_query)
    spectrum = spectrum.reshape(batch_size, count, 64)
    return FrozenAnchorBatch(
        indices_rae=indices,
        normalized_rae=normalize_rae_coordinates(indices.to(features), spatial_shape),
        parent_features=parent_features,
        parent_logits=parent_logits,
        parent_confidence=torch.sigmoid(parent_logits),
        local_cube_spectrum=spectrum,
    )


class FrozenParentRaLDRefiner(nn.Module):
    """Keep parent geometry fixed while RaLD refines point-level physical state."""

    def __init__(
        self,
        parent: CubeOccupancyNet,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        *,
        point_count: int = 10_000,
        latent_count: int = 512,
        model_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        head_dim: int = 64,
        radar_encoder: nn.Module | None = None,
        radar_token_dim: int | None = None,
        doppler_head_mode: str = "distribution",
    ) -> None:
        super().__init__()
        if range_m.shape != (256,):
            raise ValueError("Expected 256 K-Radar range bins")
        if azimuth_rad.shape != (107,) or elevation_rad.shape != (37,):
            raise ValueError("Expected 107 x 37 K-Radar angular bins")
        self.parent = parent
        if (radar_encoder is None) != (radar_token_dim is None):
            raise ValueError("Radar encoder and token dimension must be provided together")
        self.radar_encoder = radar_encoder
        self.point_count = point_count
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()
        self.refiner = RaLDAnchorLatentRefiner(
            anchor_feature_dim=parent.head.in_channels,
            latent_count=latent_count,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
            spectrum_bins=64,
            radar_token_dim=radar_token_dim,
            doppler_head_mode=doppler_head_mode,
        )
        self.register_buffer("range_m", range_m.float(), persistent=True)
        self.register_buffer("azimuth_rad", azimuth_rad.float(), persistent=True)
        self.register_buffer("elevation_rad", elevation_rad.float(), persistent=True)

    def train(self, mode: bool = True) -> FrozenParentRaLDRefiner:
        super().train(mode)
        self.parent.eval()
        return self

    def refinement_parameters(self):
        groups = [self.refiner.parameters()]
        if self.radar_encoder is not None:
            groups.append(self.radar_encoder.parameters())
        return chain.from_iterable(groups)

    def forward(self, cube_drae: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            anchors = frozen_parent_anchors(
                self.parent, cube_drae, point_count=self.point_count
            )
        radar_tokens = (
            None if self.radar_encoder is None else self.radar_encoder(cube_drae)
        )
        refined = self.refiner(
            anchors.normalized_rae,
            anchors.parent_features,
            anchors.local_cube_spectrum,
            radar_tokens=radar_tokens,
        )
        coordinates = anchors.indices_rae.to(refined["offset_bins"])
        coordinates = coordinates + refined["offset_bins"]
        batch_size, point_count, _ = coordinates.shape
        batch_index = torch.arange(batch_size, device=coordinates.device)
        batch_index = batch_index[:, None].expand(-1, point_count)
        final_query = torch.cat(
            (
                batch_index.reshape(-1, 1).to(coordinates),
                coordinates.reshape(-1, 3),
            ),
            dim=1,
        )
        point_cube_spectrum = query_cube_spectrum(cube_drae, final_query)
        point_cube_spectrum = point_cube_spectrum.reshape(batch_size, point_count, 64)
        refined.update(
            self.refiner.physical_head.physical_attributes(
                refined["query_features"], point_cube_spectrum
            )
        )
        xyz = continuous_rae_to_xyz(
            coordinates.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        ).reshape(batch_size, point_count, 3)
        anchor_xyz = continuous_rae_to_xyz(
            anchors.indices_rae.reshape(-1, 3).to(coordinates),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        ).reshape(batch_size, point_count, 3)
        confidence_logit = anchors.parent_logits + refined["confidence_logit"]
        return {
            **refined,
            "probability": refined["doppler_probability"],
            "coordinates_rae": coordinates,
            "xyz_m": xyz,
            "confidence_residual_logit": refined["confidence_logit"],
            "confidence_logit": confidence_logit,
            "confidence": torch.sigmoid(confidence_logit),
            "anchor_indices_rae": anchors.indices_rae,
            "anchor_normalized_rae": anchors.normalized_rae,
            "anchor_xyz_m": anchor_xyz,
            "anchor_features": anchors.parent_features,
            "anchor_parent_logits": anchors.parent_logits,
            "anchor_parent_confidence": anchors.parent_confidence,
            "anchor_cube_spectrum": anchors.local_cube_spectrum,
            "point_cube_spectrum": point_cube_spectrum,
            "radar_token_count": coordinates.new_tensor(
                0 if radar_tokens is None else radar_tokens.shape[1], dtype=torch.long
            ),
        }
