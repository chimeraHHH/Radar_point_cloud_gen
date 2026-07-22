"""RaLD-guided dense point queries without an occupancy geometry parent."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.rald_anchor import normalize_rae_coordinates
from models.rald_matched import FullRAEDRadarTokenEncoder, RaLDAnchorLatentRefiner


@dataclass(frozen=True)
class RadarGuidedQueries:
    coordinates_rae: torch.Tensor
    normalized_rae: torch.Tensor
    local_spectrum: torch.Tensor
    seed_score: torch.Tensor
    seed_index: torch.Tensor
    template_index: torch.Tensor


def query_templates(
    queries_per_seed: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the frozen fractional RAE templates used by G1C."""

    if queries_per_seed != 10:
        raise ValueError("G1C freezes exactly ten queries per radar seed")
    return torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.30, 0.00, 0.00],
            [-0.30, 0.00, 0.00],
            [0.00, 0.30, 0.00],
            [0.00, -0.30, 0.00],
            [0.00, 0.00, 0.30],
            [0.00, 0.00, -0.30],
            [0.20, 0.20, 0.00],
            [-0.20, -0.20, 0.00],
            [0.20, -0.20, 0.00],
        ],
        device=device,
        dtype=dtype,
    )


def _flat_to_rae(flat_index: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    _, azimuth_count, elevation_count = spatial_shape
    radius = flat_index // (azimuth_count * elevation_count)
    remainder = flat_index % (azimuth_count * elevation_count)
    azimuth = remainder // elevation_count
    elevation = remainder % elevation_count
    return torch.stack((radius, azimuth, elevation), dim=-1)


def radar_guided_queries(
    cube_drae: torch.Tensor,
    *,
    base_seed_count: int = 1_000,
    queries_per_seed: int = 10,
    nms_kernel: tuple[int, int, int] = (5, 5, 3),
) -> RadarGuidedQueries:
    """Build deterministic Full-RAED energy seeds and fractional query templates."""

    if cube_drae.ndim != 5 or cube_drae.shape[1] != 64:
        raise ValueError(f"Expected Full-RAED Cube (B,64,R,A,E), got {cube_drae.shape}")
    if base_seed_count <= 0:
        raise ValueError("G1C base seed count must be positive")
    if any(size <= 0 or size % 2 == 0 for size in nms_kernel):
        raise ValueError("G1C NMS kernels must be positive odd values")
    spatial_shape = tuple(int(size) for size in cube_drae.shape[2:])
    spatial_count = int(torch.tensor(spatial_shape).prod().item())
    if base_seed_count > spatial_count:
        raise ValueError("G1C requests more seeds than Cube spatial cells")

    # Summing log power preserves all Doppler bins without allowing one extreme
    # bin to define every spatial query.
    energy = torch.log1p(cube_drae.float().clamp_min(0.0)).sum(dim=1)
    padding = tuple(size // 2 for size in nms_kernel)
    pooled = F.max_pool3d(
        energy.unsqueeze(1), nms_kernel, stride=1, padding=padding
    ).squeeze(1)
    maxima = energy >= pooled
    nms_score = energy.masked_fill(~maxima, float("-inf"))
    maximum_count = maxima.flatten(start_dim=1).sum(dim=1)
    if torch.any(maximum_count < base_seed_count):
        raise RuntimeError(
            "G1C NMS produced fewer maxima than the frozen base seed count"
        )
    seed_score, flat_index = torch.topk(
        nms_score.flatten(start_dim=1), base_seed_count, dim=1, sorted=True
    )
    base_coordinates = _flat_to_rae(flat_index, spatial_shape).to(cube_drae)
    templates = query_templates(
        queries_per_seed, device=cube_drae.device, dtype=cube_drae.dtype
    )
    coordinates = base_coordinates[:, :, None, :] + templates[None, None]
    coordinates = coordinates.reshape(cube_drae.shape[0], -1, 3)
    maximum = coordinates.new_tensor([size - 1 for size in spatial_shape])
    coordinates = coordinates.clamp_min(0.0).minimum(maximum)
    point_count = base_seed_count * queries_per_seed
    batch = torch.arange(cube_drae.shape[0], device=cube_drae.device)
    batch = batch[:, None].expand(-1, point_count)
    query = torch.cat(
        (batch.reshape(-1, 1).to(coordinates), coordinates.reshape(-1, 3)), dim=1
    )
    spectrum = query_cube_spectrum(cube_drae, query).reshape(
        cube_drae.shape[0], point_count, 64
    )
    seed_identity = torch.arange(base_seed_count, device=cube_drae.device)
    template_identity = torch.arange(queries_per_seed, device=cube_drae.device)
    return RadarGuidedQueries(
        coordinates_rae=coordinates,
        normalized_rae=normalize_rae_coordinates(coordinates, spatial_shape),
        local_spectrum=spectrum,
        seed_score=seed_score.repeat_interleave(queries_per_seed, dim=1),
        seed_index=seed_identity.repeat_interleave(queries_per_seed),
        template_index=template_identity.repeat(base_seed_count),
    )


class RaLDGuidedQueryGenerator(nn.Module):
    """Generate fixed-count dense points from RaLD radar-guided queries."""

    def __init__(
        self,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        *,
        log_center: float,
        log_scale: float,
        base_seed_count: int = 1_000,
        queries_per_seed: int = 10,
        latent_count: int = 512,
        model_dim: int = 512,
        depth: int = 24,
        heads: int = 8,
        head_dim: int = 64,
        radar_base_channels: int = 64,
        radar_spectral_channels: int = 16,
        radar_encoded_shape: tuple[int, int, int] = (16, 7, 3),
        radar_encoded_channels: int = 16,
        radar_channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        radar_blocks_per_level: int = 2,
        offset_bounds_bins: tuple[float, float, float] = (8.0, 4.0, 2.0),
        nms_kernel: tuple[int, int, int] = (5, 5, 3),
    ) -> None:
        super().__init__()
        if range_m.ndim != 1 or azimuth_rad.ndim != 1 or elevation_rad.ndim != 1:
            raise ValueError("G1C axes must be one-dimensional")
        if base_seed_count <= 0 or queries_per_seed <= 0:
            raise ValueError("G1C query counts must be positive")
        if log_scale <= 0.0:
            raise ValueError("G1C Full-RAED normalization scale must be positive")
        self.base_seed_count = base_seed_count
        self.queries_per_seed = queries_per_seed
        self.point_count = base_seed_count * queries_per_seed
        self.nms_kernel = nms_kernel
        self.spectrum_projection = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, model_dim),
        )
        self.template_embedding = nn.Embedding(queries_per_seed, model_dim)
        self.radar_encoder = FullRAEDRadarTokenEncoder(
            log_center=log_center,
            log_scale=log_scale,
            spectral_channels=radar_spectral_channels,
            encoded_shape=radar_encoded_shape,
            encoded_channels=radar_encoded_channels,
            token_dim=model_dim,
            base_channels=radar_base_channels,
            channel_multipliers=radar_channel_multipliers,
            blocks_per_level=radar_blocks_per_level,
        )
        self.refiner = RaLDAnchorLatentRefiner(
            anchor_feature_dim=model_dim,
            latent_count=latent_count,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
            spectrum_bins=64,
            radar_token_dim=model_dim,
            doppler_head_mode="distribution",
        )
        self.register_buffer("range_m", range_m.float(), persistent=True)
        self.register_buffer("azimuth_rad", azimuth_rad.float(), persistent=True)
        self.register_buffer("elevation_rad", elevation_rad.float(), persistent=True)
        self.register_buffer(
            "offset_bounds_bins",
            torch.tensor(offset_bounds_bins, dtype=torch.float32),
            persistent=True,
        )

    def geometry_parameters(self):
        return chain(
            self.spectrum_projection.parameters(),
            self.template_embedding.parameters(),
            self.radar_encoder.parameters(),
            self.refiner.parameters(),
        )

    def forward(self, cube_drae: torch.Tensor) -> dict[str, torch.Tensor]:
        queries = radar_guided_queries(
            cube_drae,
            base_seed_count=self.base_seed_count,
            queries_per_seed=self.queries_per_seed,
            nms_kernel=self.nms_kernel,
        )
        probability = queries.local_spectrum.clamp_min(0.0)
        probability = probability / probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        template_index = queries.template_index[None].expand(cube_drae.shape[0], -1)
        anchor_features = self.spectrum_projection(probability)
        anchor_features = anchor_features + self.template_embedding(template_index)
        radar_tokens = self.radar_encoder(cube_drae)
        refined = self.refiner(
            queries.normalized_rae,
            anchor_features,
            queries.local_spectrum,
            radar_tokens=radar_tokens,
        )
        raw_offset = refined["offset_bins"]
        offset = raw_offset * (2.0 * self.offset_bounds_bins.to(raw_offset))
        coordinates = queries.coordinates_rae + offset
        maximum = coordinates.new_tensor(
            [
                self.range_m.numel() - 1,
                self.azimuth_rad.numel() - 1,
                self.elevation_rad.numel() - 1,
            ]
        )
        coordinates = coordinates.clamp_min(0.0).minimum(maximum)
        batch = torch.arange(cube_drae.shape[0], device=cube_drae.device)
        batch = batch[:, None].expand(-1, self.point_count)
        final_query = torch.cat(
            (batch.reshape(-1, 1).to(coordinates), coordinates.reshape(-1, 3)),
            dim=1,
        )
        final_spectrum = query_cube_spectrum(cube_drae, final_query).reshape(
            cube_drae.shape[0], self.point_count, 64
        )
        physical = self.refiner.physical_head.physical_attributes(
            refined["query_features"], final_spectrum
        )
        xyz = continuous_rae_to_xyz(
            coordinates.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        ).reshape(cube_drae.shape[0], self.point_count, 3)
        anchor_xyz = continuous_rae_to_xyz(
            queries.coordinates_rae.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        ).reshape(cube_drae.shape[0], self.point_count, 3)
        normalized_score = (
            queries.seed_score - queries.seed_score.mean(dim=1, keepdim=True)
        ) / queries.seed_score.std(dim=1, keepdim=True).clamp_min(1e-6)
        parent_logit = normalized_score.clamp(-8.0, 8.0)
        confidence_logit = parent_logit + physical["confidence_logit"]
        return {
            **refined,
            **physical,
            "probability": physical["doppler_probability"],
            "coordinates_rae": coordinates,
            "xyz_m": xyz,
            "confidence_logit": confidence_logit,
            "confidence": torch.sigmoid(confidence_logit),
            "offset_bins": offset,
            "raw_offset_bins": raw_offset,
            "anchor_indices_rae": queries.coordinates_rae,
            "anchor_normalized_rae": queries.normalized_rae,
            "anchor_xyz_m": anchor_xyz,
            "anchor_features": anchor_features,
            "anchor_parent_logits": parent_logit,
            "anchor_parent_confidence": torch.sigmoid(parent_logit),
            "anchor_cube_spectrum": queries.local_spectrum,
            "point_cube_spectrum": final_spectrum,
            "seed_index": queries.seed_index,
            "template_index": queries.template_index,
            "radar_token_count": coordinates.new_tensor(
                radar_tokens.shape[1], dtype=torch.long
            ),
        }
