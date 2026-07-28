"""Cube-conditioned fixed-cardinality polar rectified flow.

Inference is a direct four-step Euler transport of a fixed Sobol source.  The
model exposes no proposal selector and performs no top-k, NMS, copying, or
post-hoc filling.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cube_dense.polar_flow_target import (
    P_R_F_POINT_COUNT,
    P_R_F_SOURCE_SEED,
    PolarLogitTransform,
    fixed_scrambled_sobol_unit,
    logit_unit,
)
from models.cube_doppler import circular_mean
from models.rald_matched import FullRAEDRadarTokenEncoder


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(dim, dim * multiplier * 2)
        self.output = nn.Linear(dim * multiplier, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        content, gate = self.input(values).chunk(2, dim=-1)
        return self.output(content * F.silu(gate))


class ConditionLatentBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)

    def forward(
        self,
        latents: torch.Tensor,
        radar_tokens: torch.Tensor,
    ) -> torch.Tensor:
        query = self.cross_query_norm(latents)
        context = self.cross_context_norm(radar_tokens)
        attended = self.cross_attention(
            query,
            context,
            context,
            need_weights=False,
        )[0]
        latents = latents + attended
        normalized = self.self_norm(latents)
        latents = latents + self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        return latents + self.feed_forward(self.feed_forward_norm(latents))


class ConditionOnlyFullRAEDEncoder(nn.Module):
    """Encode condition latents from Full-RAED Cube values and nothing else."""

    def __init__(
        self,
        *,
        log_center: float,
        log_scale: float,
        model_dim: int = 256,
        latent_count: int = 256,
        depth: int = 4,
        heads: int = 8,
        radar_spectral_channels: int = 16,
        radar_encoded_shape: tuple[int, int, int] = (16, 7, 3),
        radar_encoded_channels: int = 16,
        radar_base_channels: int = 64,
        radar_channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        radar_blocks_per_level: int = 2,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("Condition model dimension must be divisible by heads")
        if latent_count <= 0 or depth <= 0:
            raise ValueError("Condition latent count and depth must be positive")
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
        self.latents = nn.Parameter(
            torch.randn(latent_count, model_dim) / math.sqrt(model_dim)
        )
        self.blocks = nn.ModuleList(
            ConditionLatentBlock(model_dim, heads) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, cube_drae: torch.Tensor) -> torch.Tensor:
        radar_tokens = self.radar_encoder(cube_drae)
        latents = self.latents.unsqueeze(0).expand(cube_drae.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, radar_tokens)
        return self.output_norm(latents)


def _fourier_features(values: torch.Tensor, frequency_count: int) -> torch.Tensor:
    frequencies = torch.pow(
        values.new_tensor(2.0),
        torch.arange(frequency_count, device=values.device, dtype=values.dtype),
    )
    phase = values.unsqueeze(-1) * frequencies * math.pi
    return torch.cat(
        (
            values,
            phase.sin().flatten(-2),
            phase.cos().flatten(-2),
        ),
        dim=-1,
    )


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, frequency_count: int = 16) -> None:
        super().__init__()
        self.frequency_count = frequency_count
        self.network = nn.Sequential(
            nn.Linear(1 + 2 * frequency_count, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("Flow time must have shape (B,)")
        return self.network(
            _fourier_features(time[:, None], self.frequency_count)
        )


class TimeAdaLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(dim, dim * 2)

    def forward(
        self,
        values: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.modulation(time_embedding).chunk(2, dim=-1)
        return self.norm(values) * (1.0 + scale[:, None]) + shift[:, None]


class ParticleConditionBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query_norm = TimeAdaLayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True,
        )
        self.feed_forward_norm = TimeAdaLayerNorm(dim)
        self.feed_forward = FeedForward(dim)

    def forward(
        self,
        particles: torch.Tensor,
        condition_latents: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_norm(particles, time_embedding)
        context = self.context_norm(condition_latents)
        particles = particles + self.cross_attention(
            query,
            context,
            context,
            need_weights=False,
        )[0]
        return particles + self.feed_forward(
            self.feed_forward_norm(particles, time_embedding)
        )


class ChunkedParticleDecoder(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        depth: int,
        heads: int,
        frequency_count: int,
        spectrum_bins: int,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("Particle model dimension must be divisible by heads")
        self.frequency_count = frequency_count
        feature_dim = 3 + 6 * frequency_count
        self.state_embedding = nn.Linear(feature_dim, model_dim)
        self.slot_embedding = nn.Linear(feature_dim, model_dim)
        self.time_embedding = TimeEmbedding(model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList(
            ParticleConditionBlock(model_dim, heads) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.velocity_head = nn.Linear(model_dim, 3)
        self.doppler_head = nn.Linear(model_dim, spectrum_bins)
        self.confidence_head = nn.Linear(model_dim, 1)

    def features(
        self,
        state: torch.Tensor,
        source_state: torch.Tensor,
        time: torch.Tensor,
        condition_latents: torch.Tensor,
    ) -> torch.Tensor:
        if state.shape != source_state.shape or state.ndim != 3:
            raise ValueError("Particle and source states must align as (B,N,3)")
        if state.shape[0] != condition_latents.shape[0]:
            raise ValueError("Particle and condition batches must match")
        time_embedding = self.time_embedding(time)
        particle = self.state_embedding(
            _fourier_features(state, self.frequency_count)
        )
        particle = particle + self.slot_embedding(
            _fourier_features(source_state, self.frequency_count)
        )
        particle = self.input_norm(particle + time_embedding[:, None])
        for block in self.blocks:
            particle = block(particle, condition_latents, time_embedding)
        return self.output_norm(particle)


class PolarRectifiedFlow(nn.Module):
    """Exactly-N polar transport conditioned only through Full-RAED latents."""

    INFERENCE_NFE = 4
    INFERENCE_METHOD = "euler"

    def __init__(
        self,
        *,
        range_knots_m: np.ndarray | torch.Tensor,
        range_cdf: np.ndarray | torch.Tensor,
        azimuth_bounds_rad: tuple[float, float],
        elevation_bounds_rad: tuple[float, float],
        doppler_axis_mps: np.ndarray | torch.Tensor,
        log_center: float,
        log_scale: float,
        point_count: int = P_R_F_POINT_COUNT,
        source_seed: int = P_R_F_SOURCE_SEED,
        model_dim: int = 256,
        latent_count: int = 256,
        condition_depth: int = 4,
        particle_depth: int = 4,
        heads: int = 8,
        frequency_count: int = 8,
        radar_spectral_channels: int = 16,
        radar_encoded_shape: tuple[int, int, int] = (16, 7, 3),
        radar_encoded_channels: int = 16,
        radar_base_channels: int = 64,
        radar_channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        radar_blocks_per_level: int = 2,
    ) -> None:
        super().__init__()
        if point_count <= 0:
            raise ValueError("P-RF point count must be positive")
        doppler_axis = torch.as_tensor(doppler_axis_mps, dtype=torch.float32)
        if doppler_axis.shape != (64,) or not torch.isfinite(doppler_axis).all():
            raise ValueError("P-RF requires a finite 64-bin Doppler axis")
        if not torch.all(doppler_axis[1:] > doppler_axis[:-1]):
            raise ValueError("Doppler axis must be strictly increasing")
        self.point_count = int(point_count)
        self.source_seed = int(source_seed)
        self.transform = PolarLogitTransform(
            range_knots_m,
            range_cdf,
            azimuth_bounds_rad=azimuth_bounds_rad,
            elevation_bounds_rad=elevation_bounds_rad,
        )
        source_unit = fixed_scrambled_sobol_unit(
            point_count,
            seed=source_seed,
            epsilon=self.transform.epsilon,
        )
        self.register_buffer(
            "source_unit",
            source_unit,
            persistent=True,
        )
        self.register_buffer(
            "source_state",
            logit_unit(source_unit, self.transform.epsilon),
            persistent=True,
        )
        self.register_buffer("doppler_axis_mps", doppler_axis, persistent=True)
        bin_step = torch.diff(doppler_axis).median()
        self.register_buffer("doppler_lower_mps", doppler_axis[0], persistent=True)
        self.register_buffer(
            "doppler_period_mps",
            bin_step * doppler_axis.numel(),
            persistent=True,
        )
        self.condition_encoder = ConditionOnlyFullRAEDEncoder(
            log_center=log_center,
            log_scale=log_scale,
            model_dim=model_dim,
            latent_count=latent_count,
            depth=condition_depth,
            heads=heads,
            radar_spectral_channels=radar_spectral_channels,
            radar_encoded_shape=radar_encoded_shape,
            radar_encoded_channels=radar_encoded_channels,
            radar_base_channels=radar_base_channels,
            radar_channel_multipliers=radar_channel_multipliers,
            radar_blocks_per_level=radar_blocks_per_level,
        )
        self.particle_decoder = ChunkedParticleDecoder(
            model_dim=model_dim,
            depth=particle_depth,
            heads=heads,
            frequency_count=frequency_count,
            spectrum_bins=doppler_axis.numel(),
        )

    def architecture_metadata(self) -> dict:
        return {
            "route": "p_rf_independent_polar_rectified_flow",
            "point_count": self.point_count,
            "source": "fixed_scrambled_sobol_open_unit_polar",
            "source_seed": self.source_seed,
            "geometry_condition_sources": ["full_raed_cube_condition_latents"],
            "forbidden_geometry_sources": [
                "g1d_checkpoint",
                "g1d_proposal",
                "g1d_selector",
                "g1d_anchor",
                "local_cube_spectrum_query",
            ],
            "particle_self_attention": False,
            "inference_method": self.INFERENCE_METHOD,
            "inference_nfe": self.INFERENCE_NFE,
            "inference_postprocessing": [],
            "endpoint_outputs": [
                "64_bin_doppler_logits",
                "confidence_logit",
            ],
        }

    def fixed_source_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("Source batch size must be positive")
        source = self.source_state
        if device is not None or dtype is not None:
            source = source.to(
                device=source.device if device is None else device,
                dtype=source.dtype if dtype is None else dtype,
            )
        return source.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def _batch_time(
        time: float | torch.Tensor,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(time, (float, int)):
            return reference.new_full((batch_size,), float(time))
        time = time.to(reference)
        if time.ndim == 0:
            return time.expand(batch_size)
        if time.shape != (batch_size,):
            raise ValueError("Flow time must be scalar or shape (B,)")
        return time

    def encode_condition(self, cube_drae: torch.Tensor) -> torch.Tensor:
        return self.condition_encoder(cube_drae)

    def _chunked_decode(
        self,
        state: torch.Tensor,
        time: float | torch.Tensor,
        condition_latents: torch.Tensor,
        *,
        chunk_size: int,
        attributes: bool,
    ) -> dict[str, torch.Tensor]:
        if state.shape[1:] != (self.point_count, 3):
            raise ValueError(
                f"Expected state (B,{self.point_count},3), got {state.shape}"
            )
        if chunk_size <= 0:
            raise ValueError("Particle chunk size must be positive")
        batch_size = state.shape[0]
        time_tensor = self._batch_time(
            time,
            batch_size=batch_size,
            reference=state,
        )
        source = self.fixed_source_state(
            batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        velocity_chunks = []
        doppler_chunks = []
        confidence_chunks = []
        for start in range(0, self.point_count, chunk_size):
            stop = min(start + chunk_size, self.point_count)
            features = self.particle_decoder.features(
                state[:, start:stop],
                source[:, start:stop],
                time_tensor,
                condition_latents,
            )
            if attributes:
                doppler_chunks.append(self.particle_decoder.doppler_head(features))
                confidence_chunks.append(
                    self.particle_decoder.confidence_head(features).squeeze(-1)
                )
            else:
                velocity_chunks.append(
                    self.particle_decoder.velocity_head(features)
                )
        if attributes:
            return {
                "doppler_logits": torch.cat(doppler_chunks, dim=1),
                "confidence_logit": torch.cat(confidence_chunks, dim=1),
            }
        return {"velocity": torch.cat(velocity_chunks, dim=1)}

    def velocity(
        self,
        state: torch.Tensor,
        time: float | torch.Tensor,
        condition_latents: torch.Tensor,
        *,
        chunk_size: int = 2_048,
    ) -> torch.Tensor:
        return self._chunked_decode(
            state,
            time,
            condition_latents,
            chunk_size=chunk_size,
            attributes=False,
        )["velocity"]

    def endpoint_attributes(
        self,
        state: torch.Tensor,
        condition_latents: torch.Tensor,
        *,
        chunk_size: int = 2_048,
    ) -> dict[str, torch.Tensor]:
        output = self._chunked_decode(
            state,
            1.0,
            condition_latents,
            chunk_size=chunk_size,
            attributes=True,
        )
        probability = torch.softmax(output["doppler_logits"], dim=-1)
        output["doppler_probability"] = probability
        output["doppler_mps"] = circular_mean(
            probability,
            self.doppler_axis_mps.to(probability),
            self.doppler_lower_mps.to(probability),
            self.doppler_period_mps.to(probability),
        )
        output["confidence"] = torch.sigmoid(output["confidence_logit"])
        return output

    def forward(
        self,
        cube_drae: torch.Tensor,
        state: torch.Tensor,
        time: float | torch.Tensor,
        *,
        chunk_size: int = 2_048,
    ) -> torch.Tensor:
        condition_latents = self.encode_condition(cube_drae)
        return self.velocity(
            state,
            time,
            condition_latents,
            chunk_size=chunk_size,
        )

    def sample(
        self,
        cube_drae: torch.Tensor,
        *,
        chunk_size: int = 2_048,
        nfe: int = INFERENCE_NFE,
    ) -> dict[str, torch.Tensor | dict]:
        if nfe != self.INFERENCE_NFE:
            raise ValueError(
                f"Protocol inference is frozen at NFE={self.INFERENCE_NFE}; got {nfe}"
            )
        condition_latents = self.encode_condition(cube_drae)
        state = self.fixed_source_state(
            cube_drae.shape[0],
            device=cube_drae.device,
            dtype=torch.float32,
        ).clone()
        step_size = 1.0 / nfe
        for step in range(nfe):
            velocity = self.velocity(
                state,
                step * step_size,
                condition_latents,
                chunk_size=chunk_size,
            )
            state = state + step_size * velocity.float()
        attributes = self.endpoint_attributes(
            state,
            condition_latents,
            chunk_size=chunk_size,
        )
        coordinates_rae = self.transform.decode_state(state)
        xyz_m = self.transform.decode_xyz(state)
        return {
            "state": state,
            "coordinates_rae": coordinates_rae,
            "xyz_m": xyz_m,
            **attributes,
            "integration": {
                "method": self.INFERENCE_METHOD,
                "nfe": nfe,
                "postprocessing": [],
            },
            "architecture_metadata": self.architecture_metadata(),
        }
