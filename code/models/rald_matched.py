"""K-Radar-native RaLD-style latent occupancy generation baseline.

The architecture follows the Apache-2.0 RaLD release at commit
ffec4b41241391734b1eda5c093de843c909eb8e, but uses only PyTorch operators and
keeps K-Radar's native RAE grid. It is a matched reimplementation, not an
official checkpoint-compatible reproduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPointEmbedding(nn.Module):
    def __init__(self, output_dim: int = 512, frequency_dim: int = 48) -> None:
        super().__init__()
        if frequency_dim % 6 != 0:
            raise ValueError("Point frequency dimension must be divisible by six")
        frequencies = torch.pow(2.0, torch.arange(frequency_dim // 6)) * math.pi
        basis = torch.zeros(3, frequency_dim // 2)
        width = frequency_dim // 6
        for axis in range(3):
            basis[axis, axis * width : (axis + 1) * width] = frequencies
        self.register_buffer("basis", basis, persistent=True)
        self.project = nn.Linear(frequency_dim + 3, output_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected point tensor (B,N,3), got {points.shape}")
        projection = torch.einsum("bnc,cf->bnf", points, self.basis)
        features = torch.cat((projection.sin(), projection.cos(), points), dim=-1)
        return self.project(features)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        head_dim: int = 64,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if query_dim <= 0 or heads <= 0 or head_dim <= 0:
            raise ValueError("Attention dimensions must be positive")
        context_dim = query_dim if context_dim is None else context_dim
        output_dim = query_dim if output_dim is None else output_dim
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.query = nn.Linear(query_dim, inner_dim, bias=False)
        self.key = nn.Linear(context_dim, inner_dim, bias=False)
        self.value = nn.Linear(context_dim, inner_dim, bias=False)
        self.output = nn.Linear(inner_dim, output_dim)

    def forward(
        self, query: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        context = query if context is None else context
        batch, query_count, _ = query.shape
        context_count = context.shape[1]

        def split_heads(values: torch.Tensor, count: int) -> torch.Tensor:
            return values.view(batch, count, self.heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.query(query), query_count)
        k = split_heads(self.key(context), context_count)
        v = split_heads(self.value(context), context_count)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, query_count, -1)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int = 4) -> None:
        super().__init__()
        hidden = dim * multiplier
        self.input = nn.Linear(dim, hidden * 2)
        self.output = nn.Linear(hidden, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        content, gate = self.input(values).chunk(2, dim=-1)
        return self.output(content * F.gelu(gate))


class PreNormAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.attention = Attention(query_dim, context_dim, heads, head_dim)

    def forward(
        self, query: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        if context is None:
            context = query
        return self.attention(self.query_norm(query), self.context_norm(context))


class PreNormFeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.norm(values))


@dataclass(frozen=True)
class GaussianPosterior:
    mean: torch.Tensor
    log_variance: torch.Tensor

    @property
    def variance(self) -> torch.Tensor:
        return self.log_variance.exp()

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + (0.5 * self.log_variance).exp() * noise

    def kl(self) -> torch.Tensor:
        return 0.5 * (
            self.mean.square() + self.variance - 1.0 - self.log_variance
        ).mean(dim=(1, 2))


class RaLDPointAutoencoder(nn.Module):
    """Order-invariant mixed-query VAE with an implicit occupancy decoder."""

    def __init__(
        self,
        point_count: int = 2_048,
        latent_count: int = 512,
        model_dim: int = 512,
        latent_dim: int = 32,
        depth: int = 24,
        heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.point_count = point_count
        self.latent_count = latent_count
        self.point_embedding = FourierPointEmbedding(model_dim)
        self.static_latents = nn.Embedding(latent_count, model_dim)
        self.dynamic_latents = nn.Embedding(latent_count, model_dim)
        self.dynamic_attention = PreNormAttention(
            model_dim, model_dim, heads=1, head_dim=model_dim
        )
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.encoder_attention = PreNormAttention(
            model_dim, model_dim, heads=1, head_dim=model_dim
        )
        self.encoder_feed_forward = PreNormFeedForward(model_dim)
        self.mean = nn.Linear(model_dim, latent_dim)
        self.log_variance = nn.Linear(model_dim, latent_dim)
        self.latent_projection = nn.Linear(latent_dim, model_dim)
        self.decoder_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    (
                        PreNormAttention(
                            model_dim, heads=heads, head_dim=head_dim
                        ),
                        PreNormFeedForward(model_dim),
                    )
                )
                for _ in range(depth)
            ]
        )
        self.decoder_attention = PreNormAttention(
            model_dim, model_dim, heads=1, head_dim=model_dim
        )
        self.occupancy = nn.Linear(model_dim, 1)

    def encode(self, points: torch.Tensor) -> GaussianPosterior:
        if points.shape[1:] != (self.point_count, 3):
            raise ValueError(
                f"Expected {self.point_count} normalized RAE points, got {points.shape}"
            )
        batch = points.shape[0]
        point_features = self.point_embedding(points)
        static = self.static_latents.weight.unsqueeze(0).expand(batch, -1, -1)
        dynamic = self.dynamic_latents.weight.unsqueeze(0).expand(batch, -1, -1)
        dynamic = dynamic + self.dynamic_attention(dynamic, point_features)
        latent = self.query_projection(static + dynamic)
        latent = latent + self.encoder_attention(latent, point_features)
        latent = latent + self.encoder_feed_forward(latent)
        mean = self.mean(latent)
        log_variance = self.log_variance(latent).clamp(-30.0, 20.0)
        return GaussianPosterior(mean, log_variance)

    def decode(self, latent: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1] != self.latent_count:
            raise ValueError(f"Unexpected latent shape {latent.shape}")
        if queries.ndim != 3 or queries.shape[-1] != 3:
            raise ValueError(f"Unexpected query shape {queries.shape}")
        features = self.latent_projection(latent)
        for attention, feed_forward in self.decoder_layers:
            features = features + attention(features)
            features = features + feed_forward(features)
        query_features = self.point_embedding(queries)
        query_features = self.decoder_attention(query_features, features)
        return self.occupancy(query_features).squeeze(-1)

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor,
        *,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, GaussianPosterior]:
        posterior = self.encode(points)
        latent = posterior.sample() if sample_posterior else posterior.mean
        return self.decode(latent, queries), posterior


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class RadarResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.main(values) + self.skip(values)


class RaLDRadarEncoder(nn.Module):
    """Native-grid 3D encoder for a normalized K-Radar RAE-Sum condition."""

    def __init__(
        self,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        blocks_per_level: int = 2,
        output_channels: int = 16,
    ) -> None:
        super().__init__()
        self.input = nn.Conv3d(1, base_channels, 3, padding=1)
        levels = []
        current = base_channels
        for level_index, multiplier in enumerate(channel_multipliers):
            output = base_channels * multiplier
            blocks = []
            for _ in range(blocks_per_level):
                blocks.append(RadarResidualBlock(current, output))
                current = output
            downsample = (
                nn.Conv3d(current, current, 3, stride=2, padding=1)
                if level_index < len(channel_multipliers) - 1
                else nn.Identity()
            )
            levels.append(nn.ModuleList((nn.Sequential(*blocks), downsample)))
        self.levels = nn.ModuleList(levels)
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(current), current),
            nn.SiLU(),
            nn.Conv3d(current, output_channels, 3, padding=1),
        )

    def forward(self, rae_sum: torch.Tensor) -> torch.Tensor:
        if rae_sum.ndim != 5 or rae_sum.shape[1] != 1:
            raise ValueError(f"Expected RAE-Sum shape (B,1,R,A,E), got {rae_sum.shape}")
        features = self.input(rae_sum)
        for blocks, downsample in self.levels:
            features = downsample(blocks(features))
        return self.output(features)


class RadarTokenEncoder(nn.Module):
    def __init__(
        self,
        encoded_shape: tuple[int, int, int] = (16, 7, 3),
        encoded_channels: int = 16,
        token_dim: int = 512,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        blocks_per_level: int = 2,
    ) -> None:
        super().__init__()
        self.encoded_shape = encoded_shape
        self.encoder = RaLDRadarEncoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            blocks_per_level=blocks_per_level,
            output_channels=encoded_channels,
        )
        self.project = nn.Linear(encoded_channels, token_dim)
        self.range_embedding = nn.Embedding(encoded_shape[0], token_dim)
        self.azimuth_embedding = nn.Embedding(encoded_shape[1], token_dim)
        self.elevation_embedding = nn.Embedding(encoded_shape[2], token_dim)

    def forward(self, rae_sum: torch.Tensor) -> torch.Tensor:
        features = self.encoder(rae_sum).permute(0, 2, 3, 4, 1)
        if tuple(features.shape[1:4]) != self.encoded_shape:
            raise ValueError(
                f"Encoded radar shape {tuple(features.shape[1:4])} does not match "
                f"configured shape {self.encoded_shape}"
            )
        tokens = self.project(features)
        range_embedding = self.range_embedding.weight[:, None, None, :]
        azimuth_embedding = self.azimuth_embedding.weight[None, :, None, :]
        elevation_embedding = self.elevation_embedding.weight[None, None, :, :]
        tokens = tokens + range_embedding + azimuth_embedding + elevation_embedding
        return tokens.flatten(1, 3)


class NoiseEmbedding(nn.Module):
    def __init__(self, frequency_dim: int, output_dim: int) -> None:
        super().__init__()
        if frequency_dim % 2 != 0:
            raise ValueError("Noise frequency dimension must be even")
        self.frequency_dim = frequency_dim
        self.layers = nn.Sequential(
            nn.Linear(frequency_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.arange(half, device=noise.device, dtype=torch.float32)
        frequencies = (1.0 / 10_000.0) ** (frequencies / max(half - 1, 1))
        phase = noise.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        return self.layers(torch.cat((phase.cos(), phase.sin()), dim=-1))


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(dim, dim * 2)

    def forward(self, values: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(noise).chunk(2, dim=-1)
        return self.norm(values) * (1.0 + scale[:, None]) + shift[:, None]


class LatentTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        condition_dim: int,
        heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.self_norm = AdaptiveLayerNorm(dim)
        self.cross_norm = AdaptiveLayerNorm(dim)
        self.feed_forward_norm = AdaptiveLayerNorm(dim)
        self.self_attention = Attention(dim, heads=heads, head_dim=head_dim)
        self.cross_attention = Attention(
            dim, condition_dim, heads=heads, head_dim=head_dim
        )
        self.feed_forward = FeedForward(dim)

    def forward(
        self, values: torch.Tensor, noise: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        values = values + self.self_attention(self.self_norm(values, noise))
        values = values + self.cross_attention(
            self.cross_norm(values, noise), condition
        )
        return values + self.feed_forward(
            self.feed_forward_norm(values, noise)
        )


class RaLDLatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        model_dim: int = 512,
        condition_dim: int = 512,
        depth: int = 24,
        heads: int = 8,
        head_dim: int = 64,
        noise_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(latent_dim, model_dim, bias=False)
        self.noise_embedding = NoiseEmbedding(noise_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                LatentTransformerBlock(
                    model_dim, condition_dim, heads, head_dim
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, latent_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        latent: torch.Tensor,
        log_sigma_over_four: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        values = self.input(latent)
        noise = self.noise_embedding(log_sigma_over_four)
        for block in self.blocks:
            values = block(values, noise, condition)
        return self.output(self.norm(values))


class RaLDEDMPreconditioner(nn.Module):
    def __init__(
        self,
        latent_count: int = 512,
        latent_dim: int = 32,
        model_dim: int = 512,
        depth: int = 24,
        heads: int = 8,
        head_dim: int = 64,
        radar_encoder: RadarTokenEncoder | None = None,
        sigma_data: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_count = latent_count
        self.latent_dim = latent_dim
        self.sigma_data = sigma_data
        self.radar_encoder = (
            RadarTokenEncoder(token_dim=model_dim)
            if radar_encoder is None
            else radar_encoder
        )
        self.denoiser = RaLDLatentDenoiser(
            latent_dim=latent_dim,
            model_dim=model_dim,
            condition_dim=model_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
        )

    def encode_condition(self, rae_sum: torch.Tensor) -> torch.Tensor:
        return self.radar_encoder(rae_sum)

    def denoise_with_condition(
        self,
        noisy_latent: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        sigma = sigma.to(noisy_latent).reshape(-1, 1, 1)
        sigma_data = torch.as_tensor(
            self.sigma_data, dtype=noisy_latent.dtype, device=noisy_latent.device
        )
        c_skip = sigma_data.square() / (sigma.square() + sigma_data.square())
        c_out = sigma * sigma_data / (sigma.square() + sigma_data.square()).sqrt()
        c_in = 1.0 / (sigma_data.square() + sigma.square()).sqrt()
        prediction = self.denoiser(
            c_in * noisy_latent,
            sigma.flatten().log() / 4.0,
            condition,
        )
        return c_skip * noisy_latent + c_out * prediction

    def forward(
        self,
        noisy_latent: torch.Tensor,
        sigma: torch.Tensor,
        rae_sum: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoise_with_condition(
            noisy_latent, sigma, self.encode_condition(rae_sum)
        )


def edm_loss(
    model: RaLDEDMPreconditioner,
    latent: torch.Tensor,
    rae_sum: torch.Tensor,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    sigma = (
        torch.randn(latent.shape[0], device=latent.device) * p_std + p_mean
    ).exp()
    noise = torch.randn_like(latent) * sigma[:, None, None]
    denoised = model(latent + noise, sigma, rae_sum)
    weight = (sigma.square() + model.sigma_data**2) / (
        sigma * model.sigma_data
    ).square()
    return (weight[:, None, None] * (denoised - latent).square()).mean()


def _seeded_noise(
    shape: tuple[int, ...], seeds: list[int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if shape[0] != len(seeds):
        raise ValueError("One deterministic seed is required per batch element")
    samples = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        samples.append(
            torch.randn(shape[1:], device=device, dtype=dtype, generator=generator)
        )
    return torch.stack(samples)


@torch.inference_mode()
def edm_sample(
    model: RaLDEDMPreconditioner,
    rae_sum: torch.Tensor,
    seeds: list[int],
    steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    if steps < 2:
        raise ValueError("EDM sampling requires at least two steps")
    condition = model.encode_condition(rae_sum)
    shape = (rae_sum.shape[0], model.latent_count, model.latent_dim)
    latent = _seeded_noise(shape, seeds, rae_sum.device, rae_sum.dtype)
    indices = torch.arange(steps, device=rae_sum.device, dtype=torch.float32)
    schedule = (
        sigma_max ** (1.0 / rho)
        + indices
        / (steps - 1)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ).pow(rho)
    schedule = torch.cat((schedule, schedule.new_zeros(1)))
    current = latent * schedule[0]
    for step, (sigma, next_sigma) in enumerate(zip(schedule[:-1], schedule[1:])):
        sigma_batch = sigma.expand(shape[0])
        denoised = model.denoise_with_condition(current, sigma_batch, condition)
        derivative = (current - denoised) / sigma
        proposed = current + (next_sigma - sigma) * derivative
        if step < steps - 1:
            next_batch = next_sigma.expand(shape[0])
            next_denoised = model.denoise_with_condition(
                proposed, next_batch, condition
            )
            next_derivative = (proposed - next_denoised) / next_sigma
            proposed = current + (next_sigma - sigma) * 0.5 * (
                derivative + next_derivative
            )
        current = proposed
    return current
