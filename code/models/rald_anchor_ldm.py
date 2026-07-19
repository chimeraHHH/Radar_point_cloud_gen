"""RaLD-faithful latent diffusion over physical radar point states.

This module follows the structural chain in the official RaLD release at
commit ffec4b41241391734b1eda5c093de843c909eb8e: mixed static/dynamic queries,
an order-invariant Gaussian point-set posterior, a Transformer latent decoder,
and radar-conditioned EDM.  Unlike RaLD's occupancy decoder, G3L only decodes
queries supplied by a frozen geometry parent and carries the complete local
Doppler distribution and confidence in every target point-state token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.rald_matched import (
    FourierPointEmbedding,
    FullRAEDRadarTokenEncoder,
    GaussianPosterior,
    PreNormAttention,
    PreNormFeedForward,
    RaLDEDMPreconditioner,
    edm_sample,
)


class RaLDPointStateEmbedding(nn.Module):
    """Embed normalized RAE, circular Doppler probability, and confidence."""

    def __init__(self, model_dim: int = 512, spectrum_bins: int = 64) -> None:
        super().__init__()
        if spectrum_bins != 64:
            raise ValueError("G3L requires the native 64-bin Doppler distribution")
        phase = torch.arange(spectrum_bins, dtype=torch.float32)
        phase = phase * (2.0 * torch.pi / spectrum_bins)
        self.register_buffer("doppler_phase", phase, persistent=True)
        self.spectrum_bins = spectrum_bins
        self.rae_embedding = FourierPointEmbedding(model_dim)
        state_dim = 3 + spectrum_bins + 2 + 1
        self.state_projection = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, model_dim),
        )

    def forward(
        self,
        normalized_rae: torch.Tensor,
        doppler_probability: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_rae.ndim != 3 or normalized_rae.shape[-1] != 3:
            raise ValueError(
                f"Expected normalized RAE (B,N,3), got {normalized_rae.shape}"
            )
        expected_spectrum_shape = (*normalized_rae.shape[:2], self.spectrum_bins)
        if doppler_probability.shape != expected_spectrum_shape:
            raise ValueError(
                "Doppler distribution must align with points and contain 64 bins"
            )
        if confidence.shape == (*normalized_rae.shape[:2], 1):
            confidence = confidence.squeeze(-1)
        if confidence.shape != normalized_rae.shape[:2]:
            raise ValueError("Confidence must have shape (B,N) or (B,N,1)")

        probability = doppler_probability.to(normalized_rae).clamp_min(0.0)
        probability = probability / probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        phase = self.doppler_phase.to(probability)
        circular_moments = torch.stack(
            (
                (probability * phase.sin()).sum(dim=-1),
                (probability * phase.cos()).sum(dim=-1),
            ),
            dim=-1,
        )
        state = torch.cat(
            (
                normalized_rae,
                probability,
                circular_moments,
                confidence.to(normalized_rae).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.rae_embedding(normalized_rae) + self.state_projection(state)


class RaLDPointStatePosteriorEncoder(nn.Module):
    """RaLD mixed queries mapped to an order-invariant Gaussian posterior."""

    def __init__(
        self,
        latent_count: int = 512,
        latent_dim: int = 32,
        model_dim: int = 512,
        spectrum_bins: int = 64,
    ) -> None:
        super().__init__()
        self.latent_count = latent_count
        self.latent_dim = latent_dim
        self.point_state_embedding = RaLDPointStateEmbedding(
            model_dim=model_dim, spectrum_bins=spectrum_bins
        )
        self.static_queries = nn.Embedding(latent_count, model_dim)
        self.dynamic_queries = nn.Embedding(latent_count, model_dim)
        self.dynamic_attention = PreNormAttention(
            model_dim, model_dim, heads=1, head_dim=model_dim
        )
        self.mixed_query_projection = nn.Linear(model_dim, model_dim)
        self.posterior_attention = PreNormAttention(
            model_dim, model_dim, heads=1, head_dim=model_dim
        )
        self.posterior_feed_forward = PreNormFeedForward(model_dim)
        self.mean = nn.Linear(model_dim, latent_dim)
        self.log_variance = nn.Linear(model_dim, latent_dim)

    def forward(
        self,
        normalized_rae: torch.Tensor,
        doppler_probability: torch.Tensor,
        confidence: torch.Tensor,
    ) -> GaussianPosterior:
        point_state = self.point_state_embedding(
            normalized_rae, doppler_probability, confidence
        )
        batch_size = point_state.shape[0]
        static = self.static_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        dynamic = self.dynamic_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        dynamic = dynamic + self.dynamic_attention(dynamic, point_state)
        mixed = self.mixed_query_projection(static + dynamic)
        mixed = mixed + self.posterior_attention(mixed, point_state)
        mixed = mixed + self.posterior_feed_forward(mixed)
        return GaussianPosterior(
            mean=self.mean(mixed),
            log_variance=self.log_variance(mixed).clamp(-30.0, 20.0),
        )


class RaLDAnchorQueryDecoder(nn.Module):
    """Decode latent state only at frozen-parent radar anchor queries."""

    def __init__(
        self,
        anchor_feature_dim: int,
        latent_dim: int = 32,
        model_dim: int = 512,
        depth: int = 24,
        heads: int = 8,
        head_dim: int = 64,
        detach_parent: bool = True,
    ) -> None:
        super().__init__()
        if anchor_feature_dim <= 0:
            raise ValueError("Anchor feature dimension must be positive")
        self.detach_parent = detach_parent
        self.latent_projection = nn.Linear(latent_dim, model_dim)
        self.latent_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    (
                        PreNormAttention(model_dim, heads=heads, head_dim=head_dim),
                        PreNormFeedForward(model_dim),
                    )
                )
                for _ in range(depth)
            ]
        )
        self.anchor_position_embedding = FourierPointEmbedding(model_dim)
        self.anchor_feature_projection = nn.Sequential(
            nn.LayerNorm(anchor_feature_dim),
            nn.Linear(anchor_feature_dim, model_dim),
        )
        self.query_cross_attention = PreNormAttention(
            model_dim, model_dim, heads=1, head_dim=model_dim
        )

    def prepare_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError(f"Expected latent array (B,M,C), got {latent.shape}")
        features = self.latent_projection(latent)
        for attention, feed_forward in self.latent_layers:
            features = features + attention(features)
            features = features + feed_forward(features)
        return features

    def forward(
        self,
        latent: torch.Tensor,
        anchor_normalized_rae: torch.Tensor,
        anchor_features: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_normalized_rae.ndim != 3 or anchor_normalized_rae.shape[-1] != 3:
            raise ValueError(
                f"Expected anchor RAE (B,N,3), got {anchor_normalized_rae.shape}"
            )
        if anchor_features.ndim != 3:
            raise ValueError("Anchor features must have shape (B,N,C)")
        if anchor_features.shape[:2] != anchor_normalized_rae.shape[:2]:
            raise ValueError("Anchor coordinates and features must align")
        if latent.shape[0] != anchor_normalized_rae.shape[0]:
            raise ValueError("Latent and anchor batches must align")

        if self.detach_parent:
            anchor_normalized_rae = anchor_normalized_rae.detach()
            anchor_features = anchor_features.detach()
        query_initialization = self.anchor_position_embedding(
            anchor_normalized_rae
        ) + self.anchor_feature_projection(anchor_features)
        prepared_latent = self.prepare_latent(latent)
        return self.query_cross_attention(query_initialization, prepared_latent)


@dataclass(frozen=True)
class RaLDAnchorLDMOutput:
    latent: torch.Tensor
    query_features: torch.Tensor
    posterior: GaussianPosterior | None


class RaLDAnchorLDM(nn.Module):
    """Minimal G3L chain: physical posterior, anchor decoder, and Full-RAED EDM."""

    OFFICIAL_RALD_COMMIT = "ffec4b41241391734b1eda5c093de843c909eb8e"

    def __init__(
        self,
        anchor_feature_dim: int,
        *,
        log_center: float | None = None,
        log_scale: float | None = None,
        latent_count: int = 512,
        latent_dim: int = 32,
        model_dim: int = 512,
        decoder_depth: int = 24,
        denoiser_depth: int = 24,
        heads: int = 8,
        head_dim: int = 64,
        edm_steps: int = 18,
        sigma_data: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        radar_encoder: FullRAEDRadarTokenEncoder | None = None,
        detach_parent: bool = True,
    ) -> None:
        super().__init__()
        if edm_steps < 2:
            raise ValueError("EDM sampling requires at least two steps")
        if radar_encoder is None:
            if log_center is None or log_scale is None:
                raise ValueError(
                    "Full-RAED train-only log normalization or a configured "
                    "encoder is required"
                )
            radar_encoder = FullRAEDRadarTokenEncoder(
                log_center=log_center,
                log_scale=log_scale,
                token_dim=model_dim,
            )
        if not isinstance(radar_encoder, FullRAEDRadarTokenEncoder):
            raise TypeError("G3L conditioning must use FullRAEDRadarTokenEncoder")
        condition_dim = radar_encoder.token_encoder.project.out_features
        if condition_dim != model_dim:
            raise ValueError(
                f"Full-RAED token dimension {condition_dim} != model dimension "
                f"{model_dim}"
            )

        self.latent_count = latent_count
        self.latent_dim = latent_dim
        self.edm_steps = edm_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.posterior_encoder = RaLDPointStatePosteriorEncoder(
            latent_count=latent_count,
            latent_dim=latent_dim,
            model_dim=model_dim,
            spectrum_bins=64,
        )
        self.decoder = RaLDAnchorQueryDecoder(
            anchor_feature_dim=anchor_feature_dim,
            latent_dim=latent_dim,
            model_dim=model_dim,
            depth=decoder_depth,
            heads=heads,
            head_dim=head_dim,
            detach_parent=detach_parent,
        )
        self.edm = RaLDEDMPreconditioner(
            latent_count=latent_count,
            latent_dim=latent_dim,
            model_dim=model_dim,
            depth=denoiser_depth,
            heads=heads,
            head_dim=head_dim,
            radar_encoder=radar_encoder,
            sigma_data=sigma_data,
        )

    def posterior_mean_path(
        self,
        target_normalized_rae: torch.Tensor,
        target_doppler_probability: torch.Tensor,
        target_confidence: torch.Tensor,
        anchor_normalized_rae: torch.Tensor,
        anchor_features: torch.Tensor,
    ) -> RaLDAnchorLDMOutput:
        """Deterministic autoencoding path that preserves the parent query support."""

        posterior = self.posterior_encoder(
            target_normalized_rae,
            target_doppler_probability,
            target_confidence,
        )
        latent = posterior.mean
        query_features = self.decoder(
            latent, anchor_normalized_rae, anchor_features
        )
        return RaLDAnchorLDMOutput(latent, query_features, posterior)

    @torch.inference_mode()
    def sampled_edm_path(
        self,
        cube_drae: torch.Tensor,
        anchor_normalized_rae: torch.Tensor,
        anchor_features: torch.Tensor,
        seeds: list[int],
        *,
        steps: int | None = None,
    ) -> RaLDAnchorLDMOutput:
        """Sample a Full-RAED-conditioned latent and decode only parent anchors."""

        latent = edm_sample(
            self.edm,
            cube_drae,
            seeds,
            steps=self.edm_steps if steps is None else steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        )
        query_features = self.decoder(
            latent, anchor_normalized_rae, anchor_features
        )
        return RaLDAnchorLDMOutput(latent, query_features, None)

    def forward(
        self,
        target_normalized_rae: torch.Tensor,
        target_doppler_probability: torch.Tensor,
        target_confidence: torch.Tensor,
        anchor_normalized_rae: torch.Tensor,
        anchor_features: torch.Tensor,
    ) -> RaLDAnchorLDMOutput:
        return self.posterior_mean_path(
            target_normalized_rae,
            target_doppler_probability,
            target_confidence,
            anchor_normalized_rae,
            anchor_features,
        )
