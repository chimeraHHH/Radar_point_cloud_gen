"""Current-Cube temporal fusion variants for G4."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.cube_cycle import CubeCycleNet
from models.cube_occupancy import ResidualBlock3d
from models.temporal_prior import (
    WarpedPrior,
    nearest_point_indices,
    prior_point_features,
)


FUSION_MODES = ("concat", "cross_attention", "draft_refinement")


class CubeTemporalNet(CubeCycleNet):
    """Fuse one physical historical prior while retaining current Cube evidence."""

    def __init__(
        self,
        fusion_mode: str,
        head_mode: str,
        doppler_mps: torch.Tensor,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        base_channels: int = 8,
        log_center: float = 11.0,
        log_scale: float = 2.0,
        static_hypothesis: str = "zero_centered",
        maximum_offset_bins: float = 0.5,
        attention_neighbors: int = 8,
    ) -> None:
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"Unsupported temporal fusion mode {fusion_mode}")
        if attention_neighbors < 1:
            raise ValueError("Temporal attention requires at least one neighbor")
        super().__init__(
            head_mode,
            doppler_mps,
            range_m,
            azimuth_rad,
            elevation_rad,
            base_channels=base_channels,
            log_center=log_center,
            log_scale=log_scale,
            static_hypothesis=static_hypothesis,
            maximum_offset_bins=maximum_offset_bins,
        )
        self.fusion_mode = fusion_mode
        self.attention_neighbors = attention_neighbors
        self.prior_feature_count = 8
        if fusion_mode == "concat":
            self.prior_grid_projection = nn.Conv3d(5, base_channels, 1)
            self.concat_fusion = ResidualBlock3d(base_channels * 2, base_channels)
        elif fusion_mode == "cross_attention":
            self.prior_token_projection = nn.Sequential(
                nn.Linear(self.prior_feature_count, base_channels),
                nn.SiLU(),
            )
            self.relative_position_projection = nn.Linear(3, base_channels)
            attention_heads = 2 if base_channels % 2 == 0 else 1
            self.prior_attention = nn.MultiheadAttention(
                base_channels, attention_heads, batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(base_channels)
        else:
            self.draft_projection = nn.Sequential(
                nn.Linear(self.prior_feature_count + 3, base_channels),
                nn.SiLU(),
                nn.Linear(base_channels, base_channels),
            )
            self.temporal_norm = nn.LayerNorm(base_channels)
            self.draft_offset_gate = nn.Linear(base_channels, 3)

    def forward_temporal(
        self,
        cube_drae: torch.Tensor,
        prior_raster_crae: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.spatial_features(cube_drae)
        if self.fusion_mode == "concat" and prior_raster_crae is not None:
            if prior_raster_crae.ndim == 4:
                prior_raster_crae = prior_raster_crae.unsqueeze(0)
            expected = (features.shape[0], 5, *features.shape[2:])
            if tuple(prior_raster_crae.shape) != expected:
                raise ValueError(
                    f"Expected prior raster {expected}, got {prior_raster_crae.shape}"
                )
            prior_features = self.prior_grid_projection(
                prior_raster_crae.to(features)
            )
            features = self.concat_fusion(
                torch.cat((features, prior_features), dim=1)
            )
        return self.head(features).squeeze(1), features

    def encoded_prior_features(self, prior: WarpedPrior) -> torch.Tensor:
        moments = prior_point_features(
            prior,
            self.doppler_mps,
            self.doppler_lower_mps,
            self.doppler_period_mps,
        )
        xyz_scale = prior.xyz_m.new_tensor([120.0, 120.0, 20.0])
        return torch.cat(
            (
                prior.xyz_m / xyz_scale,
                prior.confidence[:, None],
                moments,
            ),
            dim=1,
        )

    def query_temporal(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        query_xyz_m: torch.Tensor,
        ego_speed_mps: torch.Tensor,
        prior: WarpedPrior | None,
    ) -> dict[str, torch.Tensor]:
        gathered, (batch, _, azimuth, elevation) = self.gathered_features(
            features, indices
        )
        base_gathered = gathered
        offset_override = None
        valid_prior = None if prior is None else prior.valid & torch.isfinite(
            prior.xyz_m
        ).all(dim=1)
        if (
            self.fusion_mode != "concat"
            and valid_prior is not None
            and valid_prior.any()
        ):
            prior_xyz = prior.xyz_m[valid_prior]
            prior_coordinates = prior.coordinates_rae[valid_prior]
            prior_features = self.encoded_prior_features(prior)[valid_prior]
            neighbor_count = (
                self.attention_neighbors
                if self.fusion_mode == "cross_attention"
                else 1
            )
            neighbor_index, _ = nearest_point_indices(
                query_xyz_m, prior_xyz, neighbor_count
            )
            neighbor_xyz = prior_xyz[neighbor_index]
            relative_xyz = (
                neighbor_xyz - query_xyz_m[:, None]
            ) / query_xyz_m.new_tensor([10.0, 10.0, 5.0])
            if self.fusion_mode == "cross_attention":
                token = self.prior_token_projection(
                    prior_features[neighbor_index]
                ) + self.relative_position_projection(relative_xyz)
                attended, _ = self.prior_attention(
                    gathered[:, None], token, token, need_weights=False
                )
                gathered = self.temporal_norm(gathered + attended[:, 0])
            else:
                draft_input = torch.cat(
                    (prior_features[neighbor_index[:, 0]], relative_xyz[:, 0]),
                    dim=1,
                )
                gathered = self.temporal_norm(
                    gathered + self.draft_projection(draft_input)
                )
                learned_offset = (
                    torch.tanh(self.offset_head(gathered))
                    * self.maximum_offset_bins
                )
                prior_offset = (
                    prior_coordinates[neighbor_index[:, 0]]
                    - indices[:, -3:].to(prior_coordinates)
                ).clamp(-self.maximum_offset_bins, self.maximum_offset_bins)
                gate = torch.sigmoid(self.draft_offset_gate(gathered))
                offset_override = gate * prior_offset + (1.0 - gate) * learned_offset

        if offset_override is None:
            offset = (
                torch.tanh(self.offset_head(gathered))
                * self.maximum_offset_bins
            )
        else:
            offset = offset_override
        coordinates = indices[:, -3:].to(offset) + offset
        final_gathered = self.gathered_features_continuous(
            features, coordinates, batch
        )
        if self.fusion_mode != "concat":
            final_gathered = final_gathered + (gathered - base_gathered)
        doppler = self.query_from_projected(
            final_gathered, batch, azimuth, elevation, ego_speed_mps
        )
        return self.cycle_output_from_projected(
            doppler, final_gathered, indices, offset_override_bins=offset
        )
