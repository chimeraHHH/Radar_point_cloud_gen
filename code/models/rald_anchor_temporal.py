"""RaLD-native temporal fusion for convention-free historical point priors."""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn

from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.cube_occupancy import CubeOccupancyNet
from models.rald_anchor import frozen_parent_anchors
from models.rald_matched import (
    PreNormAttention,
    RadarTokenEncoder,
    RaLDAnchorLatentRefiner,
)
from models.temporal_prior import (
    WarpedPrior,
    nearest_point_indices,
    prior_distribution_features,
    rasterize_distribution_prior,
)


TEMPORAL_FUSION_MODES = ("token", "latent", "query")
TEMPORAL_REFINER_PREFIXES = (
    "prior_latent_projection",
    "prior_attention",
    "latent_gate",
    "prior_query_projection",
    "query_gate",
)


class RaLDTemporalAnchorLatentRefiner(RaLDAnchorLatentRefiner):
    """Insert historical evidence at a specific RaLD representation level."""

    def __init__(
        self,
        anchor_feature_dim: int,
        temporal_fusion_mode: str,
        latent_count: int = 512,
        model_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        head_dim: int = 64,
        spectrum_bins: int = 64,
        radar_token_dim: int | None = None,
        doppler_head_mode: str = "distribution",
    ) -> None:
        if temporal_fusion_mode not in TEMPORAL_FUSION_MODES:
            raise ValueError(f"Unsupported RaLD temporal mode {temporal_fusion_mode}")
        super().__init__(
            anchor_feature_dim=anchor_feature_dim,
            latent_count=latent_count,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
            spectrum_bins=spectrum_bins,
            radar_token_dim=radar_token_dim,
            doppler_head_mode=doppler_head_mode,
        )
        self.temporal_fusion_mode = temporal_fusion_mode
        if temporal_fusion_mode == "latent":
            self.prior_latent_projection = nn.Sequential(
                nn.Linear(8, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            self.prior_attention = PreNormAttention(
                model_dim, model_dim, heads=1, head_dim=model_dim
            )
            self.latent_gate = nn.Parameter(torch.zeros(()))
        elif temporal_fusion_mode == "query":
            self.prior_query_projection = nn.Sequential(
                nn.Linear(11, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            self.query_gate = nn.Parameter(torch.zeros(()))

    def encode_temporal(
        self,
        anchor_normalized_rae: torch.Tensor,
        anchor_features: torch.Tensor,
        radar_tokens: torch.Tensor,
        prior_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        anchor_tokens = self.anchor_tokens(anchor_normalized_rae, anchor_features)
        batch = anchor_normalized_rae.shape[0]
        static = self.static_latents.weight.unsqueeze(0).expand(batch, -1, -1)
        dynamic = self.dynamic_latents.weight.unsqueeze(0).expand(batch, -1, -1)
        dynamic = dynamic + self.dynamic_attention(dynamic, anchor_tokens)
        if self.radar_attention is None:
            raise ValueError("Temporal RaLD requires current Full-RAED tokens")
        dynamic = dynamic + self.radar_attention(dynamic, radar_tokens)
        if self.temporal_fusion_mode == "latent" and prior_tokens is not None:
            projected = self.prior_latent_projection(prior_tokens)
            dynamic = dynamic + torch.tanh(self.latent_gate) * self.prior_attention(
                dynamic, projected
            )
        return self.transform_latents(static + dynamic)

    def forward(
        self,
        anchor_normalized_rae: torch.Tensor,
        anchor_features: torch.Tensor,
        local_cube_spectrum: torch.Tensor,
        radar_tokens: torch.Tensor,
        prior_tokens: torch.Tensor | None = None,
        draft_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode_temporal(
            anchor_normalized_rae,
            anchor_features,
            radar_tokens,
            prior_tokens,
        )
        query_features = self.decode_queries(anchor_normalized_rae, latent)
        if self.temporal_fusion_mode == "query" and draft_features is not None:
            query_features = query_features + torch.tanh(
                self.query_gate
            ) * self.prior_query_projection(draft_features)
        output = self.physical_head(query_features, local_cube_spectrum)
        output["latent"] = latent
        output["query_features"] = query_features
        return output

    def temporal_parameters(self):
        return (
            parameter
            for name, parameter in self.named_parameters()
            if name.startswith(TEMPORAL_REFINER_PREFIXES)
        )


class FrozenParentRaLDTemporalRefiner(nn.Module):
    """Freeze the occupancy parent and place history inside the RaLD hierarchy."""

    def __init__(
        self,
        parent: CubeOccupancyNet,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        doppler_mps: torch.Tensor,
        radar_encoder: nn.Module,
        *,
        temporal_fusion_mode: str,
        point_count: int = 10_000,
        latent_count: int = 512,
        model_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        head_dim: int = 64,
        doppler_head_mode: str = "distribution",
        prior_base_channels: int = 32,
        prior_radar_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if temporal_fusion_mode not in TEMPORAL_FUSION_MODES:
            raise ValueError(f"Unsupported RaLD temporal mode {temporal_fusion_mode}")
        if range_m.shape != (256,):
            raise ValueError("Expected 256 K-Radar range bins")
        if azimuth_rad.shape != (107,) or elevation_rad.shape != (37,):
            raise ValueError("Expected 107 x 37 K-Radar angular bins")
        if doppler_mps.shape != (64,):
            raise ValueError("Expected 64 K-Radar Doppler bins")
        if temporal_fusion_mode != "token" and prior_radar_encoder is not None:
            raise ValueError("A prior radar encoder is only valid for token fusion")
        self.parent = parent
        self.radar_encoder = radar_encoder
        self.temporal_fusion_mode = temporal_fusion_mode
        self.point_count = point_count
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()
        self.refiner = RaLDTemporalAnchorLatentRefiner(
            anchor_feature_dim=parent.head.in_channels,
            temporal_fusion_mode=temporal_fusion_mode,
            latent_count=latent_count,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
            spectrum_bins=64,
            radar_token_dim=model_dim,
            doppler_head_mode=doppler_head_mode,
        )
        if temporal_fusion_mode == "token":
            self.prior_radar_encoder = (
                prior_radar_encoder
                if prior_radar_encoder is not None
                else RadarTokenEncoder(
                    token_dim=model_dim,
                    base_channels=prior_base_channels,
                    input_channels=5,
                )
            )
            self.token_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("range_m", range_m.float(), persistent=True)
        self.register_buffer("azimuth_rad", azimuth_rad.float(), persistent=True)
        self.register_buffer("elevation_rad", elevation_rad.float(), persistent=True)
        doppler_mps = doppler_mps.float()
        doppler_step = torch.median(torch.diff(doppler_mps))
        self.register_buffer("doppler_mps", doppler_mps, persistent=True)
        self.register_buffer(
            "doppler_lower_mps", doppler_mps[0], persistent=True
        )
        self.register_buffer(
            "doppler_period_mps",
            doppler_step * doppler_mps.numel(),
            persistent=True,
        )

    def train(self, mode: bool = True) -> FrozenParentRaLDTemporalRefiner:
        super().train(mode)
        self.parent.eval()
        return self

    def refinement_parameters(self):
        groups = [self.refiner.parameters(), self.radar_encoder.parameters()]
        if self.temporal_fusion_mode == "token":
            groups.extend(
                (
                    self.prior_radar_encoder.parameters(),
                    iter((self.token_gate,)),
                )
            )
        return chain.from_iterable(groups)

    def temporal_parameters(self):
        parameters = list(self.refiner.temporal_parameters())
        if self.temporal_fusion_mode == "token":
            parameters.extend(self.prior_radar_encoder.parameters())
            parameters.append(self.token_gate)
        return iter(parameters)

    def base_refinement_parameters(self):
        temporal_ids = {id(parameter) for parameter in self.temporal_parameters()}
        return (
            parameter
            for parameter in self.refinement_parameters()
            if id(parameter) not in temporal_ids
        )

    def load_single_frame_refiner(
        self,
        refiner_state: dict,
        radar_encoder_state: dict,
    ) -> list[str]:
        missing, unexpected = self.refiner.load_state_dict(
            refiner_state, strict=False
        )
        if unexpected or any(
            not name.startswith(TEMPORAL_REFINER_PREFIXES) for name in missing
        ):
            raise ValueError(
                f"Unexpected temporal refiner initialization: missing={missing}, "
                f"unexpected={unexpected}"
            )
        self.radar_encoder.load_state_dict(radar_encoder_state, strict=True)
        return missing

    def prior_features(self, prior: WarpedPrior) -> tuple[torch.Tensor, torch.Tensor]:
        valid = prior.valid & torch.isfinite(prior.xyz_m).all(dim=1)
        if not valid.any():
            return valid, prior.xyz_m.new_zeros((0, 8))
        distribution = prior_distribution_features(
            prior,
            self.doppler_mps,
            self.doppler_lower_mps,
            self.doppler_period_mps,
        )
        xyz_scale = prior.xyz_m.new_tensor([120.0, 120.0, 20.0])
        features = torch.cat(
            (
                prior.xyz_m / xyz_scale,
                prior.confidence[:, None],
                distribution,
            ),
            dim=1,
        )
        return valid, features[valid]

    def forward(
        self, cube_drae: torch.Tensor, prior: WarpedPrior | None = None
    ) -> dict[str, torch.Tensor]:
        if prior is not None and cube_drae.shape[0] != 1:
            raise ValueError("Temporal RaLD currently supports one Cube per step")
        with torch.no_grad():
            anchors = frozen_parent_anchors(
                self.parent, cube_drae, point_count=self.point_count
            )
        radar_tokens = self.radar_encoder(cube_drae)
        prior_tokens = None
        draft_features = None
        prior_count = 0
        valid = None
        features = None
        if prior is not None:
            valid, features = self.prior_features(prior)
            prior_count = int(valid.sum().item())
        if prior_count and self.temporal_fusion_mode == "token":
            raster = rasterize_distribution_prior(
                prior,
                self.doppler_mps,
                self.doppler_lower_mps,
                self.doppler_period_mps,
            )
            prior_radar_tokens = self.prior_radar_encoder(raster.unsqueeze(0))
            radar_tokens = radar_tokens + torch.tanh(
                self.token_gate
            ) * prior_radar_tokens
        elif prior_count and self.temporal_fusion_mode == "latent":
            prior_tokens = features.unsqueeze(0)
        elif prior_count and self.temporal_fusion_mode == "query":
            anchor_xyz = continuous_rae_to_xyz(
                anchors.indices_rae.reshape(-1, 3).to(anchors.parent_features),
                self.range_m,
                self.azimuth_rad,
                self.elevation_rad,
            )
            valid_xyz = prior.xyz_m[valid]
            nearest, _ = nearest_point_indices(anchor_xyz, valid_xyz, 1)
            nearest = nearest[:, 0]
            relative = (valid_xyz[nearest] - anchor_xyz) / anchor_xyz.new_tensor(
                [10.0, 10.0, 5.0]
            )
            draft_features = torch.cat((features[nearest], relative), dim=1)
            draft_features = draft_features.unsqueeze(0)
        refined = self.refiner(
            anchors.normalized_rae,
            anchors.parent_features,
            anchors.local_cube_spectrum,
            radar_tokens,
            prior_tokens=prior_tokens,
            draft_features=draft_features,
        )
        coordinates = anchors.indices_rae.to(refined["offset_bins"])
        coordinates = coordinates + refined["offset_bins"]
        batch_size, point_count, _ = coordinates.shape
        batch_index = torch.arange(batch_size, device=coordinates.device)
        batch_index = batch_index[:, None].expand(-1, point_count)
        query = torch.cat(
            (
                batch_index.reshape(-1, 1).to(coordinates),
                coordinates.reshape(-1, 3),
            ),
            dim=1,
        )
        point_cube_spectrum = query_cube_spectrum(cube_drae, query)
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
                radar_tokens.shape[1], dtype=torch.long
            ),
            "temporal_prior_count": coordinates.new_tensor(
                prior_count, dtype=torch.long
            ),
        }
