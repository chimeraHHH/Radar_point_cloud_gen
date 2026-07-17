"""Position-conditioned Doppler heads for Full-RAED Cube geometry models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cube_occupancy import CubeOccupancyNet


STATIC_HYPOTHESES = ("negative_ego", "positive_ego", "zero_centered")


def wrapped_delta(
    first: torch.Tensor, second: torch.Tensor, period: torch.Tensor
) -> torch.Tensor:
    return torch.remainder(first - second + period / 2.0, period) - period / 2.0


def wrapped_gaussian(
    centers: torch.Tensor,
    axis: torch.Tensor,
    period: torch.Tensor,
    standard_deviation: torch.Tensor,
) -> torch.Tensor:
    delta = wrapped_delta(axis.view(1, -1), centers.view(-1, 1), period)
    return torch.softmax(-0.5 * (delta / standard_deviation).square(), dim=1)


def circular_mean(
    probability: torch.Tensor,
    axis: torch.Tensor,
    lower: torch.Tensor,
    period: torch.Tensor,
) -> torch.Tensor:
    angle = 2.0 * torch.pi * (axis - lower) / period
    sine = (probability * torch.sin(angle)).sum(dim=-1)
    cosine = (probability * torch.cos(angle)).sum(dim=-1)
    mean_angle = torch.atan2(sine, cosine)
    mean_angle = torch.remainder(mean_angle, 2.0 * torch.pi)
    return lower + period * mean_angle / (2.0 * torch.pi)


def query_cube_spectrum(
    cube_drae: torch.Tensor,
    indices: torch.Tensor,
    smoothing: float = 1e-4,
) -> torch.Tensor:
    """Query normalized log-power spectra at integer RAE locations."""

    if cube_drae.ndim != 5 or cube_drae.shape[1] != 64:
        raise ValueError(f"Expected Cube shape (B,64,R,A,E), got {cube_drae.shape}")
    batch, radius, azimuth, elevation = split_query_indices(
        indices, cube_drae.shape[0]
    )
    power = cube_drae[batch, :, radius, azimuth, elevation].clamp_min(0.0)
    evidence = torch.log1p(power) + smoothing
    return evidence / evidence.sum(dim=1, keepdim=True).clamp_min(1e-12)


def split_query_indices(
    indices: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if indices.ndim != 2 or indices.shape[1] not in (3, 4):
        raise ValueError(f"Expected (N,3) or (N,4) query indices, got {indices.shape}")
    indices = indices.long()
    if indices.shape[1] == 3:
        if batch_size != 1:
            raise ValueError("Three-column indices require batch size one")
        batch = torch.zeros(indices.shape[0], dtype=torch.long, device=indices.device)
        radius, azimuth, elevation = indices.unbind(dim=1)
    else:
        batch, radius, azimuth, elevation = indices.unbind(dim=1)
    if (batch < 0).any() or (batch >= batch_size).any():
        raise IndexError("Query batch index is out of bounds")
    return batch, radius, azimuth, elevation


class CubeDopplerNet(CubeOccupancyNet):
    """Full-RAED geometry backbone with scalar or distribution Doppler query."""

    HEAD_MODES = ("scalar", "distribution", "physics_distribution")

    def __init__(
        self,
        head_mode: str,
        doppler_mps: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        base_channels: int = 8,
        log_center: float = 11.0,
        log_scale: float = 2.0,
        static_hypothesis: str = "zero_centered",
    ) -> None:
        if head_mode not in self.HEAD_MODES:
            raise ValueError(f"Unsupported Doppler head {head_mode}")
        if static_hypothesis not in STATIC_HYPOTHESES:
            raise ValueError(f"Unsupported static hypothesis {static_hypothesis}")
        super().__init__(
            "full_raed",
            doppler_mps,
            base_channels=base_channels,
            log_center=log_center,
            log_scale=log_scale,
        )
        if azimuth_rad.shape != (107,) or elevation_rad.shape != (37,):
            raise ValueError("Expected K-Radar azimuth/elevation axes of size 107/37")
        self.head_mode = head_mode
        self.static_hypothesis = static_hypothesis
        self.register_buffer("azimuth_rad", azimuth_rad.float(), persistent=True)
        self.register_buffer("elevation_rad", elevation_rad.float(), persistent=True)
        step = torch.median(torch.diff(doppler_mps.float()))
        self.register_buffer("doppler_step_mps", step, persistent=True)
        self.register_buffer(
            "doppler_period_mps", step * doppler_mps.numel(), persistent=True
        )
        self.register_buffer(
            "doppler_lower_mps", doppler_mps.float()[0], persistent=True
        )
        self.query_projection = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.SiLU(),
        )
        if head_mode == "scalar":
            self.scalar_head = nn.Linear(base_channels, 1)
        else:
            self.distribution_head = nn.Linear(base_channels, doppler_mps.numel())
        if head_mode == "physics_distribution":
            self.static_gate = nn.Linear(base_channels, 1)

    def spatial_features(self, cube_drae: torch.Tensor) -> torch.Tensor:
        level0 = self.enc0(self.encode_cube(cube_drae))
        level1 = self.enc1(self.down1(level0))
        latent = self.bottleneck(self.down2(level1))
        up1 = F.interpolate(
            latent, size=level1.shape[2:], mode="trilinear", align_corners=False
        )
        up1 = self.dec1(torch.cat((up1, level1), dim=1))
        up0 = F.interpolate(
            up1, size=level0.shape[2:], mode="trilinear", align_corners=False
        )
        return self.dec0(torch.cat((up0, level0), dim=1))

    def forward(self, cube_drae: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.spatial_features(cube_drae)
        return self.head(features).squeeze(1), features

    def gathered_features(
        self, features: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        batch, radius, azimuth, elevation = split_query_indices(
            indices, features.shape[0]
        )
        gathered = features[batch, :, radius, azimuth, elevation]
        return self.query_projection(gathered), (batch, radius, azimuth, elevation)

    def static_center(
        self,
        batch: torch.Tensor,
        azimuth: torch.Tensor,
        elevation: torch.Tensor,
        ego_speed_mps: torch.Tensor,
    ) -> torch.Tensor:
        if ego_speed_mps.ndim == 0:
            ego_speed_mps = ego_speed_mps[None]
        if ego_speed_mps.ndim != 1:
            raise ValueError("ego_speed_mps must be a scalar or one value per batch")
        radial_projection = (
            ego_speed_mps[batch]
            * torch.cos(self.azimuth_rad[azimuth])
            * torch.cos(self.elevation_rad[elevation])
        )
        if self.static_hypothesis == "negative_ego":
            center = -radial_projection
        elif self.static_hypothesis == "positive_ego":
            center = radial_projection
        else:
            center = torch.zeros_like(radial_projection)
        return torch.remainder(
            center - self.doppler_lower_mps, self.doppler_period_mps
        ) + self.doppler_lower_mps

    def query(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        ego_speed_mps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gathered, (batch, _, azimuth, elevation) = self.gathered_features(
            features, indices
        )
        return self.query_from_projected(
            gathered, batch, azimuth, elevation, ego_speed_mps
        )

    def query_from_projected(
        self,
        gathered: torch.Tensor,
        batch: torch.Tensor,
        azimuth: torch.Tensor,
        elevation: torch.Tensor,
        ego_speed_mps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply Doppler heads to already projected per-point features."""

        if gathered.ndim != 2 or gathered.shape[1] != self.project.out_channels:
            raise ValueError(
                f"Expected projected point features (*,{self.project.out_channels}), "
                f"got {gathered.shape}"
            )
        point_count = gathered.shape[0]
        if any(index.shape != (point_count,) for index in (batch, azimuth, elevation)):
            raise ValueError("Projected point indices do not match feature count")
        if self.head_mode == "scalar":
            scalar_unwrapped = self.scalar_head(gathered).squeeze(1)
            scalar = torch.remainder(
                scalar_unwrapped - self.doppler_lower_mps,
                self.doppler_period_mps,
            ) + self.doppler_lower_mps
            probability = wrapped_gaussian(
                scalar,
                self.doppler_mps,
                self.doppler_period_mps,
                self.doppler_step_mps,
            )
            return {"scalar_mps": scalar, "probability": probability}

        logits = self.distribution_head(gathered)
        learned_probability = torch.softmax(logits, dim=1)
        if self.head_mode == "distribution":
            scalar = circular_mean(
                learned_probability,
                self.doppler_mps,
                self.doppler_lower_mps,
                self.doppler_period_mps,
            )
            return {
                "logits": logits,
                "probability": learned_probability,
                "scalar_mps": scalar,
            }

        static_probability = torch.sigmoid(self.static_gate(gathered).squeeze(1))
        static_center = self.static_center(
            batch, azimuth, elevation, ego_speed_mps
        )
        analytic_probability = wrapped_gaussian(
            static_center,
            self.doppler_mps,
            self.doppler_period_mps,
            self.doppler_step_mps,
        )
        probability = (
            static_probability[:, None] * analytic_probability
            + (1.0 - static_probability[:, None]) * learned_probability
        )
        scalar = circular_mean(
            probability,
            self.doppler_mps,
            self.doppler_lower_mps,
            self.doppler_period_mps,
        )
        return {
            "logits": logits,
            "learned_probability": learned_probability,
            "analytic_probability": analytic_probability,
            "probability": probability,
            "scalar_mps": scalar,
            "static_probability": static_probability,
            "static_center_mps": static_center,
        }
