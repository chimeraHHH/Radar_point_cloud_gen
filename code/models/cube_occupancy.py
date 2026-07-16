"""Frustum-occupancy baselines for K-Radar Cube-to-dense geometry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(self.main(values) + self.skip(values))


class CubeOccupancyNet(nn.Module):
    """Predict RAE occupancy with matched spatial backbones across Cube encodings."""

    MODES = ("rae_max", "rae_moments", "full_raed")

    def __init__(
        self,
        mode: str,
        doppler_mps: torch.Tensor,
        base_channels: int = 8,
        log_center: float = 11.0,
        log_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported Cube encoding {mode}; choose from {self.MODES}")
        if doppler_mps.shape != (64,):
            raise ValueError(f"Expected 64 Doppler bins, received {doppler_mps.shape}")
        self.mode = mode
        self.log_center = log_center
        self.log_scale = log_scale
        self.register_buffer("doppler_mps", doppler_mps.float(), persistent=True)
        input_channels = {"rae_max": 1, "rae_moments": 3, "full_raed": 64}[mode]
        self.project = nn.Conv3d(input_channels, base_channels, 1)
        self.enc0 = ResidualBlock3d(base_channels, base_channels)
        self.down1 = nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.enc1 = ResidualBlock3d(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv3d(
            base_channels * 2, base_channels * 4, 3, stride=2, padding=1
        )
        self.bottleneck = ResidualBlock3d(base_channels * 4, base_channels * 4)
        self.dec1 = ResidualBlock3d(base_channels * 6, base_channels * 2)
        self.dec0 = ResidualBlock3d(base_channels * 3, base_channels)
        self.head = nn.Conv3d(base_channels, 1, 1)

    def normalized_log_power(self, cube_drae: torch.Tensor) -> torch.Tensor:
        values = (torch.log10(cube_drae.clamp_min(0.0) + 1.0) - self.log_center)
        return (values / self.log_scale).clamp(-4.0, 4.0)

    def encode_cube(self, cube_drae: torch.Tensor) -> torch.Tensor:
        if cube_drae.ndim != 5 or cube_drae.shape[1] != 64:
            raise ValueError(f"Expected Cube shape (B,64,R,A,E), got {cube_drae.shape}")
        normalized = self.normalized_log_power(cube_drae)
        if self.mode == "full_raed":
            return self.project(normalized)
        peak = normalized.amax(dim=1, keepdim=True)
        if self.mode == "rae_max":
            return self.project(peak)
        energy = cube_drae.clamp_min(0.0)
        probability = energy / energy.sum(dim=1, keepdim=True).clamp_min(1.0)
        velocity = self.doppler_mps.view(1, -1, 1, 1, 1)
        mean = (probability * velocity).sum(dim=1, keepdim=True)
        variance = (probability * (velocity - mean).square()).sum(
            dim=1, keepdim=True
        )
        velocity_scale = self.doppler_mps.abs().max().clamp_min(1e-6)
        moments = torch.cat(
            (peak, mean / velocity_scale, variance.sqrt() / velocity_scale), dim=1
        )
        return self.project(moments)

    def forward(self, cube_drae: torch.Tensor) -> torch.Tensor:
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
        return self.head(self.dec0(torch.cat((up0, level0), dim=1))).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
