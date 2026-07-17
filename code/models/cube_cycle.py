"""Continuous point parameterization for differentiable Cube-cycle training."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.cube_doppler import CubeDopplerNet


def interpolate_axis(axis: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    lower = torch.floor(coordinate).long().clamp(0, axis.numel() - 1)
    upper = (lower + 1).clamp(0, axis.numel() - 1)
    fraction = (coordinate - lower.to(coordinate)).clamp(0.0, 1.0)
    return axis[lower] * (1.0 - fraction) + axis[upper] * fraction


def continuous_rae_to_xyz(
    coordinates_rae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> torch.Tensor:
    if coordinates_rae.ndim != 2 or coordinates_rae.shape[1] != 3:
        raise ValueError(f"Expected continuous (N,3) RAE coordinates, got {coordinates_rae.shape}")
    radius = interpolate_axis(range_m, coordinates_rae[:, 0])
    azimuth = interpolate_axis(azimuth_rad, coordinates_rae[:, 1])
    elevation = interpolate_axis(elevation_rad, coordinates_rae[:, 2])
    cosine = torch.cos(elevation)
    return torch.stack(
        (
            radius * cosine * torch.cos(azimuth),
            radius * cosine * torch.sin(azimuth),
            radius * torch.sin(elevation),
        ),
        dim=1,
    )


class CubeCycleNet(CubeDopplerNet):
    """Cube Doppler model with bounded continuous offsets for each selected cell."""

    def __init__(
        self,
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
    ) -> None:
        super().__init__(
            head_mode,
            doppler_mps,
            azimuth_rad,
            elevation_rad,
            base_channels=base_channels,
            log_center=log_center,
            log_scale=log_scale,
            static_hypothesis=static_hypothesis,
        )
        if range_m.shape != (256,):
            raise ValueError("Expected a 256-bin K-Radar range axis")
        if not 0.0 < maximum_offset_bins <= 0.5:
            raise ValueError("Maximum offset must be in (0, 0.5] bins")
        self.maximum_offset_bins = maximum_offset_bins
        self.register_buffer("range_m", range_m.float(), persistent=True)
        self.offset_head = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, 3),
        )

    def query_cycle(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        ego_speed_mps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gathered, (batch, _, azimuth, elevation) = self.gathered_features(
            features, indices
        )
        result = self.query_from_projected(
            gathered, batch, azimuth, elevation, ego_speed_mps
        )
        return self.cycle_output_from_projected(result, gathered, indices)

    def cycle_output_from_projected(
        self,
        doppler_result: dict[str, torch.Tensor],
        gathered: torch.Tensor,
        indices: torch.Tensor,
        offset_override_bins: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        offset = torch.tanh(self.offset_head(gathered)) * self.maximum_offset_bins
        if offset_override_bins is not None:
            if offset_override_bins.shape != offset.shape:
                raise ValueError("Temporal offset override has the wrong shape")
            offset = offset_override_bins.clamp(
                -self.maximum_offset_bins, self.maximum_offset_bins
            )
        coordinates = indices[:, -3:].to(offset) + offset
        xyz = continuous_rae_to_xyz(
            coordinates,
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        )
        return {
            **doppler_result,
            "offset_rae_bins": offset,
            "coordinates_rae": coordinates,
            "xyz_m": xyz,
        }
