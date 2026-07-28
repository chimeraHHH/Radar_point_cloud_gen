"""Direct multi-horizon Doppler-world generator.

All horizons are produced in one forward pass. Persistent geometry is forced
through analytic ego and Doppler motion; learned motion is restricted to a
bounded tangential velocity and a small radial calibration. Birth geometry has
no point-copy path and is decoded from causal Full-RAED conditions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmhw.contracts import DIRECT_HORIZONS_SECONDS
from losses.twc_physics import (
    compose_persistent_motion,
    mandatory_radial_doppler_warp,
)
from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.temporal_prior import xyz_to_continuous_rae


def _gather(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    suffix = tensor.shape[2:]
    expanded = index.reshape(*index.shape, *([1] * len(suffix)))
    expanded = expanded.expand(*index.shape, *suffix)
    return torch.gather(tensor, 1, expanded)


class FullRAEDConditionEncoder(nn.Module):
    """Encode every Full-RAED axis with a memory-bounded global projection."""

    def __init__(
        self,
        doppler_count: int,
        hidden_dim: int,
        *,
        range_bins: int = 8,
        azimuth_bins: int = 8,
        elevation_bins: int = 4,
    ) -> None:
        super().__init__()
        self.range_bins = range_bins
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        feature_dim = (
            doppler_count + range_bins + azimuth_bins + elevation_bins
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _pool(profile: torch.Tensor, output_count: int) -> torch.Tensor:
        return F.adaptive_avg_pool1d(
            profile[:, None], output_count
        ).squeeze(1)

    def forward(self, cube_drae: torch.Tensor) -> torch.Tensor:
        if cube_drae.ndim != 5:
            raise ValueError("Full-RAED Cube must have shape (B,D,R,A,E)")
        energy = torch.log1p(cube_drae.clamp_min(0.0))
        doppler = energy.mean(dim=(2, 3, 4))
        range_profile = energy.mean(dim=(1, 3, 4))
        azimuth_profile = energy.mean(dim=(1, 2, 4))
        elevation_profile = energy.mean(dim=(1, 2, 3))
        feature = torch.cat(
            (
                doppler,
                self._pool(range_profile, self.range_bins),
                self._pool(azimuth_profile, self.azimuth_bins),
                self._pool(elevation_profile, self.elevation_bins),
            ),
            dim=1,
        )
        return self.projection(feature)


class DirectMultiHorizonDopplerWorld(nn.Module):
    """Generate three fixed-count future radar states without rollout."""

    def __init__(
        self,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        doppler_mps: torch.Tensor,
        *,
        point_count: int = 10_000,
        persistent_fraction: float = 0.70,
        hidden_dim: int = 96,
        maximum_tangential_speed_mps: float = 8.0,
        maximum_radial_correction_mps: float = 0.75,
    ) -> None:
        super().__init__()
        for name, axis, minimum in (
            ("range", range_m, 2),
            ("azimuth", azimuth_rad, 2),
            ("elevation", elevation_rad, 2),
            ("Doppler", doppler_mps, 64),
        ):
            if axis.ndim != 1 or axis.numel() < minimum:
                raise ValueError(f"Invalid D-MHW {name} axis")
            if not torch.isfinite(axis).all():
                raise ValueError(f"D-MHW {name} axis must be finite")
            if not torch.all(axis[1:] > axis[:-1]):
                raise ValueError(
                    f"D-MHW {name} axis must be strictly increasing"
                )
        if doppler_mps.numel() != 64:
            raise ValueError("D-MHW requires exactly 64 Doppler bins")
        if point_count < 2:
            raise ValueError("D-MHW requires at least two output points")
        if not 0.0 < persistent_fraction < 1.0:
            raise ValueError(
                "D-MHW persistent fraction must lie strictly in (0,1)"
            )
        if hidden_dim < 16:
            raise ValueError("D-MHW hidden dimension must be at least 16")
        if maximum_tangential_speed_mps <= 0.0:
            raise ValueError("D-MHW tangential bound must be positive")
        if maximum_radial_correction_mps <= 0.0:
            raise ValueError("D-MHW radial bound must be positive")

        persistent_count = round(point_count * persistent_fraction)
        persistent_count = min(max(persistent_count, 1), point_count - 1)
        self.point_count = point_count
        self.persistent_count = persistent_count
        self.birth_count = point_count - persistent_count
        self.maximum_tangential_speed_mps = (
            maximum_tangential_speed_mps
        )
        self.maximum_radial_correction_mps = (
            maximum_radial_correction_mps
        )
        self.register_buffer("range_m", range_m.float(), persistent=True)
        self.register_buffer(
            "azimuth_rad", azimuth_rad.float(), persistent=True
        )
        self.register_buffer(
            "elevation_rad", elevation_rad.float(), persistent=True
        )
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
        self.register_buffer(
            "horizons_seconds",
            torch.tensor(DIRECT_HORIZONS_SECONDS, dtype=torch.float32),
            persistent=True,
        )

        doppler_count = doppler_mps.numel()
        self.cube_encoder = FullRAEDConditionEncoder(
            doppler_count, hidden_dim
        )
        self.history_time_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.history_sequence_encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.horizon_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        source_feature_dim = 3 + doppler_count + 1 + doppler_count
        self.persistent_source_encoder = nn.Sequential(
            nn.Linear(source_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.persistent_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.persistent_motion_head = nn.Linear(hidden_dim, 4)
        self.persistent_doppler_head = nn.Linear(
            hidden_dim, doppler_count
        )
        self.persistent_confidence_head = nn.Linear(hidden_dim, 1)

        self.birth_queries = nn.Parameter(
            torch.randn(self.birth_count, hidden_dim) * 0.02
        )
        self.birth_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.birth_coordinate_head = nn.Linear(hidden_dim, 3)
        self.birth_doppler_head = nn.Linear(hidden_dim, doppler_count)
        self.birth_confidence_head = nn.Linear(hidden_dim, 1)

    @property
    def horizon_count(self) -> int:
        return int(self.horizons_seconds.numel())

    def _validate_inputs(
        self,
        current_cube_drae: torch.Tensor,
        history_cube_drae: torch.Tensor,
        history_time_seconds: torch.Tensor,
        observed_xyz_m: torch.Tensor,
        observed_doppler_probability: torch.Tensor,
        observed_confidence: torch.Tensor,
        observed_source_id: torch.Tensor,
        target_from_current: torch.Tensor,
    ) -> tuple[int, int, int]:
        if current_cube_drae.ndim != 5:
            raise ValueError(
                "D-MHW current Cube must have shape (B,D,R,A,E)"
            )
        batch_size = current_cube_drae.shape[0]
        expected = (
            self.doppler_mps.numel(),
            self.range_m.numel(),
            self.azimuth_rad.numel(),
            self.elevation_rad.numel(),
        )
        if tuple(current_cube_drae.shape[1:]) != expected:
            raise ValueError("D-MHW current Cube axes do not match")
        if history_cube_drae.ndim != 6:
            raise ValueError(
                "D-MHW history Cube must have shape (B,K,D,R,A,E)"
            )
        if history_cube_drae.shape[0] != batch_size:
            raise ValueError("D-MHW current/history batch mismatch")
        history_count = history_cube_drae.shape[1]
        if history_count < 2:
            raise ValueError(
                "D-MHW requires at least two strictly historical Cubes"
            )
        if tuple(history_cube_drae.shape[2:]) != expected:
            raise ValueError("D-MHW history Cube axes do not match")
        if history_time_seconds.shape != (batch_size, history_count):
            raise ValueError("D-MHW history times must have shape (B,K)")
        if not torch.isfinite(history_time_seconds).all():
            raise ValueError("D-MHW history times must be finite")
        if not (history_time_seconds < 0.0).all():
            raise ValueError("D-MHW history Cubes must precede current time")
        if not (
            history_time_seconds[:, 1:]
            > history_time_seconds[:, :-1]
        ).all():
            raise ValueError(
                "D-MHW history times must be strictly chronological"
            )
        if observed_xyz_m.ndim != 3 or observed_xyz_m.shape[-1] != 3:
            raise ValueError("D-MHW observed XYZ must have shape (B,N,3)")
        if observed_xyz_m.shape[0] != batch_size:
            raise ValueError("D-MHW observed-state batch mismatch")
        observed_count = observed_xyz_m.shape[1]
        if observed_count < self.persistent_count:
            raise ValueError(
                "D-MHW observed state cannot fill persistent quota"
            )
        if observed_doppler_probability.shape != (
            batch_size,
            observed_count,
            self.doppler_mps.numel(),
        ):
            raise ValueError("D-MHW observed Doppler shape mismatch")
        if observed_confidence.shape != (batch_size, observed_count):
            raise ValueError("D-MHW observed confidence shape mismatch")
        if observed_source_id.shape != observed_confidence.shape:
            raise ValueError("D-MHW observed source-ID shape mismatch")
        if torch.is_floating_point(observed_source_id):
            raise ValueError("D-MHW source IDs must use integer dtype")
        if target_from_current.shape != (
            batch_size,
            self.horizon_count,
            4,
            4,
        ):
            raise ValueError(
                "D-MHW target transforms must have shape (B,3,4,4)"
            )
        return batch_size, history_count, observed_count

    @staticmethod
    def _time_feature(seconds: torch.Tensor, scale: float) -> torch.Tensor:
        normalized = seconds / scale
        return torch.stack(
            (
                normalized,
                torch.sin(torch.pi * normalized),
                torch.cos(torch.pi * normalized),
            ),
            dim=-1,
        )

    def _condition(
        self,
        current_cube_drae: torch.Tensor,
        history_cube_drae: torch.Tensor,
        history_time_seconds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_count = history_cube_drae.shape[:2]
        history_latent = self.cube_encoder(
            history_cube_drae.reshape(
                batch_size * history_count,
                *history_cube_drae.shape[2:],
            )
        ).reshape(batch_size, history_count, -1)
        history_latent = history_latent + self.history_time_encoder(
            self._time_feature(history_time_seconds, scale=2.5)
        )
        _, hidden = self.history_sequence_encoder(history_latent)
        history_context = hidden[-1]
        current_context = self.cube_encoder(current_cube_drae)
        fused = self.condition_fusion(
            torch.cat(
                (
                    current_context,
                    history_context,
                    current_context * history_context,
                ),
                dim=-1,
            )
        )
        return fused, history_context

    def _select_persistent(
        self,
        observed_xyz_m: torch.Tensor,
        observed_doppler_probability: torch.Tensor,
        observed_confidence: torch.Tensor,
        observed_source_id: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        valid = (
            (observed_source_id >= 0)
            & torch.isfinite(observed_xyz_m).all(dim=-1)
            & torch.isfinite(observed_confidence)
        )
        if (valid.sum(dim=1) < self.persistent_count).any():
            raise ValueError(
                "Every D-MHW item needs enough valid persistent sources"
            )
        score = torch.where(
            valid,
            observed_confidence,
            observed_confidence.new_full((), float("-inf")),
        )
        index = torch.topk(
            score,
            self.persistent_count,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        selected = (
            _gather(observed_xyz_m, index),
            _gather(observed_doppler_probability, index),
            _gather(observed_confidence, index),
            _gather(observed_source_id, index),
        )
        for source_id in selected[3]:
            if source_id.unique().numel() != self.persistent_count:
                raise ValueError(
                    "D-MHW persistent source IDs must be unique per item"
                )
        return (*selected, index)

    def _current_local_spectrum(
        self,
        current_cube_drae: torch.Tensor,
        xyz_m: torch.Tensor,
    ) -> torch.Tensor:
        coordinates = []
        for batch_index in range(xyz_m.shape[0]):
            coordinate, _ = xyz_to_continuous_rae(
                xyz_m[batch_index],
                self.range_m,
                self.azimuth_rad,
                self.elevation_rad,
            )
            coordinates.append(coordinate)
        coordinates_rae = torch.stack(coordinates)
        batch_size, point_count = xyz_m.shape[:2]
        batch_index = torch.arange(
            batch_size, device=xyz_m.device
        )[:, None].expand(-1, point_count)
        query = torch.cat(
            (
                batch_index.reshape(-1, 1).to(coordinates_rae),
                coordinates_rae.reshape(-1, 3),
            ),
            dim=1,
        )
        spectrum = query_cube_spectrum(
            current_cube_drae, query
        ).reshape(batch_size, point_count, -1)
        return spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )

    def _horizon_token(self) -> torch.Tensor:
        feature = self._time_feature(
            self.horizons_seconds,
            scale=float(self.horizons_seconds[-1].item()),
        )
        return self.horizon_encoder(feature)

    def forward(
        self,
        *,
        current_cube_drae: torch.Tensor,
        history_cube_drae: torch.Tensor,
        history_time_seconds: torch.Tensor,
        observed_xyz_m: torch.Tensor,
        observed_doppler_probability: torch.Tensor,
        observed_confidence: torch.Tensor,
        observed_source_id: torch.Tensor,
        target_from_current: torch.Tensor,
        future_cube_drae: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, object]]:
        if future_cube_drae is not None:
            raise ValueError("D-MHW never reads a future Cube")
        batch_size, _, _ = self._validate_inputs(
            current_cube_drae,
            history_cube_drae,
            history_time_seconds,
            observed_xyz_m,
            observed_doppler_probability,
            observed_confidence,
            observed_source_id,
            target_from_current,
        )
        (
            selected_xyz,
            selected_probability,
            selected_confidence,
            selected_source_id,
            selected_index,
        ) = self._select_persistent(
            observed_xyz_m,
            observed_doppler_probability,
            observed_confidence,
            observed_source_id,
        )
        selected_probability = selected_probability.clamp_min(0.0)
        selected_probability = selected_probability / (
            selected_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        condition, history_context = self._condition(
            current_cube_drae,
            history_cube_drae,
            history_time_seconds,
        )
        local_spectrum = self._current_local_spectrum(
            current_cube_drae, selected_xyz
        )
        xyz_scale = selected_xyz.new_tensor([120.0, 120.0, 20.0])
        source_feature = torch.cat(
            (
                selected_xyz / xyz_scale,
                selected_probability,
                selected_confidence[:, :, None],
                local_spectrum,
            ),
            dim=-1,
        )
        source_token = self.persistent_source_encoder(source_feature)
        horizon_token = self._horizon_token()
        persistent_token = (
            source_token[:, None]
            + condition[:, None, None]
            + horizon_token[None, :, None]
        )
        persistent_token = self.persistent_decoder(persistent_token)

        horizon_count = self.horizon_count
        flat_batch = batch_size * horizon_count
        source_xyz_h = selected_xyz[:, None].expand(
            -1, horizon_count, -1, -1
        )
        source_probability_h = selected_probability[:, None].expand(
            -1, horizon_count, -1, -1
        )
        flat_source_xyz = source_xyz_h.reshape(
            flat_batch, self.persistent_count, 3
        )
        flat_source_probability = source_probability_h.reshape(
            flat_batch,
            self.persistent_count,
            self.doppler_mps.numel(),
        )
        flat_transform = target_from_current.reshape(flat_batch, 4, 4)
        delta_seconds = self.horizons_seconds[None].expand(
            batch_size, -1
        ).reshape(flat_batch)
        warp = mandatory_radial_doppler_warp(
            flat_source_xyz,
            flat_source_probability,
            flat_transform,
            delta_seconds,
            self.doppler_mps,
            self.doppler_lower_mps,
            self.doppler_period_mps,
        )
        raw_motion = self.persistent_motion_head(
            persistent_token
        ).reshape(flat_batch, self.persistent_count, 4)
        motion = compose_persistent_motion(
            warp,
            raw_motion[:, :, :3],
            raw_motion[:, :, 3],
            flat_transform,
            delta_seconds,
            maximum_tangential_speed_mps=(
                self.maximum_tangential_speed_mps
            ),
            maximum_radial_correction_mps=(
                self.maximum_radial_correction_mps
            ),
        )
        persistent_xyz = motion.xyz_m.reshape(
            batch_size, horizon_count, self.persistent_count, 3
        )
        persistent_doppler_logit = (
            self.persistent_doppler_head(persistent_token)
            + selected_probability[:, None].clamp_min(1e-8).log()
        )
        persistent_probability = torch.softmax(
            persistent_doppler_logit, dim=-1
        )
        persistent_confidence_logit = (
            self.persistent_confidence_head(persistent_token).squeeze(-1)
        )
        persistent_confidence = torch.sigmoid(
            persistent_confidence_logit
        )

        birth_token = (
            self.birth_queries[None, None]
            + condition[:, None, None]
            + horizon_token[None, :, None]
            + 0.25 * history_context[:, None, None]
        )
        birth_token = self.birth_decoder(birth_token)
        birth_coordinate = torch.sigmoid(
            self.birth_coordinate_head(birth_token)
        )
        coordinate_scale = birth_coordinate.new_tensor(
            [
                self.range_m.numel() - 1,
                self.azimuth_rad.numel() - 1,
                self.elevation_rad.numel() - 1,
            ]
        )
        birth_coordinate = birth_coordinate * coordinate_scale
        birth_xyz = continuous_rae_to_xyz(
            birth_coordinate.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        ).reshape(
            batch_size, horizon_count, self.birth_count, 3
        )
        birth_doppler_logit = self.birth_doppler_head(birth_token)
        birth_probability = torch.softmax(
            birth_doppler_logit, dim=-1
        )
        birth_confidence_logit = self.birth_confidence_head(
            birth_token
        ).squeeze(-1)
        birth_confidence = torch.sigmoid(birth_confidence_logit)

        persistent_source_id = selected_source_id[:, None].expand(
            -1, horizon_count, -1
        )
        birth_source_id = torch.full(
            (batch_size, horizon_count, self.birth_count),
            -1,
            dtype=selected_source_id.dtype,
            device=selected_source_id.device,
        )
        source_id = torch.cat(
            (persistent_source_id, birth_source_id), dim=2
        )
        persistent_mask = torch.zeros(
            batch_size,
            horizon_count,
            self.point_count,
            dtype=torch.bool,
            device=selected_xyz.device,
        )
        persistent_mask[:, :, : self.persistent_count] = True
        return {
            "xyz_m": torch.cat((persistent_xyz, birth_xyz), dim=2),
            "doppler_logit": torch.cat(
                (persistent_doppler_logit, birth_doppler_logit), dim=2
            ),
            "doppler_probability": torch.cat(
                (persistent_probability, birth_probability), dim=2
            ),
            "confidence_logit": torch.cat(
                (
                    persistent_confidence_logit,
                    birth_confidence_logit,
                ),
                dim=2,
            ),
            "confidence": torch.cat(
                (persistent_confidence, birth_confidence), dim=2
            ),
            "source_id": source_id,
            "persistent_mask": persistent_mask,
            "persistent_xyz_m": persistent_xyz,
            "persistent_doppler_probability": persistent_probability,
            "persistent_confidence": persistent_confidence,
            "persistent_source_id": persistent_source_id,
            "birth_xyz_m": birth_xyz,
            "birth_doppler_probability": birth_probability,
            "birth_confidence": birth_confidence,
            "selected_observed_index": selected_index,
            "selected_observed_xyz_m": selected_xyz,
            "selected_observed_doppler_probability": (
                selected_probability
            ),
            "persistent_base_xyz_m": warp.xyz_m.reshape(
                batch_size, horizon_count, self.persistent_count, 3
            ),
            "analytic_radial_velocity_mps": (
                warp.radial_velocity_mps.reshape(
                    batch_size, horizon_count, self.persistent_count
                )
            ),
            "analytic_radial_displacement_m": (
                warp.radial_displacement_m.reshape(
                    batch_size, horizon_count, self.persistent_count
                )
            ),
            "radial_direction": warp.radial_direction.reshape(
                batch_size, horizon_count, self.persistent_count, 3
            ),
            "tangential_velocity_mps": (
                motion.tangential_velocity_mps.reshape(
                    batch_size,
                    horizon_count,
                    self.persistent_count,
                    3,
                )
            ),
            "radial_correction_mps": (
                motion.radial_correction_mps.reshape(
                    batch_size, horizon_count, self.persistent_count
                )
            ),
            "learned_displacement_m": (
                motion.learned_displacement_m.reshape(
                    batch_size,
                    horizon_count,
                    self.persistent_count,
                    3,
                )
            ),
            "target_from_current": target_from_current,
            "horizons_seconds": self.horizons_seconds,
            "doppler_mps": self.doppler_mps,
            "doppler_lower_mps": self.doppler_lower_mps,
            "doppler_period_mps": self.doppler_period_mps,
            "metadata": {
                "protocol": "dmhw_direct_multi_horizon_v1",
                "horizons_seconds": list(DIRECT_HORIZONS_SECONDS),
                "direct_single_pass": True,
                "autoregressive_rollout": False,
                "uses_future_cube": False,
                "persistent_count": self.persistent_count,
                "birth_count": self.birth_count,
                "persistent_identity_preserved_across_horizons": True,
                "persistent_geometry_path": (
                    "mandatory_ego_plus_doppler_then_bounded_residual"
                ),
                "birth_condition": "history_and_current_full_raed",
            },
        }
