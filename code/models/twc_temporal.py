"""Minimal forced-temporal T-WC point-state generator.

The persistent branch has no current-only path. It starts from valid history
points, applies analytic radial Doppler displacement and ego alignment, then
predicts only tangential velocity plus a bounded radial calibration from a
current-Cube/history correlation token. A separate birth branch accounts for
newly visible support.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from losses.twc_physics import (
    compose_persistent_motion,
    mandatory_radial_doppler_warp,
)
from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.temporal_prior import xyz_to_continuous_rae


TWC_OUTPUT_MODES = ("online_enhancement", "forecast")


def _gather_points(
    tensor: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    if tensor.ndim < 2 or index.ndim != 2:
        raise ValueError("Batched gather expects tensor (B,N,...) and index (B,P)")
    if tensor.shape[0] != index.shape[0]:
        raise ValueError("Batched gather batch dimensions do not match")
    suffix = tensor.shape[2:]
    expanded = index.reshape(*index.shape, *([1] * len(suffix)))
    expanded = expanded.expand(*index.shape, *suffix)
    return torch.gather(tensor, 1, expanded)


def _batched_xyz_to_rae(
    xyz_m: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = []
    valid = []
    for batch_index in range(xyz_m.shape[0]):
        batch_coordinates, batch_valid = xyz_to_continuous_rae(
            xyz_m[batch_index],
            range_m,
            azimuth_rad,
            elevation_rad,
        )
        coordinates.append(batch_coordinates)
        valid.append(batch_valid)
    return torch.stack(coordinates), torch.stack(valid)


class ForcedTemporalTWC(nn.Module):
    """A mechanism scaffold in which persistent points require temporal state."""

    def __init__(
        self,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        doppler_mps: torch.Tensor,
        *,
        point_count: int = 10_000,
        persistent_fraction: float = 0.70,
        hidden_dim: int = 128,
        maximum_tangential_speed_mps: float = 12.0,
        maximum_radial_correction_mps: float = 1.0,
    ) -> None:
        super().__init__()
        if range_m.ndim != 1 or range_m.numel() < 2:
            raise ValueError("Range axis must contain at least two bins")
        if azimuth_rad.ndim != 1 or azimuth_rad.numel() < 2:
            raise ValueError("Azimuth axis must contain at least two bins")
        if elevation_rad.ndim != 1 or elevation_rad.numel() < 2:
            raise ValueError("Elevation axis must contain at least two bins")
        if doppler_mps.ndim != 1 or doppler_mps.numel() != 64:
            raise ValueError("T-WC requires the project 64-bin Doppler axis")
        for name, axis in (
            ("range", range_m),
            ("azimuth", azimuth_rad),
            ("elevation", elevation_rad),
            ("Doppler", doppler_mps),
        ):
            if not torch.isfinite(axis).all():
                raise ValueError(f"{name} axis must be finite")
            if not torch.all(axis[1:] > axis[:-1]):
                raise ValueError(f"{name} axis must be strictly increasing")
        if point_count < 2:
            raise ValueError("T-WC requires at least two output points")
        if not 0.0 < persistent_fraction < 1.0:
            raise ValueError("Persistent fraction must lie strictly in (0,1)")
        if hidden_dim < 8:
            raise ValueError("T-WC hidden dimension must be at least eight")
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
        doppler_count = doppler_mps.numel()
        frame_feature_dim = 3 + doppler_count + 2
        self.history_frame_encoder = nn.Sequential(
            nn.Linear(frame_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.history_sequence_encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
        )
        correlation_feature_dim = 2 * doppler_count + 1
        self.correlation_encoder = nn.Sequential(
            nn.Linear(correlation_feature_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.sequence_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.persistent_motion_head = nn.Linear(hidden_dim, 4)
        self.persistent_doppler_head = nn.Linear(
            hidden_dim, doppler_count
        )
        self.persistent_confidence_head = nn.Linear(hidden_dim, 1)
        self.cube_encoder = nn.Sequential(
            nn.Linear(doppler_count, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.birth_queries = nn.Parameter(
            torch.randn(self.birth_count, hidden_dim) * 0.02
        )
        self.birth_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.birth_coordinate_head = nn.Linear(hidden_dim, 3)
        self.birth_doppler_head = nn.Linear(hidden_dim, doppler_count)
        self.birth_confidence_head = nn.Linear(hidden_dim, 1)

    def _validate_inputs(
        self,
        conditioning_cube_drae: torch.Tensor,
        history_xyz_m: torch.Tensor,
        history_doppler_probability: torch.Tensor,
        history_confidence: torch.Tensor,
        history_source_id: torch.Tensor,
        current_from_history: torch.Tensor,
    ) -> tuple[int, int, int]:
        if conditioning_cube_drae.ndim != 5:
            raise ValueError("Conditioning Cube must have shape (B,D,R,A,E)")
        batch_size, doppler_count, range_count, azimuth_count, elevation_count = (
            conditioning_cube_drae.shape
        )
        expected_cube_shape = (
            self.doppler_mps.numel(),
            self.range_m.numel(),
            self.azimuth_rad.numel(),
            self.elevation_rad.numel(),
        )
        if (
            doppler_count,
            range_count,
            azimuth_count,
            elevation_count,
        ) != expected_cube_shape:
            raise ValueError(
                "Conditioning Cube axes do not match registered T-WC axes"
            )
        if history_xyz_m.ndim != 4 or history_xyz_m.shape[-1] != 3:
            raise ValueError("History XYZ must have shape (B,K,N,3)")
        if history_xyz_m.shape[0] != batch_size:
            raise ValueError("Cube and history batch dimensions do not match")
        _, frame_count, history_count, _ = history_xyz_m.shape
        if frame_count < 2:
            raise ValueError("T-WC requires at least two causal history frames")
        if history_count < self.persistent_count:
            raise ValueError(
                "Latest history frame cannot fill the persistent quota"
            )
        if history_doppler_probability.shape != (
            batch_size,
            frame_count,
            history_count,
            doppler_count,
        ):
            raise ValueError("History Doppler does not align with T-WC inputs")
        if history_confidence.shape != (
            batch_size,
            frame_count,
            history_count,
        ):
            raise ValueError("History confidence does not align with T-WC inputs")
        if history_source_id.shape != history_confidence.shape:
            raise ValueError("History source IDs do not align with T-WC inputs")
        if torch.is_floating_point(history_source_id):
            raise ValueError("History source IDs must use an integer dtype")
        if current_from_history.shape != (batch_size, 4, 4):
            raise ValueError("current_from_history must have shape (B,4,4)")
        return batch_size, frame_count, history_count

    def _history_context(
        self,
        history_xyz_m: torch.Tensor,
        history_doppler_probability: torch.Tensor,
        history_confidence: torch.Tensor,
        history_source_id: torch.Tensor,
    ) -> torch.Tensor:
        valid = (
            (history_source_id >= 0)
            & torch.isfinite(history_xyz_m).all(dim=-1)
            & torch.isfinite(history_confidence)
        )
        weight = (
            history_confidence.clamp_min(0.0) * valid.to(history_confidence)
        )
        denominator = weight.sum(dim=2, keepdim=True).clamp_min(1e-8)
        normalized_weight = weight / denominator
        xyz_scale = history_xyz_m.new_tensor([120.0, 120.0, 20.0])
        safe_xyz = torch.where(
            valid[:, :, :, None],
            history_xyz_m,
            torch.zeros_like(history_xyz_m),
        )
        xyz_mean = (
            safe_xyz / xyz_scale
            * normalized_weight[:, :, :, None]
        ).sum(dim=2)
        probability = torch.where(
            valid[:, :, :, None],
            history_doppler_probability,
            torch.zeros_like(history_doppler_probability),
        ).clamp_min(0.0)
        probability = probability / probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        doppler_mean = (
            probability * normalized_weight[:, :, :, None]
        ).sum(dim=2)
        safe_confidence = torch.where(
            valid,
            history_confidence,
            torch.zeros_like(history_confidence),
        )
        confidence_mean = (
            safe_confidence * normalized_weight
        ).sum(dim=2, keepdim=True)
        frame_count = history_xyz_m.shape[1]
        frame_position = torch.linspace(
            -1.0,
            1.0,
            frame_count,
            dtype=history_xyz_m.dtype,
            device=history_xyz_m.device,
        )
        frame_position = frame_position.view(1, frame_count, 1).expand(
            history_xyz_m.shape[0], -1, -1
        )
        frame_feature = torch.cat(
            (
                xyz_mean,
                doppler_mean,
                confidence_mean,
                frame_position,
            ),
            dim=-1,
        )
        encoded = self.history_frame_encoder(frame_feature)
        _, hidden = self.history_sequence_encoder(encoded)
        return hidden[-1]

    def _select_latest_persistent(
        self,
        history_xyz_m: torch.Tensor,
        history_doppler_probability: torch.Tensor,
        history_confidence: torch.Tensor,
        history_source_id: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        latest_xyz = history_xyz_m[:, -1]
        latest_probability = history_doppler_probability[:, -1]
        latest_confidence = history_confidence[:, -1]
        latest_source_id = history_source_id[:, -1]
        valid = (
            (latest_source_id >= 0)
            & torch.isfinite(latest_xyz).all(dim=-1)
            & torch.isfinite(latest_confidence)
        )
        valid_count = valid.sum(dim=1)
        if (valid_count < self.persistent_count).any():
            raise ValueError(
                "Every batch item must provide enough valid latest-frame "
                "history points for the persistent quota"
            )
        score = torch.where(
            valid,
            latest_confidence,
            latest_confidence.new_full((), float("-inf")),
        )
        selected_index = torch.topk(
            score,
            self.persistent_count,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        selected_xyz = _gather_points(latest_xyz, selected_index)
        selected_probability = _gather_points(
            latest_probability, selected_index
        )
        selected_confidence = _gather_points(
            latest_confidence, selected_index
        )
        selected_source_id = _gather_points(
            latest_source_id, selected_index
        )
        if (selected_source_id < 0).any():
            raise RuntimeError("Persistent output selected an invalid source ID")
        return (
            selected_xyz,
            selected_probability,
            selected_confidence,
            selected_source_id,
            selected_index,
        )

    def _sample_conditioning_spectrum(
        self,
        conditioning_cube_drae: torch.Tensor,
        xyz_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coordinates, valid = _batched_xyz_to_rae(
            xyz_m,
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        )
        batch_size, point_count, _ = coordinates.shape
        batch_index = torch.arange(
            batch_size, device=coordinates.device
        )[:, None].expand(-1, point_count)
        query = torch.cat(
            (
                batch_index.reshape(-1, 1).to(coordinates),
                coordinates.reshape(-1, 3),
            ),
            dim=1,
        )
        spectrum = query_cube_spectrum(
            conditioning_cube_drae, query
        ).reshape(batch_size, point_count, -1)
        return spectrum, coordinates, valid

    def _persistent_token(
        self,
        selected_probability: torch.Tensor,
        conditioning_spectrum: torch.Tensor,
        history_context: torch.Tensor,
    ) -> torch.Tensor:
        history_probability = selected_probability.clamp_min(0.0)
        history_probability = history_probability / history_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        cube_probability = conditioning_spectrum.clamp_min(0.0)
        cube_probability = cube_probability / cube_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        correlation = history_probability * cube_probability
        discrepancy = (history_probability - cube_probability).abs()
        cosine = (
            history_probability * cube_probability
        ).sum(dim=-1, keepdim=True) / (
            torch.linalg.vector_norm(
                history_probability, dim=-1, keepdim=True
            )
            * torch.linalg.vector_norm(
                cube_probability, dim=-1, keepdim=True
            )
        ).clamp_min(1e-8)
        features = torch.cat(
            (
                correlation,
                discrepancy,
                cosine,
            ),
            dim=-1,
        )
        correlation_token = self.correlation_encoder(features)
        modulation = self.sequence_modulation(history_context)[:, None]
        return correlation_token * modulation

    def _birth_output(
        self,
        conditioning_cube_drae: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cube_spectrum = torch.log1p(
            conditioning_cube_drae.clamp_min(0.0)
        ).mean(dim=(2, 3, 4))
        cube_spectrum = cube_spectrum / cube_spectrum.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        cube_context = self.cube_encoder(cube_spectrum)
        token = self.birth_queries[None] + cube_context[:, None]
        token = self.birth_decoder(token)
        coordinate_fraction = torch.sigmoid(
            self.birth_coordinate_head(token)
        )
        coordinate_scale = coordinate_fraction.new_tensor(
            [
                self.range_m.numel() - 1,
                self.azimuth_rad.numel() - 1,
                self.elevation_rad.numel() - 1,
            ]
        )
        coordinates = coordinate_fraction * coordinate_scale
        batch_size = coordinates.shape[0]
        xyz = continuous_rae_to_xyz(
            coordinates.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        ).reshape(batch_size, self.birth_count, 3)
        doppler_logit = self.birth_doppler_head(token)
        confidence_logit = self.birth_confidence_head(token).squeeze(-1)
        return {
            "xyz_m": xyz,
            "coordinates_rae": coordinates,
            "doppler_logit": doppler_logit,
            "doppler_probability": torch.softmax(
                doppler_logit, dim=-1
            ),
            "confidence_logit": confidence_logit,
            "confidence": torch.sigmoid(confidence_logit),
        }

    def forward(
        self,
        conditioning_cube_drae: torch.Tensor,
        history_xyz_m: torch.Tensor,
        history_doppler_probability: torch.Tensor,
        history_confidence: torch.Tensor,
        history_source_id: torch.Tensor,
        current_from_history: torch.Tensor,
        delta_seconds: torch.Tensor | float,
        *,
        mode: str,
        target_offset_steps: int = 0,
        future_cube_drae: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, object]]:
        """Generate current enhancement or a no-future-Cube forecast."""

        self._validate_inputs(
            conditioning_cube_drae,
            history_xyz_m,
            history_doppler_probability,
            history_confidence,
            history_source_id,
            current_from_history,
        )
        if mode not in TWC_OUTPUT_MODES:
            raise ValueError(f"Unsupported T-WC output mode {mode}")
        if future_cube_drae is not None:
            raise ValueError("T-WC never accepts a future target Cube")
        if mode == "online_enhancement" and target_offset_steps != 0:
            raise ValueError(
                "Online enhancement must target the conditioning-Cube time"
            )
        if mode == "forecast" and target_offset_steps < 1:
            raise ValueError(
                "Forecast mode requires a positive target offset"
            )
        (
            selected_xyz,
            selected_probability,
            selected_confidence,
            selected_source_id,
            selected_index,
        ) = self._select_latest_persistent(
            history_xyz_m,
            history_doppler_probability,
            history_confidence,
            history_source_id,
        )
        history_context = self._history_context(
            history_xyz_m,
            history_doppler_probability,
            history_confidence,
            history_source_id,
        )
        warp = mandatory_radial_doppler_warp(
            selected_xyz,
            selected_probability,
            current_from_history,
            delta_seconds,
            self.doppler_mps,
            self.doppler_lower_mps,
            self.doppler_period_mps,
        )
        conditioning_spectrum, base_coordinates, base_valid = (
            self._sample_conditioning_spectrum(
                conditioning_cube_drae,
                warp.xyz_m,
            )
        )
        persistent_token = self._persistent_token(
            selected_probability,
            conditioning_spectrum,
            history_context,
        )
        raw_motion = self.persistent_motion_head(persistent_token)
        motion = compose_persistent_motion(
            warp,
            raw_motion[:, :, :3],
            raw_motion[:, :, 3],
            current_from_history,
            delta_seconds,
            maximum_tangential_speed_mps=(
                self.maximum_tangential_speed_mps
            ),
            maximum_radial_correction_mps=(
                self.maximum_radial_correction_mps
            ),
        )
        persistent_coordinates, persistent_valid = _batched_xyz_to_rae(
            motion.xyz_m,
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        )
        persistent_doppler_logit = self.persistent_doppler_head(
            persistent_token
        )
        persistent_doppler_probability = torch.softmax(
            persistent_doppler_logit, dim=-1
        )
        persistent_confidence_logit = (
            self.persistent_confidence_head(persistent_token).squeeze(-1)
        )
        persistent_confidence = torch.sigmoid(
            persistent_confidence_logit
        )
        birth = self._birth_output(conditioning_cube_drae)
        batch_size = conditioning_cube_drae.shape[0]
        persistent_mask = torch.zeros(
            batch_size,
            self.point_count,
            dtype=torch.bool,
            device=conditioning_cube_drae.device,
        )
        persistent_mask[:, : self.persistent_count] = True
        birth_source_id = torch.full(
            (batch_size, self.birth_count),
            -1,
            dtype=selected_source_id.dtype,
            device=selected_source_id.device,
        )
        output_source_id = torch.cat(
            (selected_source_id, birth_source_id), dim=1
        )
        if (output_source_id[persistent_mask] < 0).any():
            raise RuntimeError(
                "Every persistent T-WC output must carry a valid source ID"
            )
        metadata = {
            "mode": mode,
            "uses_future_cube": False,
            "conditioning_cube_role": (
                "target_current_cube"
                if mode == "online_enhancement"
                else "last_observed_cube"
            ),
            "target_offset_steps": target_offset_steps,
            "persistent_count": self.persistent_count,
            "birth_count": self.birth_count,
            "persistent_head_requires_cube_history_correlation": True,
        }
        return {
            "xyz_m": torch.cat((motion.xyz_m, birth["xyz_m"]), dim=1),
            "coordinates_rae": torch.cat(
                (persistent_coordinates, birth["coordinates_rae"]), dim=1
            ),
            "doppler_logit": torch.cat(
                (persistent_doppler_logit, birth["doppler_logit"]), dim=1
            ),
            "doppler_probability": torch.cat(
                (
                    persistent_doppler_probability,
                    birth["doppler_probability"],
                ),
                dim=1,
            ),
            "confidence_logit": torch.cat(
                (
                    persistent_confidence_logit,
                    birth["confidence_logit"],
                ),
                dim=1,
            ),
            "confidence": torch.cat(
                (persistent_confidence, birth["confidence"]), dim=1
            ),
            "history_source_id": output_source_id,
            "persistent_mask": persistent_mask,
            "persistent_xyz_m": motion.xyz_m,
            "persistent_coordinates_rae": persistent_coordinates,
            "persistent_doppler_probability": (
                persistent_doppler_probability
            ),
            "persistent_confidence": persistent_confidence,
            "persistent_base_xyz_m": warp.xyz_m,
            "persistent_base_coordinates_rae": base_coordinates,
            "persistent_base_valid": base_valid,
            "persistent_valid": persistent_valid,
            "selected_history_index": selected_index,
            "selected_history_xyz_m": selected_xyz,
            "selected_history_doppler_probability": selected_probability,
            "selected_history_confidence": selected_confidence,
            "selected_history_source_id": selected_source_id,
            "conditioning_spectrum": conditioning_spectrum,
            "analytic_radial_velocity_mps": warp.radial_velocity_mps,
            "analytic_radial_displacement_m": warp.radial_displacement_m,
            "tangential_velocity_mps": motion.tangential_velocity_mps,
            "radial_correction_mps": motion.radial_correction_mps,
            "learned_displacement_m": motion.learned_displacement_m,
            "current_from_history": current_from_history,
            "delta_seconds": torch.as_tensor(
                delta_seconds,
                dtype=conditioning_cube_drae.dtype,
                device=conditioning_cube_drae.device,
            ).reshape(-1).expand(batch_size),
            "doppler_mps": self.doppler_mps,
            "doppler_lower_mps": self.doppler_lower_mps,
            "doppler_period_mps": self.doppler_period_mps,
            "metadata": metadata,
        }
