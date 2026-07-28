"""RaLD-style query-field geometry from a Full-RAED radar Cube.

G1D uses deterministic radar proposals, a strictly ordered mixed-latent
encoder, and an arbitrary-query occupancy field. Doppler is not predicted by
this geometry model: the final point spectrum is queried directly from the
measured Cube at the generated coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
import torch.nn as nn

from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.point_to_cube import trilinear_query_features
from models.rald_matched import (
    FourierPointEmbedding,
    FullRAEDRadarTokenEncoder,
    PreNormAttention,
    PreNormFeedForward,
)


@dataclass(frozen=True)
class RadarProposalSet:
    """Stable, locally suppressed Full-RAED energy proposals."""

    coordinates_rae: torch.Tensor
    flat_index: torch.Tensor
    integrated_log_energy: torch.Tensor


def integrated_log_energy(cube_drae: torch.Tensor) -> torch.Tensor:
    """Integrate nonnegative log power over all 64 Doppler bins."""

    if cube_drae.ndim != 5 or cube_drae.shape[1] != 64:
        raise ValueError(
            f"Expected Full-RAED Cube (B,64,R,A,E), got {cube_drae.shape}"
        )
    if not torch.is_floating_point(cube_drae):
        raise TypeError("Full-RAED Cube must be floating point")
    if not torch.isfinite(cube_drae).all():
        raise ValueError("Full-RAED Cube must contain only finite values")
    return torch.log1p(cube_drae.clamp_min(0.0)).sum(dim=1)


def _flat_to_rae(
    flat_index: torch.Tensor, spatial_shape: tuple[int, int, int]
) -> torch.Tensor:
    _, azimuth_count, elevation_count = spatial_shape
    radius = flat_index // (azimuth_count * elevation_count)
    remainder = flat_index % (azimuth_count * elevation_count)
    azimuth = remainder // elevation_count
    elevation = remainder % elevation_count
    return torch.stack((radius, azimuth, elevation), dim=-1)


def stable_radar_proposals(
    cube_drae: torch.Tensor,
    *,
    seed_count: int = 1_000,
    nms_kernel: tuple[int, int, int] = (5, 5, 3),
) -> RadarProposalSet:
    """Select deterministic proposals with stable greedy local suppression.

    Cells are ranked by descending integrated energy and ascending flat index.
    The ranked discrete indices are detached before greedy suppression because
    proposal selection is intentionally non-differentiable.
    """

    energy = integrated_log_energy(cube_drae)
    if seed_count <= 0:
        raise ValueError("Radar proposal count must be positive")
    if len(nms_kernel) != 3 or any(
        size <= 0 or size % 2 == 0 for size in nms_kernel
    ):
        raise ValueError("NMS kernel sizes must be positive odd integers")
    spatial_shape = tuple(int(size) for size in energy.shape[1:])
    spatial_count = energy[0].numel()
    if seed_count > spatial_count:
        raise ValueError("Radar proposal count exceeds Cube spatial cells")

    flat_energy = energy.detach().flatten(start_dim=1)
    order = torch.argsort(flat_energy, dim=1, descending=True, stable=True)
    range_count, azimuth_count, elevation_count = spatial_shape
    half_width = tuple(size // 2 for size in nms_kernel)
    selected_batches: list[list[int]] = []
    transfer_chunk = min(spatial_count, 65_536)
    for batch_index in range(flat_energy.shape[0]):
        suppressed = bytearray(spatial_count)
        selected: list[int] = []
        cursor = 0
        while cursor < spatial_count and len(selected) < seed_count:
            stop = min(cursor + transfer_chunk, spatial_count)
            ranked = order[batch_index, cursor:stop].detach().cpu().tolist()
            for raw_flat_index in ranked:
                flat_index = int(raw_flat_index)
                if suppressed[flat_index]:
                    continue
                selected.append(flat_index)
                radius = flat_index // (azimuth_count * elevation_count)
                remainder = flat_index % (azimuth_count * elevation_count)
                azimuth = remainder // elevation_count
                elevation = remainder % elevation_count
                radius_start = max(0, radius - half_width[0])
                radius_stop = min(range_count, radius + half_width[0] + 1)
                azimuth_start = max(0, azimuth - half_width[1])
                azimuth_stop = min(
                    azimuth_count, azimuth + half_width[1] + 1
                )
                elevation_start = max(0, elevation - half_width[2])
                elevation_stop = min(
                    elevation_count, elevation + half_width[2] + 1
                )
                suppression_bytes = b"\x01" * (
                    elevation_stop - elevation_start
                )
                for radius_index in range(radius_start, radius_stop):
                    radius_offset = (
                        radius_index * azimuth_count * elevation_count
                    )
                    for azimuth_index in range(
                        azimuth_start, azimuth_stop
                    ):
                        start = (
                            radius_offset
                            + azimuth_index * elevation_count
                            + elevation_start
                        )
                        suppressed[start : start + len(suppression_bytes)] = (
                            suppression_bytes
                        )
                if len(selected) == seed_count:
                    break
            cursor = stop
        if len(selected) != seed_count:
            raise RuntimeError(
                "Stable greedy radar NMS cannot pack the requested seed count"
            )
        selected_batches.append(selected)

    selected_flat = torch.tensor(
        selected_batches, dtype=torch.long, device=energy.device
    )
    selected_energy = flat_energy.gather(1, selected_flat)
    return RadarProposalSet(
        coordinates_rae=_flat_to_rae(selected_flat, spatial_shape).to(cube_drae),
        flat_index=selected_flat,
        integrated_log_energy=selected_energy.to(cube_drae),
    )


def coarse_query_templates(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the fixed 4 x 4 x 2 coarse RAE template bank."""

    values = list(
        product(
            (-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0),
            (-1.5, -0.5, 0.5, 1.5),
            (-0.5, 0.5),
        )
    )
    templates = torch.tensor(values, device=device, dtype=dtype)
    if templates.shape != (32, 3):
        raise AssertionError("G1D coarse template construction must produce 32 rows")
    return templates


def tetrahedral_local_templates(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return four centered tetrahedral offsets for final point expansion."""

    return torch.tensor(
        [
            [0.25, 0.25, 0.25],
            [0.25, -0.25, -0.25],
            [-0.25, 0.25, -0.25],
            [-0.25, -0.25, 0.25],
        ],
        device=device,
        dtype=dtype,
    )


class RaLDConditionedLatentBlock(nn.Module):
    """One latent block with self, Full-RAED cross, and feed-forward updates."""

    def __init__(self, model_dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.self_attention = PreNormAttention(
            model_dim, heads=heads, head_dim=head_dim
        )
        self.radar_attention = PreNormAttention(
            model_dim, model_dim, heads=heads, head_dim=head_dim
        )
        self.feed_forward = PreNormFeedForward(model_dim)

    def forward(
        self, latent: torch.Tensor, radar_tokens: torch.Tensor
    ) -> torch.Tensor:
        latent = latent + self.self_attention(latent)
        latent = latent + self.radar_attention(latent, radar_tokens)
        return latent + self.feed_forward(latent)


class RaLDQueryField(nn.Module):
    """Generate a fixed dense point set through a RaLD-style query field."""

    COARSE_TEMPLATE_COUNT = 32
    LOCAL_TEMPLATE_COUNT = 4
    SPECTRUM_BIN_COUNT = 64

    def __init__(
        self,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        *,
        log_center: float,
        log_scale: float,
        base_seed_count: int = 1_000,
        selected_coarse_count: int = 2_500,
        latent_count: int = 512,
        model_dim: int = 512,
        depth: int = 24,
        heads: int = 8,
        head_dim: int = 64,
        fourier_frequency_dim: int = 48,
        radar_base_channels: int = 64,
        radar_spectral_channels: int = 16,
        radar_encoded_shape: tuple[int, int, int] = (16, 7, 3),
        radar_encoded_channels: int = 16,
        radar_channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        radar_blocks_per_level: int = 2,
        offset_bounds_bins: tuple[float, float, float] = (8.0, 4.0, 2.0),
        nms_kernel: tuple[int, int, int] = (5, 5, 3),
        decode_chunk_size: int = 4_096,
    ) -> None:
        super().__init__()
        axes = (range_m, azimuth_rad, elevation_rad)
        if any(axis.ndim != 1 or axis.numel() <= 1 for axis in axes):
            raise ValueError("G1D RAE axes must be one-dimensional and nontrivial")
        if base_seed_count <= 0 or selected_coarse_count <= 0:
            raise ValueError("G1D proposal counts must be positive")
        coarse_count = base_seed_count * self.COARSE_TEMPLATE_COUNT
        if selected_coarse_count > coarse_count:
            raise ValueError("Selected coarse count exceeds the coarse query pool")
        if latent_count <= 0 or model_dim <= 0 or depth <= 0:
            raise ValueError("G1D latent dimensions and depth must be positive")
        if decode_chunk_size <= 0:
            raise ValueError("G1D decode chunk size must be positive")
        if len(offset_bounds_bins) != 3 or any(
            bound <= 0.0 for bound in offset_bounds_bins
        ):
            raise ValueError("G1D offset bounds must contain three positive values")
        if log_scale <= 0.0:
            raise ValueError("G1D Full-RAED normalization scale must be positive")

        self.base_seed_count = base_seed_count
        self.selected_coarse_count = selected_coarse_count
        self.coarse_query_count = coarse_count
        self.point_count = selected_coarse_count * self.LOCAL_TEMPLATE_COUNT
        self.latent_count = latent_count
        self.model_dim = model_dim
        self.depth = depth
        self.nms_kernel = nms_kernel
        self.decode_chunk_size = decode_chunk_size
        self.radar_encoded_shape = radar_encoded_shape
        self.expected_radar_token_count = int(
            radar_encoded_shape[0]
            * radar_encoded_shape[1]
            * radar_encoded_shape[2]
        )

        self.coordinate_embedding = FourierPointEmbedding(
            model_dim, frequency_dim=fourier_frequency_dim
        )
        self.query_state_projection = nn.Sequential(
            nn.Linear(self.SPECTRUM_BIN_COUNT + 2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
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

        self.static_latents = nn.Embedding(latent_count, model_dim)
        self.dynamic_latents = nn.Embedding(latent_count, model_dim)
        self.dynamic_proposal_attention = PreNormAttention(
            model_dim, model_dim, heads=heads, head_dim=head_dim
        )
        self.mixed_projection = nn.Linear(model_dim, model_dim)
        self.post_mix_proposal_attention = PreNormAttention(
            model_dim, model_dim, heads=heads, head_dim=head_dim
        )
        self.post_mix_feed_forward = PreNormFeedForward(model_dim)
        self.latent_blocks = nn.ModuleList(
            [
                RaLDConditionedLatentBlock(model_dim, heads, head_dim)
                for _ in range(depth)
            ]
        )

        self.decoder_attention = PreNormAttention(
            model_dim, model_dim, heads=heads, head_dim=head_dim
        )
        self.decoder_feed_forward = PreNormFeedForward(model_dim)
        self.occupancy_head = nn.Linear(model_dim, 1)
        self.confidence_head = nn.Linear(model_dim, 1)
        self.offset_head = nn.Linear(model_dim, 3)
        for head in (self.occupancy_head, self.confidence_head, self.offset_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.register_buffer("range_m", range_m.float(), persistent=True)
        self.register_buffer("azimuth_rad", azimuth_rad.float(), persistent=True)
        self.register_buffer("elevation_rad", elevation_rad.float(), persistent=True)
        self.register_buffer(
            "offset_bounds_bins",
            torch.tensor(offset_bounds_bins, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "coarse_templates",
            coarse_query_templates(),
            persistent=True,
        )
        self.register_buffer(
            "local_templates",
            tetrahedral_local_templates(),
            persistent=True,
        )

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return (
            self.range_m.numel(),
            self.azimuth_rad.numel(),
            self.elevation_rad.numel(),
        )

    def _validate_cube(self, cube_drae: torch.Tensor) -> None:
        if cube_drae.ndim != 5 or cube_drae.shape[1] != self.SPECTRUM_BIN_COUNT:
            raise ValueError(
                f"Expected Full-RAED Cube (B,64,R,A,E), got {cube_drae.shape}"
            )
        if tuple(int(size) for size in cube_drae.shape[2:]) != self.spatial_shape:
            raise ValueError(
                f"Cube spatial shape {tuple(cube_drae.shape[2:])} does not match "
                f"configured axes {self.spatial_shape}"
            )
        if not torch.is_floating_point(cube_drae):
            raise TypeError("Full-RAED Cube must be floating point")
        if not torch.isfinite(cube_drae).all():
            raise ValueError("Full-RAED Cube must contain only finite values")

    def _clamp_coordinates(self, coordinates_rae: torch.Tensor) -> torch.Tensor:
        maximum = coordinates_rae.new_tensor(
            [size - 1 for size in self.spatial_shape]
        )
        return coordinates_rae.clamp_min(0.0).minimum(maximum)

    def _normalize_coordinates(self, coordinates_rae: torch.Tensor) -> torch.Tensor:
        maximum = coordinates_rae.new_tensor(
            [size - 1 for size in self.spatial_shape]
        )
        return 2.0 * coordinates_rae / maximum - 1.0

    def _denormalize_coordinates(
        self, normalized_rae: torch.Tensor
    ) -> torch.Tensor:
        maximum = normalized_rae.new_tensor(
            [size - 1 for size in self.spatial_shape]
        )
        return (normalized_rae + 1.0) * maximum / 2.0

    @staticmethod
    def _validate_batched_coordinates(
        coordinates: torch.Tensor,
        *,
        batch_size: int,
        normalized: bool,
    ) -> None:
        if coordinates.ndim != 3 or coordinates.shape[0] != batch_size:
            raise ValueError(
                f"Expected batched query coordinates (B,N,3), got {coordinates.shape}"
            )
        if coordinates.shape[-1] != 3 or coordinates.shape[1] <= 0:
            raise ValueError("Query coordinates must contain a nonempty (N,3) set")
        if not torch.is_floating_point(coordinates):
            raise TypeError("Query coordinates must be floating point")
        if not torch.isfinite(coordinates).all():
            raise ValueError("Query coordinates must contain only finite values")
        if normalized and (
            torch.any(coordinates < -1.0) or torch.any(coordinates > 1.0)
        ):
            raise ValueError("Normalized RAE queries must lie in [-1, 1]")

    @staticmethod
    def _batch_indices(
        batch_size: int, point_count: int, device: torch.device
    ) -> torch.Tensor:
        return torch.arange(device=device, end=batch_size)[:, None].expand(
            -1, point_count
        )

    def _query_spectrum(
        self,
        cube_drae: torch.Tensor,
        coordinates_rae: torch.Tensor,
        *,
        log_power_drae: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_batched_coordinates(
            coordinates_rae,
            batch_size=cube_drae.shape[0],
            normalized=False,
        )
        batch_size, point_count, _ = coordinates_rae.shape
        batch = self._batch_indices(batch_size, point_count, cube_drae.device)
        if log_power_drae is None:
            query = torch.cat(
                (
                    batch.reshape(-1, 1).to(coordinates_rae),
                    coordinates_rae.reshape(-1, 3),
                ),
                dim=1,
            )
            spectrum = query_cube_spectrum(cube_drae, query)
        else:
            if log_power_drae.shape != cube_drae.shape:
                raise ValueError("Cached log-power Cube has the wrong shape")
            spectrum = trilinear_query_features(
                log_power_drae,
                coordinates_rae.reshape(-1, 3),
                batch.reshape(-1),
            )
            spectrum = spectrum.clamp_min(0.0) + 1e-4
            spectrum = spectrum / spectrum.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
        expected = (batch_size * point_count, self.SPECTRUM_BIN_COUNT)
        if spectrum.shape != expected:
            raise RuntimeError(
                f"Cube spectrum query returned {spectrum.shape}, expected {expected}"
            )
        return spectrum.reshape(batch_size, point_count, self.SPECTRUM_BIN_COUNT)

    def query_tokens(
        self,
        cube_drae: torch.Tensor,
        coordinates_rae: torch.Tensor,
        *,
        log_power_drae: torch.Tensor | None = None,
        energy_rae: torch.Tensor | None = None,
        validate_cube: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Embed coordinates, normalized local spectrum, energy, and range."""

        if validate_cube:
            self._validate_cube(cube_drae)
        self._validate_batched_coordinates(
            coordinates_rae,
            batch_size=cube_drae.shape[0],
            normalized=False,
        )
        coordinates_rae = self._clamp_coordinates(coordinates_rae)
        batch_size, point_count, _ = coordinates_rae.shape
        normalized = self._normalize_coordinates(coordinates_rae)
        spectrum = self._query_spectrum(
            cube_drae,
            coordinates_rae,
            log_power_drae=log_power_drae,
        )

        if energy_rae is None:
            energy_rae = integrated_log_energy(cube_drae)
        if energy_rae.shape != (cube_drae.shape[0], *self.spatial_shape):
            raise ValueError("Cached integrated-energy Cube has the wrong shape")
        energy_grid = energy_rae.unsqueeze(1)
        batch = self._batch_indices(batch_size, point_count, cube_drae.device)
        absolute_energy = trilinear_query_features(
            energy_grid,
            coordinates_rae.reshape(-1, 3),
            batch.reshape(-1),
        ).reshape(batch_size, point_count, 1)
        normalized_range = normalized[..., :1]
        absolute_log_energy = absolute_energy
        state = torch.cat((spectrum, absolute_log_energy, normalized_range), dim=-1)
        if state.shape[-1] != self.SPECTRUM_BIN_COUNT + 2:
            raise AssertionError("G1D query state must contain exactly 66 values")
        token = self.coordinate_embedding(normalized)
        token = token + self.query_state_projection(state)
        return {
            "tokens": token,
            "normalized_rae": normalized,
            "coordinates_rae": coordinates_rae,
            "local_spectrum": spectrum,
            "absolute_log_energy": absolute_log_energy,
            "normalized_range": normalized_range,
        }

    def encode_latent(
        self,
        proposal_tokens: torch.Tensor,
        radar_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the exact G1D mixed-latent topology and conditioned blocks."""

        if proposal_tokens.ndim != 3 or proposal_tokens.shape[-1] != self.model_dim:
            raise ValueError("Proposal tokens must have shape (B,N,model_dim)")
        if radar_tokens.ndim != 3 or radar_tokens.shape[-1] != self.model_dim:
            raise ValueError("Radar tokens must have shape (B,N,model_dim)")
        if proposal_tokens.shape[0] != radar_tokens.shape[0]:
            raise ValueError("Proposal and radar token batches must match")
        batch_size = proposal_tokens.shape[0]
        static = self.static_latents.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        dynamic_query = self.dynamic_latents.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Strict RaLD topology:
        # dynamic = CrossAttn(Qd, P)
        # x = Proj(Qs + dynamic)
        # x += CrossAttn(x, P)
        # x += FFN(x)
        dynamic = self.dynamic_proposal_attention(
            dynamic_query, proposal_tokens
        )
        latent = self.mixed_projection(static + dynamic)
        latent = latent + self.post_mix_proposal_attention(
            latent, proposal_tokens
        )
        latent = latent + self.post_mix_feed_forward(latent)
        for block in self.latent_blocks:
            latent = block(latent, radar_tokens)
        return latent

    def decode_query_field(
        self,
        cube_drae: torch.Tensor,
        normalized_rae: torch.Tensor,
        latent: torch.Tensor,
        *,
        chunk_size: int | None = None,
        log_power_drae: torch.Tensor | None = None,
        energy_rae: torch.Tensor | None = None,
        validate_cube: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Decode arbitrary normalized RAE queries in bounded chunks."""

        if validate_cube:
            self._validate_cube(cube_drae)
        self._validate_batched_coordinates(
            normalized_rae,
            batch_size=cube_drae.shape[0],
            normalized=True,
        )
        if latent.shape != (
            cube_drae.shape[0],
            self.latent_count,
            self.model_dim,
        ):
            raise ValueError(
                f"Expected latent shape "
                f"{(cube_drae.shape[0], self.latent_count, self.model_dim)}, "
                f"got {latent.shape}"
            )
        chunk_size = self.decode_chunk_size if chunk_size is None else chunk_size
        if chunk_size <= 0:
            raise ValueError("Query decode chunk size must be positive")

        outputs: dict[str, list[torch.Tensor]] = {
            "query_features": [],
            "occupancy_logit": [],
            "confidence_logit": [],
            "offset_bins": [],
            "query_coordinates_rae": [],
            "query_cube_spectrum": [],
            "query_absolute_log_energy": [],
        }
        for start in range(0, normalized_rae.shape[1], chunk_size):
            stop = min(start + chunk_size, normalized_rae.shape[1])
            normalized_chunk = normalized_rae[:, start:stop]
            coordinate_chunk = self._denormalize_coordinates(normalized_chunk)
            evidence = self.query_tokens(
                cube_drae,
                coordinate_chunk,
                log_power_drae=log_power_drae,
                energy_rae=energy_rae,
                validate_cube=False,
            )
            query = evidence["tokens"]
            features = query + self.decoder_attention(query, latent)
            features = features + self.decoder_feed_forward(features)
            offset = torch.tanh(self.offset_head(features))
            offset = offset * self.offset_bounds_bins.to(offset)
            outputs["query_features"].append(features)
            outputs["occupancy_logit"].append(
                self.occupancy_head(features).squeeze(-1)
            )
            outputs["confidence_logit"].append(
                self.confidence_head(features).squeeze(-1)
            )
            outputs["offset_bins"].append(offset)
            outputs["query_coordinates_rae"].append(
                evidence["coordinates_rae"]
            )
            outputs["query_cube_spectrum"].append(
                evidence["local_spectrum"]
            )
            outputs["query_absolute_log_energy"].append(
                evidence["absolute_log_energy"]
            )

        decoded = {key: torch.cat(values, dim=1) for key, values in outputs.items()}
        decoded["query_normalized_rae"] = normalized_rae
        decoded["refined_coordinates_rae"] = self._clamp_coordinates(
            decoded["query_coordinates_rae"] + decoded["offset_bins"]
        )
        expected_prefix = normalized_rae.shape[:2]
        if decoded["occupancy_logit"].shape != expected_prefix:
            raise AssertionError("G1D occupancy decoder changed query cardinality")
        if decoded["confidence_logit"].shape != expected_prefix:
            raise AssertionError("G1D confidence decoder changed query cardinality")
        if decoded["offset_bins"].shape != (*expected_prefix, 3):
            raise AssertionError("G1D offset decoder changed query cardinality")
        return decoded

    @staticmethod
    def _gather_points(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if values.ndim < 2 or index.ndim != 2 or values.shape[0] != index.shape[0]:
            raise ValueError("Batched gather values and indices must align")
        expanded = index
        for _ in range(values.ndim - 2):
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand(*index.shape, *values.shape[2:])
        return values.gather(1, expanded)

    @staticmethod
    def _stable_top_indices(score: torch.Tensor, count: int) -> torch.Tensor:
        if score.ndim != 2:
            raise ValueError("Selection score must have shape (B,N)")
        if count <= 0 or count > score.shape[1]:
            raise ValueError("Stable top selection count is out of bounds")
        return torch.argsort(
            score, dim=1, descending=True, stable=True
        )[:, :count]

    def _xyz(self, coordinates_rae: torch.Tensor) -> torch.Tensor:
        xyz = continuous_rae_to_xyz(
            coordinates_rae.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        )
        return xyz.reshape(*coordinates_rae.shape[:-1], 3)

    def forward(
        self,
        cube_drae: torch.Tensor,
        occupancy_queries_rae: torch.Tensor | None = None,
        *,
        condition_cube_drae: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        self._validate_cube(cube_drae)
        if condition_cube_drae is None:
            condition_cube_drae = cube_drae
        else:
            self._validate_cube(condition_cube_drae)
            if condition_cube_drae.shape != cube_drae.shape:
                raise ValueError("Condition Cube must match the measured Cube shape")
        log_power_drae = torch.log1p(cube_drae.clamp_min(0.0))
        energy_rae = log_power_drae.sum(dim=1)
        proposals = stable_radar_proposals(
            cube_drae,
            seed_count=self.base_seed_count,
            nms_kernel=self.nms_kernel,
        )
        proposal_evidence = self.query_tokens(
            cube_drae,
            proposals.coordinates_rae,
            log_power_drae=log_power_drae,
            energy_rae=energy_rae,
            validate_cube=False,
        )
        proposal_tokens = proposal_evidence["tokens"]
        if proposal_tokens.shape != (
            cube_drae.shape[0],
            self.base_seed_count,
            self.model_dim,
        ):
            raise AssertionError("G1D proposal token count changed unexpectedly")

        radar_tokens = self.radar_encoder(condition_cube_drae)
        expected_radar_shape = (
            cube_drae.shape[0],
            self.expected_radar_token_count,
            self.model_dim,
        )
        if radar_tokens.shape != expected_radar_shape:
            raise RuntimeError(
                f"Full-RAED encoder returned {radar_tokens.shape}, expected "
                f"{expected_radar_shape}"
            )
        latent = self.encode_latent(proposal_tokens, radar_tokens)

        coarse = (
            proposals.coordinates_rae[:, :, None, :]
            + self.coarse_templates.to(cube_drae)[None, None, :, :]
        )
        coarse = self._clamp_coordinates(
            coarse.reshape(cube_drae.shape[0], self.coarse_query_count, 3)
        )
        if coarse.shape[1] != self.base_seed_count * self.COARSE_TEMPLATE_COUNT:
            raise AssertionError("G1D coarse pool must contain 32 queries per seed")
        coarse_fields = self.decode_query_field(
            cube_drae,
            self._normalize_coordinates(coarse),
            latent,
            log_power_drae=log_power_drae,
            energy_rae=energy_rae,
            validate_cube=False,
        )
        selection_score = (
            coarse_fields["occupancy_logit"]
            + coarse_fields["confidence_logit"]
        )
        selected_index = self._stable_top_indices(
            selection_score, self.selected_coarse_count
        )
        selected_centers = self._gather_points(
            coarse_fields["refined_coordinates_rae"], selected_index
        )
        selected_zero_offset_centers = self._gather_points(
            coarse_fields["query_coordinates_rae"], selected_index
        )
        selected_score = self._gather_points(selection_score, selected_index)

        local_queries = (
            selected_centers[:, :, None, :]
            + self.local_templates.to(cube_drae)[None, None, :, :]
        )
        local_queries = self._clamp_coordinates(
            local_queries.reshape(cube_drae.shape[0], self.point_count, 3)
        )
        if local_queries.shape[1] != (
            self.selected_coarse_count * self.LOCAL_TEMPLATE_COUNT
        ):
            raise AssertionError("G1D final pool must contain four points per selection")
        zero_offset_queries = (
            selected_zero_offset_centers[:, :, None, :]
            + self.local_templates.to(cube_drae)[None, None, :, :]
        )
        zero_offset_queries = self._clamp_coordinates(
            zero_offset_queries.reshape(
                cube_drae.shape[0], self.point_count, 3
            )
        )
        final_fields = self.decode_query_field(
            cube_drae,
            self._normalize_coordinates(local_queries),
            latent,
            log_power_drae=log_power_drae,
            energy_rae=energy_rae,
            validate_cube=False,
        )
        coordinates = final_fields["refined_coordinates_rae"]
        zero_offset_coordinates = zero_offset_queries
        point_spectrum = self._query_spectrum(
            cube_drae, coordinates, log_power_drae=log_power_drae
        )
        confidence = torch.sigmoid(final_fields["confidence_logit"])

        selected_seed_index = selected_index // self.COARSE_TEMPLATE_COUNT
        selected_template_index = selected_index % self.COARSE_TEMPLATE_COUNT
        selected_proposal_flat = proposals.flat_index.gather(
            1, selected_seed_index
        )
        provenance = {
            "selected_coarse_index": selected_index,
            "selected_seed_index": selected_seed_index,
            "selected_coarse_template_index": selected_template_index,
            "selected_proposal_flat_index": selected_proposal_flat,
            "selected_score": selected_score,
            "selected_center_coordinates_rae": selected_centers,
            "selected_zero_offset_center_coordinates_rae": (
                selected_zero_offset_centers
            ),
            "point_selected_coarse_index": torch.arange(
                self.selected_coarse_count, device=cube_drae.device
            ).repeat_interleave(self.LOCAL_TEMPLATE_COUNT),
            "point_tetrahedral_template_index": torch.arange(
                self.LOCAL_TEMPLATE_COUNT, device=cube_drae.device
            ).repeat(self.selected_coarse_count),
        }
        result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "occupancy_logit": final_fields["occupancy_logit"],
            "occupancy_probability": torch.sigmoid(
                final_fields["occupancy_logit"]
            ),
            "confidence_logit": final_fields["confidence_logit"],
            "confidence": confidence,
            "offset_bins": final_fields["offset_bins"],
            "query_features": final_fields["query_features"],
            "query_coordinates_rae": final_fields["query_coordinates_rae"],
            "coordinates_rae": coordinates,
            "xyz_m": self._xyz(coordinates),
            "zero_offset_coordinates_rae": zero_offset_coordinates,
            "zero_offset_xyz_m": self._xyz(zero_offset_coordinates),
            "point_cube_spectrum": point_spectrum,
            "latent": latent,
            "proposal_coordinates_rae": proposals.coordinates_rae,
            "proposal_flat_index": proposals.flat_index,
            "proposal_integrated_log_energy": proposals.integrated_log_energy,
            "coarse_selection_provenance": provenance,
            "radar_token_count": coordinates.new_tensor(
                self.expected_radar_token_count, dtype=torch.long
            ),
            "coarse_query_count": coordinates.new_tensor(
                self.coarse_query_count, dtype=torch.long
            ),
            "selected_coarse_count": coordinates.new_tensor(
                self.selected_coarse_count, dtype=torch.long
            ),
            "final_point_count": coordinates.new_tensor(
                self.point_count, dtype=torch.long
            ),
        }

        if occupancy_queries_rae is not None:
            self._validate_batched_coordinates(
                occupancy_queries_rae,
                batch_size=cube_drae.shape[0],
                normalized=True,
            )
            training = self.decode_query_field(
                cube_drae,
                occupancy_queries_rae,
                latent,
                log_power_drae=log_power_drae,
                energy_rae=energy_rae,
                validate_cube=False,
            )
            training_coordinates = training["refined_coordinates_rae"]
            result.update(
                {
                    "training_query_input_normalized_rae": occupancy_queries_rae,
                    "training_query_input_coordinates_rae": training[
                        "query_coordinates_rae"
                    ],
                    "training_query_occupancy_logit": training[
                        "occupancy_logit"
                    ],
                    "training_query_confidence_logit": training[
                        "confidence_logit"
                    ],
                    "training_query_confidence": torch.sigmoid(
                        training["confidence_logit"]
                    ),
                    "training_query_offset_bins": training["offset_bins"],
                    "training_query_features": training["query_features"],
                    "training_query_coordinates_rae": training_coordinates,
                    "training_query_xyz_m": self._xyz(training_coordinates),
                    "training_query_point_cube_spectrum": self._query_spectrum(
                        cube_drae,
                        training_coordinates,
                        log_power_drae=log_power_drae,
                    ),
                }
            )
        return result
