"""Condition-exclusive hierarchical point allocation for G1G Stage-0.

The center allocator can consume only learned queries, normalized coordinate
templates, and the 336 global Full-RAED tokens. Local Cube spectra are sampled
only after all 2,500 centers have been allocated. A shared patch decoder then
emits four children per center with offsets constrained to one half-bin per RAE
axis and, consequently, to the physical diagonal of the center's Cube cell.
"""

from __future__ import annotations

from itertools import product
import math
from typing import Any

import torch
import torch.nn as nn

from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.rald_matched import (
    FourierPointEmbedding,
    FullRAEDRadarTokenEncoder,
    PreNormAttention,
    PreNormFeedForward,
)


class GlobalCenterAllocationBlock(nn.Module):
    """Update center queries exclusively from global Full-RAED tokens."""

    def __init__(self, model_dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.radar_attention = PreNormAttention(
            model_dim,
            model_dim,
            heads=heads,
            head_dim=head_dim,
        )
        self.feed_forward = PreNormFeedForward(model_dim)

    def forward(
        self,
        center_queries: torch.Tensor,
        radar_tokens: torch.Tensor,
    ) -> torch.Tensor:
        center_queries = center_queries + self.radar_attention(
            center_queries,
            radar_tokens,
        )
        return center_queries + self.feed_forward(center_queries)


def _center_templates(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an interior 25 x 20 x 5 normalized RAE template grid."""

    axes = (
        torch.linspace(-0.8, 0.8, 25, device=device, dtype=dtype),
        torch.linspace(-0.8, 0.8, 20, device=device, dtype=dtype),
        torch.linspace(-0.8, 0.8, 5, device=device, dtype=dtype),
    )
    mesh = torch.meshgrid(*axes, indexing="ij")
    templates = torch.stack(mesh, dim=-1).reshape(-1, 3)
    if templates.shape != (2_500, 3):
        raise AssertionError("G1G center template grid must contain 2,500 rows")
    return templates


class G1GConditionExclusiveHierarchy(nn.Module):
    """Allocate 2,500 global centers, then refine four local children each."""

    RADAR_TOKEN_COUNT = 336
    CENTER_COUNT = 2_500
    CHILDREN_PER_CENTER = 4
    POINT_COUNT = CENTER_COUNT * CHILDREN_PER_CENTER
    SPECTRUM_BIN_COUNT = 64
    PRE_ALLOCATION_SOURCES = (
        "learned_center_queries",
        "normalized_center_templates",
        "global_full_raed_tokens",
    )
    FORBIDDEN_PRE_ALLOCATION_SOURCES = (
        "local_cube_spectrum",
        "local_cube_energy",
        "local_cube_neighborhood",
        "proposal_score",
    )
    STAGE_SEQUENCE = (
        "encode_global_full_raed_tokens",
        "allocate_2500_centers",
        "sample_local_center_spectrum",
        "decode_four_bounded_children",
        "sample_final_point_spectrum",
    )

    def __init__(
        self,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        *,
        log_center: float,
        log_scale: float,
        model_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        head_dim: int = 64,
        fourier_frequency_dim: int = 48,
        radar_base_channels: int = 64,
        radar_spectral_channels: int = 16,
        radar_encoded_shape: tuple[int, int, int] = (16, 7, 3),
        radar_encoded_channels: int = 16,
        radar_channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4),
        radar_blocks_per_level: int = 2,
        center_residual_bound_normalized: tuple[float, float, float] = (
            0.2,
            0.2,
            0.2,
        ),
        child_offset_bound_bins: tuple[float, float, float] = (
            0.5,
            0.5,
            0.5,
        ),
    ) -> None:
        super().__init__()
        axes = (range_m, azimuth_rad, elevation_rad)
        if any(axis.ndim != 1 or axis.numel() <= 1 for axis in axes):
            raise ValueError("G1G RAE axes must be one-dimensional and nontrivial")
        if model_dim <= 0 or depth <= 0 or heads <= 0 or head_dim <= 0:
            raise ValueError("G1G model dimensions and depth must be positive")
        if math.prod(radar_encoded_shape) != self.RADAR_TOKEN_COUNT:
            raise ValueError("G1G requires exactly 336 global radar tokens")
        if len(center_residual_bound_normalized) != 3 or any(
            not 0.0 < value <= 1.0
            for value in center_residual_bound_normalized
        ):
            raise ValueError(
                "Center residual bounds must contain three values in (0, 1]"
            )
        if len(child_offset_bound_bins) != 3 or any(
            not 0.0 < value <= 0.5 for value in child_offset_bound_bins
        ):
            raise ValueError(
                "Child offset bounds must contain three values in (0, 0.5]"
            )
        if not math.isfinite(log_center) or not math.isfinite(log_scale):
            raise ValueError("G1G Full-RAED normalization must be finite")
        if log_scale <= 0.0:
            raise ValueError("G1G Full-RAED normalization scale must be positive")

        self.model_dim = model_dim
        self.depth = depth
        self.radar_encoded_shape = radar_encoded_shape
        self.expected_radar_token_count = math.prod(radar_encoded_shape)

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
        self.center_queries = nn.Embedding(self.CENTER_COUNT, model_dim)
        self.center_coordinate_embedding = FourierPointEmbedding(
            model_dim,
            frequency_dim=fourier_frequency_dim,
        )
        self.allocation_blocks = nn.ModuleList(
            [
                GlobalCenterAllocationBlock(model_dim, heads, head_dim)
                for _ in range(depth)
            ]
        )
        self.center_coordinate_head = nn.Linear(model_dim, 3)
        self.center_score_head = nn.Linear(model_dim, 1)

        self.center_spectrum_projection = nn.Sequential(
            nn.Linear(self.SPECTRUM_BIN_COUNT, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.child_codes = nn.Embedding(self.CHILDREN_PER_CENTER, model_dim)
        self.patch_decoder = PreNormFeedForward(model_dim)
        self.child_offset_head = nn.Linear(model_dim, 3)
        self.child_confidence_head = nn.Linear(model_dim, 1)

        self.register_buffer("range_m", range_m.float(), persistent=True)
        self.register_buffer(
            "azimuth_rad",
            azimuth_rad.float(),
            persistent=True,
        )
        self.register_buffer(
            "elevation_rad",
            elevation_rad.float(),
            persistent=True,
        )
        self.register_buffer(
            "center_templates_normalized",
            _center_templates(),
            persistent=True,
        )
        self.register_buffer(
            "center_residual_bound_normalized",
            torch.tensor(center_residual_bound_normalized, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "child_offset_bound_bins",
            torch.tensor(child_offset_bound_bins, dtype=torch.float32),
            persistent=True,
        )

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return (
            self.range_m.numel(),
            self.azimuth_rad.numel(),
            self.elevation_rad.numel(),
        )

    @classmethod
    def architecture_metadata(cls) -> dict[str, Any]:
        """Return the static dependency and cardinality contract."""

        return {
            "global_radar_token_count": cls.RADAR_TOKEN_COUNT,
            "center_query_count": cls.CENTER_COUNT,
            "children_per_center": cls.CHILDREN_PER_CENTER,
            "final_point_count": cls.POINT_COUNT,
            "pre_allocation_sources": list(cls.PRE_ALLOCATION_SOURCES),
            "forbidden_pre_allocation_sources": list(
                cls.FORBIDDEN_PRE_ALLOCATION_SOURCES
            ),
            "local_cube_sampling_stage": "after_center_allocation",
            "child_offset_contract": (
                "per-axis half-cell bound plus one-cell physical diagonal"
            ),
            "stage_sequence": list(cls.STAGE_SEQUENCE),
        }

    def _validate_cube(self, cube_drae: torch.Tensor) -> None:
        expected = (self.SPECTRUM_BIN_COUNT, *self.spatial_shape)
        if cube_drae.ndim != 5 or tuple(cube_drae.shape[1:]) != expected:
            raise ValueError(
                f"Expected Full-RAED Cube (B,{','.join(map(str, expected))}), "
                f"got {tuple(cube_drae.shape)}"
            )
        if not torch.is_floating_point(cube_drae):
            raise TypeError("Full-RAED Cube must be floating point")
        if not torch.isfinite(cube_drae).all():
            raise ValueError("Full-RAED Cube must contain only finite values")

    def _normalize_coordinates(
        self,
        coordinates_rae: torch.Tensor,
    ) -> torch.Tensor:
        maximum = coordinates_rae.new_tensor(
            [size - 1 for size in self.spatial_shape]
        )
        return 2.0 * coordinates_rae / maximum - 1.0

    def _denormalize_coordinates(
        self,
        normalized_rae: torch.Tensor,
    ) -> torch.Tensor:
        maximum = normalized_rae.new_tensor(
            [size - 1 for size in self.spatial_shape]
        )
        return (normalized_rae + 1.0) * maximum / 2.0

    def _clamp_coordinates(
        self,
        coordinates_rae: torch.Tensor,
    ) -> torch.Tensor:
        maximum = coordinates_rae.new_tensor(
            [size - 1 for size in self.spatial_shape]
        )
        return coordinates_rae.clamp_min(0.0).minimum(maximum)

    def _xyz(self, coordinates_rae: torch.Tensor) -> torch.Tensor:
        xyz = continuous_rae_to_xyz(
            coordinates_rae.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        )
        return xyz.reshape(*coordinates_rae.shape[:-1], 3)

    @staticmethod
    def _batch_indices(
        batch_size: int,
        point_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.arange(batch_size, device=device)[:, None].expand(
            -1,
            point_count,
        )

    def _sample_local_spectrum(
        self,
        cube_drae: torch.Tensor,
        coordinates_rae: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Cube spectra; callers invoke this only after center allocation."""

        batch_size, point_count, coordinate_dim = coordinates_rae.shape
        if coordinate_dim != 3 or batch_size != cube_drae.shape[0]:
            raise ValueError("Local Cube queries must have shape (B,N,3)")
        batch = self._batch_indices(
            batch_size,
            point_count,
            coordinates_rae.device,
        )
        query = torch.cat(
            (
                batch.reshape(-1, 1).to(coordinates_rae),
                coordinates_rae.reshape(-1, 3),
            ),
            dim=1,
        )
        spectrum = query_cube_spectrum(cube_drae, query)
        return spectrum.reshape(
            batch_size,
            point_count,
            self.SPECTRUM_BIN_COUNT,
        )

    def _cell_diagonal_m(
        self,
        center_coordinates_rae: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the maximum physical corner-to-corner cell diagonal."""

        corner_offsets = center_coordinates_rae.new_tensor(
            list(product((-0.5, 0.5), repeat=3))
        )
        corners = center_coordinates_rae[:, :, None, :] + corner_offsets
        corners = self._clamp_coordinates(corners)
        corner_xyz = self._xyz(corners)
        pairwise = torch.linalg.vector_norm(
            corner_xyz[:, :, :, None, :] - corner_xyz[:, :, None, :, :],
            dim=-1,
        )
        return pairwise.amax(dim=(-1, -2), keepdim=False).unsqueeze(-1)

    def allocate_centers(
        self,
        condition_cube_drae: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Allocate centers from global Full-RAED condition and no local samples."""

        self._validate_cube(condition_cube_drae)
        radar_tokens = self.radar_encoder(condition_cube_drae)
        expected = (
            condition_cube_drae.shape[0],
            self.RADAR_TOKEN_COUNT,
            self.model_dim,
        )
        if radar_tokens.shape != expected:
            raise RuntimeError(
                f"Full-RAED encoder returned {radar_tokens.shape}, "
                f"expected {expected}"
            )

        batch_size = condition_cube_drae.shape[0]
        templates = self.center_templates_normalized.to(condition_cube_drae)
        normalized_templates = templates.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        learned_queries = self.center_queries.weight.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        features = learned_queries + self.center_coordinate_embedding(
            normalized_templates
        )
        layer_features = []
        for block in self.allocation_blocks:
            features = block(features, radar_tokens)
            layer_features.append(features)

        residual = torch.tanh(self.center_coordinate_head(features))
        residual = residual * self.center_residual_bound_normalized.to(residual)
        normalized_coordinates = (normalized_templates + residual).clamp(
            -1.0,
            1.0,
        )
        coordinates = self._denormalize_coordinates(normalized_coordinates)
        if coordinates.shape != (batch_size, self.CENTER_COUNT, 3):
            raise AssertionError("G1G center allocator changed center cardinality")
        return {
            "radar_tokens": radar_tokens,
            "normalized_center_templates": normalized_templates,
            "center_query_features": features,
            "center_layer_features": torch.stack(layer_features, dim=1),
            "center_residual_normalized": residual,
            "center_coordinates_normalized": normalized_coordinates,
            "center_coordinates_rae": coordinates,
            "center_score_logit": self.center_score_head(features).squeeze(-1),
        }

    def refine_children(
        self,
        cube_drae: torch.Tensor,
        allocation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample center spectra and decode exactly four bounded children."""

        self._validate_cube(cube_drae)
        centers = allocation["center_coordinates_rae"]
        center_features = allocation["center_query_features"]
        expected_center_shape = (
            cube_drae.shape[0],
            self.CENTER_COUNT,
            3,
        )
        expected_feature_shape = (
            cube_drae.shape[0],
            self.CENTER_COUNT,
            self.model_dim,
        )
        if centers.shape != expected_center_shape:
            raise ValueError("G1G allocation has the wrong center shape")
        if center_features.shape != expected_feature_shape:
            raise ValueError("G1G allocation has the wrong feature shape")

        center_spectrum = self._sample_local_spectrum(cube_drae, centers)
        local_features = self.center_spectrum_projection(center_spectrum)
        child_codes = self.child_codes.weight[None, None, :, :]
        patch_features = (
            center_features[:, :, None, :]
            + local_features[:, :, None, :]
            + child_codes
        )
        patch_features = patch_features + self.patch_decoder(patch_features)
        offset_bins = torch.tanh(self.child_offset_head(patch_features))
        offset_bins = offset_bins * self.child_offset_bound_bins.to(offset_bins)
        child_coordinates = self._clamp_coordinates(
            centers[:, :, None, :] + offset_bins
        )
        child_xyz = self._xyz(child_coordinates)
        center_xyz = self._xyz(centers)
        physical_offset = child_xyz - center_xyz[:, :, None, :]
        cell_diagonal = self._cell_diagonal_m(centers)

        flat_coordinates = child_coordinates.reshape(
            cube_drae.shape[0],
            self.POINT_COUNT,
            3,
        )
        flat_xyz = child_xyz.reshape(
            cube_drae.shape[0],
            self.POINT_COUNT,
            3,
        )
        point_spectrum = self._sample_local_spectrum(
            cube_drae,
            flat_coordinates,
        )
        confidence_logit = self.child_confidence_head(patch_features).squeeze(
            -1
        )
        parent_index = torch.arange(
            self.CENTER_COUNT,
            device=cube_drae.device,
        ).repeat_interleave(self.CHILDREN_PER_CENTER)
        child_index = torch.arange(
            self.CHILDREN_PER_CENTER,
            device=cube_drae.device,
        ).repeat(self.CENTER_COUNT)
        return {
            "center_cube_spectrum": center_spectrum,
            "center_local_features": local_features,
            "child_patch_features": patch_features,
            "child_offset_bins": offset_bins,
            "child_physical_offset_m": physical_offset,
            "center_cell_diagonal_m": cell_diagonal,
            "child_coordinates_rae": child_coordinates,
            "child_xyz_m": child_xyz,
            "coordinates_rae": flat_coordinates,
            "xyz_m": flat_xyz,
            "point_cube_spectrum": point_spectrum,
            "confidence_logit": confidence_logit.reshape(
                cube_drae.shape[0],
                self.POINT_COUNT,
            ),
            "confidence": torch.sigmoid(confidence_logit).reshape(
                cube_drae.shape[0],
                self.POINT_COUNT,
            ),
            "point_parent_center_index": parent_index,
            "point_child_index": child_index,
        }

    def forward(
        self,
        cube_drae: torch.Tensor,
        *,
        condition_cube_drae: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        self._validate_cube(cube_drae)
        if condition_cube_drae is None:
            condition_cube_drae = cube_drae
        else:
            self._validate_cube(condition_cube_drae)
            if condition_cube_drae.shape != cube_drae.shape:
                raise ValueError(
                    "Condition Cube must match the measured Cube shape"
                )

        allocation = self.allocate_centers(condition_cube_drae)
        refinement = self.refine_children(cube_drae, allocation)
        result: dict[str, Any] = {
            **allocation,
            **refinement,
            "architecture_metadata": self.architecture_metadata(),
            "radar_token_count": refinement["coordinates_rae"].new_tensor(
                self.RADAR_TOKEN_COUNT,
                dtype=torch.long,
            ),
            "center_count": refinement["coordinates_rae"].new_tensor(
                self.CENTER_COUNT,
                dtype=torch.long,
            ),
            "children_per_center": refinement["coordinates_rae"].new_tensor(
                self.CHILDREN_PER_CENTER,
                dtype=torch.long,
            ),
            "final_point_count": refinement["coordinates_rae"].new_tensor(
                self.POINT_COUNT,
                dtype=torch.long,
            ),
        }
        return result
