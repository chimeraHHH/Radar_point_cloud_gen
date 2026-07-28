"""Condition-exclusive RaLD continuous field for the R-A1 Stage-0 preflight.

The geometry decoder accepts only normalized RAE coordinates and condition
latents. Full-RAED values are consumed exclusively by the condition encoder;
local spectra, energy, proposal scores, and source labels are intentionally
absent before a later exact-cardinality selection stage.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn as nn

from models.rald_matched import (
    FourierPointEmbedding,
    FullRAEDRadarTokenEncoder,
    PreNormAttention,
    PreNormFeedForward,
)


class RaLDWCEConditionBlock(nn.Module):
    """Update latent queries with self-attention and Full-RAED cross-attention."""

    def __init__(self, model_dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.self_attention = PreNormAttention(
            model_dim,
            heads=heads,
            head_dim=head_dim,
        )
        self.radar_attention = PreNormAttention(
            model_dim,
            model_dim,
            heads=heads,
            head_dim=head_dim,
        )
        self.feed_forward = PreNormFeedForward(model_dim)

    def forward(
        self,
        latent: torch.Tensor,
        radar_tokens: torch.Tensor,
    ) -> torch.Tensor:
        latent = latent + self.self_attention(latent)
        latent = latent + self.radar_attention(latent, radar_tokens)
        return latent + self.feed_forward(latent)


class RaLDWCEField(nn.Module):
    """Wide condition-exclusive implicit field without EDM or Doppler heads."""

    SPECTRUM_BIN_COUNT = 64
    DEFAULT_DEPTH = 6
    FORBIDDEN_PRE_SELECTION_SOURCES = (
        "local_cube_spectrum",
        "local_cube_energy",
        "local_cube_neighborhood",
        "candidate_source",
        "proposal_score",
        "target_geometry",
    )

    def __init__(
        self,
        *,
        log_center: float,
        log_scale: float,
        spatial_shape: tuple[int, int, int] = (256, 107, 37),
        latent_count: int = 512,
        model_dim: int = 512,
        depth: int = DEFAULT_DEPTH,
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
        decode_chunk_size: int = 8_192,
    ) -> None:
        super().__init__()
        if (
            len(spatial_shape) != 3
            or any(int(size) <= 1 for size in spatial_shape)
        ):
            raise ValueError("RaLD-WCE spatial shape must contain three sizes > 1")
        if latent_count <= 0 or model_dim <= 0 or depth <= 0:
            raise ValueError("RaLD-WCE latent dimensions and depth must be positive")
        if heads <= 0 or head_dim <= 0:
            raise ValueError("RaLD-WCE attention dimensions must be positive")
        if fourier_frequency_dim <= 0 or fourier_frequency_dim % 6 != 0:
            raise ValueError(
                "RaLD-WCE Fourier frequency dimension must be positive and "
                "divisible by six"
            )
        if not math.isfinite(log_center) or not math.isfinite(log_scale):
            raise ValueError("RaLD-WCE Full-RAED normalization must be finite")
        if log_scale <= 0.0:
            raise ValueError("RaLD-WCE Full-RAED normalization scale must be positive")
        if decode_chunk_size <= 0:
            raise ValueError("RaLD-WCE decode chunk size must be positive")
        if (
            len(offset_bounds_bins) != 3
            or any(float(bound) <= 0.0 for bound in offset_bounds_bins)
        ):
            raise ValueError("RaLD-WCE offset bounds must contain three positives")

        self.spatial_shape = tuple(int(size) for size in spatial_shape)
        self.latent_count = int(latent_count)
        self.model_dim = int(model_dim)
        self.depth = int(depth)
        self.decode_chunk_size = int(decode_chunk_size)
        self.radar_encoded_shape = tuple(
            int(size) for size in radar_encoded_shape
        )
        self.expected_radar_token_count = math.prod(self.radar_encoded_shape)

        self.radar_encoder = FullRAEDRadarTokenEncoder(
            log_center=log_center,
            log_scale=log_scale,
            spectral_channels=radar_spectral_channels,
            encoded_shape=self.radar_encoded_shape,
            encoded_channels=radar_encoded_channels,
            token_dim=model_dim,
            base_channels=radar_base_channels,
            channel_multipliers=radar_channel_multipliers,
            blocks_per_level=radar_blocks_per_level,
        )
        self.condition_queries = nn.Embedding(latent_count, model_dim)
        self.condition_seed_attention = PreNormAttention(
            model_dim,
            model_dim,
            heads=heads,
            head_dim=head_dim,
        )
        self.condition_blocks = nn.ModuleList(
            [
                RaLDWCEConditionBlock(model_dim, heads, head_dim)
                for _ in range(depth)
            ]
        )

        self.coordinate_embedding = FourierPointEmbedding(
            model_dim,
            frequency_dim=fourier_frequency_dim,
        )
        self.decoder_attention = PreNormAttention(
            model_dim,
            model_dim,
            heads=heads,
            head_dim=head_dim,
        )
        self.decoder_feed_forward = PreNormFeedForward(model_dim)
        self.occupancy_head = nn.Linear(model_dim, 1)
        self.residual_head = nn.Linear(model_dim, 3)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self.register_buffer(
            "offset_bounds_bins",
            torch.tensor(offset_bounds_bins, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "normalization_denominator",
            torch.tensor(
                [size - 1 for size in self.spatial_shape],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.assert_condition_exclusive_contract()

    def architecture_metadata(self) -> dict[str, Any]:
        """Return machine-checkable Stage-0 architecture and boundary metadata."""

        return {
            "protocol": "rald_wce_condition_exclusive_field_stage0_v1",
            "model_family": "RaLD-WCE",
            "stage": "R-A1 Stage 0",
            "depth": self.depth,
            "latent_count": self.latent_count,
            "model_dim": self.model_dim,
            "global_radar_token_count": self.expected_radar_token_count,
            "geometry_decoder_inputs": [
                "normalized_rae_fourier_features",
                "condition_latents",
            ],
            "forbidden_pre_selection_sources": list(
                self.FORBIDDEN_PRE_SELECTION_SOURCES
            ),
            "pre_selection_local_cube_access": False,
            "candidate_coordinate_interface": "external_normalized_rae_only",
            "low_threshold_radar_query_path_implemented": False,
            "low_threshold_radar_role_if_added": "coordinates_only",
            "same_query_wrong_condition_supported": True,
            "condition_latents_are_only_cube_path": True,
            "confidence_source": "sigmoid_occupancy_logit",
            "geometry_stage_boundary": (
                "continuous_field_before_exact_10000_selection"
            ),
            "exact_10000_selection_implemented": False,
            "doppler_head_implemented": False,
            "edm_implemented": False,
        }

    def assert_condition_exclusive_contract(self) -> None:
        """Fail fast if the public geometry decoder gains a Cube-side bypass."""

        metadata = self.architecture_metadata()
        if metadata["geometry_decoder_inputs"] != [
            "normalized_rae_fourier_features",
            "condition_latents",
        ]:
            raise AssertionError("RaLD-WCE geometry decoder inputs changed")
        if metadata["pre_selection_local_cube_access"] is not False:
            raise AssertionError("RaLD-WCE local Cube access was enabled")
        if metadata["doppler_head_implemented"] is not False:
            raise AssertionError("RaLD-WCE Stage 0 must not contain a Doppler head")
        if metadata["edm_implemented"] is not False:
            raise AssertionError("RaLD-WCE Stage 0 must not contain EDM")

        signature = inspect.signature(type(self).decode_queries)
        allowed = {
            "self",
            "normalized_rae",
            "condition_latents",
            "chunk_size",
        }
        if set(signature.parameters) != allowed:
            raise AssertionError(
                "RaLD-WCE decoder signature admits an undeclared input source"
            )

    def _validate_cube(self, cube_drae: torch.Tensor) -> None:
        expected = (self.SPECTRUM_BIN_COUNT, *self.spatial_shape)
        if cube_drae.ndim != 5 or tuple(cube_drae.shape[1:]) != expected:
            raise ValueError(
                f"Expected Full-RAED Cube (B,{','.join(map(str, expected))}), "
                f"got {tuple(cube_drae.shape)}"
            )
        if not torch.is_floating_point(cube_drae):
            raise TypeError("RaLD-WCE Full-RAED Cube must be floating point")
        if not bool(torch.isfinite(cube_drae).all()):
            raise ValueError("RaLD-WCE Full-RAED Cube must be finite")

    @staticmethod
    def _validate_queries(
        normalized_rae: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        if normalized_rae.ndim != 3 or normalized_rae.shape[-1] != 3:
            raise ValueError("RaLD-WCE queries must have shape (B,N,3)")
        if normalized_rae.shape[0] != batch_size:
            raise ValueError("RaLD-WCE query and condition batches must match")
        if normalized_rae.shape[1] <= 0:
            raise ValueError("RaLD-WCE requires at least one geometry query")
        if not torch.is_floating_point(normalized_rae):
            raise TypeError("RaLD-WCE queries must be floating point")
        if not bool(torch.isfinite(normalized_rae).all()):
            raise ValueError("RaLD-WCE queries must be finite")
        if bool((normalized_rae.abs() > 1.0 + 1e-6).any()):
            raise ValueError("RaLD-WCE normalized queries must lie in [-1,1]")

    def encode_condition(
        self,
        cube_drae: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode a Full-RAED Cube into global condition latents."""

        self._validate_cube(cube_drae)
        radar_tokens = self.radar_encoder(cube_drae)
        expected = (
            cube_drae.shape[0],
            self.expected_radar_token_count,
            self.model_dim,
        )
        if tuple(radar_tokens.shape) != expected:
            raise RuntimeError(
                f"Full-RAED encoder returned {tuple(radar_tokens.shape)}, "
                f"expected {expected}"
            )
        latent = self.condition_queries.weight.unsqueeze(0).expand(
            cube_drae.shape[0],
            -1,
            -1,
        )
        latent = latent + self.condition_seed_attention(latent, radar_tokens)
        for block in self.condition_blocks:
            latent = block(latent, radar_tokens)
        return {
            "radar_tokens": radar_tokens,
            "condition_latents": latent,
        }

    def decode_queries(
        self,
        normalized_rae: torch.Tensor,
        condition_latents: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode arbitrary coordinates without any direct Cube-side input."""

        expected_latent = (
            normalized_rae.shape[0],
            self.latent_count,
            self.model_dim,
        )
        if tuple(condition_latents.shape) != expected_latent:
            raise ValueError(
                f"Expected RaLD-WCE condition latents {expected_latent}, "
                f"got {tuple(condition_latents.shape)}"
            )
        self._validate_queries(
            normalized_rae,
            batch_size=condition_latents.shape[0],
        )
        chunk_size = (
            self.decode_chunk_size if chunk_size is None else int(chunk_size)
        )
        if chunk_size <= 0:
            raise ValueError("RaLD-WCE decode chunk size must be positive")

        occupancy_parts: list[torch.Tensor] = []
        residual_parts: list[torch.Tensor] = []
        for start in range(0, normalized_rae.shape[1], chunk_size):
            stop = min(start + chunk_size, normalized_rae.shape[1])
            coordinate_features = self.coordinate_embedding(
                normalized_rae[:, start:stop]
            )
            features = coordinate_features + self.decoder_attention(
                coordinate_features,
                condition_latents,
            )
            features = features + self.decoder_feed_forward(features)
            occupancy_parts.append(
                self.occupancy_head(features).squeeze(-1)
            )
            residual_parts.append(
                torch.tanh(self.residual_head(features))
                * self.offset_bounds_bins.to(features)
            )

        occupancy_logit = torch.cat(occupancy_parts, dim=1)
        residual_bins = torch.cat(residual_parts, dim=1)
        residual_normalized = (
            2.0
            * residual_bins
            / self.normalization_denominator.to(residual_bins)
        )
        refined = (normalized_rae + residual_normalized).clamp(-1.0, 1.0)
        confidence = torch.sigmoid(occupancy_logit)
        expected_prefix = normalized_rae.shape[:2]
        if tuple(occupancy_logit.shape) != expected_prefix:
            raise AssertionError("RaLD-WCE occupancy changed query cardinality")
        if tuple(residual_bins.shape) != (*expected_prefix, 3):
            raise AssertionError("RaLD-WCE residual changed query cardinality")
        return {
            "query_normalized_rae": normalized_rae,
            "occupancy_logit": occupancy_logit,
            "confidence_logit": occupancy_logit,
            "confidence": confidence,
            "residual_bins": residual_bins,
            "residual_normalized": residual_normalized,
            "refined_normalized_rae": refined,
        }

    def forward(
        self,
        condition_cube_drae: torch.Tensor,
        normalized_rae: torch.Tensor,
        *,
        wrong_condition_cube_drae: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        """Decode matched and optional wrong conditions on the identical queries."""

        self.assert_condition_exclusive_contract()
        matched_condition = self.encode_condition(condition_cube_drae)
        matched = self.decode_queries(
            normalized_rae,
            matched_condition["condition_latents"],
            chunk_size=chunk_size,
        )
        output: dict[str, Any] = {
            "matched": matched,
            "matched_radar_tokens": matched_condition["radar_tokens"],
            "matched_condition_latents": matched_condition["condition_latents"],
            "architecture_metadata": self.architecture_metadata(),
        }
        if wrong_condition_cube_drae is not None:
            self._validate_cube(wrong_condition_cube_drae)
            if wrong_condition_cube_drae.shape != condition_cube_drae.shape:
                raise ValueError(
                    "RaLD-WCE wrong-condition Cube must match the condition Cube"
                )
            wrong_condition = self.encode_condition(wrong_condition_cube_drae)
            wrong = self.decode_queries(
                normalized_rae,
                wrong_condition["condition_latents"],
                chunk_size=chunk_size,
            )
            if (
                matched["query_normalized_rae"].data_ptr()
                != wrong["query_normalized_rae"].data_ptr()
            ):
                raise AssertionError(
                    "RaLD-WCE wrong-condition branch did not reuse the same queries"
                )
            output.update(
                {
                    "wrong": wrong,
                    "wrong_radar_tokens": wrong_condition["radar_tokens"],
                    "wrong_condition_latents": wrong_condition[
                        "condition_latents"
                    ],
                    "same_query_wrong_condition": True,
                }
            )
        return output
