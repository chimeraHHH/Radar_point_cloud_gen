"""Physical VAE refinement on anchors from a frozen passing G3R model."""

from __future__ import annotations

from itertools import chain
from typing import Iterator

import torch
import torch.nn as nn

from models.cube_cycle import continuous_rae_to_xyz
from models.cube_doppler import query_cube_spectrum
from models.rald_anchor import FrozenParentRaLDRefiner, normalize_rae_coordinates
from models.rald_anchor_ldm import RaLDAnchorLDM
from models.rald_matched import RaLDPhysicalQueryHead


class RaLDAnchorLDMRefiner(nn.Module):
    """Decode a physical point-state VAE only on frozen G3R anchors.

    The selected G3R model remains the geometry parent. Its final continuous
    coordinates, query features, and confidence initialize the G3L support.
    G3L predicts one bounded residual offset and then re-queries the measured
    Cube at the resulting continuous position before predicting Doppler and
    confidence residuals.
    """

    def __init__(
        self,
        geometry_parent: FrozenParentRaLDRefiner,
        ldm: RaLDAnchorLDM,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
        *,
        model_dim: int = 512,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("G3L model dimension must be positive")
        if range_m.ndim != 1 or azimuth_rad.ndim != 1 or elevation_rad.ndim != 1:
            raise ValueError("G3L axes must be one-dimensional")
        if min(range_m.numel(), azimuth_rad.numel(), elevation_rad.numel()) <= 1:
            raise ValueError("G3L axes must contain at least two bins")

        self.geometry_parent = geometry_parent
        self.ldm = ldm
        self.model_dim = model_dim
        self.anchor_feature_dim = int(
            self.ldm.decoder.anchor_feature_projection[1].in_features
        )
        self.physical_head = RaLDPhysicalQueryHead(
            query_dim=model_dim,
            spectrum_bins=64,
            hidden_dim=model_dim,
            doppler_head_mode="distribution",
        )
        self.register_buffer("range_m", range_m.float(), persistent=True)
        self.register_buffer("azimuth_rad", azimuth_rad.float(), persistent=True)
        self.register_buffer("elevation_rad", elevation_rad.float(), persistent=True)

        for parameter in self.geometry_parent.parameters():
            parameter.requires_grad_(False)
        for parameter in self.ldm.edm.parameters():
            parameter.requires_grad_(False)
        self.geometry_parent.eval()
        self.ldm.edm.eval()

    def train(self, mode: bool = True) -> RaLDAnchorLDMRefiner:
        super().train(mode)
        self.geometry_parent.eval()
        self.ldm.edm.eval()
        return self

    def vae_parameters(self) -> Iterator[nn.Parameter]:
        """Return G3L-1 parameters, excluding the G3R parent and G3L-2 EDM."""

        return chain(
            self.ldm.posterior_encoder.parameters(),
            self.ldm.decoder.parameters(),
            self.physical_head.parameters(),
        )

    def vae_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "posterior_encoder": self.ldm.posterior_encoder.state_dict(),
            "anchor_decoder": self.ldm.decoder.state_dict(),
            "physical_head": self.physical_head.state_dict(),
        }

    def load_vae_state_dict(
        self, state: dict[str, dict[str, torch.Tensor]], *, strict: bool = True
    ) -> None:
        expected = {"posterior_encoder", "anchor_decoder", "physical_head"}
        if set(state) != expected:
            raise ValueError(
                f"G3L VAE state keys differ: expected {sorted(expected)}, "
                f"received {sorted(state)}"
            )
        self.ldm.posterior_encoder.load_state_dict(
            state["posterior_encoder"], strict=strict
        )
        self.ldm.decoder.load_state_dict(state["anchor_decoder"], strict=strict)
        self.physical_head.load_state_dict(state["physical_head"], strict=strict)

    @staticmethod
    def _batched_cube_query(
        cube_drae: torch.Tensor, coordinates_rae: torch.Tensor
    ) -> torch.Tensor:
        if coordinates_rae.ndim != 3 or coordinates_rae.shape[-1] != 3:
            raise ValueError("Batched Cube query coordinates must have shape (B,N,3)")
        if coordinates_rae.shape[0] != cube_drae.shape[0]:
            raise ValueError("Cube and query batch sizes differ")
        batch_size, point_count, _ = coordinates_rae.shape
        batch = torch.arange(batch_size, device=coordinates_rae.device)
        batch = batch[:, None].expand(-1, point_count)
        query = torch.cat(
            (
                batch.reshape(-1, 1).to(coordinates_rae),
                coordinates_rae.reshape(-1, 3),
            ),
            dim=1,
        )
        return query_cube_spectrum(cube_drae, query).reshape(
            batch_size, point_count, 64
        )

    @staticmethod
    def _batched_target_state(
        cube_drae: torch.Tensor,
        target_rae_index: torch.Tensor,
        target_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if target_rae_index.ndim == 2:
            if cube_drae.shape[0] != 1:
                raise ValueError("Unbatched target RAE requires a one-Cube batch")
            target_rae_index = target_rae_index.unsqueeze(0)
        if target_rae_index.ndim != 3 or target_rae_index.shape[-1] != 3:
            raise ValueError("Target RAE indices must have shape (N,3) or (B,N,3)")
        if target_rae_index.shape[0] != cube_drae.shape[0]:
            raise ValueError("Cube and target point-state batch sizes differ")
        if target_confidence.ndim == 1:
            target_confidence = target_confidence.unsqueeze(0)
        if target_confidence.shape == (*target_rae_index.shape[:2], 1):
            target_confidence = target_confidence.squeeze(-1)
        if target_confidence.shape != target_rae_index.shape[:2]:
            raise ValueError("Target confidence must align with target RAE points")

        coordinates = target_rae_index.to(
            device=cube_drae.device, dtype=cube_drae.dtype
        )
        probability = RaLDAnchorLDMRefiner._batched_cube_query(
            cube_drae, coordinates
        )
        normalized = normalize_rae_coordinates(
            coordinates, tuple(int(size) for size in cube_drae.shape[2:])
        )
        confidence = target_confidence.to(normalized).clamp(0.0, 1.0)
        return normalized, probability, confidence

    def _validate_cube_shape(self, cube_drae: torch.Tensor) -> None:
        expected = (
            self.range_m.numel(),
            self.azimuth_rad.numel(),
            self.elevation_rad.numel(),
        )
        if cube_drae.ndim != 5 or cube_drae.shape[1] != 64:
            raise ValueError(f"Expected Full-RAED Cube (B,64,R,A,E), got {cube_drae.shape}")
        if tuple(cube_drae.shape[2:]) != expected:
            raise ValueError(
                f"Cube spatial shape {tuple(cube_drae.shape[2:])} differs from "
                f"registered G3L axes {expected}"
            )

    def _parent_support(self, cube_drae: torch.Tensor) -> dict[str, object]:
        with torch.no_grad():
            parent = self.geometry_parent(cube_drae)
        parent_coordinates = parent["coordinates_rae"].detach()
        parent_features = parent["anchor_features"].detach()
        if parent_features.shape[-1] != self.anchor_feature_dim:
            raise ValueError(
                f"Frozen-parent feature dimension {parent_features.shape[-1]} "
                f"differs from G3L anchor feature dimension "
                f"{self.anchor_feature_dim}"
            )
        return {
            "parent": parent,
            "coordinates": parent_coordinates,
            "features": parent_features,
            "normalized": normalize_rae_coordinates(
                parent_coordinates,
                tuple(int(size) for size in cube_drae.shape[2:]),
            ),
        }

    def _physical_decode(
        self,
        cube_drae: torch.Tensor,
        support: dict[str, object],
        query_features: torch.Tensor,
        latent: torch.Tensor,
        posterior: object | None,
    ) -> dict[str, torch.Tensor | object]:
        parent = support["parent"]
        parent_coordinates = support["coordinates"]
        parent_features = support["features"]
        parent_normalized = support["normalized"]
        if not isinstance(parent, dict):
            raise TypeError("G3L parent support must contain the parent output")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (parent_coordinates, parent_features, parent_normalized)
        ):
            raise TypeError("G3L parent support tensors are invalid")

        anchor_spectrum = self._batched_cube_query(cube_drae, parent_coordinates)
        offset = self.physical_head(query_features, anchor_spectrum)["offset_bins"]
        coordinates = parent_coordinates.to(offset) + offset
        final_spectrum = self._batched_cube_query(cube_drae, coordinates)
        physical = self.physical_head.physical_attributes(
            query_features, final_spectrum
        )

        batch_size, point_count, _ = coordinates.shape
        xyz = continuous_rae_to_xyz(
            coordinates.reshape(-1, 3),
            self.range_m,
            self.azimuth_rad,
            self.elevation_rad,
        ).reshape(batch_size, point_count, 3)
        parent_xyz = parent.get("xyz_m")
        if parent_xyz is None:
            parent_xyz = continuous_rae_to_xyz(
                parent_coordinates.reshape(-1, 3),
                self.range_m,
                self.azimuth_rad,
                self.elevation_rad,
            ).reshape(batch_size, point_count, 3)
        parent_logit = parent["confidence_logit"].detach().to(
            physical["confidence_logit"]
        )
        confidence_logit = parent_logit + physical["confidence_logit"]
        parent_confidence = parent["confidence"].detach().to(confidence_logit)
        return {
            **physical,
            "probability": physical["doppler_probability"],
            "coordinates_rae": coordinates,
            "xyz_m": xyz,
            "offset_bins": offset,
            "confidence_residual_logit": physical["confidence_logit"],
            "confidence_logit": confidence_logit,
            "confidence": torch.sigmoid(confidence_logit),
            "anchor_indices_rae": parent_coordinates,
            "anchor_normalized_rae": parent_normalized,
            "anchor_xyz_m": parent_xyz.detach(),
            "anchor_features": parent_features,
            "anchor_parent_logits": parent_logit,
            "anchor_parent_confidence": parent_confidence,
            "anchor_cube_spectrum": anchor_spectrum,
            "point_cube_spectrum": final_spectrum,
            "latent": latent,
            "query_features": query_features,
            "posterior": posterior,
        }

    def forward(
        self,
        cube_drae: torch.Tensor,
        target_rae_index: torch.Tensor,
        target_confidence: torch.Tensor,
        *,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor | object]:
        self._validate_cube_shape(cube_drae)
        target_rae, target_probability, target_confidence = (
            self._batched_target_state(
                cube_drae, target_rae_index, target_confidence
            )
        )

        support = self._parent_support(cube_drae)
        parent_normalized = support["normalized"]
        parent_features = support["features"]
        if not isinstance(parent_normalized, torch.Tensor) or not isinstance(
            parent_features, torch.Tensor
        ):
            raise TypeError("G3L parent support tensors are invalid")

        if sample_posterior:
            posterior = self.ldm.posterior_encoder(
                target_rae, target_probability, target_confidence
            )
            latent = posterior.sample(generator=generator)
            query_features = self.ldm.decoder(
                latent, parent_normalized, parent_features
            )
        else:
            decoded = self.ldm.posterior_mean_path(
                target_rae,
                target_probability,
                target_confidence,
                parent_normalized,
                parent_features,
            )
            posterior = decoded.posterior
            latent = decoded.latent
            query_features = decoded.query_features
        if posterior is None:
            raise RuntimeError("G3L-1 requires a target-conditioned posterior")

        decoded = self._physical_decode(
            cube_drae, support, query_features, latent, posterior
        )
        return {
            **decoded,
            "target_normalized_rae": target_rae,
            "target_doppler_probability": target_probability,
            "target_confidence": target_confidence,
            "posterior_mean": posterior.mean,
            "posterior_log_variance": posterior.log_variance,
            "posterior_kl": posterior.kl(),
        }

    @torch.inference_mode()
    def sample_edm(
        self,
        cube_drae: torch.Tensor,
        seeds: list[int],
        *,
        condition_cube_drae: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> dict[str, torch.Tensor | object]:
        """Sample RaLD while preserving anchors and final measured physics.

        ``condition_cube_drae`` is used only by the preregistered condition-
        shuffle control. Parent anchors and final-position spectrum queries
        always use ``cube_drae``, isolating the EDM condition path.
        """

        self._validate_cube_shape(cube_drae)
        if len(seeds) != cube_drae.shape[0]:
            raise ValueError("G3L EDM requires exactly one fixed seed per frame")
        if condition_cube_drae is None:
            condition_cube_drae = cube_drae
        self._validate_cube_shape(condition_cube_drae)
        if condition_cube_drae.shape != cube_drae.shape:
            raise ValueError("G3L shuffled condition must match the measured Cube")

        support = self._parent_support(cube_drae)
        parent_normalized = support["normalized"]
        parent_features = support["features"]
        if not isinstance(parent_normalized, torch.Tensor) or not isinstance(
            parent_features, torch.Tensor
        ):
            raise TypeError("G3L parent support tensors are invalid")
        sampled = self.ldm.sampled_edm_path(
            condition_cube_drae,
            parent_normalized,
            parent_features,
            seeds,
            steps=steps,
        )
        output = self._physical_decode(
            cube_drae,
            support,
            sampled.query_features,
            sampled.latent,
            posterior=None,
        )
        output["sample_seeds"] = torch.as_tensor(
            seeds, dtype=torch.int64, device=cube_drae.device
        )
        output["condition_is_measured_cube"] = condition_cube_drae is cube_drae
        return output
