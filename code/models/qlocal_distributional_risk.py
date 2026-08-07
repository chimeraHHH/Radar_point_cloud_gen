"""Local Full-RAED distributional geometric-risk scoring for Q-Local-F0."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn as nn

from models.cube_doppler import query_cube_spectrum
from models.point_to_cube import trilinear_query_features
from models.rald_matched import (
    FourierPointEmbedding,
    PreNormAttention,
    PreNormFeedForward,
)
from models.rald_query_field import integrated_log_energy


DISTANCE_CENTERS_M = (0.05, 0.175, 0.375, 0.75, 1.5, 3.0)
LOCAL_SPECTRUM_BINS = 64
LOCAL_AUXILIARY_DIM = 5


def _validate_cube_coordinates(
    cube_drae: torch.Tensor,
    coordinates_rae: torch.Tensor,
) -> None:
    if cube_drae.ndim != 5 or cube_drae.shape[1] != LOCAL_SPECTRUM_BINS:
        raise ValueError(
            "Q-Local expects Cube shape (B,64,R,A,E), got "
            f"{tuple(cube_drae.shape)}"
        )
    if coordinates_rae.ndim != 3 or coordinates_rae.shape[0] != cube_drae.shape[0]:
        raise ValueError("Q-Local coordinates must have shape (B,N,3)")
    if coordinates_rae.shape[-1] != 3 or coordinates_rae.shape[1] <= 0:
        raise ValueError("Q-Local requires a nonempty RAE candidate set")
    for value, name in (
        (cube_drae, "Cube"),
        (coordinates_rae, "coordinates"),
    ):
        if not torch.is_floating_point(value):
            raise TypeError(f"Q-Local {name} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Q-Local {name} must be finite")
    maximum = coordinates_rae.new_tensor(
        [size - 1 for size in cube_drae.shape[2:]]
    )
    if bool((coordinates_rae < 0.0).any()) or bool(
        (coordinates_rae > maximum + 1e-5).any()
    ):
        raise ValueError("Q-Local coordinates escaped the Cube domain")


@torch.no_grad()
def extract_local_full_raed_features(
    cube_drae: torch.Tensor,
    coordinates_rae: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Query local spectrum, standardized energy, entropy, and RAE gradients."""

    _validate_cube_coordinates(cube_drae, coordinates_rae)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Q-Local feature epsilon must be positive")

    batch_size, point_count, _ = coordinates_rae.shape
    batch = torch.arange(
        batch_size,
        device=coordinates_rae.device,
        dtype=coordinates_rae.dtype,
    )[:, None].expand(-1, point_count)
    query = torch.cat(
        (batch.reshape(-1, 1), coordinates_rae.reshape(-1, 3)),
        dim=1,
    )
    spectrum = query_cube_spectrum(cube_drae, query).reshape(
        batch_size, point_count, LOCAL_SPECTRUM_BINS
    )
    entropy = -(
        spectrum * torch.log(spectrum.clamp_min(epsilon))
    ).sum(dim=-1, keepdim=True) / math.log(LOCAL_SPECTRUM_BINS)

    energy = integrated_log_energy(cube_drae)
    flat = energy.flatten(start_dim=1)
    center = flat.mean(dim=1).view(batch_size, 1, 1, 1)
    scale = flat.std(dim=1, unbiased=False).clamp_min(epsilon).view(
        batch_size, 1, 1, 1
    )
    normalized_energy = ((energy - center) / scale).clamp(-8.0, 8.0)
    gradients = torch.gradient(normalized_energy, dim=(1, 2, 3))
    feature_grid = torch.stack((normalized_energy, *gradients), dim=1)
    sampled = trilinear_query_features(
        feature_grid,
        coordinates_rae.reshape(-1, 3),
        batch.long().reshape(-1),
    ).reshape(batch_size, point_count, 4)

    result = {
        "local_spectrum": spectrum,
        "normalized_log_energy": sampled[..., :1],
        "spectrum_entropy": entropy,
        "energy_gradients": sampled[..., 1:],
    }
    for name, value in result.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Q-Local {name} became non-finite")
    return result


class QLocalDistributionalRiskScorer(nn.Module):
    """Predict a six-bin candidate-to-surface distance distribution."""

    FORBIDDEN_INFERENCE_INPUTS = (
        "target_xyz_confidence",
        "target_geometry",
        "future_frame",
        "test_frame",
        "selected_candidate_rows",
        "best_of_k",
    )

    def __init__(
        self,
        *,
        condition_dim: int = 512,
        model_dim: int = 128,
        heads: int = 4,
        head_dim: int = 32,
        fourier_frequency_dim: int = 48,
        decode_chunk_size: int = 8_192,
        confidence_epsilon: float = 1e-6,
        base_prior_strength: float = 0.15,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or model_dim <= 0:
            raise ValueError("Q-Local dimensions must be positive")
        if heads <= 0 or head_dim <= 0:
            raise ValueError("Q-Local attention dimensions must be positive")
        if fourier_frequency_dim <= 0 or fourier_frequency_dim % 6 != 0:
            raise ValueError(
                "Q-Local Fourier frequency dimension must be divisible by six"
            )
        if decode_chunk_size <= 0:
            raise ValueError("Q-Local decode chunk size must be positive")
        if not 0.0 < confidence_epsilon < 0.5:
            raise ValueError("Q-Local confidence epsilon must lie in (0,0.5)")
        if not math.isfinite(base_prior_strength) or base_prior_strength <= 0.0:
            raise ValueError("Q-Local base prior strength must be positive")

        self.condition_dim = int(condition_dim)
        self.model_dim = int(model_dim)
        self.decode_chunk_size = int(decode_chunk_size)
        self.confidence_epsilon = float(confidence_epsilon)
        self.base_prior_strength = float(base_prior_strength)
        self.coordinate_embedding = FourierPointEmbedding(
            model_dim,
            frequency_dim=fourier_frequency_dim,
        )
        local_state_dim = LOCAL_SPECTRUM_BINS + LOCAL_AUXILIARY_DIM + 1
        self.local_projection = nn.Sequential(
            nn.LayerNorm(local_state_dim),
            nn.Linear(local_state_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, model_dim),
        )
        self.condition_attention = PreNormAttention(
            model_dim,
            model_dim,
            heads=heads,
            head_dim=head_dim,
        )
        self.feed_forward = PreNormFeedForward(model_dim)
        self.distance_delta_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, len(DISTANCE_CENTERS_M)),
        )
        final = self.distance_delta_head[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("Q-Local final distance layer changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer(
            "distance_centers_m",
            torch.tensor(DISTANCE_CENTERS_M, dtype=torch.float32),
            persistent=True,
        )
        self.assert_inference_contract()

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "protocol": "g1_qlocal_distributional_risk_f0_v1",
            "model_family": "Fresh-WCE frozen field plus local Full-RAED risk",
            "condition_dim": self.condition_dim,
            "model_dim": self.model_dim,
            "distance_centers_m": list(DISTANCE_CENTERS_M),
            "inference_inputs": [
                "refined_normalized_rae",
                "detached_fresh_parent_confidence",
                "detached_global_condition_latents",
                "local_normalized_64bin_spectrum",
                "local_standardized_integrated_log_energy",
                "local_spectrum_entropy",
                "local_rae_energy_gradients",
            ],
            "selection_score": "negative_expected_distance_m",
            "candidate_coordinates_changed": False,
            "fresh_parent_changed": False,
            "range_quotas_changed": False,
            "minimum_distance_changed": False,
            "ground_truth_inference": False,
            "test_access": False,
            "future_access": False,
            "best_of_k": False,
            "forbidden_inference_inputs": list(self.FORBIDDEN_INFERENCE_INPUTS),
        }

    def assert_inference_contract(self) -> None:
        metadata = self.architecture_metadata()
        locked_false = (
            "candidate_coordinates_changed",
            "fresh_parent_changed",
            "range_quotas_changed",
            "minimum_distance_changed",
            "ground_truth_inference",
            "test_access",
            "future_access",
            "best_of_k",
        )
        if any(metadata[key] is not False for key in locked_false):
            raise AssertionError("Q-Local escaped its frozen scoring boundary")
        allowed = {
            "self",
            "refined_normalized_rae",
            "base_confidence",
            "condition_latents",
            "local_spectrum",
            "normalized_log_energy",
            "spectrum_entropy",
            "energy_gradients",
            "chunk_size",
        }
        signature = inspect.signature(type(self).forward)
        if set(signature.parameters) != allowed:
            raise AssertionError("Q-Local scorer admits an undeclared input")

    def _validate_inputs(
        self,
        refined_normalized_rae: torch.Tensor,
        base_confidence: torch.Tensor,
        condition_latents: torch.Tensor,
        local_spectrum: torch.Tensor,
        normalized_log_energy: torch.Tensor,
        spectrum_entropy: torch.Tensor,
        energy_gradients: torch.Tensor,
    ) -> None:
        if refined_normalized_rae.ndim != 3 or refined_normalized_rae.shape[-1] != 3:
            raise ValueError("Q-Local normalized RAE must have shape (B,N,3)")
        batch_size, point_count, _ = refined_normalized_rae.shape
        expected_scalar = (batch_size, point_count, 1)
        if base_confidence.shape != (batch_size, point_count):
            raise ValueError("Q-Local base confidence must align with candidates")
        if condition_latents.ndim != 3 or condition_latents.shape[0] != batch_size:
            raise ValueError("Q-Local condition latents must be a batched set")
        if condition_latents.shape[1] <= 0 or condition_latents.shape[2] != self.condition_dim:
            raise ValueError("Q-Local condition latent shape changed")
        if local_spectrum.shape != (
            batch_size,
            point_count,
            LOCAL_SPECTRUM_BINS,
        ):
            raise ValueError("Q-Local local spectrum shape changed")
        if normalized_log_energy.shape != expected_scalar:
            raise ValueError("Q-Local local energy shape changed")
        if spectrum_entropy.shape != expected_scalar:
            raise ValueError("Q-Local spectrum entropy shape changed")
        if energy_gradients.shape != (batch_size, point_count, 3):
            raise ValueError("Q-Local energy-gradient shape changed")
        values = (
            refined_normalized_rae,
            base_confidence,
            condition_latents,
            local_spectrum,
            normalized_log_energy,
            spectrum_entropy,
            energy_gradients,
        )
        if any(not torch.is_floating_point(value) for value in values):
            raise TypeError("Q-Local inference tensors must be floating point")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("Q-Local inference tensors must be finite")
        if bool((refined_normalized_rae.abs() > 1.0 + 1e-6).any()):
            raise ValueError("Q-Local normalized RAE escaped [-1,1]")
        if bool(((base_confidence < 0.0) | (base_confidence > 1.0)).any()):
            raise ValueError("Q-Local base confidence escaped [0,1]")
        spectrum_mass = local_spectrum.sum(dim=-1)
        if not torch.allclose(
            spectrum_mass,
            torch.ones_like(spectrum_mass),
            atol=2e-4,
            rtol=2e-4,
        ):
            raise ValueError("Q-Local local spectra are not normalized")

    def forward(
        self,
        refined_normalized_rae: torch.Tensor,
        base_confidence: torch.Tensor,
        condition_latents: torch.Tensor,
        local_spectrum: torch.Tensor,
        normalized_log_energy: torch.Tensor,
        spectrum_entropy: torch.Tensor,
        energy_gradients: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score unchanged candidates without accepting target geometry."""

        self.assert_inference_contract()
        self._validate_inputs(
            refined_normalized_rae,
            base_confidence,
            condition_latents,
            local_spectrum,
            normalized_log_energy,
            spectrum_entropy,
            energy_gradients,
        )
        chunk_size = self.decode_chunk_size if chunk_size is None else int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("Q-Local decode chunk size must be positive")

        base = base_confidence.detach().float().clamp(
            self.confidence_epsilon,
            1.0 - self.confidence_epsilon,
        )
        base_logit = torch.logit(base)
        projected_condition = self.condition_projection(condition_latents.detach())
        centers = self.distance_centers_m.float()
        scaled_centers = centers / centers[-1]
        logits_parts: list[torch.Tensor] = []
        for start in range(0, refined_normalized_rae.shape[1], chunk_size):
            stop = min(start + chunk_size, refined_normalized_rae.shape[1])
            local_state = torch.cat(
                (
                    local_spectrum[:, start:stop].detach(),
                    normalized_log_energy[:, start:stop].detach(),
                    spectrum_entropy[:, start:stop].detach(),
                    energy_gradients[:, start:stop].detach(),
                    base_logit[:, start:stop, None],
                ),
                dim=-1,
            )
            features = self.coordinate_embedding(
                refined_normalized_rae[:, start:stop].detach()
            )
            features = features + self.local_projection(local_state)
            features = features + self.condition_attention(
                features,
                projected_condition,
            )
            features = features + self.feed_forward(features)
            delta = self.distance_delta_head(features).float()
            prior = (
                -self.base_prior_strength
                * base_logit[:, start:stop, None]
                * scaled_centers.view(1, 1, -1)
            )
            logits_parts.append(prior + delta)

        distance_logits = torch.cat(logits_parts, dim=1)
        distance_probability = torch.softmax(distance_logits, dim=-1)
        expected_distance_m = (
            distance_probability * centers.view(1, 1, -1)
        ).sum(dim=-1)
        selection_score = -expected_distance_m
        expected = refined_normalized_rae.shape[:2]
        if selection_score.shape != expected:
            raise AssertionError("Q-Local changed candidate cardinality")
        if not bool(torch.isfinite(selection_score).all()):
            raise FloatingPointError("Q-Local score became non-finite")
        return {
            "distance_logits": distance_logits,
            "distance_probability": distance_probability,
            "expected_distance_m": expected_distance_m,
            "selection_score": selection_score,
            "base_confidence": base_confidence.detach(),
            "refined_normalized_rae": refined_normalized_rae.detach(),
        }


def qlocal_parameter_count(model: nn.Module) -> int:
    count = sum(parameter.numel() for parameter in model.parameters())
    if count <= 0 or count > 2_000_000:
        raise ValueError(f"Q-Local trainable parameter count is invalid: {count}")
    return count
