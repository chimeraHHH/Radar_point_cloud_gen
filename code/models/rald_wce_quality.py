"""Independent geometry-quality ranking head for frozen RaLD-WCE candidates."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn as nn

from models.rald_matched import (
    FourierPointEmbedding,
    PreNormAttention,
    PreNormFeedForward,
)


class RaLDWCEQualityHead(nn.Module):
    """Predict candidate geometry quality without changing the frozen field.

    The head consumes only a refined candidate coordinate, the frozen formal
    condition latent, and the detached formal occupancy confidence. Its output
    is a correction to the frozen occupancy logit, which makes update zero an
    exact ranking control while allowing the new head to replace the export
    score after training.
    """

    FORBIDDEN_INFERENCE_INPUTS = (
        "target_xyz_confidence",
        "target_geometry",
        "test_frame",
        "future_frame",
        "doppler_target",
        "best_of_k_sample",
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
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or model_dim <= 0:
            raise ValueError("Q1 quality dimensions must be positive")
        if heads <= 0 or head_dim <= 0:
            raise ValueError("Q1 attention dimensions must be positive")
        if fourier_frequency_dim <= 0 or fourier_frequency_dim % 6 != 0:
            raise ValueError(
                "Q1 Fourier frequency dimension must be positive and "
                "divisible by six"
            )
        if decode_chunk_size <= 0:
            raise ValueError("Q1 quality decode chunk size must be positive")
        if not 0.0 < confidence_epsilon < 0.5:
            raise ValueError("Q1 confidence epsilon must lie in (0,0.5)")

        self.condition_dim = int(condition_dim)
        self.model_dim = int(model_dim)
        self.decode_chunk_size = int(decode_chunk_size)
        self.confidence_epsilon = float(confidence_epsilon)
        self.coordinate_embedding = FourierPointEmbedding(
            model_dim,
            frequency_dim=fourier_frequency_dim,
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
        self.quality_delta_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1),
        )
        final = self.quality_delta_head[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("Q1 quality head final layer changed")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.assert_ranking_only_contract()

    def architecture_metadata(self) -> dict[str, Any]:
        """Return the machine-checkable Q1 model boundary."""

        return {
            "protocol": "g1_q1_rald_wce_quality_tiny_v1",
            "model_family": "RaLD-WCE frozen field plus independent quality head",
            "condition_dim": self.condition_dim,
            "model_dim": self.model_dim,
            "quality_inputs": [
                "refined_normalized_rae",
                "frozen_formal_condition_latents",
                "detached_formal_occupancy_confidence",
            ],
            "quality_output": "sigmoid_quality_logit",
            "initial_ranking": "bit_exact_formal_occupancy_ranking",
            "ranking_score_after_training": "quality_score_only",
            "candidate_coordinates_changed": False,
            "formal_residual_changed": False,
            "q0_q1_changed": False,
            "range_quotas_changed": False,
            "minimum_distance_changed": False,
            "ground_truth_inference": False,
            "test_access": False,
            "doppler_head": False,
            "best_of_k": False,
            "forbidden_inference_inputs": list(self.FORBIDDEN_INFERENCE_INPUTS),
        }

    def assert_ranking_only_contract(self) -> None:
        metadata = self.architecture_metadata()
        locked_false = (
            "candidate_coordinates_changed",
            "formal_residual_changed",
            "q0_q1_changed",
            "range_quotas_changed",
            "minimum_distance_changed",
            "ground_truth_inference",
            "test_access",
            "doppler_head",
            "best_of_k",
        )
        if any(metadata[key] is not False for key in locked_false):
            raise AssertionError("Q1 quality head escaped the ranking-only boundary")
        signature = inspect.signature(type(self).forward)
        allowed = {
            "self",
            "refined_normalized_rae",
            "condition_latents",
            "base_confidence",
            "chunk_size",
        }
        if set(signature.parameters) != allowed:
            raise AssertionError("Q1 quality head admits an undeclared input")

    def _validate_inputs(
        self,
        refined_normalized_rae: torch.Tensor,
        condition_latents: torch.Tensor,
        base_confidence: torch.Tensor,
    ) -> None:
        if refined_normalized_rae.ndim != 3 or (
            refined_normalized_rae.shape[-1] != 3
        ):
            raise ValueError("Q1 refined candidates must have shape (B,N,3)")
        batch_size, candidate_count, _ = refined_normalized_rae.shape
        if candidate_count <= 0:
            raise ValueError("Q1 requires at least one candidate")
        if condition_latents.ndim != 3 or (
            condition_latents.shape[0] != batch_size
            or condition_latents.shape[2] != self.condition_dim
        ):
            raise ValueError(
                "Q1 condition latents must have shape "
                f"(B,L,{self.condition_dim})"
            )
        if condition_latents.shape[1] <= 0:
            raise ValueError("Q1 condition latent set cannot be empty")
        if base_confidence.shape != (batch_size, candidate_count):
            raise ValueError("Q1 base confidence must align with candidates")
        for value, name in (
            (refined_normalized_rae, "refined candidates"),
            (condition_latents, "condition latents"),
            (base_confidence, "base confidence"),
        ):
            if not torch.is_floating_point(value):
                raise TypeError(f"Q1 {name} must be floating point")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Q1 {name} must be finite")
        if bool((refined_normalized_rae.abs() > 1.0 + 1e-6).any()):
            raise ValueError("Q1 refined candidates must lie in [-1,1]")
        if bool(((base_confidence < 0.0) | (base_confidence > 1.0)).any()):
            raise ValueError("Q1 base confidence must lie in [0,1]")

    def forward(
        self,
        refined_normalized_rae: torch.Tensor,
        condition_latents: torch.Tensor,
        base_confidence: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score one deterministic candidate set; no target input is accepted."""

        self.assert_ranking_only_contract()
        self._validate_inputs(
            refined_normalized_rae,
            condition_latents,
            base_confidence,
        )
        chunk_size = self.decode_chunk_size if chunk_size is None else int(
            chunk_size
        )
        if chunk_size <= 0:
            raise ValueError("Q1 quality decode chunk size must be positive")

        projected_condition = self.condition_projection(
            condition_latents.detach()
        )
        delta_parts: list[torch.Tensor] = []
        for start in range(0, refined_normalized_rae.shape[1], chunk_size):
            stop = min(start + chunk_size, refined_normalized_rae.shape[1])
            features = self.coordinate_embedding(
                refined_normalized_rae[:, start:stop].detach()
            )
            features = features + self.condition_attention(
                features,
                projected_condition,
            )
            features = features + self.feed_forward(features)
            delta_parts.append(
                self.quality_delta_head(features).squeeze(-1)
            )
        quality_delta_logit = torch.cat(delta_parts, dim=1)
        base = base_confidence.detach().float()
        base_logit = (
            torch.log(base + self.confidence_epsilon)
            - torch.log1p(-base + self.confidence_epsilon)
        )
        quality_logit = base_logit + quality_delta_logit.float()
        quality = torch.sigmoid(quality_logit)
        expected = refined_normalized_rae.shape[:2]
        if tuple(quality.shape) != expected:
            raise AssertionError("Q1 quality head changed candidate cardinality")
        if not bool(torch.isfinite(quality).all()):
            raise FloatingPointError("Q1 quality score became non-finite")
        return {
            "quality_logit": quality_logit,
            "quality": quality,
            "quality_delta_logit": quality_delta_logit,
            "base_confidence": base_confidence.detach(),
            "refined_normalized_rae": refined_normalized_rae.detach(),
        }


def quality_parameter_count(model: nn.Module) -> int:
    count = sum(parameter.numel() for parameter in model.parameters())
    if count <= 0 or not math.isfinite(float(count)):
        raise ValueError("Q1 quality head has no finite parameter count")
    return count
