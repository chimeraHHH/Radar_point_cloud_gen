"""Deterministic K-Radar query support for a latent-only RaLD decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.rald_anchor import normalize_rae_coordinates
from models.rald_query_field import (
    coarse_query_templates,
    stable_radar_proposals,
    tetrahedral_local_templates,
)


@dataclass(frozen=True)
class RaLDProposalSupport:
    proposal_flat_index: torch.Tensor
    coarse_coordinates_rae: torch.Tensor
    coarse_occupancy_logit: torch.Tensor
    selected_coarse_index: torch.Tensor
    selected_coarse_coordinates_rae: torch.Tensor
    final_coordinates_rae: torch.Tensor


def _clamp_coordinates(
    coordinates_rae: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    maximum = coordinates_rae.new_tensor([size - 1 for size in spatial_shape])
    return coordinates_rae.clamp_min(0.0).minimum(maximum)


def _gather_points(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2 or index.ndim != 2 or values.shape[0] != index.shape[0]:
        raise ValueError("Batched proposal values and indices must align")
    expanded = index
    for _ in range(values.ndim - 2):
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand(*index.shape, *values.shape[2:])
    return values.gather(1, expanded)


def _decode_logits(
    autoencoder,
    prepared_latent: torch.Tensor,
    normalized_queries: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("RaLD proposal decode chunk size must be positive")
    logits = []
    for start in range(0, normalized_queries.shape[1], chunk_size):
        stop = min(start + chunk_size, normalized_queries.shape[1])
        logits.append(
            autoencoder.decode_queries(
                prepared_latent,
                normalized_queries[:, start:stop],
            )
        )
    result = torch.cat(logits, dim=1)
    if result.shape != normalized_queries.shape[:2]:
        raise RuntimeError("RaLD proposal decoder changed query cardinality")
    return result


def latent_only_proposal_support(
    autoencoder,
    latent: torch.Tensor,
    cube_drae: torch.Tensor,
    *,
    base_seed_count: int = 1_000,
    selected_coarse_count: int = 2_500,
    nms_kernel: tuple[int, int, int] = (5, 5, 3),
    decode_chunk_size: int = 8_192,
) -> RaLDProposalSupport:
    """Rank deterministic Cube proposals with a coordinate-only latent decoder.

    The Cube is used only to select the non-learned proposal support. Decoder
    calls receive the prepared latent and normalized coordinates, never local
    Cube values or proposal energies.
    """

    if cube_drae.ndim != 5 or cube_drae.shape[1] != 64:
        raise ValueError(
            f"Expected Full-RAED Cube (B,64,R,A,E), got {cube_drae.shape}"
        )
    if latent.ndim != 3 or latent.shape[0] != cube_drae.shape[0]:
        raise ValueError("RaLD latent and Cube batches must align")
    if base_seed_count <= 0 or selected_coarse_count <= 0:
        raise ValueError("Proposal and selection counts must be positive")

    spatial_shape = tuple(int(size) for size in cube_drae.shape[2:])
    proposals = stable_radar_proposals(
        cube_drae,
        seed_count=base_seed_count,
        nms_kernel=nms_kernel,
    )
    coarse_templates = coarse_query_templates(
        device=cube_drae.device,
        dtype=cube_drae.dtype,
    )
    coarse = proposals.coordinates_rae[:, :, None, :] + coarse_templates[
        None, None
    ]
    coarse = _clamp_coordinates(
        coarse.reshape(cube_drae.shape[0], -1, 3),
        spatial_shape,
    )
    expected_coarse_count = base_seed_count * coarse_templates.shape[0]
    if coarse.shape[1] != expected_coarse_count:
        raise RuntimeError("RaLD proposal support changed coarse cardinality")
    if selected_coarse_count > coarse.shape[1]:
        raise ValueError("Selected proposal count exceeds the coarse support")

    normalized = normalize_rae_coordinates(coarse, spatial_shape)
    prepared_latent = autoencoder.prepare_decoder_latent(latent)
    logits = _decode_logits(
        autoencoder,
        prepared_latent,
        normalized,
        decode_chunk_size,
    ).float()
    selected_index = torch.argsort(
        logits,
        dim=1,
        descending=True,
        stable=True,
    )[:, :selected_coarse_count]
    selected = _gather_points(coarse, selected_index)

    local_templates = tetrahedral_local_templates(
        device=cube_drae.device,
        dtype=cube_drae.dtype,
    )
    final = selected[:, :, None, :] + local_templates[None, None]
    final = _clamp_coordinates(
        final.reshape(cube_drae.shape[0], -1, 3),
        spatial_shape,
    )
    expected_final_count = selected_coarse_count * local_templates.shape[0]
    if final.shape[1] != expected_final_count:
        raise RuntimeError("RaLD proposal support changed final cardinality")

    return RaLDProposalSupport(
        proposal_flat_index=proposals.flat_index,
        coarse_coordinates_rae=coarse,
        coarse_occupancy_logit=logits,
        selected_coarse_index=selected_index,
        selected_coarse_coordinates_rae=selected,
        final_coordinates_rae=final,
    )
