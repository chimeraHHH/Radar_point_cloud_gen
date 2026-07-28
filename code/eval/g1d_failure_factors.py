"""Inference-only mechanisms for the frozen G1D failure-factor diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch

from eval.dense_geometry import aggregate_geometry_reports, geometry_report
from eval.rald_guided_query import duplicate_report
from models.rald_query_field import (
    RaLDQueryField,
    integrated_log_energy,
    proposals_from_flat_index,
    stable_radar_proposals,
)
from scripts.train_rald_query_field import aggregate_scalar_reports


OFFSET_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CURRENT_RANKING_SOURCE = "occupancy_plus_confidence"
RANKING_SOURCES = (
    CURRENT_RANKING_SOURCE,
    "occupancy_only",
    "confidence_only",
    "integrated_energy",
    "validation_gt_distance_oracle",
)
FAR_COMPLETENESS_METRIC = "range_60_120m_completeness_mean_distance_m"


@dataclass(frozen=True)
class PreparedQueryField:
    """Reusable G1D state before coarse ranking and residual refinement."""

    measured_cube_drae: torch.Tensor
    log_power_drae: torch.Tensor
    energy_rae: torch.Tensor
    latent: torch.Tensor
    proposal_flat_index: torch.Tensor
    coarse_fields: dict[str, torch.Tensor]


def alpha_label(alpha: float) -> str:
    if alpha not in OFFSET_ALPHAS:
        raise ValueError(f"Unregistered G1D residual alpha: {alpha}")
    return "alpha_" + str(alpha).replace(".", "p")


def tensor_sha256(value: torch.Tensor) -> str:
    original_dtype = str(value.dtype)
    canonical = value.detach().cpu().contiguous()
    if canonical.dtype == torch.bfloat16:
        canonical = canonical.float()
    array = canonical.numpy()
    digest = hashlib.sha256()
    digest.update(original_dtype.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_query_field(
    model: RaLDQueryField,
    measured_cube_drae: torch.Tensor,
    condition_cube_drae: torch.Tensor,
    *,
    proposal_flat_index: torch.Tensor | None = None,
) -> PreparedQueryField:
    """Reproduce the checkpoint-compatible G1D path through the 32k pool."""

    model._validate_cube(measured_cube_drae)
    model._validate_cube(condition_cube_drae)
    if measured_cube_drae.shape != condition_cube_drae.shape:
        raise ValueError("Measured and global-condition Cubes must have equal shape")

    log_power_drae = torch.log1p(measured_cube_drae.clamp_min(0.0))
    energy_rae = log_power_drae.sum(dim=1)
    if proposal_flat_index is None:
        proposals = stable_radar_proposals(
            measured_cube_drae,
            seed_count=model.base_seed_count,
            nms_kernel=model.nms_kernel,
        )
    else:
        expected = (measured_cube_drae.shape[0], model.base_seed_count)
        if proposal_flat_index.shape != expected:
            raise ValueError(
                f"Proposal index shape {proposal_flat_index.shape} differs from "
                f"the model contract {expected}"
            )
        proposals = proposals_from_flat_index(
            measured_cube_drae,
            proposal_flat_index.to(device=measured_cube_drae.device),
        )

    proposal_evidence = model.query_tokens(
        measured_cube_drae,
        proposals.coordinates_rae,
        log_power_drae=log_power_drae,
        energy_rae=energy_rae,
        validate_cube=False,
    )
    radar_tokens = model.radar_encoder(condition_cube_drae)
    expected_radar_shape = (
        measured_cube_drae.shape[0],
        model.expected_radar_token_count,
        model.model_dim,
    )
    if radar_tokens.shape != expected_radar_shape:
        raise RuntimeError(
            f"Full-RAED encoder returned {radar_tokens.shape}, expected "
            f"{expected_radar_shape}"
        )
    latent = model.encode_latent(proposal_evidence["tokens"], radar_tokens)

    coarse = (
        proposals.coordinates_rae[:, :, None, :]
        + model.coarse_templates.to(measured_cube_drae)[None, None, :, :]
    )
    coarse = model._clamp_coordinates(
        coarse.reshape(
            measured_cube_drae.shape[0],
            model.coarse_query_count,
            3,
        )
    )
    decoded = model.decode_query_field(
        measured_cube_drae,
        model._normalize_coordinates(coarse),
        latent,
        log_power_drae=log_power_drae,
        energy_rae=energy_rae,
        validate_cube=False,
    )
    retained_keys = (
        "occupancy_logit",
        "confidence_logit",
        "offset_bins",
        "query_coordinates_rae",
        "refined_coordinates_rae",
        "query_absolute_log_energy",
    )
    coarse_fields = {key: decoded[key] for key in retained_keys}
    if coarse_fields["occupancy_logit"].shape != (
        measured_cube_drae.shape[0],
        model.coarse_query_count,
    ):
        raise AssertionError("G1D diagnostic did not preserve the 32k coarse pool")
    return PreparedQueryField(
        measured_cube_drae=measured_cube_drae,
        log_power_drae=log_power_drae,
        energy_rae=energy_rae,
        latent=latent,
        proposal_flat_index=proposals.flat_index,
        coarse_fields=coarse_fields,
    )


def ranking_capability(
    model: RaLDQueryField,
    coarse_fields: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    required_attributes = (
        "occupancy_head",
        "confidence_head",
        "selected_coarse_count",
        "coarse_query_count",
    )
    missing_attributes = [
        name for name in required_attributes if not hasattr(model, name)
    ]
    required_fields = (
        "occupancy_logit",
        "confidence_logit",
        "query_absolute_log_energy",
        "refined_coordinates_rae",
    )
    missing_fields = (
        []
        if coarse_fields is None
        else [name for name in required_fields if name not in coarse_fields]
    )
    supported = not missing_attributes and not missing_fields
    return {
        "status": "supported" if supported else "unsupported",
        "supported": supported,
        "missing_model_attributes": missing_attributes,
        "missing_coarse_fields": missing_fields,
        "available_sources": list(RANKING_SOURCES) if supported else [],
        "architecture_evidence": {
            "seed_source": "measured_cube_integrated_energy_with_stable_nms",
            "current_coarse_ranking": "occupancy_logit_plus_confidence_logit",
            "same_pool_size": (
                int(model.coarse_query_count)
                if hasattr(model, "coarse_query_count")
                else None
            ),
            "selection_count": (
                int(model.selected_coarse_count)
                if hasattr(model, "selected_coarse_count")
                else None
            ),
        },
    }


def _negative_nearest_target_distance(
    candidate_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    if candidate_xyz.ndim != 3 or candidate_xyz.shape[0] != 1:
        raise ValueError("D3 oracle expects one batched candidate set")
    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError("D3 oracle target must have shape (N,3)")
    if target_xyz.shape[0] == 0:
        raise ValueError("D3 oracle requires a nonempty validation target")
    if chunk_size <= 0:
        raise ValueError("D3 oracle chunk size must be positive")
    target_xyz = target_xyz.float()
    scores = []
    candidates = candidate_xyz[0].float()
    for start in range(0, candidates.shape[0], chunk_size):
        distance = torch.cdist(
            candidates[start : start + chunk_size],
            target_xyz,
        )
        scores.append(-distance.amin(dim=1))
    return torch.cat(scores).unsqueeze(0)


def ranking_scores(
    model: RaLDQueryField,
    prepared: PreparedQueryField,
    *,
    validation_target_xyz: torch.Tensor,
) -> dict[str, torch.Tensor]:
    capability = ranking_capability(model, prepared.coarse_fields)
    if not capability["supported"]:
        raise RuntimeError(f"D3 ranking intervention unsupported: {capability}")
    fields = prepared.coarse_fields
    occupancy = fields["occupancy_logit"]
    confidence = fields["confidence_logit"]
    energy = fields["query_absolute_log_energy"].squeeze(-1)
    candidate_xyz = model._xyz(fields["refined_coordinates_rae"])
    scores = {
        CURRENT_RANKING_SOURCE: occupancy + confidence,
        "occupancy_only": occupancy,
        "confidence_only": confidence,
        "integrated_energy": energy,
        "validation_gt_distance_oracle": _negative_nearest_target_distance(
            candidate_xyz,
            validation_target_xyz,
        ),
    }
    expected = (occupancy.shape[0], model.coarse_query_count)
    for name, score in scores.items():
        if score.shape != expected or not torch.isfinite(score).all():
            raise RuntimeError(
                f"D3 ranking source {name} returned invalid shape/values"
            )
    return scores


def current_ranking_score(prepared: PreparedQueryField) -> torch.Tensor:
    fields = prepared.coarse_fields
    return fields["occupancy_logit"] + fields["confidence_logit"]


def generate_selected_points(
    model: RaLDQueryField,
    prepared: PreparedQueryField,
    selection_score: torch.Tensor,
    *,
    residual_alpha: float,
) -> dict[str, Any]:
    """Scale both coarse and final residuals while freezing the top-k score."""

    if residual_alpha < 0.0 or residual_alpha > 1.0:
        raise ValueError("Residual alpha must lie in [0,1]")
    expected_score = (
        prepared.measured_cube_drae.shape[0],
        model.coarse_query_count,
    )
    if selection_score.shape != expected_score:
        raise ValueError(
            f"Selection score shape {selection_score.shape} differs from "
            f"{expected_score}"
        )
    if not torch.isfinite(selection_score).all():
        raise ValueError("Selection score must be finite")

    fields = prepared.coarse_fields
    selected_index = model._stable_top_indices(
        selection_score,
        model.selected_coarse_count,
    )
    coarse_query = model._gather_points(
        fields["query_coordinates_rae"],
        selected_index,
    )
    coarse_offset = model._gather_points(
        fields["offset_bins"],
        selected_index,
    )
    selected_centers = model._clamp_coordinates(
        coarse_query + float(residual_alpha) * coarse_offset
    )
    local_queries = (
        selected_centers[:, :, None, :]
        + model.local_templates.to(selected_centers)[None, None, :, :]
    )
    local_queries = model._clamp_coordinates(
        local_queries.reshape(
            prepared.measured_cube_drae.shape[0],
            model.point_count,
            3,
        )
    )

    final_offset_abs_mean_bins = None
    if residual_alpha == 0.0:
        coordinates = local_queries
    else:
        final_fields = model.decode_query_field(
            prepared.measured_cube_drae,
            model._normalize_coordinates(local_queries),
            prepared.latent,
            log_power_drae=prepared.log_power_drae,
            energy_rae=prepared.energy_rae,
            validate_cube=False,
        )
        coordinates = model._clamp_coordinates(
            final_fields["query_coordinates_rae"]
            + float(residual_alpha) * final_fields["offset_bins"]
        )
        final_offset_abs_mean_bins = float(
            final_fields["offset_bins"].float().abs().mean().item()
        )
    xyz_m = model._xyz(coordinates)
    if xyz_m.shape != (
        prepared.measured_cube_drae.shape[0],
        model.point_count,
        3,
    ):
        raise AssertionError("G1D diagnostic changed the final point count")
    return {
        "xyz_m": xyz_m,
        "coordinates_rae": coordinates,
        "selected_index": selected_index,
        "selected_index_sha256": tensor_sha256(selected_index),
        "selected_score_sha256": tensor_sha256(
            model._gather_points(selection_score, selected_index)
        ),
        "coarse_offset_abs_mean_bins": float(
            coarse_offset.float().abs().mean().item()
        ),
        "final_offset_abs_mean_bins": final_offset_abs_mean_bins,
        "residual_alpha": float(residual_alpha),
    }


def evaluate_generated(
    generated: dict[str, Any],
    target_xyz_confidence: torch.Tensor,
    *,
    include_duplicates: bool,
) -> dict[str, Any]:
    target = target_xyz_confidence.float()
    xyz_m = generated["xyz_m"][0].float()
    report = {
        "geometry": geometry_report(
            xyz_m,
            target[:, :3],
            target_weight=target[:, 3],
        ),
        "selected_index_sha256": generated["selected_index_sha256"],
        "selected_score_sha256": generated["selected_score_sha256"],
        "coarse_offset_abs_mean_bins": generated[
            "coarse_offset_abs_mean_bins"
        ],
        "final_offset_abs_mean_bins": generated[
            "final_offset_abs_mean_bins"
        ],
        "residual_alpha": generated["residual_alpha"],
    }
    if include_duplicates:
        report["duplicates"] = duplicate_report(xyz_m)
    return report


def aggregate_arm(
    frames: list[dict[str, Any]],
    *,
    expected_frame_count: int,
    expected_far_count: int,
    expected_point_count: int,
) -> dict[str, Any]:
    if len(frames) != expected_frame_count:
        raise ValueError(
            f"Arm has {len(frames)} frames, expected {expected_frame_count}"
        )
    identities = [
        (int(frame["sequence"]), int(frame["radar_index"])) for frame in frames
    ]
    if len(set(identities)) != expected_frame_count:
        raise ValueError("Arm frame identities are incomplete or duplicated")
    if any(
        int(frame["geometry"]["prediction_count"]) != expected_point_count
        for frame in frames
    ):
        raise ValueError("Arm did not produce the exact frozen point count")

    geometry = aggregate_geometry_reports(
        [frame["geometry"] for frame in frames]
    )
    far = geometry.get(FAR_COMPLETENESS_METRIC, {})
    if far.get("sample_count") != expected_far_count:
        raise ValueError(
            "Corrected far completeness did not cover all far-target frames"
        )
    duplicate_frames = [
        frame["duplicates"] for frame in frames if "duplicates" in frame
    ]
    if duplicate_frames and len(duplicate_frames) != expected_frame_count:
        raise ValueError("Duplicate metrics are partially populated")
    return {
        "frame_count": len(frames),
        "geometry": geometry,
        "duplicates": (
            aggregate_scalar_reports(duplicate_frames)
            if duplicate_frames
            else None
        ),
        "frames": frames,
    }


def _endpoint(frame: dict[str, Any], group: str, metric: str) -> float:
    value = frame.get(group, {}).get(metric)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric endpoint {group}.{metric}")
    return float(value)


def paired_scene_bootstrap(
    reference_frames: list[dict[str, Any]],
    candidate_frames: list[dict[str, Any]],
    *,
    group: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Return scene-first paired candidate-minus-reference intervals."""

    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    reference = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in reference_frames
    }
    candidate = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in candidate_frames
    }
    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("Paired bootstrap arms have different frame identities")

    by_scene: dict[int, list[tuple[float, float]]] = {}
    for identity in sorted(reference):
        sequence = identity[0]
        by_scene.setdefault(sequence, []).append(
            (
                _endpoint(reference[identity], group, metric),
                _endpoint(candidate[identity], group, metric),
            )
        )
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    boot_delta = np.empty(samples, dtype=np.float64)
    boot_relative = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        pairs = [
            pair
            for scene in sampled_scenes
            for pair in by_scene[int(scene)]
        ]
        ref = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        cand = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        boot_delta[index] = float(np.mean(cand - ref))
        boot_relative[index] = float(
            np.mean(cand - ref) / max(abs(float(np.mean(ref))), 1e-12)
        )

    all_pairs = [pair for scene in scenes for pair in by_scene[scene]]
    reference_values = np.asarray(
        [pair[0] for pair in all_pairs], dtype=np.float64
    )
    candidate_values = np.asarray(
        [pair[1] for pair in all_pairs], dtype=np.float64
    )
    mean_delta = float(np.mean(candidate_values - reference_values))
    relative_change = float(
        mean_delta / max(abs(float(np.mean(reference_values))), 1e-12)
    )
    return {
        "direction": "candidate_minus_reference",
        "scene_count": len(scenes),
        "frame_count": len(all_pairs),
        "reference_mean": float(np.mean(reference_values)),
        "candidate_mean": float(np.mean(candidate_values)),
        "mean_delta": mean_delta,
        "delta_ci95": np.quantile(boot_delta, (0.025, 0.975)).tolist(),
        "relative_change": relative_change,
        "relative_change_ci95": np.quantile(
            boot_relative,
            (0.025, 0.975),
        ).tolist(),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }
