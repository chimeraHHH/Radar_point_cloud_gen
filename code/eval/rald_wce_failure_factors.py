"""Frozen-candidate failure diagnostics for formal R-A1/WCE checkpoints.

Ground truth is introduced only after the matched Q0+Q1 candidate field and
the current-confidence exact export have been reconstructed. The GT-nearest
arm is unattainable and diagnostic; it is not a method result or strict upper
bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from eval.dense_geometry import aggregate_geometry_reports, geometry_report
from eval.g1a_wide_support import streaming_bidirectional_nearest
from eval.rald_wce_stage0 import (
    CAPACITY_DISTANCE_M,
    EXPORT_COUNT,
    FORMAL_OUTPUT_RANGE_QUOTAS,
    ExactWCEExport,
    WideInferenceConfig,
    _candidate_tensors,
    exact_capacity_export,
    fixed_wide_q0,
    occupancy_dependent_q1,
    range_stratum_codes,
    tensor_sha256,
)
from models.rald_wce_field import RaLDWCEField


RANGE_STRATA_M = (
    ("range_0_30m", 0.0, 30.0),
    ("range_30_60m", 30.0, 60.0),
    ("range_60_120m", 60.0, 120.0),
)
SUPPORT_THRESHOLDS_M = (0.5, 1.0, 2.0)
QUANTILE_LEVELS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
TOP_CONFIDENCE_FRACTIONS = (0.01, 0.05, 0.10, 0.25)
FAR_COMPLETENESS_METRIC = "range_60_120m_completeness_mean_distance_m"


@dataclass(frozen=True)
class WCECandidatePool:
    """Matched-condition candidate field before exact-10k ranking."""

    xyz_m: torch.Tensor
    confidence: torch.Tensor
    current_confidence_export: ExactWCEExport
    query_hashes: dict[str, str]
    report: dict[str, Any]


@dataclass(frozen=True)
class WCEFailureFactorFrame:
    """One frame of frozen-pool diagnostics and exact exports."""

    current_geometry: dict[str, float]
    validation_gt_nearest_score_geometry: dict[str, float]
    candidate_to_target_distance_m: torch.Tensor
    target_to_candidate_distance_m: torch.Tensor
    support: dict[str, Any]
    ranking: dict[str, Any]
    validation_gt_nearest_score_export: ExactWCEExport


def diagnostic_artifact_label() -> dict[str, bool | str]:
    return {
        "label": "ra1_wce_frozen_candidate_failure_factor_diagnostic",
        "diagnostic": True,
        "ground_truth_used_for_candidate_generation": False,
        "ground_truth_used_for_current_confidence_control": False,
        "ground_truth_used_for_validation_gt_nearest_score": True,
        "validation_gt_nearest_score_unattainable": True,
        "validation_gt_nearest_score_strict_upper_bound": False,
        "coverage_aware_oracle_implemented": False,
        "eligible_as_method_result": False,
        "eligible_for_checkpoint_modification": False,
    }


@torch.no_grad()
def reconstruct_matched_candidate_pool(
    model: RaLDWCEField,
    matched_cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    config: WideInferenceConfig,
) -> WCECandidatePool:
    """Rebuild the exact matched Q0+Q1 path used by ``infer_exact_wce``."""

    config.validate()
    if matched_cube_drae.ndim != 5 or matched_cube_drae.shape[0] != 1:
        raise ValueError("WCE diagnosis requires one batched DRAE Cube")
    if not bool(torch.isfinite(matched_cube_drae).all()):
        raise ValueError("WCE diagnosis Cube must be finite")

    q0 = fixed_wide_q0(
        range_m,
        azimuth_rad,
        elevation_rad,
        quotas=config.q0_range_quotas,
        seed=config.seed,
    ).to(matched_cube_drae.device)
    condition = model.encode_condition(matched_cube_drae)["condition_latents"]
    matched_q0 = model.decode_queries(
        q0,
        condition,
        chunk_size=config.decode_chunk_size,
    )
    q1, q1_report = occupancy_dependent_q1(
        matched_q0,
        range_m,
        azimuth_rad,
        elevation_rad,
        anchor_quotas=config.q1_anchor_quotas,
        samples_per_anchor=config.q1_samples_per_anchor,
        seed=config.seed,
    )
    matched_q1 = model.decode_queries(
        q1,
        condition,
        chunk_size=config.decode_chunk_size,
    )
    candidate_xyz, candidate_confidence = _candidate_tensors(
        (matched_q0, matched_q1),
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    expected_count = sum(config.q0_range_quotas) + (
        sum(config.q1_anchor_quotas) * config.q1_samples_per_anchor
    )
    if candidate_xyz.shape != (expected_count, 3):
        raise AssertionError("WCE diagnosis changed the frozen candidate count")
    if candidate_confidence.shape != (expected_count,):
        raise AssertionError("WCE diagnosis changed candidate confidence shape")

    current_export = exact_capacity_export(
        candidate_xyz,
        candidate_confidence,
        output_quotas=config.output_range_quotas,
        minimum_distance_m=config.minimum_distance_m,
    )
    query_hashes = {
        "q0_normalized_rae_sha256": tensor_sha256(q0),
        "q1_normalized_rae_sha256": tensor_sha256(q1),
    }
    return WCECandidatePool(
        xyz_m=candidate_xyz,
        confidence=candidate_confidence,
        current_confidence_export=current_export,
        query_hashes=query_hashes,
        report={
            "construction": "infer_exact_wce_matched_q0_plus_q1_exact_path",
            "candidate_count": expected_count,
            "q0_query_count": int(q0.shape[1]),
            "q1_query_count": int(q1.shape[1]),
            "q1": q1_report,
            "query_hashes": query_hashes,
            "ground_truth_accessed": False,
            "future_cube_accessed": False,
            "cfar_accessed": False,
        },
    )


def validate_current_confidence_control(
    pool: WCECandidatePool,
    formal_frame: dict[str, Any],
) -> dict[str, Any]:
    """Require bit-exact query and current-export hashes from formal metrics."""

    expected_export = formal_frame.get("matched_hashes")
    expected_queries = formal_frame.get("inference", {}).get("query_hashes")
    if not isinstance(expected_export, dict) or not isinstance(
        expected_queries, dict
    ):
        raise ValueError("Formal WCE frame lacks control hash documents")
    actual_export = pool.current_confidence_export.hashes
    export_checks = {
        key: actual_export.get(key) == expected_export.get(key)
        for key in (
            "xyz_sha256",
            "confidence_sha256",
            "selected_candidate_rows_sha256",
        )
    }
    query_checks = {
        key: pool.query_hashes.get(key) == expected_queries.get(key)
        for key in (
            "q0_normalized_rae_sha256",
            "q1_normalized_rae_sha256",
        )
    }
    failed = [
        name
        for name, passed in {**export_checks, **query_checks}.items()
        if not passed
    ]
    if failed:
        raise ValueError(
            "WCE current-confidence control does not reproduce formal hashes: "
            f"{failed}"
        )
    return {
        "passed": True,
        "bit_exact": True,
        "export_hash_checks": export_checks,
        "query_hash_checks": query_checks,
        "actual_export_hashes": actual_export,
        "expected_export_hashes": expected_export,
        "actual_query_hashes": pool.query_hashes,
        "expected_query_hashes": expected_queries,
    }


def validate_candidate_target_inputs(
    candidate_xyz_m: torch.Tensor,
    candidate_confidence: torch.Tensor,
    target_xyz_m: torch.Tensor,
    target_weight: torch.Tensor,
) -> None:
    candidate_count = candidate_xyz_m.shape[0]
    if candidate_xyz_m.ndim != 2 or candidate_xyz_m.shape[1:] != (3,):
        raise ValueError("WCE diagnostic candidate XYZ must have shape (N,3)")
    if candidate_count == 0:
        raise ValueError("WCE diagnostic candidate pool cannot be empty")
    if candidate_confidence.shape != (candidate_count,):
        raise ValueError("WCE diagnostic confidence must match candidate rows")
    if target_xyz_m.ndim != 2 or target_xyz_m.shape[1:] != (3,):
        raise ValueError("WCE diagnostic target XYZ must have shape (M,3)")
    if target_xyz_m.shape[0] == 0:
        raise ValueError("WCE diagnostic target cannot be empty")
    if target_weight.shape != (target_xyz_m.shape[0],):
        raise ValueError("WCE diagnostic target weights must match target rows")
    for value, name in (
        (candidate_xyz_m, "candidate XYZ"),
        (candidate_confidence, "candidate confidence"),
        (target_xyz_m, "target XYZ"),
        (target_weight, "target weight"),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"WCE diagnostic {name} must be finite")
    if bool((target_weight < 0.0).any()) or float(target_weight.sum()) <= 0.0:
        raise ValueError(
            "WCE diagnostic target weights must be nonnegative with positive mass"
        )
    if candidate_xyz_m.device != target_xyz_m.device:
        raise ValueError("WCE diagnostic candidates and target must share a device")


def _quantile_summary(values: torch.Tensor) -> dict[str, float]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("WCE diagnostic quantiles require a nonempty vector")
    levels = values.new_tensor(QUANTILE_LEVELS)
    quantiles = torch.quantile(values.float(), levels).detach().cpu().tolist()
    return {
        f"q{int(round(level * 100)):02d}": float(value)
        for level, value in zip(QUANTILE_LEVELS, quantiles, strict=True)
    }


def _support_fractions(
    distance_m: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> dict[str, float]:
    if distance_m.ndim != 1 or distance_m.numel() == 0:
        raise ValueError("WCE support fractions require nonempty distances")
    if weight is None:
        weight = torch.ones_like(distance_m)
    else:
        weight = weight.to(distance_m)
    denominator = weight.sum().clamp_min(1e-8)
    return {
        f"within_{str(threshold).replace('.', 'p')}m": float(
            (((distance_m <= threshold).to(weight) * weight).sum() / denominator)
            .detach()
            .cpu()
            .item()
        )
        for threshold in SUPPORT_THRESHOLDS_M
    }


def _distance_distribution(distance_m: torch.Tensor) -> dict[str, Any]:
    return {
        "count": int(distance_m.numel()),
        "mean_m": float(distance_m.float().mean().detach().cpu().item()),
        "quantiles_m": _quantile_summary(distance_m),
        "support_fraction": _support_fractions(distance_m),
    }


def support_diagnostics(
    candidate_xyz_m: torch.Tensor,
    target_xyz_m: torch.Tensor,
    target_weight: torch.Tensor,
    candidate_to_target_m: torch.Tensor,
    target_to_candidate_m: torch.Tensor,
) -> dict[str, Any]:
    candidate_codes = range_stratum_codes(candidate_xyz_m)
    target_codes = range_stratum_codes(target_xyz_m)
    per_range: dict[str, Any] = {}
    for stratum, (label, _, _) in enumerate(RANGE_STRATA_M):
        candidate_mask = candidate_codes == stratum
        target_mask = target_codes == stratum
        per_range[label] = {
            "candidate_to_target": (
                _distance_distribution(candidate_to_target_m[candidate_mask])
                if bool(candidate_mask.any())
                else None
            ),
            "target_to_candidate": (
                {
                    **_distance_distribution(
                        target_to_candidate_m[target_mask]
                    ),
                    "weighted_support_fraction": _support_fractions(
                        target_to_candidate_m[target_mask],
                        target_weight[target_mask],
                    ),
                    "target_effective_count": float(
                        target_weight[target_mask].sum().detach().cpu().item()
                    ),
                }
                if bool(target_mask.any())
                else None
            ),
        }
    return {
        "candidate_to_target": _distance_distribution(candidate_to_target_m),
        "target_to_candidate": {
            **_distance_distribution(target_to_candidate_m),
            "weighted_support_fraction": _support_fractions(
                target_to_candidate_m,
                target_weight,
            ),
            "target_effective_count": float(
                target_weight.sum().detach().cpu().item()
            ),
        },
        "per_range": per_range,
        "thresholds_m": list(SUPPORT_THRESHOLDS_M),
        "distance_scope": "global_cross_range_nearest",
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], ordered.size]
    average = 0.5 * (starts + stops - 1)
    ranks = np.repeat(average, stops - starts).astype(np.float64, copy=False)
    output = np.empty_like(ranks)
    output[order] = ranks
    return output


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def ranking_alignment_diagnostics(
    confidence: torch.Tensor,
    candidate_to_target_m: torch.Tensor,
) -> dict[str, Any]:
    """Measure whether current confidence ranks GT-near candidates highly."""

    if confidence.shape != candidate_to_target_m.shape:
        raise ValueError("WCE ranking confidence and distance shapes differ")
    if confidence.ndim != 1 or confidence.numel() == 0:
        raise ValueError("WCE ranking requires nonempty vectors")
    if not bool(torch.isfinite(confidence).all()) or not bool(
        torch.isfinite(candidate_to_target_m).all()
    ):
        raise ValueError("WCE ranking inputs must be finite")

    confidence_np = confidence.detach().float().cpu().numpy().astype(
        np.float64, copy=False
    )
    negative_distance_np = (
        -candidate_to_target_m.detach().float().cpu().numpy()
    ).astype(np.float64, copy=False)
    candidate_ids = np.arange(confidence_np.size, dtype=np.int64)
    order = np.lexsort((candidate_ids, -confidence_np))
    top_reports: dict[str, Any] = {}
    for fraction in TOP_CONFIDENCE_FRACTIONS:
        count = max(1, int(round(confidence_np.size * fraction)))
        rows = torch.from_numpy(order[:count]).to(
            device=candidate_to_target_m.device
        )
        distances = candidate_to_target_m[rows]
        label = f"top_{str(fraction).replace('.', 'p')}"
        top_reports[label] = {
            "fraction": fraction,
            "count": count,
            "confidence_quantiles": _quantile_summary(confidence[rows]),
            "candidate_to_target": _distance_distribution(distances),
        }
    return {
        "score_pair": "current_confidence_vs_negative_candidate_to_target_distance",
        "pearson": _finite_correlation(
            confidence_np,
            negative_distance_np,
        ),
        "spearman_average_ties": _finite_correlation(
            _average_ranks(confidence_np),
            _average_ranks(negative_distance_np),
        ),
        "confidence_quantiles": _quantile_summary(confidence),
        "candidate_to_target_distance_quantiles_m": _quantile_summary(
            candidate_to_target_m
        ),
        "top_confidence_subsets": top_reports,
    }


def validation_gt_nearest_score_export(
    candidate_xyz_m: torch.Tensor,
    candidate_to_target_m: torch.Tensor,
) -> ExactWCEExport:
    """Apply frozen exact export with unattainable GT-nearest ranking."""

    if candidate_to_target_m.shape != (candidate_xyz_m.shape[0],):
        raise ValueError("GT-nearest score must match candidate rows")
    if not bool(torch.isfinite(candidate_to_target_m).all()) or bool(
        (candidate_to_target_m < 0.0).any()
    ):
        raise ValueError("GT-nearest distances must be finite and nonnegative")
    result = exact_capacity_export(
        candidate_xyz_m,
        -candidate_to_target_m,
        output_quotas=FORMAL_OUTPUT_RANGE_QUOTAS,
        minimum_distance_m=CAPACITY_DISTANCE_M,
    )
    if result.xyz_m.shape != (EXPORT_COUNT, 3):
        raise AssertionError("GT-nearest diagnostic changed exact point count")
    if (
        result.report["observed_minimum_pair_distance_m"]
        < CAPACITY_DISTANCE_M - 1e-6
    ):
        raise AssertionError("GT-nearest diagnostic violated 5 cm spacing")
    result.report.update(
        {
            "arm": "validation_gt_nearest_score",
            "ground_truth_used_for_ranking": True,
            "unattainable": True,
            "strict_upper_bound": False,
            "heuristic": True,
            "coverage_aware": False,
            "eligible_as_method_result": False,
        }
    )
    return result


def diagnose_frozen_candidate_pool(
    pool: WCECandidatePool,
    target_xyz_m: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    candidate_chunk_size: int,
    target_chunk_size: int,
) -> WCEFailureFactorFrame:
    validate_candidate_target_inputs(
        pool.xyz_m,
        pool.confidence,
        target_xyz_m,
        target_weight,
    )
    candidate_ids = torch.arange(
        pool.xyz_m.shape[0],
        device=pool.xyz_m.device,
        dtype=torch.long,
    )
    nearest = streaming_bidirectional_nearest(
        pool.xyz_m.float(),
        target_xyz_m.float(),
        candidate_ids,
        candidate_chunk_size=candidate_chunk_size,
        target_chunk_size=target_chunk_size,
        target_topk_count=1,
    )
    gt_export = validation_gt_nearest_score_export(
        pool.xyz_m,
        nearest.candidate_to_target_m,
    )
    current_geometry = geometry_report(
        pool.current_confidence_export.xyz_m.to(target_xyz_m),
        target_xyz_m,
        target_weight=target_weight,
    )
    gt_geometry = geometry_report(
        gt_export.xyz_m.to(target_xyz_m),
        target_xyz_m,
        target_weight=target_weight,
    )
    return WCEFailureFactorFrame(
        current_geometry=current_geometry,
        validation_gt_nearest_score_geometry=gt_geometry,
        candidate_to_target_distance_m=nearest.candidate_to_target_m,
        target_to_candidate_distance_m=nearest.target_to_candidate_m,
        support=support_diagnostics(
            pool.xyz_m,
            target_xyz_m,
            target_weight,
            nearest.candidate_to_target_m,
            nearest.target_to_candidate_m,
        ),
        ranking=ranking_alignment_diagnostics(
            pool.confidence,
            nearest.candidate_to_target_m,
        ),
        validation_gt_nearest_score_export=gt_export,
    )


def geometry_gate_checks(aggregate: dict[str, dict]) -> dict[str, bool]:
    far = aggregate.get(FAR_COMPLETENESS_METRIC)
    return {
        "mean_chamfer_at_most_2p50m": (
            float(aggregate["chamfer_m"]["mean"]) <= 2.50
        ),
        "median_completeness_at_most_2p4946m": (
            float(aggregate["completeness_mean_distance_m"]["median"])
            <= 2.4946
        ),
        "mean_outlier_fraction_at_most_25pct": (
            float(aggregate["outlier_fraction_2m"]["mean"]) <= 0.25
        ),
        "mean_far_completeness_at_most_46p9407m": (
            far is not None
            and int(far["sample_count"]) == 23
            and float(far["mean"]) <= 46.9407
        ),
    }


def diagnosis_decision(
    current_aggregate: dict[str, dict],
    gt_nearest_aggregate: dict[str, dict],
    *,
    frame_count: int,
    far_frame_count: int,
    preflight: bool,
) -> dict[str, Any]:
    current_checks = geometry_gate_checks(current_aggregate)
    gt_checks = geometry_gate_checks(gt_nearest_aggregate)
    count_checks = {
        "exact_24_validation_frames": frame_count == 24,
        "exact_23_far_target_frames": far_frame_count == 23,
    }
    formal = not preflight and all(count_checks.values())
    current_pass = all(current_checks.values())
    gt_pass = all(gt_checks.values())
    if not formal:
        branch = "preflight_only_no_failure_factor_decision"
    elif current_pass:
        branch = "current_confidence_geometry_not_a_failure"
    elif gt_pass:
        branch = "confidence_ranking_bottleneck_indicated"
    else:
        branch = "validation_gt_nearest_score_does_not_rescue"
    return {
        "formal_failure_factor_decision_eligible": formal,
        "branch": branch,
        "current_confidence_geometry_checks": current_checks,
        "validation_gt_nearest_score_geometry_checks": gt_checks,
        "count_checks": count_checks,
        "current_confidence_geometry_passed": current_pass,
        "validation_gt_nearest_score_geometry_passed": gt_pass,
        "validation_gt_nearest_score_unattainable": True,
        "validation_gt_nearest_score_strict_upper_bound": False,
        "coverage_aware_oracle_implemented": False,
        "method_promotion_eligible": False,
        "interpretation": (
            "A GT-nearest rescue isolates confidence ranking as a bottleneck. "
            "Failure to rescue does not prove that no GT-guided or "
            "coverage-aware subset exists."
        ),
    }


def aggregate_failure_factor_frames(
    frames: list[dict[str, Any]],
    *,
    preflight: bool,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("WCE failure-factor aggregation requires frames")
    current = aggregate_geometry_reports(
        [frame["current_confidence"]["geometry"] for frame in frames]
    )
    gt_nearest = aggregate_geometry_reports(
        [
            frame["validation_gt_nearest_score"]["geometry"]
            for frame in frames
        ]
    )
    far_frame_count = sum(bool(frame["has_far_target"]) for frame in frames)
    return {
        "frame_count": len(frames),
        "far_target_frame_count": far_frame_count,
        "current_confidence": current,
        "validation_gt_nearest_score": gt_nearest,
        "decision": diagnosis_decision(
            current,
            gt_nearest,
            frame_count=len(frames),
            far_frame_count=far_frame_count,
            preflight=preflight,
        ),
    }
