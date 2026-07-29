"""Read-only cardinality diagnostics for frozen RAE occupancy logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

import numpy as np
import torch

from cube_dense.kradar import KRadarAxes
from eval.dense_geometry import (
    aggregate_geometry_reports,
    geometry_report,
    nearest_distance,
    occupancy_to_points,
    rae_indices_to_xyz,
)


FIXED_K_VALUES = (2_500, 5_000, 7_500, 10_000)
LEGACY_DENSE_K = 10_000
FAR_COMPLETENESS_METRIC = "range_60_120m_completeness_mean_distance_m"
RANGE_BINS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
LEGACY_OVERALL_METRICS = (
    "chamfer_m",
    "precision_mean_distance_m",
    "completeness_mean_distance_m",
    "outlier_fraction_2m",
    "precision_0p5m",
    "recall_0p5m",
    "fscore_0p5m",
    "precision_1p0m",
    "recall_1p0m",
    "fscore_1p0m",
    "precision_2p0m",
    "recall_2p0m",
    "fscore_2p0m",
)


@dataclass(frozen=True)
class AdaptiveCountRule:
    """A validation-time rule frozen from train-partition confidence only."""

    confidence_threshold: float
    minimum_count: int = FIXED_K_VALUES[0]
    maximum_count: int = LEGACY_DENSE_K
    fit_partition: str = "train"
    fit_target_statistic: str = "rounded_target_effective_count"
    fit_frame_count: int = 0
    preflight_only: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Adaptive confidence threshold must lie in [0,1]")
        if self.minimum_count <= 0:
            raise ValueError("Adaptive minimum count must be positive")
        if self.maximum_count < self.minimum_count:
            raise ValueError("Adaptive maximum count must not be below minimum")
        if self.fit_partition != "train":
            raise ValueError("Adaptive cardinality may only be fit on train")
        if self.fit_frame_count <= 0:
            raise ValueError("Adaptive cardinality requires train fit frames")

    def to_document(self) -> dict[str, Any]:
        return asdict(self)


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and canonical contiguous bytes."""

    canonical = value.detach().cpu().contiguous()
    if canonical.dtype == torch.bfloat16:
        canonical = canonical.float()
    array = canonical.numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def point_output_sha256(
    xyz_m: torch.Tensor,
    confidence: torch.Tensor,
    indices_rae: torch.Tensor,
) -> str:
    """Hash the frozen XYZ, confidence, and unique-cell output contract."""

    digest = hashlib.sha256()
    digest.update(b"rae_cell_xyz_confidence_indices_v1")
    for name, value in (
        ("xyz_m", xyz_m),
        ("confidence", confidence),
        ("indices_rae", indices_rae),
    ):
        digest.update(name.encode("ascii"))
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def validate_k_values(
    k_values: Iterable[int],
    cell_count: int,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in k_values)
    if not values:
        raise ValueError("At least one fixed K is required")
    if tuple(sorted(set(values))) != values:
        raise ValueError("Fixed K values must be unique and strictly increasing")
    missing = sorted(set(FIXED_K_VALUES) - set(values))
    if missing:
        raise ValueError(f"Fixed K sweep lacks required arms: {missing}")
    if values[-1] != LEGACY_DENSE_K:
        raise ValueError("The frozen cardinality sweep must end at exact 10k")
    if values[0] <= 0 or values[-1] > cell_count:
        raise ValueError(
            f"Fixed K range {values[0]}..{values[-1]} does not fit "
            f"{cell_count} RAE cells"
        )
    return values


def stable_topk_flat_indices(
    logits_rae: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Select deterministic sorted top-k cells via the frozen legacy path."""

    if logits_rae.ndim != 3 or not torch.is_floating_point(logits_rae):
        raise ValueError("RAE logits must be a floating (R,A,E) tensor")
    if not torch.isfinite(logits_rae).all():
        raise ValueError("RAE logits contain non-finite values")
    flat = torch.sigmoid(logits_rae).flatten()
    if k <= 0 or k > flat.numel():
        raise ValueError(f"top-k {k} does not fit {flat.numel()} RAE cells")

    selected = torch.topk(flat, k, sorted=True).indices
    if selected.numel() != k or torch.unique(selected).numel() != k:
        raise RuntimeError("Stable top-k did not return k unique RAE cells")
    return selected


def flat_indices_to_rae(
    flat_index: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    if flat_index.ndim != 1:
        raise ValueError("Flat RAE indices must be one-dimensional")
    range_count, azimuth_count, elevation_count = spatial_shape
    cell_count = range_count * azimuth_count * elevation_count
    if flat_index.numel() and (
        bool((flat_index < 0).any()) or bool((flat_index >= cell_count).any())
    ):
        raise ValueError("Flat RAE index is outside the spatial domain")
    range_index = flat_index // (azimuth_count * elevation_count)
    remainder = flat_index % (azimuth_count * elevation_count)
    azimuth_index = remainder // elevation_count
    elevation_index = remainder % elevation_count
    return torch.stack(
        (range_index, azimuth_index, elevation_index),
        dim=1,
    )


def _arm_label(kind: str, count: int) -> str:
    if kind == "fixed_k" and count == LEGACY_DENSE_K:
        return "frozen_exact_10k_dense_output"
    if kind == "fixed_k":
        return "reduced_cardinality_diagnostic_not_10k_dense"
    return "train_frozen_adaptive_cardinality_diagnostic"


def select_cardinality_arms(
    logits_rae: torch.Tensor,
    axes: KRadarAxes,
    *,
    k_values: Iterable[int] = FIXED_K_VALUES,
    adaptive_rule: AdaptiveCountRule | None = None,
) -> dict[str, dict[str, Any]]:
    """Decode logits-only arms; validation targets are intentionally absent."""

    cell_count = int(logits_rae.numel())
    fixed = validate_k_values(k_values, cell_count)
    maximum = max(
        fixed[-1],
        adaptive_rule.maximum_count if adaptive_rule is not None else 0,
    )
    if maximum > cell_count:
        raise ValueError("Adaptive maximum count exceeds the RAE cell count")
    ranked = stable_topk_flat_indices(logits_rae, maximum)
    ranked_confidence = torch.sigmoid(logits_rae).flatten()[ranked]

    counts: list[tuple[str, str, int, int | None]] = [
        (f"fixed_k_{count}", "fixed_k", count, count) for count in fixed
    ]
    if adaptive_rule is not None:
        threshold_count = int(
            (ranked_confidence[: adaptive_rule.maximum_count]
             >= adaptive_rule.confidence_threshold)
            .sum()
            .item()
        )
        adaptive_count = max(
            adaptive_rule.minimum_count,
            min(adaptive_rule.maximum_count, threshold_count),
        )
        counts.append(
            ("adaptive_train_confidence", "adaptive", adaptive_count, None)
        )

    arms = {}
    for name, kind, count, requested_k in counts:
        flat_index = ranked[:count]
        indices = flat_indices_to_rae(flat_index, tuple(logits_rae.shape))
        confidence = ranked_confidence[:count]
        xyz_m = rae_indices_to_xyz(indices, axes)
        arms[name] = {
            "selection_kind": kind,
            "selection_source": "model_occupancy_logits_only",
            "validation_gt_used_for_selection": False,
            "requested_k": requested_k,
            "selected_count": count,
            "density_label": _arm_label(kind, count),
            "xyz_m": xyz_m,
            "confidence": confidence,
            "indices_rae": indices,
            "flat_index": flat_index,
        }
    return arms


def freeze_train_adaptive_rule(
    confidence_cutoffs: Iterable[float],
    *,
    minimum_count: int = FIXED_K_VALUES[0],
    maximum_count: int = LEGACY_DENSE_K,
    preflight_only: bool = False,
) -> AdaptiveCountRule:
    """Freeze one global threshold from train-derived per-frame cutoffs."""

    values = np.asarray(list(confidence_cutoffs), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Adaptive fit requires finite train confidence cutoffs")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Train confidence cutoffs must lie in [0,1]")
    return AdaptiveCountRule(
        confidence_threshold=float(np.median(values)),
        minimum_count=minimum_count,
        maximum_count=maximum_count,
        fit_frame_count=int(values.size),
        preflight_only=preflight_only,
    )


def train_frame_confidence_cutoff(
    logits_rae: torch.Tensor,
    target_effective_count: float,
    *,
    minimum_count: int = FIXED_K_VALUES[0],
    maximum_count: int = LEGACY_DENSE_K,
) -> float:
    """Return a train-only confidence cutoff at the clipped effective count."""

    if not np.isfinite(target_effective_count) or target_effective_count <= 0.0:
        raise ValueError("Train target effective count must be finite and positive")
    desired = int(round(target_effective_count))
    desired = max(minimum_count, min(maximum_count, desired))
    ranked = stable_topk_flat_indices(logits_rae, maximum_count)
    confidence = torch.sigmoid(logits_rae).flatten()[ranked]
    return float(confidence[desired - 1].item())


def validate_legacy_10k_reproduction(
    logits_rae: torch.Tensor,
    axes: KRadarAxes,
    arm: dict[str, Any],
) -> dict[str, Any]:
    """Require exact agreement with the original occupancy decoder."""

    if arm.get("selection_kind") != "fixed_k":
        raise ValueError("Legacy reproduction requires a fixed-K arm")
    if int(arm.get("selected_count", -1)) != LEGACY_DENSE_K:
        raise ValueError("Legacy reproduction requires the exact-10k arm")
    legacy_xyz, legacy_confidence, legacy_indices = occupancy_to_points(
        logits_rae,
        axes,
        point_count=LEGACY_DENSE_K,
    )
    checks = {
        "indices_exact": torch.equal(arm["indices_rae"], legacy_indices),
        "confidence_exact": torch.equal(arm["confidence"], legacy_confidence),
        "xyz_exact": torch.equal(arm["xyz_m"], legacy_xyz),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stable 10k arm differs from legacy decoder: {checks}")
    stable_hash = point_output_sha256(
        arm["xyz_m"],
        arm["confidence"],
        arm["indices_rae"],
    )
    legacy_hash = point_output_sha256(
        legacy_xyz,
        legacy_confidence,
        legacy_indices,
    )
    checks["output_sha256_exact"] = stable_hash == legacy_hash
    if not checks["output_sha256_exact"]:
        raise RuntimeError("Legacy exact-10k output hash differs")
    return {
        "contract": "legacy_occupancy_to_points_sorted_topk_v1",
        "point_count": LEGACY_DENSE_K,
        "stable_output_sha256": stable_hash,
        "legacy_output_sha256": legacy_hash,
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_legacy_overall_metrics(
    current: dict[str, Any],
    archived: dict[str, Any],
    *,
    absolute_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare only uncensored legacy overall endpoints, never old far metrics."""

    if absolute_tolerance < 0.0:
        raise ValueError("Legacy metric tolerance must be non-negative")
    differences = {}
    for key in LEGACY_OVERALL_METRICS:
        if key not in current or key not in archived:
            raise ValueError(f"Legacy overall metric is missing: {key}")
        current_value = float(current[key])
        archived_value = float(archived[key])
        difference = abs(current_value - archived_value)
        differences[key] = {
            "current": current_value,
            "archived": archived_value,
            "absolute_difference": difference,
            "within_tolerance": difference <= absolute_tolerance,
        }
    passed = all(value["within_tolerance"] for value in differences.values())
    return {
        "absolute_tolerance": absolute_tolerance,
        "far_metrics_compared": False,
        "metrics": differences,
        "passed": passed,
    }


def range_coverage_report(
    prediction_xyz: torch.Tensor,
    prediction_confidence: torch.Tensor,
    target_xyz: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    distance_bins_m: tuple[tuple[float, float], ...] = RANGE_BINS_M,
    chunk_size: int = 1024,
) -> dict[str, dict[str, float | int]]:
    """Report per-range support using corrected global-nearest completeness."""

    if prediction_confidence.shape != (prediction_xyz.shape[0],):
        raise ValueError("Prediction confidence shape differs from point count")
    if target_weight.shape != (target_xyz.shape[0],):
        raise ValueError("Target weight shape differs from target count")
    target_distance = nearest_distance(
        target_xyz,
        prediction_xyz,
        chunk_size=chunk_size,
    )
    prediction_range = torch.linalg.vector_norm(prediction_xyz, dim=1)
    target_range = torch.linalg.vector_norm(target_xyz, dim=1)
    report = {}
    for lower, upper in distance_bins_m:
        prediction_mask = (prediction_range >= lower) & (prediction_range < upper)
        target_mask = (target_range >= lower) & (target_range < upper)
        target_bin_weight = target_weight[target_mask].clamp_min(0.0)
        target_effective = float(target_bin_weight.sum().item())
        name = f"range_{int(lower)}_{int(upper)}m"
        document: dict[str, float | int] = {
            "prediction_count": int(prediction_mask.sum().item()),
            "prediction_confidence_sum": float(
                prediction_confidence[prediction_mask].sum().item()
            ),
            "target_count_diagnostic_only": int(target_mask.sum().item()),
            "target_effective_count_diagnostic_only": target_effective,
        }
        if bool(target_mask.any()):
            distance = target_distance[target_mask]
            denominator = target_bin_weight.sum().clamp_min(1e-8)
            document["completeness_mean_distance_m"] = float(
                ((distance * target_bin_weight).sum() / denominator).item()
            )
            document["recall_1m"] = float(
                (
                    (distance <= 1.0).to(target_bin_weight)
                    * target_bin_weight
                ).sum().div(denominator).item()
            )
        report[name] = document
    return report


def evaluate_cardinality_arm(
    arm: dict[str, Any],
    target_xyz_confidence: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate a logits-selected arm after selection is already frozen."""

    target = target_xyz_confidence.float()
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError("Target diagnostic must have nonempty shape (N,4)")
    if not torch.isfinite(target).all():
        raise ValueError("Target diagnostic contains non-finite values")
    xyz_m = arm["xyz_m"].float()
    confidence = arm["confidence"].float()
    indices = arm["indices_rae"]
    return {
        "selection": {
            "selection_kind": arm["selection_kind"],
            "selection_source": arm["selection_source"],
            "validation_gt_used_for_selection": arm[
                "validation_gt_used_for_selection"
            ],
            "requested_k": arm["requested_k"],
            "selected_count": arm["selected_count"],
            "density_label": arm["density_label"],
            "unique_rae_cell_count": int(torch.unique(indices, dim=0).shape[0]),
            "indices_rae_sha256": tensor_sha256(indices),
            "output_sha256": point_output_sha256(xyz_m, confidence, indices),
        },
        "geometry": geometry_report(
            xyz_m,
            target[:, :3],
            target_weight=target[:, 3],
        ),
        "prediction_diagnostics": {
            "selected_count": int(xyz_m.shape[0]),
            "confidence_sum_effective_count": float(confidence.sum().item()),
            "confidence_ge_0p5_count": int((confidence >= 0.5).sum().item()),
            "confidence_mean": float(confidence.mean().item()),
        },
        "range_coverage": range_coverage_report(
            xyz_m,
            confidence,
            target[:, :3],
            target[:, 3],
        ),
        "validation_gt_diagnostic_only": {
            "selection_dependency": False,
            "target_count": int(target.shape[0]),
            "target_effective_count": float(target[:, 3].sum().item()),
        },
    }


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "sample_count": int(array.size),
    }


def aggregate_cardinality_frames(
    frames: list[dict[str, Any]],
    *,
    expected_frame_count: int,
    expected_far_target_count: int,
) -> dict[str, Any]:
    if len(frames) != expected_frame_count:
        raise ValueError(
            f"Cardinality arm has {len(frames)} frames, "
            f"expected {expected_frame_count}"
        )
    identities = [
        (
            int(frame["seed"]) if "seed" in frame else None,
            int(frame["sequence"]),
            int(frame["radar_index"]),
        )
        for frame in frames
    ]
    if len(set(identities)) != expected_frame_count:
        raise ValueError("Cardinality arm frame identities are not unique")
    geometry = aggregate_geometry_reports(
        [frame["geometry"] for frame in frames]
    )
    far_count = geometry.get(FAR_COMPLETENESS_METRIC, {}).get("sample_count", 0)
    if far_count != expected_far_target_count:
        raise ValueError(
            f"Corrected far completeness covers {far_count} frames, "
            f"expected {expected_far_target_count}"
        )

    prediction_keys = tuple(frames[0]["prediction_diagnostics"])
    prediction = {
        key: _numeric_summary(
            frame["prediction_diagnostics"][key] for frame in frames
        )
        for key in prediction_keys
    }
    target_keys = ("target_count", "target_effective_count")
    target = {
        key: _numeric_summary(
            frame["validation_gt_diagnostic_only"][key] for frame in frames
        )
        for key in target_keys
    }
    range_coverage = {}
    for bin_name in frames[0]["range_coverage"]:
        keys = sorted(
            {
                key
                for frame in frames
                for key, value in frame["range_coverage"][bin_name].items()
                if isinstance(value, (int, float))
            }
        )
        range_coverage[bin_name] = {
            key: _numeric_summary(
                frame["range_coverage"][bin_name][key]
                for frame in frames
                if key in frame["range_coverage"][bin_name]
            )
            for key in keys
        }
    return {
        "frame_count": expected_frame_count,
        "far_target_frame_count": expected_far_target_count,
        "geometry": geometry,
        "prediction_diagnostics": prediction,
        "range_coverage": range_coverage,
        "validation_gt_diagnostic_only": {
            "selection_dependency": False,
            **target,
        },
        "frames": frames,
    }


def _dominates(
    first: dict[str, float],
    second: dict[str, float],
    *,
    maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> bool:
    weak = all(first[key] >= second[key] for key in maximize) and all(
        first[key] <= second[key] for key in minimize
    )
    strict = any(first[key] > second[key] for key in maximize) or any(
        first[key] < second[key] for key in minimize
    )
    return weak and strict


def build_density_quality_pareto(
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build explicit quality-only and density-quality nondominated sets."""

    points = {}
    for name, arm in arms.items():
        geometry = arm["geometry"]
        prediction = arm["prediction_diagnostics"]
        frame = arm["frames"][0]
        points[name] = {
            "density_label": frame["selection"]["density_label"],
            "selected_count_mean": float(
                prediction["selected_count"]["mean"]
            ),
            "confidence_effective_count_mean": float(
                prediction["confidence_sum_effective_count"]["mean"]
            ),
            "chamfer_m_mean": float(geometry["chamfer_m"]["mean"]),
            "completeness_m_mean": float(
                geometry["completeness_mean_distance_m"]["mean"]
            ),
            "outlier_fraction_2m_mean": float(
                geometry["outlier_fraction_2m"]["mean"]
            ),
        }

    quality_minimize = (
        "chamfer_m_mean",
        "completeness_m_mean",
        "outlier_fraction_2m_mean",
    )
    density_minimize = (
        "completeness_m_mean",
        "outlier_fraction_2m_mean",
    )
    quality_front = []
    density_front = []
    for name, point in points.items():
        if not any(
            _dominates(
                other,
                point,
                maximize=(),
                minimize=quality_minimize,
            )
            for other_name, other in points.items()
            if other_name != name
        ):
            quality_front.append(name)
        if not any(
            _dominates(
                other,
                point,
                maximize=("selected_count_mean",),
                minimize=density_minimize,
            )
            for other_name, other in points.items()
            if other_name != name
        ):
            density_front.append(name)
    return {
        "points": points,
        "quality_only_minimize": list(quality_minimize),
        "quality_only_nondominated_arms": quality_front,
        "density_quality_objectives": {
            "maximize": ["selected_count_mean"],
            "minimize": list(density_minimize),
        },
        "density_quality_nondominated_arms": density_front,
        "interpretation_guard": (
            "Only fixed_k_10000 is an exact-10k dense output; all smaller-K "
            "arms are reduced-cardinality diagnostics."
        ),
    }


def build_exact10k_factor_diagnosis(
    arms: dict[str, dict[str, Any]],
    *,
    reference_name: str = "fixed_k_10000",
) -> dict[str, Any]:
    """Apply a frozen descriptive rubric against the exact-10k reference."""

    if reference_name not in arms:
        raise ValueError(f"Missing exact-10k reference arm {reference_name}")
    reference = arms[reference_name]["geometry"]
    reference_values = {
        "chamfer_m": float(reference["chamfer_m"]["mean"]),
        "completeness_m": float(
            reference["completeness_mean_distance_m"]["mean"]
        ),
        "outlier_fraction_2m": float(
            reference["outlier_fraction_2m"]["mean"]
        ),
        "far_completeness_m": float(
            reference[FAR_COMPLETENESS_METRIC]["mean"]
        ),
    }
    comparisons = {}
    strong_arms = []
    contributing_arms = []
    for name, arm in arms.items():
        if name == reference_name:
            continue
        geometry = arm["geometry"]
        candidate = {
            "chamfer_m": float(geometry["chamfer_m"]["mean"]),
            "completeness_m": float(
                geometry["completeness_mean_distance_m"]["mean"]
            ),
            "outlier_fraction_2m": float(
                geometry["outlier_fraction_2m"]["mean"]
            ),
            "far_completeness_m": float(
                geometry[FAR_COMPLETENESS_METRIC]["mean"]
            ),
        }
        delta = {
            key: candidate[key] - reference_values[key]
            for key in reference_values
        }
        chamfer_relative_change = (
            delta["chamfer_m"] / reference_values["chamfer_m"]
        )
        checks = {
            "outlier_at_or_below_frozen_25pct_gate": (
                candidate["outlier_fraction_2m"] <= 0.25
            ),
            "outlier_reduction_at_least_3pp": (
                delta["outlier_fraction_2m"] <= -0.03
            ),
            "chamfer_degradation_at_most_2pct": (
                chamfer_relative_change <= 0.02
            ),
            "completeness_degradation_at_most_0p10m": (
                delta["completeness_m"] <= 0.10
            ),
            "far_completeness_degradation_at_most_1m": (
                delta["far_completeness_m"] <= 1.0
            ),
        }
        strong = all(checks.values())
        contributing = (
            checks["outlier_at_or_below_frozen_25pct_gate"]
            and delta["outlier_fraction_2m"] < 0.0
            and checks["chamfer_degradation_at_most_2pct"]
            and checks["completeness_degradation_at_most_0p10m"]
        )
        if strong:
            strong_arms.append(name)
        if contributing:
            contributing_arms.append(name)
        comparisons[name] = {
            "candidate": candidate,
            "delta_candidate_minus_exact10k": delta,
            "chamfer_relative_change": chamfer_relative_change,
            "checks": checks,
            "strong_major_factor_evidence": strong,
            "bounded_contributing_factor_evidence": contributing,
        }
    return {
        "reference": reference_name,
        "reference_values": reference_values,
        "rubric_frozen_before_full_run": True,
        "strong_major_factor_arms": strong_arms,
        "bounded_contributing_factor_arms": contributing_arms,
        "exact10k_is_major_outlier_factor": bool(strong_arms),
        "exact10k_is_bounded_contributing_factor": bool(contributing_arms),
        "comparisons": comparisons,
        "interpretation_limit": (
            "This is a read-only cardinality intervention on frozen logits; "
            "it diagnoses the decoder count but does not validate a retrained "
            "smaller-output model."
        ),
    }
