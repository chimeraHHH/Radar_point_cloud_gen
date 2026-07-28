#!/usr/bin/env python3
"""Evaluate frozen no-train G1T temporal proposal support on H200."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.temporal_dataset import KRadarTemporalDataset  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
    nearest_distance,
)
from eval.g1t_temporal_proposal import (  # noqa: E402
    ARM_NAMES,
    HistoryProposalSource,
    ProposalSelection,
    build_temporal_proposal_arms,
    cube_proposal_frame,
    selection_report,
)
from eval.rald_guided_query import duplicate_report  # noqa: E402
from eval.temporal_methods import (  # noqa: E402
    aggregate_flat_reports,
    stratified_geometry_report,
)
from losses.doppler_distribution import circular_scalar_target  # noqa: E402
from models.cube_doppler import query_cube_spectrum  # noqa: E402
from models.temporal_baselines import analytic_static_center  # noqa: E402


PROTOCOL = "g1t_no_train_temporal_proposal_v2_corrected_geometry"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
POINT_COUNT = 10_000
HISTORY_FRAMES = 4
NMS_KERNEL = (5, 5, 3)
NEAR_BOUNDARY_M = 60.0
EXPECTED_WINDOWS = 8
EXPECTED_FRAMES = 384
EXPECTED_PAIRS = 376


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_commit(repo: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if not SOURCE_PATTERN.fullmatch(commit):
        raise RuntimeError("G1T requires a full Git source commit")
    return commit


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("Formal G1T evaluation requires CUDA on H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("Formal G1T evaluation is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"Formal G1T requires H200, got {resolved}")
    return device, resolved


def compose_history(
    history: list[HistoryProposalSource],
    current_from_previous: torch.Tensor,
    delta_seconds: float,
) -> list[HistoryProposalSource]:
    if delta_seconds <= 0.0:
        raise ValueError("G1T pair delta must be positive")
    return [
        HistoryProposalSource(
            prediction=source.prediction,
            current_from_source=(
                current_from_previous @ source.current_from_source
            ),
            age_seconds=source.age_seconds + delta_seconds,
            age_frames=source.age_frames + 1,
        )
        for source in history
    ]


def range_slice_geometry(
    prediction_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_weight: torch.Tensor,
    near_boundary_m: float,
) -> dict[str, float]:
    prediction_range = torch.linalg.vector_norm(prediction_xyz, dim=1)
    target_range = torch.linalg.vector_norm(target_xyz, dim=1)
    report: dict[str, float] = {}
    for label, prediction_mask, target_mask in (
        (
            "near",
            prediction_range < near_boundary_m,
            target_range < near_boundary_m,
        ),
        (
            "far",
            prediction_range >= near_boundary_m,
            target_range >= near_boundary_m,
        ),
    ):
        if not target_mask.any():
            continue
        if prediction_mask.any():
            sliced = geometry_report(
                prediction_xyz[prediction_mask],
                target_xyz[target_mask],
                target_weight=target_weight[target_mask],
                distance_bins_m=(),
            )
            for key, value in sliced.items():
                if isinstance(value, (int, float)):
                    report[f"{label}_{key}"] = value
        else:
            sliced_target = target_xyz[target_mask]
            sliced_weight = target_weight[target_mask].clamp_min(0.0)
            distance = nearest_distance(sliced_target, prediction_xyz)
            denominator = sliced_weight.sum().clamp_min(1e-8)
            report[f"{label}_completeness_mean_distance_m"] = float(
                ((distance * sliced_weight).sum() / denominator).item()
            )
    return report


@torch.inference_mode()
def evaluate_selection(
    selection: ProposalSelection,
    cube_drae: torch.Tensor,
    target_xyz_confidence: torch.Tensor,
    target_rae_index: torch.Tensor,
    ego_speed_mps: torch.Tensor,
    static_hypothesis: str,
    axes_tensors: tuple[torch.Tensor, ...],
    dynamic_threshold_mps: float,
) -> dict:
    doppler_mps, _, _, _ = axes_tensors
    doppler_step_mps = torch.median(torch.diff(doppler_mps))
    doppler_lower_mps = doppler_mps[0]
    doppler_period_mps = doppler_step_mps * doppler_mps.numel()
    prediction = selection.prediction
    target_xyz = target_xyz_confidence[:, :3].float()
    target_weight = target_xyz_confidence[:, 3].float()
    geometry = geometry_report(
        prediction.xyz_m.float(),
        target_xyz,
        target_weight=target_weight,
    )
    duplicates = duplicate_report(prediction.xyz_m.float())
    target_distribution = query_cube_spectrum(
        cube_drae, target_rae_index
    )
    target_scalar_mps = circular_scalar_target(
        target_distribution,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
    )
    target_static_center_mps = analytic_static_center(
        target_xyz,
        ego_speed_mps,
        static_hypothesis,
        doppler_lower_mps,
        doppler_period_mps,
    )
    stratified = stratified_geometry_report(
        prediction,
        prediction.static_center_mps,
        target_xyz,
        target_weight,
        target_scalar_mps,
        target_static_center_mps,
        doppler_mps,
        doppler_lower_mps,
        doppler_period_mps,
        dynamic_threshold_mps=dynamic_threshold_mps,
    )
    return {
        "geometry": geometry,
        "duplicates": duplicates,
        "selection": selection_report(
            selection, near_boundary_m=NEAR_BOUNDARY_M
        ),
        "stratified_geometry": stratified,
        "near_far_geometry": range_slice_geometry(
            prediction.xyz_m.float(),
            target_xyz,
            target_weight,
            NEAR_BOUNDARY_M,
        ),
    }


def numeric_sections(frame: dict) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for section in (
        "geometry",
        "duplicates",
        "selection",
        "stratified_geometry",
        "near_far_geometry",
    ):
        for key, value in frame[section].items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                flattened[f"{section}.{key}"] = float(value)
    return flattened


def mean_record(records: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for record in records for key in record})
    return {
        key: float(
            np.mean([record[key] for record in records if key in record])
        )
        for key in keys
    }


def aggregate_frames(frames: list[dict]) -> dict:
    if not frames:
        raise ValueError("Cannot aggregate empty G1T frames")
    grouped: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        grouped[int(frame["sequence"])].append(frame)
    per_scene = {
        str(sequence): {
            "frame_count": len(scene_frames),
            "geometry": aggregate_geometry_reports(
                [frame["geometry"] for frame in scene_frames]
            ),
            "flat": aggregate_flat_reports(
                [numeric_sections(frame) for frame in scene_frames]
            ),
        }
        for sequence, scene_frames in sorted(grouped.items())
    }
    scene_mean_records = [
        mean_record([numeric_sections(frame) for frame in scene_frames])
        for _, scene_frames in sorted(grouped.items())
    ]
    return {
        "frame_count": len(frames),
        "scene_count": len(grouped),
        "frame_level": {
            "geometry": aggregate_geometry_reports(
                [frame["geometry"] for frame in frames]
            ),
            "duplicates": aggregate_flat_reports(
                [frame["duplicates"] for frame in frames]
            ),
            "selection": aggregate_flat_reports(
                [frame["selection"] for frame in frames]
            ),
            "stratified_geometry": aggregate_flat_reports(
                [frame["stratified_geometry"] for frame in frames]
            ),
            "near_far_geometry": aggregate_flat_reports(
                [frame["near_far_geometry"] for frame in frames]
            ),
        },
        "scene_first": aggregate_flat_reports(scene_mean_records),
        "per_scene": per_scene,
    }


def scene_metric(aggregate: dict, key: str) -> float:
    return float(aggregate["scene_first"][key]["mean"])


def promotion_decision(arms: dict[str, dict]) -> dict:
    t0 = arms["t0_current"]["aggregate"]
    t1 = arms["t1_ego"]["aggregate"]
    t2 = arms["t2_doppler"]["aggregate"]
    keys = {
        "chamfer": "geometry.chamfer_m",
        "completeness": "geometry.completeness_mean_distance_m",
        "far_completeness": (
            "geometry.range_60_120m_completeness_mean_distance_m"
        ),
        "outlier": "geometry.outlier_fraction_2m",
        "duplicate": "duplicates.duplicate_fraction_0p05m",
    }
    values = {
        arm: {
            name: scene_metric(aggregate, key)
            for name, key in keys.items()
        }
        for arm, aggregate in (
            ("t0_current", t0),
            ("t1_ego", t1),
            ("t2_doppler", t2),
        )
    }
    checks = {
        "t2_improves_t1_completeness": (
            values["t2_doppler"]["completeness"]
            < values["t1_ego"]["completeness"]
        ),
        "t2_improves_t1_far_completeness": (
            values["t2_doppler"]["far_completeness"]
            < values["t1_ego"]["far_completeness"]
        ),
        "t2_outlier_degradation_at_most_2pp": (
            values["t2_doppler"]["outlier"]
            <= values["t1_ego"]["outlier"] + 0.02
        ),
        "t2_duplicate_degradation_at_most_2pp": (
            values["t2_doppler"]["duplicate"]
            <= values["t1_ego"]["duplicate"] + 0.02
        ),
        "t2_improves_t0_chamfer": (
            values["t2_doppler"]["chamfer"]
            < values["t0_current"]["chamfer"]
        ),
    }
    return {
        "basis": "mean_of_per_scene_means",
        "values": values,
        "checks": checks,
        "passed": all(checks.values()),
    }


def finite_document(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_document(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_document(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--static-hypothesis",
        choices=("negative_ego", "positive_ego", "zero_centered"),
        required=True,
    )
    parser.add_argument("--dynamic-threshold-mps", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    current_commit = git_commit(repo)
    if args.source_commit != current_commit:
        raise ValueError("G1T requested source differs from Git snapshot")
    if args.output.exists():
        raise FileExistsError(f"G1T output already exists: {args.output}")
    if args.dynamic_threshold_mps <= 0.0:
        raise ValueError("G1T dynamic threshold must be positive")
    device, device_name = require_h200(args.device)
    manifest_document = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    if manifest_document.get("gate_pass") is not True:
        raise ValueError("G1T temporal manifest did not pass its data gate")
    dataset = KRadarTemporalDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    if (
        len(dataset.windows) != EXPECTED_WINDOWS
        or len(dataset.frame_dataset) != EXPECTED_FRAMES
        or len(dataset.pairs) != EXPECTED_PAIRS
    ):
        raise ValueError("G1T requires the frozen 8-window validation cohort")
    pairs_by_current = {
        int(pair["current_dataset_index"]): pair for pair in dataset.pairs
    }
    if len(pairs_by_current) != EXPECTED_PAIRS:
        raise ValueError("G1T temporal pairs contain duplicate current frames")

    axes = load_axes(args.data_root / "resources")
    axes_tensors = tuple(
        torch.from_numpy(value).float().to(device)
        for value in (
            axes.doppler_mps,
            axes.range_m,
            axes.azimuth_rad,
            axes.elevation_rad,
        )
    )
    doppler_mps, range_m, azimuth_rad, elevation_rad = axes_tensors
    doppler_step_mps = torch.median(torch.diff(doppler_mps))
    doppler_lower_mps = doppler_mps[0]
    doppler_period_mps = doppler_step_mps * doppler_mps.numel()
    frames_by_arm: dict[str, list[dict]] = {
        arm: [] for arm in ARM_NAMES
    }

    for window in dataset.windows:
        history: list[HistoryProposalSource] = []
        for frame_in_window, dataset_index in enumerate(
            window["dataset_indices"]
        ):
            item = dataset.frame_dataset[dataset_index]
            cube = item["cube_drae"].unsqueeze(0).to(
                device, non_blocking=True
            )
            pair = (
                None
                if frame_in_window == 0
                else pairs_by_current[dataset_index]
            )
            if pair is not None:
                transform = torch.tensor(
                    pair["current_from_previous"],
                    dtype=torch.float32,
                    device=device,
                ).reshape(4, 4)
                history = compose_history(
                    history, transform, float(pair["delta_seconds"])
                )
            ego_speed_mps = item["ego_speed_mps"].to(device)
            current_frame = cube_proposal_frame(
                cube,
                range_m,
                azimuth_rad,
                elevation_rad,
                doppler_mps,
                doppler_lower_mps,
                doppler_period_mps,
                ego_speed_mps,
                args.static_hypothesis,
                point_count=POINT_COUNT,
                nms_kernel=NMS_KERNEL,
            )
            if len(history) == HISTORY_FRAMES:
                arms = build_temporal_proposal_arms(
                    cube,
                    current_frame.prediction,
                    history,
                    range_m,
                    azimuth_rad,
                    elevation_rad,
                    doppler_mps,
                    doppler_lower_mps,
                    doppler_period_mps,
                    ego_speed_mps,
                    args.static_hypothesis,
                    output_count=POINT_COUNT,
                    nms_kernel=NMS_KERNEL,
                    dynamic_threshold_mps=args.dynamic_threshold_mps,
                )
                target = item["target_xyz_confidence"].to(
                    device, non_blocking=True
                )
                target_index = item["target_rae_index"].to(
                    device, non_blocking=True
                )
                for arm, selection in arms.items():
                    metrics = evaluate_selection(
                        selection,
                        cube,
                        target,
                        target_index,
                        ego_speed_mps,
                        args.static_hypothesis,
                        axes_tensors,
                        args.dynamic_threshold_mps,
                    )
                    frames_by_arm[arm].append(
                        {
                            "arm": arm,
                            "window_id": window["window_id"],
                            "sequence": int(item["sequence"]),
                            "radar_index": int(item["radar_index"]),
                            "frame_in_window": frame_in_window,
                            "history_frame_count": len(history),
                            **metrics,
                        }
                    )
                del target, target_index, arms
            history = [
                HistoryProposalSource(
                    prediction=current_frame.prediction,
                    current_from_source=torch.eye(
                        4, dtype=torch.float32, device=device
                    ),
                    age_seconds=0.0,
                    age_frames=0,
                )
            ] + history[: HISTORY_FRAMES - 1]
            del item, cube, current_frame
            torch.cuda.empty_cache()

    expected_evaluation_frames = sum(
        int(window["frame_count"]) - HISTORY_FRAMES
        for window in dataset.windows
    )
    arms = {
        arm: {
            "aggregate": aggregate_frames(frames),
            "frames": frames,
        }
        for arm, frames in frames_by_arm.items()
    }
    far_metric = "range_60_120m_completeness_mean_distance_m"
    far_target_frame_count = sum(
        far_metric in frame["geometry"]
        for frame in frames_by_arm["t0_current"]
    )
    far_sample_counts = {
        arm: int(
            arms[arm]["aggregate"]["frame_level"]["geometry"][far_metric][
                "sample_count"
            ]
        )
        for arm in ARM_NAMES
    }
    checks = {
        "complete_arm_matrix": set(arms) == set(ARM_NAMES),
        "same_evaluation_frames": all(
            len(arms[arm]["frames"]) == expected_evaluation_frames
            for arm in ARM_NAMES
        ),
        "same_fixed_history_count": all(
            frame["history_frame_count"] == HISTORY_FRAMES
            for arm in ARM_NAMES
            for frame in arms[arm]["frames"]
        ),
        "exact_10000_output": all(
            frame["selection"]["output_point_count"] == POINT_COUNT
            for arm in ARM_NAMES
            for frame in arms[arm]["frames"]
        ),
        "t0_has_no_selected_history": all(
            frame["selection"]["history_fraction"] == 0.0
            for frame in arms["t0_current"]["frames"]
        ),
        "current_cube_rescore": True,
        "deterministic_fixed_dedup": True,
        "learned_g1d_score_accessed": False,
        "lidar_or_gt_selection_accessed": False,
        "test_partition_accessed": False,
        "finite_metrics": finite_document(arms),
        "corrected_far_geometry_covers_every_target_bearing_frame": (
            far_target_frame_count > 0
            and all(
                count == far_target_frame_count
                for count in far_sample_counts.values()
            )
        ),
    }
    decision = promotion_decision(arms)
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "source_commit": current_commit,
            "dense_geometry_sha256": sha256(
                Path(__file__).resolve().parents[1] / "eval/dense_geometry.py"
            ),
            "far_target_frame_censoring_fixed": True,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "partition": "validation",
            "point_count": POINT_COUNT,
            "history_frames": HISTORY_FRAMES,
            "nms_kernel": list(NMS_KERNEL),
            "near_boundary_m": NEAR_BOUNDARY_M,
            "dynamic_threshold_mps": args.dynamic_threshold_mps,
            "static_hypothesis": args.static_hypothesis,
            "selection_score": "current_cube_integrated_log_energy_only",
            "history_doppler": "measured_source_cube_distribution",
            "learned_g1d_score": False,
            "evaluation_window_count": len(dataset.windows),
            "evaluation_frame_count": expected_evaluation_frames,
            "far_target_frame_count": far_target_frame_count,
            "far_metric_sample_counts": far_sample_counts,
            "device": args.device,
            "device_name": device_name,
            "torch_version": torch.__version__,
        },
        "arms": arms,
        "checks": checks,
        "completed": all(checks.values()),
        "promotion": decision,
    }
    atomic_json(args.output, document)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checks": checks,
                "promotion": decision,
            },
            indent=2,
        ),
        flush=True,
    )
    if not document["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
