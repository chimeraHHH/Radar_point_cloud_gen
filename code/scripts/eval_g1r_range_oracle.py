#!/usr/bin/env python3
"""Evaluate the frozen G1R-R0 range-aware candidate-support oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
)
from eval.g1r_range_aware_support import (  # noqa: E402
    ARM_NAMES,
    BASE_SEED_COUNT,
    CANDIDATE_PARENT_QUOTAS,
    EXPORT_COUNT,
    EXPORT_QUOTAS,
    LATERAL_NMS_RADIUS_M,
    ORACLE_ARTIFACT_LABEL,
    PROPOSAL_COUNT,
    RADIAL_NMS_RADIUS_M,
    RANGE_BINS_M,
    SEED_QUOTAS,
    VANILLA_NMS_KERNEL,
    calibrated_cube_energy,
    expand_g1d_candidate_pool,
    fit_range_energy_calibration,
    range_aware_proposal_indices,
    range_count_report,
    select_fixed_quota_support_oracle,
    stable_score_proposal_indices,
)
from eval.rald_guided_query import duplicate_report  # noqa: E402
from eval.temporal_methods import aggregate_flat_reports  # noqa: E402
from models.rald_query_field import (  # noqa: E402
    integrated_log_energy,
    stable_radar_proposals,
)
from scripts.eval_g1f_oracle import (  # noqa: E402
    aggregate_range_support,
    tensor_sha256,
    validate_g1d_run,
)
from scripts.g1b_contract import sha256  # noqa: E402
from scripts.train_g1g_hierarchy import (  # noqa: E402
    FROZEN_MANIFEST_SHA256,
    FROZEN_SCENE_SPLIT_SHA256,
    frame_data_hashes,
    validate_data_contract,
)
from scripts.train_rald_query_field import (  # noqa: E402
    aggregate_scalar_reports,
    verify_source_tree,
)


PROTOCOL = "g1r_r0_range_aware_support_oracle_v1"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FAR_LABEL = "range_60_120m"
FAR_RECALL_KEY = "gt_recall_from_proposals_2p0m"
COMPLETENESS_LIMIT_M = 1.20
FAR_COMPLETENESS_LIMIT_M = 8.0
FAR_RECALL_MINIMUM = 0.30
OUTLIER_LIMIT = 0.25
DUPLICATE_LIMIT = 0.10


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit(repo: Path) -> str:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=repo,
        text=True,
    ).strip()
    if not SOURCE_PATTERN.fullmatch(commit):
        raise RuntimeError("G1R-R0 requires a full Git source commit")
    return commit


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("G1R-R0 requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G1R-R0 is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G1R-R0 requires an H200, got {resolved}")
    return device, resolved


def validate_frozen_data_contract(
    manifest_path: Path,
    scene_split_path: Path,
) -> tuple[dict, dict, dict[str, int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_split = json.loads(scene_split_path.read_text(encoding="utf-8"))
    counts = validate_data_contract(manifest, scene_split)
    actual_hashes = {
        "manifest": sha256(manifest_path),
        "scene_split": sha256(scene_split_path),
    }
    expected_hashes = {
        "manifest": FROZEN_MANIFEST_SHA256,
        "scene_split": FROZEN_SCENE_SPLIT_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(
            "G1R-R0 inputs differ from the frozen G1D 76/24 contract: "
            f"{actual_hashes}"
        )
    split_sequences = {
        partition: {
            int(value)
            for value in scene_split["splits"][partition]["sequences"]
        }
        for partition in ("train", "validation", "test")
    }
    for frame in manifest["frames"]:
        partition = str(frame["partition"])
        sequence = int(frame["sequence"])
        if sequence not in split_sequences[partition]:
            raise ValueError("G1R-R0 frame contradicts the source scene split")
        if sequence in split_sequences["test"]:
            raise ValueError("G1R-R0 development manifest touches a test scene")
    return manifest, scene_split, counts


def _cube_path(data_root: Path, record: dict) -> Path:
    return (
        data_root
        / str(int(record["sequence"]))
        / "radar_tesseract"
        / f"tesseract_{int(record['radar_index']):05d}.mat"
    )


def _cache_path(cache_root: Path, record: dict) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_"
        f"radar_{int(record['radar_index']):05d}.npz"
    )


def source_hashes(repo: Path) -> dict[str, dict[str, str]]:
    relative_paths = (
        "artifacts/idea/literature_agents/"
        "failure_driven_frontier_2026-07-28.md",
        "artifacts/idea/literature_agents/"
        "point_generation_2026-07-28.md",
        "code/eval/g1r_range_aware_support.py",
        "code/scripts/eval_g1r_range_oracle.py",
        "code/tests/test_g1r_range_aware_support.py",
        "code/tests/test_eval_g1r_range_oracle.py",
        "code/eval/g1f_candidate_support.py",
        "code/scripts/eval_g1f_oracle.py",
        "code/eval/dense_geometry.py",
        "code/eval/rald_guided_query.py",
        "code/eval/temporal_methods.py",
        "code/models/rald_query_field.py",
        "code/models/cube_cycle.py",
        "code/cube_dense/dataset.py",
        "code/cube_dense/kradar.py",
        "code/scripts/train_rald_query_field.py",
        "code/scripts/train_g1g_hierarchy.py",
        "code/scripts/g1b_contract.py",
    )
    paths = [repo / relative for relative in relative_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"G1R-R0 source map is incomplete: {missing}")
    return {
        str(path.resolve()): {"sha256": sha256(path)}
        for path in paths
    }


def axis_data_hashes(data_root: Path) -> dict[str, dict[str, str]]:
    paths = (
        data_root / "resources" / "info_arr.mat",
        data_root / "resources" / "arr_doppler.mat",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"G1R-R0 axis data is incomplete: {missing}")
    return {
        str(path.resolve()): {"sha256": sha256(path)}
        for path in paths
    }


def frame_hash_lookup(frame_hashes: list[dict]) -> dict[tuple[int, int], dict]:
    result = {}
    for record in frame_hashes:
        key = (int(record["sequence"]), int(record["radar_index"]))
        if key in result:
            raise ValueError("G1R-R0 data hashes contain duplicate frames")
        result[key] = record
    return result


def calibration_document(calibration) -> dict[str, Any]:
    median = calibration.median.detach().cpu().contiguous()
    mad = calibration.mad.detach().cpu().contiguous()
    sample_count = calibration.sample_count.detach().cpu().contiguous()
    digest = hashlib.sha256()
    for values in (median, mad, sample_count):
        digest.update(values.numpy().tobytes())
    return {
        "source_partition": "train",
        "validation_used": False,
        "training_frame_count": calibration.training_frame_count,
        "range_cell_count": int(median.numel()),
        "samples_per_range_cell": sample_count.tolist(),
        "median_integrated_log_energy": median.tolist(),
        "mad_integrated_log_energy": mad.tolist(),
        "profile_sha256": digest.hexdigest(),
    }


@torch.inference_mode()
def fit_training_profile(
    dataset: KRadarCubeDataset,
    device: torch.device,
) -> Any:
    energies = []
    for index in range(len(dataset)):
        item = dataset[index]
        cube = item["cube_drae"].unsqueeze(0).to(
            device,
            non_blocking=True,
        )
        energy = integrated_log_energy(cube)[0].float().cpu()
        energies.append(energy)
        del item, cube, energy
    profile = fit_range_energy_calibration(energies)
    del energies
    torch.cuda.empty_cache()
    return profile


def _range_parent_counts(parent_bin: torch.Tensor) -> dict[str, int]:
    return {
        label: int((parent_bin == index).sum().item())
        for index, (label, _, _) in enumerate(RANGE_BINS_M)
    }


def _arm_endpoint(arm: dict, endpoint: str) -> float:
    if endpoint == "completeness_mean_distance_m":
        return float(arm["geometry"][endpoint])
    if endpoint == "far_completeness_60_120m":
        return float(
            arm["geometry"][
                "range_60_120m_completeness_mean_distance_m"
            ]
        )
    if endpoint == "outlier_fraction_2m":
        return float(arm["geometry"][endpoint])
    if endpoint == "duplicate_fraction_0p05m":
        return float(arm["duplicates"][endpoint])
    if endpoint == "far_full_pool_weighted_gt_recall_2m":
        return float(
            arm["per_range_support"][FAR_LABEL][FAR_RECALL_KEY]
        )
    raise KeyError(endpoint)


PAIRED_ENDPOINTS = (
    "completeness_mean_distance_m",
    "far_completeness_60_120m",
    "outlier_fraction_2m",
    "duplicate_fraction_0p05m",
    "far_full_pool_weighted_gt_recall_2m",
)


def paired_deltas(arms: dict[str, dict]) -> dict[str, dict[str, float]]:
    vanilla = arms["vanilla"]
    return {
        name: {
            endpoint: _arm_endpoint(arm, endpoint)
            - _arm_endpoint(vanilla, endpoint)
            for endpoint in PAIRED_ENDPOINTS
        }
        for name, arm in arms.items()
        if name != "vanilla"
    }


@torch.inference_mode()
def evaluate_frame(
    item: dict,
    axes_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    calibration,
    device: torch.device,
    data_hash: dict,
    distance_chunk_size: int,
) -> dict:
    range_m, azimuth_rad, elevation_rad = axes_tensors
    cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
    target = item["target_xyz_confidence"].to(device, non_blocking=True)
    target_xyz = target[:, :3].float()
    target_weight = target[:, 3].float()
    z_score = calibrated_cube_energy(cube, calibration)
    proposal_indices = {
        "vanilla": stable_radar_proposals(
            cube,
            seed_count=BASE_SEED_COUNT,
            nms_kernel=VANILLA_NMS_KERNEL,
        ).flat_index,
        "z_only": stable_score_proposal_indices(
            z_score,
            seed_count=BASE_SEED_COUNT,
            nms_kernel=VANILLA_NMS_KERNEL,
        ),
        "range_aware": range_aware_proposal_indices(
            z_score,
            range_m,
            azimuth_rad,
            elevation_rad,
        ),
    }
    arms = {}
    for name in ARM_NAMES:
        pool = expand_g1d_candidate_pool(
            cube,
            proposal_indices[name],
            range_m,
            azimuth_rad,
            elevation_rad,
        )
        selection = select_fixed_quota_support_oracle(
            pool,
            target_xyz,
            target_weight,
            distance_chunk_size=distance_chunk_size,
        )
        geometry = geometry_report(
            selection.selected_xyz_m,
            target_xyz,
            target_weight=target_weight,
            chunk_size=distance_chunk_size,
        )
        duplicates = duplicate_report(selection.selected_xyz_m)
        parent_counts = _range_parent_counts(pool.parent_range_bin)
        arms[name] = {
            "candidate_count": int(pool.xyz_m.shape[0]),
            "unique_candidate_index_count": int(
                torch.unique(pool.candidate_indices).numel()
            ),
            "unique_candidate_coordinate_count": int(
                torch.unique(pool.xyz_m, dim=0).shape[0]
            ),
            "candidate_parent_range_count": parent_counts,
            "candidate_actual_range_count": range_count_report(pool.xyz_m),
            "selected_count": int(selection.selected_xyz_m.shape[0]),
            "unique_selected_index_count": int(
                torch.unique(selection.selected_candidate_indices).numel()
            ),
            "unique_selected_coordinate_count": int(
                torch.unique(selection.selected_xyz_m, dim=0).shape[0]
            ),
            "selected_actual_range_count": range_count_report(
                selection.selected_xyz_m
            ),
            "proposal_flat_index_sha256": tensor_sha256(
                pool.proposal_flat_index
            ),
            "selected_candidate_indices_sha256": tensor_sha256(
                selection.selected_candidate_indices
            ),
            "geometry": geometry,
            "duplicates": duplicates,
            "per_range_support": selection.per_range_support,
            "ground_truth_used_for_selection": True,
            "geometry_uses_original_target_confidence": True,
        }
    expected_parent_counts = {
        label: quota
        for (label, _, _), quota in zip(
            RANGE_BINS_M,
            CANDIDATE_PARENT_QUOTAS,
        )
    }
    if arms["range_aware"]["candidate_parent_range_count"] != (
        expected_parent_counts
    ):
        raise AssertionError("G1R-R0 lost its frozen parent candidate quotas")
    if arms["range_aware"]["candidate_actual_range_count"] != (
        expected_parent_counts
    ):
        raise AssertionError("G1R-R0 lost its physical candidate range quotas")
    expected_selected_counts = {
        label: quota
        for (label, _, _), quota in zip(RANGE_BINS_M, EXPORT_QUOTAS)
    }
    for arm in arms.values():
        if arm["selected_actual_range_count"] != expected_selected_counts:
            raise AssertionError("G1R-R0 lost its frozen selected range quotas")
    return {
        "partition": str(item["partition"]),
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
        "data_hashes": data_hash,
        "arms": arms,
        "paired_delta_vs_vanilla": paired_deltas(arms),
        "test_accessed": False,
    }


def _mean_numeric_reports(reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for report in reports
            for key, value in report.items()
            if isinstance(value, (int, float))
        }
    )
    return {
        key: float(np.mean([report[key] for report in reports if key in report]))
        for key in keys
    }


def _scene_arm_reports(
    frames: list[dict],
    arm_name: str,
) -> tuple[list[dict], dict[str, dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        grouped[int(frame["sequence"])].append(frame["arms"][arm_name])
    per_scene = {}
    for sequence, arms in sorted(grouped.items()):
        range_reports = []
        for label, _, _ in RANGE_BINS_M:
            range_reports.append(
                (
                    label,
                    _mean_numeric_reports(
                        [arm["per_range_support"][label] for arm in arms]
                    ),
                )
            )
        per_scene[str(sequence)] = {
            "geometry": _mean_numeric_reports(
                [arm["geometry"] for arm in arms]
            ),
            "duplicates": _mean_numeric_reports(
                [arm["duplicates"] for arm in arms]
            ),
            "per_range_support": dict(range_reports),
        }
    return list(per_scene.values()), per_scene


def aggregate_arm(frames: list[dict], arm_name: str) -> dict:
    arm_frames = [frame["arms"][arm_name] for frame in frames]
    frame_first = {
        "geometry": aggregate_geometry_reports(
            [arm["geometry"] for arm in arm_frames]
        ),
        "duplicates": aggregate_scalar_reports(
            [arm["duplicates"] for arm in arm_frames]
        ),
        "per_range_support": aggregate_range_support(arm_frames),
    }
    scene_reports, per_scene = _scene_arm_reports(frames, arm_name)
    scene_range = {}
    for label, _, _ in RANGE_BINS_M:
        scene_range[label] = aggregate_flat_reports(
            [scene["per_range_support"][label] for scene in scene_reports]
        )
    scene_first = {
        "scene_count": len(scene_reports),
        "geometry": aggregate_geometry_reports(
            [scene["geometry"] for scene in scene_reports]
        ),
        "duplicates": aggregate_scalar_reports(
            [scene["duplicates"] for scene in scene_reports]
        ),
        "per_range_support": scene_range,
        "per_scene": per_scene,
    }
    return {
        "frame_count": len(frames),
        "scene_count": len(scene_reports),
        "frame_first": frame_first,
        "scene_first": scene_first,
    }


def aggregate_paired_deltas(frames: list[dict]) -> dict:
    result = {}
    for arm_name in ("z_only", "range_aware"):
        reports = [
            frame["paired_delta_vs_vanilla"][arm_name]
            for frame in frames
        ]
        grouped: dict[int, list[dict]] = defaultdict(list)
        for frame, report in zip(frames, reports):
            grouped[int(frame["sequence"])].append(report)
        scene_reports = [
            _mean_numeric_reports(values)
            for _, values in sorted(grouped.items())
        ]
        result[arm_name] = {
            "frame_first": aggregate_scalar_reports(reports),
            "scene_first": aggregate_scalar_reports(scene_reports),
        }
    return result


def stage0_decision(metrics: dict, frames: list[dict]) -> dict[str, Any]:
    range_aware = metrics["range_aware"]["frame_first"]
    geometry = range_aware["geometry"]
    duplicates = range_aware["duplicates"]
    far_support = range_aware["per_range_support"][FAR_LABEL]
    far_geometry = geometry.get(
        "range_60_120m_completeness_mean_distance_m"
    )
    values = {
        "far_full_pool_weighted_gt_recall_2m_mean": float(
            far_support[FAR_RECALL_KEY]["mean"]
        ),
        "oracle_completeness_median_m": float(
            geometry["completeness_mean_distance_m"]["median"]
        ),
        "oracle_far_completeness_mean_m": (
            float(far_geometry["mean"])
            if far_geometry is not None
            else float("inf")
        ),
        "oracle_outlier_fraction_mean": float(
            geometry["outlier_fraction_2m"]["mean"]
        ),
        "oracle_duplicate_fraction_mean": float(
            duplicates["duplicate_fraction_0p05m"]["mean"]
        ),
    }
    checks = {
        "far_recall_at_least_30pct": (
            values["far_full_pool_weighted_gt_recall_2m_mean"]
            >= FAR_RECALL_MINIMUM
        ),
        "oracle_median_completeness_at_most_1p20m": (
            values["oracle_completeness_median_m"] <= COMPLETENESS_LIMIT_M
        ),
        "oracle_far_completeness_at_most_8m": (
            values["oracle_far_completeness_mean_m"]
            <= FAR_COMPLETENESS_LIMIT_M
        ),
        "oracle_outlier_at_most_25pct": (
            values["oracle_outlier_fraction_mean"] <= OUTLIER_LIMIT
        ),
        "oracle_duplicate_at_most_10pct": (
            values["oracle_duplicate_fraction_mean"] <= DUPLICATE_LIMIT
        ),
        "all_arms_exact_32000_candidates": all(
            arm["candidate_count"] == PROPOSAL_COUNT
            for frame in frames
            for arm in frame["arms"].values()
        ),
        "all_arms_exact_10000_outputs": all(
            arm["selected_count"] == EXPORT_COUNT
            for frame in frames
            for arm in frame["arms"].values()
        ),
        "range_aware_parent_candidate_quota_exact": all(
            tuple(
                frame["arms"]["range_aware"]["candidate_parent_range_count"][
                    label
                ]
                for label, _, _ in RANGE_BINS_M
            )
            == CANDIDATE_PARENT_QUOTAS
            for frame in frames
        ),
        "range_aware_actual_candidate_quota_exact": all(
            tuple(
                frame["arms"]["range_aware"]["candidate_actual_range_count"][
                    label
                ]
                for label, _, _ in RANGE_BINS_M
            )
            == CANDIDATE_PARENT_QUOTAS
            for frame in frames
        ),
        "all_selected_range_quotas_exact": all(
            tuple(
                arm["selected_actual_range_count"][label]
                for label, _, _ in RANGE_BINS_M
            )
            == EXPORT_QUOTAS
            for frame in frames
            for arm in frame["arms"].values()
        ),
    }
    passed = all(checks.values())
    return {
        "values": values,
        "thresholds": {
            "far_full_pool_weighted_gt_recall_2m_minimum": FAR_RECALL_MINIMUM,
            "oracle_completeness_median_m_maximum": COMPLETENESS_LIMIT_M,
            "oracle_far_completeness_mean_m_maximum": (
                FAR_COMPLETENESS_LIMIT_M
            ),
            "oracle_outlier_fraction_mean_maximum": OUTLIER_LIMIT,
            "oracle_duplicate_fraction_mean_maximum": DUPLICATE_LIMIT,
        },
        "checks": checks,
        "passed": passed,
        "training_authorized": passed,
        "decision": (
            "authorize_frozen_10_epoch_r0_training"
            if passed
            else "forbid_r0_training_oracle_gate_failed"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--g1d-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distance-chunk-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"G1R-R0 output already exists: {args.output}")
    if not SOURCE_PATTERN.fullmatch(args.source_commit):
        raise ValueError("G1R-R0 source commit must be a full lowercase Git SHA")
    if args.distance_chunk_size <= 0:
        raise ValueError("G1R-R0 distance chunk size must be positive")
    repo = Path(__file__).resolve().parents[2]
    current_commit = git_commit(repo)
    if args.source_commit != current_commit:
        raise ValueError("G1R-R0 requested source differs from Git snapshot")
    verify_source_tree(repo, args.source_commit)
    manifest, _, counts = validate_frozen_data_contract(
        args.manifest,
        args.scene_split,
    )
    g1d = validate_g1d_run(args.g1d_run)
    device, device_name = require_h200(args.device)

    all_frame_hashes = frame_data_hashes(
        manifest["frames"],
        args.data_root,
        args.cache_root,
    )
    hash_lookup = frame_hash_lookup(all_frame_hashes)
    axes = load_axes(args.data_root / "resources")
    axes_tensors = tuple(
        torch.from_numpy(values).float().to(device)
        for values in (
            axes.range_m,
            axes.azimuth_rad,
            axes.elevation_rad,
        )
    )
    training_dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train",),
    )
    validation_dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    if len(training_dataset) != 76 or len(validation_dataset) != 24:
        raise AssertionError("G1R-R0 dataset changed after contract validation")
    calibration = fit_training_profile(training_dataset, device)
    frames = []
    for index in range(len(validation_dataset)):
        item = validation_dataset[index]
        key = (int(item["sequence"]), int(item["radar_index"]))
        frame = evaluate_frame(
            item,
            axes_tensors,
            calibration,
            device,
            hash_lookup[key],
            args.distance_chunk_size,
        )
        frames.append(frame)
        del item
        torch.cuda.empty_cache()

    metrics = {
        arm_name: aggregate_arm(frames, arm_name)
        for arm_name in ARM_NAMES
    }
    decision = stage0_decision(metrics, frames)
    document = {
        "protocol": PROTOCOL,
        "artifact_label": {
            "label": ORACLE_ARTIFACT_LABEL,
            "diagnostic": True,
            "unattainable": True,
            "ground_truth_used_for_selection": True,
            "eligible_as_method_result": False,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": current_commit,
        "device": device_name,
        "frozen_design": {
            "arms": list(ARM_NAMES),
            "integrated_energy": "sum_doppler_log1p_nonnegative_power",
            "calibration": "exact_train_only_per_range_cell_median_mad",
            "vanilla": "g1d_stable_energy_nms_5x5x3",
            "z_only": "calibrated_energy_with_g1d_compatible_nms_5x5x3",
            "range_aware": (
                "calibrated_energy_fixed_range_quota_physical_nms_"
                "template_safe_parents"
            ),
            "physical_nms_lateral_radius_m": LATERAL_NMS_RADIUS_M,
            "physical_nms_radial_radius_m": RADIAL_NMS_RADIUS_M,
            "seed_quotas_0_30_30_60_60_120m": list(SEED_QUOTAS),
            "candidate_parent_quotas_0_30_30_60_60_120m": list(
                CANDIDATE_PARENT_QUOTAS
            ),
            "selected_quotas_0_30_30_60_60_120m": list(EXPORT_QUOTAS),
            "candidate_count": PROPOSAL_COUNT,
            "export_count": EXPORT_COUNT,
            "oracle_selection": "reused_g1f_gt_support_oracle",
            "geometry_metrics": "reused_eval_dense_geometry",
            "duplicate_metric": "reused_eval_rald_guided_query",
            "test_accessed": False,
        },
        "calibration": calibration_document(calibration),
        "metrics": metrics,
        "paired_delta_vs_vanilla": aggregate_paired_deltas(frames),
        "frames": frames,
        "decision": decision,
        "provenance": {
            "source_hashes": source_hashes(repo),
            "data_hashes": {
                "manifest": {
                    "path": str(args.manifest.resolve()),
                    "sha256": sha256(args.manifest),
                },
                "scene_split": {
                    "path": str(args.scene_split.resolve()),
                    "sha256": sha256(args.scene_split),
                },
                "axes": axis_data_hashes(args.data_root),
                "frames": all_frame_hashes,
                "g1d": g1d,
            },
            "data_contract": counts,
            "partitions_read": ["train", "validation"],
            "calibration_partition": "train",
            "evaluation_partition": "validation",
            "test_accessed": False,
            "output_refuses_overwrite": True,
            "torch_version": torch.__version__,
        },
    }
    atomic_json(args.output, document)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "output": str(args.output.resolve()),
                "decision": decision,
            },
            indent=2,
        )
    )
    if not decision["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
