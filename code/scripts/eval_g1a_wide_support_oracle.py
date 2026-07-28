#!/usr/bin/env python3
"""Evaluate the frozen R-A0 RaLD-wide support oracle on 24 validation frames."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
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
from eval.g1a_wide_support import (  # noqa: E402
    EXPORT_COUNT,
    ORACLE_ARTIFACT_LABEL,
    RADAR_QUERY_COUNT,
    RANDOM_QUERY_COUNT,
    RANGE_STRATA_M,
    RAW_QUERY_COUNT,
    SOURCE_LABELS,
    build_wide_query_domain,
    diagnostic_artifact_label,
    select_wide_support_oracle,
)
from eval.g1f_candidate_support import (  # noqa: E402
    PROPOSAL_COUNT as G1F_PROPOSAL_COUNT,
    select_candidate_support_oracle,
)
from eval.rald_guided_query import duplicate_report  # noqa: E402
from eval.temporal_methods import aggregate_flat_reports  # noqa: E402
from scripts.eval_g1f_oracle import (  # noqa: E402
    coarse_candidate_pool,
    validate_g1d_run,
)
from scripts.g1b_contract import sha256  # noqa: E402


PROTOCOL = "g1a_ra0_wide_support_oracle_v1"
FORMAL_SEED = 20260716
FORMAL_TRAIN_COUNT = 76
FORMAL_VALIDATION_COUNT = 24
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
OFFICIAL_RALD_COMMIT = "ffec4b41241391734b1eda5c093de843c909eb8e"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CANDIDATE_DISTANCE_CHUNK = 8_192
TARGET_DISTANCE_CHUNK = 4_096
RA0_THRESHOLDS = {
    "chamfer_median_m": 2.50,
    "outlier_fraction_2m_mean": 0.25,
    "completeness_median_m": 0.65,
    "far_completeness_mean_m": 8.0,
    "duplicate_fraction_mean": 0.10,
    "far_pool_recall_2m_mean": 0.50,
    "point_count": 10_000,
}


def atomic_json_exclusive(path: Path, document: dict) -> None:
    """Publish JSON without overwriting an existing result, including races."""

    if path.exists():
        raise FileExistsError(f"R-A0 refuses to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"R-A0 refuses to overwrite existing output: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> str:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("R-A0 source commit must be a full lowercase Git SHA")
    current = git_output(repo, "rev-parse", "HEAD")
    if current != source_commit:
        raise ValueError("R-A0 source commit differs from checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"R-A0 formal source worktree is dirty: {dirty}")
    return current


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("R-A0 requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("R-A0 is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"R-A0 requires an H200, got {resolved}")
    return device, resolved


def tensor_sha256(values: torch.Tensor) -> str:
    contiguous = values.detach().cpu().contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def validate_data_contract(manifest: dict, scene_split: dict) -> dict[str, int]:
    if scene_split.get("gate_pass") is not True:
        raise ValueError("R-A0 scene split did not pass its leakage gate")
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("R-A0 manifest must contain a frame list")
    counts = Counter(frame.get("partition") for frame in frames)
    expected = {
        "train": FORMAL_TRAIN_COUNT,
        "validation": FORMAL_VALIDATION_COUNT,
    }
    if dict(counts) != expected:
        raise ValueError(
            f"R-A0 requires the frozen 76/24 development manifest, got {dict(counts)}"
        )
    if any(frame.get("partition") == "test" for frame in frames):
        raise ValueError("R-A0 development manifest must not contain test frames")
    identities = {
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in frames
    }
    if len(identities) != len(frames):
        raise ValueError("R-A0 manifest contains duplicate frame identities")
    return {
        "manifest_frame_count": len(frames),
        "train_frame_count": counts["train"],
        "validation_frame_count": counts["validation"],
        "test_frame_count": 0,
    }


def validate_frozen_input_hashes(
    manifest_path: Path,
    scene_split_path: Path,
) -> dict[str, str]:
    actual = {
        "manifest": sha256(manifest_path),
        "scene_split": sha256(scene_split_path),
    }
    expected = {
        "manifest": FROZEN_MANIFEST_SHA256,
        "scene_split": FROZEN_SCENE_SPLIT_SHA256,
    }
    if actual != expected:
        raise ValueError(f"R-A0 inputs differ from the frozen data contract: {actual}")
    return actual


def _cache_path(cache_root: Path, record: dict) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_"
        f"radar_{int(record['radar_index']):05d}.npz"
    )


def _cube_path(data_root: Path, record: dict) -> Path:
    return (
        data_root
        / str(int(record["sequence"]))
        / "radar_tesseract"
        / f"tesseract_{int(record['radar_index']):05d}.mat"
    )


def frame_data_hashes(
    frames: list[dict],
    data_root: Path,
    cache_root: Path,
) -> list[dict[str, Any]]:
    result = []
    for record in sorted(
        frames,
        key=lambda item: (int(item["sequence"]), int(item["radar_index"])),
    ):
        cube_path = _cube_path(data_root, record)
        cache_path = _cache_path(cache_root, record)
        if not cube_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(
                f"R-A0 validation data is incomplete: {cube_path}, {cache_path}"
            )
        result.append(
            {
                "partition": str(record["partition"]),
                "sequence": int(record["sequence"]),
                "radar_index": int(record["radar_index"]),
                "cube": str(cube_path.resolve()),
                "cube_sha256": sha256(cube_path),
                "dense_cache": str(cache_path.resolve()),
                "dense_cache_sha256": sha256(cache_path),
            }
        )
    return result


def resource_data_hashes(data_root: Path) -> dict[str, dict[str, str]]:
    resources = data_root / "resources"
    result = {}
    for name in ("info_arr.mat", "arr_doppler.mat"):
        path = resources / name
        if not path.is_file():
            raise FileNotFoundError(f"R-A0 axis resource is missing: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
    return result


def source_hashes(repo: Path) -> dict[str, dict[str, str]]:
    relative_paths = (
        "code/eval/g1a_wide_support.py",
        "code/scripts/eval_g1a_wide_support_oracle.py",
        "code/eval/g1f_candidate_support.py",
        "code/scripts/eval_g1f_oracle.py",
        "code/eval/dense_geometry.py",
        "code/eval/rald_guided_query.py",
        "code/models/cube_cycle.py",
        "code/models/rald_matched.py",
        "code/models/rald_query_field.py",
        "code/cube_dense/dataset.py",
        "code/cube_dense/kradar.py",
    )
    result = {}
    for relative in relative_paths:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"R-A0 transitive source is missing: {path}")
        result[relative] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
    return result


def _source_counts(source_codes: torch.Tensor) -> dict[str, int]:
    return {
        label: int((source_codes == code).sum().item())
        for code, label in SOURCE_LABELS.items()
    }


def _g1f_overall_support(per_range: dict) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    target_mass = sum(
        float(report["target_effective_count"]) for report in per_range.values()
    )
    candidate_count = sum(int(report["candidate_count"]) for report in per_range.values())
    selected_count = sum(int(report["selected_count"]) for report in per_range.values())
    result["candidate_count"] = candidate_count
    result["target_effective_count"] = target_mass
    result["selected_count"] = selected_count
    for threshold in (0.5, 1.0, 2.0):
        suffix = str(threshold).replace(".", "p")
        for family, denominator_key in (
            ("gt_recall_from_proposals", "target_effective_count"),
            ("gt_recall_from_selected", "target_effective_count"),
            ("proposal_to_gt_support_fraction", "candidate_count"),
            ("selected_to_gt_support_fraction", "selected_count"),
        ):
            key = f"{family}_{suffix}m"
            weighted = 0.0
            denominator = 0.0
            for report in per_range.values():
                value = report.get(key)
                weight = float(report[denominator_key])
                if value is not None and weight > 0.0:
                    weighted += float(value) * weight
                    denominator += weight
            result[key] = weighted / denominator if denominator > 0.0 else None
    return result


def _paired_report(wide: dict, g1f: dict) -> dict[str, float]:
    wide_far = wide["per_range_support"]["range_60_120m"]
    g1f_far = g1f["per_range_support"]["range_60_120m"]
    return {
        "chamfer_wide_minus_g1f_m": (
            float(wide["geometry"]["chamfer_m"])
            - float(g1f["geometry"]["chamfer_m"])
        ),
        "completeness_wide_minus_g1f_m": (
            float(wide["geometry"]["completeness_mean_distance_m"])
            - float(g1f["geometry"]["completeness_mean_distance_m"])
        ),
        "far_completeness_wide_minus_g1f_m": (
            float(
                wide["geometry"][
                    "range_60_120m_completeness_mean_distance_m"
                ]
            )
            - float(
                g1f["geometry"][
                    "range_60_120m_completeness_mean_distance_m"
                ]
            )
        ),
        "outlier_wide_minus_g1f_fraction": (
            float(wide["geometry"]["outlier_fraction_2m"])
            - float(g1f["geometry"]["outlier_fraction_2m"])
        ),
        "duplicate_wide_minus_g1f_fraction": (
            float(wide["duplicates"]["duplicate_fraction_0p05m"])
            - float(g1f["duplicates"]["duplicate_fraction_0p05m"])
        ),
        "overall_pool_recall_2m_wide_minus_g1f": (
            float(wide["overall_support"]["gt_recall_from_pool_2p0m"])
            - float(
                g1f["overall_support"]["gt_recall_from_proposals_2p0m"]
            )
        ),
        "far_pool_recall_2m_wide_minus_g1f": (
            float(wide_far["gt_recall_from_pool_2p0m"])
            - float(g1f_far["gt_recall_from_proposals_2p0m"])
        ),
    }


@torch.inference_mode()
def evaluate_frame(
    item: dict,
    axes,
    device: torch.device,
    proposal_cache: dict[tuple[int, int], torch.Tensor],
) -> dict:
    cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
    target = item["target_xyz_confidence"].to(device, non_blocking=True)
    range_m = torch.as_tensor(axes.range_m, device=device, dtype=torch.float32)
    azimuth_rad = torch.as_tensor(
        axes.azimuth_rad,
        device=device,
        dtype=torch.float32,
    )
    elevation_rad = torch.as_tensor(
        axes.elevation_rad,
        device=device,
        dtype=torch.float32,
    )
    domain = build_wide_query_domain(
        cube,
        range_m,
        azimuth_rad,
        elevation_rad,
        base_seed=FORMAL_SEED,
        sequence=int(item["sequence"]),
        radar_index=int(item["radar_index"]),
    )
    wide_selection = select_wide_support_oracle(
        domain,
        target[:, :3].float(),
        target_weight=target[:, 3].float(),
        candidate_chunk_size=CANDIDATE_DISTANCE_CHUNK,
        target_chunk_size=TARGET_DISTANCE_CHUNK,
    )
    wide_geometry = geometry_report(
        wide_selection.selected_xyz_m,
        target[:, :3].float(),
        target_weight=target[:, 3].float(),
        chunk_size=1_024,
    )
    wide_duplicates = duplicate_report(wide_selection.selected_xyz_m)
    wide = {
        "artifact_label": wide_selection.artifact_label,
        "domain": domain.report,
        "domain_coordinates_rae_sha256": tensor_sha256(domain.coordinates_rae),
        "domain_xyz_sha256": tensor_sha256(domain.xyz_m),
        "domain_candidate_ids_sha256": tensor_sha256(domain.candidate_ids),
        "domain_source_codes_sha256": tensor_sha256(domain.source_codes),
        "selected_candidate_ids_sha256": tensor_sha256(
            wide_selection.selected_candidate_ids
        ),
        "selected_count": int(wide_selection.selected_candidate_ids.numel()),
        "unique_selected_candidate_id_count": int(
            torch.unique(wide_selection.selected_candidate_ids).numel()
        ),
        "selected_source_counts": _source_counts(
            wide_selection.selected_source_codes
        ),
        "geometry": wide_geometry,
        "duplicates": wide_duplicates,
        "overall_support": wide_selection.overall_support,
        "per_range_support": wide_selection.per_range_support,
        "ground_truth_used_for_proposal_generation": False,
        "ground_truth_used_for_selection": True,
    }

    g1f_xyz, g1f_indices, proposal_flat_index = coarse_candidate_pool(
        item,
        cube,
        axes,
        proposal_cache,
    )
    g1f_selection = select_candidate_support_oracle(
        g1f_xyz.float(),
        g1f_indices,
        target[:, :3].float(),
        target_weight=target[:, 3].float(),
        distance_chunk_size=1_024,
    )
    g1f_geometry = geometry_report(
        g1f_selection.selected_xyz_m,
        target[:, :3].float(),
        target_weight=target[:, 3].float(),
        chunk_size=1_024,
    )
    g1f = {
        "artifact_label": g1f_selection.artifact_label,
        "candidate_count": int(g1f_indices.numel()),
        "selected_count": int(g1f_selection.selected_candidate_indices.numel()),
        "unique_selected_candidate_id_count": int(
            torch.unique(g1f_selection.selected_candidate_indices).numel()
        ),
        "candidate_xyz_sha256": tensor_sha256(g1f_xyz),
        "proposal_flat_index_sha256": tensor_sha256(proposal_flat_index),
        "selected_candidate_ids_sha256": tensor_sha256(
            g1f_selection.selected_candidate_indices
        ),
        "geometry": g1f_geometry,
        "duplicates": duplicate_report(g1f_selection.selected_xyz_m),
        "overall_support": _g1f_overall_support(
            g1f_selection.per_range_support
        ),
        "per_range_support": g1f_selection.per_range_support,
        "ground_truth_used_for_proposal_generation": False,
        "ground_truth_used_for_selection": True,
    }
    result = {
        "partition": str(item["partition"]),
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
        "wide": wide,
        "g1f_32k": g1f,
        "paired": _paired_report(wide, g1f),
        "test_accessed": False,
    }
    del cube, target, domain, wide_selection, g1f_selection
    torch.cuda.empty_cache()
    return result


def _mean_numeric_reports(reports: list[dict]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for report in reports
            for key, value in report.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        key: float(
            np.mean(
                [
                    float(report[key])
                    for report in reports
                    if isinstance(report.get(key), (int, float))
                    and not isinstance(report.get(key), bool)
                ]
            )
        )
        for key in keys
    }


def _aggregate_range_support(reports: list[dict]) -> dict:
    labels = sorted({label for report in reports for label in report})
    return {
        label: aggregate_flat_reports(
            [
                {
                    key: float(value)
                    for key, value in report[label].items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value is not None
                }
                for report in reports
                if label in report
            ]
        )
        for label in labels
    }


def aggregate_method(frames: list[dict], method: str) -> dict:
    methods = [frame[method] for frame in frames]
    result = {
        "geometry": aggregate_geometry_reports(
            [item["geometry"] for item in methods]
        ),
        "duplicates": aggregate_flat_reports(
            [item["duplicates"] for item in methods]
        ),
        "overall_support": aggregate_flat_reports(
            [item["overall_support"] for item in methods]
        ),
        "per_range_support": _aggregate_range_support(
            [item["per_range_support"] for item in methods]
        ),
    }
    if method == "wide":
        result["domain"] = aggregate_flat_reports(
            [
                {
                    key: float(value)
                    for key, value in item["domain"].items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                }
                for item in methods
            ]
        )
    return result


def _scene_mean_method(frames: list[dict], method: str) -> dict:
    items = [frame[method] for frame in frames]
    labels = sorted(
        {
            label
            for item in items
            for label in item["per_range_support"]
        }
    )
    result = {
        "geometry": _mean_numeric_reports(
            [item["geometry"] for item in items]
        ),
        "duplicates": _mean_numeric_reports(
            [item["duplicates"] for item in items]
        ),
        "overall_support": _mean_numeric_reports(
            [item["overall_support"] for item in items]
        ),
        "per_range_support": {
            label: _mean_numeric_reports(
                [
                    item["per_range_support"][label]
                    for item in items
                    if label in item["per_range_support"]
                ]
            )
            for label in labels
        },
    }
    if method == "wide":
        result["domain"] = _mean_numeric_reports(
            [item["domain"] for item in items]
        )
    return result


def aggregate_evaluation(frames: list[dict]) -> dict:
    if len(frames) != FORMAL_VALIDATION_COUNT:
        raise ValueError("R-A0 aggregation requires all 24 validation frames")
    frame_first = {
        "frame_count": len(frames),
        "wide": aggregate_method(frames, "wide"),
        "g1f_32k": aggregate_method(frames, "g1f_32k"),
        "paired": aggregate_flat_reports([frame["paired"] for frame in frames]),
    }
    grouped: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        grouped[int(frame["sequence"])].append(frame)
    scene_records = {
        str(sequence): {
            "wide": _scene_mean_method(records, "wide"),
            "g1f_32k": _scene_mean_method(records, "g1f_32k"),
            "paired": _mean_numeric_reports(
                [record["paired"] for record in records]
            ),
            "frame_count": len(records),
        }
        for sequence, records in sorted(grouped.items())
    }
    scene_pseudo_frames = [
        {
            "wide": record["wide"],
            "g1f_32k": record["g1f_32k"],
        }
        for record in scene_records.values()
    ]
    scene_first = {
        "scene_count": len(scene_records),
        "wide": aggregate_method(scene_pseudo_frames, "wide"),
        "g1f_32k": aggregate_method(scene_pseudo_frames, "g1f_32k"),
        "paired": aggregate_flat_reports(
            [record["paired"] for record in scene_records.values()]
        ),
        "per_scene": scene_records,
    }
    return {
        "frame_first": frame_first,
        "scene_first": scene_first,
    }


def gate_checks(metrics: dict, frames: list[dict]) -> dict[str, bool]:
    """Apply the report's inclusive R-A0 gates without relaxation."""

    frame_first = metrics["frame_first"]["wide"]
    geometry = frame_first["geometry"]
    duplicates = frame_first["duplicates"]
    far_geometry = geometry.get(
        "range_60_120m_completeness_mean_distance_m"
    )
    far_support = frame_first["per_range_support"].get("range_60_120m")
    return {
        "raw_pool_exact_500k_random_plus_700k_radar": all(
            frame["wide"]["domain"]["raw_query_count"] == RAW_QUERY_COUNT
            and frame["wide"]["domain"]["random_query_count"]
            == RANDOM_QUERY_COUNT
            and frame["wide"]["domain"]["radar_query_count"]
            == RADAR_QUERY_COUNT
            for frame in frames
        ),
        "g1f_paired_same_frames_and_exact_32k": all(
            frame["g1f_32k"]["candidate_count"] == G1F_PROPOSAL_COUNT
            for frame in frames
        ),
        "exact_10000": all(
            frame["wide"]["selected_count"] == EXPORT_COUNT
            and frame["wide"]["unique_selected_candidate_id_count"]
            == EXPORT_COUNT
            for frame in frames
        ),
        "chamfer_at_most_2p50m": (
            geometry["chamfer_m"]["median"]
            <= RA0_THRESHOLDS["chamfer_median_m"]
        ),
        "outlier_at_most_25pct": (
            geometry["outlier_fraction_2m"]["mean"]
            <= RA0_THRESHOLDS["outlier_fraction_2m_mean"]
        ),
        "completeness_at_most_0p65m": (
            geometry["completeness_mean_distance_m"]["median"]
            <= RA0_THRESHOLDS["completeness_median_m"]
        ),
        "far_completeness_at_most_8p0m": (
            far_geometry is not None
            and far_geometry["mean"]
            <= RA0_THRESHOLDS["far_completeness_mean_m"]
        ),
        "duplicate_at_most_10pct": (
            duplicates["duplicate_fraction_0p05m"]["mean"]
            <= RA0_THRESHOLDS["duplicate_fraction_mean"]
        ),
        "far_pool_recall_2m_at_least_50pct": (
            far_support is not None
            and far_support["gt_recall_from_pool_2p0m"]["mean"]
            >= RA0_THRESHOLDS["far_pool_recall_2m_mean"]
        ),
        "gt_absent_from_proposal_generation": all(
            frame["wide"]["ground_truth_used_for_proposal_generation"] is False
            and frame["wide"]["domain"][
                "ground_truth_used_for_proposal_generation"
            ]
            is False
            for frame in frames
        ),
        "validation_only_test_untouched": all(
            frame["partition"] == "validation"
            and frame["test_accessed"] is False
            for frame in frames
        ),
    }


def estimated_peak_memory(max_target_count: int) -> dict[str, int | bool]:
    float_bytes = 4
    cube_bytes = 64 * 256 * 107 * 37 * float_bytes
    pool_bytes = RAW_QUERY_COUNT * (
        3 * float_bytes * 2 + 8 + 1 + 1 + 8 * 4
    )
    distance_tile_bytes = (
        CANDIDATE_DISTANCE_CHUNK
        * min(max(max_target_count, 1), TARGET_DISTANCE_CHUNK)
        * float_bytes
    )
    dense_metric_tile_bytes = 1_024 * max(max_target_count, EXPORT_COUNT) * 4
    duplicate_tile_bytes = 512 * EXPORT_COUNT * float_bytes
    conservative_runtime_overhead_bytes = 4 * 1024**3
    estimated = (
        cube_bytes
        + pool_bytes
        + distance_tile_bytes
        + dense_metric_tile_bytes
        + duplicate_tile_bytes
        + conservative_runtime_overhead_bytes
    )
    return {
        "cube_bytes": cube_bytes,
        "pool_and_sort_workspace_bytes": pool_bytes,
        "maximum_streamed_distance_tile_bytes": distance_tile_bytes,
        "dense_metric_tile_bytes": dense_metric_tile_bytes,
        "duplicate_metric_tile_bytes": duplicate_tile_bytes,
        "conservative_runtime_overhead_bytes": conservative_runtime_overhead_bytes,
        "estimated_peak_bytes": estimated,
        "limit_bytes": 32 * 1024**3,
        "below_32gb": estimated < 32 * 1024**3,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"R-A0 refuses to overwrite existing output: {args.output}"
        )
    repo = Path(__file__).resolve().parents[2]
    current_commit = verify_source_tree(repo, args.source_commit)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    data_contract = validate_data_contract(manifest, scene_split)
    frozen_hashes = validate_frozen_input_hashes(
        args.manifest,
        args.scene_split,
    )
    validation_records = [
        frame
        for frame in manifest["frames"]
        if frame["partition"] == "validation"
    ]
    data_hashes = frame_data_hashes(
        validation_records,
        args.data_root,
        args.cache_root,
    )
    axis_hashes = resource_data_hashes(args.data_root)
    g1d = validate_g1d_run(args.g1d_run)
    device, device_name = require_h200(args.device)
    axes = load_axes(args.data_root / "resources")
    dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    if len(dataset) != FORMAL_VALIDATION_COUNT:
        raise ValueError("R-A0 dataset loader changed the frozen validation count")
    proposal_cache: dict[tuple[int, int], torch.Tensor] = {}
    frames = []
    max_target_count = 0
    for index in range(len(dataset)):
        item = dataset[index]
        max_target_count = max(
            max_target_count,
            int(item["target_xyz_confidence"].shape[0]),
        )
        frames.append(
            evaluate_frame(
                item,
                axes,
                device,
                proposal_cache,
            )
        )
    metrics = aggregate_evaluation(frames)
    checks = gate_checks(metrics, frames)
    label = diagnostic_artifact_label()
    memory = estimated_peak_memory(max_target_count)
    if not memory["below_32gb"]:
        raise RuntimeError("R-A0 estimated peak exceeds the frozen 32 GB limit")
    document = {
        "protocol": PROTOCOL,
        "artifact_label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": current_commit,
        "device": device_name,
        "selection": {
            "raw_query_count": RAW_QUERY_COUNT,
            "random_query_count": RANDOM_QUERY_COUNT,
            "radar_query_count": RADAR_QUERY_COUNT,
            "export_count": EXPORT_COUNT,
            "range_strata_m": [
                {
                    "label": label_value,
                    "lower_m": lower,
                    "upper_m": upper,
                }
                for label_value, lower, upper in RANGE_STRATA_M
            ],
            "candidate_capacity": "one_per_0p05m_cartesian_cell",
            "ground_truth_used_for_proposal_generation": False,
            "ground_truth_used_for_selection": True,
            "diagnostic_unattainable": True,
            "paired_control": "same_frame_frozen_g1f_32k_oracle",
        },
        "metrics": metrics,
        "frames": frames,
        "thresholds": RA0_THRESHOLDS,
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "authorize_r_a1_one_seed_10_epoch"
            if all(checks.values())
            else "close_r_a_without_ratio_or_threshold_repair"
        ),
        "complexity": {
            "wide_support": (
                "O(frames * unique_wide_candidates * target_points)"
            ),
            "paired_g1f_support": (
                "O(frames * 32000 * target_points)"
            ),
            "distance_materialization": (
                f"at most {CANDIDATE_DISTANCE_CHUNK} x "
                f"{TARGET_DISTANCE_CHUNK} per streamed tile"
            ),
            "estimated_h200_wall_time": "45-90 minutes for 24 frames",
            "frozen_h200_budget_upper_bound": "1.5 GPU-h",
            "memory": memory,
        },
        "assumptions_requiring_mainline_confirmation": [
            (
                "Equal candidate counts per 0-40, 40-60, and 60-120 m "
                "stratum are the frozen K-Radar interpretation of "
                "range-stratified quotas."
            ),
            (
                "The per-stratum 0.75 integrated-log-energy quantile is the "
                "frozen K-Radar low-threshold analogue; it is not tunable."
            ),
            (
                "With no trained occupancy field at R-A0, secondary "
                "refinement is centered on Cube-derived low-threshold cells "
                "using the RaLD helper 1/2-cell bias, never on GT."
            ),
            (
                "The G1F target-mass capacity allocation is reused for both "
                "paired exact-10k unattainable oracles."
            ),
        ],
        "provenance": {
            "official_rald_commit": OFFICIAL_RALD_COMMIT,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": frozen_hashes["manifest"],
            "scene_split": str(args.scene_split.resolve()),
            "scene_split_sha256": frozen_hashes["scene_split"],
            "data_contract": data_contract,
            "validation_frame_data_hashes": data_hashes,
            "axis_resource_hashes": axis_hashes,
            "source_hashes": source_hashes(repo),
            "g1d": g1d,
            "g1d_proposal_cache_entries": len(proposal_cache),
            "partitions": ["validation"],
            "test_accessed": False,
            "torch_version": torch.__version__,
        },
    }
    if document["artifact_label"]["label"] != ORACLE_ARTIFACT_LABEL:
        raise AssertionError("R-A0 artifact lost its unattainable oracle label")
    atomic_json_exclusive(args.output, document)
    print(json.dumps(document, indent=2))
    if not document["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
