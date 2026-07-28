#!/usr/bin/env python3
"""Evaluate the frozen G1F-F0 diagnostic candidate-support oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
)
from eval.g1f_candidate_support import (  # noqa: E402
    EXPORT_COUNT,
    ORACLE_ARTIFACT_LABEL,
    PROPOSAL_COUNT,
    diagnostic_artifact_label,
    select_candidate_support_oracle,
)
from eval.rald_guided_query import duplicate_report  # noqa: E402
from models.cube_cycle import continuous_rae_to_xyz  # noqa: E402
from models.rald_query_field import (  # noqa: E402
    coarse_query_templates,
    proposals_from_flat_index,
)
from scripts.compare_rald_query_field import THRESHOLDS  # noqa: E402
from scripts.g1b_contract import sha256  # noqa: E402
from scripts.train_rald_query_field import (  # noqa: E402
    aggregate_scalar_reports,
    cached_proposal_flat_index,
    verify_source_tree,
)


PROTOCOL = "g1f_f0_candidate_support_oracle_v1"
G1D_PROTOCOL = "g1d_rald_query_field_geometry_v2"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
BASE_SEED_COUNT = 1_000
COARSE_TEMPLATE_COUNT = 32
NMS_KERNEL = (5, 5, 3)


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
        raise RuntimeError("G1F-F0 requires a full Git source commit")
    return commit


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("G1F-F0 requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G1F-F0 is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G1F-F0 requires an H200, got {resolved}")
    return device, resolved


def tensor_sha256(values: torch.Tensor) -> str:
    contiguous = values.detach().cpu().contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def validate_g1d_run(run: Path) -> dict:
    config_path = run / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"G1F-F0 G1D config is missing: {config_path}")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    if set(document) != {"config", "provenance"}:
        raise ValueError("G1F-F0 G1D config has an unknown schema")
    config = document["config"]
    provenance = document["provenance"]
    required_config = {
        "protocol": G1D_PROTOCOL,
        "base_seed_count": BASE_SEED_COUNT,
        "coarse_templates_per_seed": COARSE_TEMPLATE_COUNT,
        "coarse_query_count": PROPOSAL_COUNT,
        "point_count": EXPORT_COUNT,
        "test_accessed": False,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError(f"G1F-F0 requires G1D config {key}={expected!r}")
    if tuple(config.get("nms_kernel", ())) != NMS_KERNEL:
        raise ValueError("G1F-F0 requires the frozen G1D NMS kernel")
    required_provenance = {
        "test_accessed": False,
        "cfar_query_helper": False,
        "proposal_index_cache": "deterministic_flat_indices_only",
    }
    for key, expected in required_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"G1F-F0 requires G1D provenance {key}={expected!r}"
            )
    source_commit = provenance.get("git_commit")
    if not isinstance(source_commit, str) or not SOURCE_PATTERN.fullmatch(
        source_commit
    ):
        raise ValueError("G1F-F0 G1D provenance lacks a full source commit")
    return {
        "run": str(run.resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "source_commit": source_commit,
        "proposal_index_cache": provenance["proposal_index_cache"],
    }


def coarse_candidate_pool(
    item: dict,
    cube_drae: torch.Tensor,
    axes,
    proposal_cache: dict[tuple[int, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_config = SimpleNamespace(
        base_seed_count=BASE_SEED_COUNT,
        nms_kernel=NMS_KERNEL,
    )
    proposal_flat_index = cached_proposal_flat_index(
        item,
        cube_drae,
        cache_config,
        proposal_cache,
    )
    proposals = proposals_from_flat_index(cube_drae, proposal_flat_index)
    templates = coarse_query_templates(
        device=cube_drae.device,
        dtype=cube_drae.dtype,
    )
    candidate_rae = (
        proposals.coordinates_rae[:, :, None, :]
        + templates[None, None, :, :]
    ).reshape(-1, 3)
    maximum = candidate_rae.new_tensor(
        [
            cube_drae.shape[2] - 1,
            cube_drae.shape[3] - 1,
            cube_drae.shape[4] - 1,
        ]
    )
    candidate_rae = torch.minimum(
        torch.maximum(candidate_rae, torch.zeros_like(candidate_rae)),
        maximum,
    )
    if candidate_rae.shape != (PROPOSAL_COUNT, 3):
        raise AssertionError("G1F-F0 G1D expansion must produce 32,000 proposals")
    candidate_xyz = continuous_rae_to_xyz(
        candidate_rae,
        torch.as_tensor(
            axes.range_m, device=cube_drae.device, dtype=candidate_rae.dtype
        ),
        torch.as_tensor(
            axes.azimuth_rad, device=cube_drae.device, dtype=candidate_rae.dtype
        ),
        torch.as_tensor(
            axes.elevation_rad,
            device=cube_drae.device,
            dtype=candidate_rae.dtype,
        ),
    )
    candidate_indices = torch.arange(
        PROPOSAL_COUNT,
        dtype=torch.long,
        device=cube_drae.device,
    )
    return candidate_xyz, candidate_indices, proposal_flat_index[0]


@torch.inference_mode()
def evaluate_frame(
    item: dict,
    axes,
    device: torch.device,
    proposal_cache: dict[tuple[int, int], torch.Tensor],
    distance_chunk_size: int,
) -> dict:
    cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
    target = item["target_xyz_confidence"].to(device, non_blocking=True)
    candidate_xyz, candidate_indices, proposal_flat_index = (
        coarse_candidate_pool(item, cube, axes, proposal_cache)
    )
    selection = select_candidate_support_oracle(
        candidate_xyz.float(),
        candidate_indices,
        target[:, :3].float(),
        target_weight=target[:, 3].float(),
        distance_chunk_size=distance_chunk_size,
    )
    geometry = geometry_report(
        selection.selected_xyz_m,
        target[:, :3].float(),
        target_weight=target[:, 3].float(),
        chunk_size=distance_chunk_size,
    )
    duplicates = duplicate_report(selection.selected_xyz_m)
    selected_indices = selection.selected_candidate_indices
    return {
        "artifact_label": selection.artifact_label,
        "partition": item["partition"],
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
        "candidate_count": int(candidate_indices.numel()),
        "unique_candidate_index_count": int(
            torch.unique(candidate_indices).numel()
        ),
        "unique_candidate_coordinate_count": int(
            torch.unique(candidate_xyz, dim=0).shape[0]
        ),
        "selected_count": int(selected_indices.numel()),
        "unique_selected_index_count": int(
            torch.unique(selected_indices).numel()
        ),
        "unique_selected_coordinate_count": int(
            torch.unique(selection.selected_xyz_m, dim=0).shape[0]
        ),
        "selected_candidate_indices": selected_indices.cpu().tolist(),
        "selected_candidate_indices_sha256": tensor_sha256(selected_indices),
        "proposal_flat_index_sha256": tensor_sha256(proposal_flat_index),
        "geometry": geometry,
        "duplicates": duplicates,
        "per_range_support": selection.per_range_support,
        "ground_truth_used_for_selection": True,
        "learned_g1d_checkpoint_accessed": False,
        "cfar_accessed_by_oracle": False,
        "test_accessed": False,
    }


def aggregate_range_support(frames: list[dict]) -> dict:
    labels = frames[0]["per_range_support"]
    result = {}
    for label in labels:
        reports = []
        for frame in frames:
            report = {
                key: float(value)
                for key, value in frame["per_range_support"][label].items()
                if isinstance(value, (int, float)) and value is not None
            }
            reports.append(report)
        result[label] = aggregate_scalar_reports(reports)
    return result


def gate_checks(
    geometry: dict,
    duplicates: dict,
    frames: list[dict],
) -> dict[str, bool]:
    far = geometry.get("range_60_120m_completeness_mean_distance_m")
    return {
        "exact_candidate_count": all(
            frame["candidate_count"] == PROPOSAL_COUNT for frame in frames
        ),
        "unique_candidate_indices": all(
            frame["unique_candidate_index_count"] == PROPOSAL_COUNT
            for frame in frames
        ),
        "exact_export_count": all(
            frame["selected_count"] == EXPORT_COUNT for frame in frames
        ),
        "no_selected_index_reuse": all(
            frame["unique_selected_index_count"] == EXPORT_COUNT
            for frame in frames
        ),
        "chamfer": (
            geometry["chamfer_m"]["median"]
            <= THRESHOLDS["chamfer_median_m"]
        ),
        "outlier": (
            geometry["outlier_fraction_2m"]["mean"]
            <= THRESHOLDS["outlier_fraction_2m_mean"]
        ),
        "completeness": (
            geometry["completeness_mean_distance_m"]["median"]
            <= THRESHOLDS["completeness_median_m"]
        ),
        "far_completeness": (
            far is not None
            and far["mean"] <= THRESHOLDS["far_completeness_mean_m"]
        ),
        "duplicates": (
            duplicates["duplicate_fraction_0p05m"]["mean"]
            <= THRESHOLDS["duplicate_fraction_mean"]
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
    repo = Path(__file__).resolve().parents[2]
    current_commit = git_commit(repo)
    if args.source_commit != current_commit:
        raise ValueError("G1F-F0 requested source differs from the Git snapshot")
    verify_source_tree(repo, args.source_commit)
    if args.distance_chunk_size <= 0:
        raise ValueError("G1F-F0 distance chunk size must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    if scene_split.get("gate_pass") is not True:
        raise ValueError("G1F-F0 scene split did not pass its leakage gate")
    if any(frame.get("partition") == "test" for frame in manifest["frames"]):
        raise ValueError("G1F-F0 development manifest must not contain test frames")
    g1d = validate_g1d_run(args.g1d_run)
    device, device_name = require_h200(args.device)
    axes = load_axes(args.data_root / "resources")
    dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    proposal_cache: dict[tuple[int, int], torch.Tensor] = {}
    frames = [
        evaluate_frame(
            dataset[index],
            axes,
            device,
            proposal_cache,
            args.distance_chunk_size,
        )
        for index in range(len(dataset))
    ]
    geometry = aggregate_geometry_reports(
        [frame["geometry"] for frame in frames]
    )
    duplicates = aggregate_scalar_reports(
        [frame["duplicates"] for frame in frames]
    )
    checks = gate_checks(geometry, duplicates, frames)
    label = diagnostic_artifact_label()
    document = {
        "protocol": PROTOCOL,
        "artifact_label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": current_commit,
        "device": device_name,
        "selection": {
            "candidate_count": PROPOSAL_COUNT,
            "export_count": EXPORT_COUNT,
            "candidate_index_capacity": 1,
            "range_stratified": True,
            "nearest_gt_support_assignment": True,
            "ground_truth_used_for_selection": True,
            "diagnostic_unattainable": True,
            "condition_shuffle": "not_applicable_to_gt_oracle",
        },
        "metrics": {
            "geometry": geometry,
            "duplicates": duplicates,
            "per_range_support": aggregate_range_support(frames),
        },
        "frames": frames,
        "checks": checks,
        "passed": all(checks.values()),
        "provenance": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "scene_split": str(args.scene_split.resolve()),
            "scene_split_sha256": sha256(args.scene_split),
            "g1d": g1d,
            "g1d_proposal_cache_entries": len(proposal_cache),
            "g1d_proposal_cache_reconstruction": (
                "deterministic_flat_indices_then_frozen_32_template_expansion"
            ),
            "learned_g1d_checkpoint_accessed": False,
            "cfar_accessed_by_oracle": False,
            "partitions": ["validation"],
            "test_accessed": False,
            "torch_version": torch.__version__,
        },
    }
    if document["artifact_label"]["label"] != ORACLE_ARTIFACT_LABEL:
        raise AssertionError("G1F-F0 artifact lost its unattainable oracle label")
    atomic_json(args.output, document)
    print(json.dumps(document, indent=2))
    if not document["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
