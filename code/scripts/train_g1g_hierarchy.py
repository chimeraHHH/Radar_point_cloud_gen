#!/usr/bin/env python3
"""Train and evaluate the frozen one-seed G1G Stage-0 hierarchy."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
    nearest_distance,
)
from eval.rald_guided_query import duplicate_report  # noqa: E402
from losses.g1g_hierarchy import g1g_hierarchy_loss  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from models.g1g_hierarchical_allocator import (  # noqa: E402
    G1GConditionExclusiveHierarchy,
)
from scripts.g1b_contract import sha256  # noqa: E402


PROTOCOL = "g1g_condition_exclusive_hierarchy_stage0_v1"
FORMAL_SEED = 20260716
FORMAL_EPOCHS = 20
FORMAL_EVAL_EVERY = 5
FORMAL_TRAIN_COUNT = 76
FORMAL_VALIDATION_COUNT = 24
SMOKE_TRAIN_COUNT = 2
SMOKE_VALIDATION_COUNT = 2
G1D_EPOCH15_COMPLETENESS_M = 3.5811
G1D_EPOCH15_FAR_COMPLETENESS_M = 8.1239
COMPLETENESS_IMPROVEMENT_FRACTION = 0.30
STAGE0_COMPLETENESS_LIMIT_M = (
    G1D_EPOCH15_COMPLETENESS_M
    * (1.0 - COMPLETENESS_IMPROVEMENT_FRACTION)
)
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
FROZEN_NORMALIZATION_SHA256 = (
    "6a87437b80fa2ede31401beeff97f0621cc7495858ea7360a9a2f9599dd4bcd6"
)
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TrainConfig:
    protocol: str
    epochs: int
    eval_every: int
    seed: int
    smoke: bool
    train_limit: int | None
    validation_limit: int | None
    learning_rate: float
    minimum_learning_rate: float
    warmup_epochs: int
    weight_decay: float
    gradient_clip_norm: float
    ema_decay: float
    point_count: int
    center_count: int
    children_per_center: int
    radar_token_count: int
    model_dim: int
    depth: int
    heads: int
    head_dim: int
    radar_base_channels: int
    radar_spectral_channels: int
    geometry_weight: float
    outlier_weight: float
    child_existence_weight: float
    center_coverage_weight: float
    center_existence_weight: float
    center_repulsion_weight: float
    child_diversity_weight: float
    child_bound_weight: float
    outlier_threshold_m: float
    existence_radius_m: float
    center_repulsion_distance_m: float
    child_diversity_diagonal_fraction: float
    selection_metric: str
    test_accessed: bool


def frozen_config(*, smoke: bool) -> TrainConfig:
    """Return the non-overridable formal or H200 smoke configuration."""

    return TrainConfig(
        protocol=PROTOCOL,
        epochs=1 if smoke else FORMAL_EPOCHS,
        eval_every=1 if smoke else FORMAL_EVAL_EVERY,
        seed=FORMAL_SEED,
        smoke=smoke,
        train_limit=SMOKE_TRAIN_COUNT if smoke else None,
        validation_limit=SMOKE_VALIDATION_COUNT if smoke else None,
        learning_rate=1e-4,
        minimum_learning_rate=1e-6,
        warmup_epochs=1 if smoke else 2,
        weight_decay=0.05,
        gradient_clip_norm=10.0,
        ema_decay=0.999,
        point_count=10_000,
        center_count=2_500,
        children_per_center=4,
        radar_token_count=336,
        model_dim=512,
        depth=6,
        heads=8,
        head_dim=64,
        radar_base_channels=64,
        radar_spectral_channels=16,
        geometry_weight=1.0,
        outlier_weight=0.25,
        child_existence_weight=0.10,
        center_coverage_weight=0.25,
        center_existence_weight=0.05,
        center_repulsion_weight=0.05,
        child_diversity_weight=0.05,
        child_bound_weight=1.0,
        outlier_threshold_m=2.0,
        existence_radius_m=1.0,
        center_repulsion_distance_m=0.10,
        child_diversity_diagonal_fraction=0.20,
        selection_metric=(
            "median_chamfer + completeness_gate_excess + "
            "2*outlier_gate_excess + duplicate_gate_excess + "
            "0.1*far_gate_excess + 10*shuffle_gate_deficit"
        ),
        test_accessed=False,
    )


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("G1G source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("G1G source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"G1G formal source worktree is dirty: {dirty}")


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("G1G training requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G1G training is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G1G training requires H200, got {resolved}")
    return device, resolved


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected:
        raise ValueError("G1G checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def selected_indices(length: int, limit: int | None) -> list[int]:
    if limit is None:
        return list(range(length))
    if limit <= 0 or length < limit:
        raise ValueError("G1G smoke limit exceeds the available partition")
    return np.linspace(0, length - 1, limit).round().astype(int).tolist()


def smoke_cross_scene_indices(records: list[dict]) -> list[int]:
    """Select exactly two deterministic validation frames from different scenes."""

    for first in range(len(records)):
        for second in range(first + 1, len(records)):
            if int(records[first]["sequence"]) != int(records[second]["sequence"]):
                return [first, second]
    raise ValueError("G1G smoke validation needs two different scenes")


def cross_scene_condition_indices(records: list[dict]) -> list[int]:
    """Return a deterministic derangement whose pairs use different scenes."""

    count = len(records)
    if count < 2:
        raise ValueError("G1G condition shuffle requires at least two frames")
    sequences = [int(record["sequence"]) for record in records]
    for shift in range(1, count):
        candidate = [(index + shift) % count for index in range(count)]
        if all(
            sequences[index] != sequences[other]
            for index, other in enumerate(candidate)
        ):
            return candidate
    raise ValueError("No G1G cross-scene condition derangement exists")


def validate_data_contract(manifest: dict, scene_split: dict) -> dict[str, int]:
    if scene_split.get("gate_pass") is not True:
        raise ValueError("G1G scene split did not pass its leakage gate")
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("G1G manifest must contain a frame list")
    counts = Counter(frame.get("partition") for frame in frames)
    expected = {
        "train": FORMAL_TRAIN_COUNT,
        "validation": FORMAL_VALIDATION_COUNT,
    }
    if dict(counts) != expected:
        raise ValueError(
            f"G1G requires the frozen 76/24 development manifest, got {dict(counts)}"
        )
    if any(frame.get("partition") == "test" for frame in frames):
        raise ValueError("G1G development manifest must not contain test frames")
    identities = {
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in frames
    }
    if len(identities) != len(frames):
        raise ValueError("G1G manifest contains duplicate frame identities")
    return {
        "manifest_frame_count": len(frames),
        "train_frame_count": counts["train"],
        "validation_frame_count": counts["validation"],
        "test_frame_count": 0,
    }


def validate_frozen_input_hashes(
    manifest_path: Path,
    scene_split_path: Path,
    normalization_path: Path,
) -> dict[str, str]:
    actual = {
        "manifest": sha256(manifest_path),
        "scene_split": sha256(scene_split_path),
        "normalization": sha256(normalization_path),
    }
    expected = {
        "manifest": FROZEN_MANIFEST_SHA256,
        "scene_split": FROZEN_SCENE_SPLIT_SHA256,
        "normalization": FROZEN_NORMALIZATION_SHA256,
    }
    if actual != expected:
        raise ValueError(
            f"G1G inputs differ from the frozen G1D data contract: {actual}"
        )
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
    """Hash every formal Cube and dense-target cache used by Stage-0."""

    result = []
    for record in sorted(
        frames,
        key=lambda item: (
            str(item["partition"]),
            int(item["sequence"]),
            int(item["radar_index"]),
        ),
    ):
        cube_path = _cube_path(data_root, record)
        cache_path = _cache_path(cache_root, record)
        if not cube_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(
                f"G1G frame data is incomplete: {cube_path}, {cache_path}"
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


def source_hashes(repo: Path) -> dict[str, dict[str, str]]:
    relative_paths = (
        "artifacts/idea/pre_idea_drafts/g1g_condition_exclusive_hierarchy.md",
        "code/models/g1g_hierarchical_allocator.py",
        "code/losses/g1g_hierarchy.py",
        "code/scripts/train_g1g_hierarchy.py",
        "code/models/rald_matched.py",
        "code/models/cube_doppler.py",
        "code/models/cube_cycle.py",
        "code/models/cube_occupancy.py",
        "code/models/point_to_cube.py",
        "code/losses/cube_cycle.py",
        "code/losses/doppler_distribution.py",
        "code/losses/rald_anchor.py",
        "code/cube_dense/dataset.py",
        "code/cube_dense/kradar.py",
        "code/eval/dense_geometry.py",
        "code/eval/rald_guided_query.py",
        "code/scripts/g1b_contract.py",
    )
    paths = [repo / relative for relative in relative_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"G1G source map is incomplete: {missing}")
    return {
        str(path.resolve()): {"sha256": sha256(path)}
        for path in paths
    }


def architecture_anti_bypass_checks(
    metadata: dict[str, Any],
) -> dict[str, bool]:
    stage_sequence = metadata.get("stage_sequence", [])
    checks = {
        "global_radar_token_count_is_336": (
            metadata.get("global_radar_token_count") == 336
        ),
        "center_count_is_2500": metadata.get("center_query_count") == 2_500,
        "four_children_per_center": metadata.get("children_per_center") == 4,
        "final_point_count_is_10000": metadata.get("final_point_count") == 10_000,
        "pre_allocation_allowlist_exact": metadata.get(
            "pre_allocation_sources"
        )
        == [
            "learned_center_queries",
            "normalized_center_templates",
            "global_full_raed_tokens",
        ],
        "pre_allocation_forbidden_sources_exact": metadata.get(
            "forbidden_pre_allocation_sources"
        )
        == [
            "local_cube_spectrum",
            "local_cube_energy",
            "local_cube_neighborhood",
            "proposal_score",
        ],
        "local_cube_sampling_after_allocation": metadata.get(
            "local_cube_sampling_stage"
        )
        == "after_center_allocation",
        "stage_sequence_exact": stage_sequence
        == [
            "encode_global_full_raed_tokens",
            "allocate_2500_centers",
            "sample_local_center_spectrum",
            "decode_four_bounded_children",
            "sample_final_point_spectrum",
        ],
        "no_occupancy_queries": not any(
            "occupancy" in str(stage).lower() for stage in stage_sequence
        ),
        "no_proposal_cache": "proposal_score"
        not in metadata.get("pre_allocation_sources", []),
        "no_local_pre_allocation_inputs": not any(
            source
            in metadata.get("pre_allocation_sources", [])
            for source in metadata.get("forbidden_pre_allocation_sources", [])
        ),
    }
    return checks


def assert_anti_bypass_contract(
    model: G1GConditionExclusiveHierarchy,
) -> dict[str, Any]:
    metadata = model.architecture_metadata()
    checks = architecture_anti_bypass_checks(metadata)
    if not all(checks.values()):
        raise ValueError(f"G1G anti-bypass architecture contract failed: {checks}")
    return {
        "architecture_metadata": metadata,
        "static_checks": checks,
        "static_passed": True,
        "dynamic_allocation_gradient_audit": (
            "pending_until_first_two_optimizer_updates"
        ),
    }


def move_frame(item: dict, device: torch.device) -> torch.Tensor:
    return item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)


def attach_center_xyz(
    model: G1GConditionExclusiveHierarchy,
    output: dict[str, Any],
) -> dict[str, Any]:
    output["center_xyz_m"] = model._xyz(output["center_coordinates_rae"])
    return output


def gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(sum(value.square().sum() for value in values)).item())


def finite_nonzero(tensor: torch.Tensor | None) -> bool:
    return bool(
        tensor is not None
        and torch.isfinite(tensor).all()
        and torch.count_nonzero(tensor) > 0
    )


def gradient_audit(
    model: G1GConditionExclusiveHierarchy,
    measured_allocation_gradient: torch.Tensor | None,
    condition_allocation_gradient: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "measured_cube_allocation_gradient_is_none": (
            measured_allocation_gradient is None
        ),
        "condition_cube_allocation_gradient_finite_nonzero": finite_nonzero(
            condition_allocation_gradient
        ),
        "radar_encoder": gradient_norm(model.radar_encoder.parameters()),
        "center_queries": gradient_norm(model.center_queries.parameters()),
        "allocation_layer_radar_attention": [
            gradient_norm(block.radar_attention.parameters())
            for block in model.allocation_blocks
        ],
        "allocation_layer_feed_forward": [
            gradient_norm(block.feed_forward.parameters())
            for block in model.allocation_blocks
        ],
        "center_coordinate_head": gradient_norm(
            model.center_coordinate_head.parameters()
        ),
        "center_score_head": gradient_norm(model.center_score_head.parameters()),
        "post_allocation_local_spectrum_projection": gradient_norm(
            model.center_spectrum_projection.parameters()
        ),
        "patch_decoder": gradient_norm(model.patch_decoder.parameters()),
        "child_offset_head": gradient_norm(model.child_offset_head.parameters()),
        "child_confidence_head": gradient_norm(
            model.child_confidence_head.parameters()
        ),
    }


def dynamic_anti_bypass_checks(gradient_steps: list[dict]) -> dict[str, Any]:
    required = gradient_steps[:2]
    checks = {
        "two_optimizer_updates_audited": len(required) == 2,
        "measured_cube_has_no_allocation_gradient": bool(required)
        and all(
            step["gradients"]["measured_cube_allocation_gradient_is_none"]
            for step in required
        ),
        "condition_cube_drives_allocation": bool(required)
        and all(
            step["gradients"][
                "condition_cube_allocation_gradient_finite_nonzero"
            ]
            for step in required
        ),
        "radar_encoder_receives_geometry_gradient": bool(required)
        and all(step["gradients"]["radar_encoder"] > 0.0 for step in required),
        "every_allocation_layer_receives_condition_gradient": bool(required)
        and all(
            len(
                step["gradients"]["allocation_layer_radar_attention"]
            )
            == 6
            and all(
                value > 0.0
                for value in step["gradients"][
                    "allocation_layer_radar_attention"
                ]
            )
            for step in required
        ),
        "post_allocation_local_path_receives_gradient": bool(required)
        and all(
            step["gradients"]["post_allocation_local_spectrum_projection"] > 0.0
            for step in required
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


@torch.no_grad()
def update_ema(
    ema_model: G1GConditionExclusiveHierarchy,
    model: G1GConditionExclusiveHierarchy,
    decay: float,
) -> None:
    model_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        ema_parameter.mul_(decay).add_(
            model_parameters[name],
            alpha=1.0 - decay,
        )
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        ema_buffer.copy_(model_buffers[name])


def learning_rate(config: TrainConfig, epoch: int) -> float:
    if epoch <= config.warmup_epochs:
        return config.learning_rate * epoch / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(
        config.epochs - config.warmup_epochs,
        1,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + cosine * (
        config.learning_rate - config.minimum_learning_rate
    )


def aggregate_scalar_reports(reports: list[dict[str, float]]) -> dict:
    if not reports:
        raise ValueError("Cannot aggregate an empty G1G scalar report")
    keys = sorted({key for report in reports for key in report})
    return {
        key: {
            "mean": float(np.mean([report[key] for report in reports if key in report])),
            "std": float(np.std([report[key] for report in reports if key in report])),
            "median": float(
                np.median([report[key] for report in reports if key in report])
            ),
            "sample_count": sum(key in report for report in reports),
        }
        for key in keys
    }


def range_mass_report(xyz_m: torch.Tensor) -> dict[str, float]:
    radius = torch.linalg.vector_norm(xyz_m, dim=1)
    return {
        "range_0_30m_fraction": float(
            ((radius >= 0.0) & (radius < 30.0)).float().mean().item()
        ),
        "range_30_60m_fraction": float(
            ((radius >= 30.0) & (radius < 60.0)).float().mean().item()
        ),
        "range_60_120m_fraction": float(
            ((radius >= 60.0) & (radius < 120.0)).float().mean().item()
        ),
        "range_120m_plus_fraction": float(
            (radius >= 120.0).float().mean().item()
        ),
    }


def center_structure_report(
    center_xyz_m: torch.Tensor,
    output: dict[str, Any],
) -> dict[str, float]:
    quantized = torch.round(center_xyz_m / 0.05).to(torch.int64)
    unique_fraction = torch.unique(quantized, dim=0).shape[0] / center_xyz_m.shape[0]
    child_xyz = output["child_xyz_m"][0].float()
    pairwise = torch.cdist(child_xyz, child_xyz)
    mask = torch.triu(
        torch.ones(4, 4, dtype=torch.bool, device=child_xyz.device),
        diagonal=1,
    )
    sibling = pairwise[:, mask]
    parent = output["point_parent_center_index"]
    parent_counts = torch.bincount(parent, minlength=center_xyz_m.shape[0])
    return {
        "center_unique_fraction_0p05m": float(unique_fraction),
        "center_confidence_mean": float(
            torch.sigmoid(
                output["center_score_logit"][0].float()
            ).mean().item()
        ),
        "center_predicted_occupied_fraction": float(
            (output["center_score_logit"][0].float() >= 0.0)
            .float()
            .mean()
            .item()
        ),
        "children_per_center_min": float(parent_counts.min().item()),
        "children_per_center_max": float(parent_counts.max().item()),
        "child_pair_distance_mean_m": float(sibling.mean().item()),
        "child_pair_distance_median_m": float(sibling.median().item()),
        "child_pair_collapse_fraction_0p05m": float(
            (sibling < 0.05).float().mean().item()
        ),
        "child_offset_abs_mean_bins": float(
            output["child_offset_bins"][0].float().abs().mean().item()
        ),
        "child_physical_offset_mean_m": float(
            torch.linalg.vector_norm(
                output["child_physical_offset_m"][0].float(),
                dim=-1,
            ).mean().item()
        ),
    }


def repeated_control_duplicate_report(point_count: int) -> dict[str, float]:
    if point_count <= 0:
        raise ValueError("G1G control point count must be positive")
    return {
        "duplicate_fraction_0p05m": 1.0,
        "nearest_other_median_m": 0.0,
        "nearest_other_mean_m": 0.0,
    }


def hierarchy_control_point_sets(
    output: dict[str, Any],
) -> dict[str, torch.Tensor]:
    center_xyz = output["center_xyz_m"][0].float()
    child_xyz = output["child_xyz_m"][0].float()
    if center_xyz.shape != (2_500, 3) or child_xyz.shape != (2_500, 4, 3):
        raise ValueError("G1G controls require the frozen 2,500 x 4 hierarchy")
    return {
        "zero_local_refinement": center_xyz[:, None, :]
        .expand(-1, 4, -1)
        .reshape(10_000, 3),
        "child_collapse": child_xyz.mean(dim=1, keepdim=True)
        .expand(-1, 4, -1)
        .reshape(10_000, 3),
    }


def _mean_numeric_reports(reports: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for report in reports for key in report})
    return {
        key: float(np.mean([report[key] for report in reports if key in report]))
        for key in keys
        if any(isinstance(report.get(key), (int, float)) for report in reports)
    }


def scene_first_geometry(
    frames: list[dict],
    key: str,
) -> tuple[dict, dict[str, dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for frame in frames:
        grouped[int(frame["sequence"])].append(frame[key])
    per_scene = {
        str(sequence): _mean_numeric_reports(reports)
        for sequence, reports in sorted(grouped.items())
    }
    return aggregate_geometry_reports(list(per_scene.values())), per_scene


def scene_first_scalar(
    frames: list[dict],
    key: str,
) -> tuple[dict, dict[str, dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for frame in frames:
        grouped[int(frame["sequence"])].append(frame[key])
    per_scene = {
        str(sequence): _mean_numeric_reports(reports)
        for sequence, reports in sorted(grouped.items())
    }
    return aggregate_scalar_reports(list(per_scene.values())), per_scene


@torch.inference_mode()
def evaluate(
    model: G1GConditionExclusiveHierarchy,
    dataset: KRadarCubeDataset,
    indices: list[int],
    device: torch.device,
) -> dict:
    model.eval()
    records = [dataset.records[index] for index in indices]
    shuffled_positions = cross_scene_condition_indices(records)
    frames = []
    for position, index in enumerate(indices):
        item = dataset[index]
        shuffled_item = dataset[indices[shuffled_positions[position]]]
        cube = move_frame(item, device)
        shuffled_cube = move_frame(shuffled_item, device)
        target = item["target_xyz_confidence"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = attach_center_xyz(model, model(cube))
            shuffled_output = attach_center_xyz(
                model,
                model(cube, condition_cube_drae=shuffled_cube),
            )
        generated_xyz = output["xyz_m"][0].float()
        center_xyz = output["center_xyz_m"][0].float()
        controls = hierarchy_control_point_sets(output)
        zero_local_xyz = controls["zero_local_refinement"]
        collapsed_xyz = controls["child_collapse"]
        target_xyz = target[:, :3].float()
        target_weight = target[:, 3].float()
        generated = geometry_report(
            generated_xyz,
            target_xyz,
            target_weight=target_weight,
        )
        zero_local = geometry_report(
            zero_local_xyz,
            target_xyz,
            target_weight=target_weight,
        )
        child_collapse = geometry_report(
            collapsed_xyz,
            target_xyz,
            target_weight=target_weight,
        )
        condition_shuffled = geometry_report(
            shuffled_output["xyz_m"][0].float(),
            target_xyz,
            target_weight=target_weight,
        )
        centers = geometry_report(
            center_xyz,
            target_xyz,
            target_weight=target_weight,
        )
        cfar = geometry_report(
            item["cfar_xyzd_power_snr"][:, :3].to(device),
            target_xyz,
            target_weight=target_weight,
        )
        duplicates = duplicate_report(generated_xyz)
        zero_local_duplicates = repeated_control_duplicate_report(
            zero_local_xyz.shape[0]
        )
        child_collapse_duplicates = repeated_control_duplicate_report(
            collapsed_xyz.shape[0]
        )
        current_chamfer = generated["chamfer_m"]
        shuffled_chamfer = condition_shuffled["chamfer_m"]
        center_nearest_target = nearest_distance(center_xyz, target_xyz)
        structure = center_structure_report(center_xyz, output)
        structure["center_matched_fraction_1m"] = float(
            (center_nearest_target <= 1.0).float().mean().item()
        )
        frame = {
            "sequence": int(item["sequence"]),
            "radar_index": int(item["radar_index"]),
            "shuffled_condition_sequence": int(shuffled_item["sequence"]),
            "shuffled_condition_radar_index": int(
                shuffled_item["radar_index"]
            ),
            "generated": generated,
            "zero_local_refinement": zero_local,
            "child_collapse": child_collapse,
            "condition_shuffled": condition_shuffled,
            "allocated_centers": centers,
            "cfar": cfar,
            "duplicates": duplicates,
            "zero_local_refinement_duplicates": zero_local_duplicates,
            "child_collapse_duplicates": child_collapse_duplicates,
            "condition_shuffle": {
                "chamfer_fraction": float(
                    shuffled_chamfer / max(current_chamfer, 1e-12) - 1.0
                )
            },
            "point_range_mass": range_mass_report(generated_xyz),
            "center_range_mass": range_mass_report(center_xyz),
            "center_structure": structure,
            "child_confidence_mean": float(
                output["confidence"][0].float().mean().item()
            ),
            "radar_token_count": int(output["radar_token_count"].item()),
            "center_count": int(output["center_count"].item()),
            "children_per_center": int(
                output["children_per_center"].item()
            ),
            "final_point_count": int(output["final_point_count"].item()),
            "anti_bypass_static_checks": architecture_anti_bypass_checks(
                output["architecture_metadata"]
            ),
        }
        if not all(frame["anti_bypass_static_checks"].values()):
            raise RuntimeError("G1G evaluation output violates anti-bypass contract")
        frames.append(frame)
        del (
            item,
            shuffled_item,
            cube,
            shuffled_cube,
            target,
            output,
            shuffled_output,
        )
        torch.cuda.empty_cache()

    geometry_keys = (
        "generated",
        "zero_local_refinement",
        "child_collapse",
        "condition_shuffled",
        "allocated_centers",
        "cfar",
    )
    scalar_keys = (
        "duplicates",
        "zero_local_refinement_duplicates",
        "child_collapse_duplicates",
        "condition_shuffle",
        "point_range_mass",
        "center_range_mass",
        "center_structure",
    )
    report: dict[str, Any] = {
        "frame_count": len(frames),
        "scene_count": len({frame["sequence"] for frame in frames}),
        "frames": frames,
        "scene_first": {
            "scene_count": len({frame["sequence"] for frame in frames}),
            "per_scene": {},
        },
        "exact_counts": {
            "evaluated_frames": len(frames),
            "point_count_per_frame": 10_000,
            "center_count_per_frame": 2_500,
            "children_per_center": 4,
            "total_generated_points": 10_000 * len(frames),
        },
    }
    for key in geometry_keys:
        report[key] = aggregate_geometry_reports([frame[key] for frame in frames])
        aggregate, per_scene = scene_first_geometry(frames, key)
        report["scene_first"][key] = aggregate
        report["scene_first"]["per_scene"][key] = per_scene
    for key in scalar_keys:
        report[key] = aggregate_scalar_reports([frame[key] for frame in frames])
        aggregate, per_scene = scene_first_scalar(frames, key)
        report["scene_first"][key] = aggregate
        report["scene_first"]["per_scene"][key] = per_scene
    report["child_confidence_mean"] = {
        "mean": float(np.mean([frame["child_confidence_mean"] for frame in frames])),
        "median": float(
            np.median([frame["child_confidence_mean"] for frame in frames])
        ),
    }
    return report


def stage0_decision(metrics: dict) -> dict[str, Any]:
    """Apply the frozen G1G Stage-0 gates without threshold relaxation."""

    far_report = metrics["generated"].get(
        "range_60_120m_completeness_mean_distance_m"
    )
    far_completeness = (
        float(far_report["mean"])
        if far_report is not None
        else 1_000_000.0
    )
    values = {
        "condition_shuffle_chamfer_fraction_mean": float(
            metrics["condition_shuffle"]["chamfer_fraction"]["mean"]
        ),
        "duplicate_fraction_0p05m_mean": float(
            metrics["duplicates"]["duplicate_fraction_0p05m"]["mean"]
        ),
        "completeness_mean_distance_m_median": float(
            metrics["generated"]["completeness_mean_distance_m"]["median"]
        ),
        "outlier_fraction_2m_mean": float(
            metrics["generated"]["outlier_fraction_2m"]["mean"]
        ),
        "far_completeness_60_120m_mean": far_completeness,
        "center_unique_fraction_0p05m_mean": float(
            metrics["center_structure"][
                "center_unique_fraction_0p05m"
            ]["mean"]
        ),
    }
    promotion_checks = {
        "condition_shuffle_chamfer_degradation_at_least_1pct": (
            values["condition_shuffle_chamfer_fraction_mean"] >= 0.01
        ),
        "duplicate_fraction_at_most_15pct": (
            values["duplicate_fraction_0p05m_mean"] <= 0.15
        ),
        "completeness_at_least_30pct_better_than_g1d_epoch15": (
            values["completeness_mean_distance_m_median"]
            <= STAGE0_COMPLETENESS_LIMIT_M
        ),
        "outlier_fraction_at_most_25pct": (
            values["outlier_fraction_2m_mean"] <= 0.25
        ),
        "far_completeness_no_worse_than_g1d_epoch15": (
            values["far_completeness_60_120m_mean"]
            <= G1D_EPOCH15_FAR_COMPLETENESS_M
        ),
    }
    abandonment_checks = {
        "at_least_80pct_unique_center_cells_at_5cm": (
            values["center_unique_fraction_0p05m_mean"] >= 0.80
        ),
        "completeness_not_bought_by_outlier_or_far_regression": (
            not promotion_checks[
                "completeness_at_least_30pct_better_than_g1d_epoch15"
            ]
            or (
                promotion_checks["outlier_fraction_at_most_25pct"]
                and promotion_checks[
                    "far_completeness_no_worse_than_g1d_epoch15"
                ]
            )
        ),
    }
    promotion_passed = all(promotion_checks.values())
    abandonment_triggered = not all(abandonment_checks.values())
    return {
        "protocol": PROTOCOL,
        "metric_basis": "matched_frame_first_G1D_epoch15_24frame_aggregation",
        "fixed_controls": {
            "g1d_epoch15_completeness_m": G1D_EPOCH15_COMPLETENESS_M,
            "required_completeness_improvement_fraction": (
                COMPLETENESS_IMPROVEMENT_FRACTION
            ),
            "g1g_completeness_limit_m": STAGE0_COMPLETENESS_LIMIT_M,
            "g1d_epoch15_far_completeness_m": (
                G1D_EPOCH15_FAR_COMPLETENESS_M
            ),
        },
        "values": values,
        "promotion_checks": promotion_checks,
        "abandonment_checks": abandonment_checks,
        "promotion_passed": promotion_passed,
        "abandonment_triggered": abandonment_triggered,
        "passed": promotion_passed and not abandonment_triggered,
    }


def selection_score(metrics: dict) -> float:
    decision = stage0_decision(metrics)
    values = decision["values"]
    return (
        float(metrics["generated"]["chamfer_m"]["median"])
        + max(
            values["completeness_mean_distance_m_median"]
            - STAGE0_COMPLETENESS_LIMIT_M,
            0.0,
        )
        + 2.0 * max(values["outlier_fraction_2m_mean"] - 0.25, 0.0)
        + max(values["duplicate_fraction_0p05m_mean"] - 0.15, 0.0)
        + 0.1
        * max(
            values["far_completeness_60_120m_mean"]
            - G1D_EPOCH15_FAR_COMPLETENESS_M,
            0.0,
        )
        + 10.0
        * max(
            0.01 - values["condition_shuffle_chamfer_fraction_mean"],
            0.0,
        )
    )


def build_model(
    config: TrainConfig,
    axes,
    normalization: dict,
) -> G1GConditionExclusiveHierarchy:
    return G1GConditionExclusiveHierarchy(
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        log_center=float(normalization["normalization"]["center"]),
        log_scale=float(normalization["normalization"]["scale"]),
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
        radar_base_channels=config.radar_base_channels,
        radar_spectral_channels=config.radar_spectral_channels,
    )


def artifact_document(
    artifact_type: str,
    *,
    config: TrainConfig,
    provenance: dict,
    exact_counts: dict,
    anti_bypass: dict,
    payload: dict,
) -> dict:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_type": artifact_type,
        "config": asdict(config),
        "provenance": provenance,
        "exact_counts": exact_counts,
        "anti_bypass": anti_bypass,
        **payload,
    }


def save_checkpoint(
    path: Path,
    model: G1GConditionExclusiveHierarchy,
    ema_model: G1GConditionExclusiveHierarchy,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config: TrainConfig,
    provenance: dict,
    exact_counts: dict,
    anti_bypass: dict,
    gradient_steps: list[dict],
    record: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        artifact_document(
            "checkpoint",
            config=config,
            provenance=provenance,
            exact_counts=exact_counts,
            anti_bypass=anti_bypass,
            payload={
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "gradient_steps": gradient_steps,
                "record": record,
                "rng_state": capture_rng_state(),
            },
        ),
        temporary,
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)
    config = frozen_config(smoke=args.smoke)

    nonempty = args.output.exists() and any(args.output.iterdir())
    if nonempty and args.overwrite:
        shutil.rmtree(args.output)
        nonempty = False
    if nonempty and not args.resume:
        raise FileExistsError(f"G1G output is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No G1G run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    frozen_input_hashes = validate_frozen_input_hashes(
        args.manifest,
        args.scene_split,
        args.normalization,
    )
    manifest_counts = validate_data_contract(manifest, scene_split)
    data_hashes = {
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": frozen_input_hashes["manifest"],
        },
        "scene_split": {
            "path": str(args.scene_split.resolve()),
            "sha256": frozen_input_hashes["scene_split"],
        },
        "normalization": {
            "path": str(args.normalization.resolve()),
            "sha256": frozen_input_hashes["normalization"],
        },
        "range_azimuth_elevation_axes": {
            "path": str((args.data_root / "resources/info_arr.mat").resolve()),
            "sha256": sha256(args.data_root / "resources/info_arr.mat"),
        },
        "doppler_axis": {
            "path": str((args.data_root / "resources/arr_doppler.mat").resolve()),
            "sha256": sha256(args.data_root / "resources/arr_doppler.mat"),
        },
        "frames": frame_data_hashes(
            manifest["frames"],
            args.data_root,
            args.cache_root,
        ),
    }

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    axes = load_axes(args.data_root / "resources")
    model = build_model(config, axes, normalization).to(device)
    if len(model.allocation_blocks) != config.depth:
        raise AssertionError("G1G instantiated the wrong allocation depth")
    anti_bypass = assert_anti_bypass_contract(model)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_set = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train",),
    )
    validation_set = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    if len(train_set) != FORMAL_TRAIN_COUNT or len(validation_set) != (
        FORMAL_VALIDATION_COUNT
    ):
        raise ValueError("G1G dataset loader changed the frozen 76/24 split")
    train_indices = selected_indices(len(train_set), config.train_limit)
    validation_indices = (
        smoke_cross_scene_indices(validation_set.records)
        if config.smoke
        else selected_indices(len(validation_set), config.validation_limit)
    )
    cross_scene_condition_indices(
        [validation_set.records[index] for index in validation_indices]
    )
    exact_counts = {
        **manifest_counts,
        "used_train_frame_count": len(train_indices),
        "used_validation_frame_count": len(validation_indices),
        "point_count_per_frame": config.point_count,
        "center_count_per_frame": config.center_count,
        "children_per_center": config.children_per_center,
        "radar_token_count": config.radar_token_count,
        "optimizer_updates_planned": len(train_indices) * config.epochs,
        "evaluation_epochs": list(
            range(config.eval_every, config.epochs + 1, config.eval_every)
        ),
    }
    provenance = {
        "git_commit": args.source_commit,
        "worktree_clean": True,
        "source_hashes": source_hashes(repo),
        "data_hashes": data_hashes,
        "device": device_name,
        "torch_version": torch.__version__,
        "model_parameter_count": parameter_count(model),
        "partitions": ["train", "validation"],
        "test_accessed": False,
        "external_pretraining": False,
        "occupancy_queries": False,
        "proposal_cache": False,
        "local_pre_allocation_inputs": False,
        "g1d_control": {
            "epoch": 15,
            "completeness_m": G1D_EPOCH15_COMPLETENESS_M,
            "far_completeness_m": G1D_EPOCH15_FAR_COMPLETENESS_M,
            "model_loaded": False,
        },
    }
    run_document = artifact_document(
        "run_configuration",
        config=config,
        provenance=provenance,
        exact_counts=exact_counts,
        anti_bypass=anti_bypass,
        payload={"status": "configured"},
    )
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("G1G resume configuration or provenance differs")
    else:
        atomic_json(config_path, run_document)

    start_epoch = 1
    best_score = float("inf")
    prior_elapsed = 0.0
    gradient_steps: list[dict] = []
    if args.resume:
        last = torch.load(
            args.output / "last.pt",
            map_location=device,
            weights_only=False,
        )
        if (
            last.get("config") != asdict(config)
            or last.get("provenance") != provenance
            or last.get("exact_counts") != exact_counts
        ):
            raise ValueError("G1G last checkpoint metadata differs")
        model.load_state_dict(last["model"], strict=True)
        ema_model.load_state_dict(last["ema_model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        restore_rng_state(last["rng_state"])
        start_epoch = int(last["epoch"]) + 1
        prior_elapsed = float(last["record"]["elapsed_seconds"])
        gradient_steps = list(last["gradient_steps"])
        best_path = args.output / "best.pt"
        if best_path.is_file():
            best = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
            best_score = float(best["record"]["selection_score"])

    initial_path = args.output / "initial_validation_metrics.json"
    if not initial_path.is_file():
        initial_metrics = evaluate(
            ema_model,
            validation_set,
            validation_indices,
            device,
        )
        atomic_json(
            initial_path,
            artifact_document(
                "initial_validation_metrics",
                config=config,
                provenance=provenance,
                exact_counts=exact_counts,
                anti_bypass=anti_bypass,
                payload={
                    "epoch": 0,
                    "metrics": initial_metrics,
                    "stage0_decision": stage0_decision(initial_metrics),
                },
            ),
        )

    started = time.monotonic()
    update_count = max(0, start_epoch - 1) * len(train_indices)
    log_path = args.output / "train_log.jsonl"
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_learning_rate = learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = epoch_learning_rate
        order = train_indices.copy()
        random.Random(config.seed + epoch).shuffle(order)
        losses = []
        components: dict[str, list[float]] = {}
        for index in order:
            item = train_set[index]
            cube = move_frame(item, device)
            audit_update = update_count < 2
            if audit_update:
                measured_cube = cube.detach().requires_grad_(True)
                condition_cube = cube.detach().clone().requires_grad_(True)
            else:
                measured_cube = cube
                condition_cube = cube
            target = item["target_xyz_confidence"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = attach_center_xyz(
                    model,
                    model(
                        measured_cube,
                        condition_cube_drae=condition_cube,
                    ),
                )
                loss = g1g_hierarchy_loss(
                    output,
                    target,
                    expected_point_count=config.point_count,
                    expected_center_count=config.center_count,
                    expected_children_per_center=config.children_per_center,
                    geometry_weight=config.geometry_weight,
                    outlier_weight=config.outlier_weight,
                    child_existence_weight=config.child_existence_weight,
                    center_coverage_weight=config.center_coverage_weight,
                    center_existence_weight=config.center_existence_weight,
                    center_repulsion_weight=config.center_repulsion_weight,
                    child_diversity_weight=config.child_diversity_weight,
                    child_bound_weight=config.child_bound_weight,
                    outlier_threshold_m=config.outlier_threshold_m,
                    existence_radius_m=config.existence_radius_m,
                    center_repulsion_distance_m=(
                        config.center_repulsion_distance_m
                    ),
                    child_diversity_diagonal_fraction=(
                        config.child_diversity_diagonal_fraction
                    ),
                )
            if audit_update:
                allocation_probe = (
                    output["center_coordinates_rae"].float().square().mean()
                    + output["center_score_logit"].float().square().mean()
                )
                measured_gradient, condition_gradient = torch.autograd.grad(
                    allocation_probe,
                    (measured_cube, condition_cube),
                    allow_unused=True,
                    retain_graph=True,
                )
            else:
                measured_gradient = None
                condition_gradient = None
            loss.total.backward()
            update_count += 1
            if audit_update:
                gradient_steps.append(
                    {
                        "update": update_count,
                        "sequence": int(item["sequence"]),
                        "radar_index": int(item["radar_index"]),
                        "gradients": gradient_audit(
                            model,
                            measured_gradient,
                            condition_gradient,
                        ),
                    }
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            update_ema(ema_model, model, config.ema_decay)
            losses.append(float(loss.total.detach().item()))
            for name, value in loss.components.items():
                components.setdefault(name, []).append(float(value.item()))
            del (
                item,
                cube,
                measured_cube,
                condition_cube,
                target,
                output,
                loss,
            )

        dynamic_audit = dynamic_anti_bypass_checks(gradient_steps)
        current_anti_bypass = {
            **anti_bypass,
            "dynamic_allocation_gradient_audit": dynamic_audit,
        }
        record = {
            "epoch": epoch,
            "update_count": update_count,
            "train_loss_mean": float(np.mean(losses)),
            "train_components": {
                name: float(np.mean(values))
                for name, values in components.items()
            },
            "learning_rate": epoch_learning_rate,
            "elapsed_seconds": round(
                prior_elapsed + time.monotonic() - started,
                3,
            ),
        }
        is_best = False
        if epoch % config.eval_every == 0:
            metrics = evaluate(
                ema_model,
                validation_set,
                validation_indices,
                device,
            )
            decision = stage0_decision(metrics)
            score = selection_score(metrics)
            record["validation"] = metrics
            record["stage0_decision"] = decision
            record["selection_score"] = score
            metrics_document = artifact_document(
                "validation_metrics",
                config=config,
                provenance=provenance,
                exact_counts=exact_counts,
                anti_bypass=current_anti_bypass,
                payload={
                    "epoch": epoch,
                    "metrics": metrics,
                    "stage0_decision": decision,
                    "selection_score": score,
                },
            )
            atomic_json(
                args.output / f"metrics_epoch_{epoch:04d}.json",
                metrics_document,
            )
            is_best = score < best_score
        save_checkpoint(
            args.output / "last.pt",
            model,
            ema_model,
            optimizer,
            epoch=epoch,
            config=config,
            provenance=provenance,
            exact_counts=exact_counts,
            anti_bypass=current_anti_bypass,
            gradient_steps=gradient_steps,
            record=record,
        )
        if is_best:
            best_score = float(record["selection_score"])
            save_checkpoint(
                args.output / "best.pt",
                model,
                ema_model,
                optimizer,
                epoch=epoch,
                config=config,
                provenance=provenance,
                exact_counts=exact_counts,
                anti_bypass=current_anti_bypass,
                gradient_steps=gradient_steps,
                record=record,
            )
        log_document = artifact_document(
            "training_epoch_log",
            config=config,
            provenance=provenance,
            exact_counts=exact_counts,
            anti_bypass=current_anti_bypass,
            payload={"record": record},
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_document) + "\n")
        print(json.dumps(log_document), flush=True)

    best_path = args.output / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("G1G training produced no selected checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    if (
        best.get("config") != asdict(config)
        or best.get("provenance") != provenance
        or best.get("exact_counts") != exact_counts
    ):
        raise ValueError("G1G selected checkpoint metadata differs")
    ema_model.load_state_dict(best["ema_model"], strict=True)
    final_metrics = evaluate(
        ema_model,
        validation_set,
        validation_indices,
        device,
    )
    final_decision = stage0_decision(final_metrics)
    final_dynamic_audit = dynamic_anti_bypass_checks(gradient_steps)
    final_anti_bypass = {
        **anti_bypass,
        "dynamic_allocation_gradient_audit": final_dynamic_audit,
    }
    if not final_dynamic_audit["passed"]:
        final_decision = {
            **final_decision,
            "passed_before_anti_bypass": final_decision["passed"],
            "passed": False,
            "anti_bypass_failure": True,
        }
    final_payload = {
        "completed": True,
        "best_epoch": int(best["epoch"]),
        "selection_metric": config.selection_metric,
        "selection_value": best_score,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "evaluation_model": "ema_0p999",
        "metrics": final_metrics,
        "stage0_decision": final_decision,
        "gradient_steps": gradient_steps,
        "test_accessed": False,
    }
    final_document = artifact_document(
        "best_validation_metrics",
        config=config,
        provenance=provenance,
        exact_counts=exact_counts,
        anti_bypass=final_anti_bypass,
        payload=final_payload,
    )
    atomic_json(args.output / "best_validation_metrics.json", final_document)
    decision_document = artifact_document(
        "stage0_decision",
        config=config,
        provenance=provenance,
        exact_counts=exact_counts,
        anti_bypass=final_anti_bypass,
        payload={
            "best_epoch": int(best["epoch"]),
            "best_checkpoint_sha256": sha256(best_path),
            "decision": final_decision,
            "test_accessed": False,
        },
    )
    atomic_json(args.output / "stage0_decision.json", decision_document)
    print(json.dumps(final_document), flush=True)


if __name__ == "__main__":
    main()
