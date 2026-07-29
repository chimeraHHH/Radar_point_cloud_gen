#!/usr/bin/env python3
"""Run the frozen R-A2 RaLD-WCE memorization and loss/sampler pilots."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_tesseract  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
    nearest_distance,
)
from eval.rald_wce_stage0 import (  # noqa: E402
    ExactExportCapacityError,
    FORMAL_OUTPUT_RANGE_QUOTAS,
    FORMAL_Q0_RANGE_QUOTAS,
    FORMAL_Q1_ANCHOR_QUOTAS,
    FORMAL_Q1_SAMPLES_PER_ANCHOR,
    WideInferenceConfig,
    infer_exact_wce,
)
from losses.rald_wce_pilot import (  # noqa: E402
    PILOT_MODES,
    RANGE_NEGATIVE_GLOBAL_PER_CLASS,
    RANGE_NEGATIVE_PER_CLASS,
    RANGE_NEGATIVE_SHELL_PER_CLASS,
    RANGE_POSITIVE_QUOTAS,
    SOURCE_NEGATIVE_COUNT,
    SOURCE_POSITIVE_COUNT,
    rald_wce_pilot_loss,
    sample_pilot_queries,
)
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_wce_field import RaLDWCEField  # noqa: E402
from scripts.g1b_contract import sha256  # noqa: E402


PROTOCOL = "g1_ra2_rald_wce_pilot_v1"
FORMAL_SEED = 20260716
FORMAL_TRAIN_COUNT = 76
FORMAL_VALIDATION_COUNT = 24
FORMAL_FAR_VALIDATION_COUNT = 23
TINY_FRAME_COUNT = 8
TINY_MINIMUM_SPARSE_COUNT = 2
TINY_MINIMUM_FAR_COUNT = 4
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
CORRECTED_DENSE_GEOMETRY_SHA256 = (
    "e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68"
)
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_CACHE_ARRAYS = ("target_xyz_confidence", "target_rae_index")
FORBIDDEN_RECORD_KEY_FRAGMENTS = ("future", "next_cube", "test")


@dataclass(frozen=True)
class PilotConfig:
    protocol: str
    mode: str
    seed: int
    train_frame_count: int
    validation_frame_count: int
    maximum_updates: int
    epochs: int | None
    evaluation_updates: tuple[int, ...]
    checkpoint_interval_updates: int
    learning_rate: float
    minimum_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    occupancy_query_count: int
    positive_query_count: int
    negative_query_count: int
    range_positive_quotas: tuple[int, int, int] | None
    range_negative_per_class: int | None
    range_negative_global_per_class: int | None
    range_negative_shell_per_class: int | None
    latent_count: int
    model_dim: int
    depth: int
    heads: int
    head_dim: int
    decode_chunk_size: int
    q0_range_quotas: tuple[int, int, int]
    q1_anchor_quotas: tuple[int, int, int]
    q1_samples_per_anchor: int
    output_range_quotas: tuple[int, int, int]
    minimum_export_distance_m: float
    numeric_mode: str
    test_accessed: bool
    future_cube_accessed: bool
    auxiliary_radar_point_array_accessed: bool
    doppler_head: bool


def frozen_pilot_config(mode: str) -> PilotConfig:
    if mode not in PILOT_MODES:
        raise ValueError(f"Unknown R-A2 pilot mode: {mode}")
    tiny = mode == "tiny_memorization"
    range_sampler = mode == "range_class_sampler"
    positive_count = (
        sum(RANGE_POSITIVE_QUOTAS) if range_sampler else SOURCE_POSITIVE_COUNT
    )
    negative_count = (
        3 * RANGE_NEGATIVE_PER_CLASS if range_sampler else SOURCE_NEGATIVE_COUNT
    )
    return PilotConfig(
        protocol=PROTOCOL,
        mode=mode,
        seed=FORMAL_SEED,
        train_frame_count=TINY_FRAME_COUNT if tiny else FORMAL_TRAIN_COUNT,
        validation_frame_count=TINY_FRAME_COUNT if tiny else FORMAL_VALIDATION_COUNT,
        maximum_updates=500 if tiny else 380,
        epochs=None if tiny else 5,
        evaluation_updates=(100, 200, 300, 400, 500) if tiny else (228, 380),
        checkpoint_interval_updates=25 if tiny else FORMAL_TRAIN_COUNT,
        learning_rate=1e-4,
        minimum_learning_rate=1e-6,
        weight_decay=0.05,
        gradient_clip_norm=10.0,
        occupancy_query_count=positive_count + negative_count,
        positive_query_count=positive_count,
        negative_query_count=negative_count,
        range_positive_quotas=RANGE_POSITIVE_QUOTAS if range_sampler else None,
        range_negative_per_class=(
            RANGE_NEGATIVE_PER_CLASS if range_sampler else None
        ),
        range_negative_global_per_class=(
            RANGE_NEGATIVE_GLOBAL_PER_CLASS if range_sampler else None
        ),
        range_negative_shell_per_class=(
            RANGE_NEGATIVE_SHELL_PER_CLASS if range_sampler else None
        ),
        latent_count=512,
        model_dim=512,
        depth=6,
        heads=8,
        head_dim=64,
        decode_chunk_size=8_192,
        q0_range_quotas=FORMAL_Q0_RANGE_QUOTAS,
        q1_anchor_quotas=FORMAL_Q1_ANCHOR_QUOTAS,
        q1_samples_per_anchor=FORMAL_Q1_SAMPLES_PER_ANCHOR,
        output_range_quotas=FORMAL_OUTPUT_RANGE_QUOTAS,
        minimum_export_distance_m=0.05,
        numeric_mode="fp32_parameters_bf16_cuda_autocast",
        test_accessed=False,
        future_cube_accessed=False,
        auxiliary_radar_point_array_accessed=False,
        doppler_head=False,
    )


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    temporary.replace(path)


def canonical_digest(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repo,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if not SOURCE_PATTERN.fullmatch(source_commit):
        raise ValueError("R-A2 source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("R-A2 source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"R-A2 source worktree is dirty: {dirty}")


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("R-A2 pilots require CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("R-A2 pilots are H200 CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"R-A2 pilots require H200, got {resolved}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("R-A2 pilots require BF16 support")
    return device, resolved


def load_normalization(path: Path) -> tuple[float, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    values = document.get("normalization", document)
    center = float(values["center"])
    scale = float(values["scale"])
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("R-A2 normalization is invalid")
    return center, scale


def validate_manifest_document(document: dict[str, Any]) -> dict[str, int]:
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise ValueError("R-A2 manifest does not contain frames")
    counts = Counter(frame.get("partition") for frame in frames)
    expected = {"train": FORMAL_TRAIN_COUNT, "validation": FORMAL_VALIDATION_COUNT}
    if dict(counts) != expected:
        raise ValueError(f"R-A2 requires frozen 76/24 data, got {dict(counts)}")
    identities = set()
    for frame in frames:
        if frame.get("partition") not in expected:
            raise ValueError("R-A2 manifest contains a forbidden partition")
        lowered_keys = [str(key).lower() for key in frame]
        if any(
            fragment in key
            for key in lowered_keys
            for fragment in FORBIDDEN_RECORD_KEY_FRAGMENTS
        ):
            raise ValueError("R-A2 manifest contains a future/test input field")
        identity = (int(frame["sequence"]), int(frame["radar_index"]))
        if identity in identities:
            raise ValueError("R-A2 development manifest has duplicate identities")
        identities.add(identity)
    return {
        "train_frame_count": counts["train"],
        "validation_frame_count": counts["validation"],
        "test_frame_count": 0,
        "future_cube_frame_count": 0,
    }


def validate_frozen_inputs(
    manifest: Path,
    scene_split: Path,
    normalization: Path,
    repo: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    hashes = {
        "manifest_sha256": sha256(manifest),
        "scene_split_sha256": sha256(scene_split),
        "normalization_sha256": sha256(normalization),
        "dense_geometry_evaluator_sha256": sha256(
            repo / "code/eval/dense_geometry.py"
        ),
        "exact_wce_evaluator_sha256": sha256(
            repo / "code/eval/rald_wce_stage0.py"
        ),
    }
    if hashes["manifest_sha256"] != FROZEN_MANIFEST_SHA256:
        raise ValueError("R-A2 manifest hash differs from the frozen protocol")
    if hashes["scene_split_sha256"] != FROZEN_SCENE_SPLIT_SHA256:
        raise ValueError("R-A2 scene-split hash differs from the frozen protocol")
    if (
        hashes["dense_geometry_evaluator_sha256"]
        != CORRECTED_DENSE_GEOMETRY_SHA256
    ):
        raise ValueError("R-A2 corrected geometry evaluator hash changed")
    split = json.loads(scene_split.read_text(encoding="utf-8"))
    if split.get("gate_pass") is not True:
        raise ValueError("R-A2 scene split failed its leakage gate")
    if not normalization.is_file():
        raise FileNotFoundError(normalization)
    counts = validate_manifest_document(
        json.loads(manifest.read_text(encoding="utf-8"))
    )
    return hashes, counts


def _cache_path(cache_root: Path, sequence: int, radar_index: int) -> Path:
    return cache_root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


class PilotCubeDataset(Dataset):
    """Read only the current Cube and the two approved current-target arrays."""

    def __init__(
        self,
        data_root: Path,
        cache_root: Path,
        manifest: Path,
        partition: str,
    ) -> None:
        if partition not in ("train", "validation"):
            raise ValueError("R-A2 dataset permits only train or validation")
        document = json.loads(manifest.read_text(encoding="utf-8"))
        self.records = [
            record for record in document["frames"]
            if record["partition"] == partition
        ]
        if not self.records:
            raise ValueError(f"R-A2 has no {partition} records")
        self.data_root = data_root
        self.cache_root = cache_root

    def __len__(self) -> int:
        return len(self.records)

    def _target_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        path = _cache_path(
            self.cache_root,
            int(record["sequence"]),
            int(record["radar_index"]),
        )
        with np.load(path) as cache:
            target = cache[ALLOWED_CACHE_ARRAYS[0]].astype(np.float32)
            target_index = cache[ALLOWED_CACHE_ARRAYS[1]].astype(np.int64)
        return target, target_index

    def target_statistics(self, index: int) -> dict[str, Any]:
        target, _ = self._target_arrays(index)
        radius = np.linalg.norm(target[:, :3], axis=1)
        record = self.records[index]
        return {
            "index": index,
            "sequence": int(record["sequence"]),
            "radar_index": int(record["radar_index"]),
            "target_count": int(target.shape[0]),
            "has_far_target": bool(np.any((radius >= 60.0) & (radius < 120.0))),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sequence = int(record["sequence"])
        radar_index = int(record["radar_index"])
        cube = load_tesseract(
            self.data_root
            / str(sequence)
            / "radar_tesseract"
            / f"tesseract_{radar_index:05d}.mat"
        ).astype(np.float32, copy=False)
        target, target_index = self._target_arrays(index)
        return {
            "cube_drae": torch.from_numpy(cube),
            "target_xyz_confidence": torch.from_numpy(target),
            "target_rae_index": torch.from_numpy(target_index),
            "sequence": sequence,
            "radar_index": radar_index,
            "partition": str(record["partition"]),
        }


def select_tiny_subset(statistics: list[dict[str, Any]]) -> list[int]:
    """Freeze eight train frames with two global sparse extrema and four far frames."""

    if len(statistics) != FORMAL_TRAIN_COUNT:
        raise ValueError("R-A2 tiny selection requires all 76 train statistics")
    sparse_order = sorted(
        statistics,
        key=lambda row: (
            int(row["target_count"]),
            int(row["sequence"]),
            int(row["radar_index"]),
        ),
    )
    selected = [int(row["index"]) for row in sparse_order[:TINY_MINIMUM_SPARSE_COUNT]]
    by_index = {int(row["index"]): row for row in statistics}
    far_candidates = sorted(
        (row for row in statistics if bool(row["has_far_target"])),
        key=lambda row: (
            int(row["target_count"]),
            int(row["sequence"]),
            int(row["radar_index"]),
        ),
    )
    if len(far_candidates) < TINY_MINIMUM_FAR_COUNT:
        raise ValueError("R-A2 tiny selection cannot provide four far frames")
    for row in far_candidates:
        far_count = sum(bool(by_index[index]["has_far_target"]) for index in selected)
        if far_count >= TINY_MINIMUM_FAR_COUNT:
            break
        index = int(row["index"])
        if index not in selected:
            selected.append(index)

    remaining = sorted(
        (row for row in statistics if int(row["index"]) not in selected),
        key=lambda row: (
            int(row["sequence"]) in {
                int(by_index[index]["sequence"]) for index in selected
            },
            int(row["sequence"]),
            int(row["radar_index"]),
        ),
    )
    for row in remaining:
        if len(selected) == TINY_FRAME_COUNT:
            break
        selected.append(int(row["index"]))
    if len(selected) != TINY_FRAME_COUNT:
        raise ValueError("R-A2 tiny selection did not produce exactly eight frames")
    if len({int(by_index[index]["sequence"]) for index in selected}) < 2:
        raise ValueError("R-A2 tiny selection requires at least two scenes")
    if sum(bool(by_index[index]["has_far_target"]) for index in selected) < 4:
        raise AssertionError("R-A2 tiny selection lost the four-far-frame contract")
    if not set(int(row["index"]) for row in sparse_order[:2]).issubset(selected):
        raise AssertionError("R-A2 tiny selection lost a sparse extreme")
    return selected


def cross_scene_wrong_indices(
    records: list[dict[str, Any]],
    allowed_indices: list[int],
) -> dict[int, int]:
    sequences = {index: int(records[index]["sequence"]) for index in allowed_indices}
    result: dict[int, int] = {}
    for position, index in enumerate(allowed_indices):
        for shift in range(1, len(allowed_indices)):
            candidate = allowed_indices[(position + shift) % len(allowed_indices)]
            if sequences[candidate] != sequences[index]:
                result[index] = candidate
                break
        if index not in result:
            raise ValueError("R-A2 cannot build a cross-scene wrong condition")
    return result


def deterministic_frame_order(
    indices: list[int],
    *,
    seed: int,
    cycle: int,
) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed + cycle)
    order = torch.randperm(len(indices), generator=generator).tolist()
    return [indices[position] for position in order]


def frame_seed(base: int, item: dict[str, Any], update: int) -> int:
    return int(
        (
            base
            + 1_000_003 * update
            + 10_007 * int(item["sequence"])
            + 101 * int(item["radar_index"])
        )
        % (2**31 - 1)
    )


def cosine_learning_rate(config: PilotConfig, update_count: int) -> float:
    if config.mode == "tiny_memorization":
        return config.learning_rate
    epoch_index = min(update_count // FORMAL_TRAIN_COUNT, 4)
    fraction = epoch_index / 4.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def build_model(
    config: PilotConfig,
    *,
    log_center: float,
    log_scale: float,
    device: torch.device,
) -> RaLDWCEField:
    return RaLDWCEField(
        log_center=log_center,
        log_scale=log_scale,
        latent_count=config.latent_count,
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
        decode_chunk_size=config.decode_chunk_size,
    ).to(device)


def build_inference_config(config: PilotConfig) -> WideInferenceConfig:
    return WideInferenceConfig(
        q0_range_quotas=config.q0_range_quotas,
        q1_anchor_quotas=config.q1_anchor_quotas,
        q1_samples_per_anchor=config.q1_samples_per_anchor,
        output_range_quotas=config.output_range_quotas,
        minimum_distance_m=config.minimum_export_distance_m,
        seed=config.seed,
        decode_chunk_size=config.decode_chunk_size,
    )


def _aggregate_scalars(reports: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(reports[0])
    if not reports or any(sorted(report) != keys for report in reports):
        raise ValueError("R-A2 scalar reports are empty or inconsistent")
    return {
        key: {
            "mean": float(np.mean([report[key] for report in reports])),
            "median": float(np.median([report[key] for report in reports])),
            "std": float(np.std([report[key] for report in reports])),
            "sample_count": len(reports),
        }
        for key in keys
    }


def _far_recall_1m(
    prediction_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_weight: torch.Tensor,
) -> float | None:
    radius = torch.linalg.vector_norm(target_xyz, dim=1)
    mask = (radius >= 60.0) & (radius < 120.0)
    if not bool(mask.any()):
        return None
    distances = nearest_distance(target_xyz[mask], prediction_xyz)
    weights = target_weight[mask].clamp_min(0.0)
    return float(
        (((distances <= 1.0).to(weights) * weights).sum() / weights.sum().clamp_min(1e-8)).item()
    )


@torch.no_grad()
def evaluate_pilot(
    model: RaLDWCEField,
    dataset: PilotCubeDataset,
    indices: list[int],
    wrong_indices: dict[int, int],
    inference_config: WideInferenceConfig,
    axes: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Use the formal exact-10k evaluator and add uncensored far recall."""

    model.eval()
    range_m = torch.as_tensor(axes.range_m, dtype=torch.float32, device=device)
    azimuth = torch.as_tensor(axes.azimuth_rad, dtype=torch.float32, device=device)
    elevation = torch.as_tensor(
        axes.elevation_rad,
        dtype=torch.float32,
        device=device,
    )
    frames: list[dict[str, Any]] = []
    for index in indices:
        item = dataset[index]
        wrong_item = dataset[wrong_indices[index]]
        if item["partition"] not in ("train", "validation"):
            raise ValueError("R-A2 evaluator reached a forbidden partition")
        cube = item["cube_drae"].unsqueeze(0).to(device)
        wrong_cube = wrong_item["cube_drae"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference = infer_exact_wce(
                model,
                cube,
                wrong_cube,
                range_m,
                azimuth,
                elevation,
                inference_config,
            )
        target = item["target_xyz_confidence"].to(device)
        target_xyz = target[:, :3].float()
        target_weight = target[:, 3].float()
        matched_xyz = inference.matched.xyz_m.to(device)
        wrong_xyz = inference.wrong_condition.xyz_m.to(device)
        matched_geometry = geometry_report(
            matched_xyz,
            target_xyz,
            target_weight=target_weight,
        )
        wrong_geometry = geometry_report(
            wrong_xyz,
            target_xyz,
            target_weight=target_weight,
        )
        matched_far_recall = _far_recall_1m(
            matched_xyz,
            target_xyz,
            target_weight,
        )
        wrong_far_recall = _far_recall_1m(
            wrong_xyz,
            target_xyz,
            target_weight,
        )
        matched_chamfer = float(matched_geometry["chamfer_m"])
        wrong_chamfer = float(wrong_geometry["chamfer_m"])
        frames.append(
            {
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
                "partition": str(item["partition"]),
                "wrong_condition_sequence": int(wrong_item["sequence"]),
                "wrong_condition_radar_index": int(wrong_item["radar_index"]),
                "has_far_target": matched_far_recall is not None,
                "matched": matched_geometry,
                "wrong_condition": wrong_geometry,
                "matched_far_recall_1m": matched_far_recall,
                "wrong_far_recall_1m": wrong_far_recall,
                "condition_intervention": {
                    "wrong_minus_matched_chamfer_fraction": (
                        wrong_chamfer / max(matched_chamfer, 1e-12) - 1.0
                    ),
                    "matched_chamfer_better": float(
                        matched_chamfer < wrong_chamfer
                    ),
                },
                "matched_export": inference.matched.report,
                "wrong_export": inference.wrong_condition.report,
                "inference": inference.report,
            }
        )
        del item, wrong_item, cube, wrong_cube, target
        torch.cuda.empty_cache()

    far_frames = [frame for frame in frames if frame["has_far_target"]]
    if not far_frames:
        raise ValueError("R-A2 evaluation has no far-target frame")
    minimum_pair_distance = min(
        float(frame["matched_export"]["observed_minimum_pair_distance_m"])
        for frame in frames
    )
    return {
        "frame_count": len(frames),
        "scene_count": len({frame["sequence"] for frame in frames}),
        "far_target_frame_count": len(far_frames),
        "frames": frames,
        "matched": aggregate_geometry_reports(
            [frame["matched"] for frame in frames]
        ),
        "wrong_condition": aggregate_geometry_reports(
            [frame["wrong_condition"] for frame in frames]
        ),
        "condition_intervention": _aggregate_scalars(
            [frame["condition_intervention"] for frame in frames]
        ),
        "far_recall_1m": {
            "matched": _aggregate_scalars(
                [{"value": float(frame["matched_far_recall_1m"])} for frame in far_frames]
            )["value"],
            "wrong_condition": _aggregate_scalars(
                [{"value": float(frame["wrong_far_recall_1m"])} for frame in far_frames]
            )["value"],
        },
        "exact_export": {
            "point_count_per_frame": 10_000,
            "minimum_observed_pair_distance_m": minimum_pair_distance,
            "all_matched_exports_exact_10000": all(
                frame["matched_export"]["exact_point_count"] == 10_000
                for frame in frames
            ),
            "all_wrong_exports_exact_10000": all(
                frame["wrong_export"]["exact_point_count"] == 10_000
                for frame in frames
            ),
            "copy_padding_jitter_duplicate": False,
            "same_query_wrong_condition": all(
                frame["inference"]["matched_wrong_query_hashes_identical"]
                for frame in frames
            ),
        },
        "evidence_boundary": {
            "test_partition_accessed": False,
            "future_cube_accessed": False,
            "cache_arrays_read": list(ALLOWED_CACHE_ARRAYS),
            "ground_truth_accessed_for_query_or_selection": False,
        },
    }


def metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "chamfer_mean_m": float(metrics["matched"]["chamfer_m"]["mean"]),
        "outlier_fraction_mean": float(
            metrics["matched"]["outlier_fraction_2m"]["mean"]
        ),
        "completeness_median_m": float(
            metrics["matched"]["completeness_mean_distance_m"]["median"]
        ),
        "completeness_mean_m": float(
            metrics["matched"]["completeness_mean_distance_m"]["mean"]
        ),
        "far_fscore_1m_mean": float(
            metrics["matched"]["range_60_120m_fscore_1m"]["mean"]
        ),
        "far_recall_1m_mean": float(metrics["far_recall_1m"]["matched"]["mean"]),
        "wrong_condition_chamfer_degradation_mean": float(
            metrics["condition_intervention"][
                "wrong_minus_matched_chamfer_fraction"
            ]["mean"]
        ),
        "matched_condition_win_fraction": float(
            metrics["condition_intervention"]["matched_chamfer_better"]["mean"]
        ),
    }


def tiny_memorization_gate(values: dict[str, float]) -> dict[str, Any]:
    checks = {
        "chamfer_mean_at_most_1m": values["chamfer_mean_m"] <= 1.0,
        "outlier_fraction_mean_at_most_10pct": (
            values["outlier_fraction_mean"] <= 0.10
        ),
        "completeness_mean_at_most_0p75m": (
            values["completeness_mean_m"] <= 0.75
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def epoch3_screen(
    values: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, Any]:
    deltas = {
        "outlier_improvement_fraction": (
            baseline["outlier_fraction_mean"] - values["outlier_fraction_mean"]
        ),
        "far_recall_improvement_fraction": (
            values["far_recall_1m_mean"] - baseline["far_recall_1m_mean"]
        ),
    }
    checks = {
        "outlier_improves_at_least_1pp": (
            deltas["outlier_improvement_fraction"] >= 0.01
        ),
        "far_recall_improves_at_least_5pp": (
            deltas["far_recall_improvement_fraction"] >= 0.05
        ),
    }
    return {
        "deltas": deltas,
        "checks": checks,
        "continue_to_epoch5": any(checks.values()),
    }


def final_promotion(
    values: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, Any]:
    deltas = {
        "outlier_improvement_fraction": (
            baseline["outlier_fraction_mean"] - values["outlier_fraction_mean"]
        ),
        "far_fscore_relative_improvement": (
            values["far_fscore_1m_mean"]
            / max(baseline["far_fscore_1m_mean"], 1e-12)
            - 1.0
        ),
        "completeness_median_degradation_m": (
            values["completeness_median_m"]
            - baseline["completeness_median_m"]
        ),
    }
    checks = {
        "outlier_improves_at_least_2pp": (
            deltas["outlier_improvement_fraction"] >= 0.02
        ),
        "far_fscore_relative_improves_at_least_25pct": (
            deltas["far_fscore_relative_improvement"] >= 0.25
        ),
        "completeness_median_degrades_at_most_0p25m": (
            deltas["completeness_median_degradation_m"] <= 0.25
        ),
    }
    return {"deltas": deltas, "checks": checks, "promotion_passed": all(checks.values())}


def parse_formal_metric_values(document: dict[str, Any]) -> dict[str, float]:
    metrics = document.get("metrics", document)
    return {
        "outlier_fraction_mean": float(
            metrics["matched"]["outlier_fraction_2m"]["mean"]
        ),
        "completeness_median_m": float(
            metrics["matched"]["completeness_mean_distance_m"]["median"]
        ),
        "far_fscore_1m_mean": float(
            metrics["matched"]["range_60_120m_fscore_1m"]["mean"]
        ),
    }


def verify_formal_reference(
    reference_values: dict[str, float],
    formal_metrics: dict[str, Any],
    *,
    tolerance: float = 1e-4,
) -> None:
    frozen = parse_formal_metric_values(formal_metrics)
    for key, expected in frozen.items():
        if abs(float(reference_values[key]) - expected) > tolerance:
            raise ValueError(f"R-A2 formal reference differs for {key}")


def resume_contract(
    *,
    source_commit: str,
    config: PilotConfig,
    input_hashes: dict[str, str],
    source_hashes: dict[str, str],
    tiny_frame_ids: list[dict[str, int]],
    baseline_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "config": asdict(config),
        "input_hashes": input_hashes,
        "source_hashes": source_hashes,
        "tiny_frame_ids": tiny_frame_ids,
        "baseline_hashes": baseline_hashes,
    }


def validate_resume_manifest(
    manifest: dict[str, Any],
    expected_contract: dict[str, Any],
) -> None:
    if manifest.get("resume_contract_sha256") != canonical_digest(expected_contract):
        raise ValueError("R-A2 resume contract differs from the existing run")
    if manifest.get("resume_contract") != expected_contract:
        raise ValueError("R-A2 resume metadata differs from the existing run")


def initial_run_state() -> dict[str, Any]:
    return {
        "updates_completed": 0,
        "consecutive_tiny_passes": 0,
        "evaluation_updates_completed": [],
        "loss_history": [],
        "status": "running",
    }


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    contract_sha256: str,
) -> None:
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    atomic_torch_save(
        path,
        {
            "protocol": PROTOCOL,
            "resume_contract_sha256": contract_sha256,
            "state": state,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": cuda_rng,
            },
        },
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    contract_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("R-A2 checkpoint protocol differs")
    if checkpoint.get("resume_contract_sha256") != contract_sha256:
        raise ValueError("R-A2 checkpoint resume contract differs")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch_cpu"])
    if torch.cuda.is_available() and checkpoint["rng"]["torch_cuda"]:
        torch.cuda.set_rng_state_all(checkpoint["rng"]["torch_cuda"])
    return checkpoint["state"]


def prepare_output(path: Path, *, resume: bool) -> None:
    if resume:
        if not (path / "run_manifest.json").is_file():
            raise FileNotFoundError(f"No R-A2 run to resume: {path}")
        if (path / "summary.json").exists():
            raise FileExistsError("R-A2 run is already complete")
        return
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"R-A2 output is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _source_hashes(repo: Path) -> dict[str, str]:
    paths = (
        repo / "code/models/rald_wce_field.py",
        repo / "code/losses/rald_wce_pilot.py",
        repo / "code/eval/rald_wce_stage0.py",
        repo / "code/eval/dense_geometry.py",
        repo / "code/scripts/train_rald_wce_pilot.py",
    )
    return {str(path.resolve()): sha256(path) for path in paths}


def _tiny_frame_ids(
    statistics: list[dict[str, Any]],
    indices: list[int],
) -> list[dict[str, int]]:
    by_index = {int(row["index"]): row for row in statistics}
    return [
        {
            "sequence": int(by_index[index]["sequence"]),
            "radar_index": int(by_index[index]["radar_index"]),
            "target_count": int(by_index[index]["target_count"]),
            "has_far_target": int(bool(by_index[index]["has_far_target"])),
        }
        for index in indices
    ]


def _load_formal_checkpoint(
    path: Path,
    model: torch.nn.Module,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("R-A2 formal-best checkpoint has no model state")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=PILOT_MODES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--formal-best-checkpoint", type=Path)
    parser.add_argument("--formal-best-metrics", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser


def _move_queries(
    queries: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in queries.items()
    }


def _write_progress(output: Path, state: dict[str, Any]) -> None:
    atomic_json(
        output / "progress.json",
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "state": state,
        },
    )


def _formal_reference(
    *,
    args: argparse.Namespace,
    config: PilotConfig,
    log_center: float,
    log_scale: float,
    validation_dataset: PilotCubeDataset,
    validation_indices: list[int],
    validation_wrong: dict[int, int],
    inference_config: WideInferenceConfig,
    axes: Any,
    device: torch.device,
    baseline_hashes: dict[str, str],
) -> dict[str, float]:
    path = args.output_dir / "formal_reference.json"
    if args.resume:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["baseline_hashes"] != baseline_hashes:
            raise ValueError("R-A2 saved formal reference hashes differ")
        return document["values"]

    model = build_model(
        config,
        log_center=log_center,
        log_scale=log_scale,
        device=device,
    )
    checkpoint = _load_formal_checkpoint(args.formal_best_checkpoint, model)
    metrics = evaluate_pilot(
        model,
        validation_dataset,
        validation_indices,
        validation_wrong,
        inference_config,
        axes,
        device,
    )
    if metrics["frame_count"] != FORMAL_VALIDATION_COUNT:
        raise ValueError("R-A2 formal reference did not cover 24 validation frames")
    if metrics["far_target_frame_count"] != FORMAL_FAR_VALIDATION_COUNT:
        raise ValueError("R-A2 formal reference did not cover 23 far frames")
    values = metric_values(metrics)
    formal_metrics = json.loads(
        args.formal_best_metrics.read_text(encoding="utf-8")
    )
    verify_formal_reference(values, formal_metrics)
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "baseline_hashes": baseline_hashes,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "values": values,
        "metrics": metrics,
        "far_recall_recomputed_by_pilot_evaluator": True,
    }
    atomic_json(path, document)
    del model
    torch.cuda.empty_cache()
    return values


def main() -> None:
    args = build_parser().parse_args()
    config = frozen_pilot_config(args.mode)
    if config.mode != "tiny_memorization":
        if args.formal_best_checkpoint is None or args.formal_best_metrics is None:
            raise ValueError(
                "R-A2 five-epoch pilots require formal-best checkpoint and metrics"
            )
    elif args.formal_best_checkpoint is not None or args.formal_best_metrics is not None:
        raise ValueError("R-A2 tiny mode does not accept formal-best inputs")

    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    prepare_output(args.output_dir, resume=args.resume)
    device, device_name = require_h200(args.device)
    input_hashes, manifest_counts = validate_frozen_inputs(
        args.manifest,
        args.scene_split,
        args.normalization,
        repo,
    )
    baseline_hashes = None
    if config.mode != "tiny_memorization":
        baseline_hashes = {
            "formal_best_checkpoint_sha256": sha256(args.formal_best_checkpoint),
            "formal_best_metrics_sha256": sha256(args.formal_best_metrics),
        }
    log_center, log_scale = load_normalization(args.normalization)
    axes = load_axes(args.data_root / "resources")
    train_dataset = PilotCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        "train",
    )
    validation_dataset = PilotCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        "validation",
    )
    all_train_indices = list(range(len(train_dataset)))
    validation_indices = list(range(len(validation_dataset)))
    tiny_statistics: list[dict[str, Any]] = []
    if config.mode == "tiny_memorization":
        tiny_statistics = [
            train_dataset.target_statistics(index) for index in all_train_indices
        ]
        train_indices = select_tiny_subset(tiny_statistics)
        evaluation_dataset = train_dataset
        evaluation_indices = train_indices
    else:
        train_indices = all_train_indices
        evaluation_dataset = validation_dataset
        evaluation_indices = validation_indices
    train_wrong = cross_scene_wrong_indices(train_dataset.records, train_indices)
    validation_wrong = cross_scene_wrong_indices(
        validation_dataset.records,
        validation_indices,
    )
    evaluation_wrong = (
        train_wrong if config.mode == "tiny_memorization" else validation_wrong
    )
    tiny_ids = _tiny_frame_ids(tiny_statistics, train_indices) if tiny_statistics else []
    source_hashes = _source_hashes(repo)
    contract = resume_contract(
        source_commit=args.source_commit,
        config=config,
        input_hashes=input_hashes,
        source_hashes=source_hashes,
        tiny_frame_ids=tiny_ids,
        baseline_hashes=baseline_hashes,
    )
    contract_sha = canonical_digest(contract)

    if args.resume:
        run_manifest = json.loads(
            (args.output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        validate_resume_manifest(run_manifest, contract)
    else:
        architecture_probe = build_model(
            config,
            log_center=log_center,
            log_scale=log_scale,
            device=device,
        )
        run_manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "source_commit": args.source_commit,
            "resume_contract_sha256": contract_sha,
            "resume_contract": contract,
            "manifest_counts": manifest_counts,
            "architecture": architecture_probe.architecture_metadata(),
            "runtime": {
                "device_argument": args.device,
                "device_name": device_name,
                "torch_version": torch.__version__,
                "parameter_count": parameter_count(architecture_probe),
                "bf16_autocast": True,
            },
            "data_access_contract": {
                "partitions": ["train", "validation"],
                "test_partition_accessed": False,
                "future_cube_accessed": False,
                "cache_arrays_read": list(ALLOWED_CACHE_ARRAYS),
                "auxiliary_radar_point_array_accessed": False,
            },
        }
        atomic_json(args.output_dir / "run_manifest.json", run_manifest)
        del architecture_probe
        torch.cuda.empty_cache()

    inference_config = build_inference_config(config)
    baseline_values = None
    if config.mode != "tiny_memorization":
        baseline_values = _formal_reference(
            args=args,
            config=config,
            log_center=log_center,
            log_scale=log_scale,
            validation_dataset=validation_dataset,
            validation_indices=validation_indices,
            validation_wrong=validation_wrong,
            inference_config=inference_config,
            axes=axes,
            device=device,
            baseline_hashes=baseline_hashes,
        )

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    model = build_model(
        config,
        log_center=log_center,
        log_scale=log_scale,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = args.output_dir / "last.pt"
    if args.resume:
        state = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            contract_sha256=contract_sha,
        )
    else:
        state = initial_run_state()
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            state=state,
            contract_sha256=contract_sha,
        )
        _write_progress(args.output_dir, state)

    started = time.monotonic()
    stop_reason = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        while True:
            updates = int(state["updates_completed"])
            evaluation_due = (
                updates in config.evaluation_updates
                and updates not in state["evaluation_updates_completed"]
            )
            if evaluation_due:
                save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    contract_sha256=contract_sha,
                )
                metrics = evaluate_pilot(
                    model,
                    evaluation_dataset,
                    evaluation_indices,
                    evaluation_wrong,
                    inference_config,
                    axes,
                    device,
                )
                values = metric_values(metrics)
                if config.mode == "tiny_memorization":
                    decision = tiny_memorization_gate(values)
                    state["consecutive_tiny_passes"] = (
                        int(state["consecutive_tiny_passes"]) + 1
                        if decision["passed"]
                        else 0
                    )
                    decision["consecutive_passes"] = int(
                        state["consecutive_tiny_passes"]
                    )
                    decision["early_stop_passed"] = (
                        state["consecutive_tiny_passes"] >= 2
                    )
                    metrics_name = f"metrics_update{updates:04d}.json"
                elif updates == 228:
                    decision = epoch3_screen(values, baseline_values)
                    metrics_name = "metrics_epoch003.json"
                    if not decision["continue_to_epoch5"]:
                        stop_reason = "stopped_epoch3_screen_no_go"
                else:
                    decision = final_promotion(values, baseline_values)
                    metrics_name = "metrics_epoch005.json"
                    stop_reason = (
                        "promoted"
                        if decision["promotion_passed"]
                        else "completed_epoch5_no_go"
                    )
                evaluation = {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "mode": config.mode,
                    "source_commit": args.source_commit,
                    "updates_completed": updates,
                    "epoch": (
                        None
                        if config.mode == "tiny_memorization"
                        else updates // FORMAL_TRAIN_COUNT
                    ),
                    "checkpoint_sha256": sha256(checkpoint_path),
                    "values": values,
                    "metrics": metrics,
                    "decision": decision,
                    "baseline_values": baseline_values,
                }
                atomic_json(args.output_dir / metrics_name, evaluation)
                state["evaluation_updates_completed"].append(updates)
                if (
                    config.mode == "tiny_memorization"
                    and decision["early_stop_passed"]
                ):
                    stop_reason = "tiny_memorization_passed_early"
                if config.mode == "tiny_memorization" and updates == 500:
                    stop_reason = (
                        stop_reason
                        or "architecture_optimization_no_go_at_500_updates"
                    )
                save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    contract_sha256=contract_sha,
                )
                _write_progress(args.output_dir, state)
                if stop_reason is not None:
                    break

            if updates >= config.maximum_updates:
                if config.mode == "tiny_memorization":
                    stop_reason = "architecture_optimization_no_go_at_500_updates"
                else:
                    stop_reason = "completed_without_final_decision"
                break

            cycle = updates // len(train_indices) + 1
            order = deterministic_frame_order(
                train_indices,
                seed=config.seed,
                cycle=cycle,
            )
            index = order[updates % len(train_indices)]
            item = train_dataset[index]
            wrong_item = train_dataset[train_wrong[index]]
            sampled = sample_pilot_queries(
                config.mode,
                item["target_rae_index"],
                spatial_shape=tuple(item["cube_drae"].shape[1:]),
                range_m=torch.as_tensor(axes.range_m, dtype=torch.float32),
                seed=frame_seed(config.seed, item, updates + 1),
            )
            if int(sampled["query_count"]) != config.occupancy_query_count:
                raise AssertionError("R-A2 sampler changed the frozen query count")
            queries = _move_queries(sampled, device)
            cube = item["cube_drae"].unsqueeze(0).to(device)
            wrong_cube = wrong_item["cube_drae"].unsqueeze(0).to(device)
            learning_rate = cosine_learning_rate(config, updates)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            model.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    cube,
                    queries["normalized_rae"],
                    wrong_condition_cube_drae=wrong_cube,
                    chunk_size=config.decode_chunk_size,
                )
                if output.get("same_query_wrong_condition") is not True:
                    raise AssertionError("R-A2 wrong condition did not reuse queries")
                loss = rald_wce_pilot_loss(
                    output["matched"],
                    output["wrong"],
                    queries,
                    mode=config.mode,
                )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            state["updates_completed"] = updates + 1
            state["loss_history"].append(
                {
                    "update": updates + 1,
                    "learning_rate": learning_rate,
                    "total": float(loss.total.detach().item()),
                    "occupancy": float(
                        loss.components["matched_occupancy"].detach().item()
                    ),
                    "wrong_margin": float(
                        loss.components["wrong_condition_margin"].detach().item()
                    ),
                }
            )
            if state["updates_completed"] % config.checkpoint_interval_updates == 0:
                save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    contract_sha256=contract_sha,
                )
                _write_progress(args.output_dir, state)
            print(
                json.dumps(
                    {
                        "mode": config.mode,
                        "update": state["updates_completed"],
                        "loss": state["loss_history"][-1]["total"],
                    }
                ),
                flush=True,
            )
            del item, wrong_item, cube, wrong_cube, output, loss
    except ExactExportCapacityError as error:
        failure = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "mode": config.mode,
            "status": "failed_exact_10000_capacity",
            "message": str(error),
            "capacity_report": error.report,
            "copy_padding_jitter_duplicate": False,
            "scientific_decision_eligible": False,
        }
        atomic_json(args.output_dir / "terminal_capacity_failure.json", failure)
        raise SystemExit(2) from error

    state["status"] = stop_reason
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        state=state,
        contract_sha256=contract_sha,
    )
    _write_progress(args.output_dir, state)
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "mode": config.mode,
        "source_commit": args.source_commit,
        "status": stop_reason,
        "updates_completed": int(state["updates_completed"]),
        "evaluation_updates_completed": state["evaluation_updates_completed"],
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "test_partition_accessed": False,
        "future_cube_accessed": False,
        "auxiliary_radar_point_array_accessed": False,
        "doppler_head_evaluated": False,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
