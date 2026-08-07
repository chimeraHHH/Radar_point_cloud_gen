#!/usr/bin/env python3
"""Run the frozen Q1 geometry-quality ranking tiny pilot."""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
)
from eval.rald_wce_stage0 import (  # noqa: E402
    CAPACITY_DISTANCE_M,
    EXPORT_COUNT,
    FORMAL_OUTPUT_RANGE_QUOTAS,
    FORMAL_Q0_RANGE_QUOTAS,
    FORMAL_Q1_ANCHOR_QUOTAS,
    FORMAL_Q1_SAMPLES_PER_ANCHOR,
    ExactExportCapacityError,
    WideInferenceConfig,
    _candidate_tensors,
    exact_capacity_export,
    fixed_wide_q0,
    occupancy_dependent_q1,
    range_stratum_codes,
    tensor_sha256,
)
from losses.rald_wce_quality import (  # noqa: E402
    continuous_geometry_quality_target,
    rald_wce_quality_loss,
)
from models.rald_wce_field import RaLDWCEField  # noqa: E402
from models.rald_wce_quality import (  # noqa: E402
    RaLDWCEQualityHead,
    quality_parameter_count,
)
from scripts.certify_rald_wce_replay_parent import (  # noqa: E402
    certify_replay_parent,
)
from scripts.diagnose_rald_wce_failure_factors import (  # noqa: E402
    validate_formal_checkpoint,
    validate_formal_metrics,
    validate_run_manifest as validate_formal_run_manifest,
)
from scripts.train_rald_wce_pilot import (  # noqa: E402
    ALLOWED_CACHE_ARRAYS,
    PilotCubeDataset,
    cross_scene_wrong_indices,
    deterministic_frame_order,
    load_normalization,
    metric_values,
    select_tiny_subset,
    tiny_memorization_gate,
    validate_frozen_inputs,
)


PROTOCOL = "g1_q1r_rald_wce_quality_tiny_v1"
PARENT_CERTIFICATE_PROTOCOL = "g1_q1r_replay_parent_certificate_v1"
FORMAL_R_A1_PROTOCOL = "g1_ra1_rald_wce_stage0_v1"
FORMAL_PARENT_SOURCE_COMMIT = "f2a9489d40323d1ef45d85de958f4aea8126e1c8"
ORIGINAL_FORMAL_CHECKPOINT_SHA256 = (
    "5be30e0f1ca23ea3b603abb0f5e330efd3599167362a8e23ab3a5967c411a2a0"
)
ARCHIVED_FORMAL_METRICS_RELATIVE = Path(
    "artifacts/g1/wce_formal_f2a9489/metrics_epoch020.json"
)
ARCHIVED_FORMAL_MANIFEST_RELATIVE = Path(
    "artifacts/g1/wce_formal_f2a9489/run_manifest.json"
)
FORMAL_SEED = 20260716
FROZEN_ORDERED_CACHE_DIGEST_SHA256 = (
    "dd9d296cc10933fce12f4e050b4aa065f82752ab123171ec1a75e8cabbc06e4f"
)
FROZEN_ORDERED_TINY_CUBE_DIGEST_SHA256 = (
    "0bfbdb5eac17f9823033942e41abdb6d3b7303f7b55cb06807d8a0933f8a4b7c"
)
FROZEN_RESOURCE_SHA256 = {
    "info_arr.mat": (
        "53f72b22544aa11bc0057f9b8c2177a7a844fddd0e8ce3f753a989d07159767a"
    ),
    "arr_doppler.mat": (
        "f81e56889c2cedc98eb3eb8a4828e382845e3d4758a36f3b8fc0fce4839e0493"
    ),
}
FROZEN_TINY_FRAME_IDENTITIES = (
    (58, 205),
    (58, 404),
    (57, 404),
    (53, 402),
    (53, 204),
    (50, 405),
    (1, 232),
    (1, 430),
)
FORMAL_CANDIDATE_COUNT = 700_000
TINY_FRAME_COUNT = 8
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
QUALITY_SAMPLE_RANGE_QUOTAS = (12_800, 2_720, 480)
QUALITY_SAMPLE_COUNT = sum(QUALITY_SAMPLE_RANGE_QUOTAS)
QUALITY_EVALUATION_UPDATES = (100, 200, 300, 400, 500)
FORBIDDEN_ARGUMENT_FRAGMENTS = (
    "test",
    "future",
    "doppler",
    "best_of_k",
)


@dataclass(frozen=True)
class QualityTinyConfig:
    protocol: str = PROTOCOL
    seed: int = FORMAL_SEED
    train_frame_count: int = TINY_FRAME_COUNT
    validation_frame_count: int = TINY_FRAME_COUNT
    maximum_updates: int = 500
    evaluation_updates: tuple[int, ...] = QUALITY_EVALUATION_UPDATES
    checkpoint_interval_updates: int = 25
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 5.0
    quality_sample_count: int = QUALITY_SAMPLE_COUNT
    quality_sample_range_quotas: tuple[int, int, int] = (
        QUALITY_SAMPLE_RANGE_QUOTAS
    )
    high_quality_sample_fraction: float = 0.5
    high_quality_pool_fraction: float = 0.1
    distance_temperature_m: float = 1.0
    nearest_target_chunk_size: int = 4_096
    quality_model_dim: int = 128
    quality_heads: int = 4
    quality_head_dim: int = 32
    quality_decode_chunk_size: int = 8_192
    ranking_weight: float = 0.5
    minimum_target_gap: float = 0.05
    q0_range_quotas: tuple[int, int, int] = FORMAL_Q0_RANGE_QUOTAS
    q1_anchor_quotas: tuple[int, int, int] = FORMAL_Q1_ANCHOR_QUOTAS
    q1_samples_per_anchor: int = FORMAL_Q1_SAMPLES_PER_ANCHOR
    output_range_quotas: tuple[int, int, int] = FORMAL_OUTPUT_RANGE_QUOTAS
    minimum_export_distance_m: float = CAPACITY_DISTANCE_M
    numeric_mode: str = "frozen_fp32_base_bf16_quality_head"
    test_accessed: bool = False
    future_accessed: bool = False
    target_at_inference: bool = False
    doppler_head: bool = False
    best_of_k: bool = False


@dataclass(frozen=True)
class FrozenCandidateTrainingFrame:
    refined_normalized_rae: torch.Tensor
    condition_latents: torch.Tensor
    base_confidence: torch.Tensor
    quality_target: torch.Tensor
    nearest_distance_m: torch.Tensor
    range_class: torch.Tensor
    report: dict[str, Any]


def frozen_quality_config() -> QualityTinyConfig:
    config = QualityTinyConfig()
    if config.quality_sample_count != sum(config.quality_sample_range_quotas):
        raise AssertionError("Q1 sampled range quotas changed the query count")
    if config.maximum_updates != 500:
        raise AssertionError("Q1 tiny update budget changed")
    if config.evaluation_updates != QUALITY_EVALUATION_UPDATES:
        raise AssertionError("Q1 tiny evaluation cadence changed")
    if sum(config.output_range_quotas) != EXPORT_COUNT:
        raise AssertionError("Q1 output quotas changed exact cardinality")
    if any(
        (
            config.test_accessed,
            config.future_accessed,
            config.target_at_inference,
            config.doppler_head,
            config.best_of_k,
        )
    ):
        raise AssertionError("Q1 tiny evidence boundary changed")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_artifact_sha256(document: dict[str, Any]) -> str:
    payload = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
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
        raise ValueError("Q1 source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("Q1 source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"Q1 tracked source worktree is dirty: {dirty}")


def validate_replay_parent_certificate(
    certificate_path: Path,
    diagnosis_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    run_manifest_path: Path,
    *,
    expected_certifier_source_commit: str,
    repo: Path,
) -> dict[str, Any]:
    """Bind Q1-R to an explicitly certified source-equivalent replay parent."""

    document = json.loads(certificate_path.read_text(encoding="utf-8"))
    recomputed = certify_replay_parent(
        archived_metrics_path=repo / ARCHIVED_FORMAL_METRICS_RELATIVE,
        archived_manifest_path=repo / ARCHIVED_FORMAL_MANIFEST_RELATIVE,
        replay_checkpoint_path=checkpoint_path,
        replay_metrics_path=metrics_path,
        replay_manifest_path=run_manifest_path,
        replay_diagnosis_path=diagnosis_path,
        certifier_source_commit=expected_certifier_source_commit,
    )
    if document != recomputed:
        raise ValueError(
            "Q1-R replay-parent certificate differs from trainer recomputation"
        )
    identity = document.get("identity")
    checks = document.get("checks")
    if not isinstance(identity, dict) or not isinstance(checks, dict):
        raise ValueError("Q1-R replay-parent certificate is incomplete")
    actual = {
        "replay_checkpoint_sha256": sha256_file(checkpoint_path),
        "replay_metrics_sha256": sha256_file(metrics_path),
        "replay_run_manifest_sha256": sha256_file(run_manifest_path),
        "replay_failure_diagnosis_sha256": sha256_file(diagnosis_path),
    }
    required_checks = {
        "source_config_data_seed_epoch_match": True,
        "formal_metrics_recomputed": True,
        "formal_stage0_decision_preserved": True,
        "validation_frame_contract_preserved": True,
        "candidate_query_contract_preserved": True,
        "replay_diagnosis_bound": True,
        "diagnosis_aggregate_recomputed": True,
        "validation_gt_ranking_oracle_passed": True,
        "training_started": False,
        "checkpoint_modified": False,
        "test_partition_accessed": False,
        "future_cube_accessed": False,
        "cfar_accessed": False,
        "doppler_head_evaluated": False,
    }
    validations = {
        "schema_version": document.get("schema_version") == 1,
        "protocol": document.get("protocol") == PARENT_CERTIFICATE_PROTOCOL,
        "authorized_status": document.get("status")
        == "replay_parent_authorized_for_q1r_tiny",
        "q1r_tiny_authorized": document.get("q1r_tiny_authorized") is True,
        "original_parent_bound": identity.get(
            "original_checkpoint_sha256"
        )
        == ORIGINAL_FORMAL_CHECKPOINT_SHA256,
        "formal_source_bound": identity.get("formal_source_commit")
        == FORMAL_PARENT_SOURCE_COMMIT,
        "certifier_source_bound": identity.get("certifier_source_commit")
        == expected_certifier_source_commit,
        "formal_epoch_20": int(identity.get("formal_epoch", -1)) == 20,
        "replay_artifact_hashes": all(
            identity.get(key) == value for key, value in actual.items()
        ),
        "exact_parent_flag_honest": identity.get("exact_original_checkpoint")
        == (
            actual["replay_checkpoint_sha256"]
            == ORIGINAL_FORMAL_CHECKPOINT_SHA256
        ),
        "required_scientific_checks": all(
            checks.get(key) == expected
            for key, expected in required_checks.items()
        ),
        "claim_boundary": isinstance(document.get("claim_boundary"), str)
        and bool(document["claim_boundary"].strip()),
        "certificate_recomputed": recomputed.get("q1r_tiny_authorized") is True,
    }
    failed = [name for name, passed in validations.items() if not passed]
    if failed:
        raise ValueError(f"Q1-R replay-parent certificate failed: {failed}")
    return {
        "path": str(certificate_path.resolve()),
        "sha256": sha256_file(certificate_path),
        "identity": identity,
        "checks": checks,
        "validation_checks": validations,
        "claim_boundary": document.get("claim_boundary"),
        "trainer_recomputed_certificate": True,
    }


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("Q1 tiny requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("Q1 tiny is H200 CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"Q1 tiny requires H200, got {resolved}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Q1 tiny requires BF16 support")
    return device, resolved


def cache_path(cache_root: Path, record: dict[str, Any]) -> Path:
    sequence = int(record["sequence"])
    radar_index = int(record["radar_index"])
    return cache_root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


def cube_path(data_root: Path, record: dict[str, Any]) -> Path:
    sequence = int(record["sequence"])
    radar_index = int(record["radar_index"])
    return (
        data_root
        / str(sequence)
        / "radar_tesseract"
        / f"tesseract_{radar_index:05d}.mat"
    )


def frame_identity(record: dict[str, Any]) -> str:
    return (
        f"seq{int(record['sequence']):02d}/"
        f"radar{int(record['radar_index']):05d}"
    )


def ordered_cache_digest(
    records: list[dict[str, Any]],
    cache_root: Path,
) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for record in records:
        path = cache_path(cache_root, record)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "identity": frame_identity(record),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256(
        "\n".join(
            f"{row['identity']} {row['sha256']}" for row in rows
        ).encode("utf-8")
    ).hexdigest()
    return digest, rows


def ordered_cube_digest(
    records: list[dict[str, Any]],
    data_root: Path,
) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[int, int]] = set()
    for record in records:
        identity = (int(record["sequence"]), int(record["radar_index"]))
        if identity in seen:
            continue
        seen.add(identity)
        path = cube_path(data_root, record)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "identity": frame_identity(record),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256(
        "\n".join(
            f"{row['identity']} {row['sha256']}" for row in rows
        ).encode("utf-8")
    ).hexdigest()
    return digest, rows


def source_hashes(repo: Path) -> dict[str, str]:
    relative_paths = (
        "code/models/rald_wce_field.py",
        "code/models/rald_wce_quality.py",
        "code/losses/rald_wce_quality.py",
        "code/eval/rald_wce_stage0.py",
        "code/eval/dense_geometry.py",
        "code/scripts/train_rald_wce_stage0.py",
        "code/scripts/train_rald_wce_pilot.py",
        "code/scripts/certify_rald_wce_replay_parent.py",
        "code/scripts/train_rald_wce_quality_tiny.py",
        "docs/rald_wce_quality_tiny_protocol.md",
    )
    return {
        relative: sha256_file(repo / relative)
        for relative in relative_paths
    }


def tensor_mapping_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().to(device="cpu").contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def state_dict_sha256(module: torch.nn.Module) -> str:
    return tensor_mapping_sha256(module.state_dict())


def build_formal_base(
    checkpoint: dict[str, Any],
    *,
    log_center: float,
    log_scale: float,
    device: torch.device,
) -> RaLDWCEField:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Q1 formal checkpoint has no configuration")
    required = {
        "latent_count": 512,
        "model_dim": 512,
        "depth": 6,
        "heads": 8,
        "head_dim": 64,
        "decode_chunk_size": 8_192,
    }
    failed = [
        key for key, expected in required.items()
        if int(config.get(key, -1)) != expected
    ]
    if failed:
        raise ValueError(f"Q1 formal architecture differs: {failed}")
    model = RaLDWCEField(
        log_center=log_center,
        log_scale=log_scale,
        latent_count=int(config["latent_count"]),
        model_dim=int(config["model_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
        decode_chunk_size=int(config["decode_chunk_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Q1 failed to freeze the formal R-A1 field")
    return model


def build_quality_head(
    config: QualityTinyConfig,
    *,
    device: torch.device,
) -> RaLDWCEQualityHead:
    return RaLDWCEQualityHead(
        condition_dim=512,
        model_dim=config.quality_model_dim,
        heads=config.quality_heads,
        head_dim=config.quality_head_dim,
        decode_chunk_size=config.quality_decode_chunk_size,
    ).to(device)


def build_inference_config(config: QualityTinyConfig) -> WideInferenceConfig:
    return WideInferenceConfig(
        q0_range_quotas=config.q0_range_quotas,
        q1_anchor_quotas=config.q1_anchor_quotas,
        q1_samples_per_anchor=config.q1_samples_per_anchor,
        output_range_quotas=config.output_range_quotas,
        minimum_distance_m=config.minimum_export_distance_m,
        seed=config.seed,
        decode_chunk_size=config.quality_decode_chunk_size,
    )


def _axes_tensors(axes: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(axes.range_m, dtype=torch.float32, device=device),
        torch.as_tensor(axes.azimuth_rad, dtype=torch.float32, device=device),
        torch.as_tensor(axes.elevation_rad, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def frozen_candidate_field(
    base_model: RaLDWCEField,
    cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    inference_config: WideInferenceConfig,
) -> dict[str, Any]:
    """Build the unchanged formal Q0/Q1 candidate field without target access."""

    q0 = fixed_wide_q0(
        range_m,
        azimuth_rad,
        elevation_rad,
        quotas=inference_config.q0_range_quotas,
        seed=inference_config.seed,
    ).to(cube_drae.device)
    condition_latents = base_model.encode_condition(cube_drae)[
        "condition_latents"
    ]
    q0_output = base_model.decode_queries(
        q0,
        condition_latents,
        chunk_size=inference_config.decode_chunk_size,
    )
    q1, q1_report = occupancy_dependent_q1(
        q0_output,
        range_m,
        azimuth_rad,
        elevation_rad,
        anchor_quotas=inference_config.q1_anchor_quotas,
        samples_per_anchor=inference_config.q1_samples_per_anchor,
        seed=inference_config.seed,
    )
    q1_output = base_model.decode_queries(
        q1,
        condition_latents,
        chunk_size=inference_config.decode_chunk_size,
    )
    xyz_m, base_confidence = _candidate_tensors(
        (q0_output, q1_output),
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    refined = torch.cat(
        (
            q0_output["refined_normalized_rae"],
            q1_output["refined_normalized_rae"],
        ),
        dim=1,
    )
    if xyz_m.shape != (FORMAL_CANDIDATE_COUNT, 3):
        raise AssertionError("Q1 changed the formal 700k candidate count")
    if refined.shape != (1, FORMAL_CANDIDATE_COUNT, 3):
        raise AssertionError("Q1 refined candidate coordinates changed shape")
    if base_confidence.shape != (FORMAL_CANDIDATE_COUNT,):
        raise AssertionError("Q1 formal confidence changed shape")
    return {
        "refined_normalized_rae": refined,
        "condition_latents": condition_latents,
        "base_confidence": base_confidence.unsqueeze(0),
        "xyz_m": xyz_m,
        "q0": q0,
        "q1": q1,
        "q1_report": q1_report,
        "report": {
            "candidate_count": FORMAL_CANDIDATE_COUNT,
            "q0_count": int(q0.shape[1]),
            "q1_count": int(q1.shape[1]),
            "q0_sha256": tensor_sha256(q0),
            "q1_sha256": tensor_sha256(q1),
            "candidate_xyz_sha256": tensor_sha256(xyz_m),
            "base_confidence_sha256": tensor_sha256(base_confidence),
            "ground_truth_accessed": False,
            "candidate_coordinates_changed": False,
            "formal_residual_changed": False,
        },
    }


@torch.no_grad()
def initial_ranking_control(
    quality_head: RaLDWCEQualityHead,
    field: dict[str, Any],
    inference_config: WideInferenceConfig,
) -> dict[str, Any]:
    """Require update-zero quality export to select the formal ranking rows."""

    base_export = exact_capacity_export(
        field["xyz_m"],
        field["base_confidence"][0],
        output_quotas=inference_config.output_range_quotas,
        minimum_distance_m=inference_config.minimum_distance_m,
    )
    quality = quality_head(
        field["refined_normalized_rae"],
        field["condition_latents"],
        field["base_confidence"],
        chunk_size=inference_config.decode_chunk_size,
    )["quality"][0]
    quality_export = exact_capacity_export(
        field["xyz_m"],
        quality,
        output_quotas=inference_config.output_range_quotas,
        minimum_distance_m=inference_config.minimum_distance_m,
    )
    rows_identical = torch.equal(
        base_export.selected_candidate_rows,
        quality_export.selected_candidate_rows,
    )
    if not rows_identical:
        raise AssertionError("Q1 update-zero ranking differs from formal occupancy")
    return {
        "passed": True,
        "selected_candidate_rows_identical": True,
        "base_selected_rows_sha256": base_export.hashes[
            "selected_candidate_rows_sha256"
        ],
        "quality_selected_rows_sha256": quality_export.hashes[
            "selected_candidate_rows_sha256"
        ],
        "candidate_xyz_sha256": field["report"]["candidate_xyz_sha256"],
    }


@torch.no_grad()
def prepare_training_frame(
    base_model: RaLDWCEField,
    quality_head: RaLDWCEQualityHead,
    item: dict[str, Any],
    axes: Any,
    inference_config: WideInferenceConfig,
    config: QualityTinyConfig,
    device: torch.device,
) -> FrozenCandidateTrainingFrame:
    """Freeze one target-free candidate field, then attach training targets."""

    range_m, azimuth_rad, elevation_rad = _axes_tensors(axes, device)
    cube = item["cube_drae"].unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        field = frozen_candidate_field(
            base_model,
            cube,
            range_m,
            azimuth_rad,
            elevation_rad,
            inference_config,
        )
        ranking_control = initial_ranking_control(
            quality_head,
            field,
            inference_config,
        )
    target = item["target_xyz_confidence"].to(device).float()
    quality = continuous_geometry_quality_target(
        field["xyz_m"].float(),
        target,
        distance_temperature_m=config.distance_temperature_m,
        candidate_chunk_size=config.nearest_target_chunk_size,
    )
    range_class = range_stratum_codes(field["xyz_m"].float())
    if bool((range_class < 0).any()):
        raise ValueError("Q1 formal candidates escaped frozen range strata")
    report = {
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
        "partition": str(item["partition"]),
        "candidate_field": field["report"],
        "q1": field["q1_report"],
        "initial_ranking_control": ranking_control,
        "quality_target": {
            "formula": (
                "normalized_nearest_target_confidence"
                "*exp(-nearest_distance_m/1p0m)"
            ),
            "target_stop_gradient": True,
            "minimum": float(quality["quality_target"].min().item()),
            "maximum": float(quality["quality_target"].max().item()),
            "mean": float(quality["quality_target"].mean().item()),
            "nearest_distance_mean_m": float(
                quality["nearest_distance_m"].mean().item()
            ),
        },
        "target_access_timing": "after_frozen_candidate_construction",
        "target_used_for_inference": False,
    }
    return FrozenCandidateTrainingFrame(
        refined_normalized_rae=field["refined_normalized_rae"][0]
        .detach()
        .float()
        .cpu(),
        condition_latents=field["condition_latents"][0]
        .detach()
        .float()
        .cpu(),
        base_confidence=field["base_confidence"][0].detach().float().cpu(),
        quality_target=quality["quality_target"].detach().float().cpu(),
        nearest_distance_m=quality["nearest_distance_m"].detach().float().cpu(),
        range_class=range_class.detach().long().cpu(),
        report=report,
    )


def sample_quality_candidate_rows(
    quality_target: torch.Tensor,
    range_class: torch.Tensor,
    *,
    range_quotas: tuple[int, int, int],
    high_quality_fraction: float,
    high_quality_pool_fraction: float,
    seed: int,
) -> torch.Tensor:
    """Sample unique high-quality and uniform rows within each output range."""

    if quality_target.ndim != 1 or range_class.shape != quality_target.shape:
        raise ValueError("Q1 sampling inputs must be aligned vectors")
    if range_class.dtype != torch.long:
        raise TypeError("Q1 sampling range class must be torch.long")
    if len(range_quotas) != 3 or any(int(value) <= 1 for value in range_quotas):
        raise ValueError("Q1 sample range quotas must be three integers > 1")
    if not 0.0 < high_quality_fraction < 1.0:
        raise ValueError("Q1 high-quality fraction must lie in (0,1)")
    if not 0.0 < high_quality_pool_fraction <= 1.0:
        raise ValueError("Q1 high-quality pool fraction must lie in (0,1]")
    if not bool(torch.isfinite(quality_target).all()):
        raise ValueError("Q1 quality target must be finite")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected_parts: list[torch.Tensor] = []
    for code, quota in enumerate(range_quotas):
        eligible = torch.nonzero(range_class == code, as_tuple=False).flatten()
        if eligible.numel() < quota:
            raise ValueError(
                f"Q1 range {code} has {eligible.numel()} candidates below {quota}"
            )
        high_count = int(round(quota * high_quality_fraction))
        uniform_count = quota - high_count
        order = torch.argsort(
            quality_target[eligible],
            descending=True,
            stable=True,
        )
        pool_count = max(
            high_count,
            int(math.ceil(eligible.numel() * high_quality_pool_fraction)),
        )
        high_pool = eligible[order[:pool_count]]
        high_rows = high_pool[
            torch.randperm(high_pool.numel(), generator=generator)[:high_count]
        ]
        uniform_pool = eligible[order[pool_count:]]
        if uniform_pool.numel() < uniform_count:
            raise ValueError(
                f"Q1 range {code} has too few rows outside its high-quality pool"
            )
        uniform_rows = uniform_pool[
            torch.randperm(uniform_pool.numel(), generator=generator)[
                :uniform_count
            ]
        ]
        selected_parts.append(torch.cat((high_rows, uniform_rows)))
    selected = torch.cat(selected_parts)
    selected = selected[
        torch.randperm(selected.numel(), generator=generator)
    ]
    if selected.numel() != sum(range_quotas):
        raise AssertionError("Q1 sampler changed the frozen sample count")
    if torch.unique(selected).numel() != selected.numel():
        raise AssertionError("Q1 sampler introduced duplicate candidate rows")
    sampled_codes = range_class[selected]
    actual = tuple(
        int((sampled_codes == code).sum().item()) for code in range(3)
    )
    if actual != tuple(range_quotas):
        raise AssertionError("Q1 sampler changed range quotas")
    return selected


@torch.no_grad()
def infer_exact_quality(
    base_model: RaLDWCEField,
    quality_head: RaLDWCEQualityHead,
    matched_cube_drae: torch.Tensor,
    wrong_cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    inference_config: WideInferenceConfig,
) -> dict[str, Any]:
    """Run one deterministic score per unchanged candidate; no target argument."""

    matched = frozen_candidate_field(
        base_model,
        matched_cube_drae,
        range_m,
        azimuth_rad,
        elevation_rad,
        inference_config,
    )
    q0 = matched["q0"]
    q1 = matched["q1"]
    wrong_condition = base_model.encode_condition(wrong_cube_drae)[
        "condition_latents"
    ]
    wrong_q0 = base_model.decode_queries(
        q0,
        wrong_condition,
        chunk_size=inference_config.decode_chunk_size,
    )
    wrong_q1 = base_model.decode_queries(
        q1,
        wrong_condition,
        chunk_size=inference_config.decode_chunk_size,
    )
    wrong_xyz, wrong_base_confidence = _candidate_tensors(
        (wrong_q0, wrong_q1),
        range_m,
        azimuth_rad,
        elevation_rad,
    )
    wrong_refined = torch.cat(
        (
            wrong_q0["refined_normalized_rae"],
            wrong_q1["refined_normalized_rae"],
        ),
        dim=1,
    )
    matched_quality = quality_head(
        matched["refined_normalized_rae"],
        matched["condition_latents"],
        matched["base_confidence"],
        chunk_size=inference_config.decode_chunk_size,
    )["quality"][0]
    wrong_quality = quality_head(
        wrong_refined,
        wrong_condition,
        wrong_base_confidence.unsqueeze(0),
        chunk_size=inference_config.decode_chunk_size,
    )["quality"][0]
    matched_export = exact_capacity_export(
        matched["xyz_m"],
        matched_quality,
        output_quotas=inference_config.output_range_quotas,
        minimum_distance_m=inference_config.minimum_distance_m,
    )
    wrong_export = exact_capacity_export(
        wrong_xyz,
        wrong_quality,
        output_quotas=inference_config.output_range_quotas,
        minimum_distance_m=inference_config.minimum_distance_m,
    )
    return {
        "matched": matched_export,
        "wrong_condition": wrong_export,
        "report": {
            "candidate_count": FORMAL_CANDIDATE_COUNT,
            "q0_sha256": tensor_sha256(q0),
            "q1_sha256": tensor_sha256(q1),
            "matched_candidate_xyz_sha256": tensor_sha256(matched["xyz_m"]),
            "wrong_candidate_xyz_sha256": tensor_sha256(wrong_xyz),
            "ranking_score": "q1_geometry_quality",
            "candidate_coordinates_changed": False,
            "formal_residual_changed": False,
            "range_quotas_changed": False,
            "minimum_distance_changed": False,
            "ground_truth_accessed": False,
            "test_accessed": False,
            "doppler_head": False,
            "best_of_k": False,
        },
    }


def _aggregate_scalars(
    reports: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not reports:
        raise ValueError("Q1 scalar reports cannot be empty")
    keys = sorted(reports[0])
    if any(sorted(report) != keys for report in reports):
        raise ValueError("Q1 scalar reports have inconsistent keys")
    return {
        key: {
            "mean": float(np.mean([row[key] for row in reports])),
            "median": float(np.median([row[key] for row in reports])),
            "std": float(np.std([row[key] for row in reports])),
            "sample_count": len(reports),
        }
        for key in keys
    }


@torch.no_grad()
def evaluate_quality(
    base_model: RaLDWCEField,
    quality_head: RaLDWCEQualityHead,
    dataset: PilotCubeDataset,
    indices: list[int],
    wrong_indices: dict[int, int],
    inference_config: WideInferenceConfig,
    axes: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate on the frozen eight train frames with the unchanged evaluator."""

    base_model.eval()
    quality_head.eval()
    range_m, azimuth_rad, elevation_rad = _axes_tensors(axes, device)
    frames: list[dict[str, Any]] = []
    for index in indices:
        item = dataset[index]
        wrong_item = dataset[wrong_indices[index]]
        if item["partition"] != "train" or wrong_item["partition"] != "train":
            raise ValueError("Q1 tiny evaluator permits train frames only")
        cube = item["cube_drae"].unsqueeze(0).to(device)
        wrong_cube = wrong_item["cube_drae"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference = infer_exact_quality(
                base_model,
                quality_head,
                cube,
                wrong_cube,
                range_m,
                azimuth_rad,
                elevation_rad,
                inference_config,
            )
        target = item["target_xyz_confidence"].to(device).float()
        matched_geometry = geometry_report(
            inference["matched"].xyz_m.to(device),
            target[:, :3],
            target_weight=target[:, 3],
        )
        wrong_geometry = geometry_report(
            inference["wrong_condition"].xyz_m.to(device),
            target[:, :3],
            target_weight=target[:, 3],
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
                "matched": matched_geometry,
                "wrong_condition": wrong_geometry,
                "condition_intervention": {
                    "wrong_minus_matched_chamfer_fraction": (
                        wrong_chamfer / max(matched_chamfer, 1e-12) - 1.0
                    ),
                    "matched_chamfer_better": float(
                        matched_chamfer < wrong_chamfer
                    ),
                },
                "matched_export": inference["matched"].report,
                "wrong_export": inference["wrong_condition"].report,
                "inference": inference["report"],
            }
        )
        del item, wrong_item, cube, wrong_cube, target, inference
        torch.cuda.empty_cache()
    return {
        "frame_count": len(frames),
        "scene_count": len({frame["sequence"] for frame in frames}),
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
        "exact_export": {
            "point_count_per_frame": EXPORT_COUNT,
            "all_matched_exports_exact_10000": all(
                frame["matched_export"]["exact_point_count"] == EXPORT_COUNT
                for frame in frames
            ),
            "all_wrong_exports_exact_10000": all(
                frame["wrong_export"]["exact_point_count"] == EXPORT_COUNT
                for frame in frames
            ),
            "all_minimum_distance_5cm": all(
                frame["matched_export"]["observed_minimum_pair_distance_m"]
                >= CAPACITY_DISTANCE_M - 1e-6
                for frame in frames
            ),
            "all_wrong_minimum_distance_5cm": all(
                frame["wrong_export"]["observed_minimum_pair_distance_m"]
                >= CAPACITY_DISTANCE_M - 1e-6
                for frame in frames
            ),
            "copy_padding_jitter_duplicate": False,
        },
        "evidence_boundary": {
            "test_partition_accessed": False,
            "future_cube_accessed": False,
            "cache_arrays_read": list(ALLOWED_CACHE_ARRAYS),
            "ground_truth_accessed_for_inference_or_selection": False,
            "doppler_head_evaluated": False,
            "best_of_k": False,
        },
    }


def quality_metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    values = metric_values(
        {
            **metrics,
            "far_recall_1m": {
                "matched": {"mean": 0.0},
            },
        }
    )
    return {
        "chamfer_mean_m": values["chamfer_mean_m"],
        "outlier_fraction_mean": values["outlier_fraction_mean"],
        "completeness_median_m": values["completeness_median_m"],
        "completeness_mean_m": values["completeness_mean_m"],
    }


def quality_tiny_gate(
    values: dict[str, float],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    numeric = tiny_memorization_gate(
        {
            **values,
            "far_fscore_1m_mean": 0.0,
            "far_recall_1m_mean": 0.0,
            "wrong_condition_chamfer_degradation_mean": 0.0,
            "matched_condition_win_fraction": 0.0,
        }
    )
    exact = metrics["exact_export"]
    structural_checks = {
        "all_matched_exports_exact_10000": exact[
            "all_matched_exports_exact_10000"
        ]
        is True,
        "all_wrong_exports_exact_10000": exact[
            "all_wrong_exports_exact_10000"
        ]
        is True,
        "all_matched_minimum_distance_5cm": exact[
            "all_minimum_distance_5cm"
        ]
        is True,
        "all_wrong_minimum_distance_5cm": exact[
            "all_wrong_minimum_distance_5cm"
        ]
        is True,
        "no_copy_padding_jitter_duplicate": exact[
            "copy_padding_jitter_duplicate"
        ]
        is False,
    }
    checks = {**numeric["checks"], **structural_checks}
    return {
        "numeric_checks": numeric["checks"],
        "structural_checks": structural_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def initial_state() -> dict[str, Any]:
    return {
        "updates_completed": 0,
        "evaluation_updates_completed": [],
        "consecutive_gate_passes": 0,
        "loss_history": [],
        "status": "running",
    }


def save_checkpoint(
    path: Path,
    *,
    quality_head: RaLDWCEQualityHead,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    contract_sha256: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "protocol": PROTOCOL,
            "resume_contract_sha256": contract_sha256,
            "quality_head": quality_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
            "formal_base_state_stored": False,
        },
    )


def load_checkpoint(
    path: Path,
    *,
    quality_head: RaLDWCEQualityHead,
    optimizer: torch.optim.Optimizer,
    contract_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("Q1 checkpoint protocol differs")
    if checkpoint.get("resume_contract_sha256") != contract_sha256:
        raise ValueError("Q1 checkpoint resume contract differs")
    if checkpoint.get("formal_base_state_stored") is not False:
        raise ValueError("Q1 checkpoint must not store or modify the formal base")
    quality_head.load_state_dict(checkpoint["quality_head"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch_cpu"])
    if torch.cuda.is_available() and checkpoint["rng"]["torch_cuda"]:
        torch.cuda.set_rng_state_all(checkpoint["rng"]["torch_cuda"])
    return checkpoint["state"]


def ensure_evaluation_checkpoint(
    path: Path,
    *,
    resume: bool,
    quality_head: RaLDWCEQualityHead,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    contract_sha256: str,
) -> str:
    """Create once, or verify and reuse after an interrupted evaluation."""

    if not path.exists():
        save_checkpoint(
            path,
            quality_head=quality_head,
            optimizer=optimizer,
            state=state,
            contract_sha256=contract_sha256,
        )
        return sha256_file(path)
    if not resume:
        raise FileExistsError(
            f"Q1-R immutable evaluation checkpoint already exists: {path}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "protocol": checkpoint.get("protocol") == PROTOCOL,
        "resume_contract": checkpoint.get("resume_contract_sha256")
        == contract_sha256,
        "formal_base_absent": checkpoint.get("formal_base_state_stored")
        is False,
        "state": checkpoint.get("state") == state,
        "quality_head": isinstance(checkpoint.get("quality_head"), dict)
        and tensor_mapping_sha256(checkpoint["quality_head"])
        == state_dict_sha256(quality_head),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"Q1-R immutable evaluation checkpoint failed reuse: {failed}"
        )
    return sha256_file(path)


def build_evaluation_document(
    *,
    metrics: dict[str, Any],
    updates: int,
    source_commit: str,
    evaluation_checkpoint: Path,
    evaluation_checkpoint_sha256: str,
    formal_base_state_sha256: str,
    prior_consecutive_passes: int,
) -> dict[str, Any]:
    values = quality_metric_values(metrics)
    decision = quality_tiny_gate(values, metrics)
    consecutive_passes = (
        prior_consecutive_passes + 1 if decision["passed"] else 0
    )
    decision["consecutive_passes"] = consecutive_passes
    decision["early_stop_passed"] = consecutive_passes >= 2
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "updates_completed": updates,
        "quality_checkpoint": str(evaluation_checkpoint.resolve()),
        "quality_checkpoint_sha256": evaluation_checkpoint_sha256,
        "formal_base_state_sha256": formal_base_state_sha256,
        "values": values,
        "metrics": metrics,
        "decision": decision,
    }


def load_completed_evaluation(
    path: Path,
    *,
    updates: int,
    source_commit: str,
    evaluation_checkpoint: Path,
    evaluation_checkpoint_sha256: str,
    formal_base_state_sha256: str,
    prior_consecutive_passes: int,
) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Q1-R completed evaluation lacks metrics")
    recomputed = build_evaluation_document(
        metrics=metrics,
        updates=updates,
        source_commit=source_commit,
        evaluation_checkpoint=evaluation_checkpoint,
        evaluation_checkpoint_sha256=evaluation_checkpoint_sha256,
        formal_base_state_sha256=formal_base_state_sha256,
        prior_consecutive_passes=prior_consecutive_passes,
    )
    if document != recomputed:
        raise ValueError("Q1-R completed evaluation differs from recomputation")
    return recomputed


def bind_candidate_preparation(
    path: Path,
    document: dict[str, Any],
    *,
    resume: bool,
) -> str:
    if resume:
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != document:
                raise ValueError("Q1-R candidate preparation changed on resume")
        else:
            atomic_json(path, document)
    else:
        atomic_json(path, document)
    return sha256_file(path)


def write_capacity_failure(
    output_dir: Path,
    error: ExactExportCapacityError,
) -> None:
    path = output_dir / "terminal_capacity_failure.json"
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "quality_ranking_no_go_exact_10000_capacity",
        "message": str(error),
        "capacity_report": error.report,
        "scientific_decision_eligible": True,
        "copy_padding_jitter_duplicate": False,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise ValueError("Q1-R capacity failure artifact changed")
        return
    atomic_json(path, document)


def prepare_output(path: Path, *, resume: bool) -> None:
    """Validate the output target without creating scientific artifacts."""

    if resume:
        if not (path / "run_manifest.json").is_file():
            raise FileNotFoundError("Q1 resume requires run_manifest.json")
        if (path / "summary.json").exists():
            raise FileExistsError("Q1 completed run cannot be resumed")
        if (path / "terminal_capacity_failure.json").exists():
            raise FileExistsError("Q1 terminal capacity failure cannot be resumed")
        return
    if path.exists():
        raise FileExistsError(f"Q1 output already exists: {path}")


def initialize_output(path: Path, *, resume: bool) -> None:
    if resume:
        if not path.is_dir():
            raise FileNotFoundError("Q1 resume output directory disappeared")
        return
    path.mkdir(parents=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--formal-best-checkpoint", type=Path, required=True)
    parser.add_argument("--formal-best-metrics", type=Path, required=True)
    parser.add_argument("--formal-run-manifest", type=Path, required=True)
    parser.add_argument("--formal-parent-diagnosis", type=Path, required=True)
    parser.add_argument("--formal-parent-certificate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_argument_boundary(args: argparse.Namespace) -> None:
    lowered = {key.lower() for key in vars(args)}
    forbidden = [
        key
        for key in lowered
        if any(fragment in key for fragment in FORBIDDEN_ARGUMENT_FRAGMENTS)
    ]
    if forbidden:
        raise ValueError(f"Q1 parser exposes forbidden arguments: {forbidden}")


def _tiny_frame_ids(
    dataset: PilotCubeDataset,
    indices: list[int],
) -> list[dict[str, int]]:
    result = []
    for index in indices:
        record = dataset.records[index]
        result.append(
            {
                "sequence": int(record["sequence"]),
                "radar_index": int(record["radar_index"]),
            }
        )
    return result


def _write_progress(output: Path, state: dict[str, Any]) -> None:
    atomic_json(
        output / "progress.json",
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "state": state,
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_argument_boundary(args)
    config = frozen_quality_config()
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)

    input_hashes, manifest_counts = validate_frozen_inputs(
        args.manifest,
        args.scene_split,
        args.normalization,
        repo,
    )
    manifest_document = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_records = manifest_document["frames"]
    cache_digest, cache_rows = ordered_cache_digest(
        manifest_records,
        args.cache_root,
    )
    if cache_digest != FROZEN_ORDERED_CACHE_DIGEST_SHA256:
        raise ValueError(
            "Q1 ordered cache digest differs from the frozen byte set: "
            f"{cache_digest}"
        )

    train_dataset = PilotCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        "train",
    )
    statistics = [
        train_dataset.target_statistics(index)
        for index in range(len(train_dataset))
    ]
    train_indices = select_tiny_subset(statistics)
    if len(train_indices) != TINY_FRAME_COUNT:
        raise AssertionError("Q1 tiny subset changed frame count")
    tiny_identities = tuple(
        (
            int(train_dataset.records[index]["sequence"]),
            int(train_dataset.records[index]["radar_index"]),
        )
        for index in train_indices
    )
    if tiny_identities != FROZEN_TINY_FRAME_IDENTITIES:
        raise ValueError(
            f"Q1-R tiny frame identities changed: {tiny_identities}"
        )
    wrong_indices = cross_scene_wrong_indices(
        train_dataset.records,
        train_indices,
    )
    tiny_records = [train_dataset.records[index] for index in train_indices]
    cube_digest, cube_rows = ordered_cube_digest(
        tiny_records,
        args.data_root,
    )
    if cube_digest != FROZEN_ORDERED_TINY_CUBE_DIGEST_SHA256:
        raise ValueError(
            "Q1-R ordered tiny Cube digest differs from the frozen byte set: "
            f"{cube_digest}"
        )
    resources = {
        name: sha256_file(args.data_root / "resources" / name)
        for name in ("info_arr.mat", "arr_doppler.mat")
    }
    if resources != FROZEN_RESOURCE_SHA256:
        raise ValueError(f"Q1-R frozen resource hashes changed: {resources}")
    all_input_hashes = {
        **input_hashes,
        "ordered_cache_digest_sha256": cache_digest,
        "ordered_tiny_cube_digest_sha256": cube_digest,
        "resource_sha256": resources,
        "formal_best_checkpoint_sha256": sha256_file(
            args.formal_best_checkpoint
        ),
        "formal_best_metrics_sha256": sha256_file(args.formal_best_metrics),
        "formal_run_manifest_sha256": sha256_file(args.formal_run_manifest),
        "formal_parent_diagnosis_sha256": sha256_file(
            args.formal_parent_diagnosis
        ),
        "formal_parent_certificate_sha256": sha256_file(
            args.formal_parent_certificate
        ),
    }

    formal_checkpoint = torch.load(
        args.formal_best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    formal_metrics = json.loads(
        args.formal_best_metrics.read_text(encoding="utf-8")
    )
    formal_checkpoint_evidence = validate_formal_checkpoint(
        formal_checkpoint,
        checkpoint_path=args.formal_best_checkpoint,
        formal_metrics=formal_metrics,
    )
    if formal_checkpoint.get("source_commit") != FORMAL_PARENT_SOURCE_COMMIT:
        raise ValueError("Q1-R input is not from the frozen R-A1 source")
    if int(formal_checkpoint.get("epoch", -1)) != 20:
        raise ValueError("Q1-R requires the formal R-A1 epoch-20 endpoint")
    formal_metric_frames = validate_formal_metrics(formal_metrics)
    formal_metrics_evidence = {
        "frame_count": len(formal_metric_frames),
        "validation_only": all(
            frame.get("partition") == "validation"
            for frame in formal_metric_frames
        ),
        "formal_contract_validated": True,
    }
    formal_input_hashes = {
        "manifest": input_hashes["manifest_sha256"],
        "scene_split": input_hashes["scene_split_sha256"],
        "normalization": input_hashes["normalization_sha256"],
        "corrected_dense_geometry": input_hashes[
            "dense_geometry_evaluator_sha256"
        ],
    }
    formal_manifest_evidence = validate_formal_run_manifest(
        repo,
        args.formal_run_manifest,
        formal_checkpoint,
        formal_input_hashes,
    )
    if formal_checkpoint.get("protocol") != FORMAL_R_A1_PROTOCOL:
        raise ValueError("Q1 input is not a formal R-A1 checkpoint")
    replay_parent_evidence = validate_replay_parent_certificate(
        args.formal_parent_certificate,
        args.formal_parent_diagnosis,
        args.formal_best_checkpoint,
        args.formal_best_metrics,
        args.formal_run_manifest,
        expected_certifier_source_commit=args.source_commit,
        repo=repo,
    )

    log_center, log_scale = load_normalization(args.normalization)
    axes = load_axes(args.data_root / "resources")
    base_model = build_formal_base(
        formal_checkpoint,
        log_center=log_center,
        log_scale=log_scale,
        device=device,
    )
    base_state_digest = state_dict_sha256(base_model)
    inference_config = build_inference_config(config)
    prepare_output(args.output_dir, resume=args.resume)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    quality_head = build_quality_head(config, device=device)
    base_model.eval()
    quality_head.eval()
    training_frames: dict[int, FrozenCandidateTrainingFrame] = {}
    preparation_reports: list[dict[str, Any]] = []
    try:
        for index in train_indices:
            frame = prepare_training_frame(
                base_model,
                quality_head,
                train_dataset[index],
                axes,
                inference_config,
                config,
                device,
            )
            training_frames[index] = frame
            preparation_reports.append(frame.report)
            torch.cuda.empty_cache()
    except ExactExportCapacityError as error:
        initialize_output(args.output_dir, resume=args.resume)
        write_capacity_failure(args.output_dir, error)
        raise SystemExit(2) from error
    preparation_document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "frame_count": len(preparation_reports),
        "frames": preparation_reports,
        "all_initial_ranking_controls_passed": all(
            report["initial_ranking_control"]["passed"]
            for report in preparation_reports
        ),
        "target_accessed_after_candidate_construction": True,
        "target_used_for_inference": False,
    }
    preparation_sha = json_artifact_sha256(preparation_document)
    initialize_output(args.output_dir, resume=args.resume)

    current_source_hashes = source_hashes(repo)
    contract = {
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "config": asdict(config),
        "input_hashes": all_input_hashes,
        "source_hashes": current_source_hashes,
        "tiny_frame_ids": _tiny_frame_ids(train_dataset, train_indices),
        "formal_base_state_sha256": base_state_digest,
        "candidate_preparation_sha256": preparation_sha,
    }
    contract_sha = canonical_digest(contract)
    if args.resume:
        manifest = json.loads(
            (args.output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("resume_contract_sha256") != contract_sha:
            raise ValueError("Q1 resume contract digest differs")
        if manifest.get("resume_contract") != contract:
            raise ValueError("Q1 resume contract metadata differs")
    else:
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "source_commit": args.source_commit,
            "resume_contract_sha256": contract_sha,
            "resume_contract": contract,
            "manifest_counts": manifest_counts,
            "architecture": quality_head.architecture_metadata(),
            "runtime": {
                "device_argument": args.device,
                "device_name": device_name,
                "torch_version": torch.__version__,
                "quality_parameter_count": quality_parameter_count(quality_head),
                "formal_base_trainable_parameter_count": 0,
            },
            "formal_checkpoint_evidence": formal_checkpoint_evidence,
            "formal_metrics_evidence": formal_metrics_evidence,
            "formal_run_manifest_evidence": formal_manifest_evidence,
            "replay_parent_certificate_evidence": replay_parent_evidence,
            "candidate_preparation_binding": {
                "path": str(
                    (args.output_dir / "candidate_preparation.json").resolve()
                ),
                "sha256": preparation_sha,
                "resume_requires_exact_match": True,
            },
            "cache_binding": {
                "ordered_digest_sha256": cache_digest,
                "expected_ordered_digest_sha256": (
                    FROZEN_ORDERED_CACHE_DIGEST_SHA256
                ),
                "frame_count": len(cache_rows),
                "frames": cache_rows,
            },
            "cube_binding": {
                "ordered_tiny_cube_digest_sha256": cube_digest,
                "expected_ordered_tiny_cube_digest_sha256": (
                    FROZEN_ORDERED_TINY_CUBE_DIGEST_SHA256
                ),
                "frame_count": len(cube_rows),
                "frames": cube_rows,
            },
            "resource_binding": {
                "actual_sha256": resources,
                "expected_sha256": FROZEN_RESOURCE_SHA256,
            },
            "data_access_contract": {
                "training_partition": "train",
                "evaluation_partition": "same_frozen_eight_train_frames",
                "test_partition_accessed": False,
                "future_cube_accessed": False,
                "cache_arrays_read": list(ALLOWED_CACHE_ARRAYS),
                "target_used_only_for_training_supervision_and_metrics": True,
                "target_used_for_inference_or_selection": False,
                "doppler_target_or_head": False,
                "best_of_k": False,
            },
        }
        atomic_json(args.output_dir / "run_manifest.json", manifest)
    bound_preparation_sha = bind_candidate_preparation(
        args.output_dir / "candidate_preparation.json",
        preparation_document,
        resume=args.resume,
    )
    if bound_preparation_sha != preparation_sha:
        raise AssertionError("Q1-R candidate preparation SHA changed on write")

    optimizer = torch.optim.AdamW(
        quality_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = args.output_dir / "last.pt"
    if args.resume:
        state = load_checkpoint(
            checkpoint_path,
            quality_head=quality_head,
            optimizer=optimizer,
            contract_sha256=contract_sha,
        )
    else:
        state = initial_state()
        save_checkpoint(
            checkpoint_path,
            quality_head=quality_head,
            optimizer=optimizer,
            state=state,
            contract_sha256=contract_sha,
        )
        _write_progress(args.output_dir, state)

    started = time.monotonic()
    stop_reason: str | None = None
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
                save_checkpoint(
                    checkpoint_path,
                    quality_head=quality_head,
                    optimizer=optimizer,
                    state=state,
                    contract_sha256=contract_sha,
                )
                evaluation_checkpoint = (
                    args.output_dir / f"checkpoint_update{updates:04d}.pt"
                )
                evaluation_checkpoint_sha = ensure_evaluation_checkpoint(
                    evaluation_checkpoint,
                    resume=args.resume,
                    quality_head=quality_head,
                    optimizer=optimizer,
                    state=state,
                    contract_sha256=contract_sha,
                )
                current_base_digest = state_dict_sha256(base_model)
                if current_base_digest != base_state_digest:
                    raise AssertionError("Q1 modified the formal R-A1 base")
                prior_consecutive = int(state["consecutive_gate_passes"])
                metrics_path = (
                    args.output_dir / f"metrics_update{updates:04d}.json"
                )
                if metrics_path.exists():
                    if not args.resume:
                        raise FileExistsError(
                            f"Q1-R immutable evaluation metrics exist: {metrics_path}"
                        )
                    evaluation = load_completed_evaluation(
                        metrics_path,
                        updates=updates,
                        source_commit=args.source_commit,
                        evaluation_checkpoint=evaluation_checkpoint,
                        evaluation_checkpoint_sha256=evaluation_checkpoint_sha,
                        formal_base_state_sha256=current_base_digest,
                        prior_consecutive_passes=prior_consecutive,
                    )
                else:
                    metrics = evaluate_quality(
                        base_model,
                        quality_head,
                        train_dataset,
                        train_indices,
                        wrong_indices,
                        inference_config,
                        axes,
                        device,
                    )
                    evaluation = build_evaluation_document(
                        metrics=metrics,
                        updates=updates,
                        source_commit=args.source_commit,
                        evaluation_checkpoint=evaluation_checkpoint,
                        evaluation_checkpoint_sha256=evaluation_checkpoint_sha,
                        formal_base_state_sha256=current_base_digest,
                        prior_consecutive_passes=prior_consecutive,
                    )
                    atomic_json(metrics_path, evaluation)
                decision = evaluation["decision"]
                state["consecutive_gate_passes"] = int(
                    decision["consecutive_passes"]
                )
                state["evaluation_updates_completed"].append(updates)
                if decision["early_stop_passed"]:
                    stop_reason = "quality_ranking_tiny_passed_early"
                elif updates == config.maximum_updates:
                    stop_reason = "quality_ranking_no_go_at_500_updates"
                save_checkpoint(
                    checkpoint_path,
                    quality_head=quality_head,
                    optimizer=optimizer,
                    state=state,
                    contract_sha256=contract_sha,
                )
                _write_progress(args.output_dir, state)
                if stop_reason is not None:
                    break
            if updates >= config.maximum_updates:
                stop_reason = "quality_ranking_no_go_at_500_updates"
                break

            cycle = updates // len(train_indices) + 1
            order = deterministic_frame_order(
                train_indices,
                seed=config.seed,
                cycle=cycle,
            )
            index = order[updates % len(train_indices)]
            frame = training_frames[index]
            rows = sample_quality_candidate_rows(
                frame.quality_target,
                frame.range_class,
                range_quotas=config.quality_sample_range_quotas,
                high_quality_fraction=config.high_quality_sample_fraction,
                high_quality_pool_fraction=config.high_quality_pool_fraction,
                seed=(
                    config.seed
                    + 1_000_003 * (updates + 1)
                    + 10_007 * int(frame.report["sequence"])
                    + 101 * int(frame.report["radar_index"])
                )
                % (2**31 - 1),
            )
            refined = frame.refined_normalized_rae[rows].unsqueeze(0).to(device)
            base_confidence = frame.base_confidence[rows].unsqueeze(0).to(device)
            quality_target = frame.quality_target[rows].unsqueeze(0).to(device)
            range_class = frame.range_class[rows].unsqueeze(0).to(device)
            condition_latents = frame.condition_latents.unsqueeze(0).to(device)

            optimizer.zero_grad(set_to_none=True)
            quality_head.train()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = quality_head(
                    refined,
                    condition_latents,
                    base_confidence,
                    chunk_size=config.quality_decode_chunk_size,
                )
                loss = rald_wce_quality_loss(
                    output["quality_logit"],
                    quality_target,
                    range_class,
                    ranking_weight=config.ranking_weight,
                    minimum_target_gap=config.minimum_target_gap,
                )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(
                quality_head.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            if any(parameter.grad is not None for parameter in base_model.parameters()):
                raise AssertionError("Q1 propagated gradients into formal R-A1")
            state["updates_completed"] = updates + 1
            state["loss_history"].append(
                {
                    "update": updates + 1,
                    "total": float(loss.total.detach().item()),
                    "quality_calibration": float(
                        loss.components["quality_calibration"].detach().item()
                    ),
                    "quality_pairwise_ranking": float(
                        loss.components[
                            "quality_pairwise_ranking"
                        ].detach().item()
                    ),
                    "quality_pair_count": int(
                        loss.components["quality_pair_count"].detach().item()
                    ),
                }
            )
            if (
                state["updates_completed"]
                % config.checkpoint_interval_updates
                == 0
            ):
                save_checkpoint(
                    checkpoint_path,
                    quality_head=quality_head,
                    optimizer=optimizer,
                    state=state,
                    contract_sha256=contract_sha,
                )
                _write_progress(args.output_dir, state)
            print(
                json.dumps(
                    {
                        "protocol": PROTOCOL,
                        "update": state["updates_completed"],
                        "loss": state["loss_history"][-1]["total"],
                    }
                ),
                flush=True,
            )
            del (
                refined,
                base_confidence,
                quality_target,
                range_class,
                condition_latents,
                output,
                loss,
            )
    except ExactExportCapacityError as error:
        write_capacity_failure(args.output_dir, error)
        raise SystemExit(2) from error

    state["status"] = stop_reason
    save_checkpoint(
        checkpoint_path,
        quality_head=quality_head,
        optimizer=optimizer,
        state=state,
        contract_sha256=contract_sha,
    )
    _write_progress(args.output_dir, state)
    final_base_digest = state_dict_sha256(base_model)
    if final_base_digest != base_state_digest:
        raise AssertionError("Q1 terminal state changed the formal R-A1 base")
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "status": stop_reason,
        "updates_completed": int(state["updates_completed"]),
        "evaluation_updates_completed": state["evaluation_updates_completed"],
        "consecutive_gate_passes": int(state["consecutive_gate_passes"]),
        "formal_base_state_sha256": final_base_digest,
        "formal_base_unchanged": True,
        "ordered_cache_digest_sha256": cache_digest,
        "ordered_tiny_cube_digest_sha256": cube_digest,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "test_partition_accessed": False,
        "future_cube_accessed": False,
        "target_used_for_inference_or_selection": False,
        "doppler_head_evaluated": False,
        "best_of_k": False,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
