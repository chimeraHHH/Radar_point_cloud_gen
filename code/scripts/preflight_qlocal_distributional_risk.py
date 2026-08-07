#!/usr/bin/env python3
"""Run the source-bound train-only capacity and gradient gate for Q-Local-F0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import geometry_report  # noqa: E402
from eval.rald_wce_stage0 import (  # noqa: E402
    CAPACITY_DISTANCE_M,
    FORMAL_OUTPUT_RANGE_QUOTAS,
    FORMAL_Q0_RANGE_QUOTAS,
    FORMAL_Q1_ANCHOR_QUOTAS,
    FORMAL_Q1_SAMPLES_PER_ANCHOR,
    ExactExportCapacityError,
    WideInferenceConfig,
    _denormalize_bins,
    exact_capacity_export,
    range_stratum_codes,
    tensor_sha256,
)
from losses.qlocal_distributional_risk import (  # noqa: E402
    distance_distribution_target,
    qlocal_distributional_risk_loss,
)
from losses.rald_wce_quality import continuous_geometry_quality_target  # noqa: E402
from models.qlocal_distributional_risk import (  # noqa: E402
    QLocalDistributionalRiskScorer,
    extract_local_full_raed_features,
    qlocal_parameter_count,
)
from scripts.train_rald_wce_pilot import (  # noqa: E402
    PilotCubeDataset,
    load_normalization,
    validate_frozen_inputs,
)
from scripts.train_rald_wce_quality_tiny import (  # noqa: E402
    build_formal_base,
    frozen_candidate_field,
    sample_quality_candidate_rows,
    state_dict_sha256,
)


PROTOCOL = "g1_qlocal_distributional_risk_f0_preflight_v1"
FRESH_PARENT_PROTOCOL = "g1_ra1_rald_wce_stage0_v1"
FRESH_PARENT_SOURCE_COMMIT = "f2a9489d40323d1ef45d85de958f4aea8126e1c8"
FRESH_PARENT_CHECKPOINT_SHA256 = (
    "c0ebb4c1509c3d36b6834bd39977cb1cda2fcaff1747eabc97ebb41af0f06ce4"
)
FRESH_PARENT_METRICS_SHA256 = (
    "d9c0d6c41b74392b123b11e770bc122aba2ec30a835b45c157a21ad96fd14d7a"
)
FRESH_PARENT_MANIFEST_SHA256 = (
    "26ebab679d112e07c5bb364b8799cffc24b573c4abc6a71623705a4cfe962e9f"
)
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FIT_IDENTITIES = (
    (1, 232),
    (9, 440),
    (21, 402),
    (27, 403),
    (29, 399),
    (35, 403),
    (47, 514),
    (58, 404),
)
UNSEEN_IDENTITIES = (
    (5, 429),
    (26, 403),
    (42, 408),
    (50, 405),
)
WRONG_IDENTITY_MAP = {
    (1, 232): (9, 440),
    (9, 440): (21, 402),
    (21, 402): (27, 403),
    (27, 403): (29, 399),
    (29, 399): (35, 403),
    (35, 403): (47, 514),
    (47, 514): (58, 404),
    (58, 404): (1, 232),
    (5, 429): (26, 403),
    (26, 403): (42, 408),
    (42, 408): (50, 405),
    (50, 405): (5, 429),
}
EXPECTED_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
EXPECTED_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
EXPECTED_NORMALIZATION_SHA256 = (
    "4d0bca7d027a1a9f457c526f21a034a406ce4973e2dccee55ddd2d7019b41b77"
)
ORACLE_CHAMFER_GATE_M = 0.8
ORACLE_OUTLIER_GATE = 0.05
SAMPLE_RANGE_QUOTAS = (12_800, 2_720, 480)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(document: Any) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
        raise ValueError("Q-Local source commit must be a full lowercase SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("Q-Local source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"Q-Local source worktree is dirty: {dirty}")


def parse_nvidia_smi_rows(output: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 4:
            raise ValueError(f"Q-Local malformed nvidia-smi row: {line}")
        index = int(columns[0])
        if index in rows:
            raise ValueError("Q-Local nvidia-smi reported a duplicate GPU index")
        rows[index] = {
            "physical_index": index,
            "uuid": columns[1],
            "pci_bus_id": columns[2],
            "name": columns[3],
        }
    if not rows:
        raise ValueError("Q-Local nvidia-smi returned no GPUs")
    return rows


def require_h200_with_identity(device_argument: str) -> tuple[torch.device, dict[str, Any]]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Q-Local requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in ("0", "2"):
        raise RuntimeError("Q-Local requires physical H200 GPU 0 or 2")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Q-Local requires exactly one visible CUDA device")
    device = torch.device(device_argument)
    if device.type != "cuda" or device.index not in (None, 0):
        raise RuntimeError("Q-Local requires the single visible device cuda:0")
    resolved_name = torch.cuda.get_device_name(device)
    if resolved_name != "NVIDIA H200 NVL":
        raise RuntimeError(f"Q-Local requires H200, got {resolved_name}")
    output = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        stderr=subprocess.STDOUT,
    )
    rows = parse_nvidia_smi_rows(output)
    physical_index = int(visible)
    if physical_index not in rows:
        raise RuntimeError("Q-Local visible physical GPU is absent from nvidia-smi")
    provenance = rows[physical_index]
    if provenance["name"] != resolved_name:
        raise RuntimeError("Q-Local CUDA and nvidia-smi device names disagree")
    if not provenance["uuid"] or not provenance["pci_bus_id"]:
        raise RuntimeError("Q-Local GPU UUID or PCI identity is missing")
    return device, {
        **provenance,
        "cuda_visible_devices": visible,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "torch_visible_name": resolved_name,
    }


def frame_identity(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["sequence"]), int(record["radar_index"])


def resolve_frozen_cohort(
    records: list[dict[str, Any]],
) -> tuple[list[int], list[int], dict[int, int]]:
    identity_to_index: dict[tuple[int, int], int] = {}
    for index, record in enumerate(records):
        identity = frame_identity(record)
        if identity in identity_to_index:
            raise ValueError(f"Q-Local duplicate dataset identity: {identity}")
        if record.get("partition") != "train":
            raise ValueError(f"Q-Local cohort record is not train: {identity}")
        identity_to_index[identity] = index
    ordered = FIT_IDENTITIES + UNSEEN_IDENTITIES
    missing = [identity for identity in ordered if identity not in identity_to_index]
    if missing:
        raise ValueError(f"Q-Local frozen cohort is missing: {missing}")
    if len(set(ordered)) != len(ordered):
        raise AssertionError("Q-Local frozen cohort contains duplicate identities")
    if set(WRONG_IDENTITY_MAP) != set(ordered):
        raise AssertionError("Q-Local wrong-pair domain differs from the cohort")
    for source, wrong in WRONG_IDENTITY_MAP.items():
        if wrong not in identity_to_index:
            raise ValueError(f"Q-Local wrong Cube is missing: {source}->{wrong}")
        if source[0] == wrong[0]:
            raise ValueError(f"Q-Local wrong Cube stayed in one sequence: {source}")
    fit_indices = [identity_to_index[identity] for identity in FIT_IDENTITIES]
    unseen_indices = [
        identity_to_index[identity] for identity in UNSEEN_IDENTITIES
    ]
    wrong_indices = {
        identity_to_index[source]: identity_to_index[wrong]
        for source, wrong in WRONG_IDENTITY_MAP.items()
    }
    observed_fit = tuple(frame_identity(records[index]) for index in fit_indices)
    observed_unseen = tuple(
        frame_identity(records[index]) for index in unseen_indices
    )
    if observed_fit != FIT_IDENTITIES or observed_unseen != UNSEEN_IDENTITIES:
        raise AssertionError("Q-Local cohort order changed")
    return fit_indices, unseen_indices, wrong_indices


def source_hashes(repo: Path) -> dict[str, str]:
    relative_paths = (
        "code/models/qlocal_distributional_risk.py",
        "code/losses/qlocal_distributional_risk.py",
        "code/models/rald_wce_field.py",
        "code/eval/rald_wce_stage0.py",
        "code/eval/dense_geometry.py",
        "code/scripts/train_rald_wce_pilot.py",
        "code/scripts/train_rald_wce_quality_tiny.py",
        "code/scripts/preflight_qlocal_distributional_risk.py",
        "docs/qlocal_distributional_risk_tiny_protocol.md",
    )
    return {relative: sha256_file(repo / relative) for relative in relative_paths}


def cube_path(data_root: Path, record: dict[str, Any]) -> Path:
    sequence, radar_index = frame_identity(record)
    return (
        data_root
        / str(sequence)
        / "radar_tesseract"
        / f"tesseract_{radar_index:05d}.mat"
    )


def cache_path(cache_root: Path, record: dict[str, Any]) -> Path:
    sequence, radar_index = frame_identity(record)
    return cache_root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


def build_inference_config() -> WideInferenceConfig:
    return WideInferenceConfig(
        q0_range_quotas=FORMAL_Q0_RANGE_QUOTAS,
        q1_anchor_quotas=FORMAL_Q1_ANCHOR_QUOTAS,
        q1_samples_per_anchor=FORMAL_Q1_SAMPLES_PER_ANCHOR,
        output_range_quotas=FORMAL_OUTPUT_RANGE_QUOTAS,
        minimum_distance_m=CAPACITY_DISTANCE_M,
        seed=20260716,
        decode_chunk_size=8_192,
    )


def export_structural_checks(export: Any) -> dict[str, bool]:
    report = export.report
    return {
        "exact_10000": int(report["exact_point_count"]) == 10_000,
        "exact_range_quotas": report["selected_by_range"]
        == {
            "range_0_30m": 8000,
            "range_30_60m": 1700,
            "range_60_120m": 300,
        },
        "minimum_spacing_5cm": float(
            report["observed_minimum_pair_distance_m"]
        )
        >= CAPACITY_DISTANCE_M - 1e-6,
        "unique_rows": int(report["unique_selected_candidate_count"]) == 10_000,
        "no_copy_padding_jitter_duplicate": report[
            "copy_padding_jitter_duplicate"
        ]
        is False,
    }


def frame_oracle_report(
    *,
    dataset: PilotCubeDataset,
    index: int,
    base_model: torch.nn.Module,
    axes_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inference_config: WideInferenceConfig,
    device: torch.device,
    data_root: Path,
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = dataset[index]
    record = dataset.records[index]
    cube = item["cube_drae"].unsqueeze(0).to(device)
    range_m, azimuth_rad, elevation_rad = axes_tensors
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        field = frozen_candidate_field(
            base_model,
            cube,
            range_m,
            azimuth_rad,
            elevation_rad,
            inference_config,
        )

    target = item["target_xyz_confidence"].to(device).float()
    distance = continuous_geometry_quality_target(
        field["xyz_m"].float(),
        target,
        distance_temperature_m=1.0,
        candidate_chunk_size=4_096,
    )["nearest_distance_m"]
    base_export = exact_capacity_export(
        field["xyz_m"],
        field["base_confidence"][0].float(),
        output_quotas=FORMAL_OUTPUT_RANGE_QUOTAS,
        minimum_distance_m=CAPACITY_DISTANCE_M,
    )
    oracle_export = exact_capacity_export(
        field["xyz_m"],
        -distance,
        output_quotas=FORMAL_OUTPUT_RANGE_QUOTAS,
        minimum_distance_m=CAPACITY_DISTANCE_M,
    )
    base_geometry = geometry_report(
        base_export.xyz_m.to(device),
        target[:, :3],
        target_weight=target[:, 3],
    )
    oracle_geometry = geometry_report(
        oracle_export.xyz_m.to(device),
        target[:, :3],
        target_weight=target[:, 3],
    )
    base_checks = export_structural_checks(base_export)
    oracle_checks = export_structural_checks(oracle_export)
    oracle_gate = {
        "chamfer_at_most_0p8m": float(oracle_geometry["chamfer_m"])
        <= ORACLE_CHAMFER_GATE_M,
        "outlier_at_most_5pct": float(
            oracle_geometry["outlier_fraction_2m"]
        )
        <= ORACLE_OUTLIER_GATE,
        "all_structural_checks": all(oracle_checks.values()),
    }
    report = {
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
        "partition": str(item["partition"]),
        "raw_inputs": {
            "cube_path": str(cube_path(data_root, record).resolve()),
            "cube_sha256": sha256_file(cube_path(data_root, record)),
            "target_cache_path": str(cache_path(cache_root, record).resolve()),
            "target_cache_sha256": sha256_file(cache_path(cache_root, record)),
            "target_tensor_sha256": tensor_sha256(target),
        },
        "candidate_field": field["report"],
        "q1": field["q1_report"],
        "nearest_distance_sha256": tensor_sha256(distance),
        "base_confidence": {
            "geometry": base_geometry,
            "export": base_export.report,
            "hashes": base_export.hashes,
            "structural_checks": base_checks,
        },
        "gt_nearest_capacity_oracle": {
            "non_deployable": True,
            "target_used_for_selection": True,
            "geometry": oracle_geometry,
            "export": oracle_export.report,
            "hashes": oracle_export.hashes,
            "structural_checks": oracle_checks,
            "gate": oracle_gate,
            "passed": all(oracle_gate.values()),
        },
    }
    tensors = {
        "cube": cube,
        "field": field,
        "target": target,
        "nearest_distance_m": distance,
    }
    return report, tensors


def gradient_preflight(
    *,
    tensors: dict[str, Any],
    wrong_cube: torch.Tensor,
    base_model: torch.nn.Module,
    axes_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    field = tensors["field"]
    range_m, azimuth_rad, elevation_rad = axes_tensors
    spatial_shape = (
        range_m.numel(),
        azimuth_rad.numel(),
        elevation_rad.numel(),
    )
    coordinates_rae = _denormalize_bins(
        field["refined_normalized_rae"][0].float(),
        spatial_shape,
    )
    range_class = range_stratum_codes(field["xyz_m"].float())
    quality_proxy = torch.exp(-tensors["nearest_distance_m"].float())
    rows = sample_quality_candidate_rows(
        quality_proxy.detach().cpu(),
        range_class.detach().cpu(),
        range_quotas=SAMPLE_RANGE_QUOTAS,
        high_quality_fraction=0.5,
        high_quality_pool_fraction=0.1,
        seed=20260716,
    ).to(device)
    sampled_coordinates = coordinates_rae[rows].unsqueeze(0)
    matched_features = extract_local_full_raed_features(
        tensors["cube"], sampled_coordinates
    )
    wrong_features = extract_local_full_raed_features(
        wrong_cube, sampled_coordinates
    )
    wrong_condition = base_model.encode_condition(wrong_cube)["condition_latents"]
    scorer = QLocalDistributionalRiskScorer().to(device)
    parameter_count = qlocal_parameter_count(scorer)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-4, weight_decay=0.01)
    target_distance = tensors["nearest_distance_m"][rows].unsqueeze(0).float()
    target_distribution = distance_distribution_target(
        target_distance,
        scorer.distance_centers_m,
    )
    sampled_range_class = range_class[rows].unsqueeze(0).long()
    inputs = {
        "refined_normalized_rae": field["refined_normalized_rae"][:, rows].float(),
        "base_confidence": field["base_confidence"][:, rows].float(),
        "condition_latents": field["condition_latents"].float(),
        **matched_features,
    }
    losses: list[float] = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = scorer(**inputs)
        loss = qlocal_distributional_risk_loss(
            output["distance_logits"],
            target_distribution,
            target_distance,
            sampled_range_class,
            scorer.distance_centers_m,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.total.detach().item()))

    group_prefixes = {
        "coordinate_embedding": "coordinate_embedding.",
        "local_projection": "local_projection.",
        "condition_projection": "condition_projection.",
        "condition_attention": "condition_attention.",
        "feed_forward": "feed_forward.",
        "distance_delta_head": "distance_delta_head.",
    }
    gradients: dict[str, dict[str, Any]] = {}
    for group, prefix in group_prefixes.items():
        values = [
            parameter.grad.detach().float()
            for name, parameter in scorer.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        gradients[group] = {
            "tensor_count": len(values),
            "finite": bool(values) and all(
                bool(torch.isfinite(value).all()) for value in values
            ),
            "absolute_sum": float(
                sum(value.abs().sum().item() for value in values)
            ),
        }
        gradients[group]["nonzero"] = gradients[group]["absolute_sum"] > 0.0

    scorer.eval()
    with torch.no_grad():
        matched_output = scorer(**inputs)
        wrong_output = scorer(
            refined_normalized_rae=inputs["refined_normalized_rae"],
            base_confidence=inputs["base_confidence"],
            condition_latents=wrong_condition.float(),
            **wrong_features,
        )
    coordinate_hash = tensor_sha256(sampled_coordinates)
    return {
        "parameter_count": parameter_count,
        "two_update_losses": losses,
        "gradient_groups": gradients,
        "all_gradient_groups_finite_nonzero": all(
            group["finite"] and group["nonzero"] for group in gradients.values()
        ),
        "sample_count": int(rows.numel()),
        "sample_range_counts": {
            str(code): int((sampled_range_class == code).sum().item())
            for code in range(3)
        },
        "same_coordinate_intervention": {
            "matched_coordinate_sha256": coordinate_hash,
            "wrong_coordinate_sha256": coordinate_hash,
            "coordinates_bit_identical": True,
            "base_confidence_sha256": tensor_sha256(inputs["base_confidence"]),
            "matched_local_spectrum_sha256": tensor_sha256(
                matched_features["local_spectrum"]
            ),
            "wrong_local_spectrum_sha256": tensor_sha256(
                wrong_features["local_spectrum"]
            ),
            "local_spectrum_changed": not torch.equal(
                matched_features["local_spectrum"],
                wrong_features["local_spectrum"],
            ),
            "global_condition_changed": not torch.equal(
                inputs["condition_latents"], wrong_condition.float()
            ),
            "selection_score_changed": not torch.equal(
                matched_output["selection_score"],
                wrong_output["selection_score"],
            ),
        },
        "model_contract": scorer.architecture_metadata(),
    }


def atomic_commit_report(output_dir: Path, report: dict[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Q-Local preflight output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        path = staging / "preflight.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "preflight.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--fresh-parent-checkpoint", type=Path, required=True)
    parser.add_argument("--fresh-parent-metrics", type=Path, required=True)
    parser.add_argument("--fresh-parent-run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, gpu = require_h200_with_identity(args.device)
    torch.manual_seed(20260716)
    np.random.seed(20260716)

    input_hashes, manifest_counts = validate_frozen_inputs(
        args.manifest,
        args.scene_split,
        args.normalization,
        repo,
    )
    expected_inputs = {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "scene_split_sha256": EXPECTED_SCENE_SPLIT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
    }
    for key, expected in expected_inputs.items():
        if input_hashes[key] != expected:
            raise ValueError(f"Q-Local frozen {key} hash changed")

    checkpoint_hash = sha256_file(args.fresh_parent_checkpoint)
    metrics_hash = sha256_file(args.fresh_parent_metrics)
    manifest_hash = sha256_file(args.fresh_parent_run_manifest)
    if checkpoint_hash != FRESH_PARENT_CHECKPOINT_SHA256:
        raise ValueError("Q-Local fresh-parent checkpoint hash changed")
    if metrics_hash != FRESH_PARENT_METRICS_SHA256:
        raise ValueError("Q-Local fresh-parent metrics hash changed")
    if manifest_hash != FRESH_PARENT_MANIFEST_SHA256:
        raise ValueError("Q-Local fresh-parent manifest hash changed")
    checkpoint = torch.load(
        args.fresh_parent_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("protocol") != FRESH_PARENT_PROTOCOL:
        raise ValueError("Q-Local input is not a Fresh-WCE checkpoint")
    if checkpoint.get("source_commit") != FRESH_PARENT_SOURCE_COMMIT:
        raise ValueError("Q-Local fresh-parent source identity changed")
    if int(checkpoint.get("epoch", -1)) != 20:
        raise ValueError("Q-Local fresh parent must be epoch 20")
    fresh_manifest = json.loads(
        args.fresh_parent_run_manifest.read_text(encoding="utf-8")
    )
    if fresh_manifest.get("source_commit") != FRESH_PARENT_SOURCE_COMMIT:
        raise ValueError("Q-Local fresh-parent manifest source changed")
    if fresh_manifest.get("input_hashes") != {
        "manifest": EXPECTED_MANIFEST_SHA256,
        "scene_split": EXPECTED_SCENE_SPLIT_SHA256,
        "normalization": EXPECTED_NORMALIZATION_SHA256,
        "corrected_dense_geometry": input_hashes[
            "dense_geometry_evaluator_sha256"
        ],
    }:
        raise ValueError("Q-Local fresh-parent data binding changed")

    dataset = PilotCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        "train",
    )
    fit_indices, unseen_indices, wrong_indices = resolve_frozen_cohort(
        dataset.records
    )
    cohort_indices = fit_indices + unseen_indices
    cohort_binding = []
    for index in cohort_indices:
        record = dataset.records[index]
        cube = cube_path(args.data_root, record)
        cache = cache_path(args.cache_root, record)
        cohort_binding.append(
            {
                "identity": list(frame_identity(record)),
                "group": "fit" if index in fit_indices else "unseen_train",
                "cube_path": str(cube.resolve()),
                "cube_sha256": sha256_file(cube),
                "target_cache_path": str(cache.resolve()),
                "target_cache_sha256": sha256_file(cache),
                "wrong_identity": list(
                    frame_identity(dataset.records[wrong_indices[index]])
                ),
            }
        )
    cohort_digest = canonical_digest(cohort_binding)

    log_center, log_scale = load_normalization(args.normalization)
    axes = load_axes(args.data_root / "resources")
    axes_tensors = (
        torch.as_tensor(axes.range_m, dtype=torch.float32, device=device),
        torch.as_tensor(axes.azimuth_rad, dtype=torch.float32, device=device),
        torch.as_tensor(axes.elevation_rad, dtype=torch.float32, device=device),
    )
    base_model = build_formal_base(
        checkpoint,
        log_center=log_center,
        log_scale=log_scale,
        device=device,
    )
    base_state_before = state_dict_sha256(base_model)
    inference_config = build_inference_config()
    frame_reports: list[dict[str, Any]] = []
    gradient_report: dict[str, Any] | None = None
    capacity_error: dict[str, Any] | None = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for position, index in enumerate(cohort_indices):
            report, tensors = frame_oracle_report(
                dataset=dataset,
                index=index,
                base_model=base_model,
                axes_tensors=axes_tensors,
                inference_config=inference_config,
                device=device,
                data_root=args.data_root,
                cache_root=args.cache_root,
            )
            report["group"] = "fit" if index in fit_indices else "unseen_train"
            report["cohort_position"] = position
            report["wrong_identity"] = list(
                frame_identity(dataset.records[wrong_indices[index]])
            )
            frame_reports.append(report)
            if position == 0:
                wrong_item = dataset[wrong_indices[index]]
                wrong_cube = wrong_item["cube_drae"].unsqueeze(0).to(device)
                gradient_report = gradient_preflight(
                    tensors=tensors,
                    wrong_cube=wrong_cube,
                    base_model=base_model,
                    axes_tensors=axes_tensors,
                    device=device,
                )
                del wrong_item, wrong_cube
            del tensors
            torch.cuda.empty_cache()
    except ExactExportCapacityError as error:
        capacity_error = error.report

    base_state_after = state_dict_sha256(base_model)
    parent_unchanged = base_state_after == base_state_before
    all_oracle_frames_passed = (
        capacity_error is None
        and len(frame_reports) == len(cohort_indices)
        and all(
            frame["gt_nearest_capacity_oracle"]["passed"]
            for frame in frame_reports
        )
    )
    gradient_passed = bool(
        gradient_report
        and gradient_report["all_gradient_groups_finite_nonzero"]
        and gradient_report["same_coordinate_intervention"][
            "coordinates_bit_identical"
        ]
        and gradient_report["same_coordinate_intervention"][
            "local_spectrum_changed"
        ]
        and gradient_report["same_coordinate_intervention"][
            "global_condition_changed"
        ]
        and gradient_report["same_coordinate_intervention"][
            "selection_score_changed"
        ]
    )
    passed = all_oracle_frames_passed and gradient_passed and parent_unchanged
    status = (
        "qlocal_f0_preflight_passed"
        if passed
        else "qlocal_f0_capacity_no_go"
        if not all_oracle_frames_passed
        else "qlocal_f0_implementation_invalid"
    )
    elapsed = time.monotonic() - started
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "status": status,
        "passed": passed,
        "decision": {
            "training_authorized": passed,
            "all_12_oracle_frames_passed": all_oracle_frames_passed,
            "gradient_and_intervention_preflight_passed": gradient_passed,
            "fresh_parent_unchanged": parent_unchanged,
            "capacity_error": capacity_error,
        },
        "source_hashes": source_hashes(repo),
        "input_binding": {
            "frozen_inputs": input_hashes,
            "manifest_counts": manifest_counts,
            "fresh_parent_checkpoint_sha256": checkpoint_hash,
            "fresh_parent_metrics_sha256": metrics_hash,
            "fresh_parent_run_manifest_sha256": manifest_hash,
            "cohort_ordered_digest_sha256": cohort_digest,
            "cohort": cohort_binding,
        },
        "fresh_parent": {
            "name": "Fresh-WCE-20",
            "source_commit": FRESH_PARENT_SOURCE_COMMIT,
            "epoch": 20,
            "not_deleted_original_identity": True,
            "q1r_certificate_used_as_authorizer": False,
            "state_sha256_before": base_state_before,
            "state_sha256_after": base_state_after,
            "unchanged": parent_unchanged,
        },
        "cohort_contract": {
            "fit_identities": [list(identity) for identity in FIT_IDENTITIES],
            "unseen_train_identities": [
                list(identity) for identity in UNSEEN_IDENTITIES
            ],
            "wrong_identity_map": {
                f"{source[0]}:{source[1]}": list(wrong)
                for source, wrong in WRONG_IDENTITY_MAP.items()
            },
            "validation_accessed": False,
            "test_accessed": False,
            "future_accessed": False,
        },
        "oracle_gate": {
            "non_deployable": True,
            "per_frame_chamfer_m_at_most": ORACLE_CHAMFER_GATE_M,
            "per_frame_outlier_fraction_at_most": ORACLE_OUTLIER_GATE,
            "exact_output_count": 10_000,
            "output_range_quotas": list(FORMAL_OUTPUT_RANGE_QUOTAS),
            "minimum_spacing_m": CAPACITY_DISTANCE_M,
        },
        "frames": frame_reports,
        "gradient_preflight": gradient_report,
        "runtime": {
            "gpu": gpu,
            "device_argument": args.device,
            "torch_version": torch.__version__,
            "elapsed_seconds": elapsed,
            "measured_gpu_seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "evidence_boundary": {
            "target_used_only_for_capacity_oracle_labels_and_metrics": True,
            "target_used_for_deployable_inference": False,
            "validation_accessed": False,
            "test_accessed": False,
            "future_cube_accessed": False,
            "doppler_output_evaluated": False,
            "best_of_k": False,
        },
    }
    output_path = atomic_commit_report(args.output_dir, report)
    print(
        json.dumps(
            {
                "status": status,
                "passed": passed,
                "output": str(output_path),
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
