"""Target-free reconstruction of the frozen Fresh-WCE 700k candidate field."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

import torch

from eval.rald_wce_stage0 import (
    FORMAL_Q0_RANGE_QUOTAS,
    FORMAL_Q1_ANCHOR_QUOTAS,
    FORMAL_Q1_SAMPLES_PER_ANCHOR,
    WideInferenceConfig,
    _candidate_tensors,
    fixed_wide_q0,
    occupancy_dependent_q1,
    tensor_sha256,
)
from models.rald_wce_field import RaLDWCEField


PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
PROTOCOL_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
FRESH_PARENT_PROTOCOL = "g1_ra1_rald_wce_stage0_v1"
FRESH_PARENT_SOURCE_COMMIT = "f2a9489d40323d1ef45d85de958f4aea8126e1c8"
FORMAL_SEED = 20260716
FORMAL_CANDIDATE_COUNT = 700_000
FORMAL_Q0_COUNT = 500_000
FORMAL_Q1_COUNT = 200_000
FORMAL_ARCHITECTURE = {
    "latent_count": 512,
    "model_dim": 512,
    "depth": 6,
    "heads": 8,
    "head_dim": 64,
    "decode_chunk_size": 8_192,
}
FORMAL_SELECTION_METRIC = (
    "mean_chamfer + completeness_gate_excess + "
    "2*outlier_gate_excess + far_gate_excess + "
    "10*wrong_condition_gate_deficit"
)
FORMAL_CHECKPOINT_CONFIG = {
    "protocol": FRESH_PARENT_PROTOCOL,
    "seed": FORMAL_SEED,
    "smoke": False,
    "epochs": 20,
    "eval_every": 5,
    "train_limit": None,
    "validation_limit": None,
    "learning_rate": 1e-4,
    "minimum_learning_rate": 1e-6,
    "weight_decay": 0.05,
    "gradient_clip_norm": 10.0,
    "occupancy_query_count": 16_000,
    "positive_query_ratio": 0.0625,
    **FORMAL_ARCHITECTURE,
    "q0_range_quotas": FORMAL_Q0_RANGE_QUOTAS,
    "q1_anchor_quotas": FORMAL_Q1_ANCHOR_QUOTAS,
    "q1_samples_per_anchor": FORMAL_Q1_SAMPLES_PER_ANCHOR,
    "output_range_quotas": (8_000, 1_700, 300),
    "minimum_export_distance_m": 0.05,
    "selection_metric": FORMAL_SELECTION_METRIC,
    "test_accessed": False,
    "doppler_head": False,
}
PREDECESSOR_HASH_KEYS = (
    "q0_sha256",
    "q1_sha256",
    "candidate_xyz_sha256",
    "base_confidence_sha256",
)
CONTENT_HASH_KEYS = (
    "stable_candidate_id_sha256",
    "candidate_xyz_sha256",
    "base_confidence_sha256",
)
HASH_KEYS = ("stable_candidate_id_sha256", *PREDECESSOR_HASH_KEYS)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CandidateVerification:
    sequence: int
    radar_index: int
    hashes: dict[str, str]
    predecessor_expected: bool
    predecessor_hashes_match: bool | None
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and self.predecessor_hashes_match is not False

    def report(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "radar_index": self.radar_index,
            "hashes": dict(self.hashes),
            "predecessor_expected": self.predecessor_expected,
            "predecessor_hashes_match": self.predecessor_hashes_match,
            "checks": dict(self.checks),
            "passed": self.passed,
            "support_target_input": False,
            "ground_truth_accessed": False,
        }


def frozen_inference_config() -> WideInferenceConfig:
    config = WideInferenceConfig(
        q0_range_quotas=FORMAL_Q0_RANGE_QUOTAS,
        q1_anchor_quotas=FORMAL_Q1_ANCHOR_QUOTAS,
        q1_samples_per_anchor=FORMAL_Q1_SAMPLES_PER_ANCHOR,
        seed=FORMAL_SEED,
        decode_chunk_size=FORMAL_ARCHITECTURE["decode_chunk_size"],
    )
    config.validate()
    if sum(config.q0_range_quotas) != FORMAL_Q0_COUNT:
        raise AssertionError("STDA-F0 Q0 cardinality changed")
    if (
        sum(config.q1_anchor_quotas) * config.q1_samples_per_anchor
        != FORMAL_Q1_COUNT
    ):
        raise AssertionError("STDA-F0 Q1 cardinality changed")
    return config


def _same_frozen_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(actual) == len(expected) and all(
            _same_frozen_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _validate_checkpoint_config(config: Mapping[str, Any]) -> None:
    if set(config) != set(FORMAL_CHECKPOINT_CONFIG):
        missing = sorted(set(FORMAL_CHECKPOINT_CONFIG).difference(config))
        extra = sorted(set(config).difference(FORMAL_CHECKPOINT_CONFIG))
        raise ValueError(
            "STDA-F0 checkpoint configuration keys changed: "
            f"missing={missing}, extra={extra}"
        )
    mismatched = [
        key
        for key, expected in FORMAL_CHECKPOINT_CONFIG.items()
        if not _same_frozen_value(config[key], expected)
    ]
    if mismatched:
        raise ValueError(f"STDA-F0 formal configuration changed: {mismatched}")


def _validate_model_runtime(model: torch.nn.Module, device: torch.device) -> None:
    if model.training:
        raise ValueError("STDA-F0 Fresh-WCE model must be in evaluation mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("STDA-F0 Fresh-WCE parameters must be frozen")
    parameter_devices = {parameter.device for parameter in model.parameters()}
    buffer_devices = {buffer.device for buffer in model.buffers()}
    if parameter_devices.union(buffer_devices).difference({device}):
        raise ValueError("STDA-F0 Fresh-WCE tensors are on the wrong device")
    runtime_architecture = {
        "latent_count": getattr(model, "latent_count", None),
        "model_dim": getattr(model, "model_dim", None),
        "depth": getattr(model, "depth", None),
        "decode_chunk_size": getattr(model, "decode_chunk_size", None),
    }
    expected = {
        key: FORMAL_ARCHITECTURE[key]
        for key in runtime_architecture
    }
    if runtime_architecture != expected:
        raise ValueError(
            "STDA-F0 Fresh-WCE runtime architecture changed: "
            f"{runtime_architecture}"
        )


def build_frozen_base_model(
    checkpoint: Mapping[str, Any],
    *,
    log_center: float,
    log_scale: float,
    device: torch.device,
) -> RaLDWCEField:
    """Build the unchanged Fresh-WCE-20 field without importing a data loader."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("STDA-F0 checkpoint must be a mapping")
    if checkpoint.get("protocol") != FRESH_PARENT_PROTOCOL:
        raise ValueError("STDA-F0 checkpoint protocol changed")
    if checkpoint.get("source_commit") != FRESH_PARENT_SOURCE_COMMIT:
        raise ValueError("STDA-F0 checkpoint source changed")
    if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] != 20:
        raise ValueError("STDA-F0 requires the Fresh-WCE epoch-20 endpoint")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("STDA-F0 checkpoint has no configuration")
    _validate_checkpoint_config(checkpoint_config)
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("STDA-F0 checkpoint has no model state")
    if not isinstance(device, torch.device):
        raise TypeError("STDA-F0 model device must be torch.device")
    if not math.isfinite(log_center) or not math.isfinite(log_scale):
        raise ValueError("STDA-F0 normalization must be finite")
    if log_scale <= 0.0:
        raise ValueError("STDA-F0 normalization scale must be positive")

    model = RaLDWCEField(
        log_center=log_center,
        log_scale=log_scale,
        latent_count=FORMAL_ARCHITECTURE["latent_count"],
        model_dim=FORMAL_ARCHITECTURE["model_dim"],
        depth=FORMAL_ARCHITECTURE["depth"],
        heads=FORMAL_ARCHITECTURE["heads"],
        head_dim=FORMAL_ARCHITECTURE["head_dim"],
        decode_chunk_size=FORMAL_ARCHITECTURE["decode_chunk_size"],
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    _validate_model_runtime(model, device)
    return model


def axes_tensors(axes: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(axes.range_m, dtype=torch.float32, device=device),
        torch.as_tensor(axes.azimuth_rad, dtype=torch.float32, device=device),
        torch.as_tensor(axes.elevation_rad, dtype=torch.float32, device=device),
    )


def _validate_reconstruction_inputs(
    base_model: torch.nn.Module,
    cube_drae: torch.Tensor,
    axes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    if not isinstance(base_model, torch.nn.Module):
        raise TypeError("STDA-F0 base model must be a torch module")
    if not isinstance(cube_drae, torch.Tensor):
        raise TypeError("STDA-F0 Cube must be a tensor")
    if tuple(cube_drae.shape) != (1, 64, 256, 107, 37):
        raise ValueError("STDA-F0 expects one Full-RAED Cube")
    if cube_drae.dtype != torch.float32:
        raise TypeError("STDA-F0 Full-RAED Cube must be float32")
    if not bool(torch.isfinite(cube_drae).all()):
        raise ValueError("STDA-F0 Full-RAED Cube must be finite")
    expected_axis_shapes = ((256,), (107,), (37,))
    names = ("range", "azimuth", "elevation")
    for axis, expected_shape, name in zip(
        axes, expected_axis_shapes, names, strict=True
    ):
        if not isinstance(axis, torch.Tensor):
            raise TypeError(f"STDA-F0 {name} axis must be a tensor")
        if tuple(axis.shape) != expected_shape:
            raise ValueError(f"STDA-F0 {name} axis shape changed")
        if axis.dtype != torch.float32:
            raise TypeError(f"STDA-F0 {name} axis must be float32")
        if axis.device != cube_drae.device:
            raise ValueError(f"STDA-F0 {name} axis is on the wrong device")
        if not bool(torch.isfinite(axis).all()):
            raise ValueError(f"STDA-F0 {name} axis must be finite")
        if not bool((axis[1:] > axis[:-1]).all()):
            raise ValueError(f"STDA-F0 {name} axis must be strictly increasing")
    _validate_model_runtime(base_model, cube_drae.device)


def _validate_candidate_tensors(
    *,
    stable_candidate_id: torch.Tensor,
    xyz_m: torch.Tensor,
    base_confidence: torch.Tensor,
    q0: torch.Tensor,
    q1: torch.Tensor,
    device: torch.device,
) -> None:
    expected = (
        (stable_candidate_id, (FORMAL_CANDIDATE_COUNT,), torch.int64, "row ID"),
        (xyz_m, (FORMAL_CANDIDATE_COUNT, 3), torch.float32, "XYZ"),
        (
            base_confidence,
            (FORMAL_CANDIDATE_COUNT,),
            torch.float32,
            "base confidence",
        ),
        (q0, (1, FORMAL_Q0_COUNT, 3), torch.float32, "Q0"),
        (q1, (1, FORMAL_Q1_COUNT, 3), torch.float32, "Q1"),
    )
    for value, shape, dtype, name in expected:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"STDA-F0 candidate {name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"STDA-F0 candidate {name} shape changed")
        if value.dtype != dtype:
            raise TypeError(f"STDA-F0 candidate {name} dtype changed")
        if value.device != device:
            raise ValueError(f"STDA-F0 candidate {name} device changed")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"STDA-F0 candidate {name} is non-finite")
    expected_ids = torch.arange(
        FORMAL_CANDIDATE_COUNT,
        dtype=torch.int64,
        device=device,
    )
    if not torch.equal(stable_candidate_id, expected_ids):
        raise ValueError("STDA-F0 stable candidate row IDs changed")


@torch.no_grad()
def reconstruct_candidate_field(
    base_model: RaLDWCEField,
    cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    inference_config: WideInferenceConfig,
) -> dict[str, Any]:
    """Reconstruct exact Q0/Q1 candidates; callers provide the BF16 context."""

    frozen = frozen_inference_config()
    if inference_config != frozen:
        raise ValueError("STDA-F0 candidate inference configuration changed")
    _validate_reconstruction_inputs(
        base_model,
        cube_drae,
        (range_m, azimuth_rad, elevation_rad),
    )

    q0 = fixed_wide_q0(
        range_m,
        azimuth_rad,
        elevation_rad,
        quotas=inference_config.q0_range_quotas,
        seed=inference_config.seed,
    ).to(cube_drae.device)
    encoded = base_model.encode_condition(cube_drae)
    if not isinstance(encoded, Mapping):
        raise TypeError("STDA-F0 Fresh-WCE encoder returned a non-mapping")
    condition_latents = encoded.get("condition_latents")
    if not isinstance(condition_latents, torch.Tensor):
        raise TypeError("STDA-F0 Fresh-WCE condition latents are absent")
    if condition_latents.device != cube_drae.device or not bool(
        torch.isfinite(condition_latents).all()
    ):
        raise ValueError("STDA-F0 Fresh-WCE condition latents are invalid")
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
    stable_candidate_id = torch.arange(
        FORMAL_CANDIDATE_COUNT,
        dtype=torch.int64,
        device=xyz_m.device,
    )
    _validate_candidate_tensors(
        stable_candidate_id=stable_candidate_id,
        xyz_m=xyz_m,
        base_confidence=base_confidence,
        q0=q0,
        q1=q1,
        device=cube_drae.device,
    )
    return {
        "stable_candidate_id": stable_candidate_id,
        "xyz_m": xyz_m,
        "base_confidence": base_confidence,
        "q0": q0,
        "q1": q1,
        "q1_report": q1_report,
        "report": {
            "candidate_count": FORMAL_CANDIDATE_COUNT,
            "q0_count": FORMAL_Q0_COUNT,
            "q1_count": FORMAL_Q1_COUNT,
            "stable_candidate_id_sha256": tensor_sha256(stable_candidate_id),
            "q0_sha256": tensor_sha256(q0),
            "q1_sha256": tensor_sha256(q1),
            "candidate_xyz_sha256": tensor_sha256(xyz_m),
            "base_confidence_sha256": tensor_sha256(base_confidence),
            "ground_truth_accessed": False,
            "support_target_input": False,
            "candidate_coordinates_changed": False,
            "formal_residual_changed": False,
        },
    }


def verify_candidate_only(
    field: Mapping[str, Any],
    *,
    sequence: int,
    radar_index: int,
    expected_hashes: Mapping[str, str] | None,
) -> CandidateVerification:
    """Verify a reconstructed field using only candidate-side tensors/hashes."""

    if type(sequence) is not int or sequence < 0:
        raise ValueError("STDA-F0 sequence must be a nonnegative integer")
    if type(radar_index) is not int or radar_index < 0:
        raise ValueError("STDA-F0 radar index must be a nonnegative integer")
    stable_candidate_id = field.get("stable_candidate_id")
    xyz = field.get("xyz_m")
    confidence = field.get("base_confidence")
    q0 = field.get("q0")
    q1 = field.get("q1")
    report = field.get("report")
    if not isinstance(report, Mapping):
        raise ValueError("STDA-F0 candidate report is absent")
    tensors = (stable_candidate_id, xyz, confidence, q0, q1)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("STDA-F0 candidate field contains non-tensor values")
    _validate_candidate_tensors(
        stable_candidate_id=stable_candidate_id,
        xyz_m=xyz,
        base_confidence=confidence,
        q0=q0,
        q1=q1,
        device=xyz.device,
    )
    actual = {key: str(report.get(key, "")) for key in HASH_KEYS}
    recomputed = {
        "stable_candidate_id_sha256": tensor_sha256(stable_candidate_id),
        "q0_sha256": tensor_sha256(q0),
        "q1_sha256": tensor_sha256(q1),
        "candidate_xyz_sha256": tensor_sha256(xyz),
        "base_confidence_sha256": tensor_sha256(confidence),
    }
    checks = {
        "candidate_count": tuple(xyz.shape) == (FORMAL_CANDIDATE_COUNT, 3),
        "stable_candidate_id_count": tuple(stable_candidate_id.shape)
        == (FORMAL_CANDIDATE_COUNT,),
        "confidence_count": tuple(confidence.shape) == (FORMAL_CANDIDATE_COUNT,),
        "q0_shape": tuple(q0.shape) == (1, FORMAL_Q0_COUNT, 3),
        "q1_shape": tuple(q1.shape) == (1, FORMAL_Q1_COUNT, 3),
        "finite_xyz": bool(torch.isfinite(xyz).all()),
        "finite_base_confidence": bool(torch.isfinite(confidence).all()),
        "finite_q0": bool(torch.isfinite(q0).all()),
        "finite_q1": bool(torch.isfinite(q1).all()),
        "stable_candidate_id_dtype": stable_candidate_id.dtype == torch.int64,
        "candidate_xyz_dtype": xyz.dtype == torch.float32,
        "base_confidence_dtype": confidence.dtype == torch.float32,
        "q0_dtype": q0.dtype == torch.float32,
        "q1_dtype": q1.dtype == torch.float32,
        "common_device": len({value.device for value in tensors}) == 1,
        "hashes_well_formed": all(
            _SHA256_PATTERN.fullmatch(value) is not None
            for value in actual.values()
        ),
        "reported_hashes_replay": actual == recomputed,
        "reported_candidate_count": report.get("candidate_count")
        == FORMAL_CANDIDATE_COUNT,
        "reported_q0_count": report.get("q0_count") == FORMAL_Q0_COUNT,
        "reported_q1_count": report.get("q1_count") == FORMAL_Q1_COUNT,
        "ground_truth_accessed_false": report.get("ground_truth_accessed") is False,
        "support_target_input_false": report.get("support_target_input") is False,
        "candidate_coordinates_changed_false": report.get(
            "candidate_coordinates_changed"
        )
        is False,
        "formal_residual_changed_false": report.get("formal_residual_changed")
        is False,
    }
    predecessor_match: bool | None = None
    if expected_hashes is not None:
        if not isinstance(expected_hashes, Mapping):
            raise TypeError("STDA-F0 predecessor hashes must be a mapping")
        if set(expected_hashes) != set(PREDECESSOR_HASH_KEYS):
            raise ValueError("STDA-F0 predecessor hash keys changed")
        expected = {
            key: str(expected_hashes[key]) for key in PREDECESSOR_HASH_KEYS
        }
        if any(
            _SHA256_PATTERN.fullmatch(value) is None
            for value in expected.values()
        ):
            raise ValueError("STDA-F0 predecessor hash is malformed")
        predecessor_match = all(actual[key] == expected[key] for key in expected)
    result = CandidateVerification(
        sequence=int(sequence),
        radar_index=int(radar_index),
        hashes=actual,
        predecessor_expected=expected_hashes is not None,
        predecessor_hashes_match=predecessor_match,
        checks=checks,
    )
    if not result.passed:
        raise ValueError(f"STDA-F0 candidate-only verification failed: {result.report()}")
    return result
