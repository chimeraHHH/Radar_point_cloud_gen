"""Data and intervention contracts for direct multi-horizon forecasting."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


DIRECT_HORIZONS_SECONDS = (0.5, 1.5, 2.5)
INTERVENTION_NAMES = (
    "wrong_history",
    "zero_history",
    "doppler_sign_flip",
    "time_disorder",
    "cube_mismatch",
)


def _as_transform(frame: Mapping[str, object]) -> np.ndarray:
    raw = frame.get("current_radar_from_previous_radar")
    if not isinstance(raw, list) or len(raw) != 16:
        raise ValueError(
            "D-MHW requires calibrated current_radar_from_previous_radar"
        )
    transform = np.asarray(raw, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(transform).all():
        raise ValueError("D-MHW radar-frame transform must be finite")
    return transform


def _compose_target_from_current(
    frames: list[dict],
    current_index: int,
    target_index: int,
) -> np.ndarray:
    if target_index <= current_index:
        raise ValueError("D-MHW targets must be strictly in the future")
    transform = np.eye(4, dtype=np.float64)
    for index in range(current_index + 1, target_index + 1):
        transform = _as_transform(frames[index]) @ transform
    return transform


def build_direct_horizon_examples(
    manifest: Mapping[str, object],
    *,
    history_frame_count: int = 3,
    horizon_tolerance_seconds: float = 0.05,
) -> list[dict]:
    """Build causal three-horizon records without exposing future Cubes."""

    if manifest.get("gate_pass") is not True:
        raise ValueError("D-MHW temporal manifest did not pass its gate")
    checks = manifest.get("checks")
    if not isinstance(checks, Mapping) or checks.get(
        "radar_frame_ego_transforms_present"
    ) is not True:
        raise ValueError("D-MHW manifest lacks radar-frame ego transforms")
    if history_frame_count < 1:
        raise ValueError("D-MHW requires at least one strictly historical Cube")
    if horizon_tolerance_seconds <= 0.0:
        raise ValueError("D-MHW horizon tolerance must be positive")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("D-MHW temporal manifest is empty")
    if any(frame.get("partition") == "test" for frame in frames):
        raise ValueError("D-MHW development contract refuses test frames")

    by_window: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        by_window[str(frame["window_id"])].append(frame)

    examples = []
    for window_id in sorted(by_window):
        window_frames = sorted(
            by_window[window_id],
            key=lambda frame: int(frame["frame_in_window"]),
        )
        positions = [
            int(frame["frame_in_window"]) for frame in window_frames
        ]
        if positions != list(range(len(window_frames))):
            raise ValueError(f"Non-contiguous D-MHW window {window_id}")
        partition = str(window_frames[0]["partition"])
        if any(str(frame["partition"]) != partition for frame in window_frames):
            raise ValueError(f"Mixed partition D-MHW window {window_id}")

        for current_index in range(history_frame_count, len(window_frames)):
            current = window_frames[current_index]
            current_timestamp = float(current["timestamp"])
            future = window_frames[current_index + 1 :]
            target_indices = []
            errors = []
            for horizon in DIRECT_HORIZONS_SECONDS:
                if not future:
                    break
                target = min(
                    future,
                    key=lambda frame: abs(
                        float(frame["timestamp"])
                        - current_timestamp
                        - horizon
                    ),
                )
                target_index = int(target["frame_in_window"])
                error = abs(
                    float(target["timestamp"])
                    - current_timestamp
                    - horizon
                )
                if error > horizon_tolerance_seconds:
                    break
                target_indices.append(target_index)
                errors.append(error)
            if len(target_indices) != len(DIRECT_HORIZONS_SECONDS):
                continue
            if len(set(target_indices)) != len(target_indices):
                raise RuntimeError("D-MHW horizons resolved to duplicate targets")

            history = window_frames[
                current_index - history_frame_count : current_index
            ]
            targets = [window_frames[index] for index in target_indices]
            examples.append(
                {
                    "window_id": window_id,
                    "partition": partition,
                    "sequence": int(current["sequence"]),
                    "history": [
                        {
                            "radar_index": int(frame["radar_index"]),
                            "frame_in_window": int(
                                frame["frame_in_window"]
                            ),
                            "relative_seconds": (
                                float(frame["timestamp"])
                                - current_timestamp
                            ),
                        }
                        for frame in history
                    ],
                    "current": {
                        "radar_index": int(current["radar_index"]),
                        "frame_in_window": current_index,
                    },
                    "targets": [
                        {
                            "horizon_seconds": horizon,
                            "horizon_error_seconds": error,
                            "radar_index": int(target["radar_index"]),
                            "lidar64_index": int(target["lidar64_index"]),
                            "frame_in_window": target_index,
                            "target_from_current": (
                                _compose_target_from_current(
                                    window_frames,
                                    current_index,
                                    target_index,
                                )
                                .reshape(-1)
                                .tolist()
                            ),
                        }
                        for horizon, error, target, target_index in zip(
                            DIRECT_HORIZONS_SECONDS,
                            errors,
                            targets,
                            target_indices,
                            strict=True,
                        )
                    ],
                    "future_cube_exposed": False,
                }
            )
    if not examples:
        raise ValueError("No legal D-MHW three-horizon examples")
    return examples


def _cube_path(root: Path, sequence: int, radar_index: int) -> Path:
    return (
        root
        / str(sequence)
        / "radar_tesseract"
        / f"tesseract_{radar_index:05d}.mat"
    )


def _lidar_path(root: Path, sequence: int, lidar_index: int) -> Path:
    return (
        root
        / str(sequence)
        / "os2-64"
        / f"os2-64_{lidar_index:05d}.pcd"
    )


def _target_cache_path(
    root: Path, sequence: int, radar_index: int
) -> Path:
    return root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


def audit_direct_horizon_supervision(
    manifest_path: Path,
    *,
    raw_data_root: Path | None = None,
    target_cache_roots: Mapping[str, Path] | None = None,
    history_frame_count: int = 3,
    horizon_tolerance_seconds: float = 0.05,
) -> dict:
    """Audit causal inputs and future geometry labels without loading a Cube."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = build_direct_horizon_examples(
        manifest,
        history_frame_count=history_frame_count,
        horizon_tolerance_seconds=horizon_tolerance_seconds,
    )
    count_by_partition = Counter(
        example["partition"] for example in examples
    )
    errors = [
        target["horizon_error_seconds"]
        for example in examples
        for target in example["targets"]
    ]
    input_cube_paths: set[Path] = set()
    target_lidar_paths: set[Path] = set()
    target_cache_paths: dict[str, set[Path]] = defaultdict(set)
    if raw_data_root is not None:
        for example in examples:
            sequence = int(example["sequence"])
            for frame in (*example["history"], example["current"]):
                input_cube_paths.add(
                    _cube_path(
                        raw_data_root,
                        sequence,
                        int(frame["radar_index"]),
                    )
                )
            for target in example["targets"]:
                target_lidar_paths.add(
                    _lidar_path(
                        raw_data_root,
                        sequence,
                        int(target["lidar64_index"]),
                    )
                )
    if target_cache_roots:
        for example in examples:
            partition = str(example["partition"])
            root = target_cache_roots.get(partition)
            if root is None:
                continue
            sequence = int(example["sequence"])
            for target in example["targets"]:
                target_cache_paths[partition].add(
                    _target_cache_path(
                        root,
                        sequence,
                        int(target["radar_index"]),
                    )
                )

    raw_cube_missing = sum(not path.exists() for path in input_cube_paths)
    raw_lidar_missing = sum(not path.exists() for path in target_lidar_paths)
    cache_coverage = {}
    for partition in sorted(count_by_partition):
        paths = target_cache_paths.get(partition, set())
        present = sum(path.exists() for path in paths)
        cache_coverage[partition] = {
            "unique_required_target_count": len(paths),
            "present_target_count": present,
            "missing_target_count": len(paths) - present,
            "cache_root_provided": bool(
                target_cache_roots
                and target_cache_roots.get(partition) is not None
            ),
        }

    return {
        "protocol": "dmhw_direct_three_horizon_supervision_audit_v1",
        "manifest": str(manifest_path.resolve()),
        "history_frame_count": history_frame_count,
        "horizons_seconds": list(DIRECT_HORIZONS_SECONDS),
        "horizon_tolerance_seconds": horizon_tolerance_seconds,
        "example_count": len(examples),
        "example_count_by_partition": dict(sorted(count_by_partition.items())),
        "maximum_horizon_error_seconds": max(errors),
        "future_cube_exposed": False,
        "test_accessed": False,
        "raw_input_cube_unique_count": len(input_cube_paths),
        "raw_input_cube_missing_count": raw_cube_missing,
        "future_geometry_lidar_unique_count": len(target_lidar_paths),
        "future_geometry_lidar_missing_count": raw_lidar_missing,
        "target_cache_coverage": cache_coverage,
        "raw_supervision_available": (
            raw_data_root is not None
            and raw_cube_missing == 0
            and raw_lidar_missing == 0
        ),
        "prepared_training_cache_complete": (
            bool(target_cache_roots)
            and all(
                coverage["cache_root_provided"]
                and coverage["missing_target_count"] == 0
                for coverage in cache_coverage.values()
            )
        ),
    }


def clone_float_inputs(
    inputs: Mapping[str, torch.Tensor],
    *,
    requires_grad: bool,
) -> dict[str, torch.Tensor]:
    cloned = {}
    for name, value in inputs.items():
        tensor = value.detach().clone()
        if requires_grad and torch.is_floating_point(tensor):
            tensor.requires_grad_(True)
        cloned[name] = tensor
    return cloned


def _doppler_sign_indices(doppler_mps: torch.Tensor) -> torch.Tensor:
    distance = (
        doppler_mps[:, None] + doppler_mps[None, :]
    ).abs()
    return distance.argmin(dim=1)


def apply_dmhw_intervention(
    inputs: Mapping[str, torch.Tensor],
    name: str,
    doppler_mps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply one frozen intervention while retaining an autograd graph."""

    if name not in INTERVENTION_NAMES:
        raise ValueError(f"Unknown D-MHW intervention {name}")
    output = dict(inputs)
    batch_size = inputs["current_cube_drae"].shape[0]
    if name in ("wrong_history", "cube_mismatch") and batch_size < 2:
        raise ValueError(f"{name} requires batch size at least two")
    if name == "wrong_history":
        for key in (
            "history_cube_drae",
            "observed_xyz_m",
            "observed_doppler_probability",
            "observed_confidence",
            "observed_source_id",
        ):
            output[key] = torch.roll(inputs[key], shifts=1, dims=0)
    elif name == "zero_history":
        output["history_cube_drae"] = torch.zeros_like(
            inputs["history_cube_drae"]
        )
    elif name == "doppler_sign_flip":
        indices = _doppler_sign_indices(
            doppler_mps.to(inputs["observed_doppler_probability"])
        )
        output["observed_doppler_probability"] = inputs[
            "observed_doppler_probability"
        ][..., indices]
    elif name == "time_disorder":
        output["history_cube_drae"] = torch.flip(
            inputs["history_cube_drae"], dims=(1,)
        )
    elif name == "cube_mismatch":
        output["current_cube_drae"] = torch.roll(
            inputs["current_cube_drae"], shifts=1, dims=0
        )
    return output
