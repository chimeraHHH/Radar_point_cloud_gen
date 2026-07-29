#!/usr/bin/env python3
"""CPU-only, read-only audit for the R-A2 range/class sampler.

The audit deliberately does not instantiate a model or touch CUDA. It checks
the frozen manifest/cache inputs, replays one deterministic first-epoch sample
for every train frame, and records objective/evaluator mismatches that cannot be
resolved by running the current pilot for longer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROTOCOL = "g1_ra2_range_sampler_readonly_audit_v1"
FROZEN_MANIFEST_SHA256 = (
    "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
)
FROZEN_SCENE_SPLIT_SHA256 = (
    "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
)
FROZEN_NORMALIZATION_SHA256 = (
    "4d0bca7d027a1a9f457c526f21a034a406ce4973e2dccee55ddd2d7019b41b77"
)
FROZEN_EXACT_EVALUATOR_SHA256 = (
    "07dc5885d47effc6c0d41b7f2fc8f7f41c59f89a4ab1f2b73b2de44f667b8962"
)
FROZEN_PILOT_LOSS_SHA256 = (
    "6493148a4a2d6ff5a0e148b70a8922e1d48f3a87363fc11173eaef11a3cb43b8"
)
FROZEN_PILOT_TRAIN_SHA256 = (
    "caca90e03c8283255b1fa55a30979942cfabf08330cfb5f542696acc98ed68b8"
)
RANGE_BOUNDS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
EXPECTED_PARTITION_COUNTS = {"train": 76, "validation": 24}
EXPECTED_CACHE_SCHEMA_VERSION = 1
REQUIRED_TARGET_ARRAYS = (
    "target_xyz_confidence",
    "target_rae_index",
)
REQUIRED_CACHE_METADATA_ARRAYS = (
    "cache_schema_version",
    "source_manifest_sha256",
    "source_commit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def range_codes(values_m: np.ndarray) -> np.ndarray:
    values = np.asarray(values_m, dtype=np.float64)
    codes = np.full(values.shape, -1, dtype=np.int64)
    for code, (lower, upper) in enumerate(RANGE_BOUNDS_M):
        codes[(values >= lower) & (values < upper)] = code
    return codes


def nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.all(axis[1:] > axis[:-1]):
        raise ValueError("Audit axes must be strictly increasing vectors")
    right = np.searchsorted(axis, values, side="left")
    right = np.clip(right, 0, axis.size - 1)
    left = np.clip(right - 1, 0, axis.size - 1)
    choose_left = np.abs(values - axis[left]) <= np.abs(axis[right] - values)
    return np.where(choose_left, left, right).astype(np.int64)


def continuous_axis_values(axis: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(bins, dtype=np.float64),
        np.arange(len(axis), dtype=np.float64),
        np.asarray(axis, dtype=np.float64),
    )


def rae_to_xyz(
    range_m: np.ndarray,
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray,
) -> np.ndarray:
    horizontal = range_m * np.cos(elevation_rad)
    return np.stack(
        (
            horizontal * np.cos(azimuth_rad),
            horizontal * np.sin(azimuth_rad),
            range_m * np.sin(elevation_rad),
        ),
        axis=1,
    )


def xyz_to_rae_indices(
    xyz: np.ndarray,
    range_m: np.ndarray,
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray,
) -> np.ndarray:
    # The cache builder quantizes in Torch float32. Reproduce that precision so
    # exact half-bin ties use the same left-bin rule after cache serialization.
    xyz = np.asarray(xyz, dtype=np.float32)
    range_m = np.asarray(range_m, dtype=np.float32)
    azimuth_rad = np.asarray(azimuth_rad, dtype=np.float32)
    elevation_rad = np.asarray(elevation_rad, dtype=np.float32)
    radius = np.linalg.norm(xyz, axis=1)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    elevation = np.arcsin(
        np.clip(
            xyz[:, 2] / np.maximum(radius, np.finfo(np.float64).tiny),
            -1.0,
            1.0,
        )
    )
    return np.stack(
        (
            nearest_axis_indices(range_m, radius),
            nearest_axis_indices(azimuth_rad, azimuth),
            nearest_axis_indices(elevation_rad, elevation),
        ),
        axis=1,
    )


def xyz_index_consistency(
    xyz: np.ndarray,
    stored_indices: np.ndarray,
    range_m: np.ndarray,
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray,
) -> np.ndarray:
    """Accept stored nearest bins, including float32 half-bin tie roundoff."""

    xyz64 = np.asarray(xyz, dtype=np.float64)
    stored = np.asarray(stored_indices, dtype=np.int64)
    radius = np.linalg.norm(xyz64, axis=1)
    azimuth = np.arctan2(xyz64[:, 1], xyz64[:, 0])
    elevation = np.arcsin(
        np.clip(
            xyz64[:, 2]
            / np.maximum(radius, np.finfo(np.float64).tiny),
            -1.0,
            1.0,
        )
    )
    valid = np.ones(xyz64.shape[0], dtype=bool)
    float32_epsilon = np.finfo(np.float32).eps
    for axis, values, column in (
        (range_m, radius, 0),
        (azimuth_rad, azimuth, 1),
        (elevation_rad, elevation, 2),
    ):
        axis64 = np.asarray(axis, dtype=np.float64)
        nearest = nearest_axis_indices(axis64, values)
        stored_distance = np.abs(values - axis64[stored[:, column]])
        nearest_distance = np.abs(values - axis64[nearest])
        tolerance = (
            32.0
            * float32_epsilon
            * np.maximum.reduce(
                (
                    np.ones_like(values),
                    np.abs(values),
                    np.abs(axis64[stored[:, column]]),
                )
            )
        )
        valid &= stored_distance <= nearest_distance + tolerance
    return valid


def cache_path(cache_root: Path, sequence: int, radar_index: int) -> Path:
    return cache_root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


def _scalar_text(value: np.ndarray) -> str:
    scalar = np.asarray(value).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def validate_partition_contract(
    manifest: dict[str, Any],
    scene_split: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        return [], ["manifest.frames is missing or is not a list"]
    counts = Counter(str(frame.get("partition")) for frame in frames)
    if dict(counts) != EXPECTED_PARTITION_COUNTS:
        failures.append(
            f"manifest partition counts {dict(counts)} != {EXPECTED_PARTITION_COUNTS}"
        )
    identities = [
        (int(frame["sequence"]), int(frame["radar_index"])) for frame in frames
    ]
    if len(set(identities)) != len(identities):
        failures.append("manifest contains duplicate sequence/radar identities")

    split_sets = {
        partition: {
            int(sequence)
            for sequence in scene_split.get("splits", {})
            .get(partition, {})
            .get("sequences", [])
        }
        for partition in ("train", "validation", "test")
    }
    if scene_split.get("gate_pass") is not True:
        failures.append("scene split gate_pass is not true")
    if split_sets["train"] & split_sets["validation"]:
        failures.append("train and validation scene sets overlap")
    if (split_sets["train"] | split_sets["validation"]) & split_sets["test"]:
        failures.append("development and test scene sets overlap")
    for frame in frames:
        partition = str(frame.get("partition"))
        sequence = int(frame["sequence"])
        if partition not in EXPECTED_PARTITION_COUNTS:
            failures.append(f"forbidden manifest partition {partition}")
        elif sequence not in split_sets[partition]:
            failures.append(
                f"seq{sequence:02d} is labelled {partition} outside that scene split"
            )
        if sequence in split_sets["test"]:
            failures.append(f"test sequence seq{sequence:02d} entered the manifest")
    return frames, sorted(set(failures))


def audit_cache_frame(
    record: dict[str, Any],
    *,
    cache_root: Path,
    manifest_sha256: str,
    axes: Any,
) -> tuple[dict[str, Any], list[str]]:
    sequence = int(record["sequence"])
    radar_index = int(record["radar_index"])
    identity = f"seq{sequence:02d}/radar{radar_index:05d}"
    path = cache_path(cache_root, sequence, radar_index)
    failures: list[str] = []
    if not path.is_file():
        return {"identity": identity, "path": str(path)}, [f"{identity}: cache missing"]

    try:
        with np.load(path, allow_pickle=False) as cache:
            missing_target = sorted(
                set(REQUIRED_TARGET_ARRAYS) - set(cache.files)
            )
            if missing_target:
                return (
                    {
                        "identity": identity,
                        "path": str(path),
                        "missing_target_arrays": missing_target,
                        "sampler_input_valid": False,
                    },
                    [f"{identity}: missing target arrays {missing_target}"],
                )
            missing_metadata = sorted(
                set(REQUIRED_CACHE_METADATA_ARRAYS) - set(cache.files)
            )
            schema = (
                int(cache["cache_schema_version"])
                if "cache_schema_version" in cache.files
                else None
            )
            source_manifest = (
                _scalar_text(cache["source_manifest_sha256"])
                if "source_manifest_sha256" in cache.files
                else None
            )
            source_commit = (
                _scalar_text(cache["source_commit"])
                if "source_commit" in cache.files
                else None
            )
            target = np.asarray(cache["target_xyz_confidence"], dtype=np.float32)
            target_index = np.asarray(cache["target_rae_index"], dtype=np.int64)
    except Exception as error:
        return (
            {"identity": identity, "path": str(path), "load_error": repr(error)},
            [f"{identity}: cache load failed: {error}"],
        )

    spatial_shape = (
        len(axes.range_m),
        len(axes.azimuth_rad),
        len(axes.elevation_rad),
    )
    if missing_metadata:
        failures.append(
            f"{identity}: missing cache provenance arrays {missing_metadata}"
        )
    if schema is not None and schema != EXPECTED_CACHE_SCHEMA_VERSION:
        failures.append(f"{identity}: cache schema {schema} is not 1")
    if source_manifest is not None and source_manifest != manifest_sha256:
        failures.append(f"{identity}: cache source manifest hash differs")
    sampler_input_valid = True
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        failures.append(f"{identity}: target_xyz_confidence shape is {target.shape}")
        sampler_input_valid = False
    if target_index.shape != (target.shape[0], 3):
        failures.append(f"{identity}: target_rae_index shape is {target_index.shape}")
        sampler_input_valid = False
    if not np.isfinite(target).all():
        failures.append(f"{identity}: target contains non-finite values")
        sampler_input_valid = False
    if target.shape[0] and (
        np.any(target[:, 3] < 0.0) or np.any(target[:, 3] > 1.0)
    ):
        failures.append(f"{identity}: target confidence is outside [0,1]")
    maximum = np.asarray(spatial_shape, dtype=np.int64) - 1
    if target_index.size and (
        np.any(target_index < 0) or np.any(target_index > maximum)
    ):
        failures.append(f"{identity}: target RAE index is outside the Cube")
        sampler_input_valid = False

    reconstruction_mismatch = -1
    strict_reconstruction_mismatch = -1
    index_codes = np.empty(0, dtype=np.int64)
    xyz_codes = np.empty(0, dtype=np.int64)
    if sampler_input_valid:
        reconstructed = xyz_to_rae_indices(
            target[:, :3],
            np.asarray(axes.range_m),
            np.asarray(axes.azimuth_rad),
            np.asarray(axes.elevation_rad),
        )
        strict_reconstruction_mismatch = int(
            np.any(reconstructed != target_index, axis=1).sum()
        )
        consistent = xyz_index_consistency(
            target[:, :3],
            target_index,
            np.asarray(axes.range_m),
            np.asarray(axes.azimuth_rad),
            np.asarray(axes.elevation_rad),
        )
        reconstruction_mismatch = int((~consistent).sum())
        if reconstruction_mismatch:
            failures.append(
                f"{identity}: {reconstruction_mismatch} target RAE rows "
                "disagree with XYZ"
            )
        index_codes = range_codes(np.asarray(axes.range_m)[target_index[:, 0]])
        xyz_codes = range_codes(np.linalg.norm(target[:, :3], axis=1))
        if np.any(index_codes < 0) or np.any(xyz_codes < 0):
            failures.append(f"{identity}: target lies outside frozen 0-120 m strata")
            sampler_input_valid = False

    unique_indices = (
        np.unique(target_index, axis=0)
        if target_index.ndim == 2 and target_index.shape[1:] == (3,)
        else np.empty((0, 3), dtype=np.int64)
    )
    unique_codes = (
        range_codes(np.asarray(axes.range_m)[unique_indices[:, 0]])
        if unique_indices.size
        else np.empty(0, dtype=np.int64)
    )
    point_counts = [int(np.sum(index_codes == code)) for code in range(3)]
    unique_counts = [int(np.sum(unique_codes == code)) for code in range(3)]
    if str(record.get("partition")) == "train":
        for code, count in enumerate(unique_counts):
            if count == 0:
                failures.append(
                    f"{identity}: train frame has no unique positive in "
                    f"range class {code}"
                )
    report = {
        "identity": identity,
        "partition": str(record.get("partition")),
        "path": str(path),
        "cache_sha256": sha256_file(path),
        "cache_schema_version": schema,
        "source_manifest_sha256": source_manifest,
        "source_commit": source_commit,
        "missing_cache_metadata_arrays": missing_metadata,
        "sampler_input_valid": sampler_input_valid,
        "target_point_count": int(target.shape[0]),
        "unique_target_cell_count": int(unique_indices.shape[0]),
        "target_point_count_by_index_range": point_counts,
        "unique_target_cell_count_by_index_range": unique_counts,
        "xyz_vs_index_range_disagreement_count": int(
            np.sum(index_codes != xyz_codes)
        ),
        "xyz_to_index_reconstruction_mismatch_count": reconstruction_mismatch,
        "xyz_to_index_strict_nearest_mismatch_count": (
            strict_reconstruction_mismatch
        ),
    }
    return report, failures


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p05": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def audit_first_epoch_sampler(
    train_records: list[dict[str, Any]],
    cache_reports: list[dict[str, Any]],
    *,
    cache_root: Path,
    axes: Any,
) -> tuple[dict[str, Any], list[str]]:
    import torch

    from losses.rald_wce_pilot import (
        RANGE_NEGATIVE_GLOBAL_PER_CLASS,
        RANGE_NEGATIVE_SHELL_PER_CLASS,
        RANGE_POSITIVE_QUOTAS,
        sample_range_class_queries,
    )
    from scripts.train_rald_wce_pilot import (
        FORMAL_SEED,
        deterministic_frame_order,
        frame_seed,
    )

    failures: list[str] = []
    if len(train_records) != 76:
        return {}, [
            f"sampler audit requires 76 train records, got {len(train_records)}"
        ]
    order = deterministic_frame_order(
        list(range(len(train_records))),
        seed=FORMAL_SEED,
        cycle=1,
    )
    update_for_index = {index: update for update, index in enumerate(order)}
    cache_by_identity = {row["identity"]: row for row in cache_reports}
    range_m = torch.as_tensor(axes.range_m, dtype=torch.float32)
    spatial_shape = (
        len(axes.range_m),
        len(axes.azimuth_rad),
        len(axes.elevation_rad),
    )
    frame_reports: list[dict[str, Any]] = []
    shell_metric_distances: list[float] = []
    global_metric_distances: list[float] = []
    total_crossings = 0
    total_shell_within_1m = 0
    total_global_within_1m = 0
    sampler_failure_count = 0

    for index, record in enumerate(train_records):
        sequence = int(record["sequence"])
        radar_index = int(record["radar_index"])
        identity = f"seq{sequence:02d}/radar{radar_index:05d}"
        path = cache_path(cache_root, sequence, radar_index)
        if identity not in cache_by_identity or not path.is_file():
            failures.append(f"{identity}: cannot replay sampler without valid cache")
            continue
        with np.load(path, allow_pickle=False) as cache:
            target_index_np = np.asarray(cache["target_rae_index"], dtype=np.int64)
            target_xyz = np.asarray(
                cache["target_xyz_confidence"], dtype=np.float32
            )[:, :3]
        target_index = torch.from_numpy(target_index_np)
        update = update_for_index[index]
        seed = frame_seed(FORMAL_SEED, record, update + 1)
        try:
            first = sample_range_class_queries(
                target_index,
                spatial_shape=spatial_shape,
                range_m=range_m,
                seed=seed,
            )
            second = sample_range_class_queries(
                target_index,
                spatial_shape=spatial_shape,
                range_m=range_m,
                seed=seed,
            )
        except Exception as error:
            failures.append(f"{identity}: sampler failed: {error}")
            sampler_failure_count += 1
            continue

        tensor_keys = (
            "normalized_rae",
            "occupancy_target",
            "residual_target_bins",
            "range_class",
            "sample_source",
            "cell_indices",
            "fractional_offset",
        )
        deterministic = all(torch.equal(first[key], second[key]) for key in tensor_keys)
        if not deterministic:
            failures.append(
                f"{identity}: repeated CPU sampler output is not deterministic"
            )

        labels = first["occupancy_target"][0].numpy() > 0.5
        classes = first["range_class"][0].numpy().astype(np.int64)
        sources = first["sample_source"][0].numpy().astype(np.int64)
        cells = first["cell_indices"][0].numpy().astype(np.int64)
        fractional = first["fractional_offset"][0].numpy().astype(np.float64)
        unique_target = np.unique(target_index_np, axis=0)

        quota_ok = True
        for code, positive_quota in enumerate(RANGE_POSITIVE_QUOTAS):
            observed = (
                int(np.sum(labels & (classes == code))),
                int(np.sum((sources == 1) & (classes == code))),
                int(np.sum((sources == 2) & (classes == code))),
            )
            expected = (
                positive_quota,
                RANGE_NEGATIVE_GLOBAL_PER_CLASS,
                RANGE_NEGATIVE_SHELL_PER_CLASS,
            )
            if observed != expected:
                quota_ok = False
                failures.append(
                    f"{identity}: range {code} quotas {observed} != {expected}"
                )

        negative = ~labels
        nearest_index_distance = cKDTree(unique_target).query(
            cells[negative],
            k=1,
            p=np.inf,
            workers=1,
        )[0]
        ambiguity_violations = int(np.sum(nearest_index_distance <= 1.0))
        if ambiguity_violations:
            failures.append(
                f"{identity}: {ambiguity_violations} negatives enter "
                "3x3x3 ambiguity band"
            )
        shell_mask = sources == 2
        shell_index_distance = cKDTree(unique_target).query(
            cells[shell_mask],
            k=1,
            p=np.inf,
            workers=1,
        )[0]
        shell_definition_violations = int(
            np.sum((shell_index_distance < 2.0) | (shell_index_distance > 4.0))
        )
        if shell_definition_violations:
            failures.append(
                f"{identity}: {shell_definition_violations} shell negatives "
                "are outside Chebyshev radius 2-4"
            )

        query_bins = cells.astype(np.float64) + fractional
        query_range = continuous_axis_values(
            np.asarray(axes.range_m), query_bins[:, 0]
        )
        actual_classes = range_codes(query_range)
        crossing_count = int(np.sum(actual_classes != classes))
        total_crossings += crossing_count
        if crossing_count:
            failures.append(
                f"{identity}: {crossing_count} jittered queries cross their range class"
            )

        query_azimuth = continuous_axis_values(
            np.asarray(axes.azimuth_rad), query_bins[:, 1]
        )
        query_elevation = continuous_axis_values(
            np.asarray(axes.elevation_rad), query_bins[:, 2]
        )
        query_xyz = rae_to_xyz(query_range, query_azimuth, query_elevation)
        target_tree = cKDTree(target_xyz)
        shell_distance = target_tree.query(
            query_xyz[shell_mask], k=1, workers=1
        )[0]
        global_mask = sources == 1
        global_distance = target_tree.query(
            query_xyz[global_mask], k=1, workers=1
        )[0]
        shell_metric_distances.extend(shell_distance.tolist())
        global_metric_distances.extend(global_distance.tolist())
        shell_within_1m = int(np.sum(shell_distance <= 1.0))
        global_within_1m = int(np.sum(global_distance <= 1.0))
        total_shell_within_1m += shell_within_1m
        total_global_within_1m += global_within_1m

        source_unique_fraction = {}
        for source_code, source_name in ((0, "positive"), (1, "global"), (2, "shell")):
            source_cells = cells[sources == source_code]
            source_unique_fraction[source_name] = (
                float(np.unique(source_cells, axis=0).shape[0] / source_cells.shape[0])
                if source_cells.shape[0]
                else 0.0
            )
        frame_reports.append(
            {
                "identity": identity,
                "first_epoch_update": update + 1,
                "seed": seed,
                "deterministic_replay": deterministic,
                "quota_ok": quota_ok,
                "ambiguity_band_violation_count": ambiguity_violations,
                "shell_definition_violation_count": shell_definition_violations,
                "jittered_range_class_crossing_count": crossing_count,
                "shell_negative_within_1m_of_target_count": shell_within_1m,
                "global_negative_within_1m_of_target_count": global_within_1m,
                "unique_fraction_by_source": source_unique_fraction,
            }
        )

    query_count = len(frame_reports) * 16_000
    shell_count = len(shell_metric_distances)
    global_count = len(global_metric_distances)
    return {
        "attempted_frame_count": len(train_records),
        "frame_count": len(frame_reports),
        "failed_frame_count": sampler_failure_count,
        "all_frames_deterministic": (
            len(frame_reports) == len(train_records)
            and all(frame["deterministic_replay"] for frame in frame_reports)
        ),
        "all_successful_frames_deterministic": all(
            frame["deterministic_replay"] for frame in frame_reports
        ),
        "all_frozen_quotas_exact": (
            len(frame_reports) == len(train_records)
            and all(frame["quota_ok"] for frame in frame_reports)
        ),
        "all_successful_frame_quotas_exact": all(
            frame["quota_ok"] for frame in frame_reports
        ),
        "ambiguity_band_violation_count": sum(
            frame["ambiguity_band_violation_count"] for frame in frame_reports
        ),
        "shell_definition_violation_count": sum(
            frame["shell_definition_violation_count"] for frame in frame_reports
        ),
        "jittered_range_class_crossing_count": total_crossings,
        "jittered_range_class_crossing_fraction": (
            total_crossings / query_count if query_count else None
        ),
        "shell_negative_within_1m_of_target_count": total_shell_within_1m,
        "shell_negative_within_1m_of_target_fraction": (
            total_shell_within_1m / shell_count if shell_count else None
        ),
        "global_negative_within_1m_of_target_count": total_global_within_1m,
        "global_negative_within_1m_of_target_fraction": (
            total_global_within_1m / global_count if global_count else None
        ),
        "shell_negative_metric_distance_m": _percentiles(shell_metric_distances),
        "global_negative_metric_distance_m": _percentiles(global_metric_distances),
        "frames": frame_reports,
    }, sorted(set(failures))


def _normalized_source_hashes(values: dict[str, str]) -> dict[str, str]:
    normalized = {}
    for path, digest in values.items():
        marker = "/code/"
        key = path[path.index(marker) + 1 :] if marker in path else Path(path).name
        normalized[key] = digest
    return normalized


def audit_pilot_manifests(paths: list[Path]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    reports = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        contract = document.get("resume_contract", {})
        reports.append(
            {
                "path": str(path),
                "protocol": document.get("protocol"),
                "source_commit": contract.get("source_commit"),
                "mode": contract.get("config", {}).get("mode"),
                "input_hashes": contract.get("input_hashes"),
                "source_hashes": _normalized_source_hashes(
                    contract.get("source_hashes", {})
                ),
                "cache_root_bound_in_resume_contract": "cache_root" in contract,
                "data_root_bound_in_resume_contract": "data_root" in contract,
                "device_identity_bound_in_resume_contract": (
                    "device_identity" in contract
                ),
            }
        )
    if len(reports) > 1:
        reference = reports[0]
        for report in reports[1:]:
            for key in ("source_commit", "input_hashes", "source_hashes"):
                if report[key] != reference[key]:
                    failures.append(
                        f"pilot manifests are not comparable: {key} differs"
                    )
    return {
        "run_count": len(reports),
        "runs": reports,
        "query_budget_confound": (
            "source_classwise uses 10k queries/update while range_class_sampler "
            "uses 16k queries/update; equal update counts are not equal query budgets"
        ),
    }, sorted(set(failures))


def audit_run_directories(paths: list[Path]) -> tuple[dict[str, Any], list[str]]:
    import torch

    failures: list[str] = []
    reports = []
    for path in paths:
        manifest_path = path / "run_manifest.json"
        checkpoint_path = path / "last.pt"
        progress_path = path / "progress.json"
        missing = [
            str(candidate)
            for candidate in (manifest_path, checkpoint_path, progress_path)
            if not candidate.is_file()
        ]
        if missing:
            failures.append(f"run directory {path} misses {missing}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_state = checkpoint.get("state")
        progress_state = progress.get("state")
        state_identical = checkpoint_state == progress_state
        if not state_identical:
            failures.append(f"run directory {path} progress/checkpoint state differs")
        contract_hash_match = (
            checkpoint.get("resume_contract_sha256")
            == manifest.get("resume_contract_sha256")
        )
        if not contract_hash_match:
            failures.append(f"run directory {path} checkpoint contract hash differs")
        rng = checkpoint.get("rng", {})
        rng_complete = all(
            key in rng for key in ("python", "numpy", "torch_cpu", "torch_cuda")
        )
        if not rng_complete:
            failures.append(f"run directory {path} checkpoint RNG state is incomplete")

        formal_reference_path = path / "formal_reference.json"
        formal_reference_internal_match = None
        if formal_reference_path.is_file():
            from scripts.train_rald_wce_pilot import parse_formal_metric_values

            formal_reference = json.loads(
                formal_reference_path.read_text(encoding="utf-8")
            )
            parsed = parse_formal_metric_values(formal_reference["metrics"])
            formal_reference_internal_match = all(
                abs(float(formal_reference["values"][key]) - expected) <= 1e-8
                for key, expected in parsed.items()
            )
            if not formal_reference_internal_match:
                failures.append(
                    f"run directory {path} formal_reference values/metrics differ"
                )
        reports.append(
            {
                "path": str(path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_progress_state_identical": state_identical,
                "checkpoint_contract_hash_match": contract_hash_match,
                "checkpoint_rng_complete": rng_complete,
                "formal_reference_internal_match": (
                    formal_reference_internal_match
                ),
            }
        )
    return {"run_count": len(reports), "runs": reports}, sorted(set(failures))


def static_findings() -> list[dict[str, Any]]:
    return [
        {
            "severity": "high",
            "id": "pointwise_loss_vs_exact10k_ranking",
            "finding": (
                "Training receives binary occupied/empty labels and never receives "
                "target_xyz_confidence, but exact export sorts roughly 700k continuous "
                "candidates by sigmoid occupancy confidence within each range."
            ),
            "effect": (
                "Low pointwise BCE does not guarantee correct within-range top-10k "
                "ordering or calibration under the 5 cm capacity filter."
            ),
            "minimum_patch": (
                "Add a ranking-aware calibration term on a frozen candidate subset, "
                "or train confidence against target distance/importance while "
                "retaining the existing occupancy term."
            ),
        },
        {
            "severity": "high",
            "id": "cache_and_cube_not_resume_bound",
            "finding": (
                "resume_contract binds manifest/split/normalization/source hashes but "
                "does not bind data_root, cache_root, per-cache digests, or Cube files."
            ),
            "effect": (
                "A resumed run or two pilot modes can silently consume different data "
                "while passing the current resume and comparison checks."
            ),
            "minimum_patch": (
                "Bind an ordered cache digest, cache metadata source commit, data-root "
                "identity, and preferably an ordered Cube digest/index into the run "
                "manifest and resume contract."
            ),
        },
        {
            "severity": "high",
            "id": "formal_reference_resume_not_integrity_bound",
            "finding": (
                "On resume, formal_reference.json is trusted after checking only the "
                "baseline input hashes; its own digest and metric/value consistency "
                "are not part of the checkpoint contract."
            ),
            "effect": (
                "A modified reference can change epoch-3/5 decisions after resume."
            ),
            "minimum_patch": (
                "Recompute/verify formal reference values on resume and bind the "
                "reference JSON digest to the checkpoint."
            ),
        },
        {
            "severity": "medium",
            "id": "auxiliary_losses_not_range_normalized",
            "finding": (
                "Only occupancy BCE is normalized per range/class. Brier is a global "
                "mean over 15k negatives and 1k positives; residual Smooth L1 is a "
                "global mean over positives with a 700/200/100 range mix."
            ),
            "effect": (
                "Far-range confidence calibration and residual learning are not "
                "equally weighted even when the primary BCE is range balanced."
            ),
            "minimum_patch": (
                "Report auxiliary gradients by range; if material, normalize Brier and "
                "positive residual by range with weights frozen before comparison."
            ),
        },
        {
            "severity": "medium",
            "id": "unequal_query_budget",
            "finding": (
                "source_classwise uses 10k queries/update and range_class_sampler uses "
                "16k queries/update for the same 380 updates."
            ),
            "effect": (
                "A difference cannot be attributed solely to range normalization or "
                "hard-negative composition."
            ),
            "minimum_patch": (
                "Either match total decoded queries or describe R-A2 as a sampler "
                "bundle comparison rather than an isolated loss ablation."
            ),
        },
        {
            "severity": "medium",
            "id": "cuda_determinism_not_enforced",
            "finding": (
                "RNG states are checkpointed, but deterministic algorithms, SDPA "
                "backend, physical GPU identity, driver, and CUDA_VISIBLE_DEVICES "
                "are not frozen."
            ),
            "effect": (
                "Resume is seed-reproducible but not guaranteed bitwise deterministic."
            ),
            "minimum_patch": (
                "Freeze deterministic backend settings and runtime/device identity, "
                "then run an uninterrupted-versus-resumed checkpoint hash test."
            ),
        },
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot-run-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--pilot-run-dir",
        type=Path,
        action="append",
        default=[],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[2]
    from cube_dense.kradar import load_axes

    hashes = {
        "manifest_sha256": sha256_file(args.manifest),
        "scene_split_sha256": sha256_file(args.scene_split),
        "normalization_sha256": sha256_file(args.normalization),
        "exact_evaluator_sha256": sha256_file(
            repo / "code/eval/rald_wce_stage0.py"
        ),
        "pilot_loss_sha256": sha256_file(repo / "code/losses/rald_wce_pilot.py"),
        "pilot_train_sha256": sha256_file(
            repo / "code/scripts/train_rald_wce_pilot.py"
        ),
    }
    expected_hashes = {
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "scene_split_sha256": FROZEN_SCENE_SPLIT_SHA256,
        "normalization_sha256": FROZEN_NORMALIZATION_SHA256,
        "exact_evaluator_sha256": FROZEN_EXACT_EVALUATOR_SHA256,
        "pilot_loss_sha256": FROZEN_PILOT_LOSS_SHA256,
        "pilot_train_sha256": FROZEN_PILOT_TRAIN_SHA256,
    }
    failures = [
        f"{key} differs from frozen audit value"
        for key, expected in expected_hashes.items()
        if hashes[key] != expected
    ]

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    frames, partition_failures = validate_partition_contract(manifest, scene_split)
    failures.extend(partition_failures)
    axes = load_axes(args.data_root / "resources")
    cache_reports = []
    for record in frames:
        report, frame_failures = audit_cache_frame(
            record,
            cache_root=args.cache_root,
            manifest_sha256=hashes["manifest_sha256"],
            axes=axes,
        )
        cache_reports.append(report)
        failures.extend(frame_failures)

    source_commits = sorted(
        {
            report["source_commit"]
            for report in cache_reports
            if report.get("source_commit") is not None
        }
    )
    if not source_commits:
        failures.append("cache source commit is unavailable for all frames")
    elif len(source_commits) != 1:
        failures.append(f"cache source commits are mixed: {source_commits}")
    ordered_cache_digest = hashlib.sha256(
        "\n".join(
            f"{report.get('identity')} {report.get('cache_sha256')}"
            for report in cache_reports
        ).encode("utf-8")
    ).hexdigest()

    train_records = [
        record for record in frames if record.get("partition") == "train"
    ]
    sampler_report: dict[str, Any] = {}
    if all(report.get("sampler_input_valid") is True for report in cache_reports):
        sampler_report, sampler_failures = audit_first_epoch_sampler(
            train_records,
            cache_reports,
            cache_root=args.cache_root,
            axes=axes,
        )
        failures.extend(sampler_failures)
    else:
        failures.append(
            "sampler replay skipped because target cache arrays are "
            "structurally invalid"
        )

    manifest_report, manifest_failures = audit_pilot_manifests(
        args.pilot_run_manifest
    )
    failures.extend(manifest_failures)
    run_report, run_failures = audit_run_directories(args.pilot_run_dir)
    failures.extend(run_failures)
    failures = sorted(set(failures))
    import torch

    cuda_initialized = torch.cuda.is_initialized()
    if cuda_initialized:
        failures.append("CUDA was initialized during the CPU-only audit")
        failures = sorted(set(failures))
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "cpu_only": True,
        "cuda_initialized": cuda_initialized,
        "hashes": hashes,
        "expected_hashes": expected_hashes,
        "partition_counts": dict(
            Counter(str(frame.get("partition")) for frame in frames)
        ),
        "cache": {
            "frame_count": len(cache_reports),
            "source_commits": source_commits,
            "ordered_cache_digest_sha256": ordered_cache_digest,
            "frames": cache_reports,
        },
        "sampler": sampler_report,
        "pilot_manifest_comparison": manifest_report,
        "run_directory_integrity": run_report,
        "static_findings": static_findings(),
        "hard_failures": failures,
        "audit_passed": not failures,
        "scientific_decision_eligible": not failures,
    }
    atomic_json(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
