#!/usr/bin/env python3
"""Run the independent P-RF representation/model preflight on one H200."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_tesseract  # noqa: E402
from cube_dense.polar_flow_target import (  # noqa: E402
    P_R_F_POINT_COUNT,
    P_R_F_SOURCE_SEED,
    PolarLogitTransform,
    array_sha256,
    canonical_fixed_target,
    fit_empirical_range_cdf,
    fixed_scrambled_sobol_unit,
    hierarchical_one_to_one_transport,
    logit_unit,
)
from losses.polar_rectified_flow import polar_rectified_flow_loss  # noqa: E402
from models.cube_doppler import query_cube_spectrum  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from models.polar_rectified_flow import PolarRectifiedFlow  # noqa: E402


PROTOCOL = "p_rf_h200_representation_model_preflight_v1"
SOURCE_PATTERN = re.compile(r"[0-9a-f]{40}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("P-RF source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("P-RF source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"P-RF preflight source worktree is dirty: {dirty}")


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("P-RF preflight requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("P-RF preflight is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"P-RF preflight requires H200, got {resolved}")
    return device, resolved


def cache_path(cache_root: Path, record: dict) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_"
        f"radar_{int(record['radar_index']):05d}.npz"
    )


def cube_path(data_root: Path, record: dict) -> Path:
    return (
        data_root
        / str(int(record["sequence"]))
        / "radar_tesseract"
        / f"tesseract_{int(record['radar_index']):05d}.mat"
    )


def manifest_records(manifest_path: Path) -> list[dict]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in document["frames"]
        if record["partition"] in {"train", "validation"}
    ]
    if not records:
        raise ValueError("P-RF preflight found no train/validation records")
    forbidden = {
        str(record["partition"])
        for record in records
        if record["partition"] not in {"train", "validation"}
    }
    if forbidden:
        raise ValueError(f"P-RF preflight accessed forbidden partitions: {forbidden}")
    return records


def load_target(cache_root: Path, record: dict) -> np.ndarray:
    path = cache_path(cache_root, record)
    with np.load(path) as cache:
        target = cache["target_xyz_confidence"].astype(np.float32)
    if target.ndim != 2 or target.shape[1] != 4 or not np.isfinite(target).all():
        raise ValueError(f"Malformed dense target in {path}")
    return target


def load_target_rae_index(cache_root: Path, record: dict) -> np.ndarray:
    path = cache_path(cache_root, record)
    with np.load(path) as cache:
        target = cache["target_xyz_confidence"]
        target_index = cache["target_rae_index"].astype(np.int64)
    if target_index.shape != (target.shape[0], 3):
        raise ValueError(f"Malformed dense target indices in {path}")
    return target_index


def target_unique_count(target: np.ndarray) -> int:
    return int(np.unique(target[:, :3], axis=0).shape[0])


def select_preflight_frame(
    records: list[dict],
    cache_root: Path,
) -> tuple[int, list[dict]]:
    inventory = []
    for index, record in enumerate(records):
        target = load_target(cache_root, record)
        inventory.append(
            {
                "index": index,
                "sequence": int(record["sequence"]),
                "radar_index": int(record["radar_index"]),
                "partition": str(record["partition"]),
                "target_count": int(target.shape[0]),
                "unique_xyz_count": target_unique_count(target),
            }
        )
    selected = max(
        range(len(inventory)),
        key=lambda index: (
            inventory[index]["unique_xyz_count"],
            inventory[index]["target_count"],
            -index,
        ),
    )
    return selected, inventory


def select_wrong_frame(records: list[dict], selected_index: int) -> int:
    selected = records[selected_index]
    for index, record in enumerate(records):
        if index == selected_index:
            continue
        if int(record["sequence"]) != int(selected["sequence"]):
            return index
    if len(records) < 2:
        raise ValueError("Wrong-Cube sensitivity requires at least two frames")
    return 1 if selected_index == 0 else 0


def training_range_samples(
    records: list[dict],
    cache_root: Path,
) -> np.ndarray:
    ranges = []
    for record in records:
        if record["partition"] != "train":
            continue
        xyz = load_target(cache_root, record)[:, :3].astype(np.float64)
        ranges.append(np.linalg.norm(xyz, axis=1))
    if not ranges:
        raise ValueError("P-RF range CDF requires training targets")
    return np.concatenate(ranges)


def parse_normalization(path: Path) -> tuple[float, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    values = document.get("normalization", document)
    center = float(values["center"])
    scale = float(values["scale"])
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cube normalization must be finite with positive scale")
    return center, scale


def tensor_is_finite(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=2_048)
    parser.add_argument("--max-allocated-gb", type=float, default=55.0)
    parser.add_argument("--max-reserved-gb", type=float, default=65.0)
    parser.add_argument(
        "--minimum-wrong-cube-state-delta",
        type=float,
        default=1e-8,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"P-RF preflight output exists: {args.output}")
    if args.chunk_size <= 0:
        raise ValueError("P-RF chunk size must be positive")
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)
    records = manifest_records(args.manifest)
    selected_index, inventory = select_preflight_frame(
        records,
        args.cache_root,
    )
    selected_record = records[selected_index]
    wrong_index = select_wrong_frame(records, selected_index)
    wrong_record = records[wrong_index]
    selected_target = load_target(args.cache_root, selected_record)
    endpoint = canonical_fixed_target(
        selected_target,
        point_count=P_R_F_POINT_COUNT,
    )

    axes = load_axes(args.data_root / "resources")
    range_bounds_m = (
        float(np.min(axes.range_m)),
        float(np.max(axes.range_m)),
    )
    azimuth_bounds_rad = (
        float(np.min(axes.azimuth_rad)),
        float(np.max(axes.azimuth_rad)),
    )
    elevation_bounds_rad = (
        float(np.min(axes.elevation_rad)),
        float(np.max(axes.elevation_rad)),
    )
    range_knots, range_cdf = fit_empirical_range_cdf(
        training_range_samples(records, args.cache_root),
        lower_m=range_bounds_m[0],
        upper_m=range_bounds_m[1],
    )
    transform = PolarLogitTransform(
        range_knots,
        range_cdf,
        azimuth_bounds_rad=azimuth_bounds_rad,
        elevation_bounds_rad=elevation_bounds_rad,
    )
    source_unit = fixed_scrambled_sobol_unit(
        P_R_F_POINT_COUNT,
        seed=P_R_F_SOURCE_SEED,
        epsilon=transform.epsilon,
    )
    source_state_cpu = logit_unit(source_unit, transform.epsilon)
    target_state_cpu = transform.encode_xyz(endpoint.xyz_m)
    target_round_trip_max_error_m = float(
        (
            transform.decode_xyz(target_state_cpu) - endpoint.xyz_m
        ).abs().max().item()
    )
    selected_target_rae_index = torch.from_numpy(
        load_target_rae_index(args.cache_root, selected_record)
    )[endpoint.source_indices]
    transport_started = time.monotonic()
    transport = hierarchical_one_to_one_transport(
        source_state_cpu,
        target_state_cpu,
        block_size=40,
    )
    transport_seconds = time.monotonic() - transport_started

    log_center, log_scale = parse_normalization(args.normalization)
    torch.manual_seed(P_R_F_SOURCE_SEED)
    torch.cuda.manual_seed_all(P_R_F_SOURCE_SEED)
    model = PolarRectifiedFlow(
        range_knots_m=range_knots,
        range_cdf=range_cdf,
        azimuth_bounds_rad=azimuth_bounds_rad,
        elevation_bounds_rad=elevation_bounds_rad,
        doppler_axis_mps=axes.doppler_mps,
        log_center=log_center,
        log_scale=log_scale,
    ).to(device)
    cube = torch.from_numpy(
        load_tesseract(cube_path(args.data_root, selected_record)).astype(
            np.float32,
            copy=False,
        )
    )
    wrong_cube = torch.from_numpy(
        load_tesseract(cube_path(args.data_root, wrong_record)).astype(
            np.float32,
            copy=False,
        )
    )
    cube = cube.unsqueeze(0).to(device, non_blocking=True)
    wrong_cube = wrong_cube.unsqueeze(0).to(device, non_blocking=True)
    source_state = model.fixed_source_state(
        1,
        device=device,
        dtype=torch.float32,
    )
    target_state = target_state_cpu.unsqueeze(0).to(device)
    permutation = transport.source_to_target.to(device)
    target_confidence = endpoint.confidence.unsqueeze(0).to(device)
    with torch.no_grad():
        target_doppler_distribution = query_cube_spectrum(
            cube,
            selected_target_rae_index.to(device),
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    run_started = time.monotonic()
    model.train()
    training_cube = cube.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = polar_rectified_flow_loss(
            model,
            training_cube,
            source_state,
            target_state,
            permutation,
            time=torch.full((1,), 0.5, device=device),
            target_doppler_distribution=target_doppler_distribution,
            target_confidence=target_confidence,
            wrong_cube_drae=wrong_cube,
            chunk_size=args.chunk_size,
        )
    loss.total.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.monotonic() - run_started
    cube_gradient = training_cube.grad
    condition_gradients = [
        parameter.grad
        for parameter in model.condition_encoder.parameters()
        if parameter.grad is not None
    ]
    doppler_head_gradient = model.particle_decoder.doppler_head.weight.grad
    doppler_head_row_gradient = (
        doppler_head_gradient.detach().abs().sum(dim=1)
        if doppler_head_gradient is not None
        else torch.zeros(64, device=device)
    )
    per_channel_gradient = (
        cube_gradient.detach().abs().sum(dim=(0, 2, 3, 4))
        if cube_gradient is not None
        else torch.zeros(64, device=device)
    )

    model.eval()
    inference_started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        matched_output = model.sample(
            cube,
            chunk_size=args.chunk_size,
            nfe=4,
        )
        wrong_output = model.sample(
            wrong_cube,
            chunk_size=args.chunk_size,
            nfe=4,
        )
    torch.cuda.synchronize(device)
    inference_seconds = time.monotonic() - inference_started
    wrong_cube_state_delta = (
        matched_output["state"].float() - wrong_output["state"].float()
    ).abs().mean()
    source_after = model.fixed_source_state(
        1,
        device=device,
        dtype=torch.float32,
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    allocated_limit = int(args.max_allocated_gb * 1024**3)
    reserved_limit = int(args.max_reserved_gb * 1024**3)

    tensor_outputs = (
        matched_output["state"],
        matched_output["coordinates_rae"],
        matched_output["xyz_m"],
        matched_output["doppler_logits"],
        matched_output["doppler_probability"],
        matched_output["doppler_mps"],
        matched_output["confidence_logit"],
        matched_output["confidence"],
    )
    checks = {
        "exact_10000_output": (
            tuple(matched_output["xyz_m"].shape)
            == (1, P_R_F_POINT_COUNT, 3)
        ),
        "all_outputs_finite": all(tensor_is_finite(value) for value in tensor_outputs),
        "one_to_one_target_assignment": (
            torch.equal(
                torch.sort(transport.source_to_target).values,
                torch.arange(P_R_F_POINT_COUNT),
            )
        ),
        "polar_target_round_trip_within_2cm": (
            math.isfinite(target_round_trip_max_error_m)
            and target_round_trip_max_error_m <= 0.02
        ),
        "fixed_source_cube_independent": bool(
            torch.equal(source_state, source_after)
        ),
        "cube_gradient_finite_nonzero": (
            cube_gradient is not None
            and tensor_is_finite(cube_gradient)
            and bool(torch.count_nonzero(cube_gradient).item())
        ),
        "all_64_cube_channels_receive_gradient": (
            int(torch.count_nonzero(per_channel_gradient).item()) == 64
        ),
        "condition_parameter_gradients_finite_nonzero": (
            bool(condition_gradients)
            and all(tensor_is_finite(value) for value in condition_gradients)
            and any(bool(torch.count_nonzero(value).item()) for value in condition_gradients)
        ),
        "all_64_doppler_outputs_receive_gradient": (
            tensor_is_finite(doppler_head_row_gradient)
            and int(torch.count_nonzero(doppler_head_row_gradient).item()) == 64
        ),
        "wrong_cube_changes_prediction": (
            tensor_is_finite(wrong_cube_state_delta)
            and float(wrong_cube_state_delta.item())
            > args.minimum_wrong_cube_state_delta
        ),
        "nfe4_euler_without_postprocessing": (
            matched_output["integration"]
            == {"method": "euler", "nfe": 4, "postprocessing": []}
            and model.architecture_metadata()["inference_postprocessing"] == []
        ),
        "memory_within_preflight_budget": (
            peak_allocated <= allocated_limit
            and peak_reserved <= reserved_limit
        ),
        "no_test_partition_accessed": True,
    }
    source_files = (
        repo / "code/cube_dense/polar_flow_target.py",
        repo / "code/models/polar_rectified_flow.py",
        repo / "code/losses/polar_rectified_flow.py",
        Path(__file__).resolve(),
    )
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in source_files
        },
        "data": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "normalization": str(args.normalization.resolve()),
            "normalization_sha256": sha256(args.normalization),
            "frame_count": len(records),
            "train_frame_count": sum(
                record["partition"] == "train" for record in records
            ),
            "validation_frame_count": sum(
                record["partition"] == "validation" for record in records
            ),
            "selected_frame": inventory[selected_index],
            "selected_cache": str(
                cache_path(args.cache_root, selected_record).resolve()
            ),
            "selected_cache_sha256": sha256(
                cache_path(args.cache_root, selected_record)
            ),
            "wrong_cube_frame": inventory[wrong_index],
            "wrong_cube_path": str(
                cube_path(args.data_root, wrong_record).resolve()
            ),
            "range_cdf": {
                "knot_count": len(range_knots),
                "range_knots_sha256": array_sha256(range_knots),
                "range_cdf_sha256": array_sha256(range_cdf),
                "fit_partition": "train_only",
            },
        },
        "representation": {
            "point_count": P_R_F_POINT_COUNT,
            "source_seed": P_R_F_SOURCE_SEED,
            "source_unit_sha256": array_sha256(source_unit),
            "source_state_sha256": array_sha256(source_state_cpu),
            "target_hashes": endpoint.hashes,
            "target_round_trip_max_error_m": target_round_trip_max_error_m,
            "transport": {
                "block_size": transport.block_size,
                "total_squared_cost": transport.total_squared_cost,
                "hashes": transport.hashes,
                "elapsed_seconds": transport_seconds,
            },
        },
        "architecture": model.architecture_metadata(),
        "runtime": {
            "device": args.device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "model_parameter_count": parameter_count(model),
            "chunk_size": args.chunk_size,
            "backward_seconds": backward_seconds,
            "two_nfe4_inference_seconds": inference_seconds,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "max_allocated_bytes": allocated_limit,
            "max_reserved_bytes": reserved_limit,
        },
        "diagnostics": {
            "loss_total": float(loss.total.detach().item()),
            "loss_components": {
                key: float(value.item())
                for key, value in loss.components.items()
            },
            "cube_nonzero_gradient_channels": int(
                torch.count_nonzero(per_channel_gradient).item()
            ),
            "doppler_head_nonzero_gradient_rows": int(
                torch.count_nonzero(doppler_head_row_gradient).item()
            ),
            "wrong_cube_state_mean_absolute_delta": float(
                wrong_cube_state_delta.item()
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    atomic_json(args.output, document)
    print(json.dumps(document, indent=2), flush=True)
    if not document["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
