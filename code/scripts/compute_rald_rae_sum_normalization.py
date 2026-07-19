#!/usr/bin/env python3
"""Compute train-only RAE-Sum normalization for the matched RaLD baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_tesseract  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def selected_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    return np.linspace(0, length - 1, limit).round().astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partitions", nargs="+", default=["train"])
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--sample-per-frame", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpu-name")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("RAE-Sum normalization requires CUDA")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    if args.required_gpu_name and device_name != args.required_gpu_name:
        raise RuntimeError(
            f"RAE-Sum normalization requires {args.required_gpu_name}, got {device_name}"
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest["frames"]
        if record["partition"] in set(args.partitions)
    ]
    records = [
        records[index] for index in selected_indices(len(records), args.frame_limit)
    ]
    if not records:
        raise ValueError(f"No records found for partitions {args.partitions}")
    source_commit = args.source_commit or git_commit(Path(__file__).resolve().parents[2])
    if source_commit is None:
        raise RuntimeError("Source commit is required for reproducibility")

    total = 0.0
    total_square = 0.0
    voxel_count = 0
    zero_count = 0
    input_negative_count = 0
    input_value_count = 0
    minimum = float("inf")
    maximum = float("-inf")
    samples = []
    processed = []
    for position, record in enumerate(records, start=1):
        sequence = int(record["sequence"])
        radar_index = int(record["radar_index"])
        cube_path = (
            args.data_root
            / str(sequence)
            / "radar_tesseract"
            / f"tesseract_{radar_index:05d}.mat"
        )
        cube = torch.from_numpy(
            load_tesseract(cube_path).astype(np.float32, copy=False)
        ).to(device, non_blocking=True)
        input_negative_count += int((cube < 0.0).sum().item())
        input_value_count += cube.numel()
        rae_sum = cube.clamp_min(0.0).sum(dim=0)
        zero_count += int((rae_sum == 0.0).sum().item())
        log_power = torch.log10(rae_sum + 1.0)
        total += float(log_power.sum(dtype=torch.float64).item())
        total_square += float(log_power.square().sum(dtype=torch.float64).item())
        voxel_count += log_power.numel()
        minimum = min(minimum, float(log_power.amin().item()))
        maximum = max(maximum, float(log_power.amax().item()))
        sample_count = min(args.sample_per_frame, log_power.numel())
        generator = np.random.default_rng(
            args.seed + sequence * 1_000_003 + radar_index
        )
        sample_index = generator.choice(
            log_power.numel(), size=sample_count, replace=False
        )
        sample_index = torch.from_numpy(sample_index).to(device)
        samples.append(log_power.flatten()[sample_index].cpu().numpy())
        processed.append({"sequence": sequence, "radar_index": radar_index})
        print(
            json.dumps(
                {
                    "completed": position,
                    "total": len(records),
                    "sequence": sequence,
                    "radar_index": radar_index,
                }
            ),
            flush=True,
        )
        del cube, rae_sum, log_power, sample_index

    mean = total / voxel_count
    variance = max(total_square / voxel_count - mean * mean, 0.0)
    std = variance**0.5
    sampled = np.concatenate(samples)
    quantile_levels = (0.01, 0.05, 0.5, 0.95, 0.99)
    quantiles = np.quantile(sampled, quantile_levels)
    torch.cuda.synchronize(device)
    result = {
        "schema_version": 1,
        "representation": "log10_sum_doppler_power_plus_one",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "manifest_sha256": sha256(args.manifest),
        "scene_split_sha256": sha256(args.scene_split),
        "partitions": args.partitions,
        "frame_limit": args.frame_limit,
        "frame_count": len(records),
        "frames": processed,
        "device": device_name,
        "rae_sum_log_power": {
            "voxel_count": voxel_count,
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "zero_fraction": zero_count / voxel_count,
            "input_negative_fraction": input_negative_count
            / max(input_value_count, 1),
            "sample_count": int(sampled.size),
            "sample_quantiles": {
                str(level): float(value)
                for level, value in zip(quantile_levels, quantiles, strict=True)
            },
        },
        "normalization": {
            "center": mean,
            "scale": max(std, 1e-6),
            "clip": [-4.0, 4.0],
        },
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
