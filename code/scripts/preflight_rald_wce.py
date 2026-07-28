#!/usr/bin/env python3
"""Run the RaLD-WCE Stage-0 max-target H200 forward/backward preflight."""

from __future__ import annotations

import argparse
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

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from losses.rald_wce import (  # noqa: E402
    rald_wce_stage0_loss,
    sample_bounded_occupancy_queries,
)
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_wce_field import RaLDWCEField  # noqa: E402
from scripts.g1b_contract import sha256  # noqa: E402


PROTOCOL = "rald_wce_max_target_h200_preflight_v1"
FORMAL_SEED = 20260716
FORMAL_FRAME_COUNT = 100
QUERY_COUNT = 10_000
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def cache_path(cache_root: Path, record: dict) -> Path:
    return cache_root / (
        f"seq{int(record['sequence']):02d}_"
        f"radar_{int(record['radar_index']):05d}.npz"
    )


def target_count(cache_root: Path, record: dict) -> int:
    with np.load(cache_path(cache_root, record)) as cache:
        return int(cache["target_xyz_confidence"].shape[0])


def select_max_target_index(counts: list[int]) -> int:
    """Select the first largest positive target count deterministically."""

    if not counts or any(int(count) <= 0 for count in counts):
        raise ValueError("RaLD-WCE target counts must be non-empty and positive")
    return max(range(len(counts)), key=lambda index: (counts[index], -index))


def select_wrong_condition_index(
    records: list[dict],
    counts: list[int],
    matched_index: int,
) -> int:
    """Select the largest-target frame from another scene."""

    if len(records) != len(counts):
        raise ValueError("RaLD-WCE records and target counts must align")
    if matched_index < 0 or matched_index >= len(records):
        raise ValueError("RaLD-WCE matched frame index is out of bounds")
    matched_sequence = int(records[matched_index]["sequence"])
    candidates = [
        index
        for index, record in enumerate(records)
        if int(record["sequence"]) != matched_sequence
    ]
    if not candidates:
        raise ValueError("RaLD-WCE wrong condition requires another scene")
    return max(candidates, key=lambda index: (counts[index], -index))


def _flat_target_indices(
    target_rae_index: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    if target_rae_index.ndim != 2 or target_rae_index.shape[1] != 3:
        raise ValueError("RaLD-WCE target RAE indices must have shape (M,3)")
    if target_rae_index.shape[0] <= 0:
        raise ValueError("RaLD-WCE target RAE indices cannot be empty")
    target = target_rae_index.detach().to(device="cpu", dtype=torch.long)
    maximum = torch.tensor(spatial_shape, dtype=torch.long) - 1
    if bool(((target < 0) | (target > maximum)).any()):
        raise ValueError("RaLD-WCE target RAE index is outside the Cube")
    return (
        target[:, 0] * spatial_shape[1] * spatial_shape[2]
        + target[:, 1] * spatial_shape[2]
        + target[:, 2]
    )


def build_stage0_query_targets(
    target_rae_index: torch.Tensor,
    *,
    spatial_shape: tuple[int, int, int],
    query_count: int = QUERY_COUNT,
    seed: int = FORMAL_SEED,
) -> dict[str, torch.Tensor | int]:
    """Build deterministic positive-jitter and empty-cell Sobol queries."""

    return sample_bounded_occupancy_queries(
        target_rae_index,
        spatial_shape=spatial_shape,
        query_count=query_count,
        positive_query_ratio=0.5,
        seed=seed,
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
        raise ValueError("RaLD-WCE source commit must be a full lowercase Git SHA")
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("RaLD-WCE source commit does not match checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"RaLD-WCE source worktree is dirty: {dirty}")


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("RaLD-WCE preflight requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("RaLD-WCE preflight is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"RaLD-WCE preflight requires H200, got {resolved}")
    return device, resolved


def load_normalization(path: Path) -> tuple[float, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    values = document.get("normalization", document)
    center = float(values["center"])
    scale = float(values["scale"])
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("RaLD-WCE normalization is invalid")
    return center, scale


def gradient_check(module: torch.nn.Module) -> dict[str, bool | int]:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    finite = bool(
        gradients
        and all(bool(torch.isfinite(value).all()) for value in gradients)
    )
    nonzero = bool(
        gradients
        and any(bool(torch.count_nonzero(value)) for value in gradients)
    )
    return {
        "gradient_tensor_count": len(gradients),
        "finite": finite,
        "nonzero": nonzero,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-chunk-size", type=int, default=8_192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"RaLD-WCE preflight output exists: {args.output}")
    if args.decode_chunk_size <= 0:
        raise ValueError("RaLD-WCE decode chunk size must be positive")
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)
    log_center, log_scale = load_normalization(args.normalization)

    dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train", "validation"),
    )
    if len(dataset) != FORMAL_FRAME_COUNT:
        raise ValueError("RaLD-WCE preflight requires the frozen 100 frames")
    if any(record["partition"] == "test" for record in dataset.records):
        raise ValueError("RaLD-WCE preflight cannot access the test partition")
    counts = [
        target_count(args.cache_root, record)
        for record in dataset.records
    ]
    matched_index = select_max_target_index(counts)
    wrong_index = select_wrong_condition_index(
        dataset.records,
        counts,
        matched_index,
    )
    matched_item = dataset[matched_index]
    wrong_item = dataset[wrong_index]
    spatial_shape = tuple(int(size) for size in matched_item["cube_drae"].shape[1:])
    query_targets = build_stage0_query_targets(
        matched_item["target_rae_index"],
        spatial_shape=spatial_shape,
    )

    torch.manual_seed(FORMAL_SEED)
    torch.cuda.manual_seed_all(FORMAL_SEED)
    model = RaLDWCEField(
        log_center=log_center,
        log_scale=log_scale,
        spatial_shape=spatial_shape,
        depth=6,
        decode_chunk_size=args.decode_chunk_size,
    ).to(device).train()
    model.assert_condition_exclusive_contract()
    matched_cube = (
        matched_item["cube_drae"].unsqueeze(0).to(device).requires_grad_(True)
    )
    wrong_cube = (
        wrong_item["cube_drae"].unsqueeze(0).to(device).requires_grad_(True)
    )
    normalized_rae = query_targets["normalized_rae"].to(device)
    occupancy_target = query_targets["occupancy_target"].to(device)
    residual_target = query_targets["residual_target_bins"].to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            matched_cube,
            normalized_rae,
            wrong_condition_cube_drae=wrong_cube,
            chunk_size=args.decode_chunk_size,
        )
        loss = rald_wce_stage0_loss(
            output["matched"],
            output["wrong"],
            normalized_rae,
            occupancy_target,
            residual_target,
        )
    loss.total.backward()
    torch.cuda.synchronize(device)
    elapsed_seconds = time.monotonic() - started

    all_gradients = gradient_check(model)
    encoder_gradients = gradient_check(model.radar_encoder)
    decoder_gradients = gradient_check(model.decoder_attention)
    matched_cube_gradient = matched_cube.grad
    if matched_cube_gradient is None:
        doppler_channel_gradient = torch.zeros(
            64,
            dtype=torch.float32,
            device=device,
        )
    else:
        doppler_channel_gradient = matched_cube_gradient.float().abs().sum(
            dim=(0, 2, 3, 4)
        )
    doppler_gradient_report = {
        "channel_count": int(doppler_channel_gradient.numel()),
        "finite": bool(torch.isfinite(doppler_channel_gradient).all()),
        "nonzero_channel_count": int(
            torch.count_nonzero(doppler_channel_gradient).item()
        ),
        "minimum_channel_sum": float(
            doppler_channel_gradient.min().item()
        ),
        "maximum_channel_sum": float(
            doppler_channel_gradient.max().item()
        ),
    }
    alternate_chunk = max(1, args.decode_chunk_size // 2)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        alternate = model.decode_queries(
            normalized_rae,
            output["matched_condition_latents"].detach(),
            chunk_size=alternate_chunk,
        )
    chunk_deltas = {
        key: float(
            (
                output["matched"][key].detach().float()
                - alternate[key].float()
            )
            .abs()
            .max()
            .item()
        )
        for key in (
            "occupancy_logit",
            "residual_bins",
            "confidence",
            "refined_normalized_rae",
        )
    }
    maximum_chunk_delta = max(chunk_deltas.values())
    wrong_difference = float(
        (
            output["matched"]["occupancy_logit"].detach().float()
            - output["wrong"]["occupancy_logit"].detach().float()
        )
        .abs()
        .mean()
        .item()
    )
    matched_field = output["matched"]
    finite_field = all(
        bool(torch.isfinite(matched_field[key]).all())
        for key in (
            "occupancy_logit",
            "confidence",
            "residual_bins",
            "refined_normalized_rae",
        )
    )
    checks = {
        "largest_target_frame_selected": (
            int(matched_item["target_xyz_confidence"].shape[0]) == max(counts)
        ),
        "wrong_condition_is_cross_scene": (
            int(matched_item["sequence"]) != int(wrong_item["sequence"])
        ),
        "exact_10000_preflight_queries": (
            tuple(normalized_rae.shape) == (1, QUERY_COUNT, 3)
        ),
        "field_output_preserves_query_count": (
            tuple(matched_field["occupancy_logit"].shape) == (1, QUERY_COUNT)
            and tuple(matched_field["residual_bins"].shape)
            == (1, QUERY_COUNT, 3)
        ),
        "same_query_wrong_condition": bool(
            output["same_query_wrong_condition"]
        ),
        "condition_exclusive_contract": True,
        "wrong_condition_changes_untrained_field": wrong_difference > 0.0,
        "chunk_invariant_decode": maximum_chunk_delta <= 2e-2,
        "finite_field": finite_field,
        "finite_loss": bool(torch.isfinite(loss.total)),
        "finite_nonzero_gradients": (
            bool(all_gradients["finite"]) and bool(all_gradients["nonzero"])
        ),
        "condition_encoder_receives_gradient": (
            bool(encoder_gradients["finite"])
            and bool(encoder_gradients["nonzero"])
        ),
        "all_64_doppler_channels_receive_gradient": (
            bool(doppler_gradient_report["finite"])
            and doppler_gradient_report["channel_count"] == 64
            and doppler_gradient_report["nonzero_channel_count"] == 64
        ),
        "geometry_decoder_receives_gradient": (
            bool(decoder_gradients["finite"])
            and bool(decoder_gradients["nonzero"])
        ),
        "no_test_partition_accessed": True,
    }

    source_files = (
        repo / "code/models/rald_wce_field.py",
        repo / "code/losses/rald_wce.py",
        repo / "code/scripts/preflight_rald_wce.py",
        repo / "code/models/rald_matched.py",
        repo / "code/cube_dense/dataset.py",
        repo / "code/scripts/g1b_contract.py",
    )
    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "source_hashes": {
            str(path.resolve()): sha256(path)
            for path in source_files
        },
        "stage0_scope": {
            "depth": 6,
            "edm": False,
            "doppler_head": False,
            "exact_10000_selection": False,
            "preflight_query_count": QUERY_COUNT,
            "architecture_metadata": output["architecture_metadata"],
        },
        "data": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "frame_count_screened": len(dataset),
            "matched": {
                "cache_file": str(
                    cache_path(
                        args.cache_root,
                        dataset.records[matched_index],
                    ).resolve()
                ),
                "cache_sha256": sha256(
                    cache_path(
                        args.cache_root,
                        dataset.records[matched_index],
                    )
                ),
                "sequence": int(matched_item["sequence"]),
                "radar_index": int(matched_item["radar_index"]),
                "partition": str(matched_item["partition"]),
                "target_count": int(
                    matched_item["target_xyz_confidence"].shape[0]
                ),
            },
            "wrong_condition": {
                "sequence": int(wrong_item["sequence"]),
                "radar_index": int(wrong_item["radar_index"]),
                "partition": str(wrong_item["partition"]),
                "target_count": int(
                    wrong_item["target_xyz_confidence"].shape[0]
                ),
            },
            "query_targets": {
                "positive_count": int(query_targets["positive_count"]),
                "negative_count": int(query_targets["negative_count"]),
                "unique_target_cell_count": int(
                    query_targets["unique_target_cell_count"]
                ),
            },
        },
        "runtime": {
            "device": args.device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "model_parameter_count": parameter_count(model),
            "elapsed_seconds": elapsed_seconds,
            "decode_chunk_size": args.decode_chunk_size,
            "alternate_decode_chunk_size": alternate_chunk,
            "chunk_max_absolute_deltas": chunk_deltas,
            "maximum_chunk_absolute_delta": maximum_chunk_delta,
            "wrong_condition_mean_occupancy_delta": wrong_difference,
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "cuda_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        },
        "loss": {
            "total": float(loss.total.detach().item()),
            "components": {
                name: float(value.detach().item())
                for name, value in loss.components.items()
            },
        },
        "gradients": {
            "all": all_gradients,
            "condition_encoder": encoder_gradients,
            "full_raed_input_channels": doppler_gradient_report,
            "geometry_decoder_attention": decoder_gradients,
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
