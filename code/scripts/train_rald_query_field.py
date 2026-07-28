#!/usr/bin/env python3
"""Train the independently gated G1D RaLD query-field geometry model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import aggregate_geometry_reports, geometry_report  # noqa: E402
from eval.rald_guided_query import duplicate_report  # noqa: E402
from losses.rald_query_field import (  # noqa: E402
    rald_query_field_loss,
    sample_occupancy_queries,
)
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_query_field import (  # noqa: E402
    RaLDQueryField,
    stable_radar_proposals,
)
from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256  # noqa: E402


PROTOCOL = "g1d_rald_query_field_geometry_v2"
FORMAL_SEEDS = tuple(FROZEN_G1B_SEEDS)
SOURCE_PATTERN = re.compile(r"[0-9a-f]{40}")
OFFICIAL_RALD_COMMIT = "ffec4b41241391734b1eda5c093de843c909eb8e"


@dataclass(frozen=True)
class TrainConfig:
    protocol: str
    epochs: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_epochs: int
    weight_decay: float
    gradient_clip_norm: float
    ema_decay: float
    seed: int
    occupancy_query_count: int
    positive_query_ratio: float
    base_seed_count: int
    coarse_templates_per_seed: int
    coarse_query_count: int
    selected_coarse_count: int
    local_templates_per_selection: int
    point_count: int
    latent_count: int
    model_dim: int
    depth: int
    heads: int
    head_dim: int
    radar_base_channels: int
    radar_spectral_channels: int
    radar_token_count: int
    nms_kernel: tuple[int, int, int]
    offset_bounds_bins: tuple[float, float, float]
    decode_chunk_size: int
    positive_occupancy_weight: float
    negative_occupancy_weight: float
    geometry_weight: float
    outlier_weight: float
    existence_weight: float
    offset_weight: float
    repulsion_weight: float
    outlier_threshold_m: float
    existence_radius_m: float
    repulsion_distance_m: float
    eval_every: int
    train_limit: int | None
    validation_limit: int | None
    selection_metric: str
    test_accessed: bool


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("G1D training requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G1D training is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G1D training requires H200, got {resolved}")
    return device, resolved


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_source_tree(repo: Path, source_commit: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("G1D source commit does not match the checked-out HEAD")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"G1D formal source worktree is dirty: {dirty}")


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
        raise ValueError("G1D checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(sum(value.square().sum() for value in values)).item())


def finite_channel_norms(values: torch.Tensor | None, channel_dim: int) -> list[float]:
    if values is None:
        return []
    moved = values.detach().float().movedim(channel_dim, 0)
    norms = moved.reshape(moved.shape[0], -1).norm(dim=1)
    return [float(value) for value in norms.cpu()]


def gradient_audit(
    model: RaLDQueryField, cube: torch.Tensor
) -> dict[str, float | list[float]]:
    cube_channels = finite_channel_norms(cube.grad, 1)
    local_weight = model.query_state_projection[0].weight
    local_gradient = local_weight.grad
    local_columns = finite_channel_norms(
        None if local_gradient is None else local_gradient[:, :64],
        1,
    )
    energy_column = finite_channel_norms(
        None if local_gradient is None else local_gradient[:, 64:65],
        1,
    )
    range_column = finite_channel_norms(
        None if local_gradient is None else local_gradient[:, 65:66],
        1,
    )
    radar_weight = model.radar_encoder.spectral_projection.weight
    radar_columns = finite_channel_norms(radar_weight.grad, 1)
    conditioned_blocks = [
        gradient_norm(block.radar_attention.parameters())
        for block in model.latent_blocks
    ]
    mixed_modules = (
        model.static_latents,
        model.dynamic_latents,
        model.dynamic_proposal_attention,
        model.mixed_projection,
        model.post_mix_proposal_attention,
        model.post_mix_feed_forward,
    )
    query_modules = (
        model.coordinate_embedding,
        model.query_state_projection,
        model.decoder_attention,
        model.decoder_feed_forward,
    )
    heads = (model.occupancy_head, model.confidence_head, model.offset_head)
    return {
        "output_heads": gradient_norm(
            parameter for module in heads for parameter in module.parameters()
        ),
        "mixed_latent": gradient_norm(
            parameter
            for module in mixed_modules
            for parameter in module.parameters()
        ),
        "query_decoder": gradient_norm(
            parameter
            for module in query_modules
            for parameter in module.parameters()
        ),
        "full_raed_radar_encoder": gradient_norm(model.radar_encoder.parameters()),
        "cube_input_channel_norms": cube_channels,
        "local_spectrum_input_column_norms": local_columns,
        "absolute_energy_input_column_norms": energy_column,
        "normalized_range_input_column_norms": range_column,
        "radar_projection_input_column_norms": radar_columns,
        "condition_block_gradient_norms": conditioned_blocks,
    }


def aggregate_scalar_reports(reports: list[dict[str, float]]) -> dict:
    if not reports:
        raise ValueError("Cannot aggregate an empty G1D scalar report")
    keys = sorted({key for report in reports for key in report})
    if any(set(report) != set(keys) for report in reports):
        raise ValueError("G1D scalar reports do not share the same endpoints")
    return {
        key: {
            "mean": float(np.mean([report[key] for report in reports])),
            "std": float(np.std([report[key] for report in reports])),
            "median": float(np.median([report[key] for report in reports])),
            "sample_count": len(reports),
        }
        for key in keys
    }


def normalize_queries(
    coordinates_rae: torch.Tensor, spatial_shape: tuple[int, int, int]
) -> torch.Tensor:
    maximum = coordinates_rae.new_tensor([size - 1 for size in spatial_shape])
    clamped = coordinates_rae.clamp_min(0.0).minimum(maximum)
    return 2.0 * clamped / maximum - 1.0


def selected_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    return np.linspace(0, length - 1, limit).round().astype(int).tolist()


def cross_scene_condition_indices(records: list[dict]) -> list[int]:
    """Return a deterministic derangement whose pairs use different scenes."""

    count = len(records)
    if count < 2:
        raise ValueError("Condition shuffle requires at least two frames")
    sequences = [int(record["sequence"]) for record in records]
    for shift in range(1, count):
        candidate = [(index + shift) % count for index in range(count)]
        if all(
            sequences[index] != sequences[other]
            for index, other in enumerate(candidate)
        ):
            return candidate
    raise ValueError("No cross-scene condition derangement exists")


def move_frame(item: dict, device: torch.device) -> torch.Tensor:
    return item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)


def query_seed(base: int, sequence: int, radar_index: int, epoch: int = 0) -> int:
    return int(
        (base + 1_000_003 * epoch + 10_007 * sequence + 101 * radar_index)
        % (2**63 - 1)
    )


def occupancy_queries(
    item: dict,
    config: TrainConfig,
    device: torch.device,
    *,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        query_seed(
            config.seed,
            int(item["sequence"]),
            int(item["radar_index"]),
            epoch,
        )
    )
    sampled = sample_occupancy_queries(
        item["target_rae_index"].to(device),
        item["occupancy"].to(device),
        count=config.occupancy_query_count,
        positive_ratio=config.positive_query_ratio,
        generator=generator,
    )
    normalized = normalize_queries(
        sampled.coordinates_rae, tuple(int(size) for size in item["occupancy"].shape)
    )
    return normalized.unsqueeze(0), sampled.labels


def occupancy_report(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    logits = logits.float().reshape(-1)
    labels = labels.float().to(logits).reshape(-1)
    positive = labels == 1
    negative = labels == 0
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("G1D occupancy evaluation requires both query classes")
    prediction = logits >= 0.0
    positive_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[positive], labels[positive]
    )
    negative_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[negative], labels[negative]
    )
    occupancy_bce = 0.1 * positive_bce + negative_bce
    return {
        "positive_recall": float(prediction[positive].float().mean().item()),
        "empty_false_positive_rate": float(
            prediction[negative].float().mean().item()
        ),
        "positive_bce": float(positive_bce.item()),
        "negative_bce": float(negative_bce.item()),
        "occupancy_bce": float(occupancy_bce.item()),
    }


def cached_proposal_flat_index(
    item: dict,
    cube: torch.Tensor,
    config: TrainConfig,
    cache: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    key = (int(item["sequence"]), int(item["radar_index"]))
    if key not in cache:
        proposals = stable_radar_proposals(
            cube,
            seed_count=config.base_seed_count,
            nms_kernel=config.nms_kernel,
        )
        cache[key] = proposals.flat_index.detach().cpu()
    return cache[key].to(device=cube.device, non_blocking=True)


@torch.no_grad()
def update_ema(
    ema_model: RaLDQueryField, model: RaLDQueryField, decay: float
) -> None:
    model_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        source = model_parameters[name]
        ema_parameter.mul_(decay).add_(source, alpha=1.0 - decay)
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        ema_buffer.copy_(model_buffers[name])


@torch.inference_mode()
def evaluate(
    model: RaLDQueryField,
    dataset: KRadarCubeDataset,
    indices: list[int],
    device: torch.device,
    config: TrainConfig,
    proposal_cache: dict[tuple[int, int], torch.Tensor],
) -> dict:
    model.eval()
    generated_reports = []
    control_reports = []
    cfar_reports = []
    duplicate_reports = []
    occupancy_reports = []
    shuffled_occupancy_reports = []
    shuffled_geometry_reports = []
    frames = []
    validation_records = [dataset.records[index] for index in indices]
    shuffled_positions = cross_scene_condition_indices(validation_records)
    for position, index in enumerate(indices):
        item = dataset[index]
        shuffled_item = dataset[indices[shuffled_positions[position]]]
        cube = move_frame(item, device)
        shuffled_cube = move_frame(shuffled_item, device)
        target = item["target_xyz_confidence"].to(device)
        proposal_flat_index = cached_proposal_flat_index(
            item,
            cube,
            config,
            proposal_cache,
        )
        normalized_queries, labels = occupancy_queries(
            item, config, device, epoch=0
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                cube,
                normalized_queries,
                proposal_flat_index=proposal_flat_index,
            )
            shuffled_output = model(
                cube,
                normalized_queries,
                condition_cube_drae=shuffled_cube,
                proposal_flat_index=proposal_flat_index,
            )
        generated = geometry_report(
            output["xyz_m"][0].float(),
            target[:, :3].float(),
            target_weight=target[:, 3].float(),
        )
        control = geometry_report(
            output["zero_offset_xyz_m"][0].float(),
            target[:, :3].float(),
            target_weight=target[:, 3].float(),
        )
        shuffled_geometry = geometry_report(
            shuffled_output["xyz_m"][0].float(),
            target[:, :3].float(),
            target_weight=target[:, 3].float(),
        )
        cfar = geometry_report(
            item["cfar_xyzd_power_snr"][:, :3].to(device),
            target[:, :3].float(),
            target_weight=target[:, 3].float(),
        )
        duplicates = duplicate_report(output["xyz_m"][0].float())
        occupancy = occupancy_report(
            output["training_query_occupancy_logit"][0], labels
        )
        shuffled_occupancy = occupancy_report(
            shuffled_output["training_query_occupancy_logit"][0], labels
        )
        current_occupancy_bce = occupancy["occupancy_bce"]
        shuffled_occupancy_bce = shuffled_occupancy["occupancy_bce"]
        current_chamfer = generated["chamfer_m"]
        shuffled_chamfer = shuffled_geometry["chamfer_m"]
        query_coordinates = output[
            "training_query_input_coordinates_rae"
        ][0].float()
        fractional_coordinate = (
            query_coordinates - query_coordinates.round()
        ).abs().amax(dim=1) > 1e-6
        positive_mask = labels.bool()
        negative_mask = ~positive_mask
        frame = {
            "sequence": int(item["sequence"]),
            "radar_index": int(item["radar_index"]),
            "shuffled_condition_sequence": int(shuffled_item["sequence"]),
            "shuffled_condition_radar_index": int(shuffled_item["radar_index"]),
            "generated": generated,
            "zero_offset_control": control,
            "condition_shuffled": shuffled_geometry,
            "cfar": cfar,
            "duplicates": duplicates,
            "occupancy": occupancy,
            "condition_shuffled_occupancy": shuffled_occupancy,
            "condition_shuffle_occupancy_bce_fraction": float(
                shuffled_occupancy_bce / max(current_occupancy_bce, 1e-12) - 1.0
            ),
            "condition_shuffle_chamfer_fraction": float(
                shuffled_chamfer / max(current_chamfer, 1e-12) - 1.0
            ),
            "confidence_mean": float(output["confidence"].float().mean().item()),
            "occupancy_query_count": int(labels.numel()),
            "positive_occupancy_query_count": int(labels.sum().item()),
            "empty_occupancy_query_count": int(
                labels.numel() - labels.sum().item()
            ),
            "positive_fractional_coordinate_rate": float(
                fractional_coordinate[positive_mask].float().mean().item()
            ),
            "empty_fractional_coordinate_rate": float(
                fractional_coordinate[negative_mask].float().mean().item()
            ),
            "offset_abs_mean_bins": float(
                output["offset_bins"].float().abs().mean().item()
            ),
            "normalized_log_energy_abs_max": float(
                output["training_query_normalized_log_energy"]
                .float()
                .abs()
                .max()
                .item()
            ),
            "radar_token_count": int(output["radar_token_count"].item()),
            "coarse_query_count": int(output["coarse_query_count"].item()),
            "selected_coarse_count": int(
                output["selected_coarse_count"].item()
            ),
            "final_point_count": int(output["final_point_count"].item()),
            "proposal_cache_used": bool(
                output["proposal_cache_used"].item()
            ),
        }
        generated_reports.append(generated)
        control_reports.append(control)
        shuffled_geometry_reports.append(shuffled_geometry)
        cfar_reports.append(cfar)
        duplicate_reports.append(duplicates)
        occupancy_reports.append(occupancy)
        shuffled_occupancy_reports.append(shuffled_occupancy)
        frames.append(frame)
        del (
            item,
            shuffled_item,
            cube,
            shuffled_cube,
            target,
            normalized_queries,
            labels,
            proposal_flat_index,
            output,
            shuffled_output,
        )
        torch.cuda.empty_cache()
    shuffle_reports = [
        {
            "occupancy_bce_fraction": frame[
                "condition_shuffle_occupancy_bce_fraction"
            ],
            "chamfer_fraction": frame["condition_shuffle_chamfer_fraction"],
        }
        for frame in frames
    ]
    return {
        "frame_count": len(frames),
        "generated": aggregate_geometry_reports(generated_reports),
        "zero_offset_control": aggregate_geometry_reports(control_reports),
        "condition_shuffled": aggregate_geometry_reports(
            shuffled_geometry_reports
        ),
        "cfar": aggregate_geometry_reports(cfar_reports),
        "duplicates": aggregate_scalar_reports(duplicate_reports),
        "occupancy": aggregate_scalar_reports(occupancy_reports),
        "condition_shuffled_occupancy": aggregate_scalar_reports(
            shuffled_occupancy_reports
        ),
        "condition_shuffle": aggregate_scalar_reports(shuffle_reports),
        "confidence_mean": {
            "mean": float(np.mean([frame["confidence_mean"] for frame in frames])),
            "median": float(
                np.median([frame["confidence_mean"] for frame in frames])
            ),
        },
        "frames": frames,
        "proposal_cache_entry_count": len(proposal_cache),
    }


def selection_score(metrics: dict) -> float:
    chamfer = float(metrics["generated"]["chamfer_m"]["median"])
    outlier = float(metrics["generated"]["outlier_fraction_2m"]["mean"])
    recall = float(metrics["occupancy"]["positive_recall"]["mean"])
    false_positive = float(
        metrics["occupancy"]["empty_false_positive_rate"]["mean"]
    )
    return (
        chamfer
        + 2.0 * max(outlier - 0.25, 0.0)
        + max(0.80 - recall, 0.0)
        + max(false_positive - 0.20, 0.0)
    )


def learning_rate(config: TrainConfig, epoch: int) -> float:
    if epoch <= config.warmup_epochs:
        return config.learning_rate * epoch / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(
        config.epochs - config.warmup_epochs, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + cosine * (
        config.learning_rate - config.minimum_learning_rate
    )


def save_checkpoint(
    path: Path,
    model: RaLDQueryField,
    ema_model: RaLDQueryField,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config: TrainConfig,
    provenance: dict,
    gradient_steps: list[dict],
    record: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "provenance": provenance,
            "gradient_steps": gradient_steps,
            "record": record,
            "rng_state": capture_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def build_model(config: TrainConfig, axes, normalization: dict) -> RaLDQueryField:
    return RaLDQueryField(
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        log_center=float(normalization["normalization"]["center"]),
        log_scale=float(normalization["normalization"]["scale"]),
        base_seed_count=config.base_seed_count,
        selected_coarse_count=config.selected_coarse_count,
        latent_count=config.latent_count,
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
        radar_base_channels=config.radar_base_channels,
        radar_spectral_channels=config.radar_spectral_channels,
        offset_bounds_bins=config.offset_bounds_bins,
        nms_kernel=config.nms_kernel,
        decode_chunk_size=config.decode_chunk_size,
    )


def source_hashes(script_path: Path) -> dict[str, dict[str, str]]:
    code_root = script_path.parents[1]
    paths = (
        script_path,
        code_root / "models/rald_query_field.py",
        code_root / "losses/rald_query_field.py",
        code_root / "models/rald_matched.py",
        code_root / "models/cube_doppler.py",
        code_root / "models/cube_occupancy.py",
        code_root / "models/cube_cycle.py",
        code_root / "models/point_to_cube.py",
        code_root / "cube_dense/dataset.py",
        code_root / "cube_dense/kradar.py",
        code_root / "eval/dense_geometry.py",
        code_root / "eval/rald_guided_query.py",
        code_root / "scripts/g1b_contract.py",
    )
    return {
        str(path.resolve()): {"sha256": sha256(path)}
        for path in paths
        if path.is_file()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if not SOURCE_PATTERN.fullmatch(args.source_commit):
        raise ValueError("G1D source commit must be a full lowercase Git SHA")
    if not args.smoke and (
        args.train_limit is not None or args.validation_limit is not None
    ):
        raise ValueError("Formal G1D cannot limit train or validation frames")
    if args.eval_every <= 0:
        raise ValueError("G1D evaluation cadence must be positive")
    device, device_name = require_h200(args.device)
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    nonempty = args.output.exists() and any(args.output.iterdir())
    if nonempty and args.overwrite:
        shutil.rmtree(args.output)
        nonempty = False
    if nonempty and not args.resume:
        raise FileExistsError(f"G1D output is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No G1D run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    if split.get("gate_pass") is not True:
        raise ValueError("G1D scene split did not pass its leakage gate")
    if any(frame.get("partition") == "test" for frame in manifest["frames"]):
        raise ValueError("G1D development manifest must not contain test frames")
    config = TrainConfig(
        protocol=PROTOCOL,
        epochs=1 if args.smoke else 150,
        learning_rate=1e-4,
        minimum_learning_rate=1e-6,
        warmup_epochs=1 if args.smoke else 5,
        weight_decay=0.05,
        gradient_clip_norm=10.0,
        ema_decay=0.999,
        seed=args.seed,
        occupancy_query_count=10_000,
        positive_query_ratio=0.0625,
        base_seed_count=1_000,
        coarse_templates_per_seed=32,
        coarse_query_count=32_000,
        selected_coarse_count=2_500,
        local_templates_per_selection=4,
        point_count=10_000,
        latent_count=512,
        model_dim=512,
        depth=24,
        heads=8,
        head_dim=64,
        radar_base_channels=64,
        radar_spectral_channels=16,
        radar_token_count=336,
        nms_kernel=(5, 5, 3),
        offset_bounds_bins=(8.0, 4.0, 2.0),
        decode_chunk_size=4_096,
        positive_occupancy_weight=0.1,
        negative_occupancy_weight=1.0,
        geometry_weight=1.0,
        outlier_weight=0.25,
        existence_weight=0.10,
        offset_weight=0.02,
        repulsion_weight=0.02,
        outlier_threshold_m=2.0,
        existence_radius_m=1.0,
        repulsion_distance_m=0.10,
        eval_every=args.eval_every,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        selection_metric=(
            "median_chamfer + 2*max(mean_outlier_2m-0.25,0) + "
            "max(0.80-positive_recall,0) + max(empty_fpr-0.20,0)"
        ),
        test_accessed=False,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    axes = load_axes(args.data_root / "resources")
    model = build_model(config, axes, normalization).to(device)
    if len(model.latent_blocks) != 24:
        raise AssertionError("Formal G1D must instantiate 24 conditioned blocks")
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    script_path = Path(__file__).resolve()
    protocol_path = repo / "docs/g1d_rald_query_field_protocol.md"
    source_map_path = repo / "docs/g1d_rald_source_map.md"
    provenance = {
        "git_commit": args.source_commit,
        "worktree_clean": True,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "scene_split": str(args.scene_split.resolve()),
        "scene_split_sha256": sha256(args.scene_split),
        "normalization": str(args.normalization.resolve()),
        "normalization_sha256": sha256(args.normalization),
        "source_hashes": source_hashes(script_path),
        "protocol_document": str(protocol_path.resolve()),
        "protocol_document_sha256": sha256(protocol_path),
        "rald_source_map": str(source_map_path.resolve()),
        "rald_source_map_sha256": sha256(source_map_path),
        "device": device_name,
        "torch_version": torch.__version__,
        "model_parameter_count": parameter_count(model),
        "partitions": ["train", "validation"],
        "test_accessed": False,
        "external_pretraining": False,
        "cfar_query_helper": False,
        "occupancy_checkpoint": None,
        "official_rald_commit": OFFICIAL_RALD_COMMIT,
        "doppler_geometry_status": "measured_cube_spectrum_attached_not_learned",
        "proposal_index_cache": "deterministic_flat_indices_only",
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("G1D resume configuration or provenance differs")
    else:
        atomic_json(config_path, run_document)

    train_set = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("train",)
    )
    validation_set = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    train_indices = selected_indices(len(train_set), config.train_limit)
    validation_indices = selected_indices(
        len(validation_set), config.validation_limit
    )
    cross_scene_condition_indices(
        [validation_set.records[index] for index in validation_indices]
    )

    start_epoch = 1
    best_score = float("inf")
    prior_elapsed = 0.0
    gradient_steps: list[dict] = []
    proposal_cache: dict[tuple[int, int], torch.Tensor] = {}
    if args.resume:
        last = torch.load(
            args.output / "last.pt", map_location=device, weights_only=False
        )
        if last.get("config") != asdict(config) or last.get(
            "provenance"
        ) != provenance:
            raise ValueError("G1D last checkpoint metadata differs")
        model.load_state_dict(last["model"], strict=True)
        ema_model.load_state_dict(last["ema_model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        restore_rng_state(last["rng_state"])
        start_epoch = int(last["epoch"]) + 1
        prior_elapsed = float(last["record"]["elapsed_seconds"])
        gradient_steps = list(last["gradient_steps"])
        best = torch.load(
            args.output / "best.pt", map_location="cpu", weights_only=False
        )
        best_score = float(best["record"]["selection_score"])

    initial_path = args.output / "initial_validation_metrics.json"
    if initial_path.is_file():
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
    else:
        initial = evaluate(
            ema_model,
            validation_set,
            validation_indices,
            device,
            config,
            proposal_cache,
        )
        atomic_json(initial_path, initial)
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
            if update_count < 2:
                cube = cube.detach().requires_grad_(True)
            target = item["target_xyz_confidence"].to(device)
            proposal_flat_index = cached_proposal_flat_index(
                item,
                cube,
                config,
                proposal_cache,
            )
            normalized_queries, labels = occupancy_queries(
                item, config, device, epoch=epoch
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    cube,
                    normalized_queries,
                    proposal_flat_index=proposal_flat_index,
                )
                loss_output = {
                    "query_logits": output[
                        "training_query_occupancy_logit"
                    ],
                    "generated_xyz_m": output["xyz_m"],
                    "generated_confidence_logit": output["confidence_logit"],
                    "normalized_offset": output["offset_bins"]
                    / model.offset_bounds_bins.to(output["offset_bins"]),
                }
                loss = rald_query_field_loss(
                    loss_output,
                    labels,
                    target,
                    generated_point_count=config.point_count,
                    positive_weight=config.positive_occupancy_weight,
                    negative_weight=config.negative_occupancy_weight,
                    geometry_weight=config.geometry_weight,
                    outlier_weight=config.outlier_weight,
                    existence_weight=config.existence_weight,
                    offset_weight=config.offset_weight,
                    repulsion_weight=config.repulsion_weight,
                    outlier_threshold_m=config.outlier_threshold_m,
                    existence_radius_m=config.existence_radius_m,
                    repulsion_distance_m=config.repulsion_distance_m,
                )
            loss.total.backward()
            update_count += 1
            if update_count <= 2:
                gradient_steps.append(
                    {
                        "update": update_count,
                        "sequence": int(item["sequence"]),
                        "radar_index": int(item["radar_index"]),
                        "gradients": gradient_audit(model, cube),
                    }
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            update_ema(ema_model, model, config.ema_decay)
            losses.append(float(loss.total.detach().item()))
            for name, value in loss.components.items():
                components.setdefault(name, []).append(float(value.item()))
            del (
                item,
                cube,
                target,
                normalized_queries,
                labels,
                proposal_flat_index,
                output,
                loss_output,
                loss,
            )
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
                prior_elapsed + time.monotonic() - started, 3
            ),
        }
        if (
            epoch == 1
            or epoch % config.eval_every == 0
            or epoch == config.epochs
        ):
            metrics = evaluate(
                ema_model,
                validation_set,
                validation_indices,
                device,
                config,
                proposal_cache,
            )
            score = selection_score(metrics)
            record["validation"] = metrics
            record["selection_score"] = score
            atomic_json(
                args.output / f"metrics_epoch_{epoch:04d}.json", metrics
            )
            is_best = score < best_score
        else:
            is_best = False
        save_checkpoint(
            args.output / "last.pt",
            model,
            ema_model,
            optimizer,
            epoch=epoch,
            config=config,
            provenance=provenance,
            gradient_steps=gradient_steps,
            record=record,
        )
        if is_best:
            best_score = score
            save_checkpoint(
                args.output / "best.pt",
                model,
                ema_model,
                optimizer,
                epoch=epoch,
                config=config,
                provenance=provenance,
                gradient_steps=gradient_steps,
                record=record,
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    best_path = args.output / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("G1D training produced no selected checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    if best.get("config") != asdict(config) or best.get(
        "provenance"
    ) != provenance:
        raise ValueError("G1D selected checkpoint metadata differs")
    ema_model.load_state_dict(best["ema_model"], strict=True)
    final = evaluate(
        ema_model,
        validation_set,
        validation_indices,
        device,
        config,
        proposal_cache,
    )
    report = {
        "protocol": PROTOCOL,
        "completed": True,
        "best_epoch": int(best["epoch"]),
        "selection_metric": config.selection_metric,
        "selection_value": best_score,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "evaluation_model": "ema_0p999",
        "initial": initial,
        "validation": final,
        "gradient_steps": gradient_steps,
        "proposal_cache_entry_count": len(proposal_cache),
        "test_accessed": False,
        "provenance": provenance,
    }
    atomic_json(args.output / "best_validation_metrics.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
