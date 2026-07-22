#!/usr/bin/env python3
"""Train the independently gated G1C RaLD-guided query geometry model."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
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
from losses.rald_guided_query import rald_guided_geometry_loss  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_guided_query import RaLDGuidedQueryGenerator  # noqa: E402
from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256  # noqa: E402
from scripts.train_cube_doppler import move_frame, selected_indices  # noqa: E402


PROTOCOL = "g1c_rald_guided_query_geometry_v1"
FORMAL_SEEDS = tuple(FROZEN_G1B_SEEDS)
SOURCE_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class TrainConfig:
    protocol: str
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    base_seed_count: int
    queries_per_seed: int
    point_count: int
    latent_count: int
    model_dim: int
    depth: int
    heads: int
    head_dim: int
    radar_base_channels: int
    radar_spectral_channels: int
    nms_kernel: tuple[int, int, int]
    offset_bounds_bins: tuple[float, float, float]
    geometry_weight: float
    outlier_weight: float
    existence_weight: float
    offset_weight: float
    repulsion_weight: float
    outlier_threshold_m: float
    repulsion_distance_m: float
    eval_every: int
    max_eval_frames: int
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
        raise RuntimeError("G1C training requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G1C training is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G1C training requires H200, got {resolved}")
    return device, resolved


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
        raise ValueError("G1C checkpoint RNG state is incomplete")
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


def gradient_audit(model: RaLDGuidedQueryGenerator) -> dict[str, float]:
    physical = list(model.refiner.physical_head.parameters())
    physical_ids = {id(parameter) for parameter in physical}
    latent = [
        parameter
        for parameter in model.refiner.parameters()
        if id(parameter) not in physical_ids
    ]
    return {
        "physical_head": gradient_norm(physical),
        "mixed_latent_and_query_decoder": gradient_norm(latent),
        "full_raed_radar_encoder": gradient_norm(model.radar_encoder.parameters()),
        "local_64bin_spectrum_projection": gradient_norm(
            model.spectrum_projection.parameters()
        ),
        "template_embedding": gradient_norm(model.template_embedding.parameters()),
    }


def aggregate_scalar_reports(reports: list[dict[str, float]]) -> dict:
    keys = sorted({key for report in reports for key in report})
    return {
        key: {
            "mean": float(np.mean([report[key] for report in reports])),
            "std": float(np.std([report[key] for report in reports])),
            "median": float(np.median([report[key] for report in reports])),
            "sample_count": len(reports),
        }
        for key in keys
    }


@torch.inference_mode()
def evaluate(
    model: RaLDGuidedQueryGenerator,
    dataset: KRadarCubeDataset,
    indices: list[int],
    device: torch.device,
) -> dict:
    model.eval()
    generated_reports = []
    anchor_reports = []
    cfar_reports = []
    duplicate_reports = []
    frames = []
    for index in indices:
        item = dataset[index]
        cube, _ = move_frame(item, device)
        target = item["target_xyz_confidence"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(cube)
        generated = geometry_report(
            output["xyz_m"][0].float(),
            target[:, :3].float(),
            target_weight=target[:, 3].float(),
        )
        anchors = geometry_report(
            output["anchor_xyz_m"][0].float(),
            target[:, :3].float(),
            target_weight=target[:, 3].float(),
        )
        cfar = geometry_report(
            item["cfar_xyzd_power_snr"][:, :3].to(device),
            target[:, :3].float(),
            target_weight=target[:, 3].float(),
        )
        duplicates = duplicate_report(output["xyz_m"][0].float())
        frame = {
            "sequence": int(item["sequence"]),
            "radar_index": int(item["radar_index"]),
            "generated": generated,
            "radar_guided_initial_queries": anchors,
            "cfar": cfar,
            "duplicates": duplicates,
            "confidence_mean": float(output["confidence"].float().mean().item()),
            "offset_abs_mean_bins": float(
                output["offset_bins"].float().abs().mean().item()
            ),
            "radar_token_count": int(output["radar_token_count"].item()),
        }
        generated_reports.append(generated)
        anchor_reports.append(anchors)
        cfar_reports.append(cfar)
        duplicate_reports.append(duplicates)
        frames.append(frame)
        del item, cube, target, output
        torch.cuda.empty_cache()
    return {
        "frame_count": len(frames),
        "generated": aggregate_geometry_reports(generated_reports),
        "radar_guided_initial_queries": aggregate_geometry_reports(anchor_reports),
        "cfar": aggregate_geometry_reports(cfar_reports),
        "duplicates": aggregate_scalar_reports(duplicate_reports),
        "confidence_mean": {
            "mean": float(np.mean([frame["confidence_mean"] for frame in frames])),
            "median": float(np.median([frame["confidence_mean"] for frame in frames])),
        },
        "frames": frames,
    }


def selection_score(metrics: dict) -> float:
    chamfer = float(metrics["generated"]["chamfer_m"]["median"])
    outlier = float(metrics["generated"]["outlier_fraction_2m"]["mean"])
    return chamfer + 2.0 * max(outlier - 0.25, 0.0)


def save_checkpoint(
    path: Path,
    model: RaLDGuidedQueryGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler,
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
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
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


def build_model(config: TrainConfig, axes, normalization: dict) -> RaLDGuidedQueryGenerator:
    return RaLDGuidedQueryGenerator(
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        log_center=float(normalization["normalization"]["center"]),
        log_scale=float(normalization["normalization"]["scale"]),
        base_seed_count=config.base_seed_count,
        queries_per_seed=config.queries_per_seed,
        latent_count=config.latent_count,
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
        radar_base_channels=config.radar_base_channels,
        radar_spectral_channels=config.radar_spectral_channels,
        offset_bounds_bins=config.offset_bounds_bins,
        nms_kernel=config.nms_kernel,
    )


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
    parser.add_argument("--max-eval-frames", type=int, default=8)
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
        raise ValueError("G1C source commit must be a full lowercase Git SHA")
    if not args.smoke and (args.train_limit is not None or args.validation_limit is not None):
        raise ValueError("Formal G1C cannot limit train or validation frames")
    if args.eval_every <= 0 or args.max_eval_frames <= 0:
        raise ValueError("G1C evaluation cadence must be positive")
    device, device_name = require_h200(args.device)
    nonempty = args.output.exists() and any(args.output.iterdir())
    if nonempty and args.overwrite:
        shutil.rmtree(args.output)
        nonempty = False
    if nonempty and not args.resume:
        raise FileExistsError(f"G1C output is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No G1C run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    if split.get("gate_pass") is not True:
        raise ValueError("G1C scene split did not pass its leakage gate")
    if any(frame.get("partition") == "test" for frame in manifest["frames"]):
        raise ValueError("G1C development manifest must not contain test frames")
    config = TrainConfig(
        protocol=PROTOCOL,
        epochs=1 if args.smoke else 30,
        learning_rate=1e-4,
        weight_decay=1e-4,
        seed=args.seed,
        base_seed_count=1_000,
        queries_per_seed=10,
        point_count=10_000,
        latent_count=512,
        model_dim=512,
        depth=24,
        heads=8,
        head_dim=64,
        radar_base_channels=64,
        radar_spectral_channels=16,
        nms_kernel=(5, 5, 3),
        offset_bounds_bins=(8.0, 4.0, 2.0),
        geometry_weight=1.0,
        outlier_weight=0.25,
        existence_weight=0.10,
        offset_weight=0.02,
        repulsion_weight=0.02,
        outlier_threshold_m=2.0,
        repulsion_distance_m=0.10,
        eval_every=args.eval_every,
        max_eval_frames=args.max_eval_frames,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        selection_metric="median_chamfer + 2*max(mean_outlier_2m-0.25,0)",
        test_accessed=False,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    axes = load_axes(args.data_root / "resources")
    model = build_model(config, axes, normalization).to(device)
    optimizer = torch.optim.AdamW(
        model.geometry_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    script_path = Path(__file__).resolve()
    model_path = script_path.parents[1] / "models/rald_guided_query.py"
    loss_path = script_path.parents[1] / "losses/rald_guided_query.py"
    provenance = {
        "git_commit": args.source_commit,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "scene_split": str(args.scene_split.resolve()),
        "scene_split_sha256": sha256(args.scene_split),
        "normalization": str(args.normalization.resolve()),
        "normalization_sha256": sha256(args.normalization),
        "training_script": str(script_path),
        "training_script_sha256": sha256(script_path),
        "model_source": str(model_path),
        "model_source_sha256": sha256(model_path),
        "loss_source": str(loss_path),
        "loss_source_sha256": sha256(loss_path),
        "device": device_name,
        "torch_version": torch.__version__,
        "model_parameter_count": parameter_count(model),
        "partitions": ["train", "validation"],
        "test_accessed": False,
        "external_pretraining": False,
        "cfar_query_helper": False,
        "occupancy_checkpoint": None,
        "official_rald_commit": "ffec4b41241391734b1eda5c093de843c909eb8e",
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("G1C resume configuration or provenance differs")
    else:
        atomic_json(config_path, run_document)

    train_set = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("train",)
    )
    validation_set = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    train_indices = selected_indices(len(train_set), config.train_limit)
    validation_indices = selected_indices(len(validation_set), config.validation_limit)
    positions = selected_indices(
        len(validation_indices), min(config.max_eval_frames, len(validation_indices))
    )
    selection_indices = [validation_indices[position] for position in positions]

    start_epoch = 1
    best_score = float("inf")
    prior_elapsed = 0.0
    gradient_steps: list[dict] = []
    if args.resume:
        last = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        if last.get("config") != asdict(config) or last.get("provenance") != provenance:
            raise ValueError("G1C last checkpoint metadata differs")
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        restore_rng_state(last["rng_state"])
        start_epoch = int(last["epoch"]) + 1
        prior_elapsed = float(last["record"]["elapsed_seconds"])
        gradient_steps = list(last["gradient_steps"])
        best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
        best_score = float(best["record"]["selection_score"])

    initial_path = args.output / "initial_validation_metrics.json"
    if initial_path.is_file():
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
    else:
        initial = evaluate(model, validation_set, selection_indices, device)
        atomic_json(initial_path, initial)
    started = time.monotonic()
    update_count = max(0, start_epoch - 1) * len(train_indices)
    log_path = args.output / "train_log.jsonl"
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        order = train_indices.copy()
        random.Random(config.seed + epoch).shuffle(order)
        losses = []
        components: dict[str, list[float]] = {}
        for index in order:
            item = train_set[index]
            cube, _ = move_frame(item, device)
            target = item["target_xyz_confidence"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(cube)
                loss = rald_guided_geometry_loss(
                    output,
                    target,
                    queries_per_seed=config.queries_per_seed,
                    geometry_weight=config.geometry_weight,
                    outlier_weight=config.outlier_weight,
                    existence_weight=config.existence_weight,
                    offset_weight=config.offset_weight,
                    repulsion_weight=config.repulsion_weight,
                    outlier_threshold_m=config.outlier_threshold_m,
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
                        "gradients": gradient_audit(model),
                    }
                )
            torch.nn.utils.clip_grad_norm_(model.geometry_parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.total.detach().item()))
            for name, value in loss.components.items():
                components.setdefault(name, []).append(float(value.item()))
            del item, cube, target, output, loss
            torch.cuda.empty_cache()
        scheduler.step()
        record = {
            "epoch": epoch,
            "update_count": update_count,
            "train_loss_mean": float(np.mean(losses)),
            "train_components": {
                name: float(np.mean(values)) for name, values in components.items()
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
        }
        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            metrics = evaluate(model, validation_set, selection_indices, device)
            score = selection_score(metrics)
            record["validation"] = metrics
            record["selection_score"] = score
            atomic_json(args.output / f"metrics_epoch_{epoch:04d}.json", metrics)
            is_best = score < best_score
        else:
            is_best = False
        save_checkpoint(
            args.output / "last.pt",
            model,
            optimizer,
            scheduler,
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
                optimizer,
                scheduler,
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
        raise RuntimeError("G1C training produced no selected checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    if best.get("config") != asdict(config) or best.get("provenance") != provenance:
        raise ValueError("G1C selected checkpoint metadata differs")
    model.load_state_dict(best["model"], strict=True)
    final = evaluate(model, validation_set, validation_indices, device)
    report = {
        "protocol": PROTOCOL,
        "completed": True,
        "best_epoch": int(best["epoch"]),
        "selection_metric": config.selection_metric,
        "selection_value": best_score,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "initial": initial,
        "validation": final,
        "gradient_steps": gradient_steps,
        "test_accessed": False,
        "provenance": provenance,
    }
    atomic_json(args.output / "best_validation_metrics.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
