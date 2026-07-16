#!/usr/bin/env python3
"""Train matched RAE-Max or Full-RAED frustum-occupancy baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
    occupancy_to_points,
)
from losses.occupancy import occupancy_loss  # noqa: E402
from models.cube_occupancy import CubeOccupancyNet, parameter_count  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    mode: str
    epochs: int
    learning_rate: float
    weight_decay: float
    base_channels: int
    seed: int
    point_count: int
    confidence_threshold: float
    eval_every: int
    max_eval_frames: int
    train_limit: int | None
    validation_limit: int | None
    log_center: float
    log_scale: float


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


def move_frame(item: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
    occupancy = item["occupancy"].unsqueeze(0).to(device, non_blocking=True)
    return cube, occupancy


@torch.inference_mode()
def evaluate(
    model: CubeOccupancyNet,
    dataset: KRadarCubeDataset,
    indices: list[int],
    axes,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    model.eval()
    losses = []
    generated_reports = []
    cfar_reports = []
    frames = []
    for index in indices:
        item = dataset[index]
        cube, occupancy = move_frame(item, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(cube)
            loss, _ = occupancy_loss(logits, occupancy)
        target = item["target_xyz_confidence"].to(device)
        target_xyz = target[:, :3]
        target_weight = target[:, 3]
        high_confidence_count = int(
            (target_weight >= config.confidence_threshold).sum().item()
        )
        generated, confidence, _ = occupancy_to_points(
            logits[0].float(), axes, point_count=config.point_count
        )
        cfar = item["cfar_xyzd_power_snr"][:, :3].to(device)
        generated_report = geometry_report(
            generated, target_xyz, target_weight=target_weight
        )
        cfar_report = geometry_report(cfar, target_xyz, target_weight=target_weight)
        generated_reports.append(generated_report)
        cfar_reports.append(cfar_report)
        losses.append(float(loss.item()))
        frames.append(
            {
                "sequence": item["sequence"],
                "radar_index": item["radar_index"],
                "occupancy_loss": float(loss.item()),
                "prediction_confidence_mean": float(confidence.mean().item()),
                "high_confidence_target_count": high_confidence_count,
                "generated": generated_report,
                "cfar": cfar_report,
            }
        )
        del cube, occupancy, logits, target, target_xyz, target_weight
        del generated, confidence, cfar
        torch.cuda.empty_cache()
    return {
        "frame_count": len(indices),
        "occupancy_loss_mean": float(np.mean(losses)),
        "generated": aggregate_geometry_reports(generated_reports),
        "cfar": aggregate_geometry_reports(cfar_reports),
        "frames": frames,
    }


def save_checkpoint(
    path: Path,
    model: CubeOccupancyNet,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config: TrainConfig,
    provenance: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "provenance": provenance,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=CubeOccupancyNet.MODES, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Cube occupancy training requires CUDA")
    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_normalization_frames = [
        {
            "sequence": int(record["sequence"]),
            "radar_index": int(record["radar_index"]),
        }
        for record in manifest["frames"]
        if record["partition"] == "train"
    ]
    normalization = json.loads(args.normalization_stats.read_text(encoding="utf-8"))
    if normalization["partitions"] != ["train"]:
        raise ValueError("Normalization statistics must use the train partition only")
    if normalization["frame_limit"] is not None:
        raise ValueError("Normalization statistics must cover the full train partition")
    if normalization["frames"] != expected_normalization_frames:
        raise ValueError(
            "Normalization frame list does not match the full train partition"
        )
    if normalization["manifest_sha256"] != manifest_hash:
        raise ValueError(
            "Normalization manifest hash does not match the training manifest"
        )
    if normalization["scene_split_sha256"] != scene_split_hash:
        raise ValueError(
            "Normalization split hash does not match the training split"
        )
    log_center = float(normalization["normalization"]["center"])
    log_scale = float(normalization["normalization"]["scale"])
    if not np.isfinite(log_center) or not np.isfinite(log_scale) or log_scale <= 0:
        raise ValueError(
            "Normalization center and scale must be finite with positive scale"
        )
    config = TrainConfig(
        mode=args.mode,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        seed=args.seed,
        point_count=args.point_count,
        confidence_threshold=args.confidence_threshold,
        eval_every=args.eval_every,
        max_eval_frames=args.max_eval_frames,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        log_center=log_center,
        log_scale=log_scale,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Run directory is not empty: {args.output}; use --overwrite explicitly"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    source_commit = args.source_commit or git_commit(repo)
    if source_commit is None:
        raise RuntimeError(
            "Source commit is unavailable; pass --source-commit for a reproducible run"
        )
    provenance = {
        "git_commit": source_commit,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": sha256(args.normalization_stats),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    (args.output / "config.json").write_text(
        json.dumps({"config": asdict(config), "provenance": provenance}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    axes = load_axes(args.data_root / "resources")
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
    evaluation_positions = selected_indices(
        len(validation_indices), min(config.max_eval_frames, len(validation_indices))
    )
    evaluation_indices = [
        validation_indices[position] for position in evaluation_positions
    ]
    model = CubeOccupancyNet(
        config.mode,
        torch.from_numpy(axes.doppler_mps),
        base_channels=config.base_channels,
        log_center=config.log_center,
        log_scale=config.log_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    print(
        json.dumps(
            {
                "parameters": parameter_count(model),
                "train_frames": len(train_indices),
                "validation_frames": len(validation_indices),
                "evaluation_frames": len(evaluation_indices),
                "provenance": provenance,
            },
            indent=2,
        ),
        flush=True,
    )
    best_chamfer = float("inf")
    log_path = args.output / "train_log.jsonl"
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = train_indices.copy()
        random.Random(config.seed + epoch).shuffle(order)
        epoch_losses = []
        for index in order:
            item = train_set[index]
            cube, occupancy = move_frame(item, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(cube)
                loss, _ = occupancy_loss(logits, occupancy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
            del item, cube, occupancy, logits, loss
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(epoch_losses)),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        should_evaluate = epoch == 1 or epoch % config.eval_every == 0
        if should_evaluate:
            metrics = evaluate(
                model,
                validation_set,
                evaluation_indices,
                axes,
                config,
                device,
            )
            record["validation"] = metrics
            metrics_path = args.output / f"metrics_epoch_{epoch:04d}.json"
            metrics_path.write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            chamfer = metrics["generated"]["chamfer_m"]["median"]
            if chamfer < best_chamfer:
                best_chamfer = chamfer
                save_checkpoint(
                    args.output / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    config,
                    provenance,
                )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(
            args.output / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            provenance,
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
