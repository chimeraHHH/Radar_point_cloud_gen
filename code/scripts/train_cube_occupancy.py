#!/usr/bin/env python3
"""Train matched RAE-Max or Full-RAED frustum-occupancy baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
    occupancy_to_points,
)
from losses.occupancy import occupancy_loss  # noqa: E402
from models.cube_occupancy import (  # noqa: E402
    CubeOccupancyNet,
    parameter_count,
    spectral_diagnostics,
    spectral_gradient_norm,
)


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
    overfit_one_frame: bool
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
    record: dict | None = None,
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
            "record": record,
        },
        temporary,
    )
    temporary.replace(path)


def best_recorded_chamfer(output: Path, maximum_epoch: int) -> tuple[float, int]:
    values = []
    for path in sorted(output.glob("metrics_epoch_*.json")):
        epoch = int(path.stem.rsplit("_", maxsplit=1)[1])
        if epoch > maximum_epoch:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        values.append(
            (float(metrics["generated"]["chamfer_m"]["median"]), epoch)
        )
    if not values:
        raise ValueError("Resume run has no recorded validation metrics")
    return min(values)


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
    parser.add_argument("--overfit-one-frame", action="store_true")
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    if args.overfit_one_frame:
        args.train_limit = 1
        args.validation_limit = 1
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
        overfit_one_frame=args.overfit_one_frame,
        log_center=log_center,
        log_scale=log_scale,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    output_nonempty = args.output.exists() and any(args.output.iterdir())
    if output_nonempty and args.overwrite:
        shutil.rmtree(args.output)
        output_nonempty = False
    if output_nonempty and not args.resume:
        raise FileExistsError(
            f"Run directory is not empty: {args.output}; use --resume or --overwrite"
        )
    if args.resume and not output_nonempty:
        raise FileNotFoundError(f"No existing run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    source_commit = args.source_commit or git_commit(repo)
    if source_commit is None:
        raise RuntimeError(
            "Source commit is unavailable; pass --source-commit for a reproducible run"
        )
    axes = load_axes(args.data_root / "resources")
    model = CubeOccupancyNet(
        config.mode,
        torch.from_numpy(axes.doppler_mps),
        base_channels=config.base_channels,
        log_center=config.log_center,
        log_scale=config.log_scale,
    ).to(device)
    model_parameter_count = parameter_count(model)
    provenance = {
        "git_commit": source_commit,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": sha256(args.normalization_stats),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "model_parameter_count": model_parameter_count,
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
        if recorded != run_document:
            raise ValueError("Resume configuration or provenance does not match the run")
    else:
        temporary_config = config_path.with_suffix(".json.tmp")
        temporary_config.write_text(
            json.dumps(run_document, indent=2) + "\n", encoding="utf-8"
        )
        temporary_config.replace(config_path)
    train_set = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("train",)
    )
    validation_set = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train",) if config.overfit_one_frame else ("validation",),
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    start_epoch = 1
    best_chamfer = float("inf")
    prior_elapsed_seconds = 0.0
    if args.resume:
        last_checkpoint = torch.load(
            args.output / "last.pt", map_location=device, weights_only=False
        )
        if last_checkpoint["config"] != asdict(config):
            raise ValueError("Last checkpoint configuration does not match the run")
        if last_checkpoint["provenance"] != provenance:
            raise ValueError("Last checkpoint provenance does not match the run")
        model.load_state_dict(last_checkpoint["model"], strict=True)
        optimizer.load_state_dict(last_checkpoint["optimizer"])
        scheduler.load_state_dict(last_checkpoint["scheduler"])
        last_epoch = int(last_checkpoint["epoch"])
        start_epoch = last_epoch + 1
        best_chamfer, best_epoch = best_recorded_chamfer(args.output, last_epoch)
        best_path = args.output / "best.pt"
        recorded_best_epoch = None
        if best_path.exists():
            recorded_best = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            recorded_best_epoch = int(recorded_best["epoch"])
        if recorded_best_epoch != best_epoch:
            if best_epoch != last_epoch:
                raise ValueError("Best checkpoint and recorded validation metrics differ")
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                last_epoch,
                config,
                provenance,
                last_checkpoint.get("record"),
            )
        log_path = args.output / "train_log.jsonl"
        log_records = [
            json.loads(line)
            for line in (
                log_path.read_text(encoding="utf-8").splitlines()
                if log_path.exists()
                else []
            )
            if line.strip()
        ]
        logged_epoch = int(log_records[-1]["epoch"]) if log_records else 0
        if logged_epoch != last_epoch:
            checkpoint_record = last_checkpoint.get("record")
            if (
                checkpoint_record is None
                or logged_epoch != last_epoch - 1
                or int(checkpoint_record["epoch"]) != last_epoch
            ):
                raise ValueError("Training log and last checkpoint epochs differ")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(checkpoint_record) + "\n")
            log_records.append(checkpoint_record)
        prior_elapsed_seconds = float(log_records[-1]["elapsed_seconds"])
    print(
        json.dumps(
            {
                "parameters": model_parameter_count,
                "train_frames": len(train_indices),
                "validation_frames": len(validation_indices),
                "evaluation_frames": len(evaluation_indices),
                "start_epoch": start_epoch,
                "provenance": provenance,
            },
            indent=2,
        ),
        flush=True,
    )
    log_path = args.output / "train_log.jsonl"
    started = time.monotonic()
    first_step_spectral_gradient_norm = None
    first_step_spectral_gradient_recorded = start_epoch > 1
    if args.resume and log_records:
        first_step_spectral_gradient_norm = log_records[0].get(
            "spectral_diagnostics", {}
        ).get("first_step_gradient_norm")
    for epoch in range(start_epoch, config.epochs + 1):
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
            if not first_step_spectral_gradient_recorded:
                first_step_spectral_gradient_norm = spectral_gradient_norm(model)
                first_step_spectral_gradient_recorded = True
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
            del item, cube, occupancy, logits, loss
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(epoch_losses)),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": round(
                prior_elapsed_seconds + time.monotonic() - started, 3
            ),
            "spectral_diagnostics": {
                **spectral_diagnostics(model),
                "first_step_gradient_norm": first_step_spectral_gradient_norm,
            },
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
            is_best = chamfer < best_chamfer
        else:
            is_best = False
        save_checkpoint(
            args.output / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            provenance,
            record,
        )
        if is_best:
            best_chamfer = chamfer
            save_checkpoint(
                args.output / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                provenance,
                record,
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    best_checkpoint = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    best_validation = evaluate(
        model,
        validation_set,
        validation_indices,
        axes,
        config,
        device,
    )
    best_report = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "selection_metric": "generated.chamfer_m.median",
        "selection_value": best_chamfer,
        "validation": best_validation,
    }
    (args.output / "best_validation_metrics.json").write_text(
        json.dumps(best_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_validation": best_report}), flush=True)


if __name__ == "__main__":
    main()
