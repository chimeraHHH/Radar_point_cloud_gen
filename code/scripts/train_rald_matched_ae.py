#!/usr/bin/env python3
"""Train the K-Radar matched RaLD-style point latent autoencoder."""

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

from cube_dense.dataset import KRadarDenseTargetDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.rald_adapter import (  # noqa: E402
    decode_grid_topk,
    rald_occupancy_loss,
    sample_occupancy_queries,
    sample_target_points,
)
from cube_dense.training_io import (  # noqa: E402
    checkpoint_due,
    truncate_resume_artifacts,
)
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
    rae_indices_to_xyz,
)
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_matched import RaLDPointAutoencoder  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    ae_point_count: int
    latent_count: int
    model_dim: int
    latent_dim: int
    depth: int
    heads: int
    head_dim: int
    positive_query_count: int
    negative_query_count: int
    positive_label_semantics: str
    output_point_count: int
    query_chunk_size: int
    positive_loss_weight: float
    negative_loss_weight: float
    kl_weight: float
    eval_every: int
    checkpoint_every: int
    max_eval_frames: int
    train_limit: int | None
    validation_limit: int | None
    overfit_one_frame: bool


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


def frame_generator(
    device: torch.device, seed: int, epoch: int, sequence: int, radar_index: int
) -> torch.Generator:
    value = (
        seed
        + epoch * 10_000_019
        + sequence * 1_000_003
        + radar_index * 101
    )
    return torch.Generator(device=device).manual_seed(value)


def move_target(item: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        item["target_xyz_confidence"].to(device, non_blocking=True),
        item["target_rae_index"].to(device, non_blocking=True),
    )


def build_model(config: TrainConfig) -> RaLDPointAutoencoder:
    return RaLDPointAutoencoder(
        point_count=config.ae_point_count,
        latent_count=config.latent_count,
        model_dim=config.model_dim,
        latent_dim=config.latent_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
    )


@torch.inference_mode()
def evaluate(
    model: RaLDPointAutoencoder,
    dataset: KRadarDenseTargetDataset,
    indices: list[int],
    axes,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    model.eval()
    geometry_reports = []
    frames = []
    occupancy_losses = []
    kl_losses = []
    for dataset_index in indices:
        item = dataset[dataset_index]
        target, target_index = move_target(item, device)
        generator = frame_generator(
            device,
            config.seed,
            0,
            item["sequence"],
            item["radar_index"],
        )
        points = sample_target_points(
            target, axes, config.ae_point_count, generator
        ).unsqueeze(0)
        queries, labels = sample_occupancy_queries(
            target,
            target_index,
            axes,
            config.positive_query_count,
            config.negative_query_count,
            generator,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            posterior = model.encode(points)
            query_logits = model.decode(posterior.mean, queries.unsqueeze(0))[0]
            sampled_loss, terms = rald_occupancy_loss(
                query_logits,
                labels,
                posterior.kl(),
                config.positive_loss_weight,
                config.negative_loss_weight,
                config.kl_weight,
            )
            predicted_index, confidence = decode_grid_topk(
                model,
                posterior.mean,
                axes,
                point_count=config.output_point_count,
                query_chunk_size=config.query_chunk_size,
            )
        predicted_xyz = rae_indices_to_xyz(predicted_index, axes)
        report = geometry_report(
            predicted_xyz,
            target[:, :3],
            target_weight=target[:, 3],
        )
        geometry_reports.append(report)
        occupancy_losses.append(float(sampled_loss.item()))
        kl_losses.append(float(terms["kl"].item()))
        frames.append(
            {
                "sequence": item["sequence"],
                "radar_index": item["radar_index"],
                "generated": report,
                "sampled_occupancy_loss": float(sampled_loss.item()),
                "posterior_kl": float(terms["kl"].item()),
                "mean_output_confidence": float(confidence.mean().item()),
            }
        )
        del target, target_index, points, queries, labels, posterior
        del query_logits, predicted_index, predicted_xyz, confidence
    return {
        "generated": aggregate_geometry_reports(geometry_reports),
        "sampled_occupancy_loss_mean": float(np.mean(occupancy_losses)),
        "posterior_kl_mean": float(np.mean(kl_losses)),
        "frames": frames,
    }


def save_checkpoint(
    path: Path,
    model: RaLDPointAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config: TrainConfig,
    provenance: dict,
    record: dict | None,
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
        values.append((float(metrics["generated"]["chamfer_m"]["median"]), epoch))
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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--ae-point-count", type=int, default=2_048)
    parser.add_argument("--latent-count", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--positive-query-count", type=int, default=1_024)
    parser.add_argument("--negative-query-count", type=int, default=1_024)
    parser.add_argument(
        "--positive-label-semantics",
        choices=("binary_occupancy",),
        default="binary_occupancy",
    )
    parser.add_argument("--output-point-count", type=int, default=10_000)
    parser.add_argument("--query-chunk-size", type=int, default=8_192)
    parser.add_argument("--positive-loss-weight", type=float, default=0.1)
    parser.add_argument("--negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--overfit-one-frame", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpu-name")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("RaLD-style AE training requires CUDA")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    if args.required_gpu_name and device_name != args.required_gpu_name:
        raise RuntimeError(
            f"RaLD-style AE requires {args.required_gpu_name}, got {device_name}"
        )
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
    if args.overfit_one_frame:
        args.train_limit = 1
        args.validation_limit = 1
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        ae_point_count=args.ae_point_count,
        latent_count=args.latent_count,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        depth=args.depth,
        heads=args.heads,
        head_dim=args.head_dim,
        positive_query_count=args.positive_query_count,
        negative_query_count=args.negative_query_count,
        positive_label_semantics=args.positive_label_semantics,
        output_point_count=args.output_point_count,
        query_chunk_size=args.query_chunk_size,
        positive_loss_weight=args.positive_loss_weight,
        negative_loss_weight=args.negative_loss_weight,
        kl_weight=args.kl_weight,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        max_eval_frames=args.max_eval_frames,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        overfit_one_frame=args.overfit_one_frame,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    source_commit = args.source_commit or git_commit(Path(__file__).resolve().parents[2])
    if source_commit is None:
        raise RuntimeError("Source commit is required for reproducibility")
    axes = load_axes(args.data_root / "resources")
    train_set = KRadarDenseTargetDataset(
        args.cache_root, args.manifest, ("train",)
    )
    validation_set = KRadarDenseTargetDataset(
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
    evaluation_indices = [validation_indices[position] for position in evaluation_positions]
    model = build_model(config).to(device)
    provenance = {
        "git_commit": source_commit,
        "manifest_sha256": sha256(args.manifest),
        "scene_split_sha256": sha256(args.scene_split),
        "device": device_name,
        "torch_version": torch.__version__,
        "model_parameter_count": parameter_count(model),
        "external_pretraining": False,
        "upstream_rald_commit": "ffec4b41241391734b1eda5c093de843c909eb8e",
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
        if recorded != run_document:
            raise ValueError("Resume configuration or provenance does not match")
    else:
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(run_document, indent=2) + "\n", encoding="utf-8")
        temporary.replace(config_path)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    start_epoch = 1
    best_chamfer = float("inf")
    prior_elapsed_seconds = 0.0
    log_path = args.output / "train_log.jsonl"
    if args.resume:
        last = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        if last["config"] != asdict(config) or last["provenance"] != provenance:
            raise ValueError("Checkpoint does not match the requested run")
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        last_epoch = int(last["epoch"])
        start_epoch = last_epoch + 1
        best_chamfer, _ = best_recorded_chamfer(args.output, last_epoch)
        records = truncate_resume_artifacts(args.output, last_epoch)
        prior_elapsed_seconds = float(records[-1]["elapsed_seconds"])
    print(
        json.dumps(
            {
                "parameters": provenance["model_parameter_count"],
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
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        order = train_indices.copy()
        random.Random(config.seed + epoch).shuffle(order)
        losses = []
        term_values = {"positive_bce": [], "negative_bce": [], "kl": []}
        for dataset_index in order:
            item = train_set[dataset_index]
            target, target_index = move_target(item, device)
            generator = frame_generator(
                device,
                config.seed,
                epoch,
                item["sequence"],
                item["radar_index"],
            )
            points = sample_target_points(
                target, axes, config.ae_point_count, generator
            ).unsqueeze(0)
            queries, labels = sample_occupancy_queries(
                target,
                target_index,
                axes,
                config.positive_query_count,
                config.negative_query_count,
                generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, posterior = model(points, queries.unsqueeze(0))
                loss, terms = rald_occupancy_loss(
                    logits[0],
                    labels,
                    posterior.kl(),
                    config.positive_loss_weight,
                    config.negative_loss_weight,
                    config.kl_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            for name in term_values:
                term_values[name].append(float(terms[name].detach().item()))
            del target, target_index, points, queries, labels, logits, posterior, loss
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "train_terms": {
                name: float(np.mean(values)) for name, values in term_values.items()
            },
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": round(
                prior_elapsed_seconds + time.monotonic() - started, 3
            ),
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
            chamfer = float(metrics["generated"]["chamfer_m"]["median"])
            is_best = chamfer < best_chamfer
        else:
            is_best = False
        if checkpoint_due(
            epoch, config.epochs, config.checkpoint_every, should_evaluate
        ):
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

    best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    best_validation = evaluate(
        model,
        validation_set,
        validation_indices,
        axes,
        config,
        device,
    )
    best_report = {
        "best_epoch": int(best["epoch"]),
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
