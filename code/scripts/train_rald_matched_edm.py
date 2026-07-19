#!/usr/bin/env python3
"""Train the K-Radar native-grid radar-conditioned matched RaLD EDM."""

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

from cube_dense.dataset import KRadarRaLDLatentDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.rald_adapter import (  # noqa: E402
    decode_grid_topk,
    rae_sum_condition,
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
from models.rald_matched import (  # noqa: E402
    FullRAEDRadarTokenEncoder,
    RaLDEDMPreconditioner,
    RaLDPointAutoencoder,
    RadarTokenEncoder,
    edm_loss,
    edm_sample,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    model_dim: int
    depth: int
    heads: int
    head_dim: int
    radar_base_channels: int
    radar_encoded_channels: int
    radar_blocks_per_level: int
    condition_mode: str
    spectral_channels: int
    edm_steps: int
    output_point_count: int
    query_chunk_size: int
    eval_every: int
    checkpoint_every: int
    max_eval_frames: int
    train_limit: int | None
    validation_limit: int | None
    overfit_one_frame: bool
    normalization_center: float
    normalization_scale: float


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


def build_autoencoder(config: dict) -> RaLDPointAutoencoder:
    return RaLDPointAutoencoder(
        point_count=int(config["ae_point_count"]),
        latent_count=int(config["latent_count"]),
        model_dim=int(config["model_dim"]),
        latent_dim=int(config["latent_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
    )


def build_edm(config: TrainConfig, ae_config: dict) -> RaLDEDMPreconditioner:
    if config.condition_mode == "full_raed":
        radar_encoder = FullRAEDRadarTokenEncoder(
            log_center=config.normalization_center,
            log_scale=config.normalization_scale,
            spectral_channels=config.spectral_channels,
            encoded_shape=(16, 7, 3),
            encoded_channels=config.radar_encoded_channels,
            token_dim=config.model_dim,
            base_channels=config.radar_base_channels,
            channel_multipliers=(1, 1, 2, 2, 4),
            blocks_per_level=config.radar_blocks_per_level,
        )
    else:
        radar_encoder = RadarTokenEncoder(
            encoded_shape=(16, 7, 3),
            encoded_channels=config.radar_encoded_channels,
            token_dim=config.model_dim,
            base_channels=config.radar_base_channels,
            channel_multipliers=(1, 1, 2, 2, 4),
            blocks_per_level=config.radar_blocks_per_level,
        )
    return RaLDEDMPreconditioner(
        latent_count=int(ae_config["latent_count"]),
        latent_dim=int(ae_config["latent_dim"]),
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
        radar_encoder=radar_encoder,
    )


def frame_seed(seed: int, sequence: int, radar_index: int) -> int:
    return seed + sequence * 1_000_003 + radar_index * 101


def validate_normalization(
    path: Path, manifest_hash: str, scene_split_hash: str, condition_mode: str
) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if condition_mode == "rae_sum":
        if document.get("representation") != "log10_sum_doppler_power_plus_one":
            raise ValueError("RAE-Sum condition requires RAE-Sum normalization")
    elif "log10_power_plus_one" not in document:
        raise ValueError("Full-RAED condition requires per-Cube power normalization")
    if document["manifest_sha256"] != manifest_hash:
        raise ValueError("RAE-Sum normalization manifest mismatch")
    if document["scene_split_sha256"] != scene_split_hash:
        raise ValueError("RAE-Sum normalization scene-split mismatch")
    if document["partitions"] != ["train"] or document["frame_limit"] is not None:
        raise ValueError("Formal RAE-Sum normalization must cover all train frames")
    if int(document["frame_count"]) != 76:
        raise ValueError("Formal RAE-Sum normalization requires 76 train frames")
    return document


def prepare_condition(cube: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    if config.condition_mode == "full_raed":
        return cube
    return rae_sum_condition(
        cube, config.normalization_center, config.normalization_scale
    )


def validate_latent_cache(
    path: Path,
    manifest_hash: str,
    scene_split_hash: str,
    ae_checkpoint: Path,
) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["manifest_sha256"] != manifest_hash:
        raise ValueError("Latent cache manifest mismatch")
    if document["scene_split_sha256"] != scene_split_hash:
        raise ValueError("Latent cache scene-split mismatch")
    if set(document["partitions"]) != {"train", "validation"}:
        raise ValueError("Latent cache must contain train and validation only")
    if int(document["frame_count"]) != 100:
        raise ValueError("Latent cache requires all 100 development frames")
    if document["ae_checkpoint_sha256"] != sha256(ae_checkpoint):
        raise ValueError("Latent cache and AE checkpoint differ")
    return document


@torch.inference_mode()
def evaluate(
    model: RaLDEDMPreconditioner,
    autoencoder: RaLDPointAutoencoder,
    dataset: KRadarRaLDLatentDataset,
    indices: list[int],
    axes,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    model.eval()
    autoencoder.eval()
    reports = []
    frames = []
    latent_errors = []
    for dataset_index in indices:
        item = dataset[dataset_index]
        cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
        target_latent = item["latent_mean"].unsqueeze(0).to(
            device, non_blocking=True
        )
        target = item["target_xyz_confidence"].to(device, non_blocking=True)
        condition = prepare_condition(cube, config)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_latent = edm_sample(
                model,
                condition,
                [frame_seed(config.seed, item["sequence"], item["radar_index"])],
                steps=config.edm_steps,
            )
            predicted_index, confidence = decode_grid_topk(
                autoencoder,
                generated_latent,
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
        latent_rmse = float(
            (generated_latent.float() - target_latent).square().mean().sqrt().item()
        )
        reports.append(report)
        latent_errors.append(latent_rmse)
        frames.append(
            {
                "sequence": item["sequence"],
                "radar_index": item["radar_index"],
                "generated": report,
                "latent_rmse": latent_rmse,
                "mean_output_confidence": float(confidence.mean().item()),
            }
        )
        del cube, target_latent, target, condition, generated_latent
        del predicted_index, confidence, predicted_xyz
    return {
        "generated": aggregate_geometry_reports(reports),
        "latent_rmse_mean": float(np.mean(latent_errors)),
        "frames": frames,
    }


def save_checkpoint(
    path: Path,
    model: RaLDEDMPreconditioner,
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
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--ae-run", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--radar-base-channels", type=int, default=64)
    parser.add_argument("--radar-encoded-channels", type=int, default=16)
    parser.add_argument("--radar-blocks-per-level", type=int, default=2)
    parser.add_argument(
        "--condition-mode",
        choices=("rae_sum", "full_raed"),
        default="rae_sum",
    )
    parser.add_argument("--spectral-channels", type=int, default=16)
    parser.add_argument("--edm-steps", type=int, default=18)
    parser.add_argument("--output-point-count", type=int, default=10_000)
    parser.add_argument("--query-chunk-size", type=int, default=8_192)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=4)
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
        raise RuntimeError("Matched RaLD EDM training requires CUDA")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    if args.required_gpu_name and device_name != args.required_gpu_name:
        raise RuntimeError(
            f"Matched RaLD EDM requires {args.required_gpu_name}, got {device_name}"
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
    source_commit = args.source_commit or git_commit(Path(__file__).resolve().parents[2])
    if source_commit is None:
        raise RuntimeError("Source commit is required for reproducibility")
    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization = validate_normalization(
        args.normalization,
        manifest_hash,
        scene_split_hash,
        args.condition_mode,
    )
    center = float(normalization["normalization"]["center"])
    scale = float(normalization["normalization"]["scale"])
    ae_document = json.loads((args.ae_run / "config.json").read_text(encoding="utf-8"))
    ae_checkpoint_path = args.ae_run / "best.pt"
    latent_manifest_path = args.latent_root / "latent_cache_manifest.json"
    latent_manifest = validate_latent_cache(
        latent_manifest_path,
        manifest_hash,
        scene_split_hash,
        ae_checkpoint_path,
    )
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        head_dim=args.head_dim,
        radar_base_channels=args.radar_base_channels,
        radar_encoded_channels=args.radar_encoded_channels,
        radar_blocks_per_level=args.radar_blocks_per_level,
        condition_mode=args.condition_mode,
        spectral_channels=args.spectral_channels,
        edm_steps=args.edm_steps,
        output_point_count=args.output_point_count,
        query_chunk_size=args.query_chunk_size,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        max_eval_frames=args.max_eval_frames,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        overfit_one_frame=args.overfit_one_frame,
        normalization_center=center,
        normalization_scale=scale,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    autoencoder = build_autoencoder(ae_document["config"]).to(device)
    ae_checkpoint = torch.load(
        ae_checkpoint_path, map_location=device, weights_only=False
    )
    if ae_checkpoint["config"] != ae_document["config"]:
        raise ValueError("AE checkpoint configuration differs from run document")
    if ae_checkpoint["provenance"] != ae_document["provenance"]:
        raise ValueError("AE checkpoint provenance differs from run document")
    autoencoder.load_state_dict(ae_checkpoint["model"], strict=True)
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    model = build_edm(config, ae_document["config"]).to(device)
    axes = load_axes(args.data_root / "resources")
    train_set = KRadarRaLDLatentDataset(
        args.data_root,
        args.cache_root,
        args.latent_root,
        args.manifest,
        ("train",),
    )
    validation_set = KRadarRaLDLatentDataset(
        args.data_root,
        args.cache_root,
        args.latent_root,
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
    provenance = {
        "git_commit": source_commit,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": sha256(args.normalization),
        "latent_cache_manifest_sha256": sha256(latent_manifest_path),
        "ae_checkpoint_sha256": sha256(ae_checkpoint_path),
        "ae_source_commit": latent_manifest["ae_source_commit"],
        "device": device_name,
        "torch_version": torch.__version__,
        "model_parameter_count": parameter_count(model),
        "ae_parameter_count": parameter_count(autoencoder),
        "external_pretraining": False,
        "cfar_query_helper": False,
        "upstream_rald_commit": "ffec4b41241391734b1eda5c093de843c909eb8e",
        "condition_mode": config.condition_mode,
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
                "ae_parameters": provenance["ae_parameter_count"],
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
        for dataset_index in order:
            item = train_set[dataset_index]
            cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
            latent = item["latent_mean"].unsqueeze(0).to(device, non_blocking=True)
            condition = prepare_condition(cube, config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = edm_loss(model, latent, condition)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            del cube, latent, condition, loss
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": round(
                prior_elapsed_seconds + time.monotonic() - started, 3
            ),
        }
        should_evaluate = epoch == 1 or epoch % config.eval_every == 0
        if should_evaluate:
            metrics = evaluate(
                model,
                autoencoder,
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
        autoencoder,
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
