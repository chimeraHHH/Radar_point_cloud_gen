#!/usr/bin/env python3
"""Train the G3L-2 Full-RAED-conditioned EDM on frozen train latents."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_tesseract  # noqa: E402
from cube_dense.training_io import (  # noqa: E402
    checkpoint_due,
    truncate_resume_artifacts,
)
from models.cube_occupancy import parameter_count  # noqa: E402
from models.rald_matched import (  # noqa: E402
    FullRAEDRadarTokenEncoder,
    RaLDEDMPreconditioner,
    edm_loss,
)
from scripts.cache_rald_anchor_g3l_latents import (  # noqa: E402
    LATENT_SHAPE,
    PROTOCOL as LATENT_CACHE_PROTOCOL,
    latent_metadata,
    sha256,
    validate_data_contract,
    validate_g3l1_run,
    validate_latent_cache_manifest,
    validate_latent_record,
)


PROTOCOL = "rald_anchor_g3l2_full_raed_edm_training_v1"
OFFICIAL_RALD_COMMIT = "ffec4b41241391734b1eda5c093de843c909eb8e"
FORMAL_SEEDS = (20260716, 20260717, 20260718)
OFFICIAL_EPOCHS = 100
OFFICIAL_DENOISER_DEPTH = 24
OFFICIAL_EDM_STEPS = 18
OFFICIAL_P_MEAN = -1.2
OFFICIAL_P_STD = 1.2
OFFICIAL_SIGMA_DATA = 1.0
OFFICIAL_SIGMA_MIN = 0.002
OFFICIAL_SIGMA_MAX = 80.0
OFFICIAL_RHO = 7.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = OFFICIAL_EPOCHS
    learning_rate: float = 1e-4
    weight_decay: float = 0.02
    seed: int = FORMAL_SEEDS[0]
    latent_count: int = LATENT_SHAPE[0]
    latent_dim: int = LATENT_SHAPE[1]
    model_dim: int = 512
    depth: int = OFFICIAL_DENOISER_DEPTH
    heads: int = 8
    head_dim: int = 64
    radar_base_channels: int = 64
    radar_encoded_channels: int = 16
    radar_blocks_per_level: int = 2
    radar_spectral_channels: int = 16
    p_mean: float = OFFICIAL_P_MEAN
    p_std: float = OFFICIAL_P_STD
    sigma_data: float = OFFICIAL_SIGMA_DATA
    edm_steps: int = OFFICIAL_EDM_STEPS
    sigma_min: float = OFFICIAL_SIGMA_MIN
    sigma_max: float = OFFICIAL_SIGMA_MAX
    rho: float = OFFICIAL_RHO
    checkpoint_every: int = 5
    gradient_clip_norm: float = 10.0
    train_limit: int | None = None
    normalization_center: float = 0.0
    normalization_scale: float = 1.0
    condition_mode: str = "full_raed"
    inference_sampler: str = "heun"
    checkpoint_selection: str = "final_epoch_no_validation_selection"


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def official_schedule(config: TrainConfig) -> dict:
    return {
        "epochs": config.epochs,
        "denoiser_depth": config.depth,
        "p_mean": config.p_mean,
        "p_std": config.p_std,
        "sigma_data": config.sigma_data,
        "inference_steps": config.edm_steps,
        "sigma_min": config.sigma_min,
        "sigma_max": config.sigma_max,
        "rho": config.rho,
        "sampler": config.inference_sampler,
    }


def build_edm(
    config: TrainConfig,
    *,
    radar_encoder: FullRAEDRadarTokenEncoder | None = None,
) -> RaLDEDMPreconditioner:
    if (config.latent_count, config.latent_dim) != LATENT_SHAPE:
        raise ValueError("G3L-2 EDM requires the exact 512x32 latent state")
    if config.condition_mode != "full_raed":
        raise ValueError("G3L-2 EDM condition must be Full-RAED")
    if radar_encoder is None:
        radar_encoder = FullRAEDRadarTokenEncoder(
            log_center=config.normalization_center,
            log_scale=config.normalization_scale,
            spectral_channels=config.radar_spectral_channels,
            encoded_shape=(16, 7, 3),
            encoded_channels=config.radar_encoded_channels,
            token_dim=config.model_dim,
            base_channels=config.radar_base_channels,
            channel_multipliers=(1, 1, 2, 2, 4),
            blocks_per_level=config.radar_blocks_per_level,
        )
    if not isinstance(radar_encoder, FullRAEDRadarTokenEncoder):
        raise TypeError("G3L-2 requires FullRAEDRadarTokenEncoder")
    return RaLDEDMPreconditioner(
        latent_count=config.latent_count,
        latent_dim=config.latent_dim,
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
        radar_encoder=radar_encoder,
        sigma_data=config.sigma_data,
    )


def gradient_norm(parameters) -> float:
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return float(torch.sqrt(sum(value.square().sum() for value in gradients)).item())


def gradient_audit(model: RaLDEDMPreconditioner) -> dict[str, float]:
    output = list(model.denoiser.output.parameters())
    output_ids = {id(parameter) for parameter in output}
    condition = list(model.radar_encoder.parameters())
    condition_ids = {id(parameter) for parameter in condition}
    backbone = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in output_ids and id(parameter) not in condition_ids
    ]
    return {
        "denoiser_output": gradient_norm(output),
        "denoiser_backbone": gradient_norm(backbone),
        "full_raed_condition_encoder": gradient_norm(condition),
    }


def validate_two_step_gradient_audit(records: list[dict]) -> None:
    if len(records) < 2:
        raise ValueError("G3L-2 gradient audit requires the first two optimizer steps")
    if records[0]["gradients"]["denoiser_output"] <= 0.0:
        raise RuntimeError("First EDM step did not reach the zero-initialized output")
    if records[1]["gradients"]["full_raed_condition_encoder"] <= 0.0:
        raise RuntimeError("Second EDM step did not reach the Full-RAED condition encoder")


def selected_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    if limit <= 0:
        raise ValueError("Train limit must be positive")
    return np.linspace(0, length - 1, limit).round().astype(int).tolist()


def require_cache_source(configuration: dict, expected_source_commit: str) -> None:
    if configuration.get("cache_source_commit") != expected_source_commit:
        raise ValueError("G3L-2 latent cache source mismatch")


def validate_training_inputs(
    *,
    manifest_path: Path,
    scene_split_path: Path,
    normalization_path: Path,
    g3l1_run: Path,
    g3l1_report: Path,
    latent_manifest_path: Path,
    g3l1_source_commit: str,
    cache_source_commit: str,
) -> dict:
    data = validate_data_contract(
        manifest_path, scene_split_path, normalization_path
    )
    g3l1 = validate_g3l1_run(
        g3l1_run,
        g3l1_report,
        manifest_hash=data["manifest_sha256"],
        scene_split_hash=data["scene_split_sha256"],
        normalization_hash=data["normalization_sha256"],
        g3l1_source_commit=g3l1_source_commit,
    )
    latent_manifest = validate_latent_cache_manifest(
        latent_manifest_path, validate_files=False
    )
    configuration = latent_manifest["configuration"]
    require_cache_source(configuration, cache_source_commit)
    required_values = {
        "protocol": LATENT_CACHE_PROTOCOL,
        "manifest_sha256": data["manifest_sha256"],
        "scene_split_sha256": data["scene_split_sha256"],
        "normalization_sha256": data["normalization_sha256"],
        "g3l1_report_sha256": g3l1["g3l1_report_sha256"],
        "g3l1_config_sha256": g3l1["g3l1_config_sha256"],
        "g3l1_checkpoint_sha256": g3l1["g3l1_checkpoint_sha256"],
        "g3l1_source_commit": g3l1_source_commit,
        "parent_config_sha256": g3l1["parent_config_sha256"],
        "parent_checkpoint_sha256": g3l1["parent_checkpoint_sha256"],
        "cfar_query_helper": False,
        "best_of_k": False,
        "test_accessed": False,
    }
    for key, expected in required_values.items():
        if configuration.get(key) != expected:
            raise ValueError(f"G3L-2 latent cache {key} mismatch")
    latent_manifest = validate_latent_cache_manifest(
        latent_manifest_path,
        expected_configuration=configuration,
        validate_files=True,
    )
    manifest_train_keys = {
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in data["train_frames"]
    }
    latent_keys = {
        (int(record["sequence"]), int(record["radar_index"]))
        for record in latent_manifest["records"]
    }
    if latent_keys != manifest_train_keys:
        raise ValueError("G3L-2 latent cache does not exactly cover train frames")
    return {"data": data, "g3l1": g3l1, "latent_manifest": latent_manifest}


class G3LTrainLatentDataset:
    """Load each train Cube with a hash-verified frozen posterior mean."""

    def __init__(self, data_root: Path, manifest_path: Path, document: dict) -> None:
        self.data_root = data_root
        self.root = manifest_path.resolve().parent
        self.records = document["records"]
        self.metadata = latent_metadata(document["configuration"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sequence = int(record["sequence"])
        radar_index = int(record["radar_index"])
        latent = validate_latent_record(self.root, record, self.metadata)
        cube = load_tesseract(
            self.data_root
            / str(sequence)
            / "radar_tesseract"
            / f"tesseract_{radar_index:05d}.mat"
        ).astype(np.float32, copy=False)
        if cube.ndim != 4 or cube.shape[0] != 64 or not np.isfinite(cube).all():
            raise ValueError(f"Invalid Full-RAED Cube for {sequence}:{radar_index}")
        return {
            "cube_drae": torch.from_numpy(cube),
            "latent_mean": torch.from_numpy(latent),
            "sequence": sequence,
            "radar_index": radar_index,
        }


def save_checkpoint(
    path: Path,
    model: RaLDEDMPreconditioner,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    epoch: int,
    update_count: int,
    config: TrainConfig,
    provenance: dict,
    gradient_records: list[dict],
    record: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "update_count": update_count,
            "config": asdict(config),
            "provenance": provenance,
            "gradient_audit": gradient_records,
            "record": record,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        },
        temporary,
    )
    temporary.replace(path)


def require_h200_cuda(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("G3L-2 EDM training requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G3L-2 EDM training is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G3L-2 EDM training requires H200, got {resolved}")
    return device, resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--g3l1-run", type=Path, required=True)
    parser.add_argument("--g3l1-report", type=Path, required=True)
    parser.add_argument("--g3l1-source-commit", required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--cache-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=OFFICIAL_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=FORMAL_SEEDS[0])
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=OFFICIAL_DENOISER_DEPTH)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--radar-base-channels", type=int, default=64)
    parser.add_argument("--radar-encoded-channels", type=int, default=16)
    parser.add_argument("--radar-blocks-per-level", type=int, default=2)
    parser.add_argument("--radar-spectral-channels", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    output_nonempty = args.output.exists() and any(args.output.iterdir())
    if output_nonempty and args.overwrite:
        shutil.rmtree(args.output)
        output_nonempty = False
    if output_nonempty and not args.resume:
        raise FileExistsError(
            f"Run directory is not empty: {args.output}; use --resume or --overwrite"
        )
    if args.resume and not output_nonempty:
        raise FileNotFoundError(f"No G3L-2 EDM run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    device, gpu_name = require_h200_cuda(args.device)
    source_commit = args.source_commit or git_commit(Path(__file__).resolve().parents[2])
    if source_commit is None:
        raise RuntimeError("EDM source commit is required")
    latent_manifest_path = args.latent_root / "g3l2_latent_cache_manifest.json"
    inputs = validate_training_inputs(
        manifest_path=args.manifest,
        scene_split_path=args.scene_split,
        normalization_path=args.normalization,
        g3l1_run=args.g3l1_run,
        g3l1_report=args.g3l1_report,
        latent_manifest_path=latent_manifest_path,
        g3l1_source_commit=args.g3l1_source_commit,
        cache_source_commit=args.cache_source_commit,
    )
    normalization = inputs["data"]["normalization"]
    center = float(normalization["normalization"]["center"])
    scale = float(normalization["normalization"]["scale"])
    config = replace(
        TrainConfig(),
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
        radar_spectral_channels=args.radar_spectral_channels,
        checkpoint_every=args.checkpoint_every,
        train_limit=args.train_limit,
        normalization_center=center,
        normalization_scale=scale,
    )
    if config.seed != int(inputs["g3l1"]["config"]["seed"]):
        raise ValueError("G3L-2 seed must match its passing G3L-1 VAE seed")
    if config.epochs <= 0 or config.checkpoint_every <= 0:
        raise ValueError("Epoch and checkpoint cadence must be positive")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    model = build_edm(config).to(device)
    dataset = G3LTrainLatentDataset(
        args.data_root, latent_manifest_path, inputs["latent_manifest"]
    )
    train_indices = selected_indices(len(dataset), config.train_limit)
    provenance = {
        "git_commit": source_commit,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": inputs["data"]["manifest_sha256"],
        "scene_split": str(args.scene_split.resolve()),
        "scene_split_sha256": inputs["data"]["scene_split_sha256"],
        "normalization": str(args.normalization.resolve()),
        "normalization_sha256": inputs["data"]["normalization_sha256"],
        "latent_cache_manifest": str(latent_manifest_path.resolve()),
        "latent_cache_manifest_sha256": sha256(latent_manifest_path),
        "latent_cache_source_commit": args.cache_source_commit,
        "g3l1_report": str(inputs["g3l1"]["report_path"]),
        "g3l1_report_sha256": inputs["g3l1"]["g3l1_report_sha256"],
        "g3l1_config": str(inputs["g3l1"]["config_path"]),
        "g3l1_config_sha256": inputs["g3l1"]["g3l1_config_sha256"],
        "g3l1_checkpoint": str(inputs["g3l1"]["checkpoint_path"]),
        "g3l1_checkpoint_sha256": inputs["g3l1"]["g3l1_checkpoint_sha256"],
        "g3l1_source_commit": args.g3l1_source_commit,
        "parent_config": str(inputs["g3l1"]["parent_config"]),
        "parent_config_sha256": inputs["g3l1"]["parent_config_sha256"],
        "parent_checkpoint": str(inputs["g3l1"]["parent_checkpoint"]),
        "parent_checkpoint_sha256": inputs["g3l1"]["parent_checkpoint_sha256"],
        "device": gpu_name,
        "torch_version": torch.__version__,
        "model_parameter_count": parameter_count(model),
        "official_rald_commit": OFFICIAL_RALD_COMMIT,
        "official_schedule": official_schedule(config),
        "formal_seed_set": list(FORMAL_SEEDS),
        "condition_mode": "full_raed",
        "partitions": ["train"],
        "test_accessed": False,
        "external_pretraining": False,
        "cfar_query_helper": False,
        "best_of_k": False,
    }
    run_document = {
        "protocol": PROTOCOL,
        "config": asdict(config),
        "provenance": provenance,
    }
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("Resume configuration or provenance does not match")
    else:
        atomic_json(config_path, run_document)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    start_epoch = 1
    update_count = 0
    gradient_records: list[dict] = []
    prior_elapsed_seconds = 0.0
    log_path = args.output / "train_log.jsonl"
    if args.resume:
        checkpoint = torch.load(
            args.output / "last.pt", map_location=device, weights_only=False
        )
        if checkpoint.get("config") != asdict(config):
            raise ValueError("Resume checkpoint configuration differs")
        if checkpoint.get("provenance") != provenance:
            raise ValueError("Resume checkpoint provenance differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        update_count = int(checkpoint["update_count"])
        gradient_records = list(checkpoint["gradient_audit"])
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, dict):
            raise ValueError("Resume checkpoint lacks RNG state")
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state_all(rng_state["cuda"])
        retained = truncate_resume_artifacts(args.output, start_epoch - 1)
        prior_elapsed_seconds = float(retained[-1]["elapsed_seconds"])

    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "parameters": provenance["model_parameter_count"],
                "train_frames": len(train_indices),
                "start_epoch": start_epoch,
                "schedule": provenance["official_schedule"],
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
            item = dataset[dataset_index]
            cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
            latent = item["latent_mean"].unsqueeze(0).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = edm_loss(
                    model,
                    latent,
                    cube,
                    p_mean=config.p_mean,
                    p_std=config.p_std,
                )
            loss.backward()
            update_count += 1
            if update_count <= 2:
                gradient_records.append(
                    {
                        "update": update_count,
                        "sequence": item["sequence"],
                        "radar_index": item["radar_index"],
                        "gradients": gradient_audit(model),
                    }
                )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            if update_count == 2:
                validate_two_step_gradient_audit(gradient_records)
            losses.append(float(loss.detach().item()))
            del cube, latent, loss
        scheduler.step()
        record = {
            "epoch": epoch,
            "update_count": update_count,
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_median": float(np.median(losses)),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": round(
                prior_elapsed_seconds + time.monotonic() - started, 3
            ),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if checkpoint_due(
            epoch, config.epochs, config.checkpoint_every, evaluated=False
        ):
            save_checkpoint(
                args.output / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                update_count=update_count,
                config=config,
                provenance=provenance,
                gradient_records=gradient_records,
                record=record,
            )
        print(json.dumps(record), flush=True)

    last_path = args.output / "last.pt"
    if not last_path.is_file():
        raise FileNotFoundError(last_path)
    final_path = args.output / "final.pt"
    if final_path.exists() and not args.resume:
        raise FileExistsError(final_path)
    shutil.copy2(last_path, final_path)
    final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    if int(final_checkpoint["epoch"]) != config.epochs:
        raise RuntimeError("G3L-2 final checkpoint is not the configured final epoch")
    if final_checkpoint["provenance"] != provenance:
        raise RuntimeError("G3L-2 final checkpoint provenance differs")
    validate_two_step_gradient_audit(final_checkpoint["gradient_audit"])
    summary = {
        "protocol": PROTOCOL,
        "status": "completed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "epochs": config.epochs,
        "update_count": int(final_checkpoint["update_count"]),
        "checkpoint_selection": config.checkpoint_selection,
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": sha256(final_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "gradient_audit": final_checkpoint["gradient_audit"],
        "official_schedule": official_schedule(config),
        "test_accessed": False,
        "cfar_query_helper": False,
        "best_of_k": False,
    }
    atomic_json(args.output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
