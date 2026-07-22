#!/usr/bin/env python3
"""Train the G3L-1 physical VAE on a frozen passing G3R geometry parent."""

from __future__ import annotations

import argparse
import copy
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
from cube_dense.rald_run import build_single_frame_rald, load_rald_run  # noqa: E402
from losses.rald_anchor import anchor_refinement_loss  # noqa: E402
from models.rald_anchor_ldm_refiner import RaLDAnchorLDMRefiner  # noqa: E402
from models.rald_anchor_ldm import RaLDAnchorLDM  # noqa: E402
from scripts.train_cube_doppler import move_frame, selected_indices, sha256  # noqa: E402


SELECTION_PROTOCOL = "single_posterior_mean_validation_v1"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TrainConfig:
    protocol: str
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    point_count: int
    geometry_weight: float
    doppler_weight: float
    existence_weight: float
    cycle_weight: float
    offset_weight: float
    cycle_variant: str
    kl_weight: float
    kl_warmup_epochs: int
    eval_every: int
    max_eval_frames: int
    train_limit: int | None
    validation_limit: int | None
    latent_count: int
    latent_dim: int
    model_dim: int
    decoder_depth: int
    denoiser_depth: int
    heads: int
    head_dim: int
    edm_steps: int
    training_posterior_path: str
    validation_posterior_path: str
    selection_protocol: str


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("G3L-1 physical VAE training requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G3L-1 physical VAE training is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G3L-1 requires an H200, got {resolved}")
    return device, resolved


def kl_warmup_weight(epoch: int, maximum: float, warmup_epochs: int) -> float:
    """Fixed epoch schedule: zero at epoch one and full by the warmup boundary."""

    if epoch <= 0:
        raise ValueError("KL warmup epoch must be positive")
    if maximum < 0.0 or warmup_epochs < 0:
        raise ValueError("KL weight and warmup length must be non-negative")
    if warmup_epochs == 0:
        return maximum
    if warmup_epochs == 1:
        return maximum
    progress = min(max((epoch - 1) / (warmup_epochs - 1), 0.0), 1.0)
    return maximum * progress


def gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(sum(value.square().sum() for value in values)).item())


def gradient_audit(model: RaLDAnchorLDMRefiner) -> dict[str, float]:
    return {
        "posterior_encoder": gradient_norm(model.ldm.posterior_encoder.parameters()),
        "anchor_decoder": gradient_norm(model.ldm.decoder.parameters()),
        "physical_head": gradient_norm(model.physical_head.parameters()),
        "frozen_g3r_parent": gradient_norm(model.geometry_parent.parameters()),
        "frozen_edm": gradient_norm(model.ldm.edm.parameters()),
    }


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
        raise ValueError("G3L-1 checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_g3r_parent(
    summary_path: Path,
    g3r_source_commit: str,
    seed: int,
    artifact_hashes: dict[str, str],
) -> tuple[dict, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "g3r_passed"
        or summary.get("source_commit") != g3r_source_commit
        or summary.get("selected_arm") != "full"
        or summary.get("g3r_decision", {}).get("g3r_passed") is not True
    ):
        raise ValueError("G3L-1 requires a passing, source-matched G3R full summary")

    seed_key = str(seed)
    selected_runs = summary.get("selected_runs", {})
    selected_hashes = summary.get("selected_run_hashes", {})
    if seed_key not in selected_runs or seed_key not in selected_hashes:
        raise ValueError(f"G3R summary does not select seed {seed}")
    run_path = Path(selected_runs[seed_key]).resolve()
    run_hashes = selected_hashes[seed_key]
    if set(run_hashes) != {"config_sha256", "best_checkpoint_sha256"}:
        raise ValueError("G3R selected-run hash contract is incomplete")
    config_path = run_path / "config.json"
    checkpoint_path = run_path / "best.pt"
    if sha256(config_path) != run_hashes["config_sha256"]:
        raise ValueError("Selected G3R config hash differs from its gate summary")
    if sha256(checkpoint_path) != run_hashes["best_checkpoint_sha256"]:
        raise ValueError("Selected G3R checkpoint hash differs from its gate summary")

    comparison_path = Path(summary["g3r_comparison"]).resolve()
    if sha256(comparison_path) != summary["g3r_comparison_sha256"]:
        raise ValueError("G3R comparison hash differs from the queue summary")
    run = load_rald_run(run_path, expected_variant="full")
    config = run["config"]
    provenance = run["provenance"]
    if (
        int(config["seed"]) != seed
        or config["doppler_head_mode"] != "distribution"
        or provenance["git_commit"] != g3r_source_commit
    ):
        raise ValueError("Selected G3R run differs from the requested seed or source")
    if any(provenance.get(key) != value for key, value in artifact_hashes.items()):
        raise ValueError("G3L-1 data artifacts differ from the selected G3R run")
    if sha256(run["parent_checkpoint"]) != provenance["parent_g1_checkpoint_sha256"]:
        raise ValueError("Selected G3R geometry-parent checkpoint hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != config or checkpoint.get("provenance") != provenance:
        raise ValueError("Selected G3R checkpoint metadata differs from config.json")
    return summary, run


def build_model(
    run: dict,
    axes,
    config: TrainConfig,
    device: torch.device,
) -> RaLDAnchorLDMRefiner:
    parent = build_single_frame_rald(run, axes, device)
    if int(run["config"]["model_dim"]) != config.model_dim:
        raise ValueError("G3L and selected G3R model dimensions differ")
    if int(run["config"]["point_count"]) != config.point_count:
        raise ValueError("G3L and selected G3R point counts differ")
    ldm = RaLDAnchorLDM(
        anchor_feature_dim=int(parent.parent.head.in_channels),
        latent_count=config.latent_count,
        latent_dim=config.latent_dim,
        model_dim=config.model_dim,
        decoder_depth=config.decoder_depth,
        denoiser_depth=config.denoiser_depth,
        heads=config.heads,
        head_dim=config.head_dim,
        edm_steps=config.edm_steps,
        radar_encoder=copy.deepcopy(parent.radar_encoder),
        detach_parent=True,
    )
    model = RaLDAnchorLDMRefiner(
        parent,
        ldm,
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        model_dim=config.model_dim,
    ).to(device)
    if any(parameter.requires_grad for parameter in model.geometry_parent.parameters()):
        raise RuntimeError("G3L construction failed to freeze the G3R parent")
    return model


def objective_for_frame(
    model: RaLDAnchorLDMRefiner,
    cube: torch.Tensor,
    target: torch.Tensor,
    target_index: torch.Tensor,
    config: TrainConfig,
    *,
    sample_posterior: bool,
    kl_weight: float,
) -> tuple[dict, object, torch.Tensor, torch.Tensor]:
    output = model(
        cube,
        target_index,
        target[:, 3],
        sample_posterior=sample_posterior,
    )
    physical = anchor_refinement_loss(
        output,
        cube,
        target,
        target_index,
        geometry_weight=config.geometry_weight,
        doppler_weight=config.doppler_weight,
        existence_weight=config.existence_weight,
        cycle_weight=config.cycle_weight,
        offset_weight=config.offset_weight,
        cycle_variant=config.cycle_variant,
    )
    kl = output["posterior_kl"].float().mean()
    total = physical.total + kl_weight * kl
    return output, physical, kl, total


@torch.inference_mode()
def evaluate(
    model: RaLDAnchorLDMRefiner,
    dataset: KRadarCubeDataset,
    frame_indices: list[int],
    device: torch.device,
    config: TrainConfig,
) -> dict:
    model.eval()
    totals = []
    physical_totals = []
    kls = []
    posterior_variances = []
    components: dict[str, list[float]] = {}
    frames = []
    for index in frame_indices:
        item = dataset[index]
        cube, _ = move_frame(item, device)
        target = item["target_xyz_confidence"].to(device)
        target_index = item["target_rae_index"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, physical, kl, total = objective_for_frame(
                model,
                cube,
                target,
                target_index,
                config,
                sample_posterior=False,
                kl_weight=config.kl_weight,
            )
        total_value = float(total.float().item())
        physical_value = float(physical.total.float().item())
        kl_value = float(kl.float().item())
        variance = output["posterior_log_variance"].float().exp()
        totals.append(total_value)
        physical_totals.append(physical_value)
        kls.append(kl_value)
        posterior_variances.append(float(variance.mean().item()))
        frame_components = {
            name: float(value.float().item())
            for name, value in physical.components.items()
        }
        for name, value in frame_components.items():
            components.setdefault(name, []).append(value)
        frames.append(
            {
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
                "vae_total": total_value,
                "physical_total": physical_value,
                "posterior_kl": kl_value,
                "posterior_variance_mean": posterior_variances[-1],
                "components": frame_components,
            }
        )
        del item, cube, target, target_index, output, physical, kl, total
        torch.cuda.empty_cache()
    return {
        "frame_count": len(frame_indices),
        "selection_protocol": SELECTION_PROTOCOL,
        "posterior_path": "mean",
        "vae_total_mean": float(np.mean(totals)),
        "physical_total_mean": float(np.mean(physical_totals)),
        "posterior_kl_mean": float(np.mean(kls)),
        "posterior_variance_mean": float(np.mean(posterior_variances)),
        "components_mean": {
            name: float(np.mean(values)) for name, values in components.items()
        },
        "frames": frames,
    }


def save_checkpoint(
    path: Path,
    model: RaLDAnchorLDMRefiner,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config: TrainConfig,
    provenance: dict,
    gradient_steps: list[dict],
    record: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "g3l_vae": model.vae_state_dict(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--g3r-summary", type=Path, required=True)
    parser.add_argument("--g3r-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--doppler-weight", type=float, default=1.0)
    parser.add_argument("--existence-weight", type=float, default=0.1)
    parser.add_argument("--cycle-weight", type=float, default=0.1)
    parser.add_argument("--offset-weight", type=float, default=0.01)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--latent-count", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--decoder-depth", type=int, default=24)
    parser.add_argument("--denoiser-depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--edm-steps", type=int, default=18)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if not SOURCE_PATTERN.fullmatch(args.source_commit):
        raise ValueError("G3L source commit must be a full lowercase Git SHA")
    if not SOURCE_PATTERN.fullmatch(args.g3r_source_commit):
        raise ValueError("G3R source commit must be a full lowercase Git SHA")
    if args.epochs <= 0 or args.eval_every <= 0 or args.max_eval_frames <= 0:
        raise ValueError("Epoch and evaluation counts must be positive")
    if args.latent_count <= 0 or args.latent_dim <= 0:
        raise ValueError("G3L latent dimensions must be positive")
    if args.kl_weight < 0.0 or args.kl_warmup_epochs < 0:
        raise ValueError("KL weight and warmup must be non-negative")
    device, device_name = require_h200(args.device)

    nonempty = args.output.exists() and any(args.output.iterdir())
    if nonempty and args.overwrite:
        shutil.rmtree(args.output)
        nonempty = False
    if nonempty and not args.resume:
        raise FileExistsError(f"G3L-1 output is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No G3L-1 run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    artifact_hashes = {
        "manifest_sha256": sha256(args.manifest),
        "scene_split_sha256": sha256(args.scene_split),
        "normalization_sha256": sha256(args.normalization_stats),
    }
    summary, selected_run = validate_g3r_parent(
        args.g3r_summary,
        args.g3r_source_commit,
        args.seed,
        artifact_hashes,
    )
    g3r_config = selected_run["config"]
    config = TrainConfig(
        protocol="rald_anchor_g3l_physical_vae_v1",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        point_count=int(g3r_config["point_count"]),
        geometry_weight=args.geometry_weight,
        doppler_weight=args.doppler_weight,
        existence_weight=args.existence_weight,
        cycle_weight=args.cycle_weight,
        offset_weight=args.offset_weight,
        cycle_variant="full",
        kl_weight=args.kl_weight,
        kl_warmup_epochs=args.kl_warmup_epochs,
        eval_every=args.eval_every,
        max_eval_frames=args.max_eval_frames,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        latent_count=args.latent_count,
        latent_dim=args.latent_dim,
        model_dim=int(g3r_config["model_dim"]),
        decoder_depth=args.decoder_depth,
        denoiser_depth=args.denoiser_depth,
        heads=args.heads,
        head_dim=args.head_dim,
        edm_steps=args.edm_steps,
        training_posterior_path="one_reparameterized_sample",
        validation_posterior_path="posterior_mean",
        selection_protocol=SELECTION_PROTOCOL,
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    axes = load_axes(args.data_root / "resources")
    model = build_model(selected_run, axes, config, device)
    optimizer = torch.optim.AdamW(
        model.vae_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    script_path = Path(__file__).resolve()
    model_path = script_path.parents[1] / "models/rald_anchor_ldm_refiner.py"
    ldm_path = script_path.parents[1] / "models/rald_anchor_ldm.py"
    selected_config_path = selected_run["config_path"]
    selected_checkpoint_path = selected_run["checkpoint_path"]
    provenance = {
        "git_commit": args.source_commit,
        **artifact_hashes,
        "training_script": str(script_path),
        "training_script_sha256": sha256(script_path),
        "model_source": str(model_path),
        "model_source_sha256": sha256(model_path),
        "rald_anchor_ldm_source": str(ldm_path),
        "rald_anchor_ldm_source_sha256": sha256(ldm_path),
        "official_rald_commit": RaLDAnchorLDM.OFFICIAL_RALD_COMMIT,
        "g3r_summary": str(args.g3r_summary.resolve()),
        "g3r_summary_sha256": sha256(args.g3r_summary),
        "g3r_source_commit": args.g3r_source_commit,
        "g3r_comparison": str(Path(summary["g3r_comparison"]).resolve()),
        "g3r_comparison_sha256": summary["g3r_comparison_sha256"],
        "g3r_selected_run": str(selected_run["run"]),
        "g3r_selected_config": str(selected_config_path),
        "g3r_selected_config_sha256": sha256(selected_config_path),
        "g3r_selected_checkpoint": str(selected_checkpoint_path),
        "g3r_selected_checkpoint_sha256": sha256(selected_checkpoint_path),
        "g3r_geometry_parent_checkpoint": str(selected_run["parent_checkpoint"]),
        "g3r_geometry_parent_checkpoint_sha256": sha256(
            selected_run["parent_checkpoint"]
        ),
        "device": device_name,
        "torch_version": torch.__version__,
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        if not config_path.is_file():
            raise FileNotFoundError("G3L-1 resume requires config.json")
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("G3L-1 resume configuration or provenance differs")
        if not (args.output / "last.pt").is_file():
            raise FileNotFoundError("G3L-1 resume requires last.pt")
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
    evaluation_positions = selected_indices(
        len(validation_indices), min(config.max_eval_frames, len(validation_indices))
    )
    evaluation_indices = [validation_indices[index] for index in evaluation_positions]

    start_epoch = 1
    best_score = float("inf")
    gradient_steps: list[dict] = []
    prior_elapsed = 0.0
    if args.resume:
        last = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        if last.get("config") != asdict(config) or last.get("provenance") != provenance:
            raise ValueError("G3L-1 last checkpoint metadata differs")
        model.load_vae_state_dict(last["g3l_vae"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        restore_rng_state(last["rng_state"])
        start_epoch = int(last["epoch"]) + 1
        gradient_steps = list(last["gradient_steps"])
        prior_elapsed = float(last["record"]["elapsed_seconds"])
        best_path = args.output / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError("G3L-1 resume requires best.pt")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if best.get("config") != asdict(config) or best.get("provenance") != provenance:
            raise ValueError("G3L-1 best checkpoint metadata differs")
        best_score = float(best["record"]["selection_score"])

    initial_path = args.output / "initial_validation_metrics.json"
    if initial_path.is_file():
        initial_metrics = json.loads(initial_path.read_text(encoding="utf-8"))
    else:
        initial_metrics = evaluate(
            model, validation_set, evaluation_indices, device, config
        )
        atomic_json(initial_path, initial_metrics)

    print(
        json.dumps(
            {
                "train_frames": len(train_indices),
                "validation_frames": len(validation_indices),
                "evaluation_frames": len(evaluation_indices),
                "start_epoch": start_epoch,
                "selection_protocol": SELECTION_PROTOCOL,
                "provenance": provenance,
            },
            indent=2,
        ),
        flush=True,
    )

    started = time.monotonic()
    log_path = args.output / "train_log.jsonl"
    optimization_step = max(0, start_epoch - 1) * len(train_indices)
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        order = train_indices.copy()
        random.Random(config.seed + epoch).shuffle(order)
        training_kl_weight = kl_warmup_weight(
            epoch, config.kl_weight, config.kl_warmup_epochs
        )
        losses = []
        physical_losses = []
        kl_values = []
        component_values: dict[str, list[float]] = {}
        for index in order:
            item = train_set[index]
            cube, _ = move_frame(item, device)
            target = item["target_xyz_confidence"].to(device)
            target_index = item["target_rae_index"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, physical, kl, total = objective_for_frame(
                    model,
                    cube,
                    target,
                    target_index,
                    config,
                    sample_posterior=True,
                    kl_weight=training_kl_weight,
                )
            total.backward()
            if any(
                parameter.grad is not None
                for parameter in model.geometry_parent.parameters()
            ):
                raise RuntimeError("Frozen G3R parent received gradients")
            if any(parameter.grad is not None for parameter in model.ldm.edm.parameters()):
                raise RuntimeError("Frozen G3L-2 EDM received G3L-1 gradients")
            optimization_step += 1
            if optimization_step <= 2:
                gradient_steps.append(
                    {"optimization_step": optimization_step, **gradient_audit(model)}
                )
            torch.nn.utils.clip_grad_norm_(model.vae_parameters(), 5.0)
            optimizer.step()
            losses.append(float(total.detach().float().item()))
            physical_losses.append(float(physical.total.detach().float().item()))
            kl_values.append(float(kl.detach().float().item()))
            for name, value in physical.components.items():
                component_values.setdefault(name, []).append(
                    float(value.detach().float().item())
                )
            del item, cube, target, target_index, output, physical, kl, total
            torch.cuda.empty_cache()
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_vae_loss_mean": float(np.mean(losses)),
            "train_physical_loss_mean": float(np.mean(physical_losses)),
            "train_posterior_kl_mean": float(np.mean(kl_values)),
            "training_kl_weight": training_kl_weight,
            "validation_kl_weight": config.kl_weight,
            "train_components": {
                name: float(np.mean(values))
                for name, values in component_values.items()
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_steps": gradient_steps,
            "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
        }
        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            metrics = evaluate(
                model, validation_set, evaluation_indices, device, config
            )
            score = float(metrics["vae_total_mean"])
            record["validation"] = metrics
            record["selection_score"] = score
            is_best = score < best_score
            atomic_json(args.output / f"metrics_epoch_{epoch:04d}.json", metrics)
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
            gradient_steps,
            record,
        )
        if is_best:
            best_score = score
            save_checkpoint(
                args.output / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                provenance,
                gradient_steps,
                record,
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    best_path = args.output / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("G3L-1 training produced no validation-selected checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    if best.get("config") != asdict(config) or best.get("provenance") != provenance:
        raise ValueError("G3L-1 selected checkpoint metadata differs")
    model.load_vae_state_dict(best["g3l_vae"], strict=True)
    final_metrics = evaluate(
        model, validation_set, validation_indices, device, config
    )
    report = {
        "protocol": config.protocol,
        "completed": True,
        "best_epoch": int(best["epoch"]),
        "selection_protocol": SELECTION_PROTOCOL,
        "selection_value": best_score,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256(best_path),
        "initial": initial_metrics,
        "final": final_metrics,
        "gradient_steps": gradient_steps,
        "posterior_sampling": {
            "training": "one reparameterized sample per frame",
            "validation": "posterior mean only",
            "best_of_k": False,
        },
        "provenance": provenance,
    }
    atomic_json(args.output / "best_validation_metrics.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
