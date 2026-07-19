#!/usr/bin/env python3
"""Train the RaLD mixed-latent refiner behind a frozen G1 geometry parent."""

from __future__ import annotations

import argparse
import json
import random
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
from eval.cube_cycle import aggregate_cycle_reports, cube_cycle_report  # noqa: E402
from eval.dense_geometry import aggregate_geometry_reports, geometry_report  # noqa: E402
from eval.doppler_distribution import (  # noqa: E402
    aggregate_doppler_reports,
    doppler_distribution_report,
)
from losses.cube_cycle import existence_confidence_loss  # noqa: E402
from losses.rald_anchor import (  # noqa: E402
    anchor_refinement_loss,
    nearest_target_assignment,
)
from models.cube_doppler import query_cube_spectrum  # noqa: E402
from models.cube_occupancy import CubeOccupancyNet, parameter_count  # noqa: E402
from models.point_to_cube import soft_splat_raed  # noqa: E402
from models.rald_anchor import FrozenParentRaLDRefiner  # noqa: E402
from models.rald_matched import FullRAEDRadarTokenEncoder  # noqa: E402
from scripts.train_cube_doppler import move_frame, selected_indices, sha256  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
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
    eval_every: int
    max_eval_frames: int
    train_limit: int | None
    validation_limit: int | None
    rh1_one_frame: bool
    latent_count: int
    model_dim: int
    depth: int
    heads: int
    head_dim: int
    radar_base_channels: int
    radar_spectral_channels: int


def gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(sum(value.square().sum() for value in values)).item())


def gradient_audit(model: FrozenParentRaLDRefiner) -> dict[str, float]:
    physical = list(model.refiner.physical_head.parameters())
    physical_ids = {id(parameter) for parameter in physical}
    latent = [
        parameter
        for parameter in model.refiner.parameters()
        if id(parameter) not in physical_ids
    ]
    radar = (
        [] if model.radar_encoder is None else list(model.radar_encoder.parameters())
    )
    return {
        "physical_head": gradient_norm(physical),
        "set_latent_backbone": gradient_norm(latent),
        "radar_token_encoder": gradient_norm(radar),
        "all_refinement": gradient_norm(model.refinement_parameters()),
    }


def matched_spectrum(
    cube: torch.Tensor,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_rae_index: torch.Tensor,
) -> torch.Tensor:
    _, target_index = nearest_target_assignment(source_xyz, target_xyz)
    return query_cube_spectrum(cube, target_rae_index)[target_index]


@torch.inference_mode()
def evaluate(
    model: FrozenParentRaLDRefiner,
    dataset: KRadarCubeDataset,
    frame_indices: list[int],
    axes,
    device: torch.device,
) -> dict:
    model.eval()
    parent_geometry = []
    refined_geometry = []
    direct_doppler = []
    refined_doppler = []
    cycle_reports = []
    frames = []
    for index in frame_indices:
        item = dataset[index]
        cube, _ = move_frame(item, device)
        target = item["target_xyz_confidence"].to(device)
        target_index = item["target_rae_index"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(cube)
        parent_xyz = output["anchor_xyz_m"][0].float()
        generated_xyz = output["xyz_m"][0].float()
        target_xyz = target[:, :3].float()
        parent_report = geometry_report(
            parent_xyz, target_xyz, target_weight=target[:, 3]
        )
        refined_report = geometry_report(
            generated_xyz, target_xyz, target_weight=target[:, 3]
        )
        direct_target = matched_spectrum(
            cube, parent_xyz, target_xyz, target_index
        )
        refined_target = matched_spectrum(
            cube, generated_xyz, target_xyz, target_index
        )
        direct_report = doppler_distribution_report(
            output["anchor_cube_spectrum"][0].float(),
            direct_target,
            torch.as_tensor(axes.doppler_mps, device=device, dtype=torch.float32),
            torch.as_tensor(axes.doppler_mps[0], device=device, dtype=torch.float32),
            torch.as_tensor(
                np.median(np.diff(axes.doppler_mps)) * len(axes.doppler_mps),
                device=device,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                np.median(np.diff(axes.doppler_mps)),
                device=device,
                dtype=torch.float32,
            ),
            confidence=output["anchor_parent_confidence"][0].float(),
        )
        refined_report_doppler = doppler_distribution_report(
            output["doppler_probability"][0].float(),
            refined_target,
            torch.as_tensor(axes.doppler_mps, device=device, dtype=torch.float32),
            torch.as_tensor(axes.doppler_mps[0], device=device, dtype=torch.float32),
            torch.as_tensor(
                np.median(np.diff(axes.doppler_mps)) * len(axes.doppler_mps),
                device=device,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                np.median(np.diff(axes.doppler_mps)),
                device=device,
                dtype=torch.float32,
            ),
            confidence=output["anchor_parent_confidence"][0].float(),
        )
        prediction_distance, _ = nearest_target_assignment(
            generated_xyz, target_xyz
        )
        _, existence_target = existence_confidence_loss(
            output["confidence"][0].float(), prediction_distance
        )
        rendered = soft_splat_raed(
            output["coordinates_rae"][0].float(),
            output["doppler_probability"][0].float(),
            output["confidence"][0].float(),
        )
        cycle = cube_cycle_report(
            rendered,
            cube[0].float(),
            output["confidence"][0].float(),
            existence_target=existence_target,
        )
        offset = output["offset_bins"][0].float()
        cycle["offset_abs_mean_bins"] = float(offset.abs().mean().item())
        cycle["offset_saturation_fraction"] = float(
            (offset.abs() >= 0.49).float().mean().item()
        )
        parent_geometry.append(parent_report)
        refined_geometry.append(refined_report)
        direct_doppler.append(direct_report)
        refined_doppler.append(refined_report_doppler)
        cycle_reports.append(cycle)
        frames.append(
            {
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
                "parent_geometry": parent_report,
                "refined_geometry": refined_report,
                "direct_cube_doppler": direct_report,
                "refined_doppler": refined_report_doppler,
                "cycle": cycle,
            }
        )
        del item, cube, target, target_index, output, rendered
        torch.cuda.empty_cache()
    return {
        "frame_count": len(frame_indices),
        "parent_geometry": aggregate_geometry_reports(parent_geometry),
        "refined_geometry": aggregate_geometry_reports(refined_geometry),
        "direct_cube_doppler": aggregate_doppler_reports(direct_doppler),
        "refined_doppler": aggregate_doppler_reports(refined_doppler),
        "refined_cycle": aggregate_cycle_reports(cycle_reports),
        "frames": frames,
    }


def selection_score(metrics: dict) -> float:
    return float(
        metrics["refined_geometry"]["chamfer_m"]["median"]
        + 0.25 * metrics["refined_doppler"]["spectrum_nll"]["median"]
    )


def save_checkpoint(
    path: Path,
    model: FrozenParentRaLDRefiner,
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
            "refiner": model.refiner.state_dict(),
            "radar_encoder": (
                None
                if model.radar_encoder is None
                else model.radar_encoder.state_dict()
            ),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "provenance": provenance,
            "gradient_steps": gradient_steps,
            "record": record,
        },
        temporary,
    )
    temporary.replace(path)


def rh1_gate(
    initial: dict,
    final: dict,
    gradient_steps: list[dict],
) -> dict:
    parent_chamfer = final["parent_geometry"]["chamfer_m"]["median"]
    refined_chamfer = final["refined_geometry"]["chamfer_m"]["median"]
    direct_nll = final["direct_cube_doppler"]["spectrum_nll"]["median"]
    refined_nll = final["refined_doppler"]["spectrum_nll"]["median"]
    confidence = final["refined_cycle"]["confidence_mean"]["median"]
    saturation = final["refined_cycle"]["offset_saturation_fraction"]["median"]
    checks = {
        "initial_geometry_exactly_preserved": abs(
            initial["refined_geometry"]["chamfer_m"]["median"]
            - initial["parent_geometry"]["chamfer_m"]["median"]
        )
        <= 1e-6,
        "first_step_physical_gradient_nonzero": bool(
            gradient_steps and gradient_steps[0]["physical_head"] > 0.0
        ),
        "second_step_set_latent_gradient_nonzero": bool(
            len(gradient_steps) >= 2
            and gradient_steps[1]["set_latent_backbone"] > 0.0
        ),
        "second_step_radar_encoder_gradient_nonzero": bool(
            len(gradient_steps) >= 2
            and gradient_steps[1]["radar_token_encoder"] > 0.0
        ),
        "geometry_not_worse_than_parent": refined_chamfer <= parent_chamfer + 1e-6,
        "doppler_nll_improves_over_direct_cube": refined_nll < direct_nll,
        "confidence_not_collapsed": confidence >= 0.1,
        "offset_saturation_bounded": saturation <= 0.1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "values": {
            "parent_chamfer_m": parent_chamfer,
            "refined_chamfer_m": refined_chamfer,
            "direct_cube_spectrum_nll": direct_nll,
            "refined_spectrum_nll": refined_nll,
            "confidence_mean": confidence,
            "offset_saturation_fraction": saturation,
        },
        "thresholds": {
            "maximum_chamfer_increase_m": 1e-6,
            "minimum_confidence_mean": 0.1,
            "maximum_offset_saturation_fraction": 0.1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--g1-comparison", type=Path, required=True)
    parser.add_argument("--parent-g1-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--doppler-weight", type=float, default=1.0)
    parser.add_argument("--existence-weight", type=float, default=0.1)
    parser.add_argument("--cycle-weight", type=float, default=0.1)
    parser.add_argument("--offset-weight", type=float, default=0.01)
    parser.add_argument(
        "--cycle-variant", choices=("local_peak", "marginal", "full"), default="full"
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--rh1-one-frame", action="store_true")
    parser.add_argument("--latent-count", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--radar-base-channels", type=int, default=64)
    parser.add_argument("--radar-spectral-channels", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("RaLD anchor refinement requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.rh1_one_frame:
        if args.epochs < 2:
            raise ValueError("RH1 requires at least two optimization steps")
        args.train_limit = 1
        args.validation_limit = 1
        args.max_eval_frames = 1

    comparison = json.loads(args.g1_comparison.read_text(encoding="utf-8"))
    if comparison.get("decision", {}).get("g1_passed") is not True:
        raise RuntimeError("RH1/RH2 is locked until the formal G1 comparison passes")
    if args.seed not in comparison.get("seeds", []):
        raise ValueError("RaLD refiner seed must be one of the formal G1 seeds")
    parent_document = json.loads(
        (args.parent_g1_run / "config.json").read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    parent_provenance = parent_document["provenance"]
    if parent_config["mode"] != "full_raed":
        raise ValueError("RaLD anchor refinement requires the selected Full-RAED parent")
    if int(parent_config["seed"]) != args.seed:
        raise ValueError("RaLD refiner seed must match its frozen G1 parent")
    artifact_hashes = {
        "manifest_sha256": sha256(args.manifest),
        "scene_split_sha256": sha256(args.scene_split),
        "normalization_sha256": sha256(args.normalization_stats),
    }
    if any(parent_provenance[key] != value for key, value in artifact_hashes.items()):
        raise ValueError("RH data artifacts differ from the frozen G1 parent")

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        point_count=args.point_count,
        geometry_weight=args.geometry_weight,
        doppler_weight=args.doppler_weight,
        existence_weight=args.existence_weight,
        cycle_weight=args.cycle_weight,
        offset_weight=args.offset_weight,
        cycle_variant=args.cycle_variant,
        eval_every=args.eval_every,
        max_eval_frames=args.max_eval_frames,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        rh1_one_frame=args.rh1_one_frame,
        latent_count=args.latent_count,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        head_dim=args.head_dim,
        radar_base_channels=args.radar_base_channels,
        radar_spectral_channels=args.radar_spectral_channels,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    nonempty = args.output.exists() and any(args.output.iterdir())
    if nonempty and args.overwrite:
        shutil.rmtree(args.output)
        nonempty = False
    if nonempty and not args.resume:
        raise FileExistsError(f"RaLD anchor run is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No RaLD anchor run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    axes = load_axes(args.data_root / "resources")
    parent = CubeOccupancyNet(
        parent_config["mode"],
        torch.from_numpy(axes.doppler_mps),
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
    ).to(device)
    parent_path = args.parent_g1_run / "best.pt"
    parent_checkpoint = torch.load(parent_path, map_location=device, weights_only=False)
    parent.load_state_dict(parent_checkpoint["model"], strict=True)
    radar_encoder = FullRAEDRadarTokenEncoder(
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
        spectral_channels=config.radar_spectral_channels,
        token_dim=config.model_dim,
        base_channels=config.radar_base_channels,
    )
    model = FrozenParentRaLDRefiner(
        parent,
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        point_count=config.point_count,
        latent_count=config.latent_count,
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        head_dim=config.head_dim,
        radar_encoder=radar_encoder,
        radar_token_dim=config.model_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.refinement_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    provenance = {
        "git_commit": args.source_commit,
        **artifact_hashes,
        "g1_comparison": str(args.g1_comparison),
        "g1_comparison_sha256": sha256(args.g1_comparison),
        "parent_g1_checkpoint": str(parent_path),
        "parent_g1_checkpoint_sha256": sha256(parent_path),
        "parent_g1_git_commit": parent_provenance["git_commit"],
        "parent_parameter_count": parameter_count(parent),
        "refiner_parameter_count": parameter_count(model.refiner),
        "radar_encoder_parameter_count": parameter_count(model.radar_encoder),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("RH resume configuration or provenance differs")
    else:
        config_path.write_text(json.dumps(run_document, indent=2) + "\n", encoding="utf-8")

    train_set = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("train",)
    )
    validation_set = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train",) if config.rh1_one_frame else ("validation",),
    )
    train_indices = selected_indices(len(train_set), config.train_limit)
    validation_indices = selected_indices(len(validation_set), config.validation_limit)
    positions = selected_indices(
        len(validation_indices), min(config.max_eval_frames, len(validation_indices))
    )
    evaluation_indices = [validation_indices[position] for position in positions]

    start_epoch = 1
    best_score = float("inf")
    gradient_steps: list[dict] = []
    prior_elapsed = 0.0
    if args.resume:
        last = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        if last["config"] != asdict(config) or last["provenance"] != provenance:
            raise ValueError("RH checkpoint metadata differs")
        model.refiner.load_state_dict(last["refiner"], strict=True)
        model.radar_encoder.load_state_dict(last["radar_encoder"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        start_epoch = int(last["epoch"]) + 1
        gradient_steps = last["gradient_steps"]
        prior_elapsed = float(last["record"]["elapsed_seconds"])
        best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
        best_score = float(best["record"]["selection_score"])

    initial_path = args.output / "initial_metrics.json"
    if initial_path.exists():
        initial_metrics = json.loads(initial_path.read_text(encoding="utf-8"))
    else:
        initial_metrics = evaluate(
            model, validation_set, evaluation_indices, axes, device
        )
        initial_path.write_text(
            json.dumps(initial_metrics, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
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
    log_path = args.output / "train_log.jsonl"
    optimization_step = max(0, start_epoch - 1) * len(train_indices)
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        order = train_indices.copy()
        random.Random(config.seed + epoch).shuffle(order)
        losses = []
        component_values: dict[str, list[float]] = {}
        for index in order:
            item = train_set[index]
            cube, _ = move_frame(item, device)
            target = item["target_xyz_confidence"].to(device)
            target_index = item["target_rae_index"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(cube)
            objective = anchor_refinement_loss(
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
            objective.total.backward()
            if any(parameter.grad is not None for parameter in model.parent.parameters()):
                raise RuntimeError("Frozen G1 parent received gradients")
            optimization_step += 1
            if optimization_step <= 2:
                gradient_steps.append(
                    {"optimization_step": optimization_step, **gradient_audit(model)}
                )
            torch.nn.utils.clip_grad_norm_(model.refinement_parameters(), 5.0)
            optimizer.step()
            losses.append(float(objective.total.detach().item()))
            for name, value in objective.components.items():
                component_values.setdefault(name, []).append(float(value.item()))
            del item, cube, target, target_index, output, objective
            torch.cuda.empty_cache()
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "train_components": {
                name: float(np.mean(values)) for name, values in component_values.items()
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_steps": gradient_steps,
            "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
        }
        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            metrics = evaluate(model, validation_set, evaluation_indices, axes, device)
            score = selection_score(metrics)
            record["validation"] = metrics
            record["selection_score"] = score
            is_best = score < best_score
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

    best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.refiner.load_state_dict(best["refiner"], strict=True)
    model.radar_encoder.load_state_dict(best["radar_encoder"], strict=True)
    final_metrics = evaluate(model, validation_set, validation_indices, axes, device)
    report = {
        "best_epoch": int(best["epoch"]),
        "selection_value": best_score,
        "initial": initial_metrics,
        "final": final_metrics,
        "gradient_steps": gradient_steps,
    }
    if config.rh1_one_frame:
        report["rh1_gate"] = rh1_gate(initial_metrics, final_metrics, gradient_steps)
    report_path = args.output / (
        "rh1_gate.json" if config.rh1_one_frame else "best_validation_metrics.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
