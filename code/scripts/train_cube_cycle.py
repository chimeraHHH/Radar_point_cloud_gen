#!/usr/bin/env python3
"""Train matched continuous-point controls and Cube-cycle variants C0-C3."""

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
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
    nearest_distance,
    occupancy_to_points,
)
from eval.doppler_distribution import (  # noqa: E402
    aggregate_doppler_reports,
    cd_doppler_report,
    doppler_distribution_report,
)
from losses.cube_cycle import cube_cycle_loss  # noqa: E402
from losses.doppler_distribution import (  # noqa: E402
    circular_scalar_target,
    doppler_head_loss,
)
from losses.occupancy import occupancy_loss  # noqa: E402
from models.cube_cycle import CubeCycleNet  # noqa: E402
from models.cube_doppler import (  # noqa: E402
    query_cube_spectrum,
    split_query_indices,
)
from models.cube_occupancy import parameter_count  # noqa: E402
from models.point_to_cube import soft_splat_raed  # noqa: E402
from scripts.train_cube_doppler import (  # noqa: E402
    move_frame,
    selected_indices,
    sha256,
)


VARIANTS = ("none", "local_peak", "marginal", "full")


@dataclass(frozen=True)
class TrainConfig:
    variant: str
    parent_head_mode: str
    epochs: int
    head_learning_rate: float
    backbone_learning_rate: float
    weight_decay: float
    seed: int
    point_count: int
    geometry_weight: float
    eval_every: int
    max_eval_frames: int
    train_limit: int | None
    validation_limit: int | None
    base_channels: int
    log_center: float
    log_scale: float
    static_hypothesis: str
    maximum_offset_bins: float


def static_center(
    model: CubeCycleNet,
    indices: torch.Tensor,
    ego_speed: torch.Tensor,
) -> torch.Tensor:
    batch, _, azimuth, elevation = split_query_indices(indices, 1)
    return model.static_center(batch, azimuth, elevation, ego_speed)


def continuous_chamfer_loss(
    prediction_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_weight: torch.Tensor,
) -> torch.Tensor:
    prediction_to_target = nearest_distance(prediction_xyz, target_xyz)
    target_to_prediction = nearest_distance(target_xyz, prediction_xyz)
    weight = target_weight.to(target_to_prediction).clamp_min(0.0)
    completeness = (target_to_prediction * weight).sum()
    completeness = completeness / weight.sum().clamp_min(1e-8)
    return prediction_to_target.mean() + completeness


@torch.inference_mode()
def evaluate(
    model: CubeCycleNet,
    dataset: KRadarCubeDataset,
    frame_indices: list[int],
    axes,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    model.eval()
    geometry_reports = []
    cfar_reports = []
    doppler_reports = []
    cd_reports = []
    cycle_reports = []
    frames = []
    for index in frame_indices:
        item = dataset[index]
        cube, occupancy = move_frame(item, device)
        ego_speed = item["ego_speed_mps"].reshape(1).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occupancy_logits, features = model(cube)
            occupancy_value, _ = occupancy_loss(occupancy_logits, occupancy)
        _, confidence, discrete_indices = occupancy_to_points(
            occupancy_logits[0].float(), axes, point_count=config.point_count
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.query_cycle(features, discrete_indices, ego_speed)
        target_spectrum = query_cube_spectrum(cube, discrete_indices)
        center = static_center(model, discrete_indices, ego_speed)
        doppler = doppler_distribution_report(
            prediction["probability"],
            target_spectrum,
            model.doppler_mps,
            model.doppler_lower_mps,
            model.doppler_period_mps,
            model.doppler_step_mps,
            confidence=confidence,
            static_center_mps=center,
            predicted_static_probability=prediction.get("static_probability"),
        )
        rendered = soft_splat_raed(
            prediction["coordinates_rae"].float(),
            prediction["probability"].float(),
            confidence.float(),
        )
        cycle = cube_cycle_report(rendered, cube[0].float(), confidence.float())
        target = item["target_xyz_confidence"].to(device)
        target_indices = item["target_rae_index"].to(device)
        target_distribution = query_cube_spectrum(cube, target_indices)
        target_scalar = circular_scalar_target(
            target_distribution,
            model.doppler_mps,
            model.doppler_lower_mps,
            model.doppler_period_mps,
        )
        geometry = geometry_report(
            prediction["xyz_m"].float(),
            target[:, :3],
            target_weight=target[:, 3],
        )
        cd_doppler = cd_doppler_report(
            prediction["xyz_m"].float(),
            prediction["scalar_mps"].float(),
            target[:, :3],
            target_scalar,
            target_weight=target[:, 3],
        )
        cfar = item["cfar_xyzd_power_snr"][:, :3].to(device)
        cfar_geometry = geometry_report(
            cfar, target[:, :3], target_weight=target[:, 3]
        )
        offset = prediction["offset_rae_bins"].float()
        cycle["offset_abs_mean_bins"] = float(offset.abs().mean().item())
        cycle["offset_saturation_fraction"] = float(
            (offset.abs() >= config.maximum_offset_bins * 0.98).float().mean().item()
        )
        geometry_reports.append(geometry)
        cfar_reports.append(cfar_geometry)
        doppler_reports.append(doppler)
        cd_reports.append(cd_doppler)
        cycle_reports.append(cycle)
        frames.append(
            {
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
                "occupancy_loss": float(occupancy_value.item()),
                "generated_geometry": geometry,
                "cfar_geometry": cfar_geometry,
                "doppler": doppler,
                "cd_doppler": cd_doppler,
                "cycle": cycle,
            }
        )
        del item, cube, occupancy, occupancy_logits, features, occupancy_value
        del confidence, discrete_indices, prediction, target_spectrum, center
        del rendered, target, target_indices, target_distribution, target_scalar, cfar
        torch.cuda.empty_cache()
    return {
        "frame_count": len(frame_indices),
        "generated_geometry": aggregate_geometry_reports(geometry_reports),
        "cfar_geometry": aggregate_geometry_reports(cfar_reports),
        "doppler": aggregate_doppler_reports(doppler_reports),
        "cd_doppler": aggregate_doppler_reports(cd_reports),
        "cycle": aggregate_cycle_reports(cycle_reports),
        "frames": frames,
    }


def save_checkpoint(
    path: Path,
    model: CubeCycleNet,
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


def selection_score(metrics: dict) -> float:
    return float(
        metrics["cycle"]["local_spectrum_kl"]["median"]
        + 0.25 * metrics["generated_geometry"]["chamfer_m"]["median"]
    )


def best_recorded_score(output: Path, maximum_epoch: int) -> tuple[float, int]:
    values = []
    for path in sorted(output.glob("metrics_epoch_*.json")):
        epoch = int(path.stem.rsplit("_", maxsplit=1)[1])
        if epoch <= maximum_epoch:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            values.append((selection_score(metrics), epoch))
    if not values:
        raise ValueError("Resume cycle run has no validation metrics")
    return min(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--parent-g2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--maximum-offset-bins", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Cube-cycle training requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    manifest_hash = sha256(args.manifest)
    split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    parent_document = json.loads(
        (args.parent_g2_run / "config.json").read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    parent_provenance = parent_document["provenance"]
    if parent_config["head_mode"] not in ("distribution", "physics_distribution"):
        raise ValueError("Cycle parent must be E4 or E5")
    if int(parent_config["seed"]) != args.seed:
        raise ValueError("Cycle seed must match the G2 parent seed")
    if (
        parent_provenance["manifest_sha256"] != manifest_hash
        or parent_provenance["scene_split_sha256"] != split_hash
        or parent_provenance["normalization_sha256"] != normalization_hash
    ):
        raise ValueError("Cycle data artifacts differ from the G2 parent")
    config = TrainConfig(
        variant=args.variant,
        parent_head_mode=parent_config["head_mode"],
        epochs=args.epochs,
        head_learning_rate=args.head_learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        point_count=args.point_count,
        geometry_weight=args.geometry_weight,
        eval_every=args.eval_every,
        max_eval_frames=args.max_eval_frames,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
        static_hypothesis=parent_config["static_hypothesis"],
        maximum_offset_bins=args.maximum_offset_bins,
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
        raise FileExistsError(f"Cycle run is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No cycle run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    axes = load_axes(args.data_root / "resources")
    model = CubeCycleNet(
        config.parent_head_mode,
        torch.from_numpy(axes.doppler_mps),
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        base_channels=config.base_channels,
        log_center=config.log_center,
        log_scale=config.log_scale,
        static_hypothesis=config.static_hypothesis,
        maximum_offset_bins=config.maximum_offset_bins,
    ).to(device)
    parent_path = args.parent_g2_run / "best.pt"
    parent_checkpoint = torch.load(parent_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(parent_checkpoint["model"], strict=False)
    if unexpected or any(
        not key.startswith(("range_m", "offset_head")) for key in missing
    ):
        raise ValueError(
            f"Unexpected G2 initialization mismatch: missing={missing}, unexpected={unexpected}"
        )
    head_prefixes = (
        "query_projection",
        "scalar_head",
        "distribution_head",
        "static_gate",
        "offset_head",
    )
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        (head_parameters if name.startswith(head_prefixes) else backbone_parameters).append(
            parameter
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.backbone_learning_rate},
            {"params": head_parameters, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    provenance = {
        "git_commit": args.source_commit,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": split_hash,
        "normalization_sha256": normalization_hash,
        "parent_g2_checkpoint": str(parent_path),
        "parent_g2_checkpoint_sha256": sha256(parent_path),
        "parent_g2_git_commit": parent_provenance["git_commit"],
        "model_parameter_count": parameter_count(model),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("Cycle resume configuration or provenance differs")
    else:
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(run_document, indent=2) + "\n", encoding="utf-8")
        temporary.replace(config_path)

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
    evaluation_indices = [validation_indices[position] for position in positions]

    start_epoch = 1
    best_score = float("inf")
    prior_elapsed = 0.0
    log_path = args.output / "train_log.jsonl"
    if args.resume:
        last = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        if last["config"] != asdict(config) or last["provenance"] != provenance:
            raise ValueError("Last cycle checkpoint metadata differs")
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        last_epoch = int(last["epoch"])
        start_epoch = last_epoch + 1
        best_score, best_epoch = best_recorded_score(args.output, last_epoch)
        best_path = args.output / "best.pt"
        recorded_best_epoch = None
        if best_path.exists():
            recorded_best_epoch = int(
                torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]
            )
        if recorded_best_epoch != best_epoch:
            if best_epoch != last_epoch:
                raise ValueError("Best cycle checkpoint and metrics differ")
            save_checkpoint(
                best_path, model, optimizer, scheduler, last_epoch, config, provenance, last.get("record")
            )
        records = [
            json.loads(line)
            for line in (
                log_path.read_text(encoding="utf-8").splitlines()
                if log_path.exists()
                else []
            )
            if line.strip()
        ]
        logged_epoch = int(records[-1]["epoch"]) if records else 0
        if logged_epoch != last_epoch:
            checkpoint_record = last.get("record")
            if (
                checkpoint_record is None
                or logged_epoch != last_epoch - 1
                or int(checkpoint_record["epoch"]) != last_epoch
            ):
                raise ValueError("Cycle log and checkpoint epochs differ")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(checkpoint_record) + "\n")
            records.append(checkpoint_record)
        prior_elapsed = float(records[-1]["elapsed_seconds"])

    print(
        json.dumps(
            {
                "parameters": parameter_count(model),
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
        component_values: dict[str, list[float]] = {}
        for index in order:
            item = train_set[index]
            cube, occupancy = move_frame(item, device)
            ego_speed = item["ego_speed_mps"].reshape(1).to(device)
            target = item["target_xyz_confidence"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occupancy_logits, features = model(cube)
                occupancy_value, _ = occupancy_loss(occupancy_logits, occupancy)
            _, confidence, discrete_indices = occupancy_to_points(
                occupancy_logits[0].float(), axes, point_count=config.point_count
            )
            target_spectrum = query_cube_spectrum(cube, discrete_indices)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.query_cycle(features, discrete_indices, ego_speed)
                doppler_value, doppler_components = doppler_head_loss(
                    prediction,
                    target_spectrum,
                    model.doppler_mps,
                    model.doppler_lower_mps,
                    model.doppler_period_mps,
                    confidence=confidence,
                )
            geometry_value = continuous_chamfer_loss(
                prediction["xyz_m"].float(), target[:, :3], target[:, 3]
            )
            total = (
                occupancy_value
                + doppler_value
                + config.geometry_weight * geometry_value
            )
            cycle_components = {}
            if config.variant != "none":
                rendered = soft_splat_raed(
                    prediction["coordinates_rae"].float(),
                    prediction["probability"].float(),
                    confidence.float(),
                )
                cycle_value, cycle_components = cube_cycle_loss(
                    rendered,
                    cube[0].float(),
                    confidence.float(),
                    config.variant,
                )
                total = total + cycle_value
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(total.detach().item()))
            component_values.setdefault("occupancy", []).append(float(occupancy_value.item()))
            component_values.setdefault("doppler", []).append(float(doppler_value.item()))
            component_values.setdefault("continuous_chamfer", []).append(float(geometry_value.item()))
            for name, value in doppler_components.items():
                component_values.setdefault(f"doppler_{name}", []).append(float(value.item()))
            for name, value in cycle_components.items():
                component_values.setdefault(f"cycle_{name}", []).append(float(value.item()))
            del item, cube, occupancy, ego_speed, target, occupancy_logits, features
            del occupancy_value, confidence, discrete_indices, target_spectrum
            del prediction, doppler_value, geometry_value, total
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "train_components": {
                name: float(np.mean(values))
                for name, values in component_values.items()
            },
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
        }
        should_evaluate = epoch == 1 or epoch % config.eval_every == 0
        if should_evaluate:
            metrics = evaluate(
                model, validation_set, evaluation_indices, axes, config, device
            )
            score = selection_score(metrics)
            record["validation"] = metrics
            record["selection_score"] = score
            (args.output / f"metrics_epoch_{epoch:04d}.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
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
                record,
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    final_metrics = evaluate(
        model, validation_set, validation_indices, axes, config, device
    )
    report = {
        "best_epoch": int(best["epoch"]),
        "selection_metric": "cycle.local_spectrum_kl.median + 0.25 * generated_geometry.chamfer_m.median",
        "selection_value": best_score,
        "validation": final_metrics,
    }
    (args.output / "best_validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_validation": report}), flush=True)


if __name__ == "__main__":
    main()
