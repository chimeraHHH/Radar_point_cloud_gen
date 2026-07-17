#!/usr/bin/env python3
"""Train scalar, distribution, or physics-mixture Doppler heads on E2 geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
    occupancy_to_points,
)
from eval.doppler_distribution import (  # noqa: E402
    aggregate_doppler_reports,
    cd_doppler_report,
    doppler_distribution_report,
)
from losses.doppler_distribution import (  # noqa: E402
    circular_scalar_target,
    doppler_head_loss,
)
from losses.occupancy import occupancy_loss  # noqa: E402
from models.cube_doppler import (  # noqa: E402
    CubeDopplerNet,
    query_cube_spectrum,
    split_query_indices,
)
from models.cube_occupancy import parameter_count  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    head_mode: str
    epochs: int
    warmup_epochs: int
    head_learning_rate: float
    backbone_learning_rate: float
    weight_decay: float
    seed: int
    point_count: int
    eval_every: int
    max_eval_frames: int
    train_limit: int | None
    validation_limit: int | None
    base_channels: int
    log_center: float
    log_scale: float
    static_hypothesis: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def selected_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    return np.linspace(0, length - 1, limit).round().astype(int).tolist()


def move_frame(item: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
    occupancy = item["occupancy"].unsqueeze(0).to(device, non_blocking=True)
    return cube, occupancy


def static_center(
    model: CubeDopplerNet,
    indices: torch.Tensor,
    ego_speed_mps: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    batch, _, azimuth, elevation = split_query_indices(indices, batch_size)
    return model.static_center(batch, azimuth, elevation, ego_speed_mps)


@torch.inference_mode()
def evaluate(
    model: CubeDopplerNet,
    dataset: KRadarCubeDataset,
    indices: list[int],
    axes,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    model.eval()
    occupancy_losses = []
    generated_geometry = []
    cfar_geometry = []
    predicted_doppler = []
    query_reference = []
    cd_doppler_reports = []
    frames = []
    axis = model.doppler_mps
    for index in indices:
        item = dataset[index]
        cube, occupancy = move_frame(item, device)
        ego_speed = item["ego_speed_mps"].reshape(1).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occupancy_logits, features = model(cube)
            loss, _ = occupancy_loss(occupancy_logits, occupancy)
        generated_xyz, generated_confidence, generated_indices = occupancy_to_points(
            occupancy_logits[0].float(), axes, point_count=config.point_count
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.query(features, generated_indices, ego_speed)
        queried_spectrum = query_cube_spectrum(cube, generated_indices)
        generated_static_center = static_center(
            model, generated_indices, ego_speed, batch_size=1
        )
        doppler = doppler_distribution_report(
            prediction["probability"],
            queried_spectrum,
            axis,
            model.doppler_lower_mps,
            model.doppler_period_mps,
            model.doppler_step_mps,
            confidence=generated_confidence,
            static_center_mps=generated_static_center,
            predicted_static_probability=prediction.get("static_probability"),
        )
        q0 = doppler_distribution_report(
            queried_spectrum,
            queried_spectrum,
            axis,
            model.doppler_lower_mps,
            model.doppler_period_mps,
            model.doppler_step_mps,
            confidence=generated_confidence,
            static_center_mps=generated_static_center,
        )

        target = item["target_xyz_confidence"].to(device)
        target_indices = item["target_rae_index"].to(device)
        target_spectrum = query_cube_spectrum(cube, target_indices)
        target_scalar = circular_scalar_target(
            target_spectrum,
            axis,
            model.doppler_lower_mps,
            model.doppler_period_mps,
        )
        cd_doppler = cd_doppler_report(
            generated_xyz,
            prediction["scalar_mps"],
            target[:, :3],
            target_scalar,
            target_weight=target[:, 3],
        )
        generated_report = geometry_report(
            generated_xyz, target[:, :3], target_weight=target[:, 3]
        )
        cfar = item["cfar_xyzd_power_snr"][:, :3].to(device)
        cfar_report = geometry_report(
            cfar, target[:, :3], target_weight=target[:, 3]
        )
        occupancy_losses.append(float(loss.item()))
        generated_geometry.append(generated_report)
        cfar_geometry.append(cfar_report)
        predicted_doppler.append(doppler)
        query_reference.append(q0)
        cd_doppler_reports.append(cd_doppler)
        frames.append(
            {
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
                "occupancy_loss": float(loss.item()),
                "generated_geometry": generated_report,
                "cfar_geometry": cfar_report,
                "doppler": doppler,
                "q0_direct_query": q0,
                "cd_doppler": cd_doppler,
            }
        )
        del cube, occupancy, occupancy_logits, features, loss
        del generated_xyz, generated_confidence, generated_indices, prediction
        del queried_spectrum, generated_static_center, target, target_indices
        del target_spectrum, target_scalar, cfar
        torch.cuda.empty_cache()
    return {
        "frame_count": len(indices),
        "occupancy_loss_mean": float(np.mean(occupancy_losses)),
        "generated_geometry": aggregate_geometry_reports(generated_geometry),
        "cfar_geometry": aggregate_geometry_reports(cfar_geometry),
        "doppler": aggregate_doppler_reports(predicted_doppler),
        "q0_direct_query": aggregate_doppler_reports(query_reference),
        "cd_doppler": aggregate_doppler_reports(cd_doppler_reports),
        "frames": frames,
    }


def save_checkpoint(
    path: Path,
    model: CubeDopplerNet,
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


def best_recorded_nll(output: Path, maximum_epoch: int) -> tuple[float, int]:
    values = []
    for path in sorted(output.glob("metrics_epoch_*.json")):
        epoch = int(path.stem.rsplit("_", maxsplit=1)[1])
        if epoch > maximum_epoch:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        values.append((float(metrics["doppler"]["spectrum_nll"]["median"]), epoch))
    if not values:
        raise ValueError("Resume run has no recorded validation Doppler metrics")
    return min(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--parent-e2-run", type=Path, required=True)
    parser.add_argument("--static-doppler-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-mode", choices=CubeDopplerNet.HEAD_MODES, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Cube Doppler training requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("Warmup epochs must be in [0, epochs)")

    manifest_hash = sha256(args.manifest)
    split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    parent_document = json.loads(
        (args.parent_e2_run / "config.json").read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    parent_provenance = parent_document["provenance"]
    if parent_config["mode"] != "full_raed" or parent_config["overfit_one_frame"]:
        raise ValueError("P2 parent must be a formal Full-RAED E2 run")
    if int(parent_config["seed"]) != args.seed:
        raise ValueError("P2 seed must match the E2 parent seed")
    if (
        parent_provenance["manifest_sha256"] != manifest_hash
        or parent_provenance["scene_split_sha256"] != split_hash
        or parent_provenance["normalization_sha256"] != normalization_hash
    ):
        raise ValueError("P2 data artifacts differ from the E2 parent")
    static_audit = json.loads(args.static_doppler_audit.read_text(encoding="utf-8"))
    if (
        static_audit.get("passed") is not True
        and args.head_mode == "physics_distribution"
    ):
        raise ValueError("Physics-mixture training requires a passed static audit")
    static_hypothesis = static_audit["frozen_hypothesis"]
    config = TrainConfig(
        head_mode=args.head_mode,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        head_learning_rate=args.head_learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        point_count=args.point_count,
        eval_every=args.eval_every,
        max_eval_frames=args.max_eval_frames,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
        static_hypothesis=static_hypothesis,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    output_nonempty = args.output.exists() and any(args.output.iterdir())
    if output_nonempty and args.overwrite:
        shutil.rmtree(args.output)
        output_nonempty = False
    if output_nonempty and not args.resume:
        raise FileExistsError(f"Run is not empty: {args.output}")
    if args.resume and not output_nonempty:
        raise FileNotFoundError(f"No P2 run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    axes = load_axes(args.data_root / "resources")
    model = CubeDopplerNet(
        config.head_mode,
        torch.from_numpy(axes.doppler_mps),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        base_channels=config.base_channels,
        log_center=config.log_center,
        log_scale=config.log_scale,
        static_hypothesis=config.static_hypothesis,
    ).to(device)
    parent_checkpoint_path = args.parent_e2_run / "best.pt"
    parent_checkpoint = torch.load(
        parent_checkpoint_path, map_location=device, weights_only=False
    )
    missing, unexpected = model.load_state_dict(parent_checkpoint["model"], strict=False)
    if unexpected or any(
        not key.startswith(
            (
                "azimuth_rad",
                "elevation_rad",
                "doppler_step_mps",
                "doppler_period_mps",
                "doppler_lower_mps",
                "query_projection",
                "scalar_head",
                "distribution_head",
                "static_gate",
            )
        )
        for key in missing
    ):
        raise ValueError(
            f"Unexpected E2 initialization mismatch: missing={missing}, unexpected={unexpected}"
        )
    head_prefixes = (
        "query_projection",
        "scalar_head",
        "distribution_head",
        "static_gate",
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
        "parent_e2_checkpoint": str(parent_checkpoint_path),
        "parent_e2_checkpoint_sha256": sha256(parent_checkpoint_path),
        "parent_e2_git_commit": parent_provenance["git_commit"],
        "static_doppler_audit_sha256": sha256(args.static_doppler_audit),
        "static_doppler_audit_passed": bool(static_audit.get("passed")),
        "model_parameter_count": parameter_count(model),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("Resume configuration or provenance differs")
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
    validation_indices = selected_indices(
        len(validation_set), config.validation_limit
    )
    positions = selected_indices(
        len(validation_indices), min(config.max_eval_frames, len(validation_indices))
    )
    evaluation_indices = [validation_indices[position] for position in positions]

    start_epoch = 1
    best_nll = float("inf")
    prior_elapsed = 0.0
    log_path = args.output / "train_log.jsonl"
    if args.resume:
        last = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        if last["config"] != asdict(config) or last["provenance"] != provenance:
            raise ValueError("Last P2 checkpoint metadata differs")
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        last_epoch = int(last["epoch"])
        start_epoch = last_epoch + 1
        best_nll, best_epoch = best_recorded_nll(args.output, last_epoch)
        best_path = args.output / "best.pt"
        recorded_best_epoch = None
        if best_path.exists():
            recorded_best_epoch = int(
                torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]
            )
        if recorded_best_epoch != best_epoch:
            if best_epoch != last_epoch:
                raise ValueError("Best P2 checkpoint and metrics differ")
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                last_epoch,
                config,
                provenance,
                last.get("record"),
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
                raise ValueError("P2 log and checkpoint epochs differ")
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
        joint = epoch > config.warmup_epochs
        for parameter in backbone_parameters:
            parameter.requires_grad_(joint)
        model.train()
        order = train_indices.copy()
        random.Random(config.seed + epoch).shuffle(order)
        losses = []
        component_values: dict[str, list[float]] = {}
        for index in order:
            item = train_set[index]
            cube, occupancy = move_frame(item, device)
            occupied_indices = (occupancy[0] > 0).nonzero(as_tuple=False)
            confidence = occupancy[0][
                occupied_indices[:, 0],
                occupied_indices[:, 1],
                occupied_indices[:, 2],
            ]
            target_spectrum = query_cube_spectrum(cube, occupied_indices)
            ego_speed = item["ego_speed_mps"].reshape(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occupancy_logits, features = model(cube)
                prediction = model.query(features, occupied_indices, ego_speed)
                geometry_loss, _ = occupancy_loss(occupancy_logits, occupancy)
                doppler_loss, components = doppler_head_loss(
                    prediction,
                    target_spectrum,
                    model.doppler_mps,
                    model.doppler_lower_mps,
                    model.doppler_period_mps,
                    confidence=confidence,
                )
                total = doppler_loss + (geometry_loss if joint else 0.0)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(total.detach().item()))
            for name, value in components.items():
                component_values.setdefault(name, []).append(float(value.item()))
            component_values.setdefault("geometry", []).append(float(geometry_loss.item()))
            del item, cube, occupancy, occupied_indices, confidence, target_spectrum
            del ego_speed, occupancy_logits, features, prediction, geometry_loss
            del doppler_loss, total
        scheduler.step()
        record = {
            "epoch": epoch,
            "joint_finetuning": joint,
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
                model,
                validation_set,
                evaluation_indices,
                axes,
                config,
                device,
            )
            record["validation"] = metrics
            metrics_path = args.output / f"metrics_epoch_{epoch:04d}.json"
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            nll = metrics["doppler"]["spectrum_nll"]["median"]
            is_best = nll < best_nll
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
            best_nll = nll
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
        "selection_metric": "doppler.spectrum_nll.median",
        "selection_value": best_nll,
        "validation": final_metrics,
    }
    (args.output / "best_validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_validation": report}), flush=True)


if __name__ == "__main__":
    main()
