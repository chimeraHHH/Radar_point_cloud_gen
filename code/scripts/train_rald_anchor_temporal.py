#!/usr/bin/env python3
"""Train zero-gated temporal adapters inside the selected RaLD-anchor model."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.rald_prediction import (  # noqa: E402
    FrozenRaLDPredictionCache,
    RaLDPointPrediction,
    rald_prediction_from_output,
)
from cube_dense.rald_run import build_temporal_rald, load_rald_run  # noqa: E402
from cube_dense.temporal_dataset import KRadarTemporalDataset  # noqa: E402
from eval.cube_cycle import aggregate_cycle_reports, cube_cycle_report  # noqa: E402
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
)
from eval.doppler_distribution import (  # noqa: E402
    aggregate_doppler_reports,
    doppler_distribution_report,
)
from eval.temporal_cube import (  # noqa: E402
    aggregate_temporal_reports,
    ego_aligned_consistency_report,
)
from losses.cube_cycle import existence_confidence_loss  # noqa: E402
from losses.rald_anchor import (  # noqa: E402
    anchor_refinement_loss,
    nearest_target_assignment,
)
from losses.temporal_consistency import (  # noqa: E402
    ego_aligned_match,
    ego_aligned_match_loss,
)
from models.cube_doppler import query_cube_spectrum  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from models.point_to_cube import soft_splat_raed  # noqa: E402
from models.rald_anchor_temporal import TEMPORAL_FUSION_MODES  # noqa: E402
from models.temporal_prior import ego_pose_warp  # noqa: E402
from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256  # noqa: E402
from scripts.rald_gate_contract import validate_g3r_selected_runs  # noqa: E402


PROTOCOL = "rald_anchor_g4r_training_v1"


@dataclass(frozen=True)
class TrainConfig:
    fusion_mode: str
    epochs: int
    temporal_warmup_epochs: int
    temporal_learning_rate: float
    base_learning_rate: float
    weight_decay: float
    seed: int
    point_count: int
    geometry_weight: float
    doppler_weight: float
    existence_weight: float
    cycle_weight: float
    offset_weight: float
    temporal_weight: float
    scheduled_sampling_maximum: float
    eval_every: int
    max_eval_pairs: int
    train_window_limit: int | None
    validation_window_limit: int | None
    prior_base_channels: int


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_positions(length: int, limit: int | None) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    if limit <= 0:
        raise ValueError("Selection limits must be positive")
    return sorted(
        set(np.linspace(0, length - 1, limit).round().astype(int).tolist())
    )


def pairs_by_window(dataset: KRadarTemporalDataset) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for pair in dataset.pairs:
        grouped[pair["window_id"]].append(pair)
    for pairs in grouped.values():
        pairs.sort(key=lambda pair: pair["current_frame_in_window"])
    return dict(grouped)


def scheduled_probability(config: TrainConfig, epoch: int) -> float:
    if epoch <= config.temporal_warmup_epochs:
        return 0.0
    denominator = max(1, config.epochs - config.temporal_warmup_epochs)
    progress = (epoch - config.temporal_warmup_epochs) / denominator
    return config.scheduled_sampling_maximum * min(max(progress, 0.0), 1.0)


def teacher_state(
    cache: FrozenRaLDPredictionCache,
    dataset: KRadarTemporalDataset,
    pair: dict,
    device: torch.device,
) -> RaLDPointPrediction:
    record = dataset.frame_dataset.records[pair["previous_dataset_index"]]
    return cache.load(
        int(record["sequence"]), int(record["radar_index"]), device
    )


def temporal_prior(state: RaLDPointPrediction, pair: dict, model):
    transform = torch.as_tensor(
        pair["current_from_previous"],
        dtype=state.xyz_m.dtype,
        device=state.xyz_m.device,
    ).reshape(4, 4)
    return ego_pose_warp(
        state.xyz_m,
        state.probability,
        state.confidence,
        transform,
        model.doppler_mps,
        model.doppler_lower_mps,
        model.doppler_period_mps,
        model.range_m,
        model.azimuth_rad,
        model.elevation_rad,
    )


def move_current(dataset: KRadarTemporalDataset, pair: dict, device: torch.device):
    item = dataset.frame_dataset[pair["current_dataset_index"]]
    cube = item["cube_drae"].unsqueeze(0).to(device)
    target = item["target_xyz_confidence"].to(device)
    target_index = item["target_rae_index"].to(device)
    return item, cube, target, target_index


def temporal_objective(
    model,
    output: dict[str, torch.Tensor],
    cube: torch.Tensor,
    target: torch.Tensor,
    target_index: torch.Tensor,
    previous: RaLDPointPrediction,
    pair: dict,
    config: TrainConfig,
):
    base = anchor_refinement_loss(
        output,
        cube,
        target,
        target_index,
        geometry_weight=config.geometry_weight,
        doppler_weight=config.doppler_weight,
        existence_weight=config.existence_weight,
        cycle_weight=config.cycle_weight,
        offset_weight=config.offset_weight,
        cycle_variant="full",
    )
    transform = torch.as_tensor(
        pair["current_from_previous"],
        dtype=previous.xyz_m.dtype,
        device=previous.xyz_m.device,
    ).reshape(4, 4)
    match = ego_aligned_match(
        previous.xyz_m,
        previous.confidence,
        output["xyz_m"][0].float(),
        output["confidence"][0].float(),
        transform,
    )
    temporal = ego_aligned_match_loss(match)
    total = base.total + config.temporal_weight * temporal
    components = {
        name: float(value.item()) for name, value in base.components.items()
    }
    components["ego_aligned_match"] = float(temporal.item())
    components["total_with_temporal"] = float(total.detach().item())
    return total, components


@torch.inference_mode()
def evaluate(
    model,
    dataset: KRadarTemporalDataset,
    pair_indices: list[int],
    teacher_cache: FrozenRaLDPredictionCache,
    axes,
    device: torch.device,
) -> dict:
    model.eval()
    geometry_reports = []
    doppler_reports = []
    cycle_reports = []
    temporal_reports = []
    frames = []
    doppler_axis = torch.as_tensor(
        axes.doppler_mps, device=device, dtype=torch.float32
    )
    doppler_step = torch.as_tensor(
        np.median(np.diff(axes.doppler_mps)),
        device=device,
        dtype=torch.float32,
    )
    for pair_index in pair_indices:
        pair = dataset.pairs[pair_index]
        previous = teacher_state(teacher_cache, dataset, pair, device)
        item, cube, target, target_index = move_current(dataset, pair, device)
        prior = temporal_prior(previous, pair, model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(cube, prior)
        generated_xyz = output["xyz_m"][0].float()
        target_xyz = target[:, :3].float()
        geometry = geometry_report(
            generated_xyz, target_xyz, target_weight=target[:, 3]
        )
        _, matched_index = nearest_target_assignment(generated_xyz, target_xyz)
        target_spectrum = query_cube_spectrum(cube, target_index)[matched_index]
        doppler = doppler_distribution_report(
            output["doppler_probability"][0].float(),
            target_spectrum,
            doppler_axis,
            doppler_axis[0],
            doppler_step * doppler_axis.numel(),
            doppler_step,
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
        transform = torch.as_tensor(
            pair["current_from_previous"],
            dtype=previous.xyz_m.dtype,
            device=device,
        ).reshape(4, 4)
        match = ego_aligned_match(
            previous.xyz_m,
            previous.confidence,
            generated_xyz,
            output["confidence"][0].float(),
            transform,
        )
        temporal = ego_aligned_consistency_report(
            match,
            previous.confidence,
            generated_xyz,
            output["confidence"][0].float(),
            model.range_m,
            model.azimuth_rad,
            model.elevation_rad,
        )
        geometry_reports.append(geometry)
        doppler_reports.append(doppler)
        cycle_reports.append(cycle)
        temporal_reports.append(temporal)
        frames.append(
            {
                "window_id": pair["window_id"],
                "sequence": int(pair["sequence"]),
                "radar_index": int(item["radar_index"]),
                "generated_geometry": geometry,
                "doppler": doppler,
                "cycle": cycle,
                "temporal": temporal,
            }
        )
        del previous, item, cube, target, target_index, prior, output
        del rendered, match
        torch.cuda.empty_cache()
    return {
        "pair_count": len(pair_indices),
        "generated_geometry": aggregate_geometry_reports(geometry_reports),
        "doppler": aggregate_doppler_reports(doppler_reports),
        "cycle": aggregate_cycle_reports(cycle_reports),
        "temporal": aggregate_temporal_reports(temporal_reports),
        "frames": frames,
    }


def selection_score(metrics: dict) -> float:
    return float(
        metrics["temporal"]["ego_aligned_matched_distance_mean_m"]["median"]
        + 0.25 * metrics["generated_geometry"]["chamfer_m"]["median"]
        + 0.25 * metrics["cycle"]["local_spectrum_kl"]["median"]
    )


def set_training_phase(model, temporal_only: bool) -> None:
    temporal_ids = {id(parameter) for parameter in model.temporal_parameters()}
    for parameter in model.refinement_parameters():
        parameter.requires_grad_(not temporal_only or id(parameter) in temporal_ids)
    for parameter in model.parent.parameters():
        parameter.requires_grad_(False)


@torch.inference_mode()
def zero_gate_identity(model, cube, prior) -> dict[str, float | bool]:
    model.eval()
    without_history = model(cube)
    with_history = model(cube, prior)
    differences = {
        key: float(
            (without_history[key].float() - with_history[key].float())
            .abs()
            .max()
            .item()
        )
        for key in (
            "coordinates_rae",
            "xyz_m",
            "doppler_probability",
            "confidence",
        )
    }
    report = {
        f"{key}_maximum_absolute_error": value
        for key, value in differences.items()
    }
    report["exact_identity"] = max(differences.values()) == 0.0
    return report


def save_checkpoint(
    path: Path,
    model,
    optimizer,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--dense-cache-report", type=Path, required=True)
    parser.add_argument("--g3r-summary", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-prediction-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fusion-mode", choices=TEMPORAL_FUSION_MODES, required=True
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--temporal-warmup-epochs", type=int, default=5)
    parser.add_argument("--temporal-learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--scheduled-sampling-maximum", type=float, default=0.4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-pairs", type=int, default=32)
    parser.add_argument("--train-window-limit", type=int, default=None)
    parser.add_argument("--validation-window-limit", type=int, default=None)
    parser.add_argument("--prior-base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("RaLD G4R training requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if not 0 <= args.temporal_warmup_epochs <= args.epochs:
        raise ValueError("Invalid G4R temporal warmup")
    if not 0.0 <= args.scheduled_sampling_maximum <= 1.0:
        raise ValueError("Scheduled sampling must be in [0,1]")
    if args.seed not in FROZEN_G1B_SEEDS:
        raise ValueError("G4R seed is outside the frozen matrix")

    g3r_summary = json.loads(args.g3r_summary.read_text(encoding="utf-8"))
    selected_runs = validate_g3r_selected_runs(
        g3r_summary, args.source_commit
    )
    if args.parent_run.resolve() != selected_runs[args.seed]:
        raise ValueError("G4R parent differs from the selected G3R seed run")
    run = load_rald_run(args.parent_run)
    parent_config = run["config"]
    parent_provenance = run["provenance"]
    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    dense_hash = sha256(args.dense_cache_report)
    dense_report = json.loads(args.dense_cache_report.read_text(encoding="utf-8"))
    if (
        dense_report.get("completed") is not True
        or dense_report["configuration"]["source_manifest_sha256"]
        != manifest_hash
    ):
        raise ValueError("G4R dense target cache is incomplete or mismatched")
    if (
        parent_provenance["scene_split_sha256"] != scene_split_hash
        or parent_provenance["normalization_sha256"] != normalization_hash
    ):
        raise ValueError("G4R parent data provenance differs")
    temporal_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    teacher_cache = FrozenRaLDPredictionCache(
        args.parent_prediction_cache,
        expected_frames=len(temporal_manifest["frames"]),
    )
    if (
        teacher_cache.configuration["temporal_manifest_sha256"] != manifest_hash
        or teacher_cache.configuration["g3r_checkpoint_sha256"]
        != sha256(run["checkpoint_path"])
    ):
        raise ValueError("G4R teacher cache and selected parent differ")

    config = TrainConfig(
        fusion_mode=args.fusion_mode,
        epochs=args.epochs,
        temporal_warmup_epochs=args.temporal_warmup_epochs,
        temporal_learning_rate=args.temporal_learning_rate,
        base_learning_rate=args.base_learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        point_count=int(parent_config["point_count"]),
        geometry_weight=float(parent_config["geometry_weight"]),
        doppler_weight=float(parent_config["doppler_weight"]),
        existence_weight=float(parent_config["existence_weight"]),
        cycle_weight=float(parent_config["cycle_weight"]),
        offset_weight=float(parent_config["offset_weight"]),
        temporal_weight=args.temporal_weight,
        scheduled_sampling_maximum=args.scheduled_sampling_maximum,
        eval_every=args.eval_every,
        max_eval_pairs=args.max_eval_pairs,
        train_window_limit=args.train_window_limit,
        validation_window_limit=args.validation_window_limit,
        prior_base_channels=args.prior_base_channels,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if "H200" not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError("Formal G4R training requires an H200")

    nonempty = args.output.exists() and any(args.output.iterdir())
    if nonempty and args.overwrite:
        shutil.rmtree(args.output)
        nonempty = False
    if nonempty and not args.resume:
        raise FileExistsError(f"G4R run is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No G4R run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    axes = load_axes(args.data_root / "resources")
    model = build_temporal_rald(
        run,
        axes,
        device,
        config.fusion_mode,
        prior_base_channels=config.prior_base_channels,
    )
    temporal_parameters = list(model.temporal_parameters())
    temporal_ids = {id(parameter) for parameter in temporal_parameters}
    base_parameters = [
        parameter
        for parameter in model.refinement_parameters()
        if id(parameter) not in temporal_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": config.base_learning_rate},
            {
                "params": temporal_parameters,
                "lr": config.temporal_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    train_dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, ("train",)
    )
    validation_dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    train_windows = pairs_by_window(train_dataset)
    validation_windows = pairs_by_window(validation_dataset)
    train_names = sorted(train_windows)
    validation_names = sorted(validation_windows)
    train_names = [
        train_names[index]
        for index in selected_positions(len(train_names), config.train_window_limit)
    ]
    validation_names = [
        validation_names[index]
        for index in selected_positions(
            len(validation_names), config.validation_window_limit
        )
    ]
    validation_pair_indices = [
        index
        for index, pair in enumerate(validation_dataset.pairs)
        if pair["window_id"] in validation_names
    ]
    evaluation_pair_indices = [
        validation_pair_indices[index]
        for index in selected_positions(
            len(validation_pair_indices),
            min(config.max_eval_pairs, len(validation_pair_indices)),
        )
    ]

    identity_pair = validation_dataset.pairs[evaluation_pair_indices[0]]
    identity_state = teacher_state(
        teacher_cache, validation_dataset, identity_pair, device
    )
    _, identity_cube, _, _ = move_current(
        validation_dataset, identity_pair, device
    )
    identity_prior = temporal_prior(identity_state, identity_pair, model)
    identity = zero_gate_identity(model, identity_cube, identity_prior)
    if identity["exact_identity"] is not True:
        raise ValueError(f"G4R zero gate changed G3R output: {identity}")

    provenance = {
        "protocol": PROTOCOL,
        "git_commit": args.source_commit,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": normalization_hash,
        "dense_cache_report_sha256": dense_hash,
        "g3r_summary": str(args.g3r_summary.resolve()),
        "g3r_summary_sha256": sha256(args.g3r_summary),
        "g3r_comparison_sha256": g3r_summary["g3r_comparison_sha256"],
        "g3r_config_sha256": sha256(run["config_path"]),
        "g3r_checkpoint_sha256": sha256(run["checkpoint_path"]),
        "parent_config_sha256": sha256(run["config_path"]),
        "parent_checkpoint": str(run["checkpoint_path"]),
        "parent_checkpoint_sha256": sha256(run["checkpoint_path"]),
        "parent_prediction_manifest_sha256": sha256(
            teacher_cache.manifest_path
        ),
        "parent_git_commit": parent_provenance["git_commit"],
        "model_parameter_count": parameter_count(model),
        "temporal_parameter_count": sum(
            parameter.numel() for parameter in temporal_parameters
        ),
        "zero_gate_identity": identity,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("G4R resume metadata differs")
    else:
        atomic_json(config_path, run_document)

    start_epoch = 1
    best_score = float("inf")
    prior_elapsed = 0.0
    log_path = args.output / "train_log.jsonl"
    if args.resume:
        last = torch.load(
            args.output / "last.pt", map_location=device, weights_only=False
        )
        if last["config"] != asdict(config) or last["provenance"] != provenance:
            raise ValueError("G4R checkpoint metadata differs")
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        start_epoch = int(last["epoch"]) + 1
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records or int(records[-1]["epoch"]) != start_epoch - 1:
            raise ValueError("G4R log and checkpoint differ")
        prior_elapsed = float(records[-1]["elapsed_seconds"])
        best_score = min(
            float(record["selection_score"])
            for record in records
            if "selection_score" in record
        )

    print(json.dumps({"config": asdict(config), "provenance": provenance}), flush=True)
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs + 1):
        temporal_only = epoch <= config.temporal_warmup_epochs
        set_training_phase(model, temporal_only)
        model.train()
        window_order = train_names.copy()
        random.Random(config.seed + epoch).shuffle(window_order)
        sampling_rng = random.Random(config.seed * 10_000 + epoch)
        sampling_probability = scheduled_probability(config, epoch)
        losses = []
        components: dict[str, list[float]] = defaultdict(list)
        teacher_count = 0
        scheduled_count = 0
        for window_id in window_order:
            recurrent_state = None
            for pair in train_windows[window_id]:
                use_scheduled = (
                    recurrent_state is not None
                    and sampling_rng.random() < sampling_probability
                )
                if use_scheduled:
                    previous = recurrent_state
                    scheduled_count += 1
                else:
                    previous = teacher_state(
                        teacher_cache, train_dataset, pair, device
                    )
                    teacher_count += 1
                _, cube, target, target_index = move_current(
                    train_dataset, pair, device
                )
                prior = temporal_prior(previous, pair, model)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(cube, prior)
                total, values = temporal_objective(
                    model,
                    output,
                    cube,
                    target,
                    target_index,
                    previous,
                    pair,
                    config,
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.refinement_parameters(), 5.0)
                optimizer.step()
                recurrent_state = rald_prediction_from_output(output).detached()
                losses.append(float(total.detach().item()))
                for name, value in values.items():
                    components[name].append(value)
                del previous, cube, target, target_index, prior, output, total
        scheduler.step()
        record = {
            "epoch": epoch,
            "temporal_only": temporal_only,
            "scheduled_sampling_probability": sampling_probability,
            "teacher_exposure_count": teacher_count,
            "scheduled_exposure_count": scheduled_count,
            "train_loss_mean": float(np.mean(losses)),
            "train_components": {
                name: float(np.mean(values)) for name, values in components.items()
            },
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
        }
        should_evaluate = (
            epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs
        )
        is_best = False
        if should_evaluate:
            metrics = evaluate(
                model,
                validation_dataset,
                evaluation_pair_indices,
                teacher_cache,
                axes,
                device,
            )
            score = selection_score(metrics)
            record["validation"] = metrics
            record["selection_score"] = score
            atomic_json(args.output / f"metrics_epoch_{epoch:04d}.json", metrics)
            is_best = score < best_score
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

    best = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best["model"], strict=True)
    final_metrics = evaluate(
        model,
        validation_dataset,
        validation_pair_indices,
        teacher_cache,
        axes,
        device,
    )
    final_report = {
        "protocol": PROTOCOL,
        "best_epoch": int(best["epoch"]),
        "selection_metric": (
            "ego_aligned_match + 0.25 * chamfer + 0.25 * local_spectrum_kl"
        ),
        "selection_value": selection_score(final_metrics),
        "checkpoint_selection_value": best_score,
        "validation": final_metrics,
        "completed": True,
    }
    atomic_json(args.output / "best_validation_metrics.json", final_report)
    print(json.dumps({"best_validation": final_report}), flush=True)


if __name__ == "__main__":
    main()
