#!/usr/bin/env python3
"""Train concat, cross-attention, or draft-refinement Cube temporal priors."""

from __future__ import annotations

import argparse
import hashlib
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

from cube_dense.parent_prediction import (  # noqa: E402
    FrozenPredictionCache,
    PointPrediction,
    prediction_from_output,
)
from cube_dense.temporal_dataset import KRadarTemporalDataset  # noqa: E402
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
    doppler_distribution_report,
)
from eval.temporal_cube import (  # noqa: E402
    aggregate_temporal_reports,
    temporal_consistency_report,
)
from losses.cube_cycle import cube_cycle_loss  # noqa: E402
from losses.doppler_distribution import doppler_head_loss  # noqa: E402
from losses.occupancy import occupancy_loss  # noqa: E402
from losses.temporal_consistency import (  # noqa: E402
    temporal_match,
    temporal_radial_loss,
)
from models.cube_doppler import query_cube_spectrum, split_query_indices  # noqa: E402
from models.cube_occupancy import parameter_count  # noqa: E402
from models.cube_temporal import CubeTemporalNet, FUSION_MODES  # noqa: E402
from models.point_to_cube import soft_splat_raed  # noqa: E402
from models.temporal_prior import (  # noqa: E402
    gated_doppler_warp,
    rasterize_temporal_prior,
)


TEMPORAL_PARAMETER_PREFIXES = (
    "prior_grid_projection",
    "concat_fusion",
    "prior_token_projection",
    "relative_position_projection",
    "prior_attention",
    "temporal_norm",
    "draft_projection",
    "draft_offset_gate",
)


@dataclass(frozen=True)
class TrainConfig:
    fusion_mode: str
    parent_variant: str
    epochs: int
    joint_start_epoch: int
    temporal_learning_rate: float
    parent_learning_rate: float
    weight_decay: float
    seed: int
    point_count: int
    geometry_weight: float
    temporal_weight: float
    scheduled_sampling_maximum: float
    dynamic_threshold_mps: float
    eval_every: int
    max_eval_pairs: int
    train_window_limit: int | None
    validation_window_limit: int | None
    base_channels: int
    log_center: float
    log_scale: float
    static_hypothesis: str
    maximum_offset_bins: float
    attention_neighbors: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def temporal_parameter(name: str) -> bool:
    return name.startswith(TEMPORAL_PARAMETER_PREFIXES)


def static_center(
    model: CubeTemporalNet,
    prediction: dict[str, torch.Tensor],
    indices: torch.Tensor,
    ego_speed: torch.Tensor,
) -> torch.Tensor:
    if "static_center_mps" in prediction:
        return prediction["static_center_mps"]
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


def scheduled_probability(config: TrainConfig, epoch: int) -> float:
    if epoch <= config.joint_start_epoch:
        return 0.0
    denominator = max(1, config.epochs - config.joint_start_epoch)
    progress = (epoch - config.joint_start_epoch) / denominator
    return config.scheduled_sampling_maximum * min(max(progress, 0.0), 1.0)


def pairs_by_window(dataset: KRadarTemporalDataset) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for pair in dataset.pairs:
        grouped[pair["window_id"]].append(pair)
    for pairs in grouped.values():
        pairs.sort(key=lambda pair: pair["current_frame_in_window"])
    return dict(grouped)


def teacher_state(
    cache: FrozenPredictionCache,
    dataset: KRadarTemporalDataset,
    pair: dict,
    device: torch.device,
) -> PointPrediction:
    record = dataset.frame_dataset.records[pair["previous_dataset_index"]]
    return cache.load(
        int(record["sequence"]), int(record["radar_index"]), device
    )


def make_prior(
    state: PointPrediction,
    pair: dict,
    model: CubeTemporalNet,
    device: torch.device,
    dynamic_threshold_mps: float,
):
    transform = torch.tensor(
        pair["current_from_previous"], dtype=torch.float32, device=device
    ).reshape(4, 4)
    delta_seconds = torch.tensor(
        pair["delta_seconds"], dtype=torch.float32, device=device
    )
    return gated_doppler_warp(
        state.xyz_m,
        state.probability,
        state.confidence,
        transform,
        delta_seconds,
        model.doppler_mps,
        model.doppler_lower_mps,
        model.doppler_period_mps,
        model.range_m,
        model.azimuth_rad,
        model.elevation_rad,
        previous_static_center_mps=state.static_center_mps,
        dynamic_threshold_mps=dynamic_threshold_mps,
    ), transform, delta_seconds


def predict_pair(
    model: CubeTemporalNet,
    current_item: dict,
    prior_state: PointPrediction,
    pair: dict,
    axes,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    prior, transform, delta_seconds = make_prior(
        prior_state,
        pair,
        model,
        device,
        config.dynamic_threshold_mps,
    )
    prior_raster = None
    if config.fusion_mode == "concat":
        prior_raster = rasterize_temporal_prior(
            prior,
            model.doppler_mps,
            model.doppler_lower_mps,
            model.doppler_period_mps,
        )
    cube = current_item["cube_drae"].unsqueeze(0).to(device)
    occupancy = current_item["occupancy"].unsqueeze(0).to(device)
    ego_speed = current_item["ego_speed_mps"].reshape(1).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occupancy_logits, features = model.forward_temporal(cube, prior_raster)
    query_xyz, confidence, indices = occupancy_to_points(
        occupancy_logits[0].float(), axes, point_count=config.point_count
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.query_temporal(
            features, indices, query_xyz, ego_speed, prior
        )
    current_static = static_center(model, prediction, indices, ego_speed)
    return {
        "prior": prior,
        "transform": transform,
        "delta_seconds": delta_seconds,
        "cube": cube,
        "occupancy": occupancy,
        "occupancy_logits": occupancy_logits,
        "features": features,
        "query_xyz": query_xyz,
        "confidence": confidence,
        "indices": indices,
        "prediction": prediction,
        "current_static_center_mps": current_static,
        "ego_speed": ego_speed,
    }


def training_loss(
    model: CubeTemporalNet,
    current_item: dict,
    prior_state: PointPrediction,
    pair_output: dict,
    config: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    occupancy_value, _ = occupancy_loss(
        pair_output["occupancy_logits"], pair_output["occupancy"]
    )
    target_spectrum = query_cube_spectrum(
        pair_output["cube"], pair_output["indices"]
    )
    doppler_value, doppler_components = doppler_head_loss(
        pair_output["prediction"],
        target_spectrum,
        model.doppler_mps,
        model.doppler_lower_mps,
        model.doppler_period_mps,
        confidence=pair_output["confidence"],
    )
    target = current_item["target_xyz_confidence"].to(device)
    geometry_value = continuous_chamfer_loss(
        pair_output["prediction"]["xyz_m"].float(),
        target[:, :3],
        target[:, 3],
    )
    total = (
        occupancy_value
        + doppler_value
        + config.geometry_weight * geometry_value
    )
    cycle_value = total.new_zeros(())
    cycle_components = {}
    if config.parent_variant == "full":
        rendered = soft_splat_raed(
            pair_output["prediction"]["coordinates_rae"].float(),
            pair_output["prediction"]["probability"].float(),
            pair_output["confidence"].float(),
        )
        cycle_value, raw_cycle_components = cube_cycle_loss(
            rendered,
            pair_output["cube"][0].float(),
            pair_output["confidence"].float(),
            "full",
        )
        total = total + cycle_value
        cycle_components = {
            f"cycle_{key}": float(value.item())
            for key, value in raw_cycle_components.items()
        }
    match = temporal_match(
        prior_state.xyz_m,
        prior_state.probability,
        prior_state.confidence,
        pair_output["prediction"]["xyz_m"].float(),
        pair_output["prediction"]["probability"].float(),
        pair_output["confidence"].float(),
        pair_output["transform"],
        pair_output["delta_seconds"],
        model.doppler_mps,
        model.doppler_lower_mps,
        model.doppler_period_mps,
        previous_static_center_mps=prior_state.static_center_mps,
        current_static_center_mps=pair_output["current_static_center_mps"],
        dynamic_threshold_mps=config.dynamic_threshold_mps,
    )
    temporal_value = temporal_radial_loss(match)
    total = total + config.temporal_weight * temporal_value
    components = {
        "occupancy": float(occupancy_value.item()),
        "doppler": float(doppler_value.item()),
        "continuous_chamfer": float(geometry_value.item()),
        "cycle": float(cycle_value.item()),
        "temporal_radial": float(temporal_value.item()),
        **{
            f"doppler_{key}": float(value.item())
            for key, value in doppler_components.items()
        },
        **cycle_components,
    }
    return total, components


@torch.inference_mode()
def evaluate(
    model: CubeTemporalNet,
    dataset: KRadarTemporalDataset,
    pair_indices: list[int],
    teacher_cache: FrozenPredictionCache,
    axes,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    model.eval()
    geometry_reports = []
    doppler_reports = []
    cycle_reports = []
    temporal_reports = []
    frames = []
    for pair_index in pair_indices:
        pair = dataset.pairs[pair_index]
        prior_state = teacher_state(teacher_cache, dataset, pair, device)
        current_item = dataset.frame_dataset[pair["current_dataset_index"]]
        output = predict_pair(
            model, current_item, prior_state, pair, axes, config, device
        )
        target_spectrum = query_cube_spectrum(
            output["cube"], output["indices"]
        )
        doppler = doppler_distribution_report(
            output["prediction"]["probability"].float(),
            target_spectrum.float(),
            model.doppler_mps,
            model.doppler_lower_mps,
            model.doppler_period_mps,
            model.doppler_step_mps,
            confidence=output["confidence"],
            static_center_mps=output["current_static_center_mps"],
            predicted_static_probability=output["prediction"].get(
                "static_probability"
            ),
        )
        target = current_item["target_xyz_confidence"].to(device)
        geometry = geometry_report(
            output["prediction"]["xyz_m"].float(),
            target[:, :3],
            target_weight=target[:, 3],
        )
        rendered = soft_splat_raed(
            output["prediction"]["coordinates_rae"].float(),
            output["prediction"]["probability"].float(),
            output["confidence"].float(),
        )
        cycle = cube_cycle_report(
            rendered, output["cube"][0].float(), output["confidence"].float()
        )
        match = temporal_match(
            prior_state.xyz_m,
            prior_state.probability,
            prior_state.confidence,
            output["prediction"]["xyz_m"].float(),
            output["prediction"]["probability"].float(),
            output["confidence"].float(),
            output["transform"],
            output["delta_seconds"],
            model.doppler_mps,
            model.doppler_lower_mps,
            model.doppler_period_mps,
            previous_static_center_mps=prior_state.static_center_mps,
            current_static_center_mps=output["current_static_center_mps"],
            dynamic_threshold_mps=config.dynamic_threshold_mps,
        )
        temporal = temporal_consistency_report(
            match,
            prior_state.confidence,
            output["prediction"]["xyz_m"].float(),
            output["confidence"].float(),
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
                "sequence": pair["sequence"],
                "previous_frame_in_window": pair["previous_frame_in_window"],
                "current_frame_in_window": pair["current_frame_in_window"],
                "radar_index": int(current_item["radar_index"]),
                "generated_geometry": geometry,
                "doppler": doppler,
                "cycle": cycle,
                "temporal": temporal,
            }
        )
        del prior_state, current_item, output, target_spectrum, target
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
        metrics["temporal"]["temporal_radial_error_mean_m"]["median"]
        + 0.25 * metrics["generated_geometry"]["chamfer_m"]["median"]
        + 0.25 * metrics["cycle"]["local_spectrum_kl"]["median"]
    )


def save_checkpoint(
    path: Path,
    model: CubeTemporalNet,
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


def recorded_best(output: Path, maximum_epoch: int) -> tuple[float, int]:
    records = []
    for path in output.glob("metrics_epoch_*.json"):
        epoch = int(path.stem.rsplit("_", maxsplit=1)[1])
        if epoch <= maximum_epoch:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            records.append((selection_score(metrics), epoch))
    if not records:
        raise ValueError("Temporal resume has no validation metrics")
    return min(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--dense-cache-report", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-prediction-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fusion-mode", choices=FUSION_MODES, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--joint-start-epoch", type=int, default=6)
    parser.add_argument("--temporal-learning-rate", type=float, default=3e-4)
    parser.add_argument("--parent-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--scheduled-sampling-maximum", type=float, default=0.4)
    parser.add_argument("--dynamic-threshold-mps", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-pairs", type=int, default=32)
    parser.add_argument("--train-window-limit", type=int, default=None)
    parser.add_argument("--validation-window-limit", type=int, default=None)
    parser.add_argument("--attention-neighbors", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Cube temporal training requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.epochs < 1 or not 1 <= args.joint_start_epoch <= args.epochs + 1:
        raise ValueError("Invalid temporal epoch schedule")
    if not 0.0 <= args.scheduled_sampling_maximum <= 1.0:
        raise ValueError("Scheduled-sampling maximum must be in [0,1]")
    if args.eval_every < 1 or args.max_eval_pairs < 1:
        raise ValueError("Temporal evaluation intervals must be positive")

    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    dense_cache_report = json.loads(
        args.dense_cache_report.read_text(encoding="utf-8")
    )
    if dense_cache_report.get("completed") is not True:
        raise ValueError("Temporal training requires a complete dense cache")
    if (
        dense_cache_report["configuration"]["source_manifest_sha256"]
        != manifest_hash
    ):
        raise ValueError("Temporal target cache and manifest differ")
    parent_document = json.loads(
        (args.parent_run / "config.json").read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    parent_provenance = parent_document["provenance"]
    if parent_config.get("variant") not in ("none", "full"):
        raise ValueError("Temporal parent must be C0 or C3")
    if int(parent_config["seed"]) != args.seed:
        raise ValueError("Temporal seed must match the single-frame parent")
    if parent_provenance["scene_split_sha256"] != scene_split_hash:
        raise ValueError("Temporal and parent scene splits differ")
    if parent_provenance["normalization_sha256"] != normalization_hash:
        raise ValueError("Temporal and parent normalization differ")
    parent_checkpoint = (args.parent_run / "best.pt").resolve()
    parent_checkpoint_hash = sha256(parent_checkpoint)
    temporal_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    teacher_cache = FrozenPredictionCache(
        args.parent_prediction_cache, expected_frames=len(temporal_manifest["frames"])
    )
    if teacher_cache.configuration["source_manifest_sha256"] != manifest_hash:
        raise ValueError("Teacher prediction cache and manifest differ")
    if (
        teacher_cache.configuration["parent_checkpoint_sha256"]
        != parent_checkpoint_hash
    ):
        raise ValueError("Teacher cache and temporal parent checkpoints differ")
    if (
        teacher_cache.configuration["dense_cache_report_sha256"]
        != sha256(args.dense_cache_report)
    ):
        raise ValueError("Teacher and temporal dense-cache reports differ")

    config = TrainConfig(
        fusion_mode=args.fusion_mode,
        parent_variant=parent_config["variant"],
        epochs=args.epochs,
        joint_start_epoch=args.joint_start_epoch,
        temporal_learning_rate=args.temporal_learning_rate,
        parent_learning_rate=args.parent_learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        point_count=int(parent_config["point_count"]),
        geometry_weight=args.geometry_weight,
        temporal_weight=args.temporal_weight,
        scheduled_sampling_maximum=args.scheduled_sampling_maximum,
        dynamic_threshold_mps=args.dynamic_threshold_mps,
        eval_every=args.eval_every,
        max_eval_pairs=args.max_eval_pairs,
        train_window_limit=args.train_window_limit,
        validation_window_limit=args.validation_window_limit,
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
        static_hypothesis=parent_config["static_hypothesis"],
        maximum_offset_bins=float(parent_config["maximum_offset_bins"]),
        attention_neighbors=args.attention_neighbors,
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
        raise FileExistsError(f"Temporal output is not empty: {args.output}")
    if args.resume and not nonempty:
        raise FileNotFoundError(f"No temporal run to resume: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    axes = load_axes(args.data_root / "resources")
    model = CubeTemporalNet(
        config.fusion_mode,
        parent_config["parent_head_mode"],
        torch.from_numpy(axes.doppler_mps),
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        base_channels=config.base_channels,
        log_center=config.log_center,
        log_scale=config.log_scale,
        static_hypothesis=config.static_hypothesis,
        maximum_offset_bins=config.maximum_offset_bins,
        attention_neighbors=config.attention_neighbors,
    ).to(device)
    checkpoint = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or not missing or any(
        not temporal_parameter(name) for name in missing
    ):
        raise ValueError(
            f"Unexpected temporal parent initialization: missing={missing}, "
            f"unexpected={unexpected}"
        )
    parent_parameters = []
    temporal_parameters = []
    for name, parameter in model.named_parameters():
        (temporal_parameters if temporal_parameter(name) else parent_parameters).append(
            parameter
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": parent_parameters, "lr": config.parent_learning_rate},
            {"params": temporal_parameters, "lr": config.temporal_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    provenance = {
        "git_commit": args.source_commit,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": normalization_hash,
        "dense_cache_report_sha256": sha256(args.dense_cache_report),
        "parent_prediction_manifest_sha256": sha256(
            teacher_cache.manifest_path
        ),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": parent_checkpoint_hash,
        "parent_git_commit": parent_provenance["git_commit"],
        "model_parameter_count": parameter_count(model),
        "parent_model_parameter_count": int(
            parent_provenance["model_parameter_count"]
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    relative_parameter_increase = (
        provenance["model_parameter_count"]
        - provenance["parent_model_parameter_count"]
    ) / provenance["parent_model_parameter_count"]
    provenance["relative_parameter_increase"] = relative_parameter_increase
    if relative_parameter_increase > 0.05:
        raise ValueError("Temporal fusion exceeds the preregistered 5% parameter budget")
    run_document = {"config": asdict(config), "provenance": provenance}
    config_path = args.output / "config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != run_document:
            raise ValueError("Temporal resume configuration or provenance differs")
    else:
        atomic_json(config_path, run_document)

    train_dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, ("train",)
    )
    validation_dataset = KRadarTemporalDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    train_windows_all = pairs_by_window(train_dataset)
    validation_windows_all = pairs_by_window(validation_dataset)
    train_window_names = sorted(train_windows_all)
    validation_window_names = sorted(validation_windows_all)
    train_window_names = [
        train_window_names[position]
        for position in selected_positions(
            len(train_window_names), config.train_window_limit
        )
    ]
    validation_window_names = [
        validation_window_names[position]
        for position in selected_positions(
            len(validation_window_names), config.validation_window_limit
        )
    ]
    validation_pair_indices = [
        index
        for index, pair in enumerate(validation_dataset.pairs)
        if pair["window_id"] in validation_window_names
    ]
    evaluation_pair_indices = [
        validation_pair_indices[position]
        for position in selected_positions(
            len(validation_pair_indices),
            min(config.max_eval_pairs, len(validation_pair_indices)),
        )
    ]

    start_epoch = 1
    best_score = float("inf")
    prior_elapsed = 0.0
    log_path = args.output / "train_log.jsonl"
    if args.resume:
        last = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        if last["config"] != asdict(config) or last["provenance"] != provenance:
            raise ValueError("Temporal last checkpoint metadata differs")
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        last_epoch = int(last["epoch"])
        start_epoch = last_epoch + 1
        best_score, best_epoch = recorded_best(args.output, last_epoch)
        best_path = args.output / "best.pt"
        recorded_best_epoch = None
        if best_path.exists():
            recorded_best_epoch = int(
                torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]
            )
        if recorded_best_epoch != best_epoch:
            if best_epoch != last_epoch:
                raise ValueError("Temporal best checkpoint and metrics differ")
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
            for line in log_path.read_text(encoding="utf-8").splitlines()
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
                raise ValueError("Temporal log and last checkpoint differ")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(checkpoint_record) + "\n")
            records.append(checkpoint_record)
        prior_elapsed = float(records[-1]["elapsed_seconds"])

    print(
        json.dumps(
            {
                "parameters": parameter_count(model),
                "relative_parameter_increase": relative_parameter_increase,
                "train_windows": len(train_window_names),
                "validation_windows": len(validation_window_names),
                "evaluation_pairs": len(evaluation_pair_indices),
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
        parent_trainable = epoch >= config.joint_start_epoch
        for name, parameter in model.named_parameters():
            if not temporal_parameter(name):
                parameter.requires_grad_(parent_trainable)
        window_order = train_window_names.copy()
        random.Random(config.seed + epoch).shuffle(window_order)
        sampling_rng = random.Random(config.seed * 10_000 + epoch)
        sampling_probability = scheduled_probability(config, epoch)
        loss_values = []
        components: dict[str, list[float]] = defaultdict(list)
        teacher_count = 0
        scheduled_count = 0
        for window_id in window_order:
            recurrent_state = None
            for pair in train_windows_all[window_id]:
                use_scheduled = (
                    recurrent_state is not None
                    and sampling_rng.random() < sampling_probability
                )
                if use_scheduled:
                    prior_state = recurrent_state
                    scheduled_count += 1
                else:
                    prior_state = teacher_state(
                        teacher_cache, train_dataset, pair, device
                    )
                    teacher_count += 1
                current_item = train_dataset.frame_dataset[
                    pair["current_dataset_index"]
                ]
                optimizer.zero_grad(set_to_none=True)
                output = predict_pair(
                    model,
                    current_item,
                    prior_state,
                    pair,
                    axes,
                    config,
                    device,
                )
                total, component = training_loss(
                    model,
                    current_item,
                    prior_state,
                    output,
                    config,
                    device,
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                recurrent_state = prediction_from_output(
                    output["prediction"],
                    output["confidence"],
                    output["current_static_center_mps"],
                ).detached()
                loss_values.append(float(total.detach().item()))
                for name, value in component.items():
                    components[name].append(value)
                del current_item, output, total, prior_state
        scheduler.step()
        record = {
            "epoch": epoch,
            "parent_trainable": parent_trainable,
            "scheduled_sampling_probability": sampling_probability,
            "teacher_exposure_count": teacher_count,
            "scheduled_exposure_count": scheduled_count,
            "train_loss_mean": float(np.mean(loss_values)),
            "train_components": {
                name: float(np.mean(values)) for name, values in components.items()
            },
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "elapsed_seconds": round(prior_elapsed + time.monotonic() - started, 3),
        }
        should_evaluate = epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs
        if should_evaluate:
            metrics = evaluate(
                model,
                validation_dataset,
                evaluation_pair_indices,
                teacher_cache,
                axes,
                config,
                device,
            )
            score = selection_score(metrics)
            record["validation"] = metrics
            record["selection_score"] = score
            metrics_path = args.output / f"metrics_epoch_{epoch:04d}.json"
            atomic_json(metrics_path, metrics)
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
        model,
        validation_dataset,
        validation_pair_indices,
        teacher_cache,
        axes,
        config,
        device,
    )
    full_validation_score = selection_score(final_metrics)
    final_report = {
        "best_epoch": int(best["epoch"]),
        "selection_metric": "temporal_radial_error + 0.25 * current_chamfer + 0.25 * local_spectrum_kl",
        "selection_value": full_validation_score,
        "full_validation_selection_value": full_validation_score,
        "checkpoint_selection_value": best_score,
        "validation": final_metrics,
    }
    atomic_json(args.output / "best_validation_metrics.json", final_report)
    print(json.dumps({"best_validation": final_report}), flush=True)


if __name__ == "__main__":
    main()
