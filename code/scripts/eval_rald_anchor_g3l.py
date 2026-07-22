#!/usr/bin/env python3
"""Evaluate RaLD-faithful G3L posterior reconstruction or EDM generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.rald_run import load_rald_run  # noqa: E402
from eval.cube_cycle import aggregate_cycle_reports, cube_cycle_report  # noqa: E402
from eval.dense_geometry import aggregate_geometry_reports, geometry_report  # noqa: E402
from eval.doppler_distribution import (  # noqa: E402
    aggregate_doppler_reports,
    doppler_distribution_report,
)
from losses.cube_cycle import existence_confidence_loss  # noqa: E402
from losses.rald_anchor import nearest_target_assignment  # noqa: E402
from models.cube_doppler import query_cube_spectrum  # noqa: E402
from models.point_to_cube import soft_splat_raed  # noqa: E402
from models.rald_anchor_ldm import RaLDAnchorLDM  # noqa: E402
from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256  # noqa: E402
from scripts.train_cube_doppler import move_frame  # noqa: E402
from scripts.train_rald_anchor_g3l_edm import (  # noqa: E402
    OFFICIAL_DENOISER_DEPTH,
    OFFICIAL_EDM_STEPS,
    OFFICIAL_EPOCHS,
    OFFICIAL_P_MEAN,
    OFFICIAL_P_STD,
    OFFICIAL_RHO,
    OFFICIAL_SIGMA_DATA,
    OFFICIAL_SIGMA_MAX,
    OFFICIAL_SIGMA_MIN,
)
from scripts.train_rald_anchor_g3l_vae import (  # noqa: E402
    TrainConfig,
    build_model,
    require_h200,
)


PROTOCOL = "rald_anchor_g3l_evaluation_v1"
MODES = ("posterior_mean", "edm_sample")
FORMAL_SEEDS = tuple(FROZEN_G1B_SEEDS)
FRAME_SEED_PROTOCOL = "sha256(g3l-frame-seed-v1,sequence,radar_index,stream)"
VAE_PROTOCOL = "rald_anchor_g3l_physical_vae_v1"
EDM_PROTOCOL = "rald_anchor_g3l2_full_raed_edm_training_v1"


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def frame_seed(sequence: int, radar_index: int, stream: int = 0) -> int:
    payload = f"g3l-frame-seed-v1:{sequence}:{radar_index}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def cross_scene_condition_indices(records: list[dict]) -> list[int]:
    """Return a deterministic derangement whose paired frames cross scenes."""

    count = len(records)
    if count < 2:
        raise ValueError("Condition shuffle requires at least two validation frames")
    sequences = [int(record["sequence"]) for record in records]
    for shift in range(1, count):
        candidate = [(index + shift) % count for index in range(count)]
        if all(sequences[index] != sequences[other] for index, other in enumerate(candidate)):
            return candidate
    raise ValueError("No cross-scene condition derangement exists")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return document


def _train_config(document: dict) -> TrainConfig:
    expected = {field.name for field in fields(TrainConfig)}
    config = document.get("config")
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("G3L-1 configuration does not match the frozen schema")
    return TrainConfig(**config)


def validate_vae_run(
    run: Path,
    *,
    source_commit: str,
    artifact_hashes: dict[str, str],
) -> dict:
    run = run.expanduser().resolve()
    config_path = run / "config.json"
    checkpoint_path = run / "best.pt"
    metrics_path = run / "best_validation_metrics.json"
    document = _load_json(config_path)
    config = _train_config(document)
    provenance = document.get("provenance")
    if config.protocol != VAE_PROTOCOL or not isinstance(provenance, dict):
        raise ValueError("G3L-1 run protocol or provenance differs")
    if config.seed not in FORMAL_SEEDS:
        raise ValueError("G3L-1 run does not use a frozen formal seed")
    if (config.latent_count, config.latent_dim) != (512, 32):
        raise ValueError("G3L-1 latent state must be exactly 512x32")
    if config.decoder_depth != 24 or config.edm_steps != 18:
        raise ValueError("G3L-1 does not preserve the RaLD architecture contract")
    if provenance.get("git_commit") != source_commit:
        raise ValueError("G3L-1 source commit mismatch")
    for key, expected in artifact_hashes.items():
        if provenance.get(key) != expected:
            raise ValueError(f"G3L-1 {key} mismatch")
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Incomplete G3L-1 run: {run}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != document["config"]:
        raise ValueError("G3L-1 checkpoint config differs from config.json")
    if checkpoint.get("provenance") != provenance:
        raise ValueError("G3L-1 checkpoint provenance differs from config.json")
    if not isinstance(checkpoint.get("g3l_vae"), dict):
        raise ValueError("G3L-1 checkpoint lacks the physical VAE")
    metrics = _load_json(metrics_path)
    if metrics.get("completed") is not True:
        raise ValueError("G3L-1 training report is incomplete")
    if metrics.get("posterior_sampling", {}).get("best_of_k") is not False:
        raise ValueError("G3L-1 best-of-k is prohibited")

    parent_run = Path(provenance["g3r_selected_run"]).expanduser().resolve()
    parent = load_rald_run(parent_run, expected_variant="full")
    live_parent_hashes = {
        "g3r_selected_config_sha256": sha256(parent["config_path"]),
        "g3r_selected_checkpoint_sha256": sha256(parent["checkpoint_path"]),
        "g3r_geometry_parent_checkpoint_sha256": sha256(parent["parent_checkpoint"]),
    }
    for key, expected in live_parent_hashes.items():
        if provenance.get(key) != expected:
            raise ValueError(f"G3L-1 live parent {key} mismatch")
    if int(parent["config"]["seed"]) != config.seed:
        raise ValueError("G3L-1 and G3R parent seeds differ")
    return {
        "run": run,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "config": config,
        "provenance": provenance,
        "checkpoint": checkpoint,
        "parent": parent,
    }


def validate_edm_run(
    run: Path,
    *,
    source_commit: str,
    vae: dict,
    artifact_hashes: dict[str, str],
) -> dict:
    run = run.expanduser().resolve()
    config_path = run / "config.json"
    checkpoint_path = run / "final.pt"
    summary_path = run / "training_summary.json"
    document = _load_json(config_path)
    summary = _load_json(summary_path)
    config = document.get("config")
    provenance = document.get("provenance")
    if document.get("protocol") != EDM_PROTOCOL:
        raise ValueError("G3L-2 run protocol differs")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("G3L-2 configuration or provenance is invalid")
    expected_schedule = {
        "epochs": OFFICIAL_EPOCHS,
        "denoiser_depth": OFFICIAL_DENOISER_DEPTH,
        "p_mean": OFFICIAL_P_MEAN,
        "p_std": OFFICIAL_P_STD,
        "sigma_data": OFFICIAL_SIGMA_DATA,
        "inference_steps": OFFICIAL_EDM_STEPS,
        "sigma_min": OFFICIAL_SIGMA_MIN,
        "sigma_max": OFFICIAL_SIGMA_MAX,
        "rho": OFFICIAL_RHO,
        "sampler": "heun",
    }
    if summary.get("status") != "completed" or summary.get("official_schedule") != expected_schedule:
        raise ValueError("G3L-2 run did not complete the frozen RaLD schedule")
    if int(config.get("seed", -1)) != vae["config"].seed:
        raise ValueError("G3L-2 and G3L-1 seeds differ")
    if provenance.get("git_commit") != source_commit:
        raise ValueError("G3L-2 source commit mismatch")
    for key, expected in artifact_hashes.items():
        if provenance.get(key) != expected:
            raise ValueError(f"G3L-2 {key} mismatch")
    if provenance.get("g3l1_checkpoint_sha256") != sha256(vae["checkpoint_path"]):
        raise ValueError("G3L-2 uses a different G3L-1 VAE")
    if any(
        provenance.get(key) is not expected
        for key, expected in {
            "test_accessed": False,
            "external_pretraining": False,
            "cfar_query_helper": False,
            "best_of_k": False,
        }.items()
    ):
        raise ValueError("G3L-2 provenance violates the formal protocol")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if summary.get("final_checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("G3L-2 final checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != config or checkpoint.get("provenance") != provenance:
        raise ValueError("G3L-2 final checkpoint metadata differs")
    if int(checkpoint.get("epoch", -1)) != OFFICIAL_EPOCHS:
        raise ValueError("G3L-2 final checkpoint is not epoch 100")
    return {
        "run": run,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "summary_path": summary_path,
        "config": config,
        "provenance": provenance,
        "checkpoint": checkpoint,
    }


def matched_spectrum(
    cube: torch.Tensor,
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_rae_index: torch.Tensor,
) -> torch.Tensor:
    _, target_index = nearest_target_assignment(source_xyz, target_xyz)
    return query_cube_spectrum(cube, target_rae_index)[target_index]


def physical_frame_report(
    output: dict,
    cube: torch.Tensor,
    target: torch.Tensor,
    target_index: torch.Tensor,
    axes,
) -> dict:
    generated_xyz = output["xyz_m"][0].float()
    target_xyz = target[:, :3].float()
    geometry = geometry_report(
        generated_xyz, target_xyz, target_weight=target[:, 3]
    )
    target_spectrum = matched_spectrum(
        cube, generated_xyz, target_xyz, target_index
    )
    doppler_axis = torch.as_tensor(
        axes.doppler_mps, device=cube.device, dtype=torch.float32
    )
    doppler_step = torch.as_tensor(
        np.median(np.diff(axes.doppler_mps)), device=cube.device, dtype=torch.float32
    )
    doppler = doppler_distribution_report(
        output["doppler_probability"][0].float(),
        target_spectrum,
        doppler_axis,
        doppler_axis[0],
        doppler_step * len(axes.doppler_mps),
        doppler_step,
        confidence=output["confidence"][0].float(),
    )
    prediction_distance, _ = nearest_target_assignment(generated_xyz, target_xyz)
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
    return {"geometry": geometry, "doppler": doppler, "cycle": cycle}


def aggregate_reports(frames: list[dict]) -> dict:
    return {
        "frame_count": len(frames),
        "geometry": aggregate_geometry_reports([frame["geometry"] for frame in frames]),
        "doppler": aggregate_doppler_reports([frame["doppler"] for frame in frames]),
        "cycle": aggregate_cycle_reports([frame["cycle"] for frame in frames]),
        "frames": frames,
    }


def posterior_diagnostics(output: dict, model) -> dict[str, float | bool]:
    mean = output["posterior_mean"].float()
    variance = output["posterior_log_variance"].float().exp()
    rolled = output["target_doppler_probability"].roll(16, dims=-1)
    inverted_confidence = 1.0 - output["target_confidence"]
    doppler_posterior = model.ldm.posterior_encoder(
        output["target_normalized_rae"], rolled, output["target_confidence"]
    )
    confidence_posterior = model.ldm.posterior_encoder(
        output["target_normalized_rae"],
        output["target_doppler_probability"],
        inverted_confidence,
    )
    doppler_query = model.ldm.decoder(
        doppler_posterior.mean,
        output["anchor_normalized_rae"],
        output["anchor_features"],
    )
    confidence_query = model.ldm.decoder(
        confidence_posterior.mean,
        output["anchor_normalized_rae"],
        output["anchor_features"],
    )
    return {
        "finite": bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()),
        "posterior_variance_mean": float(variance.mean().item()),
        "posterior_mean_rms": float(mean.square().mean().sqrt().item()),
        "doppler_intervention_latent_rms": float(
            (doppler_posterior.mean.float() - mean).square().mean().sqrt().item()
        ),
        "confidence_intervention_latent_rms": float(
            (confidence_posterior.mean.float() - mean).square().mean().sqrt().item()
        ),
        "doppler_intervention_decoder_rms": float(
            (doppler_query.float() - output["query_features"].float())
            .square()
            .mean()
            .sqrt()
            .item()
        ),
        "confidence_intervention_decoder_rms": float(
            (confidence_query.float() - output["query_features"].float())
            .square()
            .mean()
            .sqrt()
            .item()
        ),
    }


@torch.inference_mode()
def evaluate_posterior(model, dataset, axes, device: torch.device) -> dict:
    frames = []
    posterior_means = []
    diagnostics = []
    for index in range(len(dataset)):
        item = dataset[index]
        cube, _ = move_frame(item, device)
        target = item["target_xyz_confidence"].to(device)
        target_index = item["target_rae_index"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                cube, target_index, target[:, 3], sample_posterior=False
            )
        report = physical_frame_report(output, cube, target, target_index, axes)
        diagnostic = posterior_diagnostics(output, model)
        report.update(
            {
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
                "posterior": diagnostic,
            }
        )
        frames.append(report)
        diagnostics.append(diagnostic)
        posterior_means.append(output["posterior_mean"][0].float().cpu())
        del item, cube, target, target_index, output
        torch.cuda.empty_cache()
    aggregate = aggregate_reports(frames)
    stacked = torch.stack(posterior_means)
    aggregate["posterior"] = {
        "all_finite": all(bool(item["finite"]) for item in diagnostics),
        "variance_mean": float(
            np.mean([item["posterior_variance_mean"] for item in diagnostics])
        ),
        "across_frame_mean_std": float(stacked.std(dim=0).mean().item()),
        **{
            key + "_mean": float(np.mean([item[key] for item in diagnostics]))
            for key in (
                "doppler_intervention_latent_rms",
                "confidence_intervention_latent_rms",
                "doppler_intervention_decoder_rms",
                "confidence_intervention_decoder_rms",
            )
        },
    }
    return aggregate


def diversity_summary(outputs: list[dict]) -> dict:
    latents = torch.stack([output["latent"][0].float().cpu() for output in outputs])
    xyz = torch.stack([output["xyz_m"][0].float().cpu() for output in outputs])
    latent_distances = []
    xyz_distances = []
    for first in range(len(outputs)):
        for second in range(first + 1, len(outputs)):
            latent_distances.append(
                float((latents[first] - latents[second]).square().mean().sqrt().item())
            )
            xyz_distances.append(
                float((xyz[first] - xyz[second]).square().mean().sqrt().item())
            )
    return {
        "sample_count": len(outputs),
        "latent_pairwise_rms_mean": float(np.mean(latent_distances)),
        "xyz_pairwise_rms_m_mean": float(np.mean(xyz_distances)),
    }


@torch.inference_mode()
def evaluate_edm(
    model,
    dataset,
    axes,
    device: torch.device,
    diversity_frame_limit: int,
) -> dict:
    shuffled_indices = cross_scene_condition_indices(dataset.records)
    clean_frames = []
    shuffled_frames = []
    diversity = []
    for index, condition_index in enumerate(shuffled_indices):
        item = dataset[index]
        condition_item = dataset[condition_index]
        cube, _ = move_frame(item, device)
        condition_cube, _ = move_frame(condition_item, device)
        target = item["target_xyz_confidence"].to(device)
        target_index = item["target_rae_index"].to(device)
        seed = frame_seed(int(item["sequence"]), int(item["radar_index"]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            clean = model.sample_edm(cube, [seed])
            shuffled = model.sample_edm(
                cube, [seed], condition_cube_drae=condition_cube
            )
        clean_report = physical_frame_report(clean, cube, target, target_index, axes)
        shuffled_report = physical_frame_report(
            shuffled, cube, target, target_index, axes
        )
        identity = {
            "sequence": int(item["sequence"]),
            "radar_index": int(item["radar_index"]),
            "sample_seed": seed,
        }
        clean_frames.append({**identity, **clean_report})
        shuffled_frames.append(
            {
                **identity,
                "condition_sequence": int(condition_item["sequence"]),
                "condition_radar_index": int(condition_item["radar_index"]),
                **shuffled_report,
            }
        )
        if index < diversity_frame_limit:
            outputs = [clean]
            for stream in range(1, 4):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs.append(
                        model.sample_edm(
                            cube,
                            [
                                frame_seed(
                                    int(item["sequence"]),
                                    int(item["radar_index"]),
                                    stream,
                                )
                            ],
                        )
                    )
            diversity.append({**identity, **diversity_summary(outputs)})
        del item, condition_item, cube, condition_cube, target, target_index
        del clean, shuffled
        torch.cuda.empty_cache()
    return {
        "clean": aggregate_reports(clean_frames),
        "condition_shuffle": aggregate_reports(shuffled_frames),
        "condition_shuffle_contract": {
            "cross_scene_only": True,
            "same_anchor_and_measured_cube": True,
            "same_frame_seed": True,
        },
        "diversity": {
            "descriptive_only": True,
            "frame_count": len(diversity),
            "frames": diversity,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--g3l1-run", type=Path, required=True)
    parser.add_argument("--g3l1-source-commit", required=True)
    parser.add_argument("--g3l2-run", type=Path)
    parser.add_argument("--g3l2-source-commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diversity-frame-limit", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    if args.diversity_frame_limit < 0:
        raise ValueError("Diversity frame limit must be non-negative")
    if args.mode == "edm_sample" and (
        args.g3l2_run is None or args.g3l2_source_commit is None
    ):
        raise ValueError("EDM evaluation requires its run and source commit")
    if args.mode == "posterior_mean" and (
        args.g3l2_run is not None or args.g3l2_source_commit is not None
    ):
        raise ValueError("Posterior evaluation cannot consume a G3L-2 run")
    device, device_name = require_h200(args.device)
    artifact_hashes = {
        "manifest_sha256": sha256(args.manifest),
        "scene_split_sha256": sha256(args.scene_split),
        "normalization_sha256": sha256(args.normalization),
    }
    vae = validate_vae_run(
        args.g3l1_run,
        source_commit=args.g3l1_source_commit,
        artifact_hashes=artifact_hashes,
    )
    axes = load_axes(args.data_root / "resources")
    model = build_model(vae["parent"], axes, vae["config"], device)
    model.load_vae_state_dict(vae["checkpoint"]["g3l_vae"], strict=True)
    model.eval().requires_grad_(False)
    dataset = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    if any(record.get("partition") != "validation" for record in dataset.records):
        raise ValueError("G3L evaluation must remain validation-only")

    edm = None
    if args.mode == "posterior_mean":
        evaluation = evaluate_posterior(model, dataset, axes, device)
    else:
        edm = validate_edm_run(
            args.g3l2_run,
            source_commit=args.g3l2_source_commit,
            vae=vae,
            artifact_hashes=artifact_hashes,
        )
        model.ldm.edm.load_state_dict(edm["checkpoint"]["model"], strict=True)
        evaluation = evaluate_edm(
            model, dataset, axes, device, args.diversity_frame_limit
        )
    report = {
        "protocol": PROTOCOL,
        "mode": args.mode,
        "completed": True,
        "seed": vae["config"].seed,
        "frame_seed_protocol": FRAME_SEED_PROTOCOL,
        "single_sample_per_frame": True,
        "best_of_k": False,
        "partitions": ["validation"],
        "test_accessed": False,
        "vae_run": str(vae["run"]),
        "vae_config_sha256": sha256(vae["config_path"]),
        "vae_checkpoint_sha256": sha256(vae["checkpoint_path"]),
        "edm_run": None if edm is None else str(edm["run"]),
        "edm_config_sha256": None if edm is None else sha256(edm["config_path"]),
        "edm_checkpoint_sha256": (
            None if edm is None else sha256(edm["checkpoint_path"])
        ),
        "artifact_hashes": artifact_hashes,
        "device": device_name,
        "evaluation": evaluation,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
