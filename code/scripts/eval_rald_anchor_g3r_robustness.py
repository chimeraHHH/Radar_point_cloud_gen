#!/usr/bin/env python3
"""Evaluate RaLD-anchor no-cycle/full-cycle robustness for G3R."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.cube_cycle import (  # noqa: E402
    aggregate_cycle_reports,
    cube_cycle_report,
)
from eval.dense_geometry import (  # noqa: E402
    aggregate_geometry_reports,
    geometry_report,
)
from eval.doppler_distribution import (  # noqa: E402
    aggregate_doppler_reports,
    cd_doppler_report,
    doppler_distribution_report,
)
from losses.cube_cycle import existence_confidence_loss  # noqa: E402
from losses.doppler_distribution import circular_scalar_target  # noqa: E402
from losses.rald_anchor import nearest_target_assignment  # noqa: E402
from models.cube_doppler import query_cube_spectrum  # noqa: E402
from models.cube_occupancy import CubeOccupancyNet  # noqa: E402
from models.point_to_cube import soft_splat_raed  # noqa: E402
from models.rald_anchor import FrozenParentRaLDRefiner  # noqa: E402
from models.rald_matched import FullRAEDRadarTokenEncoder  # noqa: E402
from scripts.eval_g3_cube_cycle_robustness import (  # noqa: E402
    CONDITIONS,
    frame_seed,
    perturb_cube,
)
from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256  # noqa: E402


PROTOCOL = "rald_anchor_g3r_robustness_v1"
VARIANTS = ("none", "full")
REQUIRED_AGGREGATES = (
    ("cycle", "local_spectrum_kl"),
    ("doppler", "spectrum_nll"),
    ("generated_geometry", "chamfer_m"),
    ("cd_doppler", "cd_doppler"),
    ("cycle", "covered_cell_count"),
)


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_run(path: Path, expected_variant: str) -> dict:
    path = path.resolve()
    config_path = path / "config.json"
    checkpoint_path = path / "best.pt"
    metrics_path = path / "best_validation_metrics.json"
    if not all(candidate.is_file() for candidate in (config_path, checkpoint_path, metrics_path)):
        raise FileNotFoundError(f"Incomplete G3R robustness run: {path}")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if config["cycle_variant"] != expected_variant:
        raise ValueError(f"Expected {expected_variant} run in {path}")
    if config["doppler_head_mode"] != "distribution":
        raise ValueError("G3R robustness requires distribution-head runs")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame_keys = sorted(
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in metrics["final"]["frames"]
    )
    parent_checkpoint = Path(provenance["parent_g1_checkpoint"]).resolve()
    parent_config_path = parent_checkpoint.parent / "config.json"
    if not parent_checkpoint.is_file() or not parent_config_path.is_file():
        raise FileNotFoundError(f"Missing frozen geometry parent for {path}")
    if sha256(parent_checkpoint) != provenance["parent_g1_checkpoint_sha256"]:
        raise ValueError(f"Geometry parent hash changed for {path}")
    return {
        "path": path,
        "config_path": config_path,
        "config_sha256": sha256(config_path),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256(checkpoint_path),
        "config": config,
        "provenance": provenance,
        "parent_checkpoint": parent_checkpoint,
        "parent_config_path": parent_config_path,
        "clean_frame_keys": frame_keys,
    }


def validate_runs(runs: list[dict], required_seeds: int) -> None:
    grouped: dict[str, dict[int, dict]] = {variant: {} for variant in VARIANTS}
    for run in runs:
        variant = run["config"]["cycle_variant"]
        seed = int(run["config"]["seed"])
        if seed in grouped[variant]:
            raise ValueError(f"Duplicate G3R robustness {variant} seed {seed}")
        grouped[variant][seed] = run
    expected = set(FROZEN_G1B_SEEDS)
    seed_sets = [set(grouped[variant]) for variant in VARIANTS]
    if seed_sets[0] != seed_sets[1]:
        raise ValueError("G3R robustness seed sets differ")
    if len(seed_sets[0]) != required_seeds or seed_sets[0] != expected:
        raise ValueError("G3R robustness requires the frozen three-seed matrix")
    reference_frames = None
    reference_global = None
    for seed in sorted(expected):
        reference_seed_config = None
        initial_hash = None
        parent_hash = None
        for variant in VARIANTS:
            run = grouped[variant][seed]
            seed_config = {
                key: value
                for key, value in run["config"].items()
                if key != "cycle_variant"
            }
            provenance = run["provenance"]
            global_provenance = tuple(
                provenance.get(key)
                for key in (
                    "manifest_sha256",
                    "scene_split_sha256",
                    "normalization_sha256",
                    "g1_comparison_sha256",
                    "g1b_summary_sha256",
                )
            )
            if reference_frames is None:
                reference_frames = run["clean_frame_keys"]
                reference_global = global_provenance
            if run["clean_frame_keys"] != reference_frames:
                raise ValueError("G3R robustness clean frame sets differ")
            if global_provenance != reference_global:
                raise ValueError("G3R robustness data provenance differs")
            if reference_seed_config is None:
                reference_seed_config = seed_config
                initial_hash = provenance["initial_refiner_checkpoint_sha256"]
                parent_hash = provenance["parent_g1_checkpoint_sha256"]
            if seed_config != reference_seed_config:
                raise ValueError(f"G3R robustness seed {seed} is not paired")
            if provenance["initial_refiner_checkpoint_sha256"] != initial_hash:
                raise ValueError(f"G3R robustness seed {seed} initial states differ")
            if provenance["parent_g1_checkpoint_sha256"] != parent_hash:
                raise ValueError(f"G3R robustness seed {seed} parents differ")


def build_model(run: dict, axes, device: torch.device) -> FrozenParentRaLDRefiner:
    parent_document = json.loads(
        run["parent_config_path"].read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    config = run["config"]
    parent = CubeOccupancyNet(
        parent_config["mode"],
        torch.from_numpy(axes.doppler_mps),
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
    ).to(device)
    parent_checkpoint = torch.load(
        run["parent_checkpoint"], map_location=device, weights_only=False
    )
    parent.load_state_dict(parent_checkpoint["model"], strict=True)
    radar_encoder = FullRAEDRadarTokenEncoder(
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
        spectral_channels=int(config["radar_spectral_channels"]),
        token_dim=int(config["model_dim"]),
        base_channels=int(config["radar_base_channels"]),
    )
    model = FrozenParentRaLDRefiner(
        parent,
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        point_count=int(config["point_count"]),
        latent_count=int(config["latent_count"]),
        model_dim=int(config["model_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
        radar_encoder=radar_encoder,
        radar_token_dim=int(config["model_dim"]),
        doppler_head_mode=config["doppler_head_mode"],
    ).to(device)
    checkpoint = torch.load(run["checkpoint"], map_location=device, weights_only=False)
    model.refiner.load_state_dict(checkpoint["refiner"], strict=True)
    model.radar_encoder.load_state_dict(checkpoint["radar_encoder"], strict=True)
    return model.eval()


@torch.inference_mode()
def evaluate_condition(
    model: FrozenParentRaLDRefiner,
    dataset: KRadarCubeDataset,
    frame_indices: list[int],
    axes,
    condition,
    condition_index: int,
    model_seed: int,
    maximum_log10_power: float,
    device: torch.device,
) -> dict:
    geometry_reports = []
    doppler_reports = []
    cd_reports = []
    cycle_reports = []
    frames = []
    doppler_axis = torch.as_tensor(
        axes.doppler_mps, device=device, dtype=torch.float32
    )
    doppler_lower = torch.as_tensor(
        axes.doppler_mps[0], device=device, dtype=torch.float32
    )
    doppler_step = torch.as_tensor(
        np.median(np.diff(axes.doppler_mps)), device=device, dtype=torch.float32
    )
    doppler_period = doppler_step * len(axes.doppler_mps)
    for dataset_index in frame_indices:
        item = dataset[dataset_index]
        clean_cube = item["cube_drae"].unsqueeze(0).to(device)
        sequence = int(item["sequence"])
        radar_index = int(item["radar_index"])
        model_cube, input_diagnostics = perturb_cube(
            clean_cube,
            condition,
            frame_seed(model_seed, sequence, radar_index, condition_index),
            maximum_log10_power,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(model_cube)
        target = item["target_xyz_confidence"].to(device)
        target_index = item["target_rae_index"].to(device)
        target_xyz = target[:, :3].float()
        generated_xyz = output["xyz_m"][0].float()
        _, matched_index = nearest_target_assignment(generated_xyz, target_xyz)
        all_target_spectrum = query_cube_spectrum(clean_cube, target_index)
        matched_target_spectrum = all_target_spectrum[matched_index]
        confidence = torch.sigmoid(
            output["confidence_logit"][0].float()
            / condition.confidence_temperature
        )
        doppler = doppler_distribution_report(
            output["doppler_probability"][0].float(),
            matched_target_spectrum,
            doppler_axis,
            doppler_lower,
            doppler_period,
            doppler_step,
            confidence=output["anchor_parent_confidence"][0].float(),
        )
        predicted_velocity = circular_scalar_target(
            output["doppler_probability"][0].float(),
            doppler_axis,
            doppler_lower,
            doppler_period,
        )
        target_velocity = circular_scalar_target(
            all_target_spectrum,
            doppler_axis,
            doppler_lower,
            doppler_period,
        )
        cd_report = cd_doppler_report(
            generated_xyz,
            predicted_velocity,
            target_xyz,
            target_velocity,
            target_weight=target[:, 3],
        )
        prediction_distance, _ = nearest_target_assignment(
            generated_xyz, target_xyz
        )
        _, existence_target = existence_confidence_loss(
            confidence, prediction_distance
        )
        rendered = soft_splat_raed(
            output["coordinates_rae"][0].float(),
            output["doppler_probability"][0].float(),
            confidence,
        )
        cycle = cube_cycle_report(
            rendered,
            clean_cube[0].float(),
            confidence,
            existence_target=existence_target,
        )
        offset = output["offset_bins"][0].float()
        cycle["offset_abs_mean_bins"] = float(offset.abs().mean().item())
        cycle["offset_saturation_fraction"] = float(
            (offset.abs() >= 0.49).float().mean().item()
        )
        geometry = geometry_report(
            generated_xyz, target_xyz, target_weight=target[:, 3]
        )
        geometry_reports.append(geometry)
        doppler_reports.append(doppler)
        cd_reports.append(cd_report)
        cycle_reports.append(cycle)
        frames.append(
            {
                "sequence": sequence,
                "radar_index": radar_index,
                "generated_geometry": geometry,
                "doppler": doppler,
                "cd_doppler": cd_report,
                "cycle": cycle,
                "input_diagnostics": input_diagnostics,
            }
        )
        del item, clean_cube, model_cube, output, target, target_index
        del rendered
        torch.cuda.empty_cache()
    return {
        "frame_count": len(frames),
        "generated_geometry": aggregate_geometry_reports(geometry_reports),
        "doppler": aggregate_doppler_reports(doppler_reports),
        "cd_doppler": aggregate_doppler_reports(cd_reports),
        "cycle": aggregate_cycle_reports(cycle_reports),
        "frames": frames,
    }


def endpoint_values(metrics: dict) -> dict[str, float]:
    return {
        "local_spectrum_kl": float(metrics["cycle"]["local_spectrum_kl"]["mean"]),
        "spectrum_nll": float(metrics["doppler"]["spectrum_nll"]["mean"]),
        "geometry_chamfer_m": float(
            metrics["generated_geometry"]["chamfer_m"]["mean"]
        ),
        "cd_doppler": float(metrics["cd_doppler"]["cd_doppler"]["mean"]),
        "covered_cell_count": float(
            metrics["cycle"]["covered_cell_count"]["mean"]
        ),
    }


def degradation_curve(condition_results: dict[str, dict]) -> dict[str, dict]:
    clean = endpoint_values(condition_results["clean"])
    curves = {}
    for condition in CONDITIONS:
        current = endpoint_values(condition_results[condition.condition_id])
        curves[condition.condition_id] = {
            "local_spectrum_kl_increase": current["local_spectrum_kl"]
            - clean["local_spectrum_kl"],
            "spectrum_nll_increase": current["spectrum_nll"]
            - clean["spectrum_nll"],
            "geometry_chamfer_increase_m": current["geometry_chamfer_m"]
            - clean["geometry_chamfer_m"],
            "cd_doppler_increase": current["cd_doppler"] - clean["cd_doppler"],
            "covered_cell_relative_loss": (
                clean["covered_cell_count"] - current["covered_cell_count"]
            )
            / max(clean["covered_cell_count"], 1e-12),
        }
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--none-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--full-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("G3R robustness evaluation requires CUDA")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output}")

    runs = [load_run(path, "none") for path in args.none_runs]
    runs.extend(load_run(path, "full") for path in args.full_runs)
    validate_runs(runs, args.required_seeds)
    artifact_hashes = {
        "manifest_sha256": sha256(args.manifest),
        "scene_split_sha256": sha256(args.scene_split),
        "normalization_sha256": sha256(args.normalization_stats),
    }
    for run in runs:
        if any(
            run["provenance"][key] != value
            for key, value in artifact_hashes.items()
        ):
            raise ValueError(f"G3R robustness data differs for {run['path']}")
    dataset = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    frame_indices = list(range(len(dataset)))
    frame_keys = sorted(
        (int(dataset[index]["sequence"]), int(dataset[index]["radar_index"]))
        for index in frame_indices
    )
    if frame_keys != runs[0]["clean_frame_keys"]:
        raise ValueError("G3R robustness must cover the exact clean validation set")
    configuration = {
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "artifact_hashes": artifact_hashes,
        "frame_keys": frame_keys,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "runs": [
            {
                "variant": run["config"]["cycle_variant"],
                "seed": int(run["config"]["seed"]),
                "run_path": str(run["path"]),
                "config_sha256": run["config_sha256"],
                "best_checkpoint_sha256": run["checkpoint_sha256"],
            }
            for run in runs
        ],
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["configuration"] != configuration:
            raise ValueError("G3R robustness resume configuration differs")
    else:
        progress = {"configuration": configuration, "results": {}}
        atomic_json(progress_path, progress)

    axes = load_axes(args.data_root / "resources")
    device = torch.device(args.device)
    for run in sorted(
        runs,
        key=lambda value: (
            value["config"]["cycle_variant"],
            int(value["config"]["seed"]),
        ),
    ):
        variant = run["config"]["cycle_variant"]
        seed = int(run["config"]["seed"])
        run_key = f"{variant}:seed{seed}"
        results = progress["results"].setdefault(run_key, {})
        pending = [
            (index, condition)
            for index, condition in enumerate(CONDITIONS)
            if condition.condition_id not in results
        ]
        if not pending:
            continue
        parent_config = json.loads(
            run["parent_config_path"].read_text(encoding="utf-8")
        )["config"]
        maximum_log10_power = float(parent_config["log_center"]) + 4.0 * float(
            parent_config["log_scale"]
        )
        model = build_model(run, axes, device)
        for condition_index, condition in pending:
            print(
                json.dumps(
                    {
                        "event": "condition_start",
                        "run": run_key,
                        "condition": condition.condition_id,
                    }
                ),
                flush=True,
            )
            results[condition.condition_id] = evaluate_condition(
                model,
                dataset,
                frame_indices,
                axes,
                condition,
                condition_index,
                seed,
                maximum_log10_power,
                device,
            )
            atomic_json(progress_path, progress)
        del model
        torch.cuda.empty_cache()

    required_conditions = {condition.condition_id for condition in CONDITIONS}
    expected_run_count = len(VARIANTS) * args.required_seeds
    all_conditions = len(progress["results"]) == expected_run_count and all(
        set(results) == required_conditions
        for results in progress["results"].values()
    )
    all_frames = all(
        result["frame_count"] == len(frame_keys)
        for results in progress["results"].values()
        for result in results.values()
    )
    all_metrics = all(
        source in result and metric in result[source]
        for results in progress["results"].values()
        for result in results.values()
        for source, metric in REQUIRED_AGGREGATES
    )
    metadata = {
        f"{run['config']['cycle_variant']}:seed{int(run['config']['seed'])}": run
        for run in runs
    }
    report_runs = []
    for run_key, results in sorted(progress["results"].items()):
        run = metadata[run_key]
        report_runs.append(
            {
                "variant": run["config"]["cycle_variant"],
                "seed": int(run["config"]["seed"]),
                "run_path": str(run["path"]),
                "config_sha256": run["config_sha256"],
                "best_checkpoint_sha256": run["checkpoint_sha256"],
                "conditions": results,
                "degradation_curve": degradation_curve(results),
            }
        )
    checks = {
        "all_c0_c3_seed_runs_present": len(progress["results"])
        == expected_run_count,
        "all_preregistered_conditions_present": all_conditions,
        "all_frozen_validation_frames_evaluated": all_frames,
        "all_required_metrics_present": all_metrics,
    }
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "artifact_hashes": artifact_hashes,
        "required_variants": list(VARIANTS),
        "required_seeds": args.required_seeds,
        "condition_definitions": [asdict(condition) for condition in CONDITIONS],
        "full_validation_frame_count": len(frame_keys),
        "checks": checks,
        "runs": report_runs,
        "completed": all(checks.values()),
    }
    atomic_json(args.output, report)
    if report["completed"]:
        progress_path.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "checks": checks}), flush=True)
    if not report["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
