#!/usr/bin/env python3
"""Evaluate C0/C3 Cube-cycle robustness on the frozen validation set."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    occupancy_to_points,
)
from eval.doppler_distribution import (  # noqa: E402
    aggregate_doppler_reports,
    doppler_distribution_report,
)
from models.cube_cycle import CubeCycleNet  # noqa: E402
from models.cube_doppler import query_cube_spectrum, split_query_indices  # noqa: E402
from models.point_to_cube import soft_splat_raed  # noqa: E402
from scripts.train_cube_doppler import sha256  # noqa: E402


PROTOCOL = "g3_cube_cycle_robustness_v1"
VARIANT_BY_RUN_GROUP = {"none": "none", "full": "full"}
CONFIG_PAIR_EXCLUSIONS = {"variant", "seed"}
REQUIRED_AGGREGATES = (
    ("cycle", "local_spectrum_kl"),
    ("doppler", "static_pce_median_mps"),
    ("generated_geometry", "chamfer_m"),
    ("cycle", "covered_cell_count"),
)


@dataclass(frozen=True)
class Perturbation:
    condition_id: str
    family: str
    log_power_snr_db: float | None = None
    doppler_shift_bins: int = 0
    azimuth_offset_bins: float = 0.0
    elevation_offset_bins: float = 0.0
    confidence_temperature: float = 1.0


CONDITIONS = (
    Perturbation("clean", "clean"),
    Perturbation("log_power_noise_snr20db", "log_power_noise", log_power_snr_db=20.0),
    Perturbation("log_power_noise_snr10db", "log_power_noise", log_power_snr_db=10.0),
    Perturbation("log_power_noise_snr5db", "log_power_noise", log_power_snr_db=5.0),
    Perturbation("doppler_shift_m2", "doppler_shift", doppler_shift_bins=-2),
    Perturbation("doppler_shift_m1", "doppler_shift", doppler_shift_bins=-1),
    Perturbation("doppler_shift_p1", "doppler_shift", doppler_shift_bins=1),
    Perturbation("doppler_shift_p2", "doppler_shift", doppler_shift_bins=2),
    Perturbation("azimuth_offset_p0p25_bin", "calibration_offset", azimuth_offset_bins=0.25),
    Perturbation("azimuth_offset_p0p5_bin", "calibration_offset", azimuth_offset_bins=0.5),
    Perturbation("elevation_offset_p0p25_bin", "calibration_offset", elevation_offset_bins=0.25),
    Perturbation("elevation_offset_p0p5_bin", "calibration_offset", elevation_offset_bins=0.5),
    Perturbation("confidence_temperature_0p5", "confidence_temperature", confidence_temperature=0.5),
    Perturbation("confidence_temperature_1p0", "confidence_temperature", confidence_temperature=1.0),
    Perturbation("confidence_temperature_2p0", "confidence_temperature", confidence_temperature=2.0),
)


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fractional_nonperiodic_shift(
    values: torch.Tensor, dimension: int, offset_bins: float
) -> torch.Tensor:
    if not 0.0 <= offset_bins <= 1.0:
        raise ValueError("Calibration offsets must be in [0, 1] bins")
    if offset_bins == 0.0:
        return values
    shifted = torch.roll(values, shifts=1, dims=dimension)
    boundary = [slice(None)] * values.ndim
    boundary[dimension] = 0
    shifted[tuple(boundary)] = 0.0
    return values * (1.0 - offset_bins) + shifted * offset_bins


def perturb_cube(
    clean_cube: torch.Tensor,
    condition: Perturbation,
    random_seed: int,
    maximum_log10_power: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    cube = clean_cube
    diagnostics: dict[str, float] = {}
    if condition.log_power_snr_db is not None:
        log_power = torch.log10(clean_cube.clamp_min(0.0) + 1.0)
        signal_rms = log_power.square().mean().sqrt().clamp_min(1e-8)
        requested_noise_rms = signal_rms / (
            10.0 ** (condition.log_power_snr_db / 20.0)
        )
        generator = torch.Generator(device=clean_cube.device)
        generator.manual_seed(random_seed)
        noise = torch.randn(
            log_power.shape,
            dtype=log_power.dtype,
            device=log_power.device,
            generator=generator,
        ) * requested_noise_rms
        noisy_log_power = (log_power + noise).clamp(0.0, maximum_log10_power)
        cube = torch.pow(10.0, noisy_log_power) - 1.0
        actual_noise_rms = (noisy_log_power - log_power).square().mean().sqrt()
        diagnostics = {
            "signal_log_power_rms": float(signal_rms.item()),
            "requested_noise_log_power_rms": float(requested_noise_rms.item()),
            "actual_noise_log_power_rms": float(actual_noise_rms.item()),
            "actual_log_power_snr_db": float(
                (20.0 * torch.log10(signal_rms / actual_noise_rms.clamp_min(1e-8))).item()
            ),
        }
    if condition.doppler_shift_bins:
        cube = torch.roll(cube, shifts=condition.doppler_shift_bins, dims=1)
    cube = fractional_nonperiodic_shift(cube, 3, condition.azimuth_offset_bins)
    cube = fractional_nonperiodic_shift(cube, 4, condition.elevation_offset_bins)
    return cube, diagnostics


def static_center(
    model: CubeCycleNet,
    indices: torch.Tensor,
    ego_speed: torch.Tensor,
) -> torch.Tensor:
    batch, _, azimuth, elevation = split_query_indices(indices, 1)
    return model.static_center(batch, azimuth, elevation, ego_speed)


def frame_seed(base_seed: int, sequence: int, radar_index: int, condition_index: int) -> int:
    return int(
        (
            base_seed * 1_000_003
            + sequence * 10_007
            + radar_index * 101
            + condition_index
        )
        % (2**63 - 1)
    )


@torch.inference_mode()
def evaluate_condition(
    model: CubeCycleNet,
    dataset: KRadarCubeDataset,
    frame_indices: list[int],
    axes,
    condition: Perturbation,
    condition_index: int,
    model_seed: int,
    point_count: int,
    device: torch.device,
) -> dict:
    model.eval()
    geometry_reports = []
    doppler_reports = []
    cycle_reports = []
    frames = []
    maximum_log10_power = model.log_center + 4.0 * model.log_scale
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
        ego_speed = item["ego_speed_mps"].reshape(1).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occupancy_logits, features = model(model_cube)
        calibrated_logits = occupancy_logits[0].float() / condition.confidence_temperature
        _, confidence, discrete_indices = occupancy_to_points(
            calibrated_logits, axes, point_count=point_count
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.query_cycle(features, discrete_indices, ego_speed)

        clean_target_spectrum = query_cube_spectrum(clean_cube, discrete_indices)
        center = static_center(model, discrete_indices, ego_speed)
        doppler = doppler_distribution_report(
            prediction["probability"].float(),
            clean_target_spectrum.float(),
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
        cycle = cube_cycle_report(rendered, clean_cube[0].float(), confidence.float())
        target = item["target_xyz_confidence"].to(device)
        geometry = geometry_report(
            prediction["xyz_m"].float(),
            target[:, :3],
            target_weight=target[:, 3],
        )
        geometry_reports.append(geometry)
        doppler_reports.append(doppler)
        cycle_reports.append(cycle)
        frames.append(
            {
                "sequence": sequence,
                "radar_index": radar_index,
                "generated_geometry": geometry,
                "doppler": doppler,
                "cycle": cycle,
                "input_diagnostics": input_diagnostics,
            }
        )
        del item, clean_cube, model_cube, ego_speed, occupancy_logits, features
        del calibrated_logits, confidence, discrete_indices, prediction
        del clean_target_spectrum, center, rendered, target
        torch.cuda.empty_cache()
    return {
        "frame_count": len(frames),
        "generated_geometry": aggregate_geometry_reports(geometry_reports),
        "doppler": aggregate_doppler_reports(doppler_reports),
        "cycle": aggregate_cycle_reports(cycle_reports),
        "frames": frames,
    }


def load_run_document(path: Path, expected_variant: str) -> dict:
    path = path.resolve()
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if document["config"]["variant"] != expected_variant:
        raise ValueError(
            f"Expected {expected_variant}, found {document['config']['variant']} in {path}"
        )
    checkpoint = path / "best.pt"
    metrics = path / "best_validation_metrics.json"
    if not checkpoint.is_file() or not metrics.is_file():
        raise FileNotFoundError(f"Incomplete Cube-cycle run: {path}")
    validation = json.loads(metrics.read_text(encoding="utf-8"))["validation"]
    frame_keys = sorted(
        (int(frame["sequence"]), int(frame["radar_index"]))
        for frame in validation["frames"]
    )
    return {
        "path": path.resolve(),
        "config": document["config"],
        "provenance": document["provenance"],
        "checkpoint": checkpoint.resolve(),
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(path / "config.json"),
        "clean_frame_keys": frame_keys,
    }


def validate_run_matrix(runs: list[dict], required_seeds: int) -> None:
    grouped: dict[str, dict[int, dict]] = {variant: {} for variant in VARIANT_BY_RUN_GROUP}
    for run in runs:
        variant = run["config"]["variant"]
        seed = int(run["config"]["seed"])
        if seed in grouped[variant]:
            raise ValueError(f"Duplicate {variant} run for seed {seed}")
        grouped[variant][seed] = run
    seed_sets = [set(grouped[variant]) for variant in VARIANT_BY_RUN_GROUP]
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("C0/C3 robustness seed sets differ")
    if len(seed_sets[0]) != required_seeds:
        raise ValueError(f"Robustness matrix requires {required_seeds} seeds")
    reference_global = None
    reference_config = None
    reference_frames = None
    for seed in sorted(seed_sets[0]):
        parent_hash = None
        for variant in VARIANT_BY_RUN_GROUP:
            run = grouped[variant][seed]
            paired_config = {
                key: value
                for key, value in run["config"].items()
                if key not in CONFIG_PAIR_EXCLUSIONS
            }
            global_provenance = tuple(
                run["provenance"][key]
                for key in (
                    "manifest_sha256",
                    "scene_split_sha256",
                    "normalization_sha256",
                )
            )
            if reference_global is None:
                reference_global = global_provenance
                reference_config = paired_config
                reference_frames = run["clean_frame_keys"]
            if global_provenance != reference_global:
                raise ValueError("C0/C3 robustness data provenance differs")
            if paired_config != reference_config:
                raise ValueError("C0/C3 robustness configurations are not matched")
            if run["clean_frame_keys"] != reference_frames:
                raise ValueError("C0/C3 clean validation frame sets differ")
            current_parent = run["provenance"]["parent_g2_checkpoint_sha256"]
            if parent_hash is None:
                parent_hash = current_parent
            if current_parent != parent_hash:
                raise ValueError(f"C0/C3 seed {seed} use different G2 parents")


def build_model(run: dict, axes, device: torch.device) -> CubeCycleNet:
    config = run["config"]
    model = CubeCycleNet(
        config["parent_head_mode"],
        torch.from_numpy(axes.doppler_mps),
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        base_channels=int(config["base_channels"]),
        log_center=float(config["log_center"]),
        log_scale=float(config["log_scale"]),
        static_hypothesis=config["static_hypothesis"],
        maximum_offset_bins=float(config["maximum_offset_bins"]),
    ).to(device)
    checkpoint = torch.load(run["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def endpoint_values(metrics: dict) -> dict[str, float]:
    return {
        "local_spectrum_kl": float(metrics["cycle"]["local_spectrum_kl"]["mean"]),
        "static_pce_median_mps": float(
            metrics["doppler"]["static_pce_median_mps"]["mean"]
        ),
        "geometry_chamfer_m": float(
            metrics["generated_geometry"]["chamfer_m"]["mean"]
        ),
        "covered_cell_count": float(metrics["cycle"]["covered_cell_count"]["mean"]),
    }


def degradation_curve(condition_results: dict[str, dict]) -> dict[str, dict]:
    clean = endpoint_values(condition_results["clean"])
    curves = {}
    for condition in CONDITIONS:
        current = endpoint_values(condition_results[condition.condition_id])
        curves[condition.condition_id] = {
            "local_spectrum_kl_increase": current["local_spectrum_kl"]
            - clean["local_spectrum_kl"],
            "static_pce_increase_mps": current["static_pce_median_mps"]
            - clean["static_pce_median_mps"],
            "geometry_chamfer_increase_m": current["geometry_chamfer_m"]
            - clean["geometry_chamfer_m"],
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
    parser.add_argument("--max-eval-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("G3 robustness evaluation requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    if args.overwrite:
        for path in (args.output, progress_path):
            if path.exists():
                path.unlink()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.output.exists() and args.resume and not progress_path.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("protocol") == PROTOCOL and existing.get("completed") is True:
            print(json.dumps({"output": str(args.output), "already_complete": True}))
            return
        raise ValueError("Existing robustness output is not a completed compatible report")
    if progress_path.exists() and not args.resume:
        raise FileExistsError(f"Progress already exists: {progress_path}")

    runs = [load_run_document(path, "none") for path in args.none_runs]
    runs.extend(load_run_document(path, "full") for path in args.full_runs)
    validate_run_matrix(runs, args.required_seeds)
    artifact_hashes = {
        "manifest_sha256": sha256(args.manifest),
        "scene_split_sha256": sha256(args.scene_split),
        "normalization_sha256": sha256(args.normalization_stats),
    }
    for run in runs:
        for name, value in artifact_hashes.items():
            if run["provenance"][name] != value:
                raise ValueError(f"{name} differs from Cube-cycle run {run['path']}")

    dataset = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    frame_indices = list(range(len(dataset)))
    if args.max_eval_frames is not None:
        if args.max_eval_frames <= 0:
            raise ValueError("--max-eval-frames must be positive")
        positions = np.linspace(
            0, len(frame_indices) - 1, min(args.max_eval_frames, len(frame_indices))
        ).round().astype(int)
        frame_indices = [frame_indices[position] for position in sorted(set(positions))]
    selected_frame_keys = sorted(
        (int(dataset[index]["sequence"]), int(dataset[index]["radar_index"]))
        for index in frame_indices
    )
    expected_clean_frames = runs[0]["clean_frame_keys"]
    full_validation = selected_frame_keys == expected_clean_frames
    if args.max_eval_frames is None and not full_validation:
        raise ValueError("Dataset validation frames differ from trained-run evaluation frames")

    configuration = {
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "device": args.device,
        "required_seeds": args.required_seeds,
        "point_count": int(runs[0]["config"]["point_count"]),
        "full_validation_frame_count": len(expected_clean_frames),
        "selected_frame_keys": selected_frame_keys,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "artifact_hashes": artifact_hashes,
        "runs": [
            {
                "variant": run["config"]["variant"],
                "seed": int(run["config"]["seed"]),
                "run_path": str(run["path"]),
                "config_sha256": run["config_sha256"],
                "best_checkpoint_sha256": run["checkpoint_sha256"],
                "model_source_commit": run["provenance"]["git_commit"],
            }
            for run in runs
        ],
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["configuration"] != configuration:
            raise ValueError("Robustness resume configuration differs")
    else:
        progress = {"configuration": configuration, "results": {}}
        atomic_json(progress_path, progress)

    axes = load_axes(args.data_root / "resources")
    device = torch.device(args.device)
    point_count = int(runs[0]["config"]["point_count"])
    for run in sorted(runs, key=lambda value: (value["config"]["variant"], value["config"]["seed"])):
        run_key = f"{run['config']['variant']}:seed{int(run['config']['seed'])}"
        condition_results = progress["results"].setdefault(run_key, {})
        pending = [
            (index, condition)
            for index, condition in enumerate(CONDITIONS)
            if condition.condition_id not in condition_results
        ]
        if not pending:
            continue
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
            condition_results[condition.condition_id] = evaluate_condition(
                model,
                dataset,
                frame_indices,
                axes,
                condition,
                condition_index,
                int(run["config"]["seed"]),
                point_count,
                device,
            )
            atomic_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "event": "condition_complete",
                        "run": run_key,
                        "condition": condition.condition_id,
                    }
                ),
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    required_ids = {condition.condition_id for condition in CONDITIONS}
    expected_run_count = args.required_seeds * len(VARIANT_BY_RUN_GROUP)
    all_conditions = len(progress["results"]) == expected_run_count and all(
        set(results) == required_ids for results in progress["results"].values()
    )
    all_frames = all(
        result["frame_count"] == len(expected_clean_frames)
        for results in progress["results"].values()
        for result in results.values()
    )
    all_metrics = all(
        source in result and key in result[source]
        for results in progress["results"].values()
        for result in results.values()
        for source, key in REQUIRED_AGGREGATES
    )
    report_runs = []
    run_metadata = {
        f"{run['config']['variant']}:seed{int(run['config']['seed'])}": run
        for run in runs
    }
    for run_key, results in sorted(progress["results"].items()):
        run = run_metadata[run_key]
        report_runs.append(
            {
                "variant": run["config"]["variant"],
                "seed": int(run["config"]["seed"]),
                "run_path": str(run["path"]),
                "config_sha256": run["config_sha256"],
                "best_checkpoint_sha256": run["checkpoint_sha256"],
                "model_source_commit": run["provenance"]["git_commit"],
                "conditions": results,
                "degradation_curve": degradation_curve(results),
            }
        )
    checks = {
        "all_c0_c3_seed_runs_present": len(progress["results"]) == expected_run_count,
        "all_preregistered_conditions_present": all_conditions,
        "all_frozen_validation_frames_evaluated": full_validation and all_frames,
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
        "required_variants": list(VARIANT_BY_RUN_GROUP),
        "required_seeds": args.required_seeds,
        "condition_definitions": [asdict(condition) for condition in CONDITIONS],
        "full_validation_frame_count": len(expected_clean_frames),
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
