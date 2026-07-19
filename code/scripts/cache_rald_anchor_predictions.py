#!/usr/bin/env python3
"""Cache frozen, gate-selected RaLD-anchor predictions for G4R."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from models.cube_occupancy import CubeOccupancyNet  # noqa: E402
from models.rald_anchor import FrozenParentRaLDRefiner  # noqa: E402
from models.rald_matched import FullRAEDRadarTokenEncoder  # noqa: E402
from rald_gate_contract import validate_g3r_selected_runs  # noqa: E402


PROTOCOL = "g4r_frozen_rald_anchor_predictions_v1"
MANIFEST_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 1
DOPPLER_BIN_COUNT = 64
PREDICTION_METADATA_KEYS = (
    "temporal_manifest_sha256",
    "scene_split_sha256",
    "normalization_sha256",
    "g3r_summary_sha256",
    "g3r_config_sha256",
    "g3r_checkpoint_sha256",
    "g3r_source_commit",
    "cache_source_commit",
)
PREDICTION_ARRAY_KEYS = (
    "xyz_m",
    "coordinates_rae",
    "doppler_probability",
    "confidence",
)
PREDICTION_KEYS = {
    "prediction_schema_version",
    *PREDICTION_METADATA_KEYS,
    "sequence",
    "radar_index",
    *PREDICTION_ARRAY_KEYS,
}
G3R_MODEL_CONFIG_KEYS = (
    "point_count",
    "latent_count",
    "model_dim",
    "depth",
    "heads",
    "head_dim",
    "radar_base_channels",
    "radar_spectral_channels",
)
PARENT_MODEL_CONFIG_KEYS = (
    "mode",
    "base_channels",
    "log_center",
    "log_scale",
)


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


def prediction_path(root: Path, sequence: int, radar_index: int) -> Path:
    return root / "predictions" / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"


def _require_keys(document: dict, keys: tuple[str, ...], description: str) -> None:
    missing = [key for key in keys if key not in document]
    if missing:
        raise ValueError(f"{description} lacks required keys: {missing}")


def _frame_keys(frames: list[dict]) -> list[tuple[int, int]]:
    keys = [
        (int(frame["sequence"]), int(frame["radar_index"])) for frame in frames
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Temporal manifest contains duplicate frame identities")
    return keys


def validate_cache_inputs(
    temporal_manifest_path: Path,
    scene_split_path: Path,
    normalization_path: Path,
    g3r_summary_path: Path,
    *,
    g3r_source_commit: str,
    cache_source_commit: str,
    seed: int,
) -> dict:
    """Validate the full read-only artifact chain without loading a checkpoint."""

    for path in (
        temporal_manifest_path,
        scene_split_path,
        normalization_path,
        g3r_summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    temporal_manifest_path = temporal_manifest_path.resolve()
    scene_split_path = scene_split_path.resolve()
    normalization_path = normalization_path.resolve()
    g3r_summary_path = g3r_summary_path.resolve()
    temporal_manifest = json.loads(
        temporal_manifest_path.read_text(encoding="utf-8")
    )
    if temporal_manifest.get("gate_pass") is not True:
        raise ValueError("Temporal manifest did not pass its gate")
    frames = temporal_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Temporal manifest has no cacheable frames")
    frame_keys = _frame_keys(frames)
    if any("partition" not in frame for frame in frames):
        raise ValueError("Temporal manifest frame lacks a scene partition")

    scene_split = json.loads(scene_split_path.read_text(encoding="utf-8"))
    if scene_split.get("gate_pass") is not True:
        raise ValueError("Scene split did not pass its leakage gate")
    # Parsing the normalization document makes a malformed artifact fail before CUDA.
    json.loads(normalization_path.read_text(encoding="utf-8"))

    temporal_manifest_hash = sha256(temporal_manifest_path)
    scene_split_hash = sha256(scene_split_path)
    normalization_hash = sha256(normalization_path)
    if temporal_manifest.get("source_split_sha256") != scene_split_hash:
        raise ValueError("Temporal manifest and scene split hashes differ")

    g3r_summary = json.loads(g3r_summary_path.read_text(encoding="utf-8"))
    selected_runs = validate_g3r_selected_runs(
        g3r_summary, g3r_source_commit
    )
    if seed not in selected_runs:
        raise ValueError(f"Seed {seed} is not in the gate-selected G3R matrix")
    run = selected_runs[seed]
    config_path = (run / "config.json").resolve()
    checkpoint_path = (run / "best.pt").resolve()
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    config = config_document["config"]
    provenance = config_document["provenance"]
    _require_keys(config, G3R_MODEL_CONFIG_KEYS, "Selected G3R config")
    if config.get("cycle_variant") != "full":
        raise ValueError("G4R cache requires the selected full-cycle G3R arm")
    if config.get("doppler_head_mode") != "distribution":
        raise ValueError("G4R cache requires the selected distribution G3R head")
    if int(config["seed"]) != seed:
        raise ValueError("Selected G3R seed differs")
    if int(config["point_count"]) <= 0:
        raise ValueError("Selected G3R point count must be positive")
    if provenance.get("git_commit") != g3r_source_commit:
        raise ValueError("Selected G3R source commit differs")
    if provenance.get("scene_split_sha256") != scene_split_hash:
        raise ValueError("Selected G3R and temporal scene splits differ")
    if provenance.get("normalization_sha256") != normalization_hash:
        raise ValueError("Selected G3R and temporal normalization differ")
    if not provenance.get("manifest_sha256"):
        raise ValueError("Selected G3R run lacks its training manifest hash")

    selected_hashes = g3r_summary["selected_run_hashes"][str(seed)]
    config_hash = sha256(config_path)
    checkpoint_hash = sha256(checkpoint_path)
    if config_hash != selected_hashes["config_sha256"]:
        raise ValueError("Selected G3R config hash differs")
    if checkpoint_hash != selected_hashes["best_checkpoint_sha256"]:
        raise ValueError("Selected G3R checkpoint hash differs")

    parent_checkpoint_path = Path(provenance["parent_g1_checkpoint"]).resolve()
    if not parent_checkpoint_path.is_file():
        raise FileNotFoundError(parent_checkpoint_path)
    parent_checkpoint_hash = sha256(parent_checkpoint_path)
    if parent_checkpoint_hash != provenance.get("parent_g1_checkpoint_sha256"):
        raise ValueError("Frozen geometry parent checkpoint hash differs")
    parent_config_path = (parent_checkpoint_path.parent / "config.json").resolve()
    if not parent_config_path.is_file():
        raise FileNotFoundError(parent_config_path)
    parent_document = json.loads(parent_config_path.read_text(encoding="utf-8"))
    parent_config = parent_document["config"]
    parent_provenance = parent_document["provenance"]
    _require_keys(parent_config, PARENT_MODEL_CONFIG_KEYS, "Geometry parent config")
    if parent_provenance.get("git_commit") != provenance.get(
        "parent_g1_git_commit"
    ):
        raise ValueError("Frozen geometry parent source commit differs")
    if parent_provenance.get("scene_split_sha256") != scene_split_hash:
        raise ValueError("Frozen geometry parent and temporal scene splits differ")
    if parent_provenance.get("normalization_sha256") != normalization_hash:
        raise ValueError("Frozen geometry parent and temporal normalization differ")

    comparison_path = Path(g3r_summary["g3r_comparison"]).resolve()
    configuration = {
        "protocol": PROTOCOL,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "source_commit": cache_source_commit,
        "cache_source_commit": cache_source_commit,
        "temporal_manifest": str(temporal_manifest_path),
        "temporal_manifest_sha256": temporal_manifest_hash,
        "temporal_manifest_source_commit": temporal_manifest.get("source_commit"),
        "scene_split": str(scene_split_path),
        "scene_split_sha256": scene_split_hash,
        "normalization_stats": str(normalization_path),
        "normalization_sha256": normalization_hash,
        "g3r_summary": str(g3r_summary_path),
        "g3r_summary_sha256": sha256(g3r_summary_path),
        "g3r_comparison": str(comparison_path),
        "g3r_comparison_sha256": sha256(comparison_path),
        "g3r_source_commit": g3r_source_commit,
        "g3r_seed": seed,
        "g3r_run": str(run),
        "g3r_config": str(config_path),
        "g3r_config_sha256": config_hash,
        "g3r_checkpoint": str(checkpoint_path),
        "g3r_checkpoint_sha256": checkpoint_hash,
        "g3r_training_manifest_sha256": provenance["manifest_sha256"],
        "parent_config": str(parent_config_path),
        "parent_config_sha256": sha256(parent_config_path),
        "parent_checkpoint": str(parent_checkpoint_path),
        "parent_checkpoint_sha256": parent_checkpoint_hash,
        "parent_source_commit": parent_provenance["git_commit"],
        "selected_arm": "full",
        "doppler_head_mode": "distribution",
        "point_count": int(config["point_count"]),
        "doppler_bin_count": DOPPLER_BIN_COUNT,
        "partitions": sorted({str(frame["partition"]) for frame in frames}),
        "required_frames": len(frames),
    }
    return {
        "configuration": configuration,
        "temporal_manifest": temporal_manifest,
        "frame_keys": frame_keys,
        "config": config,
        "provenance": provenance,
        "parent_config": parent_config,
        "parent_provenance": parent_provenance,
    }


def prediction_metadata(configuration: dict) -> dict[str, str]:
    return {key: str(configuration[key]) for key in PREDICTION_METADATA_KEYS}


def write_prediction(
    path: Path,
    output: dict[str, torch.Tensor],
    metadata: dict[str, str],
    *,
    sequence: int,
    radar_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "prediction_schema_version": np.asarray(
            PREDICTION_SCHEMA_VERSION, dtype=np.int16
        ),
        **{key: np.asarray(metadata[key]) for key in PREDICTION_METADATA_KEYS},
        "sequence": np.asarray(sequence, dtype=np.int16),
        "radar_index": np.asarray(radar_index, dtype=np.int32),
        "xyz_m": output["xyz_m"].detach().cpu().numpy().astype(np.float16),
        "coordinates_rae": output["coordinates_rae"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        "doppler_probability": output["doppler_probability"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        "confidence": output["confidence"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16),
    }
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def validate_prediction(
    path: Path,
    metadata: dict[str, str],
    point_count: int,
    *,
    sequence: int | None = None,
    radar_index: int | None = None,
) -> dict:
    with np.load(path, allow_pickle=False) as cache:
        keys = set(cache.files)
        if keys != PREDICTION_KEYS:
            missing = sorted(PREDICTION_KEYS - keys)
            unexpected = sorted(keys - PREDICTION_KEYS)
            raise ValueError(
                f"Prediction schema keys differ in {path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if int(cache["prediction_schema_version"]) != PREDICTION_SCHEMA_VERSION:
            raise ValueError(f"Prediction schema version differs in {path}")
        for key in PREDICTION_METADATA_KEYS:
            if str(np.asarray(cache[key]).item()) != metadata[key]:
                raise ValueError(f"Prediction metadata {key} differs in {path}")
        cached_sequence = int(cache["sequence"])
        cached_radar_index = int(cache["radar_index"])
        if sequence is not None and cached_sequence != sequence:
            raise ValueError(f"Prediction sequence differs in {path}")
        if radar_index is not None and cached_radar_index != radar_index:
            raise ValueError(f"Prediction radar index differs in {path}")
        expected_shapes = {
            "xyz_m": (point_count, 3),
            "coordinates_rae": (point_count, 3),
            "doppler_probability": (point_count, DOPPLER_BIN_COUNT),
            "confidence": (point_count,),
        }
        for key, shape in expected_shapes.items():
            if cache[key].shape != shape:
                raise ValueError(
                    f"Prediction shape {key}={cache[key].shape} in {path}; "
                    f"expected {shape}"
                )
            if not np.isfinite(cache[key]).all():
                raise ValueError(f"Non-finite prediction values in {path}:{key}")
        probability = cache["doppler_probability"].astype(np.float32)
        if probability.min() < -1e-6 or probability.max() > 1.0 + 1e-3:
            raise ValueError(f"Doppler probabilities outside [0,1] in {path}")
        probability_sum_error = float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        )
        if probability_sum_error > 2e-3:
            raise ValueError(
                f"Float16 Doppler probabilities do not sum to one in {path}: "
                f"{probability_sum_error}"
            )
        confidence = cache["confidence"].astype(np.float32)
        if confidence.min() < 0.0 or confidence.max() > 1.0:
            raise ValueError(f"Prediction confidence outside [0,1] in {path}")
        return {
            "sequence": cached_sequence,
            "radar_index": cached_radar_index,
            "point_count": point_count,
            "probability_sum_max_error": probability_sum_error,
            "confidence_mean": float(confidence.mean()),
        }


def validate_frame_record(
    frame: dict,
    metadata: dict[str, str],
    point_count: int,
) -> dict:
    path = Path(frame["prediction"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != frame.get("prediction_sha256"):
        raise ValueError(f"Prediction file hash differs: {path}")
    return validate_prediction(
        path,
        metadata,
        point_count,
        sequence=int(frame["sequence"]),
        radar_index=int(frame["radar_index"]),
    )


def require_h200_cuda(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("RaLD-anchor caching requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("RaLD-anchor caching is CUDA-only")
    resolved_name = torch.cuda.get_device_name(device)
    if "H200" not in resolved_name.upper():
        raise RuntimeError(f"RaLD-anchor caching requires H200, got {resolved_name}")
    return device, resolved_name


def build_model(contract: dict, axes, device: torch.device) -> FrozenParentRaLDRefiner:
    config = contract["config"]
    provenance = contract["provenance"]
    parent_config = contract["parent_config"]
    configuration = contract["configuration"]
    parent = CubeOccupancyNet(
        parent_config["mode"],
        torch.from_numpy(axes.doppler_mps),
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
    ).to(device)
    parent_checkpoint = torch.load(
        configuration["parent_checkpoint"],
        map_location=device,
        weights_only=False,
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
        doppler_head_mode="distribution",
    ).to(device)
    checkpoint = torch.load(
        configuration["g3r_checkpoint"],
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("config") != config or checkpoint.get("provenance") != provenance:
        raise ValueError("G3R checkpoint metadata differs from its config document")
    model.refiner.load_state_dict(checkpoint["refiner"], strict=True)
    if checkpoint.get("radar_encoder") is None:
        raise ValueError("Selected G3R checkpoint lacks Full-RAED radar encoder state")
    model.radar_encoder.load_state_dict(checkpoint["radar_encoder"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def _prepare_output(output: Path, *, resume: bool, overwrite: bool) -> Path:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if overwrite and output.exists():
        shutil.rmtree(output)
    manifest_path = output / "manifest.json"
    nonempty = output.exists() and any(output.iterdir())
    if resume and not manifest_path.is_file():
        raise FileNotFoundError(
            f"No RaLD-anchor cache manifest to resume: {manifest_path}"
        )
    if nonempty and not resume:
        raise FileExistsError(f"RaLD-anchor prediction cache is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--temporal-manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--g3r-summary", type=Path, required=True)
    parser.add_argument("--g3r-source-commit", required=True)
    parser.add_argument("--cache-source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device, resolved_device_name = require_h200_cuda(args.device)
    manifest_path = _prepare_output(
        args.output, resume=args.resume, overwrite=args.overwrite
    )
    contract = validate_cache_inputs(
        args.temporal_manifest,
        args.scene_split,
        args.normalization_stats,
        args.g3r_summary,
        g3r_source_commit=args.g3r_source_commit,
        cache_source_commit=args.cache_source_commit,
        seed=args.seed,
    )
    configuration = contract["configuration"]
    metadata = prediction_metadata(configuration)
    point_count = int(configuration["point_count"])
    if manifest_path.is_file():
        progress = json.loads(manifest_path.read_text(encoding="utf-8"))
        if progress.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("RaLD-anchor cache manifest schema differs")
        if progress.get("configuration") != configuration:
            raise ValueError("RaLD-anchor cache resume configuration differs")
    else:
        progress = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "configuration": configuration,
            "frames": [],
            "completed": False,
        }
        atomic_json(manifest_path, progress)

    completed: dict[tuple[int, int], dict] = {}
    for frame in progress["frames"]:
        key = (int(frame["sequence"]), int(frame["radar_index"]))
        if key in completed:
            raise ValueError("RaLD-anchor cache manifest contains duplicate frames")
        completed[key] = frame
    expected_keys = set(contract["frame_keys"])
    if not set(completed).issubset(expected_keys):
        raise ValueError("RaLD-anchor cache manifest contains unexpected frames")

    axes = load_axes(args.data_root / "resources")
    model = build_model(contract, axes, device)
    partitions = tuple(configuration["partitions"])
    dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.temporal_manifest,
        partitions,
    )
    dataset_keys = [
        (int(record["sequence"]), int(record["radar_index"]))
        for record in dataset.records
    ]
    if dataset_keys != contract["frame_keys"]:
        raise ValueError(
            "Temporal dataset order or frame identity differs from manifest"
        )

    with torch.inference_mode():
        for dataset_index, record in enumerate(dataset.records):
            sequence = int(record["sequence"])
            radar_index = int(record["radar_index"])
            key = (sequence, radar_index)
            path = prediction_path(args.output, sequence, radar_index).resolve()
            prior = completed.get(key)
            if prior is not None and Path(prior["prediction"]).resolve() == path:
                try:
                    validation = validate_frame_record(prior, metadata, point_count)
                except (FileNotFoundError, KeyError, ValueError):
                    pass
                else:
                    print(json.dumps({**validation, "status": "cached"}), flush=True)
                    continue

            started = time.monotonic()
            item = dataset[dataset_index]
            cube = item["cube_drae"].unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model_output = model(cube)
            output = {
                key: model_output[key][0].float()
                for key in PREDICTION_ARRAY_KEYS
            }
            write_prediction(
                path,
                output,
                metadata,
                sequence=sequence,
                radar_index=radar_index,
            )
            validation = validate_prediction(
                path,
                metadata,
                point_count,
                sequence=sequence,
                radar_index=radar_index,
            )
            frame_report = {
                **validation,
                "partition": record["partition"],
                "prediction": str(path),
                "prediction_size": path.stat().st_size,
                "prediction_sha256": sha256(path),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            completed[key] = frame_report
            progress["frames"] = sorted(
                completed.values(),
                key=lambda value: (value["sequence"], value["radar_index"]),
            )
            progress["completed"] = False
            atomic_json(manifest_path, progress)
            print(json.dumps(frame_report), flush=True)
            del item, cube, model_output, output
            torch.cuda.empty_cache()

    frame_reports = progress["frames"]
    validations = [
        validate_frame_record(frame, metadata, point_count)
        for frame in frame_reports
    ]
    checks = {
        "required_frame_count": len(frame_reports) == configuration["required_frames"],
        "exact_temporal_frame_set": {
            (int(frame["sequence"]), int(frame["radar_index"]))
            for frame in frame_reports
        }
        == expected_keys,
        "all_prediction_hashes_valid": len(validations) == len(frame_reports),
        "all_point_counts_match": all(
            validation["point_count"] == point_count for validation in validations
        ),
        "all_probability_errors_within_tolerance": all(
            validation["probability_sum_max_error"] <= 2e-3
            for validation in validations
        ),
    }
    progress.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "device_name": resolved_device_name,
            "torch_version": torch.__version__,
            "checks": checks,
            "completed": all(checks.values()),
        }
    )
    atomic_json(manifest_path, progress)
    print(json.dumps({"checks": checks, "completed": progress["completed"]}, indent=2))
    if not progress["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
