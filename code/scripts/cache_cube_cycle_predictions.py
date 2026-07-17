#!/usr/bin/env python3
"""Cache frozen C0/C3 predictions for recurrent temporal training."""

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
from eval.dense_geometry import occupancy_to_points  # noqa: E402
from models.cube_cycle import CubeCycleNet  # noqa: E402
from models.cube_doppler import split_query_indices  # noqa: E402


PREDICTION_SCHEMA_VERSION = 1


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


def write_prediction(
    path: Path,
    prediction: dict[str, torch.Tensor],
    confidence: torch.Tensor,
    discrete_indices: torch.Tensor,
    static_center_mps: torch.Tensor,
    metadata: dict,
    sequence: int,
    radar_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            prediction_schema_version=np.asarray(
                PREDICTION_SCHEMA_VERSION, dtype=np.int16
            ),
            source_manifest_sha256=np.asarray(metadata["source_manifest_sha256"]),
            parent_checkpoint_sha256=np.asarray(
                metadata["parent_checkpoint_sha256"]
            ),
            source_commit=np.asarray(metadata["source_commit"]),
            sequence=np.asarray(sequence, dtype=np.int16),
            radar_index=np.asarray(radar_index, dtype=np.int32),
            xyz_m=prediction["xyz_m"].detach().cpu().numpy().astype(np.float16),
            coordinates_rae=(
                prediction["coordinates_rae"].detach().cpu().numpy().astype(np.float16)
            ),
            doppler_probability=(
                prediction["probability"].detach().cpu().numpy().astype(np.float16)
            ),
            scalar_mps=(
                prediction["scalar_mps"].detach().cpu().numpy().astype(np.float16)
            ),
            static_center_mps=(
                static_center_mps.detach().cpu().numpy().astype(np.float16)
            ),
            confidence=confidence.detach().cpu().numpy().astype(np.float16),
            discrete_rae_index=(
                discrete_indices.detach().cpu().numpy().astype(np.int16)
            ),
        )
    temporary.replace(path)


def validate_prediction(path: Path, metadata: dict, point_count: int) -> dict:
    with np.load(path) as cache:
        required = {
            "prediction_schema_version",
            "source_manifest_sha256",
            "parent_checkpoint_sha256",
            "source_commit",
            "sequence",
            "radar_index",
            "xyz_m",
            "coordinates_rae",
            "doppler_probability",
            "scalar_mps",
            "static_center_mps",
            "confidence",
            "discrete_rae_index",
        }
        if not required.issubset(cache.files):
            raise ValueError(f"Prediction keys missing in {path}")
        if int(cache["prediction_schema_version"]) != PREDICTION_SCHEMA_VERSION:
            raise ValueError(f"Prediction schema differs in {path}")
        for key in (
            "source_manifest_sha256",
            "parent_checkpoint_sha256",
            "source_commit",
        ):
            if str(cache[key]) != metadata[key]:
                raise ValueError(f"Prediction metadata {key} differs in {path}")
        expected_shapes = {
            "xyz_m": (point_count, 3),
            "coordinates_rae": (point_count, 3),
            "doppler_probability": (point_count, 64),
            "scalar_mps": (point_count,),
            "static_center_mps": (point_count,),
            "confidence": (point_count,),
            "discrete_rae_index": (point_count, 3),
        }
        for key, shape in expected_shapes.items():
            if cache[key].shape != shape:
                raise ValueError(f"Prediction shape {key}={cache[key].shape} in {path}")
            if not np.isfinite(cache[key]).all():
                raise ValueError(f"Non-finite prediction values in {path}:{key}")
        probability_sum_error = float(
            np.max(
                np.abs(
                    cache["doppler_probability"].astype(np.float32).sum(axis=1)
                    - 1.0
                )
            )
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
            "sequence": int(cache["sequence"]),
            "radar_index": int(cache["radar_index"]),
            "point_count": point_count,
            "probability_sum_max_error": probability_sum_error,
            "confidence_mean": float(confidence.mean()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--dense-cache-report", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--required-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Parent prediction caching requires CUDA")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.output.exists() and any(args.output.iterdir()) and not (
        args.overwrite or args.resume
    ):
        raise FileExistsError(f"Prediction cache is not empty: {args.output}")
    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)

    temporal_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if temporal_manifest.get("gate_pass") is not True:
        raise ValueError("Temporal manifest did not pass its gate")
    dense_cache_report = json.loads(
        args.dense_cache_report.read_text(encoding="utf-8")
    )
    if dense_cache_report.get("completed") is not True:
        raise ValueError("Parent predictions require a complete dense cache")
    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    normalization_hash = sha256(args.normalization_stats)
    if (
        dense_cache_report["configuration"]["source_manifest_sha256"]
        != manifest_hash
    ):
        raise ValueError("Dense cache and temporal manifest differ")
    if temporal_manifest.get("source_split_sha256") != scene_split_hash:
        raise ValueError("Temporal manifest and scene split differ")

    parent_document = json.loads(
        (args.parent_run / "config.json").read_text(encoding="utf-8")
    )
    parent_config = parent_document["config"]
    parent_provenance = parent_document["provenance"]
    if parent_config.get("variant") not in ("none", "full"):
        raise ValueError("G4 parent must be the matched C0 or C3 arm")
    if parent_provenance["scene_split_sha256"] != scene_split_hash:
        raise ValueError("Parent and temporal scene splits differ")
    if parent_provenance["normalization_sha256"] != normalization_hash:
        raise ValueError("Parent and temporal normalization differ")
    parent_checkpoint = (args.parent_run / "best.pt").resolve()
    parent_hash = sha256(parent_checkpoint)
    required_frames = args.required_frames or len(temporal_manifest["frames"])
    if required_frames != len(temporal_manifest["frames"]):
        raise ValueError("Formal parent cache must cover every temporal frame")
    point_count = int(parent_config["point_count"])
    metadata = {
        "source_manifest_sha256": manifest_hash,
        "parent_checkpoint_sha256": parent_hash,
        "source_commit": args.source_commit,
    }
    configuration = {
        **metadata,
        "scene_split_sha256": scene_split_hash,
        "normalization_sha256": normalization_hash,
        "dense_cache_report_sha256": sha256(args.dense_cache_report),
        "parent_run": str(args.parent_run.resolve()),
        "parent_variant": parent_config["variant"],
        "parent_model_source_commit": parent_provenance["git_commit"],
        "parent_seed": int(parent_config["seed"]),
        "point_count": point_count,
        "required_frames": required_frames,
        "device": args.device,
    }
    progress_path = args.output / "manifest.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["configuration"] != configuration:
            raise ValueError("Parent prediction resume configuration differs")
    else:
        progress = {
            "schema_version": 1,
            "configuration": configuration,
            "frames": [],
            "completed": False,
        }
        atomic_json(progress_path, progress)
    completed = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in progress["frames"]
    }

    axes = load_axes(args.data_root / "resources")
    device = torch.device(args.device)
    model = CubeCycleNet(
        parent_config["parent_head_mode"],
        torch.from_numpy(axes.doppler_mps),
        torch.from_numpy(axes.range_m),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        base_channels=int(parent_config["base_channels"]),
        log_center=float(parent_config["log_center"]),
        log_scale=float(parent_config["log_scale"]),
        static_hypothesis=parent_config["static_hypothesis"],
        maximum_offset_bins=float(parent_config["maximum_offset_bins"]),
    ).to(device)
    checkpoint = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train", "validation"),
    )
    if len(dataset) != required_frames:
        raise ValueError(f"Temporal dataset has {len(dataset)} != {required_frames} frames")

    with torch.inference_mode():
        for dataset_index in range(len(dataset)):
            record = dataset.records[dataset_index]
            sequence = int(record["sequence"])
            radar_index = int(record["radar_index"])
            key = (sequence, radar_index)
            path = prediction_path(args.output, sequence, radar_index)
            prior = completed.get(key)
            if prior and path.is_file() and sha256(path) == prior["prediction_sha256"]:
                print(json.dumps({"sequence": sequence, "radar_index": radar_index, "status": "cached"}), flush=True)
                continue
            started = time.monotonic()
            item = dataset[dataset_index]
            cube = item["cube_drae"].unsqueeze(0).to(device)
            ego_speed = item["ego_speed_mps"].reshape(1).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occupancy_logits, features = model(cube)
            _, confidence, discrete_indices = occupancy_to_points(
                occupancy_logits[0].float(), axes, point_count=point_count
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.query_cycle(
                    features, discrete_indices, ego_speed
                )
            if "static_center_mps" in prediction:
                static_center = prediction["static_center_mps"]
            else:
                batch, _, azimuth, elevation = split_query_indices(
                    discrete_indices, 1
                )
                static_center = model.static_center(
                    batch, azimuth, elevation, ego_speed
                )
            write_prediction(
                path,
                prediction,
                confidence,
                discrete_indices,
                static_center,
                metadata,
                sequence,
                radar_index,
            )
            validation = validate_prediction(path, metadata, point_count)
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
            atomic_json(progress_path, progress)
            print(json.dumps(frame_report), flush=True)
            del item, cube, ego_speed, occupancy_logits, features
            del confidence, discrete_indices, prediction, static_center
            torch.cuda.empty_cache()

    frame_reports = progress["frames"]
    checks = {
        "required_frame_count": len(frame_reports) == required_frames,
        "all_prediction_files_present": all(
            Path(frame["prediction"]).is_file() for frame in frame_reports
        ),
        "all_point_counts_match": all(
            frame["point_count"] == point_count for frame in frame_reports
        ),
        "all_probability_errors_within_tolerance": max(
            frame["probability_sum_max_error"] for frame in frame_reports
        )
        <= 2e-3,
    }
    progress.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "checks": checks,
            "completed": all(checks.values()),
        }
    )
    atomic_json(progress_path, progress)
    print(json.dumps({"checks": checks, "completed": progress["completed"]}, indent=2))
    if not progress["completed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
