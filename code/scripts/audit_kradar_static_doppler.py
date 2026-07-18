#!/usr/bin/env python3
"""Calibrate K-Radar Doppler convention on CFAR points outside labeled boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat


HYPOTHESES = ("negative_ego", "positive_ego", "zero_centered")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_boxes(path: Path) -> list[dict]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        values = [value.strip() for value in line.split(",")]
        if len(values) < 11 or values[0] != "*":
            continue
        boxes.append(
            {
                "class": values[3],
                "center_xyz_m": np.asarray(values[4:7], dtype=np.float64),
                "yaw_rad": np.deg2rad(float(values[7])),
                "half_size_xyz_m": np.asarray(values[8:11], dtype=np.float64),
            }
        )
    return boxes


def outside_boxes(points_xyz: np.ndarray, boxes: list[dict], margin_m: float) -> np.ndarray:
    outside = np.ones(points_xyz.shape[0], dtype=bool)
    for box in boxes:
        centered = points_xyz - box["center_xyz_m"]
        cosine = np.cos(box["yaw_rad"])
        sine = np.sin(box["yaw_rad"])
        local_x = cosine * centered[:, 0] + sine * centered[:, 1]
        local_y = -sine * centered[:, 0] + cosine * centered[:, 1]
        local_z = centered[:, 2]
        extent = box["half_size_xyz_m"] + margin_m
        inside = (
            (np.abs(local_x) <= extent[0])
            & (np.abs(local_y) <= extent[1])
            & (np.abs(local_z) <= extent[2])
        )
        outside &= ~inside
    return outside


def wrap(values: np.ndarray, lower: float, period: float) -> np.ndarray:
    return np.remainder(values - lower, period) + lower


def circular_error(
    observed: np.ndarray, prediction: np.ndarray, period: float
) -> np.ndarray:
    return np.remainder(observed - prediction + period / 2.0, period) - period / 2.0


def filter_background_by_snr_quantile(
    cfar: np.ndarray, background: np.ndarray, quantile: float
) -> tuple[np.ndarray, float | None]:
    if not 0.0 <= quantile < 1.0:
        raise ValueError("Background SNR quantile must be in [0, 1)")
    if background.shape != (cfar.shape[0],) or not background.any():
        raise ValueError("Background mask is empty or malformed")
    if quantile == 0.0:
        return background.copy(), None
    threshold = float(np.quantile(cfar[background, 5], quantile))
    return background & (cfar[:, 5] >= threshold), threshold


def summarize_frames(frames: list[dict]) -> dict:
    if not frames:
        raise ValueError("Cannot summarize an empty partition")
    hypothesis = {}
    for name in HYPOTHESES:
        frame_medians = np.asarray(
            [frame["hypotheses"][name]["median_abs_error_mps"] for frame in frames],
            dtype=np.float64,
        )
        point_errors = np.concatenate(
            [np.asarray(frame["_errors"][name], dtype=np.float64) for frame in frames]
        )
        hypothesis[name] = {
            "frame_median_error_mean_mps": float(frame_medians.mean()),
            "frame_median_error_median_mps": float(np.median(frame_medians)),
            "point_abs_error_median_mps": float(np.median(point_errors)),
            "point_abs_error_p90_mps": float(np.quantile(point_errors, 0.9)),
        }
    selected = min(
        HYPOTHESES,
        key=lambda name: hypothesis[name]["frame_median_error_median_mps"],
    )
    ranked = sorted(
        HYPOTHESES,
        key=lambda name: hypothesis[name]["frame_median_error_median_mps"],
    )
    return {
        "frame_count": len(frames),
        "sequence_count": len({frame["sequence"] for frame in frames}),
        "background_point_count": int(sum(frame["background_count"] for frame in frames)),
        "hypotheses": hypothesis,
        "selected_hypothesis": selected,
        "selected_margin_to_second_mps": float(
            hypothesis[ranked[1]]["frame_median_error_median_mps"]
            - hypothesis[ranked[0]]["frame_median_error_median_mps"]
        ),
        "frame_winner_counts": dict(Counter(frame["preferred_hypothesis"] for frame in frames)),
    }


def clean_frame(frame: dict) -> dict:
    return {key: value for key, value in frame.items() if key != "_errors"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-margin-m", type=float, default=1.0)
    parser.add_argument("--minimum-range-m", type=float, default=3.0)
    parser.add_argument("--background-snr-quantile", type=float, default=0.0)
    parser.add_argument("--minimum-selection-margin-mps", type=float, default=0.05)
    parser.add_argument("--required-frames", type=int, default=100)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.background_snr_quantile < 1.0:
        raise ValueError("--background-snr-quantile must be in [0, 1)")

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["frames"]
    if args.available_only:
        records = [
            record
            for record in records
            if (
                args.cache_root
                / f"seq{int(record['sequence']):02d}_radar_{int(record['radar_index']):05d}.npz"
            ).exists()
        ]
    if len(records) != args.required_frames:
        raise ValueError(
            f"Expected {args.required_frames} manifest frames, found {len(records)}"
        )
    doppler = loadmat(args.data_root / "resources/arr_doppler.mat")[
        "arr_doppler"
    ].reshape(-1)
    if doppler.shape != (64,) or not np.all(np.diff(doppler) > 0):
        raise ValueError("Expected a strictly increasing 64-bin Doppler axis")
    step = float(np.median(np.diff(doppler)))
    period = step * len(doppler)
    lower = float(doppler[0])

    frames = []
    errors = []
    for record in records:
        sequence = int(record["sequence"])
        radar_index = int(record["radar_index"])
        cache_path = args.cache_root / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"
        try:
            with np.load(cache_path) as cache:
                cfar = cache["cfar_xyzd_power_snr"].astype(np.float64)
                ego_speed = float(cache["ego_speed_mps"])
            label_path = (
                args.data_root / str(sequence) / "info_label" / record["label"]
            )
            boxes = parse_boxes(label_path)
            radius = np.linalg.norm(cfar[:, :3], axis=1)
            unfiltered_background = outside_boxes(
                cfar[:, :3], boxes, margin_m=args.box_margin_m
            ) & (radius >= args.minimum_range_m)
            if not unfiltered_background.any():
                raise ValueError("No background CFAR points remain after masking")
            background, snr_threshold = filter_background_by_snr_quantile(
                cfar, unfiltered_background, args.background_snr_quantile
            )
            if not background.any():
                raise ValueError("No background CFAR points remain after SNR filtering")
            observed = cfar[background, 3]
            forward_projection = (
                ego_speed * cfar[background, 0] / np.maximum(radius[background], 1e-6)
            )
            predictions = {
                "negative_ego": wrap(-forward_projection, lower, period),
                "positive_ego": wrap(forward_projection, lower, period),
                "zero_centered": np.zeros_like(forward_projection),
            }
            hypothesis_report = {}
            stored_errors = {}
            for name, prediction in predictions.items():
                absolute = np.abs(circular_error(observed, prediction, period))
                stored_errors[name] = absolute.tolist()
                hypothesis_report[name] = {
                    "median_abs_error_mps": float(np.median(absolute)),
                    "p90_abs_error_mps": float(np.quantile(absolute, 0.9)),
                }
            preferred = min(
                HYPOTHESES,
                key=lambda name: hypothesis_report[name]["median_abs_error_mps"],
            )
            frames.append(
                {
                    "sequence": sequence,
                    "radar_index": radar_index,
                    "partition": record["partition"],
                    "ego_speed_mps": ego_speed,
                    "cfar_count": int(cfar.shape[0]),
                    "box_count": len(boxes),
                    "unfiltered_background_count": int(unfiltered_background.sum()),
                    "background_count": int(background.sum()),
                    "background_snr_threshold": snr_threshold,
                    "hypotheses": hypothesis_report,
                    "preferred_hypothesis": preferred,
                    "_errors": stored_errors,
                }
            )
        except Exception as error:
            errors.append(
                {
                    "sequence": sequence,
                    "radar_index": radar_index,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    partition_frames = {
        partition: [frame for frame in frames if frame["partition"] == partition]
        for partition in ("train", "validation")
    }
    train = summarize_frames(partition_frames["train"])
    validation = summarize_frames(partition_frames["validation"])
    frozen = train["selected_hypothesis"]
    validation_frozen_error = validation["hypotheses"][frozen][
        "frame_median_error_median_mps"
    ]
    random_baseline = period / 4.0
    checks = {
        "required_frame_count": len(frames) == args.required_frames,
        "no_frame_errors": not errors,
        "train_hypothesis_meets_selection_margin": train[
            "selected_margin_to_second_mps"
        ]
        >= args.minimum_selection_margin_mps,
        "frozen_hypothesis_beats_random_on_validation": validation_frozen_error
        < random_baseline,
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "manifest_sha256": sha256(args.manifest),
        "protocol": {
            "background_definition": "CFAR points outside all labeled 3D boxes",
            "box_margin_m": args.box_margin_m,
            "minimum_range_m": args.minimum_range_m,
            "background_snr_quantile": args.background_snr_quantile,
            "minimum_selection_margin_mps": args.minimum_selection_margin_mps,
            "selection_partition": "train",
            "evaluation_partition": "validation",
            "doppler_bin_step_mps": step,
            "doppler_period_mps": period,
            "random_circular_median_baseline_mps": random_baseline,
        },
        "train": train,
        "validation": validation,
        "frozen_hypothesis": frozen,
        "validation_frozen_error_mps": validation_frozen_error,
        "checks": checks,
        "passed": all(checks.values()),
        "errors": errors,
        "frames": [clean_frame(frame) for frame in frames],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
