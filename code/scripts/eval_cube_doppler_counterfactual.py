#!/usr/bin/env python3
"""Evaluate convention-aware ego-speed counterfactual response for E5."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import occupancy_to_points  # noqa: E402
from losses.doppler_distribution import (  # noqa: E402
    circular_scalar_target,
    soft_static_target,
)
from models.cube_doppler import (  # noqa: E402
    CubeDopplerNet,
    query_cube_spectrum,
    split_query_indices,
    wrapped_delta,
)


def selected_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    return np.linspace(0, length - 1, limit).round().astype(int).tolist()


def load_model(run: Path, device: torch.device, axes) -> tuple:
    document = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if config["head_mode"] != "physics_distribution":
        raise ValueError(f"Counterfactual evaluation requires E5: {run}")
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    if checkpoint["config"] != config or checkpoint["provenance"] != provenance:
        raise ValueError(f"Checkpoint metadata differs from {run}/config.json")
    model = CubeDopplerNet(
        "physics_distribution",
        torch.from_numpy(axes.doppler_mps),
        torch.from_numpy(axes.azimuth_rad),
        torch.from_numpy(axes.elevation_rad),
        base_channels=int(config["base_channels"]),
        log_center=float(config["log_center"]),
        log_scale=float(config["log_scale"]),
        static_hypothesis=config["static_hypothesis"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, provenance


def point_static_center(
    model: CubeDopplerNet,
    indices: torch.Tensor,
    ego_speed: torch.Tensor,
) -> torch.Tensor:
    batch, _, azimuth, elevation = split_query_indices(indices, 1)
    return model.static_center(batch, azimuth, elevation, ego_speed)


def through_origin_slope(expected: np.ndarray, observed: np.ndarray) -> float:
    denominator = float(np.square(expected).sum())
    if denominator <= 1e-12:
        return float("nan")
    return float((expected * observed).sum() / denominator)


def paired_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def bootstrap_scene_seed(
    values: dict[int, dict[int, list[float]]],
    samples: int,
    rng: np.random.Generator,
) -> dict:
    seeds = sorted(values)
    scenes = sorted({scene for by_scene in values.values() for scene in by_scene})
    if not seeds or not scenes:
        raise ValueError("No scene-seed counterfactual values")

    def statistic(sampled_seeds, sampled_scenes) -> float:
        selected = []
        for seed in sampled_seeds:
            for scene in sampled_scenes:
                frame_values = values[int(seed)].get(int(scene))
                if frame_values:
                    selected.append(float(np.mean(frame_values)))
        return float(np.mean(selected)) if selected else float("nan")

    point = statistic(seeds, scenes)
    bootstrap = []
    while len(bootstrap) < samples:
        value = statistic(
            rng.choice(seeds, size=len(seeds), replace=True),
            rng.choice(scenes, size=len(scenes), replace=True),
        )
        if np.isfinite(value):
            bootstrap.append(value)
    return {
        "mean": point,
        "ci95": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        "seed_count": len(seeds),
        "scene_count": len(scenes),
    }


@torch.inference_mode()
def evaluate_run(
    run: Path,
    dataset: KRadarCubeDataset,
    frame_indices: list[int],
    axes,
    device: torch.device,
    alphas: list[float],
    point_count: int,
) -> tuple[dict, dict, list[dict]]:
    model, config, provenance = load_model(run, device, axes)
    seed = int(config["seed"])
    frames = []
    grouped: dict[int, list[float]] = defaultdict(list)
    for index in frame_indices:
        item = dataset[index]
        cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
        actual_speed = item["ego_speed_mps"].reshape(1).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occupancy_logits, features = model(cube)
        _, _, query_indices = occupancy_to_points(
            occupancy_logits[0].float(), axes, point_count=point_count
        )
        target_distribution = query_cube_spectrum(cube, query_indices)
        target_scalar = circular_scalar_target(
            target_distribution,
            model.doppler_mps,
            model.doppler_lower_mps,
            model.doppler_period_mps,
        )
        reference_center = point_static_center(
            model, query_indices, actual_speed
        )
        target_static = soft_static_target(
            target_scalar, reference_center, model.doppler_period_mps
        )
        static_mask = target_static >= 0.5
        if int(static_mask.sum().item()) < 10:
            raise ValueError("Fewer than ten static query points in a validation frame")

        predictions = {}
        centers = {}
        for alpha in alphas:
            speed = actual_speed * alpha
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.query(features, query_indices, speed)
            predictions[alpha] = prediction["scalar_mps"].float()[static_mask]
            centers[alpha] = point_static_center(
                model, query_indices, speed
            )[static_mask]

        baseline_prediction = predictions[alphas[0]]
        baseline_center = centers[alphas[0]]
        dose = []
        expected_all = []
        observed_all = []
        for alpha in alphas:
            observed = wrapped_delta(
                predictions[alpha], baseline_prediction, model.doppler_period_mps
            )
            expected = wrapped_delta(
                centers[alpha], baseline_center, model.doppler_period_mps
            )
            dose.append(
                {
                    "alpha": alpha,
                    "observed_abs_change_median_mps": float(
                        observed.abs().median().item()
                    ),
                    "expected_abs_change_median_mps": float(
                        expected.abs().median().item()
                    ),
                }
            )
            reliable = (expected.abs() >= 0.1) & (
                expected.abs() <= model.doppler_period_mps / 4.0
            )
            if reliable.any():
                expected_all.append(expected[reliable].cpu().numpy())
                observed_all.append(observed[reliable].cpu().numpy())

        if config["static_hypothesis"] == "zero_centered":
            x = np.asarray(alphas, dtype=np.float64) - alphas[0]
            y = np.asarray(
                [entry["observed_abs_change_median_mps"] for entry in dose],
                dtype=np.float64,
            )
            frame_value = through_origin_slope(x, y)
            frame_correlation = float("nan")
        else:
            if not expected_all:
                raise ValueError("No reliable non-aliased counterfactual displacements")
            expected_values = np.concatenate(expected_all)
            observed_values = np.concatenate(observed_all)
            frame_value = through_origin_slope(expected_values, observed_values)
            frame_correlation = paired_correlation(expected_values, observed_values)
        grouped[int(item["sequence"])].append(frame_value)
        frames.append(
            {
                "seed": seed,
                "sequence": int(item["sequence"]),
                "radar_index": int(item["radar_index"]),
                "actual_speed_mps": float(actual_speed.item()),
                "static_point_count": int(static_mask.sum().item()),
                "response_slope": frame_value,
                "response_correlation": frame_correlation,
                "dose_response": dose,
            }
        )
        del item, cube, occupancy_logits, features, query_indices
        del target_distribution, target_scalar, reference_center, target_static
        torch.cuda.empty_cache()
    return config, provenance, frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2])
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--zero-slope-maximum", type=float, default=0.1)
    parser.add_argument("--radial-slope-minimum", type=float, default=0.0)
    parser.add_argument("--radial-correlation-minimum", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Cube Doppler counterfactual evaluation requires CUDA")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    if sorted(args.alphas) != args.alphas or args.alphas[0] != 0.0:
        raise ValueError("Counterfactual alphas must be sorted and start at zero")

    device = torch.device(args.device)
    axes = load_axes(args.data_root / "resources")
    dataset = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    frame_indices = selected_indices(len(dataset), args.frame_limit)
    all_frames = []
    configs = []
    provenances = []
    for run in args.runs:
        config, provenance, frames = evaluate_run(
            run,
            dataset,
            frame_indices,
            axes,
            device,
            args.alphas,
            args.point_count,
        )
        configs.append(config)
        provenances.append(provenance)
        all_frames.extend(frames)

    seeds = sorted({int(config["seed"]) for config in configs})
    if len(seeds) != args.required_seeds or len(seeds) != len(configs):
        raise ValueError(f"Expected {args.required_seeds} unique E5 seeds")
    conventions = {config["static_hypothesis"] for config in configs}
    if len(conventions) != 1:
        raise ValueError("Counterfactual runs use different static conventions")
    convention = next(iter(conventions))
    reference_provenance = {
        key: provenances[0][key]
        for key in (
            "git_commit",
            "manifest_sha256",
            "scene_split_sha256",
            "normalization_sha256",
            "static_doppler_audit_sha256",
            "static_doppler_audit_passed",
            "torch_version",
            "device",
        )
    }
    for provenance in provenances[1:]:
        if any(provenance[key] != value for key, value in reference_provenance.items()):
            raise ValueError("Counterfactual run provenance differs across seeds")

    grouped: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    correlation_grouped: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for frame in all_frames:
        grouped[frame["seed"]][frame["sequence"]].append(frame["response_slope"])
        if np.isfinite(frame["response_correlation"]):
            correlation_grouped[frame["seed"]][frame["sequence"]].append(
                frame["response_correlation"]
            )
    rng = np.random.default_rng(args.seed)
    slope = bootstrap_scene_seed(grouped, args.bootstrap_samples, rng)
    correlation = (
        bootstrap_scene_seed(correlation_grouped, args.bootstrap_samples, rng)
        if correlation_grouped
        else None
    )
    if convention == "zero_centered":
        checks = {
            "zero_centered_invariance": slope["ci95"][1]
            <= args.zero_slope_maximum
        }
    else:
        checks = {
            "radial_response_positive": slope["ci95"][0]
            > args.radial_slope_minimum,
            "radial_response_correlated": correlation is not None
            and correlation["ci95"][0] >= args.radial_correlation_minimum,
        }
    report = {
        "schema_version": 1,
        "convention": convention,
        "alphas": args.alphas,
        "frame_count_per_seed": len(frame_indices),
        "seeds": seeds,
        "provenance": reference_provenance,
        "thresholds": {
            "zero_slope_maximum": args.zero_slope_maximum,
            "radial_slope_minimum": args.radial_slope_minimum,
            "radial_correlation_minimum": args.radial_correlation_minimum,
        },
        "response_slope": slope,
        "response_correlation": correlation,
        "checks": checks,
        "passed": all(checks.values()),
        "frames": all_frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
