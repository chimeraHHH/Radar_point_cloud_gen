#!/usr/bin/env python3
"""Render fixed G1 panels and deterministic worst-case comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import geometry_report, occupancy_to_points  # noqa: E402
from models.cube_occupancy import CubeOccupancyNet  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(run: Path, expected_mode: str, device: torch.device, axes) -> tuple:
    document = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if config["mode"] != expected_mode:
        raise ValueError(f"Expected {expected_mode}, found {config['mode']}")
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    if checkpoint["config"] != config or checkpoint["provenance"] != provenance:
        raise ValueError(f"Checkpoint provenance differs from {run}/config.json")
    model = CubeOccupancyNet(
        expected_mode,
        torch.from_numpy(axes.doppler_mps),
        base_channels=int(config["base_channels"]),
        log_center=float(config["log_center"]),
        log_scale=float(config["log_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    metrics = json.loads(
        (run / "best_validation_metrics.json").read_text(encoding="utf-8")
    )
    return model, config, provenance, metrics


@torch.inference_mode()
def predict(model, cube: torch.Tensor, axes, point_count: int) -> tuple[np.ndarray, np.ndarray]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(cube)
    points, confidence, _ = occupancy_to_points(
        logits[0].float(), axes, point_count=point_count
    )
    return points.cpu().numpy(), confidence.cpu().numpy()


def metric_label(report: dict) -> str:
    return (
        f"CD {report['chamfer_m']:.2f} m | "
        f"F1@1m {report['fscore_1p0m']:.3f} | "
        f"out>2m {report['outlier_fraction_2m']:.3f}"
    )


def draw_panel(
    path: Path,
    frame: dict,
    target: np.ndarray,
    cfar: np.ndarray,
    rae_points: np.ndarray,
    rae_confidence: np.ndarray,
    full_points: np.ndarray,
    full_confidence: np.ndarray,
    reports: dict,
    selection: list[str],
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(18, 7.2), constrained_layout=True)
    panels = (
        ("Radar-observable LiDAR", target[:, :3], target[:, 3], "viridis"),
        ("Official CFAR", cfar, None, None),
        ("RAE-Max", rae_points, rae_confidence, "plasma"),
        ("Full-RAED", full_points, full_confidence, "plasma"),
    )
    for axis, (title, points, confidence, colormap) in zip(axes, panels, strict=True):
        if confidence is None:
            axis.scatter(points[:, 0], points[:, 1], s=2.0, c="#222222", alpha=0.65)
        else:
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=1.4 if points.shape[0] > 5_000 else 3.0,
                c=confidence,
                cmap=colormap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.55,
                rasterized=True,
            )
        axis.set_xlim(0.0, 120.0)
        axis.set_ylim(-100.0, 100.0)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linewidth=0.35, alpha=0.3)
        axis.set_xlabel("forward x (m)")
        axis.set_ylabel("left y (m)")
        axis.set_title(title, fontsize=11)
        if title in reports:
            axis.text(
                0.02,
                0.02,
                metric_label(reports[title]),
                transform=axis.transAxes,
                fontsize=7.5,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
    figure.suptitle(
        f"K-Radar seq {frame['sequence']:02d}, frame {frame['radar_index']:05d}"
        f" | selection: {', '.join(selection)}",
        fontsize=12,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--rae-max-run", type=Path, required=True)
    parser.add_argument("--full-raed-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-frame-count", type=int, default=6)
    parser.add_argument("--worst-frame-count", type=int, default=5)
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("G1 qualitative rendering requires CUDA")
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output is not empty: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    axes = load_axes(args.data_root / "resources")
    rae_model, rae_config, rae_provenance, rae_metrics = load_run(
        args.rae_max_run, "rae_max", device, axes
    )
    full_model, full_config, full_provenance, full_metrics = load_run(
        args.full_raed_run, "full_raed", device, axes
    )
    paired_config_rae = {key: value for key, value in rae_config.items() if key != "mode"}
    paired_config_full = {
        key: value for key, value in full_config.items() if key != "mode"
    }
    if paired_config_rae != paired_config_full:
        raise ValueError("Qualitative runs are not matched except for encoding mode")
    for key in ("manifest_sha256", "scene_split_sha256", "normalization_sha256"):
        if rae_provenance[key] != full_provenance[key]:
            raise ValueError(f"Qualitative run provenance differs for {key}")
    if rae_provenance["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Manifest hash differs from the qualitative runs")
    if rae_provenance["normalization_sha256"] != sha256(args.normalization_stats):
        raise ValueError("Normalization hash differs from the qualitative runs")

    dataset = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("validation",)
    )
    frame_to_index = {
        (int(record["sequence"]), int(record["radar_index"])): index
        for index, record in enumerate(dataset.records)
    }
    fixed_indices = np.linspace(
        0, len(dataset) - 1, min(args.fixed_frame_count, len(dataset))
    ).round().astype(int).tolist()
    fixed_keys = [
        (
            int(dataset.records[index]["sequence"]),
            int(dataset.records[index]["radar_index"]),
        )
        for index in fixed_indices
    ]
    full_frame_metrics = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in full_metrics["validation"]["frames"]
    }
    worst_keys = [
        key
        for key, _ in sorted(
            full_frame_metrics.items(),
            key=lambda item: float(item[1]["generated"]["chamfer_m"]),
            reverse=True,
        )[: args.worst_frame_count]
    ]
    selected_keys = list(dict.fromkeys([*fixed_keys, *worst_keys]))
    records = []
    for key in selected_keys:
        item = dataset[frame_to_index[key]]
        cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
        target = item["target_xyz_confidence"].to(device)
        cfar = item["cfar_xyzd_power_snr"][:, :3].to(device)
        rae_points, rae_confidence = predict(
            rae_model, cube, axes, args.point_count
        )
        full_points, full_confidence = predict(
            full_model, cube, axes, args.point_count
        )
        target_xyz = target[:, :3]
        target_weight = target[:, 3]
        reports = {
            "Official CFAR": geometry_report(
                cfar, target_xyz, target_weight=target_weight
            ),
            "RAE-Max": geometry_report(
                torch.from_numpy(rae_points).to(device),
                target_xyz,
                target_weight=target_weight,
            ),
            "Full-RAED": geometry_report(
                torch.from_numpy(full_points).to(device),
                target_xyz,
                target_weight=target_weight,
            ),
        }
        selection = []
        if key in fixed_keys:
            selection.append("fixed")
        if key in worst_keys:
            selection.append("Full-RAED worst-five")
        filename = f"seq{key[0]:02d}_radar_{key[1]:05d}.png"
        draw_panel(
            args.output / filename,
            {"sequence": key[0], "radar_index": key[1]},
            target.cpu().numpy(),
            cfar.cpu().numpy(),
            rae_points,
            rae_confidence,
            full_points,
            full_confidence,
            reports,
            selection,
        )
        records.append(
            {
                "sequence": key[0],
                "radar_index": key[1],
                "selection": selection,
                "figure": filename,
                "metrics": reports,
            }
        )
        del item, cube, target, cfar, target_xyz, target_weight
        torch.cuda.empty_cache()

    report = {
        "selection_protocol": {
            "fixed": "evenly spaced ordered validation frames",
            "worst": "largest Full-RAED confidence-weighted Chamfer distance",
            "fixed_frame_count": len(fixed_keys),
            "worst_frame_count": len(worst_keys),
        },
        "rae_max_run": str(args.rae_max_run),
        "full_raed_run": str(args.full_raed_run),
        "fixed_frames": [list(key) for key in fixed_keys],
        "worst_full_raed_frames": [list(key) for key in worst_keys],
        "frames": records,
    }
    (args.output / "qualitative_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
