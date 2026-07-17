#!/usr/bin/env python3
"""Verify the preregistered one-frame Cube occupancy overfit criteria."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import occupancy_to_points  # noqa: E402
from models.cube_occupancy import CubeOccupancyNet  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_logits(
    checkpoint_path: Path,
    cube: torch.Tensor,
    axes,
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]
    if checkpoint_config != config:
        raise ValueError("Checkpoint and run configuration differ")
    model = CubeOccupancyNet(
        config["mode"],
        torch.from_numpy(axes.doppler_mps),
        base_channels=int(config["base_channels"]),
        log_center=float(config["log_center"]),
        log_scale=float(config["log_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        logits = model(cube).float().cpu()
    return logits, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--logit-atol", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Cube occupancy overfit verification requires CUDA")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")

    run_document = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    config = run_document["config"]
    provenance = run_document["provenance"]
    if not config["overfit_one_frame"]:
        raise ValueError("The run is not marked as a one-frame overfit run")
    if int(config["train_limit"]) != 1 or int(config["validation_limit"]) != 1:
        raise ValueError("One-frame overfit run must use one train and validation frame")
    if provenance["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Run manifest hash does not match the supplied manifest")
    if provenance["normalization_sha256"] != sha256(args.normalization_stats):
        raise ValueError("Run normalization hash does not match the supplied statistics")

    records = [
        json.loads(line)
        for line in (args.run / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) < 2:
        raise ValueError("At least two training epochs are required to prove loss decrease")
    first_loss = float(records[0]["train_loss_mean"])
    final_loss = float(records[-1]["train_loss_mean"])
    finite_losses = all(np.isfinite(float(record["train_loss_mean"])) for record in records)

    device = torch.device(args.device)
    axes = load_axes(args.data_root / "resources")
    dataset = KRadarCubeDataset(
        args.data_root, args.cache_root, args.manifest, ("train",)
    )
    item = dataset[0]
    cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
    checkpoint_path = args.run / "best.pt"

    first_logits, first_checkpoint = load_logits(
        checkpoint_path, cube, axes, config, device
    )
    torch.cuda.empty_cache()
    second_logits, second_checkpoint = load_logits(
        checkpoint_path, cube, axes, config, device
    )
    maximum_logit_difference = float((first_logits - second_logits).abs().max().item())

    logits_device = first_logits[0].to(device)
    points, confidence, indices = occupancy_to_points(
        logits_device, axes, point_count=args.point_count
    )
    unique_index_count = int(torch.unique(indices, dim=0).shape[0])
    finite_output = bool(
        torch.isfinite(points).all().item() and torch.isfinite(confidence).all().item()
    )
    output_count = int(points.shape[0])
    checks = {
        "finite_training_losses": finite_losses,
        "training_loss_decreased": final_loss < first_loss,
        "checkpoint_epoch_consistent": int(first_checkpoint["epoch"])
        == int(second_checkpoint["epoch"]),
        "checkpoint_reload_logits_match": maximum_logit_difference
        <= args.logit_atol,
        "output_count_exact": output_count == args.point_count,
        "output_finite": finite_output,
        "output_indices_unique": unique_index_count == args.point_count,
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run),
        "mode": config["mode"],
        "seed": int(config["seed"]),
        "source_commit": provenance["git_commit"],
        "frame": {
            "sequence": int(item["sequence"]),
            "radar_index": int(item["radar_index"]),
        },
        "training": {
            "epoch_count": len(records),
            "first_loss": first_loss,
            "final_loss": final_loss,
            "relative_change": (final_loss - first_loss) / max(abs(first_loss), 1e-12),
        },
        "reload": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(first_checkpoint["epoch"]),
            "maximum_absolute_logit_difference": maximum_logit_difference,
            "absolute_tolerance": args.logit_atol,
        },
        "output": {
            "requested_point_count": args.point_count,
            "point_count": output_count,
            "unique_index_count": unique_index_count,
            "confidence_min": float(confidence.min().item()),
            "confidence_max": float(confidence.max().item()),
        },
        "checks": checks,
        "passed": all(checks.values()),
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
