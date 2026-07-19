#!/usr/bin/env python3
"""Cache frozen K-Radar target latents from a matched RaLD-style AE."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarDenseTargetDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.rald_adapter import sample_target_points  # noqa: E402
from models.rald_matched import RaLDPointAutoencoder  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_model(config: dict) -> RaLDPointAutoencoder:
    return RaLDPointAutoencoder(
        point_count=int(config["ae_point_count"]),
        latent_count=int(config["latent_count"]),
        model_dim=int(config["model_dim"]),
        latent_dim=int(config["latent_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
    )


def frame_generator(
    device: torch.device, seed: int, sequence: int, radar_index: int
) -> torch.Generator:
    value = seed + sequence * 1_000_003 + radar_index * 101
    return torch.Generator(device=device).manual_seed(value)


def valid_existing(path: Path, shape: tuple[int, int]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as cache:
            latent = cache["latent_mean"]
            posterior_kl = cache["posterior_kl"]
        return bool(
            latent.shape == shape
            and np.isfinite(latent).all()
            and np.isfinite(posterior_kl).all()
        )
    except (OSError, KeyError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--ae-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partitions", nargs="+", default=["train", "validation"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpu-name")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("RaLD latent caching requires CUDA")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device)
    if args.required_gpu_name and device_name != args.required_gpu_name:
        raise RuntimeError(
            f"RaLD latent caching requires {args.required_gpu_name}, got {device_name}"
        )
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    source_commit = args.source_commit or git_commit(Path(__file__).resolve().parents[2])
    if source_commit is None:
        raise RuntimeError("Source commit is required for reproducibility")
    manifest_hash = sha256(args.manifest)
    scene_split_hash = sha256(args.scene_split)
    ae_document = json.loads((args.ae_run / "config.json").read_text(encoding="utf-8"))
    if ae_document["provenance"]["manifest_sha256"] != manifest_hash:
        raise ValueError("AE and latent cache manifests differ")
    if ae_document["provenance"]["scene_split_sha256"] != scene_split_hash:
        raise ValueError("AE and latent cache scene splits differ")
    checkpoint_path = args.ae_run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config"] != ae_document["config"]:
        raise ValueError("AE checkpoint configuration differs from run document")
    if checkpoint["provenance"] != ae_document["provenance"]:
        raise ValueError("AE checkpoint provenance differs from run document")
    model = build_model(ae_document["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    axes = load_axes(args.data_root / "resources")
    dataset = KRadarDenseTargetDataset(
        args.cache_root, args.manifest, tuple(args.partitions)
    )
    latent_shape = (
        int(ae_document["config"]["latent_count"]),
        int(ae_document["config"]["latent_dim"]),
    )
    records = []
    for position in range(len(dataset)):
        item = dataset[position]
        sequence = int(item["sequence"])
        radar_index = int(item["radar_index"])
        output_path = args.output / f"seq{sequence:02d}_radar_{radar_index:05d}.npz"
        if not valid_existing(output_path, latent_shape):
            target = item["target_xyz_confidence"].to(device, non_blocking=True)
            generator = frame_generator(
                device,
                int(ae_document["config"]["seed"]),
                sequence,
                radar_index,
            )
            points = sample_target_points(
                target,
                axes,
                int(ae_document["config"]["ae_point_count"]),
                generator,
            ).unsqueeze(0)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                posterior = model.encode(points)
            latent = posterior.mean[0].float().cpu().numpy()
            posterior_kl = posterior.kl()[0].float().cpu().numpy()
            temporary = output_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                latent_mean=latent,
                posterior_kl=posterior_kl,
            )
            temporary.replace(output_path)
            del target, points, posterior
        records.append(
            {
                "sequence": sequence,
                "radar_index": radar_index,
                "partition": item["partition"],
                "path": output_path.name,
                "sha256": sha256(output_path),
            }
        )
        print(
            json.dumps(
                {
                    "completed": position + 1,
                    "total": len(dataset),
                    "sequence": sequence,
                    "radar_index": radar_index,
                }
            ),
            flush=True,
        )
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "manifest_sha256": manifest_hash,
        "scene_split_sha256": scene_split_hash,
        "partitions": args.partitions,
        "frame_count": len(records),
        "latent_shape": list(latent_shape),
        "ae_run": str(args.ae_run),
        "ae_source_commit": ae_document["provenance"]["git_commit"],
        "ae_checkpoint": str(checkpoint_path),
        "ae_checkpoint_sha256": sha256(checkpoint_path),
        "device": device_name,
        "records": records,
    }
    summary_path = args.output / "latent_cache_manifest.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
