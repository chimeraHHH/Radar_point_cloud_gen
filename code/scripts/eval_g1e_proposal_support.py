#!/usr/bin/env python3
"""Run the frozen G1E-D0 proposal-support diagnosis on archived RaLD VAEs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from cube_dense.rald_adapter import sample_target_points  # noqa: E402
from eval.dense_geometry import geometry_report  # noqa: E402
from eval.rald_guided_query import duplicate_report  # noqa: E402
from models.cube_cycle import continuous_rae_to_xyz  # noqa: E402
from models.rald_matched import RaLDPointAutoencoder  # noqa: E402
from models.rald_proposal_support import latent_only_proposal_support  # noqa: E402


PROTOCOL = "g1e_rald_latent_edm_d0_proposal_support_v1"
OFFICIAL_RALD_COMMIT = "ffec4b41241391734b1eda5c093de843c909eb8e"
SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MAXIMUM_CHAMFER_M = 5.0
MAXIMUM_OUTLIER_FRACTION = 0.25
MINIMUM_CHAMFER_IMPROVEMENT = 0.30
BASE_SEED_COUNT = 1_000
COARSE_QUERY_COUNT = 32_000
SELECTED_COARSE_COUNT = 2_500
FINAL_POINT_COUNT = 10_000
NMS_KERNEL = (5, 5, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit(repo: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if not SOURCE_PATTERN.fullmatch(commit):
        raise RuntimeError("G1E-D0 requires a full Git source commit")
    return commit


def require_h200(device_name: str) -> tuple[torch.device, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("G1E-D0 requires CUDA on an H200")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("G1E-D0 is CUDA-only")
    resolved = torch.cuda.get_device_name(device)
    if "H200" not in resolved.upper():
        raise RuntimeError(f"G1E-D0 requires an H200, got {resolved}")
    return device, resolved


def frame_generator(
    device: torch.device,
    seed: int,
    sequence: int,
    radar_index: int,
) -> torch.Generator:
    value = seed + sequence * 1_000_003 + radar_index * 101
    return torch.Generator(device=device).manual_seed(value)


def build_autoencoder(config: dict) -> RaLDPointAutoencoder:
    return RaLDPointAutoencoder(
        point_count=int(config["ae_point_count"]),
        latent_count=int(config["latent_count"]),
        model_dim=int(config["model_dim"]),
        latent_dim=int(config["latent_dim"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        head_dim=int(config["head_dim"]),
    )


def validate_archived_run(gate_path: Path) -> tuple[dict, Path, dict, dict, Path]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    run = Path(gate["run"]).resolve()
    config_path = run / "config.json"
    checkpoint_path = run / "best.pt"
    metrics_path = run / "best_validation_metrics.json"
    for path in (config_path, checkpoint_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Archived G1E-D0 input is missing: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != config:
        raise ValueError("Archived RaLD checkpoint config differs from config.json")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("git_commit") != gate.get("source_commit"):
        raise ValueError("Archived RaLD checkpoint source differs from gate record")
    required = {
        "overfit_one_frame": True,
        "train_limit": 1,
        "validation_limit": 1,
        "positive_query_count": 625,
        "negative_query_count": 9375,
        "output_point_count": 10_000,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Archived RaLD config {key} is not {expected}")
    if gate.get("required_epoch") != 100 or config.get("epochs") != 100:
        raise ValueError("G1E-D0 requires an archived run completed to epoch 100")
    archived_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    archived_chamfer = float(
        archived_metrics["validation"]["generated"]["chamfer_m"]["median"]
    )
    if not np.isclose(
        archived_chamfer,
        float(gate["metrics"]["chamfer_m"]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Archived RaLD metrics differ from the gate record")
    return gate, run, config, checkpoint, metrics_path


@torch.inference_mode()
def evaluate_run(
    gate_path: Path,
    dataset: KRadarCubeDataset,
    axes,
    device: torch.device,
) -> dict:
    gate, run, config, checkpoint, metrics_path = validate_archived_run(gate_path)
    item = dataset[0]
    cube = item["cube_drae"].unsqueeze(0).to(device, non_blocking=True)
    target = item["target_xyz_confidence"].to(device, non_blocking=True)
    generator = frame_generator(
        device,
        int(config["seed"]),
        int(item["sequence"]),
        int(item["radar_index"]),
    )
    points = sample_target_points(
        target,
        axes,
        int(config["ae_point_count"]),
        generator,
        sampling_mode=config["surface_sampling_mode"],
    ).unsqueeze(0)

    model = build_autoencoder(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        posterior = model.encode(points)
        support = latent_only_proposal_support(
            model,
            posterior.mean,
            cube,
            base_seed_count=BASE_SEED_COUNT,
            selected_coarse_count=SELECTED_COARSE_COUNT,
            nms_kernel=NMS_KERNEL,
            decode_chunk_size=int(config["query_chunk_size"]),
        )
    final_rae = support.final_coordinates_rae[0].float()
    predicted_xyz = continuous_rae_to_xyz(
        final_rae,
        torch.as_tensor(axes.range_m, device=device),
        torch.as_tensor(axes.azimuth_rad, device=device),
        torch.as_tensor(axes.elevation_rad, device=device),
    )
    geometry = geometry_report(
        predicted_xyz,
        target[:, :3].float(),
        target_weight=target[:, 3].float(),
    )
    duplicates = duplicate_report(predicted_xyz)
    archived_chamfer = float(gate["metrics"]["chamfer_m"])
    proposal_chamfer = float(geometry["chamfer_m"])
    improvement = (archived_chamfer - proposal_chamfer) / archived_chamfer
    checks = {
        "archived_source_matched": (
            checkpoint["provenance"]["git_commit"] == gate["source_commit"]
        ),
        "one_frame_configuration": bool(config["overfit_one_frame"]),
        "proposal_chamfer": proposal_chamfer <= MAXIMUM_CHAMFER_M,
        "proposal_outlier": (
            float(geometry["outlier_fraction_2m"]) <= MAXIMUM_OUTLIER_FRACTION
        ),
        "minimum_relative_improvement": (
            improvement >= MINIMUM_CHAMFER_IMPROVEMENT
        ),
        "base_seed_count": support.proposal_flat_index.shape[1]
        == BASE_SEED_COUNT,
        "coarse_query_count": support.coarse_coordinates_rae.shape[1]
        == COARSE_QUERY_COUNT,
        "selected_coarse_count": (
            support.selected_coarse_coordinates_rae.shape[1]
            == SELECTED_COARSE_COUNT
        ),
        "final_point_count": final_rae.shape[0] == FINAL_POINT_COUNT,
        "coordinate_and_latent_decoder_only": True,
        "learned_g1d_checkpoint_accessed": False,
        "cfar_query_helper": False,
        "test_accessed": False,
    }
    return {
        "gate_path": str(gate_path.resolve()),
        "gate_sha256": sha256(gate_path),
        "archived_run": str(run),
        "archived_source_commit": gate["source_commit"],
        "config_sha256": sha256(run / "config.json"),
        "checkpoint": str(run / "best.pt"),
        "checkpoint_sha256": sha256(run / "best.pt"),
        "archived_metrics": str(metrics_path),
        "archived_metrics_sha256": sha256(metrics_path),
        "frame": {
            "partition": item["partition"],
            "sequence": int(item["sequence"]),
            "radar_index": int(item["radar_index"]),
        },
        "query_decoder_inputs": ["normalized_rae", "latent"],
        "counts": {
            "base_seed": int(support.proposal_flat_index.shape[1]),
            "coarse": int(support.coarse_coordinates_rae.shape[1]),
            "selected_coarse": int(
                support.selected_coarse_coordinates_rae.shape[1]
            ),
            "final": int(final_rae.shape[0]),
        },
        "archived_full_grid": {
            "chamfer_m": archived_chamfer,
            "outlier_fraction_2m": float(
                gate["metrics"]["outlier_fraction_2m"]
            ),
        },
        "proposal_support": {
            "geometry": geometry,
            "duplicates": duplicates,
            "chamfer_improvement_fraction": improvement,
            "occupancy_logit": {
                "minimum": float(
                    support.coarse_occupancy_logit.float().amin().item()
                ),
                "median": float(
                    support.coarse_occupancy_logit.float().median().item()
                ),
                "maximum": float(
                    support.coarse_occupancy_logit.float().amax().item()
                ),
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    current_commit = git_commit(repo)
    if args.source_commit != current_commit:
        raise ValueError("G1E-D0 requested source differs from the Git snapshot")
    device, device_name = require_h200(args.device)
    axes = load_axes(args.data_root / "resources")
    dataset = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("train",),
    )
    runs = [
        evaluate_run(gate_path, dataset, axes, device)
        for gate_path in args.gate
    ]
    document = {
        "protocol": PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": current_commit,
        "official_rald_commit": OFFICIAL_RALD_COMMIT,
        "device": device_name,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "thresholds": {
            "maximum_chamfer_m": MAXIMUM_CHAMFER_M,
            "maximum_outlier_fraction_2m": MAXIMUM_OUTLIER_FRACTION,
            "minimum_chamfer_improvement_fraction": (
                MINIMUM_CHAMFER_IMPROVEMENT
            ),
        },
        "runs": runs,
        "passed": any(run["passed"] for run in runs),
        "test_accessed": False,
    }
    atomic_json(args.output, document)
    print(json.dumps(document, indent=2))
    if not document["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
