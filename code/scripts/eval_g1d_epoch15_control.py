#!/usr/bin/env python3
"""Re-evaluate the frozen G1D epoch-15 EMA with corrected geometry metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from scripts.g1b_contract import sha256  # noqa: E402
from scripts.train_g1g_hierarchy import (  # noqa: E402
    FORMAL_VALIDATION_COUNT,
    G1D_CONTROL_EPOCH,
    G1D_CONTROL_PROTOCOL,
    G1D_CONTROL_SOURCE_COMMIT,
    canonical_data_contract,
    frame_data_hashes,
    validate_data_contract,
    validate_frozen_input_hashes,
)
from scripts.train_rald_query_field import (  # noqa: E402
    PROTOCOL as G1D_PROTOCOL,
    TrainConfig,
    build_model,
    evaluate,
    require_h200,
    verify_source_tree,
)


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_checkpoint(checkpoint: dict[str, Any]) -> TrainConfig:
    if int(checkpoint.get("epoch", -1)) != G1D_CONTROL_EPOCH:
        raise ValueError("G1D control requires the frozen epoch-15 checkpoint")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("git_commit") != G1D_CONTROL_SOURCE_COMMIT:
        raise ValueError("G1D control checkpoint source commit differs")
    config_document = checkpoint.get("config")
    if not isinstance(config_document, dict):
        raise ValueError("G1D control checkpoint has no configuration")
    config = TrainConfig(**config_document)
    if config.protocol != G1D_PROTOCOL:
        raise ValueError("G1D control checkpoint protocol differs")
    if config.seed != 20260716 or config.point_count != 10_000:
        raise ValueError("G1D control checkpoint is not the frozen formal arm")
    if "ema_model" not in checkpoint:
        raise ValueError("G1D control checkpoint has no EMA state")
    return config


def build_document(
    *,
    source_commit: str,
    source_script: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    data_contract: dict[str, Any],
    validation_records: list[dict[str, Any]],
    metrics: dict[str, Any],
    device_name: str,
) -> dict[str, Any]:
    if metrics.get("frame_count") != FORMAL_VALIDATION_COUNT:
        raise ValueError("G1D control did not evaluate all validation frames")
    far_metric = "range_60_120m_completeness_mean_distance_m"
    far_target_identities = [
        {
            "sequence": int(frame["sequence"]),
            "radar_index": int(frame["radar_index"]),
        }
        for frame in metrics.get("frames", [])
        if far_metric in frame.get("generated", {})
    ]
    far = metrics.get("generated", {}).get(
        far_metric, {}
    )
    if (
        not far_target_identities
        or far.get("sample_count") != len(far_target_identities)
    ):
        raise ValueError(
            "Corrected far completeness must cover every far-target frame"
        )
    return {
        "schema_version": 1,
        "protocol": G1D_CONTROL_PROTOCOL,
        "artifact_type": "matched_g1d_epoch15_corrected_geometry_control",
        "source": {
            "git_commit": source_commit,
            "script": str(source_script.resolve()),
            "script_sha256": sha256(source_script),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "source_commit": checkpoint["provenance"]["git_commit"],
            "epoch": int(checkpoint["epoch"]),
            "evaluation_state": "ema_model",
            "config": checkpoint["config"],
        },
        "evaluator": {
            "dense_geometry_path": str(
                (source_script.parents[1] / "eval/dense_geometry.py").resolve()
            ),
            "dense_geometry_sha256": sha256(
                source_script.parents[1] / "eval/dense_geometry.py"
            ),
            "far_target_frame_censoring_fixed": True,
        },
        "data_contract": data_contract,
        "validation_frame_identities": [
            {
                "sequence": int(record["sequence"]),
                "radar_index": int(record["radar_index"]),
            }
            for record in validation_records
        ],
        "far_target_frame_identities": far_target_identities,
        "metrics": metrics,
        "device": device_name,
        "torch_version": torch.__version__,
        "test_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"G1D control output already exists: {args.output}")
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    validate_data_contract(manifest, scene_split)
    frozen_hashes = validate_frozen_input_hashes(
        args.manifest,
        args.scene_split,
        args.normalization,
    )
    data_hashes = {
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": frozen_hashes["manifest"],
        },
        "scene_split": {
            "path": str(args.scene_split.resolve()),
            "sha256": frozen_hashes["scene_split"],
        },
        "normalization": {
            "path": str(args.normalization.resolve()),
            "sha256": frozen_hashes["normalization"],
        },
        "range_azimuth_elevation_axes": {
            "path": str((args.data_root / "resources/info_arr.mat").resolve()),
            "sha256": sha256(args.data_root / "resources/info_arr.mat"),
        },
        "doppler_axis": {
            "path": str((args.data_root / "resources/arr_doppler.mat").resolve()),
            "sha256": sha256(args.data_root / "resources/arr_doppler.mat"),
        },
        "frames": frame_data_hashes(
            manifest["frames"],
            args.data_root,
            args.cache_root,
        ),
    }

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = validate_checkpoint(checkpoint)
    axes = load_axes(args.data_root / "resources")
    model = build_model(config, axes, normalization).to(device)
    model.load_state_dict(checkpoint["ema_model"], strict=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    validation_set = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    if len(validation_set) != FORMAL_VALIDATION_COUNT:
        raise ValueError("G1D control requires exactly 24 validation frames")
    proposal_cache: dict[tuple[int, int], torch.Tensor] = {}
    metrics = evaluate(
        model,
        validation_set,
        list(range(len(validation_set))),
        device,
        config,
        proposal_cache,
    )
    document = build_document(
        source_commit=args.source_commit,
        source_script=Path(__file__).resolve(),
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        data_contract=canonical_data_contract(data_hashes),
        validation_records=validation_set.records,
        metrics=metrics,
        device_name=device_name,
    )
    atomic_json(args.output, document)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "checkpoint_sha256": document["checkpoint"]["sha256"],
                "metrics": {
                    "completeness_median_m": metrics["generated"][
                        "completeness_mean_distance_m"
                    ]["median"],
                    "far_completeness_mean_m": metrics["generated"][
                        "range_60_120m_completeness_mean_distance_m"
                    ]["mean"],
                    "far_sample_count": metrics["generated"][
                        "range_60_120m_completeness_mean_distance_m"
                    ]["sample_count"],
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
