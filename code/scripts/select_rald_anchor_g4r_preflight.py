#!/usr/bin/env python3
"""Freeze one RaLD representation level after the bounded G4R preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1b_contract import sha256


PROTOCOL = "rald_anchor_g4r_preflight_selection_v1"
MODES = ("token", "latent", "query")
TIE_PRIORITY = {"latent": 0, "token": 1, "query": 2}


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_run(path: Path, mode: str, source_commit: str, seed: int) -> dict:
    path = path.resolve()
    config_path = path / "config.json"
    checkpoint_path = path / "best.pt"
    metrics_path = path / "best_validation_metrics.json"
    if not all(
        candidate.is_file()
        for candidate in (config_path, checkpoint_path, metrics_path)
    ):
        raise FileNotFoundError(f"Incomplete G4R preflight run: {path}")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if (
        config["fusion_mode"] != mode
        or int(config["seed"]) != seed
        or int(config["epochs"]) != 5
        or provenance["git_commit"] != source_commit
        or provenance["zero_gate_identity"]["exact_identity"] is not True
        or metrics.get("completed") is not True
    ):
        raise ValueError(f"G4R preflight contract differs for {mode}")
    score = float(metrics["selection_value"])
    validation = metrics["validation"]
    confidence = float(validation["cycle"]["confidence_mean"]["median"])
    coverage = float(validation["cycle"]["covered_cell_count"]["median"])
    chamfer = float(validation["generated_geometry"]["chamfer_m"]["median"])
    local_kl = float(validation["cycle"]["local_spectrum_kl"]["median"])
    if not all(np.isfinite(value) for value in (score, confidence, coverage, chamfer, local_kl)):
        raise ValueError(f"G4R preflight contains non-finite metrics for {mode}")
    return {
        "path": str(path),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "metrics_sha256": sha256(metrics_path),
        "parent_checkpoint_sha256": provenance["parent_checkpoint_sha256"],
        "parent_prediction_manifest_sha256": provenance[
            "parent_prediction_manifest_sha256"
        ],
        "score": score,
        "confidence": confidence,
        "coverage": coverage,
        "chamfer": chamfer,
        "local_spectrum_kl": local_kl,
    }


def select(runs: dict[str, dict]) -> str:
    reference = runs["token"]
    for mode, run in runs.items():
        if (
            run["parent_checkpoint_sha256"]
            != reference["parent_checkpoint_sha256"]
            or run["parent_prediction_manifest_sha256"]
            != reference["parent_prediction_manifest_sha256"]
        ):
            raise ValueError(f"G4R preflight parent differs for {mode}")
    maximum_confidence = max(run["confidence"] for run in runs.values())
    maximum_coverage = max(run["coverage"] for run in runs.values())
    eligible = {
        mode: run
        for mode, run in runs.items()
        if run["confidence"] >= 0.90 * maximum_confidence
        and run["coverage"] >= 0.90 * maximum_coverage
    }
    if not eligible:
        raise ValueError("All G4R preflight arms collapse confidence or coverage")
    return min(
        eligible,
        key=lambda mode: (eligible[mode]["score"], TIE_PRIORITY[mode]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-run", type=Path, required=True)
    parser.add_argument("--latent-run", type=Path, required=True)
    parser.add_argument("--query-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-seed", type=int, default=20260716)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    paths = {
        "token": args.token_run,
        "latent": args.latent_run,
        "query": args.query_run,
    }
    runs = {
        mode: load_run(path, mode, args.source_commit, args.required_seed)
        for mode, path in paths.items()
    }
    selected_mode = select(runs)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "required_seed": args.required_seed,
        "runs": runs,
        "selected_fusion_mode": selected_mode,
        "selected_arm": f"TR{4 + MODES.index(selected_mode)}",
        "selection_rule": (
            "minimum frozen score after 90% confidence and coverage retention"
        ),
        "completed": True,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
