#!/usr/bin/env python3
"""Apply the preregistered one-frame matched RaLD AE overfit gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = {
    "maximum_chamfer_m": 5.0,
    "maximum_outlier_fraction_2m": 0.5,
    "minimum_fscore_1p0m": 0.2,
    "minimum_mean_output_confidence": 0.05,
    "minimum_train_loss_reduction_fraction": 0.30,
}


def verify(run: Path, required_epoch: int) -> dict:
    document = json.loads((run / "config.json").read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    records = [
        json.loads(line)
        for line in (run / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("RaLD AE overfit log is empty")
    best = json.loads(
        (run / "best_validation_metrics.json").read_text(encoding="utf-8")
    )
    frames = best["validation"]["frames"]
    if len(frames) != 1:
        raise ValueError("RaLD AE overfit must evaluate exactly one frame")
    geometry = best["validation"]["generated"]
    initial_loss = float(records[0]["train_loss_mean"])
    final_loss = float(records[-1]["train_loss_mean"])
    loss_reduction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    metrics = {
        "chamfer_m": float(geometry["chamfer_m"]["median"]),
        "outlier_fraction_2m": float(
            geometry["outlier_fraction_2m"]["mean"]
        ),
        "fscore_1p0m": float(geometry["fscore_1p0m"]["mean"]),
        "mean_output_confidence": float(frames[0]["mean_output_confidence"]),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "train_loss_reduction_fraction": loss_reduction,
    }
    checks = {
        "full_required_epoch": int(records[-1]["epoch"]) == required_epoch,
        "one_frame_configuration": config["overfit_one_frame"] is True
        and int(config["train_limit"]) == 1
        and int(config["validation_limit"]) == 1,
        "no_external_pretraining": provenance["external_pretraining"] is False,
        "chamfer": metrics["chamfer_m"] <= THRESHOLDS["maximum_chamfer_m"],
        "outlier": metrics["outlier_fraction_2m"]
        <= THRESHOLDS["maximum_outlier_fraction_2m"],
        "fscore": metrics["fscore_1p0m"]
        >= THRESHOLDS["minimum_fscore_1p0m"],
        "confidence": metrics["mean_output_confidence"]
        >= THRESHOLDS["minimum_mean_output_confidence"],
        "loss_reduction": metrics["train_loss_reduction_fraction"]
        >= THRESHOLDS["minimum_train_loss_reduction_fraction"],
    }
    return {
        "protocol": "matched RaLD AE one-frame overfit gate",
        "run": str(run),
        "source_commit": provenance["git_commit"],
        "required_epoch": required_epoch,
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-epoch", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    report = verify(args.run, args.required_epoch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
