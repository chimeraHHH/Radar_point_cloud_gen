"""Crash-safe checkpoint cadence helpers for large radar models."""

from __future__ import annotations

import json
from pathlib import Path


def checkpoint_due(
    epoch: int, total_epochs: int, checkpoint_every: int, evaluated: bool
) -> bool:
    if checkpoint_every <= 0:
        raise ValueError("Checkpoint cadence must be positive")
    return evaluated or epoch == total_epochs or epoch % checkpoint_every == 0


def truncate_resume_artifacts(output: Path, checkpoint_epoch: int) -> list[dict]:
    """Drop log/metric records newer than the last crash-safe checkpoint."""
    log_path = output / "train_log.jsonl"
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retained = [record for record in records if int(record["epoch"]) <= checkpoint_epoch]
    if not retained or int(retained[-1]["epoch"]) != checkpoint_epoch:
        raise ValueError("Training log does not contain the last checkpoint epoch")
    if len(retained) != len(records):
        temporary = log_path.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(record) + "\n" for record in retained),
            encoding="utf-8",
        )
        temporary.replace(log_path)
    for path in output.glob("metrics_epoch_*.json"):
        epoch = int(path.stem.rsplit("_", maxsplit=1)[1])
        if epoch > checkpoint_epoch:
            path.unlink()
    return retained
