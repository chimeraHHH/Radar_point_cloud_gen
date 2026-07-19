"""Immutable authorization contract for G1/G1B geometry parents."""

from __future__ import annotations

import hashlib
from pathlib import Path


FROZEN_G1B_SEEDS = (20260716, 20260717, 20260718)
FROZEN_G1B_EPOCHS = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_original_parent(decision: dict) -> tuple[str, str] | None:
    g1_passed = decision.get("g1_passed")
    if not isinstance(g1_passed, bool):
        raise ValueError("Formal G1 decision must contain boolean g1_passed")
    if g1_passed:
        return "full_raed", "formal_g1_passed"

    rae_max_beats_cfar = decision.get("rae_max_beats_cfar")
    if not isinstance(rae_max_beats_cfar, bool):
        raise ValueError(
            "Failed G1 decision must contain boolean rae_max_beats_cfar"
        )
    if rae_max_beats_cfar:
        return "rae_max", "late_fusion_recovery_after_g1_failure"
    return None


def validate_g1b_summary(
    summary: dict,
    training_source_commit: str,
    decision_source_commit: str,
    run_root: Path,
    required_seeds: tuple[int, ...] = FROZEN_G1B_SEEDS,
) -> tuple[str, dict[int, Path]]:
    if tuple(required_seeds) != FROZEN_G1B_SEEDS:
        raise ValueError("G1B consumers require the frozen three-seed protocol")
    if summary.get("status") != "g1b_passed" or not summary.get("candidate_mode"):
        raise ValueError("G1B did not authorize an independent geometry parent")
    if summary.get("training_source_commit") != training_source_commit:
        raise ValueError("G1B training source commit differs from the contract")
    if summary.get("decision_source_commit") != decision_source_commit:
        raise ValueError("G1B decision source commit differs from the contract")
    if tuple(summary.get("seeds", [])) != FROZEN_G1B_SEEDS:
        raise ValueError("G1B summary does not use the frozen three seeds")
    if summary.get("epochs") != FROZEN_G1B_EPOCHS:
        raise ValueError("G1B summary does not use the frozen epoch budget")

    for path_key, hash_key in (
        ("screen_report", "screen_report_sha256"),
        ("launch_decision", "launch_decision_sha256"),
        ("comparison", "comparison_sha256"),
    ):
        path_value = summary.get(path_key)
        expected_hash = summary.get(hash_key)
        if not path_value or not expected_hash:
            raise ValueError(f"G1B summary lacks {path_key} provenance")
        path = Path(path_value)
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"G1B {path_key} provenance does not match")

    mode = str(summary["candidate_mode"])
    if mode == "rae_max":
        raise ValueError("G1B candidate cannot be its own RAE-Max baseline")
    tag = training_source_commit[:8]
    parents = {
        seed: run_root / f"g1b_stage_b_{mode}_seed{seed}_{tag}"
        for seed in FROZEN_G1B_SEEDS
    }
    baselines = {
        run_root / f"g1b_stage_b_rae_max_seed{seed}_{tag}"
        for seed in FROZEN_G1B_SEEDS
    }
    candidate_runs = {Path(path) for path in summary.get("candidate_runs", [])}
    baseline_runs = {Path(path) for path in summary.get("baseline_runs", [])}
    all_runs = {Path(path) for path in summary.get("runs", [])}
    if candidate_runs != set(parents.values()):
        raise ValueError("G1B summary does not authorize the exact candidate parents")
    if baseline_runs != baselines:
        raise ValueError("G1B summary does not record the exact baseline runs")
    if all_runs != candidate_runs | baseline_runs:
        raise ValueError("G1B summary run set differs from its formal arms")
    return mode, parents
