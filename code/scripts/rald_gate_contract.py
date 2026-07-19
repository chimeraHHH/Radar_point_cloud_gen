"""Strict contracts for downstream use of passing RaLD gate summaries."""

from __future__ import annotations

import json
from pathlib import Path

from g1b_contract import FROZEN_G1B_SEEDS, sha256


def validate_g3r_selected_runs(
    summary: dict,
    source_commit: str,
    seeds: tuple[int, ...] = FROZEN_G1B_SEEDS,
) -> dict[int, Path]:
    if summary.get("status") != "g3r_passed":
        raise ValueError("Downstream RaLD work requires a passing G3R summary")
    if summary.get("source_commit") != source_commit:
        raise ValueError("G3R summary source commit differs")
    if tuple(summary.get("seeds", ())) != seeds:
        raise ValueError("G3R summary seed matrix differs")
    if summary.get("selected_arm") != "full":
        raise ValueError("G3R downstream parent must be the full-cycle arm")
    comparison_path = Path(summary["g3r_comparison"]).resolve()
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    if sha256(comparison_path) != summary.get("g3r_comparison_sha256"):
        raise ValueError("G3R comparison hash differs")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("decision", {}).get("g3r_passed") is not True:
        raise ValueError("G3R comparison decision is not passing")
    selected_runs = summary.get("selected_runs", {})
    selected_hashes = summary.get("selected_run_hashes", {})
    expected_keys = {str(seed) for seed in seeds}
    if set(selected_runs) != expected_keys or set(selected_hashes) != expected_keys:
        raise ValueError("G3R selected run matrix is incomplete")
    if comparison.get("runs", {}).get("full") != selected_runs:
        raise ValueError("G3R summary and comparison selected runs differ")
    if comparison.get("run_hashes", {}).get("full") != selected_hashes:
        raise ValueError("G3R summary and comparison selected hashes differ")
    validated = {}
    for seed in seeds:
        key = str(seed)
        run = Path(selected_runs[key]).resolve()
        config_path = run / "config.json"
        checkpoint_path = run / "best.pt"
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Incomplete selected G3R run: {run}")
        hashes = selected_hashes[key]
        if (
            sha256(config_path) != hashes.get("config_sha256")
            or sha256(checkpoint_path) != hashes.get("best_checkpoint_sha256")
        ):
            raise ValueError(f"Selected G3R artifacts changed for seed {seed}")
        document = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            int(document["config"]["seed"]) != seed
            or document["config"]["cycle_variant"] != "full"
            or document["config"]["doppler_head_mode"] != "distribution"
            or document["provenance"]["git_commit"] != source_commit
        ):
            raise ValueError(f"Selected G3R run contract differs for seed {seed}")
        validated[seed] = run
    return validated

