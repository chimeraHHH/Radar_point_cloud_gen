#!/usr/bin/env python3
"""Freeze the T4-T6 fusion choice from matched five-epoch preflight runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL = "g4_temporal_preflight_v1"
ARMS = {
    "concat": "T4",
    "cross_attention": "T5",
    "draft_refinement": "T6",
}
CONFIG_EXCLUSIONS = {"fusion_mode"}
PROVENANCE_EXCLUSIONS = {
    "model_parameter_count",
    "relative_parameter_increase",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def selection_score(metrics: dict) -> float:
    return float(
        metrics["temporal"]["temporal_radial_error_mean_m"]["median"]
        + 0.25 * metrics["generated_geometry"]["chamfer_m"]["median"]
        + 0.25 * metrics["cycle"]["local_spectrum_kl"]["median"]
    )


def load_run(path: Path, expected_mode: str, required_seed: int) -> dict:
    path = path.resolve()
    config_path = path / "config.json"
    checkpoint_path = path / "best.pt"
    metrics_path = path / "best_validation_metrics.json"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    if config.get("fusion_mode") != expected_mode:
        raise ValueError(
            f"Expected {expected_mode}, found {config.get('fusion_mode')} in {path}"
        )
    expected_schedule = {
        "epochs": 5,
        "joint_start_epoch": 6,
        "seed": required_seed,
        "train_window_limit": None,
        "validation_window_limit": None,
    }
    differences = {
        key: (config.get(key), expected)
        for key, expected in expected_schedule.items()
        if config.get(key) != expected
    }
    if differences:
        raise ValueError(f"Invalid G4 preflight schedule in {path}: {differences}")
    if float(provenance["relative_parameter_increase"]) > 0.05:
        raise ValueError(f"G4 preflight arm exceeds the 5% parameter budget: {path}")
    metrics_document = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not 1 <= int(metrics_document["best_epoch"]) <= 5:
        raise ValueError(f"Invalid best checkpoint epoch in {path}")
    metrics = metrics_document["validation"]
    score = selection_score(metrics)
    reported_score = float(metrics_document["full_validation_selection_value"])
    if abs(score - reported_score) > 1e-12:
        raise ValueError(f"Full-validation selection score is inconsistent in {path}")
    frames = {
        (
            int(frame["sequence"]),
            int(frame["previous_frame_in_window"]),
            int(frame["current_frame_in_window"]),
        )
        for frame in metrics["frames"]
    }
    if len(frames) != int(metrics["pair_count"]):
        raise ValueError(f"Duplicate or missing validation pair identities in {path}")
    return {
        "arm": ARMS[expected_mode],
        "fusion_mode": expected_mode,
        "run_path": str(path),
        "config": config,
        "provenance": provenance,
        "config_sha256": sha256(config_path),
        "best_checkpoint_sha256": sha256(checkpoint_path),
        "best_validation_metrics_sha256": sha256(metrics_path),
        "best_epoch": int(metrics_document["best_epoch"]),
        "selection_value": score,
        "validation_pair_count": int(metrics["pair_count"]),
        "validation_pairs": frames,
    }


def validate_matched(runs: list[dict]) -> dict[str, bool]:
    reference = runs[0]
    reference_config = {
        key: value
        for key, value in reference["config"].items()
        if key not in CONFIG_EXCLUSIONS
    }
    reference_provenance = {
        key: value
        for key, value in reference["provenance"].items()
        if key not in PROVENANCE_EXCLUSIONS
    }
    checks = {
        "all_three_fusion_families_present": {run["fusion_mode"] for run in runs}
        == set(ARMS),
        "matched_configuration": all(
            {
                key: value
                for key, value in run["config"].items()
                if key not in CONFIG_EXCLUSIONS
            }
            == reference_config
            for run in runs
        ),
        "matched_data_parent_and_runtime_provenance": all(
            {
                key: value
                for key, value in run["provenance"].items()
                if key not in PROVENANCE_EXCLUSIONS
            }
            == reference_provenance
            for run in runs
        ),
        "matched_complete_validation_pairs": all(
            run["validation_pairs"] == reference["validation_pairs"] for run in runs
        ),
        "parameter_budget_respected": all(
            float(run["provenance"]["relative_parameter_increase"]) <= 0.05
            for run in runs
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"G4 preflight arms are not matched: {checks}")
    return checks


def markdown(report: dict) -> str:
    rows = [
        "# G4 Temporal Fusion Preflight Decision",
        "",
        f"Protocol: `{report['protocol']}`",
        "",
        "| Rank | Arm | Fusion | Full-validation score | Parameters added |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, run in enumerate(report["ranking"], start=1):
        rows.append(
            f"| {rank} | {run['arm']} | `{run['fusion_mode']}` | "
            f"{run['selection_value']:.8f} | "
            f"{100.0 * run['relative_parameter_increase']:.3f}% |"
        )
    rows.extend(
        [
            "",
            "## Frozen Selection",
            "",
            f"Selected **{report['selected_arm']} "
            f"(`{report['selected_fusion_mode']}`)** for the three-seed formal run.",
            "",
            "The choice minimizes the preregistered full-validation score. Exact "
            "score ties are resolved by lower parameter increase, then arm ID. This "
            "preflight is a model-family selection step and is not inferential evidence "
            "for G4.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concat-run", type=Path, required=True)
    parser.add_argument("--cross-attention-run", type=Path, required=True)
    parser.add_argument("--draft-refinement-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-markdown", type=Path, required=True)
    parser.add_argument("--required-seed", type=int, default=20260716)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.output, args.decision_markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}")
    runs = [
        load_run(args.concat_run, "concat", args.required_seed),
        load_run(args.cross_attention_run, "cross_attention", args.required_seed),
        load_run(args.draft_refinement_run, "draft_refinement", args.required_seed),
    ]
    checks = validate_matched(runs)
    checks["selector_source_matches_model_source"] = all(
        run["provenance"]["git_commit"] == args.source_commit for run in runs
    )
    if not checks["selector_source_matches_model_source"]:
        raise ValueError("G4 selector and preflight model source commits differ")
    ranking = sorted(
        runs,
        key=lambda run: (
            run["selection_value"],
            float(run["provenance"]["relative_parameter_increase"]),
            run["arm"],
        ),
    )
    selected = ranking[0]
    compact_ranking = [
        {
            "arm": run["arm"],
            "fusion_mode": run["fusion_mode"],
            "run_path": run["run_path"],
            "config_sha256": run["config_sha256"],
            "best_checkpoint_sha256": run["best_checkpoint_sha256"],
            "best_validation_metrics_sha256": run[
                "best_validation_metrics_sha256"
            ],
            "best_epoch": run["best_epoch"],
            "selection_value": run["selection_value"],
            "relative_parameter_increase": float(
                run["provenance"]["relative_parameter_increase"]
            ),
            "validation_pair_count": run["validation_pair_count"],
        }
        for run in ranking
    ]
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_commit": args.source_commit,
        "required_seed": args.required_seed,
        "selection_metric": "temporal_radial_error + 0.25 * current_chamfer + 0.25 * local_spectrum_kl",
        "selection_scope": "full frozen validation cohort",
        "tie_break": "lower parameter increase, then arm ID",
        "checks": checks,
        "ranking": compact_ranking,
        "selected_arm": selected["arm"],
        "selected_fusion_mode": selected["fusion_mode"],
        "selected_run_path": selected["run_path"],
        "completed": True,
    }
    atomic_text(args.output, json.dumps(report, indent=2) + "\n")
    atomic_text(args.decision_markdown, markdown(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
