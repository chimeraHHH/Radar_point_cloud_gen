#!/usr/bin/env python3
"""Queue the source-bound RaLD-native G4R experiment chain."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from g1b_contract import FROZEN_G1B_SEEDS, sha256
from gpu_runtime import validate_gpu_candidates
from queue_g4_temporal import (
    GPUJob,
    atomic_json,
    emit,
    run_cpu_command,
    run_gpu_jobs,
    wait_for_download_completion,
    wait_for_json,
)
from rald_gate_contract import validate_g3r_selected_runs


FUSION_MODES = ("token", "latent", "query")


def train_command(
    python: Path,
    args,
    parent: Path,
    parent_cache: Path,
    output: Path,
    fusion_mode: str,
    seed: int,
    epochs: int,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/train_rald_anchor_temporal.py"),
        "--data-root",
        str(args.data_root),
        "--cache-root",
        str(args.cache_root),
        "--manifest",
        str(args.manifest),
        "--scene-split",
        str(args.scene_split),
        "--normalization-stats",
        str(args.normalization),
        "--dense-cache-report",
        str(args.dense_cache_report),
        "--g3r-summary",
        str(args.g3r_summary),
        "--parent-run",
        str(parent),
        "--parent-prediction-cache",
        str(parent_cache),
        "--output",
        str(output),
        "--fusion-mode",
        fusion_mode,
        "--epochs",
        str(epochs),
        "--temporal-warmup-epochs",
        str(min(5, epochs)),
        "--seed",
        str(seed),
        "--eval-every",
        "5",
        "--max-eval-pairs",
        "32",
        "--device",
        "cuda:0",
        "--source-commit",
        args.source_commit,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-manifest-dir", type=Path, required=True)
    parser.add_argument("--download-verification", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--odometry-root", type=Path, required=True)
    parser.add_argument("--g0-report", type=Path, required=True)
    parser.add_argument("--g3r-summary", type=Path, required=True)
    parser.add_argument("--dense-cache-report", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--required-gpu-name", default="NVIDIA H200 NVL")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(FROZEN_G1B_SEEDS)
    )
    parser.add_argument("--required-frames", type=int, default=2160)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()
    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)
    if tuple(args.seeds) != FROZEN_G1B_SEEDS:
        raise ValueError("Formal G4R requires the frozen three-seed matrix")
    if args.required_frames != 2160:
        raise ValueError("Formal G4R requires the frozen 2160-frame cohort")
    python = Path(os.environ.get("PYTHON", "python"))
    args.run_root.mkdir(parents=True, exist_ok=True)
    tag = args.source_commit[:8]

    temporal_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_sequences = {
        int(frame["sequence"]) for frame in temporal_manifest["frames"]
    }
    if len(expected_sequences) != 45:
        raise ValueError("Formal G4R manifest requires exactly 45 sequences")
    download_summary = wait_for_download_completion(
        args.download_manifest_dir / "summary.json",
        expected_sequences,
        args.poll_seconds,
    )
    emit(
        "g4r_download_finished",
        completed_sequences=len(download_summary.get("completed_sequences", [])),
        failures=download_summary.get("failures", []),
    )
    verify_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/verify_kradar_g0_download.py"),
        "--audit-manifest",
        str(args.manifest),
        "--data-root",
        str(args.data_root),
        "--download-manifest-dir",
        str(args.download_manifest_dir),
        "--output",
        str(args.download_verification),
        "--workers",
        "8",
    ]
    if args.download_verification.exists():
        verify_command.append("--overwrite")
    verification = run_cpu_command(
        "g4r_download_verification",
        verify_command,
        args.download_verification,
        completion_field="passed",
    )
    if (
        verification.get("passed") is not True
        or int(verification["expected_frame_count"]) != args.required_frames
    ):
        raise SystemExit("G4R download verification failed")

    g0 = wait_for_json(args.g0_report, args.poll_seconds, "waiting_for_g0")
    if g0.get("aggregate", {}).get("gate_pass") is not True:
        raise SystemExit("G0 did not pass; G4R cannot build dense targets")
    g3r = wait_for_json(
        args.g3r_summary, args.poll_seconds, "waiting_for_g3r"
    )
    selected_runs = validate_g3r_selected_runs(
        g3r, args.source_commit, tuple(args.seeds)
    )
    emit(
        "g4r_dependencies_passed",
        selected_runs={str(seed): str(run) for seed, run in selected_runs.items()},
        g3r_summary_sha256=sha256(args.g3r_summary),
    )

    dense_job = GPUJob(
        name="g4r_dense_target_cache",
        command=[
            str(python),
            "-u",
            str(args.repo_root / "code/scripts/build_kradar_dense_cache.py"),
            "--data-root",
            str(args.data_root),
            "--cache-root",
            str(args.cache_root),
            "--manifest",
            str(args.manifest),
            "--scene-split",
            str(args.scene_split),
            "--odometry-root",
            str(args.odometry_root),
            "--g0-report",
            str(args.g0_report),
            "--lidar-time-reference",
            "none",
            "--output",
            str(args.dense_cache_report),
            "--device",
            "cuda:0",
            "--required-frames",
            str(args.required_frames),
            "--source-commit",
            args.source_commit,
        ],
        log_path=args.run_root / f"g4r_dense_cache_{tag}.log",
        marker=args.dense_cache_report,
        resume_evidence=args.dense_cache_report,
    )
    run_gpu_jobs([dense_job], args)

    parent_cache_paths = {
        seed: args.run_root / f"g4r_parent_cache_seed{seed}_{tag}"
        for seed in args.seeds
    }
    parent_jobs = []
    for seed in args.seeds:
        output = parent_cache_paths[seed]
        parent_jobs.append(
            GPUJob(
                name=f"g4r_parent_cache_seed{seed}",
                command=[
                    str(python),
                    "-u",
                    str(
                        args.repo_root
                        / "code/scripts/cache_rald_anchor_predictions.py"
                    ),
                    "--data-root",
                    str(args.data_root),
                    "--cache-root",
                    str(args.cache_root),
                    "--temporal-manifest",
                    str(args.manifest),
                    "--scene-split",
                    str(args.scene_split),
                    "--normalization-stats",
                    str(args.normalization),
                    "--g3r-summary",
                    str(args.g3r_summary),
                    "--g3r-source-commit",
                    args.source_commit,
                    "--cache-source-commit",
                    args.source_commit,
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                    "--device",
                    "cuda:0",
                ],
                log_path=args.run_root / f"g4r_parent_cache_seed{seed}_{tag}.log",
                marker=output / "manifest.json",
                resume_evidence=output / "manifest.json",
            )
        )
    run_gpu_jobs(parent_jobs, args)

    baseline_paths = {
        seed: args.run_root / f"g4r_baselines_seed{seed}_{tag}"
        for seed in args.seeds
    }
    baseline_jobs = []
    for seed in args.seeds:
        output = baseline_paths[seed]
        baseline_jobs.append(
            GPUJob(
                name=f"g4r_baselines_seed{seed}",
                command=[
                    str(python),
                    "-u",
                    str(
                        args.repo_root
                        / "code/scripts/eval_rald_anchor_g4r_baselines.py"
                    ),
                    "--data-root",
                    str(args.data_root),
                    "--cache-root",
                    str(args.cache_root),
                    "--manifest",
                    str(args.manifest),
                    "--scene-split",
                    str(args.scene_split),
                    "--normalization-stats",
                    str(args.normalization),
                    "--dense-cache-report",
                    str(args.dense_cache_report),
                    "--parent-prediction-cache",
                    str(parent_cache_paths[seed]),
                    "--output",
                    str(output),
                    "--device",
                    "cuda:0",
                    "--source-commit",
                    args.source_commit,
                ],
                log_path=args.run_root / f"g4r_baselines_seed{seed}_{tag}.log",
                marker=output / "report.json",
                resume_evidence=output,
            )
        )

    preflight_seed = min(args.seeds)
    preflight_paths = {
        mode: args.run_root / f"g4r_preflight_{mode}_seed{preflight_seed}_{tag}"
        for mode in FUSION_MODES
    }
    preflight_jobs = [
        GPUJob(
            name=f"g4r_preflight_{mode}",
            command=train_command(
                python,
                args,
                selected_runs[preflight_seed],
                parent_cache_paths[preflight_seed],
                output,
                mode,
                preflight_seed,
                5,
            ),
            log_path=args.run_root
            / f"g4r_preflight_{mode}_seed{preflight_seed}_{tag}.log",
            marker=output,
            completion_field=None,
            training_epochs=5,
            resume_evidence=output / "last.pt",
        )
        for mode, output in preflight_paths.items()
    ]
    run_gpu_jobs([*baseline_jobs, *preflight_jobs], args)

    selection_path = args.run_root / f"g4r_preflight_selection_{tag}.json"
    selection_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/select_rald_anchor_g4r_preflight.py"),
        "--token-run",
        str(preflight_paths["token"]),
        "--latent-run",
        str(preflight_paths["latent"]),
        "--query-run",
        str(preflight_paths["query"]),
        "--output",
        str(selection_path),
        "--required-seed",
        str(preflight_seed),
        "--source-commit",
        args.source_commit,
    ]
    if selection_path.exists():
        selection_command.append("--overwrite")
    selection = run_cpu_command(
        "g4r_preflight_selection",
        selection_command,
        selection_path,
        args.source_commit,
        "completed",
    )
    selected_mode = selection["selected_fusion_mode"]
    emit("g4r_preflight_selected", fusion_mode=selected_mode)

    formal_paths = {
        seed: args.run_root / f"g4r_{selected_mode}_seed{seed}_{tag}"
        for seed in args.seeds
    }
    formal_jobs = [
        GPUJob(
            name=f"g4r_formal_{selected_mode}_seed{seed}",
            command=train_command(
                python,
                args,
                selected_runs[seed],
                parent_cache_paths[seed],
                formal_paths[seed],
                selected_mode,
                seed,
                20,
            ),
            log_path=args.run_root
            / f"g4r_{selected_mode}_seed{seed}_{tag}.log",
            marker=formal_paths[seed],
            completion_field=None,
            training_epochs=20,
            resume_evidence=formal_paths[seed] / "last.pt",
        )
        for seed in args.seeds
    ]
    run_gpu_jobs(formal_jobs, args)

    rollout_paths = {
        seed: args.run_root / f"g4r_rollout_{selected_mode}_seed{seed}_{tag}"
        for seed in args.seeds
    }
    rollout_jobs = []
    for seed in args.seeds:
        output = rollout_paths[seed]
        rollout_jobs.append(
            GPUJob(
                name=f"g4r_rollout_seed{seed}",
                command=[
                    str(python),
                    "-u",
                    str(
                        args.repo_root
                        / "code/scripts/eval_rald_anchor_g4r_rollout.py"
                    ),
                    "--data-root",
                    str(args.data_root),
                    "--cache-root",
                    str(args.cache_root),
                    "--manifest",
                    str(args.manifest),
                    "--scene-split",
                    str(args.scene_split),
                    "--normalization-stats",
                    str(args.normalization),
                    "--dense-cache-report",
                    str(args.dense_cache_report),
                    "--parent-prediction-cache",
                    str(parent_cache_paths[seed]),
                    "--preflight-selection",
                    str(selection_path),
                    "--temporal-run",
                    str(formal_paths[seed]),
                    "--output",
                    str(output),
                    "--device",
                    "cuda:0",
                    "--source-commit",
                    args.source_commit,
                ],
                log_path=args.run_root / f"g4r_rollout_seed{seed}_{tag}.log",
                marker=output / "report.json",
                resume_evidence=output,
            )
        )
    run_gpu_jobs(rollout_jobs, args)

    comparison_path = args.run_root / f"g4r_comparison_{tag}.json"
    comparison_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/compare_rald_anchor_g4r.py"),
        "--baseline-reports",
        *[str(baseline_paths[seed] / "report.json") for seed in args.seeds],
        "--temporal-reports",
        *[str(rollout_paths[seed] / "report.json") for seed in args.seeds],
        "--output",
        str(comparison_path),
        "--source-commit",
        args.source_commit,
    ]
    if comparison_path.exists():
        comparison_command.append("--overwrite")
    comparison = run_cpu_command(
        "g4r_comparison",
        comparison_command,
        comparison_path,
        args.source_commit,
        "completed",
    )
    passed = comparison["decision"]["g4r_passed"] is True
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "g4r_passed" if passed else "g4r_failed",
        "source_commit": args.source_commit,
        "g3r_summary": str(args.g3r_summary),
        "g3r_summary_sha256": sha256(args.g3r_summary),
        "selected_fusion_mode": selected_mode,
        "preflight_selection": str(selection_path),
        "preflight_selection_sha256": sha256(selection_path),
        "formal_runs": {str(seed): str(path) for seed, path in formal_paths.items()},
        "formal_run_hashes": {
            str(seed): {
                "config_sha256": sha256(path / "config.json"),
                "best_checkpoint_sha256": sha256(path / "best.pt"),
            }
            for seed, path in formal_paths.items()
        },
        "baseline_reports": {
            str(seed): str(path / "report.json")
            for seed, path in baseline_paths.items()
        },
        "rollout_reports": {
            str(seed): str(path / "report.json")
            for seed, path in rollout_paths.items()
        },
        "comparison": str(comparison_path),
        "comparison_sha256": sha256(comparison_path),
        "decision": comparison["decision"],
        "completed": True,
    }
    summary_path = args.run_root / f"g4r_queue_summary_{tag}.json"
    atomic_json(summary_path, summary)
    emit("g4r_queue_complete", summary=str(summary_path), passed=passed)


if __name__ == "__main__":
    main()
