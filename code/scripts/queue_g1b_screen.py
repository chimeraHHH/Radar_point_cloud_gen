#!/usr/bin/env python3
"""Run the independent G1B screen only after the original G1 closes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from gpu_runtime import validate_gpu_candidates

try:
    from compare_g1b_screen import MODES
    from queue_g1_formal import (
        Job,
        available_gpu,
        completed_run,
        emit,
        launch_job,
        prepare_run,
        resource_failure,
        tail_text,
        train_command,
    )
except ModuleNotFoundError:  # Imported as scripts.queue_g1b_screen in tests.
    from scripts.compare_g1b_screen import MODES
    from scripts.queue_g1_formal import (
        Job,
        available_gpu,
        completed_run,
        emit,
        launch_job,
        prepare_run,
        resource_failure,
        tail_text,
        train_command,
    )


def wait_for_failed_g1(report_path: Path, poll_seconds: int) -> dict:
    while not report_path.exists():
        emit("waiting_for_original_g1_decision", missing=str(report_path))
        time.sleep(poll_seconds)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    passed = report.get("decision", {}).get("g1_passed")
    if passed is True:
        emit("g1b_not_authorized", reason="original_g1_passed")
        raise SystemExit(0)
    if passed is not False:
        raise ValueError("Original G1 report lacks a final boolean decision")
    emit("g1b_authorized", reason="original_g1_failed")
    return report


def run_screen(jobs: list[Job], python: Path, script: Path, output: Path) -> dict:
    if output.exists():
        emit("g1b_screen_exists", report=str(output))
        return json.loads(output.read_text(encoding="utf-8"))
    command = [
        str(python),
        "-u",
        str(script),
        "--runs",
        *(str(job.run_path) for job in jobs),
        "--output",
        str(output),
        "--required-seed",
        str(jobs[0].seed),
    ]
    emit("g1b_screen_comparison_started", command=command)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    report = json.loads(output.read_text(encoding="utf-8"))
    emit(
        "g1b_screen_comparison_finished",
        survivors=report["survivors"],
        selected_candidate=report["selected_candidate"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-g1-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--required-gpu-name")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=24)
    parser.add_argument("--required-train-frames", type=int, default=76)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()

    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)
    wait_for_failed_g1(args.original_g1_report, args.poll_seconds)
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    if (
        normalization["partitions"] != ["train"]
        or normalization["frame_limit"] is not None
        or int(normalization["frame_count"]) != args.required_train_frames
    ):
        raise ValueError("G1B requires complete train-only normalization")

    python = Path(os.environ.get("PYTHON", "python"))
    train_script = args.repo_root / "code/scripts/train_cube_occupancy.py"
    compare_script = args.repo_root / "code/scripts/compare_g1b_screen.py"
    tag = args.source_commit[:8]
    jobs = [
        Job(
            mode=mode,
            seed=args.seed,
            run_path=args.run_root / f"g1b_{mode}_seed{args.seed}_{tag}",
            log_path=args.run_root / f"g1b_{mode}_seed{args.seed}_{tag}.log",
        )
        for mode in MODES
    ]
    pending = [job for job in jobs if not completed_run(job.run_path, args.epochs)]
    for job in jobs:
        if job not in pending:
            emit("g1b_run_already_complete", mode=job.mode, seed=job.seed)
    running = []

    while pending or running:
        for active in running.copy():
            return_code = active.process.poll()
            if return_code is None:
                continue
            active.handle.close()
            running.remove(active)
            emit(
                "g1b_run_finished",
                mode=active.job.mode,
                seed=active.job.seed,
                gpu=active.gpu,
                return_code=return_code,
            )
            if return_code == 0 and completed_run(
                active.job.run_path, args.epochs
            ):
                continue
            if (
                resource_failure(return_code, active.job.log_path)
                and active.job.attempts <= args.maximum_resource_retries
            ):
                emit(
                    "g1b_run_resource_retry_queued",
                    mode=active.job.mode,
                    seed=active.job.seed,
                    attempts=active.job.attempts,
                )
                pending.append(active.job)
                continue
            emit(
                "g1b_run_failed",
                mode=active.job.mode,
                seed=active.job.seed,
                return_code=return_code,
                log_tail=tail_text(active.job.log_path),
            )
            raise SystemExit(return_code or 4)

        assigned = {active.gpu for active in running}
        while pending:
            gpu, states = available_gpu(
                args.gpu_candidates, assigned, args.maximum_used_memory_mib
            )
            if gpu is None:
                if not running:
                    emit("waiting_for_g1b_gpu", states=states)
                break
            job = pending.pop(0)
            resume = prepare_run(job)
            command = train_command(
                job, python, train_script, args, args.normalization, resume
            )
            running.append(launch_job(job, gpu, command))
            assigned.add(gpu)
        if pending or running:
            time.sleep(args.poll_seconds)

    report_path = args.run_root / f"g1b_screen_{tag}.json"
    report = run_screen(jobs, python, compare_script, report_path)
    if report["stage_b_authorized"] is not True:
        emit("g1b_stage_b_not_authorized", report=str(report_path))
        raise SystemExit(3)
    emit(
        "g1b_stage_b_candidate_frozen",
        candidate=report["selected_candidate"],
        report=str(report_path),
    )


if __name__ == "__main__":
    main()
