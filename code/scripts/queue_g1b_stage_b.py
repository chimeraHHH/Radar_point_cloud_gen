#!/usr/bin/env python3
"""Run the explicitly recorded G1B Stage B after the one-seed screen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.g1b_contract import (  # noqa: E402
    FROZEN_G1B_SEEDS,
    sha256,
)
from scripts.gpu_runtime import validate_gpu_candidates  # noqa: E402
from scripts.queue_g1_formal import (  # noqa: E402
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


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for_screen(path: Path, poll_seconds: int) -> dict:
    while not path.exists():
        emit("waiting_for_g1b_stage_a", missing=str(path))
        time.sleep(poll_seconds)
    report = json.loads(path.read_text(encoding="utf-8"))
    if "stage_b_authorized" not in report:
        raise ValueError("G1B Stage A report lacks a final decision")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-repo-root", type=Path, required=True)
    parser.add_argument("--decision-repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--decision-source-commit", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260716, 20260717, 20260718]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=24)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--required-gpu-name")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()

    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)
    if tuple(args.seeds) != FROZEN_G1B_SEEDS:
        raise ValueError("G1B Stage B requires the frozen three seeds in order")
    if args.epochs != 50:
        raise ValueError("G1B Stage B requires the frozen 50-epoch budget")
    args.run_root.mkdir(parents=True, exist_ok=True)
    tag = args.source_commit[:8]
    decision_tag = args.decision_source_commit[:8]
    summary_path = args.run_root / f"g1b_stage_b_summary_{decision_tag}.json"
    if summary_path.exists():
        emit("g1b_stage_b_summary_exists", summary=str(summary_path))
        return
    screen = wait_for_screen(args.screen_report, args.poll_seconds)
    candidate = screen.get("selected_candidate")
    if screen["stage_b_authorized"] is not True or candidate is None:
        summary = {
            "status": "skipped_no_stage_a_survivor",
            "screen_report": str(args.screen_report),
            "screen_decision": {
                "stage_b_authorized": screen["stage_b_authorized"],
                "selected_candidate": candidate,
            },
        }
        atomic_json(summary_path, summary)
        emit("g1b_stage_b_skipped", summary=summary)
        return
    if candidate == "rae_max":
        raise ValueError("G1B Stage A cannot select its own baseline")

    launch_decision_path = (
        args.run_root / f"g1b_stage_b_launch_decision_{decision_tag}.json"
    )
    launch_decision = {
        "protocol": "G1B Stage B independent three-seed decision",
        "authorized": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_mode": candidate,
        "screen_report": str(args.screen_report),
        "screen_report_sha256": sha256(args.screen_report),
        "training_source_commit": args.source_commit,
        "decision_source_commit": args.decision_source_commit,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "thresholds_unchanged": True,
        "original_g1_reopened": False,
        "downstream_g2b_g3b_unlocked": False,
    }
    if launch_decision_path.exists():
        recorded = json.loads(launch_decision_path.read_text(encoding="utf-8"))
        stable_keys = (
            "protocol",
            "authorized",
            "candidate_mode",
            "screen_report",
            "screen_report_sha256",
            "training_source_commit",
            "decision_source_commit",
            "seeds",
            "epochs",
            "thresholds_unchanged",
            "original_g1_reopened",
            "downstream_g2b_g3b_unlocked",
        )
        if any(recorded.get(key) != launch_decision.get(key) for key in stable_keys):
            raise ValueError("Recorded G1B Stage B launch decision differs")
        launch_decision = recorded
    else:
        atomic_json(launch_decision_path, launch_decision)
    emit("g1b_stage_b_authorized", decision=launch_decision)

    jobs = [
        Job(
            mode=mode,
            seed=seed,
            run_path=args.run_root / f"g1b_stage_b_{mode}_seed{seed}_{tag}",
            log_path=args.run_root / f"g1b_stage_b_{mode}_seed{seed}_{tag}.log",
        )
        for mode in ("rae_max", candidate)
        for seed in args.seeds
    ]
    pending = [job for job in jobs if not completed_run(job.run_path, args.epochs)]
    running = []
    python = Path(os.environ.get("PYTHON", "python"))
    train_script = args.train_repo_root / "code/scripts/train_cube_occupancy.py"
    while pending or running:
        for active in running.copy():
            returncode = active.process.poll()
            if returncode is None:
                continue
            active.handle.close()
            running.remove(active)
            emit(
                "g1b_stage_b_run_finished",
                mode=active.job.mode,
                seed=active.job.seed,
                gpu=active.gpu,
                returncode=returncode,
            )
            if returncode == 0 and completed_run(active.job.run_path, args.epochs):
                continue
            if (
                resource_failure(returncode, active.job.log_path)
                and active.job.attempts <= args.maximum_resource_retries
            ):
                pending.append(active.job)
                continue
            for item in running:
                item.process.terminate()
                item.handle.close()
            atomic_json(
                summary_path,
                {
                    "status": "stage_b_process_failed",
                    "mode": active.job.mode,
                    "seed": active.job.seed,
                    "returncode": returncode,
                    "log_tail": tail_text(active.job.log_path),
                },
            )
            raise SystemExit(returncode or 4)

        assigned = {active.gpu for active in running}
        while pending:
            gpu, states = available_gpu(
                args.gpu_candidates, assigned, args.maximum_used_memory_mib
            )
            if gpu is None:
                if not running:
                    emit("waiting_for_g1b_stage_b_gpu", states=states)
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

    baseline_runs = [job for job in jobs if job.mode == "rae_max"]
    candidate_runs = [job for job in jobs if job.mode == candidate]
    comparison_path = args.run_root / f"g1b_stage_b_comparison_{decision_tag}.json"
    if not comparison_path.exists():
        command = [
            str(python),
            "-u",
            str(args.decision_repo_root / "code/scripts/compare_g1b_formal.py"),
            "--baseline-runs",
            *[str(job.run_path) for job in baseline_runs],
            "--candidate-runs",
            *[str(job.run_path) for job in candidate_runs],
            "--candidate-mode",
            candidate,
            "--screen-report",
            str(args.screen_report),
            "--launch-decision",
            str(launch_decision_path),
            "--output",
            str(comparison_path),
            "--required-seeds",
            str(len(args.seeds)),
        ]
        emit("g1b_stage_b_comparison_started", command=command)
        subprocess.run(command, check=True, cwd=args.decision_repo_root)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    passed = comparison.get("g1b_passed") is True
    summary = {
        "status": "g1b_passed" if passed else "g1b_failed",
        "candidate_mode": candidate,
        "training_source_commit": args.source_commit,
        "decision_source_commit": args.decision_source_commit,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "screen_report": str(args.screen_report),
        "screen_report_sha256": sha256(args.screen_report),
        "launch_decision": str(launch_decision_path),
        "launch_decision_sha256": sha256(launch_decision_path),
        "runs": [str(job.run_path) for job in jobs],
        "baseline_runs": [str(job.run_path) for job in baseline_runs],
        "candidate_runs": [str(job.run_path) for job in candidate_runs],
        "comparison": str(comparison_path),
        "comparison_sha256": sha256(comparison_path),
        "checks": comparison.get("checks"),
        "g2b_g3b_unlocked": False,
    }
    atomic_json(summary_path, summary)
    emit("g1b_stage_b_finished", summary=summary)


if __name__ == "__main__":
    main()
