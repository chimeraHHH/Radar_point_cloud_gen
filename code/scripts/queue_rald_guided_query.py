#!/usr/bin/env python3
"""Queue G1C Stage A and conditionally authorized three-seed Stage B."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_runtime import cuda_environment, validate_gpu_candidates
from scripts.g1b_contract import FROZEN_G1B_SEEDS, sha256
from scripts.queue_g1_formal import available_gpu, resource_failure, tail_text


@dataclass
class Job:
    seed: int
    run_path: Path
    log_path: Path
    attempts: int = 0


@dataclass
class RunningJob:
    job: Job
    gpu: int
    process: subprocess.Popen
    handle: object


def emit(event: str, **values) -> None:
    print(
        json.dumps(
            {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **values,
            }
        ),
        flush=True,
    )


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_run(job: Job, source_commit: str, expected_epochs: int) -> bool:
    config_path = job.run_path / "config.json"
    metrics_path = job.run_path / "best_validation_metrics.json"
    checkpoint_path = job.run_path / "best.pt"
    if not all(path.is_file() for path in (config_path, metrics_path, checkpoint_path)):
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return (
        config.get("provenance", {}).get("git_commit") == source_commit
        and int(config.get("config", {}).get("seed", -1)) == job.seed
        and int(config.get("config", {}).get("epochs", -1)) == expected_epochs
        and metrics.get("completed") is True
        and metrics.get("test_accessed") is False
        and metrics.get("best_checkpoint_sha256") == sha256(checkpoint_path)
    )


def prepare_run(job: Job) -> bool:
    if not job.run_path.exists() or not any(job.run_path.iterdir()):
        return False
    if (job.run_path / "config.json").is_file() and (job.run_path / "last.pt").is_file():
        return True
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = job.run_path.with_name(f"{job.run_path.name}.incomplete.{timestamp}")
    job.run_path.rename(archived)
    emit("g1c_incomplete_run_archived", source=str(job.run_path), destination=str(archived))
    return False


def train_command(
    job: Job, args, python: Path, resume: bool, *, smoke: bool
) -> list[str]:
    command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/train_rald_guided_query.py"),
        "--data-root",
        str(args.data_root),
        "--cache-root",
        str(args.cache_root),
        "--manifest",
        str(args.manifest),
        "--scene-split",
        str(args.scene_split),
        "--normalization",
        str(args.normalization),
        "--output",
        str(job.run_path),
        "--seed",
        str(job.seed),
        "--source-commit",
        args.source_commit,
        "--device",
        "cuda:0",
        "--eval-every",
        "5",
        "--max-eval-frames",
        "8",
    ]
    if resume:
        command.append("--resume")
    if smoke:
        command.extend(
            ("--smoke", "--train-limit", "2", "--validation-limit", "1")
        )
    return command


def launch(job: Job, gpu: int, command: list[str]) -> RunningJob:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8")
    job.attempts += 1
    emit(
        "g1c_run_started",
        seed=job.seed,
        gpu=gpu,
        attempt=job.attempts,
        log=str(job.log_path),
        command=command,
    )
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=cuda_environment(gpu),
    )
    return RunningJob(job, gpu, process, handle)


def run_jobs(
    jobs: list[Job], args, python: Path, *, smoke: bool = False
) -> None:
    expected_epochs = 1 if smoke else 30
    pending = [
        job
        for job in jobs
        if not completed_run(job, args.source_commit, expected_epochs)
    ]
    running: list[RunningJob] = []
    while pending or running:
        for active in running.copy():
            returncode = active.process.poll()
            if returncode is None:
                continue
            active.handle.close()
            running.remove(active)
            emit(
                "g1c_run_finished",
                seed=active.job.seed,
                gpu=active.gpu,
                returncode=returncode,
            )
            if returncode == 0 and completed_run(
                active.job, args.source_commit, expected_epochs
            ):
                continue
            if (
                resource_failure(returncode, active.job.log_path)
                and active.job.attempts <= args.maximum_resource_retries
            ):
                pending.append(active.job)
                emit("g1c_resource_retry_queued", seed=active.job.seed)
                continue
            for remaining in running:
                remaining.process.terminate()
                remaining.handle.close()
            raise RuntimeError(
                f"G1C seed {active.job.seed} failed: "
                f"{tail_text(active.job.log_path)}"
            )

        assigned = {active.gpu for active in running}
        while pending:
            gpu, states = available_gpu(
                args.gpu_candidates, assigned, args.maximum_used_memory_mib
            )
            if gpu is None:
                if not running:
                    emit("waiting_for_g1c_h200", states=states)
                break
            job = pending.pop(0)
            command = train_command(
                job, args, python, prepare_run(job), smoke=smoke
            )
            running.append(launch(job, gpu, command))
            assigned.add(gpu)
        if pending or running:
            time.sleep(args.poll_seconds)


def compare(stage: str, jobs: list[Job], args, python: Path, output: Path, stage_a: Path | None = None) -> dict:
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/compare_rald_guided_query.py"),
        "--stage",
        stage,
        "--runs",
        *(str(job.run_path) for job in jobs),
        "--output",
        str(output),
    ]
    if stage_a is not None:
        command.extend(("--stage-a-report", str(stage_a)))
    completed = subprocess.run(command, check=False, cwd=args.repo_root)
    if not output.is_file():
        raise RuntimeError(
            f"G1C {stage} comparison failed without a report: {completed.returncode}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--required-gpu-name", required=True)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def validate_preflight(job: Job, source_commit: str, output: Path) -> dict:
    if not completed_run(job, source_commit, 1):
        raise ValueError("G1C preflight run is incomplete")
    metrics_path = job.run_path / "best_validation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frames = metrics["validation"]["frames"]
    gradients = metrics["gradient_steps"]
    checks = {
        "two_optimizer_steps": len(gradients) == 2,
        "fixed_10000_points": bool(
            frames and frames[0]["generated"]["prediction_count"] == 10_000
        ),
        "full_raed_336_tokens": bool(
            frames and frames[0]["radar_token_count"] == 336
        ),
        "first_step_physical_gradient": bool(
            gradients and gradients[0]["gradients"]["physical_head"] > 0.0
        ),
        "second_step_mixed_latent_gradient": bool(
            len(gradients) == 2
            and gradients[1]["gradients"]["mixed_latent_and_query_decoder"] > 0.0
        ),
        "second_step_full_raed_gradient": bool(
            len(gradients) == 2
            and gradients[1]["gradients"]["full_raed_radar_encoder"] > 0.0
        ),
        "second_step_local_spectrum_gradient": bool(
            len(gradients) == 2
            and gradients[1]["gradients"]["local_64bin_spectrum_projection"] > 0.0
        ),
    }
    report = {
        "protocol": "g1c_rald_guided_query_preflight_v1",
        "source_commit": source_commit,
        "run": str(job.run_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "checks": checks,
        "passed": all(checks.values()),
        "test_accessed": False,
    }
    atomic_json(output, report)
    if not report["passed"]:
        raise ValueError(f"G1C preflight failed: {checks}")
    return report


def main() -> None:
    args = parse_args()
    if tuple(args.gpu_candidates) != (0, 2):
        raise ValueError("G1C may use only physical H200 GPUs 0 and 2")
    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)
    args.run_root.mkdir(parents=True, exist_ok=True)
    python = Path(os.environ.get("PYTHON", "python"))
    tag = args.source_commit[:8]
    jobs = {
        seed: Job(
            seed,
            args.run_root / f"g1c_seed{seed}_{tag}",
            args.run_root / f"g1c_seed{seed}_{tag}.log",
        )
        for seed in FROZEN_G1B_SEEDS
    }
    summary_path = args.run_root / f"g1c_summary_{tag}.json"
    if summary_path.is_file():
        emit("g1c_summary_exists", summary=str(summary_path))
        return

    preflight_job = Job(
        FROZEN_G1B_SEEDS[0],
        args.run_root / f"g1c_preflight_{tag}",
        args.run_root / f"g1c_preflight_{tag}.log",
    )
    preflight_path = args.run_root / f"g1c_preflight_{tag}.json"
    if not preflight_path.is_file():
        run_jobs([preflight_job], args, python, smoke=True)
        validate_preflight(preflight_job, args.source_commit, preflight_path)
    else:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("passed") is not True:
            raise ValueError("Recorded G1C preflight did not pass")

    seed_a = FROZEN_G1B_SEEDS[0]
    run_jobs([jobs[seed_a]], args, python)
    stage_a_path = args.run_root / f"g1c_stage_a_{tag}.json"
    stage_a = compare("stage_a", [jobs[seed_a]], args, python, stage_a_path)
    if stage_a.get("decision", {}).get("passed") is not True:
        summary = {
            "status": "g1c_failed_stage_a",
            "source_commit": args.source_commit,
            "stage_a": str(stage_a_path),
            "stage_a_sha256": sha256(stage_a_path),
            "stage_b_started": False,
            "preflight": str(preflight_path),
            "preflight_sha256": sha256(preflight_path),
            "test_accessed": False,
        }
        atomic_json(summary_path, summary)
        emit("g1c_closed_after_stage_a", summary=summary)
        return

    run_jobs([jobs[seed] for seed in FROZEN_G1B_SEEDS[1:]], args, python)
    stage_b_path = args.run_root / f"g1c_stage_b_{tag}.json"
    stage_b = compare(
        "stage_b",
        list(jobs.values()),
        args,
        python,
        stage_b_path,
        stage_a=stage_a_path,
    )
    passed = stage_b.get("decision", {}).get("passed") is True
    summary = {
        "status": "g1c_passed" if passed else "g1c_failed_stage_b",
        "source_commit": args.source_commit,
        "preflight": str(preflight_path),
        "preflight_sha256": sha256(preflight_path),
        "stage_a": str(stage_a_path),
        "stage_a_sha256": sha256(stage_a_path),
        "stage_b": str(stage_b_path),
        "stage_b_sha256": sha256(stage_b_path),
        "runs": {str(seed): str(job.run_path) for seed, job in jobs.items()},
        "run_hashes": stage_b.get("run_hashes"),
        "test_accessed": False,
        "successors_unlocked": passed,
    }
    atomic_json(summary_path, summary)
    emit("g1c_finished", summary=summary)


if __name__ == "__main__":
    main()
