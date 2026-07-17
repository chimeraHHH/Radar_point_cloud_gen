#!/usr/bin/env python3
"""Queue the preregistered six-run G1 experiment after preflight passes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


RESOURCE_FAILURE_MARKERS = (
    "out of memory",
    "cuda error",
    "cuda driver error",
    "device is busy",
    "device-side assert",
    "cublas_status_alloc_failed",
)


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


def gpu_state(index: int) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, utilization = output.split(",")
    return int(memory.strip()), int(utilization.strip())


def available_gpu(
    candidates: list[int], assigned: set[int], maximum_used_memory_mib: int
) -> tuple[int | None, dict]:
    states = {}
    for index in candidates:
        if index in assigned:
            states[str(index)] = {"assigned": True}
            continue
        try:
            memory, utilization = gpu_state(index)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            states[str(index)] = {"error": str(error)}
            continue
        states[str(index)] = {
            "memory_mib": memory,
            "utilization_percent": utilization,
        }
        if memory <= maximum_used_memory_mib and utilization < 5:
            return index, states
    return None, states


def wait_for_reports(paths: list[Path], poll_seconds: int) -> None:
    while True:
        missing = [str(path) for path in paths if not path.exists()]
        if not missing:
            break
        emit("waiting_for_g1_preflight", missing=missing)
        time.sleep(poll_seconds)
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            emit("g1_preflight_failed", report=str(path), checks=report.get("checks"))
            raise SystemExit(3)
    emit("g1_preflight_passed", reports=[str(path) for path in paths])


def completed_run(run_path: Path, epochs: int) -> bool:
    metrics = run_path / "best_validation_metrics.json"
    log = run_path / "train_log.jsonl"
    if not metrics.exists() or not log.exists():
        return False
    records = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return bool(records) and int(records[-1]["epoch"]) == epochs


def tail_text(path: Path, maximum_bytes: int = 64_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace")


def resource_failure(return_code: int, log_path: Path) -> bool:
    if return_code < 0:
        return True
    log = tail_text(log_path).lower()
    return any(marker in log for marker in RESOURCE_FAILURE_MARKERS)


@dataclass
class Job:
    mode: str
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


def prepare_run(job: Job) -> bool:
    if not job.run_path.exists() or not any(job.run_path.iterdir()):
        return False
    if (job.run_path / "config.json").exists() and (
        job.run_path / "last.pt"
    ).exists():
        return True
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = job.run_path.with_name(f"{job.run_path.name}.incomplete.{timestamp}")
    job.run_path.rename(archived)
    emit(
        "incomplete_run_archived",
        mode=job.mode,
        seed=job.seed,
        source=str(job.run_path),
        destination=str(archived),
    )
    return False


def train_command(
    job: Job,
    python: Path,
    train_script: Path,
    args,
    normalization: Path,
    resume: bool,
) -> list[str]:
    command = [
        str(python),
        "-u",
        str(train_script),
        "--data-root",
        str(args.data_root),
        "--cache-root",
        str(args.cache_root),
        "--manifest",
        str(args.manifest),
        "--scene-split",
        str(args.scene_split),
        "--output",
        str(job.run_path),
        "--mode",
        job.mode,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(job.seed),
        "--eval-every",
        str(args.eval_every),
        "--max-eval-frames",
        str(args.max_eval_frames),
        "--normalization-stats",
        str(normalization),
        "--device",
        "cuda:0",
        "--source-commit",
        args.source_commit,
    ]
    if resume:
        command.append("--resume")
    return command


def launch_job(
    job: Job,
    gpu: int,
    command: list[str],
) -> RunningJob:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8")
    job.attempts += 1
    emit(
        "g1_run_started",
        mode=job.mode,
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
        env=environment,
    )
    return RunningJob(job=job, gpu=gpu, process=process, handle=handle)


def run_comparison(
    jobs: list[Job], python: Path, compare_script: Path, report_path: Path
) -> dict:
    if report_path.exists():
        emit("g1_comparison_exists", report=str(report_path))
        return json.loads(report_path.read_text(encoding="utf-8"))
    rae_runs = [str(job.run_path) for job in jobs if job.mode == "rae_max"]
    full_runs = [str(job.run_path) for job in jobs if job.mode == "full_raed"]
    command = [
        str(python),
        "-u",
        str(compare_script),
        "--rae-max-runs",
        *rae_runs,
        "--full-raed-runs",
        *full_runs,
        "--output",
        str(report_path),
    ]
    emit("g1_comparison_started", command=command)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    emit("g1_comparison_finished", decision=report["decision"])
    return report


def run_qualitative(
    jobs: list[Job], python: Path, qualitative_script: Path, args
) -> Path:
    seed = min(args.seeds)
    rae_run = next(
        job.run_path for job in jobs if job.mode == "rae_max" and job.seed == seed
    )
    full_run = next(
        job.run_path for job in jobs if job.mode == "full_raed" and job.seed == seed
    )
    tag = args.source_commit[:8]
    output = args.run_root / f"g1_qualitative_{tag}"
    report = output / "qualitative_report.json"
    if report.exists():
        emit("g1_qualitative_exists", report=str(report))
        return report
    if output.exists() and any(output.iterdir()):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = output.with_name(f"{output.name}.incomplete.{timestamp}")
        output.rename(archived)
        emit(
            "g1_qualitative_incomplete_archived",
            source=str(output),
            destination=str(archived),
        )
    command = [
        str(python),
        "-u",
        str(qualitative_script),
        "--data-root",
        str(args.data_root),
        "--cache-root",
        str(args.cache_root),
        "--manifest",
        str(args.manifest),
        "--normalization-stats",
        str(args.normalization),
        "--rae-max-run",
        str(rae_run),
        "--full-raed-run",
        str(full_run),
        "--output",
        str(output),
        "--device",
        "cuda:0",
    ]
    log_path = args.run_root / f"g1_qualitative_{tag}.log"
    while True:
        gpu, states = available_gpu(
            args.gpu_candidates, set(), args.maximum_used_memory_mib
        )
        if gpu is None:
            emit("waiting_for_qualitative_gpu", states=states)
            time.sleep(args.poll_seconds)
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        emit("g1_qualitative_started", gpu=gpu, command=command, log=str(log_path))
        with log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        emit("g1_qualitative_finished", return_code=completed.returncode)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        if not report.exists():
            raise RuntimeError("Qualitative command succeeded without a report")
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--preflight-source-commit", required=True)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260716, 20260717, 20260718])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=24)
    parser.add_argument("--required-train-frames", type=int, default=76)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()

    python = Path(os.environ.get("PYTHON", "python"))
    train_script = args.repo_root / "code/scripts/train_cube_occupancy.py"
    compare_script = args.repo_root / "code/scripts/compare_g1_cube_occupancy.py"
    qualitative_script = args.repo_root / "code/scripts/render_g1_qualitative.py"
    preflight_tag = args.preflight_source_commit[:8]
    preflight_reports = [
        args.run_root
        / f"g1_overfit_{mode}_{preflight_tag}"
        / "overfit_verification.json"
        for mode in ("rae_max", "full_raed")
    ]
    wait_for_reports(preflight_reports, args.poll_seconds)

    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    if (
        normalization["partitions"] != ["train"]
        or normalization["frame_limit"] is not None
        or int(normalization["frame_count"]) != args.required_train_frames
    ):
        raise ValueError("Formal G1 requires complete train-only normalization")

    tag = args.source_commit[:8]
    jobs = [
        Job(
            mode=mode,
            seed=seed,
            run_path=args.run_root / f"g1_{mode}_seed{seed}_{tag}",
            log_path=args.run_root / f"g1_{mode}_seed{seed}_{tag}.log",
        )
        for mode in ("rae_max", "full_raed")
        for seed in args.seeds
    ]
    pending = [job for job in jobs if not completed_run(job.run_path, args.epochs)]
    for job in jobs:
        if job not in pending:
            emit("g1_run_already_complete", mode=job.mode, seed=job.seed)
    running: list[RunningJob] = []

    while pending or running:
        for active in running.copy():
            return_code = active.process.poll()
            if return_code is None:
                continue
            active.handle.close()
            running.remove(active)
            emit(
                "g1_run_finished",
                mode=active.job.mode,
                seed=active.job.seed,
                gpu=active.gpu,
                return_code=return_code,
            )
            if return_code == 0 and completed_run(active.job.run_path, args.epochs):
                continue
            if (
                resource_failure(return_code, active.job.log_path)
                and active.job.attempts <= args.maximum_resource_retries
            ):
                emit(
                    "g1_run_resource_retry_queued",
                    mode=active.job.mode,
                    seed=active.job.seed,
                    attempts=active.job.attempts,
                )
                pending.append(active.job)
                continue
            emit(
                "g1_run_failed",
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
                    emit("waiting_for_gpu", states=states)
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

    comparison_path = args.run_root / f"g1_comparison_{tag}.json"
    report = run_comparison(jobs, python, compare_script, comparison_path)
    qualitative_report = run_qualitative(jobs, python, qualitative_script, args)
    emit("g1_qualitative_complete", report=str(qualitative_report))
    if report["decision"]["g1_passed"] is not True:
        raise SystemExit(3)
    emit("g1_formal_complete", report=str(comparison_path))


if __name__ == "__main__":
    main()
