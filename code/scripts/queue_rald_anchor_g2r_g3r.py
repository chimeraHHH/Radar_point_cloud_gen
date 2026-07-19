#!/usr/bin/env python3
"""Queue the independent RaLD-anchor G2R and G3R decision chain."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from g1b_contract import (
    FROZEN_G1B_SEEDS,
    sha256,
    validate_g1b_summary,
)
from gpu_runtime import cuda_environment, validate_gpu_candidates


RESOURCE_FAILURE_MARKERS = (
    "out of memory",
    "cuda error",
    "device is busy",
    "cublas_status_alloc_failed",
)
G2R_HEADS = ("scalar", "distribution")
G3R_VARIANTS = ("none", "local_peak", "marginal", "full")


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


def wait_for_json(path: Path, poll_seconds: int, event: str) -> dict:
    while not path.exists():
        emit(event, missing=str(path))
        time.sleep(poll_seconds)
    return json.loads(path.read_text(encoding="utf-8"))


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
            states[str(index)] = {
                "memory_mib": memory,
                "utilization_percent": utilization,
            }
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            states[str(index)] = {"error": str(error)}
            continue
        if memory <= maximum_used_memory_mib and utilization < 5:
            return index, states
    return None, states


@dataclass
class Job:
    stage: str
    seed: int
    arm: str
    parent: Path
    run: Path
    log: Path
    initial_refiner: Path | None = None
    attempts: int = 0


@dataclass
class RunningJob:
    job: Job
    gpu: int
    process: subprocess.Popen
    handle: object


def resource_failure(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    with log_path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 64_000))
        tail = handle.read().decode("utf-8", errors="replace").lower()
    return any(marker in tail for marker in RESOURCE_FAILURE_MARKERS)


def completed(job: Job, args) -> bool:
    config_path = job.run / "config.json"
    checkpoint_path = job.run / "best.pt"
    metrics_path = job.run / "best_validation_metrics.json"
    log_path = job.run / "train_log.jsonl"
    if not all(
        path.is_file()
        for path in (config_path, checkpoint_path, metrics_path, log_path)
    ):
        return False
    document = json.loads(config_path.read_text(encoding="utf-8"))
    config = document["config"]
    provenance = document["provenance"]
    expected_epochs = args.g2r_epochs if job.stage == "g2r" else args.g3r_epochs
    expected_head = job.arm if job.stage == "g2r" else "distribution"
    expected_cycle = "none" if job.stage == "g2r" else job.arm
    expected_warmup = args.g2r_warmup_epochs if job.stage == "g2r" else 0
    if (
        provenance["git_commit"] != args.source_commit
        or int(config["seed"]) != job.seed
        or int(config["epochs"]) != expected_epochs
        or config["doppler_head_mode"] != expected_head
        or config["cycle_variant"] != expected_cycle
        or int(config["physical_head_warmup_epochs"]) != expected_warmup
        or provenance["parent_g1_checkpoint"] != str(job.parent / "best.pt")
        or provenance["parent_g1_checkpoint_sha256"]
        != sha256(job.parent / "best.pt")
    ):
        return False
    if job.initial_refiner is None:
        if config.get("initial_refiner_run") is not None:
            return False
    elif (
        config.get("initial_refiner_run") != str(job.initial_refiner)
        or provenance.get("initial_refiner_checkpoint_sha256")
        != sha256(job.initial_refiner / "best.pt")
        or provenance.get("initial_refiner_source_commit") != args.source_commit
    ):
        return False
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return bool(records) and int(records[-1]["epoch"]) == expected_epochs


def train_command(job: Job, args) -> list[str]:
    epochs = args.g2r_epochs if job.stage == "g2r" else args.g3r_epochs
    head = job.arm if job.stage == "g2r" else "distribution"
    cycle = "none" if job.stage == "g2r" else job.arm
    warmup = args.g2r_warmup_epochs if job.stage == "g2r" else 0
    command = [
        str(args.python),
        "-u",
        str(args.repo_root / "code/scripts/train_rald_anchor_refiner.py"),
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
        "--g1-comparison",
        str(args.g1_comparison),
        "--g1-source-commit",
        args.g1_source_commit,
        "--g1b-summary",
        str(args.g1b_summary),
        "--g1b-training-source-commit",
        args.g1b_training_source_commit,
        "--g1b-decision-source-commit",
        args.g1b_decision_source_commit,
        "--parent-g1-run",
        str(job.parent),
        "--output",
        str(job.run),
        "--epochs",
        str(epochs),
        "--seed",
        str(job.seed),
        "--doppler-head-mode",
        head,
        "--cycle-variant",
        cycle,
        "--physical-head-warmup-epochs",
        str(warmup),
        "--eval-every",
        str(args.eval_every),
        "--max-eval-frames",
        str(args.max_eval_frames),
        "--device",
        "cuda:0",
        "--source-commit",
        args.source_commit,
    ]
    if job.initial_refiner is not None:
        command.extend(
            [
                "--initial-refiner-run",
                str(job.initial_refiner),
                "--initial-refiner-source-commit",
                args.source_commit,
            ]
        )
    if (job.run / "last.pt").exists():
        command.append("--resume")
    return command


def launch(job: Job, gpu: int, args) -> RunningJob:
    job.log.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log.open("a", encoding="utf-8")
    command = train_command(job, args)
    job.attempts += 1
    emit(
        "rald_gate_run_started",
        stage=job.stage,
        arm=job.arm,
        seed=job.seed,
        gpu=gpu,
        attempt=job.attempts,
        command=command,
    )
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=cuda_environment(gpu),
    )
    return RunningJob(job=job, gpu=gpu, process=process, handle=handle)


def run_jobs(jobs: list[Job], args) -> None:
    pending = [job for job in jobs if not completed(job, args)]
    running: list[RunningJob] = []
    while pending or running:
        launched = False
        while pending:
            gpu, states = available_gpu(
                args.gpu_candidates,
                {item.gpu for item in running},
                args.maximum_used_memory_mib,
            )
            if gpu is None:
                if not running:
                    emit("waiting_for_h200", states=states)
                break
            running.append(launch(pending.pop(0), gpu, args))
            launched = True
        for item in list(running):
            returncode = item.process.poll()
            if returncode is None:
                continue
            item.handle.close()
            running.remove(item)
            emit(
                "rald_gate_run_finished",
                stage=item.job.stage,
                arm=item.job.arm,
                seed=item.job.seed,
                gpu=item.gpu,
                returncode=returncode,
            )
            if returncode == 0 and completed(item.job, args):
                continue
            if (
                resource_failure(item.job.log)
                and item.job.attempts < args.maximum_resource_retries
            ):
                if item.job.run.exists() and not (item.job.run / "last.pt").exists():
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    item.job.run.rename(
                        item.job.run.with_name(
                            f"{item.job.run.name}.incomplete.{timestamp}"
                        )
                    )
                emit(
                    "rald_gate_resource_retry",
                    stage=item.job.stage,
                    arm=item.job.arm,
                    seed=item.job.seed,
                    attempt=item.job.attempts,
                )
                pending.append(item.job)
                continue
            for active in running:
                active.process.terminate()
                active.handle.close()
            raise RuntimeError(
                f"{item.job.stage} {item.job.arm} seed {item.job.seed} failed"
            )
        if pending or running:
            time.sleep(1 if launched else args.poll_seconds)


def wait_for_available_gpu(args, event: str) -> int:
    while True:
        gpu, states = available_gpu(
            args.gpu_candidates, set(), args.maximum_used_memory_mib
        )
        if gpu is not None:
            return gpu
        emit(event, states=states)
        time.sleep(args.poll_seconds)


def run_gpu_command(command: list[str], log: Path, args, event: str) -> None:
    gpu = wait_for_available_gpu(args, f"waiting_for_h200_{event}")
    log.parent.mkdir(parents=True, exist_ok=True)
    emit(event, gpu=gpu, command=command)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=cuda_environment(gpu),
        )


def run_comparison(command: list[str], output: Path, event: str, args) -> dict:
    if not output.exists():
        emit(event, command=command)
        result = subprocess.run(command, check=False, cwd=args.repo_root)
        if result.returncode not in (0, 3):
            raise RuntimeError(f"{event} failed with {result.returncode}")
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--g1-comparison", type=Path, required=True)
    parser.add_argument("--g1-source-commit", required=True)
    parser.add_argument("--g1b-summary", type=Path, required=True)
    parser.add_argument("--g1b-run-root", type=Path, required=True)
    parser.add_argument("--g1b-training-source-commit", required=True)
    parser.add_argument("--g1b-decision-source-commit", required=True)
    parser.add_argument("--rh2-summary", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(FROZEN_G1B_SEEDS)
    )
    parser.add_argument("--g2r-epochs", type=int, default=30)
    parser.add_argument("--g2r-warmup-epochs", type=int, default=5)
    parser.add_argument("--g3r-epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--required-gpu-name", default="NVIDIA H200 NVL")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()

    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)
    if tuple(args.seeds) != FROZEN_G1B_SEEDS:
        raise ValueError("G2R/G3R requires the exact frozen seeds in order")
    if (args.g2r_epochs, args.g2r_warmup_epochs, args.g3r_epochs) != (30, 5, 20):
        raise ValueError("G2R/G3R training budgets are frozen at 30/5/20")
    args.run_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.run_root / f"g2r_g3r_summary_{args.source_commit[:8]}.json"
    if summary_path.exists():
        emit("rald_gate_summary_exists", summary=str(summary_path))
        return

    rh2 = wait_for_json(args.rh2_summary, args.poll_seconds, "waiting_for_rh2")
    if rh2.get("status") != "rh2_passed":
        atomic_json(
            summary_path,
            {
                "status": "skipped_rh2_not_passed",
                "source_commit": args.source_commit,
                "rh2_summary": str(args.rh2_summary),
                "rh2_status": rh2.get("status"),
            },
        )
        return
    if rh2.get("source_commit") != args.source_commit:
        raise ValueError("G2R/G3R requires RH2 from the same source commit")
    g1b = wait_for_json(args.g1b_summary, args.poll_seconds, "waiting_for_g1b")
    parent_mode, parents = validate_g1b_summary(
        g1b,
        args.g1b_training_source_commit,
        args.g1b_decision_source_commit,
        args.g1b_run_root,
        tuple(args.seeds),
    )
    if (
        rh2.get("parent_mode") != parent_mode
        or rh2.get("parent_route") != "independent_g1b_parent"
        or rh2.get("g1b_summary_sha256") != sha256(args.g1b_summary)
        or rh2.get("seeds") != args.seeds
    ):
        raise ValueError("RH2 and G1B family contracts differ")

    g2r_jobs = [
        Job(
            stage="g2r",
            seed=seed,
            arm=head,
            parent=parents[seed],
            run=args.run_root
            / f"g2r_{head}_seed{seed}_{args.source_commit[:8]}",
            log=args.run_root / f"g2r_{head}_seed{seed}.log",
        )
        for head in G2R_HEADS
        for seed in args.seeds
    ]
    run_jobs(g2r_jobs, args)
    scalar = [job for job in g2r_jobs if job.arm == "scalar"]
    distribution = [job for job in g2r_jobs if job.arm == "distribution"]
    g2r_path = args.run_root / f"g2r_comparison_{args.source_commit[:8]}.json"
    g2r = run_comparison(
        [
            str(args.python),
            "-u",
            str(args.repo_root / "code/scripts/compare_rald_anchor_g2r.py"),
            "--scalar-runs",
            *[str(job.run) for job in scalar],
            "--distribution-runs",
            *[str(job.run) for job in distribution],
            "--output",
            str(g2r_path),
        ],
        g2r_path,
        "g2r_comparison_started",
        args,
    )
    if g2r.get("decision", {}).get("g2r_passed") is not True:
        atomic_json(
            summary_path,
            {
                "status": "g2r_gate_failed",
                "source_commit": args.source_commit,
                "rh2_summary": str(args.rh2_summary),
                "g2r_comparison": str(g2r_path),
                "decision": g2r.get("decision"),
            },
        )
        return

    distribution_by_seed = {job.seed: job.run for job in distribution}
    g3r_jobs = [
        Job(
            stage="g3r",
            seed=seed,
            arm=variant,
            parent=parents[seed],
            initial_refiner=distribution_by_seed[seed],
            run=args.run_root
            / f"g3r_{variant}_seed{seed}_{args.source_commit[:8]}",
            log=args.run_root / f"g3r_{variant}_seed{seed}.log",
        )
        for variant in G3R_VARIANTS
        for seed in args.seeds
    ]
    run_jobs(g3r_jobs, args)
    jobs_by_variant = {
        variant: [job for job in g3r_jobs if job.arm == variant]
        for variant in G3R_VARIANTS
    }
    renderer_path = args.run_root / f"renderer_{args.source_commit[:8]}.json"
    if not renderer_path.exists():
        run_gpu_command(
            [
                str(args.python),
                "-u",
                str(args.repo_root / "code/scripts/verify_point_to_cube_renderer.py"),
                "--output",
                str(renderer_path),
                "--device",
                "cuda:0",
                "--required-gpu-name",
                args.required_gpu_name,
                "--source-commit",
                args.source_commit,
            ],
            args.run_root / "renderer.log",
            args,
            "renderer_verification_started",
        )
    robustness_path = (
        args.run_root / f"g3r_robustness_{args.source_commit[:8]}.json"
    )
    if not robustness_path.exists():
        robustness_command = [
            str(args.python),
            "-u",
            str(
                args.repo_root
                / "code/scripts/eval_rald_anchor_g3r_robustness.py"
            ),
            "--none-runs",
            *[str(job.run) for job in jobs_by_variant["none"]],
            "--full-runs",
            *[str(job.run) for job in jobs_by_variant["full"]],
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
            "--output",
            str(robustness_path),
            "--device",
            "cuda:0",
            "--source-commit",
            args.source_commit,
        ]
        if robustness_path.with_suffix(
            robustness_path.suffix + ".progress.json"
        ).exists():
            robustness_command.append("--resume")
        run_gpu_command(
            robustness_command,
            args.run_root / "g3r_robustness.log",
            args,
            "g3r_robustness_started",
        )
    g3r_path = args.run_root / f"g3r_comparison_{args.source_commit[:8]}.json"
    g3r = run_comparison(
        [
            str(args.python),
            "-u",
            str(args.repo_root / "code/scripts/compare_rald_anchor_g3r.py"),
            "--none-runs",
            *[str(job.run) for job in jobs_by_variant["none"]],
            "--local-runs",
            *[str(job.run) for job in jobs_by_variant["local_peak"]],
            "--marginal-runs",
            *[str(job.run) for job in jobs_by_variant["marginal"]],
            "--full-runs",
            *[str(job.run) for job in jobs_by_variant["full"]],
            "--g2r-comparison",
            str(g2r_path),
            "--renderer-test-report",
            str(renderer_path),
            "--robustness-report",
            str(robustness_path),
            "--output",
            str(g3r_path),
        ],
        g3r_path,
        "g3r_comparison_started",
        args,
    )
    passed = g3r.get("decision", {}).get("g3r_passed") is True
    selected_arm = "full" if passed else None
    selected_runs = (
        g3r.get("runs", {}).get("full", {}) if passed else {}
    )
    selected_run_hashes = (
        g3r.get("run_hashes", {}).get("full", {}) if passed else {}
    )
    if passed and (
        set(selected_runs) != {str(seed) for seed in args.seeds}
        or set(selected_run_hashes) != {str(seed) for seed in args.seeds}
    ):
        raise ValueError("Passing G3R report lacks the complete selected run matrix")
    atomic_json(
        summary_path,
        {
            "status": "g3r_passed" if passed else "g3r_gate_failed",
            "source_commit": args.source_commit,
            "parent_mode": parent_mode,
            "parent_route": "independent_g1b_parent",
            "seeds": args.seeds,
            "rh2_summary": str(args.rh2_summary),
            "rh2_summary_sha256": sha256(args.rh2_summary),
            "g1b_summary": str(args.g1b_summary),
            "g1b_summary_sha256": sha256(args.g1b_summary),
            "g2r_comparison": str(g2r_path),
            "g2r_comparison_sha256": sha256(g2r_path),
            "g3r_comparison": str(g3r_path),
            "g3r_comparison_sha256": sha256(g3r_path),
            "g2r_decision": g2r.get("decision"),
            "g3r_decision": g3r.get("decision"),
            "selected_arm": selected_arm,
            "selected_runs": selected_runs,
            "selected_run_hashes": selected_run_hashes,
        },
    )
    emit("rald_gate_chain_finished", summary=str(summary_path), passed=passed)


if __name__ == "__main__":
    main()
