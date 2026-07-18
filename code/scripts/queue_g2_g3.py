#!/usr/bin/env python3
"""Queue the preregistered G2 and G3 experiments after G1 completes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from gpu_runtime import cuda_environment, validate_gpu_candidates


RESOURCE_FAILURE_MARKERS = (
    "out of memory",
    "cuda error",
    "cuda driver error",
    "device is busy",
    "device-side assert",
    "cublas_status_alloc_failed",
)

G2_ARMS = {
    "scalar": "e3_scalar",
    "distribution": "e4_distribution",
    "physics_distribution": "e5_physics",
}
G3_ARMS = {
    "none": "c0_none",
    "local_peak": "c1_local",
    "marginal": "c2_marginal",
    "full": "c3_full",
}


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


def g1_parent_runs(
    comparison_path: Path, source_commit: str, seeds: list[int]
) -> dict[int, Path]:
    """Resolve G1 parents beside the comparison, not inside the G2 run root."""
    tag = source_commit[:8]
    return {
        seed: comparison_path.parent / f"g1_full_raed_seed{seed}_{tag}"
        for seed in seeds
    }


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


def completed_run(run_path: Path, epochs: int, source_commit: str) -> bool:
    config_path = run_path / "config.json"
    metrics_path = run_path / "best_validation_metrics.json"
    log_path = run_path / "train_log.jsonl"
    if not all(path.is_file() for path in (config_path, metrics_path, log_path)):
        return False
    document = json.loads(config_path.read_text(encoding="utf-8"))
    if document["provenance"]["git_commit"] != source_commit:
        return False
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return bool(records) and int(records[-1]["epoch"]) == epochs


@dataclass
class Job:
    phase: str
    arm: str
    seed: int
    parent: Path
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
    if (job.run_path / "config.json").is_file() and (
        job.run_path / "last.pt"
    ).is_file():
        return True
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = job.run_path.with_name(f"{job.run_path.name}.incomplete.{timestamp}")
    job.run_path.rename(archived)
    emit(
        "incomplete_run_archived",
        phase=job.phase,
        arm=job.arm,
        seed=job.seed,
        source=str(job.run_path),
        destination=str(archived),
    )
    return False


def train_command(job: Job, python: Path, repo_root: Path, args, resume: bool) -> list[str]:
    common = [
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
        str(job.run_path),
        "--seed",
        str(job.seed),
        "--eval-every",
        str(args.eval_every),
        "--max-eval-frames",
        str(args.max_eval_frames),
        "--device",
        "cuda:0",
        "--source-commit",
        args.source_commit,
    ]
    if job.phase == "g2":
        command = [
            str(python),
            "-u",
            str(repo_root / "code/scripts/train_cube_doppler.py"),
            *common,
            "--parent-e2-run",
            str(job.parent),
            "--static-doppler-audit",
            str(args.static_doppler_audit),
            "--head-mode",
            job.arm,
            "--epochs",
            str(args.g2_epochs),
            "--warmup-epochs",
            str(args.g2_warmup_epochs),
        ]
    else:
        command = [
            str(python),
            "-u",
            str(repo_root / "code/scripts/train_cube_cycle.py"),
            *common,
            "--parent-g2-run",
            str(job.parent),
            "--variant",
            job.arm,
            "--epochs",
            str(args.g3_epochs),
        ]
    if resume:
        command.append("--resume")
    return command


def launch_job(job: Job, gpu: int, command: list[str]) -> RunningJob:
    environment = cuda_environment(gpu)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8")
    job.attempts += 1
    emit(
        f"{job.phase}_run_started",
        arm=job.arm,
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


def run_jobs(
    jobs: list[Job],
    epochs: int,
    python: Path,
    repo_root: Path,
    args,
) -> None:
    pending = [
        job
        for job in jobs
        if not completed_run(job.run_path, epochs, args.source_commit)
    ]
    for job in jobs:
        if job not in pending:
            emit(
                f"{job.phase}_run_already_complete", arm=job.arm, seed=job.seed
            )
    running: list[RunningJob] = []
    while pending or running:
        for active in running.copy():
            return_code = active.process.poll()
            if return_code is None:
                continue
            active.handle.close()
            running.remove(active)
            emit(
                f"{active.job.phase}_run_finished",
                arm=active.job.arm,
                seed=active.job.seed,
                gpu=active.gpu,
                return_code=return_code,
            )
            if return_code == 0 and completed_run(
                active.job.run_path, epochs, args.source_commit
            ):
                continue
            if (
                resource_failure(return_code, active.job.log_path)
                and active.job.attempts <= args.maximum_resource_retries
            ):
                emit(
                    f"{active.job.phase}_resource_retry_queued",
                    arm=active.job.arm,
                    seed=active.job.seed,
                    attempts=active.job.attempts,
                )
                pending.append(active.job)
                continue
            emit(
                f"{active.job.phase}_run_failed",
                arm=active.job.arm,
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
                    emit("waiting_for_gpu", phase=pending[0].phase, states=states)
                break
            job = pending.pop(0)
            resume = prepare_run(job)
            command = train_command(job, python, repo_root, args, resume)
            running.append(launch_job(job, gpu, command))
            assigned.add(gpu)
        if pending or running:
            time.sleep(args.poll_seconds)


def run_gpu_command(
    phase: str,
    command: list[str],
    output: Path,
    log_path: Path,
    args,
    accepted_codes: tuple[int, ...] = (0,),
    resume: bool = False,
) -> dict:
    if output.is_file():
        emit(f"{phase}_already_complete", output=str(output))
        return json.loads(output.read_text(encoding="utf-8"))
    attempts = 0
    while True:
        gpu, states = available_gpu(
            args.gpu_candidates, set(), args.maximum_used_memory_mib
        )
        if gpu is None:
            emit("waiting_for_gpu", phase=phase, states=states)
            time.sleep(args.poll_seconds)
            continue
        current = command.copy()
        if resume and output.with_suffix(output.suffix + ".progress.json").exists():
            current.append("--resume")
        environment = cuda_environment(gpu)
        attempts += 1
        emit(
            f"{phase}_started",
            gpu=gpu,
            attempt=attempts,
            log=str(log_path),
            command=current,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                current,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        emit(f"{phase}_finished", return_code=completed.returncode)
        if completed.returncode in accepted_codes and output.is_file():
            return json.loads(output.read_text(encoding="utf-8"))
        if (
            resource_failure(completed.returncode, log_path)
            and attempts <= args.maximum_resource_retries
        ):
            emit(f"{phase}_resource_retry_queued", attempts=attempts)
            continue
        raise SystemExit(completed.returncode or 4)


def run_cpu_decision(
    phase: str,
    command: list[str],
    output: Path,
    accepted_codes: tuple[int, ...] = (0, 3),
) -> dict:
    if output.is_file():
        emit(f"{phase}_already_complete", output=str(output))
        return json.loads(output.read_text(encoding="utf-8"))
    emit(f"{phase}_started", command=command)
    completed = subprocess.run(command, check=False)
    emit(f"{phase}_finished", return_code=completed.returncode)
    if completed.returncode not in accepted_codes or not output.is_file():
        raise SystemExit(completed.returncode or 4)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--static-doppler-audit", type=Path, required=True)
    parser.add_argument("--g1-comparison", type=Path, required=True)
    parser.add_argument("--g1-qualitative-report", type=Path, required=True)
    parser.add_argument("--g1-source-commit", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--required-gpu-name")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260716, 20260717, 20260718]
    )
    parser.add_argument("--g2-epochs", type=int, default=30)
    parser.add_argument("--g2-warmup-epochs", type=int, default=5)
    parser.add_argument("--g3-epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()

    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)

    if len(set(args.seeds)) != 3:
        raise ValueError("Formal G2/G3 requires exactly three unique seeds")
    python = Path(os.environ.get("PYTHON", "python"))
    g1_report = wait_for_json(
        args.g1_comparison, args.poll_seconds, "waiting_for_g1_comparison"
    )
    if g1_report.get("decision", {}).get("g1_passed") is not True:
        raise SystemExit("G1 did not pass; G2/G3 cannot start")
    wait_for_json(
        args.g1_qualitative_report,
        args.poll_seconds,
        "waiting_for_g1_qualitative_report",
    )
    static_audit = wait_for_json(
        args.static_doppler_audit,
        args.poll_seconds,
        "waiting_for_static_doppler_audit",
    )
    physics_enabled = static_audit.get("passed") is True
    emit(
        "g2_g3_dependencies_passed",
        g1_comparison=str(args.g1_comparison),
        g1_qualitative_report=str(args.g1_qualitative_report),
        static_doppler_audit=str(args.static_doppler_audit),
        physics_enabled=physics_enabled,
    )

    source_tag = args.source_commit[:8]
    g1_parents = g1_parent_runs(
        args.g1_comparison, args.g1_source_commit, args.seeds
    )
    for seed, parent in g1_parents.items():
        if not completed_run(parent, 50, args.g1_source_commit):
            raise ValueError(f"Incomplete G1 parent for seed {seed}: {parent}")

    active_g2_arms = {
        mode: label
        for mode, label in G2_ARMS.items()
        if physics_enabled or mode != "physics_distribution"
    }
    g2_jobs = [
        Job(
            phase="g2",
            arm=mode,
            seed=seed,
            parent=g1_parents[seed],
            run_path=args.run_root / f"g2_{label}_seed{seed}_{source_tag}",
            log_path=args.run_root / f"g2_{label}_seed{seed}_{source_tag}.log",
        )
        for mode, label in active_g2_arms.items()
        for seed in args.seeds
    ]
    run_jobs(g2_jobs, args.g2_epochs, python, args.repo_root, args)

    g2_by_arm = {
        mode: [job for job in g2_jobs if job.arm == mode]
        for mode in active_g2_arms
    }
    counterfactual_path = None
    counterfactual = None
    if physics_enabled:
        counterfactual_path = args.run_root / f"g2_counterfactual_{source_tag}.json"
        counterfactual_command = [
            str(python),
            "-u",
            str(args.repo_root / "code/scripts/eval_cube_doppler_counterfactual.py"),
            "--runs",
            *[str(job.run_path) for job in g2_by_arm["physics_distribution"]],
            "--data-root",
            str(args.data_root),
            "--cache-root",
            str(args.cache_root),
            "--manifest",
            str(args.manifest),
            "--output",
            str(counterfactual_path),
            "--device",
            "cuda:0",
        ]
        counterfactual = run_gpu_command(
            "g2_counterfactual",
            counterfactual_command,
            counterfactual_path,
            args.run_root / f"g2_counterfactual_{source_tag}.log",
            args,
            accepted_codes=(0, 3),
        )

    g2_comparison_path = args.run_root / f"g2_comparison_{source_tag}.json"
    g2_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/compare_g2_cube_doppler.py"),
        "--scalar-runs",
        *[str(job.run_path) for job in g2_by_arm["scalar"]],
        "--distribution-runs",
        *[str(job.run_path) for job in g2_by_arm["distribution"]],
        "--output",
        str(g2_comparison_path),
    ]
    if physics_enabled:
        g2_command.extend(
            [
                "--physics-runs",
                *[
                    str(job.run_path)
                    for job in g2_by_arm["physics_distribution"]
                ],
                "--counterfactual-report",
                str(counterfactual_path),
            ]
        )
    g2_report = run_cpu_decision(
        "g2_comparison", g2_command, g2_comparison_path
    )
    distribution_passed = bool(g2_report["distribution_passed"])
    if physics_enabled and g2_report["physics_passed"] is True:
        selected_g2_arm = "physics_distribution"
    elif distribution_passed:
        selected_g2_arm = "distribution"
    else:
        failure_path = args.run_root / f"g2_g3_queue_summary_{source_tag}.json"
        failure_report = {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "g1_source_commit": args.g1_source_commit,
            "seeds": args.seeds,
            "g2": {
                "comparison": str(g2_comparison_path),
                "counterfactual": None
                if counterfactual_path is None
                else str(counterfactual_path),
                "physics_evaluated": physics_enabled,
                "passed": False,
                "distribution_passed": False,
                "checks": g2_report["checks"],
            },
            "g3": {"started": False},
            "completed": False,
            "failure": "E4 distribution head did not beat E3 scalar baseline",
        }
        atomic_json(failure_path, failure_report)
        emit("g2_distribution_gate_failed", summary=str(failure_path))
        raise SystemExit(3)
    selected_g2_parents = {
        job.seed: job.run_path for job in g2_by_arm[selected_g2_arm]
    }
    emit(
        "g2_decision_complete",
        passed=g2_report["g2_passed"],
        selected_arm=selected_g2_arm,
        counterfactual_passed=None
        if counterfactual is None
        else counterfactual["passed"],
    )

    renderer_path = args.run_root / f"g3_renderer_test_{source_tag}.json"
    renderer_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/verify_point_to_cube_renderer.py"),
        "--output",
        str(renderer_path),
        "--device",
        "cuda:0",
        "--source-commit",
        args.source_commit,
    ]
    run_gpu_command(
        "g3_renderer_test",
        renderer_command,
        renderer_path,
        args.run_root / f"g3_renderer_test_{source_tag}.log",
        args,
    )

    g3_jobs = [
        Job(
            phase="g3",
            arm=variant,
            seed=seed,
            parent=selected_g2_parents[seed],
            run_path=args.run_root / f"g3_{label}_seed{seed}_{source_tag}",
            log_path=args.run_root / f"g3_{label}_seed{seed}_{source_tag}.log",
        )
        for variant, label in G3_ARMS.items()
        for seed in args.seeds
    ]
    run_jobs(g3_jobs, args.g3_epochs, python, args.repo_root, args)
    g3_by_arm = {
        variant: [job for job in g3_jobs if job.arm == variant]
        for variant in G3_ARMS
    }

    robustness_path = args.run_root / f"g3_robustness_{source_tag}.json"
    robustness_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/eval_g3_cube_cycle_robustness.py"),
        "--none-runs",
        *[str(job.run_path) for job in g3_by_arm["none"]],
        "--full-runs",
        *[str(job.run_path) for job in g3_by_arm["full"]],
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
    run_gpu_command(
        "g3_robustness",
        robustness_command,
        robustness_path,
        args.run_root / f"g3_robustness_{source_tag}.log",
        args,
        resume=True,
    )

    g3_comparison_path = args.run_root / f"g3_comparison_{source_tag}.json"
    g3_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/compare_g3_cube_cycle.py"),
        "--none-runs",
        *[str(job.run_path) for job in g3_by_arm["none"]],
        "--local-runs",
        *[str(job.run_path) for job in g3_by_arm["local_peak"]],
        "--marginal-runs",
        *[str(job.run_path) for job in g3_by_arm["marginal"]],
        "--full-runs",
        *[str(job.run_path) for job in g3_by_arm["full"]],
        "--renderer-test-report",
        str(renderer_path),
        "--robustness-report",
        str(robustness_path),
        "--output",
        str(g3_comparison_path),
    ]
    g3_report = run_cpu_decision(
        "g3_comparison", g3_command, g3_comparison_path
    )
    selected_g3_arm = "full" if g3_report["g3_passed"] else "none"
    final_report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "g1_source_commit": args.g1_source_commit,
        "seeds": args.seeds,
        "g2": {
            "comparison": str(g2_comparison_path),
            "counterfactual": None
            if counterfactual_path is None
            else str(counterfactual_path),
            "passed": bool(g2_report["g2_passed"]),
            "distribution_passed": distribution_passed,
            "physics_evaluated": physics_enabled,
            "physics_passed": g2_report["physics_passed"],
            "selected_arm": selected_g2_arm,
            "selected_runs": {
                str(seed): str(path) for seed, path in selected_g2_parents.items()
            },
        },
        "g3": {
            "comparison": str(g3_comparison_path),
            "renderer_test": str(renderer_path),
            "robustness": str(robustness_path),
            "passed": bool(g3_report["g3_passed"]),
            "selected_arm": selected_g3_arm,
            "selected_runs": {
                str(job.seed): str(job.run_path)
                for job in g3_by_arm[selected_g3_arm]
            },
        },
        "completed": True,
    }
    final_path = args.run_root / f"g2_g3_queue_summary_{source_tag}.json"
    atomic_json(final_path, final_report)
    emit("g2_g3_queue_complete", summary=str(final_path), report=final_report)


if __name__ == "__main__":
    main()
