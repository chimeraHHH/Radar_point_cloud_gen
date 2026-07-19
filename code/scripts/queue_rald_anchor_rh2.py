#!/usr/bin/env python3
"""Queue three-seed RH2 after RH1 and the core G2/G3 protocol finish."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from gpu_runtime import cuda_environment, validate_gpu_candidates


RESOURCE_FAILURE_MARKERS = (
    "out of memory",
    "cuda error",
    "device is busy",
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
    seed: int
    parent: Path
    run: Path
    log: Path
    attempts: int = 0


@dataclass
class RunningJob:
    job: Job
    gpu: int
    process: subprocess.Popen
    handle: object


def completed(job: Job, epochs: int, source_commit: str) -> bool:
    metrics = job.run / "best_validation_metrics.json"
    config = job.run / "config.json"
    log = job.run / "train_log.jsonl"
    if not all(path.is_file() for path in (metrics, config, log)):
        return False
    document = json.loads(config.read_text(encoding="utf-8"))
    if document["provenance"]["git_commit"] != source_commit:
        return False
    records = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return bool(records) and int(records[-1]["epoch"]) == epochs


def g1b_parent_runs(
    summary: dict, seeds: list[int], run_root: Path, source_commit: str
) -> dict[int, Path]:
    if summary.get("status") != "g1b_passed" or not summary.get("candidate_mode"):
        raise ValueError("G1B did not authorize an independent geometry parent")
    if summary.get("training_source_commit") != source_commit:
        raise ValueError("G1B training source commit differs from the queue contract")
    if set(seeds) - set(summary.get("seeds", [])):
        raise ValueError("RH2 seeds are absent from the G1B Stage B decision")
    mode = str(summary["candidate_mode"])
    authorized = {Path(path) for path in summary.get("candidate_runs", [])}
    parents = {
        seed: run_root
        / f"g1b_stage_b_{mode}_seed{seed}_{source_commit[:8]}"
        for seed in seeds
    }
    if set(parents.values()) - authorized:
        raise ValueError("G1B summary does not authorize all RH2 candidate parents")
    return parents


def train_command(job: Job, args) -> list[str]:
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
        "--g1b-summary",
        str(args.g1b_summary),
        "--parent-g1-run",
        str(job.parent),
        "--output",
        str(job.run),
        "--epochs",
        str(args.rh2_epochs),
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
    if (job.run / "last.pt").exists():
        command.append("--resume")
    return command


def launch(job: Job, gpu: int, args) -> RunningJob:
    job.log.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log.open("a", encoding="utf-8")
    command = train_command(job, args)
    job.attempts += 1
    emit(
        "rh2_run_started",
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


def resource_failure(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    with log_path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 64_000))
        tail = handle.read().decode("utf-8", errors="replace").lower()
    return any(marker in tail for marker in RESOURCE_FAILURE_MARKERS)


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
    parser.add_argument("--g1b-source-commit", required=True)
    parser.add_argument("--rh1-summary", type=Path, required=True)
    parser.add_argument("--core-g2-g3-summary", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260716, 20260717, 20260718]
    )
    parser.add_argument("--rh2-epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-frames", type=int, default=8)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--required-gpu-name", default="NVIDIA H200 NVL")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()

    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)
    args.run_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.run_root / f"rh2_queue_summary_{args.source_commit[:8]}.json"
    if summary_path.exists():
        emit("rh2_summary_exists", summary=str(summary_path))
        return
    rh1 = wait_for_json(args.rh1_summary, args.poll_seconds, "waiting_for_rh1")
    if rh1.get("status") != "rh1_passed":
        atomic_json(
            summary_path,
            {
                "status": "skipped_rh1_not_passed",
                "source_commit": args.source_commit,
                "rh1_summary": str(args.rh1_summary),
                "rh1_status": rh1.get("status"),
            },
        )
        return
    g1 = wait_for_json(args.g1_comparison, args.poll_seconds, "waiting_for_g1")
    decision = g1.get("decision", {})
    parent_mode = str(rh1.get("parent_mode"))
    parent_route = str(rh1.get("route"))
    if parent_route == "formal_g1_passed" and decision.get("g1_passed") is not True:
        raise ValueError("RH1 Full-RAED route contradicts G1")
    if (
        parent_route == "late_fusion_recovery_after_g1_failure"
        and decision.get("rae_max_beats_cfar") is not True
    ):
        raise ValueError("RH1 RAE-Max route contradicts G1")
    if parent_route == "independent_g1b_parent":
        g1b = wait_for_json(args.g1b_summary, args.poll_seconds, "waiting_for_g1b")
        if (
            g1b.get("status") != "g1b_passed"
            or g1b.get("candidate_mode") != parent_mode
        ):
            raise ValueError("RH1 G1B route contradicts Stage B")
    elif parent_route not in {
        "formal_g1_passed",
        "late_fusion_recovery_after_g1_failure",
    }:
        raise ValueError("RH1 summary has an unknown parent route")
    if parent_route == "formal_g1_passed":
        core = wait_for_json(
            args.core_g2_g3_summary, args.poll_seconds, "waiting_for_core_g2_g3"
        )
        if not isinstance(core, dict) or "source_commit" not in core:
            raise ValueError("Core G2/G3 summary is malformed")
        core_dependency = {
            "status": "finished",
            "summary": str(args.core_g2_g3_summary),
            "source_commit": core["source_commit"],
        }
    else:
        core_dependency = {
            "status": "not_unlocked_after_g1_failure",
            "summary": None,
            "source_commit": None,
        }

    if parent_route == "independent_g1b_parent":
        parents = g1b_parent_runs(
            g1b, args.seeds, args.g1b_run_root, args.g1b_source_commit
        )
    else:
        parents = {
            seed: args.g1_comparison.parent
            / f"g1_{parent_mode}_seed{seed}_{args.g1_source_commit[:8]}"
            for seed in args.seeds
        }
    jobs = [
        Job(
            seed=seed,
            parent=parents[seed],
            run=args.run_root
            / f"rh2_{parent_mode}_seed{seed}_{args.source_commit[:8]}",
            log=args.run_root / f"rh2_seed{seed}.log",
        )
        for seed in args.seeds
    ]
    pending = [
        job
        for job in jobs
        if not completed(job, args.rh2_epochs, args.source_commit)
    ]
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
                "rh2_run_finished",
                seed=item.job.seed,
                gpu=item.gpu,
                returncode=returncode,
            )
            if returncode != 0 or not completed(
                item.job, args.rh2_epochs, args.source_commit
            ):
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
                        "rh2_resource_retry",
                        seed=item.job.seed,
                        attempt=item.job.attempts,
                    )
                    pending.append(item.job)
                    continue
                atomic_json(
                    summary_path,
                    {
                        "status": "rh2_process_failed",
                        "source_commit": args.source_commit,
                        "seed": item.job.seed,
                        "returncode": returncode,
                        "run": str(item.job.run),
                    },
                )
                for active in running:
                    active.process.terminate()
                    active.handle.close()
                raise SystemExit(returncode or 3)
        if pending or running:
            time.sleep(1 if launched else args.poll_seconds)

    comparison_path = args.run_root / f"rh2_comparison_{args.source_commit[:8]}.json"
    if not comparison_path.exists():
        command = [
            str(args.python),
            "-u",
            str(args.repo_root / "code/scripts/compare_rald_anchor_rh2.py"),
            "--runs",
            *[str(job.run) for job in jobs],
            "--output",
            str(comparison_path),
            "--required-seeds",
            str(len(args.seeds)),
        ]
        emit("rh2_comparison_started", command=command)
        subprocess.run(command, check=True, cwd=args.repo_root)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    passed = comparison.get("decision", {}).get("rh2_passed") is True
    summary = {
        "status": "rh2_passed" if passed else "rh2_gate_failed",
        "source_commit": args.source_commit,
        "parent_mode": parent_mode,
        "parent_route": parent_route,
        "core_g2_g3_dependency": core_dependency,
        "runs": [str(job.run) for job in jobs],
        "comparison": str(comparison_path),
        "decision": comparison.get("decision"),
    }
    atomic_json(summary_path, summary)
    emit("rh2_finished", summary=summary)


if __name__ == "__main__":
    main()
