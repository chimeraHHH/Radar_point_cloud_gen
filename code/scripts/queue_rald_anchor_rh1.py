#!/usr/bin/env python3
"""Queue RH0.5 and RH1 behind the frozen formal G1 decision."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from gpu_runtime import cuda_environment, validate_gpu_candidates


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


def wait_for_json(path: Path, poll_seconds: int) -> dict:
    while not path.exists():
        emit("waiting_for_g1_comparison", missing=str(path))
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


def wait_for_gpu(
    candidates: list[int], maximum_used_memory_mib: int, poll_seconds: int
) -> int:
    while True:
        states = {}
        for index in candidates:
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
                emit("h200_available", gpu=index, states=states)
                return index
        emit("waiting_for_h200", states=states)
        time.sleep(poll_seconds)


def run_logged(command: list[str], gpu: int, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    emit("job_started", gpu=gpu, log=str(log_path), command=command)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=cuda_environment(gpu),
            check=False,
        )
    emit("job_finished", gpu=gpu, log=str(log_path), returncode=completed.returncode)
    return completed.returncode


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
    parser.add_argument("--parent-g1-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--rh1-epochs", type=int, default=20)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--required-gpu-name", default="NVIDIA H200 NVL")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    args = parser.parse_args()

    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)
    args.run_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.run_root / f"rh1_queue_summary_{args.source_commit[:8]}.json"
    if summary_path.exists():
        emit("queue_summary_exists", summary=str(summary_path))
        return
    comparison = wait_for_json(args.g1_comparison, args.poll_seconds)
    if comparison.get("decision", {}).get("g1_passed") is not True:
        summary = {
            "status": "skipped_g1_failed",
            "source_commit": args.source_commit,
            "g1_comparison": str(args.g1_comparison),
            "g1_decision": comparison.get("decision"),
            "rh05_started": False,
            "rh1_started": False,
        }
        atomic_json(summary_path, summary)
        emit("rald_anchor_skipped", summary=summary)
        return
    if args.seed not in comparison.get("seeds", []):
        raise ValueError("RH1 seed is absent from the formal G1 comparison")
    parent_config = args.parent_g1_run / "config.json"
    parent_checkpoint = args.parent_g1_run / "best.pt"
    if not parent_config.is_file() or not parent_checkpoint.is_file():
        raise FileNotFoundError("Frozen G1 parent is incomplete")

    integration_path = args.run_root / "rh05_native_integration.json"
    if not integration_path.exists():
        gpu = wait_for_gpu(
            args.gpu_candidates, args.maximum_used_memory_mib, args.poll_seconds
        )
        integration_command = [
            str(args.python),
            "-u",
            str(args.repo_root / "code/scripts/verify_rald_anchor_training_chain.py"),
            "--data-root",
            str(args.data_root),
            "--parent-g1-run",
            str(args.parent_g1_run),
            "--output",
            str(integration_path),
            "--device",
            "cuda:0",
            "--required-gpu-name",
            args.required_gpu_name,
            "--source-commit",
            args.source_commit,
        ]
        returncode = run_logged(
            integration_command, gpu, args.run_root / "rh05_native_integration.log"
        )
        if returncode != 0:
            atomic_json(
                summary_path,
                {
                    "status": "rh05_failed",
                    "source_commit": args.source_commit,
                    "returncode": returncode,
                    "integration_report": str(integration_path),
                },
            )
            raise SystemExit(returncode)
    integration = json.loads(integration_path.read_text(encoding="utf-8"))
    if integration.get("passed") is not True:
        atomic_json(
            summary_path,
            {
                "status": "rh05_gate_failed",
                "source_commit": args.source_commit,
                "integration_report": str(integration_path),
                "checks": integration.get("checks"),
            },
        )
        return

    run_path = args.run_root / f"rh1_seed{args.seed}_{args.source_commit[:8]}"
    gate_path = run_path / "rh1_gate.json"
    if not gate_path.exists():
        gpu = wait_for_gpu(
            args.gpu_candidates, args.maximum_used_memory_mib, args.poll_seconds
        )
        train_command = [
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
            "--parent-g1-run",
            str(args.parent_g1_run),
            "--output",
            str(run_path),
            "--epochs",
            str(args.rh1_epochs),
            "--seed",
            str(args.seed),
            "--rh1-one-frame",
            "--device",
            "cuda:0",
            "--source-commit",
            args.source_commit,
        ]
        if (run_path / "last.pt").exists():
            train_command.append("--resume")
        returncode = run_logged(
            train_command, gpu, args.run_root / f"rh1_seed{args.seed}.log"
        )
        if returncode != 0:
            atomic_json(
                summary_path,
                {
                    "status": "rh1_process_failed",
                    "source_commit": args.source_commit,
                    "returncode": returncode,
                    "run": str(run_path),
                },
            )
            raise SystemExit(returncode)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    passed = gate.get("rh1_gate", {}).get("passed") is True
    summary = {
        "status": "rh1_passed" if passed else "rh1_gate_failed",
        "source_commit": args.source_commit,
        "g1_comparison": str(args.g1_comparison),
        "parent_g1_run": str(args.parent_g1_run),
        "rh05_report": str(integration_path),
        "rh1_run": str(run_path),
        "rh1_gate": gate.get("rh1_gate"),
        "rh2_unlocked": passed,
    }
    atomic_json(summary_path, summary)
    emit("rald_anchor_rh1_finished", summary=summary)


if __name__ == "__main__":
    main()
