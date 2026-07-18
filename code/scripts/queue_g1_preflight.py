#!/usr/bin/env python3
"""Queue G1 normalization and one-frame overfit checks after G0 passes."""

from __future__ import annotations

import argparse
import json
import os
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
    candidates: list[int], poll_seconds: int, maximum_used_memory_mib: int
) -> int:
    while True:
        states = {}
        for index in candidates:
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
                emit("gpu_selected", gpu=index, states=states)
                return index
        emit("waiting_for_gpu", states=states)
        time.sleep(poll_seconds)


def run_logged(command: list[str], log_path: Path, gpu: int) -> None:
    environment = cuda_environment(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    emit("command_started", gpu=gpu, log=str(log_path), command=command)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    emit(
        "command_finished",
        gpu=gpu,
        log=str(log_path),
        return_code=completed.returncode,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def wait_for_g0(report_path: Path, required_frames: int, poll_seconds: int) -> dict:
    while True:
        if not report_path.exists():
            emit("waiting_for_g0", report=str(report_path))
            time.sleep(poll_seconds)
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            emit(
                "waiting_for_g0_report",
                report=str(report_path),
                error=f"{type(error).__name__}: {error}",
            )
            time.sleep(poll_seconds)
            continue
        aggregate = report.get("aggregate")
        if aggregate is None:
            emit(
                "waiting_for_g0_completion",
                report=str(report_path),
                completed_frames=len(report.get("frames", [])),
                required_frames=required_frames,
            )
            time.sleep(poll_seconds)
            continue
        break
    checks = {
        "required_frame_count": int(aggregate["successful_frames"])
        == required_frames,
        "no_failed_frames": int(aggregate["failed_frames"]) == 0,
        "g0_gate_pass": bool(aggregate["gate_pass"]),
    }
    emit("g0_checked", checks=checks, aggregate=aggregate)
    if not all(checks.values()):
        raise SystemExit(3)
    return report


def completed_overfit(run_path: Path, epochs: int) -> bool:
    log = run_path / "train_log.jsonl"
    if not (run_path / "best.pt").exists() or not log.exists():
        return False
    records = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return bool(records) and int(records[-1]["epoch"]) == epochs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0-report", type=Path, required=True)
    parser.add_argument("--required-g0-frames", type=int, default=100)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--required-gpu-name")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--overfit-epochs", type=int, default=50)
    args = parser.parse_args()

    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)

    python = Path(os.environ.get("PYTHON", "python"))
    train_script = args.repo_root / "code/scripts/train_cube_occupancy.py"
    verify_script = args.repo_root / "code/scripts/verify_cube_occupancy_overfit.py"
    normalization = args.run_root / "g1_cube_normalization_train.json"
    args.run_root.mkdir(parents=True, exist_ok=True)

    wait_for_g0(args.g0_report, args.required_g0_frames, args.poll_seconds)

    if not normalization.exists():
        gpu = wait_for_gpu(
            args.gpu_candidates,
            args.poll_seconds,
            args.maximum_used_memory_mib,
        )
        run_logged(
            [
                str(python),
                "-u",
                str(args.repo_root / "code/scripts/compute_cube_normalization.py"),
                "--data-root",
                str(args.data_root),
                "--manifest",
                str(args.manifest),
                "--scene-split",
                str(args.scene_split),
                "--output",
                str(normalization),
                "--device",
                "cuda:0",
                "--source-commit",
                args.source_commit,
            ],
            args.run_root / "g1_cube_normalization_train.log",
            gpu,
        )
    else:
        emit("normalization_exists", path=str(normalization))

    for mode in ("rae_max", "full_raed"):
        run_path = args.run_root / f"g1_overfit_{mode}_{args.source_commit[:8]}"
        verification = run_path / "overfit_verification.json"
        if verification.exists():
            report = json.loads(verification.read_text(encoding="utf-8"))
            if report.get("passed") is True:
                emit("overfit_already_verified", mode=mode, report=str(verification))
                continue
            raise SystemExit(3)

        if not completed_overfit(run_path, args.overfit_epochs):
            resume = False
            if run_path.exists() and any(run_path.iterdir()):
                if (run_path / "config.json").exists() and (
                    run_path / "last.pt"
                ).exists():
                    resume = True
                    emit("overfit_resuming", mode=mode, run=str(run_path))
                else:
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    archived = run_path.with_name(f"{run_path.name}.incomplete.{timestamp}")
                    run_path.rename(archived)
                    emit(
                        "overfit_incomplete_archived",
                        mode=mode,
                        source=str(run_path),
                        destination=str(archived),
                    )
            gpu = wait_for_gpu(
                args.gpu_candidates,
                args.poll_seconds,
                args.maximum_used_memory_mib,
            )
            train_command = [
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
                str(run_path),
                "--mode",
                mode,
                "--epochs",
                str(args.overfit_epochs),
                "--eval-every",
                "5",
                "--max-eval-frames",
                "1",
                "--overfit-one-frame",
                "--normalization-stats",
                str(normalization),
                "--device",
                "cuda:0",
                "--source-commit",
                args.source_commit,
            ]
            if resume:
                train_command.append("--resume")
            run_logged(
                train_command,
                args.run_root / f"g1_overfit_{mode}_{args.source_commit[:8]}.log",
                gpu,
            )

        gpu = wait_for_gpu(
            args.gpu_candidates,
            args.poll_seconds,
            args.maximum_used_memory_mib,
        )
        run_logged(
            [
                str(python),
                "-u",
                str(verify_script),
                "--data-root",
                str(args.data_root),
                "--cache-root",
                str(args.cache_root),
                "--manifest",
                str(args.manifest),
                "--normalization-stats",
                str(normalization),
                "--run",
                str(run_path),
                "--output",
                str(verification),
                "--device",
                "cuda:0",
            ],
            args.run_root / f"g1_verify_{mode}_{args.source_commit[:8]}.log",
            gpu,
        )

    emit("g1_preflight_complete", source_commit=args.source_commit)


if __name__ == "__main__":
    main()
