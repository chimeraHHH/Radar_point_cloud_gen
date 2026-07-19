#!/usr/bin/env python3
"""Queue RH0.5 and RH1 behind the frozen formal G1 decision."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from g1b_contract import select_original_parent, sha256, validate_g1b_summary
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


def wait_for_named_json(path: Path, poll_seconds: int, event: str) -> dict:
    while not path.exists():
        emit(event, missing=str(path))
        time.sleep(poll_seconds)
    return json.loads(path.read_text(encoding="utf-8"))


def select_parent(
    decision: dict, full_parent: Path, rae_parent: Path
) -> tuple[str, Path, str] | None:
    selected = select_original_parent(decision)
    if selected is None:
        return None
    mode, route = selected
    return mode, full_parent if mode == "full_raed" else rae_parent, route


def select_g1b_parent(
    summary: dict,
    seed: int,
    run_root: Path,
    training_source_commit: str,
    decision_source_commit: str,
) -> tuple[str, Path] | None:
    if summary.get("status") != "g1b_passed" or not summary.get("candidate_mode"):
        return None
    mode, parents = validate_g1b_summary(
        summary,
        training_source_commit,
        decision_source_commit,
        run_root,
    )
    if seed not in parents:
        raise ValueError("RH1 seed is absent from the G1B Stage B decision")
    return mode, parents[seed]


def validate_integration_report(
    report: dict,
    source_commit: str,
    parent_run: Path,
    parent_mode: str,
    parent_source_commit: str,
    parent_checkpoint_sha256: str,
) -> None:
    parent = report.get("parent", {})
    expected = {
        "source_commit": source_commit,
        "parent_run": str(parent_run),
        "parent_mode": parent_mode,
        "parent_source_commit": parent_source_commit,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    observed = {
        "source_commit": report.get("source_commit"),
        "parent_run": parent.get("run"),
        "parent_mode": parent.get("mode"),
        "parent_source_commit": parent.get("git_commit"),
        "parent_checkpoint_sha256": parent.get("checkpoint_sha256"),
    }
    if observed != expected:
        raise ValueError("RH0.5 report provenance differs from the selected parent")


def validate_existing_summary(summary: dict, expected: dict) -> None:
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing RH1 summary differs from the queue contract")


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
    parser.add_argument("--g1-source-commit", required=True)
    parser.add_argument("--full-parent-g1-run", type=Path, required=True)
    parser.add_argument("--rae-parent-g1-run", type=Path, required=True)
    parser.add_argument("--g1b-summary", type=Path, required=True)
    parser.add_argument("--g1b-run-root", type=Path, required=True)
    parser.add_argument("--g1b-source-commit", required=True)
    parser.add_argument("--g1b-decision-source-commit", required=True)
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
    if args.seed != 20260716:
        raise ValueError("RH1 requires the frozen one-frame seed 20260716")
    if args.rh1_epochs != 20:
        raise ValueError("RH1 requires the frozen 20-epoch budget")
    args.run_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.run_root / f"rh1_queue_summary_{args.source_commit[:8]}.json"
    comparison = wait_for_json(args.g1_comparison, args.poll_seconds)
    decision = comparison.get("decision", {})
    selection = select_parent(
        decision, args.full_parent_g1_run, args.rae_parent_g1_run
    )
    if selection is None:
        g1b = wait_for_named_json(
            args.g1b_summary, args.poll_seconds, "waiting_for_g1b_stage_b"
        )
        g1b_selection = select_g1b_parent(
            g1b,
            args.seed,
            args.g1b_run_root,
            args.g1b_source_commit,
            args.g1b_decision_source_commit,
        )
        if g1b_selection is None:
            summary = {
                "status": "skipped_no_geometry_parent_passed",
                "source_commit": args.source_commit,
                "g1_comparison": str(args.g1_comparison),
                "g1_decision": decision,
                "g1b_summary": str(args.g1b_summary),
                "g1b_status": g1b.get("status"),
                "rh05_started": False,
                "rh1_started": False,
            }
            atomic_json(summary_path, summary)
            emit("rald_anchor_skipped", summary=summary)
            return
        parent_mode, parent_g1_run = g1b_selection
        route = "independent_g1b_parent"
    else:
        parent_mode, parent_g1_run, route = selection
        expected_parent = args.g1_comparison.parent / (
            f"g1_{parent_mode}_seed{args.seed}_{args.g1_source_commit[:8]}"
        )
        if parent_g1_run != expected_parent:
            raise ValueError("Original G1 parent path differs from the formal run")
    if args.seed not in comparison.get("seeds", []):
        raise ValueError("RH1 seed is absent from the formal G1 comparison")
    parent_config = parent_g1_run / "config.json"
    parent_checkpoint = parent_g1_run / "best.pt"
    if not parent_config.is_file() or not parent_checkpoint.is_file():
        raise FileNotFoundError("Frozen G1 parent is incomplete")
    parent_document = json.loads(parent_config.read_text(encoding="utf-8"))
    parent_source_commit = str(parent_document["provenance"]["git_commit"])
    expected_parent_source = (
        args.g1b_source_commit
        if route == "independent_g1b_parent"
        else args.g1_source_commit
    )
    if parent_source_commit != expected_parent_source:
        raise ValueError("Selected parent source commit differs from the queue contract")
    if parent_document["config"].get("mode") != parent_mode:
        raise ValueError("Selected parent mode differs from its config")
    if int(parent_document["config"].get("seed", -1)) != args.seed:
        raise ValueError("Selected parent seed differs from RH1")
    parent_config_hash = sha256(parent_config)
    parent_checkpoint_hash = sha256(parent_checkpoint)
    g1b_hash = (
        sha256(args.g1b_summary) if route == "independent_g1b_parent" else None
    )
    summary_contract = {
        "source_commit": args.source_commit,
        "seed": args.seed,
        "route": route,
        "parent_mode": parent_mode,
        "g1_comparison": str(args.g1_comparison),
        "g1_comparison_sha256": sha256(args.g1_comparison),
        "g1_source_commit": args.g1_source_commit,
        "g1b_summary": (
            str(args.g1b_summary) if route == "independent_g1b_parent" else None
        ),
        "g1b_summary_sha256": g1b_hash,
        "g1b_training_source_commit": (
            args.g1b_source_commit if route == "independent_g1b_parent" else None
        ),
        "g1b_decision_source_commit": (
            args.g1b_decision_source_commit
            if route == "independent_g1b_parent"
            else None
        ),
        "parent_g1_run": str(parent_g1_run),
        "parent_config_sha256": parent_config_hash,
        "parent_checkpoint_sha256": parent_checkpoint_hash,
        "parent_training_source_commit": parent_source_commit,
    }
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        validate_existing_summary(existing, summary_contract)
        emit("queue_summary_exists", summary=str(summary_path))
        return

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
            str(parent_g1_run),
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
                    **summary_contract,
                    "status": "rh05_failed",
                    "returncode": returncode,
                    "integration_report": str(integration_path),
                },
            )
            raise SystemExit(returncode)
    integration = json.loads(integration_path.read_text(encoding="utf-8"))
    validate_integration_report(
        integration,
        args.source_commit,
        parent_g1_run,
        parent_mode,
        parent_source_commit,
        parent_checkpoint_hash,
    )
    if integration.get("passed") is not True:
        atomic_json(
            summary_path,
            {
                **summary_contract,
                "status": "rh05_gate_failed",
                "integration_report": str(integration_path),
                "checks": integration.get("checks"),
            },
        )
        return

    run_path = (
        args.run_root
        / f"rh1_{parent_mode}_seed{args.seed}_{args.source_commit[:8]}"
    )
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
            "--g1-source-commit",
            args.g1_source_commit,
            "--g1b-summary",
            str(args.g1b_summary),
            "--g1b-training-source-commit",
            args.g1b_source_commit,
            "--g1b-decision-source-commit",
            args.g1b_decision_source_commit,
            "--parent-g1-run",
            str(parent_g1_run),
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
                    **summary_contract,
                    "status": "rh1_process_failed",
                    "returncode": returncode,
                    "run": str(run_path),
                },
            )
            raise SystemExit(returncode)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    passed = gate.get("rh1_gate", {}).get("passed") is True
    summary = {
        **summary_contract,
        "status": "rh1_passed" if passed else "rh1_gate_failed",
        "rh05_report": str(integration_path),
        "rh05_report_sha256": sha256(integration_path),
        "rh1_run": str(run_path),
        "rh1_gate": gate.get("rh1_gate"),
        "rh2_unlocked": passed,
    }
    atomic_json(summary_path, summary)
    emit("rald_anchor_rh1_finished", summary=summary)


if __name__ == "__main__":
    main()
