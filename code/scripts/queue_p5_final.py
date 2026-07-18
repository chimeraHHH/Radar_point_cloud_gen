#!/usr/bin/env python3
"""Queue the frozen P5 test evaluation after the G4 selection is complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from gpu_runtime import cuda_environment, validate_gpu_candidates
from queue_g2_g3 import atomic_json, available_gpu, emit, resource_failure, tail_text


FORMAL_SEEDS = (20260716, 20260717, 20260718)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_source(document: dict) -> str | None:
    return (
        document.get("source_commit")
        or document.get("configuration", {}).get("source_commit")
        or document.get("provenance", {}).get("git_commit")
    )


def wait_for_json(path: Path, poll_seconds: int, event: str) -> dict:
    while True:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                emit(event, error=f"{type(error).__name__}: {error}")
        else:
            emit(event, missing=str(path))
        time.sleep(poll_seconds)


def wait_for_download_completion(
    path: Path, expected_sequences: set[int], poll_seconds: int
) -> dict:
    while True:
        document = wait_for_json(path, poll_seconds, "waiting_for_p5_download")
        completed = {int(value) for value in document.get("completed_sequences", [])}
        failures = document.get("failures", [])
        if completed == expected_sequences and not failures:
            return document
        emit(
            "waiting_for_p5_download_recovery",
            completed_sequences=len(completed),
            expected_sequences=len(expected_sequences),
            missing_sequences=sorted(expected_sequences - completed),
            unexpected_sequences=sorted(completed - expected_sequences),
            failure_count=len(failures),
        )
        time.sleep(poll_seconds)


def validate_g4_release(summary: dict, seeds: tuple[int, ...]) -> dict:
    if summary.get("completed") is not True:
        raise ValueError("G4 is incomplete; P5 test remains sealed")
    formal_runs = {
        int(seed): Path(path) for seed, path in summary.get("formal_runs", {}).items()
    }
    baseline_reports = {
        int(seed): Path(path)
        for seed, path in summary.get("baseline_reports", {}).items()
    }
    rollout_reports = {
        int(seed): Path(path)
        for seed, path in summary.get("rollout_reports", {}).items()
    }
    expected = set(seeds)
    if any(set(values) != expected for values in (formal_runs, baseline_reports, rollout_reports)):
        raise ValueError("G4 summary does not cover the three formal seeds")
    required = ("parent_summary", "preflight_selection", "comparison")
    if any(not summary.get(field) for field in required):
        raise ValueError("G4 summary lacks frozen selection provenance")
    return {
        "formal_runs": formal_runs,
        "validation_baseline_reports": baseline_reports,
        "validation_rollout_reports": rollout_reports,
        "parent_summary": Path(summary["parent_summary"]),
        "preflight_selection": Path(summary["preflight_selection"]),
        "comparison": Path(summary["comparison"]),
        "selected_arm": summary["selected_arm"],
        "selected_fusion_mode": summary["selected_fusion_mode"],
        "g4_passed": bool(summary.get("g4_passed")),
    }


def json_complete(path: Path, source_commit: str, field: str = "completed") -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return document.get(field) is True and report_source(document) == source_commit


@dataclass
class GPUJob:
    name: str
    command: list[str]
    log_path: Path
    marker: Path
    completion_field: str = "completed"
    resume_evidence: Path | None = None
    attempts: int = 0


@dataclass
class RunningJob:
    job: GPUJob
    gpu: int
    process: subprocess.Popen
    handle: object


def job_complete(job: GPUJob, source_commit: str) -> bool:
    return json_complete(job.marker, source_commit, job.completion_field)


def launch_job(job: GPUJob, gpu: int) -> RunningJob:
    command = job.command.copy()
    if job.resume_evidence is not None and job.resume_evidence.exists():
        command.append("--resume")
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8")
    job.attempts += 1
    emit(
        "p5_job_started",
        name=job.name,
        gpu=gpu,
        attempt=job.attempts,
        command=command,
        log=str(job.log_path),
    )
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=cuda_environment(gpu),
    )
    return RunningJob(job, gpu, process, handle)


def run_gpu_jobs(jobs: list[GPUJob], args) -> None:
    pending = [job for job in jobs if not job_complete(job, args.source_commit)]
    for job in jobs:
        if job not in pending:
            emit("p5_job_already_complete", name=job.name, marker=str(job.marker))
    running: list[RunningJob] = []
    while pending or running:
        for active in running.copy():
            return_code = active.process.poll()
            if return_code is None:
                continue
            active.handle.close()
            running.remove(active)
            emit(
                "p5_job_finished",
                name=active.job.name,
                gpu=active.gpu,
                return_code=return_code,
            )
            if return_code == 0 and job_complete(active.job, args.source_commit):
                continue
            if (
                resource_failure(return_code, active.job.log_path)
                and active.job.attempts <= args.maximum_resource_retries
            ):
                emit(
                    "p5_resource_retry_queued",
                    name=active.job.name,
                    attempts=active.job.attempts,
                )
                pending.append(active.job)
                continue
            emit(
                "p5_job_failed",
                name=active.job.name,
                return_code=return_code,
                marker_complete=job_complete(active.job, args.source_commit),
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
                    emit("waiting_for_gpu", phase="p5", states=states)
                break
            running.append(launch_job(pending.pop(0), gpu))
            assigned.add(gpu)
        if pending or running:
            time.sleep(args.poll_seconds)


def run_json_command(
    name: str,
    command: list[str],
    output: Path,
    source_commit: str | None,
    completion_field: str,
) -> dict:
    if output.is_file():
        document = json.loads(output.read_text(encoding="utf-8"))
        source_matches = source_commit is None or report_source(document) == source_commit
        if source_matches and document.get(completion_field) is True:
            emit(f"{name}_already_complete", output=str(output))
            return document
    emit(f"{name}_started", command=command)
    completed = subprocess.run(command, check=False)
    emit(f"{name}_finished", return_code=completed.returncode)
    if completed.returncode != 0 or not output.is_file():
        raise SystemExit(completed.returncode or 4)
    document = json.loads(output.read_text(encoding="utf-8"))
    if source_commit is not None and report_source(document) != source_commit:
        raise ValueError(f"{name} output source commit differs")
    if document.get(completion_field) is not True:
        raise ValueError(f"{name} output is incomplete")
    return document


def archive_incomplete_directory(
    directory: Path, report: Path, source_commit: str
) -> None:
    if not directory.is_dir() or not any(directory.iterdir()):
        return
    if json_complete(report, source_commit):
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = directory.with_name(f"{directory.name}.incomplete.{timestamp}")
    directory.rename(archived)
    emit(
        "p5_incomplete_output_archived",
        source=str(directory),
        destination=str(archived),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--odometry-root", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--g0-report", type=Path, required=True)
    parser.add_argument("--static-doppler-audit", type=Path, required=True)
    parser.add_argument("--g4-summary", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--download-manifest-dir", type=Path, required=True)
    parser.add_argument("--download-verification", type=Path, required=True)
    parser.add_argument("--dense-cache-report", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--lidar-time-reference",
        choices=("none", "start", "center", "end"),
        default="none",
    )
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--required-gpu-name")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()

    validate_gpu_candidates(args.gpu_candidates, args.required_gpu_name)
    python = Path(os.environ.get("PYTHON", "python"))
    tag = args.source_commit[:8]
    args.run_root.mkdir(parents=True, exist_ok=True)

    g4_summary = wait_for_json(args.g4_summary, args.poll_seconds, "waiting_for_g4_release")
    g4 = validate_g4_release(g4_summary, FORMAL_SEEDS)
    parent_summary = wait_for_json(
        g4["parent_summary"], args.poll_seconds, "waiting_for_p5_parent_summary"
    )
    if parent_summary.get("completed") is not True:
        raise ValueError("G2/G3 parent summary is incomplete")
    parent_runs = {
        int(seed): Path(path)
        for seed, path in parent_summary["g3"]["selected_runs"].items()
    }
    if set(parent_runs) != set(FORMAL_SEEDS):
        raise ValueError("P5 parent summary has the wrong seed set")
    for path in (
        g4["preflight_selection"],
        g4["comparison"],
        *parent_runs.values(),
        *g4["formal_runs"].values(),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    emit(
        "p5_test_released",
        g4_passed=g4["g4_passed"],
        selected_arm=g4["selected_arm"],
        selected_fusion_mode=g4["selected_fusion_mode"],
    )

    manifest_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/build_kradar_temporal_manifest.py"),
        "--split",
        str(args.scene_split),
        "--metadata-root",
        str(args.metadata_root),
        "--odometry-root",
        str(args.odometry_root),
        "--output",
        str(args.test_manifest),
        "--partitions",
        "test",
        "--g4-release-summary",
        str(args.g4_summary),
        "--window-length",
        "48",
        "--windows-per-sequence",
        "1",
        "--source-commit",
        args.source_commit,
    ]
    manifest = run_json_command(
        "p5_test_manifest",
        manifest_command,
        args.test_manifest,
        args.source_commit,
        "gate_pass",
    )
    expected_sequences = {int(frame["sequence"]) for frame in manifest["frames"]}
    if len(expected_sequences) != 8 or len(manifest["frames"]) != 384:
        raise ValueError("P5 manifest must contain 8 sequences / 384 frames")

    wait_for_download_completion(
        args.download_manifest_dir / "summary.json",
        expected_sequences,
        args.poll_seconds,
    )
    verify_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/verify_kradar_g0_download.py"),
        "--audit-manifest",
        str(args.test_manifest),
        "--data-root",
        str(args.data_root),
        "--download-manifest-dir",
        str(args.download_manifest_dir),
        "--output",
        str(args.download_verification),
        "--workers",
        "8",
    ]
    if args.download_verification.exists():
        verify_command.append("--overwrite")
    verification = run_json_command(
        "p5_download_verification",
        verify_command,
        args.download_verification,
        None,
        "passed",
    )
    if int(verification["expected_frame_count"]) != 384:
        raise ValueError("P5 download verification covers the wrong frame count")

    dense_job = GPUJob(
        "p5_dense_target_cache",
        [
            str(python),
            "-u",
            str(args.repo_root / "code/scripts/build_kradar_dense_cache.py"),
            "--data-root",
            str(args.data_root),
            "--cache-root",
            str(args.cache_root),
            "--manifest",
            str(args.test_manifest),
            "--scene-split",
            str(args.scene_split),
            "--odometry-root",
            str(args.odometry_root),
            "--g0-report",
            str(args.g0_report),
            "--lidar-time-reference",
            args.lidar_time_reference,
            "--output",
            str(args.dense_cache_report),
            "--device",
            "cuda:0",
            "--required-frames",
            "384",
            "--source-commit",
            args.source_commit,
        ],
        args.run_root / f"p5_dense_cache_{tag}.log",
        args.dense_cache_report,
        resume_evidence=args.dense_cache_report,
    )
    run_gpu_jobs([dense_job], args)

    parent_cache_paths = {
        seed: args.run_root / f"p5_parent_cache_seed{seed}_{tag}"
        for seed in FORMAL_SEEDS
    }
    parent_jobs = []
    for seed in FORMAL_SEEDS:
        output = parent_cache_paths[seed]
        parent_jobs.append(
            GPUJob(
                f"p5_parent_cache_seed{seed}",
                [
                    str(python),
                    "-u",
                    str(args.repo_root / "code/scripts/cache_cube_cycle_predictions.py"),
                    "--data-root",
                    str(args.data_root),
                    "--cache-root",
                    str(args.cache_root),
                    "--manifest",
                    str(args.test_manifest),
                    "--scene-split",
                    str(args.scene_split),
                    "--normalization-stats",
                    str(args.normalization),
                    "--dense-cache-report",
                    str(args.dense_cache_report),
                    "--parent-run",
                    str(parent_runs[seed]),
                    "--output",
                    str(output),
                    "--device",
                    "cuda:0",
                    "--required-frames",
                    "384",
                    "--source-commit",
                    args.source_commit,
                ],
                args.run_root / f"p5_parent_cache_seed{seed}_{tag}.log",
                output / "manifest.json",
                resume_evidence=output / "manifest.json",
            )
        )
    run_gpu_jobs(parent_jobs, args)

    baseline_paths = {
        seed: args.run_root / f"p5_baselines_seed{seed}_{tag}"
        for seed in FORMAL_SEEDS
    }
    baseline_jobs = []
    for seed in FORMAL_SEEDS:
        output = baseline_paths[seed]
        baseline_jobs.append(
            GPUJob(
                f"p5_t0_t3_seed{seed}",
                [
                    str(python),
                    "-u",
                    str(args.repo_root / "code/scripts/eval_g4_temporal_baselines.py"),
                    "--data-root",
                    str(args.data_root),
                    "--cache-root",
                    str(args.cache_root),
                    "--manifest",
                    str(args.test_manifest),
                    "--scene-split",
                    str(args.scene_split),
                    "--normalization-stats",
                    str(args.normalization),
                    "--dense-cache-report",
                    str(args.dense_cache_report),
                    "--parent-run",
                    str(parent_runs[seed]),
                    "--parent-prediction-cache",
                    str(parent_cache_paths[seed]),
                    "--output",
                    str(output),
                    "--partition",
                    "test",
                    "--history-frames",
                    "4",
                    "--device",
                    "cuda:0",
                    "--source-commit",
                    args.source_commit,
                ],
                args.run_root / f"p5_baselines_seed{seed}_{tag}.log",
                output / "report.json",
                resume_evidence=output / "progress.json",
            )
        )

    temporal_paths = {
        seed: args.run_root / f"p5_{g4['selected_arm'].lower()}_seed{seed}_{tag}"
        for seed in FORMAL_SEEDS
    }
    temporal_jobs = []
    for seed in FORMAL_SEEDS:
        output = temporal_paths[seed]
        temporal_jobs.append(
            GPUJob(
                f"p5_temporal_seed{seed}",
                [
                    str(python),
                    "-u",
                    str(args.repo_root / "code/scripts/eval_g4_temporal_rollout.py"),
                    "--data-root",
                    str(args.data_root),
                    "--cache-root",
                    str(args.cache_root),
                    "--manifest",
                    str(args.test_manifest),
                    "--scene-split",
                    str(args.scene_split),
                    "--normalization-stats",
                    str(args.normalization),
                    "--dense-cache-report",
                    str(args.dense_cache_report),
                    "--parent-prediction-cache",
                    str(parent_cache_paths[seed]),
                    "--preflight-selection",
                    str(g4["preflight_selection"]),
                    "--temporal-run",
                    str(g4["formal_runs"][seed]),
                    "--output",
                    str(output),
                    "--partition",
                    "test",
                    "--warmup-iterations",
                    "3",
                    "--device",
                    "cuda:0",
                    "--source-commit",
                    args.source_commit,
                ],
                args.run_root / f"p5_temporal_seed{seed}_{tag}.log",
                output / "report.json",
                resume_evidence=output / "progress.json",
            )
        )
    run_gpu_jobs([*baseline_jobs, *temporal_jobs], args)

    baseline_reports = [baseline_paths[seed] / "report.json" for seed in FORMAL_SEEDS]
    temporal_reports = [temporal_paths[seed] / "report.json" for seed in FORMAL_SEEDS]
    object_root = args.run_root / f"p5_object_velocity_{tag}"
    object_report = object_root / "report.json"
    archive_incomplete_directory(object_root, object_report, args.source_commit)
    object_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/eval_p5_object_velocity.py"),
        "--data-root",
        str(args.data_root),
        "--cache-root",
        str(args.cache_root),
        "--manifest",
        str(args.test_manifest),
        "--scene-split",
        str(args.scene_split),
        "--static-doppler-audit",
        str(args.static_doppler_audit),
        "--baseline-report",
        *[str(path) for path in baseline_reports],
        "--temporal-report",
        *[str(path) for path in temporal_reports],
        "--output",
        str(object_root),
        "--bootstrap-samples",
        "10000",
        "--bootstrap-seed",
        "20260718",
        "--source-commit",
        args.source_commit,
    ]
    run_json_command(
        "p5_object_velocity",
        object_command,
        object_report,
        args.source_commit,
        "completed",
    )

    efficiency_report = args.run_root / f"p5_efficiency_{tag}.json"
    efficiency_job = GPUJob(
        "p5_efficiency",
        [
            str(python),
            "-u",
            str(args.repo_root / "code/scripts/benchmark_p5_efficiency.py"),
            "--data-root",
            str(args.data_root),
            "--cache-root",
            str(args.cache_root),
            "--manifest",
            str(args.test_manifest),
            "--baseline-reports",
            *[str(path) for path in baseline_reports],
            "--temporal-reports",
            *[str(path) for path in temporal_reports],
            "--output",
            str(efficiency_report),
            "--device",
            "cuda:0",
            "--source-commit",
            args.source_commit,
        ],
        args.run_root / f"p5_efficiency_{tag}.log",
        efficiency_report,
    )
    run_gpu_jobs([efficiency_job], args)

    final_root = args.run_root / f"p5_final_{tag}"
    final_report = final_root / "report.json"
    archive_incomplete_directory(final_root, final_report, args.source_commit)
    final_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/compare_p5_final.py"),
        "--manifest",
        str(args.test_manifest),
        "--scene-split",
        str(args.scene_split),
        "--baseline-reports",
        *[str(path) for path in baseline_reports],
        "--temporal-reports",
        *[str(path) for path in temporal_reports],
        "--object-velocity-report",
        str(object_report),
        "--efficiency-report",
        str(efficiency_report),
        "--output",
        str(final_root),
        "--bootstrap-samples",
        "10000",
        "--bootstrap-seed",
        "20260718",
        "--source-commit",
        args.source_commit,
    ]
    final = run_json_command(
        "p5_final_comparison",
        final_command,
        final_report,
        args.source_commit,
        "completed",
    )
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "g4_summary": str(args.g4_summary),
        "g4_summary_sha256": sha256(args.g4_summary),
        "g4_passed": g4["g4_passed"],
        "selected_temporal_arm": g4["selected_arm"],
        "test_manifest": str(args.test_manifest),
        "test_manifest_sha256": sha256(args.test_manifest),
        "download_verification": str(args.download_verification),
        "dense_cache_report": str(args.dense_cache_report),
        "baseline_reports": {
            str(seed): str(baseline_paths[seed] / "report.json")
            for seed in FORMAL_SEEDS
        },
        "temporal_reports": {
            str(seed): str(temporal_paths[seed] / "report.json")
            for seed in FORMAL_SEEDS
        },
        "object_velocity_report": str(object_report),
        "efficiency_report": str(efficiency_report),
        "final_report": str(final_report),
        "final_completed": final["completed"],
        "completed": True,
    }
    summary_path = args.run_root / f"p5_queue_summary_{tag}.json"
    atomic_json(summary_path, summary)
    emit("p5_queue_complete", summary=str(summary_path), report=summary)


if __name__ == "__main__":
    main()
