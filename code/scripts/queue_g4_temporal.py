#!/usr/bin/env python3
"""Queue the preregistered G4 current-Cube temporal experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from queue_g2_g3 import (
    atomic_json,
    available_gpu,
    emit,
    resource_failure,
    tail_text,
)


PREFLIGHT_ARMS = {
    "concat": "t4_concat",
    "cross_attention": "t5_cross_attention",
    "draft_refinement": "t6_draft_refinement",
}


def wait_for_json(path: Path, poll_seconds: int, event: str) -> dict:
    while not path.is_file():
        emit(event, missing=str(path))
        time.sleep(poll_seconds)
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_download_completion(
    path: Path,
    expected_sequences: set[int],
    poll_seconds: int,
) -> dict:
    while True:
        if not path.is_file():
            emit("waiting_for_g4_download", missing=str(path))
            time.sleep(poll_seconds)
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            emit("waiting_for_g4_download_summary", error=f"{type(error).__name__}: {error}")
            time.sleep(poll_seconds)
            continue
        completed = {int(value) for value in document.get("completed_sequences", [])}
        failures = document.get("failures", [])
        if completed == expected_sequences and not failures:
            return document
        emit(
            "waiting_for_g4_download_recovery",
            completed_sequences=len(completed),
            expected_sequences=len(expected_sequences),
            missing_sequences=sorted(expected_sequences - completed),
            unexpected_sequences=sorted(completed - expected_sequences),
            failure_count=len(failures),
        )
        time.sleep(poll_seconds)


def report_source(document: dict) -> str | None:
    return (
        document.get("source_commit")
        or document.get("configuration", {}).get("source_commit")
        or document.get("provenance", {}).get("git_commit")
    )


def json_complete(path: Path, source_commit: str, field: str) -> bool:
    if not path.is_file():
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get(field) is True and report_source(document) == source_commit


def training_complete(run_path: Path, epochs: int, source_commit: str) -> bool:
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
class GPUJob:
    name: str
    command: list[str]
    log_path: Path
    marker: Path
    completion_field: str | None = "completed"
    training_epochs: int | None = None
    resume_evidence: Path | None = None
    attempts: int = 0


@dataclass
class RunningJob:
    job: GPUJob
    gpu: int
    process: subprocess.Popen
    handle: object


def job_complete(job: GPUJob, source_commit: str) -> bool:
    if job.training_epochs is not None:
        return training_complete(job.marker, job.training_epochs, source_commit)
    if job.completion_field is None:
        return job.marker.is_file()
    return json_complete(job.marker, source_commit, job.completion_field)


def job_command(job: GPUJob) -> list[str]:
    command = job.command.copy()
    if job.resume_evidence is not None and job.resume_evidence.exists():
        command.append("--resume")
    return command


def prepare_job(job: GPUJob) -> None:
    if (
        job.training_epochs is None
        or not job.marker.is_dir()
        or not any(job.marker.iterdir())
        or (job.resume_evidence is not None and job.resume_evidence.exists())
    ):
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = job.marker.with_name(f"{job.marker.name}.incomplete.{timestamp}")
    job.marker.rename(archived)
    emit(
        "g4_incomplete_training_archived",
        name=job.name,
        source=str(job.marker),
        destination=str(archived),
    )


def launch_job(job: GPUJob, gpu: int) -> RunningJob:
    prepare_job(job)
    command = job_command(job)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8")
    job.attempts += 1
    emit(
        "g4_job_started",
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
        env=environment,
    )
    return RunningJob(job=job, gpu=gpu, process=process, handle=handle)


def run_gpu_jobs(jobs: list[GPUJob], args) -> None:
    pending = [job for job in jobs if not job_complete(job, args.source_commit)]
    for job in jobs:
        if job not in pending:
            emit("g4_job_already_complete", name=job.name, marker=str(job.marker))
    running: list[RunningJob] = []
    while pending or running:
        for active in running.copy():
            return_code = active.process.poll()
            if return_code is None:
                continue
            active.handle.close()
            running.remove(active)
            emit(
                "g4_job_finished",
                name=active.job.name,
                gpu=active.gpu,
                return_code=return_code,
            )
            if return_code == 0 and job_complete(
                active.job, args.source_commit
            ):
                continue
            if (
                resource_failure(return_code, active.job.log_path)
                and active.job.attempts <= args.maximum_resource_retries
            ):
                emit(
                    "g4_resource_retry_queued",
                    name=active.job.name,
                    attempts=active.job.attempts,
                )
                pending.append(active.job)
                continue
            emit(
                "g4_job_failed",
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
                    emit("waiting_for_gpu", phase="g4", states=states)
                break
            job = pending.pop(0)
            running.append(launch_job(job, gpu))
            assigned.add(gpu)
        if pending or running:
            time.sleep(args.poll_seconds)


def run_cpu_command(
    name: str,
    command: list[str],
    output: Path,
    source_commit: str | None = None,
    completion_field: str | None = None,
    required_outputs: tuple[Path, ...] = (),
) -> dict:
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        source_matches = source_commit is None or report_source(existing) == source_commit
        complete = completion_field is None or existing.get(completion_field) is True
        if source_matches and complete and all(path.is_file() for path in required_outputs):
            emit(f"{name}_already_complete", output=str(output))
            return existing
    emit(f"{name}_started", command=command)
    completed = subprocess.run(command, check=False)
    emit(f"{name}_finished", return_code=completed.returncode)
    if completed.returncode != 0 or not output.is_file():
        raise SystemExit(completed.returncode or 4)
    document = json.loads(output.read_text(encoding="utf-8"))
    if source_commit is not None and report_source(document) != source_commit:
        raise ValueError(f"{name} output source commit differs")
    if completion_field is not None and document.get(completion_field) is not True:
        raise ValueError(f"{name} output is incomplete")
    missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise ValueError(f"{name} required outputs are missing: {missing_outputs}")
    return document


def train_command(
    python: Path,
    repo_root: Path,
    args,
    parent: Path,
    parent_cache: Path,
    output: Path,
    fusion_mode: str,
    seed: int,
    epochs: int,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(repo_root / "code/scripts/train_cube_temporal.py"),
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
        "--dense-cache-report",
        str(args.dense_cache_report),
        "--parent-run",
        str(parent),
        "--parent-prediction-cache",
        str(parent_cache),
        "--output",
        str(output),
        "--fusion-mode",
        fusion_mode,
        "--epochs",
        str(epochs),
        "--joint-start-epoch",
        "6",
        "--seed",
        str(seed),
        "--eval-every",
        "5",
        "--max-eval-pairs",
        "32",
        "--device",
        "cuda:0",
        "--source-commit",
        args.source_commit,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-manifest-dir", type=Path, required=True)
    parser.add_argument("--download-verification", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--odometry-root", type=Path, required=True)
    parser.add_argument("--g0-report", type=Path, required=True)
    parser.add_argument("--g2-g3-summary", type=Path, required=True)
    parser.add_argument("--dense-cache-report", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-candidates", type=int, nargs="+", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260716, 20260717, 20260718]
    )
    parser.add_argument("--required-frames", type=int, default=2160)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=100)
    parser.add_argument("--maximum-resource-retries", type=int, default=3)
    args = parser.parse_args()

    if len(set(args.seeds)) != 3:
        raise ValueError("Formal G4 requires exactly three unique seeds")
    if args.required_frames != 2160:
        raise ValueError("Formal G4 requires the frozen 2160-frame cohort")
    python = Path(os.environ.get("PYTHON", "python"))
    temporal_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_sequences = {
        int(frame["sequence"]) for frame in temporal_manifest["frames"]
    }
    if len(expected_sequences) != 45:
        raise ValueError("Formal G4 manifest requires exactly 45 sequences")
    download_summary = wait_for_download_completion(
        args.download_manifest_dir / "summary.json",
        expected_sequences,
        args.poll_seconds,
    )
    emit(
        "g4_download_finished",
        completed_sequences=len(download_summary.get("completed_sequences", [])),
        failures=download_summary.get("failures", []),
    )
    verify_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/verify_kradar_g0_download.py"),
        "--audit-manifest",
        str(args.manifest),
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
    verification = run_cpu_command(
        "g4_download_verification",
        verify_command,
        args.download_verification,
        completion_field="passed",
    )
    if verification.get("passed") is not True:
        raise SystemExit("G4 temporal download did not pass CRC verification")
    if int(verification["expected_frame_count"]) != args.required_frames:
        raise ValueError("G4 download verification covers the wrong frame count")

    g0_report = wait_for_json(args.g0_report, args.poll_seconds, "waiting_for_g0")
    if g0_report.get("aggregate", {}).get("gate_pass") is not True:
        raise SystemExit("G0 did not pass; G4 cannot build dense targets")
    parent_summary = wait_for_json(
        args.g2_g3_summary, args.poll_seconds, "waiting_for_g2_g3"
    )
    if parent_summary.get("completed") is not True:
        raise SystemExit("G2/G3 queue is incomplete; G4 parent is unavailable")
    selected_runs = {
        int(seed): Path(path)
        for seed, path in parent_summary["g3"]["selected_runs"].items()
    }
    if set(selected_runs) != set(args.seeds):
        raise ValueError("G4 parent summary has the wrong seed set")
    for seed, run in selected_runs.items():
        required = (
            run / "config.json",
            run / "best.pt",
            run / "best_validation_metrics.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"G4 parent artifacts are missing for seed {seed}: {missing}"
            )
    emit(
        "g4_dependencies_passed",
        parent_arm=parent_summary["g3"]["selected_arm"],
        g3_passed=parent_summary["g3"]["passed"],
        selected_runs={str(seed): str(path) for seed, path in selected_runs.items()},
    )

    tag = args.source_commit[:8]
    prior_report = args.run_root / f"g4_temporal_prior_{tag}.json"
    prior_job = GPUJob(
        name="temporal_prior_verification",
        command=[
            str(python),
            "-u",
            str(args.repo_root / "code/scripts/verify_temporal_prior.py"),
            "--output",
            str(prior_report),
            "--device",
            "cuda:0",
            "--source-commit",
            args.source_commit,
        ],
        log_path=args.run_root / f"g4_temporal_prior_{tag}.log",
        marker=prior_report,
        completion_field="passed",
    )
    run_gpu_jobs([prior_job], args)

    dense_job = GPUJob(
        name="dense_target_cache",
        command=[
            str(python),
            "-u",
            str(args.repo_root / "code/scripts/build_kradar_dense_cache.py"),
            "--data-root",
            str(args.data_root),
            "--cache-root",
            str(args.cache_root),
            "--manifest",
            str(args.manifest),
            "--scene-split",
            str(args.scene_split),
            "--odometry-root",
            str(args.odometry_root),
            "--g0-report",
            str(args.g0_report),
            "--output",
            str(args.dense_cache_report),
            "--device",
            "cuda:0",
            "--required-frames",
            str(args.required_frames),
            "--source-commit",
            args.source_commit,
        ],
        log_path=args.run_root / f"g4_dense_cache_{tag}.log",
        marker=args.dense_cache_report,
        resume_evidence=args.dense_cache_report,
    )
    run_gpu_jobs([dense_job], args)

    parent_cache_paths = {
        seed: args.run_root / f"g4_parent_cache_seed{seed}_{tag}"
        for seed in args.seeds
    }
    parent_cache_jobs = []
    for seed in args.seeds:
        output = parent_cache_paths[seed]
        parent_cache_jobs.append(
            GPUJob(
                name=f"parent_prediction_cache_seed{seed}",
                command=[
                    str(python),
                    "-u",
                    str(
                        args.repo_root
                        / "code/scripts/cache_cube_cycle_predictions.py"
                    ),
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
                    "--dense-cache-report",
                    str(args.dense_cache_report),
                    "--parent-run",
                    str(selected_runs[seed]),
                    "--output",
                    str(output),
                    "--device",
                    "cuda:0",
                    "--required-frames",
                    str(args.required_frames),
                    "--source-commit",
                    args.source_commit,
                ],
                log_path=args.run_root
                / f"g4_parent_cache_seed{seed}_{tag}.log",
                marker=output / "manifest.json",
                resume_evidence=output / "manifest.json",
            )
        )
    run_gpu_jobs(parent_cache_jobs, args)

    baseline_paths = {
        seed: args.run_root / f"g4_baselines_seed{seed}_{tag}"
        for seed in args.seeds
    }
    baseline_jobs = []
    for seed in args.seeds:
        output = baseline_paths[seed]
        baseline_jobs.append(
            GPUJob(
                name=f"t0_t3_baselines_seed{seed}",
                command=[
                    str(python),
                    "-u",
                    str(
                        args.repo_root
                        / "code/scripts/eval_g4_temporal_baselines.py"
                    ),
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
                    "--dense-cache-report",
                    str(args.dense_cache_report),
                    "--parent-run",
                    str(selected_runs[seed]),
                    "--parent-prediction-cache",
                    str(parent_cache_paths[seed]),
                    "--output",
                    str(output),
                    "--history-frames",
                    "4",
                    "--device",
                    "cuda:0",
                    "--source-commit",
                    args.source_commit,
                ],
                log_path=args.run_root / f"g4_baselines_seed{seed}_{tag}.log",
                marker=output / "report.json",
                resume_evidence=output / "progress.json",
            )
        )

    preflight_seed = min(args.seeds)
    preflight_paths = {
        mode: args.run_root
        / f"g4_preflight_{label}_seed{preflight_seed}_{tag}"
        for mode, label in PREFLIGHT_ARMS.items()
    }
    preflight_jobs = [
        GPUJob(
            name=f"preflight_{mode}",
            command=train_command(
                python,
                args.repo_root,
                args,
                selected_runs[preflight_seed],
                parent_cache_paths[preflight_seed],
                output,
                mode,
                preflight_seed,
                5,
            ),
            log_path=args.run_root
            / f"g4_preflight_{PREFLIGHT_ARMS[mode]}_seed{preflight_seed}_{tag}.log",
            marker=output,
            completion_field=None,
            training_epochs=5,
            resume_evidence=output / "last.pt",
        )
        for mode, output in preflight_paths.items()
    ]
    run_gpu_jobs([*baseline_jobs, *preflight_jobs], args)

    selection_path = args.run_root / f"g4_preflight_selection_{tag}.json"
    selection_markdown = args.run_root / f"g4_preflight_selection_{tag}.md"
    selection_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/select_g4_temporal_preflight.py"),
        "--concat-run",
        str(preflight_paths["concat"]),
        "--cross-attention-run",
        str(preflight_paths["cross_attention"]),
        "--draft-refinement-run",
        str(preflight_paths["draft_refinement"]),
        "--output",
        str(selection_path),
        "--decision-markdown",
        str(selection_markdown),
        "--required-seed",
        str(preflight_seed),
        "--source-commit",
        args.source_commit,
    ]
    if selection_path.exists() or selection_markdown.exists():
        selection_command.append("--overwrite")
    selection = run_cpu_command(
        "g4_preflight_selection",
        selection_command,
        selection_path,
        args.source_commit,
        "completed",
        (selection_markdown,),
    )
    selected_mode = selection["selected_fusion_mode"]
    selected_arm = selection["selected_arm"].lower()
    emit(
        "g4_preflight_selected",
        selected_arm=selection["selected_arm"],
        selected_fusion_mode=selected_mode,
    )

    formal_paths = {
        seed: args.run_root / f"g4_{selected_arm}_{selected_mode}_seed{seed}_{tag}"
        for seed in args.seeds
    }
    formal_jobs = [
        GPUJob(
            name=f"formal_{selected_mode}_seed{seed}",
            command=train_command(
                python,
                args.repo_root,
                args,
                selected_runs[seed],
                parent_cache_paths[seed],
                formal_paths[seed],
                selected_mode,
                seed,
                20,
            ),
            log_path=args.run_root
            / f"g4_{selected_arm}_{selected_mode}_seed{seed}_{tag}.log",
            marker=formal_paths[seed],
            completion_field=None,
            training_epochs=20,
            resume_evidence=formal_paths[seed] / "last.pt",
        )
        for seed in args.seeds
    ]
    run_gpu_jobs(formal_jobs, args)

    rollout_paths = {
        seed: args.run_root
        / f"g4_rollout_{selected_arm}_{selected_mode}_seed{seed}_{tag}"
        for seed in args.seeds
    }
    rollout_jobs = []
    for seed in args.seeds:
        output = rollout_paths[seed]
        rollout_jobs.append(
            GPUJob(
                name=f"strict_rollout_seed{seed}",
                command=[
                    str(python),
                    "-u",
                    str(
                        args.repo_root
                        / "code/scripts/eval_g4_temporal_rollout.py"
                    ),
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
                    "--dense-cache-report",
                    str(args.dense_cache_report),
                    "--parent-prediction-cache",
                    str(parent_cache_paths[seed]),
                    "--preflight-selection",
                    str(selection_path),
                    "--temporal-run",
                    str(formal_paths[seed]),
                    "--output",
                    str(output),
                    "--warmup-iterations",
                    "3",
                    "--device",
                    "cuda:0",
                    "--source-commit",
                    args.source_commit,
                ],
                log_path=args.run_root
                / f"g4_rollout_{selected_arm}_{selected_mode}_seed{seed}_{tag}.log",
                marker=output / "report.json",
                resume_evidence=output / "progress.json",
            )
        )
    run_gpu_jobs(rollout_jobs, args)

    comparison_path = args.run_root / f"g4_comparison_{tag}.json"
    comparison_markdown = args.run_root / f"g4_comparison_{tag}.md"
    comparison_command = [
        str(python),
        "-u",
        str(args.repo_root / "code/scripts/compare_g4_temporal.py"),
        "--baseline-reports",
        *[str(baseline_paths[seed] / "report.json") for seed in args.seeds],
        "--temporal-reports",
        *[str(rollout_paths[seed] / "report.json") for seed in args.seeds],
        "--output",
        str(comparison_path),
        "--decision-markdown",
        str(comparison_markdown),
        "--bootstrap-samples",
        "10000",
        "--bootstrap-seed",
        "20260718",
        "--source-commit",
        args.source_commit,
    ]
    if comparison_path.exists() or comparison_markdown.exists():
        comparison_command.append("--overwrite")
    comparison = run_cpu_command(
        "g4_comparison",
        comparison_command,
        comparison_path,
        args.source_commit,
        "completed",
        (comparison_markdown,),
    )
    final_report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "download_verification": str(args.download_verification),
        "dense_cache_report": str(args.dense_cache_report),
        "parent_summary": str(args.g2_g3_summary),
        "parent_arm": parent_summary["g3"]["selected_arm"],
        "preflight_selection": str(selection_path),
        "selected_arm": selection["selected_arm"],
        "selected_fusion_mode": selected_mode,
        "baseline_reports": {
            str(seed): str(path / "report.json")
            for seed, path in baseline_paths.items()
        },
        "formal_runs": {str(seed): str(path) for seed, path in formal_paths.items()},
        "rollout_reports": {
            str(seed): str(path / "report.json")
            for seed, path in rollout_paths.items()
        },
        "comparison": str(comparison_path),
        "g4_passed": bool(comparison["decision"]["g4_passed"]),
        "completed": True,
    }
    final_path = args.run_root / f"g4_queue_summary_{tag}.json"
    atomic_json(final_path, final_report)
    emit("g4_queue_complete", summary=str(final_path), report=final_report)


if __name__ == "__main__":
    main()
