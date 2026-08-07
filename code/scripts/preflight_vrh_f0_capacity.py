#!/usr/bin/env python3
"""Run the frozen train-only VRH-F0 paired capacity gate on one H200."""

from __future__ import annotations

import argparse
import ast
import gc
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes  # noqa: E402
from eval.dense_geometry import geometry_report  # noqa: E402
from eval.rald_wce_stage0 import tensor_sha256  # noqa: E402
from eval.vrh_f0_decode import (  # noqa: E402
    decode_canonical_streams,
    export_flat_control,
    export_sequential_frontier,
    write_canonical_streams,
    write_exposure_trace,
    write_selected_events,
)
from eval.vrh_f0_fit import (  # noqa: E402
    fit_gt_oracle_marks,
    load_target_only,
    write_fit_sidecar,
)
from eval.vrh_f0_metrics import (  # noqa: E402
    build_target_return_groups,
    evaluate_first_later_returns,
    selected_events_inside_cell_interiors,
    write_return_assignments,
)
from eval.vrh_f0_support import (  # noqa: E402
    CELL_COUNT,
    MINIMUM_DISTANCE_M,
    NATURAL_RANGE_UPPER_M,
    SUBRAY_COUNT,
    build_vrh_support,
    commit_support,
    write_model_marks,
    write_vrh_support,
)
from scripts.preflight_qlocal_distributional_risk import (  # noqa: E402
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_NORMALIZATION_SHA256,
    EXPECTED_SCENE_SPLIT_SHA256,
    FIT_IDENTITIES,
    UNSEEN_IDENTITIES,
    cache_path,
    frame_identity,
    git_output,
    require_h200_with_identity,
    resolve_frozen_cohort,
    sha256_file,
    verify_source_tree,
)
from scripts.preflight_qlocal_global_export import (  # noqa: E402
    retention_gate,
    target_stratum_retention,
)
from scripts.train_rald_wce_pilot import validate_frozen_inputs  # noqa: E402


PROTOCOL = "g1_vrh_f0_variable_return_capacity_v1"
PROTOCOL_SHA256 = (
    "78c62f29e8f325287d79eefea141bf2b014f1388448d4ed1a3b5365071cd7946"
)
PROTOCOL_FREEZE_COMMIT = "d034dcaba1d3c015239a9623b281c5b01fafe2c6"
ARCHIVED_F0R_SHA256 = (
    "505644667ba3f73b58ee138608307e82632fe92bc3336f538c74767ab9c148ae"
)
ARCHIVED_F0R_SOURCE_COMMIT = "34579a245efecb0a10baeec3bcd1929a4744d502"
CORRECTED_DENSE_GEOMETRY_SHA256 = (
    "e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68"
)
ORACLE_CHAMFER_GATE_M = 0.8
ORACLE_OUTLIER_GATE = 0.05
RETENTION_TOLERANCE = 1e-6
EXACT_OUTPUT_COUNT = 10_000
MAX_WALL_SECONDS = 2 * 60 * 60
MAX_CUDA_BYTES = 60 * 1024**3
EXPECTED_FRAME_COUNT = 12
RANGE_BOUNDS_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
RANGE_LABELS = ("range_0_30m", "range_30_60m", "range_60_120m")


def source_hashes(repo: Path) -> dict[str, str]:
    paths = (
        "code/eval/dense_geometry.py",
        "code/eval/vrh_f0_support.py",
        "code/eval/vrh_f0_fit.py",
        "code/eval/vrh_f0_decode.py",
        "code/eval/vrh_f0_metrics.py",
        "code/scripts/preflight_qlocal_distributional_risk.py",
        "code/scripts/preflight_qlocal_global_export.py",
        "code/scripts/preflight_vrh_f0_capacity.py",
        "docs/vrh_f0_variable_return_capacity_protocol.md",
    )
    return {path: sha256_file(repo / path) for path in paths}


def archived_f0r_frame_map(document: dict[str, Any]) -> dict[tuple[int, int], dict]:
    if document.get("protocol") != "g1_qlocal_f0r_global_export_capacity_v1":
        raise ValueError("VRH archived F0R protocol changed")
    if document.get("source_commit") != ARCHIVED_F0R_SOURCE_COMMIT:
        raise ValueError("VRH archived F0R source changed")
    if document.get("status") != "qlocal_f0r_capacity_no_go":
        raise ValueError("VRH archived F0R terminal changed")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != EXPECTED_FRAME_COUNT:
        raise ValueError("VRH archived F0R report must contain exactly 12 frames")
    mapping = {
        (int(frame["sequence"]), int(frame["radar_index"])): frame
        for frame in frames
    }
    if len(mapping) != EXPECTED_FRAME_COUNT:
        raise ValueError("VRH archived F0R identities are not unique")
    expected = set(FIT_IDENTITIES + UNSEEN_IDENTITIES)
    if set(mapping) != expected:
        raise ValueError("VRH archived F0R cohort changed")
    return mapping


def frozen_api_audit(repo: Path) -> dict[str, Any]:
    signatures = {
        "fit_gt_oracle_marks": tuple(inspect.signature(fit_gt_oracle_marks).parameters),
        "decode_canonical_streams": tuple(
            inspect.signature(decode_canonical_streams).parameters
        ),
        "export_sequential_frontier": tuple(
            inspect.signature(export_sequential_frontier).parameters
        ),
        "export_flat_control": tuple(inspect.signature(export_flat_control).parameters),
    }
    expected = {
        "fit_gt_oracle_marks": ("support", "target_xyz_confidence", "query_chunk_size"),
        "decode_canonical_streams": ("support", "model_marks"),
        "export_sequential_frontier": (
            "canonical_streams",
            "exact_count",
            "minimum_distance_m",
        ),
        "export_flat_control": (
            "canonical_streams",
            "exact_count",
            "minimum_distance_m",
        ),
    }
    signature_checks = {name: signatures[name] == value for name, value in expected.items()}
    decode_path = repo / "code/eval/vrh_f0_decode.py"
    tree = ast.parse(decode_path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    forbidden_imports = sorted(
        name for name in imports if name.endswith("vrh_f0_fit") or name.endswith("vrh_f0_metrics")
    )
    checks = {
        **signature_checks,
        "decoder_imports_no_fitter_or_metric_sidecar": not forbidden_imports,
    }
    return {
        "signatures": {name: list(value) for name, value in signatures.items()},
        "decoder_imports": sorted(imports),
        "forbidden_decoder_imports": forbidden_imports,
        "checks": checks,
        "passed": all(checks.values()),
    }


def begin_transaction(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"VRH output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )


def publish_transaction(staging: Path, output_dir: Path, report: dict[str, Any]) -> Path:
    report_path = staging / "preflight.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    os.replace(staging, output_dir)
    return output_dir / "preflight.json"


def _relative_file_report(staging: Path, path: Path, digest: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(staging)),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _fixed_reference(frame: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    rows = frame["global_gt_nearest_capacity_oracle"]["target_stratum_retention"]
    return {label: dict(row["fixed"]) for label, row in rows.items()}


def _selected_by_range(selected_xyz: np.ndarray) -> dict[str, int]:
    radius = np.linalg.norm(selected_xyz, axis=1)
    return {
        label: int(((radius >= lower) & (radius < upper)).sum())
        for label, (lower, upper) in zip(RANGE_LABELS, RANGE_BOUNDS_M)
    }


def _evaluate_arm(
    *,
    export: Any,
    replay: Any,
    support: Any,
    target: np.ndarray,
    target_groups: list[Any],
    fixed_reference: dict[str, dict[str, float | int]],
    fit_report: dict[str, Any],
    api_audit: dict[str, Any],
    device: torch.device,
    frame_dir: Path,
    staging: Path,
) -> dict[str, Any]:
    replay_checks = {
        "stream_sha256": replay.streams_sha256 == export.streams_sha256,
        "trace_sha256": replay.trace.digest_sha256 == export.trace.digest_sha256,
        "selected_set_sha256": replay.selected_set_sha256 == export.selected_set_sha256,
        "accepted_cell_id_sha256": replay.accepted_cell_id_sha256
        == export.accepted_cell_id_sha256,
        "selected_cell_ids": np.array_equal(
            replay.selected_cell_id, export.selected_cell_id
        ),
    }
    return_result = evaluate_first_later_returns(target_groups, export)
    assignment_path = frame_dir / f"{export.arm}_return_assignments.bin"
    assignment_file_hash = write_return_assignments(
        assignment_path, return_result, export.arm
    )
    if assignment_file_hash != return_result.digest_sha256:
        raise AssertionError("VRH return-assignment file hash changed")

    trace_path = frame_dir / f"{export.arm}_exposure_trace.bin"
    trace_file_hash = write_exposure_trace(trace_path, export.trace, export.arm)
    if trace_file_hash != export.trace.digest_sha256:
        raise AssertionError("VRH exposure-trace file hash changed")
    selected_path = frame_dir / f"{export.arm}_selected_events.bin"
    selected_file_hash = write_selected_events(selected_path, export)
    if selected_file_hash != export.selected_set_sha256:
        raise AssertionError("VRH selected-event file hash changed")

    if export.selected_count > 0:
        prediction = torch.from_numpy(export.selected_xyz).to(
            device=device, dtype=torch.float32
        )
        target_tensor = torch.from_numpy(target.astype(np.float32, copy=False)).to(
            device=device
        )
        geometry = geometry_report(
            prediction,
            target_tensor[:, :3],
            target_weight=target_tensor[:, 3],
        )
        current_retention = target_stratum_retention(prediction, target_tensor)
        retention, retention_passed = retention_gate(
            fixed_reference,
            current_retention,
        )
        del prediction, target_tensor
    else:
        geometry = None
        retention = None
        retention_passed = False

    structural_checks = {
        "exact_10000": export.selected_count == EXACT_OUTPUT_COUNT,
        "unique_stable_cell_ids": int(export.report["unique_selected_cell_count"])
        == export.selected_count,
        "unique_finite_xyz": bool(export.report["finite_xyz"])
        and int(export.report["unique_selected_xyz_count"]) == export.selected_count,
        "inside_source_cell_interiors": selected_events_inside_cell_interiors(
            support, export
        ),
        "all_fitted_marks_inside_source_cell_interiors": bool(
            fit_report["mark_interior_verification"][
                "all_marks_inside_source_cell_interiors"
            ]
        ),
        "decoder_export_replay": all(replay_checks.values()),
        "independent_kdtree_minimum_spacing": bool(
            export.report["independent_kdtree_spacing_passed"]
        ),
        "zero_positive_target_weight_outside_support": float(
            fit_report["target_support"][
                "positive_confidence_out_of_support_weight"
            ]
        )
        == 0.0,
        "target_free_decoder_exporter_api": bool(api_audit["passed"]),
        "no_padding_jitter_duplicate_or_range_quota": all(
            (
                export.report["padding_used"] is False,
                export.report["random_jitter_used"] is False,
                export.report["duplicate_fallback_used"] is False,
                export.report["range_quota_used"] is False,
            )
        ),
    }
    gate = {
        "all_structural_checks": all(structural_checks.values()),
        "chamfer_at_most_0p8m": geometry is not None
        and float(geometry["chamfer_m"]) <= ORACLE_CHAMFER_GATE_M,
        "outlier_at_most_5pct": geometry is not None
        and float(geometry["outlier_fraction_2m"]) <= ORACLE_OUTLIER_GATE,
        "all_target_strata_retained": retention_passed,
        "all_first_later_classes_passed": bool(
            return_result.report["all_nonempty_classes_passed"]
        ),
    }
    return {
        "passed": all(gate.values()),
        "gate": gate,
        "structural_checks": structural_checks,
        "replay_checks": replay_checks,
        "geometry": geometry,
        "target_stratum_retention": retention,
        "first_later_returns": return_result.report,
        "export": export.report,
        "selected_by_range": _selected_by_range(export.selected_xyz),
        "hashes": {
            "canonical_stream_sha256": export.streams_sha256,
            "exposure_trace_sha256": export.trace.digest_sha256,
            "selected_set_sha256": export.selected_set_sha256,
            "accepted_cell_id_sha256": export.accepted_cell_id_sha256,
            "return_assignment_sha256": return_result.digest_sha256,
        },
        "files": {
            "exposure_trace": _relative_file_report(
                staging, trace_path, trace_file_hash
            ),
            "selected_events": _relative_file_report(
                staging, selected_path, selected_file_hash
            ),
            "return_assignments": _relative_file_report(
                staging, assignment_path, assignment_file_hash
            ),
        },
        "decoder_target_input": False,
        "exporter_target_input": False,
        "metric_evaluator_target_input": True,
    }


def _frame_report(
    *,
    position: int,
    record: dict[str, Any],
    group: str,
    cache_file: Path,
    archived: dict[str, Any],
    support: Any,
    support_commit: Any,
    api_audit: dict[str, Any],
    device: torch.device,
    staging: Path,
) -> dict[str, Any]:
    frame_started = time.monotonic()
    target = load_target_only(cache_file, support_commit=support_commit)
    target_tensor_sha = tensor_sha256(
        torch.from_numpy(target.astype(np.float32, copy=False))
    )
    expected_target_sha = archived["raw_inputs"]["target_tensor_sha256"]
    if target_tensor_sha != expected_target_sha:
        raise ValueError("VRH target tensor hash changed from archived F0R")
    fit = fit_gt_oracle_marks(support, target, query_chunk_size=131_072)
    frame_dir = staging / "frames" / f"{position:02d}_{record['sequence']}_{record['radar_index']}"
    frame_dir.mkdir(parents=True, exist_ok=False)

    marks_path = frame_dir / "model_marks.bin"
    marks_file_hash = write_model_marks(marks_path, fit.model_marks, support)
    if marks_file_hash != fit.report["model_marks_sha256"]:
        raise AssertionError("VRH ModelMarks file hash changed")
    sidecar_path = frame_dir / "fit_sidecar.bin"
    sidecar_file_hash = write_fit_sidecar(
        sidecar_path, fit.fit_sidecar, support
    )
    if sidecar_file_hash != fit.report["fit_sidecar_sha256"]:
        raise AssertionError("VRH FitSidecar file hash changed")

    streams = decode_canonical_streams(support, fit.model_marks)
    streams_path = frame_dir / "canonical_streams.bin"
    streams_file_hash = write_canonical_streams(streams_path, streams)
    if streams_file_hash != streams.digest_sha256:
        raise AssertionError("VRH canonical-stream file hash changed")
    decision = export_sequential_frontier(streams)
    control = export_flat_control(streams)

    replay_streams = decode_canonical_streams(support, fit.model_marks)
    if replay_streams.digest_sha256 != streams.digest_sha256:
        raise AssertionError("VRH decoder replay changed canonical streams")
    decision_replay = export_sequential_frontier(replay_streams)
    control_replay = export_flat_control(replay_streams)
    groups = build_target_return_groups(support, fit.canonical_targets)
    fixed_reference = _fixed_reference(archived)
    decision_report = _evaluate_arm(
        export=decision,
        replay=decision_replay,
        support=support,
        target=target,
        target_groups=groups,
        fixed_reference=fixed_reference,
        fit_report=fit.report,
        api_audit=api_audit,
        device=device,
        frame_dir=frame_dir,
        staging=staging,
    )
    control_report = _evaluate_arm(
        export=control,
        replay=control_replay,
        support=support,
        target=target,
        target_groups=groups,
        fixed_reference=fixed_reference,
        fit_report=fit.report,
        api_audit=api_audit,
        device=device,
        frame_dir=frame_dir,
        staging=staging,
    )
    utility_differences = [
        key
        for key in decision_report["gate"]
        if bool(decision_report["gate"][key])
        and not bool(control_report["gate"][key])
    ]
    return {
        "cohort_position": position,
        "sequence": int(record["sequence"]),
        "radar_index": int(record["radar_index"]),
        "partition": str(record["partition"]),
        "group": group,
        "raw_inputs": {
            "target_cache_path": str(cache_file.resolve()),
            "target_cache_sha256": sha256_file(cache_file),
            "target_tensor_sha256": target_tensor_sha,
            "radar_cube_accessed": False,
            "cfar_array_accessed": False,
        },
        "fit": fit.report,
        "canonical_stream": {
            "event_count": streams.event_count,
            "nonempty_ray_count": int((np.diff(streams.ray_offsets) > 0).sum()),
            "maximum_return_count": int(np.diff(streams.ray_offsets).max(initial=0)),
            "sha256": streams.digest_sha256,
        },
        "decision_sequential_frontier": decision_report,
        "control_flat_exposure": control_report,
        "paired_control": {
            "same_canonical_stream_sha256": decision.streams_sha256
            == control.streams_sha256,
            "accepted_cell_id_hash_differs": decision.accepted_cell_id_sha256
            != control.accepted_cell_id_sha256,
            "decision_depth_ge_2_count": decision.report[
                "accepted_depth_ge_2_count"
            ],
            "decision_pass_control_fail_gate_keys": utility_differences,
        },
        "files": {
            "model_marks": _relative_file_report(
                staging, marks_path, marks_file_hash
            ),
            "fit_sidecar": _relative_file_report(
                staging, sidecar_path, sidecar_file_hash
            ),
            "canonical_streams": _relative_file_report(
                staging, streams_path, streams_file_hash
            ),
        },
        "elapsed_seconds": time.monotonic() - frame_started,
    }


def terminal_status(
    *,
    implementation_valid: bool,
    resource_valid: bool,
    all_decision_frames_passed: bool,
    renewal_activity: bool,
    renewal_utility: bool,
) -> str:
    if not resource_valid:
        return "vrh_f0_resource_invalid"
    if not implementation_valid:
        return "vrh_f0_implementation_invalid"
    if not all_decision_frames_passed:
        return "vrh_f0_capacity_no_go"
    if renewal_activity and renewal_utility:
        return "vrh_f0_capacity_passed"
    return "vrh_f0_lattice_only"


def _load_train_records(manifest: Path) -> list[dict[str, Any]]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    return [row for row in document["frames"] if row["partition"] == "train"]


def run_gate(args: argparse.Namespace, staging: Path) -> dict[str, Any]:
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    verify_source_tree(repo, args.source_commit)
    device, gpu = require_h200_with_identity(args.device)
    torch.manual_seed(20260716)
    np.random.seed(20260716)
    protocol_path = repo / "docs/vrh_f0_variable_return_capacity_protocol.md"
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("VRH frozen protocol bytes changed")
    protocol_blob = git_output(repo, "hash-object", str(protocol_path.relative_to(repo)))
    frozen_blob = git_output(
        repo,
        "rev-parse",
        f"{PROTOCOL_FREEZE_COMMIT}:docs/vrh_f0_variable_return_capacity_protocol.md",
    )
    if protocol_blob != frozen_blob:
        raise ValueError("VRH protocol blob differs from the audited freeze commit")
    api_audit = frozen_api_audit(repo)
    if not api_audit["passed"]:
        raise ValueError(f"VRH target-boundary API audit failed: {api_audit}")

    input_hashes, manifest_counts = validate_frozen_inputs(
        args.manifest,
        args.scene_split,
        args.normalization,
        repo,
    )
    expected_inputs = {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "scene_split_sha256": EXPECTED_SCENE_SPLIT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "dense_geometry_evaluator_sha256": CORRECTED_DENSE_GEOMETRY_SHA256,
    }
    for key, expected in expected_inputs.items():
        if input_hashes[key] != expected:
            raise ValueError(f"VRH frozen input changed: {key}")
    if sha256_file(args.archived_f0r) != ARCHIVED_F0R_SHA256:
        raise ValueError("VRH archived F0R report bytes changed")
    archived_document = json.loads(args.archived_f0r.read_text(encoding="utf-8"))
    archived_frames = archived_f0r_frame_map(archived_document)

    records = _load_train_records(args.manifest)
    fit_indices, unseen_indices, _ = resolve_frozen_cohort(records)
    cohort_indices = fit_indices + unseen_indices
    cohort_binding: list[dict[str, Any]] = []
    for index in cohort_indices:
        record = records[index]
        identity = frame_identity(record)
        archived = archived_frames[identity]
        path = cache_path(args.cache_root, record)
        observed_hash = sha256_file(path)
        expected_path = archived["raw_inputs"]["target_cache_path"]
        expected_hash = archived["raw_inputs"]["target_cache_sha256"]
        if str(path.resolve()) != expected_path or observed_hash != expected_hash:
            raise ValueError(f"VRH target-cache binding changed for {identity}")
        cohort_binding.append(
            {
                "identity": list(identity),
                "group": "fit" if index in fit_indices else "unseen_train",
                "target_cache_path": str(path.resolve()),
                "target_cache_sha256": observed_hash,
                "target_tensor_sha256": archived["raw_inputs"][
                    "target_tensor_sha256"
                ],
            }
        )

    resources = args.data_root / "resources"
    axis_files = {
        "info_arr.mat": sha256_file(resources / "info_arr.mat"),
        "arr_doppler.mat": sha256_file(resources / "arr_doppler.mat"),
    }
    axes = load_axes(resources)
    support = build_vrh_support(axes)
    if support.cell_count != CELL_COUNT or support.ray_count != SUBRAY_COUNT:
        raise AssertionError("VRH support cardinality changed")
    if support.range_axis.edges[-1] != NATURAL_RANGE_UPPER_M:
        raise AssertionError("VRH natural range support changed")
    support_path = staging / "support.bin"
    support_file_hash = write_vrh_support(support_path, support)
    support_commitment = commit_support(support)

    frames: list[dict[str, Any]] = []
    for position, index in enumerate(cohort_indices):
        if time.monotonic() - started > MAX_WALL_SECONDS:
            raise TimeoutError("VRH exceeded the frozen two-hour wall budget")
        record = records[index]
        identity = frame_identity(record)
        frames.append(
            _frame_report(
                position=position,
                record=record,
                group="fit" if index in fit_indices else "unseen_train",
                cache_file=cache_path(args.cache_root, record),
                archived=archived_frames[identity],
                support=support,
                support_commit=support_commitment,
                api_audit=api_audit,
                device=device,
                staging=staging,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if elapsed > MAX_WALL_SECONDS:
        raise TimeoutError("VRH exceeded the frozen two-hour wall budget")
    if max(peak_allocated, peak_reserved) > MAX_CUDA_BYTES:
        raise MemoryError("VRH exceeded the frozen 60-GiB CUDA memory budget")
    complete = len(frames) == EXPECTED_FRAME_COUNT
    all_decision_passed = complete and all(
        frame["decision_sequential_frontier"]["passed"] for frame in frames
    )
    same_streams = complete and all(
        frame["paired_control"]["same_canonical_stream_sha256"]
        for frame in frames
    )
    selected_hash_differs = any(
        frame["paired_control"]["accepted_cell_id_hash_differs"]
        for frame in frames
    )
    decision_depth_ge_2 = any(
        int(frame["paired_control"]["decision_depth_ge_2_count"]) > 0
        for frame in frames
    )
    renewal_activity = same_streams and selected_hash_differs and decision_depth_ge_2
    renewal_utility = all_decision_passed and any(
        bool(frame["paired_control"]["decision_pass_control_fail_gate_keys"])
        for frame in frames
    )
    status = terminal_status(
        implementation_valid=True,
        resource_valid=True,
        all_decision_frames_passed=all_decision_passed,
        renewal_activity=renewal_activity,
        renewal_utility=renewal_utility,
    )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "protocol_git_blob": protocol_blob,
        "source_commit": args.source_commit,
        "status": status,
        "passed": status == "vrh_f0_capacity_passed",
        "decision": {
            "implementation_valid": True,
            "resource_valid": True,
            "complete_12_frame_two_arm_run": complete,
            "all_decision_frames_passed": all_decision_passed,
            "renewal_activity_certified": renewal_activity,
            "renewal_utility_certified": renewal_utility,
            "learned_renewal_route_authorized": status
            == "vrh_f0_capacity_passed",
            "lattice_only_evidence": status == "vrh_f0_lattice_only",
            "partial_transport_gate_next": status
            in ("vrh_f0_lattice_only", "vrh_f0_capacity_no_go"),
        },
        "source_hashes": source_hashes(repo),
        "api_boundary_audit": api_audit,
        "input_binding": {
            "frozen_inputs": input_hashes,
            "manifest_counts": manifest_counts,
            "archived_f0r_sha256": ARCHIVED_F0R_SHA256,
            "axis_source_sha256": axis_files,
            "cohort": cohort_binding,
        },
        "support": {
            "shape_rae": list(support.shape_rae),
            "ray_count": support.ray_count,
            "cell_count": support.cell_count,
            "natural_range_upper_m": float(support.range_axis.edges[-1]),
            "digest_sha256": support.digest_sha256,
            "schema_header_sha256": support.schema_header_sha256,
            "committed_before_target_load": True,
            "file": _relative_file_report(
                staging, support_path, support_file_hash
            ),
        },
        "cohort_contract": {
            "fit_identities": [list(value) for value in FIT_IDENTITIES],
            "unseen_train_identities": [list(value) for value in UNSEEN_IDENTITIES],
            "validation_accessed": False,
            "test_accessed": False,
            "future_accessed": False,
            "radar_cube_accessed": False,
            "cfar_array_accessed": False,
        },
        "target_use_boundary": {
            "target_used_for_mark_hazard_fit": True,
            "target_used_for_mark_offsets": True,
            "target_used_for_mark_priority": True,
            "target_used_for_seed_reservation": False,
            "decoder_target_input": False,
            "exporter_target_input": False,
            "metric_evaluator_target_input": True,
        },
        "gate_contract": {
            "exact_output_count": EXACT_OUTPUT_COUNT,
            "minimum_spacing_m": MINIMUM_DISTANCE_M,
            "per_frame_chamfer_m_at_most": ORACLE_CHAMFER_GATE_M,
            "per_frame_outlier_fraction_at_most": ORACLE_OUTLIER_GATE,
            "target_stratum_retention_tolerance": RETENTION_TOLERANCE,
            "first_later_completeness_m_at_most": 1.0,
            "first_later_recall_1m_at_least": 0.80,
            "range_quota_used": False,
            "best_of_k": False,
        },
        "frames": frames,
        "renewal_activity": {
            "same_canonical_stream_hash_all_frames": same_streams,
            "any_selected_cell_hash_differs": selected_hash_differs,
            "decision_accepts_any_depth_ge_2": decision_depth_ge_2,
            "certified": renewal_activity,
        },
        "renewal_utility": {
            "decision_passes_all_frames": all_decision_passed,
            "control_fails_corresponding_gate_passed_by_decision": renewal_utility,
            "certified": renewal_utility,
        },
        "runtime": {
            "gpu": gpu,
            "device_argument": args.device,
            "torch_version": torch.__version__,
            "elapsed_seconds": elapsed,
            "wall_budget_seconds": MAX_WALL_SECONDS,
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "cuda_memory_ceiling_bytes": MAX_CUDA_BYTES,
        },
        "evidence_boundary": {
            "non_deployable_gt_aided_capacity_oracle": True,
            "model_trained": False,
            "radar_observability_claim": False,
            "doppler_claim": False,
            "temporal_claim": False,
            "strict_mathematical_upper_bound": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--archived-f0r", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _resource_error(error: BaseException) -> bool:
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(error, (MemoryError, TimeoutError, cuda_oom)) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def main() -> None:
    args = build_parser().parse_args()
    staging = begin_transaction(args.output_dir)
    started = time.monotonic()

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("VRH exceeded the frozen two-hour wall budget")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_WALL_SECONDS)
    exit_code = 0
    try:
        report = run_gate(args, staging)
        if report["status"] == "vrh_f0_capacity_no_go":
            exit_code = 2
    except BaseException as error:
        resource_invalid = _resource_error(error)
        report = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "source_commit": args.source_commit,
            "status": (
                "vrh_f0_resource_invalid"
                if resource_invalid
                else "vrh_f0_implementation_invalid"
            ),
            "passed": False,
            "decision": {
                "implementation_valid": False,
                "resource_valid": not resource_invalid,
                "scientific_routing_permitted": False,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "runtime": {
                "elapsed_seconds": time.monotonic() - started,
                "wall_budget_seconds": MAX_WALL_SECONDS,
                "peak_host_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
            },
        }
        exit_code = 3
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    output = publish_transaction(staging, args.output_dir, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "output": str(output),
                "elapsed_seconds": report["runtime"]["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
