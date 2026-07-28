#!/usr/bin/env python3
"""Run the frozen D-MHW G-RM label-only legality and provenance audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import load_axes, load_pcd_ascii  # noqa: E402
from dmhw.birth_radial_moment import (  # noqa: E402
    PROTOCOL,
    build_frame_radial_moment_labels,
    parse_label_boxes,
    radar_roi_mask,
    sha256,
    validate_static_sign_audit,
)
from dmhw.contracts import (  # noqa: E402
    DIRECT_HORIZONS_SECONDS,
    build_direct_horizon_examples,
)


SOURCE_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_commit(repo: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if not SOURCE_PATTERN.fullmatch(commit):
        raise RuntimeError("G-RM audit requires a full Git commit")
    return commit


def transform_points(points_xyz_m: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if transform.shape != (4, 4):
        raise ValueError("Radar/LiDAR transform must be 4x4")
    return points_xyz_m @ transform[:3, :3].T + transform[:3, 3]


def coverage_report(valid: int, denominator: int) -> dict:
    return {
        "valid_label_count": int(valid),
        "eligible_geometry_count": int(denominator),
        "coverage": None if denominator == 0 else float(valid / denominator),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--static-doppler-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-train-anchors", type=int, default=740)
    parser.add_argument("--expected-validation-anchors", type=int, default=160)
    parser.add_argument("--boundary-margin-m", type=float, default=0.1)
    parser.add_argument("--dynamic-threshold-mps", type=float, default=0.5)
    parser.add_argument("--minimum-overall-coverage", type=float, default=0.75)
    parser.add_argument("--minimum-horizon-coverage", type=float, default=0.60)
    parser.add_argument("--minimum-dynamic-windows", type=int, default=10)
    parser.add_argument("--maximum-p5-mae-mps", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    current_commit = git_commit(repo)
    if args.source_commit != current_commit:
        raise ValueError("G-RM source commit differs from the checked-out snapshot")
    if args.output.exists() or args.ledger_output.exists():
        raise FileExistsError("G-RM output already exists")
    if args.output == args.ledger_output:
        raise ValueError("Summary and ledger outputs must differ")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("gate_pass") is not True:
        raise ValueError("G-RM temporal manifest did not pass")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("G-RM temporal manifest is empty")
    test_records = [frame for frame in frames if frame.get("partition") == "test"]
    if test_records:
        raise ValueError("G-RM development audit refuses test records")
    if manifest.get("checks", {}).get(
        "radar_frame_ego_transforms_present"
    ) is not True:
        raise ValueError("G-RM manifest lacks radar-frame ego transforms")

    examples = build_direct_horizon_examples(manifest)
    anchor_count = Counter(example["partition"] for example in examples)
    expected_anchor_count = {
        "train": args.expected_train_anchors,
        "validation": args.expected_validation_anchors,
    }
    frame_by_window_index = {
        (str(frame["window_id"]), int(frame["frame_in_window"])): frame
        for frame in frames
    }
    if len(frame_by_window_index) != len(frames):
        raise ValueError("G-RM manifest contains duplicate window-frame identities")

    target_usage: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for example in examples:
        for target in example["targets"]:
            key = (str(example["window_id"]), int(target["frame_in_window"]))
            target_usage[key].append(
                {
                    "anchor_partition": str(example["partition"]),
                    "anchor_current_radar_index": int(
                        example["current"]["radar_index"]
                    ),
                    "horizon_seconds": float(target["horizon_seconds"]),
                }
            )

    axes = load_axes(args.data_root / "resources")
    doppler_step = float(np.median(np.diff(axes.doppler_mps)))
    doppler_lower = float(axes.doppler_mps[0])
    doppler_period = doppler_step * int(axes.doppler_mps.size)
    range_bounds = (float(np.min(axes.range_m)), float(np.max(axes.range_m)))
    azimuth_bounds = (
        float(np.min(axes.azimuth_rad)),
        float(np.max(axes.azimuth_rad)),
    )
    elevation_bounds = (
        float(np.min(axes.elevation_rad)),
        float(np.max(axes.elevation_rad)),
    )

    static_document = json.loads(
        args.static_doppler_audit.read_text(encoding="utf-8")
    )
    sign_audit = validate_static_sign_audit(static_document)
    if not np.isclose(
        sign_audit["doppler_period_mps"],
        doppler_period,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("G-RM and P5 Doppler periods differ")

    file_access_count = Counter(
        {
            "temporal_manifest": 1,
            "static_doppler_audit": 1,
            "axis_resource": 2,
        }
    )
    file_access_bytes = Counter(
        {
            "temporal_manifest": args.manifest.stat().st_size,
            "static_doppler_audit": args.static_doppler_audit.stat().st_size,
            "axis_resource": sum(
                (
                    args.data_root / "resources" / name
                ).stat().st_size
                for name in ("info_arr.mat", "arr_doppler.mat")
            ),
        }
    )
    frame_summaries: dict[tuple[str, int], dict] = {}
    invalid_unique = Counter()
    all_box_comparisons = []
    dynamic_windows = set()
    frame_errors = []
    valid_provenance_count = 0
    ledger_line_count = 0
    maximum_pose_residual = 0.0
    calibration_hash_by_sequence = {}

    args.ledger_output.parent.mkdir(parents=True, exist_ok=True)
    ledger_temporary = args.ledger_output.with_suffix(
        args.ledger_output.suffix + ".tmp"
    )
    with ledger_temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as ledger:
                for target_number, key in enumerate(sorted(target_usage), start=1):
                    target_frame = frame_by_window_index[key]
                    source_key = (key[0], key[1] - 1)
                    source_frame = frame_by_window_index.get(source_key)
                    if source_frame is None:
                        frame_errors.append(
                            {
                                "window_id": key[0],
                                "frame_in_window": key[1],
                                "error": "missing immediately preceding frame",
                            }
                        )
                        continue
                    sequence = int(target_frame["sequence"])
                    sequence_root = args.data_root / str(sequence)
                    source_label_path = (
                        sequence_root
                        / "info_label"
                        / str(source_frame["label"])
                    )
                    target_label_path = (
                        sequence_root
                        / "info_label"
                        / str(target_frame["label"])
                    )
                    lidar_path = (
                        sequence_root
                        / "os2-64"
                        / f"os2-64_{int(target_frame['lidar64_index']):05d}.pcd"
                    )
                    calibration_path = (
                        sequence_root
                        / "info_calib"
                        / "calib_radar_lidar.txt"
                    )
                    try:
                        source_label_sha = sha256(source_label_path)
                        target_label_sha = sha256(target_label_path)
                        target_lidar_sha = sha256(lidar_path)
                        if sequence not in calibration_hash_by_sequence:
                            calibration_hash_by_sequence[sequence] = sha256(
                                calibration_path
                            )
                            file_access_count["calibration"] += 1
                            file_access_bytes["calibration"] += (
                                calibration_path.stat().st_size
                            )
                        calibration_sha = calibration_hash_by_sequence[sequence]
                        file_access_count.update(
                            {
                                "label": 2,
                                "future_lidar_geometry": 1,
                            }
                        )
                        file_access_bytes.update(
                            {
                                "label": (
                                    source_label_path.stat().st_size
                                    + target_label_path.stat().st_size
                                ),
                                "future_lidar_geometry": lidar_path.stat().st_size,
                            }
                        )
                        raw_lidar, fields = load_pcd_ascii(lidar_path)
                        if not all(name in fields for name in ("x", "y", "z")):
                            raise ValueError("Future LiDAR PCD lacks XYZ fields")
                        xyz_columns = [fields.index(name) for name in ("x", "y", "z")]
                        lidar_xyz = raw_lidar[:, xyz_columns].astype(np.float64)
                        calibration = np.asarray(
                            target_frame["radar_from_lidar64"],
                            dtype=np.float64,
                        ).reshape(4, 4)
                        radar_xyz = transform_points(lidar_xyz, calibration)
                        eligible = radar_roi_mask(
                            radar_xyz,
                            range_bounds_m=range_bounds,
                            azimuth_bounds_rad=azimuth_bounds,
                            elevation_bounds_rad=elevation_bounds,
                        )
                        eligible_xyz = radar_xyz[eligible]
                        original_indices = np.flatnonzero(eligible)
                        source_boxes = parse_label_boxes(
                            source_label_path.read_text(encoding="utf-8"),
                            calibration[:3, 3],
                        )
                        target_boxes = parse_label_boxes(
                            target_label_path.read_text(encoding="utf-8"),
                            calibration[:3, 3],
                        )
                        labels = build_frame_radial_moment_labels(
                            points_xyz_m=eligible_xyz,
                            original_point_indices=original_indices,
                            source_frame=source_frame,
                            target_frame=target_frame,
                            source_boxes=source_boxes,
                            target_boxes=target_boxes,
                            doppler_lower_mps=doppler_lower,
                            doppler_period_mps=doppler_period,
                            p5_sign_hypothesis=sign_audit["hypothesis"],
                            source_label_sha256=source_label_sha,
                            target_label_sha256=target_label_sha,
                            target_lidar_sha256=target_lidar_sha,
                            calibration_sha256=calibration_sha,
                            boundary_margin_m=args.boundary_margin_m,
                            dynamic_threshold_mps=args.dynamic_threshold_mps,
                        )
                    except Exception as error:
                        frame_errors.append(
                            {
                                "sequence": sequence,
                                "window_id": key[0],
                                "frame_in_window": key[1],
                                "radar_index": int(target_frame["radar_index"]),
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                        continue

                    usage = target_usage[key]
                    horizons = sorted(
                        {float(item["horizon_seconds"]) for item in usage}
                    )
                    for record in labels.records:
                        record["anchor_usage_count"] = len(usage)
                        record["used_horizons_seconds"] = horizons
                        required = (
                            record.get("source_frame"),
                            record.get("target_frame"),
                            record.get("delta_seconds"),
                            record.get("pose_sha256"),
                            record.get("label_pair_sha256"),
                            record.get("provenance"),
                        )
                        if all(value is not None for value in required):
                            valid_provenance_count += 1
                        ledger.write(
                            json.dumps(
                                record,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        ledger_line_count += 1
                    invalid_unique.update(labels.invalid_reason_counts)
                    all_box_comparisons.extend(labels.box_comparisons)
                    if labels.dynamic_track_ids:
                        dynamic_windows.add(str(target_frame["window_id"]))
                    maximum_pose_residual = max(
                        maximum_pose_residual,
                        labels.pose_residual_max_abs,
                    )
                    frame_summaries[key] = {
                        "sequence": sequence,
                        "partition": str(target_frame["partition"]),
                        "window_id": str(target_frame["window_id"]),
                        "source_radar_index": int(source_frame["radar_index"]),
                        "target_radar_index": int(target_frame["radar_index"]),
                        "target_lidar64_index": int(
                            target_frame["lidar64_index"]
                        ),
                        "target_use_count": len(usage),
                        "used_horizons_seconds": horizons,
                        "eligible_geometry_count": labels.eligible_point_count,
                        "valid_label_count": labels.valid_point_count,
                        "coverage": (
                            labels.valid_point_count
                            / labels.eligible_point_count
                        ),
                        "invalid_reason_counts": labels.invalid_reason_counts,
                        "dynamic_track_count": len(labels.dynamic_track_ids),
                        "pose_residual_max_abs": labels.pose_residual_max_abs,
                        "pose_sha256": labels.pose_hash,
                        "source_label_sha256": source_label_sha,
                        "target_label_sha256": target_label_sha,
                        "target_lidar_sha256": target_lidar_sha,
                    }
                    if (
                        args.progress_every > 0
                        and target_number % args.progress_every == 0
                    ):
                        print(
                            json.dumps(
                                {
                                    "processed_unique_targets": target_number,
                                    "required_unique_targets": len(target_usage),
                                    "ledger_labels": ledger_line_count,
                                    "frame_errors": len(frame_errors),
                                }
                            ),
                            flush=True,
                        )
    ledger_temporary.replace(args.ledger_output)

    aggregate_valid = 0
    aggregate_denominator = 0
    by_partition_counts = defaultdict(lambda: [0, 0])
    by_horizon_counts = defaultdict(lambda: [0, 0])
    invalid_occurrence_weighted = Counter()
    missing_usage_count = 0
    for example in examples:
        for target in example["targets"]:
            key = (str(example["window_id"]), int(target["frame_in_window"]))
            summary = frame_summaries.get(key)
            if summary is None:
                missing_usage_count += 1
                continue
            valid = int(summary["valid_label_count"])
            denominator = int(summary["eligible_geometry_count"])
            aggregate_valid += valid
            aggregate_denominator += denominator
            partition = str(example["partition"])
            horizon = f"{float(target['horizon_seconds']):.1f}"
            by_partition_counts[partition][0] += valid
            by_partition_counts[partition][1] += denominator
            by_horizon_counts[horizon][0] += valid
            by_horizon_counts[horizon][1] += denominator
            invalid_occurrence_weighted.update(summary["invalid_reason_counts"])

    overall_coverage = coverage_report(
        aggregate_valid,
        aggregate_denominator,
    )
    coverage_by_partition = {
        partition: coverage_report(values[0], values[1])
        for partition, values in sorted(by_partition_counts.items())
    }
    coverage_by_horizon = {
        horizon: coverage_report(values[0], values[1])
        for horizon, values in sorted(by_horizon_counts.items())
    }
    p5_errors = np.asarray(
        [
            item["point_box_abs_circular_error_median_mps"]
            for item in all_box_comparisons
        ],
        dtype=np.float64,
    )
    p5_report = {
        "comparison_count": int(p5_errors.size),
        "mean_absolute_error_mps": (
            None if p5_errors.size == 0 else float(np.mean(p5_errors))
        ),
        "median_absolute_error_mps": (
            None if p5_errors.size == 0 else float(np.median(p5_errors))
        ),
        "p90_absolute_error_mps": (
            None if p5_errors.size == 0 else float(np.quantile(p5_errors, 0.9))
        ),
        "gate_maximum_mae_mps": args.maximum_p5_mae_mps,
        "comparison_unit": (
            "one tracked box; median point-to-box circular difference per box, "
            "then mean across boxes"
        ),
    }

    all_horizons_present = set(coverage_by_horizon) == {
        f"{value:.1f}" for value in DIRECT_HORIZONS_SECONDS
    }
    checks = {
        "manifest_gate_passed": manifest.get("gate_pass") is True,
        "radar_pose_contract_present": manifest.get("checks", {}).get(
            "radar_frame_ego_transforms_present"
        )
        is True,
        "exact_740_train_160_validation_anchors": dict(anchor_count)
        == expected_anchor_count,
        "all_anchor_horizon_usages_audited": missing_usage_count == 0,
        "no_frame_errors": not frame_errors,
        "test_records_read_zero": not test_records,
        "future_cube_path_count_zero": True,
        "future_cube_bytes_read_zero": True,
        "every_valid_label_has_required_provenance": (
            valid_provenance_count == ledger_line_count
        ),
        "ledger_line_count_matches_unique_valid_labels": (
            ledger_line_count
            == sum(
                int(summary["valid_label_count"])
                for summary in frame_summaries.values()
            )
        ),
        "pose_composition_residual_le_1e-8": maximum_pose_residual <= 1e-8,
        "overall_coverage_ge_75pct": (
            overall_coverage["coverage"] is not None
            and overall_coverage["coverage"] >= args.minimum_overall_coverage
        ),
        "each_horizon_coverage_ge_60pct": (
            all_horizons_present
            and all(
                value["coverage"] is not None
                and value["coverage"] >= args.minimum_horizon_coverage
                for value in coverage_by_horizon.values()
            )
        ),
        "dynamic_tracked_windows_ge_10": len(dynamic_windows)
        >= args.minimum_dynamic_windows,
        "p5_box_center_range_rate_mae_le_0p5_mps": (
            p5_report["mean_absolute_error_mps"] is not None
            and p5_report["mean_absolute_error_mps"]
            <= args.maximum_p5_mae_mps
        ),
        "p5_sign_selected_on_train_only": sign_audit["selection_partition"]
        == "train",
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "source_commit": current_commit,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "data_root": str(args.data_root.resolve()),
            "static_doppler_audit": str(
                args.static_doppler_audit.resolve()
            ),
            "static_doppler_audit_sha256": sha256(
                args.static_doppler_audit
            ),
            "boundary_margin_m": args.boundary_margin_m,
            "dynamic_threshold_mps": args.dynamic_threshold_mps,
            "expected_anchor_count": expected_anchor_count,
            "minimum_overall_coverage": args.minimum_overall_coverage,
            "minimum_horizon_coverage": args.minimum_horizon_coverage,
            "minimum_dynamic_windows": args.minimum_dynamic_windows,
            "maximum_p5_mae_mps": args.maximum_p5_mae_mps,
            "radar_roi": {
                "range_bounds_m": list(range_bounds),
                "azimuth_bounds_rad": list(azimuth_bounds),
                "elevation_bounds_rad": list(elevation_bounds),
                "definition": (
                    "finite future LiDAR XYZ transformed into target radar "
                    "coordinates and clipped only by static radar axes"
                ),
            },
            "doppler_axis": {
                "bin_count": int(axes.doppler_mps.size),
                "lower_mps": doppler_lower,
                "step_mps": doppler_step,
                "period_mps": doppler_period,
            },
        },
        "legality": {
            "future_cube_path_count": 0,
            "future_cube_bytes_read": 0,
            "test_record_count_read": 0,
            "future_lidar_is_label_only": True,
            "radar_only_inference_claim_unchanged": True,
            "training_uses_lidar_and_annotation_derived_supervision": True,
            "file_access_count_by_role": dict(sorted(file_access_count.items())),
            "file_access_bytes_by_role": dict(sorted(file_access_bytes.items())),
        },
        "sign_convention": sign_audit,
        "anchor_audit": {
            "example_count": len(examples),
            "example_count_by_partition": dict(sorted(anchor_count.items())),
            "target_occurrence_count": sum(
                len(example["targets"]) for example in examples
            ),
            "unique_target_frame_count": len(target_usage),
            "audited_unique_target_frame_count": len(frame_summaries),
            "missing_target_usage_count": missing_usage_count,
        },
        "label_coverage": {
            "overall": overall_coverage,
            "by_partition": coverage_by_partition,
            "by_horizon_seconds": coverage_by_horizon,
            "invalid_reason_counts_unique_targets": dict(
                sorted(invalid_unique.items())
            ),
            "invalid_reason_counts_anchor_weighted": dict(
                sorted(invalid_occurrence_weighted.items())
            ),
            "unverified_background_policy": (
                "invalid; no static-background label is inferred from ego pose"
            ),
        },
        "dynamic_tracked_subset": {
            "independent_window_count": len(dynamic_windows),
            "window_ids": sorted(dynamic_windows),
            "minimum_required_windows": args.minimum_dynamic_windows,
            "box_comparison_count": len(all_box_comparisons),
        },
        "p5_box_center_agreement": p5_report,
        "provenance": {
            "ledger": str(args.ledger_output.resolve()),
            "ledger_sha256": sha256(args.ledger_output),
            "ledger_line_count": ledger_line_count,
            "valid_provenance_count": valid_provenance_count,
            "calibration_sha256_by_sequence": {
                str(key): value
                for key, value in sorted(calibration_hash_by_sequence.items())
            },
            "maximum_pose_composition_residual": maximum_pose_residual,
            "birth_distribution_ground_truth": False,
            "birth_distribution_claim_enabled": False,
            "birth_radial_moment_claim_enabled": passed,
            "deployment_eligible": False,
        },
        "checks": checks,
        "passed": passed,
        "decision": "go_label_contract_only" if passed else "no_go",
        "frame_errors": frame_errors,
        "frame_summaries": [
            frame_summaries[key] for key in sorted(frame_summaries)
        ],
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ledger": str(args.ledger_output),
                "anchor_audit": report["anchor_audit"],
                "overall_coverage": overall_coverage,
                "dynamic_tracked_subset": report["dynamic_tracked_subset"],
                "p5_box_center_agreement": p5_report,
                "checks": checks,
                "passed": passed,
                "decision": report["decision"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
