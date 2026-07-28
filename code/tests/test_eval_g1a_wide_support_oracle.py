import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

import scripts.eval_g1a_wide_support_oracle as eval_ra0
from scripts.eval_g1a_wide_support_oracle import (
    FORMAL_VALIDATION_COUNT,
    FROZEN_MANIFEST_SHA256,
    FROZEN_SCENE_SPLIT_SHA256,
    PROTOCOL,
    RA0_THRESHOLDS,
    aggregate_evaluation,
    aggregate_resource_usage,
    atomic_json_exclusive,
    diagnostic_decision,
    estimated_peak_memory,
    gate_checks,
    parse_args,
    preflight_checks,
    select_max_target_frame,
    validate_data_contract,
)


def passing_metrics() -> dict:
    return {
        "frame_first": {
            "wide": {
                "geometry": {
                    "chamfer_m": {"median": 2.50},
                    "outlier_fraction_2m": {"mean": 0.25},
                    "completeness_mean_distance_m": {"median": 0.65},
                    "range_60_120m_completeness_mean_distance_m": {
                        "mean": 8.0,
                    },
                },
                "duplicates": {
                    "duplicate_fraction_0p05m": {"mean": 0.10},
                },
                "per_range_support": {
                    "range_60_120m": {
                        "gt_recall_from_pool_2p0m": {"mean": 0.50},
                    },
                },
            },
        },
    }


def passing_frames() -> list[dict]:
    return [
        {
            "partition": "validation",
            "sequence": index % 4,
            "radar_index": index,
            "wide": {
                "domain": {
                    "raw_query_count": 1_200_000,
                    "random_query_count": 500_000,
                    "radar_query_count": 700_000,
                    "unique_range_capacity_at_least_export_count": True,
                    "ground_truth_used_for_proposal_generation": False,
                },
                "artifact_label": {
                    "full_rald_wide_family_closure_eligible": False,
                },
                "selection": {
                    "range_capacity_validated_after_unique_pool": True,
                    "heuristic": True,
                    "strict_upper_bound": False,
                },
                "overall_support": {
                    "pool_support_scope": "global_cross_range_nearest",
                },
                "duplicates": {
                    "duplicate_fraction_0p05m": 0.0,
                },
                "selected_count": 10_000,
                "unique_selected_candidate_id_count": 10_000,
                "ground_truth_used_for_proposal_generation": False,
            },
            "g1f_32k": {"candidate_count": 32_000},
            "test_accessed": False,
        }
        for index in range(FORMAL_VALIDATION_COUNT)
    ]


def test_frozen_ra0_gate_boundaries_are_inclusive_and_unrelaxed() -> None:
    assert RA0_THRESHOLDS == {
        "chamfer_median_m": 2.50,
        "outlier_fraction_2m_mean": 0.25,
        "completeness_median_m": 0.65,
        "far_completeness_mean_m": 8.0,
        "duplicate_fraction_mean": 0.10,
        "far_pool_recall_2m_mean": 0.50,
        "point_count": 10_000,
    }
    checks = gate_checks(passing_metrics(), passing_frames())
    assert all(checks.values())

    failures = (
        ("chamfer_m", "median", 2.50001),
        ("outlier_fraction_2m", "mean", 0.25001),
        ("completeness_mean_distance_m", "median", 0.65001),
        (
            "range_60_120m_completeness_mean_distance_m",
            "mean",
            8.00001,
        ),
    )
    for metric, statistic, value in failures:
        changed = copy.deepcopy(passing_metrics())
        changed["frame_first"]["wide"]["geometry"][metric][statistic] = value
        assert not all(gate_checks(changed, passing_frames()).values())

    changed = copy.deepcopy(passing_metrics())
    changed["frame_first"]["wide"]["duplicates"][
        "duplicate_fraction_0p05m"
    ]["mean"] = 0.10001
    assert not all(gate_checks(changed, passing_frames()).values())
    changed = copy.deepcopy(passing_metrics())
    changed["frame_first"]["wide"]["per_range_support"]["range_60_120m"][
        "gt_recall_from_pool_2p0m"
    ]["mean"] = 0.49999
    assert not all(gate_checks(changed, passing_frames()).values())


def test_data_contract_requires_exact_76_24_and_never_test() -> None:
    frames = [
        {"partition": "train", "sequence": index, "radar_index": index}
        for index in range(76)
    ]
    frames.extend(
        {
            "partition": "validation",
            "sequence": 1_000 + index,
            "radar_index": index,
        }
        for index in range(24)
    )
    result = validate_data_contract(
        {"frames": frames},
        {"gate_pass": True},
    )
    assert result == {
        "manifest_frame_count": 100,
        "train_frame_count": 76,
        "validation_frame_count": 24,
        "test_frame_count": 0,
    }
    assert FROZEN_MANIFEST_SHA256 == (
        "645307a8bae351db51b55128043dae69bce5b928169d5fa25c1b9c55083de4e4"
    )
    assert FROZEN_SCENE_SPLIT_SHA256 == (
        "61596bd50ce0bdab633c9ff0ec5ab5148c2c78a10dfc06a71d0d6f0a6c9427cc"
    )

    invalid = [
        *frames,
        {"partition": "test", "sequence": 9_999, "radar_index": 1},
    ]
    with pytest.raises(ValueError, match="76/24"):
        validate_data_contract({"frames": invalid}, {"gate_pass": True})
    with pytest.raises(ValueError, match="leakage"):
        validate_data_contract({"frames": frames}, {"gate_pass": False})


def test_cli_exposes_no_scientific_count_ratio_or_threshold_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_g1a_wide_support_oracle.py",
            "--data-root",
            "/data",
            "--cache-root",
            "/cache",
            "--manifest",
            "/manifest.json",
            "--scene-split",
            "/split.json",
            "--g1d-run",
            "/g1d",
            "--output",
            "/output.json",
            "--source-commit",
            "a" * 40,
            "--preflight-max-frame",
        ],
    )
    args = parse_args()
    assert vars(args).keys() == {
        "data_root",
        "cache_root",
        "manifest",
        "scene_split",
        "g1d_run",
        "output",
        "source_commit",
        "device",
        "preflight_max_frame",
    }
    assert args.preflight_max_frame is True


def test_protocol_is_an_initial_query_domain_diagnostic_not_an_upper_bound() -> None:
    assert PROTOCOL == (
        "g1a_ra0_rald_inspired_initial_query_domain_diagnostic_v2"
    )
    for frame in passing_frames():
        assert frame["wide"]["selection"]["heuristic"] is True
        assert frame["wide"]["selection"]["strict_upper_bound"] is False
        assert (
            frame["wide"]["artifact_label"][
                "full_rald_wide_family_closure_eligible"
            ]
            is False
        )


def test_failed_diagnostic_does_not_close_full_rald_wide_support_family() -> None:
    assert diagnostic_decision(passed=False, preflight=False) == (
        "initial_query_domain_diagnostic_failed_"
        "no_conclusion_about_full_rald_wide_support_family"
    )
    assert diagnostic_decision(passed=False, preflight=True) == (
        "initial_query_domain_preflight_failed_"
        "no_conclusion_about_full_rald_wide_support_family"
    )


def test_existing_output_is_rejected_before_h200_or_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "ra0.json"
    output.write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(output=output)
    monkeypatch.setattr(eval_ra0, "parse_args", lambda: args)
    monkeypatch.setattr(
        eval_ra0,
        "require_h200",
        lambda _: (_ for _ in ()).throw(AssertionError("H200 accessed")),
    )

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        eval_ra0.main()
    assert output.read_text(encoding="utf-8") == "{}\n"


def test_exclusive_json_writer_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    atomic_json_exclusive(output, {"first": True})
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        atomic_json_exclusive(output, {"first": False})
    assert output.read_text(encoding="utf-8") == '{\n  "first": true\n}\n'


def method_record(value: float, *, wide: bool) -> dict:
    result = {
        "geometry": {
            "chamfer_m": value,
            "outlier_fraction_2m": value / 10.0,
            "completeness_mean_distance_m": value / 2.0,
            "range_60_120m_completeness_mean_distance_m": value,
        },
        "duplicates": {"duplicate_fraction_0p05m": value / 20.0},
        "overall_support": {"gt_recall_from_pool_2p0m": value / 10.0},
        "per_range_support": {
            "range_60_120m": {
                "gt_recall_from_pool_2p0m": value / 10.0,
            },
        },
    }
    if wide:
        result["domain"] = {
            "raw_query_count": 1_200_000,
            "unique_capacity_cell_count": 1_100_000,
        }
    return result


def test_aggregation_reports_frame_first_and_scene_first() -> None:
    frames = []
    for index in range(24):
        value = 1.0 if index < 18 else 3.0
        wide = method_record(value, wide=True)
        g1f = method_record(value + 1.0, wide=False)
        frames.append(
            {
                "sequence": 1 if index < 18 else 2,
                "wide": wide,
                "g1f_32k": g1f,
                "paired": {
                    "chamfer_wide_minus_g1f_m": -1.0,
                },
            }
        )
    aggregate = aggregate_evaluation(frames)

    assert aggregate["frame_first"]["frame_count"] == 24
    assert aggregate["scene_first"]["scene_count"] == 2
    assert aggregate["frame_first"]["wide"]["geometry"]["chamfer_m"]["mean"] == 1.5
    assert aggregate["scene_first"]["wide"]["geometry"]["chamfer_m"]["mean"] == 2.0
    assert set(aggregate["scene_first"]["per_scene"]) == {"1", "2"}


def test_max_target_frame_preflight_selection_is_stable(tmp_path: Path) -> None:
    records = [
        {"sequence": 6, "radar_index": 183},
        {"sequence": 55, "radar_index": 376},
        {"sequence": 2, "radar_index": 99},
    ]
    target_counts = [4, 7, 7]
    for record, target_count in zip(records, target_counts, strict=True):
        cache = tmp_path / (
            f"seq{record['sequence']:02d}_"
            f"radar_{record['radar_index']:05d}.npz"
        )
        np.savez(
            cache,
            target_xyz_confidence=np.zeros(
                (target_count, 4),
                dtype=np.float32,
            ),
        )

    selected = select_max_target_frame(records, tmp_path)

    assert selected["dataset_index"] == 2
    assert selected["sequence"] == 2
    assert selected["radar_index"] == 99
    assert selected["target_count"] == 7


def test_preflight_checks_require_global_support_capacity_nms_and_resources() -> None:
    frame = passing_frames()[0]
    frame["wide"]["geometry"] = {"target_count": 123}
    frame["resources"] = {
        "wall_time_seconds": 2.0,
        "cuda_peak_allocated_bytes": 1_024,
        "cuda_peak_reserved_bytes": 2_048,
    }
    selected_frame = {
        "sequence": frame["sequence"],
        "radar_index": frame["radar_index"],
        "target_count": 123,
    }

    checks = preflight_checks(frame, selected_frame)
    assert all(checks.values())

    changed = copy.deepcopy(frame)
    changed["wide"]["overall_support"]["pool_support_scope"] = (
        "range_local_nearest"
    )
    assert not all(preflight_checks(changed, selected_frame).values())
    changed = copy.deepcopy(frame)
    changed["wide"]["duplicates"]["duplicate_fraction_0p05m"] = 0.001
    assert not all(preflight_checks(changed, selected_frame).values())


def test_resource_aggregation_records_peak_and_wall_time() -> None:
    frames = [
        {
            "resources": {
                "wall_time_seconds": 2.0,
                "cuda_peak_allocated_bytes": 1_000,
                "cuda_peak_reserved_bytes": 2_000,
                "cuda_peak_allocated_delta_bytes": 800,
                "cuda_peak_reserved_delta_bytes": 1_500,
            }
        },
        {
            "resources": {
                "wall_time_seconds": 3.0,
                "cuda_peak_allocated_bytes": 1_200,
                "cuda_peak_reserved_bytes": 2_400,
                "cuda_peak_allocated_delta_bytes": 900,
                "cuda_peak_reserved_delta_bytes": 1_700,
            }
        },
    ]

    report = aggregate_resource_usage(frames)

    assert report["frame_count"] == 2
    assert report["total_wall_time_seconds"] == 5.0
    assert report["maximum_frame_wall_time_seconds"] == 3.0
    assert report["maximum_cuda_peak_allocated_bytes"] == 1_200
    assert report["maximum_cuda_peak_reserved_bytes"] == 2_400


def test_streaming_peak_estimate_is_below_32gb() -> None:
    report = estimated_peak_memory(max_target_count=50_000)

    assert report["below_32gb"] is True
    assert report["estimated_peak_bytes"] < report["limit_bytes"]
    assert report["maximum_streamed_distance_tile_bytes"] == 8192 * 4096 * 4
    assert report["target_topk_shortlist_bytes"] == 50_000 * 4 * (4 + 8)
