import argparse
import copy
from pathlib import Path
import sys

import pytest

import scripts.eval_g1a_wide_support_oracle as eval_ra0
from scripts.eval_g1a_wide_support_oracle import (
    FORMAL_VALIDATION_COUNT,
    FROZEN_MANIFEST_SHA256,
    FROZEN_SCENE_SPLIT_SHA256,
    RA0_THRESHOLDS,
    aggregate_evaluation,
    atomic_json_exclusive,
    estimated_peak_memory,
    gate_checks,
    parse_args,
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
                    "ground_truth_used_for_proposal_generation": False,
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
    }


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


def test_streaming_peak_estimate_is_below_32gb() -> None:
    report = estimated_peak_memory(max_target_count=50_000)

    assert report["below_32gb"] is True
    assert report["estimated_peak_bytes"] < report["limit_bytes"]
    assert report["maximum_streamed_distance_tile_bytes"] == 8192 * 4096 * 4
