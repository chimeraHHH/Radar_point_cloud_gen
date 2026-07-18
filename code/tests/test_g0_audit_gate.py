from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from scripts.audit_kradar_g0 import aggregate_report  # noqa: E402


def frame(observable_count: int, observable_fraction: float) -> dict:
    return {
        "sequence": 1,
        "partition": "train",
        "schema": {
            "logical_shape": [64, 256, 107, 37],
            "raw_dtype": "float64",
            "lidar64_fields": ["x", "y", "z", "t"],
        },
        "synchronization": {"os2_delta_ms": 0.0, "os1_delta_ms": 25.0},
        "motion": {"nearest_timestamp_delta_ms": 0.0, "speed_mps": 5.0},
        "cfar_roundtrip": {
            "exact_bin_fraction": 1.0,
            "exact_bin_count": 100,
            "point_count": 100,
        },
        "observability": {
            "observable_count": observable_count,
            "observable_fraction_of_surface": observable_fraction,
        },
        "alignment_null": {"correct_minus_mirror_margin": 0.1},
        "doppler_alias": {
            "negative_ego_hypothesis_median_error_mps": 0.2,
            "positive_ego_hypothesis_median_error_mps": 0.3,
            "zero_centered_hypothesis_median_error_mps": 0.1,
            "random_circular_median_baseline_mps": 0.9,
        },
        "lidar_scan_timing": {
            "selected_reference": "none",
            "margin_median_by_reference": {"none": -0.5},
        },
    }


def test_nonempty_check_accepts_low_coverage_with_positive_targets() -> None:
    report = aggregate_report([frame(1, 0.001)], required_frames=1)

    assert report["checks"]["observable_target_nonempty"] is True
    assert report["gate_pass"] is True


def test_nonempty_check_rejects_zero_positive_targets() -> None:
    report = aggregate_report([frame(0, 0.0)], required_frames=1)

    assert report["checks"]["observable_target_nonempty"] is False
    assert report["gate_pass"] is False
