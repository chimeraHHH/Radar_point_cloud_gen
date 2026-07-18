from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from scripts.build_kradar_dense_cache import (  # noqa: E402
    lidar_time_origin_shift,
    validate_g0_lidar_time_reference,
)


def report(*references: str | None) -> dict:
    frames = []
    for reference in references:
        frame = {"lidar_scan_timing": {}}
        if reference is not None:
            frame["lidar_scan_timing"]["selected_reference"] = reference
        frames.append(frame)
    return {"frames": frames}


def test_matching_g0_reference_is_accepted() -> None:
    validate_g0_lidar_time_reference(report("none", "none"), "none")


def test_mismatched_g0_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="differs from G0"):
        validate_g0_lidar_time_reference(report("none", "start"), "none")


def test_missing_g0_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_g0_lidar_time_reference(report(None), "none")


@pytest.mark.parametrize(
    ("reference", "expected"),
    (("none", None), ("start", 0.0), ("center", -0.05), ("end", -0.1)),
)
def test_lidar_time_origin_shift(reference: str, expected: float | None) -> None:
    assert lidar_time_origin_shift(reference, 0.1) == expected


def test_lidar_time_origin_shift_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        lidar_time_origin_shift("invalid", 0.1)
    with pytest.raises(ValueError, match="negative"):
        lidar_time_origin_shift("none", -0.1)
