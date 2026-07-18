from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from scripts.build_kradar_dense_cache import (  # noqa: E402
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
