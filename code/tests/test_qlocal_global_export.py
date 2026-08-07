from __future__ import annotations

import inspect

import pytest
import torch

from eval.rald_wce_stage0 import global_exact_capacity_export
from preflight_qlocal_global_export import (
    archived_frame_map,
    atomic_commit_report,
    retention_gate,
    target_stratum_retention,
    terminal_status,
)


def _archived_document() -> dict:
    return {
        "protocol": "g1_qlocal_distributional_risk_f0_preflight_v1",
        "source_commit": "bf1a166c48be60f82c5283b5937edf0cf6ad64c2",
        "status": "qlocal_f0_capacity_no_go",
        "frames": [
            {"sequence": index + 1, "radar_index": 100 + index}
            for index in range(12)
        ],
    }


def test_global_exporter_api_has_no_target_or_quota_input() -> None:
    parameters = inspect.signature(global_exact_capacity_export).parameters

    assert set(parameters) == {
        "candidate_xyz_m",
        "candidate_confidence",
        "minimum_distance_m",
    }
    assert "target" not in " ".join(parameters).lower()
    assert "quota" not in " ".join(parameters).lower()


def test_archived_frame_map_rejects_duplicate_or_wrong_terminal() -> None:
    document = _archived_document()
    assert len(archived_frame_map(document)) == 12

    document["frames"][-1] = document["frames"][0]
    with pytest.raises(ValueError):
        archived_frame_map(document)

    document = _archived_document()
    document["status"] = "qlocal_f0_preflight_passed"
    with pytest.raises(ValueError):
        archived_frame_map(document)


def test_retention_gate_requires_both_completeness_and_recall() -> None:
    fixed = {
        "range_30_60m": {
            "target_count": 4,
            "target_effective_count": 2.0,
            "completeness_mean_distance_m": 0.4,
            "recall_1m": 0.75,
        }
    }
    global_report = {
        "range_30_60m": {
            "target_count": 4,
            "target_effective_count": 2.0,
            "completeness_mean_distance_m": 0.39,
            "recall_1m": 0.76,
        }
    }
    checks, passed = retention_gate(fixed, global_report)
    assert passed
    assert checks["range_30_60m"]["passed"]

    global_report["range_30_60m"]["recall_1m"] = 0.70
    _, passed = retention_gate(fixed, global_report)
    assert not passed

    global_report["range_30_60m"]["target_effective_count"] = 1.5
    with pytest.raises(ValueError):
        retention_gate(fixed, global_report)


def test_target_stratum_retention_rejects_zero_effective_weight() -> None:
    prediction = torch.tensor([[1.0, 0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        target_stratum_retention(prediction, target)


def test_terminal_status_does_not_route_from_invalid_implementation() -> None:
    assert terminal_status(
        all_frames_passed=True,
        parent_unchanged=True,
    ) == "qlocal_f0r_capacity_passed"
    assert terminal_status(
        all_frames_passed=False,
        parent_unchanged=True,
    ) == "qlocal_f0r_capacity_no_go"
    assert terminal_status(
        all_frames_passed=False,
        parent_unchanged=False,
    ) == "qlocal_f0r_implementation_invalid"


def test_atomic_report_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "terminal"
    path = atomic_commit_report(output, {"status": "pass"})

    assert path == output / "preflight.json"
    assert path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        atomic_commit_report(output, {"status": "forged"})


def test_global_export_is_stable_under_equal_scores() -> None:
    x = 10.0 + 0.06 * torch.arange(22, dtype=torch.float32)
    y = -0.63 + 0.06 * torch.arange(22, dtype=torch.float32)
    z = -0.60 + 0.06 * torch.arange(21, dtype=torch.float32)
    xyz = torch.cartesian_prod(x, y, z)
    confidence = torch.ones(xyz.shape[0])

    export = global_exact_capacity_export(
        xyz,
        confidence,
        minimum_distance_m=0.05,
    )

    assert torch.equal(
        export.selected_candidate_rows,
        torch.arange(10_000),
    )
