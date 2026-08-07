import inspect
from pathlib import Path

import pytest
import torch

from scripts.diagnose_rb2_candidate_support import (
    BASE_CANDIDATE_QUOTAS,
    BANK_SCALES,
    DIAGNOSED_PARENT_SOURCE_COMMIT,
    LATTICE_ORIGIN_XYZ_M,
    MINIMUM_CONFIDENCE_COVERAGE,
    MINIMUM_OCCUPIED_VOXEL_RECALL,
    NEIGHBORHOOD_RADIUS,
    SCORE_MODES,
    aggregate_doppler_score,
    arm_gate,
    build_cube_only_candidates,
    candidate_support_metrics,
    frozen_sweep_grid,
    records_for_mode,
    sweep_decision,
    verify_source_tree,
)


def test_frozen_sweep_grid_covers_score_bank_cross_product() -> None:
    grid = frozen_sweep_grid()
    assert len(grid) == len(SCORE_MODES) * len(BANK_SCALES) == 9
    assert {arm.score_mode for arm in grid} == set(SCORE_MODES)
    assert {arm.candidate_count for arm in grid} == {20_000, 40_000, 80_000}
    assert all(arm.neighborhood_radius == NEIGHBORHOOD_RADIUS == 4 for arm in grid)
    assert all(arm.seed_multiplier == 8 for arm in grid)
    assert grid[0].candidate_quotas == BASE_CANDIDATE_QUOTAS
    assert grid[-1].candidate_quotas == (64_000, 13_600, 2_400)


def test_doppler_aggregations_are_explicit_and_deterministic() -> None:
    cube = torch.tensor(
        [
            [[[-1.0]], [[3.0]]],
            [[[8.0]], [[1.0]]],
        ]
    )
    maximum = aggregate_doppler_score(cube, "max_d")
    summed = aggregate_doppler_score(cube, "sum_d")
    log_summed = aggregate_doppler_score(cube, "log_sum_d")
    assert maximum[:, 0, 0].tolist() == pytest.approx(
        [torch.log1p(torch.tensor(8.0)).item(), torch.log1p(torch.tensor(3.0)).item()]
    )
    assert summed[:, 0, 0].tolist() == pytest.approx([8.0, 4.0])
    assert log_summed[:, 0, 0].tolist() == pytest.approx(
        [
            torch.log1p(torch.tensor(8.0)).item(),
            torch.log1p(torch.tensor(3.0)).item()
            + torch.log1p(torch.tensor(1.0)).item(),
        ]
    )
    with pytest.raises(ValueError, match="Unknown frozen"):
        aggregate_doppler_score(cube, "target_ranked")


def test_candidate_builder_has_no_target_or_forbidden_input() -> None:
    parameters = set(inspect.signature(build_cube_only_candidates).parameters)
    assert parameters == {
        "cube_drae",
        "range_m",
        "azimuth_rad",
        "elevation_rad",
        "arm",
    }
    assert not {"target", "cfar", "future_cube", "test"} & parameters


def test_execution_commit_is_separate_from_diagnosed_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_commit = "a" * 40
    responses = iter((execution_commit + "\n", ""))

    def fake_check_output(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return next(responses)

    monkeypatch.setattr(
        "scripts.diagnose_rb2_candidate_support.subprocess.check_output",
        fake_check_output,
    )
    verify_source_tree(Path("."), execution_commit)
    assert execution_commit != DIAGNOSED_PARENT_SOURCE_COMMIT

    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        verify_source_tree(Path("."), "8fe9280")


def test_candidate_support_reports_recall_and_weighted_coverage() -> None:
    candidates = torch.tensor([[0, 0, 0], [2, 0, 0]], dtype=torch.long)
    origin = torch.tensor(LATTICE_ORIGIN_XYZ_M)
    target_xyz = origin + torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [1.0, 0.2, 0.2],
        ]
    )
    target = torch.cat((target_xyz, torch.tensor([[1.0], [2.0], [7.0]])), dim=1)
    report = candidate_support_metrics(candidates, target)
    assert report["target_point_count"] == 3
    assert report["target_occupied_voxel_count"] == 2
    assert report["represented_target_occupied_voxel_count"] == 2
    assert report["target_occupied_voxel_recall"] == pytest.approx(1.0)
    assert report["target_confidence_coverage"] == pytest.approx(1.0)

    partial = candidate_support_metrics(candidates[:1], target)
    assert partial["target_occupied_voxel_recall"] == pytest.approx(0.5)
    assert partial["target_confidence_coverage"] == pytest.approx(0.3)


def test_duplicate_candidate_ids_are_rejected() -> None:
    candidates = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)
    target = torch.cat(
        (
            torch.tensor(LATTICE_ORIGIN_XYZ_M)[None] + 0.2,
            torch.ones((1, 1)),
        ),
        dim=1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        candidate_support_metrics(candidates, target)


def test_arm_gate_is_strict_over_every_frame() -> None:
    passing = [
        {
            "target_occupied_voxel_recall": MINIMUM_OCCUPIED_VOXEL_RECALL,
            "target_confidence_coverage": MINIMUM_CONFIDENCE_COVERAGE,
        },
        {
            "target_occupied_voxel_recall": 0.40,
            "target_confidence_coverage": 0.60,
        },
    ]
    assert arm_gate(passing)["passed"] is True
    failing = [*passing, {**passing[0], "target_occupied_voxel_recall": 0.199}]
    assert arm_gate(failing)["passed"] is False
    construction_failure = [
        *passing,
        {"construction_error": {"type": "CandidateCapacityError"}},
    ]
    result = arm_gate(construction_failure)
    assert result["passed"] is False
    assert result["construction_failure_count"] == 1


def _summary(name: str, count: int, passed: bool) -> dict:
    return {
        "arm": {"name": name, "candidate_count": count},
        "gate": {"passed": passed},
    }


def test_decision_reports_all_scores_at_minimum_passing_bank() -> None:
    decision = sweep_decision(
        [
            _summary("max_d_bank20k_r4", 20_000, False),
            _summary("sum_d_bank40k_r4", 40_000, True),
            _summary("log_sum_d_bank40k_r4", 40_000, True),
            _summary("max_d_bank80k_r4", 80_000, True),
        ],
        mode="one-frame",
    )
    assert decision["minimum_passing_candidate_count"] == 40_000
    assert decision["eligible_arms_at_minimum_bank"] == [
        "sum_d_bank40k_r4",
        "log_sum_d_bank40k_r4",
    ]
    assert decision["training_authorized"] is False


def test_decision_closes_only_current_activation_family() -> None:
    decision = sweep_decision(
        [_summary("max_d_bank80k_r4", 80_000, False)],
        mode="full-train",
    )
    assert decision["decision"] == "close_current_cube_activation_family"
    assert decision["minimum_passing_candidate_count"] is None
    assert "does not close all Cube-only representations" in decision["claim_boundary"]


def test_record_selection_never_uses_validation_or_test() -> None:
    train = [
        {"sequence": 1, "radar_index": 232, "partition": "train"},
        *[
            {"sequence": 2, "radar_index": index, "partition": "train"}
            for index in range(75)
        ],
    ]
    assert records_for_mode(train, "one-frame") == [train[0]]
    assert len(records_for_mode(train, "full-train")) == 76
    with pytest.raises(ValueError, match="all 76"):
        records_for_mode(train[:-1], "full-train")
    with pytest.raises(ValueError, match="Unsupported"):
        records_for_mode(train, "validation")
