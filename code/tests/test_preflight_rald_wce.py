import pytest
import torch

from scripts.preflight_rald_wce import (
    build_stage0_query_targets,
    select_max_target_index,
    select_wrong_condition_index,
)


def test_max_target_selection_is_deterministic_and_first_on_ties() -> None:
    assert select_max_target_index([10, 30, 20, 30]) == 1


@pytest.mark.parametrize("counts", ([], [1, 0], [-1]))
def test_max_target_selection_rejects_invalid_counts(
    counts: list[int],
) -> None:
    with pytest.raises(ValueError, match="positive"):
        select_max_target_index(counts)


def test_wrong_condition_selection_is_cross_scene_and_largest() -> None:
    records = [
        {"sequence": 1, "radar_index": 10},
        {"sequence": 2, "radar_index": 20},
        {"sequence": 3, "radar_index": 30},
        {"sequence": 2, "radar_index": 40},
    ]
    counts = [100, 80, 90, 90]

    assert select_wrong_condition_index(records, counts, 0) == 2
    assert select_wrong_condition_index(records, counts, 2) == 0


def test_wrong_condition_rejects_single_scene() -> None:
    records = [
        {"sequence": 1, "radar_index": 10},
        {"sequence": 1, "radar_index": 20},
    ]
    with pytest.raises(ValueError, match="another scene"):
        select_wrong_condition_index(records, [10, 20], 0)


def test_stage0_query_targets_are_deterministic_and_balanced() -> None:
    target = torch.tensor(
        [
            [1, 1, 1],
            [2, 2, 1],
            [3, 3, 1],
            [4, 4, 1],
            [5, 5, 1],
            [6, 1, 1],
        ],
        dtype=torch.long,
    )
    first = build_stage0_query_targets(
        target,
        spatial_shape=(8, 7, 3),
        query_count=10,
        seed=37,
    )
    second = build_stage0_query_targets(
        target,
        spatial_shape=(8, 7, 3),
        query_count=10,
        seed=37,
    )

    assert first["normalized_rae"].shape == (1, 10, 3)
    assert first["occupancy_target"].shape == (1, 10)
    assert first["residual_target_bins"].shape == (1, 10, 3)
    assert first["positive_count"] == 5
    assert first["negative_count"] == 5
    assert first["unique_target_cell_count"] == 6
    assert torch.count_nonzero(first["occupancy_target"]) == 5
    assert torch.all(first["normalized_rae"].abs() <= 1.0)
    assert torch.count_nonzero(first["residual_target_bins"]) > 0
    torch.testing.assert_close(
        first["normalized_rae"],
        second["normalized_rae"],
    )
    torch.testing.assert_close(
        first["occupancy_target"],
        second["occupancy_target"],
    )
    torch.testing.assert_close(
        first["residual_target_bins"],
        second["residual_target_bins"],
    )


def test_stage0_query_targets_reject_out_of_bounds_target() -> None:
    target = torch.tensor([[8, 1, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="outside"):
        build_stage0_query_targets(
            target,
            spatial_shape=(8, 7, 3),
            query_count=10,
        )
