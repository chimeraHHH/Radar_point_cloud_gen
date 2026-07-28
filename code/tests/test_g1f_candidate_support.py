import torch

from eval.g1f_candidate_support import (
    EXPORT_COUNT,
    ORACLE_ARTIFACT_LABEL,
    PROPOSAL_COUNT,
    diagnostic_artifact_label,
    nearest_candidate_assignment,
    select_candidate_support_oracle,
)


def synthetic_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_indices = torch.arange(PROPOSAL_COUNT, dtype=torch.long) * 3 + 7
    candidate_xyz = torch.stack(
        (
            torch.linspace(1.0, 29.0, PROPOSAL_COUNT),
            torch.zeros(PROPOSAL_COUNT),
            torch.zeros(PROPOSAL_COUNT),
        ),
        dim=1,
    )
    target_xyz = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [26.0, 0.0, 0.0],
        ]
    )
    return candidate_xyz, candidate_indices, target_xyz


def test_oracle_exports_exact_count_with_unique_candidate_indices() -> None:
    candidate_xyz, candidate_indices, target_xyz = synthetic_inputs()
    result = select_candidate_support_oracle(
        candidate_xyz,
        candidate_indices,
        target_xyz,
    )

    assert result.selected_xyz_m.shape == (EXPORT_COUNT, 3)
    assert result.selected_pool_indices.shape == (EXPORT_COUNT,)
    assert result.selected_candidate_indices.shape == (EXPORT_COUNT,)
    assert torch.unique(result.selected_candidate_indices).numel() == EXPORT_COUNT


def test_oracle_selection_is_deterministic() -> None:
    inputs = synthetic_inputs()
    first = select_candidate_support_oracle(*inputs)
    second = select_candidate_support_oracle(*inputs)

    torch.testing.assert_close(
        first.selected_pool_indices,
        second.selected_pool_indices,
    )
    torch.testing.assert_close(
        first.selected_candidate_indices,
        second.selected_candidate_indices,
    )


def test_oracle_never_reuses_a_selected_pool_index() -> None:
    candidate_xyz, candidate_indices, target_xyz = synthetic_inputs()
    result = select_candidate_support_oracle(
        candidate_xyz,
        candidate_indices.flip(0),
        target_xyz,
    )

    assert torch.unique(result.selected_pool_indices).numel() == EXPORT_COUNT
    assert torch.unique(result.selected_candidate_indices).numel() == EXPORT_COUNT


def test_nearest_assignment_tie_uses_candidate_index_across_chunks() -> None:
    target_xyz = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    candidate_xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    candidate_indices = torch.tensor([9, 3])

    distance, assigned = nearest_candidate_assignment(
        target_xyz,
        candidate_xyz,
        candidate_indices,
        chunk_size=1,
    )

    torch.testing.assert_close(distance, torch.ones(2))
    assert assigned.tolist() == [1, 1]


def test_coverage_assignment_keeps_support_for_separated_targets() -> None:
    clustered_x = torch.linspace(5.0, 5.1, PROPOSAL_COUNT - 1)
    candidate_xyz = torch.stack(
        (
            torch.cat((clustered_x, torch.tensor([25.2]))),
            torch.zeros(PROPOSAL_COUNT),
            torch.zeros(PROPOSAL_COUNT),
        ),
        dim=1,
    )
    candidate_indices = torch.arange(PROPOSAL_COUNT, dtype=torch.long)
    target_xyz = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [25.0, 0.0, 0.0],
        ]
    )

    naive_distance = torch.minimum(
        (candidate_xyz[:, 0] - target_xyz[0, 0]).abs(),
        (candidate_xyz[:, 0] - target_xyz[1, 0]).abs(),
    )
    naive_selected = torch.argsort(
        naive_distance,
        stable=True,
    )[:EXPORT_COUNT]
    separated_candidate = PROPOSAL_COUNT - 1
    assert not bool((naive_selected == separated_candidate).any())

    result = select_candidate_support_oracle(
        candidate_xyz,
        candidate_indices,
        target_xyz,
    )

    assert bool((result.selected_pool_indices == separated_candidate).any())
    near = result.per_range_support["range_0_30m"]
    assert near["selected_covered_target_mass_fraction"] == 1.0
    assert near["gt_recall_from_selected_0p5m"] == 1.0


def test_oracle_artifact_is_explicitly_diagnostic_and_unattainable() -> None:
    label = diagnostic_artifact_label()

    assert label["label"] == ORACLE_ARTIFACT_LABEL
    assert label["diagnostic"] is True
    assert label["unattainable"] is True
    assert label["ground_truth_used_for_selection"] is True
    assert label["eligible_as_method_result"] is False
