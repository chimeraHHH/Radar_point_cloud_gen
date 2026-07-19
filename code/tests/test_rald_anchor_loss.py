import torch

from losses.rald_anchor import anchor_refinement_loss, nearest_target_assignment


def test_nearest_target_assignment_matches_full_cdist() -> None:
    source = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    target = torch.tensor([[1.9, 0.0, 0.0], [4.5, 0.0, 0.0]])

    distance, index = nearest_target_assignment(source, target, chunk_size=2)
    expected_distance, expected_index = torch.cdist(source, target).min(dim=1)

    torch.testing.assert_close(distance, expected_distance)
    torch.testing.assert_close(index, expected_index)


def test_scalar_anchor_loss_supports_no_cycle_ablation() -> None:
    cube = torch.rand(1, 64, 4, 4, 4)
    probability = torch.softmax(torch.randn(1, 3, 64), dim=-1)
    output = {
        "xyz_m": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        ),
        "anchor_xyz_m": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        ),
        "anchor_parent_confidence": torch.full((1, 3), 0.8),
        "anchor_cube_spectrum": probability,
        "point_cube_spectrum": probability,
        "doppler_probability": probability,
        "doppler_scalar_bin": torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True),
        "confidence": torch.full((1, 3), 0.8, requires_grad=True),
        "coordinates_rae": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]
        ),
        "offset_bins": torch.zeros(1, 3, 3, requires_grad=True),
    }
    target = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]
    )
    target_index = torch.tensor([[0, 0, 0], [2, 2, 2]])

    objective = anchor_refinement_loss(
        output,
        cube,
        target,
        target_index,
        cycle_variant="none",
    )

    assert torch.isfinite(objective.total)
    assert "doppler_scalar_smooth_l1_bins" in objective.components
    assert objective.components["cycle"].item() == 0.0
