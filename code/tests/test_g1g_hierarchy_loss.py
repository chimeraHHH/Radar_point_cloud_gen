import torch

from losses.g1g_hierarchy import (
    bounded_child_diversity_loss,
    center_repulsion_loss,
    g1g_hierarchy_loss,
    physical_child_bound_loss,
)


def hierarchy_output(
    *,
    collapsed: bool = False,
    confidence: float = 0.8,
    center_logit: float = 2.0,
) -> dict[str, torch.Tensor]:
    center_xyz = torch.tensor(
        [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    if collapsed:
        offsets = torch.zeros(1, 2, 4, 3)
    else:
        offsets = torch.tensor(
            [
                [
                    [
                        [-0.20, 0.00, 0.00],
                        [0.20, 0.00, 0.00],
                        [0.00, -0.20, 0.00],
                        [0.00, 0.20, 0.00],
                    ],
                    [
                        [-0.20, 0.00, 0.00],
                        [0.20, 0.00, 0.00],
                        [0.00, -0.20, 0.00],
                        [0.00, 0.20, 0.00],
                    ],
                ]
            ]
        )
    offsets = offsets.requires_grad_(True)
    child_xyz = center_xyz[:, :, None, :] + offsets
    confidence_logit = torch.full(
        (1, 8),
        torch.logit(torch.tensor(confidence)).item(),
        requires_grad=True,
    )
    center_score_logit = torch.full(
        (1, 2),
        center_logit,
        requires_grad=True,
    )
    return {
        "xyz_m": child_xyz.reshape(1, 8, 3),
        "center_coordinates_rae": torch.zeros(1, 2, 3),
        "center_xyz_m": center_xyz,
        "child_xyz_m": child_xyz,
        "child_physical_offset_m": offsets,
        "center_cell_diagonal_m": torch.ones(1, 2, 1),
        "confidence_logit": confidence_logit,
        "confidence": torch.sigmoid(confidence_logit),
        "center_score_logit": center_score_logit,
    }


def target() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
        ]
    )


def compute(output: dict[str, torch.Tensor]):
    return g1g_hierarchy_loss(
        output,
        target(),
        expected_point_count=8,
        expected_center_count=2,
        expected_children_per_center=4,
        nearest_other_chunk_size=1,
    )


def test_loss_directly_supervises_geometry_existence_and_centers() -> None:
    output = hierarchy_output()
    loss = compute(output)

    assert set(loss.existence_targets) == {"child", "center"}
    assert loss.existence_targets["child"].shape == (8,)
    assert loss.existence_targets["center"].shape == (2,)
    assert loss.components["geometry_chamfer"] > 0.0
    assert loss.components["center_coverage_mean_distance_m"] == 0.0
    assert loss.components["child_existence_confidence"] > 0.0
    assert loss.components["center_existence_confidence"] > 0.0
    assert loss.components["child_physical_bound_violation"] == 0.0

    loss.total.backward()
    assert output["center_xyz_m"].grad is not None
    assert torch.count_nonzero(output["center_xyz_m"].grad) > 0
    assert output["child_physical_offset_m"].grad is not None
    assert torch.count_nonzero(output["child_physical_offset_m"].grad) > 0
    assert output["confidence_logit"].grad is not None
    assert torch.count_nonzero(output["confidence_logit"].grad) > 0
    assert output["center_score_logit"].grad is not None
    assert torch.count_nonzero(output["center_score_logit"].grad) > 0


def test_existence_confidence_rewards_matched_predictions() -> None:
    correct = compute(hierarchy_output(confidence=0.9, center_logit=3.0))
    wrong = compute(hierarchy_output(confidence=0.1, center_logit=-3.0))

    assert (
        correct.components["child_existence_confidence"]
        < wrong.components["child_existence_confidence"]
    )
    assert (
        correct.components["center_existence_confidence"]
        < wrong.components["center_existence_confidence"]
    )


def test_child_existence_requires_logits_for_stable_bce() -> None:
    output = hierarchy_output()
    del output["confidence_logit"]

    try:
        compute(output)
    except KeyError as error:
        assert error.args == ("confidence_logit",)
    else:
        raise AssertionError("G1G loss accepted probabilities without logits")


def test_outlier_hinge_activates_only_beyond_two_metres() -> None:
    near = hierarchy_output()
    far = hierarchy_output()
    far["xyz_m"] = far["xyz_m"] + torch.tensor([0.0, 5.0, 0.0])

    near_loss = compute(near)
    far_loss = compute(far)
    assert near_loss.components["outlier_hinge_2m"] == 0.0
    assert far_loss.components["outlier_hinge_2m"] > 0.0


def test_center_repulsion_penalizes_collapsed_allocations() -> None:
    collapsed = torch.zeros(4, 3)
    separated = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    collapsed_loss, collapsed_nearest = center_repulsion_loss(
        collapsed,
        minimum_distance_m=0.1,
        chunk_size=2,
    )
    separated_loss, separated_nearest = center_repulsion_loss(
        separated,
        minimum_distance_m=0.1,
        chunk_size=2,
    )

    assert collapsed_loss > separated_loss
    assert torch.all(collapsed_nearest == 0.0)
    assert torch.all(separated_nearest > 0.1)


def test_final_repulsion_detects_overlap_across_different_parents() -> None:
    output = hierarchy_output()
    baseline = compute(output)
    overlapped = hierarchy_output()
    overlapped_children = overlapped["child_xyz_m"].clone()
    overlapped_children[0, 1, 0] = overlapped_children[0, 0, 0]
    overlapped["child_xyz_m"] = overlapped_children
    overlapped["xyz_m"] = overlapped_children.reshape(1, 8, 3)

    collapsed = compute(overlapped)

    assert (
        collapsed.components["final_point_repulsion"]
        > baseline.components["final_point_repulsion"]
    )


def test_child_diversity_is_scale_normalized_and_detects_collapse() -> None:
    collapsed = torch.zeros(2, 4, 3)
    spread = hierarchy_output()["child_physical_offset_m"][0].detach()
    diagonal = torch.ones(2, 1)
    collapsed_loss, collapsed_fraction = bounded_child_diversity_loss(
        collapsed,
        diagonal,
        minimum_diagonal_fraction=0.2,
    )
    spread_loss, spread_fraction = bounded_child_diversity_loss(
        spread,
        diagonal,
        minimum_diagonal_fraction=0.2,
    )

    assert collapsed_loss > spread_loss
    assert torch.all(collapsed_fraction == 0.0)
    assert torch.all(spread_fraction > 0.2)


def test_physical_bound_loss_reports_only_excess() -> None:
    within = torch.zeros(2, 4, 3)
    outside = within.clone()
    outside[0, 0, 0] = 1.5
    diagonal = torch.ones(2, 1)

    within_loss, within_excess = physical_child_bound_loss(within, diagonal)
    outside_loss, outside_excess = physical_child_bound_loss(outside, diagonal)
    assert within_loss == 0.0
    assert torch.count_nonzero(within_excess) == 0
    assert outside_loss > 0.0
    assert outside_excess[0, 0] == 0.5


def test_loss_rejects_nonformal_cardinality_without_explicit_test_override() -> None:
    output = hierarchy_output()
    try:
        g1g_hierarchy_loss(output, target())
    except ValueError as error:
        assert "frozen point count" in str(error)
    else:
        raise AssertionError("G1G loss accepted a nonformal test cardinality")
