import pytest
import torch

from losses.rb2_voxel_slot import (
    build_voxel_slot_targets,
    candidate_support_report,
    rb2_voxel_slot_loss,
)
from models.rb2_voxel_slot_model import (
    CandidateCapacityError,
    GUARANTEED_MINIMUM_DISTANCE_M,
    LATTICE_ORIGIN_XYZ_M,
    MINIMUM_EXPORT_DISTANCE_M,
    OUTPUT_POINT_QUOTAS,
    OUTPUT_VOXEL_QUOTAS,
    RB2VoxelSlotNet,
    SLOT_RESIDUAL_LIMIT_XYZ_M,
    SLOTS_PER_VOXEL,
    VOXEL_SIZE_M,
    candidate_voxels_from_cube,
    export_exact_voxel_slots,
    lattice_indices_to_centers,
    range_stratum_codes,
)


def toy_axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([5.0, 10.0, 20.0, 40.0, 50.0, 80.0, 100.0]),
        torch.tensor([-0.40, -0.20, 0.0, 0.20, 0.40]),
        torch.tensor([-0.10, 0.0, 0.10]),
    )


def toy_cube() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260716)
    return torch.rand((1, 4, 7, 5, 3), generator=generator) * 100.0


def toy_candidates(
    quotas: tuple[int, int, int] = (8, 4, 2),
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    axes = toy_axes()
    candidates = candidate_voxels_from_cube(
        toy_cube(),
        *axes,
        candidate_voxel_quotas=quotas,
        seed_multiplier=4,
        maximum_neighbor_radius=3,
    )
    return candidates, axes


def test_frozen_representation_contract() -> None:
    assert VOXEL_SIZE_M == 0.40
    assert SLOTS_PER_VOXEL == 4
    assert OUTPUT_POINT_QUOTAS == (8_000, 1_700, 300)
    assert OUTPUT_VOXEL_QUOTAS == (2_000, 425, 75)
    assert sum(OUTPUT_POINT_QUOTAS) == 10_000
    assert GUARANTEED_MINIMUM_DISTANCE_M >= MINIMUM_EXPORT_DISTANCE_M


def test_cube_only_candidate_builder_is_deterministic_and_stratified() -> None:
    candidates, _ = toy_candidates()
    repeated, _ = toy_candidates()
    assert torch.equal(candidates, repeated)
    assert candidates.shape == (1, 14, 3)
    assert torch.unique(candidates[0], dim=0).shape[0] == 14
    codes = range_stratum_codes(lattice_indices_to_centers(candidates))[0]
    assert [(codes == code).sum().item() for code in range(3)] == [8, 4, 2]


def test_model_predicts_bounded_slots_inside_immutable_voxels() -> None:
    candidates, axes = toy_candidates()
    cube = toy_cube()
    model = RB2VoxelSlotNet(doppler_bins=4, hidden_dim=32, depth=2)
    prediction = model(cube, candidates, *axes)
    assert prediction.occupancy_logits.shape == (1, 14)
    assert prediction.slot_confidence_logits.shape == (1, 14, 4)
    assert prediction.local_offsets_xyz_m.shape == (1, 14, 4, 3)
    assert prediction.slot_xyz_m.shape == (1, 14, 4, 3)
    residual_limit = torch.tensor(SLOT_RESIDUAL_LIMIT_XYZ_M)
    assert torch.all(
        (prediction.slot_xyz_m - prediction.candidate_centers_xyz_m[:, :, None])
        .abs()
        <= (0.10 + residual_limit.max() + 1e-6)
    )
    origin = torch.tensor(LATTICE_ORIGIN_XYZ_M)
    lower = origin + candidates.float() * VOXEL_SIZE_M
    upper = lower + VOXEL_SIZE_M
    assert torch.all(prediction.slot_xyz_m >= lower[:, :, None] + 0.029)
    assert torch.all(prediction.slot_xyz_m <= upper[:, :, None] - 0.029)
    within_voxel = torch.cdist(
        prediction.slot_xyz_m[0],
        prediction.slot_xyz_m[0],
    )
    diagonal = torch.eye(SLOTS_PER_VOXEL, dtype=torch.bool)[None]
    minimum = within_voxel.masked_fill(diagonal, torch.inf).min().detach()
    assert float(minimum) >= 0.059


def test_export_selects_whole_voxels_without_repair() -> None:
    candidates, axes = toy_candidates((4, 2, 2))
    model = RB2VoxelSlotNet(doppler_bins=4, hidden_dim=24, depth=2)
    prediction = model(toy_cube(), candidates, *axes)
    exported = export_exact_voxel_slots(
        prediction,
        point_quotas=(8, 4, 4),
    )[0]
    assert exported.xyz_m.shape == (16, 3)
    assert exported.selected_candidate_indices_xyz.shape == (4, 3)
    assert exported.report["range_point_quotas"] == [8, 4, 4]
    assert exported.report["copy_used"] is False
    assert exported.report["padding_used"] is False
    assert exported.report["jitter_used"] is False
    assert exported.report["ground_truth_used_for_candidate_or_export"] is False
    assert (
        torch.unique(exported.selected_candidate_indices_xyz, dim=0).shape[0]
        == exported.selected_candidate_indices_xyz.shape[0]
    )


def test_export_capacity_shortfall_is_a_hard_failure() -> None:
    candidates, axes = toy_candidates((4, 2, 1))
    model = RB2VoxelSlotNet(doppler_bins=4, hidden_dim=24, depth=2)
    prediction = model(toy_cube(), candidates, *axes)
    with pytest.raises(CandidateCapacityError, match="below the frozen export quota"):
        export_exact_voxel_slots(prediction, point_quotas=(8, 4, 8))


def test_target_assignment_and_loss_are_differentiable() -> None:
    candidates, axes = toy_candidates((4, 2, 1))
    centers = lattice_indices_to_centers(candidates)
    target_xyz = torch.stack(
        (
            centers[0, 0] + torch.tensor([-0.12, -0.12, 0.02]),
            centers[0, 0] + torch.tensor([-0.12, 0.12, -0.02]),
            centers[0, 1] + torch.tensor([0.12, -0.12, 0.01]),
            centers[0, 1] + torch.tensor([0.12, 0.12, -0.01]),
        )
    )
    target = torch.cat((target_xyz, torch.ones((4, 1))), dim=1)
    targets = build_voxel_slot_targets(candidates, target)
    report = candidate_support_report(targets)
    assert targets.occupancy.sum().item() == 2
    assert targets.slot_confidence.sum().item() == 4
    assert report["minimum_target_occupied_voxel_recall"] == 1.0
    assert report["minimum_target_confidence_coverage"] == 1.0

    model = RB2VoxelSlotNet(doppler_bins=4, hidden_dim=24, depth=2)
    prediction = model(toy_cube(), candidates, *axes)
    loss, components = rb2_voxel_slot_loss(prediction, targets)
    assert torch.isfinite(loss)
    assert set(components) == {
        "total",
        "occupancy",
        "slot_confidence",
        "local_offset",
        "positive_voxel_fraction",
        "positive_slot_fraction",
    }
    loss.backward()
    assert model.offset_head.weight.grad is not None
    assert torch.isfinite(model.offset_head.weight.grad).all()


def test_duplicate_candidate_ids_are_rejected_by_supervision() -> None:
    candidates, _ = toy_candidates((4, 2, 1))
    candidates[0, 1] = candidates[0, 0]
    target = torch.tensor([[10.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="duplicate voxel IDs"):
        build_voxel_slot_targets(candidates, target)
