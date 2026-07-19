import torch

from eval.temporal_cube import ego_aligned_consistency_report
from losses.temporal_consistency import ego_aligned_match, ego_aligned_match_loss


def test_ego_aligned_match_uses_pose_without_doppler_convention() -> None:
    transform = torch.eye(4)
    transform[0, 3] = 2.0
    previous = torch.tensor([[10.0, 0.0, 0.0], [20.0, 1.0, 0.0]])
    current = torch.tensor([[12.0, 0.0, 0.0], [22.0, 1.0, 0.0]])
    confidence = torch.ones(2)

    match = ego_aligned_match(
        previous, confidence, current, confidence, transform
    )

    torch.testing.assert_close(match.match_distance_m, torch.zeros(2))
    torch.testing.assert_close(ego_aligned_match_loss(match), torch.tensor(0.0))


def test_ego_aligned_report_exposes_matching_and_flicker() -> None:
    previous = torch.tensor([[10.0, 0.0, 0.0], [20.0, 1.0, 0.0]])
    current = previous.clone()
    confidence = torch.ones(2)
    match = ego_aligned_match(
        previous, confidence, current, confidence, torch.eye(4)
    )

    report = ego_aligned_consistency_report(
        match,
        confidence,
        current,
        confidence,
        torch.linspace(0.0, 120.0, 256),
        torch.linspace(-1.0, 1.0, 107),
        torch.linspace(-0.3, 0.3, 37),
    )

    assert report["ego_aligned_matched_distance_mean_m"] == 0.0
    assert report["occupancy_flicker"] <= 1e-6
