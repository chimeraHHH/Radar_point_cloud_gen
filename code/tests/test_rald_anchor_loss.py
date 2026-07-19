import torch

from losses.rald_anchor import nearest_target_assignment


def test_nearest_target_assignment_matches_full_cdist() -> None:
    source = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    target = torch.tensor([[1.9, 0.0, 0.0], [4.5, 0.0, 0.0]])

    distance, index = nearest_target_assignment(source, target, chunk_size=2)
    expected_distance, expected_index = torch.cdist(source, target).min(dim=1)

    torch.testing.assert_close(distance, expected_distance)
    torch.testing.assert_close(index, expected_index)
