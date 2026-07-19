import numpy as np
import torch

from cube_dense.kradar import KRadarAxes
from cube_dense.rald_adapter import (
    decode_grid_topk,
    indices_to_normalized_rae,
    rae_sum_condition,
    rald_occupancy_loss,
    sample_empty_indices,
    sample_occupancy_queries,
    sample_target_points,
    xyz_to_normalized_rae,
)


def axes() -> KRadarAxes:
    return KRadarAxes(
        doppler_mps=np.linspace(-8.0, 8.0, 64),
        range_m=np.array([1.0, 2.0, 3.0, 4.0]),
        azimuth_rad=np.array([-0.5, 0.0, 0.5]),
        elevation_rad=np.array([-0.2, 0.2]),
    )


def targets() -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.tensor([[0, 0, 0], [1, 1, 1], [3, 2, 0]])
    axis = axes()
    radius = torch.as_tensor(axis.range_m)[indices[:, 0]]
    azimuth = torch.as_tensor(axis.azimuth_rad)[indices[:, 1]]
    elevation = torch.as_tensor(axis.elevation_rad)[indices[:, 2]]
    cosine = torch.cos(elevation)
    xyz = torch.stack(
        (
            radius * cosine * torch.cos(azimuth),
            radius * cosine * torch.sin(azimuth),
            radius * torch.sin(elevation),
        ),
        dim=1,
    ).float()
    confidence = torch.tensor([0.9, 0.7, 0.5]).unsqueeze(1)
    return torch.cat((xyz, confidence), dim=1), indices


def test_xyz_and_grid_indices_share_normalized_coordinates() -> None:
    target, indices = targets()

    continuous = xyz_to_normalized_rae(target[:, :3], axes())
    grid = indices_to_normalized_rae(indices, axes())

    torch.testing.assert_close(continuous, grid, rtol=1e-5, atol=1e-6)


def test_rae_sum_condition_uses_all_doppler_bins() -> None:
    cube = torch.zeros(1, 64, 2, 2, 2)
    cube[:, 3] = 9.0
    cube[:, 17] = 90.0

    condition = rae_sum_condition(cube, center=2.0, scale=1.0)

    expected = torch.log10(torch.tensor(100.0)) - 2.0
    torch.testing.assert_close(condition.unique(), expected.reshape(1))


def test_sampling_is_deterministic_and_empty_queries_exclude_targets() -> None:
    target, indices = targets()
    first_generator = torch.Generator().manual_seed(31)
    second_generator = torch.Generator().manual_seed(31)

    first_points = sample_target_points(target, axes(), 5, first_generator)
    second_points = sample_target_points(target, axes(), 5, second_generator)
    torch.testing.assert_close(first_points, second_points)

    empty = sample_empty_indices(
        indices, (4, 3, 2), 12, torch.Generator().manual_seed(37)
    )
    occupied = {tuple(values) for values in indices.tolist()}
    assert all(tuple(values) not in occupied for values in empty.tolist())


def test_occupancy_queries_contain_soft_positive_and_zero_negative_labels() -> None:
    target, indices = targets()

    queries, labels = sample_occupancy_queries(
        target,
        indices,
        axes(),
        positive_count=4,
        negative_count=5,
        generator=torch.Generator().manual_seed(41),
    )

    assert queries.shape == (9, 3)
    assert labels.shape == (9,)
    assert torch.count_nonzero(labels).item() == 4
    assert torch.count_nonzero(labels == 0.0).item() == 5


def test_occupancy_loss_preserves_positive_negative_and_kl_terms() -> None:
    logits = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    labels = torch.tensor([[0.8, 0.0, 0.6]])

    loss, terms = rald_occupancy_loss(logits, labels, torch.tensor([0.3]))
    loss.backward()

    assert torch.isfinite(loss)
    assert set(terms) == {"positive_bce", "negative_bce", "kl"}
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 3


class QueryScoreDecoder:
    def decode(self, latent: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        del latent
        return 3.0 * queries[..., 0] + queries[..., 1] - queries[..., 2]


def test_chunked_grid_decoder_returns_exact_global_topk() -> None:
    latent = torch.zeros(1, 2, 2)

    indices, confidence = decode_grid_topk(
        QueryScoreDecoder(), latent, axes(), point_count=5, query_chunk_size=7
    )

    all_indices = torch.cartesian_prod(
        torch.arange(4), torch.arange(3), torch.arange(2)
    )
    queries = indices_to_normalized_rae(all_indices, axes())
    expected = torch.topk(
        3.0 * queries[:, 0] + queries[:, 1] - queries[:, 2], 5
    ).indices
    assert {tuple(row) for row in indices.tolist()} == {
        tuple(row) for row in all_indices[expected].tolist()
    }
    assert confidence.shape == (5,)
