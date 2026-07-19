import json

import numpy as np
import pytest
import torch

from cube_dense.dataset import KRadarDenseTargetDataset, KRadarRaLDLatentDataset
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

    uniform_points = sample_target_points(
        target,
        axes(),
        5,
        torch.Generator().manual_seed(31),
        sampling_mode="uniform",
    )
    assert uniform_points.shape == (5, 3)

    empty = sample_empty_indices(
        indices, (4, 3, 2), 12, torch.Generator().manual_seed(37)
    )
    occupied = {tuple(values) for values in indices.tolist()}
    assert all(tuple(values) not in occupied for values in empty.tolist())


def test_unknown_target_sampling_mode_is_rejected() -> None:
    target, _ = targets()

    with pytest.raises(ValueError, match="Unsupported target sampling mode"):
        sample_target_points(
            target,
            axes(),
            3,
            torch.Generator().manual_seed(1),
            sampling_mode="range_magic",
        )


def test_empty_query_sampling_is_not_biased_to_low_flat_indices() -> None:
    shape = (100, 10, 10)
    empty = sample_empty_indices(
        torch.tensor([[50, 5, 5]]),
        shape,
        1_000,
        torch.Generator().manual_seed(39),
    )
    flat = empty[:, 0] * shape[1] * shape[2] + empty[:, 1] * shape[2] + empty[:, 2]

    assert 3_500 < flat.float().mean().item() < 6_500


def test_occupancy_queries_contain_binary_positive_and_zero_negative_labels() -> None:
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
    assert torch.count_nonzero(labels == 1.0).item() == 4
    assert torch.count_nonzero(labels == 0.0).item() == 5


def test_occupied_voxel_queries_are_deterministic_and_inside_cells() -> None:
    target, indices = targets()
    kwargs = {
        "target_xyz_confidence": target,
        "target_rae_index": indices,
        "axes": axes(),
        "positive_count": 3,
        "negative_count": 2,
        "positive_sampling_mode": "uniform",
        "positive_query_location_mode": "occupied_voxel_uniform",
    }

    first = sample_occupancy_queries(
        **kwargs, generator=torch.Generator().manual_seed(43)
    )
    second = sample_occupancy_queries(
        **kwargs, generator=torch.Generator().manual_seed(43)
    )

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert torch.count_nonzero(first[1] == 1.0).item() == 3
    positive = first[0][first[1] == 1.0]
    centers = indices_to_normalized_rae(indices, axes())
    half_cell = torch.tensor([1.0 / 3.0, 1.0 / 2.0, 1.0])
    inside = (
        (positive[:, None] - centers[None]).abs()
        <= half_cell.view(1, 1, 3) + 1e-6
    ).all(dim=2)
    assert inside.any(dim=1).all()


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
    def __init__(self) -> None:
        self.prepare_calls = 0

    def prepare_decoder_latent(self, latent: torch.Tensor) -> torch.Tensor:
        self.prepare_calls += 1
        return latent

    def decode_queries(
        self, latent: torch.Tensor, queries: torch.Tensor
    ) -> torch.Tensor:
        del latent
        return 3.0 * queries[..., 0] + queries[..., 1] - queries[..., 2]

    def decode(self, latent: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        return self.decode_queries(latent, queries)


def test_chunked_grid_decoder_returns_exact_global_topk() -> None:
    latent = torch.zeros(1, 2, 2)

    decoder = QueryScoreDecoder()
    indices, confidence = decode_grid_topk(
        decoder, latent, axes(), point_count=5, query_chunk_size=7
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
    assert decoder.prepare_calls == 1


def test_target_only_dataset_does_not_require_cube_files(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    target, indices = targets()
    np.savez(
        cache_root / "seq01_radar_00002.npz",
        target_xyz_confidence=target.numpy(),
        target_rae_index=indices.numpy(),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "sequence": 1,
                        "radar_index": 2,
                        "partition": "train",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = KRadarDenseTargetDataset(cache_root, manifest, ("train",))
    item = dataset[0]

    assert len(dataset) == 1
    torch.testing.assert_close(item["target_xyz_confidence"], target)
    torch.testing.assert_close(item["target_rae_index"], indices)


def test_latent_dataset_joins_cube_target_and_frozen_latent(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    latent_root = tmp_path / "latent"
    cache_root.mkdir()
    latent_root.mkdir()
    target, indices = targets()
    np.savez(
        cache_root / "seq01_radar_00002.npz",
        target_xyz_confidence=target.numpy(),
        target_rae_index=indices.numpy(),
    )
    latent = np.arange(32, dtype=np.float32).reshape(8, 4)
    np.savez(
        latent_root / "seq01_radar_00002.npz",
        latent_mean=latent,
        posterior_kl=np.float32(0.25),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "sequence": 1,
                        "radar_index": 2,
                        "partition": "train",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    expected_cube = np.ones((64, 4, 3, 2), dtype=np.float32)
    monkeypatch.setattr("cube_dense.dataset.load_tesseract", lambda _: expected_cube)

    dataset = KRadarRaLDLatentDataset(
        data_root, cache_root, latent_root, manifest, ("train",)
    )
    item = dataset[0]

    torch.testing.assert_close(item["cube_drae"], torch.from_numpy(expected_cube))
    torch.testing.assert_close(item["latent_mean"], torch.from_numpy(latent))
    assert item["posterior_kl"].item() == 0.25
