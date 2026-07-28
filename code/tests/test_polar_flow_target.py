import numpy as np
import pytest
import torch

from cube_dense.polar_flow_target import (
    PolarLogitTransform,
    canonical_fixed_target,
    continuous_fixed_target,
    fit_empirical_range_cdf,
    fixed_scrambled_sobol_unit,
    hierarchical_one_to_one_transport,
    logit_unit,
)


def transform() -> PolarLogitTransform:
    return PolarLogitTransform(
        np.array([0.0, 10.0, 30.0, 60.0], dtype=np.float32),
        np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float32),
        azimuth_bounds_rad=(-0.8, 0.8),
        elevation_bounds_rad=(-0.25, 0.25),
    )


def target_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.linspace(1.0, 30.0, 117, dtype=np.float64),
        np.linspace(-1.0, 1.0, 161, dtype=np.float64),
        np.linspace(-0.4, 0.4, 81, dtype=np.float64),
    )


def nearest_indices(
    xyz: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    radius = np.linalg.norm(xyz, axis=1)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    elevation = np.arcsin(xyz[:, 2] / np.maximum(radius, 1e-8))
    coordinates = (radius, azimuth, elevation)
    return np.column_stack(
        [
            np.abs(axis[None, :] - values[:, None]).argmin(axis=1)
            for axis, values in zip(axes, coordinates, strict=True)
        ]
    ).astype(np.int64)


def planar_sparse_target() -> tuple[np.ndarray, np.ndarray]:
    first, second = np.meshgrid(
        np.linspace(-0.45, 0.45, 7),
        np.linspace(-0.30, 0.30, 7),
        indexing="ij",
    )
    xyz = np.column_stack(
        (
            np.full(first.size, 10.0),
            first.reshape(-1),
            second.reshape(-1),
        )
    ).astype(np.float32)
    confidence = np.linspace(0.3, 0.9, xyz.shape[0], dtype=np.float32)
    target = np.column_stack((xyz, confidence))
    return target, nearest_indices(xyz, target_axes())


def test_scrambled_sobol_source_is_fixed_and_open_unit() -> None:
    first = fixed_scrambled_sobol_unit(128, seed=20260716)
    second = fixed_scrambled_sobol_unit(128, seed=20260716)
    different = fixed_scrambled_sobol_unit(128, seed=20260717)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, different)
    assert first.shape == (128, 3)
    assert torch.all((first > 0.0) & (first < 1.0))
    assert torch.isfinite(logit_unit(first)).all()


def test_empirical_range_cdf_is_strict_and_invertible() -> None:
    samples = np.concatenate(
        (
            np.linspace(0.0, 20.0, 200),
            np.linspace(40.0, 60.0, 30),
        )
    )
    knots, cdf = fit_empirical_range_cdf(
        samples,
        lower_m=0.0,
        upper_m=60.0,
        knot_count=65,
    )

    assert np.all(np.diff(knots) > 0.0)
    assert np.all(np.diff(cdf) > 0.0)
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0


def test_polar_logit_transform_round_trips_physical_coordinates() -> None:
    physical = torch.tensor(
        [
            [2.0, -0.7, -0.2],
            [17.0, 0.0, 0.0],
            [52.0, 0.65, 0.2],
        ]
    )
    state = transform().encode_rae(physical)
    reconstructed = transform().decode_state(state)

    torch.testing.assert_close(reconstructed, physical, rtol=1e-5, atol=1e-5)


def test_xyz_round_trip_preserves_points_inside_fov() -> None:
    coordinates = torch.tensor(
        [
            [8.0, -0.4, 0.1],
            [25.0, 0.2, -0.15],
            [50.0, 0.65, 0.2],
        ]
    )
    radius, azimuth, elevation = coordinates.unbind(dim=1)
    xyz = torch.stack(
        (
            radius * torch.cos(elevation) * torch.cos(azimuth),
            radius * torch.cos(elevation) * torch.sin(azimuth),
            radius * torch.sin(elevation),
        ),
        dim=1,
    )

    reconstructed = transform().decode_xyz(transform().encode_xyz(xyz))

    torch.testing.assert_close(reconstructed, xyz, rtol=1e-5, atol=1e-5)


def test_canonical_target_is_exact_unique_and_deterministic() -> None:
    generator = np.random.default_rng(19)
    xyz = generator.normal(size=(80, 3)).astype(np.float32)
    confidence = generator.uniform(size=(80, 1)).astype(np.float32)
    target = np.concatenate((xyz, confidence), axis=1)

    first = canonical_fixed_target(target, point_count=32)
    second = canonical_fixed_target(target[::-1].copy(), point_count=32)

    assert first.xyz_m.shape == (32, 3)
    assert np.unique(first.xyz_m.numpy(), axis=0).shape[0] == 32
    assert first.hashes["xyz_sha256"] == second.hashes["xyz_sha256"]


def test_canonical_target_refuses_copy_fill() -> None:
    target = torch.zeros(12, 4)
    target[:, 0] = torch.arange(12)

    with pytest.raises(ValueError, match="without replacement"):
        canonical_fixed_target(target, point_count=16)


def test_continuous_target_preserves_legacy_dense_endpoint() -> None:
    generator = np.random.default_rng(31)
    xyz = generator.normal(size=(80, 3)).astype(np.float32)
    xyz[:, 0] += 12.0
    target = np.column_stack(
        (xyz, generator.uniform(size=80).astype(np.float32))
    )
    axes = target_axes()
    target_index = nearest_indices(xyz, axes)

    legacy = canonical_fixed_target(target, point_count=32)
    endpoint = continuous_fixed_target(
        target,
        target_index,
        range_axis_m=axes[0],
        azimuth_axis_rad=axes[1],
        elevation_axis_rad=axes[2],
        point_count=32,
    )

    torch.testing.assert_close(endpoint.xyz_m, legacy.xyz_m, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        endpoint.confidence,
        legacy.confidence,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(endpoint.source_indices, legacy.source_indices)
    assert endpoint.hashes == legacy.hashes
    assert endpoint.rae_indices is None
    assert endpoint.lifting is None


def test_continuous_target_lifts_sparse_planar_support_deterministically() -> None:
    target, target_index = planar_sparse_target()
    axes = target_axes()

    first = continuous_fixed_target(
        target,
        target_index,
        range_axis_m=axes[0],
        azimuth_axis_rad=axes[1],
        elevation_axis_rad=axes[2],
        point_count=64,
        minimum_spacing_m=0.05,
    )
    second = continuous_fixed_target(
        target[::-1].copy(),
        target_index[::-1].copy(),
        range_axis_m=axes[0],
        azimuth_axis_rad=axes[1],
        elevation_axis_rad=axes[2],
        point_count=64,
        minimum_spacing_m=0.05,
    )

    assert first.xyz_m.shape == (64, 3)
    assert first.rae_indices is not None
    assert first.rae_indices.shape == (64, 3)
    assert first.is_lifted is not None
    assert bool(first.is_lifted.any())
    torch.testing.assert_close(first.xyz_m, second.xyz_m, rtol=0.0, atol=0.0)
    assert torch.equal(first.rae_indices, second.rae_indices)

    distances = torch.cdist(first.xyz_m, first.xyz_m)
    distances.fill_diagonal_(float("inf"))
    assert float(distances.min().item()) >= 0.05 - 1e-6

    occupied = {tuple(index) for index in target_index.tolist()}
    assert all(tuple(index) in occupied for index in first.rae_indices.tolist())
    assert first.lifting is not None
    assert first.lifting["mode"] == "local_tangent_occupied_rae_lifting"
    assert first.lifting["doppler_label_source"] == (
        "current_frame_cube_spectrum_at_endpoint_rae_index"
    )
    assert first.lifting["exact_point_count"] == 64
    assert first.lifting["minimum_pair_distance_m"] >= 0.05 - 1e-6
    assert first.hashes["xyz_sha256"] == second.hashes["xyz_sha256"]
    assert (
        first.hashes["doppler_rae_indices_sha256"]
        == second.hashes["doppler_rae_indices_sha256"]
    )


def test_continuous_target_provenance_tracks_parent_and_tangent_support() -> None:
    target, target_index = planar_sparse_target()
    axes = target_axes()
    endpoint = continuous_fixed_target(
        target,
        target_index,
        range_axis_m=axes[0],
        azimuth_axis_rad=axes[1],
        elevation_axis_rad=axes[2],
        point_count=64,
    )

    assert endpoint.is_lifted is not None
    assert endpoint.tangent_source_indices is not None
    assert endpoint.tangent_offset_uv_m is not None
    lifted = endpoint.is_lifted
    parents = endpoint.source_indices[lifted]
    supports = endpoint.tangent_source_indices[lifted]
    offsets = endpoint.tangent_offset_uv_m[lifted]

    assert torch.all((parents >= 0) & (parents < target.shape[0]))
    assert torch.all(supports[:, 0] >= 0)
    assert torch.all((supports >= -1) & (supports < target.shape[0]))
    assert torch.all(torch.linalg.vector_norm(offsets, dim=1) <= 0.6 + 1e-6)
    assert torch.all(torch.linalg.vector_norm(offsets, dim=1) > 0.0)


def test_continuous_target_refuses_unsupported_capacity() -> None:
    target = np.array([[10.0, 0.0, 0.0, 0.8]], dtype=np.float32)
    axes = target_axes()
    target_index = nearest_indices(target[:, :3], axes)

    with pytest.raises(ValueError, match="capacity"):
        continuous_fixed_target(
            target,
            target_index,
            range_axis_m=axes[0],
            azimuth_axis_rad=axes[1],
            elevation_axis_rad=axes[2],
            point_count=4,
        )


def test_hierarchical_transport_is_deterministic_one_to_one() -> None:
    source = logit_unit(fixed_scrambled_sobol_unit(32, seed=7))
    target = source.roll(5, dims=0) + torch.tensor([0.1, -0.2, 0.05])

    first = hierarchical_one_to_one_transport(
        source,
        target,
        block_size=4,
    )
    second = hierarchical_one_to_one_transport(
        source,
        target,
        block_size=4,
    )

    first.validate(32)
    assert torch.equal(
        torch.sort(first.source_to_target).values,
        torch.arange(32),
    )
    assert torch.equal(first.source_to_target, second.source_to_target)
    assert first.hashes == second.hashes
    assert first.total_squared_cost == second.total_squared_cost


def test_hierarchical_transport_rejects_nondivisible_blocks() -> None:
    source = torch.randn(30, 3)
    target = torch.randn(30, 3)

    with pytest.raises(ValueError, match="divide"):
        hierarchical_one_to_one_transport(source, target, block_size=8)
