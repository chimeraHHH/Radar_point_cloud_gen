import numpy as np
import pytest
import torch

from cube_dense.polar_flow_target import (
    PolarLogitTransform,
    canonical_fixed_target,
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
