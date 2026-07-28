"""Deterministic target representation for polar rectified flow.

This module deliberately contains no G1D proposal, selector, anchor, or
checkpoint dependency.  It defines the fixed source measure, an invertible
physical-polar transform, and a per-frame one-to-one transport contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
import torch.nn as nn


P_R_F_POINT_COUNT = 10_000
P_R_F_SOURCE_SEED = 20260716


def array_sha256(values: np.ndarray | torch.Tensor) -> str:
    """Hash numeric values with an explicit dtype/shape header."""

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.ascontiguousarray(values)
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fixed_scrambled_sobol_unit(
    point_count: int = P_R_F_POINT_COUNT,
    *,
    seed: int = P_R_F_SOURCE_SEED,
    epsilon: float = 1e-5,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the Cube-independent source measure in the open unit cube."""

    if point_count <= 0:
        raise ValueError("Sobol point count must be positive")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("Sobol epsilon must lie in (0, 0.5)")
    source = torch.quasirandom.SobolEngine(
        dimension=3,
        scramble=True,
        seed=seed,
    ).draw(point_count, dtype=dtype)
    return source.mul(1.0 - 2.0 * epsilon).add(epsilon)


def logit_unit(values: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    if not 0.0 < epsilon < 0.5:
        raise ValueError("Logit epsilon must lie in (0, 0.5)")
    values = values.clamp(epsilon, 1.0 - epsilon)
    return torch.log(values) - torch.log1p(-values)


def fit_empirical_range_cdf(
    sample_ranges_m: np.ndarray | torch.Tensor,
    *,
    lower_m: float,
    upper_m: float,
    knot_count: int = 257,
    uniform_mixture: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a strictly invertible empirical range CDF.

    A small uniform component prevents empty physical intervals from producing
    flat CDF segments.  Only training-set ranges may be supplied by callers.
    """

    if not math.isfinite(lower_m) or not math.isfinite(upper_m):
        raise ValueError("Range bounds must be finite")
    if upper_m <= lower_m:
        raise ValueError("Range upper bound must exceed lower bound")
    if knot_count < 3:
        raise ValueError("Range CDF requires at least three knots")
    if not 0.0 < uniform_mixture < 1.0:
        raise ValueError("Uniform mixture must lie in (0, 1)")
    if isinstance(sample_ranges_m, torch.Tensor):
        samples = sample_ranges_m.detach().cpu().numpy()
    else:
        samples = np.asarray(sample_ranges_m)
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    samples = samples[(samples >= lower_m) & (samples <= upper_m)]
    if samples.size == 0:
        raise ValueError("Cannot fit range CDF without in-FOV training samples")
    samples.sort()
    knots = np.linspace(lower_m, upper_m, knot_count, dtype=np.float64)
    empirical = np.searchsorted(samples, knots, side="right") / samples.size
    uniform = (knots - lower_m) / (upper_m - lower_m)
    cdf = (1.0 - uniform_mixture) * empirical + uniform_mixture * uniform
    cdf[0] = 0.0
    cdf[-1] = 1.0
    if not np.all(np.diff(cdf) > 0.0):
        raise AssertionError("Uniform-mixture range CDF is not strictly increasing")
    return knots, cdf


class PolarLogitTransform(nn.Module):
    """Invertible XYZ <-> empirical-polar-logit state transform."""

    def __init__(
        self,
        range_knots_m: np.ndarray | torch.Tensor,
        range_cdf: np.ndarray | torch.Tensor,
        *,
        azimuth_bounds_rad: tuple[float, float],
        elevation_bounds_rad: tuple[float, float],
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        knots = torch.as_tensor(range_knots_m, dtype=torch.float32)
        cdf = torch.as_tensor(range_cdf, dtype=torch.float32)
        if knots.ndim != 1 or cdf.shape != knots.shape or knots.numel() < 3:
            raise ValueError("Range knots and CDF must be aligned one-dimensional arrays")
        if not torch.isfinite(knots).all() or not torch.isfinite(cdf).all():
            raise ValueError("Range transform values must be finite")
        if not torch.all(knots[1:] > knots[:-1]):
            raise ValueError("Range knots must be strictly increasing")
        if not torch.all(cdf[1:] > cdf[:-1]):
            raise ValueError("Range CDF must be strictly increasing")
        if not torch.isclose(cdf[0], cdf.new_tensor(0.0)):
            raise ValueError("Range CDF must start at zero")
        if not torch.isclose(cdf[-1], cdf.new_tensor(1.0)):
            raise ValueError("Range CDF must end at one")
        azimuth_lower, azimuth_upper = map(float, azimuth_bounds_rad)
        elevation_lower, elevation_upper = map(float, elevation_bounds_rad)
        if not azimuth_upper > azimuth_lower:
            raise ValueError("Azimuth bounds must be strictly ordered")
        if not elevation_upper > elevation_lower:
            raise ValueError("Elevation bounds must be strictly ordered")
        if not 0.0 < epsilon < 0.5:
            raise ValueError("Transform epsilon must lie in (0, 0.5)")
        self.register_buffer("range_knots_m", knots, persistent=True)
        self.register_buffer("range_cdf", cdf, persistent=True)
        self.register_buffer(
            "angular_bounds_rad",
            torch.tensor(
                [
                    [azimuth_lower, azimuth_upper],
                    [elevation_lower, elevation_upper],
                ],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.epsilon = float(epsilon)

    @staticmethod
    def _piecewise_linear(
        values: torch.Tensor,
        input_knots: torch.Tensor,
        output_knots: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = values.dtype
        input_knots = input_knots.to(device=values.device, dtype=values.dtype)
        output_knots = output_knots.to(device=values.device, dtype=values.dtype)
        bounded = values.clamp(input_knots[0], input_knots[-1]).contiguous()
        right = torch.searchsorted(input_knots, bounded, right=True)
        right = right.clamp(1, input_knots.numel() - 1)
        left = right - 1
        fraction = (bounded - input_knots[left]) / (
            input_knots[right] - input_knots[left]
        ).clamp_min(torch.finfo(values.dtype).eps)
        result = output_knots[left] + fraction * (
            output_knots[right] - output_knots[left]
        )
        return result.to(original_dtype)

    def physical_to_unit(self, coordinates_rae: torch.Tensor) -> torch.Tensor:
        if coordinates_rae.shape[-1] != 3:
            raise ValueError("Polar coordinates must end in (range, azimuth, elevation)")
        if not torch.isfinite(coordinates_rae).all():
            raise ValueError("Polar coordinates must be finite")
        bounds = self.angular_bounds_rad.to(coordinates_rae)
        unit_range = self._piecewise_linear(
            coordinates_rae[..., 0],
            self.range_knots_m,
            self.range_cdf,
        )
        unit_azimuth = (
            coordinates_rae[..., 1] - bounds[0, 0]
        ) / (bounds[0, 1] - bounds[0, 0])
        unit_elevation = (
            coordinates_rae[..., 2] - bounds[1, 0]
        ) / (bounds[1, 1] - bounds[1, 0])
        return torch.stack(
            (unit_range, unit_azimuth, unit_elevation),
            dim=-1,
        ).clamp(self.epsilon, 1.0 - self.epsilon)

    def unit_to_physical(self, unit: torch.Tensor) -> torch.Tensor:
        if unit.shape[-1] != 3:
            raise ValueError("Unit polar coordinates must end in three values")
        if not torch.isfinite(unit).all():
            raise ValueError("Unit polar coordinates must be finite")
        bounded = unit.clamp(self.epsilon, 1.0 - self.epsilon)
        bounds = self.angular_bounds_rad.to(unit)
        range_m = self._piecewise_linear(
            bounded[..., 0],
            self.range_cdf,
            self.range_knots_m,
        )
        azimuth = bounds[0, 0] + bounded[..., 1] * (
            bounds[0, 1] - bounds[0, 0]
        )
        elevation = bounds[1, 0] + bounded[..., 2] * (
            bounds[1, 1] - bounds[1, 0]
        )
        return torch.stack((range_m, azimuth, elevation), dim=-1)

    def encode_rae(self, coordinates_rae: torch.Tensor) -> torch.Tensor:
        return logit_unit(self.physical_to_unit(coordinates_rae), self.epsilon)

    def decode_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != 3 or not torch.isfinite(state).all():
            raise ValueError("Polar flow state must be a finite (...,3) tensor")
        return self.unit_to_physical(torch.sigmoid(state))

    def encode_xyz(self, xyz_m: torch.Tensor) -> torch.Tensor:
        if xyz_m.shape[-1] != 3 or not torch.isfinite(xyz_m).all():
            raise ValueError("XYZ points must be a finite (...,3) tensor")
        radius = torch.linalg.vector_norm(xyz_m, dim=-1)
        azimuth = torch.atan2(xyz_m[..., 1], xyz_m[..., 0])
        elevation = torch.asin(
            (xyz_m[..., 2] / radius.clamp_min(1e-8)).clamp(-1.0, 1.0)
        )
        return self.encode_rae(torch.stack((radius, azimuth, elevation), dim=-1))

    def decode_xyz(self, state: torch.Tensor) -> torch.Tensor:
        coordinates = self.decode_state(state)
        radius, azimuth, elevation = coordinates.unbind(dim=-1)
        cos_elevation = torch.cos(elevation)
        return torch.stack(
            (
                radius * cos_elevation * torch.cos(azimuth),
                radius * cos_elevation * torch.sin(azimuth),
                radius * torch.sin(elevation),
            ),
            dim=-1,
        )


def _quantize_for_morton(values: np.ndarray, bits: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Morton input must be a finite (N,3) array")
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    span = upper - lower
    normalized = np.divide(
        values - lower,
        span,
        out=np.zeros_like(values),
        where=span > 0.0,
    )
    maximum = (1 << bits) - 1
    return np.rint(normalized * maximum).astype(np.uint64)


def morton_codes(values: np.ndarray | torch.Tensor, bits: int = 20) -> np.ndarray:
    """Compute deterministic 3D Morton codes for stable spatial ordering."""

    if not 1 <= bits <= 21:
        raise ValueError("Morton bit count must lie in [1, 21]")
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    quantized = _quantize_for_morton(values, bits)
    codes = np.zeros(quantized.shape[0], dtype=np.uint64)
    for bit in range(bits):
        codes |= ((quantized[:, 0] >> bit) & 1) << (3 * bit)
        codes |= ((quantized[:, 1] >> bit) & 1) << (3 * bit + 1)
        codes |= ((quantized[:, 2] >> bit) & 1) << (3 * bit + 2)
    return codes


@dataclass(frozen=True)
class FixedTargetEndpoint:
    xyz_m: torch.Tensor
    confidence: torch.Tensor
    source_indices: torch.Tensor
    hashes: dict[str, str]


def canonical_fixed_target(
    target_xyz_confidence: np.ndarray | torch.Tensor,
    *,
    point_count: int = P_R_F_POINT_COUNT,
) -> FixedTargetEndpoint:
    """Select a deterministic unique endpoint without replacement.

    This is a preflight target adapter, not the final continuous target-lifting
    algorithm.  It refuses insufficient unique support rather than copying
    points.  Evenly sampling a stable Morton order avoids input-order leakage.
    """

    if isinstance(target_xyz_confidence, torch.Tensor):
        target = target_xyz_confidence.detach().cpu().numpy()
    else:
        target = np.asarray(target_xyz_confidence)
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2 or target.shape[1] != 4:
        raise ValueError("Target endpoint input must have shape (N,4)")
    if point_count <= 0:
        raise ValueError("Target point count must be positive")
    if not np.isfinite(target).all():
        raise ValueError("Target endpoint input must be finite")
    unique_xyz, first = np.unique(target[:, :3], axis=0, return_index=True)
    if unique_xyz.shape[0] < point_count:
        raise ValueError(
            f"Target has {unique_xyz.shape[0]} unique XYZ points; "
            f"{point_count} are required without replacement"
        )
    first = first.astype(np.int64, copy=False)
    codes = morton_codes(unique_xyz)
    order = np.lexsort((first, codes))
    positions = np.floor(
        (np.arange(point_count, dtype=np.float64) + 0.5)
        * order.size
        / point_count
    ).astype(np.int64)
    selected_unique = order[positions]
    selected_original = first[selected_unique]
    xyz = np.ascontiguousarray(target[selected_original, :3])
    confidence = np.ascontiguousarray(target[selected_original, 3])
    if np.unique(xyz, axis=0).shape[0] != point_count:
        raise AssertionError("Canonical target selection introduced duplicates")
    indices = np.ascontiguousarray(selected_original)
    return FixedTargetEndpoint(
        xyz_m=torch.from_numpy(xyz),
        confidence=torch.from_numpy(confidence),
        source_indices=torch.from_numpy(indices),
        hashes={
            "xyz_sha256": array_sha256(xyz),
            "confidence_sha256": array_sha256(confidence),
            "source_indices_sha256": array_sha256(indices),
        },
    )


@dataclass(frozen=True)
class HierarchicalTransport:
    """Source-index to target-index one-to-one assignment."""

    source_to_target: torch.Tensor
    total_squared_cost: float
    block_size: int
    hashes: dict[str, str]

    def validate(self, point_count: int) -> None:
        permutation = self.source_to_target
        if permutation.shape != (point_count,) or permutation.dtype != torch.long:
            raise ValueError("Transport permutation has the wrong shape or dtype")
        expected = torch.arange(point_count, dtype=torch.long)
        if not torch.equal(torch.sort(permutation.cpu()).values, expected):
            raise ValueError("Transport is not a one-to-one target permutation")
        if not math.isfinite(self.total_squared_cost):
            raise ValueError("Transport cost must be finite")


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError(
            "P-RF hierarchical transport requires scipy.optimize."
            "linear_sum_assignment; install scipy>=1.10"
        ) from error
    rows, columns = linear_sum_assignment(cost)
    return rows.astype(np.int64), columns.astype(np.int64)


def _squared_cost(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = first[:, None, :] - second[None, :, :]
    cost = np.einsum("...d,...d->...", difference, difference)
    if not np.isfinite(cost).all():
        raise ValueError("Transport cost matrix contains non-finite values")
    return cost


def hierarchical_one_to_one_transport(
    source_state: np.ndarray | torch.Tensor,
    target_state: np.ndarray | torch.Tensor,
    *,
    block_size: int = 40,
) -> HierarchicalTransport:
    """Build a deterministic two-level one-to-one per-frame assignment."""

    if isinstance(source_state, torch.Tensor):
        source = source_state.detach().cpu().numpy()
    else:
        source = np.asarray(source_state)
    if isinstance(target_state, torch.Tensor):
        target = target_state.detach().cpu().numpy()
    else:
        target = np.asarray(target_state)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Source transport state must have shape (N,3)")
    if target.shape != source.shape:
        raise ValueError("Source and target transport states must have equal shape")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Transport states must be finite")
    point_count = source.shape[0]
    if point_count <= 0:
        raise ValueError("Transport point count must be positive")
    if block_size <= 0 or point_count % block_size:
        raise ValueError("Transport block size must divide the point count")
    source_order = np.lexsort(
        (np.arange(point_count, dtype=np.int64), morton_codes(source))
    )
    target_order = np.lexsort(
        (np.arange(point_count, dtype=np.int64), morton_codes(target))
    )
    block_count = point_count // block_size
    source_blocks = source_order.reshape(block_count, block_size)
    target_blocks = target_order.reshape(block_count, block_size)
    source_centers = source[source_blocks].mean(axis=1)
    target_centers = target[target_blocks].mean(axis=1)
    block_rows, block_columns = _linear_sum_assignment(
        _squared_cost(source_centers, target_centers)
    )
    if not np.array_equal(block_rows, np.arange(block_count)):
        raise AssertionError("Block assignment did not cover every source block")
    permutation = np.full(point_count, -1, dtype=np.int64)
    total_cost = 0.0
    for source_block_index, target_block_index in zip(
        block_rows, block_columns, strict=True
    ):
        source_indices = source_blocks[source_block_index]
        target_indices = target_blocks[target_block_index]
        local_cost = _squared_cost(
            source[source_indices],
            target[target_indices],
        )
        local_rows, local_columns = _linear_sum_assignment(local_cost)
        permutation[source_indices[local_rows]] = target_indices[local_columns]
        total_cost += float(local_cost[local_rows, local_columns].sum())
    if not np.array_equal(np.sort(permutation), np.arange(point_count)):
        raise AssertionError("Hierarchical transport is not a target permutation")
    cost_value = np.asarray([total_cost], dtype="<f8")
    transport = HierarchicalTransport(
        source_to_target=torch.from_numpy(permutation),
        total_squared_cost=total_cost,
        block_size=block_size,
        hashes={
            "source_state_sha256": array_sha256(source.astype("<f8")),
            "target_state_sha256": array_sha256(target.astype("<f8")),
            "source_to_target_sha256": array_sha256(permutation),
            "total_squared_cost_sha256": array_sha256(cost_value),
        },
    )
    transport.validate(point_count)
    return transport
