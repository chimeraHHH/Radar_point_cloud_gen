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
    rae_indices: torch.Tensor | None = None
    is_lifted: torch.Tensor | None = None
    tangent_source_indices: torch.Tensor | None = None
    tangent_offset_uv_m: torch.Tensor | None = None
    lifting: dict[str, object] | None = None


def canonical_fixed_target(
    target_xyz_confidence: np.ndarray | torch.Tensor,
    *,
    point_count: int = P_R_F_POINT_COUNT,
) -> FixedTargetEndpoint:
    """Select a deterministic unique endpoint without replacement.

    This compatibility path refuses insufficient unique support rather than
    copying points. Evenly sampling a stable Morton order avoids input-order
    leakage. ``continuous_fixed_target`` calls it unchanged whenever enough
    original unique points are available.
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


def _strict_axis(values: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < 2 or not np.isfinite(axis).all():
        raise ValueError(f"{name} must contain at least two finite values")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _nearest_axis_indices(axis: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(axis, query, side="left")
    right = np.clip(right, 0, axis.size - 1)
    left = np.clip(right - 1, 0, axis.size - 1)
    use_left = np.abs(query - axis[left]) <= np.abs(axis[right] - query)
    return np.where(use_left, left, right).astype(np.int64)


def _xyz_to_rae_indices(
    xyz_m: np.ndarray,
    *,
    range_axis_m: np.ndarray,
    azimuth_axis_rad: np.ndarray,
    elevation_axis_rad: np.ndarray,
) -> np.ndarray:
    xyz = np.asarray(xyz_m, dtype=np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    elevation = np.arcsin(
        np.divide(
            xyz[:, 2],
            radius,
            out=np.zeros_like(radius),
            where=radius > 0.0,
        ).clip(-1.0, 1.0)
    )
    return np.column_stack(
        (
            _nearest_axis_indices(range_axis_m, radius),
            _nearest_axis_indices(azimuth_axis_rad, azimuth),
            _nearest_axis_indices(elevation_axis_rad, elevation),
        )
    )


def _orient_tangent_basis(basis: np.ndarray) -> np.ndarray:
    oriented = np.asarray(basis, dtype=np.float64).copy()
    for dimension in range(oriented.shape[1]):
        pivot = int(np.argmax(np.abs(oriented[:, dimension])))
        if oriented[pivot, dimension] < 0.0:
            oriented[:, dimension] *= -1.0
    return oriented


def continuous_fixed_target(
    target_xyz_confidence: np.ndarray | torch.Tensor,
    target_rae_index: np.ndarray | torch.Tensor,
    *,
    range_axis_m: np.ndarray | torch.Tensor,
    azimuth_axis_rad: np.ndarray | torch.Tensor,
    elevation_axis_rad: np.ndarray | torch.Tensor,
    point_count: int = P_R_F_POINT_COUNT,
    minimum_spacing_m: float = 0.05,
    tangent_neighbor_count: int = 12,
    minimum_tangent_support: int = 6,
    tangent_neighbor_radius_m: float = 1.5,
    maximum_patch_radius_m: float = 0.6,
    patch_neighbor_fraction: float = 0.6,
    maximum_planarity_ratio: float = 0.25,
) -> FixedTargetEndpoint:
    """Lift sparse observed target support to a fixed-cardinality endpoint.

    The sparse path never opens an unobserved RAE cell. Each synthetic point is
    a bounded offset on a locally fitted tangent plane, anchored to an observed
    target point, and receives its Doppler label from the current-frame Cube at
    the endpoint RAE index. A global Euclidean spacing gate certifies the 5 cm
    capacity. If these constraints cannot supply ``point_count`` points, the
    function fails instead of copying points or widening support.

    Frames with at least ``point_count`` exact unique XYZ samples use
    ``canonical_fixed_target`` directly, preserving the legacy endpoint bytes.
    """

    if isinstance(target_xyz_confidence, torch.Tensor):
        target_xyz_confidence = target_xyz_confidence.detach().cpu().numpy()
    if isinstance(target_rae_index, torch.Tensor):
        target_rae_index = target_rae_index.detach().cpu().numpy()
    target = np.asarray(target_xyz_confidence, dtype=np.float32)
    target_indices = np.asarray(target_rae_index, dtype=np.int64)
    if target.ndim != 2 or target.shape[1] != 4 or target.shape[0] == 0:
        raise ValueError("Continuous target input must have non-empty shape (N,4)")
    if target_indices.shape != (target.shape[0], 3):
        raise ValueError("Continuous target RAE indices must have shape (N,3)")
    if not np.isfinite(target).all():
        raise ValueError("Continuous target values must be finite")
    if point_count <= 0:
        raise ValueError("Continuous target point count must be positive")
    if not math.isfinite(minimum_spacing_m) or minimum_spacing_m <= 0.0:
        raise ValueError("Continuous target spacing must be finite and positive")
    if tangent_neighbor_count < minimum_tangent_support:
        raise ValueError("Tangent neighbor count is smaller than minimum support")
    if minimum_tangent_support < 3:
        raise ValueError("Tangent fitting requires at least three support points")
    positive_parameters = {
        "tangent_neighbor_radius_m": tangent_neighbor_radius_m,
        "maximum_patch_radius_m": maximum_patch_radius_m,
        "patch_neighbor_fraction": patch_neighbor_fraction,
        "maximum_planarity_ratio": maximum_planarity_ratio,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_parameters.values()):
        raise ValueError("Continuous target lifting parameters must be finite and positive")

    range_axis = _strict_axis(range_axis_m, "Range axis")
    azimuth_axis = _strict_axis(azimuth_axis_rad, "Azimuth axis")
    elevation_axis = _strict_axis(elevation_axis_rad, "Elevation axis")
    recomputed_indices = _xyz_to_rae_indices(
        target[:, :3],
        range_axis_m=range_axis,
        azimuth_axis_rad=azimuth_axis,
        elevation_axis_rad=elevation_axis,
    )
    if not np.array_equal(recomputed_indices, target_indices):
        mismatch_count = int(np.any(recomputed_indices != target_indices, axis=1).sum())
        raise ValueError(
            f"Continuous target has {mismatch_count} XYZ/RAE provenance mismatches"
        )

    unique_xyz, first = np.unique(target[:, :3], axis=0, return_index=True)
    if unique_xyz.shape[0] >= point_count:
        return canonical_fixed_target(target, point_count=point_count)

    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError(
            "Continuous target lifting requires scipy.spatial.cKDTree"
        ) from error

    first = first.astype(np.int64, copy=False)
    unique_xyz = unique_xyz.astype(np.float64, copy=False)
    unique_confidence = target[first, 3].astype(np.float32, copy=False)
    unique_rae = target_indices[first]
    occupied_cells = {tuple(index) for index in unique_rae.tolist()}
    point_tree = cKDTree(unique_xyz)
    neighbor_count = min(tangent_neighbor_count, unique_xyz.shape[0])
    neighbor_distances, neighbor_indices = point_tree.query(
        unique_xyz,
        k=neighbor_count,
    )
    neighbor_distances = np.atleast_2d(neighbor_distances)
    neighbor_indices = np.atleast_2d(neighbor_indices)
    if unique_xyz.shape[0] == 1:
        neighbor_distances = neighbor_distances.reshape(1, 1)
        neighbor_indices = neighbor_indices.reshape(1, 1)

    selected_xyz: list[np.ndarray] = []
    selected_confidence: list[float] = []
    selected_parent: list[int] = []
    selected_rae: list[np.ndarray] = []
    selected_lifted: list[bool] = []
    selected_support: list[np.ndarray] = []
    selected_offset: list[np.ndarray] = []
    spatial_buckets: dict[tuple[int, int, int], list[int]] = {}
    accepted_spacing_m = minimum_spacing_m + 1e-6
    accepted_spacing_squared = accepted_spacing_m**2
    attempted_candidates = 0
    rejected_spacing = 0
    rejected_unoccupied_cell = 0

    def accept_candidate(
        xyz: np.ndarray,
        *,
        parent_unique_index: int,
        rae_index: np.ndarray,
        lifted: bool,
        support_unique_indices: np.ndarray,
        tangent_offset_uv_m: np.ndarray,
    ) -> bool:
        nonlocal rejected_spacing
        candidate = np.asarray(xyz, dtype=np.float32).astype(np.float64)
        bucket = tuple(np.floor(candidate / accepted_spacing_m).astype(np.int64))
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    neighbor_bucket = (
                        bucket[0] + delta_x,
                        bucket[1] + delta_y,
                        bucket[2] + delta_z,
                    )
                    for selected_index in spatial_buckets.get(neighbor_bucket, ()):
                        difference = candidate - selected_xyz[selected_index]
                        if float(difference @ difference) < accepted_spacing_squared:
                            rejected_spacing += 1
                            return False
        support = np.full(tangent_neighbor_count, -1, dtype=np.int64)
        source_support = first[support_unique_indices]
        support[: source_support.size] = source_support
        selected_index = len(selected_xyz)
        selected_xyz.append(candidate)
        selected_confidence.append(float(unique_confidence[parent_unique_index]))
        selected_parent.append(int(first[parent_unique_index]))
        selected_rae.append(np.asarray(rae_index, dtype=np.int64))
        selected_lifted.append(bool(lifted))
        selected_support.append(support)
        selected_offset.append(np.asarray(tangent_offset_uv_m, dtype=np.float32))
        spatial_buckets.setdefault(bucket, []).append(selected_index)
        return True

    parent_codes = morton_codes(unique_xyz)
    parent_order = np.lexsort(
        (
            unique_xyz[:, 2],
            unique_xyz[:, 1],
            unique_xyz[:, 0],
            parent_codes,
        )
    )
    for parent in parent_order:
        accept_candidate(
            unique_xyz[parent],
            parent_unique_index=int(parent),
            rae_index=unique_rae[parent],
            lifted=False,
            support_unique_indices=np.asarray([parent], dtype=np.int64),
            tangent_offset_uv_m=np.zeros(2, dtype=np.float32),
        )

    observed_selected_count = len(selected_xyz)
    tangent_eligible_parent_count = 0
    candidate_pitch_m = minimum_spacing_m / 2.0
    for parent in parent_order:
        distances = neighbor_distances[parent]
        support_mask = distances <= tangent_neighbor_radius_m + 1e-12
        support_indices = neighbor_indices[parent][support_mask].astype(
            np.int64,
            copy=False,
        )
        support_distances = distances[support_mask]
        if support_indices.size < minimum_tangent_support:
            continue
        centered = unique_xyz[support_indices] - unique_xyz[parent]
        scale = max(float(support_distances[-1]), np.finfo(np.float64).eps)
        weights = np.exp(-0.5 * (support_distances / scale) ** 2)
        covariance = (
            (centered * weights[:, None]).T @ centered
        ) / max(float(weights.sum()), np.finfo(np.float64).eps)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if eigenvalues[1] <= 1e-8:
            continue
        planarity_ratio = float(eigenvalues[2] / eigenvalues[1])
        if planarity_ratio > maximum_planarity_ratio:
            continue
        tangent_basis = _orient_tangent_basis(eigenvectors[:, :2])
        patch_radius = min(
            maximum_patch_radius_m,
            patch_neighbor_fraction * float(support_distances[-1]),
        )
        if patch_radius < candidate_pitch_m:
            continue
        tangent_eligible_parent_count += 1
        lattice_radius = int(math.floor(patch_radius / candidate_pitch_m))
        offsets = []
        for first_offset in range(-lattice_radius, lattice_radius + 1):
            for second_offset in range(-lattice_radius, lattice_radius + 1):
                if first_offset == 0 and second_offset == 0:
                    continue
                squared_index_radius = first_offset**2 + second_offset**2
                if (
                    squared_index_radius * candidate_pitch_m**2
                    <= patch_radius**2 + 1e-12
                ):
                    offsets.append(
                        (squared_index_radius, first_offset, second_offset)
                    )
        offsets.sort()
        for _, first_offset, second_offset in offsets:
            attempted_candidates += 1
            offset_uv = candidate_pitch_m * np.asarray(
                [first_offset, second_offset],
                dtype=np.float64,
            )
            candidate = (
                unique_xyz[parent]
                + tangent_basis[:, 0] * offset_uv[0]
                + tangent_basis[:, 1] * offset_uv[1]
            )
            candidate = np.asarray(candidate, dtype=np.float32).astype(np.float64)
            candidate_rae = _xyz_to_rae_indices(
                candidate.reshape(1, 3),
                range_axis_m=range_axis,
                azimuth_axis_rad=azimuth_axis,
                elevation_axis_rad=elevation_axis,
            )[0]
            if tuple(candidate_rae) not in occupied_cells:
                rejected_unoccupied_cell += 1
                continue
            if accept_candidate(
                candidate,
                parent_unique_index=int(parent),
                rae_index=candidate_rae,
                lifted=True,
                support_unique_indices=support_indices,
                tangent_offset_uv_m=offset_uv,
            ) and len(selected_xyz) == point_count:
                break
        if len(selected_xyz) == point_count:
            break

    if len(selected_xyz) < point_count:
        raise ValueError(
            "Continuous target certified 5 cm capacity "
            f"{len(selected_xyz)} is below required {point_count}; "
            "refusing to widen observed RAE support"
        )

    xyz = np.ascontiguousarray(
        np.asarray(selected_xyz[:point_count], dtype=np.float32)
    )
    confidence = np.ascontiguousarray(
        np.asarray(selected_confidence[:point_count], dtype=np.float32)
    )
    source_indices = np.ascontiguousarray(
        np.asarray(selected_parent[:point_count], dtype=np.int64)
    )
    rae_indices = np.ascontiguousarray(
        np.asarray(selected_rae[:point_count], dtype=np.int64)
    )
    lifted_mask = np.ascontiguousarray(
        np.asarray(selected_lifted[:point_count], dtype=np.bool_)
    )
    tangent_sources = np.ascontiguousarray(
        np.asarray(selected_support[:point_count], dtype=np.int64)
    )
    tangent_offsets = np.ascontiguousarray(
        np.asarray(selected_offset[:point_count], dtype=np.float32)
    )
    final_tree = cKDTree(xyz.astype(np.float64))
    nearest_distance = final_tree.query(xyz.astype(np.float64), k=2)[0][:, 1]
    minimum_pair_distance_m = float(nearest_distance.min())
    if minimum_pair_distance_m + 1e-9 < minimum_spacing_m:
        raise AssertionError("Continuous target violated its Euclidean spacing contract")
    if not all(tuple(index) in occupied_cells for index in rae_indices.tolist()):
        raise AssertionError("Continuous target opened an unobserved RAE cell")

    hashes = {
        "xyz_sha256": array_sha256(xyz),
        "confidence_sha256": array_sha256(confidence),
        "source_indices_sha256": array_sha256(source_indices),
        "doppler_rae_indices_sha256": array_sha256(rae_indices),
        "lifted_mask_sha256": array_sha256(lifted_mask),
        "tangent_source_indices_sha256": array_sha256(tangent_sources),
        "tangent_offset_uv_m_sha256": array_sha256(tangent_offsets),
    }
    return FixedTargetEndpoint(
        xyz_m=torch.from_numpy(xyz),
        confidence=torch.from_numpy(confidence),
        source_indices=torch.from_numpy(source_indices),
        hashes=hashes,
        rae_indices=torch.from_numpy(rae_indices),
        is_lifted=torch.from_numpy(lifted_mask),
        tangent_source_indices=torch.from_numpy(tangent_sources),
        tangent_offset_uv_m=torch.from_numpy(tangent_offsets),
        lifting={
            "mode": "local_tangent_occupied_rae_lifting",
            "raw_input_count": int(target.shape[0]),
            "raw_unique_count": int(unique_xyz.shape[0]),
            "observed_selected_count": observed_selected_count,
            "lifted_count": int(lifted_mask.sum()),
            "exact_point_count": point_count,
            "minimum_spacing_contract_m": minimum_spacing_m,
            "minimum_pair_distance_m": minimum_pair_distance_m,
            "occupied_rae_cell_count": len(occupied_cells),
            "used_rae_cell_count": len({tuple(index) for index in rae_indices.tolist()}),
            "tangent_eligible_parent_count": tangent_eligible_parent_count,
            "attempted_lifted_candidate_count": attempted_candidates,
            "rejected_spacing_candidate_count": rejected_spacing,
            "rejected_unoccupied_cell_candidate_count": rejected_unoccupied_cell,
            "tangent_neighbor_count": tangent_neighbor_count,
            "minimum_tangent_support": minimum_tangent_support,
            "tangent_neighbor_radius_m": tangent_neighbor_radius_m,
            "maximum_patch_radius_m": maximum_patch_radius_m,
            "patch_neighbor_fraction": patch_neighbor_fraction,
            "maximum_planarity_ratio": maximum_planarity_ratio,
            "candidate_pitch_m": candidate_pitch_m,
            "geometry_support_source": (
                "observed_target_local_weighted_pca_with_occupied_rae_cell_gate"
            ),
            "confidence_source": "anchored_observed_target_point",
            "doppler_label_source": (
                "current_frame_cube_spectrum_at_endpoint_rae_index"
            ),
            "capacity_semantics": (
                "certified_lower_bound_reached_exact_point_count"
            ),
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
