"""Cube-only Cartesian voxel-slot model for the R-B2 Stage-0 pilot.

The representation is fixed: 0.40 m Cartesian voxels and four bounded slots
per voxel. Candidate voxels are derived from the current radar Cube only.
Ground truth is never accepted by the candidate builder or export path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn


PROTOCOL = "rb2_cube_only_voxel_slot_tiny_v1"
VOXEL_SIZE_M = 0.40
SLOTS_PER_VOXEL = 4
EXPORT_POINT_COUNT = 10_000
RANGE_STRATA_M = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
OUTPUT_POINT_QUOTAS = (8_000, 1_700, 300)
OUTPUT_VOXEL_QUOTAS = tuple(value // SLOTS_PER_VOXEL for value in OUTPUT_POINT_QUOTAS)
DEFAULT_CANDIDATE_VOXEL_QUOTAS = tuple(value * 8 for value in OUTPUT_VOXEL_QUOTAS)
MINIMUM_EXPORT_DISTANCE_M = 0.05

RANGE_MAX_M = 118.037109
AZIMUTH_MIN_RAD = math.radians(-53.0)
AZIMUTH_MAX_RAD = math.radians(53.0)
ELEVATION_MIN_RAD = math.radians(-18.0)
ELEVATION_MAX_RAD = math.radians(18.0)
LATERAL_MAX_M = RANGE_MAX_M * math.sin(AZIMUTH_MAX_RAD)
VERTICAL_MAX_M = RANGE_MAX_M * math.sin(ELEVATION_MAX_RAD)
LATTICE_ORIGIN_XYZ_M = (0.0, -LATERAL_MAX_M, -VERTICAL_MAX_M)
LATTICE_UPPER_XYZ_M = (RANGE_MAX_M, LATERAL_MAX_M, VERTICAL_MAX_M)
LATTICE_SHAPE_XYZ = tuple(
    int(math.ceil((upper - lower) / VOXEL_SIZE_M))
    for lower, upper in zip(
        LATTICE_ORIGIN_XYZ_M,
        LATTICE_UPPER_XYZ_M,
        strict=True,
    )
)

# Four XY quadrants plus bounded residuals leave at least 6 cm between any two
# slots in one cell or in face-adjacent cells. This is a representation
# constraint, not output jitter or a post-hoc separation operation.
SLOT_ANCHORS_XYZ_M = (
    (-0.10, -0.10, 0.0),
    (-0.10, 0.10, 0.0),
    (0.10, -0.10, 0.0),
    (0.10, 0.10, 0.0),
)
SLOT_RESIDUAL_LIMIT_XYZ_M = (0.07, 0.07, 0.17)
GUARANTEED_MINIMUM_DISTANCE_M = 0.06


class CandidateCapacityError(RuntimeError):
    """Raised when Cube-only support cannot fill a frozen range quota."""


@dataclass(frozen=True)
class VoxelSlotPrediction:
    candidate_indices_xyz: torch.Tensor
    candidate_centers_xyz_m: torch.Tensor
    occupancy_logits: torch.Tensor
    slot_confidence_logits: torch.Tensor
    local_offsets_xyz_m: torch.Tensor
    slot_xyz_m: torch.Tensor


@dataclass(frozen=True)
class CubeOnlyExport:
    xyz_m: torch.Tensor
    confidence: torch.Tensor
    selected_candidate_indices_xyz: torch.Tensor
    selected_candidate_positions: torch.Tensor
    report: dict[str, object]


def _origin(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(LATTICE_ORIGIN_XYZ_M, device=device, dtype=dtype)


def lattice_indices_to_centers(
    indices_xyz: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert immutable integer lattice identities to Cartesian centers."""

    if indices_xyz.ndim < 2 or indices_xyz.shape[-1] != 3:
        raise ValueError("R-B2 lattice indices must end in XYZ")
    if indices_xyz.dtype not in (torch.int32, torch.int64):
        raise ValueError("R-B2 lattice indices must be integer tensors")
    maximum = torch.tensor(
        LATTICE_SHAPE_XYZ,
        device=indices_xyz.device,
        dtype=indices_xyz.dtype,
    )
    if bool(((indices_xyz < 0) | (indices_xyz >= maximum)).any()):
        raise ValueError("R-B2 lattice index is outside the frozen support")
    return (
        _origin(device=indices_xyz.device, dtype=dtype)
        + (indices_xyz.to(dtype) + 0.5) * VOXEL_SIZE_M
    )


def cartesian_to_rae(xyz_m: torch.Tensor) -> torch.Tensor:
    if xyz_m.shape[-1] != 3:
        raise ValueError("R-B2 Cartesian coordinates must end in XYZ")
    radius = torch.linalg.vector_norm(xyz_m, dim=-1)
    azimuth = torch.atan2(xyz_m[..., 1], xyz_m[..., 0])
    horizontal = torch.linalg.vector_norm(xyz_m[..., :2], dim=-1)
    elevation = torch.atan2(xyz_m[..., 2], horizontal)
    return torch.stack((radius, azimuth, elevation), dim=-1)


def points_in_fov(xyz_m: torch.Tensor) -> torch.Tensor:
    rae = cartesian_to_rae(xyz_m)
    return (
        (rae[..., 0] >= 0.0)
        & (rae[..., 0] <= RANGE_MAX_M)
        & (rae[..., 1] >= AZIMUTH_MIN_RAD)
        & (rae[..., 1] <= AZIMUTH_MAX_RAD)
        & (rae[..., 2] >= ELEVATION_MIN_RAD)
        & (rae[..., 2] <= ELEVATION_MAX_RAD)
    )


def range_stratum_codes(xyz_m: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(xyz_m, dim=-1)
    codes = torch.full_like(radius, -1, dtype=torch.long)
    for code, (lower, upper) in enumerate(RANGE_STRATA_M):
        codes[(radius >= lower) & (radius < upper)] = code
    return codes


def _safe_center_mask(
    centers_xyz_m: torch.Tensor,
    stratum: int,
) -> torch.Tensor:
    """Require every possible bounded slot position to remain in one stratum/FOV."""

    limit = torch.tensor(
        (0.17, 0.17, 0.17),
        device=centers_xyz_m.device,
        dtype=centers_xyz_m.dtype,
    )
    corners = torch.tensor(
        [
            (sx, sy, sz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        device=centers_xyz_m.device,
        dtype=centers_xyz_m.dtype,
    )
    possible = centers_xyz_m[:, None, :] + corners[None, :, :] * limit
    lower, upper = RANGE_STRATA_M[stratum]
    radius = torch.linalg.vector_norm(possible, dim=-1)
    return (
        points_in_fov(possible).all(dim=1)
        & (radius >= lower).all(dim=1)
        & (radius < upper).all(dim=1)
    )


def _linear_ids(indices_xyz: torch.Tensor) -> torch.Tensor:
    _, ny, nz = LATTICE_SHAPE_XYZ
    return (indices_xyz[..., 0] * ny + indices_xyz[..., 1]) * nz + indices_xyz[..., 2]


def _indices_from_linear_ids(
    linear_ids: torch.Tensor,
) -> torch.Tensor:
    _, ny, nz = LATTICE_SHAPE_XYZ
    ix = torch.div(linear_ids, ny * nz, rounding_mode="floor")
    remainder = linear_ids - ix * ny * nz
    iy = torch.div(remainder, nz, rounding_mode="floor")
    iz = remainder - iy * nz
    return torch.stack((ix, iy, iz), dim=-1)


def _ordered_unique_indices(indices_xyz: torch.Tensor) -> torch.Tensor:
    """Keep the first Cube-ranked occurrence of each immutable voxel ID."""

    ordered: list[int] = []
    seen: set[int] = set()
    for value in _linear_ids(indices_xyz).detach().cpu().tolist():
        integer = int(value)
        if integer not in seen:
            seen.add(integer)
            ordered.append(integer)
    return _indices_from_linear_ids(
        torch.tensor(ordered, dtype=torch.long, device=indices_xyz.device)
    )


def _neighbor_offsets(maximum_radius: int) -> list[tuple[int, int, int]]:
    offsets = [
        (dx, dy, dz)
        for dx in range(-maximum_radius, maximum_radius + 1)
        for dy in range(-maximum_radius, maximum_radius + 1)
        for dz in range(-maximum_radius, maximum_radius + 1)
    ]
    return sorted(
        offsets,
        key=lambda value: (
            abs(value[0]) + abs(value[1]) + abs(value[2]),
            value[0],
            value[1],
            value[2],
        ),
    )


def _polar_xyz(
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> torch.Tensor:
    cosine = torch.cos(elevation_rad)
    return torch.stack(
        (
            range_m * cosine * torch.cos(azimuth_rad),
            range_m * cosine * torch.sin(azimuth_rad),
            range_m * torch.sin(elevation_rad),
        ),
        dim=-1,
    )


def _candidate_voxels_one_frame(
    spatial_score_rae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    candidate_voxel_quotas: Sequence[int],
    *,
    seed_multiplier: int,
    maximum_neighbor_radius: int,
) -> torch.Tensor:
    r_count, a_count, e_count = spatial_score_rae.shape
    selected_by_range: list[torch.Tensor] = []
    maximum = torch.tensor(
        LATTICE_SHAPE_XYZ,
        device=spatial_score_rae.device,
        dtype=torch.long,
    )
    origin = _origin(device=spatial_score_rae.device, dtype=torch.float32)
    offsets = _neighbor_offsets(maximum_neighbor_radius)

    for stratum, required in enumerate(candidate_voxel_quotas):
        if required <= 0:
            raise ValueError("R-B2 candidate quotas must be positive")
        lower, upper = RANGE_STRATA_M[stratum]
        eligible_range = (range_m >= lower) & (range_m < min(upper, RANGE_MAX_M))
        masked = spatial_score_rae.masked_fill(
            ~eligible_range[:, None, None],
            -torch.inf,
        )
        available = int(eligible_range.sum().item()) * a_count * e_count
        seed_count = min(available, max(required * seed_multiplier, required))
        if seed_count == 0:
            raise CandidateCapacityError(
                f"R-B2 Cube axes contain no cells in range stratum {stratum}"
            )
        flat = masked.reshape(-1)
        top = torch.topk(flat, k=seed_count, sorted=True).indices
        range_index = torch.div(top, a_count * e_count, rounding_mode="floor")
        remainder = top - range_index * a_count * e_count
        azimuth_index = torch.div(remainder, e_count, rounding_mode="floor")
        elevation_index = remainder - azimuth_index * e_count
        xyz = _polar_xyz(
            range_m[range_index],
            azimuth_rad[azimuth_index],
            elevation_rad[elevation_index],
        )
        indices = torch.floor((xyz - origin[None, :]) / VOXEL_SIZE_M).long()
        valid = ((indices >= 0) & (indices < maximum[None, :])).all(dim=1)
        indices = _ordered_unique_indices(indices[valid])
        centers = lattice_indices_to_centers(indices)
        indices = indices[_safe_center_mask(centers, stratum)]
        if indices.shape[0] > required:
            indices = indices[:required]

        accepted = indices.detach().cpu().tolist()
        accepted_ids = {
            int(value)
            for value in _linear_ids(indices).detach().cpu().tolist()
        }
        seed_rows = list(accepted)
        for seed in seed_rows:
            if len(accepted) >= required:
                break
            for delta in offsets:
                candidate = (
                    seed[0] + delta[0],
                    seed[1] + delta[1],
                    seed[2] + delta[2],
                )
                if any(
                    value < 0 or value >= LATTICE_SHAPE_XYZ[axis]
                    for axis, value in enumerate(candidate)
                ):
                    continue
                candidate_tensor = torch.tensor(
                    [candidate],
                    dtype=torch.long,
                    device=spatial_score_rae.device,
                )
                linear = int(_linear_ids(candidate_tensor)[0].item())
                if linear in accepted_ids:
                    continue
                center = lattice_indices_to_centers(candidate_tensor)
                if not bool(_safe_center_mask(center, stratum)[0]):
                    continue
                accepted_ids.add(linear)
                accepted.append(list(candidate))
                if len(accepted) >= required:
                    break
        if len(accepted) < required:
            raise CandidateCapacityError(
                "R-B2 Cube-only candidate support is insufficient for range "
                f"{stratum}: {len(accepted)} < {required}"
            )
        selected_by_range.append(
            torch.tensor(
                accepted[:required],
                dtype=torch.long,
                device=spatial_score_rae.device,
            )
        )

    return torch.cat(selected_by_range, dim=0)


@torch.no_grad()
def candidate_voxels_from_cube(
    cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    candidate_voxel_quotas: Sequence[int] = DEFAULT_CANDIDATE_VOXEL_QUOTAS,
    seed_multiplier: int = 8,
    maximum_neighbor_radius: int = 4,
) -> torch.Tensor:
    """Build a fixed-lattice candidate bank from the current Cube only."""

    if cube_drae.ndim != 5:
        raise ValueError("R-B2 Cube must have shape (B,D,R,A,E)")
    if cube_drae.shape[2:] != (
        range_m.numel(),
        azimuth_rad.numel(),
        elevation_rad.numel(),
    ):
        raise ValueError("R-B2 Cube and RAE axes have inconsistent shapes")
    if len(candidate_voxel_quotas) != len(RANGE_STRATA_M):
        raise ValueError("R-B2 requires one candidate quota per range stratum")
    if seed_multiplier <= 0 or maximum_neighbor_radius < 0:
        raise ValueError("R-B2 candidate expansion parameters are invalid")
    score = torch.log1p(cube_drae.float().clamp_min(0.0)).amax(dim=1)
    batches = [
        _candidate_voxels_one_frame(
            score[index],
            range_m.float(),
            azimuth_rad.float(),
            elevation_rad.float(),
            candidate_voxel_quotas,
            seed_multiplier=seed_multiplier,
            maximum_neighbor_radius=maximum_neighbor_radius,
        )
        for index in range(cube_drae.shape[0])
    ]
    return torch.stack(batches, dim=0)


def _nearest_axis_indices(axis: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if axis.ndim != 1 or axis.numel() < 2:
        raise ValueError("R-B2 axes must be one-dimensional with at least two bins")
    ascending = bool(torch.all(axis[1:] >= axis[:-1]))
    descending = bool(torch.all(axis[1:] <= axis[:-1]))
    if not ascending and not descending:
        raise ValueError("R-B2 axes must be monotonic")
    working_axis = axis if ascending else axis.flip(0)
    right = torch.searchsorted(working_axis, query.contiguous())
    right = right.clamp(0, working_axis.numel() - 1)
    left = (right - 1).clamp(0, working_axis.numel() - 1)
    choose_left = (query - working_axis[left]).abs() <= (
        working_axis[right] - query
    ).abs()
    result = torch.where(choose_left, left, right)
    return result if ascending else working_axis.numel() - 1 - result


def gather_candidate_spectra(
    cube_drae: torch.Tensor,
    centers_xyz_m: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> torch.Tensor:
    """Gather the current Cube spectrum nearest each candidate center."""

    if centers_xyz_m.ndim != 3 or centers_xyz_m.shape[0] != cube_drae.shape[0]:
        raise ValueError("R-B2 candidate centers must have shape (B,N,3)")
    rae = cartesian_to_rae(centers_xyz_m)
    range_index = _nearest_axis_indices(range_m, rae[..., 0])
    azimuth_index = _nearest_axis_indices(azimuth_rad, rae[..., 1])
    elevation_index = _nearest_axis_indices(elevation_rad, rae[..., 2])
    a_count = azimuth_rad.numel()
    e_count = elevation_rad.numel()
    linear = (range_index * a_count + azimuth_index) * e_count + elevation_index
    flattened = cube_drae.flatten(start_dim=2)
    gather_index = linear[:, None, :].expand(-1, cube_drae.shape[1], -1)
    return torch.gather(flattened, 2, gather_index).transpose(1, 2)


class RB2VoxelSlotNet(nn.Module):
    """Predict fixed-lattice occupancy, slot confidence, and local XYZ."""

    def __init__(
        self,
        doppler_bins: int = 64,
        hidden_dim: int = 192,
        depth: int = 4,
        log_center: float = 11.0,
        log_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if doppler_bins <= 0 or hidden_dim <= 0 or depth < 2:
            raise ValueError("R-B2 model dimensions are invalid")
        if not math.isfinite(log_center) or not math.isfinite(log_scale):
            raise ValueError("R-B2 normalization must be finite")
        if log_scale <= 0.0:
            raise ValueError("R-B2 normalization scale must be positive")
        self.doppler_bins = doppler_bins
        self.log_center = float(log_center)
        self.log_scale = float(log_scale)
        layers: list[nn.Module] = [
            nn.Linear(doppler_bins + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                (
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.occupancy_head = nn.Linear(hidden_dim, 1)
        self.slot_confidence_head = nn.Linear(hidden_dim, SLOTS_PER_VOXEL)
        self.offset_head = nn.Linear(hidden_dim, SLOTS_PER_VOXEL * 3)
        self.register_buffer(
            "slot_anchors_xyz_m",
            torch.tensor(SLOT_ANCHORS_XYZ_M, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "slot_residual_limit_xyz_m",
            torch.tensor(SLOT_RESIDUAL_LIMIT_XYZ_M, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        cube_drae: torch.Tensor,
        candidate_indices_xyz: torch.Tensor,
        range_m: torch.Tensor,
        azimuth_rad: torch.Tensor,
        elevation_rad: torch.Tensor,
    ) -> VoxelSlotPrediction:
        if cube_drae.ndim != 5 or cube_drae.shape[1] != self.doppler_bins:
            raise ValueError(
                "R-B2 expected Cube shape "
                f"(B,{self.doppler_bins},R,A,E), got {tuple(cube_drae.shape)}"
            )
        if candidate_indices_xyz.ndim != 3:
            raise ValueError("R-B2 candidate indices must have shape (B,N,3)")
        if candidate_indices_xyz.shape[0] != cube_drae.shape[0]:
            raise ValueError("R-B2 Cube and candidate batch sizes differ")
        centers = lattice_indices_to_centers(
            candidate_indices_xyz,
            dtype=cube_drae.dtype,
        )
        spectra = gather_candidate_spectra(
            cube_drae,
            centers,
            range_m,
            azimuth_rad,
            elevation_rad,
        )
        normalized_spectra = (
            (torch.log10(spectra.clamp_min(0.0) + 1.0) - self.log_center)
            / self.log_scale
        ).clamp(-4.0, 4.0)
        rae = cartesian_to_rae(centers)
        coordinates = torch.cat(
            (
                centers / RANGE_MAX_M,
                rae[..., 0:1] / RANGE_MAX_M,
                rae[..., 1:2] / AZIMUTH_MAX_RAD,
                rae[..., 2:3] / ELEVATION_MAX_RAD,
            ),
            dim=-1,
        )
        features = self.backbone(torch.cat((normalized_spectra, coordinates), dim=-1))
        occupancy = self.occupancy_head(features).squeeze(-1)
        slot_confidence = self.slot_confidence_head(features)
        residual = torch.tanh(
            self.offset_head(features).view(
                cube_drae.shape[0],
                candidate_indices_xyz.shape[1],
                SLOTS_PER_VOXEL,
                3,
            )
        ) * self.slot_residual_limit_xyz_m
        local_offsets = self.slot_anchors_xyz_m + residual
        slot_xyz = centers[:, :, None, :] + local_offsets
        return VoxelSlotPrediction(
            candidate_indices_xyz=candidate_indices_xyz,
            candidate_centers_xyz_m=centers,
            occupancy_logits=occupancy,
            slot_confidence_logits=slot_confidence,
            local_offsets_xyz_m=local_offsets,
            slot_xyz_m=slot_xyz,
        )


def export_exact_voxel_slots(
    prediction: VoxelSlotPrediction,
    *,
    point_quotas: Sequence[int] = OUTPUT_POINT_QUOTAS,
) -> list[CubeOnlyExport]:
    """Rank whole voxels and export all four learned slots without repair."""

    if len(point_quotas) != len(RANGE_STRATA_M):
        raise ValueError("R-B2 export requires three range quotas")
    if any(value <= 0 or value % SLOTS_PER_VOXEL for value in point_quotas):
        raise ValueError("R-B2 point quotas must be positive multiples of four")
    voxel_quotas = tuple(value // SLOTS_PER_VOXEL for value in point_quotas)
    outputs: list[CubeOnlyExport] = []
    for batch in range(prediction.slot_xyz_m.shape[0]):
        centers = prediction.candidate_centers_xyz_m[batch]
        strata = range_stratum_codes(centers)
        score = (
            prediction.occupancy_logits[batch]
            + prediction.slot_confidence_logits[batch].mean(dim=-1)
        )
        selected_parts: list[torch.Tensor] = []
        for stratum, required in enumerate(voxel_quotas):
            available = torch.nonzero(strata == stratum, as_tuple=False).flatten()
            if available.numel() < required:
                raise CandidateCapacityError(
                    "R-B2 candidate capacity is below the frozen export quota "
                    f"for range {stratum}: {available.numel()} < {required}"
                )
            order = torch.argsort(
                score[available],
                descending=True,
                stable=True,
            )
            selected_parts.append(available[order[:required]])
        selected = torch.cat(selected_parts)
        selected_indices = prediction.candidate_indices_xyz[batch, selected]
        selected_ids = _linear_ids(selected_indices)
        if torch.unique(selected).numel() != selected.numel():
            raise RuntimeError("R-B2 export selected a candidate position twice")
        if torch.unique(selected_ids).numel() != selected_ids.numel():
            raise RuntimeError("R-B2 export selected a duplicate voxel identity")
        xyz = prediction.slot_xyz_m[batch, selected].reshape(-1, 3)
        confidence = torch.sigmoid(
            prediction.slot_confidence_logits[batch, selected]
        ).reshape(-1)
        expected = int(sum(point_quotas))
        if xyz.shape != (expected, 3):
            raise AssertionError("R-B2 export did not produce the frozen point count")
        exported_strata = range_stratum_codes(xyz)
        observed_quotas = tuple(
            int((exported_strata == code).sum().item())
            for code in range(len(RANGE_STRATA_M))
        )
        if observed_quotas != tuple(point_quotas):
            raise RuntimeError(
                "R-B2 bounded slots crossed a range boundary: "
                f"{observed_quotas} != {tuple(point_quotas)}"
            )
        if not bool(points_in_fov(xyz).all()):
            raise RuntimeError("R-B2 bounded slot left the frozen radar FOV")
        outputs.append(
            CubeOnlyExport(
                xyz_m=xyz,
                confidence=confidence,
                selected_candidate_indices_xyz=selected_indices,
                selected_candidate_positions=selected,
                report={
                    "exact_point_count": int(xyz.shape[0]),
                    "range_point_quotas": list(observed_quotas),
                    "slots_per_selected_voxel": SLOTS_PER_VOXEL,
                    "selected_voxel_count": int(selected.numel()),
                    "guaranteed_minimum_pair_distance_lower_bound_m": (
                        GUARANTEED_MINIMUM_DISTANCE_M
                    ),
                    "minimum_distance_requirement_m": MINIMUM_EXPORT_DISTANCE_M,
                    "copy_used": False,
                    "padding_used": False,
                    "jitter_used": False,
                    "duplicate_candidate_id_used": False,
                    "ground_truth_used_for_candidate_or_export": False,
                    "cube_only_inference": True,
                },
            )
        )
    return outputs
