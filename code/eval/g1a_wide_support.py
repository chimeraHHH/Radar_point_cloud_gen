"""R-A0 RaLD-inspired initial-query-domain support diagnostic.

The wide domain is constructed from Cube-only evidence. Ground-truth geometry
is accepted only by ``select_wide_support_diagnostic`` for an explicitly
unattainable heuristic diagnostic. The selector is not a strict upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from models.cube_cycle import continuous_rae_to_xyz


RANDOM_QUERY_COUNT = 500_000
RADAR_QUERY_COUNT = 700_000
RAW_QUERY_COUNT = RANDOM_QUERY_COUNT + RADAR_QUERY_COUNT
EXPORT_COUNT = 10_000
LOW_THRESHOLD_QUANTILE = 0.75
CAPACITY_CELL_M = 0.05
RANGE_STRATA_M = (
    ("range_0_30m", 0.0, 30.0),
    ("range_30_60m", 30.0, 60.0),
    ("range_60_120m", 60.0, 120.0),
)
RANDOM_RANGE_QUOTAS = (166_667, 166_667, 166_666)
RADAR_RANGE_QUOTAS = (233_334, 233_333, 233_333)
SUPPORT_THRESHOLDS_M = (0.5, 1.0, 2.0)
TARGET_REASSIGNMENT_TOPK = 4
SOURCE_RANDOM_FRUSTUM = 0
SOURCE_RADAR_LOW_THRESHOLD = 1
SOURCE_RADAR_HELPER_AUGMENTED = 2
SOURCE_LABELS = {
    SOURCE_RANDOM_FRUSTUM: "random_frustum",
    SOURCE_RADAR_LOW_THRESHOLD: "radar_low_threshold",
    SOURCE_RADAR_HELPER_AUGMENTED: "radar_helper_augmented",
}
DIAGNOSTIC_ARTIFACT_LABEL = (
    "g1a_ra0_rald_inspired_initial_query_domain_heuristic_diagnostic"
)


@dataclass(frozen=True)
class WideQueryDomain:
    """One fixed-count raw union and its radar-preferred 5 cm cell domain."""

    coordinates_rae: torch.Tensor
    xyz_m: torch.Tensor
    candidate_ids: torch.Tensor
    source_codes: torch.Tensor
    range_stratum_codes: torch.Tensor
    report: dict


@dataclass(frozen=True)
class StreamingNearest:
    """Bidirectional nearest-neighbour results without a full distance matrix."""

    candidate_to_target_m: torch.Tensor
    target_to_candidate_m: torch.Tensor
    target_nearest_candidate_row: torch.Tensor
    target_topk_candidate_rows: torch.Tensor
    target_topk_distance_m: torch.Tensor


@dataclass(frozen=True)
class WideSupportDiagnosticSelection:
    """One exact-10k, GT-guided heuristic diagnostic export."""

    selected_pool_indices: torch.Tensor
    selected_candidate_ids: torch.Tensor
    selected_xyz_m: torch.Tensor
    selected_source_codes: torch.Tensor
    selected_range_stratum_codes: torch.Tensor
    overall_support: dict
    per_range_support: dict[str, dict]
    selection_report: dict
    artifact_label: dict[str, bool | str]


def diagnostic_artifact_label() -> dict[str, bool | str]:
    return {
        "label": DIAGNOSTIC_ARTIFACT_LABEL,
        "diagnostic": True,
        "unattainable": True,
        "strict_upper_bound": False,
        "selection_is_heuristic": True,
        "rald_scope": "inspired_initial_query_domain_only",
        "full_rald_wide_family_closure_eligible": False,
        "ground_truth_used_for_selection": True,
        "ground_truth_used_for_proposal_generation": False,
        "eligible_as_method_result": False,
    }


def _validate_axis(axis: torch.Tensor, name: str) -> None:
    if axis.ndim != 1 or axis.numel() < 2:
        raise ValueError(f"R-A0 {name} axis must contain at least two values")
    if not torch.is_floating_point(axis):
        raise TypeError(f"R-A0 {name} axis must be floating point")
    if not torch.isfinite(axis).all():
        raise ValueError(f"R-A0 {name} axis must contain only finite values")
    if not bool((axis[1:] > axis[:-1]).all()):
        raise ValueError(f"R-A0 {name} axis must be strictly increasing")


def _validate_axes(
    spatial_shape: tuple[int, int, int],
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
) -> None:
    for axis, name in (
        (range_m, "range"),
        (azimuth_rad, "azimuth"),
        (elevation_rad, "elevation"),
    ):
        _validate_axis(axis, name)
    if spatial_shape != (
        range_m.numel(),
        azimuth_rad.numel(),
        elevation_rad.numel(),
    ):
        raise ValueError("R-A0 Cube shape and RAE axes differ")
    if float(range_m[0].item()) >= 30.0 or float(range_m[-1].item()) < 60.0:
        raise ValueError("R-A0 range axis does not cover the frozen strata")


def _frame_seed(base_seed: int, sequence: int, radar_index: int, stream: int) -> int:
    modulus = 2**31 - 1
    return int(
        (
            int(base_seed) * 1_000_003
            + int(sequence) * 10_009
            + int(radar_index) * 101
            + int(stream) * 17
        )
        % modulus
    )


def _sobol(
    count: int,
    dimension: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0:
        return torch.empty((0, dimension), device=device, dtype=dtype)
    values = torch.quasirandom.SobolEngine(
        dimension=dimension,
        scramble=True,
        seed=seed,
    ).draw(count, dtype=torch.float32)
    return values.to(device=device, dtype=dtype, non_blocking=True)


def _physical_to_fractional_index(
    axis: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    right = torch.searchsorted(axis, values)
    right = right.clamp(1, axis.numel() - 1)
    left = right - 1
    lower = axis[left]
    upper = axis[right]
    fraction = (values - lower) / (upper - lower).clamp_min(
        torch.finfo(values.dtype).eps
    )
    return (left.to(values) + fraction).clamp(0.0, axis.numel() - 1)


def _stratum_range_indices(
    range_m: torch.Tensor,
    lower_m: float,
    upper_m: float,
) -> torch.Tensor:
    mask = (range_m >= lower_m) & (range_m < upper_m)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError(
            f"R-A0 range axis has no cells in [{lower_m},{upper_m}) m"
        )
    return indices


def stratified_random_queries(
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    quotas: tuple[int, int, int],
    base_seed: int,
    sequence: int,
    radar_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate GT-free low-discrepancy queries with exact range quotas."""

    if len(quotas) != len(RANGE_STRATA_M) or any(count <= 0 for count in quotas):
        raise ValueError("R-A0 random quotas must be three positive integers")
    device = range_m.device
    dtype = range_m.dtype
    coordinates = []
    stratum_codes = []
    for stratum, ((_, lower_m, upper_m), count) in enumerate(
        zip(RANGE_STRATA_M, quotas, strict=True)
    ):
        valid_range = _stratum_range_indices(range_m, lower_m, upper_m)
        physical_lower = max(lower_m, float(range_m[valid_range[0]].item()))
        physical_upper = min(
            upper_m,
            float(range_m[valid_range[-1]].item()),
        )
        if physical_upper <= physical_lower:
            raise ValueError("R-A0 random-query stratum has zero physical width")
        unit = _sobol(
            count,
            3,
            seed=_frame_seed(base_seed, sequence, radar_index, stratum),
            device=device,
            dtype=dtype,
        )
        physical_range = physical_lower + unit[:, 0] * (
            physical_upper - physical_lower
        )
        range_index = _physical_to_fractional_index(range_m, physical_range)
        azimuth_index = unit[:, 1] * (azimuth_rad.numel() - 1)
        elevation_index = unit[:, 2] * (elevation_rad.numel() - 1)
        coordinates.append(
            torch.stack((range_index, azimuth_index, elevation_index), dim=1)
        )
        stratum_codes.append(
            torch.full(
                (count,),
                stratum,
                device=device,
                dtype=torch.int8,
            )
        )
    return torch.cat(coordinates), torch.cat(stratum_codes)


def _flat_to_rae(
    flat_index: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    _, azimuth_count, elevation_count = spatial_shape
    radius = flat_index // (azimuth_count * elevation_count)
    remainder = flat_index % (azimuth_count * elevation_count)
    azimuth = remainder // elevation_count
    elevation = remainder % elevation_count
    return torch.stack((radius, azimuth, elevation), dim=1)


def low_threshold_radar_queries(
    cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    *,
    quotas: tuple[int, int, int],
    base_seed: int,
    sequence: int,
    radar_index: int,
    threshold_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Build per-stratum RaLD-inspired helper queries from Cube-only evidence.

    Each stratum independently takes all cells at or above a broad integrated
    log-energy quantile. If fewer helpers exist than the fixed quota, the
    remainder uses the RaLD helper-augmentation pattern: replacement sampling
    and a 1- or 2-cell bounded bias. This is initial-query augmentation, not
    RaLD's occupancy-dependent second decoding pass. No GT is used.
    """

    if cube_drae.ndim != 4 or cube_drae.shape[0] != 64:
        raise ValueError("R-A0 expects one Full-RAED Cube with shape (64,R,A,E)")
    if len(quotas) != len(RANGE_STRATA_M) or any(count <= 0 for count in quotas):
        raise ValueError("R-A0 radar quotas must be three positive integers")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("R-A0 low-threshold quantile must lie in (0,1)")
    spatial_shape = tuple(int(size) for size in cube_drae.shape[1:])
    energy = torch.log1p(cube_drae.float().clamp_min(0.0)).sum(dim=0)
    coordinates = []
    sources = []
    strata = []
    reports = []
    for stratum, ((label, lower_m, upper_m), quota) in enumerate(
        zip(RANGE_STRATA_M, quotas, strict=True)
    ):
        radius_indices = _stratum_range_indices(range_m, lower_m, upper_m)
        block = energy[radius_indices]
        threshold = torch.quantile(block.flatten(), threshold_quantile)
        eligible_local = torch.nonzero(
            block >= threshold,
            as_tuple=False,
        )
        if eligible_local.shape[0] == 0:
            raise RuntimeError(f"R-A0 low threshold produced no helpers in {label}")
        helper = eligible_local.to(cube_drae)
        helper[:, 0] = radius_indices[eligible_local[:, 0]].to(helper)
        eligible_count = int(helper.shape[0])
        if eligible_count >= quota:
            sample_index = (
                torch.arange(quota, device=helper.device, dtype=torch.long)
                * eligible_count
                // quota
            )
            selected = helper[sample_index]
            source = torch.full(
                (quota,),
                SOURCE_RADAR_LOW_THRESHOLD,
                device=helper.device,
                dtype=torch.int8,
            )
            augmentation_count = 0
        else:
            augmentation_count = quota - eligible_count
            unit = _sobol(
                augmentation_count,
                5,
                seed=_frame_seed(
                    base_seed,
                    sequence,
                    radar_index,
                    10 + stratum,
                ),
                device=helper.device,
                dtype=helper.dtype,
            )
            selected_helper = (
                (unit[:, 0] * eligible_count)
                .floor()
                .to(torch.long)
                .clamp_max(eligible_count - 1)
            )
            scale = (unit[:, 1] * 2.0).floor() + 1.0
            bias = (unit[:, 2:5] * 2.0 - 1.0) * scale[:, None]
            refined = helper[selected_helper] + bias
            minimum = refined.new_tensor(
                [float(radius_indices[0].item()), 0.0, 0.0]
            )
            maximum = refined.new_tensor(
                [
                    float(radius_indices[-1].item()),
                    float(spatial_shape[1] - 1),
                    float(spatial_shape[2] - 1),
                ]
            )
            refined = torch.minimum(torch.maximum(refined, minimum), maximum)
            selected = torch.cat((helper, refined))
            source = torch.cat(
                (
                    torch.full(
                        (eligible_count,),
                        SOURCE_RADAR_LOW_THRESHOLD,
                        device=helper.device,
                        dtype=torch.int8,
                    ),
                    torch.full(
                        (augmentation_count,),
                        SOURCE_RADAR_HELPER_AUGMENTED,
                        device=helper.device,
                        dtype=torch.int8,
                    ),
                )
            )
        if selected.shape != (quota, 3):
            raise AssertionError("R-A0 radar helper expansion changed its quota")
        coordinates.append(selected)
        sources.append(source)
        strata.append(
            torch.full(
                (quota,),
                stratum,
                device=helper.device,
                dtype=torch.int8,
            )
        )
        reports.append(
            {
                "label": label,
                "quota": quota,
                "threshold_quantile": threshold_quantile,
                "threshold_integrated_log_energy": float(threshold.item()),
                "eligible_low_threshold_cell_count": eligible_count,
                "retained_low_threshold_count": quota - augmentation_count,
                "helper_augmentation_count": augmentation_count,
                "occupancy_dependent_second_pass": False,
            }
        )
    return (
        torch.cat(coordinates),
        torch.cat(sources),
        torch.cat(strata),
        reports,
    )


def _rae_to_xyz_chunked(
    coordinates_rae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("R-A0 coordinate conversion chunk must be positive")
    parts = []
    for start in range(0, coordinates_rae.shape[0], chunk_size):
        parts.append(
            continuous_rae_to_xyz(
                coordinates_rae[start : start + chunk_size],
                range_m,
                azimuth_rad,
                elevation_rad,
            )
        )
    return torch.cat(parts)


def capacity_cell_keys(
    xyz_m: torch.Tensor,
    *,
    cell_m: float = CAPACITY_CELL_M,
) -> torch.Tensor:
    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
        raise ValueError("R-A0 capacity cells require XYZ shape (N,3)")
    if cell_m <= 0.0:
        raise ValueError("R-A0 capacity-cell width must be positive")
    cell = torch.floor(xyz_m / cell_m).to(torch.int64)
    offset = 4096
    base = 8192
    shifted = cell + offset
    if bool(((shifted < 0) | (shifted >= base)).any()):
        raise ValueError("R-A0 XYZ coordinate exceeds the frozen cell hash domain")
    return (
        shifted[:, 0] * base * base
        + shifted[:, 1] * base
        + shifted[:, 2]
    )


def stable_capacity_one_indices(
    xyz_m: torch.Tensor,
    *,
    source_codes: torch.Tensor | None = None,
    cell_m: float = CAPACITY_CELL_M,
) -> torch.Tensor:
    """Keep one representative per cell, preferring measured radar evidence."""

    keys = capacity_cell_keys(xyz_m, cell_m=cell_m)
    order = torch.arange(xyz_m.shape[0], device=xyz_m.device)
    if source_codes is not None:
        if source_codes.shape != (xyz_m.shape[0],):
            raise ValueError("R-A0 source labels must match candidate rows")
        priority = torch.full_like(source_codes, 2, dtype=torch.long)
        priority[source_codes == SOURCE_RADAR_HELPER_AUGMENTED] = 1
        priority[source_codes == SOURCE_RADAR_LOW_THRESHOLD] = 0
        order = order[torch.argsort(priority, stable=True)]
    order = order[torch.argsort(keys[order], stable=True)]
    sorted_keys = keys[order]
    first = torch.ones_like(sorted_keys, dtype=torch.bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return torch.sort(order[first]).values


def _source_counts(source_codes: torch.Tensor) -> dict[str, int]:
    return {
        label: int((source_codes == code).sum().item())
        for code, label in SOURCE_LABELS.items()
    }


def _range_counts(range_codes: torch.Tensor) -> dict[str, int]:
    return {
        label: int((range_codes == index).sum().item())
        for index, (label, _, _) in enumerate(RANGE_STRATA_M)
    }


def _range_codes_from_xyz(xyz_m: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(xyz_m, dim=1)
    codes = torch.full(
        (xyz_m.shape[0],),
        -1,
        device=xyz_m.device,
        dtype=torch.int8,
    )
    for index, (_, lower_m, upper_m) in enumerate(RANGE_STRATA_M):
        mask = (radius >= lower_m) & (radius < upper_m)
        codes[mask] = index
    if bool((codes < 0).any()):
        raise ValueError("R-A0 coordinates must lie in the frozen [0,120) m strata")
    return codes


def build_wide_query_domain(
    cube_drae: torch.Tensor,
    range_m: torch.Tensor,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    base_seed: int,
    sequence: int,
    radar_index: int,
) -> WideQueryDomain:
    """Construct the frozen 1.2M initial-query union and unique domain."""

    if cube_drae.ndim == 5:
        if cube_drae.shape[0] != 1:
            raise ValueError("R-A0 wide-domain construction is frame-streamed")
        cube_drae = cube_drae[0]
    if cube_drae.ndim != 4 or cube_drae.shape[0] != 64:
        raise ValueError("R-A0 expects one Full-RAED Cube")
    range_m = range_m.to(device=cube_drae.device, dtype=torch.float32)
    azimuth_rad = azimuth_rad.to(device=cube_drae.device, dtype=torch.float32)
    elevation_rad = elevation_rad.to(device=cube_drae.device, dtype=torch.float32)
    _validate_axes(
        tuple(int(size) for size in cube_drae.shape[1:]),
        range_m,
        azimuth_rad,
        elevation_rad,
    )

    random_rae, random_strata = stratified_random_queries(
        range_m,
        azimuth_rad,
        elevation_rad,
        quotas=RANDOM_RANGE_QUOTAS,
        base_seed=base_seed,
        sequence=sequence,
        radar_index=radar_index,
    )
    radar_rae, radar_sources, radar_strata, radar_reports = (
        low_threshold_radar_queries(
            cube_drae,
            range_m,
            quotas=RADAR_RANGE_QUOTAS,
            base_seed=base_seed,
            sequence=sequence,
            radar_index=radar_index,
            threshold_quantile=LOW_THRESHOLD_QUANTILE,
        )
    )
    raw_rae = torch.cat((random_rae, radar_rae))
    raw_sources = torch.cat(
        (
            torch.full(
                (RANDOM_QUERY_COUNT,),
                SOURCE_RANDOM_FRUSTUM,
                device=cube_drae.device,
                dtype=torch.int8,
            ),
            radar_sources,
        )
    )
    raw_strata = torch.cat((random_strata, radar_strata))
    raw_ids = torch.arange(
        RAW_QUERY_COUNT,
        device=cube_drae.device,
        dtype=torch.long,
    )
    if raw_rae.shape != (RAW_QUERY_COUNT, 3):
        raise AssertionError("R-A0 raw query union must contain exactly 1.2M rows")
    if _source_counts(raw_sources)["random_frustum"] != RANDOM_QUERY_COUNT:
        raise AssertionError("R-A0 random source count changed")
    if (
        _source_counts(raw_sources)["radar_low_threshold"]
        + _source_counts(raw_sources)["radar_helper_augmented"]
        != RADAR_QUERY_COUNT
    ):
        raise AssertionError("R-A0 radar-guided source count changed")
    xyz = _rae_to_xyz_chunked(
        raw_rae,
        range_m,
        azimuth_rad,
        elevation_rad,
        chunk_size=131_072,
    )
    keep = stable_capacity_one_indices(xyz, source_codes=raw_sources)
    unique_rae = raw_rae[keep]
    unique_xyz = xyz[keep]
    unique_ids = raw_ids[keep]
    unique_sources = raw_sources[keep]
    unique_strata = _range_codes_from_xyz(unique_xyz)
    if unique_xyz.shape[0] < EXPORT_COUNT:
        raise RuntimeError("R-A0 unique wide domain cannot fill exact-10k export")
    unique_range_counts = _range_counts(unique_strata)
    if any(count < EXPORT_COUNT for count in unique_range_counts.values()):
        raise RuntimeError(
            "R-A0 each post-dedup range stratum must independently fill 10k"
        )
    report = {
        "raw_query_count": RAW_QUERY_COUNT,
        "random_query_count": RANDOM_QUERY_COUNT,
        "radar_query_count": RADAR_QUERY_COUNT,
        "unique_capacity_cell_count": int(unique_xyz.shape[0]),
        "capacity_cell_collision_count": int(
            RAW_QUERY_COUNT - unique_xyz.shape[0]
        ),
        "capacity_cell_collision_fraction": float(
            1.0 - unique_xyz.shape[0] / RAW_QUERY_COUNT
        ),
        "capacity_cell_m": CAPACITY_CELL_M,
        "deduplication": (
            "one_per_0p05m_cartesian_cell_radar_evidence_preferred"
        ),
        "representative_priority": [
            "radar_low_threshold",
            "radar_helper_augmented",
            "random_frustum",
        ],
        "raw_source_counts": _source_counts(raw_sources),
        "unique_source_counts": _source_counts(unique_sources),
        "raw_range_counts": _range_counts(raw_strata),
        "unique_range_counts": unique_range_counts,
        "unique_range_capacity_at_least_export_count": True,
        "random_range_quotas": dict(
            zip(
                (label for label, _, _ in RANGE_STRATA_M),
                RANDOM_RANGE_QUOTAS,
                strict=True,
            )
        ),
        "radar_range_quotas": dict(
            zip(
                (label for label, _, _ in RANGE_STRATA_M),
                RADAR_RANGE_QUOTAS,
                strict=True,
            )
        ),
        "radar_strata": radar_reports,
        "proposal_generation_inputs": [
            "frame_identity_seed",
            "rae_axes",
            "full_raed_cube_integrated_log_energy",
        ],
        "ground_truth_used_for_proposal_generation": False,
        "global_energy_ranking_used": False,
        "rald_scope": "inspired_initial_query_domain_only",
        "occupancy_dependent_second_pass": False,
    }
    return WideQueryDomain(
        coordinates_rae=unique_rae,
        xyz_m=unique_xyz,
        candidate_ids=unique_ids,
        source_codes=unique_sources,
        range_stratum_codes=unique_strata,
        report=report,
    )


def streaming_bidirectional_nearest(
    candidate_xyz_m: torch.Tensor,
    target_xyz_m: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    candidate_chunk_size: int,
    target_chunk_size: int,
    target_topk_count: int = 1,
) -> StreamingNearest:
    """Compute global directed support and a bounded target top-k shortlist."""

    if (
        candidate_xyz_m.ndim != 2
        or target_xyz_m.ndim != 2
        or candidate_xyz_m.shape[1] != 3
        or target_xyz_m.shape[1] != 3
        or candidate_xyz_m.shape[0] == 0
        or target_xyz_m.shape[0] == 0
    ):
        raise ValueError("R-A0 streaming nearest requires nonempty XYZ matrices")
    if candidate_ids.shape != (candidate_xyz_m.shape[0],):
        raise ValueError("R-A0 candidate IDs must match candidate rows")
    if candidate_chunk_size <= 0 or target_chunk_size <= 0:
        raise ValueError("R-A0 distance chunks must be positive")
    if not 1 <= target_topk_count <= candidate_xyz_m.shape[0]:
        raise ValueError("R-A0 target top-k must fit the candidate domain")
    if candidate_xyz_m.device != target_xyz_m.device:
        raise ValueError("R-A0 candidate and target tensors must share a device")
    target_xyz_m = target_xyz_m.to(candidate_xyz_m)
    candidate_to_target = candidate_xyz_m.new_full(
        (candidate_xyz_m.shape[0],),
        float("inf"),
    )
    target_to_candidate = target_xyz_m.new_full(
        (target_xyz_m.shape[0],),
        float("inf"),
    )
    target_nearest_row = torch.full(
        (target_xyz_m.shape[0],),
        -1,
        device=candidate_xyz_m.device,
        dtype=torch.long,
    )
    target_topk_distance = target_xyz_m.new_full(
        (target_xyz_m.shape[0], target_topk_count),
        float("inf"),
    )
    target_topk_row = torch.full(
        (target_xyz_m.shape[0], target_topk_count),
        -1,
        device=candidate_xyz_m.device,
        dtype=torch.long,
    )
    for candidate_start in range(
        0,
        candidate_xyz_m.shape[0],
        candidate_chunk_size,
    ):
        candidate_stop = min(
            candidate_start + candidate_chunk_size,
            candidate_xyz_m.shape[0],
        )
        rows = torch.arange(
            candidate_start,
            candidate_stop,
            device=candidate_xyz_m.device,
        )
        id_order = torch.argsort(candidate_ids[rows], stable=True)
        ordered_rows = rows[id_order]
        candidate_chunk = candidate_xyz_m[ordered_rows]
        chunk_min = candidate_xyz_m.new_full(
            (candidate_chunk.shape[0],),
            float("inf"),
        )
        for target_start in range(
            0,
            target_xyz_m.shape[0],
            target_chunk_size,
        ):
            target_stop = min(
                target_start + target_chunk_size,
                target_xyz_m.shape[0],
            )
            distance = torch.cdist(
                candidate_chunk,
                target_xyz_m[target_start:target_stop],
            )
            chunk_min = torch.minimum(chunk_min, distance.amin(dim=1))
            local_distance, local_row = distance.min(dim=0)
            global_row = ordered_rows[local_row]
            current_distance = target_to_candidate[target_start:target_stop]
            current_row = target_nearest_row[target_start:target_stop]
            better = local_distance < current_distance
            current_ids = torch.full_like(
                global_row,
                torch.iinfo(candidate_ids.dtype).max,
            )
            valid_current = current_row >= 0
            current_ids[valid_current] = candidate_ids[
                current_row[valid_current]
            ]
            tied_better = (local_distance == current_distance) & (
                candidate_ids[global_row] < current_ids
            )
            update = better | tied_better
            target_to_candidate[target_start:target_stop] = torch.where(
                update,
                local_distance,
                current_distance,
            )
            target_nearest_row[target_start:target_stop] = torch.where(
                update,
                global_row,
                current_row,
            )
            local_k = min(target_topk_count, candidate_chunk.shape[0])
            local_topk_distance, local_topk_index = torch.topk(
                distance,
                local_k,
                dim=0,
                largest=False,
                sorted=True,
            )
            local_topk_row = ordered_rows[local_topk_index].transpose(0, 1)
            local_topk_distance = local_topk_distance.transpose(0, 1)
            current_topk_distance = target_topk_distance[
                target_start:target_stop
            ]
            current_topk_row = target_topk_row[target_start:target_stop]
            combined_distance = torch.cat(
                (current_topk_distance, local_topk_distance),
                dim=1,
            )
            combined_row = torch.cat(
                (current_topk_row, local_topk_row),
                dim=1,
            )
            combined_id = torch.full_like(
                combined_row,
                torch.iinfo(candidate_ids.dtype).max,
            )
            valid = combined_row >= 0
            combined_id[valid] = candidate_ids[combined_row[valid]]
            by_id = torch.argsort(combined_id, dim=1, stable=True)
            combined_distance = torch.gather(combined_distance, 1, by_id)
            combined_row = torch.gather(combined_row, 1, by_id)
            by_distance = torch.argsort(
                combined_distance,
                dim=1,
                stable=True,
            )[:, :target_topk_count]
            target_topk_distance[target_start:target_stop] = torch.gather(
                combined_distance,
                1,
                by_distance,
            )
            target_topk_row[target_start:target_stop] = torch.gather(
                combined_row,
                1,
                by_distance,
            )
            del distance
        candidate_to_target[ordered_rows] = chunk_min
    if not torch.isfinite(candidate_to_target).all():
        raise RuntimeError("R-A0 candidate-to-target streaming distance is incomplete")
    if not torch.isfinite(target_to_candidate).all():
        raise RuntimeError("R-A0 target-to-candidate streaming distance is incomplete")
    if bool((target_nearest_row < 0).any()):
        raise RuntimeError("R-A0 target assignment is incomplete")
    if not torch.isfinite(target_topk_distance).all():
        raise RuntimeError("R-A0 target top-k support is incomplete")
    if bool((target_topk_row < 0).any()):
        raise RuntimeError("R-A0 target top-k assignment is incomplete")
    return StreamingNearest(
        candidate_to_target_m=candidate_to_target,
        target_to_candidate_m=target_to_candidate,
        target_nearest_candidate_row=target_nearest_row,
        target_topk_candidate_rows=target_topk_row,
        target_topk_distance_m=target_topk_distance,
    )


def _range_masks(xyz_m: torch.Tensor) -> list[torch.Tensor]:
    codes = _range_codes_from_xyz(xyz_m)
    return [codes == index for index in range(len(RANGE_STRATA_M))]


def _capacity_constrained_quotas(
    target_mass: list[float],
    candidate_capacity: list[int],
) -> list[int]:
    if sum(candidate_capacity) < EXPORT_COUNT:
        raise ValueError("R-A0 candidate capacity cannot fill exact-10k export")
    quotas = [0 for _ in candidate_capacity]
    for _ in range(EXPORT_COUNT):
        eligible = [
            index
            for index, capacity in enumerate(candidate_capacity)
            if quotas[index] < capacity
        ]
        if not eligible:
            raise RuntimeError("R-A0 exhausted range capacity")
        supported = [index for index in eligible if target_mass[index] > 0.0]
        allocation = supported or eligible
        if supported:
            chosen = min(
                allocation,
                key=lambda index: (
                    (quotas[index] + 1) / target_mass[index],
                    index,
                ),
            )
        else:
            chosen = min(
                allocation,
                key=lambda index: (
                    (quotas[index] + 1)
                    / max(candidate_capacity[index], 1),
                    index,
                ),
            )
        quotas[chosen] += 1
    return quotas


def _stable_distance_order(
    distance_m: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    by_id = torch.argsort(candidate_ids, stable=True)
    return by_id[torch.argsort(distance_m[by_id], stable=True)]


def capacity_aware_topk_greedy(
    candidate_xyz_m: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_range_codes: torch.Tensor,
    candidate_to_target_m: torch.Tensor,
    target_topk_candidate_rows: torch.Tensor,
    target_topk_distance_m: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    output_quotas: list[int],
    minimum_distance_m: float = CAPACITY_CELL_M,
) -> tuple[torch.Tensor, dict]:
    """Select a deterministic heuristic set with reassignment and true NMS.

    Targets are processed by descending weight and may fall back through their
    global top-k shortlist when a candidate is already used, its range quota is
    full, or it violates the true Euclidean minimum distance. Remaining quota is
    filled by global candidate-to-target distance. This is capacity-aware but is
    not a globally optimal assignment.
    """

    candidate_count = candidate_xyz_m.shape[0]
    if candidate_xyz_m.shape != (candidate_count, 3):
        raise ValueError("R-A0 greedy candidate XYZ must have shape (N,3)")
    for values, name in (
        (candidate_ids, "candidate IDs"),
        (candidate_range_codes, "candidate range codes"),
        (candidate_to_target_m, "candidate support distances"),
    ):
        if values.shape != (candidate_count,):
            raise ValueError(f"R-A0 greedy {name} must match candidate rows")
    if (
        target_topk_candidate_rows.ndim != 2
        or target_topk_distance_m.shape != target_topk_candidate_rows.shape
        or target_topk_candidate_rows.shape[0] != target_weight.shape[0]
    ):
        raise ValueError("R-A0 greedy target top-k tensors differ")
    if len(output_quotas) != len(RANGE_STRATA_M):
        raise ValueError("R-A0 greedy output quotas must match range strata")
    if any(quota < 0 for quota in output_quotas):
        raise ValueError("R-A0 greedy output quotas must be nonnegative")
    if minimum_distance_m <= 0.0:
        raise ValueError("R-A0 greedy minimum distance must be positive")
    if bool((candidate_range_codes < 0).any()) or bool(
        (candidate_range_codes >= len(RANGE_STRATA_M)).any()
    ):
        raise ValueError("R-A0 greedy candidate range code is invalid")

    capacities = [
        int((candidate_range_codes == index).sum().item())
        for index in range(len(RANGE_STRATA_M))
    ]
    if any(
        quota > capacity
        for quota, capacity in zip(output_quotas, capacities, strict=True)
    ):
        raise ValueError("R-A0 post-dedup range capacity cannot fill its quota")

    xyz_cpu = candidate_xyz_m.detach().cpu()
    ids_cpu = candidate_ids.detach().cpu()
    range_cpu = candidate_range_codes.detach().cpu()
    topk_rows_cpu = target_topk_candidate_rows.detach().cpu()
    topk_distance_cpu = target_topk_distance_m.detach().cpu()
    weight_cpu = target_weight.detach().cpu()
    selected_rows: list[int] = []
    selected_set: set[int] = set()
    selected_cells: dict[
        tuple[int, int, int],
        list[tuple[float, float, float]],
    ] = {}
    selected_range_counts = [0 for _ in RANGE_STRATA_M]
    rejected_reuse = 0
    rejected_distance = 0
    rejected_quota = 0
    topk_attempt_count = 0
    topk_assignment_count = 0
    minimum_distance_squared = minimum_distance_m**2

    def try_candidate(row: int) -> bool:
        nonlocal rejected_reuse, rejected_distance, rejected_quota
        if row in selected_set:
            rejected_reuse += 1
            return False
        stratum = int(range_cpu[row].item())
        if selected_range_counts[stratum] >= output_quotas[stratum]:
            rejected_quota += 1
            return False
        point = tuple(float(value) for value in xyz_cpu[row].tolist())
        cell = tuple(
            math.floor(value / minimum_distance_m) for value in point
        )
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    neighbours = selected_cells.get(
                        (
                            cell[0] + delta_x,
                            cell[1] + delta_y,
                            cell[2] + delta_z,
                        )
                    )
                    if neighbours is None:
                        continue
                    for neighbour in neighbours:
                        distance_squared = sum(
                            (left - right) ** 2
                            for left, right in zip(
                                point,
                                neighbour,
                                strict=True,
                            )
                        )
                        if distance_squared < minimum_distance_squared:
                            rejected_distance += 1
                            return False
        selected_rows.append(row)
        selected_set.add(row)
        selected_cells.setdefault(cell, []).append(point)
        selected_range_counts[stratum] += 1
        return True

    target_order = torch.argsort(
        weight_cpu,
        descending=True,
        stable=True,
    ).tolist()
    for target_row in target_order:
        shortlist = [
            (
                float(topk_distance_cpu[target_row, rank].item()),
                int(ids_cpu[int(topk_rows_cpu[target_row, rank].item())].item()),
                int(topk_rows_cpu[target_row, rank].item()),
            )
            for rank in range(topk_rows_cpu.shape[1])
        ]
        for _, _, candidate_row in sorted(shortlist):
            topk_attempt_count += 1
            if try_candidate(candidate_row):
                topk_assignment_count += 1
                break

    for stratum in range(len(RANGE_STRATA_M)):
        if selected_range_counts[stratum] >= output_quotas[stratum]:
            continue
        candidate_pool = torch.nonzero(
            candidate_range_codes == stratum,
            as_tuple=False,
        ).flatten()
        fill_order = _stable_distance_order(
            candidate_to_target_m[candidate_pool],
            candidate_ids[candidate_pool],
        )
        for candidate_row in candidate_pool[fill_order].detach().cpu().tolist():
            if try_candidate(int(candidate_row)) and (
                selected_range_counts[stratum] >= output_quotas[stratum]
            ):
                break

    if selected_range_counts != output_quotas:
        raise RuntimeError(
            "R-A0 true 5 cm Euclidean selection cannot fill output quotas"
        )
    selected = torch.tensor(
        selected_rows,
        device=candidate_xyz_m.device,
        dtype=torch.long,
    )
    if selected.numel() != sum(output_quotas):
        raise AssertionError("R-A0 greedy selection changed output cardinality")
    report = {
        "method": "capacity_aware_global_topk_greedy_with_true_5cm_nms",
        "heuristic": True,
        "globally_optimal": False,
        "strict_upper_bound": False,
        "target_topk_count": int(target_topk_candidate_rows.shape[1]),
        "topk_attempt_count": topk_attempt_count,
        "topk_assignment_count": topk_assignment_count,
        "fill_candidate_count": int(selected.numel()) - topk_assignment_count,
        "rejected_candidate_reuse_count": rejected_reuse,
        "rejected_range_quota_count": rejected_quota,
        "rejected_euclidean_duplicate_count": rejected_distance,
        "minimum_euclidean_distance_m": minimum_distance_m,
        "range_capacity_validated_after_unique_pool": True,
        "range_capacity_after_unique_pool": dict(
            zip(
                (label for label, _, _ in RANGE_STRATA_M),
                capacities,
                strict=True,
            )
        ),
        "output_quotas": dict(
            zip(
                (label for label, _, _ in RANGE_STRATA_M),
                output_quotas,
                strict=True,
            )
        ),
        "selected_range_counts": dict(
            zip(
                (label for label, _, _ in RANGE_STRATA_M),
                selected_range_counts,
                strict=True,
            )
        ),
    }
    return selected, report


def _weighted_fraction(
    distance_m: torch.Tensor,
    weight: torch.Tensor,
    threshold_m: float,
) -> float:
    return float(
        (
            ((distance_m <= threshold_m).to(weight) * weight).sum()
            / weight.sum().clamp_min(1e-8)
        ).item()
    )


def _validate_diagnostic_inputs(
    domain: WideQueryDomain,
    target_xyz_m: torch.Tensor,
    target_weight: torch.Tensor | None,
) -> torch.Tensor:
    count = domain.xyz_m.shape[0]
    if domain.xyz_m.ndim != 2 or domain.xyz_m.shape[1] != 3:
        raise ValueError("R-A0 wide domain must contain XYZ rows")
    if domain.coordinates_rae.shape != (count, 3):
        raise ValueError("R-A0 wide domain RAE/XYZ counts differ")
    for values, name in (
        (domain.candidate_ids, "candidate IDs"),
        (domain.source_codes, "source labels"),
        (domain.range_stratum_codes, "range labels"),
    ):
        if values.shape != (count,):
            raise ValueError(f"R-A0 {name} must match candidate rows")
    if domain.candidate_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("R-A0 candidate IDs must be integers")
    if torch.unique(domain.candidate_ids).numel() != count:
        raise ValueError("R-A0 candidate IDs must be unique")
    if torch.unique(capacity_cell_keys(domain.xyz_m)).numel() != count:
        raise ValueError("R-A0 diagnostic requires a cell-unique candidate domain")
    actual_range_codes = _range_codes_from_xyz(domain.xyz_m)
    if not torch.equal(actual_range_codes, domain.range_stratum_codes):
        raise ValueError("R-A0 stored and physical range labels differ")
    if (
        target_xyz_m.ndim != 2
        or target_xyz_m.shape[1] != 3
        or target_xyz_m.shape[0] == 0
    ):
        raise ValueError("R-A0 target XYZ must have nonempty shape (N,3)")
    if not torch.isfinite(target_xyz_m).all():
        raise ValueError("R-A0 target XYZ must be finite")
    if target_weight is None:
        target_weight = torch.ones(
            target_xyz_m.shape[0],
            device=target_xyz_m.device,
            dtype=target_xyz_m.dtype,
        )
    if target_weight.shape != (target_xyz_m.shape[0],):
        raise ValueError("R-A0 target weights must have shape (N,)")
    target_weight = target_weight.to(target_xyz_m)
    if not torch.isfinite(target_weight).all() or bool((target_weight < 0).any()):
        raise ValueError("R-A0 target weights must be finite and nonnegative")
    if float(target_weight.sum().item()) <= 0.0:
        raise ValueError("R-A0 target weights must have positive mass")
    return target_weight


def select_wide_support_diagnostic(
    domain: WideQueryDomain,
    target_xyz_m: torch.Tensor,
    *,
    target_weight: torch.Tensor | None = None,
    candidate_chunk_size: int = 8_192,
    target_chunk_size: int = 4_096,
) -> WideSupportDiagnosticSelection:
    """Select an exact-10k GT-guided heuristic from the initial query domain."""

    target_xyz_m = target_xyz_m.to(domain.xyz_m)
    target_weight = _validate_diagnostic_inputs(
        domain,
        target_xyz_m,
        target_weight,
    )
    candidate_masks = _range_masks(domain.xyz_m)
    target_masks = _range_masks(target_xyz_m)
    capacities = [int(mask.sum().item()) for mask in candidate_masks]
    target_mass = [
        float(target_weight[mask].sum().item()) for mask in target_masks
    ]
    quotas = _capacity_constrained_quotas(target_mass, capacities)
    if any(
        quota > capacity
        for quota, capacity in zip(quotas, capacities, strict=True)
    ):
        raise RuntimeError("R-A0 post-dedup range capacity validation failed")

    global_nearest = streaming_bidirectional_nearest(
        domain.xyz_m,
        target_xyz_m,
        domain.candidate_ids,
        candidate_chunk_size=candidate_chunk_size,
        target_chunk_size=target_chunk_size,
        target_topk_count=TARGET_REASSIGNMENT_TOPK,
    )
    candidate_range_codes = _range_codes_from_xyz(domain.xyz_m)
    selected_pool_indices, selection_report = capacity_aware_topk_greedy(
        domain.xyz_m,
        domain.candidate_ids,
        candidate_range_codes,
        global_nearest.candidate_to_target_m,
        global_nearest.target_topk_candidate_rows,
        global_nearest.target_topk_distance_m,
        target_weight,
        output_quotas=quotas,
        minimum_distance_m=CAPACITY_CELL_M,
    )
    if selected_pool_indices.shape != (EXPORT_COUNT,):
        raise AssertionError("R-A0 must export exactly 10,000 candidates")
    selected_candidate_ids = domain.candidate_ids[selected_pool_indices]
    if torch.unique(selected_candidate_ids).numel() != EXPORT_COUNT:
        raise AssertionError("R-A0 selected one candidate more than once")
    selected_nearest = streaming_bidirectional_nearest(
        domain.xyz_m[selected_pool_indices],
        target_xyz_m,
        selected_candidate_ids,
        candidate_chunk_size=candidate_chunk_size,
        target_chunk_size=target_chunk_size,
        target_topk_count=1,
    )

    per_range: dict[str, dict[str, float | int | None | dict]] = {}
    for stratum, (label, _, _) in enumerate(RANGE_STRATA_M):
        candidate_pool = torch.nonzero(
            candidate_masks[stratum],
            as_tuple=False,
        ).flatten()
        target_indices = torch.nonzero(
            target_masks[stratum],
            as_tuple=False,
        ).flatten()
        selected_pool = selected_pool_indices[
            candidate_range_codes[selected_pool_indices] == stratum
        ]
        local_weight = target_weight[target_indices]
        report: dict[str, float | int | None | dict | bool | str] = {
            "candidate_count": int(candidate_pool.numel()),
            "target_count": int(target_indices.numel()),
            "target_effective_count": float(local_weight.sum().item()),
            "output_quota": int(quotas[stratum]),
            "selected_count": int(selected_pool.numel()),
            "candidate_source_counts": _source_counts(
                domain.source_codes[candidate_pool]
            ),
            "selected_source_counts": _source_counts(
                domain.source_codes[selected_pool]
            ),
            "pool_support_scope": "global_cross_range_nearest",
            "range_used_only_for_reporting_and_output_quota": True,
        }
        for threshold_m in SUPPORT_THRESHOLDS_M:
            suffix = str(threshold_m).replace(".", "p")
            report[f"candidate_to_gt_support_fraction_{suffix}m"] = float(
                (
                    global_nearest.candidate_to_target_m[candidate_pool]
                    <= threshold_m
                )
                .float()
                .mean()
                .item()
            )
            report[f"gt_recall_from_pool_{suffix}m"] = (
                _weighted_fraction(
                    global_nearest.target_to_candidate_m[target_indices],
                    local_weight,
                    threshold_m,
                )
                if target_indices.numel()
                else None
            )
            report[f"gt_recall_from_selected_{suffix}m"] = (
                _weighted_fraction(
                    selected_nearest.target_to_candidate_m[target_indices],
                    local_weight,
                    threshold_m,
                )
                if target_indices.numel()
                else None
            )
        per_range[label] = report

    if not torch.isfinite(global_nearest.target_to_candidate_m).all():
        raise RuntimeError("R-A0 global full-pool target support is incomplete")
    if not torch.isfinite(global_nearest.candidate_to_target_m).all():
        raise RuntimeError("R-A0 global full-pool candidate support is incomplete")
    if not torch.isfinite(selected_nearest.target_to_candidate_m).all():
        raise RuntimeError("R-A0 selected target support is incomplete")
    overall: dict[str, float | int | None | bool | str] = {
        "candidate_count": int(domain.xyz_m.shape[0]),
        "target_count": int(target_xyz_m.shape[0]),
        "target_effective_count": float(target_weight.sum().item()),
        "selected_count": EXPORT_COUNT,
        "pool_support_scope": "global_cross_range_nearest",
        "selection_is_heuristic": True,
        "strict_upper_bound": False,
    }
    for threshold_m in SUPPORT_THRESHOLDS_M:
        suffix = str(threshold_m).replace(".", "p")
        overall[f"candidate_to_gt_support_fraction_{suffix}m"] = float(
            (
                global_nearest.candidate_to_target_m <= threshold_m
            ).float().mean().item()
        )
        overall[f"gt_recall_from_pool_{suffix}m"] = _weighted_fraction(
            global_nearest.target_to_candidate_m,
            target_weight,
            threshold_m,
        )
        overall[f"gt_recall_from_selected_{suffix}m"] = _weighted_fraction(
            selected_nearest.target_to_candidate_m,
            target_weight,
            threshold_m,
        )
    return WideSupportDiagnosticSelection(
        selected_pool_indices=selected_pool_indices,
        selected_candidate_ids=selected_candidate_ids,
        selected_xyz_m=domain.xyz_m[selected_pool_indices],
        selected_source_codes=domain.source_codes[selected_pool_indices],
        selected_range_stratum_codes=candidate_range_codes[selected_pool_indices],
        overall_support=overall,
        per_range_support=per_range,
        selection_report=selection_report,
        artifact_label=diagnostic_artifact_label(),
    )
