"""GPU operators for K-Radar CFAR extraction and observable LiDAR targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .kradar import KRadarAxes


@dataclass(frozen=True)
class CFARConfig:
    train_kernel: int = 9
    guard_kernel: int = 3
    local_kernel: int = 3
    false_alarm_rate: float = 1e-3
    max_points: int = 10_000

    def validate(self) -> None:
        kernels = (self.train_kernel, self.guard_kernel, self.local_kernel)
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError(f"CFAR kernels must be positive odd integers: {kernels}")
        if self.train_kernel <= self.guard_kernel:
            raise ValueError("train_kernel must exceed guard_kernel")
        if not 0.0 < self.false_alarm_rate < 1.0:
            raise ValueError("false_alarm_rate must be in (0, 1)")
        if self.max_points < 1:
            raise ValueError("max_points must be positive")


@dataclass
class CFARResult:
    indices_drae: torch.Tensor
    points_xyzd_power_snr: torch.Tensor
    peak_power: torch.Tensor
    noise_power: torch.Tensor
    threshold_scale: float
    candidate_count: int


@dataclass
class ObservableTarget:
    points_xyz: torch.Tensor
    confidence: torch.Tensor
    threshold_margin: torch.Tensor
    surface_mask: torch.Tensor
    indices_rae: torch.Tensor
    source_indices: torch.Tensor


def _interpolate_pose(
    timestamps: torch.Tensor,
    positions: torch.Tensor,
    headings: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    right = torch.searchsorted(timestamps, query.contiguous()).clamp(
        1, timestamps.numel() - 1
    )
    left = right - 1
    scale = (query - timestamps[left]) / (timestamps[right] - timestamps[left])
    position_scale = scale.to(positions.dtype)
    heading_scale = scale.to(headings.dtype)
    position = positions[left] + position_scale[:, None] * (
        positions[right] - positions[left]
    )
    heading = headings[left] + heading_scale * (headings[right] - headings[left])
    return position, heading


def deskew_lidar_to_reference(
    points_xyz: torch.Tensor,
    point_offsets_s: torch.Tensor,
    reference_timestamp: float,
    timestamp_origin_shift_s: float,
    calibration_xyz_m: torch.Tensor,
    odometry_timestamps: torch.Tensor,
    odometry_positions: torch.Tensor,
    odometry_headings: torch.Tensor,
) -> torch.Tensor:
    """Deskew a rotating LiDAR scan into the radar frame at a reference time."""

    if not points_xyz.is_cuda:
        raise ValueError("LiDAR deskew must run on CUDA")
    point_times = (
        torch.full(
            point_offsets_s.shape,
            reference_timestamp,
            dtype=odometry_timestamps.dtype,
            device=point_offsets_s.device,
        )
        + timestamp_origin_shift_s
        + point_offsets_s.to(odometry_timestamps.dtype)
    )
    point_position, point_heading = _interpolate_pose(
        odometry_timestamps,
        odometry_positions,
        odometry_headings,
        point_times,
    )
    reference_query = torch.full(
        (1,),
        reference_timestamp,
        dtype=odometry_timestamps.dtype,
        device=points_xyz.device,
    )
    reference_position, reference_heading = _interpolate_pose(
        odometry_timestamps,
        odometry_positions,
        odometry_headings,
        reference_query,
    )

    calibrated = points_xyz + calibration_xyz_m
    point_cos = torch.cos(point_heading)
    point_sin = torch.sin(point_heading)
    global_x = point_cos * calibrated[:, 0] - point_sin * calibrated[:, 1]
    global_y = point_sin * calibrated[:, 0] + point_cos * calibrated[:, 1]
    global_xyz = torch.stack(
        (global_x, global_y, calibrated[:, 2]), dim=1
    ) + point_position
    relative = global_xyz - reference_position
    reference_cos = torch.cos(reference_heading[0])
    reference_sin = torch.sin(reference_heading[0])
    reference_x = reference_cos * relative[:, 0] + reference_sin * relative[:, 1]
    reference_y = -reference_sin * relative[:, 0] + reference_cos * relative[:, 1]
    return torch.stack((reference_x, reference_y, relative[:, 2]), dim=1)


def _axis_tensor(values: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def nearest_bin(axis: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Return nearest indices for a monotonically increasing one-dimensional axis."""

    query = query.contiguous()
    right = torch.searchsorted(axis, query, right=False).clamp(0, axis.numel() - 1)
    left = (right - 1).clamp(0, axis.numel() - 1)
    choose_left = (query - axis[left]).abs() <= (axis[right] - query).abs()
    return torch.where(choose_left, left, right)


def cartesian_to_polar(points_xyz: torch.Tensor) -> tuple[torch.Tensor, ...]:
    radius = torch.linalg.vector_norm(points_xyz, dim=1)
    azimuth = torch.atan2(points_xyz[:, 1], points_xyz[:, 0])
    elevation = torch.asin(
        (points_xyz[:, 2] / radius.clamp_min(torch.finfo(radius.dtype).tiny)).clamp(
            -1.0, 1.0
        )
    )
    return radius, azimuth, elevation


def ca_cfar_points(
    cube_drae: torch.Tensor,
    axes: KRadarAxes,
    config: CFARConfig,
) -> CFARResult:
    """Extract 3D CA-CFAR spatial peaks and retain each peak's Doppler argmax."""

    config.validate()
    expected = (
        len(axes.doppler_mps),
        len(axes.range_m),
        len(axes.azimuth_rad),
        len(axes.elevation_rad),
    )
    if tuple(cube_drae.shape) != expected:
        raise ValueError(f"Cube shape {tuple(cube_drae.shape)} != axes {expected}")
    if not cube_drae.is_cuda:
        raise ValueError("G0 CFAR computation must run on CUDA")

    peak_power, doppler_index = cube_drae.max(dim=0)
    volume = peak_power[None, None]
    train_cells = config.train_kernel**3 - config.guard_kernel**3
    train_sum = (
        F.avg_pool3d(
            volume,
            config.train_kernel,
            stride=1,
            padding=config.train_kernel // 2,
        )
        * config.train_kernel**3
    )
    guard_sum = (
        F.avg_pool3d(
            volume,
            config.guard_kernel,
            stride=1,
            padding=config.guard_kernel // 2,
        )
        * config.guard_kernel**3
    )
    noise_power = ((train_sum - guard_sum) / train_cells).squeeze(0).squeeze(0)
    noise_power = noise_power.clamp_min(torch.finfo(peak_power.dtype).tiny)
    threshold_scale = train_cells * (
        config.false_alarm_rate ** (-1.0 / train_cells) - 1.0
    )
    local_max = F.max_pool3d(
        volume,
        config.local_kernel,
        stride=1,
        padding=config.local_kernel // 2,
    ).squeeze(0).squeeze(0)
    candidates = (peak_power >= local_max) & (
        peak_power > noise_power * threshold_scale
    )
    border = config.train_kernel // 2
    valid = torch.zeros_like(candidates)
    valid[border:-border, border:-border, border:-border] = True
    candidates &= valid
    candidate_count = int(candidates.sum().item())
    if candidate_count == 0:
        raise RuntimeError("CA-CFAR returned no candidates")

    count = min(candidate_count, config.max_points)
    scores = peak_power.masked_fill(~candidates, -torch.inf)
    selected_power, linear = torch.topk(scores.flatten(), count)
    num_azimuth = peak_power.shape[1]
    num_elevation = peak_power.shape[2]
    range_index = linear // (num_azimuth * num_elevation)
    remainder = linear % (num_azimuth * num_elevation)
    azimuth_index = remainder // num_elevation
    elevation_index = remainder % num_elevation
    selected_doppler = doppler_index[
        range_index, azimuth_index, elevation_index
    ]

    range_axis = _axis_tensor(axes.range_m, cube_drae)
    azimuth_axis = _axis_tensor(axes.azimuth_rad, cube_drae)
    elevation_axis = _axis_tensor(axes.elevation_rad, cube_drae)
    doppler_axis = _axis_tensor(axes.doppler_mps, cube_drae)
    radius = range_axis[range_index]
    azimuth = azimuth_axis[azimuth_index]
    elevation = elevation_axis[elevation_index]
    cos_elevation = torch.cos(elevation)
    xyz = torch.stack(
        (
            radius * cos_elevation * torch.cos(azimuth),
            radius * cos_elevation * torch.sin(azimuth),
            radius * torch.sin(elevation),
        ),
        dim=1,
    )
    selected_noise = noise_power[range_index, azimuth_index, elevation_index]
    snr = selected_power / selected_noise
    indices = torch.stack(
        (selected_doppler, range_index, azimuth_index, elevation_index), dim=1
    )
    points = torch.cat(
        (
            xyz,
            doppler_axis[selected_doppler, None],
            selected_power[:, None],
            snr[:, None],
        ),
        dim=1,
    )
    return CFARResult(
        indices_drae=indices,
        points_xyzd_power_snr=points,
        peak_power=peak_power,
        noise_power=noise_power,
        threshold_scale=float(threshold_scale),
        candidate_count=candidate_count,
    )


def validate_cfar_roundtrip(
    cube_drae: torch.Tensor,
    axes: KRadarAxes,
    result: CFARResult,
) -> dict[str, float | int]:
    """Project extracted XYZ+Doppler points back to DRAE and compare exact bins."""

    points = result.points_xyzd_power_snr
    radius, azimuth, elevation = cartesian_to_polar(points[:, :3])
    doppler_axis = _axis_tensor(axes.doppler_mps, cube_drae)
    range_axis = _axis_tensor(axes.range_m, cube_drae)
    azimuth_axis = _axis_tensor(axes.azimuth_rad, cube_drae)
    elevation_axis = _axis_tensor(axes.elevation_rad, cube_drae)
    recovered = torch.stack(
        (
            nearest_bin(doppler_axis, points[:, 3]),
            nearest_bin(range_axis, radius),
            nearest_bin(azimuth_axis, azimuth),
            nearest_bin(elevation_axis, elevation),
        ),
        dim=1,
    )
    exact = (recovered == result.indices_drae).all(dim=1)
    exact_count = int(exact.sum().item())
    d, r, a, e = recovered.unbind(dim=1)
    recovered_power = cube_drae[d, r, a, e]
    reference_power = points[:, 4]
    relative_error = (recovered_power - reference_power).abs() / reference_power.abs().clamp_min(
        torch.finfo(reference_power.dtype).tiny
    )
    return {
        "point_count": int(points.shape[0]),
        "exact_bin_count": exact_count,
        "exact_bin_fraction": exact_count / int(points.shape[0]),
        "max_relative_power_error": float(relative_error.max().item()),
    }


def observable_lidar_target(
    points_xyz: torch.Tensor,
    axes: KRadarAxes,
    result: CFARResult,
    surface_tolerance_m: float = 1.0,
    margin_temperature: float = 0.5,
) -> ObservableTarget:
    """Assign radar observability using local CFAR margin and angular first surfaces."""

    if not points_xyz.is_cuda:
        raise ValueError("G0 observability computation must run on CUDA")
    radius, azimuth, elevation = cartesian_to_polar(points_xyz)
    range_axis = _axis_tensor(axes.range_m, points_xyz)
    azimuth_axis = _axis_tensor(axes.azimuth_rad, points_xyz)
    elevation_axis = _axis_tensor(axes.elevation_rad, points_xyz)
    finite = torch.isfinite(points_xyz).all(dim=1)
    in_fov = (
        finite
        & (radius >= range_axis[0])
        & (radius <= range_axis[-1])
        & (azimuth >= azimuth_axis[0])
        & (azimuth <= azimuth_axis[-1])
        & (elevation >= elevation_axis[0])
        & (elevation <= elevation_axis[-1])
        & (points_xyz[:, 0] > 0.0)
    )
    source_indices = torch.nonzero(in_fov, as_tuple=False).flatten()
    points_xyz = points_xyz[in_fov]
    radius = radius[in_fov]
    azimuth = azimuth[in_fov]
    elevation = elevation[in_fov]
    range_index = nearest_bin(range_axis, radius)
    azimuth_index = nearest_bin(azimuth_axis, azimuth)
    elevation_index = nearest_bin(elevation_axis, elevation)

    local_peak = F.max_pool3d(
        result.peak_power[None, None], 3, stride=1, padding=1
    ).squeeze(0).squeeze(0)
    peak = local_peak[range_index, azimuth_index, elevation_index]
    noise = result.noise_power[range_index, azimuth_index, elevation_index]
    threshold = noise * result.threshold_scale
    margin = torch.log((peak + 1.0) / (threshold + 1.0))
    confidence = torch.sigmoid(margin / margin_temperature)

    angular_index = azimuth_index * len(axes.elevation_rad) + elevation_index
    nearest_surface = torch.full(
        (len(axes.azimuth_rad) * len(axes.elevation_rad),),
        torch.inf,
        dtype=radius.dtype,
        device=radius.device,
    )
    nearest_surface.scatter_reduce_(
        0, angular_index, radius, reduce="amin", include_self=True
    )
    surface_mask = radius <= nearest_surface[angular_index] + surface_tolerance_m
    confidence = confidence * surface_mask.to(confidence.dtype)
    indices = torch.stack((range_index, azimuth_index, elevation_index), dim=1)
    return ObservableTarget(
        points_xyz=points_xyz,
        confidence=confidence,
        threshold_margin=margin,
        surface_mask=surface_mask,
        indices_rae=indices,
        source_indices=source_indices,
    )
