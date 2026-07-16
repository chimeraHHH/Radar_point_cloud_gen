"""Strict loaders and coordinate transforms for full K-Radar DRAE tensors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


LABEL_HEADER = re.compile(
    r"idx\(tesseract_os2-64_cam-front_os1-128_cam-lrr\)="
    r"(?P<radar>\d+)_(?P<lidar64>\d+)_(?P<cam_front>\d+)_"
    r"(?P<lidar128>\d+)_(?P<cam_other>\d+), timestamp=(?P<timestamp>[0-9.]+)"
)


@dataclass(frozen=True)
class KRadarAxes:
    doppler_mps: np.ndarray
    range_m: np.ndarray
    azimuth_rad: np.ndarray
    elevation_rad: np.ndarray


@dataclass(frozen=True)
class FrameIndices:
    radar: int
    lidar64: int
    cam_front: int
    lidar128: int
    cam_other: int
    timestamp: float


@dataclass(frozen=True)
class Calibration:
    frame_offset: int
    translation_xyz_m: np.ndarray


@dataclass(frozen=True)
class KRadarFrame:
    indices: FrameIndices
    cube_drae: np.ndarray
    lidar64: np.ndarray
    lidar64_fields: tuple[str, ...]
    lidar128: np.ndarray | None
    lidar128_fields: tuple[str, ...] | None
    calibration: Calibration


def load_axes(resources: Path) -> KRadarAxes:
    rae = loadmat(resources / "info_arr.mat")
    doppler = loadmat(resources / "arr_doppler.mat")["arr_doppler"].reshape(-1)
    axes = KRadarAxes(
        doppler_mps=doppler.astype(np.float64),
        range_m=rae["arrRange"].reshape(-1).astype(np.float64),
        azimuth_rad=np.deg2rad(rae["arrAzimuth"].reshape(-1).astype(np.float64)),
        elevation_rad=np.deg2rad(rae["arrElevation"].reshape(-1).astype(np.float64)),
    )
    expected = (64, 256, 107, 37)
    actual = tuple(map(len, (axes.doppler_mps, axes.range_m, axes.azimuth_rad, axes.elevation_rad)))
    if actual != expected:
        raise ValueError(f"Unexpected K-Radar axes {actual}; expected {expected}")
    return axes


def load_tesseract(path: Path, reverse_angular_axes: bool = True) -> np.ndarray:
    on_disk = loadmat(path)["arrDREA"]
    if on_disk.shape != (64, 256, 37, 107):
        raise ValueError(f"Unexpected arrDREA shape {on_disk.shape} in {path}")
    cube = np.transpose(on_disk, (0, 1, 3, 2))
    if reverse_angular_axes:
        cube = np.flip(np.flip(cube, axis=2), axis=3)
    if not np.isfinite(cube).all():
        raise ValueError(f"Non-finite values in {path}")
    return np.ascontiguousarray(cube)


def load_pcd_ascii(path: Path) -> tuple[np.ndarray, list[str]]:
    fields: list[str] | None = None
    header_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            tokens = line.strip().split()
            if not tokens or tokens[0].startswith("#"):
                continue
            key, *values = tokens
            if key == "FIELDS":
                fields = values
            if key == "DATA":
                if values != ["ascii"]:
                    raise ValueError(f"Only ASCII PCD is supported: {path}")
                break
    if not fields:
        raise ValueError(f"PCD fields missing in {path}")
    values = np.loadtxt(path, dtype=np.float32, skiprows=header_lines)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(fields):
        raise ValueError(f"PCD field count mismatch in {path}")
    return values, fields


def parse_label_header(path: Path) -> FrameIndices:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    match = LABEL_HEADER.search(first_line)
    if not match:
        raise ValueError(f"Unrecognized K-Radar label header: {first_line}")
    values = match.groupdict()
    return FrameIndices(
        radar=int(values["radar"]),
        lidar64=int(values["lidar64"]),
        cam_front=int(values["cam_front"]),
        lidar128=int(values["lidar128"]),
        cam_other=int(values["cam_other"]),
        timestamp=float(values["timestamp"]),
    )


def load_calibration(path: Path, z_offset_m: float = 0.7) -> Calibration:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Malformed calibration file: {path}")
    values = [float(value) for value in lines[1].split(",")]
    if len(values) < 3:
        raise ValueError(f"Malformed calibration row: {lines[1]}")
    return Calibration(
        frame_offset=int(values[0]),
        translation_xyz_m=np.array([values[1], values[2], z_offset_m], dtype=np.float64),
    )


def lidar_to_radar(points_xyz: np.ndarray, calibration: Calibration) -> np.ndarray:
    return points_xyz + calibration.translation_xyz_m.reshape(1, 3)


def cartesian_to_polar(points_xyz: np.ndarray) -> np.ndarray:
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    radius = np.linalg.norm(points_xyz, axis=1)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(np.divide(z, radius, out=np.zeros_like(z), where=radius > 0))
    return np.column_stack((radius, azimuth, elevation))


def polar_to_cartesian(
    range_m: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray
) -> np.ndarray:
    cos_elevation = np.cos(elevation)
    return np.column_stack(
        (
            range_m * cos_elevation * np.cos(azimuth),
            range_m * cos_elevation * np.sin(azimuth),
            range_m * np.sin(elevation),
        )
    )


def nearest_bin(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(values, query, side="left")
    right = np.clip(right, 0, len(values) - 1)
    left = np.clip(right - 1, 0, len(values) - 1)
    choose_left = np.abs(query - values[left]) <= np.abs(values[right] - query)
    return np.where(choose_left, left, right)


def load_frame(
    sequence_root: Path,
    label_path: Path,
    resources: Path,
    z_offset_m: float = 0.7,
) -> KRadarFrame:
    axes = load_axes(resources)
    indices = parse_label_header(label_path)
    calibration = load_calibration(
        sequence_root / "info_calib" / "calib_radar_lidar.txt",
        z_offset_m=z_offset_m,
    )
    lidar64, lidar64_fields = load_pcd_ascii(
        sequence_root / "os2-64" / f"os2-64_{indices.lidar64:05d}.pcd"
    )
    lidar128_path = sequence_root / "os1-128" / f"os1-128_{indices.lidar128:05d}.pcd"
    if lidar128_path.exists():
        lidar128, lidar128_fields = load_pcd_ascii(lidar128_path)
    else:
        lidar128, lidar128_fields = None, None
    cube = load_tesseract(
        sequence_root / "radar_tesseract" / f"tesseract_{indices.radar:05d}.mat"
    )
    expected_shape = tuple(
        map(
            len,
            (
                axes.doppler_mps,
                axes.range_m,
                axes.azimuth_rad,
                axes.elevation_rad,
            ),
        )
    )
    if cube.shape != expected_shape:
        raise ValueError(f"Cube/axis shape mismatch: {cube.shape} != {expected_shape}")
    return KRadarFrame(
        indices=indices,
        cube_drae=cube,
        lidar64=lidar64,
        lidar64_fields=tuple(lidar64_fields),
        lidar128=lidar128,
        lidar128_fields=None
        if lidar128_fields is None
        else tuple(lidar128_fields),
        calibration=calibration,
    )
