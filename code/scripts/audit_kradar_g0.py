#!/usr/bin/env python3
"""Run the K-Radar G0 schema, synchronization, CFAR, and target audit on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.io import whosmat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar import KRadarAxes, load_axes, load_frame  # noqa: E402
from cube_dense.observability import (  # noqa: E402
    CFARConfig,
    CFARResult,
    ObservableTarget,
    ca_cfar_points,
    deskew_lidar_to_reference,
    observable_lidar_target,
    validate_cfar_roundtrip,
)


QUANTILES = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)


def load_sensor_times(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                result[row[0].strip()] = float(row[1])
    return result


def load_odometry(path: Path) -> dict[str, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True)
    timestamp = np.asarray(values["timestamp"], dtype=np.float64) / 1e6
    position = np.column_stack(
        (values["local_x"], values["local_y"], values["local_z"])
    ).astype(np.float64)
    velocity = np.column_stack(
        [np.gradient(position[:, axis], timestamp) for axis in range(3)]
    )
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    yaw_rate = np.gradient(heading, timestamp)
    return {
        "timestamp": timestamp,
        "position": position,
        "velocity": velocity,
        "heading": heading,
        "yaw_rate": yaw_rate,
    }


def load_pose_odometry(
    path: Path,
    labels: list[str],
    os2_times: dict[str, float],
) -> dict[str, np.ndarray]:
    """Attach official KITTI-format poses to label-defined OS2 timestamps."""

    labels = sorted(
        labels,
        key=lambda name: int(name.removesuffix(".txt").split("_", maxsplit=1)[1]),
    )
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape != (len(labels), 12):
        raise ValueError(
            f"Odometry/label mismatch in {path}: {values.shape} vs {len(labels)} labels"
        )
    poses = values.reshape(-1, 3, 4)
    lidar_indices = [
        int(name.removesuffix(".txt").split("_", maxsplit=1)[1])
        for name in labels
    ]
    timestamp = np.asarray(
        [os2_times[f"os2-64_{index:05d}.pcd"] for index in lidar_indices],
        dtype=np.float64,
    )
    position = poses[:, :, 3]
    heading = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    velocity = np.column_stack(
        [np.gradient(position[:, axis], timestamp) for axis in range(3)]
    )
    yaw_rate = np.gradient(heading, timestamp)
    return {
        "timestamp": timestamp,
        "position": position,
        "velocity": velocity,
        "heading": heading,
        "yaw_rate": yaw_rate,
    }


def interpolate_motion(odometry: dict[str, np.ndarray], timestamp: float) -> dict:
    time_axis = odometry["timestamp"]
    nearest = int(np.argmin(np.abs(time_axis - timestamp)))
    right = int(np.searchsorted(time_axis, timestamp, side="left"))
    right = min(max(right, 1), len(time_axis) - 1)
    left = right - 1
    position = [
        float(np.interp(timestamp, time_axis, odometry["position"][:, axis]))
        for axis in range(3)
    ]
    velocity = [
        float(np.interp(timestamp, time_axis, odometry["velocity"][:, axis]))
        for axis in range(3)
    ]
    speed = float(np.linalg.norm(velocity))
    return {
        "position_xyz_m": position,
        "velocity_xyz_mps": velocity,
        "speed_mps": speed,
        "heading_rad": float(
            np.interp(timestamp, time_axis, odometry["heading"])
        ),
        "yaw_rate_radps": float(
            np.interp(timestamp, time_axis, odometry["yaw_rate"])
        ),
        "nearest_timestamp_delta_ms": float(
            abs(time_axis[nearest] - timestamp) * 1e3
        ),
        "interpolation_bracket_ms": float((time_axis[right] - time_axis[left]) * 1e3),
    }


def tensor_quantiles(values: torch.Tensor) -> list[float]:
    flat = values.flatten()
    step = max(1, math.ceil(flat.numel() / 1_000_000))
    sample = flat[::step]
    levels = torch.tensor(QUANTILES, dtype=sample.dtype, device=sample.device)
    return [float(value) for value in torch.quantile(sample, levels).cpu().tolist()]


def axis_report(values: np.ndarray) -> dict:
    delta = np.diff(values)
    return {
        "count": int(len(values)),
        "min": float(values[0]),
        "max": float(values[-1]),
        "median_step": float(np.median(delta)),
        "strictly_increasing": bool(np.all(delta > 0)),
    }


def target_report(target: ObservableTarget, source_count: int) -> dict:
    surface = target.surface_mask
    surface_confidence = target.confidence[surface]
    surface_margin = target.threshold_margin[surface]
    observable = surface_confidence >= 0.5
    return {
        "source_count": int(source_count),
        "in_fov_count": int(target.points_xyz.shape[0]),
        "in_fov_fraction": float(target.points_xyz.shape[0] / source_count),
        "surface_count": int(surface.sum().item()),
        "surface_fraction_of_fov": float(surface.float().mean().item()),
        "observable_count": int(observable.sum().item()),
        "observable_fraction_of_surface": float(observable.float().mean().item()),
        "confidence_quantiles_surface": tensor_quantiles(surface_confidence),
        "margin_median_surface": float(surface_margin.median().item()),
    }


def doppler_alias_report(
    result: CFARResult, axes: KRadarAxes, ego_speed_mps: float
) -> dict:
    points = result.points_xyzd_power_snr
    observed = points[:, 3]
    radius = torch.linalg.vector_norm(points[:, :3], dim=1).clamp_min(1e-6)
    forward_projection = ego_speed_mps * points[:, 0] / radius
    doppler_axis = torch.as_tensor(
        axes.doppler_mps, dtype=points.dtype, device=points.device
    )
    step = torch.median(torch.diff(doppler_axis))
    period = step * doppler_axis.numel()
    lower = doppler_axis[0]

    def wrap(value: torch.Tensor) -> torch.Tensor:
        return torch.remainder(value - lower, period) + lower

    def circular_error(prediction: torch.Tensor) -> torch.Tensor:
        return torch.remainder(observed - prediction + period / 2, period) - period / 2

    negative_error = circular_error(wrap(-forward_projection)).abs()
    positive_error = circular_error(wrap(forward_projection)).abs()
    zero_error = circular_error(torch.zeros_like(forward_projection)).abs()
    negative_median = float(negative_error.median().item())
    positive_median = float(positive_error.median().item())
    zero_median = float(zero_error.median().item())
    hypotheses = {
        "negative_ego": negative_median,
        "positive_ego": positive_median,
        "zero_centered": zero_median,
    }
    return {
        "doppler_bin_step_mps": float(step.item()),
        "alias_period_mps": float(period.item()),
        "negative_ego_hypothesis_median_error_mps": negative_median,
        "positive_ego_hypothesis_median_error_mps": positive_median,
        "zero_centered_hypothesis_median_error_mps": zero_median,
        "preferred_static_hypothesis": min(hypotheses, key=hypotheses.get),
        "interpretation": (
            "Zero-centered power dominates; this audit does not support applying a raw "
            "ego-radial correction without a separate compensation study."
            if zero_median == min(hypotheses.values())
            else "An ego-radial hypothesis fits better than a zero-centered spectrum."
        ),
        "random_circular_median_baseline_mps": float(period.item() / 4.0),
    }


def save_cache(
    path: Path,
    result: CFARResult,
    target: ObservableTarget,
    motion: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    surface = target.surface_mask
    target_values = torch.cat(
        (target.points_xyz[surface], target.confidence[surface, None]), dim=1
    )
    np.savez_compressed(
        path,
        cfar_xyzd_power_snr=result.points_xyzd_power_snr.detach().cpu().numpy().astype(np.float32),
        cfar_drae_index=result.indices_drae.detach().cpu().numpy().astype(np.int16),
        target_xyz_confidence=target_values.detach().cpu().numpy().astype(np.float32),
        target_rae_index=target.indices_rae[surface].detach().cpu().numpy().astype(np.int16),
        ego_velocity_xyz_mps=np.asarray(motion["velocity_xyz_mps"], dtype=np.float32),
        ego_speed_mps=np.asarray(motion["speed_mps"], dtype=np.float32),
        ego_yaw_rate_radps=np.asarray(motion["yaw_rate_radps"], dtype=np.float32),
    )


def save_figure(
    path: Path,
    axes: KRadarAxes,
    result: CFARResult,
    target: ObservableTarget,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak_ra = torch.log10(result.peak_power.amax(dim=2).clamp_min(1.0)).cpu().numpy()
    cfar = result.points_xyzd_power_snr.detach().cpu().numpy()
    surface = target.surface_mask
    lidar = target.points_xyz[surface].detach().cpu().numpy()
    confidence = target.confidence[surface].detach().cpu().numpy()
    fig, panel = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    image = panel[0, 0].imshow(
        peak_ra,
        origin="lower",
        aspect="auto",
        extent=(
            np.rad2deg(axes.azimuth_rad[0]),
            np.rad2deg(axes.azimuth_rad[-1]),
            axes.range_m[0],
            axes.range_m[-1],
        ),
        cmap="magma",
    )
    panel[0, 0].set(title="Doppler/elevation max", xlabel="Azimuth (deg)", ylabel="Range (m)")
    fig.colorbar(image, ax=panel[0, 0], label="log10 power")

    stride = max(1, len(lidar) // 20_000)
    panel[0, 1].scatter(
        lidar[::stride, 1],
        lidar[::stride, 0],
        color="#8a8f98",
        s=1,
        alpha=0.18,
        label="LiDAR surface",
    )
    observable = confidence >= 0.5
    panel[0, 1].scatter(
        lidar[observable, 1],
        lidar[observable, 0],
        color="#1b9e77",
        s=2,
        alpha=0.55,
        label="Observable target",
    )
    panel[0, 1].legend(loc="upper right", frameon=False, markerscale=4)
    scatter = panel[0, 1].scatter(
        cfar[:, 1], cfar[:, 0], c=cfar[:, 3], s=4, cmap="coolwarm", alpha=0.8
    )
    panel[0, 1].set(
        title="Observable LiDAR and CFAR peaks",
        xlabel="Lateral y (m)",
        ylabel="Forward x (m)",
        xlim=(-50, 50),
        ylim=(0, 100),
        aspect="equal",
    )
    fig.colorbar(scatter, ax=panel[0, 1], label="Doppler (m/s)")

    panel[1, 0].hist(cfar[:, 3], bins=len(axes.doppler_mps), color="#2878b5")
    panel[1, 0].set(title="CFAR Doppler distribution", xlabel="Doppler (m/s)", ylabel="Count")
    panel[1, 1].hist(confidence, bins=40, range=(0, 1), color="#d85a40")
    panel[1, 1].axvline(0.5, color="black", linestyle="--", linewidth=1)
    panel[1, 1].set(title="Surface observability", xlabel="Confidence", ylabel="Count")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_report(frames: list[dict], required_frames: int) -> dict:
    successful = [frame for frame in frames if "error" not in frame]
    if not successful:
        return {"successful_frames": 0, "gate_pass": False, "checks": {}}

    os2_delta = [abs(frame["synchronization"]["os2_delta_ms"]) for frame in successful]
    os1_delta = [abs(frame["synchronization"]["os1_delta_ms"]) for frame in successful]
    odometry_delta = [
        frame["motion"]["nearest_timestamp_delta_ms"] for frame in successful
    ]
    roundtrip = [frame["cfar_roundtrip"]["exact_bin_fraction"] for frame in successful]
    observable = [
        frame["observability"]["observable_fraction_of_surface"]
        for frame in successful
    ]
    alignment = [
        frame["alignment_null"]["correct_minus_mirror_margin"]
        for frame in successful
    ]
    best_doppler_error = [
        min(
            frame["doppler_alias"]["negative_ego_hypothesis_median_error_mps"],
            frame["doppler_alias"]["positive_ego_hypothesis_median_error_mps"],
            frame["doppler_alias"]["zero_centered_hypothesis_median_error_mps"],
        )
        for frame in successful
    ]
    random_doppler_error = [
        frame["doppler_alias"]["random_circular_median_baseline_mps"]
        for frame in successful
    ]
    selected_minus_none = [
        frame["lidar_scan_timing"]["margin_median_by_reference"][
            frame["lidar_scan_timing"]["selected_reference"]
        ]
        - frame["lidar_scan_timing"]["margin_median_by_reference"]["none"]
        for frame in successful
    ]
    checks = {
        "required_frame_count": len(successful) >= required_frames,
        "schema_consistent": all(
            frame["schema"]["logical_shape"] == [64, 256, 107, 37]
            and frame["schema"]["raw_dtype"] == "float64"
            for frame in successful
        ),
        "os2_label_sync_le_1ms": max(os2_delta) <= 1.0,
        "odometry_support_le_60ms": max(odometry_delta) <= 60.0,
        "lidar_point_time_present": all(
            "t" in frame["schema"]["lidar64_fields"] for frame in successful
        ),
        "selected_deskew_not_worse_than_none": float(np.mean(selected_minus_none))
        >= 0.0,
        "cfar_exact_roundtrip": all(
            frame["cfar_roundtrip"]["exact_bin_count"]
            == frame["cfar_roundtrip"]["point_count"]
            for frame in successful
        ),
        "observable_target_nonempty": min(observable) > 0.005,
        "observable_target_stable": float(np.std(observable)) < 0.15,
        "correct_azimuth_beats_mirror": float(np.mean(alignment)) > 0.0,
        "doppler_hypothesis_beats_random": float(np.mean(best_doppler_error))
        < float(np.mean(random_doppler_error)),
    }
    return {
        "successful_frames": len(successful),
        "failed_frames": len(frames) - len(successful),
        "sequence_count": len({frame.get("sequence", 1) for frame in successful}),
        "partition_frame_count": {
            partition: sum(
                frame.get("partition", "feasibility") == partition
                for frame in successful
            )
            for partition in sorted(
                {frame.get("partition", "feasibility") for frame in successful}
            )
        },
        "ego_speed_mps_min": float(
            min(frame["motion"]["speed_mps"] for frame in successful)
        ),
        "ego_speed_mps_max": float(
            max(frame["motion"]["speed_mps"] for frame in successful)
        ),
        "os2_abs_delta_ms_max": float(max(os2_delta)),
        "os1_abs_delta_ms_mean": float(np.mean(os1_delta)),
        "os1_abs_delta_ms_max": float(max(os1_delta)),
        "odometry_nearest_delta_ms_max": float(max(odometry_delta)),
        "cfar_roundtrip_fraction_min": float(min(roundtrip)),
        "observable_fraction_mean": float(np.mean(observable)),
        "observable_fraction_std": float(np.std(observable)),
        "correct_minus_mirror_margin_mean": float(np.mean(alignment)),
        "selected_minus_no_deskew_margin_mean": float(np.mean(selected_minus_none)),
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
    }


def stratified_report(frames: list[dict]) -> dict:
    successful = [frame for frame in frames if "error" not in frame]
    groups: dict[str, list[dict]] = defaultdict(list)
    for frame in successful:
        group_name = f"partition:{frame.get('partition', 'feasibility')}"
        groups[group_name].append(frame)
        for tag in frame.get("description", []):
            groups[f"tag:{tag}"].append(frame)

    def summarize(group: list[dict]) -> dict:
        observable = [
            frame["observability"]["observable_fraction_of_surface"]
            for frame in group
        ]
        alignment = [
            frame["alignment_null"]["correct_minus_mirror_margin"]
            for frame in group
        ]
        deskew = [
            frame["lidar_scan_timing"]["margin_median_by_reference"][
                frame["lidar_scan_timing"]["selected_reference"]
            ]
            - frame["lidar_scan_timing"]["margin_median_by_reference"]["none"]
            for frame in group
        ]
        speed = [frame["motion"]["speed_mps"] for frame in group]
        return {
            "frame_count": len(group),
            "sequence_count": len({frame.get("sequence", 1) for frame in group}),
            "observable_fraction_mean": float(np.mean(observable)),
            "observable_fraction_std": float(np.std(observable)),
            "correct_minus_mirror_margin_mean": float(np.mean(alignment)),
            "selected_minus_no_deskew_margin_mean": float(np.mean(deskew)),
            "ego_speed_mps_min": float(min(speed)),
            "ego_speed_mps_max": float(max(speed)),
        }

    return {name: summarize(group) for name, group in sorted(groups.items())}


def write_markdown(path: Path, payload: dict) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# K-Radar G0 Audit",
        "",
        f"Frames completed: {aggregate.get('successful_frames', 0)}",
        f"Gate pass: **{aggregate.get('gate_pass', False)}**",
        "",
        "## Gate Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in aggregate.get("checks", {}).items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Aggregate Evidence",
            "",
            f"- Sequences covered: {aggregate.get('sequence_count', 0)}",
            f"- Partition frame counts: {aggregate.get('partition_frame_count', {})}",
            f"- Ego-speed range: {aggregate.get('ego_speed_mps_min', float('nan')):.3f} to {aggregate.get('ego_speed_mps_max', float('nan')):.3f} m/s",
            f"- OS2/label maximum timestamp delta: {aggregate.get('os2_abs_delta_ms_max', float('nan')):.6f} ms",
            f"- OS1/label mean absolute delta: {aggregate.get('os1_abs_delta_ms_mean', float('nan')):.3f} ms",
            f"- Odometry nearest-sample maximum delta: {aggregate.get('odometry_nearest_delta_ms_max', float('nan')):.3f} ms",
            f"- Minimum exact CFAR round-trip rate: {aggregate.get('cfar_roundtrip_fraction_min', float('nan')):.6f}",
            f"- Observable surface fraction: {aggregate.get('observable_fraction_mean', float('nan')):.4f} +/- {aggregate.get('observable_fraction_std', float('nan')):.4f}",
            f"- Correct-minus-mirrored angular margin: {aggregate.get('correct_minus_mirror_margin_mean', float('nan')):.4f}",
            f"- Selected deskew-minus-no-deskew margin: {aggregate.get('selected_minus_no_deskew_margin_mean', float('nan')):.4f}",
            "",
            "The primary geometry target uses OS2-64 because its timestamp is the label timestamp. OS1-128 is retained as an asynchronous auxiliary source.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--required-frames", type=int, default=8)
    parser.add_argument("--max-cfar-points", type=int, default=10_000)
    parser.add_argument("--false-alarm-rate", type=float, default=1e-3)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--audit-manifest", type=Path, default=None)
    parser.add_argument("--scene-split", type=Path, default=None)
    parser.add_argument("--odometry-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--lidar-time-reference",
        choices=("none", "start", "center", "end"),
        default="start",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("G0 audit requires an available CUDA device")
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    resources = args.root / "resources"
    axes = load_axes(resources)
    if args.audit_manifest is None:
        sequence_root = args.root / "1"
        label_paths = (
            [sequence_root / "info_label" / name for name in args.labels]
            if args.labels
            else sorted((sequence_root / "info_label").glob("*.txt"))
        )[: args.max_frames]
        frame_specs = [
            {"sequence": 1, "partition": "feasibility", "label": path.name}
            for path in label_paths
        ]
        scene_split = None
    else:
        if args.scene_split is None or args.odometry_root is None:
            raise ValueError(
                "--scene-split and --odometry-root are required with --audit-manifest"
            )
        manifest = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
        frame_specs = manifest["frames"][: args.max_frames]
        scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))

    contexts: dict[int, dict] = {}

    def sequence_context(sequence: int, partition: str) -> dict:
        if sequence in contexts:
            return contexts[sequence]
        sequence_root = args.root / str(sequence)
        os1_times = load_sensor_times(sequence_root / "time_info" / "os1-128.txt")
        os2_times = load_sensor_times(sequence_root / "time_info" / "os2-64.txt")
        if scene_split is None:
            odometry = load_odometry(resources / "seq_1_local.csv")
        else:
            labels = scene_split["splits"][partition]["labels"][str(sequence)]
            odometry = load_pose_odometry(
                args.odometry_root / f"gt_{sequence:02d}.txt",
                labels,
                os2_times,
            )
        odometry_cuda = {
            key: torch.as_tensor(
                value,
                dtype=torch.float64 if key == "timestamp" else torch.float32,
                device=args.device,
            )
            for key, value in odometry.items()
            if key in {"timestamp", "position", "heading"}
        }
        context = {
            "root": sequence_root,
            "os1_times": os1_times,
            "os2_times": os2_times,
            "odometry": odometry,
            "odometry_cuda": odometry_cuda,
        }
        contexts[sequence] = context
        return context
    config = CFARConfig(
        false_alarm_rate=args.false_alarm_rate,
        max_points=args.max_cfar_points,
    )
    report_path = args.output / "g0_audit.json"
    frames: list[dict] = []
    if report_path.exists() and not args.overwrite:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        frames = existing.get("frames", [])
    payload = {
        "device": args.device,
        "protocol": (
            "cross-scene manifest audit"
            if args.audit_manifest is not None
            else "sequence-1 feasibility audit"
        ),
        "axes": {
            "doppler_mps": axis_report(axes.doppler_mps),
            "range_m": axis_report(axes.range_m),
            "azimuth_rad": axis_report(axes.azimuth_rad),
            "elevation_rad": axis_report(axes.elevation_rad),
        },
        "frames": frames,
    }

    for spec in frame_specs:
        sequence = int(spec["sequence"])
        partition = spec["partition"]
        context = sequence_context(sequence, partition)
        sequence_root = context["root"]
        os1_times = context["os1_times"]
        os2_times = context["os2_times"]
        odometry = context["odometry"]
        odometry_cuda = context["odometry_cuda"]
        label_path = sequence_root / "info_label" / spec["label"]
        if any(
            frame.get("sequence", 1) == sequence
            and frame.get("label") == label_path.name
            and "error" not in frame
            for frame in frames
        ):
            print(
                json.dumps(
                    {
                        "sequence": sequence,
                        "label": label_path.name,
                        "status": "cached",
                    }
                ),
                flush=True,
            )
            continue
        frames = [
            frame
            for frame in frames
            if not (
                frame.get("sequence", 1) == sequence
                and frame.get("label") == label_path.name
            )
        ]
        started = time.monotonic()
        try:
            frame = load_frame(sequence_root, label_path, resources)
            cube_path = (
                sequence_root
                / "radar_tesseract"
                / f"tesseract_{frame.indices.radar:05d}.mat"
            )
            raw_metadata = next(item for item in whosmat(cube_path) if item[0] == "arrDREA")
            cube = torch.as_tensor(
                frame.cube_drae, dtype=torch.float32, device=args.device
            )
            result = ca_cfar_points(cube, axes, config)
            roundtrip = validate_cfar_roundtrip(cube, axes, result)
            lidar = torch.as_tensor(
                frame.lidar64, dtype=torch.float32, device=args.device
            )
            calibration = torch.as_tensor(
                frame.calibration.translation_xyz_m,
                dtype=lidar.dtype,
                device=lidar.device,
            )
            point_offsets_s = lidar[:, frame.lidar64_fields.index("t")] * 1e-9
            scan_duration_s = float(point_offsets_s.max().item())
            origin_shifts = {
                "start": 0.0,
                "center": -scan_duration_s / 2.0,
                "end": -scan_duration_s,
            }
            calibrated_candidates = {"none": lidar[:, :3] + calibration}
            for name, origin_shift in origin_shifts.items():
                calibrated_candidates[name] = deskew_lidar_to_reference(
                    points_xyz=lidar[:, :3],
                    point_offsets_s=point_offsets_s,
                    reference_timestamp=frame.indices.timestamp,
                    timestamp_origin_shift_s=origin_shift,
                    calibration_xyz_m=calibration,
                    odometry_timestamps=odometry_cuda["timestamp"],
                    odometry_positions=odometry_cuda["position"],
                    odometry_headings=odometry_cuda["heading"],
                )
            timing_targets = {
                name: observable_lidar_target(points, axes, result)
                for name, points in calibrated_candidates.items()
            }
            calibrated = calibrated_candidates[args.lidar_time_reference]
            target = timing_targets[args.lidar_time_reference]
            mirrored = calibrated.clone()
            mirrored[:, 1] *= -1
            mirror_target = observable_lidar_target(mirrored, axes, result)
            motion = interpolate_motion(odometry, frame.indices.timestamp)

            sensitivity = {}
            for z_offset in (0.0, 0.7, 1.0):
                offset = calibration.clone()
                offset[2] = z_offset
                if args.lidar_time_reference == "none":
                    sensitivity_points = lidar[:, :3] + offset
                else:
                    sensitivity_points = deskew_lidar_to_reference(
                        points_xyz=lidar[:, :3],
                        point_offsets_s=point_offsets_s,
                        reference_timestamp=frame.indices.timestamp,
                        timestamp_origin_shift_s=origin_shifts[
                            args.lidar_time_reference
                        ],
                        calibration_xyz_m=offset,
                        odometry_timestamps=odometry_cuda["timestamp"],
                        odometry_positions=odometry_cuda["position"],
                        odometry_headings=odometry_cuda["heading"],
                    )
                sensitivity[f"z_{z_offset:.1f}m"] = target_report(
                    observable_lidar_target(sensitivity_points, axes, result),
                    len(lidar),
                )
            correct_margin = float(
                target.threshold_margin[target.surface_mask].median().item()
            )
            mirror_margin = float(
                mirror_target.threshold_margin[mirror_target.surface_mask].median().item()
            )
            frame_report = {
                "sequence": sequence,
                "partition": partition,
                "description": spec.get("description", []),
                "label": label_path.name,
                "schema": {
                    "on_disk_shape": list(raw_metadata[1]),
                    "logical_shape": list(frame.cube_drae.shape),
                    "raw_dtype": str(frame.cube_drae.dtype),
                    "raw_nbytes": int(frame.cube_drae.nbytes),
                    "gpu_dtype": str(cube.dtype),
                    "power_quantile_levels": list(QUANTILES),
                    "power_quantiles": tensor_quantiles(cube),
                    "lidar64_shape": list(frame.lidar64.shape),
                    "lidar64_fields": list(frame.lidar64_fields),
                    "lidar128_shape": None
                    if frame.lidar128 is None
                    else list(frame.lidar128.shape),
                    "lidar128_fields": None
                    if frame.lidar128_fields is None
                    else list(frame.lidar128_fields),
                },
                "synchronization": {
                    "label_timestamp": frame.indices.timestamp,
                    "os2_delta_ms": (
                        os2_times[f"os2-64_{frame.indices.lidar64:05d}.pcd"]
                        - frame.indices.timestamp
                    )
                    * 1e3,
                    "os1_delta_ms": (
                        os1_times[f"os1-128_{frame.indices.lidar128:05d}.pcd"]
                        - frame.indices.timestamp
                    )
                    * 1e3,
                },
                "motion": motion,
                "lidar_scan_timing": {
                    "point_time_field": "t",
                    "point_time_unit": "nanoseconds from scan origin",
                    "duration_ms": scan_duration_s * 1e3,
                    "selected_reference": args.lidar_time_reference,
                    "translation_during_scan_m": motion["speed_mps"]
                    * scan_duration_s,
                    "margin_median_by_reference": {
                        name: float(
                            candidate.threshold_margin[
                                candidate.surface_mask
                            ].median().item()
                        )
                        for name, candidate in timing_targets.items()
                    },
                },
                "calibration_xyz_m": frame.calibration.translation_xyz_m.tolist(),
                "cfar": {
                    "candidate_count": result.candidate_count,
                    "retained_count": int(result.indices_drae.shape[0]),
                    "threshold_scale": result.threshold_scale,
                },
                "cfar_roundtrip": roundtrip,
                "observability": target_report(target, len(lidar)),
                "alignment_null": {
                    "correct_margin_median": correct_margin,
                    "mirror_margin_median": mirror_margin,
                    "correct_minus_mirror_margin": correct_margin - mirror_margin,
                },
                "calibration_sensitivity": sensitivity,
                "doppler_alias": doppler_alias_report(result, axes, motion["speed_mps"]),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            save_cache(
                args.cache_root
                / f"seq{sequence:02d}_radar_{frame.indices.radar:05d}.npz",
                result,
                target,
                motion,
            )
            save_figure(
                args.output
                / "figures"
                / f"seq{sequence:02d}_radar_{frame.indices.radar:05d}.png",
                axes,
                result,
                target,
            )
            frames.append(frame_report)
            print(json.dumps(frame_report, indent=2), flush=True)
            del cube, result, target, mirror_target, frame
            torch.cuda.empty_cache()
        except Exception as error:  # Continue the audit so all failures are visible.
            frames.append(
                {
                    "sequence": sequence,
                    "partition": partition,
                    "label": label_path.name,
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            print(json.dumps(frames[-1], indent=2), flush=True)
        frames.sort(key=lambda item: (item.get("sequence", 1), item["label"]))
        payload["frames"] = frames
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    payload["aggregate"] = aggregate_report(frames, args.required_frames)
    payload["stratified"] = stratified_report(frames)
    payload["dataset_descriptions"] = {
        str(sequence): (args.root / str(sequence) / "description.txt")
        .read_text(encoding="utf-8")
        .strip()
        for sequence in sorted({int(spec["sequence"]) for spec in frame_specs})
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output / "g0_audit.md", payload)
    print(json.dumps(payload["aggregate"], indent=2), flush=True)
    if any("error" in frame for frame in frames):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
