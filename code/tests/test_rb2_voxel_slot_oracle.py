from __future__ import annotations

import numpy as np
import pytest

from eval.rb2_voxel_slot_oracle import (
    DEFAULT_CONFIGS,
    EXPORT_COUNT,
    OUTPUT_RANGE_QUOTAS,
    VoxelSlotCapacityError,
    VoxelSlotConfig,
    construct_gt_aided_slots,
    make_lattice,
    run_gt_aided_voxel_slot_oracle,
)
from scripts.preflight_rb2_voxel_slot import (
    FORMAL_TRAIN_COUNT,
    FORMAL_VALIDATION_COUNT,
    select_mode_records,
    validate_manifest_contract,
)


def synthetic_target() -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(
        [5.0, 12.0, 22.0, 29.0, 34.0, 45.0, 57.0, 65.0, 82.0, 105.0],
        dtype=np.float64,
    )
    azimuth = np.radians(
        np.asarray([-30.0, -12.0, 0.0, 20.0, 35.0], dtype=np.float64)
    )
    elevation = np.radians(
        np.asarray([-8.0, 0.0, 7.0], dtype=np.float64)
    )
    points = []
    for value in radius:
        for azimuth_value in azimuth:
            for elevation_value in elevation:
                cosine = np.cos(elevation_value)
                points.append(
                    (
                        value * cosine * np.cos(azimuth_value),
                        value * cosine * np.sin(azimuth_value),
                        value * np.sin(elevation_value),
                    )
                )
    xyz = np.asarray(points, dtype=np.float64)
    weight = np.linspace(0.5, 1.0, xyz.shape[0], dtype=np.float64)
    return xyz, weight


def manifest_frames() -> list[dict]:
    train = [
        {
            "partition": "train",
            "sequence": 1 + index // 10,
            "radar_index": index,
        }
        for index in range(FORMAL_TRAIN_COUNT)
    ]
    validation = [
        {
            "partition": "validation",
            "sequence": 40 + index // 10,
            "radar_index": 1_000 + index,
        }
        for index in range(FORMAL_VALIDATION_COUNT)
    ]
    return train + validation


def test_cartesian_voxel_indices_are_unique_and_stable() -> None:
    config = DEFAULT_CONFIGS[0]
    lattice = make_lattice(config)
    points = np.asarray(
        [
            [10.01, 0.01, 0.01],
            [10.02, 0.02, 0.02],
            [10.61, 0.01, 0.01],
        ],
        dtype=np.float64,
    )
    first = lattice.point_indices(points)
    second = lattice.point_indices(points.copy())
    assert np.array_equal(first, second)
    assert tuple(first[0]) == tuple(first[1])
    assert tuple(first[0]) != tuple(first[2])
    ids = [lattice.linear_id(tuple(int(value) for value in row)) for row in first]
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


@pytest.mark.parametrize("config", DEFAULT_CONFIGS, ids=lambda item: item.name)
def test_slots_are_bounded_mutually_exclusive_and_fixed(
    config: VoxelSlotConfig,
) -> None:
    xyz, _ = synthetic_target()
    lattice = make_lattice(config)
    index = tuple(int(value) for value in lattice.point_indices(xyz[:1])[0])
    placement = construct_gt_aided_slots(
        lattice,
        index,
        xyz,
        config,
        local_target_xyz_m=xyz[:1],
    )
    assert placement.xyz_m.shape == (config.slots_per_voxel, 3)
    lower, upper = lattice.cell_bounds(index)
    margin = 0.51 * config.minimum_distance_m
    assert np.all(placement.xyz_m >= lower + margin - 1e-10)
    assert np.all(placement.xyz_m <= upper - margin + 1e-10)
    distances = np.linalg.norm(
        placement.xyz_m[:, None] - placement.xyz_m[None],
        axis=2,
    )
    distances[np.diag_indices_from(distances)] = np.inf
    assert distances.min() >= config.minimum_distance_m - 1e-9


def test_oracle_is_stable_exact_10k_and_explicitly_gt_aided() -> None:
    xyz, weight = synthetic_target()
    first = run_gt_aided_voxel_slot_oracle(xyz, weight, DEFAULT_CONFIGS[0])
    second = run_gt_aided_voxel_slot_oracle(xyz, weight, DEFAULT_CONFIGS[0])

    assert first.selected_xyz_m.shape == (EXPORT_COUNT, 3)
    assert np.array_equal(first.selected_xyz_m, second.selected_xyz_m)
    assert first.report["hashes"] == second.report["hashes"]
    assert first.report["selection"]["selected_by_range"] == {
        "range_0_30m": OUTPUT_RANGE_QUOTAS[0],
        "range_30_60m": OUTPUT_RANGE_QUOTAS[1],
        "range_60_120m": OUTPUT_RANGE_QUOTAS[2],
    }
    assert (
        first.report["selection"]["observed_minimum_pair_distance_m"]
        >= 0.05 - 1e-9
    )
    label = first.report["artifact_label"]
    assert label["unattainable_gt_aided_heuristic"] is True
    assert label["strict_upper_bound"] is False
    assert label["eligible_as_model_result"] is False
    assert label["ground_truth_used_for_activation"] is True
    assert label["ground_truth_used_for_slot_offsets"] is True
    assert label["ground_truth_used_for_selection"] is True
    assert first.report["forbidden_operations"] == {
        "copy": False,
        "padding": False,
        "jitter": False,
        "free_centers": False,
    }


def test_capacity_shortfall_is_a_hard_failure() -> None:
    xyz, weight = synthetic_target()
    impossible = VoxelSlotConfig(
        name="impossible_coarse_capacity",
        voxel_size_xyz_m=(40.0, 40.0, 40.0),
        slots_per_voxel=1,
        template_resolution=2,
    )
    with pytest.raises(VoxelSlotCapacityError, match="insufficient range capacity"):
        run_gt_aided_voxel_slot_oracle(xyz, weight, impossible)


def test_target_outside_fov_is_rejected() -> None:
    xyz, weight = synthetic_target()
    xyz[0] = np.asarray([-1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="outside the frozen FOV"):
        run_gt_aided_voxel_slot_oracle(xyz, weight, DEFAULT_CONFIGS[0])


def test_manifest_rejects_test_and_mode_counts_are_fixed() -> None:
    frames = manifest_frames()
    validated = validate_manifest_contract({"frames": frames})
    assert len(select_mode_records(validated, "preflight")) == 2
    assert len(select_mode_records(validated, "capacity")) == 100
    assert len(select_mode_records(validated, "geometry")) == 24
    assert {
        frame["partition"] for frame in select_mode_records(validated, "preflight")
    } == {"train", "validation"}

    contaminated = frames[:-1] + [
        {"partition": "test", "sequence": 99, "radar_index": 99}
    ]
    with pytest.raises(ValueError, match="76/24"):
        validate_manifest_contract({"frames": contaminated})
