import inspect

import numpy as np
import pytest
import torch

from cube_dense.kradar import KRadarAxes
from eval.geometry_cardinality import (
    FIXED_K_VALUES,
    AdaptiveCountRule,
    aggregate_cardinality_frames,
    evaluate_cardinality_arm,
    select_cardinality_arms,
    stable_topk_flat_indices,
    validate_k_values,
    validate_legacy_10k_reproduction,
)


SPATIAL_SHAPE = (20, 25, 20)


def axes() -> KRadarAxes:
    return KRadarAxes(
        doppler_mps=np.linspace(-5.0, 5.0, 64),
        range_m=np.linspace(1.0, 100.0, SPATIAL_SHAPE[0]),
        azimuth_rad=np.linspace(-0.6, 0.6, SPATIAL_SHAPE[1]),
        elevation_rad=np.linspace(-0.2, 0.2, SPATIAL_SHAPE[2]),
    )


def logits() -> torch.Tensor:
    values = torch.linspace(-4.0, 4.0, np.prod(SPATIAL_SHAPE))
    return values.reshape(SPATIAL_SHAPE)


def test_stable_topk_is_repeatable_sorted_and_unique() -> None:
    values = torch.tensor(
        [
            [[2.0, 2.0], [1.0, 0.0]],
            [[2.0, -1.0], [0.5, 0.5]],
        ]
    )

    first = stable_topk_flat_indices(values, 5)
    second = stable_topk_flat_indices(values.clone(), 5)
    selected_confidence = torch.sigmoid(values).flatten()[first]

    assert torch.equal(first, second)
    assert torch.unique(first).numel() == 5
    assert bool(
        (selected_confidence[:-1] >= selected_confidence[1:]).all()
    )


def test_fixed_k_arms_are_unique_and_exact_10k_matches_legacy() -> None:
    output = select_cardinality_arms(logits(), axes())

    assert tuple(output) == tuple(f"fixed_k_{value}" for value in FIXED_K_VALUES)
    for count in FIXED_K_VALUES:
        arm = output[f"fixed_k_{count}"]
        assert arm["selected_count"] == count
        assert torch.unique(arm["indices_rae"], dim=0).shape[0] == count
    assert (
        output["fixed_k_2500"]["density_label"]
        == "reduced_cardinality_diagnostic_not_10k_dense"
    )
    assert (
        output["fixed_k_10000"]["density_label"]
        == "frozen_exact_10k_dense_output"
    )
    reproduction = validate_legacy_10k_reproduction(
        logits(),
        axes(),
        output["fixed_k_10000"],
    )
    assert reproduction["passed"] is True
    assert (
        reproduction["stable_output_sha256"]
        == reproduction["legacy_output_sha256"]
    )


@pytest.mark.parametrize(
    "values,cell_count",
    [
        ((2_500, 5_000, 10_000), 10_000),
        ((2_500, 5_000, 7_500, 10_000, 12_500), 20_000),
        ((5_000, 2_500, 7_500, 10_000), 10_000),
        ((2_500, 5_000, 7_500, 10_000), 9_999),
    ],
)
def test_k_range_rejects_incomplete_or_invalid_sweeps(
    values: tuple[int, ...],
    cell_count: int,
) -> None:
    with pytest.raises(ValueError):
        validate_k_values(values, cell_count)


def test_validation_selection_api_has_no_gt_input() -> None:
    parameters = inspect.signature(select_cardinality_arms).parameters

    assert not any("target" in name or name == "gt" for name in parameters)
    output = select_cardinality_arms(
        logits(),
        axes(),
        adaptive_rule=AdaptiveCountRule(
            confidence_threshold=0.5,
            fit_frame_count=76,
        ),
    )
    assert all(
        arm["validation_gt_used_for_selection"] is False
        for arm in output.values()
    )
    assert (
        output["adaptive_train_confidence"]["selection_source"]
        == "model_occupancy_logits_only"
    )


def test_corrected_far_semantics_keeps_frames_without_far_predictions() -> None:
    arm = {
        "selection_kind": "fixed_k",
        "selection_source": "model_occupancy_logits_only",
        "validation_gt_used_for_selection": False,
        "requested_k": 1,
        "selected_count": 1,
        "density_label": "unit_test",
        "xyz_m": torch.tensor([[20.0, 0.0, 0.0]]),
        "confidence": torch.tensor([0.8]),
        "indices_rae": torch.tensor([[0, 0, 0]]),
    }
    target = torch.tensor([[80.0, 0.0, 0.0, 1.0]])
    frames = []
    for index in range(23):
        evaluated = evaluate_cardinality_arm(arm, target)
        assert (
            evaluated["geometry"][
                "range_60_120m_completeness_mean_distance_m"
            ]
            == 60.0
        )
        assert (
            evaluated["range_coverage"]["range_60_120m"][
                "prediction_count"
            ]
            == 0
        )
        assert (
            evaluated["range_coverage"]["range_60_120m"][
                "completeness_mean_distance_m"
            ]
            == 60.0
        )
        frames.append(
            {
                "sequence": index,
                "radar_index": index,
                **evaluated,
            }
        )

    aggregate = aggregate_cardinality_frames(
        frames,
        expected_frame_count=23,
        expected_far_target_count=23,
    )

    assert (
        aggregate["geometry"][
            "range_60_120m_completeness_mean_distance_m"
        ]["sample_count"]
        == 23
    )
