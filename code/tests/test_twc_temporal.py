from __future__ import annotations

import pytest
import torch

from losses.wrong_condition import WRONG_CONDITIONS, apply_wrong_condition
from models.cube_doppler import circular_mean
from models.twc_temporal import ForcedTemporalTWC


def tiny_case() -> tuple[
    ForcedTemporalTWC,
    dict[str, torch.Tensor],
    tuple[torch.Tensor, ...],
]:
    torch.manual_seed(41)
    range_m = torch.linspace(0.0, 60.0, 16)
    azimuth_rad = torch.linspace(-1.0, 1.0, 9)
    elevation_rad = torch.linspace(-0.3, 0.3, 5)
    doppler_mps = torch.linspace(-6.0, 6.0, 8)
    model = ForcedTemporalTWC(
        range_m,
        azimuth_rad,
        elevation_rad,
        doppler_mps,
        point_count=12,
        persistent_fraction=0.75,
        hidden_dim=32,
        maximum_tangential_speed_mps=5.0,
        maximum_radial_correction_mps=0.5,
    ).eval()
    batch_size, frame_count, history_count = 2, 3, 12
    point_index = torch.arange(history_count).float()
    base = torch.stack(
        (
            10.0 + point_index,
            -1.0 + 0.1 * point_index,
            torch.zeros_like(point_index),
        ),
        dim=1,
    )
    history_xyz_m = torch.stack(
        [
            torch.stack(
                [
                    base + torch.tensor([0.5 * frame, 0.2 * batch, 0.0])
                    for frame in range(frame_count)
                ]
            )
            for batch in range(batch_size)
        ]
    )
    doppler_index = (torch.arange(history_count) + 5) % doppler_mps.numel()
    probability = torch.nn.functional.one_hot(
        doppler_index, doppler_mps.numel()
    ).float()
    history_probability = probability.view(
        1, 1, history_count, -1
    ).expand(batch_size, frame_count, -1, -1).clone()
    history_probability[1] = torch.roll(
        history_probability[1], shifts=2, dims=-1
    )
    history_confidence = torch.linspace(
        0.5, 1.0, history_count
    ).view(1, 1, -1).expand(batch_size, frame_count, -1).clone()
    history_source_id = torch.arange(
        batch_size * frame_count * history_count
    ).reshape(batch_size, frame_count, history_count)
    current_from_history = torch.eye(4)[None].repeat(
        batch_size, 1, 1
    )
    current_from_history[:, 0, 3] = 0.25
    inputs = {
        "conditioning_cube_drae": torch.rand(
            batch_size,
            doppler_mps.numel(),
            range_m.numel(),
            azimuth_rad.numel(),
            elevation_rad.numel(),
        ),
        "history_xyz_m": history_xyz_m,
        "history_doppler_probability": history_probability,
        "history_confidence": history_confidence,
        "history_source_id": history_source_id,
        "current_from_history": current_from_history,
        "delta_seconds": torch.full((batch_size,), 0.1),
    }
    return model, inputs, (
        range_m,
        azimuth_rad,
        elevation_rad,
        doppler_mps,
    )


def test_forced_temporal_output_has_persistent_birth_contract() -> None:
    model, inputs, _ = tiny_case()

    with torch.inference_mode():
        output = model(
            **inputs,
            mode="online_enhancement",
        )

    assert output["xyz_m"].shape == (2, 12, 3)
    assert output["doppler_probability"].shape == (2, 12, 8)
    assert output["persistent_mask"][:, :9].all()
    assert not output["persistent_mask"][:, 9:].any()
    assert (
        output["history_source_id"][output["persistent_mask"]] >= 0
    ).all()
    assert (
        output["history_source_id"][~output["persistent_mask"]] == -1
    ).all()
    assert output["metadata"] == {
        "mode": "online_enhancement",
        "uses_future_cube": False,
        "conditioning_cube_role": "target_current_cube",
        "target_offset_steps": 0,
        "persistent_count": 9,
        "birth_count": 3,
        "persistent_head_requires_cube_history_correlation": True,
    }


def test_persistent_motion_is_tangential_plus_bounded_radial_correction() -> None:
    model, inputs, _ = tiny_case()

    with torch.inference_mode():
        output = model(
            **inputs,
            mode="online_enhancement",
        )

    selected = output["selected_history_xyz_m"]
    radial_direction = selected / torch.linalg.vector_norm(
        selected, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    radial_dot = (
        output["tangential_velocity_mps"] * radial_direction
    ).sum(dim=-1)
    torch.testing.assert_close(
        radial_dot, torch.zeros_like(radial_dot), atol=1e-5, rtol=0.0
    )
    assert (
        output["radial_correction_mps"].abs().max()
        <= model.maximum_radial_correction_mps
    )
    assert (
        output["analytic_radial_displacement_m"].abs() > 1e-6
    ).any()
    assert all("gate" not in name for name, _ in model.named_parameters())


def test_current_cube_history_correlation_changes_persistent_output() -> None:
    model, inputs, _ = tiny_case()
    changed_inputs = dict(inputs)
    changed_inputs["conditioning_cube_drae"] = torch.roll(
        inputs["conditioning_cube_drae"], shifts=3, dims=1
    )

    with torch.inference_mode():
        matched = model(**inputs, mode="online_enhancement")
        changed = model(**changed_inputs, mode="online_enhancement")

    assert not torch.allclose(
        matched["persistent_xyz_m"], changed["persistent_xyz_m"]
    )
    assert not torch.allclose(
        matched["conditioning_spectrum"], changed["conditioning_spectrum"]
    )


def test_forecast_metadata_forbids_future_cube() -> None:
    model, inputs, _ = tiny_case()

    with torch.inference_mode():
        output = model(
            **inputs,
            mode="forecast",
            target_offset_steps=5,
        )

    assert output["metadata"]["mode"] == "forecast"
    assert output["metadata"]["conditioning_cube_role"] == "last_observed_cube"
    assert output["metadata"]["target_offset_steps"] == 5
    assert output["metadata"]["uses_future_cube"] is False
    with pytest.raises(ValueError, match="never accepts a future"):
        model(
            **inputs,
            mode="forecast",
            target_offset_steps=1,
            future_cube_drae=inputs["conditioning_cube_drae"],
        )


def test_invalid_latest_history_source_cannot_fill_persistent_quota() -> None:
    model, inputs, _ = tiny_case()
    inputs["history_source_id"][:, -1, :4] = -1

    with pytest.raises(ValueError, match="persistent quota"):
        model(**inputs, mode="online_enhancement")


def test_axes_must_be_strictly_increasing() -> None:
    _, _, axes = tiny_case()
    range_m, azimuth_rad, elevation_rad, doppler_mps = axes
    invalid_range = range_m.clone()
    invalid_range[3] = invalid_range[2]

    with pytest.raises(ValueError, match="range axis"):
        ForcedTemporalTWC(
            invalid_range,
            azimuth_rad,
            elevation_rad,
            doppler_mps,
            point_count=12,
            hidden_dim=32,
        )


@pytest.mark.parametrize("condition", WRONG_CONDITIONS)
def test_wrong_conditions_modify_the_causal_history(condition: str) -> None:
    _, inputs, axes = tiny_case()

    wrong = apply_wrong_condition(
        history_xyz_m=inputs["history_xyz_m"],
        history_doppler_probability=(
            inputs["history_doppler_probability"]
        ),
        history_confidence=inputs["history_confidence"],
        history_source_id=inputs["history_source_id"],
        condition=condition,
        doppler_mps=axes[-1],
    )

    if condition in ("history_replacement", "time_reversal"):
        assert not torch.equal(
            wrong["history_source_id"], inputs["history_source_id"]
        )
    else:
        assert not torch.equal(
            wrong["history_doppler_probability"],
            inputs["history_doppler_probability"],
        )


def test_doppler_zero_intervention_has_zero_circular_mean() -> None:
    _, inputs, axes = tiny_case()
    doppler_mps = axes[-1]
    step = torch.median(torch.diff(doppler_mps))
    period = step * doppler_mps.numel()

    wrong = apply_wrong_condition(
        history_xyz_m=inputs["history_xyz_m"],
        history_doppler_probability=(
            inputs["history_doppler_probability"]
        ),
        history_confidence=inputs["history_confidence"],
        history_source_id=inputs["history_source_id"],
        condition="doppler_zero",
        doppler_mps=doppler_mps,
    )
    mean = circular_mean(
        wrong["history_doppler_probability"],
        doppler_mps,
        doppler_mps[0],
        period,
    )

    torch.testing.assert_close(
        mean, torch.zeros_like(mean), atol=1e-5, rtol=0.0
    )
