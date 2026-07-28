from __future__ import annotations

import json

import pytest
import torch

from dmhw.contracts import (
    INTERVENTION_NAMES,
    apply_dmhw_intervention,
    audit_direct_horizon_supervision,
    build_direct_horizon_examples,
    clone_float_inputs,
)
from dmhw.synthetic_contract import make_dmhw_synthetic_contract
from losses.dmhw_objective import dmhw_stage0_loss
from models.dmhw_direct_world import DirectMultiHorizonDopplerWorld


def make_case(
    *,
    point_count: int = 32,
) -> tuple[
    DirectMultiHorizonDopplerWorld,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    inputs, targets, axes = make_dmhw_synthetic_contract(
        torch.device("cpu"),
        point_count=point_count,
        persistent_fraction=0.625,
        full_raed=False,
        seed=11,
    )
    model = DirectMultiHorizonDopplerWorld(
        *axes,
        point_count=point_count,
        persistent_fraction=0.625,
        hidden_dim=32,
        maximum_tangential_speed_mps=3.0,
        maximum_radial_correction_mps=0.4,
    )
    return model, inputs, targets


def test_direct_output_is_exact_count_and_identity_preserving() -> None:
    model, inputs, _ = make_case()
    output = model(**inputs)

    assert output["xyz_m"].shape == (2, 3, 32, 3)
    assert output["doppler_probability"].shape == (2, 3, 32, 64)
    assert output["confidence"].shape == (2, 3, 32)
    assert output["source_id"].shape == (2, 3, 32)
    assert output["persistent_mask"].shape == (2, 3, 32)
    assert torch.equal(
        output["persistent_source_id"][:, 0],
        output["persistent_source_id"][:, 1],
    )
    assert torch.equal(
        output["persistent_source_id"][:, 1],
        output["persistent_source_id"][:, 2],
    )
    assert torch.all(output["source_id"][:, :, 20:] == -1)
    assert output["metadata"]["direct_single_pass"] is True
    assert output["metadata"]["autoregressive_rollout"] is False
    assert output["metadata"]["uses_future_cube"] is False


def test_future_cube_and_noncausal_history_are_rejected() -> None:
    model, inputs, _ = make_case()
    with pytest.raises(ValueError, match="never reads a future Cube"):
        model(**inputs, future_cube_drae=inputs["current_cube_drae"])

    noncausal = dict(inputs)
    noncausal["history_time_seconds"] = torch.tensor(
        [[-0.3, 0.0, 0.1], [-0.3, 0.0, 0.1]]
    )
    with pytest.raises(ValueError, match="precede current time"):
        model(**noncausal)


def test_persistent_motion_is_forced_and_bounded() -> None:
    model, inputs, _ = make_case()
    output = model(**inputs)

    tangent_radial_dot = (
        output["tangential_velocity_mps"]
        * output["radial_direction"]
    ).sum(dim=-1)
    assert float(tangent_radial_dot.abs().max().detach()) < 1e-5
    tangent_norm = torch.linalg.vector_norm(
        output["tangential_velocity_mps"], dim=-1
    )
    assert float(tangent_norm.max().detach()) <= 3.0 + 1e-5
    assert (
        float(output["radial_correction_mps"].abs().max().detach())
        <= 0.4 + 1e-5
    )
    expected_displacement = (
        output["analytic_radial_velocity_mps"]
        * output["horizons_seconds"][None, :, None]
    )
    torch.testing.assert_close(
        output["analytic_radial_displacement_m"],
        expected_displacement,
    )


def test_five_interventions_are_measurable_and_loss_backpropagates() -> None:
    model, inputs, targets = make_case()
    baseline = model(**inputs)
    for name in INTERVENTION_NAMES:
        intervention = apply_dmhw_intervention(
            inputs, name, model.doppler_mps
        )
        changed = model(**intervention)
        per_horizon = (
            changed["xyz_m"] - baseline["xyz_m"]
        ).abs().mean(dim=(0, 2, 3))
        assert torch.all(per_horizon > 1e-7), name

    differentiable = clone_float_inputs(inputs, requires_grad=True)
    output = model(**differentiable)
    objective = dmhw_stage0_loss(
        output,
        **targets,
        aligned_synthetic_attributes=True,
    )
    objective.total.backward()
    assert torch.isfinite(objective.total)
    assert differentiable["current_cube_drae"].grad is not None
    assert differentiable["history_cube_drae"].grad is not None
    current_channel_gradient = differentiable[
        "current_cube_drae"
    ].grad.abs().sum(dim=(0, 2, 3, 4))
    history_channel_gradient = differentiable[
        "history_cube_drae"
    ].grad.abs().sum(dim=(0, 1, 3, 4, 5))
    assert int((current_channel_gradient > 0).sum()) == 64
    assert int((history_channel_gradient > 0).sum()) == 64
    assert differentiable["observed_xyz_m"].grad is not None
    assert float(differentiable["observed_xyz_m"].grad.abs().sum()) > 0.0


def test_real_objective_refuses_point_aligned_attribute_claims() -> None:
    model, inputs, targets = make_case()
    output = model(**inputs)
    with pytest.raises(ValueError, match="synthetic-preflight only"):
        dmhw_stage0_loss(
            output,
            target_xyz_m=targets["target_xyz_m"],
            target_doppler_probability=targets[
                "target_doppler_probability"
            ],
        )


def synthetic_manifest() -> dict:
    frames = []
    for index in range(48):
        transform = torch.eye(4, dtype=torch.float64)
        transform[0, 3] = 0.1
        frames.append(
            {
                "sequence": 1,
                "partition": "train",
                "window_id": "seq01_w00",
                "frame_in_window": index,
                "radar_index": 100 + index,
                "lidar64_index": 200 + index,
                "timestamp": 0.1 * index,
                "current_radar_from_previous_radar": (
                    transform.reshape(-1).tolist()
                ),
            }
        )
    return {
        "gate_pass": True,
        "checks": {"radar_frame_ego_transforms_present": True},
        "frames": frames,
    }


def test_manifest_contract_builds_strict_three_horizon_supervision(
    tmp_path,
) -> None:
    manifest = synthetic_manifest()
    examples = build_direct_horizon_examples(manifest)
    assert len(examples) == 20
    assert all(example["future_cube_exposed"] is False for example in examples)
    first = examples[0]
    assert [target["horizon_seconds"] for target in first["targets"]] == [
        0.5,
        1.5,
        2.5,
    ]
    translations = [
        target["target_from_current"][3] for target in first["targets"]
    ]
    assert translations == pytest.approx([0.5, 1.5, 2.5])

    path = tmp_path / "dmhw_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    audit = audit_direct_horizon_supervision(path)
    assert audit["example_count_by_partition"] == {"train": 20}
    assert audit["future_cube_exposed"] is False
    assert audit["test_accessed"] is False
