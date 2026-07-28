from __future__ import annotations

import torch

from losses.twc_physics import (
    compose_persistent_motion,
    inverse_radial_loss,
    mandatory_radial_doppler_warp,
    motion_cycle_loss,
)
from losses.wrong_condition import wrong_condition_margin_loss


def doppler_axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    doppler_mps = torch.linspace(-8.0, 8.0, 64)
    step = torch.median(torch.diff(doppler_mps))
    return doppler_mps, doppler_mps[0], step * doppler_mps.numel()


def test_mandatory_radial_warp_applies_doppler_then_ego_transform() -> None:
    doppler_mps, lower, period = doppler_axes()
    selected_bin = torch.argmin((doppler_mps - 4.0).abs())
    probability = torch.nn.functional.one_hot(
        selected_bin.reshape(1, 1), doppler_mps.numel()
    ).float()
    xyz = torch.tensor([[[10.0, 0.0, 0.0]]])
    transform = torch.eye(4)[None]
    transform[:, 0, 3] = 2.0

    warp = mandatory_radial_doppler_warp(
        xyz,
        probability,
        transform,
        0.5,
        doppler_mps,
        lower,
        period,
    )

    expected_displacement = doppler_mps[selected_bin] * 0.5
    torch.testing.assert_close(
        warp.radial_displacement_m,
        expected_displacement.reshape(1, 1),
    )
    torch.testing.assert_close(
        warp.xyz_m[:, :, 0],
        (12.0 + expected_displacement).reshape(1, 1),
    )


def test_composed_motion_is_tangential_and_radially_bounded() -> None:
    doppler_mps, lower, period = doppler_axes()
    probability = torch.nn.functional.one_hot(
        torch.tensor([[40, 42]]), doppler_mps.numel()
    ).float()
    xyz = torch.tensor([[[10.0, 0.0, 0.0], [12.0, 2.0, 0.0]]])
    transform = torch.eye(4)[None]
    warp = mandatory_radial_doppler_warp(
        xyz,
        probability,
        transform,
        0.1,
        doppler_mps,
        lower,
        period,
    )

    motion = compose_persistent_motion(
        warp,
        torch.tensor([[[3.0, 2.0, 1.0], [1.0, -4.0, 2.0]]]),
        torch.tensor([[20.0, -20.0]]),
        transform,
        0.1,
        maximum_tangential_speed_mps=5.0,
        maximum_radial_correction_mps=0.5,
    )

    radial_dot = (
        motion.tangential_velocity_mps * warp.radial_direction
    ).sum(dim=-1)
    torch.testing.assert_close(
        radial_dot, torch.zeros_like(radial_dot), atol=1e-5, rtol=0.0
    )
    assert motion.radial_correction_mps.abs().max() <= 0.5


def test_inverse_radial_and_cycle_losses_are_zero_for_exact_motion() -> None:
    doppler_mps, lower, period = doppler_axes()
    selected_bin = torch.argmin((doppler_mps - 3.0).abs())
    probability = torch.nn.functional.one_hot(
        selected_bin.reshape(1, 1), doppler_mps.numel()
    ).float()
    xyz = torch.tensor([[[10.0, 0.0, 0.0]]])
    transform = torch.eye(4)[None]
    delta = 0.2
    warp = mandatory_radial_doppler_warp(
        xyz,
        probability,
        transform,
        delta,
        doppler_mps,
        lower,
        period,
    )
    tangential = torch.zeros_like(xyz)
    radial_correction = torch.zeros(1, 1)

    inverse = inverse_radial_loss(
        xyz,
        warp.xyz_m,
        transform,
        delta,
        probability,
        doppler_mps,
        lower,
        period,
    )
    cycle = motion_cycle_loss(
        xyz,
        warp.xyz_m,
        transform,
        delta,
        warp.radial_velocity_mps,
        tangential,
        radial_correction,
    )

    torch.testing.assert_close(inverse, torch.tensor(0.0), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(cycle, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_wrong_condition_margin_orders_physical_discrepancy() -> None:
    matched = torch.tensor([0.10, 0.20])
    adequately_wrong = torch.tensor([0.40, 0.50])
    insufficiently_wrong = torch.tensor([0.15, 0.22])

    passed = wrong_condition_margin_loss(
        matched, adequately_wrong, margin=0.20
    )
    failed = wrong_condition_margin_loss(
        matched, insufficiently_wrong, margin=0.20
    )

    torch.testing.assert_close(passed.total, torch.tensor(0.0))
    assert failed.total > 0.0
