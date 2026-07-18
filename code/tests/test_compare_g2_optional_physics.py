from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from scripts.compare_g2_cube_doppler import g2_decision  # noqa: E402


def test_distribution_can_pass_when_physics_is_not_evaluated() -> None:
    decision = g2_decision(
        {
            "distribution_spectrum_nll_gain": True,
            "distribution_secondary_gain": True,
        },
        physics_evaluated=False,
    )

    assert decision == {
        "distribution_passed": True,
        "physics_passed": None,
        "g2_passed": True,
    }


def test_physics_checks_remain_required_when_e5_is_evaluated() -> None:
    decision = g2_decision(
        {
            "distribution_spectrum_nll_gain": True,
            "distribution_secondary_gain": True,
            "physics_static_pce_gain": False,
            "physics_secondary_gain": True,
            "geometry_chamfer_nondegradation": True,
            "dynamic_fraction_not_collapsed": True,
            "counterfactual_convention_response": True,
        },
        physics_evaluated=True,
    )

    assert decision["distribution_passed"] is True
    assert decision["physics_passed"] is False
    assert decision["g2_passed"] is False
