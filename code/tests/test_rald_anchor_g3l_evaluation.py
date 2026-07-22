import copy

from scripts.compare_rald_anchor_g3l import g3l1_decision, g3l2_decision
from scripts.eval_rald_anchor_g3l import (
    cross_scene_condition_indices,
    frame_seed,
)


def comparison(relative_upper: float = 0.0, relative_lower: float = 0.0) -> dict:
    return {
        "chamfer_m": {"relative_change_ci95": [relative_lower, relative_upper]},
        "local_spectrum_kl": {
            "relative_change_ci95": [relative_lower, relative_upper]
        },
        "circular_w1_mps": {
            "relative_change_ci95": [relative_lower, relative_upper]
        },
        "confidence_mean": {
            "relative_change_ci95": [relative_lower, relative_upper]
        },
        "covered_cell_count": {
            "relative_change_ci95": [relative_lower, relative_upper]
        },
    }


def posterior_diagnostic() -> dict:
    return {
        "all_finite": True,
        "variance_mean": 0.2,
        "across_frame_mean_std": 0.01,
        "doppler_intervention_latent_rms_mean": 0.01,
        "doppler_intervention_decoder_rms_mean": 0.01,
        "confidence_intervention_latent_rms_mean": 0.01,
        "confidence_intervention_decoder_rms_mean": 0.01,
    }


def test_frame_seed_is_identity_derived_and_deterministic() -> None:
    assert frame_seed(1, 2) == frame_seed(1, 2)
    assert frame_seed(1, 2) != frame_seed(1, 3)
    assert frame_seed(1, 2, 0) != frame_seed(1, 2, 1)
    assert 0 <= frame_seed(58, 9999) < 2**63


def test_condition_shuffle_is_a_cross_scene_derangement() -> None:
    records = [
        {"sequence": 1},
        {"sequence": 1},
        {"sequence": 2},
        {"sequence": 2},
        {"sequence": 3},
        {"sequence": 3},
    ]
    shuffled = cross_scene_condition_indices(records)
    assert sorted(shuffled) == list(range(len(records)))
    assert all(
        records[index]["sequence"] != records[other]["sequence"]
        for index, other in enumerate(shuffled)
    )


def test_g3l1_threshold_directions_are_frozen() -> None:
    runs = {seed: {"posterior": posterior_diagnostic()} for seed in (1, 2, 3)}
    passing = comparison(relative_upper=0.01, relative_lower=-0.05)
    assert g3l1_decision(passing, runs)["g3l1_passed"] is True

    chamfer_failure = copy.deepcopy(passing)
    chamfer_failure["chamfer_m"]["relative_change_ci95"][1] = 0.021
    assert g3l1_decision(chamfer_failure, runs)["g3l1_passed"] is False

    confidence_failure = copy.deepcopy(passing)
    confidence_failure["confidence_mean"]["relative_change_ci95"][0] = -0.101
    assert g3l1_decision(confidence_failure, runs)["g3l1_passed"] is False


def test_g3l2_requires_confident_condition_regression() -> None:
    retained = comparison(relative_upper=0.04, relative_lower=-0.05)
    no_effect = {
        "chamfer_m": {"improvement_ci95": [-0.1, 0.01]},
        "local_spectrum_kl": {"improvement_ci95": [-0.1, 0.02]},
    }
    assert g3l2_decision(retained, no_effect)["g3l2_passed"] is False

    confident_effect = copy.deepcopy(no_effect)
    confident_effect["local_spectrum_kl"]["improvement_ci95"] = [-0.2, -0.01]
    assert g3l2_decision(retained, confident_effect)["g3l2_passed"] is True

    geometry_failure = copy.deepcopy(retained)
    geometry_failure["chamfer_m"]["relative_change_ci95"][1] = 0.051
    assert g3l2_decision(geometry_failure, confident_effect)["g3l2_passed"] is False
