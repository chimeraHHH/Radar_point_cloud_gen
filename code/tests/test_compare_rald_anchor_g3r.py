from scripts.compare_rald_anchor_g3r import gate_decision


def report(improvement_lower: float, relative_lower: float = -0.05) -> dict:
    return {
        "improvement_ci95": [improvement_lower, improvement_lower + 0.1],
        "relative_change_ci95": [relative_lower, 0.01],
    }


def test_g3r_gate_requires_two_metric_classes_and_anti_collapse() -> None:
    primary = {
        "local_spectrum_kl": report(0.02),
        "spectrum_nll": report(0.01),
        "circular_w1_mps": report(-0.01),
        "cd_doppler": report(-0.01),
        "geometry_chamfer_m": report(-0.01),
        "geometry_fscore_1m": report(-0.01),
        "confidence_mean": report(-0.01),
        "covered_cell_count": report(-0.01),
        "confidence_ece": report(-0.01),
    }
    runs = {
        1: {
            "frames": {
                (1, 1): {"cycle": {"offset_saturation_fraction": 0.02}}
            }
        }
    }

    decision = gate_decision(primary, runs)

    assert decision["g3r_statistical_gate_passed"] is True
    primary["spectrum_nll"] = report(-0.01)
    assert gate_decision(primary, runs)["g3r_statistical_gate_passed"] is False

