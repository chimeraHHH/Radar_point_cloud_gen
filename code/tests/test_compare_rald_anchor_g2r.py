from scripts.compare_rald_anchor_g2r import gate_decision


def report(improvement_lower: float, relative_upper: float = -0.01) -> dict:
    return {
        "improvement_ci95": [improvement_lower, improvement_lower + 0.1],
        "relative_change_ci95": [-0.1, relative_upper],
    }


def test_g2r_gate_requires_distribution_and_direct_query_gains() -> None:
    head = {
        "spectrum_nll": report(0.02),
        "circular_w1_mps": report(0.01),
        "cd_doppler": report(-0.01),
        "geometry_chamfer_m": report(-0.01, relative_upper=0.01),
    }
    direct = {
        "spectrum_nll": report(0.01),
        "circular_w1_mps": report(-0.01),
        "soft_ece_10bin": report(-0.01),
        "cd_doppler": report(-0.01),
    }
    runs = {
        1: {
            "frames": {
                (1, 1): {
                    "cycle": {
                        "confidence_mean": 0.7,
                        "offset_saturation_fraction": 0.02,
                    }
                }
            }
        }
    }

    assert gate_decision(head, direct, runs)["g2r_passed"] is True
    direct["spectrum_nll"] = report(-0.01)
    assert gate_decision(head, direct, runs)["g2r_passed"] is False

