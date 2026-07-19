from scripts.compare_rald_anchor_rh2 import gate_decision


def report(improvement_lower: float, relative_upper: float = -0.01) -> dict:
    return {
        "improvement_ci95": [improvement_lower, improvement_lower + 0.1],
        "relative_change_ci95": [-0.1, relative_upper],
    }


def test_rh2_gate_requires_geometry_doppler_and_physical_stability() -> None:
    geometry = {
        "chamfer_m": report(0.01),
        "fscore_1p0m": report(-0.01),
    }
    doppler = {"spectrum_nll": report(0.02)}
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

    decision = gate_decision(geometry, doppler, runs)

    assert decision["rh2_passed"] is True
    doppler["spectrum_nll"] = report(-0.01)
    assert gate_decision(geometry, doppler, runs)["rh2_passed"] is False
