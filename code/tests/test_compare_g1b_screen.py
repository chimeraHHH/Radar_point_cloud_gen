from scripts.compare_g1b_screen import MODES, evaluate_screen


def run(mode: str, chamfer: float, outlier: float, far: float, fscore: float) -> dict:
    parameters = {
        "rae_max": 100_000,
        "rae_moments": 100_016,
        "rae_circular_harmonics": 100_032,
        "full_raed_rank2": 100_152,
    }
    return {
        "path": f"/runs/{mode}",
        "mode": mode,
        "config": {"mode": mode, "seed": 20260716, "epochs": 15},
        "provenance": {
            "git_commit": "a" * 40,
            "manifest_sha256": "manifest",
            "scene_split_sha256": "split",
            "normalization_sha256": "normalization",
            "device": "NVIDIA H200 NVL",
            "torch_version": "2.test",
            "model_parameter_count": parameters[mode],
        },
        "best_epoch": 15,
        "metrics": {
            "chamfer_m": {"median": chamfer},
            "outlier_fraction_2m": {"mean": outlier},
            "range_60_120m_completeness_mean_distance_m": {"mean": far},
            "range_60_120m_fscore_1m": {"mean": fscore},
        },
        "spectral_diagnostics": (
            {}
            if mode == "rae_max"
            else {
                "first_step_gradient_norm": 0.1,
                "spectral_branch_weight_rms": 0.2,
                "spectral_to_trunk_weight_rms_ratio": 0.5,
            }
        ),
    }


def test_screen_selects_best_survivor_by_preregistered_order() -> None:
    values = {
        "rae_max": (2.0, 0.24, 6.0, 0.10),
        "rae_moments": (2.01, 0.24, 5.8, 0.10),
        "rae_circular_harmonics": (1.95, 0.24, 5.9, 0.11),
        "full_raed_rank2": (1.95, 0.24, 5.7, 0.10),
    }
    report = evaluate_screen(
        [run(mode, *values[mode]) for mode in MODES], 20260716
    )

    assert report["survivors"] == list(MODES[1:])
    assert report["selected_candidate"] == "full_raed_rank2"
    assert report["stage_b_authorized"] is True


def test_screen_closes_stage_b_when_all_candidates_fail() -> None:
    values = {
        "rae_max": (2.0, 0.24, 6.0, 0.10),
        "rae_moments": (2.2, 0.24, 6.1, 0.09),
        "rae_circular_harmonics": (2.2, 0.24, 6.1, 0.09),
        "full_raed_rank2": (2.2, 0.26, 6.1, 0.09),
    }
    report = evaluate_screen(
        [run(mode, *values[mode]) for mode in MODES], 20260716
    )

    assert report["survivors"] == []
    assert report["selected_candidate"] is None
    assert report["stage_b_authorized"] is False
