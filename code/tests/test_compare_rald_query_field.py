from copy import deepcopy

from scripts.compare_rald_query_field import (
    FORMAL_CONFIG,
    frame_count_checks,
    gradient_checks,
    stage_a_decision,
    stage_b_decision,
)


def gradients() -> list[dict]:
    return [
        {
            "update": 1,
            "gradients": {
                "output_heads": 1.0,
            },
        },
        {
            "update": 2,
            "gradients": {
                "output_heads": 1.0,
                "mixed_latent": 1.0,
                "query_decoder": 1.0,
                "full_raed_radar_encoder": 1.0,
                "cube_input_channel_norms": [1.0] * 64,
                "local_spectrum_input_column_norms": [1.0] * 64,
                "radar_projection_input_column_norms": [1.0] * 64,
                "condition_block_gradient_norms": [1.0] * 24,
            },
        },
    ]


def frame(sequence: int = 1, radar_index: int = 1) -> dict:
    return {
        "sequence": sequence,
        "radar_index": radar_index,
        "generated": {
            "prediction_count": 10_000,
            "chamfer_m": 2.0,
            "outlier_fraction_2m": 0.20,
            "completeness_mean_distance_m": 0.50,
        },
        "zero_offset_control": {
            "chamfer_m": 2.2,
            "outlier_fraction_2m": 0.22,
            "completeness_mean_distance_m": 0.55,
        },
        "duplicates": {"duplicate_fraction_0p05m": 0.05},
        "occupancy": {
            "positive_recall": 0.85,
            "empty_false_positive_rate": 0.10,
        },
        "radar_token_count": 336,
        "coarse_query_count": 32_000,
        "selected_coarse_count": 2_500,
        "final_point_count": 10_000,
        "occupancy_query_count": 10_000,
        "positive_occupancy_query_count": 625,
        "empty_occupancy_query_count": 9_375,
    }


def run(seed: int = 20260716) -> dict:
    one_frame = frame()
    return {
        "config": {**FORMAL_CONFIG, "seed": seed},
        "metrics": {
            "gradient_steps": gradients(),
            "validation": {
                "generated": {
                    "chamfer_m": {"mean": 2.0, "median": 2.0},
                    "outlier_fraction_2m": {"mean": 0.20, "median": 0.20},
                    "completeness_mean_distance_m": {
                        "mean": 0.50,
                        "median": 0.50,
                    },
                    "range_60_120m_completeness_mean_distance_m": {
                        "mean": 5.0,
                        "median": 5.0,
                    },
                },
                "zero_offset_control": {
                    "chamfer_m": {"mean": 2.2, "median": 2.2},
                    "outlier_fraction_2m": {"mean": 0.22, "median": 0.22},
                },
                "duplicates": {
                    "duplicate_fraction_0p05m": {
                        "mean": 0.05,
                        "median": 0.05,
                    }
                },
                "confidence_mean": {"mean": 0.50, "median": 0.50},
                "occupancy": {
                    "positive_recall": {"mean": 0.85, "median": 0.85},
                    "empty_false_positive_rate": {
                        "mean": 0.10,
                        "median": 0.10,
                    },
                },
                "condition_shuffle": {
                    "occupancy_bce_fraction": {"mean": 0.02, "median": 0.02},
                    "chamfer_fraction": {"mean": 0.0, "median": 0.0},
                },
            },
        },
        "frames": {(one_frame["sequence"], one_frame["radar_index"]): one_frame},
    }


def test_stage_a_accepts_complete_rald_query_field_contract() -> None:
    decision = stage_a_decision(run())

    assert decision["passed"] is True
    assert all(decision["checks"].values())


def test_stage_a_rejects_missing_condition_block_and_query_count() -> None:
    candidate = run()
    candidate["metrics"]["gradient_steps"][1]["gradients"][
        "condition_block_gradient_norms"
    ] = [1.0] * 23
    next(iter(candidate["frames"].values()))[
        "positive_occupancy_query_count"
    ] = 624

    decision = stage_a_decision(candidate)

    assert decision["passed"] is False
    assert decision["checks"]["second_step_all_24_condition_blocks"] is False
    assert decision["checks"]["all_frames_positive_queries_625"] is False
    assert gradient_checks(candidate)["second_step_all_24_condition_blocks"] is False
    assert frame_count_checks(candidate)["all_frames_positive_queries_625"] is False


def test_stage_b_requires_paired_improvement_over_zero_offset_control() -> None:
    runs = {
        seed: run(seed)
        for seed in (20260716, 20260717, 20260718)
    }
    statistics, decision = stage_b_decision(
        runs,
        bootstrap_samples=50,
        bootstrap_seed=7,
    )

    assert decision["passed"] is True
    assert statistics["paired_control"]["chamfer_m"]["ci95"][1] < 0.0
    assert statistics["paired_control"]["outlier_fraction_2m"]["ci95"][1] < 0.0

    regressed = deepcopy(runs)
    for candidate in regressed.values():
        one_frame = next(iter(candidate["frames"].values()))
        one_frame["zero_offset_control"]["chamfer_m"] = 1.9
        one_frame["zero_offset_control"]["outlier_fraction_2m"] = 0.19
    _, failed = stage_b_decision(
        regressed,
        bootstrap_samples=50,
        bootstrap_seed=7,
    )

    assert failed["passed"] is False
    assert failed["checks"]["paired_control_both_upper_nonpositive"] is False
