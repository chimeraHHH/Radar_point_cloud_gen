from scripts.train_rald_wce_stage0 import (
    FORMAL_FAR_VALIDATION_COUNT,
    FORMAL_VALIDATION_COUNT,
    frozen_config,
    stage0_decision,
)


def synthetic_metrics() -> dict:
    return {
        "frame_count": FORMAL_VALIDATION_COUNT,
        "far_target_frame_count": FORMAL_FAR_VALIDATION_COUNT,
        "matched": {
            "chamfer_m": {"mean": 2.0},
            "completeness_mean_distance_m": {"median": 2.0},
            "outlier_fraction_2m": {"mean": 0.20},
            "range_60_120m_completeness_mean_distance_m": {"mean": 20.0},
        },
        "condition_intervention": {
            "wrong_minus_matched_chamfer_fraction": {"mean": 0.03},
            "matched_chamfer_better": {"mean": 0.80},
        },
        "exact_export": {
            "minimum_observed_pair_distance_m": 0.05,
            "all_matched_exports_exact_10000": True,
            "all_wrong_exports_exact_10000": True,
            "copy_padding_jitter_duplicate": False,
        },
    }


def test_formal_config_freezes_full_wide_query_and_xyz_only_scope() -> None:
    config = frozen_config(smoke=False)

    assert config.epochs == 20
    assert config.occupancy_query_count == 16_000
    assert config.positive_query_ratio == 0.0625
    assert sum(config.q0_range_quotas) == 500_000
    assert sum(config.q1_anchor_quotas) * config.q1_samples_per_anchor == 200_000
    assert sum(config.output_range_quotas) == 10_000
    assert config.minimum_export_distance_m == 0.05
    assert config.test_accessed is False
    assert config.doppler_head is False


def test_formal_decision_requires_all_scientific_and_structural_gates() -> None:
    decision = stage0_decision(
        synthetic_metrics(),
        peak_allocated_bytes=40 * 2**30,
        peak_reserved_bytes=50 * 2**30,
        formal=True,
    )

    assert decision["eligible_for_stage0_scientific_decision"] is True
    assert decision["promotion_passed"] is True
    assert all(decision["structural_checks"].values())
    assert all(decision["scientific_checks"].values())
    assert all(decision["count_checks"].values())
    assert decision["doppler_head_evaluated"] is False


def test_smoke_and_wrong_condition_failure_cannot_promote() -> None:
    metrics = synthetic_metrics()
    metrics["condition_intervention"][
        "wrong_minus_matched_chamfer_fraction"
    ]["mean"] = 0.0
    decision = stage0_decision(
        metrics,
        peak_allocated_bytes=10,
        peak_reserved_bytes=10,
        formal=False,
    )

    assert decision["eligible_for_stage0_scientific_decision"] is False
    assert decision["promotion_passed"] is False
    assert decision["scientific_checks"][
        "wrong_condition_degrades_chamfer_at_least_1pct"
    ] is False
    assert decision["smoke_only"] is True


def test_formal_decision_requires_exact_24_and_23_counts() -> None:
    metrics = synthetic_metrics()
    metrics["far_target_frame_count"] = FORMAL_FAR_VALIDATION_COUNT - 1
    decision = stage0_decision(
        metrics,
        peak_allocated_bytes=10,
        peak_reserved_bytes=10,
        formal=True,
    )

    assert decision["eligible_for_stage0_scientific_decision"] is False
    assert decision["promotion_passed"] is False
    assert decision["count_checks"]["exact_23_far_target_frames"] is False
