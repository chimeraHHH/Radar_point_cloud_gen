from pathlib import Path

import pytest
import torch

from scripts.train_g1g_hierarchy import (
    FORMAL_EPOCHS,
    FORMAL_EVAL_EVERY,
    FROZEN_NORMALIZATION_SHA256,
    FORMAL_SEED,
    STAGE0_COMPLETENESS_LIMIT_M,
    architecture_anti_bypass_checks,
    artifact_document,
    cross_scene_condition_indices,
    dynamic_anti_bypass_checks,
    frozen_config,
    hierarchy_control_point_sets,
    repeated_control_duplicate_report,
    smoke_cross_scene_indices,
    stage0_decision,
    validate_data_contract,
)


def passing_metrics() -> dict:
    return {
        "condition_shuffle": {
            "chamfer_fraction": {"mean": 0.01},
        },
        "duplicates": {
            "duplicate_fraction_0p05m": {"mean": 0.15},
        },
        "generated": {
            "chamfer_m": {"median": 2.0},
            "completeness_mean_distance_m": {
                "median": STAGE0_COMPLETENESS_LIMIT_M,
            },
            "outlier_fraction_2m": {"mean": 0.25},
            "range_60_120m_completeness_mean_distance_m": {
                "mean": 8.1239,
            },
        },
        "center_structure": {
            "center_unique_fraction_0p05m": {"mean": 0.80},
        },
    }


def frozen_metadata() -> dict:
    return {
        "global_radar_token_count": 336,
        "center_query_count": 2_500,
        "children_per_center": 4,
        "final_point_count": 10_000,
        "pre_allocation_sources": [
            "learned_center_queries",
            "normalized_center_templates",
            "global_full_raed_tokens",
        ],
        "forbidden_pre_allocation_sources": [
            "local_cube_spectrum",
            "local_cube_energy",
            "local_cube_neighborhood",
            "proposal_score",
        ],
        "local_cube_sampling_stage": "after_center_allocation",
        "stage_sequence": [
            "encode_global_full_raed_tokens",
            "allocate_2500_centers",
            "sample_local_center_spectrum",
            "decode_four_bounded_children",
            "sample_final_point_spectrum",
        ],
    }


def test_formal_and_smoke_configs_are_nonoverridable_protocol_constants() -> None:
    formal = frozen_config(smoke=False)
    smoke = frozen_config(smoke=True)

    assert formal.seed == FORMAL_SEED == 20260716
    assert formal.epochs == FORMAL_EPOCHS == 20
    assert formal.eval_every == FORMAL_EVAL_EVERY == 5
    assert formal.train_limit is None
    assert formal.validation_limit is None
    assert formal.point_count == 10_000
    assert formal.center_count == 2_500
    assert formal.children_per_center == 4
    assert smoke.seed == formal.seed
    assert smoke.epochs == 1
    assert smoke.eval_every == 1
    assert smoke.train_limit == 2
    assert smoke.validation_limit == 2
    assert FROZEN_NORMALIZATION_SHA256 == (
        "4d0bca7d027a1a9f457c526f21a034a406ce4973e2dccee55ddd2d7019b41b77"
    )


def test_data_contract_requires_exact_76_24_manifest_and_no_test() -> None:
    frames = [
        {"partition": "train", "sequence": index, "radar_index": index}
        for index in range(76)
    ]
    frames.extend(
        {
            "partition": "validation",
            "sequence": 1_000 + index,
            "radar_index": index,
        }
        for index in range(24)
    )
    counts = validate_data_contract(
        {"frames": frames},
        {"gate_pass": True},
    )
    assert counts == {
        "manifest_frame_count": 100,
        "train_frame_count": 76,
        "validation_frame_count": 24,
        "test_frame_count": 0,
    }

    invalid = [*frames, {"partition": "test", "sequence": 9_999, "radar_index": 1}]
    with pytest.raises(ValueError, match="76/24"):
        validate_data_contract({"frames": invalid}, {"gate_pass": True})
    with pytest.raises(ValueError, match="leakage"):
        validate_data_contract({"frames": frames}, {"gate_pass": False})


def test_cross_scene_shuffle_is_deterministic_and_never_same_scene() -> None:
    records = [
        {"sequence": 1, "radar_index": 10},
        {"sequence": 2, "radar_index": 20},
        {"sequence": 3, "radar_index": 30},
        {"sequence": 4, "radar_index": 40},
    ]
    first = cross_scene_condition_indices(records)
    second = cross_scene_condition_indices(records)

    assert first == second
    assert sorted(first) == list(range(len(records)))
    assert all(
        records[index]["sequence"] != records[other]["sequence"]
        for index, other in enumerate(first)
    )
    smoke = smoke_cross_scene_indices(
        [
            {"sequence": 1, "radar_index": 10},
            {"sequence": 1, "radar_index": 11},
            {"sequence": 2, "radar_index": 20},
        ]
    )
    assert smoke == [0, 2]


def test_anti_bypass_contract_rejects_each_local_preallocation_source() -> None:
    checks = architecture_anti_bypass_checks(frozen_metadata())
    assert all(checks.values())

    for forbidden in (
        "local_cube_spectrum",
        "local_cube_energy",
        "local_cube_neighborhood",
        "proposal_score",
    ):
        metadata = frozen_metadata()
        metadata["pre_allocation_sources"] = [
            *metadata["pre_allocation_sources"],
            forbidden,
        ]
        checks = architecture_anti_bypass_checks(metadata)
        assert checks["pre_allocation_allowlist_exact"] is False


def test_dynamic_anti_bypass_requires_two_complete_gradient_audits() -> None:
    gradients = {
        "measured_cube_allocation_gradient_is_none": True,
        "condition_cube_allocation_gradient_finite_nonzero": True,
        "radar_encoder": 1.0,
        "allocation_layer_radar_attention": [1.0] * 6,
        "post_allocation_local_spectrum_projection": 1.0,
    }
    audit = dynamic_anti_bypass_checks(
        [
            {"gradients": dict(gradients)},
            {"gradients": dict(gradients)},
        ]
    )
    assert audit["passed"] is True

    failed_gradients = dict(gradients)
    failed_gradients["measured_cube_allocation_gradient_is_none"] = False
    failed = dynamic_anti_bypass_checks(
        [
            {"gradients": gradients},
            {"gradients": failed_gradients},
        ]
    )
    assert failed["passed"] is False


def test_stage0_decision_uses_exact_inclusive_frozen_boundaries() -> None:
    decision = stage0_decision(passing_metrics())
    assert all(decision["promotion_checks"].values())
    assert all(decision["abandonment_checks"].values())
    assert decision["promotion_passed"] is True
    assert decision["abandonment_triggered"] is False
    assert decision["passed"] is True
    assert decision["fixed_controls"]["g1d_epoch15_completeness_m"] == 3.5811
    assert decision["fixed_controls"]["g1d_epoch15_far_completeness_m"] == 8.1239


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("condition_shuffle", "chamfer_fraction", "mean"), 0.009999),
        (("duplicates", "duplicate_fraction_0p05m", "mean"), 0.150001),
        (
            ("generated", "completeness_mean_distance_m", "median"),
            STAGE0_COMPLETENESS_LIMIT_M + 1e-6,
        ),
        (("generated", "outlier_fraction_2m", "mean"), 0.250001),
        (
            (
                "generated",
                "range_60_120m_completeness_mean_distance_m",
                "mean",
            ),
            8.123901,
        ),
    ],
)
def test_stage0_decision_fails_any_promotion_gate(
    path: tuple[str, str, str],
    value: float,
) -> None:
    metrics = passing_metrics()
    metrics[path[0]][path[1]][path[2]] = value
    decision = stage0_decision(metrics)
    assert decision["promotion_passed"] is False
    assert decision["passed"] is False


def test_stage0_center_collapse_is_an_explicit_abandonment() -> None:
    metrics = passing_metrics()
    metrics["center_structure"]["center_unique_fraction_0p05m"]["mean"] = 0.799999
    decision = stage0_decision(metrics)
    assert decision["promotion_passed"] is True
    assert decision["abandonment_triggered"] is True
    assert decision["passed"] is False


def test_smoke_decision_fails_closed_when_no_far_slice_exists() -> None:
    metrics = passing_metrics()
    del metrics["generated"][
        "range_60_120m_completeness_mean_distance_m"
    ]
    decision = stage0_decision(metrics)
    assert decision["values"]["far_completeness_60_120m_mean"] == 1_000_000.0
    assert decision["promotion_checks"][
        "far_completeness_no_worse_than_g1d_epoch15"
    ] is False
    assert decision["passed"] is False


def test_zero_local_and_child_collapse_controls_are_distinct_fixed_sets() -> None:
    center_xyz = torch.arange(7_500, dtype=torch.float32).reshape(1, 2_500, 3)
    offsets = torch.tensor(
        [
            [-0.2, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]
    )
    child_xyz = center_xyz[:, :, None, :] + offsets[None, None]
    controls = hierarchy_control_point_sets(
        {
            "center_xyz_m": center_xyz,
            "child_xyz_m": child_xyz,
        }
    )

    assert controls["zero_local_refinement"].shape == (10_000, 3)
    assert controls["child_collapse"].shape == (10_000, 3)
    assert not torch.equal(
        controls["zero_local_refinement"],
        controls["child_collapse"],
    )
    zero_groups = controls["zero_local_refinement"].reshape(2_500, 4, 3)
    collapsed_groups = controls["child_collapse"].reshape(2_500, 4, 3)
    assert torch.all(zero_groups == zero_groups[:, :1])
    assert torch.all(collapsed_groups == collapsed_groups[:, :1])
    assert repeated_control_duplicate_report(10_000)[
        "duplicate_fraction_0p05m"
    ] == 1.0


def test_every_artifact_envelope_carries_hashes_counts_and_anti_bypass() -> None:
    config = frozen_config(smoke=True)
    provenance = {
        "git_commit": "a" * 40,
        "source_hashes": {"/source.py": {"sha256": "b" * 64}},
        "data_hashes": {
            "manifest": {"path": "/manifest.json", "sha256": "c" * 64}
        },
        "test_accessed": False,
    }
    counts = {
        "train_frame_count": 76,
        "validation_frame_count": 24,
        "used_train_frame_count": 2,
        "used_validation_frame_count": 2,
        "point_count_per_frame": 10_000,
    }
    anti_bypass = {
        "static_checks": architecture_anti_bypass_checks(frozen_metadata()),
        "static_passed": True,
    }
    document = artifact_document(
        "unit_test",
        config=config,
        provenance=provenance,
        exact_counts=counts,
        anti_bypass=anti_bypass,
        payload={"path": str(Path("/tmp/output"))},
    )

    assert document["provenance"]["source_hashes"]
    assert document["provenance"]["data_hashes"]
    assert document["exact_counts"] == counts
    assert all(document["anti_bypass"]["static_checks"].values())
    assert document["config"]["epochs"] == 1
    assert document["config"]["test_accessed"] is False
