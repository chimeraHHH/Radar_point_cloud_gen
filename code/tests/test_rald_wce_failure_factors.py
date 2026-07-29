import json

import pytest
import torch

from eval.rald_wce_failure_factors import (
    WCECandidatePool,
    _average_ranks,
    diagnosis_decision,
    diagnostic_artifact_label,
    ranking_alignment_diagnostics,
    validate_candidate_target_inputs,
    validate_current_confidence_control,
    validation_gt_nearest_score_export,
)
from eval.rald_wce_stage0 import ExactWCEExport, tensor_sha256
from scripts.diagnose_rald_wce_failure_factors import (
    _json_normalize,
    validation_records,
)


def _point_grid(
    count_xyz: tuple[int, int, int],
    origin: tuple[float, float, float],
) -> torch.Tensor:
    x = origin[0] + 0.06 * torch.arange(count_xyz[0])
    y = origin[1] + 0.06 * torch.arange(count_xyz[1])
    z = origin[2] + 0.06 * torch.arange(count_xyz[2])
    return torch.cartesian_prod(x, y, z)


def exact_candidate_set() -> torch.Tensor:
    near = _point_grid((20, 20, 20), (10.0, -0.6, -0.6))
    middle = _point_grid((17, 10, 10), (40.0, -0.3, -0.3))
    far = _point_grid((3, 10, 10), (70.0, -0.3, -0.3))
    return torch.cat((near, middle, far), dim=0)


def test_current_confidence_control_requires_formal_bit_exact_hashes() -> None:
    xyz = torch.tensor([[1.0, 0.0, 0.0]])
    confidence = torch.tensor([0.75])
    rows = torch.tensor([0])
    export = ExactWCEExport(
        xyz_m=xyz,
        confidence=confidence,
        selected_candidate_rows=rows,
        report={},
        hashes={
            "xyz_sha256": tensor_sha256(xyz),
            "confidence_sha256": tensor_sha256(confidence),
            "selected_candidate_rows_sha256": tensor_sha256(rows),
        },
    )
    query_hashes = {
        "q0_normalized_rae_sha256": "q0",
        "q1_normalized_rae_sha256": "q1",
    }
    pool = WCECandidatePool(
        xyz_m=xyz,
        confidence=confidence,
        current_confidence_export=export,
        query_hashes=query_hashes,
        report={},
    )
    reference = {
        "matched_hashes": dict(export.hashes),
        "inference": {"query_hashes": dict(query_hashes)},
    }

    result = validate_current_confidence_control(pool, reference)
    assert result["passed"] is True
    assert result["bit_exact"] is True
    assert all(result["export_hash_checks"].values())
    assert all(result["query_hash_checks"].values())

    reference["matched_hashes"]["xyz_sha256"] = "wrong"
    with pytest.raises(ValueError, match="formal hashes"):
        validate_current_confidence_control(pool, reference)


def test_gt_nearest_arm_is_unattainable_not_an_upper_bound() -> None:
    label = diagnostic_artifact_label()

    assert label["ground_truth_used_for_candidate_generation"] is False
    assert label["ground_truth_used_for_validation_gt_nearest_score"] is True
    assert label["validation_gt_nearest_score_unattainable"] is True
    assert label["validation_gt_nearest_score_strict_upper_bound"] is False
    assert label["coverage_aware_oracle_implemented"] is False
    assert label["eligible_as_method_result"] is False


def test_candidate_target_validation_rejects_shape_and_nonfinite() -> None:
    candidate = torch.zeros(4, 3)
    confidence = torch.ones(4)
    target = torch.zeros(2, 3)
    weight = torch.ones(2)
    validate_candidate_target_inputs(candidate, confidence, target, weight)

    with pytest.raises(ValueError, match="shape"):
        validate_candidate_target_inputs(
            candidate[:, :2],
            confidence,
            target,
            weight,
        )
    with pytest.raises(ValueError, match="confidence"):
        validate_candidate_target_inputs(
            candidate,
            confidence[:3],
            target,
            weight,
        )
    nonfinite = candidate.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_candidate_target_inputs(
            nonfinite,
            confidence,
            target,
            weight,
        )
    with pytest.raises(ValueError, match="positive mass"):
        validate_candidate_target_inputs(
            candidate,
            confidence,
            target,
            torch.zeros_like(weight),
        )


def test_gt_nearest_export_keeps_exact_10k_quotas_and_5cm() -> None:
    xyz = exact_candidate_set()
    distance = torch.linspace(0.0, 3.0, xyz.shape[0])

    export = validation_gt_nearest_score_export(xyz, distance)

    assert export.xyz_m.shape == (10_000, 3)
    assert export.confidence.shape == (10_000,)
    assert torch.unique(export.selected_candidate_rows).numel() == 10_000
    assert export.report["output_quotas"] == {
        "range_0_30m": 8_000,
        "range_30_60m": 1_700,
        "range_60_120m": 300,
    }
    assert export.report["observed_minimum_pair_distance_m"] >= 0.05 - 1e-6
    assert export.report["arm"] == "validation_gt_nearest_score"
    assert export.report["ground_truth_used_for_ranking"] is True
    assert export.report["strict_upper_bound"] is False
    assert export.report["coverage_aware"] is False

    with pytest.raises(ValueError, match="match candidate"):
        validation_gt_nearest_score_export(xyz, distance[:-1])
    bad_distance = distance.clone()
    bad_distance[0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validation_gt_nearest_score_export(xyz, bad_distance)


def test_ranking_alignment_reports_order_and_quantile_statistics() -> None:
    distance = torch.arange(1.0, 101.0)
    confidence = -distance

    report = ranking_alignment_diagnostics(confidence, distance)

    assert report["pearson"] == pytest.approx(1.0)
    assert report["spearman_average_ties"] == pytest.approx(1.0)
    assert set(report["confidence_quantiles"]) == {
        "q00",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "q100",
    }
    assert report["top_confidence_subsets"]["top_0p01"]["count"] == 1
    assert (
        report["top_confidence_subsets"]["top_0p25"][
            "candidate_to_target"
        ]["mean_m"]
        == pytest.approx(13.0)
    )


def test_average_ranks_handles_ties_without_python_row_loop() -> None:
    ranks = _average_ranks(
        torch.tensor([3.0, 1.0, 3.0, 2.0, 1.0]).numpy()
    )

    assert ranks.tolist() == pytest.approx([3.5, 0.5, 3.5, 2.0, 0.5])


def test_json_normalize_equates_nested_tuples_and_lists() -> None:
    tuple_document = {
        "outer": (1, {"inner": (2, 3)}),
        "scalar": "unchanged",
    }
    list_document = {
        "outer": [1, {"inner": [2, 3]}],
        "scalar": "unchanged",
    }

    assert _json_normalize(tuple_document) == _json_normalize(list_document)


def aggregate(
    *,
    chamfer: float,
    completeness: float,
    outlier: float,
    far: float,
) -> dict:
    return {
        "chamfer_m": {"mean": chamfer, "median": chamfer, "sample_count": 24},
        "completeness_mean_distance_m": {
            "mean": completeness,
            "median": completeness,
            "sample_count": 24,
        },
        "outlier_fraction_2m": {
            "mean": outlier,
            "median": outlier,
            "sample_count": 24,
        },
        "range_60_120m_completeness_mean_distance_m": {
            "mean": far,
            "median": far,
            "sample_count": 23,
        },
    }


def test_failure_factor_decision_forks_without_upper_bound_claim() -> None:
    passed = aggregate(
        chamfer=2.0,
        completeness=2.0,
        outlier=0.20,
        far=40.0,
    )
    failed = aggregate(
        chamfer=4.0,
        completeness=3.0,
        outlier=0.35,
        far=50.0,
    )

    ranking = diagnosis_decision(
        failed,
        passed,
        frame_count=24,
        far_frame_count=23,
        preflight=False,
    )
    assert ranking["branch"] == "confidence_ranking_bottleneck_indicated"
    assert ranking["formal_failure_factor_decision_eligible"] is True
    assert ranking["validation_gt_nearest_score_strict_upper_bound"] is False
    assert ranking["method_promotion_eligible"] is False

    no_rescue = diagnosis_decision(
        failed,
        failed,
        frame_count=24,
        far_frame_count=23,
        preflight=False,
    )
    assert no_rescue["branch"] == (
        "validation_gt_nearest_score_does_not_rescue"
    )

    current_pass = diagnosis_decision(
        passed,
        passed,
        frame_count=24,
        far_frame_count=23,
        preflight=False,
    )
    assert current_pass["branch"] == (
        "current_confidence_geometry_not_a_failure"
    )

    preflight = diagnosis_decision(
        failed,
        passed,
        frame_count=2,
        far_frame_count=2,
        preflight=True,
    )
    assert preflight["branch"] == (
        "preflight_only_no_failure_factor_decision"
    )
    assert preflight["formal_failure_factor_decision_eligible"] is False


def test_frame_selection_supports_two_or_24_and_rejects_test(tmp_path) -> None:
    train = [
        {"sequence": 1, "radar_index": index, "partition": "train"}
        for index in range(76)
    ]
    validation = [
        {"sequence": 2, "radar_index": index, "partition": "validation"}
        for index in range(24)
    ]
    references = [
        {
            "sequence": frame["sequence"],
            "radar_index": frame["radar_index"],
            "partition": "validation",
        }
        for frame in validation
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"frames": train + validation}),
        encoding="utf-8",
    )

    assert len(validation_records(manifest, references, preflight=True)) == 2
    assert len(validation_records(manifest, references, preflight=False)) == 24

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["frames"][0]["partition"] = "test"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="test"):
        validation_records(manifest, references, preflight=True)
