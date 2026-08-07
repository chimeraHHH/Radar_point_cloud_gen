import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from eval.rald_wce_stage0 import WideInferenceConfig
from losses.rald_wce_quality import (
    continuous_geometry_quality_target,
    rald_wce_quality_loss,
)
from models.rald_wce_quality import RaLDWCEQualityHead
from scripts import train_rald_wce_quality_tiny as train


def _quality_head() -> RaLDWCEQualityHead:
    return RaLDWCEQualityHead(
        condition_dim=8,
        model_dim=16,
        heads=2,
        head_dim=8,
        decode_chunk_size=32,
    )


def _exact_candidate_set() -> torch.Tensor:
    def grid(
        shape: tuple[int, int, int],
        origin: tuple[float, float, float],
    ) -> torch.Tensor:
        x = origin[0] + 0.06 * torch.arange(shape[0])
        y = origin[1] + 0.06 * torch.arange(shape[1])
        z = origin[2] + 0.06 * torch.arange(shape[2])
        return torch.cartesian_prod(x, y, z)

    near = grid((20, 20, 20), (10.0, -0.6, -0.6))
    middle = grid((17, 10, 10), (40.0, -0.3, -0.3))
    far = grid((3, 10, 10), (70.0, -0.3, -0.3))
    return torch.cat((near, middle, far), dim=0)


def test_quality_head_is_ranking_only_and_initializes_to_base_score() -> None:
    head = _quality_head()
    refined = torch.rand(1, 17, 3) * 2.0 - 1.0
    condition = torch.randn(1, 5, 8)
    base = torch.linspace(0.05, 0.95, 17).unsqueeze(0)

    output = head(refined, condition, base)

    torch.testing.assert_close(output["quality"], base)
    torch.testing.assert_close(
        output["quality_delta_logit"],
        torch.zeros_like(base),
    )
    metadata = head.architecture_metadata()
    assert metadata["candidate_coordinates_changed"] is False
    assert metadata["formal_residual_changed"] is False
    assert metadata["ground_truth_inference"] is False
    assert metadata["doppler_head"] is False
    assert metadata["best_of_k"] is False
    assert "target_xyz_confidence" not in inspect.signature(head.forward).parameters


def test_quality_head_rejects_misaligned_or_nonfinite_inputs() -> None:
    head = _quality_head()
    refined = torch.zeros(1, 4, 3)
    condition = torch.zeros(1, 2, 8)
    base = torch.full((1, 4), 0.5)
    head(refined, condition, base)

    with pytest.raises(ValueError, match="align"):
        head(refined, condition, base[:, :3])
    bad = refined.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        head(bad, condition, base)
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        head(torch.full_like(refined, 2.0), condition, base)


def test_continuous_quality_target_uses_distance_and_confidence_without_grad() -> None:
    candidate = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 0.25]],
        requires_grad=True,
    )

    result = continuous_geometry_quality_target(
        candidate,
        target,
        distance_temperature_m=1.0,
        candidate_chunk_size=2,
    )

    expected = torch.tensor([1.0, torch.exp(torch.tensor(-1.0)), 0.25])
    torch.testing.assert_close(result["quality_target"], expected)
    torch.testing.assert_close(
        result["nearest_distance_m"],
        torch.tensor([0.0, 1.0, 0.0]),
    )
    assert result["quality_target"].requires_grad is False
    assert result["nearest_distance_m"].requires_grad is False
    assert candidate.grad is None
    assert target.grad is None


def test_continuous_quality_target_requires_positive_confidence_mass() -> None:
    candidate = torch.zeros(2, 3)
    target = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="positive mass"):
        continuous_geometry_quality_target(candidate, target)


def test_quality_loss_rewards_correct_within_range_ordering() -> None:
    target = torch.tensor(
        [[0.05, 0.25, 0.75, 0.95] * 3],
        dtype=torch.float32,
    )
    codes = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]],
        dtype=torch.long,
    )
    good = torch.tensor(
        [[-3.0, -1.0, 1.0, 3.0] * 3],
        requires_grad=True,
    )
    bad = -good.detach()

    good_loss = rald_wce_quality_loss(good, target, codes)
    bad_loss = rald_wce_quality_loss(bad, target, codes)

    assert good_loss.total < bad_loss.total
    assert good_loss.components["quality_pair_count"].item() == 6
    good_loss.total.backward()
    assert good.grad is not None
    assert torch.isfinite(good.grad).all()


def test_quality_loss_rejects_missing_range_class() -> None:
    logits = torch.zeros(1, 6)
    target = torch.zeros(1, 6)
    codes = torch.tensor([[0, 0, 0, 1, 1, 1]])
    with pytest.raises(ValueError, match="range class 2"):
        rald_wce_quality_loss(logits, target, codes)


def test_quality_sampler_is_unique_deterministic_and_range_frozen() -> None:
    quality = torch.linspace(0.0, 1.0, 350)
    codes = torch.cat(
        (
            torch.zeros(200, dtype=torch.long),
            torch.ones(100, dtype=torch.long),
            torch.full((50,), 2, dtype=torch.long),
        )
    )
    kwargs = {
        "range_quotas": (20, 10, 4),
        "high_quality_fraction": 0.5,
        "high_quality_pool_fraction": 0.2,
        "seed": 123,
    }

    first = train.sample_quality_candidate_rows(quality, codes, **kwargs)
    second = train.sample_quality_candidate_rows(quality, codes, **kwargs)

    assert torch.equal(first, second)
    assert first.numel() == 34
    assert torch.unique(first).numel() == 34
    assert tuple(int((codes[first] == code).sum()) for code in range(3)) == (
        20,
        10,
        4,
    )
    for code, quota in enumerate(kwargs["range_quotas"]):
        eligible = torch.nonzero(codes == code, as_tuple=False).flatten()
        order = torch.argsort(quality[eligible], descending=True, stable=True)
        pool_count = max(
            quota // 2,
            int(math.ceil(eligible.numel() * 0.2)),
        )
        high_pool = eligible[order[:pool_count]]
        selected = first[codes[first] == code]
        assert torch.isin(selected, high_pool).sum().item() == quota // 2


def test_ordered_cache_digest_binds_identity_order_and_bytes(
    tmp_path: Path,
) -> None:
    records = [
        {"sequence": 1, "radar_index": 2},
        {"sequence": 3, "radar_index": 4},
    ]
    for record, payload in zip(records, (b"first", b"second"), strict=True):
        train.cache_path(tmp_path, record).write_bytes(payload)

    digest, rows = train.ordered_cache_digest(records, tmp_path)
    reversed_digest, _ = train.ordered_cache_digest(
        list(reversed(records)),
        tmp_path,
    )

    assert len(digest) == 64
    assert rows[0]["identity"] == "seq01/radar00002"
    assert digest != reversed_digest
    train.cache_path(tmp_path, records[0]).write_bytes(b"changed")
    changed_digest, _ = train.ordered_cache_digest(records, tmp_path)
    assert changed_digest != digest


def test_initial_ranking_control_preserves_exact_export_rows() -> None:
    head = _quality_head()
    xyz = _exact_candidate_set()
    confidence = torch.linspace(0.01, 0.99, xyz.shape[0])
    field = {
        "xyz_m": xyz,
        "base_confidence": confidence.unsqueeze(0),
        "refined_normalized_rae": (
            torch.rand(1, xyz.shape[0], 3) * 2.0 - 1.0
        ),
        "condition_latents": torch.randn(1, 5, 8),
        "report": {"candidate_xyz_sha256": "synthetic"},
    }

    report = train.initial_ranking_control(
        head,
        field,
        WideInferenceConfig(),
    )

    assert report["passed"] is True
    assert report["selected_candidate_rows_identical"] is True
    assert (
        report["base_selected_rows_sha256"]
        == report["quality_selected_rows_sha256"]
    )


def test_frozen_config_and_inference_signature_lock_tiny_protocol() -> None:
    config = train.frozen_quality_config()
    assert config.protocol == "g1_q1r_rald_wce_quality_tiny_v1"
    assert config.maximum_updates == 500
    assert config.evaluation_updates == (100, 200, 300, 400, 500)
    assert config.quality_sample_count == 16_000
    assert config.output_range_quotas == (8_000, 1_700, 300)
    assert config.minimum_export_distance_m == 0.05
    assert config.test_accessed is False
    assert config.doppler_head is False
    assert config.best_of_k is False
    assert train.FROZEN_ORDERED_TINY_CUBE_DIGEST_SHA256 == (
        "0bfbdb5eac17f9823033942e41abdb6d3b7303f7b55cb06807d8a0933f8a4b7c"
    )
    assert len(train.FROZEN_TINY_FRAME_IDENTITIES) == 8
    signature = inspect.signature(train.infer_exact_quality)
    assert "target_xyz_confidence" not in signature.parameters
    assert "target" not in signature.parameters


def test_quality_gate_requires_numeric_and_both_export_structures() -> None:
    values = {
        "chamfer_mean_m": 0.8,
        "outlier_fraction_mean": 0.08,
        "completeness_median_m": 0.6,
        "completeness_mean_m": 0.7,
    }
    exact = {
        "all_matched_exports_exact_10000": True,
        "all_wrong_exports_exact_10000": True,
        "all_minimum_distance_5cm": True,
        "all_wrong_minimum_distance_5cm": True,
        "copy_padding_jitter_duplicate": False,
    }
    passed = train.quality_tiny_gate(values, {"exact_export": exact})
    assert passed["passed"] is True

    wrong_failed = dict(exact)
    wrong_failed["all_wrong_minimum_distance_5cm"] = False
    failed = train.quality_tiny_gate(
        values,
        {"exact_export": wrong_failed},
    )
    assert failed["passed"] is False
    assert failed["numeric_checks"] == passed["numeric_checks"]


def test_replay_parent_certificate_binds_every_parent_artifact(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    metrics = tmp_path / "metrics.json"
    manifest = tmp_path / "manifest.json"
    diagnosis = tmp_path / "diagnosis.json"
    for path, payload in (
        (checkpoint, b"checkpoint"),
        (metrics, b"metrics"),
        (manifest, b"manifest"),
        (diagnosis, b"diagnosis"),
    ):
        path.write_bytes(payload)
    identity = {
        "original_checkpoint_sha256": (
            train.ORIGINAL_FORMAL_CHECKPOINT_SHA256
        ),
        "replay_checkpoint_sha256": train.sha256_file(checkpoint),
        "replay_metrics_sha256": train.sha256_file(metrics),
        "replay_run_manifest_sha256": train.sha256_file(manifest),
        "replay_failure_diagnosis_sha256": train.sha256_file(diagnosis),
        "formal_source_commit": train.FORMAL_PARENT_SOURCE_COMMIT,
        "formal_epoch": 20,
        "exact_original_checkpoint": False,
        "certifier_source_commit": "a" * 40,
    }
    checks = {
        "source_config_data_seed_epoch_match": True,
        "formal_stage0_decision_preserved": True,
        "validation_frame_contract_preserved": True,
        "candidate_query_contract_preserved": True,
        "replay_diagnosis_bound": True,
        "validation_gt_ranking_oracle_passed": True,
        "test_partition_accessed": False,
        "future_cube_accessed": False,
        "cfar_accessed": False,
        "doppler_head_evaluated": False,
    }
    certificate = tmp_path / "certificate.json"
    certificate.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": train.PARENT_CERTIFICATE_PROTOCOL,
                "status": "replay_parent_authorized_for_q1r_tiny",
                "q1r_tiny_authorized": True,
                "identity": identity,
                "checks": checks,
                "claim_boundary": "source-equivalent replay, not original",
            }
        ),
        encoding="utf-8",
    )

    evidence = train.validate_replay_parent_certificate(
        certificate,
        diagnosis,
        checkpoint,
        metrics,
        manifest,
        expected_certifier_source_commit="a" * 40,
    )
    assert evidence["validation_checks"]["replay_artifact_hashes"] is True

    diagnosis.write_bytes(b"changed")
    with pytest.raises(ValueError, match="replay_artifact_hashes"):
        train.validate_replay_parent_certificate(
            certificate,
            diagnosis,
            checkpoint,
            metrics,
            manifest,
            expected_certifier_source_commit="a" * 40,
        )


def test_cpu_test_does_not_initialize_cuda() -> None:
    assert torch.cuda.is_initialized() is False
