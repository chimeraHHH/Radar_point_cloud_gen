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


def _fake_parent_certificate(
    *,
    checkpoint: Path,
    metrics: Path,
    manifest: Path,
    diagnosis: Path,
    certifier_source_commit: str,
) -> dict:
    checks = {
        "source_config_data_seed_epoch_match": True,
        "formal_metrics_recomputed": True,
        "formal_stage0_decision_preserved": True,
        "validation_frame_contract_preserved": True,
        "candidate_query_contract_preserved": True,
        "replay_diagnosis_bound": True,
        "diagnosis_aggregate_recomputed": True,
        "diagnosis_hash_geometry_chain_recomputed": True,
        "diagnostic_source_and_input_binding": True,
        "physical_gpu_policy": True,
        "validation_gt_ranking_oracle_passed": True,
        "training_started": False,
        "checkpoint_modified": False,
        "test_partition_accessed": False,
        "future_cube_accessed": False,
        "cfar_accessed": False,
        "doppler_head_evaluated": False,
    }
    checkpoint_sha = train.sha256_file(checkpoint)
    return {
        "schema_version": 1,
        "protocol": train.PARENT_CERTIFICATE_PROTOCOL,
        "status": "replay_parent_authorized_for_q1r_tiny",
        "q1r_tiny_authorized": True,
        "identity": {
            "original_checkpoint_sha256": (
                train.ORIGINAL_FORMAL_CHECKPOINT_SHA256
            ),
            "replay_checkpoint_sha256": checkpoint_sha,
            "replay_metrics_sha256": train.sha256_file(metrics),
            "replay_run_manifest_sha256": train.sha256_file(manifest),
            "replay_failure_diagnosis_sha256": train.sha256_file(diagnosis),
            "formal_source_commit": train.FORMAL_PARENT_SOURCE_COMMIT,
            "formal_epoch": 20,
            "exact_original_checkpoint": (
                checkpoint_sha == train.ORIGINAL_FORMAL_CHECKPOINT_SHA256
            ),
            "certifier_source_commit": certifier_source_commit,
        },
        "checks": checks,
        "claim_boundary": "source-equivalent replay, not original",
    }


def test_replay_parent_certificate_is_recomputed_and_binds_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    source_commit = "a" * 40

    def fake_certifier(**kwargs: object) -> dict:
        return _fake_parent_certificate(
            checkpoint=Path(kwargs["replay_checkpoint_path"]),
            metrics=Path(kwargs["replay_metrics_path"]),
            manifest=Path(kwargs["replay_manifest_path"]),
            diagnosis=Path(kwargs["replay_diagnosis_path"]),
            certifier_source_commit=str(kwargs["certifier_source_commit"]),
        )

    monkeypatch.setattr(train, "certify_replay_parent", fake_certifier)
    document = fake_certifier(
        replay_checkpoint_path=checkpoint,
        replay_metrics_path=metrics,
        replay_manifest_path=manifest,
        replay_diagnosis_path=diagnosis,
        certifier_source_commit=source_commit,
    )
    certificate = tmp_path / "certificate.json"
    certificate.write_text(json.dumps(document), encoding="utf-8")

    evidence = train.validate_replay_parent_certificate(
        certificate,
        diagnosis,
        checkpoint,
        metrics,
        manifest,
        expected_certifier_source_commit=source_commit,
        repo=tmp_path,
    )
    assert evidence["validation_checks"]["replay_artifact_hashes"] is True
    assert evidence["trainer_recomputed_certificate"] is True

    diagnosis.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from trainer recomputation"):
        train.validate_replay_parent_certificate(
            certificate,
            diagnosis,
            checkpoint,
            metrics,
            manifest,
            expected_certifier_source_commit=source_commit,
            repo=tmp_path,
        )


def test_replay_parent_certificate_rejects_forged_all_true_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {}
    for name in ("checkpoint", "metrics", "manifest", "diagnosis"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    source_commit = "a" * 40
    recomputed = _fake_parent_certificate(
        checkpoint=paths["checkpoint"],
        metrics=paths["metrics"],
        manifest=paths["manifest"],
        diagnosis=paths["diagnosis"],
        certifier_source_commit=source_commit,
    )
    monkeypatch.setattr(train, "certify_replay_parent", lambda **_: recomputed)
    forged = json.loads(json.dumps(recomputed))
    forged["comparison"] = {"invented": True}
    certificate = tmp_path / "certificate.json"
    certificate.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from trainer recomputation"):
        train.validate_replay_parent_certificate(
            certificate,
            paths["diagnosis"],
            paths["checkpoint"],
            paths["metrics"],
            paths["manifest"],
            expected_certifier_source_commit=source_commit,
            repo=tmp_path,
        )


def test_immutable_evaluation_checkpoint_is_reused_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _quality_head()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    state = train.initial_state()
    path = tmp_path / "checkpoint_update0100.pt"

    def cpu_safe_save(
        output: Path,
        *,
        quality_head: RaLDWCEQualityHead,
        optimizer: torch.optim.Optimizer,
        state: dict,
        contract_sha256: str,
    ) -> None:
        torch.save(
            {
                "protocol": train.PROTOCOL,
                "resume_contract_sha256": contract_sha256,
                "quality_head": quality_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "state": state,
                "formal_base_state_stored": False,
            },
            output,
        )

    monkeypatch.setattr(train, "save_checkpoint", cpu_safe_save)
    first_sha = train.ensure_evaluation_checkpoint(
        path,
        resume=False,
        quality_head=head,
        optimizer=optimizer,
        state=state,
        contract_sha256="contract",
    )
    second_sha = train.ensure_evaluation_checkpoint(
        path,
        resume=True,
        quality_head=head,
        optimizer=optimizer,
        state=state,
        contract_sha256="contract",
    )
    assert first_sha == second_sha

    with torch.no_grad():
        next(head.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="quality_head"):
        train.ensure_evaluation_checkpoint(
            path,
            resume=True,
            quality_head=head,
            optimizer=optimizer,
            state=state,
            contract_sha256="contract",
        )


def _quality_metrics() -> dict:
    frames = []
    for index in range(8):
        matched = {
            "chamfer_m": 0.8,
            "outlier_fraction_2m": 0.08,
            "completeness_mean_distance_m": 0.7,
            "range_60_120m_fscore_1m": 0.1,
            "prediction_count": 10_000,
            "target_count": 1_000,
        }
        wrong = {
            "chamfer_m": 0.9,
            "outlier_fraction_2m": 0.09,
            "completeness_mean_distance_m": 0.75,
            "range_60_120m_fscore_1m": 0.08,
            "prediction_count": 10_000,
            "target_count": 1_000,
        }
        export = {
            "exact_point_count": 10_000,
            "observed_minimum_pair_distance_m": 0.06,
            "copy_padding_jitter_duplicate": False,
        }
        frames.append(
            {
                "sequence": index + 1,
                "radar_index": 100 + index,
                "partition": "train",
                "wrong_condition_sequence": 20 + index,
                "wrong_condition_radar_index": 200 + index,
                "matched": matched,
                "wrong_condition": wrong,
                "condition_intervention": {
                    "wrong_minus_matched_chamfer_fraction": 0.125,
                    "matched_chamfer_better": 1.0,
                },
                "matched_export": export,
                "wrong_export": export,
                "inference": {
                    "ground_truth_accessed": False,
                    "test_accessed": False,
                    "doppler_head": False,
                    "best_of_k": False,
                },
            }
        )
    return train.recompute_quality_metrics({"frames": frames})


def test_completed_evaluation_is_recomputed_before_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_update0100.pt"
    checkpoint.write_bytes(b"immutable")
    path = tmp_path / "metrics_update0100.json"
    document = train.build_evaluation_document(
        metrics=_quality_metrics(),
        updates=100,
        source_commit="a" * 40,
        evaluation_checkpoint=checkpoint,
        evaluation_checkpoint_sha256=train.sha256_file(checkpoint),
        formal_base_state_sha256="base",
        prior_consecutive_passes=0,
    )
    train.atomic_json(path, document)

    loaded = train.load_completed_evaluation(
        path,
        updates=100,
        source_commit="a" * 40,
        evaluation_checkpoint=checkpoint,
        evaluation_checkpoint_sha256=train.sha256_file(checkpoint),
        formal_base_state_sha256="base",
        prior_consecutive_passes=0,
    )
    assert loaded == document

    document["decision"]["passed"] = False
    train.atomic_json(path, document)
    with pytest.raises(ValueError, match="differs from recomputation"):
        train.load_completed_evaluation(
            path,
            updates=100,
            source_commit="a" * 40,
            evaluation_checkpoint=checkpoint,
            evaluation_checkpoint_sha256=train.sha256_file(checkpoint),
            formal_base_state_sha256="base",
            prior_consecutive_passes=0,
        )


def test_completed_evaluation_rejects_forged_aggregate_and_decision(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint_update0100.pt"
    checkpoint.write_bytes(b"immutable")
    path = tmp_path / "metrics_update0100.json"
    document = train.build_evaluation_document(
        metrics=_quality_metrics(),
        updates=100,
        source_commit="a" * 40,
        evaluation_checkpoint=checkpoint,
        evaluation_checkpoint_sha256=train.sha256_file(checkpoint),
        formal_base_state_sha256="base",
        prior_consecutive_passes=0,
    )
    document["metrics"]["matched"]["chamfer_m"]["mean"] = 0.1
    document["values"] = train.quality_metric_values(document["metrics"])
    document["decision"] = train.quality_tiny_gate(
        document["values"], document["metrics"]
    )
    document["decision"]["consecutive_passes"] = 1
    document["decision"]["early_stop_passed"] = False
    train.atomic_json(path, document)

    with pytest.raises(ValueError, match="differs from recomputation"):
        train.load_completed_evaluation(
            path,
            updates=100,
            source_commit="a" * 40,
            evaluation_checkpoint=checkpoint,
            evaluation_checkpoint_sha256=train.sha256_file(checkpoint),
            formal_base_state_sha256="base",
            prior_consecutive_passes=0,
        )


def test_candidate_preparation_is_immutable_across_resume(tmp_path: Path) -> None:
    path = tmp_path / "candidate_preparation.json"
    document = {"protocol": train.PROTOCOL, "frames": [{"id": 1}]}
    expected_sha = train.json_artifact_sha256(document)

    assert train.bind_candidate_preparation(
        path,
        document,
        resume=False,
    ) == expected_sha
    assert train.bind_candidate_preparation(
        path,
        document,
        resume=True,
    ) == expected_sha

    path.unlink()
    assert train.bind_candidate_preparation(
        path,
        document,
        resume=True,
    ) == expected_sha
    changed = {"protocol": train.PROTOCOL, "frames": [{"id": 2}]}
    with pytest.raises(ValueError, match="changed on resume"):
        train.bind_candidate_preparation(path, changed, resume=True)


def test_output_validation_has_no_creation_side_effect(tmp_path: Path) -> None:
    output = tmp_path / "new-run"

    train.prepare_output(output, resume=False)

    assert output.exists() is False
    train.initialize_output(output, resume=False)
    assert output.is_dir()


def test_staging_initialization_exposes_only_complete_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    staging = train.create_staging_output(output)

    assert output.exists() is False
    (staging / "run_manifest.json").write_text("{}", encoding="utf-8")
    (staging / "candidate_preparation.json").write_text("{}", encoding="utf-8")
    (staging / "last.pt").write_bytes(b"checkpoint")
    staging.replace(output)

    assert (output / "run_manifest.json").is_file()
    assert (output / "candidate_preparation.json").is_file()
    assert (output / "last.pt").is_file()


def test_terminal_resume_validates_last_immutable_evaluation(
    tmp_path: Path,
) -> None:
    head = _quality_head()
    pre_state = train.initial_state()
    pre_state["updates_completed"] = 100
    pre_state["consecutive_gate_passes"] = 1
    evaluation_checkpoint = tmp_path / "checkpoint_update0100.pt"
    torch.save(
        {
            "protocol": train.PROTOCOL,
            "resume_contract_sha256": "contract",
            "quality_head": head.state_dict(),
            "state": pre_state,
            "formal_base_state_stored": False,
        },
        evaluation_checkpoint,
    )
    evaluation = train.build_evaluation_document(
        metrics=_quality_metrics(),
        updates=100,
        source_commit="a" * 40,
        evaluation_checkpoint=evaluation_checkpoint,
        evaluation_checkpoint_sha256=train.sha256_file(evaluation_checkpoint),
        formal_base_state_sha256="base",
        prior_consecutive_passes=1,
    )
    assert evaluation["decision"]["early_stop_passed"] is True
    train.atomic_json(tmp_path / "metrics_update0100.json", evaluation)
    terminal = json.loads(json.dumps(pre_state))
    terminal["evaluation_updates_completed"] = [100]
    terminal["consecutive_gate_passes"] = 2
    terminal["status"] = "quality_ranking_tiny_passed_early"
    terminal["terminal_runtime"] = {
        "elapsed_seconds": 12.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }

    status = train.validate_terminal_resume(
        tmp_path,
        state=terminal,
        quality_head=head,
        contract_sha256="contract",
        source_commit="a" * 40,
        formal_base_state_sha256="base",
        maximum_updates=500,
    )

    assert status == "quality_ranking_tiny_passed_early"
    terminal["consecutive_gate_passes"] = 1
    with pytest.raises(ValueError, match="consecutive_passes"):
        train.validate_terminal_resume(
            tmp_path,
            state=terminal,
            quality_head=head,
            contract_sha256="contract",
            source_commit="a" * 40,
            formal_base_state_sha256="base",
            maximum_updates=500,
        )


def test_h200_runtime_requires_frozen_physical_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _: "NVIDIA H200 NVL",
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    device, name = train.require_h200("cuda:0")

    assert device == torch.device("cuda:0")
    assert name == "NVIDIA H200 NVL"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="physical H200 GPU 0 or 2"):
        train.require_h200("cuda:0")


def test_cpu_test_does_not_initialize_cuda() -> None:
    assert torch.cuda.is_initialized() is False
