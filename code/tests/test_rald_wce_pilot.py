import json
from pathlib import Path

import numpy as np
import pytest
import torch

from losses.rald_wce_pilot import (
    RANGE_NEGATIVE_GLOBAL_PER_CLASS,
    RANGE_NEGATIVE_SHELL_PER_CLASS,
    RANGE_POSITIVE_QUOTAS,
    SOURCE_NEGATIVE_COUNT,
    SOURCE_POSITIVE_COUNT,
    rald_wce_pilot_loss,
    sample_range_class_queries,
    sample_source_classwise_queries,
)
from scripts import train_rald_wce_pilot as train


def source_target() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.tensor(
        [
            [10, 2, 2],
            [11, 3, 2],
            [40, 4, 3],
            [41, 5, 3],
            [70, 6, 4],
            [71, 7, 4],
        ],
        dtype=torch.long,
    )
    return target, torch.arange(120, dtype=torch.float32)


def test_frozen_mode_parameters_and_query_counts() -> None:
    tiny = train.frozen_pilot_config("tiny_memorization")
    source = train.frozen_pilot_config("source_classwise")
    ranged = train.frozen_pilot_config("range_class_sampler")

    assert tiny.train_frame_count == 8
    assert tiny.maximum_updates == 500
    assert tiny.evaluation_updates == (100, 200, 300, 400, 500)
    assert source.epochs == 5
    assert source.maximum_updates == 380
    assert source.evaluation_updates == (228, 380)
    assert source.occupancy_query_count == 10_000
    assert source.positive_query_count == 625
    assert source.negative_query_count == 9_375
    assert ranged.occupancy_query_count == 16_000
    assert ranged.range_positive_quotas == (700, 200, 100)
    assert ranged.range_negative_per_class == 5_000
    assert ranged.numeric_mode == "fp32_parameters_bf16_cuda_autocast"
    assert ranged.test_accessed is False
    assert ranged.future_cube_accessed is False


def test_source_sampler_is_deterministic_exact_and_full_cell() -> None:
    target, range_m = source_target()
    first = sample_source_classwise_queries(
        target,
        spatial_shape=(120, 16, 8),
        range_m=range_m,
        seed=37,
    )
    second = sample_source_classwise_queries(
        target,
        spatial_shape=(120, 16, 8),
        range_m=range_m,
        seed=37,
    )

    assert first["query_count"] == 10_000
    assert first["positive_count"] == SOURCE_POSITIVE_COUNT
    assert first["negative_count"] == SOURCE_NEGATIVE_COUNT
    assert first["normalized_rae"].shape == (1, 10_000, 3)
    assert first["metadata"]["sampling_with_replacement"] is True
    assert first["metadata"]["coordinate_phase_shortcut_allowed"] is False
    assert first["metadata"]["jitter_distribution_positive"] == (
        first["metadata"]["jitter_distribution_negative"]
    )
    assert torch.all(first["fractional_offset"].abs() <= 0.5)
    torch.testing.assert_close(
        first["normalized_rae"],
        second["normalized_rae"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first["occupancy_target"],
        second["occupancy_target"],
        rtol=0.0,
        atol=0.0,
    )


def test_positive_and_negative_jitter_have_no_phase_offset() -> None:
    target, range_m = source_target()
    sampled = sample_source_classwise_queries(
        target,
        spatial_shape=(120, 16, 8),
        range_m=range_m,
        seed=41,
    )
    labels = sampled["occupancy_target"][0] > 0.5
    jitter = sampled["fractional_offset"][0]
    mean_gap = (jitter[labels].mean(0) - jitter[~labels].mean(0)).abs()

    assert torch.all(mean_gap < 0.06)
    assert torch.all(jitter[labels].amin(0) < -0.40)
    assert torch.all(jitter[labels].amax(0) > 0.40)
    assert torch.all(jitter[~labels].amin(0) < -0.40)
    assert torch.all(jitter[~labels].amax(0) > 0.40)


def test_range_sampler_fills_frozen_class_and_source_quotas() -> None:
    target, range_m = source_target()
    sampled = sample_range_class_queries(
        target,
        spatial_shape=(120, 16, 8),
        range_m=range_m,
        seed=43,
    )
    labels = sampled["occupancy_target"][0] > 0.5
    ranges = sampled["range_class"][0]
    sources = sampled["sample_source"][0]

    assert sampled["query_count"] == 16_000
    for code, positive_quota in enumerate(RANGE_POSITIVE_QUOTAS):
        assert int((labels & (ranges == code)).sum()) == positive_quota
        assert int((~labels & (ranges == code)).sum()) == 5_000
        assert int(((sources == 1) & (ranges == code)).sum()) == (
            RANGE_NEGATIVE_GLOBAL_PER_CLASS
        )
        assert int(((sources == 2) & (ranges == code)).sum()) == (
            RANGE_NEGATIVE_SHELL_PER_CLASS
        )


def test_range_sampler_refuses_a_missing_positive_range_class() -> None:
    target, range_m = source_target()
    target_without_far = target[target[:, 0] < 60]

    with pytest.raises(ValueError, match="range class 2 positive"):
        sample_range_class_queries(
            target_without_far,
            spatial_shape=(120, 16, 8),
            range_m=range_m,
            seed=44,
        )


def test_range_sampler_excludes_occupied_3x3x3_ambiguity_band() -> None:
    target, range_m = source_target()
    sampled = sample_range_class_queries(
        target,
        spatial_shape=(120, 16, 8),
        range_m=range_m,
        seed=47,
    )
    labels = sampled["occupancy_target"][0] > 0.5
    negatives = sampled["cell_indices"][0][~labels]
    distance = (
        negatives[:, None, :] - torch.unique(target, dim=0)[None, :, :]
    ).abs().amax(dim=2)

    assert bool((distance > 1).all())


def fake_queries(range_mode: bool = False) -> dict:
    if range_mode:
        occupancy = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        range_class = torch.tensor([[0, 0, 1, 1, 2, 2]])
    else:
        occupancy = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        range_class = torch.tensor([[0, 1, 1, 2]])
    count = occupancy.shape[1]
    normalized = torch.zeros(1, count, 3)
    return {
        "normalized_rae": normalized,
        "occupancy_target": occupancy,
        "residual_target_bins": torch.zeros(1, count, 3),
        "range_class": range_class,
    }


def fake_output(queries: dict, logit: float = 0.0) -> dict:
    count = queries["occupancy_target"].shape[1]
    logits = torch.full((1, count), logit)
    return {
        "query_normalized_rae": queries["normalized_rae"],
        "occupancy_logit": logits,
        "confidence": torch.sigmoid(logits),
        "residual_bins": torch.zeros(1, count, 3),
    }


def test_source_classwise_loss_matches_frozen_formula() -> None:
    queries = fake_queries()
    output = fake_output(queries)
    loss = rald_wce_pilot_loss(
        output,
        output,
        queries,
        mode="source_classwise",
        wrong_condition_weight=0.0,
        residual_weight=0.0,
        brier_weight=0.0,
    )

    expected = 1.1 * torch.log(torch.tensor(2.0))
    torch.testing.assert_close(loss.total, expected)
    torch.testing.assert_close(loss.components["matched_occupancy"], expected)


def test_range_class_loss_normalizes_each_range_and_class() -> None:
    queries = fake_queries(range_mode=True)
    output = fake_output(queries)
    loss = rald_wce_pilot_loss(
        output,
        output,
        queries,
        mode="range_class_sampler",
        wrong_condition_weight=0.0,
        residual_weight=0.0,
        brier_weight=0.0,
    )

    assert all(
        key in loss.components
        for key in (
            "matched_range_0_positive_bce",
            "matched_range_1_negative_bce",
            "matched_range_2_positive_bce",
        )
    )
    torch.testing.assert_close(
        loss.total,
        1.1 * torch.log(torch.tensor(2.0)),
    )


def test_wrong_condition_must_reuse_identical_queries() -> None:
    queries = fake_queries()
    matched = fake_output(queries)
    wrong = fake_output(queries)
    wrong["query_normalized_rae"] = queries["normalized_rae"].clone()
    wrong["query_normalized_rae"][0, 0, 0] = 0.1

    with pytest.raises(ValueError, match="changed the frozen query set"):
        rald_wce_pilot_loss(
            matched,
            wrong,
            queries,
            mode="source_classwise",
        )


def synthetic_manifest() -> dict:
    frames = []
    for index in range(76):
        frames.append(
            {
                "partition": "train",
                "sequence": 1 + index % 8,
                "radar_index": index,
            }
        )
    for index in range(24):
        frames.append(
            {
                "partition": "validation",
                "sequence": 20 + index % 8,
                "radar_index": index,
            }
        )
    return {"frames": frames}


def test_manifest_contract_rejects_test_and_future_fields() -> None:
    assert train.validate_manifest_document(synthetic_manifest()) == {
        "train_frame_count": 76,
        "validation_frame_count": 24,
        "test_frame_count": 0,
        "future_cube_frame_count": 0,
    }
    bad = synthetic_manifest()
    bad["frames"][0]["future_cube_path"] = "forbidden"
    with pytest.raises(ValueError, match="future/test"):
        train.validate_manifest_document(bad)

    bad = synthetic_manifest()
    bad["frames"][0]["partition"] = "test"
    with pytest.raises(ValueError, match="76/24"):
        train.validate_manifest_document(bad)


def test_dataset_reads_only_approved_target_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "partition": "train",
                        "sequence": 1,
                        "radar_index": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    accessed = []

    class FakeCache:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __getitem__(self, key):
            accessed.append(key)
            if key == "target_xyz_confidence":
                return np.ones((4, 4), dtype=np.float32)
            if key == "target_rae_index":
                return np.ones((4, 3), dtype=np.int64)
            raise AssertionError(f"forbidden cache array read: {key}")

    monkeypatch.setattr(train.np, "load", lambda _path: FakeCache())
    monkeypatch.setattr(
        train,
        "load_tesseract",
        lambda _path: np.ones((64, 2, 2, 2), dtype=np.float32),
    )
    dataset = train.PilotCubeDataset(
        tmp_path,
        tmp_path,
        manifest,
        "train",
    )
    item = dataset[0]

    assert tuple(accessed) == train.ALLOWED_CACHE_ARRAYS
    assert set(item) == {
        "cube_drae",
        "target_xyz_confidence",
        "target_rae_index",
        "sequence",
        "radar_index",
        "partition",
    }


def test_tiny_selection_freezes_sparse_and_far_contract() -> None:
    statistics = [
        {
            "index": index,
            "sequence": 1 + index % 8,
            "radar_index": 100 + index,
            "target_count": 100 + index,
            "has_far_target": index in {2, 3, 4, 5, 6, 7},
        }
        for index in range(76)
    ]
    selected = train.select_tiny_subset(statistics)

    assert len(selected) == 8
    assert {0, 1}.issubset(selected)
    assert sum(statistics[index]["has_far_target"] for index in selected) >= 4
    assert len({statistics[index]["sequence"] for index in selected}) >= 2


def test_wrong_condition_is_always_cross_scene() -> None:
    records = [
        {"sequence": 1, "radar_index": 1},
        {"sequence": 2, "radar_index": 2},
        {"sequence": 1, "radar_index": 3},
        {"sequence": 3, "radar_index": 4},
    ]
    mapping = train.cross_scene_wrong_indices(records, [0, 1, 2, 3])

    assert set(mapping) == {0, 1, 2, 3}
    assert all(
        records[index]["sequence"] != records[wrong]["sequence"]
        for index, wrong in mapping.items()
    )


def test_tiny_and_five_epoch_decisions_use_frozen_thresholds() -> None:
    tiny = train.tiny_memorization_gate(
        {
            "chamfer_mean_m": 1.0,
            "outlier_fraction_mean": 0.10,
            "completeness_mean_m": 0.75,
        }
    )
    baseline = {
        "outlier_fraction_mean": 0.31,
        "far_recall_1m_mean": 0.10,
        "far_fscore_1m_mean": 0.02,
        "completeness_median_m": 1.80,
    }
    current = {
        "outlier_fraction_mean": 0.29,
        "far_recall_1m_mean": 0.14,
        "far_fscore_1m_mean": 0.026,
        "completeness_median_m": 2.00,
    }

    assert tiny["passed"] is True
    assert train.epoch3_screen(current, baseline)["continue_to_epoch5"] is True
    assert train.final_promotion(current, baseline)["promotion_passed"] is True

    current["far_fscore_1m_mean"] = 0.024
    assert train.final_promotion(current, baseline)["promotion_passed"] is False


def test_atomic_json_replaces_temp_file(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    train.atomic_json(output, {"value": 1})
    train.atomic_json(output, {"value": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 2}
    assert not output.with_suffix(".json.tmp").exists()


def test_resume_contract_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = train.frozen_pilot_config("tiny_memorization")
    contract = train.resume_contract(
        source_commit="a" * 40,
        config=config,
        input_hashes={"manifest": "m"},
        source_hashes={"source": "s"},
        tiny_frame_ids=[{"sequence": 1, "radar_index": 2}],
        baseline_hashes=None,
    )
    manifest = {
        "resume_contract_sha256": train.canonical_digest(contract),
        "resume_contract": contract,
    }
    train.validate_resume_manifest(manifest, contract)
    changed = dict(contract)
    changed["source_commit"] = "b" * 40
    with pytest.raises(ValueError, match="resume contract"):
        train.validate_resume_manifest(manifest, changed)

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    state = train.initial_run_state()
    state["updates_completed"] = 25
    checkpoint = tmp_path / "last.pt"
    digest = train.canonical_digest(contract)
    train.save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        state=state,
        contract_sha256=digest,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    restored = train.load_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        contract_sha256=digest,
    )

    assert restored["updates_completed"] == 25
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    assert not checkpoint.with_suffix(".pt.tmp").exists()


def test_formal_reference_verification_requires_metric_identity() -> None:
    formal = {
        "metrics": {
            "matched": {
                "outlier_fraction_2m": {"mean": 0.30},
                "completeness_mean_distance_m": {"median": 1.8},
                "range_60_120m_fscore_1m": {"mean": 0.02},
            }
        }
    }
    values = {
        "outlier_fraction_mean": 0.30,
        "completeness_median_m": 1.8,
        "far_fscore_1m_mean": 0.02,
    }
    train.verify_formal_reference(values, formal)
    values["outlier_fraction_mean"] = 0.31
    with pytest.raises(ValueError, match="outlier_fraction_mean"):
        train.verify_formal_reference(values, formal)
