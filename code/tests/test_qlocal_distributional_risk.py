from __future__ import annotations

import inspect

import pytest
import torch

from losses.qlocal_distributional_risk import (
    distance_distribution_target,
    qlocal_distributional_risk_loss,
)
from models.qlocal_distributional_risk import (
    DISTANCE_CENTERS_M,
    QLocalDistributionalRiskScorer,
    extract_local_full_raed_features,
    qlocal_parameter_count,
)
from scripts.preflight_qlocal_distributional_risk import (
    FIT_IDENTITIES,
    UNSEEN_IDENTITIES,
    WRONG_IDENTITY_MAP,
    atomic_commit_report,
    parse_nvidia_smi_rows,
    resolve_frozen_cohort,
)


def _model_inputs(batch: int = 2, points: int = 24) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    spectrum = torch.rand(batch, points, 64, generator=generator)
    spectrum = spectrum / spectrum.sum(dim=-1, keepdim=True)
    return {
        "refined_normalized_rae": torch.rand(
            batch, points, 3, generator=generator
        )
        * 2.0
        - 1.0,
        "base_confidence": torch.rand(batch, points, generator=generator),
        "condition_latents": torch.randn(
            batch, 12, 512, generator=generator
        ),
        "local_spectrum": spectrum,
        "normalized_log_energy": torch.randn(
            batch, points, 1, generator=generator
        ),
        "spectrum_entropy": torch.rand(
            batch, points, 1, generator=generator
        ),
        "energy_gradients": torch.randn(
            batch, points, 3, generator=generator
        ),
    }


def test_scorer_contract_and_shapes() -> None:
    model = QLocalDistributionalRiskScorer(model_dim=64, heads=2, head_dim=16)
    inputs = _model_inputs()
    output = model(**inputs, chunk_size=7)
    assert output["distance_logits"].shape == (2, 24, 6)
    assert output["distance_probability"].shape == (2, 24, 6)
    assert output["selection_score"].shape == (2, 24)
    assert torch.allclose(
        output["distance_probability"].sum(dim=-1),
        torch.ones(2, 24),
    )
    assert qlocal_parameter_count(model) < 2_000_000
    metadata = model.architecture_metadata()
    assert metadata["candidate_coordinates_changed"] is False
    assert metadata["ground_truth_inference"] is False
    assert "target_xyz_confidence" not in inspect.signature(model.forward).parameters


def test_zero_delta_prior_is_monotone_with_parent_confidence() -> None:
    model = QLocalDistributionalRiskScorer(model_dim=64, heads=2, head_dim=16)
    inputs = _model_inputs(batch=1, points=9)
    confidence = torch.linspace(0.01, 0.99, 9).unsqueeze(0)
    inputs["base_confidence"] = confidence
    score = model(**inputs)["selection_score"][0]
    assert torch.all(score[1:] > score[:-1])


def test_scorer_rejects_unnormalized_spectrum() -> None:
    model = QLocalDistributionalRiskScorer(model_dim=64, heads=2, head_dim=16)
    inputs = _model_inputs(batch=1)
    inputs["local_spectrum"] = torch.ones_like(inputs["local_spectrum"])
    with pytest.raises(ValueError, match="not normalized"):
        model(**inputs)


def test_local_full_raed_features_are_finite_and_cube_dependent() -> None:
    generator = torch.Generator().manual_seed(19)
    cube = torch.rand(1, 64, 8, 7, 6, generator=generator)
    wrong = torch.flip(cube, dims=(1, 2))
    coordinates = torch.tensor(
        [[[0.5, 1.25, 2.0], [4.2, 3.1, 1.7], [7.0, 6.0, 5.0]]]
    )
    matched = extract_local_full_raed_features(cube, coordinates)
    intervened = extract_local_full_raed_features(wrong, coordinates)
    assert matched["local_spectrum"].shape == (1, 3, 64)
    assert matched["normalized_log_energy"].shape == (1, 3, 1)
    assert matched["spectrum_entropy"].shape == (1, 3, 1)
    assert matched["energy_gradients"].shape == (1, 3, 3)
    assert not torch.equal(
        matched["local_spectrum"], intervened["local_spectrum"]
    )


def test_distance_distribution_interpolates_and_clamps() -> None:
    centers = torch.tensor(DISTANCE_CENTERS_M)
    distance = torch.tensor([[0.0, 0.05, 0.1125, 0.375, 4.0]])
    target = distance_distribution_target(distance, centers)
    assert target.shape == (1, 5, 6)
    assert torch.allclose(target.sum(dim=-1), torch.ones(1, 5))
    assert target[0, 0, 0] == 1.0
    assert target[0, 1, 0] == 1.0
    assert torch.allclose(target[0, 2, :2], torch.tensor([0.5, 0.5]))
    assert target[0, 3, 2] == 1.0
    assert target[0, 4, -1] == 1.0


def test_distributional_listwise_loss_has_finite_gradients() -> None:
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(1, 18, 6, generator=generator, requires_grad=True)
    distance = torch.linspace(0.0, 4.0, 18).unsqueeze(0)
    centers = torch.tensor(DISTANCE_CENTERS_M)
    target = distance_distribution_target(distance, centers)
    range_class = torch.tensor([[0] * 6 + [1] * 6 + [2] * 6])
    loss = qlocal_distributional_risk_loss(
        logits,
        target,
        distance,
        range_class,
        centers,
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def _cohort_records() -> list[dict[str, object]]:
    return [
        {
            "sequence": sequence,
            "radar_index": radar_index,
            "partition": "train",
        }
        for sequence, radar_index in FIT_IDENTITIES + UNSEEN_IDENTITIES
    ]


def test_frozen_cohort_enforces_identity_order_and_wrong_pairs() -> None:
    records = _cohort_records()
    fit, unseen, wrong = resolve_frozen_cohort(records)
    assert fit == list(range(8))
    assert unseen == list(range(8, 12))
    for source, destination in WRONG_IDENTITY_MAP.items():
        source_index = (FIT_IDENTITIES + UNSEEN_IDENTITIES).index(source)
        destination_index = (FIT_IDENTITIES + UNSEEN_IDENTITIES).index(
            destination
        )
        assert wrong[source_index] == destination_index


def test_frozen_cohort_rejects_drop_duplicate_and_partition_tamper() -> None:
    records = _cohort_records()
    with pytest.raises(ValueError, match="missing"):
        resolve_frozen_cohort(records[:-1])
    duplicate = records + [dict(records[0])]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_frozen_cohort(duplicate)
    tampered = [dict(record) for record in records]
    tampered[3]["partition"] = "validation"
    with pytest.raises(ValueError, match="not train"):
        resolve_frozen_cohort(tampered)


def test_nvidia_smi_parser_binds_uuid_and_pci() -> None:
    rows = parse_nvidia_smi_rows(
        "0, GPU-a, 00000000:01:00.0, NVIDIA H200 NVL\n"
        "2, GPU-b, 00000000:D1:00.0, NVIDIA H200 NVL\n"
    )
    assert rows[0]["uuid"] == "GPU-a"
    assert rows[2]["pci_bus_id"] == "00000000:D1:00.0"
    with pytest.raises(ValueError, match="malformed"):
        parse_nvidia_smi_rows("0,too,few\n")


def test_atomic_report_commit_is_fail_closed(tmp_path) -> None:
    output = tmp_path / "result"
    report = {"status": "qlocal_f0_capacity_no_go", "passed": False}
    path = atomic_commit_report(output, report)
    assert path.is_file()
    assert not list(tmp_path.glob(".result.staging-*"))
    with pytest.raises(FileExistsError):
        atomic_commit_report(output, report)
