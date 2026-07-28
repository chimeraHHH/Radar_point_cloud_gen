import math

import pytest
import torch
import torch.nn.functional as F

from losses.rald_query_field import (
    chunked_nearest_other_distance,
    global_knn_repulsion,
    rald_query_field_loss,
    sample_occupancy_queries,
)


def _dilated(occupancy: torch.Tensor) -> torch.Tensor:
    return (
        F.max_pool3d(
            (occupancy > 0)[None, None].float(),
            kernel_size=3,
            stride=1,
            padding=1,
        )[0, 0]
        > 0
    )


def _loss_output(
    query_logits: torch.Tensor,
    xyz: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "query_logits": query_logits,
        "generated_xyz_m": xyz,
        "generated_confidence_logit": torch.full(
            (xyz.shape[0],),
            torch.logit(torch.tensor(0.75)).item(),
            dtype=xyz.dtype,
            device=xyz.device,
        ),
        "normalized_offset": torch.zeros_like(xyz),
    }


def test_sampler_returns_exact_625_9375_and_unambiguous_balanced_negatives() -> None:
    occupancy = torch.zeros(24, 24, 24)
    target = torch.tensor(
        [
            [3, 4, 5],
            [11, 12, 13],
            [20, 18, 17],
        ]
    )
    occupancy[target[:, 0], target[:, 1], target[:, 2]] = torch.tensor(
        [1.0, 2.0, 4.0]
    )
    generator = torch.Generator().manual_seed(20260728)

    queries = sample_occupancy_queries(
        target,
        occupancy,
        generator=generator,
    )

    assert queries.coordinates_rae.shape == (10_000, 3)
    assert queries.labels.shape == (10_000,)
    assert queries.positive_count == 625
    assert queries.negative_count == 9_375
    assert int(queries.labels.sum()) == 625
    assert queries.range_stratum.shape == (10_000,)
    assert set(queries.range_stratum.tolist()) == {0, 1, 2}

    negative = queries.coordinates_rae[queries.labels == 0].long()
    assert not bool(
        _dilated(occupancy)[
            negative[:, 0],
            negative[:, 1],
            negative[:, 2],
        ].any()
    )
    negative_strata = queries.range_stratum[queries.labels == 0]
    assert torch.bincount(negative_strata, minlength=3).tolist() == [
        3_125,
        3_125,
        3_125,
    ]

    positive = queries.coordinates_rae[queries.labels == 1]
    positive_cell = positive.round().long()
    assert bool(
        (occupancy[
            positive_cell[:, 0],
            positive_cell[:, 1],
            positive_cell[:, 2],
        ] > 0).all()
    )
    assert bool(((positive - positive_cell).abs() <= 0.5).all())


@pytest.mark.parametrize(
    ("target", "occupancy", "count", "ratio", "message"),
    [
        (
            torch.empty(0, 3, dtype=torch.long),
            torch.zeros(6, 6, 6),
            16,
            0.25,
            "At least one occupied",
        ),
        (
            torch.tensor([[1, 1, 1]]),
            torch.zeros(6, 6),
            16,
            0.25,
            "occupancy must have shape",
        ),
        (
            torch.tensor([[1, 1, 1]]),
            torch.ones(2, 6, 6),
            16,
            0.25,
            "three non-empty strata",
        ),
        (
            torch.tensor([[1, 1, 1]]),
            torch.ones(6, 6, 6),
            10,
            0.0625,
            "must be an integer",
        ),
    ],
)
def test_sampler_strictly_validates_empty_shape_ratio_and_strata(
    target: torch.Tensor,
    occupancy: torch.Tensor,
    count: int,
    ratio: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_occupancy_queries(
            target,
            occupancy,
            count=count,
            positive_ratio=ratio,
        )


def test_sampler_reports_insufficient_unambiguous_negatives() -> None:
    occupancy = torch.ones(6, 6, 6)
    with pytest.raises(
        ValueError,
        match="Insufficient unambiguous negative cells",
    ):
        sample_occupancy_queries(
            torch.tensor([[1, 1, 1]]),
            occupancy,
            count=16,
            positive_ratio=0.25,
        )


def test_global_knn_repulsion_detects_cross_seed_collapse() -> None:
    cross_seed_collapse = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    separated = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]
    )

    collapsed_loss = global_knn_repulsion(
        cross_seed_collapse,
        minimum_distance_m=0.5,
        chunk_size=2,
    )
    separated_loss = global_knn_repulsion(
        separated,
        minimum_distance_m=0.5,
        chunk_size=2,
    )
    assert collapsed_loss > 0
    assert separated_loss == 0
    assert collapsed_loss > separated_loss
    torch.testing.assert_close(
        chunked_nearest_other_distance(
            cross_seed_collapse,
            chunk_size=2,
        )[[0, 2]],
        torch.zeros(2),
    )


def test_query_field_loss_splits_positive_and_negative_bce() -> None:
    logits = torch.tensor([0.5, -0.25, 1.5, -2.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    target = torch.cat((xyz, torch.ones(4, 1)), dim=1)

    result = rald_query_field_loss(
        _loss_output(logits, xyz),
        labels,
        target,
        generated_point_count=4,
        geometry_weight=0.0,
        outlier_weight=0.0,
        existence_weight=0.0,
        offset_weight=0.0,
        repulsion_weight=0.0,
        distance_chunk_size=2,
    )

    expected_positive = F.binary_cross_entropy_with_logits(
        logits[:2],
        labels[:2],
    )
    expected_negative = F.binary_cross_entropy_with_logits(
        logits[2:],
        labels[2:],
    )
    torch.testing.assert_close(
        result.components["occupancy_positive_bce"],
        expected_positive,
    )
    torch.testing.assert_close(
        result.components["occupancy_negative_bce"],
        expected_negative,
    )
    expected_all = F.binary_cross_entropy_with_logits(logits, labels)
    torch.testing.assert_close(
        result.components["occupancy_bce"],
        expected_all,
    )
    torch.testing.assert_close(result.total, expected_all)


def test_query_field_occupancy_bce_preserves_rald_sample_ratio() -> None:
    labels = torch.cat((torch.ones(1), torch.zeros(15)))
    logits = torch.zeros(16, requires_grad=True)
    xyz = torch.arange(16, dtype=torch.float32)[:, None].repeat(1, 3)
    target = torch.cat((xyz[:4], torch.ones(4, 1)), dim=1)

    result = rald_query_field_loss(
        _loss_output(logits, xyz),
        labels,
        target,
        generated_point_count=16,
        geometry_weight=0.0,
        outlier_weight=0.0,
        existence_weight=0.0,
        offset_weight=0.0,
        repulsion_weight=0.0,
        distance_chunk_size=4,
    )
    result.total.backward()

    torch.testing.assert_close(result.total, torch.tensor(math.log(2.0)))
    assert logits.grad is not None
    assert float(logits.grad.sum()) > 0.0


def test_query_field_loss_gradients_reach_logits_and_generated_xyz() -> None:
    logits = torch.tensor(
        [0.1, -0.2, 0.3, -0.4],
        requires_grad=True,
    )
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    xyz = torch.tensor(
        [
            [0.2, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [3.2, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    target_xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    target = torch.cat((target_xyz, torch.ones(4, 1)), dim=1)

    result = rald_query_field_loss(
        _loss_output(logits, xyz),
        labels,
        target,
        generated_point_count=4,
        distance_chunk_size=2,
    )
    result.total.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == logits.numel()
    assert xyz.grad is not None
    assert torch.count_nonzero(xyz.grad) > 0
    assert set(result.distances) == {
        "prediction_to_target_m",
        "target_to_prediction_m",
        "nearest_other_m",
        "existence_target",
    }
