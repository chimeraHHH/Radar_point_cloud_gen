import torch

from eval.dense_geometry import (
    aggregate_geometry_reports,
    geometry_report,
)


def test_far_completeness_is_reported_without_far_predictions() -> None:
    prediction = torch.tensor([[20.0, 0.0, 0.0]])
    target = torch.tensor([[80.0, 0.0, 0.0]])

    report = geometry_report(prediction, target)

    assert report["range_60_120m_completeness_mean_distance_m"] == 60.0
    assert report["range_60_120m_fscore_1m"] == 0.0
    assert "range_60_120m_precision_mean_distance_m" not in report


def test_far_completeness_aggregate_keeps_every_target_bearing_frame() -> None:
    reports = [
        geometry_report(
            torch.tensor([[20.0 + float(index), 0.0, 0.0]]),
            torch.tensor([[80.0, 0.0, 0.0]]),
        )
        for index in range(24)
    ]

    aggregate = aggregate_geometry_reports(reports)

    assert aggregate[
        "range_60_120m_completeness_mean_distance_m"
    ]["sample_count"] == 24
