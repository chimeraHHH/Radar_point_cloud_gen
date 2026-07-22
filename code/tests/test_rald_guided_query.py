import torch

from eval.rald_guided_query import duplicate_report, nearest_other_distance
from losses.rald_guided_query import (
    rald_guided_geometry_loss,
    within_seed_repulsion,
)
from models.rald_guided_query import (
    RaLDGuidedQueryGenerator,
    query_templates,
    radar_guided_queries,
)
from scripts.compare_rald_guided_query import stage_a_decision


SPATIAL_SHAPE = (8, 5, 3)


def cube() -> torch.Tensor:
    values = torch.full((1, 64, *SPATIAL_SHAPE), 1e-4)
    values[:, 7, 2, 1, 1] = 12.0
    values[:, 23, 6, 3, 1] = 10.0
    values[:, 51, 4, 4, 2] = 8.0
    return values


def axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.linspace(0.0, 7.0, SPATIAL_SHAPE[0]),
        torch.linspace(-0.4, 0.4, SPATIAL_SHAPE[1]),
        torch.linspace(-0.2, 0.2, SPATIAL_SHAPE[2]),
    )


def tiny_model() -> RaLDGuidedQueryGenerator:
    range_m, azimuth, elevation = axes()
    return RaLDGuidedQueryGenerator(
        range_m,
        azimuth,
        elevation,
        log_center=0.0,
        log_scale=1.0,
        base_seed_count=2,
        queries_per_seed=10,
        latent_count=4,
        model_dim=16,
        depth=1,
        heads=4,
        head_dim=4,
        radar_base_channels=4,
        radar_spectral_channels=4,
        radar_encoded_shape=SPATIAL_SHAPE,
        radar_encoded_channels=4,
        radar_channel_multipliers=(1,),
        radar_blocks_per_level=1,
        offset_bounds_bins=(2.0, 1.0, 1.0),
        nms_kernel=(3, 3, 3),
    )


def test_radar_guided_queries_are_deterministic_full_spectrum_queries() -> None:
    measured = cube()
    first = radar_guided_queries(
        measured, base_seed_count=2, nms_kernel=(3, 3, 3)
    )
    second = radar_guided_queries(
        measured, base_seed_count=2, nms_kernel=(3, 3, 3)
    )

    assert first.coordinates_rae.shape == (1, 20, 3)
    assert first.local_spectrum.shape == (1, 20, 64)
    assert first.seed_index.unique().numel() == 2
    assert first.template_index.unique().numel() == 10
    torch.testing.assert_close(first.coordinates_rae, second.coordinates_rae)
    torch.testing.assert_close(first.local_spectrum, second.local_spectrum)
    assert query_templates(10).unique(dim=0).shape[0] == 10


def test_geometry_gradient_reaches_rald_and_full_raed_after_zero_head_update() -> None:
    torch.manual_seed(91)
    model = tiny_model()
    optimizer = torch.optim.SGD(model.geometry_parameters(), lr=0.05)
    with torch.no_grad():
        initial = model(cube())
        target_xyz = initial["anchor_xyz_m"][0] + torch.tensor([0.2, 0.0, 0.0])
        target = torch.cat((target_xyz, torch.ones(target_xyz.shape[0], 1)), dim=1)

    first = model(cube())
    first_loss = rald_guided_geometry_loss(
        first, target, queries_per_seed=10
    )
    first_loss.total.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.refiner.physical_head.parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = model(cube())
    second_loss = rald_guided_geometry_loss(
        second, target, queries_per_seed=10
    )
    second_loss.total.backward()
    for parameters in (
        model.radar_encoder.parameters(),
        model.spectrum_projection.parameters(),
    ):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in parameters
        )


def test_repulsion_and_duplicate_metrics_detect_collapsed_queries() -> None:
    separated = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    collapsed = torch.zeros(10, 3)
    assert within_seed_repulsion(
        separated, queries_per_seed=10
    ) < within_seed_repulsion(collapsed, queries_per_seed=10)
    report = duplicate_report(collapsed)
    assert report["duplicate_fraction_0p05m"] == 1.0
    assert torch.all(nearest_other_distance(collapsed) == 0.0)


def test_stage_a_gate_does_not_relax_outlier_threshold() -> None:
    run = {
        "metrics": {
            "validation": {
                "generated": {
                    "chamfer_m": {"median": 2.0},
                    "outlier_fraction_2m": {"mean": 0.24},
                    "completeness_mean_distance_m": {"median": 0.6},
                    "range_60_120m_completeness_mean_distance_m": {"mean": 7.5},
                },
                "duplicates": {
                    "duplicate_fraction_0p05m": {"mean": 0.05}
                },
                "confidence_mean": {"mean": 0.5},
            },
            "gradient_steps": [
                {
                    "gradients": {
                        "physical_head": 1.0,
                        "mixed_latent_and_query_decoder": 0.0,
                        "full_raed_radar_encoder": 0.0,
                        "local_64bin_spectrum_projection": 0.0,
                    }
                },
                {
                    "gradients": {
                        "physical_head": 1.0,
                        "mixed_latent_and_query_decoder": 1.0,
                        "full_raed_radar_encoder": 1.0,
                        "local_64bin_spectrum_projection": 1.0,
                    }
                },
            ],
        }
    }
    assert stage_a_decision(run)["passed"] is True
    run["metrics"]["validation"]["generated"]["outlier_fraction_2m"]["mean"] = 0.2501
    assert stage_a_decision(run)["passed"] is False
