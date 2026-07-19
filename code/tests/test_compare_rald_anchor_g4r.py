import pytest

from scripts.compare_rald_anchor_g4r import (
    COMMON_ARTIFACT_FIELDS,
    COMMON_SHA256_FIELDS,
    TEMPORAL_ARTIFACT_FIELDS,
    TEMPORAL_SHA256_FIELDS,
    configuration_hashes_complete,
    gate_decision,
    sha256,
)


def test_temporal_provenance_hashes_live_artifacts(tmp_path) -> None:
    configuration = {
        "point_count": 10_000,
        "partition": "validation",
        "source_commit": "a" * 40,
        "model_source_commit": "a" * 40,
        "fusion_mode": "latent",
        "strict_recurrent_rollout": True,
    }
    for index, (path_field, hash_field) in enumerate(
        COMMON_ARTIFACT_FIELDS + TEMPORAL_ARTIFACT_FIELDS
    ):
        artifact = tmp_path / f"artifact-{index}"
        artifact.write_bytes(f"artifact-{index}".encode())
        configuration[path_field] = str(artifact)
        configuration[hash_field] = sha256(artifact)
    for hash_field in COMMON_SHA256_FIELDS + TEMPORAL_SHA256_FIELDS:
        configuration.setdefault(hash_field, "b" * 64)

    assert configuration_hashes_complete(configuration, temporal=True)
    (tmp_path / "artifact-0").write_bytes(b"changed")
    assert not configuration_hashes_complete(configuration, temporal=True)


def endpoint(
    improvement_lower: float = 0.01,
    relative_upper: float = 0.01,
    retention_lower: float = 0.95,
) -> dict:
    return {
        "improvement_ci95": [improvement_lower, improvement_lower + 0.02],
        "relative_change_ci95": [-0.02, relative_upper],
        "retention_ratio_ci95": [retention_lower, retention_lower + 0.02],
    }


def passing_inputs() -> tuple[dict, dict, dict]:
    versus_t0 = {
        "ego_aligned_matched_distance_m": endpoint(),
        "occupancy_flicker": endpoint(),
        "geometry_chamfer_m": endpoint(relative_upper=0.02),
        "local_spectrum_kl": endpoint(relative_upper=0.05),
        "circular_w1_mps": endpoint(relative_upper=0.05),
    }
    versus_history = {
        "geometry_chamfer_m": endpoint(),
        "local_spectrum_kl": endpoint(),
        "circular_w1_mps": endpoint(improvement_lower=-0.01),
    }
    step25_versus_t0 = {
        "confidence_mean": endpoint(retention_lower=0.90),
        "covered_cell_count": endpoint(retention_lower=0.90),
    }
    return versus_t0, versus_history, step25_versus_t0


def test_g4r_gate_accepts_all_preregistered_boundaries() -> None:
    inputs = passing_inputs()

    decision = gate_decision(*inputs, complete_provenance=True)

    assert decision["g4r_passed"] is True
    assert all(value is True for value in decision.values())


@pytest.mark.parametrize(
    ("group", "endpoint_name", "field", "value"),
    [
        ("t0", "ego_aligned_matched_distance_m", "improvement_ci95", [0.0, 0.1]),
        ("t0", "occupancy_flicker", "improvement_ci95", [-0.01, 0.1]),
        ("history", "geometry_chamfer_m", "improvement_ci95", [0.0, 0.1]),
        ("t0", "geometry_chamfer_m", "relative_change_ci95", [-0.01, 0.0201]),
        ("t0", "local_spectrum_kl", "relative_change_ci95", [-0.01, 0.0501]),
        ("t0", "circular_w1_mps", "relative_change_ci95", [-0.01, 0.0501]),
        ("step25", "confidence_mean", "retention_ratio_ci95", [0.8999, 1.0]),
        ("step25", "covered_cell_count", "retention_ratio_ci95", [0.8999, 1.0]),
    ],
)
def test_g4r_gate_rejects_each_required_metric_boundary(
    group: str,
    endpoint_name: str,
    field: str,
    value: list[float],
) -> None:
    versus_t0, versus_history, step25_versus_t0 = passing_inputs()
    groups = {
        "t0": versus_t0,
        "history": versus_history,
        "step25": step25_versus_t0,
    }
    groups[group][endpoint_name][field] = value

    decision = gate_decision(
        versus_t0,
        versus_history,
        step25_versus_t0,
        complete_provenance=True,
    )

    assert decision["g4r_passed"] is False


def test_g4r_gate_requires_one_history_spectrum_gain_and_provenance() -> None:
    versus_t0, versus_history, step25_versus_t0 = passing_inputs()
    versus_history["local_spectrum_kl"]["improvement_ci95"] = [-0.01, 0.1]

    no_spectrum_gain = gate_decision(
        versus_t0,
        versus_history,
        step25_versus_t0,
        complete_provenance=True,
    )
    no_provenance = gate_decision(
        *passing_inputs(), complete_provenance=False
    )

    assert no_spectrum_gain["spectrum_improves_vs_history_aggregation"] is False
    assert no_spectrum_gain["g4r_passed"] is False
    assert no_provenance["complete_provenance"] is False
    assert no_provenance["g4r_passed"] is False
