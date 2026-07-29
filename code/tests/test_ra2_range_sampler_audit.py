import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audit_ra2_range_sampler as audit


def axes() -> SimpleNamespace:
    return SimpleNamespace(
        range_m=np.arange(120, dtype=np.float32),
        azimuth_rad=np.linspace(-0.5, 0.5, 16, dtype=np.float32),
        elevation_rad=np.linspace(-0.2, 0.2, 8, dtype=np.float32),
    )


def test_nearest_axis_indices_matches_left_tie_rule() -> None:
    axis = np.asarray([0.0, 1.0, 3.0])
    values = np.asarray([-1.0, 0.5, 2.0, 4.0])
    np.testing.assert_array_equal(
        audit.nearest_axis_indices(axis, values),
        np.asarray([0, 0, 1, 2]),
    )


def test_xyz_index_consistency_accepts_float32_half_bin_tie() -> None:
    test_axes = axes()
    radius = np.asarray([10.0], dtype=np.float32)
    azimuth = np.asarray(
        [
            0.5
            * (
                test_axes.azimuth_rad[7]
                + test_axes.azimuth_rad[8]
            )
        ],
        dtype=np.float32,
    )
    elevation = np.asarray([test_axes.elevation_rad[3]], dtype=np.float32)
    xyz = audit.rae_to_xyz(radius, azimuth, elevation).astype(np.float32)
    stored = np.asarray([[10, 7, 3]], dtype=np.int64)
    consistent = audit.xyz_index_consistency(
        xyz,
        stored,
        test_axes.range_m,
        test_axes.azimuth_rad,
        test_axes.elevation_rad,
    )
    assert consistent.tolist() == [True]


def test_partition_contract_rejects_cross_split_scene() -> None:
    manifest = {
        "frames": [
            {"partition": "train", "sequence": 1, "radar_index": index}
            for index in range(76)
        ]
        + [
            {"partition": "validation", "sequence": 2, "radar_index": index}
            for index in range(24)
        ]
    }
    split = {
        "gate_pass": True,
        "splits": {
            "train": {"sequences": [1]},
            "validation": {"sequences": [1, 2]},
            "test": {"sequences": [3]},
        },
    }
    _, failures = audit.validate_partition_contract(manifest, split)
    assert "train and validation scene sets overlap" in failures


def test_cache_audit_checks_metadata_geometry_and_range_coverage(
    tmp_path: Path,
) -> None:
    test_axes = axes()
    indices = np.asarray(
        [
            [10, 7, 3],
            [40, 8, 4],
            [70, 9, 4],
        ],
        dtype=np.int16,
    )
    radius = test_axes.range_m[indices[:, 0]]
    azimuth = test_axes.azimuth_rad[indices[:, 1]]
    elevation = test_axes.elevation_rad[indices[:, 2]]
    xyz = audit.rae_to_xyz(radius, azimuth, elevation).astype(np.float32)
    target = np.concatenate(
        (xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)),
        axis=1,
    )
    path = audit.cache_path(tmp_path, 1, 2)
    np.savez_compressed(
        path,
        cache_schema_version=np.asarray(1, dtype=np.int16),
        source_manifest_sha256=np.asarray(audit.FROZEN_MANIFEST_SHA256),
        source_commit=np.asarray("a" * 40),
        target_xyz_confidence=target,
        target_rae_index=indices,
    )
    report, failures = audit.audit_cache_frame(
        {"partition": "train", "sequence": 1, "radar_index": 2},
        cache_root=tmp_path,
        manifest_sha256=audit.FROZEN_MANIFEST_SHA256,
        axes=test_axes,
    )
    assert failures == []
    assert report["unique_target_cell_count_by_index_range"] == [1, 1, 1]
    assert report["xyz_to_index_reconstruction_mismatch_count"] == 0


def test_cache_audit_rejects_stale_manifest_metadata(tmp_path: Path) -> None:
    test_axes = axes()
    target = np.asarray([[10.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    path = audit.cache_path(tmp_path, 1, 2)
    np.savez_compressed(
        path,
        cache_schema_version=np.asarray(1, dtype=np.int16),
        source_manifest_sha256=np.asarray("stale"),
        source_commit=np.asarray("a" * 40),
        target_xyz_confidence=target,
        target_rae_index=np.asarray([[10, 7, 3]], dtype=np.int16),
    )
    _, failures = audit.audit_cache_frame(
        {"partition": "validation", "sequence": 1, "radar_index": 2},
        cache_root=tmp_path,
        manifest_sha256=audit.FROZEN_MANIFEST_SHA256,
        axes=test_axes,
    )
    assert any("source manifest hash differs" in failure for failure in failures)


def test_cache_audit_reports_missing_provenance_but_keeps_sampler_input(
    tmp_path: Path,
) -> None:
    test_axes = axes()
    indices = np.asarray([[10, 7, 3]], dtype=np.int16)
    radius = test_axes.range_m[indices[:, 0]]
    azimuth = test_axes.azimuth_rad[indices[:, 1]]
    elevation = test_axes.elevation_rad[indices[:, 2]]
    xyz = audit.rae_to_xyz(radius, azimuth, elevation).astype(np.float32)
    target = np.concatenate(
        (xyz, np.ones((1, 1), dtype=np.float32)),
        axis=1,
    )
    path = audit.cache_path(tmp_path, 1, 2)
    np.savez_compressed(
        path,
        target_xyz_confidence=target,
        target_rae_index=indices,
    )
    report, failures = audit.audit_cache_frame(
        {"partition": "validation", "sequence": 1, "radar_index": 2},
        cache_root=tmp_path,
        manifest_sha256=audit.FROZEN_MANIFEST_SHA256,
        axes=test_axes,
    )
    assert report["sampler_input_valid"] is True
    assert report["source_commit"] is None
    assert any("missing cache provenance arrays" in failure for failure in failures)


def test_pilot_manifest_comparison_ignores_snapshot_prefixes(
    tmp_path: Path,
) -> None:
    base = {
        "protocol": "g1_ra2_rald_wce_pilot_v1",
        "resume_contract": {
            "source_commit": "a" * 40,
            "input_hashes": {"manifest": "m"},
            "source_hashes": {"/snap/one/code/losses/x.py": "x"},
            "config": {"mode": "source_classwise"},
        },
    }
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(base), encoding="utf-8")
    changed = json.loads(json.dumps(base))
    changed["resume_contract"]["source_hashes"] = {
        "/snap/two/code/losses/x.py": "x"
    }
    changed["resume_contract"]["config"]["mode"] = "range_class_sampler"
    right.write_text(json.dumps(changed), encoding="utf-8")
    report, failures = audit.audit_pilot_manifests([left, right])
    assert failures == []
    assert report["run_count"] == 2
    assert all(
        run["cache_root_bound_in_resume_contract"] is False
        for run in report["runs"]
    )


def test_static_findings_cover_ranking_resume_and_auxiliary_losses() -> None:
    identifiers = {finding["id"] for finding in audit.static_findings()}
    assert {
        "pointwise_loss_vs_exact10k_ranking",
        "cache_and_cube_not_resume_bound",
        "formal_reference_resume_not_integrity_bound",
        "auxiliary_losses_not_range_normalized",
        "unequal_query_budget",
        "cuda_determinism_not_enforced",
    }.issubset(identifiers)
