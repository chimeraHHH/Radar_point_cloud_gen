from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts import stda_f0_metric_phase as metric_phase


EXPECTED_PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
EXPECTED_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
EXPECTED_DENSE_GEOMETRY_SHA256 = (
    "e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68"
)


def test_metric_entrypoint_imports_under_isolated_no_site_interpreter() -> None:
    script = Path(metric_phase.__file__).resolve()
    completed = subprocess.run(
        (sys.executable, "-I", "-S", str(script), "--help"),
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stdout
    assert "--expected-gpu-uuid" in completed.stdout


def _namespace(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "decision_export": None,
        "decision_sha256": None,
        "pointwise_export": None,
        "pointwise_sha256": None,
        "greedy_export": None,
        "greedy_sha256": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _save_npy(path: Path, array: np.ndarray) -> str:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_binds_frozen_protocol_evaluator_and_freeze_commit() -> None:
    assert metric_phase.PROTOCOL_SHA256 == EXPECTED_PROTOCOL_SHA256
    assert metric_phase.PROTOCOL_FREEZE_COMMIT == EXPECTED_FREEZE_COMMIT
    assert metric_phase.DENSE_GEOMETRY_SHA256 == EXPECTED_DENSE_GEOMETRY_SHA256
    assert metric_phase.ARM_NAMES == ("decision", "pointwise", "greedy")


def test_source_has_no_structure_fitting_or_solver_imports() -> None:
    source = inspect.getsource(metric_phase)
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    forbidden = (
        "stda_f0_fit",
        "stda_f0_round",
        "stda_f0_structure",
        "stda_f0_verify",
    )
    assert not any(any(token in name for token in forbidden) for name in imported)


def test_archived_evaluator_is_hashed_before_dynamic_import() -> None:
    source = inspect.getsource(metric_phase.load_archived_dense_geometry)
    assert source.index("sha256_bytes(payload)") < source.index(
        "importlib.import_module"
    )
    assert "in sys.modules" in source
    assert "post_import_payload" in source
    assert metric_phase.DENSE_GEOMETRY_MODULE == "eval.dense_geometry"


def test_archived_metric_calls_preserve_default_chunking_and_separate_distance() -> None:
    source = inspect.getsource(metric_phase.evaluate_arm)
    tree = ast.parse(source)
    geometry_calls = []
    nearest_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "geometry_report":
            geometry_calls.append(node)
        if node.func.attr == "nearest_distance":
            nearest_calls.append(node)
    assert len(geometry_calls) == 1
    assert len(geometry_calls[0].args) == 3
    assert geometry_calls[0].keywords == []
    assert len(nearest_calls) == 1
    assert len(nearest_calls[0].args) == 2
    assert nearest_calls[0].keywords == []
    assert source.count("torch.cuda.synchronize(device)") == 2


def test_canonical_json_uses_exact_frozen_encoding() -> None:
    document = {"z": True, "ascii": "ok", "list": [2, 1]}
    expected = b'{"ascii":"ok","list":[2,1],"z":true}\n'
    assert metric_phase.canonical_json_bytes(document) == expected
    assert json.loads(expected) == document
    with pytest.raises(ValueError):
        metric_phase.canonical_json_bytes({"bad": float("nan")})


def test_float32_npy_snapshot_is_hash_bound_and_read_only(tmp_path: Path) -> None:
    target_path = tmp_path / "target.npy"
    target = np.asarray(
        [[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.75]],
        dtype="<f4",
    )
    digest = _save_npy(target_path, target)

    snapshot = metric_phase.load_float32_npy_snapshot(
        target_path,
        digest,
        columns=4,
        label="original target",
    )

    np.testing.assert_array_equal(snapshot.array, target)
    assert snapshot.array.dtype.str == "<f4"
    assert not snapshot.array.flags.writeable
    assert snapshot.file_sha256 == digest
    assert snapshot.file_size_bytes == target_path.stat().st_size
    assert snapshot.array_sha256 == hashlib.sha256(
        target.tobytes(order="C")
    ).hexdigest()


def test_snapshot_rejects_wrong_sha_dtype_trailing_bytes_and_negative_mass(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.npy"
    valid_sha = _save_npy(
        valid_path,
        np.asarray([[1.0, 2.0, 3.0]], dtype="<f4"),
    )
    with pytest.raises(metric_phase.MetricPhaseError, match="mismatch"):
        metric_phase.load_float32_npy_snapshot(
            valid_path,
            "0" * 64,
            columns=3,
            label="decision export",
        )

    big_endian_path = tmp_path / "big_endian.npy"
    big_endian_sha = _save_npy(
        big_endian_path,
        np.asarray([[1.0, 2.0, 3.0]], dtype=">f4"),
    )
    with pytest.raises(metric_phase.MetricPhaseError, match="little-endian"):
        metric_phase.load_float32_npy_snapshot(
            big_endian_path,
            big_endian_sha,
            columns=3,
            label="decision export",
        )

    trailing_path = tmp_path / "trailing.npy"
    _save_npy(trailing_path, np.asarray([[1.0, 2.0, 3.0]], dtype="<f4"))
    with trailing_path.open("ab") as handle:
        handle.write(b"unexpected")
    trailing_sha = hashlib.sha256(trailing_path.read_bytes()).hexdigest()
    with pytest.raises(metric_phase.MetricPhaseError, match="trailing bytes"):
        metric_phase.load_float32_npy_snapshot(
            trailing_path,
            trailing_sha,
            columns=3,
            label="decision export",
        )

    negative_path = tmp_path / "negative.npy"
    negative_sha = _save_npy(
        negative_path,
        np.asarray([[1.0, 2.0, 3.0, -0.5]], dtype="<f4"),
    )
    with pytest.raises(metric_phase.MetricPhaseError, match="negative confidence"):
        metric_phase.load_float32_npy_snapshot(
            negative_path,
            negative_sha,
            columns=4,
            label="original target",
        )


def test_arm_pairs_are_optional_but_complete_and_frozen_ordered(tmp_path: Path) -> None:
    arms = metric_phase.collect_arm_inputs(
        _namespace(
            decision_export=tmp_path / "decision.npy",
            decision_sha256="1" * 64,
            greedy_export=tmp_path / "greedy.npy",
            greedy_sha256="3" * 64,
        )
    )
    assert [arm.name for arm in arms] == ["decision", "greedy"]
    assert [arm.expected_sha256 for arm in arms] == ["1" * 64, "3" * 64]

    with pytest.raises(metric_phase.MetricPhaseError, match="supplied together"):
        metric_phase.collect_arm_inputs(
            _namespace(pointwise_export=tmp_path / "pointwise.npy")
        )
    with pytest.raises(metric_phase.MetricPhaseError, match="At least one"):
        metric_phase.collect_arm_inputs(_namespace())


def test_range_metrics_use_positive_confidence_and_half_open_strata() -> None:
    target_xyz = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
            [119.0, 0.0, 0.0],
            [120.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    confidence = torch.tensor([1.0, 3.0, 0.0, 1.0, 7.0], dtype=torch.float32)
    distance = torch.tensor([0.5, 2.0, 0.2, 1.0, 0.0], dtype=torch.float32)

    report = metric_phase.range_metrics_from_distances(
        target_xyz,
        confidence,
        distance,
    )

    assert report["range_0_30"] == {
        "applicable": True,
        "completeness_mean_distance_m": 0.5,
        "positive_confidence_row_count": 1,
        "positive_confidence_weight": 1.0,
        "recall_1m": 1.0,
    }
    assert report["range_30_60"]["completeness_mean_distance_m"] == 2.0
    assert report["range_30_60"]["recall_1m"] == 0.0
    assert report["range_60_120"]["positive_confidence_row_count"] == 1
    assert report["range_60_120"]["positive_confidence_weight"] == 1.0
    assert report["range_60_120"]["recall_1m"] == 1.0


def test_nonbearing_range_is_nonapplicable_and_neutral_in_semantics() -> None:
    ranges = {
        "range_0_30": {
            "applicable": True,
            "completeness_mean_distance_m": 0.75,
            "recall_1m": 0.80,
        },
        "range_30_60": {
            "applicable": True,
            "completeness_mean_distance_m": 1.01,
            "recall_1m": 0.79,
        },
        "range_60_120": {
            "applicable": False,
            "completeness_mean_distance_m": None,
            "recall_1m": None,
        },
    }
    semantics = metric_phase.semantic_metric_booleans(
        {"chamfer_m": 0.8, "outlier_fraction_2m": 0.05},
        ranges,
    )

    assert len(semantics) == 8
    assert semantics["chamfer_le_0p8"] == {"applicable": True, "passed": True}
    assert semantics["outlier_le_0p05"] == {
        "applicable": True,
        "passed": True,
    }
    assert semantics["range_30_60_completeness_le_1m"]["passed"] is False
    assert semantics["range_30_60_recall_1m_ge_0p8"]["passed"] is False
    assert semantics["range_60_120_completeness_le_1m"] == {
        "applicable": False,
        "passed": True,
    }
    assert semantics["range_60_120_recall_1m_ge_0p8"] == {
        "applicable": False,
        "passed": True,
    }
    assert all(
        type(value) is bool
        for gate in semantics.values()
        for value in gate.values()
    )


def test_gpu_identity_normalization_is_exact() -> None:
    assert metric_phase.normalize_gpu_uuid(
        "GPU-000B6236-3632-A001-9667-1F02CBB61C8B"
    ) == "GPU-000b6236-3632-a001-9667-1f02cbb61c8b"
    assert metric_phase.normalize_pci_bus_id("0000:d1:00.0") == (
        "00000000:D1:00.0"
    )
    with pytest.raises(metric_phase.MetricPhaseError):
        metric_phase.normalize_gpu_uuid("000b6236")
    with pytest.raises(metric_phase.MetricPhaseError):
        metric_phase.normalize_pci_bus_id("d1:00")


def test_canonical_output_is_exclusive_fsynced_and_replayable(tmp_path: Path) -> None:
    output = tmp_path / "metric.json"
    document = {"schema": metric_phase.SCHEMA, "verified": True}

    digest, size = metric_phase.write_canonical_json_exclusive(output, document)

    expected = metric_phase.canonical_json_bytes(document)
    assert output.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert size == len(expected)
    with pytest.raises(metric_phase.MetricPhaseError, match="already exists"):
        metric_phase.write_canonical_json_exclusive(output, document)


def test_parser_requires_hashes_gpu_identity_and_output() -> None:
    parser = metric_phase.build_parser()
    parsed = parser.parse_args(
        [
            "--frame-key",
            "seq01/radar00002",
            "--source-commit",
            "a" * 40,
            "--target",
            "/tmp/target.npy",
            "--target-sha256",
            "b" * 64,
            "--decision-export",
            "/tmp/decision.npy",
            "--decision-sha256",
            "c" * 64,
            "--expected-gpu-uuid",
            "GPU-000b6236-3632-a001-9667-1f02cbb61c8b",
            "--expected-gpu-pci",
            "00000000:D1:00.0",
            "--expected-gpu-name",
            "NVIDIA H200 NVL",
            "--output",
            "/tmp/metric.json",
        ]
    )
    assert parsed.frame_key == "seq01/radar00002"
    assert parsed.target_sha256 == "b" * 64
    assert parsed.expected_gpu_name == "NVIDIA H200 NVL"
    assert parsed.output == Path("/tmp/metric.json")


@pytest.mark.skipif(
    os.environ.get("STDA_F0_CUDA_SMOKE") != "1",
    reason="explicit synthetic H200 smoke only",
)
def test_synthetic_single_h200_metric_child(tmp_path: Path) -> None:
    expected_uuid = os.environ["STDA_EXPECTED_GPU_UUID"]
    expected_pci = os.environ["STDA_EXPECTED_GPU_PCI"]
    expected_name = os.environ.get("STDA_EXPECTED_GPU_NAME", "NVIDIA H200 NVL")
    target = np.asarray(
        [
            [10.0, 0.0, 0.0, 1.0],
            [40.0, 0.0, 0.0, 2.0],
            [70.0, 0.0, 0.0, 1.0],
        ],
        dtype="<f4",
    )
    exports = {
        "decision": target[:, :3].copy(),
        "pointwise": target[:, :3]
        + np.asarray([0.25, 0.0, 0.0], dtype="<f4"),
        "greedy": target[:2, :3].copy(),
    }
    target_path = tmp_path / "target.npy"
    target_sha = _save_npy(target_path, target)
    arm_paths: dict[str, Path] = {}
    arm_hashes: dict[str, str] = {}
    for name, export in exports.items():
        path = tmp_path / f"{name}.npy"
        arm_paths[name] = path
        arm_hashes[name] = _save_npy(path, export)
    output = tmp_path / "metric.json"
    arguments = argparse.Namespace(
        frame_key="seq01/radar00002",
        source_commit=EXPECTED_FREEZE_COMMIT,
        target=target_path,
        target_sha256=target_sha,
        decision_export=arm_paths["decision"],
        decision_sha256=arm_hashes["decision"],
        pointwise_export=arm_paths["pointwise"],
        pointwise_sha256=arm_hashes["pointwise"],
        greedy_export=arm_paths["greedy"],
        greedy_sha256=arm_hashes["greedy"],
        expected_gpu_uuid=expected_uuid,
        expected_gpu_pci=expected_pci,
        expected_gpu_name=expected_name,
        output=output,
    )

    report, output_sha, output_size = metric_phase.run(arguments)

    assert report["available_arms"] == ["decision", "pointwise", "greedy"]
    assert report["gpu_provenance"]["logical_device"] == "cuda:0"
    assert report["gpu_provenance"]["physical_identity_match"] is True
    assert report["runtime"]["metric_ns"] >= sum(
        report["runtime"]["arm_metric_ns"].values()
    )
    assert set(report["metrics"]) == {"decision", "pointwise", "greedy"}
    assert all(
        arm["float32_cuda"] is True for arm in report["metrics"].values()
    )
    assert output_sha == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output_size == output.stat().st_size
    assert output.read_bytes() == metric_phase.canonical_json_bytes(report)
