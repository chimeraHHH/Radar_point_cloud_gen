#!/usr/bin/env python3
"""Single-CUDA-child geometry metrics for frozen STDA-F0 exports."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from types import ModuleType
from typing import Any, Mapping


def _bootstrap_isolated_site_packages() -> str | None:
    """Expose the pinned environment packages without importing ``site``."""

    if not sys.flags.no_site:
        return None
    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not site_packages.is_dir():
        raise RuntimeError(f"isolated site-packages path is absent: {site_packages}")
    resolved = str(site_packages.resolve())
    if resolved not in sys.path:
        sys.path.append(resolved)
    return resolved


ISOLATED_SITE_PACKAGES = _bootstrap_isolated_site_packages()


import numpy as np
import torch


SCHEMA = "stda_f0_metric_phase_v1"
PROTOCOL = "stda_f0_sparse_target_demand_assignment"
PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)
PROTOCOL_FREEZE_COMMIT = "eb839e1e806c44dd1668051085a307faf4fe83a0"
DENSE_GEOMETRY_SHA256 = (
    "e1f970c1e827c7801c7c9018378603b89a7709b0b4bf261e740eab1575631e68"
)
DENSE_GEOMETRY_RELATIVE_PATH = Path("code/eval/dense_geometry.py")
DENSE_GEOMETRY_MODULE = "eval.dense_geometry"

ARM_NAMES = ("decision", "pointwise", "greedy")
RANGE_STRATA = (
    ("range_0_30", 0.0, 30.0),
    ("range_30_60", 30.0, 60.0),
    ("range_60_120", 60.0, 120.0),
)
CHAMFER_GATE_M = 0.8
OUTLIER_GATE = 0.05
COMPLETENESS_GATE_M = 1.0
RECALL_GATE = 0.80
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FRAME_KEY_RE = re.compile(r"^seq[0-9]{2}/radar[0-9]{5}$")
PCI_RE = re.compile(
    r"^(?P<domain>[0-9a-fA-F]{4,8}):(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\.(?P<function>[0-7])$"
)
GPU_UUID_RE = re.compile(
    r"^GPU-(?P<a>[0-9a-fA-F]{8})-(?P<b>[0-9a-fA-F]{4})-"
    r"(?P<c>[0-9a-fA-F]{4})-(?P<d>[0-9a-fA-F]{4})-"
    r"(?P<e>[0-9a-fA-F]{12})$"
)


class MetricPhaseError(RuntimeError):
    """Raised when a frozen metric-child contract is violated."""


@dataclass(frozen=True)
class ArraySnapshot:
    """One hash-verified immutable NPY payload."""

    path: Path
    file_sha256: str
    file_size_bytes: int
    device_id: int
    inode: int
    array: np.ndarray
    array_sha256: str

    def evidence(self) -> dict[str, object]:
        return {
            "absolute_path": str(self.path),
            "array_sha256": self.array_sha256,
            "device_id": self.device_id,
            "dtype": self.array.dtype.str,
            "file_sha256": self.file_sha256,
            "file_sha256_verified": True,
            "file_size_bytes": self.file_size_bytes,
            "inode": self.inode,
            "shape": [int(value) for value in self.array.shape],
        }


@dataclass(frozen=True)
class ArmInput:
    """One available matched-arm export and its caller-bound file hash."""

    name: str
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class CudaIdentity:
    """Observed identity of the sole CUDA-visible physical device."""

    uuid: str
    pci_bus_id: str
    name: str
    driver_version: int
    total_memory_bytes: int
    compute_capability: tuple[int, int]
    cuda_visible_devices: str


class _CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    """Return the exact Section 14 canonical JSON byte representation."""

    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise MetricPhaseError(f"{label} must be a lowercase SHA-256 digest")
    return value


def normalize_gpu_uuid(value: str) -> str:
    match = GPU_UUID_RE.fullmatch(value)
    if match is None:
        raise MetricPhaseError("GPU UUID must use NVIDIA GPU-xxxxxxxx-... syntax")
    groups = [match.group(key).lower() for key in ("a", "b", "c", "d", "e")]
    return "GPU-" + "-".join(groups)


def normalize_pci_bus_id(value: str) -> str:
    match = PCI_RE.fullmatch(value)
    if match is None:
        raise MetricPhaseError("GPU PCI ID must use domain:bus:device.function syntax")
    domain = int(match.group("domain"), 16)
    bus = int(match.group("bus"), 16)
    device = int(match.group("device"), 16)
    function = int(match.group("function"), 10)
    return f"{domain:08X}:{bus:02X}:{device:02X}.{function}"


def _read_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    absolute = path.expanduser()
    if not absolute.is_absolute():
        raise MetricPhaseError(f"Input path must be absolute: {path}")
    absolute = absolute.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MetricPhaseError(f"Input is not a regular file: {absolute}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise MetricPhaseError(f"Input changed while being read: {absolute}")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise MetricPhaseError(f"Short read for input: {absolute}")
    return payload, before


def _load_npy_payload(payload: bytes, label: str) -> np.ndarray:
    if not payload.startswith(b"\x93NUMPY"):
        raise MetricPhaseError(f"{label} is not an NPY file")
    stream = io.BytesIO(payload)
    try:
        loaded = np.load(stream, allow_pickle=False)
    except (OSError, TypeError, ValueError) as error:
        raise MetricPhaseError(f"{label} is not a valid non-pickle NPY file") from error
    if not isinstance(loaded, np.ndarray):
        raise MetricPhaseError(f"{label} must contain one NPY array")
    if stream.tell() != len(payload):
        raise MetricPhaseError(f"{label} contains trailing bytes")
    if loaded.dtype.str != "<f4":
        raise MetricPhaseError(f"{label} must have exact little-endian float32 dtype")
    if not loaded.flags.c_contiguous:
        raise MetricPhaseError(f"{label} must use C-contiguous NPY storage")
    array = np.array(loaded, dtype="<f4", order="C", copy=True)
    array.setflags(write=False)
    return array


def load_float32_npy_snapshot(
    path: Path,
    expected_sha256: str,
    *,
    columns: int,
    label: str,
) -> ArraySnapshot:
    """Read exact hashed NPY bytes once and expose an immutable float32 array."""

    expected = _validate_sha256(expected_sha256, f"{label} SHA-256")
    payload, file_stat = _read_regular_file(path)
    observed = sha256_bytes(payload)
    if observed != expected:
        raise MetricPhaseError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    array = _load_npy_payload(payload, label)
    if array.ndim != 2 or array.shape[1] != columns or array.shape[0] == 0:
        raise MetricPhaseError(
            f"{label} must be non-empty with shape (N,{columns}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise MetricPhaseError(f"{label} contains non-finite float32 values")
    if columns == 4:
        if np.any(array[:, 3] < 0.0):
            raise MetricPhaseError("Original target contains negative confidence")
        if not np.any(array[:, 3] > 0.0):
            raise MetricPhaseError("Original target has no positive confidence")
    return ArraySnapshot(
        path=path.expanduser().resolve(strict=True),
        file_sha256=observed,
        file_size_bytes=int(file_stat.st_size),
        device_id=int(file_stat.st_dev),
        inode=int(file_stat.st_ino),
        array=array,
        array_sha256=sha256_bytes(array.tobytes(order="C")),
    )


def collect_arm_inputs(arguments: argparse.Namespace) -> tuple[ArmInput, ...]:
    """Validate optional path/hash pairs and preserve the frozen arm order."""

    arms: list[ArmInput] = []
    for name in ARM_NAMES:
        path = getattr(arguments, f"{name}_export")
        digest = getattr(arguments, f"{name}_sha256")
        if (path is None) != (digest is None):
            raise MetricPhaseError(
                f"--{name}-export and --{name}-sha256 must be supplied together"
            )
        if path is not None and digest is not None:
            arms.append(
                ArmInput(
                    name=name,
                    path=path,
                    expected_sha256=_validate_sha256(
                        digest, f"{name} export SHA-256"
                    ),
                )
            )
    if not arms:
        raise MetricPhaseError("At least one matched-arm export is required")
    return tuple(arms)


def _assert_distinct_paths(
    output_path: Path,
    target: ArraySnapshot,
    arms: Mapping[str, ArraySnapshot],
) -> None:
    if not output_path.expanduser().is_absolute():
        raise MetricPhaseError("Output path must be absolute")
    resolved_output = output_path.expanduser().resolve(strict=False)
    snapshots = [("target", target), *arms.items()]
    path_keys: set[Path] = set()
    for label, snapshot in snapshots:
        if snapshot.path == resolved_output:
            raise MetricPhaseError(f"Output aliases {label} input path")
        if snapshot.path in path_keys:
            raise MetricPhaseError("Metric inputs must use distinct absolute paths")
        path_keys.add(snapshot.path)


def _driver_call(result: int, operation: str) -> None:
    if result != 0:
        raise MetricPhaseError(f"CUDA driver call {operation} failed with code {result}")


def _cuda_driver_identity(logical_ordinal: int = 0) -> tuple[str, str, int]:
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise MetricPhaseError("Unable to load libcuda.so.1") from error
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceGetUuid.argtypes = [ctypes.POINTER(_CUuuid), ctypes.c_int]
    driver.cuDeviceGetUuid.restype = ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int

    _driver_call(driver.cuInit(0), "cuInit")
    device = ctypes.c_int()
    _driver_call(
        driver.cuDeviceGet(ctypes.byref(device), logical_ordinal), "cuDeviceGet"
    )
    uuid = _CUuuid()
    _driver_call(
        driver.cuDeviceGetUuid(ctypes.byref(uuid), device.value), "cuDeviceGetUuid"
    )
    pci_buffer = ctypes.create_string_buffer(32)
    _driver_call(
        driver.cuDeviceGetPCIBusId(pci_buffer, len(pci_buffer), device.value),
        "cuDeviceGetPCIBusId",
    )
    version = ctypes.c_int()
    _driver_call(
        driver.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion"
    )
    raw = bytes(uuid.bytes).hex()
    observed_uuid = normalize_gpu_uuid(
        f"GPU-{raw[:8]}-{raw[8:12]}-{raw[12:16]}-"
        f"{raw[16:20]}-{raw[20:]}"
    )
    observed_pci = normalize_pci_bus_id(pci_buffer.value.decode("ascii"))
    return observed_uuid, observed_pci, int(version.value)


def bind_single_h200(
    *,
    expected_uuid: str,
    expected_pci_bus_id: str,
    expected_name: str,
) -> tuple[torch.device, CudaIdentity]:
    """Bind cuda:0 to the one caller-authorized physical H200 identity."""

    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise MetricPhaseError("CUDA_DEVICE_ORDER must equal PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible or "," in visible or visible.strip() != visible:
        raise MetricPhaseError("Exactly one CUDA_VISIBLE_DEVICES token is required")
    normalized_expected_uuid = normalize_gpu_uuid(expected_uuid)
    normalized_expected_pci = normalize_pci_bus_id(expected_pci_bus_id)
    if visible.isdecimal():
        if visible not in {"0", "2"}:
            raise MetricPhaseError("Only physical H200 GPU 0 or GPU 2 is allowed")
    elif visible.startswith("GPU-"):
        if normalize_gpu_uuid(visible) != normalized_expected_uuid:
            raise MetricPhaseError("Visible GPU UUID token does not match expectation")
    else:
        raise MetricPhaseError("CUDA_VISIBLE_DEVICES must be GPU 0, GPU 2, or a GPU UUID")
    if "H200" not in expected_name.upper():
        raise MetricPhaseError("Expected GPU name is not an H200")
    if not torch.cuda.is_available():
        raise MetricPhaseError("STDA metric child requires CUDA")
    if torch.cuda.device_count() != 1:
        raise MetricPhaseError("STDA metric child requires one visible CUDA device")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.current_device() != 0:
        raise MetricPhaseError("The sole visible H200 must map to cuda:0")
    observed_name = torch.cuda.get_device_name(device)
    observed_uuid, observed_pci, driver_version = _cuda_driver_identity(0)
    if observed_name != expected_name:
        raise MetricPhaseError(
            f"GPU name mismatch: expected {expected_name}, observed {observed_name}"
        )
    if observed_uuid != normalized_expected_uuid:
        raise MetricPhaseError(
            f"GPU UUID mismatch: expected {normalized_expected_uuid}, "
            f"observed {observed_uuid}"
        )
    if observed_pci != normalized_expected_pci:
        raise MetricPhaseError(
            f"GPU PCI mismatch: expected {normalized_expected_pci}, "
            f"observed {observed_pci}"
        )
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    return device, CudaIdentity(
        uuid=observed_uuid,
        pci_bus_id=observed_pci,
        name=observed_name,
        driver_version=driver_version,
        total_memory_bytes=int(properties.total_memory),
        compute_capability=(int(capability[0]), int(capability[1])),
        cuda_visible_devices=visible,
    )


def load_archived_dense_geometry(repo_root: Path) -> ModuleType:
    """Hash the archived evaluator before importing it into this fresh child."""

    evaluator_path = (repo_root / DENSE_GEOMETRY_RELATIVE_PATH).resolve(strict=True)
    payload, _ = _read_regular_file(evaluator_path)
    observed = sha256_bytes(payload)
    if observed != DENSE_GEOMETRY_SHA256:
        raise MetricPhaseError(
            "Archived dense_geometry.py SHA-256 mismatch: "
            f"expected {DENSE_GEOMETRY_SHA256}, observed {observed}"
        )
    if DENSE_GEOMETRY_MODULE in sys.modules:
        raise MetricPhaseError("Archived dense_geometry.py was imported before verification")
    code_root = str((repo_root / "code").resolve(strict=True))
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    importlib.invalidate_caches()
    module = importlib.import_module(DENSE_GEOMETRY_MODULE)
    module_path = Path(module.__file__ or "").resolve(strict=True)
    if module_path != evaluator_path:
        raise MetricPhaseError(
            f"Imported evaluator path mismatch: {module_path} vs {evaluator_path}"
        )
    post_import_payload, _ = _read_regular_file(evaluator_path)
    if sha256_bytes(post_import_payload) != DENSE_GEOMETRY_SHA256:
        raise MetricPhaseError("Archived evaluator changed during import")
    return module


def range_metrics_from_distances(
    target_xyz: torch.Tensor,
    target_weight: torch.Tensor,
    target_to_prediction: torch.Tensor,
) -> dict[str, dict[str, object]]:
    """Compute positive-confidence weighted range completeness and recall."""

    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise MetricPhaseError("Target XYZ tensor must have shape (N,3)")
    if target_weight.shape != (target_xyz.shape[0],):
        raise MetricPhaseError("Target confidence shape mismatch")
    if target_to_prediction.shape != target_weight.shape:
        raise MetricPhaseError("Target-to-prediction distance shape mismatch")
    if not (
        target_xyz.dtype == torch.float32
        and target_weight.dtype == torch.float32
        and target_to_prediction.dtype == torch.float32
    ):
        raise MetricPhaseError("Range metrics require float32 tensors")
    target_range = torch.linalg.vector_norm(target_xyz, dim=1)
    positive = target_weight > 0.0
    records: dict[str, dict[str, object]] = {}
    for key, lower, upper in RANGE_STRATA:
        mask = positive & (target_range >= lower) & (target_range < upper)
        positive_row_count = int(mask.sum().item())
        if positive_row_count == 0:
            records[key] = {
                "applicable": False,
                "completeness_mean_distance_m": None,
                "positive_confidence_row_count": 0,
                "positive_confidence_weight": 0.0,
                "recall_1m": None,
            }
            continue
        weights = target_weight[mask]
        distances = target_to_prediction[mask]
        weight_sum = weights.sum()
        weight_value = float(weight_sum.item())
        if not math.isfinite(weight_value) or weight_value <= 0.0:
            raise MetricPhaseError(f"Invalid positive confidence in {key}")
        completeness = float(((distances * weights).sum() / weight_sum).item())
        recall = float(
            (
                ((distances <= COMPLETENESS_GATE_M).to(weights) * weights).sum()
                / weight_sum
            ).item()
        )
        if not math.isfinite(completeness) or not math.isfinite(recall):
            raise MetricPhaseError(f"Non-finite range metric in {key}")
        records[key] = {
            "applicable": True,
            "completeness_mean_distance_m": completeness,
            "positive_confidence_row_count": positive_row_count,
            "positive_confidence_weight": weight_value,
            "recall_1m": recall,
        }
    return records


def semantic_metric_booleans(
    geometry: Mapping[str, float],
    ranges: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, bool]]:
    """Map exact metric predicates to the frozen shared-vector key names."""

    chamfer = float(geometry["chamfer_m"])
    outlier = float(geometry["outlier_fraction_2m"])
    if not math.isfinite(chamfer) or not math.isfinite(outlier):
        raise MetricPhaseError("Archived geometry report contains non-finite gates")
    result: dict[str, dict[str, bool]] = {
        "chamfer_le_0p8": {
            "applicable": True,
            "passed": bool(chamfer <= CHAMFER_GATE_M),
        },
        "outlier_le_0p05": {
            "applicable": True,
            "passed": bool(outlier <= OUTLIER_GATE),
        },
    }
    for key, _, _ in RANGE_STRATA:
        record = ranges[key]
        applicable = bool(record["applicable"])
        if applicable:
            completeness = float(record["completeness_mean_distance_m"])
            recall = float(record["recall_1m"])
            completeness_passed = bool(completeness <= COMPLETENESS_GATE_M)
            recall_passed = bool(recall >= RECALL_GATE)
        else:
            completeness_passed = True
            recall_passed = True
        result[f"{key}_completeness_le_1m"] = {
            "applicable": applicable,
            "passed": completeness_passed,
        }
        result[f"{key}_recall_1m_ge_0p8"] = {
            "applicable": applicable,
            "passed": recall_passed,
        }
    if len(result) != 8:
        raise AssertionError("STDA metric child must emit exactly eight metric gates")
    return result


def _validated_geometry_report(report: Mapping[str, Any]) -> dict[str, float | int]:
    required = {
        "chamfer_m",
        "outlier_fraction_2m",
        "prediction_count",
        "target_count",
    }
    if not required.issubset(report):
        missing = sorted(required.difference(report))
        raise MetricPhaseError(f"Archived geometry report is missing keys: {missing}")
    validated: dict[str, float | int] = {}
    for key, value in report.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricPhaseError(f"Unexpected geometry value for {key}: {value!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise MetricPhaseError(f"Non-finite geometry value for {key}")
        validated[str(key)] = value
    return validated


def evaluate_arm(
    *,
    module: ModuleType,
    snapshot: ArraySnapshot,
    target_xyz: torch.Tensor,
    target_weight: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, object], int]:
    """Evaluate one arm in one explicitly synchronized CUDA interval."""

    if device != torch.device("cuda:0"):
        raise MetricPhaseError("Metric evaluation is restricted to cuda:0")
    torch.cuda.synchronize(device)
    started_ns = time.perf_counter_ns()
    with torch.no_grad():
        prediction = torch.from_numpy(np.array(snapshot.array, copy=True)).to(
            device=device,
            dtype=torch.float32,
        )
        if prediction.device != device or prediction.dtype != torch.float32:
            raise MetricPhaseError("Prediction was not materialized as float32 cuda:0")
        geometry = module.geometry_report(prediction, target_xyz, target_weight)
        target_to_prediction = module.nearest_distance(target_xyz, prediction)
        if (
            target_to_prediction.device != device
            or target_to_prediction.dtype != torch.float32
        ):
            raise MetricPhaseError("Archived nearest_distance changed dtype or device")
        ranges = range_metrics_from_distances(
            target_xyz,
            target_weight,
            target_to_prediction,
        )
        validated_geometry = _validated_geometry_report(geometry)
        semantics = semantic_metric_booleans(validated_geometry, ranges)
    torch.cuda.synchronize(device)
    elapsed_ns = time.perf_counter_ns() - started_ns
    if elapsed_ns <= 0:
        raise MetricPhaseError("Per-arm metric interval must be positive")
    return (
        {
            "archived_geometry_report_called_unchanged": True,
            "archived_nearest_distance_called_separately": True,
            "default_chunking_preserved": True,
            "float32_cuda": True,
            "geometry": validated_geometry,
            "metric_ns": elapsed_ns,
            "range_metrics": ranges,
            "semantic_booleans": semantics,
        },
        elapsed_ns,
    )


def _verify_snapshots_unchanged(
    snapshots: Mapping[str, ArraySnapshot],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for label, snapshot in snapshots.items():
        payload, after = _read_regular_file(snapshot.path)
        observed = sha256_bytes(payload)
        same_identity = (
            int(after.st_dev) == snapshot.device_id
            and int(after.st_ino) == snapshot.inode
        )
        if observed != snapshot.file_sha256 or not same_identity:
            raise MetricPhaseError(f"{label} input changed during metric execution")
        results[label] = {
            "post_metric_sha256": observed,
            "same_device_and_inode": True,
            "unchanged_during_metric": True,
        }
    return results


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_canonical_json_exclusive(
    path: Path, document: Mapping[str, object]
) -> tuple[str, int]:
    """Publish canonical JSON without overwriting and fsync file and directory."""

    output = path.expanduser()
    if not output.is_absolute():
        raise MetricPhaseError("Output JSON path must be absolute")
    output = output.resolve(strict=False)
    parent = output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise MetricPhaseError(f"Output already exists: {output}")
    payload = canonical_json_bytes(document)
    temporary = parent / f".{output.name}.tmp.{os.getpid()}.{time.time_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    linked = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        os.link(temporary, output)
        linked = True
        os.unlink(temporary)
        _fsync_directory(parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        if linked and output.exists():
            output.unlink()
            _fsync_directory(parent)
        raise
    observed, _ = _read_regular_file(output)
    if observed != payload:
        raise MetricPhaseError("Published metric JSON failed byte replay")
    return sha256_bytes(observed), len(observed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen STDA-F0 single-child CUDA metric phase"
    )
    parser.add_argument("--frame-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    for name in ARM_NAMES:
        parser.add_argument(f"--{name}-export", type=Path)
        parser.add_argument(f"--{name}-sha256")
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-pci", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> tuple[dict[str, object], str, int]:
    if FRAME_KEY_RE.fullmatch(arguments.frame_key) is None:
        raise MetricPhaseError("Frame key must use seqNN/radarNNNNN syntax")
    if SOURCE_COMMIT_RE.fullmatch(arguments.source_commit) is None:
        raise MetricPhaseError("Source commit must be a lowercase 40-hex Git hash")
    arms = collect_arm_inputs(arguments)
    target = load_float32_npy_snapshot(
        arguments.target,
        arguments.target_sha256,
        columns=4,
        label="original target",
    )
    arm_snapshots = {
        arm.name: load_float32_npy_snapshot(
            arm.path,
            arm.expected_sha256,
            columns=3,
            label=f"{arm.name} export",
        )
        for arm in arms
    }
    _assert_distinct_paths(arguments.output, target, arm_snapshots)

    repo_root = Path(__file__).resolve().parents[2]
    dense_geometry = load_archived_dense_geometry(repo_root)
    device, gpu = bind_single_h200(
        expected_uuid=arguments.expected_gpu_uuid,
        expected_pci_bus_id=arguments.expected_gpu_pci,
        expected_name=arguments.expected_gpu_name,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    metric_started_ns = time.perf_counter_ns()
    with torch.no_grad():
        target_xyz = torch.from_numpy(np.array(target.array[:, :3], copy=True)).to(
            device=device,
            dtype=torch.float32,
        )
        target_weight = torch.from_numpy(np.array(target.array[:, 3], copy=True)).to(
            device=device,
            dtype=torch.float32,
        )
        if (
            target_xyz.device != device
            or target_xyz.dtype != torch.float32
            or target_weight.device != device
            or target_weight.dtype != torch.float32
        ):
            raise MetricPhaseError("Target was not materialized as float32 cuda:0")
        arm_reports: dict[str, dict[str, object]] = {}
        arm_metric_ns: dict[str, int] = {}
        for arm in arms:
            report, elapsed_ns = evaluate_arm(
                module=dense_geometry,
                snapshot=arm_snapshots[arm.name],
                target_xyz=target_xyz,
                target_weight=target_weight,
                device=device,
            )
            arm_reports[arm.name] = report
            arm_metric_ns[arm.name] = elapsed_ns
    torch.cuda.synchronize(device)
    metric_ns = time.perf_counter_ns() - metric_started_ns
    if metric_ns <= 0 or metric_ns < sum(arm_metric_ns.values()):
        raise MetricPhaseError("Invalid aggregate metric interval")
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    all_snapshots = {"target": target, **arm_snapshots}
    unchanged = _verify_snapshots_unchanged(all_snapshots)
    input_evidence = {"target": target.evidence()}
    input_evidence["target"].update(unchanged["target"])
    arm_input_evidence: dict[str, dict[str, object]] = {}
    for arm in arms:
        evidence = arm_snapshots[arm.name].evidence()
        evidence.update(unchanged[arm.name])
        arm_input_evidence[arm.name] = evidence

    report_document: dict[str, object] = {
        "available_arms": [arm.name for arm in arms],
        "contract_booleans": {
            "archived_dense_geometry_sha256_verified_before_import": True,
            "exactly_one_visible_cuda_device": True,
            "expected_physical_gpu_identity_bound_to_cuda_0": True,
            "input_file_sha256_verified_before_load": True,
            "input_files_unchanged_during_metric": True,
            "metric_child_contains_no_structure_or_target_fitting": True,
            "per_arm_intervals_synchronized": True,
            "target_is_original_float32_xyz_confidence": True,
        },
        "dense_geometry": {
            "absolute_path": str(
                (repo_root / DENSE_GEOMETRY_RELATIVE_PATH).resolve(strict=True)
            ),
            "sha256": DENSE_GEOMETRY_SHA256,
        },
        "frame_key": arguments.frame_key,
        "gpu_provenance": {
            "compute_capability": list(gpu.compute_capability),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "cuda_visible_devices": gpu.cuda_visible_devices,
            "driver_version": gpu.driver_version,
            "expected": {
                "name": arguments.expected_gpu_name,
                "pci_bus_id": normalize_pci_bus_id(arguments.expected_gpu_pci),
                "uuid": normalize_gpu_uuid(arguments.expected_gpu_uuid),
            },
            "logical_device": "cuda:0",
            "name": gpu.name,
            "pci_bus_id": gpu.pci_bus_id,
            "physical_identity_match": True,
            "total_memory_bytes": gpu.total_memory_bytes,
            "uuid": gpu.uuid,
            "visible_cuda_device_count": 1,
        },
        "inputs": {
            "arms": arm_input_evidence,
            "target": input_evidence["target"],
        },
        "metrics": arm_reports,
        "protocol": PROTOCOL,
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime": {
            "arm_metric_ns": arm_metric_ns,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "interpreter": {
                "isolated": bool(sys.flags.isolated),
                "no_site": bool(sys.flags.no_site),
                "site_packages_bootstrap": ISOLATED_SITE_PACKAGES,
            },
            "metric_ns": metric_ns,
            "numpy_version": np.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        },
        "schema": SCHEMA,
        "source_commit": arguments.source_commit,
    }
    output_sha256, output_size = write_canonical_json_exclusive(
        arguments.output, report_document
    )
    return report_document, output_sha256, output_size


def main() -> None:
    arguments = build_parser().parse_args()
    _, output_sha256, output_size = run(arguments)
    summary = {
        "output": str(arguments.output.expanduser().resolve(strict=True)),
        "output_sha256": output_sha256,
        "output_size_bytes": output_size,
        "schema": SCHEMA,
    }
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
