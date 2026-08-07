from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import sys
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from eval import stda_f0_candidate as candidate


EXPECTED_ARCHITECTURE = {
    "latent_count": 512,
    "model_dim": 512,
    "depth": 6,
    "heads": 8,
    "head_dim": 64,
    "decode_chunk_size": 8_192,
}
EXPECTED_PREDECESSOR_KEYS = {
    "q0_sha256",
    "q1_sha256",
    "candidate_xyz_sha256",
    "base_confidence_sha256",
}


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _formal_checkpoint() -> dict[str, Any]:
    return {
        "protocol": candidate.FRESH_PARENT_PROTOCOL,
        "source_commit": candidate.FRESH_PARENT_SOURCE_COMMIT,
        "epoch": 20,
        "config": copy.deepcopy(candidate.FORMAL_CHECKPOINT_CONFIG),
        "model": {"synthetic": torch.tensor([1.0])},
    }


class RecordingField(torch.nn.Module):
    constructor_kwargs: dict[str, Any] | None = None
    loaded_state: dict[str, Any] | None = None
    loaded_strict: bool | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        type(self).constructor_kwargs = dict(kwargs)
        self.latent_count = int(kwargs["latent_count"])
        self.model_dim = int(kwargs["model_dim"])
        self.depth = int(kwargs["depth"])
        self.decode_chunk_size = int(kwargs["decode_chunk_size"])
        self.synthetic = torch.nn.Parameter(torch.zeros(()))

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> None:
        type(self).loaded_state = dict(state_dict)
        type(self).loaded_strict = strict


class SyntheticFrozenField(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.latent_count = 512
        self.model_dim = 512
        self.depth = 6
        self.decode_chunk_size = 8_192
        self.register_buffer("device_anchor", torch.zeros((), device=device))
        self.autocast_observations: list[bool] = []
        self.eval()

    def _record_autocast(self) -> None:
        self.autocast_observations.append(torch.is_autocast_enabled("cuda"))

    def encode_condition(self, cube_drae: torch.Tensor) -> dict[str, torch.Tensor]:
        self._record_autocast()
        return {
            "condition_latents": cube_drae.new_zeros((1, 1, 1)),
        }

    def decode_queries(
        self,
        normalized_rae: torch.Tensor,
        condition_latents: torch.Tensor,
        *,
        chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        del condition_latents
        self._record_autocast()
        assert chunk_size == 8_192
        confidence = (
            0.75
            - 0.10 * normalized_rae[..., 0]
            + 0.03 * normalized_rae[..., 1]
            - 0.01 * normalized_rae[..., 2]
        ).float()
        return {
            "refined_normalized_rae": normalized_rae.float().clone(),
            "confidence": confidence,
        }


def _synthetic_axes(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.linspace(0.5, 120.0, 256, dtype=torch.float32, device=device),
        torch.linspace(-0.9, 0.9, 107, dtype=torch.float32, device=device),
        torch.linspace(-0.3, 0.3, 37, dtype=torch.float32, device=device),
    )


def _manual_field(device: torch.device) -> dict[str, Any]:
    count = candidate.FORMAL_CANDIDATE_COUNT
    row_id = torch.arange(count, dtype=torch.int64, device=device)
    base = torch.arange(count, dtype=torch.float32, device=device)
    xyz = torch.stack(
        (base * 1e-4, base.remainder(101) * 1e-3, -base.remainder(53) * 1e-3),
        dim=1,
    )
    confidence = (base.remainder(997) / 997.0).float()
    q0 = torch.arange(
        3 * candidate.FORMAL_Q0_COUNT,
        dtype=torch.float32,
        device=device,
    ).reshape(1, candidate.FORMAL_Q0_COUNT, 3)
    q1 = torch.arange(
        3 * candidate.FORMAL_Q1_COUNT,
        dtype=torch.float32,
        device=device,
    ).reshape(1, candidate.FORMAL_Q1_COUNT, 3)
    report = {
        "candidate_count": count,
        "q0_count": candidate.FORMAL_Q0_COUNT,
        "q1_count": candidate.FORMAL_Q1_COUNT,
        "stable_candidate_id_sha256": _tensor_sha256(row_id),
        "q0_sha256": _tensor_sha256(q0),
        "q1_sha256": _tensor_sha256(q1),
        "candidate_xyz_sha256": _tensor_sha256(xyz),
        "base_confidence_sha256": _tensor_sha256(confidence),
        "ground_truth_accessed": False,
        "support_target_input": False,
        "candidate_coordinates_changed": False,
        "formal_residual_changed": False,
    }
    return {
        "stable_candidate_id": row_id,
        "xyz_m": xyz,
        "base_confidence": confidence,
        "q0": q0,
        "q1": q1,
        "q1_report": {"source": "synthetic"},
        "report": report,
    }


def test_static_target_free_api_imports_and_caller_owned_autocast() -> None:
    source_path = ROOT / "code/eval/stda_f0_candidate.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_import_fragments = (
        "dataset",
        "target",
        "ground_truth",
        "lidar",
        "dense_geometry",
        "stda_f0_fit",
        "stda_f0_structure",
        "stda_f0_round",
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "autocast"
    assert not any(
        fragment in module.lower()
        for module in imported
        for fragment in forbidden_import_fragments
    )

    forbidden_parameters = ("target", "ground_truth", "gt", "lidar", "nearest")
    for function_name in (
        "frozen_inference_config",
        "build_frozen_base_model",
        "axes_tensors",
        "reconstruct_candidate_field",
        "verify_candidate_only",
    ):
        names = inspect.signature(getattr(candidate, function_name)).parameters
        assert not any(
            fragment == name.lower() or fragment in name.lower().split("_")
            for name in names
            for fragment in forbidden_parameters
        )
    assert tuple(
        inspect.signature(candidate.reconstruct_candidate_field).parameters
    ) == (
        "base_model",
        "cube_drae",
        "range_m",
        "azimuth_rad",
        "elevation_rad",
        "inference_config",
    )
    assert tuple(inspect.signature(candidate.verify_candidate_only).parameters) == (
        "field",
        "sequence",
        "radar_index",
        "expected_hashes",
    )


def test_frozen_counts_architecture_and_full_formal_config() -> None:
    config = candidate.frozen_inference_config()
    assert candidate.FORMAL_ARCHITECTURE == EXPECTED_ARCHITECTURE
    assert sum(config.q0_range_quotas) == 500_000
    assert sum(config.q1_anchor_quotas) == 25_000
    assert config.q1_samples_per_anchor == 8
    assert sum(config.q1_anchor_quotas) * config.q1_samples_per_anchor == 200_000
    assert candidate.FORMAL_CANDIDATE_COUNT == 700_000
    assert config.seed == 20260716
    assert config.decode_chunk_size == 8_192
    assert candidate.FORMAL_CHECKPOINT_CONFIG["smoke"] is False
    assert candidate.FORMAL_CHECKPOINT_CONFIG["epochs"] == 20
    assert candidate.FORMAL_CHECKPOINT_CONFIG["train_limit"] is None
    assert candidate.FORMAL_CHECKPOINT_CONFIG["validation_limit"] is None
    assert candidate.FORMAL_CHECKPOINT_CONFIG["test_accessed"] is False
    assert candidate.FORMAL_CHECKPOINT_CONFIG["doppler_head"] is False


def test_build_frozen_model_uses_strict_fresh_wce_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate, "RaLDWCEField", RecordingField)
    device = torch.device("cuda:0")
    model = candidate.build_frozen_base_model(
        _formal_checkpoint(),
        log_center=0.25,
        log_scale=1.5,
        device=device,
    )
    assert isinstance(model, RecordingField)
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert {parameter.device for parameter in model.parameters()} == {device}
    assert RecordingField.constructor_kwargs == {
        "log_center": 0.25,
        "log_scale": 1.5,
        **EXPECTED_ARCHITECTURE,
    }
    assert RecordingField.loaded_state == {"synthetic": torch.tensor([1.0])}
    assert RecordingField.loaded_strict is True


@pytest.mark.parametrize(
    ("path", "bad_value"),
    (
        (("protocol",), "wrong"),
        (("source_commit",), "0" * 40),
        (("epoch",), 19),
        (("epoch",), 20.0),
        (("config", "smoke"), True),
        (("config", "epochs"), 19),
        (("config", "latent_count"), 256),
        (("config", "heads"), 4),
        (("config", "q0_range_quotas"), (166_666, 166_667, 166_666)),
        (("config", "q1_anchor_quotas"), (20_000, 4_249, 750)),
        (("config", "q1_samples_per_anchor"), 4),
        (("config", "decode_chunk_size"), 4_096),
        (("config", "test_accessed"), True),
        (("config", "doppler_head"), True),
        (("model",), None),
    ),
)
def test_wrong_checkpoint_or_config_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    bad_value: Any,
) -> None:
    monkeypatch.setattr(candidate, "RaLDWCEField", RecordingField)
    checkpoint = _formal_checkpoint()
    owner: dict[str, Any] = checkpoint
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = bad_value
    with pytest.raises((TypeError, ValueError)):
        candidate.build_frozen_base_model(
            checkpoint,
            log_center=0.0,
            log_scale=1.0,
            device=torch.device("cuda:0"),
        )


def test_checkpoint_config_missing_or_extra_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate, "RaLDWCEField", RecordingField)
    for mutate in (
        lambda config: config.pop("depth"),
        lambda config: config.__setitem__("unknown_architecture", 1),
    ):
        checkpoint = _formal_checkpoint()
        mutate(checkpoint["config"])
        with pytest.raises(ValueError, match="configuration keys changed"):
            candidate.build_frozen_base_model(
                checkpoint,
                log_center=0.0,
                log_scale=1.0,
                device=torch.device("cuda:0"),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H200 GPU2 is required")
def test_full_700k_synthetic_reconstruction_hashes_and_determinism() -> None:
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name(0) == "NVIDIA H200 NVL"
    assert str(torch.cuda.get_device_properties(0).uuid) == (
        "000b6236-3632-a001-9667-1f02cbb61c8b"
    )
    device = torch.device("cuda:0")
    torch.manual_seed(20260716)
    torch.cuda.manual_seed_all(20260716)
    generator = torch.Generator(device=device).manual_seed(20260716)
    cube = torch.empty(
        (1, 64, 256, 107, 37),
        dtype=torch.float32,
        device=device,
    ).uniform_(-1.0, 1.0, generator=generator)
    axes = _synthetic_axes(device)
    model = SyntheticFrozenField(device)
    config = candidate.frozen_inference_config()

    outputs: list[dict[str, Any]] = []
    for _ in range(2):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs.append(
                candidate.reconstruct_candidate_field(
                    model,
                    cube,
                    *axes,
                    config,
                )
            )
    assert not torch.is_autocast_enabled("cuda")
    assert model.autocast_observations and all(model.autocast_observations)

    first, second = outputs
    expected_shapes_and_dtypes = {
        "stable_candidate_id": ((700_000,), torch.int64),
        "xyz_m": ((700_000, 3), torch.float32),
        "base_confidence": ((700_000,), torch.float32),
        "q0": ((1, 500_000, 3), torch.float32),
        "q1": ((1, 200_000, 3), torch.float32),
    }
    for key, (shape, dtype) in expected_shapes_and_dtypes.items():
        value = first[key]
        assert tuple(value.shape) == shape
        assert value.dtype == dtype
        assert value.device == device
        assert torch.equal(value, second[key])
        if value.is_floating_point():
            assert bool(torch.isfinite(value).all())
    assert first["stable_candidate_id"][0].item() == 0
    assert first["stable_candidate_id"][-1].item() == 699_999
    for key, tensor_key in (
        ("stable_candidate_id_sha256", "stable_candidate_id"),
        ("candidate_xyz_sha256", "xyz_m"),
        ("base_confidence_sha256", "base_confidence"),
        ("q0_sha256", "q0"),
        ("q1_sha256", "q1"),
    ):
        assert first["report"][key] == _tensor_sha256(first[tensor_key])
        assert first["report"][key] == second["report"][key]


def test_candidate_only_verifier_12_expected_and_64_fresh_branches() -> None:
    field = _manual_field(torch.device("cpu"))
    expected = {
        key: field["report"][key] for key in candidate.PREDECESSOR_HASH_KEYS
    }
    assert set(expected) == EXPECTED_PREDECESSOR_KEYS

    identities = tuple((index + 1, 10_000 + index) for index in range(76))
    predecessor_identities = frozenset(identities[:12])
    predecessor_count = 0
    fresh_count = 0
    last_fresh = None
    for sequence, radar_index in identities:
        predecessor_hashes = (
            expected
            if (sequence, radar_index) in predecessor_identities
            else None
        )
        result = candidate.verify_candidate_only(
            field,
            sequence=sequence,
            radar_index=radar_index,
            expected_hashes=predecessor_hashes,
        )
        assert result.passed
        assert set(result.hashes) == set(candidate.HASH_KEYS)
        if predecessor_hashes is not None:
            predecessor_count += 1
            assert result.predecessor_expected is True
            assert result.predecessor_hashes_match is True
        else:
            fresh_count += 1
            last_fresh = result
            assert result.predecessor_expected is False
            assert result.predecessor_hashes_match is None
    assert predecessor_count == 12
    assert fresh_count == 64
    assert last_fresh is not None
    report = last_fresh.report()
    assert report["support_target_input"] is False
    assert report["ground_truth_accessed"] is False

    wrong = dict(expected)
    wrong["candidate_xyz_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate-only verification failed"):
        candidate.verify_candidate_only(
            field,
            sequence=1,
            radar_index=232,
            expected_hashes=wrong,
        )


def test_candidate_only_verifier_rejects_content_and_contract_tampering() -> None:
    field = _manual_field(torch.device("cpu"))

    mutations: list[tuple[str, Any]] = []
    bad_ids = field["stable_candidate_id"].clone()
    bad_ids[1] = 0
    mutations.append(("stable_candidate_id", bad_ids))
    bad_xyz = field["xyz_m"].clone()
    bad_xyz[0, 0] = torch.nan
    mutations.append(("xyz_m", bad_xyz))
    mutations.append(("base_confidence", field["base_confidence"].double()))
    mutations.append(("q0", field["q0"][:, :-1]))
    if torch.cuda.is_available():
        mutations.append(("q1", field["q1"].to("cuda:0")))

    for key, bad_value in mutations:
        changed = dict(field)
        changed[key] = bad_value
        with pytest.raises((TypeError, ValueError)):
            candidate.verify_candidate_only(
                changed,
                sequence=1,
                radar_index=232,
                expected_hashes=None,
            )

    changed_report = dict(field)
    changed_report["report"] = dict(field["report"])
    changed_report["report"]["base_confidence_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="candidate-only verification failed"):
        candidate.verify_candidate_only(
            changed_report,
            sequence=1,
            radar_index=232,
            expected_hashes=None,
        )

    expected = {
        key: field["report"][key] for key in candidate.PREDECESSOR_HASH_KEYS
    }
    expected["stable_candidate_id_sha256"] = field["report"][
        "stable_candidate_id_sha256"
    ]
    with pytest.raises(ValueError, match="predecessor hash keys changed"):
        candidate.verify_candidate_only(
            field,
            sequence=1,
            radar_index=232,
            expected_hashes=expected,
        )


def test_reconstruction_rejects_wrong_config_shape_dtype_finite_and_device() -> None:
    device = torch.device("cuda:0")
    model = SyntheticFrozenField(device)
    axes = _synthetic_axes(device)
    cube = torch.zeros((1, 64, 256, 107, 37), device=device)

    bad_config = candidate.frozen_inference_config()
    object.__setattr__(bad_config, "decode_chunk_size", 4_096)
    with pytest.raises(ValueError, match="configuration changed"):
        candidate.reconstruct_candidate_field(model, cube, *axes, bad_config)

    bad_inputs = (
        (cube[:, :, :-1], axes, ValueError),
        (cube.double(), axes, TypeError),
        (cube, (axes[0][:-1], axes[1], axes[2]), ValueError),
        (cube, (axes[0].double(), axes[1], axes[2]), TypeError),
        (cube, (axes[0], axes[1].cpu(), axes[2]), ValueError),
    )
    for bad_cube, bad_axes, error_type in bad_inputs:
        with pytest.raises(error_type):
            candidate.reconstruct_candidate_field(
                model,
                bad_cube,
                *bad_axes,
                candidate.frozen_inference_config(),
            )

    nonfinite_cube = cube.clone()
    nonfinite_cube[0, 0, 0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="must be finite"):
        candidate.reconstruct_candidate_field(
            model,
            nonfinite_cube,
            *axes,
            candidate.frozen_inference_config(),
        )
