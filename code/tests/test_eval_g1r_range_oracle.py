import ast
import json
from pathlib import Path

import pytest

from scripts.eval_g1r_range_oracle import (
    ARM_NAMES,
    CANDIDATE_PARENT_QUOTAS,
    EXPORT_QUOTAS,
    FAR_RECALL_KEY,
    PROTOCOL,
    aggregate_paired_deltas,
    paired_deltas,
    source_hashes,
    stage0_decision,
    validate_frozen_data_contract,
)


def _geometry(
    *,
    completeness: float,
    far: float,
    outlier: float,
) -> dict[str, float]:
    return {
        "chamfer_m": completeness + 0.2,
        "precision_mean_distance_m": 0.2,
        "completeness_mean_distance_m": completeness,
        "outlier_fraction_2m": outlier,
        "range_60_120m_completeness_mean_distance_m": far,
        "prediction_count": 10_000,
        "target_count": 4,
        "target_effective_count": 4.0,
    }


def _arm(
    *,
    completeness: float = 1.0,
    far: float = 7.5,
    outlier: float = 0.20,
    duplicate: float = 0.08,
    far_recall: float = 0.35,
    range_aware: bool = False,
) -> dict:
    parent = (
        dict(zip(("range_0_30m", "range_30_60m", "range_60_120m"), CANDIDATE_PARENT_QUOTAS))
        if range_aware
        else {
            "range_0_30m": 20_000,
            "range_30_60m": 8_000,
            "range_60_120m": 4_000,
        }
    )
    selected = dict(
        zip(
            ("range_0_30m", "range_30_60m", "range_60_120m"),
            EXPORT_QUOTAS,
        )
    )
    return {
        "candidate_count": 32_000,
        "selected_count": 10_000,
        "candidate_parent_range_count": parent,
        "selected_actual_range_count": selected,
        "geometry": _geometry(
            completeness=completeness,
            far=far,
            outlier=outlier,
        ),
        "duplicates": {
            "duplicate_fraction_0p05m": duplicate,
            "nearest_other_median_m": 0.2,
            "nearest_other_mean_m": 0.3,
        },
        "per_range_support": {
            "range_60_120m": {
                FAR_RECALL_KEY: far_recall,
            }
        },
    }


def _frame(sequence: int = 6) -> dict:
    arms = {
        "vanilla": _arm(completeness=1.1),
        "z_only": _arm(completeness=1.05),
        "range_aware": _arm(range_aware=True),
    }
    return {
        "sequence": sequence,
        "radar_index": 100 + sequence,
        "arms": arms,
        "paired_delta_vs_vanilla": paired_deltas(arms),
    }


def _aggregate_metrics(frame: dict) -> dict:
    arm = frame["arms"]["range_aware"]
    return {
        "range_aware": {
            "frame_first": {
                "geometry": {
                    "completeness_mean_distance_m": {
                        "mean": arm["geometry"][
                            "completeness_mean_distance_m"
                        ],
                        "median": arm["geometry"][
                            "completeness_mean_distance_m"
                        ],
                    },
                    "outlier_fraction_2m": {
                        "mean": arm["geometry"]["outlier_fraction_2m"],
                        "median": arm["geometry"]["outlier_fraction_2m"],
                    },
                    "range_60_120m_completeness_mean_distance_m": {
                        "mean": arm["geometry"][
                            "range_60_120m_completeness_mean_distance_m"
                        ],
                        "median": arm["geometry"][
                            "range_60_120m_completeness_mean_distance_m"
                        ],
                    },
                },
                "duplicates": {
                    "duplicate_fraction_0p05m": {
                        "mean": arm["duplicates"][
                            "duplicate_fraction_0p05m"
                        ],
                        "median": arm["duplicates"][
                            "duplicate_fraction_0p05m"
                        ],
                    }
                },
                "per_range_support": {
                    "range_60_120m": {
                        FAR_RECALL_KEY: {
                            "mean": arm["per_range_support"][
                                "range_60_120m"
                            ][FAR_RECALL_KEY],
                            "median": arm["per_range_support"][
                                "range_60_120m"
                            ][FAR_RECALL_KEY],
                        }
                    }
                },
            }
        }
    }


def test_stage0_gate_passes_only_when_every_frozen_condition_passes() -> None:
    frame = _frame()
    decision = stage0_decision(_aggregate_metrics(frame), [frame])

    assert decision["passed"] is True
    assert decision["training_authorized"] is True
    assert all(decision["checks"].values())

    frame["arms"]["range_aware"]["per_range_support"]["range_60_120m"][
        FAR_RECALL_KEY
    ] = 0.299
    decision = stage0_decision(_aggregate_metrics(frame), [frame])
    assert decision["passed"] is False
    assert decision["training_authorized"] is False
    assert decision["decision"] == "forbid_r0_training_oracle_gate_failed"


def test_paired_comparison_preserves_same_frame_and_scene_first_units() -> None:
    first = _frame(sequence=6)
    second = _frame(sequence=12)
    first_delta = first["paired_delta_vs_vanilla"]["range_aware"][
        "completeness_mean_distance_m"
    ]

    assert first_delta == pytest.approx(-0.1)
    aggregate = aggregate_paired_deltas([first, second])
    assert aggregate["range_aware"]["frame_first"][
        "completeness_mean_distance_m"
    ]["sample_count"] == 2
    assert aggregate["range_aware"]["scene_first"][
        "completeness_mean_distance_m"
    ]["sample_count"] == 2


def test_data_contract_delegates_to_frozen_76_24_and_rejects_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        {
            "partition": "train" if index < 76 else "validation",
            "sequence": index + 1,
            "radar_index": index,
        }
        for index in range(100)
    ]
    manifest = {"frames": frames}
    split = {
        "gate_pass": True,
        "splits": {
            "train": {"sequences": list(range(1, 77))},
            "validation": {"sequences": list(range(77, 101))},
            "test": {"sequences": [999]},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.eval_g1r_range_oracle.FROZEN_MANIFEST_SHA256",
        __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        "scripts.eval_g1r_range_oracle.FROZEN_SCENE_SPLIT_SHA256",
        __import__("hashlib").sha256(split_path.read_bytes()).hexdigest(),
    )
    _, _, counts = validate_frozen_data_contract(manifest_path, split_path)
    assert counts == {
        "manifest_frame_count": 100,
        "train_frame_count": 76,
        "validation_frame_count": 24,
        "test_frame_count": 0,
    }

    manifest["frames"][0]["partition"] = "test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="76/24"):
        validate_frozen_data_contract(manifest_path, split_path)


def test_cli_has_source_lock_h200_guard_and_no_overwrite_escape() -> None:
    path = Path(__file__).parents[1] / "scripts" / "eval_g1r_range_oracle.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    option_strings = {
        constant.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for constant in node.args
        if isinstance(constant, ast.Constant)
        and isinstance(constant.value, str)
    }

    assert PROTOCOL in source
    assert "--source-commit" in option_strings
    assert "--overwrite" not in option_strings
    assert "if args.output.exists()" in source
    assert "require_h200" in source
    assert "verify_source_tree" in source
    assert '("train",)' in source
    assert '("validation",)' in source
    assert "test_accessed" in source
    assert set(ARM_NAMES) == {"vanilla", "z_only", "range_aware"}


def test_source_map_covers_all_four_new_files_and_reused_metrics() -> None:
    repo = Path(__file__).parents[2]
    hashes = source_hashes(repo)
    paths = set(hashes)

    for relative in (
        "code/eval/g1r_range_aware_support.py",
        "code/scripts/eval_g1r_range_oracle.py",
        "code/tests/test_g1r_range_aware_support.py",
        "code/tests/test_eval_g1r_range_oracle.py",
        "code/eval/g1f_candidate_support.py",
        "code/eval/dense_geometry.py",
        "code/models/rald_query_field.py",
    ):
        assert str((repo / relative).resolve()) in paths
    assert all(record["sha256"] for record in hashes.values())

