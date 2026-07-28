#!/usr/bin/env python3
"""Run frozen D1/D2/D3 inference diagnostics on the G1D epoch-150 EMA."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.dataset import KRadarCubeDataset  # noqa: E402
from cube_dense.kradar import load_axes  # noqa: E402
from eval.g1d_failure_factors import (  # noqa: E402
    CURRENT_RANKING_SOURCE,
    FAR_COMPLETENESS_METRIC,
    OFFSET_ALPHAS,
    RANKING_SOURCES,
    aggregate_arm,
    alpha_label,
    current_ranking_score,
    evaluate_generated,
    generate_selected_points,
    paired_scene_bootstrap,
    prepare_query_field,
    ranking_capability,
    ranking_scores,
)
from scripts.g1b_contract import sha256  # noqa: E402
from scripts.train_g1g_hierarchy import (  # noqa: E402
    FORMAL_VALIDATION_COUNT,
    G1D_CONTROL_PROTOCOL,
    canonical_data_contract,
    frame_data_hashes,
    range_target_frame_identities,
    validate_data_contract,
    validate_frozen_input_hashes,
)
from scripts.train_rald_query_field import (  # noqa: E402
    PROTOCOL as G1D_PROTOCOL,
    TrainConfig,
    build_model,
    cached_proposal_flat_index,
    cross_scene_condition_indices,
    move_frame,
    require_h200,
    verify_source_tree,
)


PROTOCOL = "g1d_epoch150_failure_factors_d1_d2_d3_v1"
EXPECTED_CHECKPOINT_EPOCH = 150
EXPECTED_SEED = 20260716
EXPECTED_POINT_COUNT = 10_000
EXPECTED_FAR_FRAME_COUNT = 23
FORMAL_BOOTSTRAP_SAMPLES = 10_000
FORMAL_BOOTSTRAP_SEED = 20260716
CHECKPOINT_SOURCE_FILES = (
    "scripts/train_rald_query_field.py",
    "models/rald_query_field.py",
    "losses/rald_query_field.py",
    "models/rald_matched.py",
    "models/cube_doppler.py",
    "models/cube_occupancy.py",
    "models/cube_cycle.py",
    "models/point_to_cube.py",
    "cube_dense/dataset.py",
    "cube_dense/kradar.py",
    "eval/rald_guided_query.py",
    "scripts/g1b_contract.py",
)


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _identity(item: dict[str, Any]) -> dict[str, int]:
    return {
        "sequence": int(item["sequence"]),
        "radar_index": int(item["radar_index"]),
    }


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
) -> TrainConfig:
    epoch = int(checkpoint.get("epoch", -1))
    config_document = checkpoint.get("config")
    provenance = checkpoint.get("provenance")
    if epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(
            f"G1D diagnosis requires epoch {EXPECTED_CHECKPOINT_EPOCH}, got {epoch}"
        )
    if not isinstance(config_document, dict):
        raise ValueError("G1D checkpoint has no configuration document")
    if not isinstance(provenance, dict):
        raise ValueError("G1D checkpoint has no provenance document")
    config = TrainConfig(**config_document)
    checks = {
        "protocol_matches": config.protocol == G1D_PROTOCOL,
        "epoch_is_training_endpoint": epoch == int(config.epochs),
        "formal_seed": config.seed == EXPECTED_SEED,
        "exact_10000_points": config.point_count == EXPECTED_POINT_COUNT,
        "full_validation_configured": config.validation_limit is None,
        "config_test_accessed_false": config.test_accessed is False,
        "provenance_test_accessed_false": (
            provenance.get("test_accessed") is False
        ),
        "development_partitions_only": provenance.get("partitions")
        == ["train", "validation"],
        "ema_state_present": isinstance(checkpoint.get("ema_model"), dict),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"G1D epoch-150 checkpoint contract failed: {failed}")
    record = checkpoint.get("record")
    if isinstance(record, dict) and "epoch" in record:
        if int(record["epoch"]) != epoch:
            raise ValueError("G1D checkpoint record epoch differs from checkpoint")
    return config


def checkpoint_source_compatibility(
    repo: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    recorded = checkpoint["provenance"].get("source_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("G1D checkpoint has no source-hash map")
    evidence = {}
    for relative in CHECKPOINT_SOURCE_FILES:
        suffix = f"/code/{relative}"
        matches = [
            document
            for path, document in recorded.items()
            if str(path).replace("\\", "/").endswith(suffix)
        ]
        if len(matches) != 1 or not isinstance(matches[0], dict):
            raise ValueError(
                f"G1D checkpoint source map lacks one unique {relative}"
            )
        expected = matches[0].get("sha256")
        current_path = repo / "code" / relative
        actual = sha256(current_path)
        if actual != expected:
            raise ValueError(
                f"Checkpoint-incompatible source file {relative}: "
                f"{actual} != {expected}"
            )
        evidence[relative] = {
            "path": str(current_path.resolve()),
            "sha256": actual,
            "matches_checkpoint": True,
        }
    return {
        "status": "compatible",
        "strict_state_dict_load_required": True,
        "files": evidence,
    }


def validate_corrected_reference(
    corrected: dict[str, Any],
    *,
    corrected_path: Path,
    data_contract: dict[str, Any],
    validation_identities: list[dict[str, int]],
    far_identities: list[dict[str, int]],
    evaluator_path: Path,
) -> dict[str, Any]:
    evaluator_sha = sha256(evaluator_path)
    checks = {
        "reference_protocol": corrected.get("protocol")
        == G1D_CONTROL_PROTOCOL,
        "reference_test_accessed_false": corrected.get("test_accessed")
        is False,
        "data_contract_matches": corrected.get("data_contract")
        == data_contract,
        "validation_identities_match": corrected.get(
            "validation_frame_identities"
        )
        == validation_identities,
        "far_identities_match": corrected.get("far_target_frame_identities")
        == far_identities,
        "corrected_evaluator_hash_matches": corrected.get("evaluator", {}).get(
            "dense_geometry_sha256"
        )
        == evaluator_sha,
        "far_censoring_fix_declared": corrected.get("evaluator", {}).get(
            "far_target_frame_censoring_fixed"
        )
        is True,
        "reference_frame_count_24": corrected.get("metrics", {}).get(
            "frame_count"
        )
        == FORMAL_VALIDATION_COUNT,
        "reference_far_sample_count_23": corrected.get("metrics", {})
        .get("generated", {})
        .get(FAR_COMPLETENESS_METRIC, {})
        .get("sample_count")
        == EXPECTED_FAR_FRAME_COUNT,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Corrected evaluator reference failed: {failed}")
    return {
        "path": str(corrected_path.resolve()),
        "sha256": sha256(corrected_path),
        "protocol": corrected["protocol"],
        "evaluator_path": str(evaluator_path.resolve()),
        "evaluator_sha256": evaluator_sha,
        "far_target_frame_censoring_fixed": True,
        "checks": checks,
    }


def source_hashes(repo: Path, script_path: Path) -> dict[str, dict[str, str]]:
    paths = (
        script_path,
        repo / "code/eval/g1d_failure_factors.py",
        repo / "code/models/rald_query_field.py",
        repo / "code/scripts/train_rald_query_field.py",
        repo / "code/eval/dense_geometry.py",
        repo / "code/eval/rald_guided_query.py",
        repo / "code/cube_dense/dataset.py",
        repo / "code/scripts/train_g1g_hierarchy.py",
    )
    return {
        str(path.resolve()): {"sha256": sha256(path)}
        for path in paths
    }


def _contrast_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return FORMAL_BOOTSTRAP_SEED + int.from_bytes(digest[:2], "big")


def _paired_endpoints(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    name: str,
    include_duplicates: bool,
) -> dict[str, Any]:
    endpoints = {
        "chamfer_m": ("geometry", "chamfer_m"),
        "completeness_mean_distance_m": (
            "geometry",
            "completeness_mean_distance_m",
        ),
        "outlier_fraction_2m": ("geometry", "outlier_fraction_2m"),
    }
    if include_duplicates:
        endpoints["duplicate_fraction_0p05m"] = (
            "duplicates",
            "duplicate_fraction_0p05m",
        )
    return {
        endpoint: paired_scene_bootstrap(
            reference["frames"],
            candidate["frames"],
            group=group,
            metric=metric,
            samples=FORMAL_BOOTSTRAP_SAMPLES,
            seed=_contrast_seed(f"{name}:{endpoint}"),
        )
        for endpoint, (group, metric) in endpoints.items()
    }


def build_d1_contrasts(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference_name = alpha_label(1.0)
    reference = arms[reference_name]
    comparisons = {}
    for alpha in OFFSET_ALPHAS:
        if alpha == 1.0:
            continue
        name = alpha_label(alpha)
        endpoints = _paired_endpoints(
            reference,
            arms[name],
            name=f"d1:{name}",
            include_duplicates=True,
        )
        checks = {
            "chamfer_paired_ci_upper_below_zero": endpoints["chamfer_m"][
                "delta_ci95"
            ][1]
            < 0.0,
            "outlier_reduction_at_least_3pp": endpoints[
                "outlier_fraction_2m"
            ]["mean_delta"]
            <= -0.03,
            "duplicate_reduction_at_least_3pp": endpoints[
                "duplicate_fraction_0p05m"
            ]["mean_delta"]
            <= -0.03,
            "completeness_degradation_at_most_0p10m": endpoints[
                "completeness_mean_distance_m"
            ]["mean_delta"]
            <= 0.10,
        }
        comparisons[name] = {
            "reference": reference_name,
            "candidate": name,
            "endpoints": endpoints,
            "bounded_offset_repair_authorized": all(checks.values()),
            "checks": checks,
        }
    return comparisons


def build_d2_contrasts(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_name = "measured_current__condition_current"
    global_name = "measured_current__condition_shuffled"
    measured_name = "measured_shuffled__condition_current"
    both_name = "measured_shuffled__condition_shuffled"
    contrasts = {
        "global_condition_effect_at_current_measurement": _paired_endpoints(
            arms[baseline_name],
            arms[global_name],
            name="d2:global",
            include_duplicates=False,
        ),
        "measured_path_effect_at_current_condition": _paired_endpoints(
            arms[baseline_name],
            arms[measured_name],
            name="d2:measured",
            include_duplicates=False,
        ),
        "joint_wrong_effect": _paired_endpoints(
            arms[baseline_name],
            arms[both_name],
            name="d2:joint",
            include_duplicates=False,
        ),
    }
    global_chamfer = contrasts[
        "global_condition_effect_at_current_measurement"
    ]["chamfer_m"]
    measured_chamfer = contrasts[
        "measured_path_effect_at_current_condition"
    ]["chamfer_m"]
    checks = {
        "absolute_global_condition_chamfer_change_below_1pct": abs(
            global_chamfer["relative_change"]
        )
        < 0.01,
        "wrong_measured_path_chamfer_degradation_above_10pct": (
            measured_chamfer["relative_change"] > 0.10
        ),
    }
    return {
        "contrasts": contrasts,
        "local_path_bypass_confirmed": all(checks.values()),
        "checks": checks,
    }


def build_d3_contrasts(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = arms[CURRENT_RANKING_SOURCE]
    comparisons = {}
    for name in RANKING_SOURCES:
        if name == CURRENT_RANKING_SOURCE:
            continue
        comparisons[name] = {
            "reference": CURRENT_RANKING_SOURCE,
            "candidate": name,
            "endpoints": _paired_endpoints(
                reference,
                arms[name],
                name=f"d3:{name}",
                include_duplicates=True,
            ),
        }
    oracle = comparisons["validation_gt_distance_oracle"]["endpoints"]
    oracle_checks = {
        "chamfer_or_completeness_improves_at_least_10pct": max(
            -oracle["chamfer_m"]["relative_change"],
            -oracle["completeness_mean_distance_m"]["relative_change"],
        )
        >= 0.10,
        "outlier_no_worse_than_plus_2pp": oracle["outlier_fraction_2m"][
            "mean_delta"
        ]
        <= 0.02,
        "duplicate_no_worse_than_plus_2pp": oracle[
            "duplicate_fraction_0p05m"
        ]["mean_delta"]
        <= 0.02,
    }
    return {
        "comparisons": comparisons,
        "learned_ranking_repair_authorized": all(oracle_checks.values()),
        "oracle_checks": oracle_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corrected-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=FORMAL_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=FORMAL_BOOTSTRAP_SEED,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"G1D diagnosis output exists: {args.output}")
    if (
        args.bootstrap_samples != FORMAL_BOOTSTRAP_SAMPLES
        or args.bootstrap_seed != FORMAL_BOOTSTRAP_SEED
    ):
        raise ValueError(
            "Formal G1D diagnosis freezes bootstrap at "
            f"{FORMAL_BOOTSTRAP_SAMPLES}/{FORMAL_BOOTSTRAP_SEED}"
        )

    repo = Path(__file__).resolve().parents[2]
    script_path = Path(__file__).resolve()
    verify_source_tree(repo, args.source_commit)
    device, device_name = require_h200(args.device)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_split = json.loads(args.scene_split.read_text(encoding="utf-8"))
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    manifest_counts = validate_data_contract(manifest, scene_split)
    frozen_hashes = validate_frozen_input_hashes(
        args.manifest,
        args.scene_split,
        args.normalization,
    )
    data_hashes = {
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": frozen_hashes["manifest"],
        },
        "scene_split": {
            "path": str(args.scene_split.resolve()),
            "sha256": frozen_hashes["scene_split"],
        },
        "normalization": {
            "path": str(args.normalization.resolve()),
            "sha256": frozen_hashes["normalization"],
        },
        "range_azimuth_elevation_axes": {
            "path": str(
                (args.data_root / "resources/info_arr.mat").resolve()
            ),
            "sha256": sha256(
                args.data_root / "resources/info_arr.mat"
            ),
        },
        "doppler_axis": {
            "path": str(
                (args.data_root / "resources/arr_doppler.mat").resolve()
            ),
            "sha256": sha256(
                args.data_root / "resources/arr_doppler.mat"
            ),
        },
        "frames": frame_data_hashes(
            manifest["frames"],
            args.data_root,
            args.cache_root,
        ),
    }
    data_contract = canonical_data_contract(data_hashes)

    validation_set = KRadarCubeDataset(
        args.data_root,
        args.cache_root,
        args.manifest,
        ("validation",),
    )
    if len(validation_set) != FORMAL_VALIDATION_COUNT:
        raise ValueError("G1D diagnosis requires exactly 24 validation frames")
    validation_identities = [_identity(record) for record in validation_set.records]
    far_identities = range_target_frame_identities(
        args.cache_root,
        validation_set.records,
    )
    if len(far_identities) != EXPECTED_FAR_FRAME_COUNT:
        raise ValueError(
            f"G1D diagnosis requires 23 far-target frames, got "
            f"{len(far_identities)}"
        )

    corrected = json.loads(
        args.corrected_control.read_text(encoding="utf-8")
    )
    corrected_reference = validate_corrected_reference(
        corrected,
        corrected_path=args.corrected_control,
        data_contract=data_contract,
        validation_identities=validation_identities,
        far_identities=far_identities,
        evaluator_path=repo / "code/eval/dense_geometry.py",
    )

    checkpoint_sha = sha256(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = validate_checkpoint_metadata(checkpoint)
    source_compatibility = checkpoint_source_compatibility(repo, checkpoint)
    checkpoint_summary = {
        "path": str(args.checkpoint.resolve()),
        "sha256": checkpoint_sha,
        "epoch": int(checkpoint["epoch"]),
        "evaluation_state": "ema_model",
        "source_commit": checkpoint["provenance"]["git_commit"],
        "provenance_test_accessed": checkpoint["provenance"][
            "test_accessed"
        ],
        "config": asdict(config),
        "source_compatibility": source_compatibility,
    }

    axes = load_axes(args.data_root / "resources")
    model = build_model(config, axes, normalization).to(device)
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    model.eval().requires_grad_(False)
    del checkpoint

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    shuffled_positions = cross_scene_condition_indices(
        validation_set.records
    )
    derangement = [
        {
            "current": _identity(record),
            "shuffled": _identity(
                validation_set.records[shuffled_positions[index]]
            ),
        }
        for index, record in enumerate(validation_set.records)
    ]

    d1_frames = {alpha_label(alpha): [] for alpha in OFFSET_ALPHAS}
    d2_names = (
        "measured_current__condition_current",
        "measured_current__condition_shuffled",
        "measured_shuffled__condition_current",
        "measured_shuffled__condition_shuffled",
    )
    d2_frames = {name: [] for name in d2_names}
    d3_capability = ranking_capability(model)
    d3_frames = (
        {name: [] for name in RANKING_SOURCES}
        if d3_capability["supported"]
        else {}
    )
    proposal_cache: dict[tuple[int, int], torch.Tensor] = {}

    for position in range(len(validation_set)):
        current_item = validation_set[position]
        shuffled_item = validation_set[shuffled_positions[position]]
        current_identity = _identity(current_item)
        shuffled_identity = _identity(shuffled_item)
        current_cube = move_frame(current_item, device)
        shuffled_cube = move_frame(shuffled_item, device)
        target = current_item["target_xyz_confidence"].to(
            device,
            non_blocking=True,
        )
        current_proposals = cached_proposal_flat_index(
            current_item,
            current_cube,
            config,
            proposal_cache,
        )
        shuffled_proposals = cached_proposal_flat_index(
            shuffled_item,
            shuffled_cube,
            config,
            proposal_cache,
        )

        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            current_context = prepare_query_field(
                model,
                current_cube,
                current_cube,
                proposal_flat_index=current_proposals,
            )
        combined_score = current_ranking_score(current_context)

        baseline_generated = None
        baseline_evaluated = None
        d1_hashes = set()
        for alpha in OFFSET_ALPHAS:
            with torch.inference_mode(), torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                generated = generate_selected_points(
                    model,
                    current_context,
                    combined_score,
                    residual_alpha=alpha,
                )
            evaluated = evaluate_generated(
                generated,
                target,
                include_duplicates=True,
            )
            frame = {
                **current_identity,
                "measured_identity": current_identity,
                "condition_identity": current_identity,
                **evaluated,
            }
            d1_frames[alpha_label(alpha)].append(frame)
            d1_hashes.add(evaluated["selected_index_sha256"])
            if alpha == 1.0:
                baseline_generated = generated
                baseline_evaluated = evaluated
        if (
            len(d1_hashes) != 1
            or baseline_generated is None
            or baseline_evaluated is None
        ):
            raise AssertionError("D1 did not freeze top-k indices across alpha")

        d2_frames["measured_current__condition_current"].append(
            {
                **current_identity,
                "measured_identity": current_identity,
                "condition_identity": current_identity,
                **{
                    key: value
                    for key, value in baseline_evaluated.items()
                    if key != "duplicates"
                },
            }
        )

        if d3_capability["supported"]:
            frame_capability = ranking_capability(
                model,
                current_context.coarse_fields,
            )
            if not frame_capability["supported"]:
                if any(d3_frames.values()):
                    raise RuntimeError(
                        "D3 capability changed after partial evaluation"
                    )
                d3_capability = frame_capability
                d3_frames = {}
            else:
                d3_capability = frame_capability
                scores = ranking_scores(
                    model,
                    current_context,
                    validation_target_xyz=target[:, :3],
                )
                for name in RANKING_SOURCES:
                    if name == CURRENT_RANKING_SOURCE:
                        generated = baseline_generated
                        evaluated = baseline_evaluated
                    else:
                        with torch.inference_mode(), torch.autocast(
                            "cuda",
                            dtype=torch.bfloat16,
                        ):
                            generated = generate_selected_points(
                                model,
                                current_context,
                                scores[name],
                                residual_alpha=1.0,
                            )
                        evaluated = evaluate_generated(
                            generated,
                            target,
                            include_duplicates=True,
                        )
                    d3_frames[name].append(
                        {
                            **current_identity,
                            "ranking_source": name,
                            "validation_target_used_for_ranking": (
                                name == "validation_gt_distance_oracle"
                            ),
                            **evaluated,
                        }
                    )

        d2_contexts = (
            (
                "measured_current__condition_shuffled",
                current_cube,
                shuffled_cube,
                current_proposals,
                current_identity,
                shuffled_identity,
            ),
            (
                "measured_shuffled__condition_current",
                shuffled_cube,
                current_cube,
                shuffled_proposals,
                shuffled_identity,
                current_identity,
            ),
            (
                "measured_shuffled__condition_shuffled",
                shuffled_cube,
                shuffled_cube,
                shuffled_proposals,
                shuffled_identity,
                shuffled_identity,
            ),
        )
        for (
            name,
            measured_cube,
            condition_cube,
            proposals,
            measured_identity,
            condition_identity,
        ) in d2_contexts:
            with torch.inference_mode(), torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                prepared = prepare_query_field(
                    model,
                    measured_cube,
                    condition_cube,
                    proposal_flat_index=proposals,
                )
                generated = generate_selected_points(
                    model,
                    prepared,
                    current_ranking_score(prepared),
                    residual_alpha=1.0,
                )
            evaluated = evaluate_generated(
                generated,
                target,
                include_duplicates=False,
            )
            d2_frames[name].append(
                {
                    **current_identity,
                    "measured_identity": measured_identity,
                    "condition_identity": condition_identity,
                    **evaluated,
                }
            )
            del prepared, generated

        print(
            json.dumps(
                {
                    "completed_validation_frame": position + 1,
                    "validation_frame_count": FORMAL_VALIDATION_COUNT,
                    "current": current_identity,
                    "shuffled": shuffled_identity,
                }
            ),
            flush=True,
        )
        del (
            current_item,
            shuffled_item,
            current_cube,
            shuffled_cube,
            target,
            current_context,
            combined_score,
            baseline_generated,
        )
        torch.cuda.empty_cache()

    aggregate_kwargs = {
        "expected_frame_count": FORMAL_VALIDATION_COUNT,
        "expected_far_count": EXPECTED_FAR_FRAME_COUNT,
        "expected_point_count": EXPECTED_POINT_COUNT,
    }
    d1_arms = {
        name: aggregate_arm(frames, **aggregate_kwargs)
        for name, frames in d1_frames.items()
    }
    d2_arms = {
        name: aggregate_arm(frames, **aggregate_kwargs)
        for name, frames in d2_frames.items()
    }
    if d3_capability["supported"]:
        d3_arms = {
            name: aggregate_arm(frames, **aggregate_kwargs)
            for name, frames in d3_frames.items()
        }
        d3_document = {
            **d3_capability,
            "status": "completed",
            "arms": d3_arms,
            "contrasts": build_d3_contrasts(d3_arms),
            "validation_gt_distance_oracle": {
                "uses_validation_target_geometry": True,
                "uses_test_data": False,
                "deployment_eligible": False,
                "purpose": "diagnostic_upper_bound_only",
            },
        }
    else:
        d3_document = {
            **d3_capability,
            "status": "unsupported",
            "arms": {},
            "contrasts": None,
            "reason": (
                "Checkpoint architecture does not expose separable coarse "
                "occupancy, confidence, and energy ranking sources"
            ),
        }

    access_checks = {
        "manifest_has_zero_test_frames": manifest_counts["test_frame_count"]
        == 0,
        "dataset_constructed_with_validation_partition_only": True,
        "checkpoint_config_test_accessed_false": (
            checkpoint_summary["config"]["test_accessed"] is False
        ),
        "checkpoint_provenance_test_accessed_false": checkpoint_summary[
            "provenance_test_accessed"
        ]
        is False,
        "corrected_reference_test_accessed_false": corrected.get(
            "test_accessed"
        )
        is False,
        "d3_oracle_uses_validation_targets_only": True,
        "test_dataset_not_constructed": True,
        "test_labels_not_read": True,
        "test_metrics_not_computed": True,
    }
    if not all(access_checks.values()):
        raise AssertionError("G1D prohibited test-access contract failed")

    document = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "artifact_type": "g1d_epoch150_failure_factor_diagnosis",
        "completed": d3_document["status"] in {"completed", "unsupported"},
        "source": {
            "git_commit": args.source_commit,
            "script": str(script_path.resolve()),
            "script_sha256": sha256(script_path),
            "source_hashes": source_hashes(repo, script_path),
        },
        "checkpoint": checkpoint_summary,
        "evaluator": corrected_reference,
        "data_contract": data_contract,
        "validation_contract": {
            "partition": "validation",
            "frame_count": FORMAL_VALIDATION_COUNT,
            "frame_identities": validation_identities,
            "far_range_m": [60.0, 120.0],
            "far_target_frame_count": EXPECTED_FAR_FRAME_COUNT,
            "far_target_frame_identities": far_identities,
            "cross_scene_derangement": derangement,
        },
        "prohibited_test_access": {
            "test_accessed": False,
            "prohibited_partitions": ["test"],
            "checks": access_checks,
        },
        "experiments": {
            "D1_residual_offset_alpha": {
                "status": "completed",
                "alpha_values": list(OFFSET_ALPHAS),
                "jointly_scaled_residuals": [
                    "coarse_center_offset_bins",
                    "final_local_offset_bins",
                ],
                "topk_indices_frozen_across_alpha": True,
                "arms": d1_arms,
                "contrasts_vs_alpha_1": build_d1_contrasts(d1_arms),
            },
            "D2_measured_global_condition_2x2": {
                "status": "completed",
                "measured_path_components": [
                    "integrated_energy_seed_proposals",
                    "local_spectrum",
                    "local_energy",
                    "query_tokens",
                ],
                "global_condition_component": "full_raed_radar_encoder_tokens",
                "arms": d2_arms,
                "diagnosis": build_d2_contrasts(d2_arms),
            },
            "D3_ranking_source": d3_document,
        },
        "runtime": {
            "device": device_name,
            "torch_version": torch.__version__,
            "autocast": "bfloat16",
            "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
            "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
            "proposal_cache_entry_count": len(proposal_cache),
        },
        "test_accessed": False,
    }
    atomic_json(args.output, document)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "completed": document["completed"],
                "checkpoint_sha256": checkpoint_sha,
                "evaluator_sha256": corrected_reference[
                    "evaluator_sha256"
                ],
                "validation_frame_count": FORMAL_VALIDATION_COUNT,
                "far_target_frame_count": EXPECTED_FAR_FRAME_COUNT,
                "d3_status": d3_document["status"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
