#!/usr/bin/env python3
"""Build a deterministic sequence-isolated K-Radar train/validation/test split."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
FRAME_RATIO_TOLERANCE = 0.025


def read_official_frames(split_dir: Path) -> dict[int, list[str]]:
    frames: dict[int, set[str]] = defaultdict(set)
    for name in ("train.txt", "test.txt"):
        for line in (split_dir / name).read_text(encoding="utf-8").splitlines():
            sequence_text, label = line.strip().split(",", maxsplit=1)
            frames[int(sequence_text)].add(label)
    return {sequence: sorted(labels) for sequence, labels in frames.items()}


def read_descriptions(metadata_root: Path, sequences: list[int]) -> dict[int, tuple[str, ...]]:
    descriptions = {}
    for sequence in sequences:
        path = metadata_root / str(sequence) / "description.txt"
        tags = tuple(part.strip().lower() for part in path.read_text(encoding="utf-8").split(","))
        if len(tags) != 3:
            raise ValueError(f"Expected road,time,weather in {path}: {tags}")
        descriptions[sequence] = tags
    return descriptions


def assign_sequences(
    frames: dict[int, list[str]],
    descriptions: dict[int, tuple[str, ...]],
    seed: int,
) -> tuple[dict[str, list[int]], dict]:
    """Solve a sequence-level multilabel split with frame-ratio constraints."""

    rng = random.Random(seed)
    sequences = sorted(frames)
    splits = list(SPLIT_RATIOS)
    ratios = np.asarray([SPLIT_RATIOS[split] for split in splits])
    sizes = np.asarray([len(frames[sequence]) for sequence in sequences], dtype=float)
    tags = sorted({tag for values in descriptions.values() for tag in values})
    metrics = ["__all_frames__", *tags]
    binary_count = len(sequences) * len(splits)
    deviation_count = len(metrics) * len(splits) * 2
    variable_count = binary_count + deviation_count

    objective = np.zeros(variable_count)
    integrality = np.zeros(variable_count)
    integrality[:binary_count] = 1
    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.full(variable_count, np.inf)
    upper_bounds[:binary_count] = 1
    rows: list[np.ndarray] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def assignment_index(sequence_index: int, split_index: int) -> int:
        return sequence_index * len(splits) + split_index

    def deviation_index(metric_index: int, split_index: int, sign: int) -> int:
        return binary_count + (metric_index * len(splits) + split_index) * 2 + sign

    for sequence_index in range(len(sequences)):
        row = np.zeros(variable_count)
        for split_index in range(len(splits)):
            row[assignment_index(sequence_index, split_index)] = 1
            objective[assignment_index(sequence_index, split_index)] = (
                rng.random() * 1e-9
            )
        rows.append(row)
        row_lower.append(1)
        row_upper.append(1)

    for metric_index, metric in enumerate(metrics):
        coefficients = (
            sizes
            if metric == "__all_frames__"
            else np.asarray(
                [
                    sizes[index] if metric in descriptions[sequence] else 0.0
                    for index, sequence in enumerate(sequences)
                ]
            )
        )
        metric_total = float(coefficients.sum())
        weight = (20.0 if metric == "__all_frames__" else 1.0) / max(
            metric_total, 1.0
        )
        for split_index, ratio in enumerate(ratios):
            row = np.zeros(variable_count)
            for sequence_index, value in enumerate(coefficients):
                row[assignment_index(sequence_index, split_index)] = value
            positive = deviation_index(metric_index, split_index, 0)
            negative = deviation_index(metric_index, split_index, 1)
            row[positive] = -1
            row[negative] = 1
            objective[positive] = weight
            objective[negative] = weight
            rows.append(row)
            target = metric_total * ratio
            row_lower.append(target)
            row_upper.append(target)

    tag_sequence_count = Counter(
        tag for sequence in sequences for tag in descriptions[sequence]
    )
    for tag in tags:
        if tag_sequence_count[tag] < len(splits):
            continue
        member_indices = [
            index
            for index, sequence in enumerate(sequences)
            if tag in descriptions[sequence]
        ]
        for split_index in range(len(splits)):
            row = np.zeros(variable_count)
            for sequence_index in member_indices:
                row[assignment_index(sequence_index, split_index)] = 1
            rows.append(row)
            row_lower.append(1)
            row_upper.append(np.inf)

    total_frames = float(sizes.sum())
    for split_index, ratio in enumerate(ratios):
        row = np.zeros(variable_count)
        for sequence_index, size in enumerate(sizes):
            row[assignment_index(sequence_index, split_index)] = size
        rows.append(row)
        row_lower.append(total_frames * (ratio - FRAME_RATIO_TOLERANCE))
        row_upper.append(total_frames * (ratio + FRAME_RATIO_TOLERANCE))

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), row_lower, row_upper),
        options={"time_limit": 120},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Scene split optimization failed: {result.message}")
    binary = result.x[:binary_count].reshape(len(sequences), len(splits))
    assignment = {
        split: [
            sequence
            for sequence_index, sequence in enumerate(sequences)
            if binary[sequence_index, split_index] > 0.5
        ]
        for split_index, split in enumerate(splits)
    }
    frame_counts = {
        split: sum(len(frames[sequence]) for sequence in assigned)
        for split, assigned in assignment.items()
    }
    solver_report = {
        "method": "scipy.optimize.milp (HiGHS)",
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "objective": float(result.fun),
        "frame_ratio_tolerance": FRAME_RATIO_TOLERANCE,
        "achieved_frame_ratios": {
            split: frame_counts[split] / total_frames for split in splits
        },
        "attributes_required_in_all_splits": sorted(
            tag for tag, count in tag_sequence_count.items() if count >= len(splits)
        ),
    }
    return assignment, solver_report


def split_summary(
    sequences: list[int],
    frames: dict[int, list[str]],
    descriptions: dict[int, tuple[str, ...]],
) -> dict:
    tags = Counter()
    for sequence in sequences:
        for tag in descriptions[sequence]:
            tags[tag] += len(frames[sequence])
    return {
        "sequence_count": len(sequences),
        "frame_count": sum(len(frames[sequence]) for sequence in sequences),
        "sequences": sequences,
        "frame_count_by_tag": dict(sorted(tags.items())),
        "labels": {
            str(sequence): frames[sequence]
            for sequence in sequences
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--official-split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--exclude-sequences", type=int, nargs="*", default=[])
    parser.add_argument(
        "--exclusion-reason",
        default="explicitly excluded before split construction",
    )
    args = parser.parse_args()
    all_frames = read_official_frames(args.official_split_dir)
    excluded = sorted(set(args.exclude_sequences))
    unknown = sorted(set(excluded) - set(all_frames))
    if unknown:
        raise ValueError(f"Excluded sequences absent from official split: {unknown}")
    frames = {
        sequence: labels
        for sequence, labels in all_frames.items()
        if sequence not in excluded
    }
    descriptions = read_descriptions(args.metadata_root, sorted(frames))
    assignment, optimizer = assign_sequences(frames, descriptions, args.seed)
    sets = {split: set(sequences) for split, sequences in assignment.items()}
    overlap = {
        "train_validation": sorted(sets["train"] & sets["validation"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "validation_test": sorted(sets["validation"] & sets["test"]),
    }
    split_payload = {
        split: split_summary(sequences, frames, descriptions)
        for split, sequences in assignment.items()
    }
    assigned = set().union(*sets.values())
    required_attributes = optimizer["attributes_required_in_all_splits"]
    total_included_frames = sum(map(len, frames.values()))
    checks = {
        "all_included_sequences_assigned": assigned == set(frames),
        "frame_conservation": sum(
            summary["frame_count"] for summary in split_payload.values()
        )
        == total_included_frames,
        "frame_ratios_within_tolerance": all(
            abs(
                split_payload[split]["frame_count"] / total_included_frames
                - SPLIT_RATIOS[split]
            )
            <= FRAME_RATIO_TOLERANCE
            for split in SPLIT_RATIOS
        ),
        "required_attributes_in_all_splits": all(
            split_payload[split]["frame_count_by_tag"].get(attribute, 0) > 0
            for attribute in required_attributes
            for split in SPLIT_RATIOS
        ),
        "zero_sequence_overlap": sum(map(len, overlap.values())) == 0,
    }
    payload = {
        "protocol": "sequence-isolated labelled-frame split",
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "source_sequence_count": len(all_frames),
        "source_frame_count": sum(map(len, all_frames.values())),
        "included_sequence_count": len(frames),
        "included_frame_count": total_included_frames,
        "excluded_sequences": {
            str(sequence): {
                "frame_count": len(all_frames[sequence]),
                "reason": args.exclusion_reason,
            }
            for sequence in excluded
        },
        "optimizer": optimizer,
        "description_schema": ["road", "time", "weather"],
        "sequence_descriptions": {
            str(sequence): list(tags) for sequence, tags in descriptions.items()
        },
        "splits": split_payload,
        "leakage_audit": {
            "sequence_overlap": overlap,
            "sequence_overlap_count": sum(map(len, overlap.values())),
            "adjacent_frame_cross_split_possible": False,
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
    }
    if payload["leakage_audit"]["sequence_overlap_count"] != 0:
        raise RuntimeError(f"Sequence leakage detected: {overlap}")
    if any(not sequences for sequences in assignment.values()):
        raise RuntimeError(f"An empty split was generated: {assignment}")
    if not payload["gate_pass"]:
        raise RuntimeError(f"Scene split checks failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                split: {
                    "sequence_count": payload["splits"][split]["sequence_count"],
                    "frame_count": payload["splits"][split]["frame_count"],
                    "sequences": payload["splits"][split]["sequences"],
                    "frame_count_by_tag": payload["splits"][split]["frame_count_by_tag"],
                }
                for split in SPLIT_RATIOS
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
