#!/usr/bin/env python3
"""Build a deterministic sequence-isolated K-Radar train/validation/test split."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


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
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    sequences = list(frames)
    rng.shuffle(sequences)
    tag_frequency = Counter(tag for tags in descriptions.values() for tag in tags)
    sequences.sort(
        key=lambda sequence: (
            min(tag_frequency[tag] for tag in descriptions[sequence]),
            -len(frames[sequence]),
        )
    )
    total_frames = sum(map(len, frames.values()))
    total_by_tag = Counter()
    for sequence, tags in descriptions.items():
        for tag in tags:
            total_by_tag[tag] += len(frames[sequence])

    assignment = {split: [] for split in SPLIT_RATIOS}
    frame_count = Counter()
    tag_count: dict[str, Counter] = defaultdict(Counter)
    for sequence in sequences:
        size = len(frames[sequence])
        tags = descriptions[sequence]
        best_split = None
        best_cost = None
        for split, ratio in SPLIT_RATIOS.items():
            projected_frames = frame_count[split] + size
            global_error = abs(projected_frames - total_frames * ratio) / total_frames
            tag_error = sum(
                abs(tag_count[split][tag] + size - total_by_tag[tag] * ratio)
                / max(total_by_tag[tag], 1)
                for tag in tags
            )
            overflow = max(0.0, projected_frames - total_frames * (ratio + 0.05))
            cost = global_error + tag_error + 10.0 * overflow / total_frames
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = split
        assert best_split is not None
        assignment[best_split].append(sequence)
        frame_count[best_split] += size
        for tag in tags:
            tag_count[best_split][tag] += size
    return {split: sorted(values) for split, values in assignment.items()}


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
    args = parser.parse_args()
    frames = read_official_frames(args.official_split_dir)
    descriptions = read_descriptions(args.metadata_root, sorted(frames))
    assignment = assign_sequences(frames, descriptions, args.seed)
    sets = {split: set(sequences) for split, sequences in assignment.items()}
    overlap = {
        "train_validation": sorted(sets["train"] & sets["validation"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "validation_test": sorted(sets["validation"] & sets["test"]),
    }
    payload = {
        "protocol": "sequence-isolated labelled-frame split",
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "description_schema": ["road", "time", "weather"],
        "sequence_descriptions": {
            str(sequence): list(tags) for sequence, tags in descriptions.items()
        },
        "splits": {
            split: split_summary(sequences, frames, descriptions)
            for split, sequences in assignment.items()
        },
        "leakage_audit": {
            "sequence_overlap": overlap,
            "sequence_overlap_count": sum(map(len, overlap.values())),
            "adjacent_frame_cross_split_possible": False,
        },
    }
    if payload["leakage_audit"]["sequence_overlap_count"] != 0:
        raise RuntimeError(f"Sequence leakage detected: {overlap}")
    if any(not sequences for sequences in assignment.values()):
        raise RuntimeError(f"An empty split was generated: {assignment}")
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
