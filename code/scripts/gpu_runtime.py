#!/usr/bin/env python3
"""GPU selection helpers for heterogeneous servers."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping


def gpu_name(index: int) -> str:
    name = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if not name:
        raise ValueError(f"GPU {index} has no reported model name")
    return name


def validate_gpu_candidates(candidates: list[int], required_name: str | None) -> None:
    if not candidates:
        raise ValueError("At least one GPU candidate is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("GPU candidates must be unique")
    if required_name is None:
        return
    names = {index: gpu_name(index) for index in candidates}
    mismatches = {index: name for index, name in names.items() if name != required_name}
    if mismatches:
        raise ValueError(
            f"GPU candidates must all be {required_name!r}; found {mismatches}"
        )


def cuda_environment(
    index: int, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # nvidia-smi uses PCI order. Pin CUDA to the same order before applying an index.
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(index)
    return environment
