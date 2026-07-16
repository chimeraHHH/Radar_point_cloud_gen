#!/usr/bin/env python3
"""Fetch small per-sequence K-Radar metadata needed for scene-isolated splits."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar_archive import (  # noqa: E402
    SynologySession,
    credentials_from_environment,
    fetch_members,
)


def metadata_members(sequence: int) -> list[str]:
    prefix = f"{sequence}/"
    return [
        f"{prefix}description.txt",
        f"{prefix}info_calib/calib_radar_lidar.txt",
        f"{prefix}time_info/os1-128.txt",
        f"{prefix}time_info/os2-64.txt",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--sequence-workers", type=int, default=4)
    parser.add_argument("--range-workers", type=int, default=2)
    parser.add_argument("--sequences", type=int, nargs="+", default=list(range(1, 59)))
    parser.add_argument(
        "--base-url", default="https://kaistavelab.tw5.quickconnect.to"
    )
    args = parser.parse_args()
    account, password = credentials_from_environment()
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    with SynologySession(
        args.base_url, account, password, proxy=args.proxy
    ) as client, ThreadPoolExecutor(max_workers=args.sequence_workers) as executor:
        futures = {
            executor.submit(
                fetch_members,
                client,
                sequence,
                metadata_members(sequence),
                args.output,
                args.manifest_dir / f"sequence_{sequence:02d}.json",
                args.range_workers,
            ): sequence
            for sequence in args.sequences
        }
        completed = []
        failures = []
        for future in as_completed(futures):
            sequence = futures[future]
            try:
                records = future.result()
            except Exception as error:  # Preserve progress across irregular archives.
                failures.append(
                    {
                        "sequence": sequence,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(json.dumps(failures[-1]), flush=True)
                continue
            completed.append(sequence)
            print(
                json.dumps(
                    {
                        "sequence": sequence,
                        "members": len(records),
                        "completed": len(completed),
                        "total": len(futures),
                    }
                ),
                flush=True,
            )
        failure_path = args.manifest_dir / "failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        if failures:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
