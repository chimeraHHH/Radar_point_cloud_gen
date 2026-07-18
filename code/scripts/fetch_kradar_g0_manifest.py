#!/usr/bin/env python3
"""Fetch the selective K-Radar members required by a G0 audit manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar_archive import (  # noqa: E402
    SynologySession,
    credentials_from_environment,
    fetch_members,
)


def sequence_members(sequence: int, records: list[dict]) -> list[str]:
    prefix = f"{sequence}/"
    members = [
        f"{prefix}description.txt",
        f"{prefix}info_calib/calib_radar_lidar.txt",
        f"{prefix}time_info/os1-128.txt",
        f"{prefix}time_info/os2-64.txt",
    ]
    for record in records:
        members.extend(
            [
                f"{prefix}info_label/{record['label']}",
                f"{prefix}radar_tesseract/tesseract_{record['radar_index']:05d}.mat",
                f"{prefix}os2-64/os2-64_{record['lidar64_index']:05d}.pcd",
            ]
        )
    return list(dict.fromkeys(members))


def write_summary(
    path: Path,
    requested: set[int],
    completed: set[int],
    failures: dict[int, str],
    active: set[int],
    round_index: int,
) -> dict:
    document = {
        "requested_sequences": sorted(requested),
        "completed_sequences": sorted(completed),
        "failures": [
            {"sequence": sequence, "error": failures[sequence]}
            for sequence in sorted(failures)
            if sequence not in completed
        ],
        "pending_sequences": sorted(requested - completed),
        "active_sequences": sorted(active),
        "retry_round": round_index,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-manifest-dir", type=Path, required=True)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--sequence-workers", type=int, default=3)
    parser.add_argument("--range-workers", type=int, default=3)
    parser.add_argument("--retry-rounds", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=int, default=30)
    parser.add_argument("--sequences", type=int, nargs="*", default=None)
    parser.add_argument(
        "--base-url", default="https://kaistavelab.tw5.quickconnect.to"
    )
    args = parser.parse_args()
    if args.sequence_workers < 1 or args.range_workers < 1:
        raise ValueError("Download worker counts must be positive")
    if args.retry_rounds < 1 or args.retry_delay_seconds < 0:
        raise ValueError("Invalid retry schedule")
    account, password = credentials_from_environment()
    manifest = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in manifest["frames"]:
        grouped[int(record["sequence"])].append(record)
    if args.sequences is not None:
        requested = set(args.sequences)
        unknown = sorted(requested - set(grouped))
        if unknown:
            raise ValueError(f"Sequences absent from audit manifest: {unknown}")
        grouped = {
            sequence: records
            for sequence, records in grouped.items()
            if sequence in requested
        }
    args.download_manifest_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.download_manifest_dir / "summary.json"

    def fetch_sequence(sequence: int, records: list[dict]):
        with SynologySession(
            args.base_url,
            account,
            password,
            proxy=args.proxy,
        ) as client:
            return fetch_members(
                client,
                sequence,
                sequence_members(sequence, records),
                args.output,
                args.download_manifest_dir / f"sequence_{sequence:02d}.json",
                workers=args.range_workers,
            )

    requested = set(grouped)
    completed: set[int] = set()
    failures: dict[int, str] = {}
    last_round = 0
    for round_index in range(1, args.retry_rounds + 1):
        last_round = round_index
        active = requested - completed
        write_summary(
            summary_path, requested, completed, failures, active, round_index
        )
        if not active:
            break
        print(
            json.dumps(
                {
                    "event": "download_round_started",
                    "round": round_index,
                    "sequences": sorted(active),
                }
            ),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.sequence_workers) as executor:
            futures = {
                executor.submit(fetch_sequence, sequence, grouped[sequence]): sequence
                for sequence in sorted(active)
            }
            for future in as_completed(futures):
                sequence = futures[future]
                active.remove(sequence)
                try:
                    records = future.result()
                except Exception as error:
                    failures[sequence] = f"{type(error).__name__}: {error}"
                    print(
                        json.dumps(
                            {
                                "sequence": sequence,
                                "round": round_index,
                                "error": failures[sequence],
                            }
                        ),
                        flush=True,
                    )
                else:
                    completed.add(sequence)
                    failures.pop(sequence, None)
                    print(
                        json.dumps(
                            {
                                "sequence": sequence,
                                "round": round_index,
                                "members": len(records),
                                "completed": len(completed),
                                "total": len(requested),
                            }
                        ),
                        flush=True,
                    )
                write_summary(
                    summary_path,
                    requested,
                    completed,
                    failures,
                    active,
                    round_index,
                )
        if completed == requested:
            break
        if round_index < args.retry_rounds:
            print(
                json.dumps(
                    {
                        "event": "download_retry_wait",
                        "round": round_index,
                        "pending_sequences": sorted(requested - completed),
                        "delay_seconds": args.retry_delay_seconds,
                    }
                ),
                flush=True,
            )
            time.sleep(args.retry_delay_seconds)

    write_summary(
        summary_path,
        requested,
        completed,
        failures,
        set(),
        last_round,
    )
    if completed != requested:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
