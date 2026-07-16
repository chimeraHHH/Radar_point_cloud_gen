#!/usr/bin/env python3
"""Fetch the selective K-Radar members required by a G0 audit manifest."""

from __future__ import annotations

import argparse
import json
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-manifest-dir", type=Path, required=True)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--sequence-workers", type=int, default=3)
    parser.add_argument("--range-workers", type=int, default=3)
    parser.add_argument("--sequences", type=int, nargs="*", default=None)
    parser.add_argument(
        "--base-url", default="https://kaistavelab.tw5.quickconnect.to"
    )
    args = parser.parse_args()
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

    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.sequence_workers) as executor:
        futures = {
            executor.submit(fetch_sequence, sequence, records): sequence
            for sequence, records in sorted(grouped.items())
        }
        for future in as_completed(futures):
            sequence = futures[future]
            try:
                records = future.result()
            except Exception as error:
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
    summary = {
        "requested_sequences": sorted(grouped),
        "completed_sequences": sorted(completed),
        "failures": sorted(failures, key=lambda item: item["sequence"]),
    }
    (args.download_manifest_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
