#!/usr/bin/env python3
"""Verify a selective K-Radar G0 download against its frozen audit manifest."""

from __future__ import annotations

import argparse
import binascii
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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


def crc32(path: Path) -> str:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            checksum = binascii.crc32(chunk, checksum)
    return f"{checksum & 0xFFFFFFFF:08x}"


def verify_record(record: dict) -> dict:
    path = Path(record["output"])
    result = {
        "path": str(path),
        "expected_size": int(record["size"]),
        "expected_crc32": record["crc32"].lower(),
        "exists": path.is_file(),
    }
    if not result["exists"]:
        result.update({"actual_size": None, "actual_crc32": None, "valid": False})
        return result
    result["actual_size"] = path.stat().st_size
    result["actual_crc32"] = crc32(path)
    result["valid"] = (
        result["actual_size"] == result["expected_size"]
        and result["actual_crc32"] == result["expected_crc32"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download-manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")

    audit = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for frame in audit["frames"]:
        grouped[int(frame["sequence"])].append(frame)
    expected_sequences = sorted(grouped)
    summary_path = args.download_manifest_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else None
    )

    errors = []
    records_to_verify = []
    sequence_reports = []
    for sequence in expected_sequences:
        expected_members = sequence_members(sequence, grouped[sequence])
        expected_outputs = {
            str(args.data_root / member): member for member in expected_members
        }
        manifest_path = args.download_manifest_dir / f"sequence_{sequence:02d}.json"
        if not manifest_path.is_file():
            errors.append(f"Missing sequence manifest: {manifest_path}")
            sequence_reports.append(
                {
                    "sequence": sequence,
                    "expected_members": len(expected_members),
                    "manifest_members": 0,
                    "member_set_matches": False,
                }
            )
            continue
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_outputs = {record["output"] for record in records}
        expected_output_set = set(expected_outputs)
        missing = sorted(expected_output_set - manifest_outputs)
        unexpected = sorted(manifest_outputs - expected_output_set)
        duplicate_count = len(records) - len(manifest_outputs)
        if missing:
            errors.append(
                f"Sequence {sequence} missing {len(missing)} manifest members"
            )
        if unexpected:
            errors.append(
                f"Sequence {sequence} has {len(unexpected)} unexpected manifest members"
            )
        if duplicate_count:
            errors.append(
                f"Sequence {sequence} has {duplicate_count} duplicate manifest members"
            )
        sequence_reports.append(
            {
                "sequence": sequence,
                "expected_members": len(expected_members),
                "manifest_members": len(records),
                "member_set_matches": (
                    not missing and not unexpected and not duplicate_count
                ),
                "missing": missing,
                "unexpected": unexpected,
                "duplicate_count": duplicate_count,
            }
        )
        records_to_verify.extend(
            record for record in records if record["output"] in expected_output_set
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        file_reports = list(executor.map(verify_record, records_to_verify))
    invalid_files = [report for report in file_reports if not report["valid"]]
    if invalid_files:
        errors.append(f"Invalid local files: {len(invalid_files)}")

    expected_frame_count = len(audit["frames"])
    cube_count = sum(
        1 for report in file_reports if "/radar_tesseract/" in report["path"]
    )
    lidar_count = sum(1 for report in file_reports if "/os2-64/" in report["path"])
    label_count = sum(1 for report in file_reports if "/info_label/" in report["path"])
    if (cube_count, lidar_count, label_count) != (
        expected_frame_count,
        expected_frame_count,
        expected_frame_count,
    ):
        errors.append(
            "Frame artifact counts do not match the frozen audit manifest: "
            f"cube={cube_count}, lidar={lidar_count}, label={label_count}, "
            f"expected={expected_frame_count}"
        )

    summary_valid = bool(
        summary
        and summary.get("requested_sequences") == expected_sequences
        and summary.get("completed_sequences") == expected_sequences
        and not summary.get("failures")
    )
    if not summary_valid:
        errors.append("Downloader summary is absent, incomplete, or contains failures")
    report = {
        "passed": not errors,
        "expected_frame_count": expected_frame_count,
        "cube_count": cube_count,
        "lidar_count": lidar_count,
        "label_count": label_count,
        "expected_sequences": expected_sequences,
        "summary_valid": summary_valid,
        "verified_file_count": len(file_reports),
        "invalid_files": invalid_files,
        "sequence_reports": sequence_reports,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
