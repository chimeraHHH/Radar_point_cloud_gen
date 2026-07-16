#!/usr/bin/env python3
"""Record which official K-Radar sequence archives are currently published."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_dense.kradar_archive import (  # noqa: E402
    SynologySession,
    credentials_from_environment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--expected-sequences", type=int, default=58)
    parser.add_argument(
        "--base-url", default="https://kaistavelab.tw5.quickconnect.to"
    )
    args = parser.parse_args()
    account, password = credentials_from_environment()
    with SynologySession(
        args.base_url, account, password, proxy=args.proxy
    ) as client:
        response = client.session.get(
            f"{client.base_url}/webapi/entry.cgi",
            params={
                "api": "SYNO.FileStation.List",
                "version": "2",
                "method": "list",
                "folder_path": "/KRadar",
                "offset": 0,
                "limit": 1000,
                "_sid": client.sid,
            },
            timeout=client.timeout,
        )
        response.raise_for_status()
        listing = response.json()
    if not listing.get("success"):
        raise RuntimeError(f"File Station list failed: {listing}")
    available = []
    for item in listing["data"]["files"]:
        name = item["name"]
        stem, suffix = Path(name).stem, Path(name).suffix.lower()
        if suffix == ".zip" and stem.isdigit():
            available.append(int(stem))
    available = sorted(set(available))
    expected = set(range(1, args.expected_sequences + 1))
    payload = {
        "protocol": "official Synology File Station /KRadar folder listing",
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_sequences": args.expected_sequences,
        "available_sequence_count": len(available),
        "available_sequences": available,
        "missing_sequence_count": len(expected - set(available)),
        "missing_sequences": sorted(expected - set(available)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
