#!/usr/bin/env python3
"""Pure-stdlib verifier for an immutable STDA-F0 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_NAME = "payload_manifest.json"
COMPLETE_NAME = "BUNDLE_COMPLETE.json"
BUNDLE_DOMAIN = b"stda_f0_bundle_v2\0"
PROTOCOL_SHA256 = (
    "a1bacd619ab462b92e8b0fbfccf142e9999ab18130996762f6964a3ccd2edf84"
)


def canonical_json_bytes(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_document(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"noncanonical JSON: {path}") from error
    if not isinstance(document, dict) or canonical_json_bytes(document) != payload:
        raise ValueError(f"noncanonical JSON bytes: {path}")
    return document, payload


def _safe_relative(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("bundle manifest path must be text")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe bundle manifest path: {value!r}")
    return path.as_posix()


def verify_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest, manifest_bytes = _canonical_document(root / MANIFEST_NAME)
    complete, complete_bytes = _canonical_document(root / COMPLETE_NAME)
    if manifest.get("schema") != "stda_f0_payload_manifest_v2":
        raise ValueError("STDA-F0 payload manifest schema changed")
    if complete.get("schema") != "stda_f0_bundle_complete_v2":
        raise ValueError("STDA-F0 complete-bundle schema changed")
    if complete.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("STDA-F0 complete-bundle protocol hash changed")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("STDA-F0 payload manifest has no file list")
    expected_paths: list[str] = []
    payload_bytes = 0
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("STDA-F0 payload row is not an object")
        relative = _safe_relative(row.get("path"))
        if relative in (MANIFEST_NAME, COMPLETE_NAME):
            raise ValueError("STDA-F0 metadata cannot list itself as payload")
        if type(row.get("size")) is not int or int(row["size"]) < 0:
            raise ValueError("STDA-F0 payload size is invalid")
        expected_hash = row.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError("STDA-F0 payload SHA-256 is invalid")
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"STDA-F0 payload is absent or symbolic: {relative}")
        if path.stat().st_size != row["size"]:
            raise ValueError(f"STDA-F0 payload size changed: {relative}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"STDA-F0 payload hash changed: {relative}")
        expected_paths.append(relative)
        payload_bytes += int(row["size"])
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
        raise ValueError("STDA-F0 payload paths are not unique canonical order")
    observed_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in (MANIFEST_NAME, COMPLETE_NAME)
    )
    if observed_paths != expected_paths:
        raise ValueError("STDA-F0 bundle contains unlisted or missing payloads")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    checks = {
        "payload_count": manifest.get("payload_count") == len(files),
        "payload_bytes": manifest.get("payload_bytes") == payload_bytes,
        "complete_payload_count": complete.get("payload_count") == len(files),
        "complete_manifest_sha256": complete.get("payload_manifest_sha256") == manifest_sha,
        "source_commit_full": isinstance(complete.get("source_commit"), str)
        and len(complete["source_commit"]) == 40,
    }
    if not all(checks.values()):
        raise ValueError(f"STDA-F0 complete-bundle checks failed: {checks}")
    root_hash = hashlib.sha256(
        BUNDLE_DOMAIN + manifest_bytes + b"\0" + complete_bytes
    ).hexdigest()
    return {
        "schema": "stda_f0_bundle_verification_v2",
        "root_path": str(root),
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": complete["source_commit"],
        "payload_count": len(files),
        "payload_bytes": payload_bytes,
        "complete_bundle_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "payload_manifest_sha256": manifest_sha,
        "bundle_complete_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "bundle_root": root_hash,
        "checks": checks,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(canonical_json_bytes(verify_bundle(args.root)).decode("ascii"), end="")


if __name__ == "__main__":
    main()
