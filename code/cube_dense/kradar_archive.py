"""Selective, resumable access to K-Radar archives on Synology File Station."""

from __future__ import annotations

import binascii
import json
import os
import shutil
import struct
import time
import zlib
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from zipfile import ZipInfo

import requests
from remotezip import RemoteZip


LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_SIGNATURE = 0x04034B50
SUPPORTED_COMPRESSION = {0, 8}


@dataclass(frozen=True)
class DownloadRecord:
    member: str
    output: str
    size: int
    compressed_size: int
    crc32: str
    elapsed_seconds: float
    resumed_bytes: int


class SynologySession(AbstractContextManager["SynologySession"]):
    """Authenticated File Station session with no credentials persisted to disk."""

    def __init__(
        self,
        base_url: str,
        account: str,
        password: str,
        proxy: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.account = account
        self.password = password
        self.timeout = timeout
        self.session = requests.Session()
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})
        self.sid: str | None = None

    def login(self) -> None:
        response = self.session.post(
            f"{self.base_url}/webapi/entry.cgi",
            data={
                "api": "SYNO.API.Auth",
                "version": "7",
                "method": "login",
                "account": self.account,
                "passwd": self.password,
                "session": "FileStation",
                "format": "sid",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"Synology login failed: {payload}")
        self.sid = payload["data"]["sid"]

    def logout(self) -> None:
        if not self.sid:
            return
        try:
            self.session.post(
                f"{self.base_url}/webapi/entry.cgi",
                data={
                    "api": "SYNO.API.Auth",
                    "version": "7",
                    "method": "logout",
                    "session": "FileStation",
                    "_sid": self.sid,
                },
                timeout=self.timeout,
            )
        finally:
            self.sid = None

    def __enter__(self) -> "SynologySession":
        self.login()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.logout()
        self.session.close()

    def archive_url(self, sequence: int) -> str:
        if not self.sid:
            raise RuntimeError("Synology session is not authenticated")
        query = urlencode(
            {
                "api": "SYNO.FileStation.Download",
                "version": "2",
                "method": "download",
                "path": f"/KRadar/{sequence}.zip",
                "mode": "open",
                "_sid": self.sid,
            }
        )
        return f"{self.base_url}/webapi/entry.cgi?{query}"


def _request_range(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    timeout: int,
    attempts: int = 8,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, timeout),
            )
            if response.status_code != 206:
                body = response.text[:300]
                response.close()
                raise RuntimeError(
                    f"Range request {start}-{end} returned {response.status_code}: {body}"
                )
            return response
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Range request failed after {attempts} attempts") from last_error


def _data_offset(
    session: requests.Session,
    url: str,
    header_offset: int,
    timeout: int,
) -> int:
    response = _request_range(session, url, header_offset, header_offset + 4095, timeout)
    header = response.content
    response.close()
    if len(header) < LOCAL_FILE_HEADER.size:
        raise RuntimeError(f"Truncated local ZIP header at offset {header_offset}")
    signature, _, _, _, _, _, _, _, _, name_len, extra_len = LOCAL_FILE_HEADER.unpack_from(header)
    if signature != LOCAL_FILE_SIGNATURE:
        raise RuntimeError(f"Invalid local ZIP signature at offset {header_offset}")
    return header_offset + LOCAL_FILE_HEADER.size + name_len + extra_len


def _download_compressed_range(
    session: requests.Session,
    url: str,
    start: int,
    size: int,
    output: Path,
    timeout: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    resumed = output.stat().st_size if output.exists() else 0
    if resumed > size:
        output.unlink()
        resumed = 0

    with output.open("a+b") as handle:
        while handle.tell() < size:
            local_offset = handle.tell()
            request_start = start + local_offset
            request_end = min(start + size - 1, request_start + chunk_size - 1)
            expected = request_end - request_start + 1
            last_error: Exception | None = None
            for attempt in range(8):
                response: requests.Response | None = None
                received = 0
                try:
                    response = _request_range(
                        session, url, request_start, request_end, timeout
                    )
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        received += len(chunk)
                    handle.flush()
                    if received != expected:
                        raise RuntimeError(
                            f"Truncated range {request_start}-{request_end}: "
                            f"{received} != {expected}"
                        )
                    break
                except (requests.RequestException, RuntimeError) as exc:
                    last_error = exc
                    handle.seek(local_offset)
                    handle.truncate()
                    if attempt == 7:
                        raise RuntimeError(
                            f"Range {request_start}-{request_end} failed after retries"
                        ) from last_error
                    time.sleep(min(2**attempt, 30))
                finally:
                    if response is not None:
                        response.close()
    return resumed


def _crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            crc = binascii.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def _inflate_and_verify(compressed: Path, output: Path, info: ZipInfo) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    crc = 0
    written = 0
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if info.compress_type == 8 else None

    with compressed.open("rb") as source, temporary.open("wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            data = decompressor.decompress(chunk) if decompressor else chunk
            if data:
                target.write(data)
                written += len(data)
                crc = binascii.crc32(data, crc)
        if decompressor:
            tail = decompressor.flush()
            target.write(tail)
            written += len(tail)
            crc = binascii.crc32(tail, crc)

    crc &= 0xFFFFFFFF
    if written != info.file_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Uncompressed size mismatch for {info.filename}: {written} != {info.file_size}"
        )
    if crc != info.CRC:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"CRC mismatch for {info.filename}: {crc:08x} != {info.CRC:08x}")
    temporary.replace(output)


def fetch_members(
    client: SynologySession,
    sequence: int,
    members: Iterable[str],
    output_root: Path,
    manifest_path: Path,
) -> list[DownloadRecord]:
    """Fetch selected ZIP members with byte-range resume and CRC verification."""

    url = client.archive_url(sequence)
    requested = list(dict.fromkeys(members))
    with RemoteZip(url, session=client.session) as archive:
        index = {info.filename: info for info in archive.infolist()}

    missing = sorted(set(requested) - set(index))
    if missing:
        raise FileNotFoundError(f"Members absent from sequence {sequence}: {missing}")

    records: list[DownloadRecord] = []
    scratch = output_root / ".compressed"
    for member in requested:
        info = index[member]
        if info.compress_type not in SUPPORTED_COMPRESSION:
            raise RuntimeError(
                f"Unsupported compression type {info.compress_type} for {info.filename}"
            )
        output = output_root / member
        if (
            output.exists()
            and output.stat().st_size == info.file_size
            and _crc32_file(output) == info.CRC
        ):
            records.append(
                DownloadRecord(
                    member=member,
                    output=str(output),
                    size=info.file_size,
                    compressed_size=info.compress_size,
                    crc32=f"{info.CRC:08x}",
                    elapsed_seconds=0.0,
                    resumed_bytes=info.compress_size,
                )
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps([asdict(record) for record in records], indent=2) + "\n",
                encoding="utf-8",
            )
            continue

        started = time.monotonic()
        data_offset = _data_offset(
            client.session, url, info.header_offset, client.timeout
        )
        compressed = scratch / f"{info.CRC:08x}-{Path(member).name}.part"
        resumed = _download_compressed_range(
            client.session,
            url,
            data_offset,
            info.compress_size,
            compressed,
            client.timeout,
        )
        _inflate_and_verify(compressed, output, info)
        compressed.unlink(missing_ok=True)
        records.append(
            DownloadRecord(
                member=member,
                output=str(output),
                size=info.file_size,
                compressed_size=info.compress_size,
                crc32=f"{info.CRC:08x}",
                elapsed_seconds=round(time.monotonic() - started, 3),
                resumed_bytes=resumed,
            )
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps([asdict(record) for record in records], indent=2) + "\n",
            encoding="utf-8",
        )

    shutil.rmtree(scratch, ignore_errors=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def credentials_from_environment() -> tuple[str, str]:
    account = os.environ.get("KRADAR_NAS_USER")
    password = os.environ.get("KRADAR_NAS_PASSWORD")
    if not account or not password:
        raise RuntimeError("Set KRADAR_NAS_USER and KRADAR_NAS_PASSWORD")
    return account, password
