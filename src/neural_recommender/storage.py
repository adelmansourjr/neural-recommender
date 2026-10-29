"""Atomic local artifacts and TLS-verified public downloads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _bounded_fingerprint(path: Path, limit: int) -> tuple[int, str]:
    """Enforce the read budget even when a file grows after its size check."""
    hasher, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(min(1024 * 1024, limit - total + 1)):
            total += len(chunk)
            if total > limit:
                raise ValueError("Dataset exceeds the configured size limit")
            hasher.update(chunk)
    if not total:
        raise ValueError("Dataset is empty")
    return total, hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def download(url: str, path: Path, max_bytes: int = 30_000_000) -> dict[str, Any]:
    """Download to a temporary file, retaining content hashes for cached data."""
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("Dataset downloads require HTTPS")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Dataset size budget must be a positive integer")
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists():
        if not receipt_path.exists():
            raise ValueError(f"Unverified cached file: {path}. Remove it and download again.")
        size = path.stat().st_size
        if not 0 < size <= max_bytes:
            raise ValueError("Cached dataset is empty or exceeds the configured size limit")
        try:
            with receipt_path.open("rb") as stream:
                raw = stream.read(16_385)
            if len(raw) > 16_384:
                raise ValueError("Receipt exceeds its metadata size budget")
            receipt = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError("Cached dataset receipt is malformed") from error
        if (
            not isinstance(receipt, dict)
            or not isinstance(receipt.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", receipt["sha256"])
            or type(receipt.get("bytes")) is not int
        ):
            raise ValueError("Cached dataset receipt is malformed")
        actual_size, digest = _bounded_fingerprint(path, max_bytes)
        if (
            receipt.get("url") != url
            or receipt["bytes"] != actual_size
            or receipt["sha256"] != digest
        ):
            raise ValueError(f"Cached data failed provenance verification: {path}")
        return receipt
    if shutil.which("curl") is None:
        raise RuntimeError("Install curl to download data with system TLS verification")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".download")
    os.close(descriptor)
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "2",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--max-time",
                "120",
                "--max-filesize",
                str(max_bytes),
                "--output",
                temporary,
                url,
            ],
            check=True,
            timeout=380,
        )
        downloaded = Path(temporary)
        size = downloaded.stat().st_size
        if not 0 < size <= max_bytes:
            raise ValueError("Dataset download is empty or exceeds the configured size limit")
        size, digest = _bounded_fingerprint(downloaded, max_bytes)
        receipt = {
            "url": url,
            "sha256": digest,
            "bytes": size,
            "downloaded_at": datetime.now(UTC).isoformat(),
        }
        os.replace(temporary, path)
        write_json(receipt_path, receipt)
        return receipt
    finally:
        Path(temporary).unlink(missing_ok=True)
