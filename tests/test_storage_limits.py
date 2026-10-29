"""Bound actual cached reads when a file changes after its initial stat."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_recommender.storage import download


def test_growing_cached_file_cannot_read_past_budget_plus_one(tmp_path, monkeypatch):
    path = tmp_path / "cached.bin"
    path.write_bytes(b"abcdefgh" * 2)
    path.with_suffix(".bin.receipt.json").write_text(
        json.dumps(
            {
                "url": "https://example.com/data",
                "bytes": 8,
                "sha256": hashlib.sha256(b"abcdefgh").hexdigest(),
            }
        )
    )
    original_stat, original_open = Path.stat, Path.open
    reads = []

    class Reader:
        def __init__(self):
            self.stream = io.BytesIO(b"abcdefgh" * 2)

        def readable(self):
            return True

        def read(self, size):
            value = self.stream.read(size)
            reads.append(len(value))
            return value

        def readinto(self, buffer):
            size = self.stream.readinto(buffer)
            reads.append(size)
            return size

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

    # Model a concurrent append: the earlier stat saw eight bytes, the open
    # stream now exposes sixteen. Limits must apply to actual reads as well.
    monkeypatch.setattr(
        Path,
        "stat",
        lambda target, *args, **kwargs: SimpleNamespace(st_size=8)
        if target == path
        else original_stat(target, *args, **kwargs),
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda target, *args, **kwargs: Reader()
        if target == path
        else original_open(target, *args, **kwargs),
    )
    with pytest.raises(ValueError, match="size limit"):
        download("https://example.com/data", path, max_bytes=8)
    assert sum(reads) <= 9


def test_deeply_nested_cached_receipt_is_a_controlled_validation_error(tmp_path):
    path = tmp_path / "cached.bin"
    path.write_bytes(b"data")
    path.with_suffix(".bin.receipt.json").write_bytes(b"[" * 4000 + b"]" * 4000)
    with pytest.raises(ValueError, match="receipt is malformed"):
        download("https://example.com/data", path)
