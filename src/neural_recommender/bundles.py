"""Atomic, content-verified model bundles with explicit legacy compatibility.

A publisher writes an immutable generation first, then replaces a small active
pointer. Readers consequently resolve either the previous complete generation or
the new one. Hash verification checks consistency; it is not a trust signature.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from .als import ALSConfig
from .conditional import validate_selector
from .storage import write_json


@contextmanager
def _bundle_write_lock(artifact_dir: Path) -> Iterator[None]:
    """Reject overlapping publishers/rollbacks before either can change pointers.

    Exclusive file creation works across processes and threads without a new
    dependency. A crashed writer leaves a fail-closed lock; an operator must
    confirm that no writer remains before removing it and retrying.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lock = artifact_dir / ".bundle-writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(
            "Another bundle writer is active or was interrupted; confirm it has stopped "
            "before removing .bundle-writer.lock"
        ) from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def file_hash(path: Path) -> str:
    """Stream artifacts rather than holding a dense ranker twice in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_bundle(artifact_dir: Path) -> Path:
    """Use the active generation, or an explicitly supported legacy directory."""
    pointer = artifact_dir / "active.json"
    if not pointer.exists():
        return artifact_dir
    try:
        value = json.loads(pointer.read_text())
        name = value["bundle"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Active bundle pointer is malformed") from error
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", name):
        raise ValueError("Active bundle name is invalid")
    resolved = (artifact_dir / "bundles" / name).resolve()
    if resolved.parent != (artifact_dir / "bundles").resolve() or not resolved.is_dir():
        raise ValueError("Active bundle does not resolve to an existing generation")
    return resolved


def verify_manifest(directory: Path, metadata: dict[str, Any]) -> None:
    """Validate the exact expected file list before deserializing weights."""
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")
    version = metadata.get("format_version")
    if type(version) is not int:
        raise ValueError("Checkpoint format version must be an integer")
    if version == 1:
        manifest = {"model.pt": metadata.get("checkpoint_sha256")}
    elif version in {2, 3}:
        seeds = metadata.get("seeds")
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= 10:
            raise ValueError("Checkpoint requires one to ten recorded neural seeds")
        expected = {"rankers.npz"} | {f"neural_{index}.pt" for index in range(len(seeds))}
        if version == 3:
            expected.add("als.npz")
        manifest = metadata.get("artifact_hashes", {})
        if not isinstance(manifest, dict) or set(manifest) != expected:
            raise ValueError("Checkpoint manifest is incomplete or contains unexpected paths")
    else:
        raise ValueError("Unsupported checkpoint format; use a matching release or retrain")
    for name, digest in manifest.items():
        if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
            raise ValueError("Checkpoint manifest contains an invalid digest")
        path = directory / name
        if not path.is_file() or path.is_symlink() or file_hash(path) != digest:
            raise ValueError("Checkpoint hash differs from its metadata or a file is missing")


def atomic_torch_save(value: object, path: Path) -> None:
    """Replace a checkpoint without truncating a hard-linked older generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".weights.tmp")
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write NumPy arrays without extending or rewriting linked files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(descriptor)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_bundle(
    artifact_dir: Path,
    source: Path,
    metadata: dict[str, Any],
    als_factors: np.ndarray,
) -> Path:
    """Publish format 3, retaining the old active pointer as a rollback target.

    Baseline arrays are hard-linked when possible so retaining rollback versions
    does not duplicate the roughly 60 MB EASE matrix. Old writers use atomic
    replacement, so subsequent experiment runs cannot mutate the linked inode.
    """
    with _bundle_write_lock(artifact_dir):
        return _publish_bundle(artifact_dir, source, metadata, als_factors)


def _publish_bundle(
    artifact_dir: Path,
    source: Path,
    metadata: dict[str, Any],
    als_factors: np.ndarray,
) -> Path:
    """Stage and activate one generation while holding its writer lock."""
    source_metadata = json.loads((source / "metadata.json").read_text())
    verify_manifest(source, source_metadata)
    if source_metadata["format_version"] not in {2, 3}:
        raise ValueError("Publishing requires an optimized format 2 or 3 source")
    if metadata.get("format_version") != 3:
        raise ValueError("The bundle publisher requires format 3 metadata")
    inherited = (
        "data_sha256",
        "movie_ids",
        "config",
        "seeds",
        "fusion_weights",
        "discovery_penalty",
        "protocol_sha256",
    )
    if any(metadata.get(key) != source_metadata.get(key) for key in inherited):
        raise ValueError("Published metadata differs from its frozen source model identity")
    config = ALSConfig(**metadata["als_config"])
    config.validate()
    if (
        als_factors.shape != (len(metadata["movie_ids"]), config.factors)
        or not np.isfinite(als_factors).all()
    ):
        raise ValueError("ALS factors do not match the published metadata")
    components = {"ease", "item_knn", "popularity", "neural_ensemble", "previous_hybrid", "als"}
    validate_selector(metadata.get("history_selector"), components)
    validate_selector(metadata.get("history_blends"), components)
    bundle_root = artifact_dir / "bundles"
    bundle_root.mkdir(exist_ok=True)
    label = f"v3-{uuid4().hex[:16]}"
    temporary = Path(tempfile.mkdtemp(prefix="staging-", dir=bundle_root))
    destination = bundle_root / label
    try:
        names = ["rankers.npz"] + [f"neural_{index}.pt" for index in range(len(metadata["seeds"]))]
        for name in names:
            try:
                os.link(source / name, temporary / name)
            except OSError:
                shutil.copyfile(source / name, temporary / name)
        atomic_savez(temporary / "als.npz", item_factors=als_factors)
        metadata = dict(metadata)
        metadata["artifact_hashes"] = {
            # Require the bytes verified before staging, rather than blessing
            # whatever a concurrently replaced source path happens to contain.
            **{name: source_metadata["artifact_hashes"][name] for name in names},
            "als.npz": file_hash(temporary / "als.npz"),
        }
        write_json(temporary / "metadata.json", metadata)
        verify_manifest(temporary, metadata)
        os.replace(temporary, destination)
        previous = artifact_dir / "active.json"
        if previous.exists():
            write_json(artifact_dir / "previous.json", json.loads(previous.read_text()))
        else:
            write_json(artifact_dir / "previous.json", {"legacy": True})
        write_json(previous, {"bundle": label})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def rollback_bundle(artifact_dir: Path) -> Path:
    """Restore the previous validated pointer without deleting any generation."""
    with _bundle_write_lock(artifact_dir):
        return _rollback_bundle(artifact_dir)


def _rollback_bundle(artifact_dir: Path) -> Path:
    """Validate and restore one rollback target while excluding other writers."""
    previous = artifact_dir / "previous.json"
    if not previous.exists():
        raise ValueError("No rollback target was recorded")
    target = json.loads(previous.read_text())
    if not isinstance(target, dict):
        raise ValueError("Rollback pointer must be a JSON object")
    if target == {"legacy": True}:
        directory = artifact_dir
        metadata = json.loads((directory / "metadata.json").read_text())
        verify_manifest(directory, metadata)
        (artifact_dir / "active.json").unlink(missing_ok=True)
        return directory
    name = target.get("bundle")
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", name):
        raise ValueError("Rollback pointer is invalid")
    directory = artifact_dir / "bundles" / name
    if directory.resolve().parent != (artifact_dir / "bundles").resolve():
        raise ValueError("Rollback bundle resolves outside the artifact directory")
    metadata = json.loads((directory / "metadata.json").read_text())
    verify_manifest(directory, metadata)
    write_json(artifact_dir / "active.json", target)
    return resolve_bundle(artifact_dir)
