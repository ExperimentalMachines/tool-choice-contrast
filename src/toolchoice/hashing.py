"""Hashing. Two record digests, and they are not interchangeable.

``content_hash`` covers the measurement surface only -- question, tools, gold.
A typo fix in an adjudication rationale does not move it, so a score recorded
months ago stays attributable to the corpus that produced it.

``record_hash`` covers everything. It is file identity.

A ``content_fingerprint`` drift is a new corpus version. A ``record_fingerprint``
drift on its own is reported as CONTENT_STABLE_RECORD_DRIFT and is not blocking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTENT_FIELDS = ("question", "tools", "gold_decision", "gold_call")


def stable_json(value: Any) -> str:
    """Upstream's exact serialization, so a digest computed here matches one computed there."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def content_hash(record: dict[str, Any]) -> str:
    return semantic_hash({k: record[k] for k in CONTENT_FIELDS})


def record_hash(record: dict[str, Any]) -> str:
    return semantic_hash({k: v for k, v in record.items() if k != "record_hash"})


def corpus_fingerprint(hashes: list[str]) -> str:
    """Order-independent, matching upstream's compute_dataset_fingerprint."""
    return hashlib.sha256("\n".join(sorted(hashes)).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ranking_key(seed: str, identifier: str) -> str:
    """Deterministic ordering that does not depend on input order or on a language RNG."""
    return hashlib.sha256(f"{seed}\x00{identifier}".encode()).hexdigest()
