"""Atomic, deterministic artifact writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def dumps_artifact(payload: Any) -> str:
    """The house serialization for every derived JSON artifact."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_json_atomic(path: Path, payload: Any) -> Path:
    return write_text_atomic(path, dumps_artifact(payload))


def write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path
