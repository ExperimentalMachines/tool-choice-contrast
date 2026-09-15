"""Rendering. A pure function of (question, tools) -- it never sees the item.

Both executors render identically because rendering is the tokenizer's job; only `generate`
differs between them. Each run records prompt_sha256 per item so a local and a GPU run of the same
population can be proven to have shown the model the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def render(question: str, tools: list[dict[str, Any]] | None) -> str:
    """A tokenizer-free rendering used by the mock path and by hashing.

    Real backends call ``tokenizer.apply_chat_template(messages, tools=tools, ...)``. This function
    exists so the mock path and the prompt digest do not depend on a tokenizer being installed.
    """
    if not tools:
        return question
    catalogue = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{question}\n\n<tools>{catalogue}</tools>"


def prompt_sha256(question: str, tools: list[dict[str, Any]] | None) -> str:
    return hashlib.sha256(render(question, tools).encode("utf-8")).hexdigest()
