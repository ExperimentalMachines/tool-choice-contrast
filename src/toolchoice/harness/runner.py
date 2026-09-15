"""Run a population through a backend. Produces evidence, never verdicts.

Scoring is deliberately not here and never runs on a GPU: the run emits generations, and those are
scored on a host afterwards, for free, as many times as needed. A runtime must not be the thing
that decides whether a model passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from toolchoice.harness.prompt import prompt_sha256, render
from toolchoice.harness.protocol import InferenceBackend

#: The complete set of metadata a backend receives. Nothing here names the gold decision, the
#: gold answer, or the arm -- see test_backend_never_receives_gold.
VISIBLE_METADATA = ("example_id",)


def run_population(
    backend: InferenceBackend,
    items: Sequence[dict[str, Any]],
    *,
    generation_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        tools = item["tools"] or None
        result = backend.generate(
            render(item["question"], tools),
            tools=tools,
            generation_config=generation_config,
            metadata={"example_id": item["example_id"]},
        )
        rows.append({
            "example_id": item["example_id"],
            "output": result.text,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "prompt_sha256": prompt_sha256(item["question"], tools),
            "backend": backend.name,
            "backend_metadata": result.backend_metadata,
        })
    return rows
