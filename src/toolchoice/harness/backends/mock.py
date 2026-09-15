"""A fully scripted model: a lookup from example_id to the exact text it will emit.

The script is resolved BEFORE the run, from the population, by the control that wants it. At run
time this backend knows nothing but an id, so it cannot condition on gold even by accident -- the
same property `harness.protocol` buys structurally for real backends.
"""

from __future__ import annotations

from typing import Any

from toolchoice.harness.protocol import GenerationResult

REFUSAL_TEXT = "I'm sorry, I'm unable to help with that."
CLARIFY_TEXT = "Which account should I look at?"
MALFORMED_TEXT = '<tool_call>{"name": "get_account_balance", "arguments": {'


def call_text(tool_name: str) -> str:
    import json

    return f'<tool_call>{json.dumps({"name": tool_name, "arguments": {}})}</tool_call>'


def answer_text(value: str) -> str:
    return f"The answer is {value}."


class MockBackend:
    """Deterministic, no model, no network, no knowledge of gold."""

    name = "mock"

    def __init__(self, script: dict[str, str]) -> None:
        self.script = script
        self.call_count = 0
        self.seen_metadata: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        generation_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        self.call_count += 1
        meta = dict(metadata or {})
        self.seen_metadata.append(meta)
        example_id = str(meta.get("example_id", ""))
        if example_id not in self.script:
            raise KeyError(f"no scripted response for {example_id!r}")
        text = self.script[example_id]
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="length" if text == MALFORMED_TEXT else "stop",
        )
