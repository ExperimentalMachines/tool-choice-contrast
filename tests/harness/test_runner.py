from __future__ import annotations

from typing import Any

import pytest

from toolchoice.harness.backends.mock import MockBackend
from toolchoice.harness.prompt import prompt_sha256, render
from toolchoice.harness.runner import VISIBLE_METADATA, run_population
from toolchoice.scoring.controls import CONTROLS_BY_ID
from toolchoice.stems.synthetic import population

FORBIDDEN = ("gold", "expected", "arm", "answer", "decision", "family", "stem")


def test_backend_never_receives_gold() -> None:
    """The reason this package does not use upstream's evaluation-side backend protocol."""
    items = population(s_count=4, e_count=2)
    backend = MockBackend(CONTROLS_BY_ID["C1"].script(items))
    run_population(backend, items)
    assert backend.seen_metadata
    for seen in backend.seen_metadata:
        assert set(seen) == {"example_id"}
        for key in seen:
            assert not any(word in key.lower() for word in FORBIDDEN if word != "example")
    assert VISIBLE_METADATA == ("example_id",)


def test_prompt_is_a_pure_function_of_question_and_tools() -> None:
    """Two arms of different stems with the same question and tools render identically."""
    question = "What is 6 times 7?"
    tools: list[dict[str, Any]] = [{"name": "t", "description": "d", "parameters": {}}]
    assert render(question, tools) == render(question, list(tools))
    assert prompt_sha256(question, None) != prompt_sha256(question, tools)


def test_no_tools_renders_without_a_tool_block() -> None:
    """S-none differs from S-five in prompt structure, which is why S-one exists."""
    assert "<tools>" not in render("q", None)
    assert "<tools>" not in render("q", [])
    assert "<tools>" in render("q", [{"name": "t", "description": "d", "parameters": {}}])


def test_generation_rows_carry_the_prompt_digest() -> None:
    items = population(s_count=2, e_count=1)
    rows = run_population(MockBackend(CONTROLS_BY_ID["C1"].script(items)), items)
    assert len(rows) == len(items)
    assert all(len(row["prompt_sha256"]) == 64 for row in rows)


def test_unscripted_example_is_an_error_not_a_default() -> None:
    items = population(s_count=2, e_count=1)
    with pytest.raises(KeyError):
        run_population(MockBackend({}), items)
