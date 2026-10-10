"""A run is scored with the parser for the model's native tool format, and says which one."""

from __future__ import annotations

from typing import Any

from tests.formatting.test_lfm import R1, R2
from toolchoice.scoring.score import score
from toolchoice.stems.synthetic import population


def _decision(report: dict[str, Any], arm: str) -> str:
    """The one decision an arm holding a single item was scored as."""
    matrix = report["routing"][arm]["confusion_matrix"]
    (decided,) = [label for row in matrix.values() for label, count in row.items() if count]
    return decided


def test_lfm_calls_are_calls_only_under_the_lfm_parser() -> None:
    # Whole stems, because the paired contrast refuses an incomplete one.
    items = population(s_count=1, e_count=1)
    replies = {"tcc-v1/E/0001/applicable": R1, "tcc-v1/S/0001/one": R2}
    generations = [
        {"example_id": i["example_id"], "output": replies.get(i["example_id"], "12"),
         "finish_reason": "stop"}
        for i in items
    ]

    lfm = score(items, generations, tool_format="lfm")
    assert lfm["tool_format"] == "lfm"
    assert (_decision(lfm, "E-applicable"), _decision(lfm, "S-one")) == ("CALL", "CALL")

    default = score(items, generations)
    assert default["tool_format"] == "qwen"
    assert (_decision(default, "E-applicable"), _decision(default, "S-one")) == ("ANSWER", "ANSWER")
