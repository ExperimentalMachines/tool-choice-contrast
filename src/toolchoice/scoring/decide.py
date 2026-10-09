"""Turn a raw generation into a decision, keeping "did not parse" separate from "answered".

``parse_qwen_native_output`` returns decision "ANSWER" on every FORMAT_ERROR path. Reusing its
decision verbatim would let a truncated generation count as a direct answer and inflate the
headline metric, so status is consulted first and a format error never becomes a decision.

Each model family writes tool calls in its own native format, so the run's tool format picks the
parser. Every parser returns the same ``ParsedNativeOutput`` and the rule above holds for all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from opengrad.data.behavior import DECISIONS
from opengrad.evaluation.capability import RefusalVerdict, detect_refusal
from opengrad.formatting.parser import ParsedNativeOutput, parse_qwen_native_output

from toolchoice.formatting.lfm import parse_lfm_native_output

ToolFormat = Literal["qwen", "lfm"]

PARSERS: Mapping[ToolFormat, Callable[..., ParsedNativeOutput]] = {
    "qwen": parse_qwen_native_output,
    "lfm": parse_lfm_native_output,
}

PARSE_FAILURE = "PARSE_FAILURE"


@dataclass(frozen=True)
class Decision:
    """What the model did, with the evidence that says so."""

    decision: str  # one of DECISIONS, or PARSE_FAILURE
    parse_status: str
    refusal: RefusalVerdict
    content: str
    call_names: tuple[str, ...]

    @property
    def parsed(self) -> bool:
        return self.decision in DECISIONS


def decide(
    raw: str | None, *, truncated: bool = False, tool_format: ToolFormat = "qwen"
) -> Decision:
    if raw is None or not raw.strip():
        return Decision(PARSE_FAILURE, "EMPTY", detect_refusal(raw), "", ())
    parsed = PARSERS[tool_format](raw, truncated=truncated)
    refusal = detect_refusal(parsed.content or raw)
    names = tuple(call.name for call in parsed.calls)
    if parsed.status != "RAW_VALID":
        # The parser still reports a decision here. We drop it: a malformed payload tells us the
        # model tried to call something, not that it answered.
        return Decision(PARSE_FAILURE, parsed.status, refusal, parsed.content or "", names)
    decision = parsed.decision
    if decision not in DECISIONS:
        return Decision(PARSE_FAILURE, "UNMAPPED_DECISION", refusal, parsed.content or "", names)
    return Decision(decision, parsed.status, refusal, parsed.content or "", names)
