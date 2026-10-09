"""Parse LFM2.5's native tool calls into the boundary type the vendored Qwen parser returns.

LFM2.5 writes a call as a Python call list between special tokens,
``<|tool_call_start|>[get_weather(city="Manila")]<|tool_call_end|>``. The vendored parser only
reads Qwen's ``<tool_call>`` tags, so without this module every LFM call scores as an ANSWER.

A reply with no envelope goes to ``parse_qwen_native_output`` untouched, so ANSWER, CLARIFY and
UNSUPPORTED come from one set of prose rules for both formats and the two stay comparable.

Strict like upstream: one malformed call makes the whole emission a FORMAT_ERROR, never a repaired
or partial call. OpenWeights' ``ToolCallParser.parseLfmStyle`` reads the same protocol on-device
but drops a call it cannot parse and keeps the rest. Here that would let a broken call score as a
clean ANSWER or a shorter CALL, so it is refused instead.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from opengrad.formatting.parser import ParsedCall, ParsedNativeOutput, parse_qwen_native_output

_START = "<|tool_call_start|>"
_END = "<|tool_call_end|>"
_ENVELOPE = re.compile(re.escape(_START) + "(.*?)" + re.escape(_END), re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class _MalformedCall(Exception):
    """An envelope body broke the call grammar; the message names how."""


def parse_lfm_native_output(raw: str, *, truncated: bool = False) -> ParsedNativeOutput:
    envelopes = list(_ENVELOPE.finditer(raw))
    if not envelopes:
        if _START in raw or _END in raw:
            content = _THINK.sub("", raw).strip()
            return ParsedNativeOutput(
                "ANSWER", [], content, "FORMAT_ERROR", ["UNCLOSED_TOOL_CALL"], truncated
            )
        return parse_qwen_native_output(raw, truncated=truncated)

    content = _THINK.sub("", _ENVELOPE.sub("", raw)).strip()
    calls: list[ParsedCall] = []
    errors: list[str] = []
    for index, envelope in enumerate(envelopes):
        try:
            calls.extend(_envelope_calls(envelope.group(1)))
        except _MalformedCall as exc:
            errors.append(f"MALFORMED_TOOL_CALL[{index}]: {exc}")
    if errors or not calls:
        return ParsedNativeOutput(
            "ANSWER", [], content, "FORMAT_ERROR", errors or ["EMPTY_TOOL_CALL"], truncated
        )
    return ParsedNativeOutput("CALL", calls, content, "RAW_VALID", [], truncated)


def _envelope_calls(body: str) -> list[ParsedCall]:
    """Read one envelope body; raises _MalformedCall naming the first violation."""
    try:
        payload = ast.parse(body.strip(), mode="eval").body
    except (SyntaxError, ValueError) as exc:
        raise _MalformedCall(f"not a Python expression ({exc})") from exc
    if not isinstance(payload, ast.List):
        raise _MalformedCall("payload is not a list of calls")
    return [_call(node) for node in payload.elts]


def _call(node: ast.expr) -> ParsedCall:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise _MalformedCall("element is not a call to a plain function name")
    name = node.func.id
    if node.args:
        raise _MalformedCall(f"{name}() has positional arguments")
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise _MalformedCall(f"{name}() unpacks **kwargs")
        # The grammar accepts a repeated keyword; keeping either value would repair the call.
        if keyword.arg in arguments:
            raise _MalformedCall(f"{name}() repeats argument {keyword.arg!r}")
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise _MalformedCall(f"{name}() argument {keyword.arg!r} is not a literal") from exc
    return ParsedCall(name, arguments)
