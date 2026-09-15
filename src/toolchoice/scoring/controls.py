"""Scripted models with exactly known behaviour, and what the scorer must say about each.

A benchmark that has never been shown to detect the thing it claims to detect is not validated.
Each control is a policy plus a pre-registered window. The windows are committed before any real
model runs, so a control cannot be adjusted afterwards to match what the scorer happened to emit.

C4 is the one that matters: a model that answers bare questions and refuses the moment tools
appear is the regression this corpus exists to catch. If C4 does not fire, nothing downstream is
worth reporting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from toolchoice.harness.backends.mock import (
    CLARIFY_TEXT,
    MALFORMED_TEXT,
    REFUSAL_TEXT,
    answer_text,
    call_text,
)

Check = Callable[[dict[str, Any]], bool]
#: A script maps every example_id to the exact text the fake model will emit. It is built from the
#: population before the run, so the backend itself never sees gold.
Script = Callable[[list[dict[str, Any]]], dict[str, str]]


@dataclass(frozen=True)
class Control:
    control_id: str
    description: str
    script: Script
    proves: str
    expectations: dict[str, Check]


def _gold_text(item: dict[str, Any]) -> str:
    gold = item["gold_decision"]
    if gold == "ANSWER":
        return answer_text(str((item.get("gold_answer") or {}).get("value", "42")))
    if gold == "CALL":
        return call_text(item["gold_call"]["name"])
    if gold == "CLARIFY":
        return CLARIFY_TEXT
    return REFUSAL_TEXT


def _oracle(items: list[dict[str, Any]]) -> dict[str, str]:
    return {i["example_id"]: _gold_text(i) for i in items}


def _always_call(items: list[dict[str, Any]]) -> dict[str, str]:
    return {
        i["example_id"]: (call_text(i["tools"][0]["name"]) if i["tools"] else _gold_text(i))
        for i in items
    }


def _always_refuse(items: list[dict[str, Any]]) -> dict[str, str]:
    return {i["example_id"]: REFUSAL_TEXT for i in items}


def _refuse_if_tools(items: list[dict[str, Any]]) -> dict[str, str]:
    """The regression simulant: answers bare, refuses the moment any tool is offered."""
    out = {}
    for item in items:
        if item["gold_decision"] == "ANSWER":
            out[item["example_id"]] = REFUSAL_TEXT if item["tools"] else _gold_text(item)
        else:
            out[item["example_id"]] = _gold_text(item)
    return out


def _malformed_on_s_five(items: list[dict[str, Any]]) -> dict[str, str]:
    return {
        i["example_id"]: (MALFORMED_TEXT if i["example_id"].endswith("/five") else _gold_text(i))
        for i in items
    }


def _walk(result: dict[str, Any], path: str) -> Any:
    node: Any = result
    for depth, part in enumerate(path.split(".")):
        if node is None:
            return None
        if not isinstance(node, dict) or part not in node:
            available = sorted(node) if isinstance(node, dict) else type(node).__name__
            raise KeyError(
                f"control path {path!r} does not resolve: no {part!r} at depth {depth}; "
                f"available: {available}"
            )
        node = node[part]
    return node


def _is(path: str, expected: Any) -> Check:
    def check(result: dict[str, Any]) -> bool:
        return bool(_walk(result, path) == expected)

    return check


def _at_least(path: str, threshold: float) -> Check:
    def check(result: dict[str, Any]) -> bool:
        node = _walk(result, path)
        return node is not None and float(node) >= threshold

    return check


def _at_most(path: str, threshold: float) -> Check:
    def check(result: dict[str, Any]) -> bool:
        node = _walk(result, path)
        return node is not None and float(node) <= threshold

    return check


CONTROLS: tuple[Control, ...] = (
    Control(
        "C1", "oracle: emits each arm's gold behaviour", _oracle,
        "the scorer can score a perfect policy",
        {
            "daa_baseline_is_one": _at_least("contrast.S-five.daa_baseline.point", 1.0),
            "no_flips": _at_most("contrast.S-five.tps_flip.point", 0.01),
            "call_recall_is_one": _at_least("routing.E-applicable.must_call_accuracy", 1.0),
            "clarify_accuracy_is_one": _at_least("routing.E-underspecified.clarification_accuracy", 1.0),
            "unsupported_accuracy_is_one": _at_least("routing.E-absent.unsupported_accuracy", 1.0),
        },
    ),
    Control(
        "C2", "always calls when any tool is offered", _always_call,
        "detects the spurious-call flip direction, distinct from refusal",
        {
            "flips_almost_always": _at_least("contrast.S-five.tps_flip.point", 0.95),
            "flips_are_calls": _at_least("contrast.S-five.flip_destinations.CALL", 1),
            "no_refusal_flips": _at_most("contrast.S-five.flip_destinations.REFUSAL", 0),
            "no_call_accuracy_collapses": _at_most("routing.S-five.no_call_accuracy", 0.01),
        },
    ),
    Control(
        "C3", "refuses everything, the M0 shape", _always_refuse,
        "the headline metric reports UNDEFINED instead of a misleading 0.0",
        {
            "daa_is_zero": _at_most("contrast.S-five.daa_baseline.point", 0.01),
            "tps_is_undefined": _is("contrast.S-five.status", "UNDEFINED"),
            "tps_point_is_null": _is("contrast.S-five.tps_flip.point", None),
            "unsupported_still_right": _at_least("routing.E-absent.unsupported_accuracy", 1.0),
        },
    ),
    Control(
        "C4", "answers bare questions, refuses once tools appear", _refuse_if_tools,
        "THE control: the corpus detects the regression it exists to catch",
        {
            "daa_baseline_is_one": _at_least("contrast.S-five.daa_baseline.point", 1.0),
            "daa_collapses_with_tools": _at_most("contrast.S-five.daa_arm.point", 0.01),
            "flips_almost_always": _at_least("contrast.S-five.tps_flip.point", 0.95),
            "flips_are_refusals": _at_least("contrast.S-five.flip_destinations.REFUSAL", 1),
            "no_spurious_calls": _at_most("contrast.S-five.flip_destinations.CALL", 0),
            "one_tool_is_enough": _at_least("contrast.S-one.tps_flip.point", 0.95),
        },
    ),
    Control(
        "C6", "correct behaviour, malformed payloads on S-five", _malformed_on_s_five,
        "a parse failure is not counted as a direct answer",
        {
            "parse_failures_are_visible": _at_least("coverage.S-five.parse_failures", 1),
            "arm_reports_unmeasurable": _is("routing.S-five.status", "UNMEASURABLE"),
            "nothing_scored_on_that_arm": _is("routing.S-five.scored", 0),
            "flips_are_parse_failures": _at_least("contrast.S-five.flip_destinations.PARSE_FAILURE", 1),
        },
    ),
)

CONTROLS_BY_ID = {control.control_id: control for control in CONTROLS}
