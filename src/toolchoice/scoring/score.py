"""Score a run: per-arm routing, coverage, and the paired contrast.

A reply that calls no tool is decided by the vendored keyword rules unless judgments from
``tcc judge`` are given. Scoring only reads judgments; it never calls a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

from opengrad.evaluation.routing import routing_metrics

from toolchoice.constants import ARM_TO_GOLD, FAMILY_ARMS
from toolchoice.ids import parse_example_id
from toolchoice.scoring.contrast import (
    BASELINE_ARM,
    Outcome,
    group_by_stem,
    tool_presence_sensitivity,
)
from toolchoice.scoring.decide import PROSE_DECISIONS, Decision, ToolFormat, decide_generation
from toolchoice.scoring.judge import Judgment, judge_identity, output_sha256

ARTIFACT_KIND = "CONTRAST_SCORES"

#: Below this, an arm's metrics are reported but flagged: a run that scored a fifth of its items
#: is not a measurement of the model, it is a measurement of the harness.
MIN_COVERAGE = 0.95


def _answer_is_correct(content: str, gold_answer: dict[str, Any] | None) -> bool:
    if not gold_answer:
        return True  # non-S arms are scored on the decision only
    value = str(gold_answer.get("value", "")).strip()
    return bool(value) and value in content


def _judged(decision: Decision, row: dict[str, Any], judgment: Judgment | None) -> Decision:
    example_id = row["example_id"]
    if judgment is None:
        raise ValueError(f"{example_id} decides without a tool call but has no judgment")
    if judgment.output_sha256 != output_sha256(row["output"]):
        raise ValueError(f"{example_id}: the judgment is stale, it judged a different reply")
    return replace(decision, decision=judgment.decision)


def score(
    items: Sequence[dict[str, Any]],
    generations: Sequence[dict[str, Any]],
    *,
    tool_format: ToolFormat = "qwen",
    judgments: Sequence[Judgment] | None = None,
) -> dict[str, Any]:
    by_id = {row["example_id"]: row for row in generations}
    missing = [item["example_id"] for item in items if item["example_id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} items have no generation, e.g. {missing[:3]}")

    if judgments is None:
        classifier: dict[str, Any] = {"kind": "keyword"}
        unused: dict[str, Judgment] = {}
    else:
        classifier = {"kind": "judge", **asdict(judge_identity(judgments))}
        unused = {j.example_id: j for j in judgments}

    outcomes: list[Outcome] = []
    for item in items:
        row = by_id[item["example_id"]]
        decision = decide_generation(row, tool_format=tool_format)
        if judgments is not None and decision.decision in PROSE_DECISIONS:
            decision = _judged(decision, row, unused.pop(item["example_id"], None))
        stem_id, arm = parse_example_id(item["example_id"])
        outcomes.append(Outcome(
            example_id=item["example_id"], stem_id=stem_id, arm=arm,
            gold_decision=item["gold_decision"], decision=decision,
            correct_answer=_answer_is_correct(decision.content, item.get("gold_answer")),
        ))
    if unused:
        raise ValueError(
            f"judgments for replies that are not prose decisions in this population: {sorted(unused)}"
        )

    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "tool_format": tool_format,
        "prose_classifier": classifier,
        "items": len(outcomes),
        "routing": _routing(outcomes),
        "coverage": _coverage(outcomes),
        "contrast": _contrast(outcomes),
    }


def _routing(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """routing_metrics per arm, never pooled.

    Refusal is correct in E-absent and wrong in S-none, so a pooled figure over the corpus is a
    nonsense number. Parse failures are excluded here -- routing_metrics raises on a label outside
    the canonical four -- and counted in `coverage` instead.
    """
    out: dict[str, Any] = {}
    for arm in ARM_TO_GOLD:
        arm_outcomes = [o for o in outcomes if o.arm == arm]
        scored = [o for o in arm_outcomes if o.decision.parsed]
        if not scored:
            out[arm] = {"status": "UNMEASURABLE", "discovered": len(arm_outcomes), "scored": 0}
            continue
        metrics = dict(routing_metrics([o.gold_decision for o in scored],
                                       [o.decision.decision for o in scored]))
        # Upstream returns 0.0 for a class with an empty denominator. That 0.0 is exactly how a
        # missing ANSWER population stayed invisible, so it is replaced by an explicit null.
        matrix = metrics["confusion_matrix"]
        assert isinstance(matrix, dict)
        for key, gold in (("no_call_accuracy", "ANSWER"), ("clarification_accuracy", "CLARIFY"),
                          ("unsupported_accuracy", "UNSUPPORTED"), ("must_call_accuracy", "CALL")):
            if sum(matrix[gold].values()) == 0:
                metrics[key] = None
        metrics["status"] = "MEASURED"
        metrics["discovered"] = len(arm_outcomes)
        metrics["scored"] = len(scored)
        out[arm] = metrics
    return out


def _coverage(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARM_TO_GOLD:
        arm_outcomes = [o for o in outcomes if o.arm == arm]
        if not arm_outcomes:
            continue
        failures = sum(1 for o in arm_outcomes if not o.decision.parsed)
        rate = (len(arm_outcomes) - failures) / len(arm_outcomes)
        out[arm] = {
            "items": len(arm_outcomes),
            "parse_failures": failures,
            "scored_fraction": rate,
            "status": "OK" if rate >= MIN_COVERAGE else "BLOCKED_LOW_COVERAGE",
        }
    return out


def _contrast(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    stems = group_by_stem(outcomes, "S")
    if not stems:
        return {}
    return {
        arm: tool_presence_sensitivity(stems, arm)
        for arm in FAMILY_ARMS["S"] if arm != BASELINE_ARM
    }
