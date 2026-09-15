"""Identity, seeds and the arm taxonomy. Everything here is chosen once and frozen."""

from __future__ import annotations

CORPUS_ID = "tcc-v1"
SCHEMA_VERSION = 1

# Chosen once, not searched. Changing either is a new corpus version.
SPLIT_SEED = "tool-choice-contrast-v1"
DISTRACTOR_SEED = "tool-choice-contrast-distractors-v1"

UPSTREAM_PIN = {
    "repo": "https://github.com/arjhinety/OpenGrad",
    "commit": "62a5721b44cb07ebdb26de706be4915d9ddd8c71",
}

# The arm is definitional: it determines the gold decision. An adjudicator who
# disagrees with an arm's gold has found an authoring bug, not a label, and the
# build raises rather than accepting the override.
#
# Gold decisions use the canonical upstream vocabulary. `routing_metrics` raises
# ValueError on anything outside DECISIONS, and `no_call_accuracy` is keyed on
# matrix["ANSWER"]["ANSWER"] -- storing "DIRECT" would forfeit the metric this
# corpus exists to supply. "DIRECT" survives only as a prose alias.
ARM_TO_GOLD: dict[str, str] = {
    "S-none": "ANSWER",
    "S-one": "ANSWER",
    "S-five": "ANSWER",
    "E-applicable": "CALL",
    "E-underspecified": "CLARIFY",
    "E-absent": "UNSUPPORTED",
}

GOLD_TO_ALIAS: dict[str, str] = {
    "ANSWER": "DIRECT",
    "CALL": "CALL",
    "CLARIFY": "CLARIFY",
    "UNSUPPORTED": "UNSUPPORTED",
}

FAMILY_ARMS: dict[str, tuple[str, ...]] = {
    "S": ("S-none", "S-one", "S-five"),
    "E": ("E-applicable", "E-underspecified", "E-absent"),
}

ARMS: tuple[str, ...] = tuple(a for arms in FAMILY_ARMS.values() for a in arms)

# S-none carries no tool block at all, which is the point of the dose series:
# the renderer only emits a tool section when the list is non-empty, so S-none
# differs from S-five in template structure as well as tool content. S-one
# separates "a tool block exists" from "several tools exist".
ARM_TOOL_COUNTS: dict[str, tuple[int, int]] = {
    "S-none": (0, 0),
    "S-one": (1, 1),
    "S-five": (3, 5),
}

TARGET_STEMS: dict[str, int] = {"S": 100, "E": 200}

# Pre-registered before any run. 300 stems cannot support a 5pp guard; see the
# power table in the plan. Claiming 5pp on this corpus would be false.
GUARD_CLAIM_PP: dict[str, float] = {"S": 8.8, "E": 6.2}
