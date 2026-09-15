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


# ── held-out material the corpus must not overlap ──────────────────────────────────────────────
#
# OpenGrad pins these as derived parquet files under a gitignored .cache directory, and nothing in
# that repository produces them -- so those hashes cannot be reproduced from a clean clone. We pin
# the UPSTREAM files instead, which are the bytes on the Hub and therefore genuinely reproducible.
#
# Verified equivalent: normalizing `question` over these two files yields 2,295 distinct MCQ
# prompts and 286 llm-judge prompts, all 286 contained in the MCQ set -- matching Study 002 doc 32
# exactly. See reports/data/heldout-index-pins.json.
HELDOUT_REPOSITORY = "nvidia/When2Call"
HELDOUT_REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"
HELDOUT_FILES: dict[str, str] = {
    "test/when2call_test_mcq.jsonl":
        "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c",
    "test/when2call_test_llm_judge.jsonl":
        "0b710578e3b02479e5acf5688140ec5edf71383fa5c093481c7536b57fd13b25",
}


def normalize_prompt(text: str) -> str:
    """Case-folded, whitespace-collapsed text: the P-DET-v1 prompt-matching rule.

    Reimplemented rather than vendored: it lives in ``opengrad.data.classifier_input``, which
    imports the normalization-v3 builder and its pyarrow dependency. The cross-check above is what
    makes the reimplementation safe -- it reproduces upstream's published counts exactly.
    """
    return " ".join(str(text).split()).casefold()


# Every E arm offers the capability-bearing tool plus this many distractors, so len(tools) is
# exactly equal across the three arms and catalogue size cannot explain a behaviour change
# (rubric E.4.5). Gate G5 asserts the equality rather than testing it statistically.
E_DISTRACTOR_COUNT = 2
