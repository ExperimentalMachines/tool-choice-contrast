"""The headline metric: does offering irrelevant tools stop the model answering?

Two numbers, reported as a pair and never apart.

  DAA(arm)      how often the model actually attempts a direct answer on that arm
  TPS_flip(arm) among stems it answered with NO tools, how often it stops once tools appear

TPS_flip is UNDEFINED, not 0.0, when fewer than MIN_CONDITIONING_STEMS stems were answered bare.
This is the metric's central failure mode and it is not an edge case: a model that already refuses
everything -- the exact regression this corpus exists to catch -- has nothing left to flip, so it
scores the same 0.0 a healthy model would. DAA(S-none) is what detects that. Control C3 exists to
prove the undefined path fires rather than quietly emitting a number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from toolchoice.constants import FAMILY_ARMS
from toolchoice.scoring.decide import Decision
from toolchoice.scoring.stats import Interval, cluster_bootstrap, mcnemar_exact, wilson

BASELINE_ARM = "S-none"
MIN_CONDITIONING_STEMS = 30


@dataclass(frozen=True)
class Outcome:
    """One scored item."""

    example_id: str
    stem_id: str
    arm: str
    gold_decision: str
    decision: Decision
    correct_answer: bool = False

    @property
    def routed(self) -> bool:
        """Chose to answer directly, and the generation actually parsed."""
        return self.decision.decision == "ANSWER"

    @property
    def attempted(self) -> bool:
        """Routed to an answer AND produced one we could extract and check."""
        return self.routed and self.correct_answer is not None and self.decision.parsed

    @property
    def correct(self) -> bool:
        return self.routed and bool(self.correct_answer)


@dataclass(frozen=True)
class StemOutcome:
    stem_id: str
    arms: dict[str, Outcome]


def group_by_stem(outcomes: Sequence[Outcome], family: str) -> list[StemOutcome]:
    """Group into stems and drop nothing silently: an incomplete stem raises."""
    expected = set(FAMILY_ARMS[family])
    buckets: dict[str, dict[str, Outcome]] = {}
    for outcome in outcomes:
        if outcome.arm in expected:
            buckets.setdefault(outcome.stem_id, {})[outcome.arm] = outcome
    stems = []
    for stem_id, arms in sorted(buckets.items()):
        if set(arms) != expected:
            # Shrinking the denominator instead would leave per-arm accuracy looking fine while
            # the paired contrast silently measured a different population.
            raise ValueError(
                f"{stem_id} has arms {sorted(arms)}, expected {sorted(expected)}; "
                "a paired contrast cannot be computed over an incomplete stem"
            )
        stems.append(StemOutcome(stem_id, arms))
    return stems


def direct_answer_availability(stems: Sequence[StemOutcome], arm: str) -> Interval:
    attempts = sum(1 for s in stems if s.arms[arm].correct)
    return wilson(attempts, len(stems))


def tool_presence_sensitivity(stems: Sequence[StemOutcome], arm: str) -> dict[str, Any]:
    """Flip rate away from a direct answer, conditioned on having answered bare."""
    conditioning = [s for s in stems if s.arms[BASELINE_ARM].routed]
    flipped = [s for s in conditioning if not s.arms[arm].routed]
    b = len(flipped)
    c = sum(1 for s in stems if not s.arms[BASELINE_ARM].routed and s.arms[arm].routed)

    destinations = {"CALL": 0, "CLARIFY": 0, "UNSUPPORTED": 0, "PARSE_FAILURE": 0, "REFUSAL": 0}
    for stem in flipped:
        decision = stem.arms[arm].decision
        if decision.refusal.is_refusal:
            destinations["REFUSAL"] += 1
        destinations[decision.decision] = destinations.get(decision.decision, 0) + 1

    measurable = len(conditioning) >= MIN_CONDITIONING_STEMS
    if measurable:
        def statistic(sample: Sequence[Any]) -> float | None:
            cond = [s for s in sample if s.arms[BASELINE_ARM].routed]
            if not cond:
                return None
            return sum(1 for s in cond if not s.arms[arm].routed) / len(cond)

        flip = wilson(b, len(conditioning))
        boot = cluster_bootstrap(list(stems), statistic)
    else:
        flip = Interval(None, None, None, f"UNDEFINED[conditioning={len(conditioning)}<{MIN_CONDITIONING_STEMS}]")
        boot = Interval(None, None, None, "UNDEFINED[insufficient_conditioning]")

    return {
        "arm": arm,
        "baseline_arm": BASELINE_ARM,
        "status": "MEASURED" if measurable else "UNDEFINED",
        "conditioning_stems": len(conditioning),
        "min_conditioning_stems": MIN_CONDITIONING_STEMS,
        "tps_flip": flip.as_dict(),
        "tps_flip_bootstrap": boot.as_dict(),
        "tps_net": (b - c) / len(stems) if stems else None,
        "mcnemar": mcnemar_exact(b, c),
        "flip_destinations": destinations,
        # Never report the sensitivity without the level: a model that answers nothing bare has
        # nothing to flip, and scores the same as a model that flips nothing.
        "daa_baseline": direct_answer_availability(stems, BASELINE_ARM).as_dict(),
        "daa_arm": direct_answer_availability(stems, arm).as_dict(),
    }
