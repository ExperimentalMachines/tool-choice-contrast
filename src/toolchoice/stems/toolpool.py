"""Deterministic distractor selection.

Selection is a pure function of (seed, stem_id, arm) and the pool contents. Same inputs, same
tools, forever -- no RNG, no dependence on file iteration order.
"""

from __future__ import annotations

from toolchoice.constants import DISTRACTOR_SEED
from toolchoice.hashing import ranking_key
from toolchoice.stems.model import PoolTool


class PoolError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)


def eligible_distractors(
    pool: list[PoolTool], *, stem_domain: str, family: str, exclude: frozenset[str] = frozenset()
) -> list[PoolTool]:
    """Pool entries admissible as distractors for this stem, in stable pool_id order."""
    out = []
    for tool in pool:
        if tool.pool_id in exclude or tool.name in exclude:
            continue
        # S.3.4 / E.4.7: a distractor from the stem's own domain risks being applicable.
        if tool.domain == stem_domain:
            continue
        # S.3.3: general-purpose tools are barred from Family S permanently.
        if family == "S" and tool.general_purpose:
            continue
        out.append(tool)
    return sorted(out, key=lambda t: t.pool_id)


def select(
    pool: list[PoolTool],
    *,
    stem_id: str,
    arm: str,
    count: int,
    stem_domain: str,
    family: str,
    exclude: frozenset[str] = frozenset(),
) -> list[PoolTool]:
    """Pick `count` distractors deterministically.

    The arm is part of the ranking key for Family S only: its arms differ in how many tools they
    offer, and S-one's single tool should not be forced to equal S-five's first. Family E arms
    share one selection, because their distractors must be identical across arms -- only the
    capability-bearing tool is allowed to vary.
    """
    if count == 0:
        return []
    candidates = eligible_distractors(pool, stem_domain=stem_domain, family=family, exclude=exclude)
    if len(candidates) < count:
        raise PoolError(
            "POOL_UNDERSUPPLIED",
            f"{stem_id}/{arm} needs {count} distractors, {len(candidates)} eligible",
        )
    key_arm = arm if family == "S" else "E-shared"
    ranked = sorted(candidates, key=lambda t: ranking_key(DISTRACTOR_SEED, f"{stem_id}\x00{key_arm}\x00{t.pool_id}"))
    return sorted(ranked[:count], key=lambda t: t.pool_id)
