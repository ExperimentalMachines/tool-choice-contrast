"""One authored stem becomes its three arm items.

The question is copied byte for byte into every arm. Gate G3 re-checks that afterwards, but the
invariant is established here: no code path in this module may touch ``stem.question``.
"""

from __future__ import annotations

import copy
from typing import Any

from toolchoice.constants import ARM_TO_GOLD, ARM_TOOL_COUNTS, E_DISTRACTOR_COUNT, FAMILY_ARMS
from toolchoice.ids import example_id
from toolchoice.ids import stem_id as make_stem_id
from toolchoice.stems.model import PoolTool, Stem
from toolchoice.stems.toolpool import PoolError, select


def _underspecified_variant(stem: Stem) -> dict[str, Any]:
    """Same tool name and purpose, required set a strict superset (rubric E.4.2)."""
    assert stem.gold_tool is not None and stem.underspecified_extra is not None
    extra = stem.underspecified_extra
    params = copy.deepcopy(stem.gold_tool.parameters)
    params.setdefault("properties", {})[extra.parameter] = copy.deepcopy(extra.schema)
    required = list(params.get("required", []))
    if extra.parameter in required:
        raise PoolError("EXTRA_ALREADY_REQUIRED", f"{stem.stem_ordinal}:{extra.parameter}")
    params["required"] = sorted([*required, extra.parameter])
    if "default" in extra.schema:
        # Rubric E.4.2 condition 2. A schema default makes the parameter derivable, so CLARIFY
        # stops being the only defensible gold.
        raise PoolError("EXTRA_HAS_SCHEMA_DEFAULT", extra.parameter)
    return {
        "name": stem.gold_tool.name,
        "description": stem.gold_tool.description,
        "parameters": params,
    }


def expand(stem: Stem, pool: list[PoolTool]) -> list[dict[str, Any]]:
    """Return one partial record per arm. Hashes and provenance are attached by the builder."""
    sid = make_stem_id(stem.family, stem.stem_ordinal)
    items: list[dict[str, Any]] = []

    if stem.family == "S":
        for arm in FAMILY_ARMS["S"]:
            low, high = ARM_TOOL_COUNTS[arm]
            # A range picks its own size deterministically rather than always taking the low end.
            count = low if low == high else low + (int(_arm_hash(sid, arm), 16) % (high - low + 1))
            chosen = select(
                pool, stem_id=sid, arm=arm, count=count, stem_domain=stem.domain, family="S"
            )
            items.append(_item(stem, sid, arm, [t.as_tool() for t in chosen],
                              {t.name: "DISTRACTOR" for t in chosen},
                              [t.pool_id for t in chosen], None))
        return items

    if stem.gold_tool is None or stem.absent_substitute_pool_id is None:
        raise PoolError("E_STEM_INCOMPLETE", sid)

    # One selection shared by all three E arms: their distractors must be identical, so that the
    # only thing moving across arms is the capability-bearing tool.
    distractors = select(
        pool, stem_id=sid, arm="E-shared", count=E_DISTRACTOR_COUNT,
        stem_domain=stem.domain, family="E",
        exclude=frozenset({stem.gold_tool.name, stem.absent_substitute_pool_id}),
    )
    substitute = next((t for t in pool if t.pool_id == stem.absent_substitute_pool_id), None)
    if substitute is None:
        raise PoolError("SUBSTITUTE_NOT_IN_POOL", stem.absent_substitute_pool_id)
    if substitute.name == stem.gold_tool.name:
        raise PoolError("SUBSTITUTE_SHADOWS_GOLD", substitute.name)

    tail = [t.as_tool() for t in distractors]
    tail_roles = {t.name: "DISTRACTOR" for t in distractors}
    tail_ids = [t.pool_id for t in distractors]

    gold_call = {
        "name": stem.gold_tool.name,
        "required_arguments": stem.gold_tool.required,
        "argument_bindings": dict(stem.gold_tool.bindings),
        "argument_match_policy": "STRING_EXACT_AFTER_STRIP",
    }
    heads: dict[str, tuple[dict[str, Any], str, dict[str, Any] | None]] = {
        "E-applicable": (stem.gold_tool.as_tool(), "APPLICABLE", gold_call),
        "E-underspecified": (_underspecified_variant(stem), "UNDERSPECIFIED_VARIANT", None),
        "E-absent": (substitute.as_tool(), "DISTRACTOR", None),
    }
    for arm in FAMILY_ARMS["E"]:
        head, role, call = heads[arm]
        items.append(_item(
            stem, sid, arm, [head, *tail], {head["name"]: role, **tail_roles},
            [stem.absent_substitute_pool_id, *tail_ids] if arm == "E-absent" else tail_ids, call,
        ))
    return items


def _arm_hash(sid: str, arm: str) -> str:
    from toolchoice.constants import DISTRACTOR_SEED
    from toolchoice.hashing import ranking_key

    return ranking_key(DISTRACTOR_SEED, f"{sid}\x00{arm}\x00size")


def _item(
    stem: Stem, sid: str, arm: str, tools: list[dict[str, Any]],
    roles: dict[str, str], pool_ids: list[str], gold_call: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "example_id": example_id(sid, arm),
        # Copied, never rewritten. G3 asserts byte-identity across a stem's arms.
        "question": stem.question,
        "tools": tools,
        "tool_roles": roles,
        "tool_pool_ids": sorted(pool_ids),
        "gold_decision": ARM_TO_GOLD[arm],
        "gold_answer": stem.gold_answer,
        "gold_call": gold_call,
        "contrast": {
            "controlled_variable": "tool_availability",
            "held_constant": ["question", "domain", "entity_bindings"],
            "question_identical_to_siblings": True,
        },
        "stem_provenance": {
            "origin": "ADAPTED_FROM_PUBLIC" if stem.stem_source else "AUTHORED_ORIGINAL",
            "source_file": stem.source_file,
            "source_anchor": stem.source_anchor,
            "domain": stem.domain,
            "capability": stem.capability,
            "upstream": stem.stem_source,
        },
    }
