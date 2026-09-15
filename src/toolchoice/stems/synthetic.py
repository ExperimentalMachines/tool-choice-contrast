"""A synthetic population for the control suite.

Not authored content and never published: it exists so the scorer and the controls can be
exercised before any human writes a stem. Authoring is the expensive, irreversible step, and it
should follow proof that the measurement works, not precede it.
"""

from __future__ import annotations

from typing import Any

from toolchoice.stems.expand import expand
from toolchoice.stems.model import GoldTool, PoolTool, Stem, UnderspecifiedExtra

_DOMAINS = ("travel", "calendar", "messaging", "weather", "media", "smarthome", "logistics")


def pool() -> list[PoolTool]:
    tools = []
    for domain in _DOMAINS:
        for index in range(4):
            name = f"{domain}_action_{index}"
            tools.append(PoolTool(
                pool_id=f"{domain}/{name}@v1", domain=domain, name=name,
                description=f"Perform {domain} action {index}.",
                parameters={
                    "type": "object",
                    "properties": {"target": {"type": "string", "description": "What to act on."}},
                    "required": ["target"],
                    "additionalProperties": False,
                },
            ))
    # One barred general-purpose tool, so S.3.3 filtering is exercised rather than assumed.
    tools.append(PoolTool(
        pool_id="general/calculator@v1", domain="general", name="calculator",
        description="Evaluate an arithmetic expression.",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}},
                    "required": ["expression"], "additionalProperties": False},
        general_purpose=True,
    ))
    return sorted(tools, key=lambda t: t.pool_id)


def stems(*, s_count: int = 40, e_count: int = 12) -> list[Stem]:
    out: list[Stem] = []
    for i in range(1, s_count + 1):
        out.append(Stem(
            stem_ordinal=i, family="S", domain="math",
            question=f"A shop sold {i} crates of 12 apples each. How many apples is that?",
            source_file="synthetic", source_anchor=f"stems[{i}]",
            gold_answer={"value": str(i * 12), "type": "numeric"},
        ))
    for i in range(1, e_count + 1):
        out.append(Stem(
            stem_ordinal=i, family="E", domain="finance",
            question=f"What is the current balance on account 44{i:02d}-9920?",
            source_file="synthetic", source_anchor=f"stems[{i}]",
            capability="retrieve a bank account balance",
            gold_tool=GoldTool(
                name="get_account_balance",
                description="Return the current cleared balance for one account.",
                parameters={
                    "type": "object",
                    "properties": {"account_id": {"type": "string", "description": "Account number."}},
                    "required": ["account_id"], "additionalProperties": False,
                },
                bindings={"account_id": f"44{i:02d}-9920"},
            ),
            underspecified_extra=UnderspecifiedExtra(
                parameter="authorising_person_id",
                schema={"type": "string", "description": "Who is authorising the lookup."},
                justification=(
                    "No conventional default: the authorising person cannot be inferred from the "
                    "question and there is no sensible assumption (rubric E.4.2 condition 3)."
                ),
            ),
            absent_substitute_pool_id="messaging/messaging_action_0@v1",
        ))
    return out


def population(**kwargs: Any) -> list[dict[str, Any]]:
    tool_pool = pool()
    items: list[dict[str, Any]] = []
    for stem in stems(**kwargs):
        items.extend(expand(stem, tool_pool))
    return items
