"""Authored stem and tool-pool types. Frozen: a stem is authored once, then expanded."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PoolTool:
    """A distractor drawn from the shared pool.

    ``general_purpose`` marks the tools rubric S.3.3 bars from Family S permanently --
    calculators, code execution, search. A calculator offered against an arithmetic question is
    applicable, not a distractor, and the item would have no defensible single gold.
    """

    pool_id: str
    domain: str
    name: str
    description: str
    parameters: dict[str, Any]
    general_purpose: bool = False

    def as_tool(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass(frozen=True)
class GoldTool:
    """The tool an E stem's applicable arm offers, with every required parameter bound."""

    name: str
    description: str
    parameters: dict[str, Any]
    bindings: dict[str, str]

    def as_tool(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required", []))


@dataclass(frozen=True)
class UnderspecifiedExtra:
    """The extra required parameter that makes the underspecified arm underspecified.

    Rubric E.4.2: no evidence span, no schema default, and no conventional default a reasonable
    assistant would assume. ``justification`` records why condition 3 holds for this parameter,
    because that is the condition authors violate (E.4.3).
    """

    parameter: str
    schema: dict[str, Any]
    justification: str


@dataclass(frozen=True)
class Stem:
    stem_ordinal: int
    family: str
    domain: str
    question: str
    source_file: str
    source_anchor: str
    # Family S
    gold_answer: dict[str, Any] | None = None
    stem_source: dict[str, Any] | None = None
    # Family E
    capability: str | None = None
    gold_tool: GoldTool | None = None
    underspecified_extra: UnderspecifiedExtra | None = None
    absent_substitute_pool_id: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)
