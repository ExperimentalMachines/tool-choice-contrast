"""Shared fixtures. The factory builds a valid record so every test mutates one field."""

from __future__ import annotations

from typing import Any

import pytest

from toolchoice.constants import ARM_TO_GOLD
from toolchoice.ids import example_id, stem_id
from toolchoice.population.record import finalize


def _tool(name: str, required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Test tool {name}.",
        "parameters": {
            "type": "object",
            "properties": {p: {"type": "string"} for p in required},
            "required": required,
            "additionalProperties": False,
        },
    }


def make_record(
    *,
    family: str = "E",
    ordinal: int = 1,
    arm: str = "E-applicable",
    question: str = "What is the balance on account 4471-9920?",
    proposed: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    stem = stem_id(family, ordinal)
    gold = ARM_TO_GOLD[arm]
    if arm == "E-applicable":
        tools = [_tool("get_balance", ["account_id"]), _tool("convert_currency", ["amount"])]
        roles = {"get_balance": "APPLICABLE", "convert_currency": "DISTRACTOR"}
        gold_call = {
            "name": "get_balance",
            "required_arguments": ["account_id"],
            "argument_bindings": {"account_id": "4471-9920"},
            "argument_match_policy": "STRING_EXACT_AFTER_STRIP",
        }
    elif arm == "E-underspecified":
        tools = [_tool("get_balance", ["account_id", "as_of_date"]), _tool("convert_currency", ["amount"])]
        roles = {"get_balance": "UNDERSPECIFIED_VARIANT", "convert_currency": "DISTRACTOR"}
        gold_call = None
    elif arm == "E-absent":
        tools = [_tool("send_email", ["to"]), _tool("convert_currency", ["amount"])]
        roles = {"send_email": "DISTRACTOR", "convert_currency": "DISTRACTOR"}
        gold_call = None
    elif arm == "S-none":
        tools, roles, gold_call = [], {}, None
    else:
        count = 1 if arm == "S-one" else 3
        tools = [_tool(f"distractor_{i}", ["x"]) for i in range(count)]
        roles = {t["name"]: "DISTRACTOR" for t in tools}
        gold_call = None

    final = overrides.pop("gold_decision", gold)
    proposal_decision = proposed if proposed is not None else final
    record: dict[str, Any] = {
        "example_id": example_id(stem, arm),
        "question": question,
        "tools": tools,
        "tool_roles": roles,
        "gold_decision": final,
        "gold_call": gold_call,
        "contrast": {
            "controlled_variable": "tool_availability",
            "held_constant": ["question_intent", "domain", "entity_bindings"],
            "question_identical_to_siblings": True,
        },
        "stem_provenance": {
            "origin": "AUTHORED_ORIGINAL",
            "source_file": "data/stems/E/finance.yaml",
            "source_file_sha256": "0" * 64,
            "source_anchor": f"stems[{ordinal}]",
            "domain": "finance",
            "authored_by": "zeraphim",
        },
        "label_provenance": {
            "label_authority": "HUMAN",
            "status": "LABELED",
            "proposal": {
                "stage": "MODEL_PROPOSED",
                "proposer_model_id": "test/mock",
                "proposed_decision": proposal_decision,
                "raw_response_sha256": "1" * 64,
            },
            "adjudication": {
                "stage": "HUMAN_ADJUDICATED",
                "adjudicator_id": "zeraphim",
                "verdict": "ACCEPTED" if proposal_decision == final else "CHANGED",
                "final_decision": final,
                "rubric_version": "tcc-adjudication-v1",
                "rubric_clause": "E.2",
                "rationale": "" if proposal_decision == final else "Parameter is bound by the question.",
            },
            "second_pass": None,
            "agreement": {
                "model_matched_human": proposal_decision == final,
                "model_decision": proposal_decision,
                "human_decision": final,
            },
        },
        "metadata": {"eligibility": "evaluation_only", "split": "evaluation", "partition": "pilot"},
    }
    record.update(overrides)
    return finalize(record)


@pytest.fixture
def record_factory():
    return make_record
