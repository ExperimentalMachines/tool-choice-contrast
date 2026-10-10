"""A pinned local model decides replies that call no tool, and its judgments are an artifact.

The declines are verbatim LFM2.5 replies. Generation rows hold a reply without its final
``<|im_end|>``, so R3 is stored here as the row holds it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.formatting.test_lfm import R1, R3
from tests.scoring.test_tool_format import _decision
from toolchoice.ids import parse_example_id
from toolchoice.scoring.judge import (
    PROMPT_SHA256,
    JudgeIdentity,
    Judgment,
    OllamaJudge,
    judge_generations,
    output_sha256,
    read_judgments,
    write_judgments,
)
from toolchoice.scoring.score import score
from toolchoice.stems.synthetic import population

#: Keyword rules call this ANSWER: no refusal phrase, no question mark.
DECLINE = (
    "I don't have access to banking or financial systems to check account balances. You'll need "
    "to contact your bank directly or use their online portal to view the cleared balance for "
    "account 4471-9920."
)
#: Keyword rules call this CLARIFY, for its closing offer.
DECLINE_WITH_OFFER = R3.removesuffix("<|im_end|>")

IDENTITY = JudgeIdentity(model="fake", digest="d" * 64, prompt_sha256=PROMPT_SHA256)

ITEMS = population(s_count=1, e_count=1)
REPLIES = {
    "tcc-v1/S/0001/none": "12",
    "tcc-v1/S/0001/one": "12",
    "tcc-v1/S/0001/five": "",
    "tcc-v1/E/0001/applicable": R1,
    "tcc-v1/E/0001/underspecified": DECLINE_WITH_OFFER,
    "tcc-v1/E/0001/absent": DECLINE,
}
GENERATIONS = [
    {"example_id": item["example_id"], "output": REPLIES[item["example_id"]], "finish_reason": "stop"}
    for item in ITEMS
]
LABELS = {"12": "ANSWER", DECLINE_WITH_OFFER: "UNSUPPORTED", DECLINE: "UNSUPPORTED"}


class FakeJudge:
    def __init__(self, labels: Mapping[str, str], identity: JudgeIdentity = IDENTITY) -> None:
        self.identity = identity
        self.labels = labels
        self.seen: list[tuple[str, Sequence[Mapping[str, Any]], str]] = []

    def classify(self, question: str, tools: Sequence[Mapping[str, Any]], reply: str) -> str:
        self.seen.append((question, tools, reply))
        return self.labels[reply]


def _judge(labels: Mapping[str, str] = LABELS) -> list[Judgment]:
    return judge_generations(ITEMS, GENERATIONS, tool_format="lfm", judge=FakeJudge(labels))


def test_only_replies_without_a_call_reach_the_judge_with_question_tools_and_reply() -> None:
    judge = FakeJudge(LABELS)
    judgments = judge_generations(ITEMS, GENERATIONS, tool_format="lfm", judge=judge)

    judged = ["tcc-v1/S/0001/none", "tcc-v1/S/0001/one", "tcc-v1/E/0001/underspecified",
              "tcc-v1/E/0001/absent"]
    assert [(j.example_id, j.decision) for j in judgments] == [
        ("tcc-v1/S/0001/none", "ANSWER"),
        ("tcc-v1/S/0001/one", "ANSWER"),
        ("tcc-v1/E/0001/underspecified", "UNSUPPORTED"),
        ("tcc-v1/E/0001/absent", "UNSUPPORTED"),
    ]
    by_id = {item["example_id"]: item for item in ITEMS}
    assert judge.seen == [
        (by_id[i]["question"], by_id[i]["tools"], REPLIES[i]) for i in judged
    ]
    sent = json.dumps(judge.seen)
    for item in ITEMS:
        _, arm = parse_example_id(item["example_id"])
        for secret in (item["example_id"], arm, item["gold_decision"]):
            assert secret not in sent


def test_judgments_survive_a_round_trip_through_jsonl(tmp_path: Path) -> None:
    judgments = _judge()
    path = write_judgments(tmp_path / "judgments.jsonl", judgments)
    assert read_judgments(path) == judgments


def test_a_file_mixing_judges_is_rejected(tmp_path: Path) -> None:
    first, second, *_ = _judge()
    other = Judgment(second.example_id, second.output_sha256, second.decision,
                     JudgeIdentity("other", "e" * 64, PROMPT_SHA256))
    path = write_judgments(tmp_path / "judgments.jsonl", [first, other])
    with pytest.raises(ValueError, match="2 judges"):
        read_judgments(path)


def test_a_file_judging_one_reply_twice_is_rejected(tmp_path: Path) -> None:
    first, *_ = _judge()
    path = write_judgments(tmp_path / "judgments.jsonl", [first, first])
    with pytest.raises(ValueError, match="tcc-v1/S/0001/none is judged more than once"):
        read_judgments(path)


def test_a_file_with_an_illegal_decision_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    path.write_text(json.dumps({
        "example_id": "tcc-v1/E/0001/applicable", "output_sha256": "0" * 64, "decision": "CALL",
        "judge": {"model": "fake", "digest": "d" * 64, "prompt_sha256": PROMPT_SHA256},
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="illegal prose decision 'CALL'"):
        read_judgments(path)


def test_a_judge_returning_an_illegal_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="illegal prose decision 'REFUSE'"):
        _judge({**LABELS, DECLINE: "REFUSE"})


DECIDED_ARMS = ("S-none", "S-one", "E-applicable", "E-underspecified", "E-absent")


def test_judgments_replace_keyword_decisions_only_where_no_tool_was_called() -> None:
    keyword = score(ITEMS, GENERATIONS, tool_format="lfm")
    judged = score(ITEMS, GENERATIONS, tool_format="lfm", judgments=_judge())

    assert keyword["prose_classifier"] == {"kind": "keyword"}
    assert [_decision(keyword, arm) for arm in DECIDED_ARMS] == [
        "ANSWER", "ANSWER", "CALL", "CLARIFY", "ANSWER",
    ]
    assert judged["prose_classifier"] == {
        "kind": "judge", "model": "fake", "digest": "d" * 64, "prompt_sha256": PROMPT_SHA256,
    }
    assert [_decision(judged, arm) for arm in DECIDED_ARMS] == [
        "ANSWER", "ANSWER", "CALL", "UNSUPPORTED", "UNSUPPORTED",
    ]
    assert judged["coverage"]["S-five"]["parse_failures"] == 1


def test_a_judgment_of_a_different_reply_is_stale() -> None:
    judgments = [
        replace(j, output_sha256=output_sha256("12")) if j.example_id == "tcc-v1/E/0001/absent" else j
        for j in _judge()
    ]
    with pytest.raises(ValueError, match="tcc-v1/E/0001/absent: the judgment is stale"):
        score(ITEMS, GENERATIONS, tool_format="lfm", judgments=judgments)


def test_a_reply_without_a_call_must_have_a_judgment() -> None:
    judgments = [j for j in _judge() if j.example_id != "tcc-v1/E/0001/absent"]
    with pytest.raises(ValueError, match="tcc-v1/E/0001/absent decides without a tool call"):
        score(ITEMS, GENERATIONS, tool_format="lfm", judgments=judgments)


@pytest.mark.parametrize("example_id, output", [
    ("tcc-v1/E/0001/applicable", R1),
    ("tcc-v1/S/0001/five", ""),
    ("tcc-v1/E/0002/absent", DECLINE),
])
def test_a_judgment_of_a_call_a_parse_failure_or_an_unknown_item_is_rejected(
    example_id: str, output: str
) -> None:
    extra = Judgment(example_id, output_sha256(output), "UNSUPPORTED", IDENTITY)
    with pytest.raises(ValueError, match=example_id):
        score(ITEMS, GENERATIONS, tool_format="lfm", judgments=[*_judge(), extra])


def test_score_rejects_judgments_from_two_judges() -> None:
    first, *rest = _judge()
    other = replace(first, judge=JudgeIdentity("other", "e" * 64, PROMPT_SHA256))
    with pytest.raises(ValueError, match="2 judges"):
        score(ITEMS, GENERATIONS, tool_format="lfm", judgments=[other, *rest])


@pytest.mark.model
def test_ollama_judge_reads_a_real_decline_as_unsupported() -> None:
    judge = OllamaJudge("qwen3.5:9b")
    assert judge.identity == JudgeIdentity(
        "qwen3.5:9b", "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
        PROMPT_SHA256,
    )
    tools = [
        {"name": "send_message", "description": "Send a direct message to one recipient."},
        {"name": "track_parcel", "description": "Return the current transit status of one parcel."},
        {"name": "set_thermostat", "description": "Set the target temperature for one thermostat."},
    ]
    question = "What is the current cleared balance on account 4471-9920?"
    assert judge.classify(question, tools, DECLINE_WITH_OFFER) == "UNSUPPORTED"
