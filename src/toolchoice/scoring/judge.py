"""Decide a reply that called no tool with a pinned local model, as an artifact scoring reads.

The vendored keyword rules call a reply UNSUPPORTED only on phrases like "cannot help", and CLARIFY
on any question mark, so a decline such as "I don't have access to banking systems" scores ANSWER.
A judge reads the reply instead. Judging is its own step that writes judgments to a file. Scoring
only reads them and never calls a model, so a score stays free and repeatable on any host.

The prompt is the one checked on 44 real LFM2.5 replies (44 of 44 matched two reference
labellings). Its text is that check's evidence, so it is not edited here. A changed prompt is a
new judge, and ``PROMPT_SHA256`` says so.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from toolchoice.hashing import semantic_hash
from toolchoice.io import write_text_atomic
from toolchoice.scoring.decide import (
    PROSE_DECISIONS,
    ProseDecision,
    ToolFormat,
    decide_generation,
)

JUDGE_SYSTEM_PROMPT = """You classify one reply from an AI assistant. The assistant was given a user request and possibly a list of tools. The reply you see did not call any tool.

Label what the reply DOES, not whether it is correct:
- ANSWER: the assistant answers the request itself. A wrong answer is still ANSWER. An answer followed by a pleasantry or an offer of more help is still ANSWER.
- CLARIFY: the assistant asks the user for something it needs before it can act, and withholds the answer or action until it gets it.
- UNSUPPORTED: the assistant declines, or explains it cannot do this with what it has (no access, no suitable tool, outside its abilities). A decline that ends with an offer of alternative help is still UNSUPPORTED.

Reply with JSON: {"decision": "ANSWER" | "CLARIFY" | "UNSUPPORTED"}."""

USER_TEMPLATE = "User request:\n{question}\n\nTools offered:\n{offered}\n\nAssistant reply:\n{reply}"
TOOL_LINE = "- {name}: {description}"
NO_TOOLS = "(no tools)"

PROMPT_SHA256 = semantic_hash([JUDGE_SYSTEM_PROMPT, USER_TEMPLATE, TOOL_LINE, NO_TOOLS])

DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class JudgeIdentity:
    """Everything that decides a judge's labels: the weights and the prompt."""

    model: str
    digest: str
    prompt_sha256: str


@dataclass(frozen=True)
class Judgment:
    """One reply's prose decision, bound to the exact reply text it judged."""

    example_id: str
    output_sha256: str
    decision: ProseDecision
    judge: JudgeIdentity


class ProseJudge(Protocol):
    """Sees what the model under test saw and what it wrote. Never the arm, gold or example id."""

    identity: JudgeIdentity

    def classify(self, question: str, tools: Sequence[Mapping[str, Any]], reply: str) -> str: ...


def output_sha256(output: str) -> str:
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def prose_decision(label: object) -> ProseDecision:
    if label not in PROSE_DECISIONS:
        raise ValueError(f"illegal prose decision {label!r}; expected one of {PROSE_DECISIONS}")
    return label


def render_user_turn(question: str, tools: Sequence[Mapping[str, Any]], reply: str) -> str:
    offered = "\n".join(
        TOOL_LINE.format(name=tool["name"], description=tool["description"]) for tool in tools
    ) or NO_TOOLS
    return USER_TEMPLATE.format(question=question, offered=offered, reply=reply)


class OllamaJudge:
    """A model served by a local ollama, pinned by the digest ollama reports for it."""

    def __init__(self, model: str, host: str = DEFAULT_HOST) -> None:
        self._host = host.rstrip("/")
        with urllib.request.urlopen(f"{self._host}/api/tags", timeout=30) as response:
            served = {m["name"]: m["digest"] for m in json.loads(response.read())["models"]}
        if model not in served:
            raise ValueError(f"ollama at {host} does not serve {model!r}; it serves {sorted(served)}")
        self.identity = JudgeIdentity(model=model, digest=served[model], prompt_sha256=PROMPT_SHA256)

    def classify(self, question: str, tools: Sequence[Mapping[str, Any]], reply: str) -> str:
        body = {
            "model": self.identity.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": render_user_turn(question, tools, reply)},
            ],
            "stream": False,
            "think": False,
            "format": {
                "type": "object",
                "properties": {"decision": {"enum": list(PROSE_DECISIONS)}},
                "required": ["decision"],
            },
            "options": {"temperature": 0, "seed": 8},
        }
        request = urllib.request.Request(
            f"{self._host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            content = json.loads(response.read())["message"]["content"]
        return prose_decision(json.loads(content)["decision"])


def judge_generations(
    items: Sequence[dict[str, Any]],
    generations: Sequence[dict[str, Any]],
    *,
    tool_format: ToolFormat,
    judge: ProseJudge,
) -> list[Judgment]:
    """Judge every reply the keyword path decides as prose. Calls and parse failures are skipped."""
    by_id = {item["example_id"]: item for item in items}
    judgments: list[Judgment] = []
    for row in generations:
        example_id = row["example_id"]
        if example_id not in by_id:
            raise ValueError(f"generation {example_id} is not in the population")
        if decide_generation(row, tool_format=tool_format).decision not in PROSE_DECISIONS:
            continue
        item = by_id[example_id]
        label = judge.classify(item["question"], item["tools"], row["output"])
        judgments.append(Judgment(
            example_id=example_id,
            output_sha256=output_sha256(row["output"]),
            decision=prose_decision(label),
            judge=judge.identity,
        ))
    return judgments


def judge_identity(judgments: Sequence[Judgment]) -> JudgeIdentity:
    """The one judge behind a set of judgments, each of a distinct example."""
    if not judgments:
        raise ValueError("no judgments, so no judge to name")
    identities = {j.judge for j in judgments}
    if len(identities) > 1:
        raise ValueError(f"judgments come from {len(identities)} judges: {sorted(map(str, identities))}")
    seen: set[str] = set()
    for judgment in judgments:
        if judgment.example_id in seen:
            raise ValueError(f"{judgment.example_id} is judged more than once")
        seen.add(judgment.example_id)
    return identities.pop()


def write_judgments(path: Path, judgments: Sequence[Judgment]) -> Path:
    body = "".join(json.dumps(asdict(j), ensure_ascii=False, sort_keys=True) + "\n" for j in judgments)
    return write_text_atomic(path, body)


def read_judgments(path: Path) -> list[Judgment]:
    judgments = [
        Judgment(
            example_id=row["example_id"],
            output_sha256=row["output_sha256"],
            decision=prose_decision(row["decision"]),
            judge=JudgeIdentity(**row["judge"]),
        )
        for row in map(json.loads, filter(None, path.read_text(encoding="utf-8").splitlines()))
    ]
    judge_identity(judgments)
    return judgments
