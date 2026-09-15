"""Gate behaviour, including the two that must fail on the current seed corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from opengrad.verification.accounting import PASS

from toolchoice.verification.gates import main as gates_main
from toolchoice.verification.gates import run_gates

ROOT = Path(__file__).resolve().parents[2]
RECORDS = [
    json.loads(line)
    for line in (ROOT / "data/frozen/tcc-v1/population.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
PARTITION = json.loads((ROOT / "reports/data/population-partition.json").read_text(encoding="utf-8"))


def _by_name(minimum: int = 150) -> dict[str, object]:
    report = run_gates(RECORDS, minimum_per_mode=minimum, partition_payload=PARTITION)
    return {result.name: result for result in report.results}


@pytest.mark.parametrize(
    "gate", ["G1_structural", "G2_arm_completeness", "G3_stem_invariance",
             "G5_tool_context", "G11_partition"],
)
def test_structural_gates_pass_on_the_seed_corpus(gate: str) -> None:
    assert _by_name()[gate].status == PASS


def test_mode_balance_fails_because_the_corpus_is_a_seed() -> None:
    """Expected to fail today. It goes green when authoring finishes, not before."""
    result = _by_name()["G4_mode_balance"]
    assert result.status != PASS
    assert result.detail["ANSWER"] < 150


def test_adjudication_fails_because_nothing_is_labelled() -> None:
    result = _by_name()["G7_adjudication"]
    assert result.status != PASS
    assert result.detail["pending"] == len(RECORDS)


def test_a_gate_that_discovered_nothing_is_a_failure_not_a_pass() -> None:
    """The non-vacuity rule. A gate that examined nothing has not verified anything."""
    report = run_gates([], minimum_per_mode=150, partition_payload=None)
    structural = next(r for r in report.results if r.name == "G1_structural")
    assert structural.status != PASS
    assert report.overall != PASS


def test_stem_invariance_catches_an_edited_question() -> None:
    """G3: without it the contrast is confounded and every number is void."""
    tampered = [dict(r) for r in RECORDS]
    tampered[0] = {**tampered[0], "question": tampered[0]["question"] + " (edited)"}
    result = next(r for r in run_gates(tampered, minimum_per_mode=1,
                                       partition_payload=PARTITION).results
                  if r.name == "G3_stem_invariance")
    assert result.status != PASS
    assert result.failed == 1


def test_partition_gate_blocks_rather_than_passes_when_absent() -> None:
    result = next(r for r in run_gates(RECORDS, minimum_per_mode=1).results
                  if r.name == "G11_partition")
    assert result.blocked == 1
    assert result.status != PASS


def test_missing_population_exits_two_not_zero() -> None:
    assert gates_main(["--population", str(ROOT / "nope.jsonl")]) == 2
