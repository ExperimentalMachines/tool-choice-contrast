"""`tcc-verify`: the gate suite, tri-state, with execution accounting.

A gate must prove both that its assertions passed and that they actually ran. A gate that
discovered nothing is a FAIL, not a quiet PASS -- that distinction is the entire reason the
upstream accounting module exists and it is reused rather than re-derived.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opengrad.verification.accounting import (
    BLOCKED_OPTIONAL_DEPENDENCY,
    CONDITIONALLY_REQUIRED,
    OPTIONAL,
    PASS,
    REQUIRED_NONEMPTY,
    ValidationResult,
    VerificationReport,
)

from toolchoice.constants import ARM_TO_GOLD, E_DISTRACTOR_COUNT, FAMILY_ARMS
from toolchoice.ids import parse_example_id

CONTRACT = 1


def gate_structural(items: Sequence[dict[str, Any]]) -> ValidationResult:
    from toolchoice.population.record import RecordError, validate

    errors = []
    for item in items:
        try:
            validate(item)
        except RecordError as exc:
            errors.append(f"{item.get('example_id', '?')}: {exc.reason_code}")
    return ValidationResult(
        name="G1_structural", policy=REQUIRED_NONEMPTY,
        discovered=len(items), checked=len(items),
        passed=len(items) - len(errors), failed=len(errors), errors=errors[:20],
    )


def gate_arm_completeness(items: Sequence[dict[str, Any]]) -> ValidationResult:
    stems: dict[str, set[str]] = {}
    for item in items:
        stem, arm = parse_example_id(item["example_id"])
        stems.setdefault(stem, set()).add(arm)
    errors = []
    for stem, arms in sorted(stems.items()):
        family = stem.split("/")[1]
        if arms != set(FAMILY_ARMS[family]):
            errors.append(f"{stem}: has {sorted(arms)}, expected {sorted(FAMILY_ARMS[family])}")
    return ValidationResult(
        name="G2_arm_completeness", policy=REQUIRED_NONEMPTY,
        discovered=len(stems), checked=len(stems),
        passed=len(stems) - len(errors), failed=len(errors), errors=errors[:20],
    )


def gate_stem_invariance(items: Sequence[dict[str, Any]]) -> ValidationResult:
    """The most important gate: without it the contrast is confounded and every number is void."""
    questions: dict[str, set[str]] = {}
    for item in items:
        stem, _ = parse_example_id(item["example_id"])
        questions.setdefault(stem, set()).add(item["question"])
    errors = [f"{stem}: {len(qs)} distinct questions across its arms"
              for stem, qs in sorted(questions.items()) if len(qs) != 1]
    return ValidationResult(
        name="G3_stem_invariance", policy=REQUIRED_NONEMPTY,
        discovered=len(questions), checked=len(questions),
        passed=len(questions) - len(errors), failed=len(errors), errors=errors[:20],
    )


def gate_tool_context(items: Sequence[dict[str, Any]]) -> ValidationResult:
    """Per-arm catalogue shape, including the E-arm equality that replaces a statistical check."""
    from toolchoice.constants import ARM_TOOL_COUNTS

    errors: list[str] = []
    e_sizes: dict[str, set[int]] = {}
    for item in items:
        arm = item["arm"]
        count = len(item["tools"])
        if arm in ARM_TOOL_COUNTS:
            low, high = ARM_TOOL_COUNTS[arm]
            if not low <= count <= high:
                errors.append(f"{item['example_id']}: {count} tools, expected {low}..{high}")
        else:
            stem, _ = parse_example_id(item["example_id"])
            e_sizes.setdefault(stem, set()).add(count)
            if count != E_DISTRACTOR_COUNT + 1:
                errors.append(f"{item['example_id']}: {count} tools, expected {E_DISTRACTOR_COUNT + 1}")
    for stem, sizes in sorted(e_sizes.items()):
        if len(sizes) != 1:
            errors.append(f"{stem}: E arms differ in catalogue size {sorted(sizes)}")
    return ValidationResult(
        name="G5_tool_context", policy=REQUIRED_NONEMPTY,
        discovered=len(items), checked=len(items),
        passed=len(items) - len(errors), failed=len(errors), errors=errors[:20],
    )


def gate_mode_balance(items: Sequence[dict[str, Any]], minimum: int) -> ValidationResult:
    counts = {decision: 0 for decision in set(ARM_TO_GOLD.values())}
    for item in items:
        counts[item["gold_decision"]] += 1
    errors = [f"{decision}: {n} items, need >= {minimum}"
              for decision, n in sorted(counts.items()) if n < minimum]
    return ValidationResult(
        name="G4_mode_balance", policy=REQUIRED_NONEMPTY,
        discovered=len(counts), checked=len(counts),
        passed=len(counts) - len(errors), failed=len(errors), errors=errors,
        detail=counts,
    )


def gate_adjudication(items: Sequence[dict[str, Any]]) -> ValidationResult:
    """A frozen release contains zero PENDING items. Unlabelled is a build state, not a release."""
    from toolchoice.population.record import PENDING

    pending = [i["example_id"] for i in items if i["label_provenance"]["status"] == PENDING]
    return ValidationResult(
        name="G7_adjudication", policy=REQUIRED_NONEMPTY,
        discovered=len(items), checked=len(items),
        passed=len(items) - len(pending), failed=len(pending),
        errors=[f"{e}: PENDING" for e in pending[:10]],
        detail={"pending": len(pending)},
    )


def gate_partition(payload: dict[str, Any] | None) -> ValidationResult:
    from toolchoice.population.partition import check_integrity

    if payload is None:
        return ValidationResult(
            name="G11_partition", policy=CONDITIONALLY_REQUIRED,
            discovered=0, blocked=1, blocked_status=BLOCKED_OPTIONAL_DEPENDENCY,
            blocked_reasons=["no partition artifact"], precondition="a partition has been drawn",
        )
    problems = check_integrity(payload)
    stems = payload["sides"]["pilot"]["stems"] + payload["sides"]["reported"]["stems"]
    return ValidationResult(
        name="G11_partition", policy=REQUIRED_NONEMPTY,
        discovered=stems, checked=stems,
        passed=stems - len(problems), failed=len(problems), errors=problems[:10],
    )


def run_gates(
    items: Sequence[dict[str, Any]], *, minimum_per_mode: int,
    partition_payload: dict[str, Any] | None = None,
) -> VerificationReport:
    report = VerificationReport(contract=CONTRACT)
    for result in (
        gate_structural(items),
        gate_arm_completeness(items),
        gate_stem_invariance(items),
        gate_mode_balance(items, minimum_per_mode),
        gate_tool_context(items),
        gate_adjudication(items),
        gate_partition(partition_payload),
    ):
        report.add(result)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcc-verify", description="Run the corpus gates.")
    parser.add_argument("--population", type=Path, default=Path("data/frozen/tcc-v1/population.jsonl"))
    parser.add_argument("--minimum-per-mode", type=int, default=150)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.population.exists():
        # Nothing to verify is not a pass. It is a blocked gate with a named reason.
        blocked = ValidationResult(
            name="G0_population_present", policy=REQUIRED_NONEMPTY,
            discovered=0, blocked=1, blocked_status=BLOCKED_OPTIONAL_DEPENDENCY,
            blocked_reasons=[f"no population at {args.population}"],
        )
        report = VerificationReport(contract=CONTRACT)
        report.add(blocked)
        print(json.dumps({"overall": report.overall}, indent=2) if args.json else report.render())
        return 2

    items = [json.loads(line) for line in args.population.read_text(encoding="utf-8").splitlines() if line]
    partition_path = Path("reports/data/population-partition.json")
    partition_payload = (
        json.loads(partition_path.read_text(encoding="utf-8")) if partition_path.exists() else None
    )
    report = run_gates(items, minimum_per_mode=args.minimum_per_mode,
                       partition_payload=partition_payload)
    if args.json:
        print(json.dumps({
            "schema_version": 1, "artifact_kind": "GATE_REPORT",
            "overall": report.overall, "contract": CONTRACT,
            "results": [r.counts() for r in report.results],
        }, indent=2, sort_keys=True))
    else:
        print(report.render())
    return 0 if report.overall == PASS else 1


__all__ = ["OPTIONAL", "main", "run_gates"]

if __name__ == "__main__":
    # Without this, `python -m toolchoice.verification.gates` imports the module, runs nothing and
    # exits 0 -- indistinguishable from a passing verification.
    raise SystemExit(main())
