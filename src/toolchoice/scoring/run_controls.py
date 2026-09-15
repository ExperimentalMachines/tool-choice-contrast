"""Execute the controls against a synthetic population and check the pre-registered windows."""

from __future__ import annotations

from typing import Any

from opengrad.verification.accounting import (
    PASS,
    REQUIRED_NONEMPTY,
    ValidationResult,
    VerificationReport,
)

from toolchoice.harness.backends.mock import MockBackend
from toolchoice.harness.runner import run_population
from toolchoice.scoring.controls import CONTROLS, Control
from toolchoice.scoring.score import score

CONTRACT = 1
ARTIFACT_KIND = "CONTROL_SUITE"


def run_control(control: Control, items: list[dict[str, Any]]) -> dict[str, Any]:
    backend = MockBackend(control.script(list(items)))
    generations = run_population(backend, items)
    result = score(items, generations)
    checks = {name: bool(check(result)) for name, check in control.expectations.items()}
    return {
        "control_id": control.control_id,
        "description": control.description,
        "proves": control.proves,
        "backend": "MOCK",
        "checks": checks,
        "passed": all(checks.values()),
        "scores": result,
    }


def run_all(items: list[dict[str, Any]]) -> dict[str, Any]:
    report = VerificationReport(contract=CONTRACT)
    outcomes = []
    for control in CONTROLS:
        outcome = run_control(control, items)
        outcomes.append(outcome)
        passed = sum(1 for ok in outcome["checks"].values() if ok)
        failed = len(outcome["checks"]) - passed
        report.add(ValidationResult(
            name=f"control_{control.control_id}",
            policy=REQUIRED_NONEMPTY,
            discovered=len(outcome["checks"]),
            checked=len(outcome["checks"]),
            passed=passed,
            failed=failed,
            errors=[f"{control.control_id}: {n} did not hold"
                    for n, ok in outcome["checks"].items() if not ok],
        ))
    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "contract": CONTRACT,
        "population_items": len(items),
        "overall": report.overall,
        "all_passed": report.overall == PASS,
        "controls": outcomes,
        "render": report.render(),
    }
