"""The control suite. If these stop holding, no real model result may be reported."""

from __future__ import annotations

import pytest

from toolchoice.scoring.controls import CONTROLS, CONTROLS_BY_ID
from toolchoice.scoring.run_controls import run_all, run_control
from toolchoice.stems.synthetic import population


@pytest.fixture(scope="module")
def items() -> list[dict]:
    return population()


@pytest.fixture(scope="module")
def suite(items: list[dict]) -> dict:
    return run_all(items)


def test_every_control_passes_its_preregistered_window(suite: dict) -> None:
    failures = {
        c["control_id"]: [n for n, ok in c["checks"].items() if not ok]
        for c in suite["controls"] if not c["passed"]
    }
    assert not failures, failures
    assert suite["all_passed"] is True


def test_c4_detects_the_regression_this_corpus_exists_for(items: list[dict]) -> None:
    """A model that answers bare questions and refuses once tools appear."""
    result = run_control(CONTROLS_BY_ID["C4"], items)["scores"]["contrast"]["S-five"]
    assert result["status"] == "MEASURED"
    assert result["daa_baseline"]["point"] == pytest.approx(1.0)
    assert result["daa_arm"]["point"] == pytest.approx(0.0)
    assert result["tps_flip"]["point"] == pytest.approx(1.0)
    assert result["mcnemar"]["p_value"] < 1e-6
    assert result["flip_destinations"]["REFUSAL"] > 0
    assert result["flip_destinations"]["CALL"] == 0


def test_c3_reports_undefined_rather_than_a_misleading_zero(items: list[dict]) -> None:
    """The model that motivated the corpus is the one the headline metric cannot measure."""
    result = run_control(CONTROLS_BY_ID["C3"], items)["scores"]["contrast"]["S-five"]
    assert result["status"] == "UNDEFINED"
    assert result["tps_flip"]["point"] is None
    assert result["conditioning_stems"] == 0
    # A healthy model also scores no flips. Only the level distinguishes them.
    assert result["daa_baseline"]["point"] == pytest.approx(0.0)


def test_spurious_calls_and_refusals_are_distinguishable(items: list[dict]) -> None:
    """C2 and C4 produce the same TPS_flip. Only the decomposition separates them."""
    call_model = run_control(CONTROLS_BY_ID["C2"], items)["scores"]["contrast"]["S-five"]
    refuse_model = run_control(CONTROLS_BY_ID["C4"], items)["scores"]["contrast"]["S-five"]
    assert call_model["tps_flip"]["point"] == refuse_model["tps_flip"]["point"]
    assert call_model["flip_destinations"]["CALL"] > 0
    assert call_model["flip_destinations"]["REFUSAL"] == 0
    assert refuse_model["flip_destinations"]["REFUSAL"] > 0
    assert refuse_model["flip_destinations"]["CALL"] == 0


def test_control_paths_resolve_against_the_current_scorer(items: list[dict]) -> None:
    """A renamed metric must break the controls loudly, not silently pass."""
    for control in CONTROLS:
        run_control(control, items)  # raises KeyError naming the path if the shape moved
