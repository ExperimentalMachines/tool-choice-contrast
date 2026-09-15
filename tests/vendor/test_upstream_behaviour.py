"""Behavioural backstops on the vendored functions.

Hash parity says the bytes are the same. These say what the bytes *do*, so that
if a vendored file ever has to be forked, the diff in behaviour is visible.
"""

from __future__ import annotations

import pytest
from opengrad.evaluation.capability import detect_refusal
from opengrad.evaluation.routing import routing_metrics


def test_routing_metrics_rejects_the_word_direct() -> None:
    """The reason gold_decision stores ANSWER and never DIRECT."""
    with pytest.raises(ValueError):
        routing_metrics(["DIRECT"], ["DIRECT"])


def test_no_call_accuracy_is_keyed_on_answer() -> None:
    metrics = routing_metrics(["ANSWER", "ANSWER", "CALL"], ["ANSWER", "CALL", "CALL"])
    assert metrics["no_call_accuracy"] == pytest.approx(0.5)
    assert metrics["over_call_rate"] == pytest.approx(0.5)


def test_empty_class_returns_zero_not_null() -> None:
    """Upstream emits 0.0 for an absent class. This is the defect the wrapper must mask:
    a corpus with no ANSWER items scores no_call_accuracy 0.0, indistinguishable from a
    model that answered nothing correctly."""
    metrics = routing_metrics(["CALL"], ["CALL"])
    assert metrics["no_call_accuracy"] == 0.0


def test_detect_refusal_ignores_midresponse_hedges() -> None:
    assert detect_refusal("I can't be certain, but the answer is 16").is_refusal is False
    verdict = detect_refusal("I'm sorry, I'm unable to help with that.")
    assert verdict.is_refusal is True
    assert verdict.method == "HEURISTIC_REGEX_v1"


def test_detect_refusal_only_looks_at_the_first_400_chars() -> None:
    padded = "The answer is 42. " * 40 + "I am unable to do that."
    assert len(padded) > 400
    assert detect_refusal(padded).is_refusal is False
