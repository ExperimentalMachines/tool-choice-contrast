"""The held-out pins must stay reproducible, and our normalize_prompt must stay upstream's.

The offline tests keep the recorded artifact honest. The network test is the one that matters:
it re-downloads the pinned upstream files and reproduces all three counts Study 002 published.
That reproduction is the whole licence for reimplementing normalize_prompt rather than vendoring
it, so if it ever stops reproducing, the reimplementation stops being safe.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from toolchoice.constants import (
    HELDOUT_FILES,
    HELDOUT_REPOSITORY,
    HELDOUT_REVISION,
    normalize_prompt,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
PINS = json.loads((ROOT / "reports/data/heldout-index-pins.json").read_text(encoding="utf-8"))


def test_constants_agree_with_the_recorded_artifact() -> None:
    assert PINS["repository"] == HELDOUT_REPOSITORY
    assert PINS["revision"] == HELDOUT_REVISION
    assert {k: v["sha256"] for k, v in PINS["files"].items()} == HELDOUT_FILES


def test_the_unreproducible_pin_is_recorded_not_silently_dropped() -> None:
    """Declining to use upstream's parquet hash is a decision, and it carries its reason."""
    assert PINS["not_pinned_and_why"], "a rejected pin with no recorded reason is an unexplained gap"
    for reason in PINS["not_pinned_and_why"].values():
        assert "pyarrow" in reason and "reproduc" in reason


def test_normalize_prompt_collapses_case_and_whitespace() -> None:
    assert normalize_prompt("  What   is\tthe BALANCE?\n") == "what is the balance?"
    assert normalize_prompt("A B") == normalize_prompt("a  b")


@pytest.mark.network
def test_pinned_heldout_files_reproduce_upstreams_published_counts() -> None:
    import urllib.error
    import urllib.request

    counts: dict[str, set[str]] = {}
    for path, expected_sha in sorted(HELDOUT_FILES.items()):
        url = f"https://huggingface.co/datasets/{HELDOUT_REPOSITORY}/resolve/{HELDOUT_REVISION}/{path}"
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            pytest.skip(f"BLOCKED_NETWORK: {url}: {exc}")

        assert hashlib.sha256(body).hexdigest() == expected_sha, f"{path} moved at {HELDOUT_REVISION}"
        rows = [json.loads(line) for line in body.decode("utf-8").splitlines() if line]
        assert len(rows) == PINS["files"][path]["rows"]
        counts[path] = {normalize_prompt(row["question"]) for row in rows}

    mcq = counts["test/when2call_test_mcq.jsonl"]
    judge = counts["test/when2call_test_llm_judge.jsonl"]
    assert len(mcq) == 2295
    assert len(judge) == 286
    # Doc 32: every llm-judge prompt is already an MCQ prompt. Summing the splits double-counts.
    assert judge <= mcq
