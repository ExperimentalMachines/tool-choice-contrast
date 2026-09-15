"""The vendor drop must stay byte-identical to its pinned upstream.

Two offline checks catch a local edit. One network check catches a stale pin.
The network check reports BLOCKED rather than PASS when it cannot reach
GitHub -- "we could not check" and "we checked and it is fine" are different
claims, and a gate that conflates them is worse than no gate.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "opengrad"
PROVENANCE = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))


def test_provenance_declares_the_pin() -> None:
    assert len(PROVENANCE["upstream_commit"]) == 40
    assert PROVENANCE["modified"] is False


@pytest.mark.parametrize("rel", sorted(PROVENANCE["files"]))
def test_vendored_file_hash_matches_provenance(rel: str) -> None:
    path = VENDOR / rel
    assert path.exists(), f"{rel} is recorded in PROVENANCE.json but absent from the tree"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == PROVENANCE["files"][rel]["sha256"], f"{rel} was edited locally"


def test_replaced_files_are_declared_not_silent() -> None:
    """The one deviation from byte-identity must be recorded with a reason."""
    replaced = PROVENANCE["replaced_files"]
    assert replaced, "a replaced file with no record is an undeclared fork"
    for rel, meta in replaced.items():
        path = VENDOR / rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
        assert meta["reason"].strip()
        assert rel not in PROVENANCE["files"], f"{rel} cannot be both identical and replaced"


def test_vendored_tree_imports_as_opengrad() -> None:
    from opengrad.data.behavior import DECISIONS
    from opengrad.evaluation.routing import routing_metrics
    from opengrad.formatting.parser import parse_qwen_native_output
    from opengrad.verification.accounting import ValidationResult

    assert DECISIONS == frozenset({"CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"})
    assert callable(routing_metrics)
    assert callable(parse_qwen_native_output)
    assert ValidationResult is not None


@pytest.mark.network
def test_provenance_matches_upstream_at_pinned_commit() -> None:
    import urllib.error
    import urllib.request

    commit = PROVENANCE["upstream_commit"]
    base = f"https://raw.githubusercontent.com/arjhinety/OpenGrad/{commit}"
    for rel, meta in sorted(PROVENANCE["files"].items()):
        url = f"{base}/{meta['upstream_path']}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            pytest.skip(f"BLOCKED_NETWORK: {url}: {exc}")
        assert hashlib.sha256(body).hexdigest() == meta["sha256"], (
            f"{rel} differs from upstream at {commit[:12]}"
        )
