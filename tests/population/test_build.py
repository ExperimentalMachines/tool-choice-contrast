"""The population must rebuild to the same bytes, and say so distinctly when it does not."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from toolchoice.hashing import content_hash, corpus_fingerprint, record_hash
from toolchoice.population.build import MANIFEST_PATH, POPULATION_PATH, assemble, build, verify
from toolchoice.population.record import PENDING, validate
from toolchoice.stems.loader import load_pool, load_stems

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def records() -> list[dict]:
    stems = load_stems(ROOT / "data/stems", ROOT)
    pool = load_pool(ROOT / "data/tool_pool")
    return assemble(stems, pool)


def test_committed_population_verifies(tmp_path: Path) -> None:
    """The CI regression gate: a rebuild reproduces both fingerprints from a clean checkout."""
    assert verify(ROOT, stems_dir=Path("data/stems"), pool_dir=Path("data/tool_pool"),
                  out=POPULATION_PATH, manifest=MANIFEST_PATH) == 0


def test_build_is_byte_identical_across_two_runs(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for out in (first, second):
        build(ROOT, stems_dir=Path("data/stems"), pool_dir=Path("data/tool_pool"),
              out=out.relative_to(ROOT) if out.is_relative_to(ROOT) else out,
              manifest=tmp_path / f"{out.stem}-manifest.json")
    assert first.read_bytes() == second.read_bytes()


def test_rows_are_sorted_by_example_id(records: list[dict]) -> None:
    """File bytes must not depend on stem iteration order."""
    ids = [r["example_id"] for r in records]
    assert ids == sorted(ids)


def test_every_record_validates(records: list[dict]) -> None:
    for record in records:
        validate(record)


def test_unlabelled_population_is_a_build_state_not_a_release_state(records: list[dict]) -> None:
    assert all(r["label_provenance"]["status"] == PENDING for r in records)
    manifest = json.loads((ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest["frozen"] is False


def test_pending_record_may_not_carry_labels(records: list[dict]) -> None:
    from toolchoice.population.record import RecordError

    record = copy.deepcopy(records[0])
    record["label_provenance"]["proposal"] = {"proposed_decision": "CALL"}
    with pytest.raises(RecordError) as exc:
        validate(record)
    assert exc.value.reason_code == "REC_PENDING_WITH_LABELS"


def test_content_stable_record_drift_is_distinguishable(records: list[dict]) -> None:
    """A reworded rationale must not invalidate a score recorded months earlier."""
    before_content = corpus_fingerprint([r["content_hash"] for r in records])
    before_record = corpus_fingerprint([r["record_hash"] for r in records])

    edited = copy.deepcopy(records)
    edited[0]["stem_provenance"]["capability"] = "reworded, same meaning"
    edited[0]["content_hash"] = content_hash(edited[0])
    edited[0].pop("record_hash")
    edited[0]["record_hash"] = record_hash(edited[0])

    assert corpus_fingerprint([r["content_hash"] for r in edited]) == before_content
    assert corpus_fingerprint([r["record_hash"] for r in edited]) != before_record


def test_verify_detects_an_edited_question(tmp_path: Path) -> None:
    population = tmp_path / "population.jsonl"
    manifest = tmp_path / "manifest.json"
    build(ROOT, stems_dir=Path("data/stems"), pool_dir=Path("data/tool_pool"),
          out=population, manifest=manifest)
    rows = [json.loads(line) for line in population.read_text(encoding="utf-8").splitlines() if line]
    rows[0]["question"] = rows[0]["question"] + " (tampered)"
    population.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    assert verify(ROOT, stems_dir=Path("data/stems"), pool_dir=Path("data/tool_pool"),
                  out=population, manifest=manifest) == 1
