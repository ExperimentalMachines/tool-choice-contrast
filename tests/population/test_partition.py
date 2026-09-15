from __future__ import annotations

import json
from pathlib import Path

from toolchoice.ids import parse_example_id
from toolchoice.population.partition import PILOT, REPORTED, check_integrity, partition

ROOT = Path(__file__).resolve().parents[2]
RECORDS = [
    json.loads(line)
    for line in (ROOT / "data/frozen/tcc-v1/population.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]


def test_partition_unit_is_stem_not_example() -> None:
    """The load-bearing one. Splitting on items destroys the paired contrast silently."""
    payload = partition(RECORDS)
    for side in (PILOT, REPORTED):
        stems = set(payload["sides"][side]["stem_ids"])
        for example_id in payload["sides"][side]["example_ids"]:
            stem, _ = parse_example_id(example_id)
            assert stem in stems


def test_no_stem_straddles_the_two_sides() -> None:
    payload = partition(RECORDS)
    pilot = set(payload["sides"][PILOT]["stem_ids"])
    reported = set(payload["sides"][REPORTED]["stem_ids"])
    assert not pilot & reported
    assert pilot | reported == {r["stem_id"] for r in RECORDS}


def test_partition_is_deterministic_for_a_fixed_seed() -> None:
    first, second = partition(RECORDS), partition(list(reversed(RECORDS)))
    for side in (PILOT, REPORTED):
        assert first["sides"][side]["fingerprint"] == second["sides"][side]["fingerprint"]


def test_integrity_check_catches_a_straddling_stem() -> None:
    payload = partition(RECORDS)
    stolen = payload["sides"][REPORTED]["stem_ids"][0]
    payload["sides"][PILOT]["stem_ids"].append(stolen)
    assert check_integrity(payload)


def test_committed_partition_reproduces() -> None:
    committed = json.loads((ROOT / "reports/data/population-partition.json").read_text(encoding="utf-8"))
    rebuilt = partition(RECORDS, pilot_fraction=committed["pilot_fraction"])
    for side in (PILOT, REPORTED):
        assert committed["sides"][side]["fingerprint"] == rebuilt["sides"][side]["fingerprint"]


def test_pilot_is_declared_a_development_surface_not_a_measurement() -> None:
    payload = partition(RECORDS)
    assert "not a second measurement" in payload["roles"][PILOT]
