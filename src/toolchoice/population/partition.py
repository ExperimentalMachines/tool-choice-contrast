"""Split the corpus into a pilot and a reported side.

The unit is the STEM, never the item. Splitting on items would put E-applicable in pilot and
E-absent in reported for the same stem, destroying the paired contrast -- the entire reason the
corpus exists -- with no assertion firing anywhere and per-arm accuracy still looking healthy.

Ordering is a keyed hash, not an RNG: reproducible regardless of input order or language version.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from toolchoice.constants import SPLIT_SEED
from toolchoice.hashing import corpus_fingerprint, ranking_key
from toolchoice.ids import parse_example_id

ARTIFACT_KIND = "EVAL_PARTITION"
ALGORITHM = "stratified_by_family_and_gold_multiset__sha256(seed||stem_id)_ascending"
PILOT, REPORTED = "pilot", "reported"


def _strata(records: Sequence[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    """Each stem's stratum: its family plus the multiset of gold decisions across its arms."""
    golds: dict[str, list[str]] = {}
    for record in records:
        golds.setdefault(record["stem_id"], []).append(record["gold_decision"])
    return {stem: tuple(sorted(values)) for stem, values in golds.items()}


def partition(records: Sequence[dict[str, Any]], *, pilot_fraction: float = 0.25) -> dict[str, Any]:
    strata = _strata(records)
    buckets: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for stem, gold_multiset in strata.items():
        family = stem.split("/")[1]
        buckets.setdefault((family, gold_multiset), []).append(stem)

    pilot: list[str] = []
    reported: list[str] = []
    per_class: dict[str, dict[str, int]] = {}
    for (family, gold_multiset), stems in sorted(buckets.items()):
        ordered = sorted(stems, key=lambda s: ranking_key(SPLIT_SEED, s))
        take = round(len(ordered) * pilot_fraction)
        # Never empty a stratum on either side unless it genuinely has one member.
        take = min(max(take, 1 if len(ordered) > 1 else 0), len(ordered) - 1 if len(ordered) > 1 else len(ordered))
        pilot.extend(ordered[:take])
        reported.extend(ordered[take:])
        per_class[f"{family}:{'+'.join(gold_multiset)}"] = {
            "stems": len(ordered), "pilot": take, "reported": len(ordered) - take,
        }

    pilot, reported = sorted(pilot), sorted(reported)
    by_stem: dict[str, list[str]] = {record["stem_id"]: [] for record in records}
    for record in records:
        by_stem[record["stem_id"]].append(record["example_id"])

    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "algorithm": ALGORITHM,
        "seed": SPLIT_SEED,
        "partition_unit": "stem_id",
        "sibling_integrity": (
            "every arm of a stem lands on the same side; asserted at build and re-asserted at verify"
        ),
        "pilot_fraction": pilot_fraction,
        "roles": {
            PILOT: (
                "development surface. The corpus author saw these items while authoring, so a pilot "
                "score is not a second measurement."
            ),
            REPORTED: "evaluated once, on the reported run, over the pinned checkpoints.",
        },
        "per_class": dict(sorted(per_class.items())),
        "sides": {
            PILOT: {
                "stems": len(pilot), "items": sum(len(by_stem[s]) for s in pilot),
                "fingerprint": corpus_fingerprint(pilot), "stem_ids": pilot,
                "example_ids": sorted(e for s in pilot for e in by_stem[s]),
            },
            REPORTED: {
                "stems": len(reported), "items": sum(len(by_stem[s]) for s in reported),
                "fingerprint": corpus_fingerprint(reported), "stem_ids": reported,
                "example_ids": sorted(e for s in reported for e in by_stem[s]),
            },
        },
    }


def check_integrity(payload: dict[str, Any]) -> list[str]:
    """Every arm of a stem on one side; sides disjoint and exhaustive."""
    problems: list[str] = []
    pilot = set(payload["sides"][PILOT]["stem_ids"])
    reported = set(payload["sides"][REPORTED]["stem_ids"])
    if pilot & reported:
        problems.append(f"{len(pilot & reported)} stems appear on both sides")
    for side, stems in ((PILOT, pilot), (REPORTED, reported)):
        for example_id in payload["sides"][side]["example_ids"]:
            stem, _ = parse_example_id(example_id)
            if stem not in stems:
                problems.append(f"{example_id} is on the {side} side but its stem is not")
                break
    return problems
