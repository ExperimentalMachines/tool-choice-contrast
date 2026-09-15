"""Build the frozen population, and prove a rebuild reproduced it.

`build()` writes; `--verify` recomputes everything in memory, writes nothing, and diffs. The two
fingerprints are reported separately: a moved record fingerprint with a stable content fingerprint
means prose or provenance changed and previously recorded scores remain comparable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from toolchoice.constants import CORPUS_ID, DISTRACTOR_SEED, UPSTREAM_PIN
from toolchoice.hashing import corpus_fingerprint, file_sha256
from toolchoice.io import write_text_atomic
from toolchoice.population.record import PENDING, finalize, validate
from toolchoice.stems.expand import expand
from toolchoice.stems.loader import load_pool, load_stems
from toolchoice.stems.model import PoolTool, Stem

POPULATION_PATH = Path("data/frozen") / CORPUS_ID / "population.jsonl"
MANIFEST_PATH = Path("reports/data/population-manifest.json")
ARTIFACT_KIND = "POPULATION_MANIFEST"


def _pending_provenance() -> dict[str, Any]:
    return {
        "label_authority": "HUMAN",
        "status": PENDING,
        "proposal": None,
        "adjudication": None,
        "second_pass": None,
        "agreement": None,
    }


def assemble(stems: Sequence[Stem], pool: Sequence[PoolTool]) -> list[dict[str, Any]]:
    """Stems to finalized records, in a deterministic order."""
    records: list[dict[str, Any]] = []
    for stem in sorted(stems, key=lambda s: (s.family, s.stem_ordinal)):
        for partial in expand(stem, list(pool)):
            partial.setdefault("label_provenance", _pending_provenance())
            partial.setdefault("metadata", {"eligibility": "evaluation_only", "split": "evaluation"})
            records.append(finalize(partial))
    # Sorted by id so the file bytes cannot depend on stem iteration order.
    records.sort(key=lambda r: r["example_id"])
    for record in records:
        validate(record)
    return records


def build_manifest(records: Sequence[dict[str, Any]], stems: Sequence[Stem], *, path: Path) -> dict[str, Any]:
    by_arm: dict[str, int] = {}
    by_gold: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for record in records:
        by_arm[record["arm"]] = by_arm.get(record["arm"], 0) + 1
        by_gold[record["gold_decision"]] = by_gold.get(record["gold_decision"], 0) + 1
        status = record["label_provenance"]["status"]
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "corpus_id": CORPUS_ID,
        "frozen": statuses.get(PENDING, 0) == 0,
        "record_count": len(records),
        "stem_count": len({r["stem_id"] for r in records}),
        "content_fingerprint": corpus_fingerprint([r["content_hash"] for r in records]),
        "record_fingerprint": corpus_fingerprint([r["record_hash"] for r in records]),
        "population_file": str(path).replace("\\", "/"),
        "population_file_sha256": file_sha256(path) if path.exists() else None,
        "counts": {
            "by_arm": dict(sorted(by_arm.items())),
            "by_gold_decision": dict(sorted(by_gold.items())),
            "by_label_status": dict(sorted(statuses.items())),
        },
        "determinism": {
            "distractor_seed": DISTRACTOR_SEED,
            "distractor_algorithm": "sha256(seed||stem_id||arm||pool_id)_ascending",
            "row_order": "sorted_by_example_id",
        },
        "stem_source_files": {
            source: {"sha256": sha, "stems": sum(1 for s in stems if s.source_file == source)}
            for source, sha in sorted(_source_hashes(stems).items())
        },
        "upstream_pin": UPSTREAM_PIN,
    }


def _source_hashes(stems: Sequence[Stem]) -> dict[str, str]:
    out: dict[str, str] = {}
    for stem in stems:
        if stem.source_file not in out:
            candidate = Path(stem.source_file)
            out[stem.source_file] = file_sha256(candidate) if candidate.is_file() else "SYNTHETIC"
    return out


def _load(root: Path, stems_dir: Path, pool_dir: Path) -> tuple[list[Stem], list[PoolTool]]:
    return load_stems(root / stems_dir, root), load_pool(root / pool_dir)


def build(root: Path, *, stems_dir: Path, pool_dir: Path, out: Path, manifest: Path) -> int:
    stems, pool = _load(root, stems_dir, pool_dir)
    if not stems:
        print(f"no stems under {stems_dir}; nothing to build")
        return 2
    records = assemble(stems, pool)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    write_text_atomic(root / out, body)
    payload = build_manifest(records, stems, path=root / out)
    write_text_atomic(root / manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{len(records)} records, {payload['stem_count']} stems")
    print(f"  content fingerprint {payload['content_fingerprint'][:16]}")
    print(f"  record  fingerprint {payload['record_fingerprint'][:16]}")
    return 0


def verify(root: Path, *, stems_dir: Path, pool_dir: Path, out: Path, manifest: Path) -> int:
    """Rebuild in memory and diff. Writes nothing."""
    manifest_path, population_path = root / manifest, root / out
    if not manifest_path.exists() or not population_path.exists():
        print("VERIFY BLOCKED - no committed population to verify against")
        return 2
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    stems, pool = _load(root, stems_dir, pool_dir)
    records = assemble(stems, pool)

    content = corpus_fingerprint([r["content_hash"] for r in records])
    record_fp = corpus_fingerprint([r["record_hash"] for r in records])
    on_disk = [json.loads(line) for line in population_path.read_text(encoding="utf-8").splitlines() if line]

    problems: list[str] = []
    if content != recorded["content_fingerprint"]:
        problems.append(f"content fingerprint moved: {recorded['content_fingerprint'][:16]} -> {content[:16]}")
    if record_fp != recorded["record_fingerprint"]:
        problems.append(f"record fingerprint moved: {recorded['record_fingerprint'][:16]} -> {record_fp[:16]}")
    if file_sha256(population_path) != recorded["population_file_sha256"]:
        problems.append("population file sha256 does not match the manifest")
    if len(on_disk) != len(records):
        problems.append(f"record count moved: {len(on_disk)} on disk, {len(records)} rebuilt")

    rebuilt_by_id = {r["example_id"]: r for r in records}
    drifted = [
        r["example_id"] for r in on_disk
        if r["example_id"] not in rebuilt_by_id
        or rebuilt_by_id[r["example_id"]]["record_hash"] != r["record_hash"]
    ]
    if drifted:
        problems.append(f"{len(drifted)} records differ, e.g. {drifted[:10]}")

    for source, meta in recorded.get("stem_source_files", {}).items():
        path = root / source
        if meta["sha256"] != "SYNTHETIC" and path.is_file() and file_sha256(path) != meta["sha256"]:
            problems.append(f"authored stem file changed without a rebuild: {source}")

    if problems:
        print("VERIFY FAILED")
        for problem in problems:
            print(f"  {problem}")
        # A stable measurement surface under a moved file identity is a reportable, non-blocking
        # state: prose or provenance changed and previously recorded scores stay comparable.
        if content == recorded["content_fingerprint"] and record_fp != recorded["record_fingerprint"]:
            print("  status: CONTENT_STABLE_RECORD_DRIFT")
        return 1
    print(f"VERIFY PASSED - {len(records)} records reproduce both fingerprints")
    return 0
