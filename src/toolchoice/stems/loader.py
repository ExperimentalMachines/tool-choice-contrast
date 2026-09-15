"""Read authored stems and the tool pool from YAML, recording where each came from."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolchoice.constants import FAMILY_ARMS
from toolchoice.stems.model import GoldTool, PoolTool, Stem, UnderspecifiedExtra


class LoadError(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)


def _yaml(path: Path) -> Any:
    import yaml  # optional dependency: authoring only, never needed to score a run

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_pool(directory: Path) -> list[PoolTool]:
    tools: list[PoolTool] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        doc = _yaml(path) or {}
        for entry in doc.get("tools", []):
            pool_id = entry["pool_id"]
            if pool_id in seen:
                raise LoadError("POOL_DUPLICATE_ID", pool_id)
            seen.add(pool_id)
            tools.append(PoolTool(
                pool_id=pool_id, domain=entry["domain"], name=entry["name"],
                description=entry["description"], parameters=entry["parameters"],
                general_purpose=bool(entry.get("general_purpose", False)),
            ))
    # Sorted by pool_id so selection cannot depend on filesystem iteration order.
    return sorted(tools, key=lambda t: t.pool_id)


def load_stems(directory: Path, root: Path) -> list[Stem]:
    stems: list[Stem] = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(directory.rglob("*.yaml")):
        doc = _yaml(path) or {}
        rel = str(path.relative_to(root))
        domain = doc["domain"]
        for index, entry in enumerate(doc.get("stems", [])):
            family = entry["family"]
            if family not in FAMILY_ARMS:
                raise LoadError("STEM_UNKNOWN_FAMILY", f"{rel}:{family}")
            key = (family, int(entry["id"]))
            if key in seen:
                raise LoadError("STEM_DUPLICATE_ID", f"{family}/{entry['id']}")
            seen.add(key)
            stems.append(_stem(entry, family, domain, rel, index))
    return sorted(stems, key=lambda s: (s.family, s.stem_ordinal))


def _stem(entry: dict[str, Any], family: str, domain: str, rel: str, index: int) -> Stem:
    gold_tool = extra = None
    if family == "E":
        gt = entry["applicable_tool"]
        gold_tool = GoldTool(
            name=gt["name"], description=gt["description"],
            parameters=gt["parameters"], bindings=dict(gt.get("bindings", {})),
        )
        ue = entry["underspecified_extra"]
        extra = UnderspecifiedExtra(
            parameter=ue["parameter"], schema=ue["schema"], justification=ue["justification"],
        )
    return Stem(
        stem_ordinal=int(entry["id"]), family=family, domain=domain,
        question=entry["question"], source_file=rel, source_anchor=f"stems[{index}]",
        gold_answer=entry.get("gold_answer"), stem_source=entry.get("source"),
        capability=entry.get("capability"), gold_tool=gold_tool, underspecified_extra=extra,
        absent_substitute_pool_id=entry.get("absent_substitute"),
        notes=entry.get("notes", {}),
    )
