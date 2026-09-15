"""The single source of identity. Every id in the corpus is minted here."""

from __future__ import annotations

from toolchoice.constants import ARM_TO_GOLD, CORPUS_ID, FAMILY_ARMS


def stem_id(family: str, ordinal: int) -> str:
    if family not in FAMILY_ARMS:
        raise ValueError(f"unknown family: {family}")
    return f"{CORPUS_ID}/{family}/{ordinal:04d}"


def example_id(stem: str, arm: str) -> str:
    if arm not in ARM_TO_GOLD:
        raise ValueError(f"unknown arm: {arm}")
    return f"{stem}/{arm.split('-', 1)[1]}"


def parse_stem_id(stem: str) -> tuple[str, str, int]:
    corpus, family, ordinal = stem.split("/")
    return corpus, family, int(ordinal)


def parse_example_id(example: str) -> tuple[str, str]:
    """Return (stem_id, arm). The inverse of example_id."""
    stem, suffix = example.rsplit("/", 1)
    _, family, _ = parse_stem_id(stem)
    arm = f"{family}-{suffix}"
    if arm not in ARM_TO_GOLD:
        raise ValueError(f"example id does not resolve to a known arm: {example}")
    return stem, arm


def sibling_example_ids(stem: str) -> list[str]:
    _, family, _ = parse_stem_id(stem)
    return sorted(example_id(stem, arm) for arm in FAMILY_ARMS[family])
