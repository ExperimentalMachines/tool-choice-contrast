"""The record schema. This is the only place the item shape is written down."""

from __future__ import annotations

from typing import Any

from opengrad.data.behavior import DECISIONS

from toolchoice.constants import ARM_TO_GOLD, CORPUS_ID, GOLD_TO_ALIAS, SCHEMA_VERSION
from toolchoice.hashing import content_hash, record_hash
from toolchoice.ids import parse_example_id, sibling_example_ids

ARTIFACT_KIND = "TOOL_CHOICE_CONTRAST_ITEM"

REQUIRED_FIELDS = (
    "schema_version", "artifact_kind", "corpus_id",
    "example_id", "stem_id", "family", "arm", "sibling_example_ids",
    "question", "tools", "tool_roles",
    "gold_decision", "gold_decision_alias", "gold_call",
    "contrast", "stem_provenance", "label_provenance", "metadata",
    "content_hash", "record_hash",
)

TOOL_ROLES = frozenset({"APPLICABLE", "DISTRACTOR", "UNDERSPECIFIED_VARIANT"})
ADJUDICATION_VERDICTS = frozenset({"ACCEPTED", "CHANGED", "ESCALATED_RESOLVED"})
#: PENDING exists so a population can be built and gated before anyone labels it. A frozen
#: release must contain zero PENDING items -- gate G7 enforces that, and it is the only thing
#: standing between "expanded" and "adjudicated".
PENDING = "PENDING"
LABEL_STATUSES = frozenset({PENDING, "LABELED", "AMBIGUOUS_EXCLUDED", "AMBIGUOUS_DUAL_GOLD"})


class RecordError(ValueError):
    """A record violates the schema. Carries a reason code, never a bare message."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)


def finalize(record: dict[str, Any]) -> dict[str, Any]:
    """Attach the derived fields and both digests. Order matters: content, then record."""
    stem, arm = parse_example_id(record["example_id"])
    record["schema_version"] = SCHEMA_VERSION
    record["artifact_kind"] = ARTIFACT_KIND
    record["corpus_id"] = CORPUS_ID
    record["stem_id"] = stem
    record["arm"] = arm
    record["family"] = arm.split("-", 1)[0]
    record["sibling_example_ids"] = sibling_example_ids(stem)
    # `.get`, not `[]`: an unrecognised decision must surface from validate() with a
    # reason code, not as a bare KeyError from the hashing path.
    record["gold_decision_alias"] = GOLD_TO_ALIAS.get(record["gold_decision"])
    record["content_hash"] = content_hash(record)
    record.pop("record_hash", None)
    record["record_hash"] = record_hash(record)
    return record


def validate(record: dict[str, Any]) -> None:
    """Structural validation. Raises RecordError with a reason code, never a bare assert."""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise RecordError("REC_MISSING_FIELD", ",".join(sorted(missing)))

    arm = record["arm"]
    if arm not in ARM_TO_GOLD:
        raise RecordError("REC_UNKNOWN_ARM", arm)

    gold = record["gold_decision"]
    if gold not in DECISIONS:
        # "DIRECT" lands here on purpose: routing_metrics would raise on it later,
        # and a corpus that cannot be scored is worse than one that fails to build.
        raise RecordError("REC_NONCANONICAL_DECISION", gold)
    if gold != ARM_TO_GOLD[arm]:
        raise RecordError("REC_ARM_GOLD_MISMATCH", f"{arm} declares {ARM_TO_GOLD[arm]}, got {gold}")

    if record["metadata"].get("eligibility") != "evaluation_only":
        raise RecordError("REC_NOT_EVALUATION_ONLY", record["example_id"])

    if record["example_id"] not in record["sibling_example_ids"]:
        raise RecordError("REC_SIBLINGS_EXCLUDE_SELF", record["example_id"])

    for name, role in record["tool_roles"].items():
        if role not in TOOL_ROLES:
            raise RecordError("REC_UNKNOWN_TOOL_ROLE", f"{name}={role}")
    offered = {t["name"] for t in record["tools"]}
    if offered != set(record["tool_roles"]):
        raise RecordError("REC_TOOL_ROLE_MISMATCH", record["example_id"])

    _validate_label_provenance(record)
    _validate_gold_call(record, offered)

    if record["content_hash"] != content_hash(record):
        raise RecordError("REC_CONTENT_HASH_DRIFT", record["example_id"])
    if record["record_hash"] != record_hash(record):
        raise RecordError("REC_RECORD_HASH_DRIFT", record["example_id"])


def _validate_label_provenance(record: dict[str, Any]) -> None:
    lp = record["label_provenance"]
    if lp.get("status") not in LABEL_STATUSES:
        raise RecordError("REC_LABEL_STATUS", str(lp.get("status")))

    proposal, adjudication = lp.get("proposal"), lp.get("adjudication")
    if lp["status"] == PENDING:
        # Unlabelled is a legitimate build state; it is not a legitimate release state.
        if proposal is not None or adjudication is not None:
            raise RecordError("REC_PENDING_WITH_LABELS", record["example_id"])
        if lp.get("label_authority") != "HUMAN":
            raise RecordError("REC_LABEL_AUTHORITY", str(lp.get("label_authority")))
        return
    if not isinstance(proposal, dict):
        raise RecordError("REC_NO_MODEL_PROPOSAL", record["example_id"])
    if not isinstance(adjudication, dict):
        raise RecordError("REC_NO_HUMAN_ADJUDICATION", record["example_id"])
    if lp.get("label_authority") != "HUMAN":
        raise RecordError("REC_LABEL_AUTHORITY", str(lp.get("label_authority")))
    # The two are siblings and are never merged. There is no top-level `label`.
    if "label" in lp:
        raise RecordError("REC_MERGED_LABEL", record["example_id"])
    if adjudication.get("verdict") not in ADJUDICATION_VERDICTS:
        raise RecordError("REC_UNKNOWN_VERDICT", str(adjudication.get("verdict")))
    if adjudication.get("verdict") == "CHANGED" and not adjudication.get("rationale"):
        raise RecordError("REC_CHANGED_WITHOUT_RATIONALE", record["example_id"])
    if record["gold_decision"] != adjudication.get("final_decision"):
        raise RecordError("REC_GOLD_NOT_FINAL", record["example_id"])

    agreement = lp.get("agreement", {})
    expected = proposal.get("proposed_decision") == adjudication.get("final_decision")
    if agreement.get("model_matched_human") is not expected:
        raise RecordError("REC_AGREEMENT_NOT_DERIVED", record["example_id"])


def _validate_gold_call(record: dict[str, Any], offered: set[str]) -> None:
    arm, gold_call = record["arm"], record["gold_call"]
    if arm == "E-applicable":
        if not isinstance(gold_call, dict):
            raise RecordError("REC_CALL_ARM_WITHOUT_GOLD_CALL", record["example_id"])
        if gold_call["name"] not in offered:
            raise RecordError("REC_GOLD_CALL_NOT_OFFERED", gold_call["name"])
        bindings = gold_call.get("argument_bindings", {})
        unbound = [p for p in gold_call.get("required_arguments", []) if p not in bindings]
        if unbound:
            raise RecordError("REC_UNBOUND_REQUIRED_ARG", ",".join(sorted(unbound)))
    elif gold_call is not None:
        raise RecordError("REC_GOLD_CALL_ON_NONCALL_ARM", f"{arm}:{record['example_id']}")
