from __future__ import annotations

import pytest

from tests.conftest import make_record
from toolchoice.constants import ARM_TO_GOLD, FAMILY_ARMS
from toolchoice.hashing import content_hash, record_hash
from toolchoice.population.record import RecordError, finalize, validate


@pytest.mark.parametrize("arm", sorted(ARM_TO_GOLD))
def test_every_arm_produces_a_valid_record(arm: str) -> None:
    validate(make_record(family=arm.split("-", 1)[0], arm=arm))


def test_gold_decision_uses_canonical_vocabulary_not_direct() -> None:
    with pytest.raises(RecordError) as exc:
        validate(make_record(arm="S-none", family="S", gold_decision="DIRECT"))
    assert exc.value.reason_code == "REC_NONCANONICAL_DECISION"


def test_arm_gold_is_definitional() -> None:
    with pytest.raises(RecordError) as exc:
        validate(make_record(arm="E-applicable", gold_decision="CLARIFY"))
    assert exc.value.reason_code == "REC_ARM_GOLD_MISMATCH"


def test_every_item_is_evaluation_only() -> None:
    record = make_record()
    record["metadata"]["eligibility"] = "training"
    with pytest.raises(RecordError) as exc:
        validate(record)
    assert exc.value.reason_code == "REC_NOT_EVALUATION_ONLY"


def test_sibling_ids_are_complete_and_include_self() -> None:
    for family, arms in FAMILY_ARMS.items():
        record = make_record(family=family, arm=arms[0])
        assert len(record["sibling_example_ids"]) == len(arms)
        assert record["example_id"] in record["sibling_example_ids"]


def test_proposal_and_adjudication_are_never_merged() -> None:
    record = make_record()
    record["label_provenance"]["label"] = "CALL"
    with pytest.raises(RecordError) as exc:
        validate(record)
    assert exc.value.reason_code == "REC_MERGED_LABEL"


def test_gold_decision_must_equal_human_final_decision() -> None:
    record = make_record()
    record["label_provenance"]["adjudication"]["final_decision"] = "ANSWER"
    with pytest.raises(RecordError) as exc:
        validate(record)
    assert exc.value.reason_code == "REC_GOLD_NOT_FINAL"


def test_agreement_is_derived_not_asserted() -> None:
    record = make_record(proposed="CLARIFY")
    assert record["label_provenance"]["agreement"]["model_matched_human"] is False
    record["label_provenance"]["agreement"]["model_matched_human"] = True
    with pytest.raises(RecordError) as exc:
        validate(record)
    assert exc.value.reason_code == "REC_AGREEMENT_NOT_DERIVED"


def test_changed_verdict_requires_a_rationale() -> None:
    record = make_record(proposed="CLARIFY")
    record["label_provenance"]["adjudication"]["rationale"] = ""
    with pytest.raises(RecordError) as exc:
        validate(record)
    assert exc.value.reason_code == "REC_CHANGED_WITHOUT_RATIONALE"


def test_call_arm_requires_every_required_argument_bound() -> None:
    record = make_record(arm="E-applicable")
    record["gold_call"]["required_arguments"] = ["account_id", "as_of_date"]
    with pytest.raises(RecordError) as exc:
        validate(finalize(record))
    assert exc.value.reason_code == "REC_UNBOUND_REQUIRED_ARG"


def test_non_call_arm_must_not_carry_a_gold_call() -> None:
    record = make_record(arm="E-absent")
    record["gold_call"] = {"name": "send_email"}
    with pytest.raises(RecordError) as exc:
        validate(finalize(record))
    assert exc.value.reason_code == "REC_GOLD_CALL_ON_NONCALL_ARM"


def test_content_hash_excludes_label_provenance() -> None:
    """A rationale typo must not invalidate a score recorded months earlier."""
    before = make_record()
    after = make_record()
    after["label_provenance"]["adjudication"]["rationale"] = "Reworded, same verdict."
    assert content_hash(after) == before["content_hash"]
    assert record_hash(after) != before["record_hash"]


def test_content_hash_covers_the_tool_catalogue() -> None:
    a = make_record(arm="E-applicable")
    b = make_record(arm="E-underspecified")
    assert a["content_hash"] != b["content_hash"]
