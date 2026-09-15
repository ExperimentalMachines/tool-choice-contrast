from __future__ import annotations

import pytest

from toolchoice.constants import E_DISTRACTOR_COUNT, FAMILY_ARMS
from toolchoice.stems.expand import expand
from toolchoice.stems.synthetic import pool, population, stems


def test_question_is_byte_identical_across_a_stems_arms() -> None:
    """Gate G3's invariant, established here: no code path may touch stem.question."""
    tool_pool = pool()
    for stem in stems(s_count=3, e_count=3):
        questions = {item["question"] for item in expand(stem, tool_pool)}
        assert len(questions) == 1


def test_e_arms_offer_exactly_equal_catalogue_sizes() -> None:
    """Substitution, not deletion, so catalogue size cannot explain a behaviour change."""
    tool_pool = pool()
    for stem in stems(s_count=0, e_count=5):
        sizes = {len(item["tools"]) for item in expand(stem, tool_pool)}
        assert sizes == {E_DISTRACTOR_COUNT + 1}


def test_e_arms_share_their_distractors() -> None:
    """Only the capability-bearing tool may vary across E arms."""
    items = [i for i in population(s_count=0, e_count=1)]
    by_arm = {i["example_id"].rsplit("/", 1)[1]: i for i in items}
    distractors = {
        arm: {n for n, role in item["tool_roles"].items() if role == "DISTRACTOR"}
        for arm, item in by_arm.items()
    }
    assert distractors["applicable"] == distractors["underspecified"]


def test_underspecified_required_set_is_a_strict_superset() -> None:
    items = population(s_count=0, e_count=1)
    by_arm = {i["example_id"].rsplit("/", 1)[1]: i for i in items}
    gold = next(t for t in by_arm["applicable"]["tools"] if t["name"] == "get_account_balance")
    variant = next(t for t in by_arm["underspecified"]["tools"] if t["name"] == "get_account_balance")
    assert set(gold["parameters"]["required"]) < set(variant["parameters"]["required"])
    extra = set(variant["parameters"]["required"]) - set(gold["parameters"]["required"])
    for name in extra:
        # Rubric E.4.2 condition 2: a schema default makes the parameter derivable.
        assert "default" not in variant["parameters"]["properties"][name]


def test_general_purpose_tools_are_barred_from_family_s() -> None:
    """Rubric S.3.3: a calculator offered against an arithmetic question is applicable."""
    for item in population(s_count=6, e_count=0):
        assert "calculator" not in item["tool_roles"]


def test_s_distractors_never_share_the_stem_domain() -> None:
    tool_pool = {t.name: t for t in pool()}
    for item in population(s_count=6, e_count=0):
        for name in item["tool_roles"]:
            assert tool_pool[name].domain != "math"


def test_distractor_selection_is_seed_stable() -> None:
    first = {i["example_id"]: i["tool_pool_ids"] for i in population(s_count=5, e_count=2)}
    second = {i["example_id"]: i["tool_pool_ids"] for i in population(s_count=5, e_count=2)}
    assert first == second


def test_every_stem_expands_to_its_full_arm_set() -> None:
    tool_pool = pool()
    for stem in stems(s_count=2, e_count=2):
        arms = {i["example_id"].rsplit("/", 1)[1] for i in expand(stem, tool_pool)}
        expected = {a.split("-", 1)[1] for a in FAMILY_ARMS[stem.family]}
        assert arms == expected


def test_incomplete_stem_raises_rather_than_shrinking_the_denominator() -> None:
    from toolchoice.scoring.contrast import Outcome, group_by_stem
    from toolchoice.scoring.decide import decide

    outcomes = [
        Outcome("tcc-v1/S/0001/none", "tcc-v1/S/0001", "S-none", "ANSWER", decide("The answer is 12.")),
    ]
    with pytest.raises(ValueError, match="incomplete stem"):
        group_by_stem(outcomes, "S")
