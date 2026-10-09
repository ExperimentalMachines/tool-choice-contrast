"""LFM2.5's native tool calls, read strictly.

R1 to R3 are verbatim LFM2.5 replies captured 2026-10-10 from the ExecuTorch exports on a Mac.
The remaining fixtures are adapted from OpenWeights' ToolCallParserTest.
"""

from __future__ import annotations

import pytest
from opengrad.formatting.parser import ParsedCall, ParsedNativeOutput, parse_qwen_native_output

from toolchoice.formatting.lfm import parse_lfm_native_output

R1 = (
    '<|tool_call_start|>[get_account_balance(account_id="4471-9920")]<|tool_call_end|>'
    "I am retrieving the current cleared balance for account ID 4471-9920.<|im_end|>"
)
R2 = (
    '<|tool_call_start|>[send_message(to="James", body="James runs 3 sprints 3 times a week, '
    "with each sprint covering 60 meters. Total weekly distance: 3 sprints * 3 times * 60 meters "
    '= 540 meters.")]<|tool_call_end|>I will calculate the total distance James runs in a week '
    "based on the information provided: 3 sprints per week, 3 repetitions per sprint, and 60 "
    "meters per sprint. Then I will send you the result via message.<|im_end|>"
)
R3 = (
    "I don't have access to banking or financial account information. To check the cleared "
    "balance for account 4471-9920, you'll need to contact your bank directly or use their online "
    "banking portal. Would you like help drafting a message to your bank?<|im_end|>"
)


def _envelope(body: str) -> str:
    return f"<|tool_call_start|>{body}<|tool_call_end|>"


def test_real_call_reply_is_a_call_with_the_trailing_prose_as_content() -> None:
    assert parse_lfm_native_output(R1) == ParsedNativeOutput(
        "CALL",
        [ParsedCall("get_account_balance", {"account_id": "4471-9920"})],
        "I am retrieving the current cleared balance for account ID 4471-9920.<|im_end|>",
        "RAW_VALID",
        [],
        False,
    )


def test_real_call_keeps_commas_and_operators_inside_a_string_argument() -> None:
    parsed = parse_lfm_native_output(R2)
    assert parsed.decision == "CALL"
    assert parsed.calls == [ParsedCall("send_message", {
        "to": "James",
        "body": "James runs 3 sprints 3 times a week, with each sprint covering 60 meters. "
                "Total weekly distance: 3 sprints * 3 times * 60 meters = 540 meters.",
    })]


def test_prose_reply_is_decided_by_the_shared_qwen_rules() -> None:
    parsed = parse_lfm_native_output(R3)
    assert parsed == parse_qwen_native_output(R3)
    # The trailing question mark makes this CLARIFY under the shared rule, not UNSUPPORTED.
    assert (parsed.decision, parsed.status) == ("CLARIFY", "RAW_VALID")


def test_arguments_keep_python_literal_types() -> None:
    parsed = parse_lfm_native_output(
        _envelope("[search(query='rain', limit=5, exact=True, tag=None)]")
    )
    assert parsed.calls == [
        ParsedCall("search", {"query": "rain", "limit": 5, "exact": True, "tag": None})
    ]


def test_comma_inside_a_string_does_not_split_arguments() -> None:
    parsed = parse_lfm_native_output(_envelope("[note(text='Manila, Philippines')]"))
    assert parsed.calls == [ParsedCall("note", {"text": "Manila, Philippines"})]


def test_one_envelope_may_hold_several_calls() -> None:
    parsed = parse_lfm_native_output(_envelope("[a(x=1), b(y='two')]"))
    assert parsed.calls == [ParsedCall("a", {"x": 1}), ParsedCall("b", {"y": "two"})]


def test_backslashes_and_quotes_follow_python_string_rules() -> None:
    parsed = parse_lfm_native_output(_envelope(r"""[open(path='C:\\tmp', note='say "hi"')]"""))
    assert parsed.calls == [ParsedCall("open", {"path": "C:\\tmp", "note": 'say "hi"'})]


def test_think_block_is_not_content() -> None:
    raw = "<think>need the tool</think>" + _envelope("[a(x=1)]") + "On it."
    parsed = parse_lfm_native_output(raw)
    assert (parsed.decision, parsed.content) == ("CALL", "On it.")


def test_generation_cut_mid_call_is_an_unclosed_call_not_an_answer() -> None:
    raw = "Looking it up.<|tool_call_start|>[get_weather(city='Man"
    assert parse_lfm_native_output(raw, truncated=True) == ParsedNativeOutput(
        "ANSWER", [], raw, "FORMAT_ERROR", ["UNCLOSED_TOOL_CALL"], True
    )


@pytest.mark.parametrize(("body", "error"), [
    ("[f(1)]", "MALFORMED_TOOL_CALL[0]: f() has positional arguments"),
    ("[f(x=1+2)]", "MALFORMED_TOOL_CALL[0]: f() argument 'x' is not a literal"),
    ("[a.b(x=1)]", "MALFORMED_TOOL_CALL[0]: element is not a call to a plain function name"),
    ("f(x=1)", "MALFORMED_TOOL_CALL[0]: payload is not a list of calls"),
    ("[f(**kw)]", "MALFORMED_TOOL_CALL[0]: f() unpacks **kwargs"),
    ("[f(x=1, x=2)]", "MALFORMED_TOOL_CALL[0]: f() repeats argument 'x'"),
    ("[f(x=", "MALFORMED_TOOL_CALL[0]: not a Python expression"),
    ("[]", "EMPTY_TOOL_CALL"),
])
def test_malformed_payload_is_a_format_error_and_never_a_partial_call(
    body: str, error: str
) -> None:
    parsed = parse_lfm_native_output("Sure." + _envelope(body))
    assert (parsed.decision, parsed.calls, parsed.content, parsed.status) == (
        "ANSWER", [], "Sure.", "FORMAT_ERROR"
    )
    assert parsed.errors is not None and len(parsed.errors) == 1
    assert parsed.errors[0].startswith(error)


def test_one_bad_envelope_voids_the_good_one() -> None:
    parsed = parse_lfm_native_output(_envelope("[a(x=1)]") + _envelope("[b(1)]"))
    assert (parsed.status, parsed.calls) == ("FORMAT_ERROR", [])
    assert parsed.errors == ["MALFORMED_TOOL_CALL[1]: b() has positional arguments"]
