import json

from prompts import INSTRUCTIONS, POLICY, build_case_input


def test_case_input_is_deterministic_and_round_trips() -> None:
    a = {"procedure": {"procedure_risk": "LOW"}, "documents": [{"text": "x"}]}
    b = {"documents": [{"text": "x"}], "procedure": {"procedure_risk": "LOW"}}

    assert build_case_input(a) == build_case_input(b)
    assert json.loads(build_case_input(a).split("\n", 1)[1]) == a


def test_instructions_are_static_and_policy_first() -> None:
    assert INSTRUCTIONS.startswith(POLICY)
    assert "{" not in INSTRUCTIONS
