import json

from prompts import INSTRUCTIONS, POLICY, build_case_input


def test_case_input_is_deterministic_and_round_trips() -> None:
    a = {"procedure": {"procedure_risk": "LOW"}, "documents": [{"text": "x"}]}
    b = {"documents": [{"text": "x"}], "procedure": {"procedure_risk": "LOW"}}

    assert build_case_input(a) == build_case_input(b)
    assert json.loads(build_case_input(a).split("\n", 1)[1]) == {
        **a,
        "documents": [{"text": "x", "index": 0}],
        "labs": [],
    }


def test_case_input_labels_items_without_mutating_submission() -> None:
    submission = {"documents": [{"date": "2026-02-22"}, {"date": "2026-02-21"}]}

    payload = json.loads(build_case_input(submission).split("\n", 1)[1])

    assert [d["index"] for d in payload["documents"]] == [0, 1]
    assert "index" not in submission["documents"][0]


def test_instructions_are_static_and_policy_first() -> None:
    assert INSTRUCTIONS.startswith(POLICY)
    assert "{" not in INSTRUCTIONS
