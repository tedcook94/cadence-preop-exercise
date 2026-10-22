"""Full pipeline tests with a mocked OpenAI client returning hand-written CaseFacts."""

import copy
import math
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import extract
from core import triage_submission
from schemas import (
    AnticoagulantFacts,
    CaseFacts,
    ConsentStatus,
    DocumentFacts,
    LabFacts,
    NoteVital,
    TriageOutput,
)

BASE_SUBMISSION: dict[str, Any] = {
    "procedure": {"case_id": "e2e", "procedure_risk": "MODERATE", "procedure_date": "2026-05-20"},
    "vitals": [
        {"type": "blood_pressure", "systolic": 124, "diastolic": 78, "date": "2026-05-10T09:00:00Z"},
        {"type": "temperature", "value_f": 98.4, "date": "2026-05-10T09:05:00Z"},
    ],
    "labs": [{"code": "CBC", "effective_at": "2026-05-05T08:00:00Z", "status": "final"}],
    "medications": [{"name": "lisinopril", "active": True}],
    "conditions": [],
    "documents": [
        {"type": "History and Physical", "date": "2026-05-01", "text": "Pre-op H&P."},
        {"type": "Surgical Consent", "date": "2026-05-02", "text": "Signed by patient."},
        {"type": "Nursing Intake", "date": "2026-05-18", "text": "Intake complete."},
    ],
}


def _doc(index: int, kind: str, status: ConsentStatus = "N/A") -> DocumentFacts:
    return DocumentFacts.model_validate(
        {
            "index": index,
            "doc_kind": kind,
            "consent_status": status,
            "consent_matches_procedure": "YES" if kind == "CONSENT" else "N/A",
            "evidence": f"documents[{index}] says so",
        }
    )


READY_FACTS = CaseFacts(
    documents=[_doc(0, "HP_PREOP"), _doc(1, "CONSENT", "SIGNED"), _doc(2, "NURSING")],
    labs=[LabFacts(index=0, test="CBC", evidence="labs[0] is a CBC")],
    note_vitals=[],
    anticoagulants=[],
)


def _submission(**updates: Any) -> dict[str, Any]:
    submission = copy.deepcopy(BASE_SUBMISSION)
    for key, value in updates.items():
        if key in ("procedure_date", "procedure_risk"):
            submission["procedure"][key] = value
        else:
            submission[key] = value
    return submission


def _response(facts: CaseFacts, low: dict[str, float] | None = None) -> SimpleNamespace:
    """Build a parse() response; `low` maps a regex for one value to its probability."""

    text = facts.model_dump_json()
    output: list[Any] = []
    if low is not None:
        tokens: list[SimpleNamespace] = []
        position = 0
        for pattern, probability in low.items():
            match = re.search(pattern, text[position:])
            assert match is not None
            start, end = position + match.start(1), position + match.end(1)
            tokens += [
                SimpleNamespace(token=text[position:start], logprob=0.0),
                SimpleNamespace(token=text[start:end], logprob=math.log(probability)),
            ]
            position = end
        tokens.append(SimpleNamespace(token=text[position:], logprob=0.0))
        content = SimpleNamespace(type="output_text", logprobs=tokens)
        output = [SimpleNamespace(content=[content])]
    return SimpleNamespace(output_parsed=facts, output=output, output_text=text)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock()
    monkeypatch.setattr(extract, "_client", mock)
    return mock


def _run(client: Mock, submission: dict[str, Any], response: SimpleNamespace) -> TriageOutput:
    client.responses.parse.return_value = response
    return triage_submission(submission, model="test-model")


def _categories(output: TriageOutput) -> list[str]:
    return [issue.category for issue in output.issues]


def test_ready(client: Mock) -> None:
    output = _run(client, _submission(), _response(READY_FACTS))

    assert (output.decision, output.issues) == ("READY", [])
    assert "satisfied" in output.explanation


def test_needs_follow_up_reports_every_issue(client: Mock) -> None:
    submission = _submission(medications=[{"name": "apixaban", "active": True}])
    submission["documents"][0]["date"] = "2026-04-01"
    submission["documents"].append({"type": "Cardiology", "date": "2026-05-12", "text": "Hold 2 days."})
    facts = READY_FACTS.model_copy(
        update={
            "documents": [*READY_FACTS.documents, _doc(3, "ANTICOAG_PLAN")],
            "anticoagulants": [
                AnticoagulantFacts(
                    drug="apixaban",
                    status="ACTIVE",
                    plan_doc_index=3,
                    plan_has_hold=True,
                    plan_has_resume=False,
                    plan_matches_drug=True,
                    evidence="documents[3]: hold 2 days; no restart given",
                )
            ],
        }
    )

    output = _run(client, submission, _response(facts))

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert _categories(output) == ["REQUIRED_DOCUMENTATION", "ANTICOAGULATION_MANAGEMENT"]
    assert output.explanation == " | ".join(f"{i.category}: {i.description}" for i in output.issues)


def test_not_cleared_still_lists_other_issues(client: Mock) -> None:
    facts = READY_FACTS.model_copy(
        update={
            "documents": [_doc(0, "HP_PREOP"), _doc(1, "CONSENT", "PENDING"), _doc(2, "NURSING")],
            "note_vitals": [
                NoteVital(
                    doc_index=2,
                    kind="BP",
                    systolic=186,
                    diastolic=95,
                    value=None,
                    unit=None,
                    timestamp="2026-05-18T10:00:00Z",
                    evidence="documents[2]: BP 186/95",
                )
            ],
        }
    )

    output = _run(client, _submission(), _response(facts))

    assert output.decision == "NOT_CLEARED"
    assert _categories(output) == ["REQUIRED_DOCUMENTATION", "ACUTE_SAFETY_EXCLUSION"]
    assert output.issues[1].evidence.source == "documents[2]"


def test_missing_procedure_date_skips_window_checks(client: Mock) -> None:
    submission = _submission(procedure_date=None)
    submission["documents"][0]["date"] = "2025-01-01"

    output = _run(client, submission, _response(READY_FACTS))

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert _categories(output) == ["MISSING_REQUIRED_DATA"]
    assert output.issues[0].evidence.source == "procedure.procedure_date"


def test_low_confidence_signed_consent_needs_follow_up(client: Mock) -> None:
    response = _response(READY_FACTS, low={r'"consent_status":"(SIGNED)"': 0.4})

    output = _run(client, _submission(), response)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert [i.description for i in output.issues] == ["Surgical consent not signed by patient"]


def _resolves(submission: dict[str, Any], path: str) -> bool:
    node: Any = submission
    for name, index in re.findall(r"(\w+)(?:\[(\d+)\])?", path):
        if not isinstance(node, dict) or name not in node:
            return False
        node = node[name]
        if index:
            if not isinstance(node, list) or int(index) >= len(node):
                return False
            node = node[int(index)]
    return True


def test_output_contract_and_determinism(client: Mock) -> None:
    submission = _submission(procedure_risk=None, vitals=[], labs=[])

    facts = READY_FACTS.model_copy(update={"labs": []})

    first = _run(client, submission, _response(facts))
    second = _run(client, submission, _response(facts))

    assert TriageOutput.model_validate(first.model_dump()) == first
    assert first.model_dump_json() == second.model_dump_json()
    assert first.issues
    assert all(_resolves(submission, issue.evidence.source) for issue in first.issues)
