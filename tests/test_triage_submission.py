from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import extract
from core import build_explanation, decide, triage_submission
from schemas import (
    CaseFacts,
    DocumentFacts,
    IssueEvidence,
    LabFacts,
    PatientSubmission,
    TriageIssue,
    TriageOutput,
)


def _doc(index: int, kind: str, status: str = "N/A", match: str = "N/A") -> DocumentFacts:
    return DocumentFacts.model_validate(
        {
            "index": index,
            "doc_kind": kind,
            "consent_status": status,
            "consent_matches_procedure": match,
            "evidence": f"documents[{index}]",
        }
    )


READY_FACTS = CaseFacts(
    documents=[_doc(0, "HP_PREOP"), _doc(1, "CONSENT", "SIGNED", "YES")],
    labs=[LabFacts(index=0, test="CBC", evidence="labs[0] CBC")],
    note_vitals=[],
    anticoagulants=[],
)


@pytest.fixture
def submission_payload() -> dict[str, object]:
    return {
        "patient": {"id": "patient-1"},
        "procedure": {"case_id": "case-1", "procedure_risk": "LOW", "procedure_date": "2026-02-01"},
        "vitals": [
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-01-25"},
            {"type": "temperature", "value_f": 98.6, "date": "2026-01-25"},
        ],
        "labs": [{"code": "CBC", "display": "Complete blood count", "effective_at": "2026-01-20", "status": "final"}],
        "medications": [],
        "conditions": [],
        "documents": [
            {"type": "history_and_physical", "date": "2026-01-20", "text": "History and physical completed."},
            {"type": "surgical_consent", "date": "2026-01-22", "text": "Signed surgical consent."},
        ],
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock()
    mock.responses.parse.return_value = SimpleNamespace(output_parsed=READY_FACTS, output=[])
    monkeypatch.setattr(extract, "_client", mock)
    return mock


def _issue(category: str) -> TriageIssue:
    return TriageIssue.model_validate(
        {"category": category, "description": "d", "evidence": IssueEvidence(source="s", details="x")}
    )


def test_ready_submission(client: Mock, submission_payload: dict[str, object]) -> None:
    output = triage_submission(submission_payload, model="test-model")

    assert isinstance(output, TriageOutput)
    assert (output.decision, output.issues) == ("READY", [])
    assert client.responses.parse.call_count == 1


def test_accepts_validated_submission(client: Mock, submission_payload: dict[str, object]) -> None:
    submission = PatientSubmission.model_validate(submission_payload)

    assert triage_submission(submission, model="test-model").decision == "READY"


def test_all_rules_report_together(client: Mock, submission_payload: dict[str, object]) -> None:
    submission_payload["vitals"] = [{"type": "blood_pressure", "systolic": 185, "diastolic": 80, "date": "2026-01-30"}]
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=READY_FACTS.model_copy(update={"labs": []}), output=[]
    )
    submission_payload["labs"] = []

    output = triage_submission(submission_payload, model="test-model")

    assert output.decision == "NOT_CLEARED"
    assert [i.category for i in output.issues] == [
        "MISSING_REQUIRED_DATA",
        "REQUIRED_TESTING",
        "ACUTE_SAFETY_EXCLUSION",
    ]
    assert output.explanation.startswith("MISSING_REQUIRED_DATA: Missing temperature reading | ")


@pytest.mark.parametrize(
    ("categories", "expected"),
    [
        ([], "READY"),
        (["REQUIRED_TESTING"], "NEEDS_FOLLOW_UP"),
        (["MISSING_REQUIRED_DATA", "ACUTE_SAFETY_EXCLUSION"], "NOT_CLEARED"),
    ],
)
def test_decision_precedence(categories: list[str], expected: str) -> None:
    assert decide([_issue(c) for c in categories]) == expected


def test_explanation_lists_issues() -> None:
    assert build_explanation([_issue("REQUIRED_TESTING")]) == "REQUIRED_TESTING: d"
    assert "satisfied" in build_explanation([])
